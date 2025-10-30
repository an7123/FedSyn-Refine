#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-level DP Top-K Seed Finder (Tokens + Phrases) – rigorous, label-private, across-class capped.

What this script does:
- Builds a single set of *global clients* (label-private simulation).
- For each user, computes per-class sums S and counts W from per-example L2-clipped embeddings.
- Applies optional per-class cap (--tau) and an across-class cap (--tau_tot) to bound user-level sensitivity.
- Secure-aggregates (simulated by sums), then adds server-side Gaussian noise for user-level DP:
    ΔS = 2 * R * tau_tot,  ΔW = 2 * tau_tot
  (full epsilon used across classes; split only between S and W).
- Scores a PUBLIC vocabulary by centered cosine vs. the DP mean; returns Top-K tokens.
- Builds phrase seeds (hill-climb + compositional) and MMR-reranks.
- Saves per-class files compatible with your downstream pipeline.

Dependencies:
    pip install sentence-transformers gensim nltk datasets torch numpy
"""

import os, re, math, argparse, random
from itertools import combinations
from typing import List, Tuple, Dict
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords

# ----------------------------
# Imports and setup
# ----------------------------
try:
    from sentence_transformers import SentenceTransformer
    import gensim.downloader as api
except ImportError:
    raise SystemExit("Please pip install: sentence-transformers gensim nltk datasets torch numpy")

# ----------------------------
# Globals & Constants
# ----------------------------
AG_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
PUBMED_LABELS = {0: "BACKGROUND", 1: "OBJECTIVE", 2: "METHODS", 3: "RESULTS", 4: "CONCLUSIONS"}
PUBMED_LABEL_TO_ID = {name: i for i, name in PUBMED_LABELS.items()}

try:
    EN_STOP = set(stopwords.words("english"))
except Exception:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    EN_STOP = set(stopwords.words("english"))

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def build_public_dictionary(max_vocab=20000, model_name="glove-wiki-gigaword-100"):
    """Public vocabulary (letters-only by default; relax regex if you want numeric/biomed tokens)."""
    print(f"Loading PUBLIC vocabulary from {model_name} ...")
    model = api.load(model_name)
    words = []
    rx = re.compile(r"^[a-z]+$")  # consider relaxing to include digits/units if needed
    for w in model.index_to_key:
        if len(words) >= max_vocab:
            break
        if rx.fullmatch(w) and (w not in EN_STOP):
            words.append(w)
    return words

def embed_texts(texts, st_model, template: str=None):
    if template:
        texts = [template.format(t=t) for t in texts]
    E = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / np.maximum(norms, 1e-9)).astype(np.float32)

def build_doc_embeddings(docs, st_model, clip_norm=20.0, clip_mode="clip"):
    """
    clip_mode="clip": true L2 clip to radius R (never up-scales).
    clip_mode="scale": normalize then scale to exactly R (higher SNR; still DP-safe since ||x||<=R).
    """
    X = st_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if clip_mode == "scale":
        X = (X / np.maximum(norms, 1e-9)) * clip_norm
    else:
        scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-9))
        X = X * scale
    return X.astype(np.float32)

def gaussian_noise_sigma(sensitivity, epsilon, delta):
    epsilon = max(float(epsilon), 1e-12)
    delta = max(float(delta), 1e-18)
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

# ----------------------------
# Client construction (label-private)
# ----------------------------
def make_global_clients(docs_by_class: Dict[int, List[str]], client_size: int = 20) -> List[Dict[int, List[str]]]:
    """
    Build clients once from the whole dataset, then each client holds a dict lid->list_of_docs (possibly empty).
    This simulates label-participation privacy (clients conceptually send a masked vector per class).
    """
    all_pairs = [(lid, t) for lid, docs in docs_by_class.items() for t in docs]
    random.shuffle(all_pairs)
    chunks = [all_pairs[i:i + client_size] for i in range(0, len(all_pairs), client_size)]
    lids = sorted(docs_by_class.keys())
    clients = []
    for ch in chunks:
        per_class = {lid: [] for lid in lids}
        for lid, text in ch:
            per_class[lid].append(text)
        clients.append(per_class)
    return clients  # list of dicts: lid -> docs (maybe empty)

# ----------------------------
# Phrase utilities
# ----------------------------
def mmr_rerank(candidates: List[str], cand_embs: np.ndarray, relevance_scores: np.ndarray, k: int, lambda_val: float = 0.5) -> List[str]:
    if not candidates or k <= 0: return []
    num_cands = cand_embs.shape[0]
    k = min(k, num_cands)
    all_indices = np.arange(num_cands)
    best_idx = int(np.argmax(relevance_scores))
    selected_indices = [best_idx]
    is_remaining = np.ones(num_cands, dtype=bool); is_remaining[best_idx] = False
    while len(selected_indices) < k and np.any(is_remaining):
        selected_embs = cand_embs[selected_indices]
        remaining_indices = all_indices[is_remaining]
        remaining_embs = cand_embs[remaining_indices]
        sim_matrix = remaining_embs @ selected_embs.T
        max_sim = np.max(sim_matrix, axis=1) if sim_matrix.size else np.array([])
        rem_rel = relevance_scores[remaining_indices]
        mmr_scores = lambda_val * rem_rel - (1 - lambda_val) * max_sim
        if not remaining_indices.size: break
        pick = int(np.argmax(mmr_scores))
        add_idx = int(remaining_indices[pick])
        selected_indices.append(add_idx)
        is_remaining[add_idx] = False
    return [candidates[i] for i in selected_indices]

def hillclimb_phrases(top_tokens, st_model, target_vec, max_len=3, cand_per_len=200, steps=200, rng=None):
    rng = rng or np.random.default_rng(0)
    def score(text):
        e = embed_texts([text], st_model, template="{t}")[0]
        return float(e @ target_vec)
    tokens = top_tokens[: min(200, len(top_tokens))]
    phrases = set(); results = []
    for t in tokens[:50]:
        s = score(t); results.append((s, t)); phrases.add(t)
    for n in [2, 3]:
        seeds = set()
        if len(tokens) < n: continue
        for _ in range(cand_per_len):
            cand = tuple(rng.choice(tokens, size=n, replace=False))
            seeds.add(cand)
        for cand in list(seeds):
            best = list(cand); best_s = score(" ".join(best))
            for _ in range(steps):
                pos = rng.integers(0, n)
                new_tok = rng.choice(tokens)
                if new_tok in best and len(set(best)) == len(best):
                    avail = [t for t in tokens if t not in best or t == best[pos]]
                    if not avail: continue
                    new_tok = rng.choice(avail)
                new = best.copy(); new[pos] = new_tok
                if len(set(new)) < n: continue
                s = score(" ".join(new))
                if s > best_s: best, best_s = new, s
            text = " ".join(best)
            if text not in phrases:
                phrases.add(text)
                results.append((best_s, text))
    results.sort(key=lambda x: x[0], reverse=True)
    seen = set(); ranked = []
    for s, t in results:
        if t in seen: continue
        seen.add(t); ranked.append((s, t))
    return [t for _, t in ranked]

def build_compositional_phrases(top_tokens: List[str], public_vocab_map: dict, public_emb: np.ndarray, ngram_sizes: List[int] = [2, 3], max_candidates: int = 10000) -> Tuple[List[str], np.ndarray]:
    phrase_cands = []
    eligible = [t for t in top_tokens if t in public_vocab_map]
    for n in ngram_sizes:
        if len(eligible) < n: continue
        combs = list(combinations(eligible, n))
        phrase_cands.extend(combs)
    if len(phrase_cands) > max_candidates:
        random.shuffle(phrase_cands)
        phrase_cands = phrase_cands[:max_candidates]
    final_phrases, composed_embeddings = [], []
    for toks in phrase_cands:
        idxs = [public_vocab_map[t] for t in toks if t in public_vocab_map]
        if len(idxs) != len(toks): continue
        token_embs = public_emb[idxs]
        vec = token_embs.sum(axis=0)
        nrm = np.linalg.norm(vec)
        if nrm > 1e-9:
            vec /= nrm
            final_phrases.append(" ".join(toks))
            composed_embeddings.append(vec)
    if not composed_embeddings:
        return [], np.array([])
    return final_phrases, np.vstack(composed_embeddings).astype(np.float32)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="pubmed", choices=["ag_news", "pubmed"])
    ap.add_argument("--epsilon", type=float, default=5.0)
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--clip_norm", type=float, default=20.0, help="R: per-document L2 clipping radius")
    ap.add_argument("--clip_mode", choices=["clip","scale"], default="clip",
                    help="clip: true L2 clip to R; scale: normalize then scale to R (higher SNR).")
    ap.add_argument("--tau", type=float, default=5.0,
                    help="Optional per-class cap per user (sum-norm cap tau*R and count cap tau). Set 0 to disable.")
    ap.add_argument("--tau_tot", type=float, default=None,
                    help="Across-class cap per user (sum of per-class counts). If None, defaults to --tau. Set >0 to avoid sqrt(C) sensitivity.")
    ap.add_argument("--st_model", type=str, default="pritamdeka/S-BioBert-snli-multinli-stsb")
    ap.add_argument("--max_public_vocab", type=int, default=20000)
    ap.add_argument("--K_tokens", type=int, default=1000)
    ap.add_argument("--K_phrases", type=int, default=5000)
    ap.add_argument("--M_seed", type=int, default=400)
    ap.add_argument("--N_compositional", type=int, default=1000)
    ap.add_argument("--client_size", type=int, default=20)
    ap.add_argument("--limit_per_class", type=int, default=None)
    ap.add_argument("--mmr_lambda", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--central_equiv_debug", action="store_true",
                    help="Disable DP noise; caps still applied as configured. For exact centralized on clipped data: also set --tau=0 --tau_tot=0 and --clip_mode clip/scale as desired.")
    args = ap.parse_args()

    # Derive tau_tot default
    if args.tau_tot is None:
        args.tau_tot = float(args.tau) if args.tau > 0 else 0.0

    # --- Dataset-specific configuration ---
    if args.dataset == "ag_news":
        LABELS = AG_LABELS
        ds = load_dataset("ag_news", split="train")
        if args.limit_per_class is None: args.limit_per_class = 4000
        if args.output_dir is None: args.output_dir = "dp_outputs_agnews_refined"
    elif args.dataset == "pubmed":
        LABELS = PUBMED_LABELS
        ds = load_dataset("armanc/pubmed-rct20k", split="train")
        if args.limit_per_class is None: args.limit_per_class = 10000
        if args.output_dir is None: args.output_dir = "dp_outputs_pubmed_refined"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Running on dataset: {args.dataset.upper()}")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load data
    docs_by_class = {i: [] for i in LABELS}
    for ex in ds:
        if args.dataset == "pubmed":
            label_str = ex["label"].upper()
            lid = PUBMED_LABEL_TO_ID[label_str]
        else:
            lid = int(ex["label"])
        if args.limit_per_class and len(docs_by_class[lid]) >= args.limit_per_class:
            continue
        docs_by_class[lid].append(ex["text"])

    # 2) Public models/resources
    print(f"Loading SentenceTransformer: {args.st_model} ...")
    st = SentenceTransformer(args.st_model)
    d = st.get_sentence_embedding_dimension()
    print("Building PUBLIC vocabulary ...")
    public_tokens = build_public_dictionary(max_vocab=args.max_public_vocab)
    public_vocab_map = {token: i for i, token in enumerate(public_tokens)}
    print(f"Encoding PUBLIC vocab (|V_pub|={len(public_tokens)}) ...")
    public_emb = embed_texts(public_tokens, st, template="{t}")

    # 3) Clients (label-private simulation)
    global_clients = make_global_clients(docs_by_class, client_size=args.client_size)
    lids_sorted = sorted(LABELS.keys())

    # 4) Aggregate per user with per-class and across-class caps
    S_global = {lid: np.zeros(d, dtype=np.float32) for lid in lids_sorted}
    W_global = {lid: 0.0 for lid in lids_sorted}

    for cli in global_clients:
        # First, build raw per-class stats for this user
        s_per_c = []
        w_per_c = []
        for lid in lids_sorted:
            docs = cli[lid]
            if docs:
                X = build_doc_embeddings(docs, st, clip_norm=args.clip_norm, clip_mode=args.clip_mode)
                s = X.sum(axis=0)
                w = float(len(docs))
                # Optional per-class cap
                if args.tau > 0:
                    cap_s = args.tau * args.clip_norm
                    nrm = float(np.linalg.norm(s))
                    if nrm > cap_s:
                        s = (cap_s / nrm) * s
                    w = min(w, float(args.tau))
            else:
                s = np.zeros(d, dtype=np.float32)
                w = 0.0
            s_per_c.append(s.astype(np.float32))
            w_per_c.append(w)

        s_per_c = np.vstack(s_per_c)                    # (C, d)
        w_per_c = np.array(w_per_c, dtype=np.float32)   # (C,)

        # Across-class cap (simplex): sum_c w_jc ≤ tau_tot
        tau_tot = float(args.tau_tot)
        total_w = float(w_per_c.sum())
        if tau_tot > 0.0 and total_w > tau_tot:
            alpha = tau_tot / total_w
            w_per_c *= alpha
            s_per_c *= alpha

        # Accumulate into global sums
        for idx, lid in enumerate(lids_sorted):
            S_global[lid] += s_per_c[idx]
            W_global[lid] += float(w_per_c[idx])

    # 5) DP calibration (full epsilon; split only between S and W)
    gamma = 1e-3 * max(args.tau_tot, 1.0)  # small floor to stabilize the ratio
    if args.central_equiv_debug:
        sigma_S = sigma_W = 0.0
        print("[DEBUG] DP noise disabled. Caps as configured (per-class and across-class).")
    else:
        if args.tau_tot <= 0:
            raise SystemExit("For user-level DP across classes, set --tau_tot > 0 (across-class cap).")
        eps_S = args.epsilon / 2.0
        eps_W = args.epsilon / 2.0
        delta_S = args.delta / 2.0
        delta_W = args.delta / 2.0
        Delta_S = 2.0 * args.clip_norm * args.tau_tot
        Delta_W = 2.0 * args.tau_tot
        sigma_S = gaussian_noise_sigma(Delta_S, eps_S, delta_S)
        sigma_W = gaussian_noise_sigma(Delta_W, eps_W, delta_W)
        print(f"[DP] σ_S={sigma_S:.4g} (Δ_S={Delta_S:.4g}, ε_S={eps_S}, δ_S={delta_S})  "
              f"σ_W={sigma_W:.4g} (Δ_W={Delta_W:.4g}, ε_W={eps_W}, δ_W={delta_W})")

    # 6) Per-class token/phrase scoring and saving
    for lid, cname in LABELS.items():
        S_c = S_global[lid]
        W_c = W_global[lid]
        if W_c == 0.0:
            print(f"\n=== Class {lid}: {cname} (docs=0) - SKIPPING ===")
            continue

        # Add DP noise (server-side, post-SecAgg)
        Z_c = rng.normal(0.0, sigma_S, size=S_c.shape).astype(np.float32) if sigma_S > 0 else 0.0
        zeta_c = rng.normal(0.0, sigma_W) if sigma_W > 0 else 0.0  # scalar draw (no deprecation)

        denom = max(W_c + zeta_c, gamma)
        mu_hat = (S_c + Z_c) / denom  # keep raw (scoring will center/normalize)

        # Token scoring: centered cosine with CSLS-like adjustment
        cand_mean = public_emb.mean(0, keepdims=True)
        public_emb_centered = public_emb - cand_mean
        public_emb_centered /= np.maximum(np.linalg.norm(public_emb_centered, axis=1, keepdims=True), 1e-9)
        q = (mu_hat - cand_mean[0])
        q /= max(np.linalg.norm(q), 1e-9)
        sims = public_emb_centered @ q
        r_query = float(np.mean(np.sort(sims)[-min(50, len(sims)):]))

        csls_scores = 2.0 * sims - r_query
        topk_idx = np.argsort(csls_scores)[::-1][: args.K_tokens]
        topk_tokens = [public_tokens[i] for i in topk_idx]
        print(f"\n=== Class {lid}: {cname} ===")
        print(f"Top-10 TOKENS: {topk_tokens[:10]}")

        # Phrases (hill-climb + compositional), then MMR
        rng_local = np.random.default_rng(args.seed + lid)
        seed_M = min(args.M_seed, len(topk_tokens))
        phrase_cands_A = hillclimb_phrases(topk_tokens[:seed_M], st, q, rng=rng_local)

        seed_N = min(args.N_compositional, len(topk_tokens))
        phrase_cands_B, p_emb_B = build_compositional_phrases(
            top_tokens=topk_tokens[:seed_N], public_vocab_map=public_vocab_map, public_emb=public_emb
        )
        p_emb_A = embed_texts(phrase_cands_A, st, template="{t}") if phrase_cands_A else np.array([])

        combined_phrases = phrase_cands_A + phrase_cands_B
        if p_emb_A.size > 0 and p_emb_B.size > 0: combined_embs = np.vstack([p_emb_A, p_emb_B])
        elif p_emb_A.size > 0: combined_embs = p_emb_A
        else: combined_embs = p_emb_B

        unique_map = {p: i for i, p in enumerate(combined_phrases)}
        phrase_pool = list(unique_map.keys())
        p_emb_pool = np.array([combined_embs[unique_map[p]] for p in phrase_pool]) if combined_phrases else np.array([])

        if phrase_pool and p_emb_pool.size > 0:
            p_center = p_emb_pool - cand_mean
            p_center /= np.maximum(np.linalg.norm(p_center, axis=1, keepdims=True), 1e-9)
            sims_phrases = p_center @ q
            final_phrases = mmr_rerank(
                candidates=phrase_pool, cand_embs=p_center,
                relevance_scores=sims_phrases, k=args.K_phrases,
                lambda_val=args.mmr_lambda
            )
        else:
            final_phrases = []

        print(f"Top-10 PHRASES: {final_phrases[:10]}")

        # Save
        out_path = os.path.join(args.output_dir, f"class{lid}_{cname.replace('/', '-')}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Class {lid} ({cname})\n")
            f.write("=== Top-K TOKENS ===\n")
            for w in topk_tokens: f.write(w + "\n")
            f.write("\n=== Top-K PHRASES ===\n")
            for p in final_phrases: f.write(p + "\n")
        print(f"[Saved results to {out_path}]")

if __name__ == "__main__":
    main()

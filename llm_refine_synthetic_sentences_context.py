#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM refinement for generating synthetic sentences from DP-generated seeds.

This script takes outputs from the DP seed generation pipeline
(e.g., dp_outputs_pubmed/class{cid}_{Name}.txt) which contain both
Top-K Tokens and Top-K Phrases. It uses a pretrained LLM (e.g., Mistral)
to refine these seeds into fluent, class-conditioned synthetic sentences.

**IMPROVEMENT**: This version uses seed-specific context. For each seed used
in a prompt, the "related context terms" are sampled from the *other* top
seeds (tokens and phrases) from the same class, making the prompt more coherent.

Inputs per class from DP pipeline:
  - {output_dir_from_dp_script}/class{cid}_{Name}.txt (containing tokens and phrases)

Output per class:
  - {out_dir}/refined_class{cid}_{Name}.jsonl
    with objects: {"seed","class","sentence"}

Dependencies:
    pip install transformers torch tqdm
"""

import os, re, json, argparse, logging, random
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Constants for parsing DP output files ---
TOKEN_SECTION_HEADER_RX = re.compile(r"^===\s*Top-K TOKENS\s*===", re.IGNORECASE)
PHRASE_SECTION_HEADER_RX = re.compile(r"^===\s*Top-K PHRASES\s*===", re.IGNORECASE)
CLASS_HEADER_RX = re.compile(r"^Class\s*(\d+)\s*\(([^)]+)\)")

# --- LLM Prompting Constants ---
SYSTEM_PROMPT = (
    "You are an expert scientific writer. Given a label and a seed topic describing the rhetorical role "
    "of a sentence in a scientific abstract (e.g., background, objective, methods, results, conclusions), "
    "and a set of related context terms, write one realistic, concise, and academic sentence that clearly matches the role. "
    "Avoid patient identifiers, PHI, and specific grant numbers. Keep it generic and medically plausible."
)

STYLE_GUIDELINES = (
    "Guidelines:\n"
    "- Write exactly ONE sentence (15–40 words).\n"
    "- Keep the tone academic and concise.\n"
    "- Do not fabricate exact statistics unless the role demands them; prefer approximate phrasing.\n"
    "- Avoid rare drug brand names; use generic terms (e.g., 'antihypertensive therapy').\n"
    "- Ensure the sentence directly relates to the seed/topic and role label."
)

USER_PROMPT_TEMPLATE = (
    "Role label: {label}\n"
    "Seed/topic: {seed}\n"
    "Related context terms: {ctx}\n\n"
    "{style}\n\n"
    "Return exactly one line in this format (no extra text):\n"
    "Sentence: <your single sentence here>"
)

SENTENCE_EXTRACTION_RX = re.compile(r"^\s*Sentence\s*:\s*(.+)", re.IGNORECASE)

# --- Helper Functions ---

def read_dp_output_file(path: str) -> Tuple[List[str], List[str]]:
    """
    Reads the DP output file and extracts Top-K Tokens and Top-K Phrases.
    Assumes the file format from dp_topk_userlevel_seeds_refined.py.
    """
    tokens = []
    phrases = []
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        in_tokens_section = False
        in_phrases_section = False
        for line in f:
            s = line.strip()
            if not s: continue

            if TOKEN_SECTION_HEADER_RX.search(s):
                in_tokens_section = True
                in_phrases_section = False
                continue
            elif PHRASE_SECTION_HEADER_RX.search(s):
                in_tokens_section = False
                in_phrases_section = True
                continue
            elif CLASS_HEADER_RX.search(s):
                in_tokens_section = False
                in_phrases_section = False
                continue

            # Handle content based on current section
            # The first block before any header is assumed to be tokens
            if in_tokens_section or (not in_phrases_section and not s.startswith("===")):
                tokens.append(s)
            elif in_phrases_section:
                phrases.append(s)

    # Clean up tokens that might have been caught before the header
    final_tokens = [t for t in tokens if not PHRASE_SECTION_HEADER_RX.search(t)]

    # Remove duplicates while preserving order
    unique_tokens = list(dict.fromkeys(final_tokens))
    unique_phrases = list(dict.fromkeys(phrases))

    return unique_tokens, unique_phrases


def discover_class_files(in_dir: str) -> Dict[Tuple[int, str], str]:
    """
    Discovers class-specific output files from the DP pipeline.
    Returns a dictionary mapping (class_id, class_name) to file_path.
    """
    class_files = {}
    for f_name in os.listdir(in_dir):
        if f_name.startswith("class") and f_name.endswith(".txt"):
            path = os.path.join(in_dir, f_name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    match = CLASS_HEADER_RX.match(first_line)
                    if match:
                        cid = int(match.group(1))
                        cname = match.group(2).replace('-', '/') # Revert name sanitization if needed
                        class_files[(cid, cname)] = path
            except Exception as e:
                logging.warning(f"Could not parse file {path}: {e}")
    return class_files

def build_llm_messages(seed: str, label: str, ctx_terms: List[str]) -> List[Dict[str,str]]:
    """Constructs the chat messages for the LLM prompt."""
    ctx = ", ".join(ctx_terms) if ctx_terms else "N/A"
    user_content = USER_PROMPT_TEMPLATE.format(label=label, seed=seed, ctx=ctx, style=STYLE_GUIDELINES)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

def apply_chat_template_batch(messages_list: List[List[Dict[str,str]]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Applies the chat template and tokenizes a batch of prompts."""
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_list]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the LLM model and tokenizer."""
    logging.info(f"Loading LLM model and tokenizer from: {model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    logging.info("Model and tokenizer loaded.")
    return model, tok

def generate_sentences_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[List[Dict[str,str]]],
    generation_config: Dict
) -> List[str]:
    """Generates sentences from a batch of prompts using the LLM."""
    inputs = apply_chat_template_batch(prompts, tokenizer)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    num_beams = generation_config.get("num_beams", 1)

    out = model.generate(
        **inputs,
        do_sample=(num_beams == 1),
        temperature=generation_config.get("temperature", 0.6),
        top_p=generation_config.get("top_p", 0.9),
        top_k=generation_config.get("top_k", 100),
        max_new_tokens=generation_config.get("max_new_tokens", 60),
        repetition_penalty=generation_config.get("repetition_penalty", 1.05),
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    start = inputs["input_ids"].shape[1]
    texts = tokenizer.batch_decode(out[:, start:], skip_special_tokens=True)
    return [t.strip() for t in texts]

def extract_single_sentence_from_llm_output(raw_output: str) -> str:
    """
    Extracts a single, cleaned sentence from the raw LLM output.
    Looks for the "Sentence: " prefix, otherwise takes the first non-empty line.
    """
    for line in [l.strip() for l in raw_output.splitlines() if l.strip()]:
        match = SENTENCE_EXTRACTION_RX.match(line)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    first_line = next((l.strip() for l in raw_output.splitlines() if l.strip()), "")
    return re.sub(r"\s+", " ", first_line).strip()

def main():
    ap = argparse.ArgumentParser(description="LLM refinement for generating synthetic sentences from DP-generated seeds.")
    ap.add_argument("--in_dir", type=str, required=True,
                    help="Input directory containing DP output files (e.g., class{cid}_{Name}.txt).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for refined JSONL files.")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to the HuggingFace LLM model (e.g., mistralai/Mistral-7B-Instruct-v0.3).")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for LLM inference.")
    ap.add_argument("--context_size", type=int, default=8,
                    help="Number of related context terms to sample for each LLM prompt.")
    ap.add_argument("--seed_type", type=str, default="phrases", choices=["tokens", "phrases", "both"],
                    help="Which type of seeds to use for LLM prompting: 'tokens', 'phrases', or 'both'.")
    ap.add_argument("--num_outputs_per_class", type=int, default=1000,
                    help="Target number of synthetic sentences to generate per class.")
    ap.add_argument("--seed_limit_per_class", type=int, default=200,
                    help="Limit the number of DP seeds used per class (0 for no limit). Prioritizes phrases if 'both' is selected.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # LLM Generation parameters
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--num_beams", type=int, default=1)

    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    generation_cfg = {k: v for k, v in vars(args).items() if k in ["temperature", "top_p", "max_new_tokens", "repetition_penalty", "num_beams"]}

    os.makedirs(args.out_dir, exist_ok=True)
    class_files = discover_class_files(args.in_dir)
    if not class_files:
        logging.error(f"No class files found in {args.in_dir} matching the expected format (class{{cid}}_{{Name}}.txt).")
        return

    for (cid, label_name), dp_output_path in sorted(class_files.items(), key=lambda x: x[0][0]):
        logging.info(f"\nProcessing Class {cid}: {label_name} from {dp_output_path}")

        all_tokens, all_phrases = read_dp_output_file(dp_output_path)

        # 1. Define the pool of seeds to be used for prompting
        seeds_for_prompting = []
        if args.seed_type == "phrases":
            seeds_for_prompting = all_phrases
        elif args.seed_type == "tokens":
            seeds_for_prompting = all_tokens
        elif args.seed_type == "both":
            # Prioritize phrases, then add unique tokens
            seeds_for_prompting = list(dict.fromkeys(all_phrases + all_tokens))

        if not seeds_for_prompting:
            logging.warning(f"No seeds of type '{args.seed_type}' found for class {cid} ({label_name}) -- skipping.")
            continue

        if args.seed_limit_per_class > 0:
            seeds_for_prompting = seeds_for_prompting[:args.seed_limit_per_class]
        logging.info(f"Using {len(seeds_for_prompting)} unique seeds for prompting.")

        # 2. Define the pool of terms for generating context
        # Use all available high-quality terms for context
        context_pool = list(dict.fromkeys(all_tokens + all_phrases))

        # 3. Prepare the generation plan: seeds and their specific contexts
        prompts_to_generate = []
        seeds_for_generation = []
        num_prompts_needed = args.num_outputs_per_class

        if num_prompts_needed > 0:
            # Cycle through the available seeds to meet the target output count
            for i in range(num_prompts_needed):
                seed = seeds_for_prompting[i % len(seeds_for_prompting)]
                seeds_for_generation.append(seed)

                # Create seed-specific context: sample from all other terms
                relevant_context_pool = [term for term in context_pool if term != seed]
                random.shuffle(relevant_context_pool)
                ctx_terms = relevant_context_pool[:args.context_size]

                # Build the prompt message
                prompt_message = build_llm_messages(seed=seed, label=label_name, ctx_terms=ctx_terms)
                prompts_to_generate.append(prompt_message)

        if not prompts_to_generate:
            logging.warning(f"No prompts were generated for class {cid}. Check --num_outputs_per_class.")
            continue

        # 4. Generate sentences in batches
        out_file_path = os.path.join(args.out_dir, f"refined_class{cid}_{label_name.replace('/', '-')}.jsonl")
        results = []
        pbar = tqdm(range(0, len(prompts_to_generate), args.batch_size), desc=f"Generating for class {cid}")
        for b_idx in pbar:
            batch_prompts = prompts_to_generate[b_idx : b_idx + args.batch_size]
            batch_seeds = seeds_for_generation[b_idx : b_idx + args.batch_size]

            raw_llm_outputs = generate_sentences_batch(model, tokenizer, batch_prompts, generation_cfg)

            for seed, raw_output in zip(batch_seeds, raw_llm_outputs):
                sentence = extract_single_sentence_from_llm_output(raw_output)
                if 15 <= len(sentence.split()) <= 40:
                    results.append({"seed": seed, "class": label_name, "sentence": sentence})
                elif sentence:
                    logging.debug(f"Skipping sentence due to length ({len(sentence.split())} words): '{sentence}'")
        
        with open(out_file_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info(f"Successfully generated and saved {len(results)} sentences to {out_file_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a classifier on synthetic PubMed RCT (5-way sentence roles) and
evaluate on the official HuggingFace split.

Synthetic JSONL expected fields per line:
  {"seed": "...", "class": "methods", "sentence": "We randomized 120 patients ..."}

HF dataset columns:
  text (sentence), section_label (role name)
"""

import os, json, glob, argparse
from typing import List, Dict
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

import transformers
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

def set_seed(seed: int):
    import random, torch
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows

def load_synth_from_dir(train_dir: str, label_names: List[str]) -> Dataset:
    label2id = {name.lower(): i for i, name in enumerate(label_names)}
    paths = sorted(glob.glob(os.path.join(train_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"No .jsonl files found in {train_dir}")
    data = []
    skipped = 0
    for p in paths:
        for r in read_jsonl(p):
            lab = (r.get("class") or "").strip().lower()
            sent = (r.get("sentence") or "").strip()
            if not sent or lab not in label2id:
                skipped += 1; continue
            data.append({"text": sent, "label": label2id[lab]})
    if not data:
        raise SystemExit("After filtering, no usable rows from synthetic files.")
    print(f"Loaded synthetic: {len(data)} rows (skipped {skipped}).")
    return Dataset.from_list(data)

def build_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }

def make_training_args(output_dir, batch_size, lr, epochs, seed):
    try:
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=epochs,
            logging_steps=50,
            seed=seed,
            report_to="none",
            save_strategy="no",
            #evaluation_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=epochs,
            logging_steps=50,
            seed=seed,
            do_eval=True,
            save_steps=0,
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", type=str, default="armanc/pubmed-rct20k")
    ap.add_argument("--hf_config", type=str, default=None,
                    help="e.g., 'PubMed_200k_RCT' if required")
    ap.add_argument("--train_dir", type=str, required=True,
                    help="Dir with synthetic *.jsonl (fields: sentence, class)")

    ap.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny")
    ap.add_argument("--output_dir", type=str, default="runs/synth2pubmed_tinybert")
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--num_train_epochs", type=int, default=20)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--val_fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"transformers version: {transformers.__version__}")

    # Load HF dataset for label order + eval
    if args.hf_config:
        hf = load_dataset(args.hf_dataset, args.hf_config)
    else:
        hf = load_dataset(args.hf_dataset)
    col_text = "text"
    col_label = "label"
    split_test = "test" if "test" in hf else list(hf.keys())[-1]
    hf_test = hf[split_test]

    labels = sorted(list({str(x).strip().lower() for x in hf_test[col_label]}))
    id2label = {i: lab for i, lab in enumerate(labels)}
    num_labels = len(labels)
    print("HF labels:", labels)

    # Synthetic load + small val split
    synth = load_synth_from_dir(args.train_dir, labels)
    n = len(synth)
    k_val = max(1, int(args.val_fraction * n))
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    ds_train = synth.select(idx[k_val:].tolist())
    ds_val   = synth.select(idx[:k_val].tolist())

    tok = build_tokenizer(args.model_name_or_path)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds_train_tok = ds_train.map(tok_fn, batched=True).rename_column("label","labels")
    ds_val_tok   = ds_val.map(tok_fn, batched=True).rename_column("label","labels")

    # HF test -> map labels to ids
    def tok_hf(batch):
        return tok(batch[col_text], truncation=True, max_length=args.max_length)
    hf_test_tok = hf_test.map(tok_hf, batched=True)
    hf_test_tok = hf_test_tok.rename_column(col_label, "labels")
    def map_lab(ex):
        ex["labels"] = labels.index(str(ex["labels"]).lower().strip())
        return ex
    hf_test_tok = hf_test_tok.map(map_lab)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v:k for k,v in id2label.items()}
    )

    training_args = make_training_args(
        output_dir=args.output_dir,
        batch_size=min(args.per_device_train_batch_size, args.per_device_eval_batch_size),
        lr=args.learning_rate,
        epochs=args.num_train_epochs,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("\nValidation metrics:", trainer.evaluate())
    print("Test metrics:", trainer.evaluate(hf_test_tok))

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_order.txt"), "w", encoding="utf-8") as f:
        for i in range(num_labels):
            f.write(f"{i}\t{id2label[i]}\n")

if __name__ == "__main__":
    main()

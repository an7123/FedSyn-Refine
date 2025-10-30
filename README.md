# FedSyn-Refine

Implementation of FedSyn-Refine.

## Quick Start

Run the example on the PubMed dataset:

`bash run.sh`

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Hugging Face Hub

You may also need to get access to the Mistral model via a Huggingface request. 

## Repository Structure

- `seed_gen.py` — Federated synthetic data seed generation
- `llm_refine_synthetic_sentences_context.py` — LLM-based refinement with contextual processing
- `train_classifier.py` — Classifier training and evaluation
- `run.sh` — Example execution script for PubMed dataset



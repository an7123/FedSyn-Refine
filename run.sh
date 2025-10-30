#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end run for PubMed RCT 5-way sentence role classification.
# Adjust variables below as needed.

# --- Dataset Configuration (make these configurable for different datasets) ---
DATASET_NAME="pubmed"                    # Used for output directory naming
HF_DATASET="armanc/pubmed-rct20k"        # Hugging Face dataset name
HF_CONFIG=""                             # e.g., "PubMed_200k_RCT" if required (leave empty for default)
DATASET_LIMIT_PER_CLASS=10000            # Limit for DP script

# --- Model Configuration ---
ST_MODEL="all-MiniLM-L6-v2"
LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
CLASSIFIER_MODEL="prajjwal1/bert-tiny"   # Base model for training classifier

# --- DP Parameters (for the first step) ---
DP_EPSILON=10.0
DP_DELTA=1e-2
DP_CLIP_NORM=20.0 
DP_TAU=5    

# --- Output Directories ---
OUT_DP_SEEDS="dp_outputs_${DATASET_NAME}_pca"
OUT_SYNTH_SENTENCES="synth_sentences_${DATASET_NAME}"
RUN_DIR="runs/llm_refined_synth_classifier_${DATASET_NAME}"

# --- LLM Refinement Parameters (for the second step) ---
LLM_NUM_OUTPUTS_PER_CLASS=10000
LLM_BATCH_SIZE=32
LLM_CONTEXT_PACK_SIZE=8
LLM_SEED_TYPE="both" # Choose 'tokens', 'phrases', or 'both' for LLM prompting

# --- Classifier Training Parameters (for the third step) ---
TRAIN_MAX_LENGTH=128
TRAIN_NUM_EPOCHS=5
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=128

# echo "--- Starting FedSyn-Refine Pipeline for ${DATASET_NAME} ---"

#1) DP Top-K word discovery + seeds (using the refined DP script)
echo "Step 1: Running DP Seed Generation..."
python seed_gen.py \
  --dataset "${DATASET_NAME}" \
  --epsilon "${DP_EPSILON}" \
  --delta "${DP_DELTA}" \
  --clip_norm "${DP_CLIP_NORM}" \
  --tau "${DP_TAU}" \
  --st_model "${ST_MODEL}" \
  --limit_per_class "${DATASET_LIMIT_PER_CLASS}" \
  --output_dir "${OUT_DP_SEEDS}"
echo "DP Seed Generation complete. Outputs in ${OUT_DP_SEEDS}"

# 2) LLM refinement to generate synthetic sentences per role
echo "Step 2: Running LLM Refinement to Generate Synthetic Sentences..."
python llm_refine_synthetic_sentences_context.py \
  --in_dir "${OUT_DP_SEEDS}" \
  --out_dir "${OUT_SYNTH_SENTENCES}" \
  --model_path "${LLM_MODEL}" \
  --num_outputs_per_class "${LLM_NUM_OUTPUTS_PER_CLASS}" \
  --batch_size "${LLM_BATCH_SIZE}" \
  #--seed_type "${LLM_SEED_TYPE}" # Use the newly added option

    #--context_pack_size "${LLM_CONTEXT_PACK_SIZE}" \
echo "LLM Refinement complete. Synthetic sentences in ${OUT_SYNTH_SENTENCES}"

# 3) Train classifier on synthetic, evaluate on HF test
echo "Step 3: Training Classifier on Synthetic Data and Evaluating..."
python train_classifier_pubmed.py \
  --hf_dataset "${HF_DATASET}" \
  $( [ -n "${HF_CONFIG}" ] && echo --hf_config "${HF_CONFIG}" ) \
  --train_dir "${OUT_SYNTH_SENTENCES}" \
  --model_name_or_path "${CLASSIFIER_MODEL}" \
  --output_dir "${RUN_DIR}" \
  --max_length "${TRAIN_MAX_LENGTH}" \
  --num_train_epochs "${TRAIN_NUM_EPOCHS}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
echo "Classifier Training and Evaluation complete. Results in ${RUN_DIR}"

echo "--- Pipeline Finished ---"
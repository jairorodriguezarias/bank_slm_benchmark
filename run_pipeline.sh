#!/bin/bash
set -e

# =================================================
# PIPELINE CONFIGURATION (PARAMETERIZED MODELS)
# =================================================
# Change these variables to run the pipeline on a different model or dataset!

BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Example alternatives:
# BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# BASE_MODEL="HuggingFaceTB/SmolLM-1.7B-Instruct"

MODEL_PREFIX="tinyllama" # Used for folder naming (e.g., bank_expert_tinyllama_slm)
DATASET="data/train_final_5500.jsonl"
TEST_DATASET="data/train_final_5500_test.jsonl"
LIMIT=10 # Set to a small number (e.g., 10) for quick testing. Change to 0 or remove the flag for a full run.

CHECKPOINT_DIR=".checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "================================================="
echo "  Bank SLM Benchmark - Full Automated Pipeline   "
echo "================================================="
echo "Base Model: $BASE_MODEL"
echo "Checkpoints: $CHECKPOINT_DIR (delete to restart)"
echo "================================================="

MODELS_TO_BENCHMARK="$BASE_MODEL"

# STEP 1: SFT Training (General Banking)
if [ ! -f "$CHECKPOINT_DIR/step1_sft.done" ]; then
    echo -e "\n[1/6] Starting Training (SFT) for General Banking..."
    echo "This will automatically perform a 90/10 split on $DATASET"
    ./venv/bin/python src/1_sft_train.py \
        --base_model "$BASE_MODEL" \
        --data "$DATASET" \
        --output_dir "models/tuned/bank_expert_${MODEL_PREFIX}_slm" \
        --limit $LIMIT
    touch "$CHECKPOINT_DIR/step1_sft.done"
else
    echo -e "\n[1/6] Skipping Training (SFT) - Checkpoint found."
fi
MODELS_TO_BENCHMARK="$MODELS_TO_BENCHMARK models/tuned/bank_expert_${MODEL_PREFIX}_slm"

# STEP 2: DPO Tuning
if [ -f "data/dpo_dataset.jsonl" ]; then
    if [ ! -f "$CHECKPOINT_DIR/step2_dpo.done" ]; then
        echo -e "\n[2/6] Starting Preference Tuning (DPO)..."
        echo "Found DPO dataset. Running Direct Preference Optimization on the SFT model."
        ./venv/bin/python src/2_dpo_train.py \
            --base_model "models/tuned/bank_expert_${MODEL_PREFIX}_slm" \
            --data "data/dpo_dataset.jsonl" \
            --output_dir "models/tuned/bank_expert_${MODEL_PREFIX}_dpo" \
            --limit $LIMIT
        touch "$CHECKPOINT_DIR/step2_dpo.done"
    else
        echo -e "\n[2/6] Skipping Preference Tuning (DPO) - Checkpoint found."
    fi
    MODELS_TO_BENCHMARK="$MODELS_TO_BENCHMARK models/tuned/bank_expert_${MODEL_PREFIX}_dpo"
else
    echo -e "\n[2/6] Skipping DPO Tuning..."
    echo "No data/dpo_dataset.jsonl found."
fi

# STEP 3: ORPO Unified Tuning
if [ -f "data/dpo_dataset.jsonl" ]; then
    if [ ! -f "$CHECKPOINT_DIR/step3_orpo.done" ]; then
        echo -e "\n[3/6] Starting Odds Ratio Preference Optimization (ORPO)..."
        echo "Running ORPO directly on the Base Model (skipping SFT)."
        ./venv/bin/python src/3_orpo_train.py \
            --base_model "$BASE_MODEL" \
            --data "data/dpo_dataset.jsonl" \
            --output_dir "models/tuned/bank_expert_${MODEL_PREFIX}_orpo" \
            --limit $LIMIT
        touch "$CHECKPOINT_DIR/step3_orpo.done"
    else
        echo -e "\n[3/6] Skipping ORPO Tuning - Checkpoint found."
    fi
    MODELS_TO_BENCHMARK="$MODELS_TO_BENCHMARK models/tuned/bank_expert_${MODEL_PREFIX}_orpo"
else
    echo -e "\n[3/6] Skipping ORPO Tuning..."
    echo "No data/dpo_dataset.jsonl found."
fi

# STEP 4: Secondary SFT & LoRA Merging (Blockchain)
if [ -f "data/blockchain_sft_dataset.jsonl" ]; then
    if [ ! -f "$CHECKPOINT_DIR/step4_merge.done" ]; then
        echo -e "\n[4/6] Starting Multi-LoRA Merging Workflow..."
        echo "Training a secondary adapter specifically for Blockchain/MiCA..."
        ./venv/bin/python src/1_sft_train.py \
            --base_model "$BASE_MODEL" \
            --data "data/blockchain_sft_dataset.jsonl" \
            --output_dir "models/tuned/blockchain_${MODEL_PREFIX}_slm" \
            --limit $LIMIT
        
        echo "Merging General Banking adapter and Blockchain adapter..."
        ./venv/bin/python src/6_multi_lora_merge.py \
            --base_model "$BASE_MODEL" \
            --adapter1 "models/tuned/bank_expert_${MODEL_PREFIX}_slm" \
            --adapter2 "models/tuned/blockchain_${MODEL_PREFIX}_slm" \
            --output_dir "models/tuned/merged_financial_expert_${MODEL_PREFIX}" \
            --ratio 0.5
            
        touch "$CHECKPOINT_DIR/step4_merge.done"
    else
        echo -e "\n[4/6] Skipping Multi-LoRA Merging - Checkpoint found."
    fi
    MODELS_TO_BENCHMARK="$MODELS_TO_BENCHMARK models/tuned/merged_financial_expert_${MODEL_PREFIX}"
else
     echo -e "\n[4/6] Skipping Multi-LoRA Merging..."
     echo "No data/blockchain_sft_dataset.jsonl found to create a secondary adapter."
fi


# STEP 5: Benchmarking
if [ ! -f "$CHECKPOINT_DIR/step5_benchmark.done" ]; then
    echo -e "\n[5/6] Starting Benchmarking..."
    echo "Evaluating ALL generated models on the unseen test split..."
    echo "Models to benchmark: $MODELS_TO_BENCHMARK"
    ./venv/bin/python src/100_benchmark.py \
        --dataset "$TEST_DATASET" \
        --models $MODELS_TO_BENCHMARK \
        --run-name "full_pipeline_${MODEL_PREFIX}_test"
    touch "$CHECKPOINT_DIR/step5_benchmark.done"
else
    echo -e "\n[5/6] Skipping Benchmarking - Checkpoint found."
fi

# STEP 6: Evaluation Report
if [ ! -f "$CHECKPOINT_DIR/step6_evaluate.done" ]; then
    echo -e "\n[6/6] Generating Evaluation Report..."
    ./venv/bin/python src/99_evaluate.py --dataset "$TEST_DATASET"
    touch "$CHECKPOINT_DIR/step6_evaluate.done"
else
    echo -e "\n[6/6] Skipping Evaluation Report - Checkpoint found."
fi

echo -e "\n================================================="
echo "  Full Pipeline Complete! Check results/latest   "
echo "  (To restart from scratch, run: rm -rf $CHECKPOINT_DIR)"
echo "================================================="
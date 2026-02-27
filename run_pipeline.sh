#!/bin/bash
set -e

echo "==========================================="
echo "  Bank SLM Benchmark - Automated Pipeline  "
echo "==========================================="

echo -e "\n[1/4] Starting Training (SFT)..."
echo "This will automatically perform a 90/10 split on data/train_final_5500.jsonl"
./venv/bin/python src/sft_train.py \
    --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --data "data/train_final_5500.jsonl" \
    --output_dir "models/tuned/bank_expert_tinyllama_slm" \
    --limit 10

if [ -f "data/dpo_dataset.jsonl" ]; then
    echo -e "\n[2/4] Starting Preference Tuning (DPO)..."
    echo "Found DPO dataset. Running Direct Preference Optimization."
    ./venv/bin/python src/dpo_train.py \
        --base_model "models/tuned/bank_expert_tinyllama_slm" \
        --data "data/dpo_dataset.jsonl" \
        --output_dir "models/tuned/bank_expert_tinyllama_dpo" \
        --limit 10
    
    MODELS_TO_BENCHMARK="models/tuned/bank_expert_tinyllama_slm models/tuned/bank_expert_tinyllama_dpo"
else
    echo -e "\n[2/4] Skipping DPO Tuning..."
    echo "No data/dpo_dataset.jsonl found. Run 'python src/utils/generate_dpo_data.py' to generate it."
    MODELS_TO_BENCHMARK="models/tuned/bank_expert_tinyllama_slm"
fi

echo -e "\n[3/4] Starting Benchmarking..."
echo "Evaluating the model(s) on the 10% unseen test split..."
./venv/bin/python src/benchmark.py \
    --dataset "data/train_final_5500_test.jsonl" \
    --models $MODELS_TO_BENCHMARK \
    --run-name "tinyllama_expert_test"

echo -e "\n[4/4] Generating Evaluation Report..."
./venv/bin/python src/evaluate.py

echo -e "\n==========================================="
echo "  Pipeline Complete! Check results/latest  "
echo "==========================================="


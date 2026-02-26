#!/bin/bash
set -e

echo "==========================================="
echo "  Bank SLM Benchmark - Automated Pipeline  "
echo "==========================================="

echo -e "\n[1/3] Starting Training (SFT)..."
echo "This will automatically perform a 90/10 split on data/train_final_5500.jsonl"
./venv/bin/python src/train.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data "data/train_final_5500.jsonl" \
    --output_dir "models/tuned/bank_expert_slm"

echo -e "\n[2/3] Starting Benchmarking..."
echo "Evaluating the model on the 10% unseen test split..."
./venv/bin/python src/benchmark.py \
    --dataset "data/train_final_5500_test.jsonl" \
    --models "models/tuned/bank_expert_slm" \
    --run-name "expert_model_test"

echo -e "\n[3/3] Generating Evaluation Report..."
./venv/bin/python src/evaluate.py

echo -e "\n==========================================="
echo "  Pipeline Complete! Check results/latest  "
echo "==========================================="

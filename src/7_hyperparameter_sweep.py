"""
Hyperparameter Sweep for LoRA Adapters

This script tests the impact of different LoRA configurations on training time
and model performance. 

In LoRA, the "Rank" (r) determines the dimensionality of the adapter matrices.
- Low rank (e.g., r=8): Very fast to train, uses almost no memory, but might not capture complex knowledge.
- High rank (e.g., r=64 or 128): Slower to train, larger file size, but can capture deeper nuances of the data.

This script loops through a defined set of hyperparameters, trains a model for each,
and saves them to separate directories so they can be benchmarked against each other.
"""
import os
import json
import time
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def run_sweep(base_model_name, data_path, base_output_dir, limit=None):
    print(f"Starting Hyperparameter Sweep on {base_model_name}")
    print(f"Dataset: {data_path}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Dataset
    if data_path.endswith(".jsonl"):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
            
    full_dataset = Dataset.from_list(data)
    if limit is not None:
        print(f"Limiting dataset to {limit} examples for quick sweep testing.")
        full_dataset = full_dataset.select(range(min(limit, len(full_dataset))))

    # Apply formatting using tokenizer's chat template
    def format_example(example):
        if 'instruction' in example and 'output' in example:
            query, answer = example['instruction'], example['output']
        elif 'prompt' in example and 'completion' in example:
            query, answer = example['prompt'], example['completion']
        else:
            query, answer = example.get('query', ''), example.get('reference_answer', '')
            
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        
    formatted_dataset = full_dataset.map(format_example)

    # 3. Define the Sweep Configurations
    # Alpha is usually scaled with r (a common heuristic is alpha = 2 * r)
    sweep_configs = [
        {"r": 4, "alpha": 8, "dropout": 0.05},
        {"r": 16, "alpha": 32, "dropout": 0.05}, # The standard default
        {"r": 64, "alpha": 128, "dropout": 0.05} # High capacity
    ]

    # Device check
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    sweep_results = []

    # 4. Run the loop
    for i, config in enumerate(sweep_configs):
        r, alpha, dropout = config["r"], config["alpha"], config["dropout"]
        run_name = f"sweep_r{r}_alpha{alpha}"
        output_dir = os.path.join(base_output_dir, run_name)
        
        print(f"\n{'='*40}")
        print(f"Running Sweep {i+1}/{len(sweep_configs)}: {run_name}")
        print(f"Config: Rank={r}, Alpha={alpha}, Dropout={dropout}")
        print(f"{'='*40}")

        # Reload the base model fresh for each sweep to prevent memory leaks
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            trust_remote_code=True
        )

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1, # Keep to 1 epoch for sweep speed
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="no", # Don't save intermediate checkpoints to save disk space
            fp16=False,
            bf16=False,
            report_to="none",
            gradient_checkpointing=True,
            dataset_text_field="text",
            max_seq_length=256
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
        )

        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        
        print(f"Training for {run_name} finished in {train_time:.2f} seconds.")
        trainer.save_model(output_dir)
        
        sweep_results.append({
            "run_name": run_name,
            "r": r,
            "alpha": alpha,
            "train_time_seconds": round(train_time, 2)
        })
        
        # Cleanup memory before next loop
        del model
        del trainer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # 5. Output Summary
    print("\n" + "="*40)
    print(" SWEEP COMPLETE - TIMING SUMMARY ")
    print("="*40)
    for res in sweep_results:
        print(f"Rank (r): {res['r']:<4} | Alpha: {res['alpha']:<4} | Time: {res['train_time_seconds']}s")
    
    summary_path = os.path.join(base_output_dir, "sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(sweep_results, f, indent=4)
    print(f"\nSaved summary to {summary_path}")
    print("You can now pass these directories to 100_benchmark.py to see which Rank performs best!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep LoRA hyperparameters")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=str, default="data/train_final_5500.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/tuned/sweeps")
    parser.add_argument("--limit", type=int, default=100, help="Limit dataset size for fast sweeping")
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    run_sweep(args.base_model, data_path, output_dir, args.limit)

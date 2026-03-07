"""
Odds Ratio Preference Optimization (ORPO) Training Script

ORPO is a novel fine-tuning technique that combines Supervised Fine-Tuning (SFT) 
and Preference Alignment (like DPO) into a single, unified training step.

Why use ORPO instead of SFT + DPO?
1. Efficiency: It skips the need for a separate SFT phase. You go straight from the 
   base model to the final preference-aligned model.
2. Memory: Unlike DPO, ORPO does NOT require a frozen "Reference Model" to be loaded 
   in memory. It achieves alignment by modifying the odds ratio of chosen vs. rejected 
   generations directly during the adaptation phase.
3. LoRA is still used here to ensure the training fits comfortably on consumer hardware
   (Mac/Single GPU) by only training adapter weights.
"""
import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import ORPOTrainer, ORPOConfig

def setup_orpo_training_args(output_dir):
    config = ORPOConfig(
        output_dir=output_dir,
        beta=0.1,  # The odds ratio weight (similar to DPO's beta, controls preference strength)
        learning_rate=5e-5, # Standard learning rate for preference tuning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2, # Slightly more epochs than DPO since it also acts as SFT
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to="none",
        gradient_checkpointing=True,
    )
    config.max_prompt_length = 128
    config.max_length = 256
    return config

def train_orpo(base_model_name, data_path, output_dir, limit=None):
    print(f"Loading Base Model for ORPO: {base_model_name}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Preference Dataset (Chosen vs Rejected)
    print(f"Loading ORPO data from: {data_path}")
    data = []
    if data_path.endswith(".jsonl"):
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
            
    full_dataset = Dataset.from_list(data)

    if limit is not None:
        print(f"Limiting dataset to {limit} examples for quick testing")
        full_dataset = full_dataset.select(range(min(limit, len(full_dataset))))

    # Convert the raw data into Tokenizer Chat Template formats expected by ORPO
    # ORPO expects 'prompt', 'chosen', and 'rejected' columns, just like DPO
    def format_orpo(example):
        return {
            "prompt": tokenizer.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=False, add_generation_prompt=True),
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }
        
    formatted_dataset = full_dataset.map(format_orpo)

    # Split into train/test (90/10)
    dataset_dict = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    print(f"Dataset split: {len(train_dataset)} training examples, {len(test_dataset)} test examples.")

    # 3. Load Model (Notice we DO NOT load a reference model for ORPO)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        trust_remote_code=True
    )

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # 5. Setup Training Arguments & Trainer
    training_args = setup_orpo_training_args(output_dir)
    training_args.eval_strategy = "epoch"

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting ORPO Unified Training...")
    trainer.train()
    
    print(f"Saving ORPO model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note: Unlike DPO, the base model for ORPO is the RAW, pre-trained model (e.g., TinyLlama)
    # You do NOT need to run SFT first.
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=str, default="data/dpo_dataset.jsonl") 
    parser.add_argument("--output_dir", type=str, default="models/tuned/bank_expert_orpo")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset examples for quick tests")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    base_model_path = os.path.join(project_root, args.base_model) if not os.path.isabs(args.base_model) else args.base_model
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    train_orpo(base_model_path, data_path, output_dir, args.limit)

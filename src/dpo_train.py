import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

def setup_dpo_training_args(output_dir):
    config = DPOConfig(
        output_dir=output_dir,
        beta=0.1,  # Controls deviation from the reference model (lower = closer to ref)
        learning_rate=5e-5, # Slightly lower learning rate for DPO
        per_device_train_batch_size=1, # Very small batch size to fit in memory
        gradient_accumulation_steps=8,
        num_train_epochs=1, # DPO requires very few epochs
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to="none",
        gradient_checkpointing=True,
    )
    config.max_prompt_length = 128
    config.max_length = 256
    return config

def train_dpo(base_model_name, data_path, output_dir):
    print(f"Loading Base SFT Model for DPO: {base_model_name}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load DPO Preference Dataset
    print(f"Loading DPO data from: {data_path}")
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

    # Convert the raw chosen/rejected data into Tokenizer Chat Template formats
    def format_dpo(example):
        return {
            "prompt": tokenizer.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=False, add_generation_prompt=True),
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }
        
    formatted_dataset = full_dataset.map(format_dpo)

    # 3. Load Models
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We load the model we want to train
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        trust_remote_code=True
    )
    
    # We load a second copy of the exact same model to act as a "reference" 
    # The reference model ensures the training doesn't deviate too far into gibberish
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        trust_remote_code=True
    )

    # 4. LoRA Config (crucial for consumer hardware)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # Setup Training Arguments
    training_args = setup_dpo_training_args(output_dir)

    # 5. Initialize the DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO Preference Tuning...")
    trainer.train()
    
    print(f"Saving DPO model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The default base model should be the one you JUST finished SFT on
    parser.add_argument("--base_model", type=str, default="models/tuned/bank_expert_slm")
    parser.add_argument("--data", type=str, default="data/dpo_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/tuned/bank_expert_dpo")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    base_model_path = os.path.join(project_root, args.base_model) if not os.path.isabs(args.base_model) else args.base_model
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    train_dpo(base_model_path, data_path, output_dir)

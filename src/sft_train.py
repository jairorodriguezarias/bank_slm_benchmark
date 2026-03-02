import json
import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def setup_training_args(output_dir):
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        packing=False,
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="text", # Moved here
    )
    config.max_seq_length = 256
    return config

def format_instruction(sample):
    return f"User: {sample['query']}\nAssistant: {sample['reference_answer']}"

def train(base_model_name, data_path, output_dir, limit=None):
    print(f"Loading Base Model: {base_model_name}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # Load Dataset
    if data_path.endswith(".jsonl"):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
    
    # Convert to HuggingFace Dataset
    full_dataset = Dataset.from_list(data)
    
    if limit is not None:
        print(f"Limiting dataset to {limit} examples for quick testing")
        full_dataset = full_dataset.select(range(min(limit, len(full_dataset))))
    
    # Split into train/test (90/10)
    dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    print(f"Dataset split: {len(train_dataset)} training examples, {len(test_dataset)} test examples.")

    # Save the test split for benchmarking later
    if data_path.endswith(".jsonl"):
        test_file_path = data_path.replace(".jsonl", "_test.jsonl")
    else:
        test_file_path = data_path.replace(".json", "_test.json")
    with open(test_file_path, 'w') as f:
        for item in test_dataset:
            f.write(json.dumps(item) + '\n')
    print(f"Test split saved to: {test_file_path}")
    
    # Apply formatting using tokenizer's chat template
    def format_example(example):
        # Handle various schemas: instruction/output, query/reference_answer, prompt/completion
        if 'instruction' in example and 'output' in example:
            query = example['instruction']
            answer = example['output']
        elif 'prompt' in example and 'completion' in example:
            query = example['prompt']
            answer = example['completion']
        else:
            query = example.get('query', '')
            answer = example.get('reference_answer', '')
            
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
        
        # Use the model's native chat template
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": formatted_text}
        
    train_dataset = train_dataset.map(format_example)
    test_dataset = test_dataset.map(format_example)

    # Load Model
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

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # Training Arguments
    training_args = setup_training_args(output_dir)
    training_args.eval_strategy = "epoch"

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting Training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=str, default="data/train_final_5500.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/tuned/bank_expert_slm")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset examples for quick tests")
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    train(args.base_model, data_path, output_dir, args.limit)

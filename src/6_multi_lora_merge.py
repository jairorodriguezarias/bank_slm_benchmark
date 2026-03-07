"""
Multi-LoRA Merging Script

This script demonstrates how to solve the "Knowledge Interference" problem
by training separate LoRA adapters for distinct domains and merging them mathematically.

Instead of training one model on a mixed dataset (which can cause confusion between
traditional banking and blockchain regulations), this script:
1. Assumes you have trained Adapter A (e.g., Banking) and Adapter B (e.g., Blockchain).
2. Loads both adapters onto a single frozen base model.
3. Uses advanced merging math (like 'linear' or 'ties') to combine their weights.
4. Saves a new, unified "Frankenstein" adapter that knows both domains.

Why do this?
- Prevents Catastrophic Forgetting: Traditional banking knowledge isn't overwritten by new crypto terminology.
- Customizable Ratios: You can weight the merge (e.g., 70% Banking / 30% Blockchain) if one domain is more important.
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_loras(base_model_path, adapter_1_path, adapter_2_path, output_dir, ratio=0.5, merge_type="linear"):
    print(f"Loading Base Model: {base_model_path}")
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 1. Load the underlying base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
        device_map=device
    )
    
    # 2. Load the first adapter (e.g., Banking)
    print(f"\nLoading Adapter 1: {adapter_1_path} (Weight: {ratio})")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_1_path, 
        adapter_name="adapter_1"
    )

    # 3. Load the second adapter (e.g., Blockchain) alongside it
    print(f"Loading Adapter 2: {adapter_2_path} (Weight: {1.0 - ratio})")
    model.load_adapter(adapter_2_path, adapter_name="adapter_2")

    # 4. Mathematically merge the two adapters
    print(f"\nMerging adapters using '{merge_type}' method...")
    
    # weights parameter defines how much influence each adapter has
    # e.g., ratio=0.7 means [0.7, 0.3] -> 70% Adapter 1, 30% Adapter 2
    model.add_weighted_adapter(
        adapters=["adapter_1", "adapter_2"],
        weights=[ratio, 1.0 - ratio],
        adapter_name="merged_expert",
        combination_type=merge_type
    )

    # 5. Set the active adapter to our newly created one
    model.set_adapter("merged_expert")

    # 6. Save the merged adapter
    print(f"\nSaving newly merged adapter to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Also copy the tokenizer from the base model for convenience
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    print("Merge complete! You can now use this new adapter directory in benchmark.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two distinct LoRA adapters")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="The original base model used to train the adapters")
    parser.add_argument("--adapter1", type=str, required=True, help="Path to the first trained adapter (e.g., models/tuned/banking_lora)")
    parser.add_argument("--adapter2", type=str, required=True, help="Path to the second trained adapter (e.g., models/tuned/blockchain_lora)")
    parser.add_argument("--output_dir", type=str, default="models/tuned/merged_lora", help="Where to save the combined adapter")
    parser.add_argument("--ratio", type=float, default=0.5, help="Weight given to adapter1 (0.0 to 1.0). Adapter2 gets 1.0 - ratio. Default is 0.5 (50/50 split).")
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "ties", "dare_ties"], help="The mathematical method used to merge weights.")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_path = os.path.join(base_dir, args.base_model) if not os.path.isabs(args.base_model) else args.base_model
    adapter1_path = os.path.join(base_dir, args.adapter1) if not os.path.isabs(args.adapter1) else args.adapter1
    adapter2_path = os.path.join(base_dir, args.adapter2) if not os.path.isabs(args.adapter2) else args.adapter2
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    merge_loras(base_model_path, adapter1_path, adapter2_path, output_dir, args.ratio, args.method)

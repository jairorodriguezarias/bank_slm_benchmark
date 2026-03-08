"""
Doc-to-LoRA Zero-Shot Knowledge Injection

This script implements the cutting-edge "Doc-to-LoRA" (D2L) methodology developed by Sakana AI.
Instead of using gradient descent (SFT) over hours to train a model on a new document, 
D2L uses a pre-trained "Hypernetwork" to instantly predict and inject the optimal LoRA weights 
for a document directly into the model's memory in seconds.

IMPORTANT: D2L Hypernetworks are architecturally bound to specific base models. 
This script utilizes the `google/gemma-2-2b-it` architecture, as that is the primary 
open-source checkpoint provided by the researchers. It cannot be used out-of-the-box 
on TinyLlama or Qwen without retraining the entire hypernetwork.

Prerequisites:
You must clone and install the Sakana AI repository manually:
    git clone https://github.com/SakanaAI/doc-to-lora.git
    cd doc-to-lora
    pip install -e .
"""

import os
import sys
import argparse
import torch
import time
import shutil

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("\nError: 'huggingface_hub' is missing. Please run: pip install huggingface_hub")
    sys.exit(1)

try:
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    HAS_D2L = True
except ImportError:
    HAS_D2L = False

def download_checkpoint_if_needed(checkpoint_path):
    """Downloads the 5GB PyTorch checkpoint from HuggingFace if it doesn't exist locally."""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
        
    print(f"\nCheckpoint not found at {checkpoint_path}.")
    print("Initiating automatic download from HuggingFace (SakanaAI/doc-to-lora-gemma-2-2b-it)...")
    print("NOTE: This is a ~5GB file and may take several minutes depending on your connection.")
    
    dest_dir = os.path.dirname(checkpoint_path)
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="SakanaAI/doc-to-lora-gemma-2-2b-it", 
            filename="pytorch_model.bin",
            local_dir=dest_dir
        )
        print(f"Download complete: {downloaded_path}\n")
        return downloaded_path
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        sys.exit(1)

def run_d2l_injection(document_path, checkpoint_path, query):
    if not HAS_D2L:
        print("\n" + "="*60)
        print(" ERROR: Sakana AI 'ctx-to-lora' package not found!")
        print("="*60)
        print("This script requires a custom library. Please install it by running:")
        print("  1. git clone https://github.com/SakanaAI/doc-to-lora.git")
        print("  2. cd doc-to-lora")
        print("  3. pip install -e .")
        print("="*60 + "\n")
        sys.exit(1)

    # Automatically download the checkpoint if it is missing
    actual_checkpoint_path = download_checkpoint_if_needed(checkpoint_path)

    print(f"Loading Doc-to-LoRA Hypernetwork from: {actual_checkpoint_path}")
    print("This may take a moment to load the Gemma-2b base model and hypernetwork weights...")
    
    # 1. Load the Hypernetwork Model and Tokenizer
    try:
        state_dict = torch.load(actual_checkpoint_path, weights_only=False)
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict, train=False, use_sequence_packing=False
        )
        model.reset() # Ensure no residual weights are active
        tokenizer = get_tokenizer(model.base_model.name_or_path)
    except Exception as e:
        print(f"\nFailed to load checkpoint: {e}")
        print("Ensure you have downloaded the correct PyTorch bin file from Sakana AI.")
        sys.exit(1)

    # 2. Load the Raw Text Document
    if not os.path.exists(document_path):
        print(f"\nError: Document not found at {document_path}")
        sys.exit(1)
        
    print(f"\nReading document: {os.path.basename(document_path)}")
    with open(document_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # 3. Prepare the Query
    print(f"\nUser Query: '{query}'")
    chat = [{"role": "user", "content": query}]
    chat_ids = tokenizer.apply_chat_template(
        chat, 
        add_special_tokens=False, 
        return_attention_mask=False, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # 4. TEST A: INFERENCE WITH ZERO-SHOT LORA INJECTION
    print("\n" + "="*50)
    print(" TEST A: INJECTING KNOWLEDGE VIA HYPERNETWORK ")
    print("="*50)
    start_time = time.time()
    
    # The Magic Step: Instantly generates and applies LoRA weights based on the text
    model.internalize(doc_text) 
    
    injection_time = time.time() - start_time
    print(f"Knowledge internalized in {injection_time:.2f} seconds!")
    print("Generating response...")
    
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
    response_with_knowledge = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Model Response (WITH INJECTED KNOWLEDGE) ---")
    print(response_with_knowledge)

    # 5. TEST B: INFERENCE WITHOUT KNOWLEDGE (BASELINE)
    print("\n\n" + "="*50)
    print(" TEST B: BASELINE (HALLUCINATION CHECK) ")
    print("="*50)
    
    # Remove the dynamically generated LoRA adapter
    model.reset() 
    
    print("Generating response without internalized knowledge...")
    outputs_baseline = model.generate(input_ids=chat_ids, max_new_tokens=512)
    response_baseline = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
    
    print("\n--- Model Response (BASELINE/HALLUCINATING) ---")
    print(response_baseline)
    print("\n" + "="*50)
    
    # 6. CLEANUP PROMPT (As requested)
    print("\n" + "="*50)
    print(" CLEANUP ")
    print("="*50)
    user_input = input(f"Do you want to delete the ~5GB model checkpoint at '{actual_checkpoint_path}' to free up disk space? (y/N): ")
    if user_input.lower().strip() in ['y', 'yes']:
        try:
            os.remove(actual_checkpoint_path)
            print(f"Successfully deleted: {actual_checkpoint_path}")
            print("Disk space freed.")
        except Exception as e:
            print(f"Failed to delete the file: {e}")
    else:
        print("Kept the model checkpoint for future use.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot LoRA Injection via Sakana AI D2L")
    parser.add_argument("--document", type=str, default="data/raw_txt/Markets in Crypto-assets, and amending Directive .txt", help="Path to the raw text document to internalize")
    parser.add_argument("--checkpoint", type=str, default="models/d2l_weights/pytorch_model.bin", help="Path to save/load the D2L pytorch_model.bin checkpoint")
    parser.add_argument("--query", type=str, default="According to the document, what are the primary objectives of the MiCA regulation regarding crypto-asset service providers?", help="The question to ask the model")
    
    args = parser.parse_args()
    
    # Resolve project root paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_path = os.path.join(base_dir, args.document) if not os.path.isabs(args.document) else args.document
    chk_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    
    run_d2l_injection(doc_path, chk_path, args.query)
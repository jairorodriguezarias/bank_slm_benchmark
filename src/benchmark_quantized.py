import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"  # We'll test quantization on this model
QUANTIZED_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct-Int4"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bank_queries.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def run_quantized_benchmark():
    device = get_device()
    print(f"Running Quantized Benchmark on device: {device}")
    
    # Check for CUDA if using bitsandbytes 4-bit
    if device == "mps":
        print("Warning: Running 4-bit quantization on MPS (Apple Silicon).")
        print("If bitsandbytes fails, consider using GGUF/llama.cpp instead.")
    elif device == "cpu":
        print("Warning: Running 4-bit quantization on CPU. This will be slow.")

    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    print(f"\n{'='*20}\nTesting Model: {MODEL_NAME} (4-bit Quantization)\n{'='*20}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Configuring 4-bit Quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading quantized model...")
        # Note: device_map="auto" is often required for quantization to dispatch layers correctly
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto", 
            trust_remote_code=True
        )
        
        model_results = []
        
        for item in tqdm(queries, desc=f"Querying {QUANTIZED_MODEL_NAME}"):
            start_time = time.time()
            
            # Construct prompt
            if tokenizer.chat_template:
                try:
                    messages = [
                        {"role": "system", "content": "You are a helpful customer support assistant for a bank."},
                        {"role": "user", "content": item['query']}
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                        prompt = f"User: {item['query']}\nAssistant:"
            else:
                prompt = f"User: {item['query']}\nAssistant:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Token metrics
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            num_tokens_out = len(generated_tokens)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            end_time = time.time()
            inference_time = end_time - start_time
            tokens_per_second = num_tokens_out / inference_time if inference_time > 0 else 0
            
            result_entry = {
                "model": QUANTIZED_MODEL_NAME,
                "query_id": item['id'],
                "category": item['category'],
                "query": item['query'],
                "response": generated_text.strip(),
                "inference_time_seconds": round(inference_time, 4),
                "tokens_out": num_tokens_out,
                "tokens_per_second": round(tokens_per_second, 2)
            }
            model_results.append(result_entry)
        
        # Save results
        df = pd.DataFrame(model_results)
        safe_name = QUANTIZED_MODEL_NAME.replace("/", "_")
        csv_path = f"{RESULTS_PATH}/{safe_name}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error testing {QUANTIZED_MODEL_NAME}: {e}")
        print("\nPossible fix for Mac: Try using 'llama-cpp-python' for GGUF models instead of bitsandbytes.")

if __name__ == "__main__":
    run_quantized_benchmark()

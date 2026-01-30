import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen1.5-0.5B-Chat"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "banking77_short.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "banking77")
os.makedirs(RESULTS_PATH, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def run_benchmark():
    device = get_device()
    print(f"Running Banking77 Benchmark on device: {device}")
    
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        
        safe_name = model_name.replace("/", "_")
        csv_path = f"{RESULTS_PATH}/{safe_name}_results.csv"
        
        if os.path.exists(csv_path):
            print(f"Skipping {model_name} (results already exist)")
            continue
            
        try:
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device != "cpu" else torch.float32, 
                trust_remote_code=True,
                device_map=device
            )
            
            model_results = []
            
            for item in tqdm(queries, desc=f"Querying {model_name}"):
                start_time = time.time()
                
                # Simple prompt for Banking77
                prompt = f"User: {item['query']}\nAssistant:"

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        do_sample=True, 
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                num_tokens_out = len(generated_tokens)
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                tokens_per_second = num_tokens_out / inference_time if inference_time > 0 else 0
                
                model_results.append({
                    "model": model_name,
                    "query_id": item['id'],
                    "category": item['category'],
                    "query": item['query'],
                    "response": generated_text.strip(),
                    "inference_time_seconds": round(inference_time, 4),
                    "tokens_out": num_tokens_out,
                    "tokens_per_second": round(tokens_per_second, 2)
                })
            
            df = pd.DataFrame(model_results)
            df.to_csv(csv_path, index=False)
            
            # Cleanup
            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

if __name__ == "__main__":
    run_benchmark()

import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os
import gc
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen1.5-0.5B-Chat",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "facebook/opt-1.3b"
]

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

def run_benchmark():
    device = get_device()
    print(f"Running on device: {device}")
    
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    # We don't initialize all_results list here if we are merging later, 
    # but we'll keep it for the current run's results.
    current_run_results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        
        try:
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Fix for models missing pad_token
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

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=150, 
                        do_sample=True, 
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Calculate token metrics
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                num_tokens_out = len(generated_tokens)
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                tokens_per_second = num_tokens_out / inference_time if inference_time > 0 else 0
                
                result_entry = {
                    "model": model_name,
                    "query_id": item['id'],
                    "category": item['category'],
                    "query": item['query'],
                    "response": generated_text.strip(),
                    "inference_time_seconds": round(inference_time, 4),
                    "tokens_out": num_tokens_out,
                    "tokens_per_second": round(tokens_per_second, 2)
                }
                model_results.append(result_entry)
                current_run_results.append(result_entry)
            
            # Save intermediate results
            df = pd.DataFrame(model_results)
            safe_name = model_name.replace("/", "_")
            df.to_csv(f"{RESULTS_PATH}/{safe_name}_results.csv", index=False)
            
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

    # Final Aggregation from all CSVs in results folder
    print("\nAggregating all results...")
    all_dfs = []
    import glob
    csv_files = glob.glob(f"{RESULTS_PATH}/*_results.csv")
    for filename in csv_files:
        try:
            df = pd.read_csv(filename)
            all_dfs.append(df)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(f"{RESULTS_PATH}/all_models_benchmark.csv", index=False)
        print(f"\nBenchmark Complete! Combined results saved to {RESULTS_PATH}/all_models_benchmark.csv")
        print(f"Total models in report: {final_df['model'].nunique()}")
        print(f"Models: {final_df['model'].unique()}")
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    run_benchmark()

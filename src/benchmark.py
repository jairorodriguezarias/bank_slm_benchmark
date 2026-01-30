import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os
import gc
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
MODELS_TO_TEST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen1.5-0.5B-Chat",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "facebook/opt-1.3b",
    "facebook/opt-125m"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bank_queries.json")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_run_directory():
    """Creates a timestamped directory for the current run and updates history."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Update history log
    history_file = os.path.join(RESULTS_ROOT, "history.json")
    history_entry = {
        "run_id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "path": run_dir,
        "models": MODELS_TO_TEST
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    history.append(history_entry)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
        
    return run_dir, timestamp

def update_latest_link(run_dir):
    """Updates the 'latest' symlink or directory copy to point to the current run."""
    latest_dir = os.path.join(RESULTS_ROOT, "latest")
    if os.path.exists(latest_dir):
        if os.path.islink(latest_dir):
            os.unlink(latest_dir)
        else:
            shutil.rmtree(latest_dir)
            
    # Try to create a symlink, fall back to copy if not supported (e.g., some Windows envs)
    try:
        os.symlink(run_dir, latest_dir)
    except OSError:
        shutil.copytree(run_dir, latest_dir)

def run_benchmark():
    device = get_device()
    print(f"Running on device: {device}")
    
    run_dir, run_id = setup_run_directory()
    print(f"Starting Benchmark Run: {run_id}")
    print(f"Results will be saved to: {run_dir}")
    
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    current_run_results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        
        safe_name = model_name.replace("/", "_")
        # Check if results exist in this specific run (unlikely unless resumed)
        if os.path.exists(f"{run_dir}/{safe_name}_results.csv"):
            print(f"Skipping {model_name} (results already exist in this run)")
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
            
            df = pd.DataFrame(model_results)
            df.to_csv(f"{run_dir}/{safe_name}_results.csv", index=False)
            
            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    print("\nAggregating run results...")
    if current_run_results:
        final_df = pd.DataFrame(current_run_results)
        final_df.to_csv(f"{run_dir}/all_models_benchmark.csv", index=False)
        print(f"Combined results saved to {run_dir}/all_models_benchmark.csv")
        
        # Update 'latest' pointer
        update_latest_link(run_dir)
        print("Updated 'results/latest' to point to this run.")
    else:
        print("No results generated in this run.")

if __name__ == "__main__":
    run_benchmark()

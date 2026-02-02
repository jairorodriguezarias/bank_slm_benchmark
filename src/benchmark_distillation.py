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
# load_dotenv()

# Models to Compare: Standard vs Distilled
MODELS_TO_TEST = [
    "Qwen/Qwen2-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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
    return "cpu"

def run_benchmark():
    device = get_device()
    print(f"Running on device: {device}")
    
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    all_results = []

    for model_name in MODELS_TO_TEST:
        safe_name = model_name.replace("/", "_")
        csv_path = f"{RESULTS_PATH}/{safe_name}_results.csv"
        
        # We might want to re-run if the user is asking for "next" steps
        # but let's keep the skip logic for efficiency unless the file is empty
        if os.path.exists(csv_path):
            print(f"Checking existing results for {model_name}...")
            try:
                df = pd.read_csv(csv_path)
                if len(df) >= 10: # If we already have 10+ results, skip
                    print(f"Skipping {model_name} (Sufficient results already exist)")
                    all_results.extend(df.to_dict('records'))
                    continue
                else:
                    print(f"Existing results for {model_name} are insufficient ({len(df)}). Re-running...")
            except:
                print(f"Error reading {csv_path}, re-running...")

        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        
        try:
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Use float16 for GPU/MPS, float32 for CPU
            dtype = torch.float16 if device != "cpu" else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                device_map="auto" if device != "cpu" else None
            )
            
            if device == "cpu":
                model = model.to(device)
            
            model_results = []
            
            # Run 10 queries for a better benchmark
            num_queries = min(10, len(queries))
            for item in tqdm(queries[:num_queries], desc=f"Querying {model_name}"):
                start_time = time.time()
                
                # Chat Template Handling
                if tokenizer.chat_template:
                    try:
                        messages = [
                            {"role": "system", "content": "You are a helpful customer support assistant for a bank. Provide detailed and accurate answers."},
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
                        max_new_tokens=256, 
                        do_sample=True, 
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                result_entry = {
                    "model": model_name,
                    "query_id": item['id'],
                    "category": item['category'],
                    "query": item['query'],
                    "response": generated_text.strip(),
                    "inference_time_seconds": round(inference_time, 4)
                }
                model_results.append(result_entry)
                all_results.append(result_entry)
            
            # Save intermediate
            df = pd.DataFrame(model_results)
            df.to_csv(csv_path, index=False)
            
            # Cleanup memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    # Final Aggregation
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{RESULTS_PATH}/comparison_benchmark.csv", index=False)
    print(f"\nBenchmark Complete! Results saved to {RESULTS_PATH}/comparison_benchmark.csv")

if __name__ == "__main__":
    run_benchmark()

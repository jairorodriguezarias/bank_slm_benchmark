import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc

# Lightweight benchmark script
MODELS = [
    "Qwen/Qwen2-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct" 
    # Skipping DeepSeek-R1-Distill-Qwen-1.5B to ensure completion within timeout
]

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bank_queries.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

def run():
    print("Loading queries...")
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)[:2] # Only 2 queries

    results = []
    
    for model_name in MODELS:
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cpu")
            
            for item in queries:
                print(f"Querying {model_name} with ID {item['id']}")
                start = time.time()
                prompt = f"User: {item['query']}\nAssistant:"
                inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
                outputs = model.generate(**inputs, max_new_tokens=50)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response: {resp[:50]}...")
                
                results.append({
                    "model": model_name,
                    "query": item['query'],
                    "response": resp,
                    "time": time.time() - start
                })
            
            del model
            del tokenizer
            gc.collect()
        except Exception as e:
            print(f"Error: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_PATH}/manual_benchmark.csv", index=False)
    print("Saved manual_benchmark.csv")

if __name__ == "__main__":
    run()

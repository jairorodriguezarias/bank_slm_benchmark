import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import gc
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TEST = [
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
    else:
        return "cpu"

def run_benchmark():
    device = get_device()
    print(f"Running on device: {device}")
    
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        try:
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map=device)
            
            model_results = []
            for item in tqdm(queries, desc=f"Querying {model_name}"):
                start_time = time.time()
                prompt = f"User: {item['query']}\nAssistant:"
                if tokenizer.chat_template:
                    try:
                        messages = [{"role": "user", "content": item['query']}] # R1 models often take just user prompt or specific format
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except: pass

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=250, do_sample=True, temperature=0.6, pad_token_id=tokenizer.eos_token_id)
                
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                inference_time = time.time() - start_time
                
                model_results.append({
                    "model": model_name,
                    "query_id": item['id'],
                    "category": item['category'],
                    "query": item['query'],
                    "response": generated_text.strip(),
                    "inference_time_seconds": round(inference_time, 4)
                })
            
            df = pd.DataFrame(model_results)
            safe_name = model_name.replace("/", "_")
            df.to_csv(f"{RESULTS_PATH}/{safe_name}_results.csv", index=False)
            print(f"Saved {safe_name}_results.csv")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_benchmark()

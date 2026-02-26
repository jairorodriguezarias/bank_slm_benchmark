import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import os
import gc
import shutil
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models", "tuned")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_run_directory(name_prefix="run"):
    """Creates a timestamped directory for the current run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{name_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def update_latest_link(run_dir, link_name="latest"):
    """Updates a symlink to point to the current run."""
    latest_link = os.path.join(RESULTS_ROOT, link_name)
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            shutil.rmtree(latest_link)
    try:
        os.symlink(run_dir, latest_link)
    except OSError:
        shutil.copytree(run_dir, latest_link)

def load_hf_model(model_name, device, use_4bit=False):
    """Loads a standard Hugging Face model or a local PEFT adapter."""
    is_local = os.path.exists(model_name)
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer_path = model_name
    if is_local:
        config = PeftConfig.from_pretrained(model_name)
        tokenizer_path = config.base_model_name_or_path
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        print("Configuring 4-bit Quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if device != "cpu" else torch.float32,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {model_name}...")
    if is_local:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if use_4bit else device
        )
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if use_4bit else device
        )
    
    return model, tokenizer

def load_gguf_model(model_path):
    """Loads a GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
        print(f"Loading GGUF model from {model_path} with Metal/GPU support...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Use all layers on GPU if available
            n_ctx=2048,
            verbose=False
        )
        return llm
    except ImportError:
        print("Error: llama-cpp-python not installed. Cannot load GGUF models.")
        return None

def run_inference(model, tokenizer, query, device, is_gguf=False):
    """Runs a single inference and returns the response and metrics."""
    start_time = time.time()
    
    if is_gguf:
        prompt = f"""<|system|>
You are a helpful customer support assistant for a bank.</s>
<|user|>
{query}</s>
<|assistant|>
"""
        response = model(prompt, max_tokens=150, stop=["</s>"], echo=False)
        generated_text = response['choices'][0]['text'].strip()
        num_tokens_out = response['usage']['completion_tokens']
    else:
        if tokenizer.chat_template:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful customer support assistant for a bank."},
                    {"role": "user", "content": query}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"User: {query}\nAssistant:"
        else:
            prompt = f"User: {query}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device if not hasattr(model, "hf_device_map") else "next(model.parameters()).device")
        
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
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    end_time = time.time()
    latency = end_time - start_time
    tps = num_tokens_out / latency if latency > 0 else 0
    
    return generated_text, latency, num_tokens_out, tps

def main():
    parser = argparse.ArgumentParser(description='Unified SLM Benchmark')
    parser.add_argument('--models', nargs='+', help='List of Hugging Face models or local paths')
    parser.add_argument('--gguf-models', nargs='+', help='List of paths to GGUF model files')
    parser.add_argument('--dataset', type=str, default='data/bank_queries.json', help='Path to dataset JSON')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization for HF models')
    parser.add_argument('--run-name', type=str, default='benchmark', help='Prefix for the results directory')
    args = parser.parse_args()

    device = get_device()
    print(f"Starting benchmark on device: {device}")
    
    run_dir, run_id = setup_run_directory(args.run_name)
    print(f"Results will be saved to: {run_dir}")

    # Load dataset
    if args.dataset.endswith(".jsonl"):
        with open(args.dataset, 'r') as f:
            queries = [json.loads(line) for line in f if line.strip()]
    else:
        with open(args.dataset, 'r') as f:
            queries = json.load(f)

    # Standardize queries for benchmarking
    standardized_queries = []
    for i, item in enumerate(queries):
        query_text = item.get('query', item.get('prompt', item.get('instruction', '')))
        query_id = item.get('id', item.get('query_id', str(i)))
        category = item.get('category', 'general')
        
        if query_text:
            standardized_queries.append({
                "id": query_id,
                "query": query_text,
                "category": category
            })
    
    queries = standardized_queries
    hf_models = args.models or []
    # If no models specified, use a default list (matching previous behavior)
    if not hf_models and not args.gguf_models:
        hf_models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2-1.5B-Instruct",
            "HuggingFaceTB/SmolLM-1.7B-Instruct",
            "facebook/opt-1.3b",
            "mtgv/MobileLLaMA-1.4B-Base",
            "google/gemma-2b-it"
        ]
        # Auto-discover local SFT models if no specific list provided
        if os.path.exists(MODELS_ROOT):
             for name in os.listdir(MODELS_ROOT):
                 full_path = os.path.join(MODELS_ROOT, name)
                 if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "adapter_config.json")):
                     hf_models.append(full_path)

    all_run_results = []

    # Run HF Models
    for model_name in hf_models:
        print(f"\n{'='*20}\nTesting Model: {model_name}\n{'='*20}")
        try:
            model, tokenizer = load_hf_model(model_name, device, args.use_4bit)
            model_results = []
            
            display_name = os.path.basename(model_name)
            if args.use_4bit:
                display_name += "-Int4"

            for item in tqdm(queries, desc=f"Querying {display_name}"):
                resp, latency, tokens, tps = run_inference(model, tokenizer, item['query'], device)
                model_results.append({
                    "model": display_name,
                    "query_id": item['id'],
                    "category": item['category'],
                    "query": item['query'],
                    "response": resp,
                    "latency_s": round(latency, 4),
                    "tokens_out": tokens,
                    "tokens_per_second": round(tps, 2)
                })
            
            df = pd.DataFrame(model_results)
            safe_name = display_name.replace("/", "_")
            df.to_csv(f"{run_dir}/{safe_name}_results.csv", index=False)
            all_run_results.extend(model_results)
            
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

    # Run GGUF Models
    if args.gguf_models:
        for model_path in args.gguf_models:
            display_name = os.path.basename(model_path)
            print(f"\n{'='*20}\nTesting GGUF Model: {display_name}\n{'='*20}")
            try:
                llm = load_gguf_model(model_path)
                if not llm: continue
                
                model_results = []
                for item in tqdm(queries, desc=f"Querying {display_name}"):
                    resp, latency, tokens, tps = run_inference(llm, None, item['query'], device, is_gguf=True)
                    model_results.append({
                        "model": display_name,
                        "query_id": item['id'],
                        "category": item['category'],
                        "query": item['query'],
                        "response": resp,
                        "latency_s": round(latency, 4),
                        "tokens_out": tokens,
                        "tokens_per_second": round(tps, 2)
                    })
                
                df = pd.DataFrame(model_results)
                df.to_csv(f"{run_dir}/{display_name.replace('.', '_')}_results.csv", index=False)
                all_run_results.extend(model_results)
                del llm
                gc.collect()
            except Exception as e:
                print(f"Error testing GGUF {display_name}: {e}")

    if all_run_results:
        final_df = pd.DataFrame(all_run_results)
        final_df.to_csv(f"{run_dir}/all_models_benchmark.csv", index=False)
        update_latest_link(run_dir, "latest")
        print(f"\nBenchmark complete. Consolidated results in {run_dir}/all_models_benchmark.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
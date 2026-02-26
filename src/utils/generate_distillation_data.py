import json
import os
import time
import torch
import argparse
import concurrent.futures
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Google GenAI SDK (New v0.x)
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Robust path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bank_queries.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "distilled_training_data.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "data", "distilled_checkpoint.json")

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_teacher_model(model_id):
    """Loads a local model to act as the Teacher."""
    print(f"Loading local Teacher model: {model_id}...")
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "left" # Batch inference requires left padding
        
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)
        
    return model, tokenizer

def generate_local_batch(model, tokenizer, items, context=""):
    """Generates responses for a batch of queries."""
    system_prompt = "You are a banking expert. Provide a detailed reasoning followed by 'Answer: [final response]'."
    if context:
        system_prompt += f" Context: {context}"

    prompts = []
    for item in items:
        p = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nQuery: {item['query']}<|im_end|>\n<|im_start|>assistant\nThought:"
        prompts.append(p)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for i in range(len(items)):
        full_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if "Answer:" in full_text:
            parts = full_text.split("Answer:", 1)
            reasoning, answer = parts[0].strip(), parts[1].strip()
        else:
            reasoning, answer = "CoT generated", full_text.strip()
        results.append((reasoning, answer))
    return results

def generate_gemini_single(client, model_name, item, context=""):
    """Wrapper for parallel execution of Gemini calls."""
    query = item['query']
    category = item['category']
    
    full_prompt = f"""
    You are a banking expert. Answer the following query about '{category}'.
    Context: {context}
    Query: {query}
    Provide a detailed Chain-of-Thought reasoning followed by the final answer.
    """

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        temperature=0.7
    )

    try:
        response = client.models.generate_content(model=model_name, contents=contents, config=generate_content_config)
        return "Gemini Thinking Process", response.text if response.text else "No response"
    except Exception as e:
        return "Error", f"Gemini API Error: {str(e)}"

def create_distillation_dataset(source_path, provider, model_id, limit=None, context="", batch_size=4, max_workers=5):
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            distilled_data = json.load(f)
        print(f"Resuming from checkpoint with {len(distilled_data)} items.")
    elif os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            distilled_data = json.load(f)
        print(f"Loaded {len(distilled_data)} existing items from {OUTPUT_PATH}. Appending new data.")
    else:
        distilled_data = []

    print(f"Loading queries from {source_path}...")
    with open(source_path, 'r') as f:
        queries = json.load(f)
    
    if limit:
        # If the source is large (like banking77), we might want a random sample or just the head
        if len(queries) > limit:
            import random
            queries = random.sample(queries, limit)
            print(f"Randomly sampled {limit} queries from {len(queries) + limit} total.")
        else:
            queries = queries[:limit]
            print(f"Using all {len(queries)} available queries.")
    
    # Filter out already processed queries
    processed_queries = {d['instruction'] for d in distilled_data}
    remaining_queries = [q for q in queries if q.get('query', q.get('instruction')) not in processed_queries]
    
    if not remaining_queries:
        print("All queries already processed.")
        return

    print(f"Generating synthetic data for {len(remaining_queries)} remaining queries...")

    if provider == "local":
        model, tokenizer = load_teacher_model(model_id)
        for i in tqdm(range(0, len(remaining_queries), batch_size)):
            batch = remaining_queries[i:i+batch_size]
            results = generate_local_batch(model, tokenizer, batch, context)
            
            for item, (reasoning, answer) in zip(batch, results):
                distilled_data.append({
                    "instruction": item['query'],
                    "input": "",
                    "output": answer,
                    "teacher_reasoning": reasoning,
                    "model_source": f"{provider}-{model_id}",
                    "category": item['category']
                })
            
            # Save checkpoint every batch
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(distilled_data, f, indent=2)

    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(generate_gemini_single, client, model_id, q, context): q for q in remaining_queries}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_query), total=len(remaining_queries)):
                item = future_to_query[future]
                reasoning, answer = future.result()
                
                distilled_data.append({
                    "instruction": item['query'],
                    "input": "",
                    "output": answer,
                    "teacher_reasoning": reasoning,
                    "model_source": f"{provider}-{model_id}",
                    "category": item['category']
                })
                
                # Save checkpoint every 5 items for Gemini
                if len(distilled_data) % 5 == 0:
                    with open(CHECKPOINT_PATH, 'w') as f:
                        json.dump(distilled_data, f, indent=2)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(distilled_data, f, indent=2)
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        
    print(f"\nSuccess! Distillation dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/bank_queries.json", help="Source JSON file with queries")
    parser.add_argument("--provider", type=str, choices=["local", "gemini"], default="local")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=5)
    
    args = parser.parse_args()
    
    # Resolve source path
    source_path = os.path.join(PROJECT_ROOT, args.source)
    
    if args.provider == "gemini" and args.model == "Qwen/Qwen2.5-1.5B-Instruct":
        args.model = "gemini-2.0-flash-thinking-exp-01-21"
    
    create_distillation_dataset(source_path, args.provider, args.model, args.limit, args.context, args.batch_size, args.workers)
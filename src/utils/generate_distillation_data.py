import json
import os
import time
import torch
import argparse
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

def generate_local_response(model, tokenizer, query, category, context=""):
    """Generates response using local HF model."""
    system_prompt = "You are a highly experienced banking expert. Provide a detailed, step-by-step response including your internal reasoning (Chain-of-Thought)."
    if context:
        system_prompt += f" Use the following context for your answer: {context}"

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nQuery about {category}: {query}<|im_end|>\n<|im_start|>assistant\nThought:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    if "Answer:" in full_text:
        parts = full_text.split("Answer:", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
    else:
        reasoning = "Chain of Thought generated."
        answer = full_text.strip()
    
    return reasoning, answer

def generate_gemini_response(client, model_name, query, category, context=""):
    """Generates response using the new Google GenAI SDK with Thinking Config."""
    
    full_prompt = f"""
    You are a banking expert. Answer the following query about '{category}'.
    
    Context: {context}
    
    Query: {query}
    
    Please provide a detailed Chain-of-Thought reasoning followed by the final answer.
    """

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_prompt),
            ],
        ),
    ]
    
    # Configure tools and thinking
    tools = [
        types.Tool(google_search=types.GoogleSearch()) # Enable Google Search
    ]
    
    # Use thinking config
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH", 
        ),
        tools=tools,
        temperature=0.7
    )

    try:
        # Using stream=False for simple data collection
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract text and potential thoughts
        # Note: The structure of response with thinking_config might separate thoughts
        text_content = response.text if response.text else ""
        
        # Try to parse thought/answer if the model output follows it, or use the whole text
        # Since 'Thinking' is a specific feature, it might come in a separate part depending on API version.
        # For now, we treat the whole output as the content.
        
        return "Gemini Thinking Process", text_content

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Error", "Error generating response."

def create_distillation_dataset(provider, model_id, limit=None, context=""):
    print(f"Loading queries from {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)
    
    if limit:
        queries = queries[:limit]
        print(f"Limiting to first {limit} queries.")

    # Initialize Provider
    local_model, local_tokenizer = None, None
    gemini_client = None
    
    if provider == "local":
        local_model, local_tokenizer = load_teacher_model(model_id)
    elif provider == "gemini":
        if not HAS_GEMINI:
             raise ImportError("google-genai library is not installed. Run 'pip install google-genai'")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        print(f"Using Google GenAI Client with model: {model_id}")
        gemini_client = genai.Client(api_key=api_key)

    distilled_data = []
    print(f"Generating synthetic data using {provider} ({model_id})...")
    
    for item in tqdm(queries):
        if provider == "local":
            reasoning, answer = generate_local_response(local_model, local_tokenizer, item['query'], item['category'], context)
        elif provider == "gemini":
            reasoning, answer = generate_gemini_response(gemini_client, model_id, item['query'], item['category'], context)
            time.sleep(1) # Basic rate limiting
        
        training_example = {
            "instruction": item['query'],
            "input": "",
            "output": answer,
            "teacher_reasoning": reasoning,
            "model_source": f"{provider}-{model_id}",
            "category": item['category']
        }
        distilled_data.append(training_example)
        
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(distilled_data, f, indent=2)
    
    print(f"\nSuccess! Distillation dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distillation data using a Teacher model.")
    parser.add_argument("--provider", type=str, choices=["local", "gemini"], default="local", help="Inference provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--context", type=str, default="", help="Additional knowledge/context")
    
    args = parser.parse_args()
    
    # Default Gemini model
    if args.provider == "gemini" and args.model == "Qwen/Qwen2.5-1.5B-Instruct":
        args.model = "gemini-3-flash-preview"
    
    create_distillation_dataset(args.provider, args.model, args.limit, args.context)
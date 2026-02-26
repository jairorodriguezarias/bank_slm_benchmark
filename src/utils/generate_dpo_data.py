import json
import os
import argparse
import time
from tqdm import tqdm
import google.generativeai as genai

# Try to get the Gemini API key from the environment
# You need to run: export GEMINI_API_KEY="your_api_key_here" before running this script
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY environment variable not set.")
    print("Please set it before running the script: export GEMINI_API_KEY='your_api_key'")

# Use Gemini 2.5 Flash as it is fast and cheap for this kind of bulk generation
MODEL_NAME = "gemini-2.5-flash"

def generate_rejected_answer(prompt, chosen_answer, model):
    """
    Asks Gemini to generate a 'bad' or 'rejected' answer for the given prompt.
    The goal is to create contrast for DPO training.
    """
    system_instruction = """
    You are an AI tasked with generating a *poor* or *rejected* response to a user's prompt.
    The response should be subtly flawed, not completely nonsensical. 
    Examples of what makes a good 'rejected' response for a banking/finance context:
    1. Being slightly impolite or overly abrupt.
    2. Hallucinating a fake policy or regulation name.
    3. Being unnecessarily verbose but missing the core point.
    4. Refusing to answer when an answer is actually possible.
    5. Giving dangerously incorrect financial advice.
    
    You will be provided the user's prompt and the correct ('chosen') answer.
    Generate ONLY the text of the rejected answer. Do not include any explanations.
    """
    
    generation_prompt = f"""
    User Prompt: {prompt}
    Correct Answer (For Context): {chosen_answer}
    
    Generate the rejected answer now:
    """
    
    try:
        response = model.generate_content(
            generation_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=500,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating rejected answer: {e}")
        return None

def process_dataset(input_path, output_path, limit=None):
    if not api_key:
        print("Exiting because GEMINI_API_KEY is missing.")
        return
        
    print(f"Loading data from {input_path}")
    
    data = []
    if input_path.endswith(".jsonl"):
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Limiting to first {limit} examples for testing.")

    # Initialize the Gemini model with system instructions
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction="You are an AI tasked with generating a *poor* or *rejected* response to a user's prompt. The response should be subtly flawed, not completely nonsensical. Generate ONLY the text of the rejected answer. Do not include any explanations."
    )

    dpo_dataset = []
    
    print(f"Generating rejected answers to save to {output_path}...")
    
    with open(output_path, 'w') as out_f:
        for i, item in enumerate(tqdm(data)):
            # Handle different schema variations gracefully
            prompt = item.get('query', item.get('prompt', item.get('instruction', '')))
            chosen = item.get('reference_answer', item.get('completion', item.get('output', '')))
            
            if not prompt or not chosen:
                continue

            rejected = generate_rejected_answer(prompt, chosen, model)
            
            if rejected:
                dpo_example = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }
                dpo_dataset.append(dpo_example)
                
                # Write immediately so we don't lose data if it crashes
                out_f.write(json.dumps(dpo_example) + "\n")
                out_f.flush()
                
            # Sleep slightly to avoid hitting rate limits
            time.sleep(1.0)

    print(f"\nSuccessfully created DPO dataset with {len(dpo_dataset)} examples at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO preference data using Gemini")
    parser.add_argument("--input", type=str, default="data/train_final_5500.jsonl", help="Path to input JSON/JSONL file")
    parser.add_argument("--output", type=str, default="data/dpo_dataset.jsonl", help="Path to output DPO JSONL file")
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to process (default 100 for testing, use 0 for all)")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    limit = None if args.limit == 0 else args.limit
    
    process_dataset(input_path, output_path, limit)

import json
import os
from tqdm import tqdm

# This script demonstrates the process of creating a "Distilled" dataset.
# In a real scenario, you would use the 'Teacher' (DeepSeek-V3) to generate 
# high-quality Chain-of-Thought (CoT) reasoning and answers for these queries.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bank_queries.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "distilled_training_data.json")

def simulate_teacher_generation(query, category):
    """
    Simulates a Teacher model (like DeepSeek-V3) generating a response.
    In production, this would be an API call to the Teacher model.
    """
    # Simulated CoT + Answer
    reasoning = f"Thought: The user is asking about {category}. I need to provide a clear, step-by-step guide."
    answer = f"To resolve your issue regarding {category}, please follow these steps..."
    
    return {
        "reasoning": reasoning,
        "content": answer
    }

def create_distillation_dataset():
    print("Loading queries...")
    with open(DATA_PATH, 'r') as f:
        queries = json.load(f)
    
    distilled_data = []
    
    print("Generating synthetic data from Teacher (DeepSeek-V3)...")
    for item in tqdm(queries):
        # 1. Prompt the Teacher
        teacher_output = simulate_teacher_generation(item['query'], item['category'])
        
        # 2. Format for Student Training (e.g., SFT format)
        training_example = {
            "instruction": item['query'],
            "input": "",
            "output": teacher_output['content'],
            "teacher_reasoning": teacher_output['reasoning'], # Optional: for CoT distillation
            "model_source": "DeepSeek-V3"
        }
        distilled_data.append(training_example)
        
    # 3. Save Dataset
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(distilled_data, f, indent=2)
    
    print(f"Distillation dataset saved to {OUTPUT_PATH}")
    print("Next Step: Fine-tune a small model (Student) on this dataset.")

if __name__ == "__main__":
    create_distillation_dataset()

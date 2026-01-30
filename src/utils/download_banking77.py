import pandas as pd
import json
import os

# Robust path handling
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_banking77_pandas():
    print("Downloading full Banking77 dataset using pandas and direct Hugging Face links...")
    
    # Try source: the 'mteb' version of banking77 which is often cleaner
    try:
        from datasets import load_dataset
        print("Trying 'mteb/banking77'...")
        dataset = load_dataset("mteb/banking77")
        
        full_data = []
        global_id = 1
        
        for split in dataset.keys():
            print(f"Processing {split} split...")
            for item in dataset[split]:
                full_data.append({
                    "id": global_id,
                    "category": f"Banking77 Intent: {item.get('label_text', 'unknown')}",
                    "query": item['text'],
                    "reference_answer": f"The user is asking about: {item.get('label_text', 'unknown').replace('_', ' ')}."
                })
                global_id += 1
                
        output_path = os.path.join(DATA_DIR, "banking77_full.json")
        with open(output_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        print(f"Successfully saved {len(full_data)} queries to {output_path}")
        return
    except Exception as e:
        print(f"MTEB load failed: {e}")

    # Alternative: Use the 'bitext/banking-intent-2000' as a similar alternative
    print("Trying 'bitext/banking-intent-2000' as an alternative...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("bitext/banking-intent-2000")
        full_data = []
        global_id = 1
        for item in dataset['train']:
            full_data.append({
                "id": global_id,
                "category": f"Intent: {item['intent']}",
                "query": item['utterance'],
                "reference_answer": item['expansion']
            })
            global_id += 1
        output_path = os.path.join(DATA_DIR, "banking_intents_full.json")
        with open(output_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        print(f"Saved {len(full_data)} alternative banking queries to {output_path}")
    except Exception as e:
        print(f"Alternative failed: {e}")

if __name__ == "__main__":
    download_banking77_pandas()

if __name__ == "__main__":
    download_banking77_pandas()
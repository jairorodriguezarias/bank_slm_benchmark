import json
import os

def augment_line(item):
    p = item["prompt"]
    c = item["completion"]
    
    variations = []
    
    # 1. Formal Inquiry
    variations.append({
        "prompt": f"Please provide a formal explanation of the following: {p}",
        "completion": f"In accordance with the relevant regulatory framework, {c[0].lower() + c[1:]}"
    })
    
    # 2. Casual/Chat
    variations.append({
        "prompt": f"Quick question: {p.replace('What are', 'what are').replace('List', 'can you list')}",
        "completion": f"Sure! {c}"
    })
    
    # 3. Role-Play (Compliance Officer)
    variations.append({
        "prompt": f"As a compliance officer, I need to know: {p}",
        "completion": f"From a regulatory compliance standpoint, {c}"
    })
    
    # 4. Role-Play (End User)
    variations.append({
        "prompt": f"I'm a user looking for info. {p}",
        "completion": f"Based on the documents, {c}"
    })
    
    # 5. Instructional (Step-by-Step/Bulleted)
    # Simple conversion to bullet points if there are commas
    bulleted_c = c
    if "," in c:
        parts = [p.strip() for p in c.split(",")]
        bulleted_c = "The key requirements/details include:\n- " + "\n- ".join(parts)
    
    variations.append({
        "prompt": f"Can you break this down for me? {p}",
        "completion": bulleted_c
    })
    
    # 6. Summary/Concise
    variations.append({
        "prompt": f"Give me a brief summary: {p}",
        "completion": c[:150] + "..." if len(c) > 150 else c
    })
    
    # 7. Deep-Dive/Detailed
    variations.append({
        "prompt": f"I need a detailed answer regarding: {p}",
        "completion": f"To provide a comprehensive overview, {c} This is essential for understanding the regulatory impact."
    })
    
    # 8. Clarification Seeking
    variations.append({
        "prompt": f"Could you clarify this for me? {p}",
        "completion": f"Certainly. {c}"
    })
    
    # 9. Problem-Solving
    variations.append({
        "prompt": f"I'm trying to understand the regulations. {p}",
        "completion": f"To help with your understanding: {c}"
    })
    
    # 10. Technical/Developer
    variations.append({
        "prompt": f"Identify the technical parameters for: {p}",
        "completion": f"The technical and regulatory specifications are as follows: {c}"
    })
    
    return variations

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pdf_synthetic_dataset.jsonl")
    parser.add_argument("--output", default="data/pdf_synthetic_dataset_augmented.jsonl")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    with open(input_file, "r") as f:
        lines = f.readlines()

    all_variations = []
    for line in lines:
        try:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            all_variations.extend(augment_line(item))
        except json.JSONDecodeError:
            continue

    with open(output_file, "w") as f:
        for v in all_variations:
            f.write(json.dumps(v) + "\n")
            
    print(f"Augmentation complete. Total lines in augmented dataset: {len(all_variations)}")

if __name__ == "__main__":
    main()

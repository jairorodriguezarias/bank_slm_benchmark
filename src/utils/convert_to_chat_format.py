import json
import os

def convert_to_chat_format(input_path, output_path):
    print(f"Converting {input_path} to chat format...")
    converted_count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Some lines might have multiple JSON objects due to previous concatenation errors
                # We try to handle potential issues by finding all JSON objects in a line if they exist
                # But typically json.loads handles one. If there are multiple, we need to split.
                
                # Simple split and load for potential concatenated objects without newlines
                # (This is a common cause of "Extra data" errors)
                
                # Robust parsing: 
                start_idx = 0
                while start_idx < len(line):
                    try:
                        obj, end_idx = json.JSONDecoder().raw_decode(line[start_idx:])
                        
                        # Extract prompt and completion
                        prompt_text = ""
                        completion_text = ""
                        
                        if 'prompt' in obj:
                            prompt_text = obj['prompt']
                        elif 'instruction' in obj:
                            prompt_text = obj['instruction']
                        elif 'query' in obj:
                            prompt_text = obj['query']
                            
                        if 'completion' in obj:
                            completion_text = obj['completion']
                        elif 'output' in obj:
                            completion_text = obj['output']
                        elif 'reference_answer' in obj:
                            completion_text = obj['reference_answer']
                            
                        if prompt_text and completion_text:
                            new_obj = {
                                "prompt": [
                                    {"role": "user", "content": prompt_text}
                                ],
                                "completion": [
                                    {"role": "assistant", "content": completion_text}
                                ]
                            }
                            f_out.write(json.dumps(new_obj) + "\n")
                            converted_count += 1
                        
                        # Move to next object in line if any
                        next_part = line[start_idx + end_idx:].strip()
                        if next_part:
                            start_idx += end_idx + (len(line[start_idx+end_idx:]) - len(next_part))
                        else:
                            break
                    except json.JSONDecodeError:
                        break
                        
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Finished. Converted {converted_count} items to {output_path}")

if __name__ == "__main__":
    input_file = "data/train_final_5500.jsonl"
    output_file = "data/train_final_chat_format.jsonl"
    convert_to_chat_format(input_file, output_file)

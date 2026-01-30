import os
import json
import fitz  # PyMuPDF
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 2
MAX_CHUNKS_PER_FILE = 50
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
SKIP_PAGES_START = 8
SKIP_PAGES_END = 5

def get_project_root():
    """Returns the absolute path to the project root."""
    return Path(__file__).parent.parent.parent

def pdf_to_txt(pdf_path, txt_path, skip_start=SKIP_PAGES_START, skip_end=SKIP_PAGES_END):
    """Converts a PDF to a TXT file, skipping typical index and reference pages."""
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            start_page = min(skip_start, total_pages // 2)
            end_page = max(0, total_pages - skip_end)
            
            text = ""
            for page_num in range(start_page, end_page):
                text += doc[page_num].get_text()
            
            # Normalize whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return False

def get_device():
    """Detects the best available hardware acceleration."""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_local_slm(model_id):
    """Loads the SLM with appropriate precision and hardware mapping."""
    print(f"Loading SLM: {model_id}...")
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float16 for performance on GPU/MPS, float32 on CPU
    dtype = torch.float16 if device != "cpu" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def clean_text(text):
    """Aggressively cleans text to minimize token waste."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_synthetic_data_batch(model, tokenizer, chunks):
    """Generates high-quality Instruction-Output pairs for SFT."""
    prompts = [
        f"Context: {chunk}\n\nTask: Acting as a technical expert, create one complex Instruction-Output pair based ONLY on the context above. The instruction should be a specific question or task, and the output should be a comprehensive, professional answer.\n\nInstruction:"
        for chunk in chunks
    ]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    all_pairs = []
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for i in range(len(prompts)):
            output_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Flexible parsing of the model output
            if "Output:" in output_text:
                parts = output_text.split("Output:", 1)
                inst = parts[0].strip()
                out = parts[1].strip()
            else:
                lines = [l for l in output_text.split('\n') if l.strip()]
                if len(lines) >= 2:
                    inst = lines[0].replace("Instruction:", "").strip()
                    out = " ".join(lines[1:]).replace("Output:", "").strip()
                else:
                    continue
                
            if len(inst) > 15 and len(out) > 30:
                all_pairs.append({"instruction": inst, "output": out})
    except Exception as e:
        print(f"Error during generation: {e}")
    return all_pairs

def main():
    root = get_project_root()
    PDF_DIR = root / "data" / "raw_pdfs"
    TXT_DIR = root / "data" / "raw_txt"
    OUTPUT_FILE = root / "data" / "blockchain_sft_dataset.jsonl"

    os.makedirs(TXT_DIR, exist_ok=True)

    # 1. Convert PDFs to Text
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"--- Step 1: Converting {len(pdf_files)} PDFs to TXT ---")
    for pdf in tqdm(pdf_files, desc="Converting"):
        txt_name = pdf.rsplit('.', 1)[0] + ".txt"
        dest_path = TXT_DIR / txt_name
        if not dest_path.exists():
            pdf_to_txt(PDF_DIR / pdf, dest_path)

    # 2. Load Model
    model, tokenizer = load_local_slm(MODEL_ID)

    # 3. Process Text Files and Generate SFT Data
    txt_files = sorted([f for f in os.listdir(TXT_DIR) if f.endswith(".txt")])
    total_pairs = 0

    print(f"\n--- Step 2: Generating SFT Data from {len(txt_files)} files ---")
    for idx, txt_file in enumerate(txt_files):
        print(f"[{idx+1}/{len(txt_files)}] {txt_file}")
        
        chunks = []
        with open(TXT_DIR / txt_file, 'r', encoding='utf-8') as f:
            text_buffer = ""
            while len(chunks) < MAX_CHUNKS_PER_FILE:
                new_data = f.read(2048)
                if not new_data: break
                text_buffer += new_data
                
                while len(text_buffer) >= CHUNK_SIZE and len(chunks) < MAX_CHUNKS_PER_FILE:
                    raw_chunk = text_buffer[:CHUNK_SIZE]
                    chunks.append(clean_text(raw_chunk))
                    text_buffer = text_buffer[(CHUNK_SIZE - CHUNK_OVERLAP):]
            
            if text_buffer and len(chunks) < MAX_CHUNKS_PER_FILE:
                chunks.append(clean_text(text_buffer))
        
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Generating", leave=False):
            batch = chunks[i:i + BATCH_SIZE]
            pairs = generate_synthetic_data_batch(model, tokenizer, batch)
            
            if pairs:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                    for p in pairs:
                        f_out.write(json.dumps(p) + '\n')
                total_pairs += len(pairs)

    print(f"\nSuccess! Generated {total_pairs} pairs for SFT.")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
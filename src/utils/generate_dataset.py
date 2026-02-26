import os
import json
import fitz  # PyMuPDF
import re
import argparse
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# --- Configuration ---
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
CHUNK_SIZE = 3000  # Approx characters per chunk for Gemini context
CHUNK_OVERLAP = 500
MAX_WORKERS = 10    # Parallel calls to Gemini

def get_project_root():
    return Path(__file__).parent.parent.parent

def pdf_to_txt(pdf_path, txt_path):
    """Converts a PDF to a TXT file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Basic cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return False

def get_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks

def generate_pairs_from_chunk(client, model_name, chunk):
    """Uses Gemini to generate multiple prompt/completion pairs from a text chunk."""
    prompt_instruction = f"""
    Context: {chunk}    
    Task: Based ONLY on the context above, generate 5 diverse high-quality instruction-completion pairs.
    Each pair must be a JSON object with "prompt" and "completion" keys.
    The "prompt" should be a specific question or request about the context.
    The "completion" should be a detailed, professional, and accurate response.
    
    Return ONLY a JSON list of objects. No other text.
    Example:
    [
      {{"prompt": "What is the primary goal of the MiCA regulation?", "completion": "The primary goal is..."}},
      ...
    ]
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_instruction,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response
        if response.text:
            data = json.loads(response.text)
            if isinstance(data, list):
                return data
        return []
    except Exception as e:
        # print(f"Gemini API Error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from PDFs using Gemini.")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help="Gemini model ID")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--limit-pdfs", type=int, default=None, help="Limit number of PDFs to process")
    args = parser.parse_args()

    root = get_project_root()
    PDF_DIR = root / "data" / "raw_pdfs"
    TXT_DIR = root / "data" / "raw_txt"
    OUTPUT_FILE = root / "data" / "pdf_synthetic_dataset.jsonl"

    os.makedirs(TXT_DIR, exist_ok=True)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
    
    client = genai.Client(api_key=api_key)

    # 1. PDF to TXT
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    if args.limit_pdfs:
        pdf_files = pdf_files[:args.limit_pdfs]

    print(f"--- Step 1: Converting {len(pdf_files)} PDFs ---")
    all_chunks = []
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        txt_name = pdf.rsplit('.', 1)[0] + ".txt"
        txt_path = TXT_DIR / txt_name
        
        if not txt_path.exists():
            pdf_to_txt(PDF_DIR / pdf, txt_path)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            all_chunks.extend(get_chunks(text))

    print(f"\n--- Step 2: Generating Dataset from {len(all_chunks)} chunks ---")
    
    total_pairs = 0
    # Use ThreadPoolExecutor for parallel Gemini calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_pairs_from_chunk, client, args.model, chunk): chunk for chunk in all_chunks}
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_chunks), desc="Generating"):
                pairs = future.result()
                if pairs:
                    for p in pairs:
                        # Ensure keys are "prompt" and "completion" as requested
                        formatted_pair = {
                            "prompt": p.get("prompt", p.get("instruction", "")),
                            "completion": p.get("completion", p.get("output", ""))
                        }
                        if formatted_pair["prompt"] and formatted_pair["completion"]:
                            f_out.write(json.dumps(formatted_pair) + '\n')
                            total_pairs += 1

    print(f"\nSuccess! Generated {total_pairs} pairs.")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

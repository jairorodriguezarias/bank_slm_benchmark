import os
import json
import requests
from tqdm import tqdm

# Constants
BASE_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
FILES_TO_DOWNLOAD = ["train.json", "dev.json", "test.json", "private_test.json"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "finqa")

def download_file(filename, output_dir):
    """Downloads a single file from the FinQA repository."""
    url = BASE_URL + filename
    output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading {filename} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
        print(f"Successfully saved to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None

def verify_dataset(file_path):
    """Loads the JSON file to verify integrity and count records."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = len(data)
        print(f"✅ Verified {os.path.basename(file_path)}: {count} records found.")
        
        # Print a sample to ensure structure is as expected
        if count > 0:
            print(f"   Sample keys: {list(data[0].keys())}")
            
        return True
    except json.JSONDecodeError:
        print(f"❌ Error: {os.path.basename(file_path)} is not valid JSON.")
        return False
    except Exception as e:
        print(f"❌ Error verifying {os.path.basename(file_path)}: {e}")
        return False

def main():
    # Create target directory
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)
    else:
        print(f"Directory exists: {DATA_DIR}")

    # Download and verify loop
    success_count = 0
    for filename in FILES_TO_DOWNLOAD:
        file_path = download_file(filename, DATA_DIR)
        if file_path:
            if verify_dataset(file_path):
                success_count += 1
        print("-" * 50)

    print(f"\nSummary: {success_count}/{len(FILES_TO_DOWNLOAD)} files downloaded and verified.")
    print(f"Data stored in: {DATA_DIR}")

if __name__ == "__main__":
    main()

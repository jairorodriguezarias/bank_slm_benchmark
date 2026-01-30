from huggingface_hub import hf_hub_download
import os

# Robust path handling
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print(f"Downloading {filename} from {model_id}...")
hf_hub_download(
    repo_id=model_id,
    filename=filename,
    local_dir=MODELS_DIR,
    local_dir_use_symlinks=False
)
print(f"Downloaded to {os.path.join(MODELS_DIR, filename)}")

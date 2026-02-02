import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "Qwen/Qwen2-1.5B-Instruct"
device = "cpu" # Try CPU to be safe/simple

print(f"Loading {model_name} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

print("Inference...")
inputs = tokenizer("Hello, are you working?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
print("Done.")

# Bank SLM Benchmark & Distillation

A comprehensive suite for **Benchmarking**, **Distilling**, and **Fine-Tuning** Small Language Models (SLMs) specifically for the banking domain. This project unifies standard performance metrics with advanced knowledge distillation workflows and financial reasoning datasets.

## 🚀 Key Features

### 1. Automated Benchmarking
- **Metrics**: Latency, Throughput (tokens/sec), ROUGE-L, and Semantic Similarity (Cosine).
- **Mac-Optimized**: Native Metal (MPS) support for PyTorch and GGUF models via `llama-cpp-python`.
- **Visualization**: Automated PDF report generation with Accuracy vs. Latency scatter plots.

### 2. Specialized Datasets
- **Bank Queries**: 195 high-quality custom banking queries (Primary test set).
- **FinQA**: Financial numerical reasoning over earnings reports (6,000+ records). [Original Repository](https://github.com/czyssrs/FinQA).
- **Banking77**: Large-scale intent detection (13,000+ records).

### 3. Knowledge Distillation (Teacher-Student)
- **DeepSeek Integration**: Uses **DeepSeek-V3** (via API) to generate high-quality Chain-of-Thought (CoT) reasoning.
- **Data Synthesis**: Converts raw banking queries into rich training examples.
- **Comparison Pipeline**: Benchmark "Student" models (e.g., Qwen2, Phi-3) against "Distilled" variants.

### 4. Supervised Fine-Tuning (SFT) & Dataset Generation
- **PDF to SFT**: Converts technical banking PDFs into Q&A pairs.
- **LoRA Training**: Efficient fine-tuning scripts for banking domain adaptation.

---

## 📂 Project Structure

```text
├── data/
│   ├── bank_queries.json            # Primary test set (195 queries)
│   ├── finqa/                       # FinQA Dataset (numerical reasoning)
│   ├── distilled_training_data.json # Generated Teacher CoT data
│   ├── banking77_full.json          # Large-scale intent dataset
│   └── raw_txt/                     # Source texts (PDF extractions)
├── src/
│   ├── benchmark.py                 # Main unified benchmarking script
│   ├── generate_distillation_data.py # Generates training data from DeepSeek
│   ├── evaluate.py                  # Calcs metrics & generates PDF reports
│   ├── train.py                     # SFT/LoRA training script
│   └── utils/                       # Helpers (PDF, Downloads, Dataset Gen)
│       ├── download_banking77.py    # Banking77 dataset downloader
│       ├── download_finqa.py        # FinQA downloader & verifier
│       └── generate_dataset.py      # PDF to SFT pair converter
├── models/                          # Local model storage (GGUF/HF)
└── results/                         # CSV logs, plots, and PDF reports
```

---

## 🛠️ Installation

1. **Clone and Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**:
   ```bash
   # Download the FinQA dataset
   python src/utils/download_finqa.py
   
   # Download the Banking77 dataset
   python src/utils/download_banking77.py
   ```

3. **Mac Silicon Optimization (Optional but Recommended)**:
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```

4. **Environment Configuration**:
   Create a `.env` file:
   ```env
   HF_TOKEN=your_huggingface_token
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

---

## 📖 Usage Guide

### A. Data & Distillation Workflow

1. **Generate Distilled Data**:
   ```bash
   python src/generate_distillation_data.py
   ```

2. **Fine-Tune (Optional)**:
   ```bash
   python src/train.py
   ```

### B. Benchmarking

Run any model (HuggingFace or GGUF) using the unified script:

```bash
# General benchmark
python src/benchmark.py

# Benchmark specific models
python src/benchmark.py --models Qwen/Qwen2-1.5B-Instruct microsoft/Phi-3-mini-4k-instruct
```

### C. Evaluation & Reporting

```bash
python src/evaluate.py
```

### D. Dataset Generation (PDF to SFT)

To convert raw PDFs (placed in `data/raw_pdfs/`) into a high-quality SFT dataset:

```bash
python src/utils/generate_dataset.py
```
*Note: This requires a local SLM (default: Qwen2.5-0.5B) to generate synthetic Q&A pairs.*

---

## 📊 Evaluation Methodology

- **ROUGE-L**: Structural overlap with reference answers.
- **Semantic Similarity**: `all-MiniLM-L6-v2` embedding distance.
- **Latency**: End-to-end response time (ms).
- **Throughput**: Generation speed (tokens/second).

## 🖥️ Hardware Notes

- **Apple Silicon (M1/M2/M3)**: Optimized for Metal via `mps` and `llama.cpp`.
- **CUDA**: Full support for NVIDIA GPUs.
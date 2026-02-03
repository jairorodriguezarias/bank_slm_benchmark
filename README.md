# Bank SLM Benchmark & Distillation

A comprehensive suite for **Benchmarking**, **Distilling**, and **Fine-Tuning** Small Language Models (SLMs) specifically for the banking domain. This project unifies standard performance metrics with advanced knowledge distillation workflows and financial reasoning datasets.

---

## 🚀 Key Features

### 1. Unified Benchmarking Engine (`src/benchmark.py`)
- **Multi-Framework Support**: Seamlessly benchmark Hugging Face (HF) transformers and GGUF models.
- **Hardware Acceleration**: Automatic detection and utilization of **Apple Silicon Metal (MPS)**, **NVIDIA CUDA**, or CPU.
- **Quantization Support**: Test models in `float16` or `int4` (via `bitsandbytes`) to analyze the accuracy vs. performance trade-offs.
- **Batch Processing**: Automated timestamped results management with a "latest" symlink for easy access.

### 2. Specialized Financial Datasets
- **Bank Queries (Primary)**: 195 hand-crafted queries covering Cards, Accounts, Security, and Loans, each with a gold-standard reference answer.
- **FinQA**: Integration of financial numerical reasoning tasks (6,000+ records) from SEC filings.
- **Banking77**: Full-scale intent classification dataset (13,000+ queries) converted for generative tasks.

### 3. Knowledge Distillation Workflow
- **Teacher-Student Pipeline**: Use high-capacity models (DeepSeek-V3/R1) to generate high-quality Chain-of-Thought (CoT) reasoning for the banking domain.
- **Synthetic Data Generation**: Automates the creation of training sets that teach smaller models *how* to reason, not just what to answer.

### 4. Specialization Toolkit
- **PDF to SFT**: A unique utility that converts technical banking PDFs/Directives into structured Q&A pairs for Supervised Fine-Tuning.
- **Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation) scripts to train SLMs on consumer hardware in minutes.

---

## 📂 Project Structure

```text
├── data/
│   ├── bank_queries.json            # Primary test set (195 queries)
│   ├── finqa/                       # FinQA Dataset (numerical reasoning)
│   ├── distilled_training_data.json # Teacher-generated CoT training data
│   ├── banking77_full.json          # Intent detection dataset
│   ├── raw_pdfs/                    # Source technical documents
│   └── blockchain_sft_dataset.jsonl # Generated dataset for SFT
├── src/
│   ├── benchmark.py                 # Main benchmarking entry point
│   ├── evaluate.py                  # Scoring & PDF Report generation
│   ├── train.py                     # LoRA Fine-tuning script
│   └── utils/                       # Downloaders & Dataset converters
│       ├── generate_distillation_data.py # DeepSeek Teacher simulation
├── models/                          # Local storage for GGUF & Adapters
└── results/                         # Raw CSV logs, Plots, and PDF reports
```

---

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Mac Optimization (Highly Recommended)**:
   For local GGUF support with Metal acceleration:
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```

3. **Data Preparation**:
   ```bash
   python src/utils/download_finqa.py
   python src/utils/download_banking77.py
   ```

4. **API Keys**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_token
   DEEPSEEK_API_KEY=your_key
   ```

---

## 📖 Detailed Usage Guide

### Phase 1: Data Preparation & Distillation
Before training or benchmarking, you need high-quality data. This project provides two primary methods for generating synthetic training datasets:

#### 1. Knowledge Distillation (Teacher-to-Student)
This method uses a high-capacity "Teacher" model (conceptually DeepSeek-V3) to generate high-quality responses and reasoning paths for your queries.
- **Goal**: To teach a small model *how* to reason by showing it the step-by-step logic (Chain-of-Thought) of a larger model.
- **Process**: Runs through `data/bank_queries.json` and generates reasoning + final answers.
- **Output**: `data/distilled_training_data.json`

```bash
python src/utils/generate_distillation_data.py
```

#### 2. Domain Specialization (PDF-to-SFT)
Turn technical banking documents (Directives, Annexes, Whitepapers) into structured Q&A pairs.
- **Extraction**: Automatically converts PDFs in `data/raw_pdfs/` to text, skipping irrelevant pages like indexes.
- **Synthetic Generation**: A local SLM (Qwen2.5-0.5B) processes chunks of the technical text to create complex, professional instruction-output pairs.
- **Output**: `data/blockchain_sft_dataset.jsonl`

```bash
python src/utils/generate_dataset.py
```

### Phase 2: Fine-Tuning (SFT)
Train a base model on your specific banking data. Recommended SLMs for this task:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Fastest, low VRAM)
- `Qwen/Qwen2-1.5B-Instruct` (Excellent reasoning)
- `microsoft/Phi-3-mini-4k-instruct` (Strong performance)
- `HuggingFaceTB/SmolLM-1.7B-Instruct` (Highly optimized)
- `Qwen/Qwen1.5-0.5B-Chat` (Ultra-lightweight)

```bash
python src/train.py \
    --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --data "data/distilled_training_data.json" \
    --output_dir "models/tuned/bank_distilled_adapter"
```

### Phase 3: Benchmarking
Run the benchmarking suite. The standard command works across all hardware (Mac, NVIDIA GPU, or CPU):

```bash
# Universal command (runs on any hardware)
python src/benchmark.py \
    --models Qwen/Qwen2-1.5B-Instruct HuggingFaceTB/SmolLM-1.7B-Instruct \
    --run-name "comparison_test"
```

**Default Models Tested:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `Qwen/Qwen2-1.5B-Instruct`
- `HuggingFaceTB/SmolLM-1.7B-Instruct`
- `facebook/opt-1.3b`
- `mtgv/MobileLLaMA-1.4B-Base`
- `google/gemma-2b-it`

**Hardware-Specific Tips:**
- **Mac (Apple Silicon)**: Use the command above; it will automatically use `mps` (Metal). You can also benchmark local **GGUF** files by adding `--gguf-models models/your_model.gguf`.
- **NVIDIA GPU**: Add the `--use-4bit` flag to enable memory-efficient 4-bit quantization via `bitsandbytes`.


### Phase 4: Evaluation & Reporting
After benchmarking, run the evaluation to generate the visual report:
```bash
python src/evaluate.py
```
This will produce:
- **`results/latest/all_models_benchmark.csv`**: Raw metrics.
- **`results/latest/benchmark_report.pdf`**: A professional report with charts.
- **`BENCHMARK_REPORT.md`**: A quick-view summary of the latest run.

---

## 📊 Evaluation Metrics

We use a multi-dimensional scoring system to evaluate SLMs:
- **ROUGE-L**: Measures longest common subsequence between model output and reference.
- **Semantic Similarity**: Uses `all-MiniLM-L6-v2` embeddings to calculate cosine similarity (captures meaning even if words differ).
- **Latency (ms)**: Time taken to generate the first token + subsequent ones.
- **Throughput (Tokens/sec)**: Generation speed, critical for real-time customer support.

---

## 🖥️ Hardware Notes
- **Apple Silicon**: Uses `mps` for HF models and `llama-cpp` Metal for GGUF.
- **NVIDIA GPU**: Uses `cuda` with `bitsandbytes` for 4-bit optimization.
- **CPU**: Fallback mode for testing in restricted environments.

---

## 📈 Roadmap & Next Steps
1. **API Integration**: Deploy the best performing GGUF model as a FastAPI microservice.
2. **RAG Integration**: Connect the benchmarking suite to a Vector DB for Retrieval-Augmented Generation.
3. **Advanced Distillation**: Implement "Offline Distillation" where the teacher scores student outputs to provide a reward signal (DPO).

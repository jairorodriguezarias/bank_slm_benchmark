# Bank SLM Benchmark & Distillation

A comprehensive suite for **Benchmarking**, **Distilling**, and **Fine-Tuning** Small Language Models (SLMs) specifically for the banking domain. This project unifies standard performance metrics with advanced knowledge distillation workflows and financial reasoning datasets.

---

## 🚀 Key Features

### 1. Unified Benchmarking Engine (`src/benchmark.py`)
- **Multi-Format Dataset Support**: Seamlessly reads `.json` and `.jsonl` files, dynamically handling varying schema keys (e.g., `prompt`, `instruction`, `query`).
- **Multi-Framework Support**: Seamlessly benchmark Hugging Face (HF) transformers and GGUF models.
- **Hardware Acceleration**: Automatic detection and utilization of **Apple Silicon Metal (MPS)**, **NVIDIA CUDA**, or CPU.
- **Quantization Support**: Test models in `float16` or `int4` (via `bitsandbytes`) to analyze the accuracy vs. performance trade-offs.
- **Batch Processing**: Automated timestamped results management with a "latest" symlink for easy access.

### 2. End-to-End Automation
- **Pipeline Script**: Includes a `run_pipeline.sh` script for single-command execution of the entire workflow (Training -> Benchmarking -> Evaluation).

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
│   ├── banking77_full.json          # Intent detection dataset (13k queries)
│   ├── raw_pdfs/                    # Source technical documents (MiCA, Data Act)
│   ├── pdf_synthetic_dataset.jsonl  # Q&A pairs extracted from PDFs
│   ├── new_search_dataset.jsonl     # New data from Google Search (DORA, PSD3, TFR)
│   └── train_final_5500.jsonl       # Final merged dataset (5,500+ entries) for SFT
├── src/
│   ├── benchmark.py                 # Main benchmarking entry point
│   ├── evaluate.py                  # Scoring & PDF Report generation
│   ├── train.py                     # LoRA Fine-tuning script
│   └── utils/                       # Downloaders & Dataset converters
│       ├── generate_distillation_data.py # DeepSeek Teacher simulation
│       ├── generate_new_data.py          # Internet/Search data generator
│       └── augment_dataset.py            # 10x Dataset Augmenter
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
   # Download public datasets
   python src/utils/download_finqa.py
   python src/utils/download_banking77.py
   
   # Generate synthetic & augmented data
   python src/utils/generate_dataset.py       # From PDFs
   python src/utils/generate_new_data.py      # From Internet/Search
   python src/utils/augment_dataset.py        # 10x Augmentation
   ```

4. **API Keys**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_token
   GEMINI_API_KEY=your_key
   ```

---

## 📖 Detailed Usage Guide

### Phase 1: Data Preparation & Distillation
Before training or benchmarking, you need high-quality data. This project utilizes a multi-source data pipeline:

#### 1. Knowledge Distillation (Teacher-to-Student)
This method uses a high-capacity "Teacher" model (DeepSeek-V3 or Gemini) to generate high-quality responses and reasoning paths.
- **Goal**: Teach a small model *how* to reason via Chain-of-Thought (CoT).
- **Process**: Runs through `data/bank_queries.json` and generates reasoning + final answers.
- **Output**: `data/distilled_training_data.json`

```bash
python src/utils/generate_distillation_data.py
```

#### 2. Domain Specialization (PDF & Internet Search)
Turn technical banking documents and recent regulatory updates into structured Q&A pairs.
- **PDF Extraction**: Converts `data/raw_pdfs/` (MiCA, Data Act) into structured Q&A.
- **Internet Search**: Uses Google Search to fetch information on the latest regulations (DORA, PSD3, TFR).
- **Scripts**: `src/utils/generate_dataset.py` and `src/utils/generate_new_data.py`.

#### 3. Dataset Augmentation (`src/utils/augment_dataset.py`)
Expand your synthetic or search-based datasets by generating 10 variations for each query.
- **Goal**: Improve model robustness by training on different tones (Formal, Casual, Technical) and roles (Compliance Officer, End User).
- **Output**: Multiplies your dataset size by 10x (e.g., from 500 to 5,000+ entries).

```bash
# Example: Augmenting the search data
python src/utils/augment_dataset.py --input data/new_search_dataset.jsonl --output data/new_search_dataset_augmented.jsonl
```

#### 4. Final Merged Dataset
All sources are combined into a single, high-density training file:
- **Sources**: Banking77, PDF-derived data, Search-derived data, and their 10x augmentations.
- **Final File**: `data/train_final_5500.jsonl` (~5,500 entries).

### Phase 2: Fine-Tuning (SFT)
Train a base model on the final merged dataset. Recommended SLMs:
- `Qwen/Qwen2.5-0.5B-Instruct` (Excellent reasoning/efficiency)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Fastest)

**Automated Split:** The training script automatically performs a 90/10 train/test split. It trains on 90% of the data and saves the remaining 10% to `data/train_final_5500_test.jsonl` to ensure unbiased benchmarking later.

```bash
python src/train.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data "data/train_final_5500.jsonl" \
    --output_dir "models/tuned/bank_expert_slm"
```

### Automated Pipeline Execution
For the simplest end-to-end experience (Training -> Benchmarking -> Evaluation), you can use the included pipeline script. This will train the model, test it against the unseen 10% split, and generate the final PDF report:

```bash
./run_pipeline.sh
```

### Phase 3: Benchmarking (Manual)
If you prefer to run the benchmarking suite manually, use the standard command. Ensure you point it to the *test* split generated during training:

```bash
# Universal command (runs on any hardware)
python src/benchmark.py \
    --dataset "data/train_final_5500_test.jsonl" \
    --models "models/tuned/bank_expert_slm" \
    --run-name "expert_model_test"
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
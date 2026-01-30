# Bank SLM Benchmark

A comprehensive benchmarking suite for testing and specialized fine-tuning of Small Language Models (SLMs) on banking customer support tasks.

## Features

- **Automated Benchmarking**: Measures latency, throughput (tokens/sec), and response quality (ROUGE-L, Semantic Similarity).
- **Dual Dataset Support**: Focuses on a high-quality 195-query banking set (`bank_queries.json`) and supports the full 13,000+ query `Banking77` dataset for large-scale intent testing.
- **Mac-Optimized Quantization**: Support for GGUF models via `llama-cpp-python` leveraging Metal acceleration on Apple Silicon.
- **Visual Analysis**: Integrated plotting for Accuracy vs. Latency and Throughput comparisons.
- **Supervised Fine-Tuning (SFT)**: Built-in support for LoRA-based fine-tuning to specialize models for banking domain knowledge.
- **Dataset Synthesis**: Tooling to convert technical PDFs into structured instruction-output datasets for specialized training.

## Project Structure

- `src/benchmark.py`: Main execution script. Supports Hugging Face models, 4-bit quantization, and GGUF models.
- `src/evaluate.py`: Unified evaluation script. Calculates metrics (ROUGE-L, Semantic Similarity), generates plots, and updates reports.
- `src/train.py`: SFT training script using `peft` and `trl` for efficient LoRA adaptation.
- `src/utils/`: Directory for utility scripts (downloading datasets/models and generating synthetic data).

## Workflow & Usage

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For Mac users (Metal support for GGUF)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 2. Dataset Synthesis (PDF to Instruction)
Convert technical documents (PDFs) into high-quality SFT training datasets. This tool extracts text and uses a local SLM to generate complex Instruction-Output pairs.

1. Place your PDFs in `data/raw_pdfs/`.
2. Run the generator (defaults to `Qwen2.5-0.5B-Instruct` for speed):
```bash
python src/utils/generate_dataset.py
```
This script performs a 2-step process:
- **PDF to TXT**: Clean extraction skipping index/reference pages.
- **SFT Generation**: Batch-processed, hardware-accelerated generation of training pairs saved in `data/blockchain_sft_dataset.jsonl`.

### 3. Fine-Tuning a Model (SFT)
Specialize a base model for the banking or blockchain domain using the generated dataset:
```bash
python src/train.py --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output_dir "models/tuned/TinyLlama-Banking"
```

### 4. Running a Benchmark
Execute the unified benchmark on 195 high-quality banking queries:
```bash
# Standard benchmark (HF models)
python src/benchmark.py

# GGUF benchmark (Mac optimized)
python src/benchmark.py --gguf-models models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Using a specific dataset (e.g. Banking77)
python src/benchmark.py --dataset data/banking77_full.json
```

### 5. Evaluating and Visualizing Results
Calculate metrics (ROUGE-L, Semantic Similarity) and generate comparison charts:
```bash
python src/evaluate.py
```
This updates `BENCHMARK_REPORT.md` and generates a professional `benchmark_report.pdf` in the results directory.

## Analysis & Reports

The project produces a comprehensive PDF report (`benchmark_report.pdf`) and a markdown summary (`BENCHMARK_REPORT.md`) that include:
- **Performance Tables**: Comprehensive metrics (Latency, TPS, ROUGE-L, Similarity) for all tested models.
- **Visualizations**: Scatter plots for Accuracy vs. Latency and bar charts for Throughput and Quality scores.

This automated reporting ensures that stakeholders can quickly interpret which Small Language Model is best suited for specific banking use cases.

## Hardware Support
The project is optimized for:
- **macOS**: Native acceleration via Metal (MPS) for HF models and GGUF.
- **Linux/Windows**: Full CUDA support for NVIDIA GPUs.
- **CPU**: Fallback mode for environment-agnostic execution.
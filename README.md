# Bank SLM Benchmark

A comprehensive benchmarking suite for testing and specialized fine-tuning of Small Language Models (SLMs) on banking customer support tasks.

## Features

- **Automated Benchmarking**: Measures latency, throughput (tokens/sec), and response quality (ROUGE-L, Semantic Similarity).
- **Supervised Fine-Tuning (SFT)**: Built-in support for LoRA-based fine-tuning to specialize models for banking domain knowledge.
- **Result History**: Every run is timestamped and logged, allowing for comparison across different training iterations.
- **Professional Reporting**: Automatically generates visualization charts and a detailed PDF report comparing model performance and quality trade-offs.

## Project Structure

- `src/benchmark.py`: Main execution script. Discovers and tests base models and local SFT adapters.
- `src/train.py`: SFT training script using `peft` and `trl` for efficient LoRA adaptation.
- `src/evaluate_models.py`: Computes quality metrics and generates a Markdown summary.
- `src/generate_pdf_report.py`: Creates the final PDF report with charts and executive analysis.
- `results/`: Contains timestamped directories for every benchmark run.
- `models/tuned/`: Storage location for your fine-tuned SFT adapters.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running a Benchmark
Execute the benchmark on the default set of models. The script will automatically detect any models you have trained in `models/tuned/`.
```bash
python src/benchmark.py
```

### 3. Fine-Tuning a Model (SFT)
Specialize a base model for the banking dataset:
```bash
python src/train.py --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output_dir "models/tuned/TinyLlama-Banking"
```

### 4. Generating Reports
After a benchmark run, evaluate the quality and generate the PDF:
```bash
python src/evaluate_models.py
python src/generate_pdf_report.py
```
Reports are saved in `results/latest/` (a symlink to the most recent run).

## Hardware Support
The project is optimized for:
- **macOS**: Native acceleration via Metal (MPS).
- **Linux/Windows**: Full CUDA support for NVIDIA GPUs.
- **CPU**: Fallback mode for environment-agnostic execution.

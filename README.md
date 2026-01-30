# Bank SLM Benchmark Project

This project benchmarks 5 open-source Small Language Models (SLMs) on a set of banking-related customer support queries.

## Models Tested
1. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. `Qwen/Qwen2-1.5B-Instruct`
3. `Qwen/Qwen1.5-0.5B-Chat`
4. `HuggingFaceTB/SmolLM-1.7B-Instruct`
5. `facebook/opt-1.3b` (Baseline)

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Benchmark:**
   The benchmark script downloads models (cached) and runs inference.
   ```bash
   python src/benchmark.py
   ```
   *Note: This script handles model loading, inference, and result aggregation.*

3. **Generate Report:**
   Analyze the results and generate a Markdown report.
   ```bash
   python src/analyze_results.py
   ```

## Directory Structure
- `data/`: Contains `bank_queries.json` (test dataset).
- `src/`: Source code for benchmarking and analysis.
- `results/`: CSV files containing raw outputs and metrics.

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
CSV_PATH = os.path.join(RESULTS_PATH, "all_models_benchmark.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "BENCHMARK_REPORT.md")

def generate_report():
    if not os.path.exists(CSV_PATH):
        print("No results found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Calculate stats
    stats = df.groupby("model")["inference_time_seconds"].agg(['mean', 'min', 'max']).reset_index()
    stats = stats.sort_values("mean")
    
    lines = []
    lines.append("# Bank SLM Benchmark Report\n")
    lines.append("## Performance Metrics\n")
    lines.append("| Model | Average Time (s) | Min Time (s) | Max Time (s) |")
    lines.append("|---|---|---|---|")
    
    for _, row in stats.iterrows():
        lines.append(f"| {row['model']} | {row['mean']:.2f} | {row['min']:.2f} | {row['max']:.2f} |")
        
    lines.append("\n## Qualitative Analysis (Sample: 'I lost my credit card')\n")
    
    sample_query_id = 1
    sample_df = df[df['query_id'] == sample_query_id]
    
    for _, row in sample_df.iterrows():
        lines.append(f"### {row['model']}")
        resp = str(row['response']).replace('\n', '\n> ')
        lines.append(f"**Response:**\n> {resp}\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Report generated at {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
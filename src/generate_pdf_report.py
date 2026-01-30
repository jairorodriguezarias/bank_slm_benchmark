import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import argparse

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'SLM Benchmark Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

def process_data(df):
    """Aggregates raw data by model."""
    # Ensure numeric columns are actually numeric
    cols_to_avg = ['inference_time_seconds', 'tokens_per_second', 'semantic_similarity']
    for col in cols_to_avg:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by model and calculate mean
    agg_df = df.groupby('model')[cols_to_avg].mean().reset_index()
    
    # Rename columns for clarity in plots/report
    agg_df = agg_df.rename(columns={
        'model': 'Model',
        'inference_time_seconds': 'Avg Latency (s)',
        'tokens_per_second': 'Avg Tokens/Sec',
        'semantic_similarity': 'Avg Semantic Similarity'
    })
    
    return agg_df

def generate_charts(df, output_dir):
    # Set style
    plt.style.use('ggplot')
    
    chart_latency = os.path.join(output_dir, "chart_latency.png")
    chart_throughput = os.path.join(output_dir, "chart_throughput.png")
    chart_quality = os.path.join(output_dir, "chart_quality.png")
    
    # 1. Latency Chart
    plt.figure(figsize=(10, 6))
    plt.barh(df['Model'], df['Avg Latency (s)'], color='salmon')
    plt.xlabel('Average Latency (seconds)')
    plt.title('Model Latency Comparison (Lower is Better)')
    plt.tight_layout()
    plt.savefig(chart_latency)
    plt.close()

    # 2. Throughput Chart
    plt.figure(figsize=(10, 6))
    plt.barh(df['Model'], df['Avg Tokens/Sec'], color='skyblue')
    plt.xlabel('Tokens Per Second')
    plt.title('Model Throughput Comparison (Higher is Better)')
    plt.tight_layout()
    plt.savefig(chart_throughput)
    plt.close()

    # 3. Quality Chart (Semantic Similarity)
    plt.figure(figsize=(10, 6))
    plt.barh(df['Model'], df['Avg Semantic Similarity'], color='lightgreen')
    plt.xlabel('Semantic Similarity Score')
    plt.title('Response Quality Comparison (Higher is Better)')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(chart_quality)
    plt.close()

    return chart_latency, chart_throughput, chart_quality

def create_pdf(df, output_dir, charts):
    pdf_file = os.path.join(output_dir, "benchmark_report.pdf")
    chart_latency, chart_throughput, chart_quality = charts

    pdf = PDFReport()
    pdf.add_page()

    # Executive Summary
    pdf.chapter_title("1. Executive Summary")
    best_perf = df.loc[df['Avg Tokens/Sec'].idxmax()]
    best_qual = df.loc[df['Avg Semantic Similarity'].idxmax()]
    
    summary = (
        f"This report compares the performance of Small Language Models (SLMs) "
        f"for banking-related tasks. The benchmark covers latency, throughput, and semantic quality.\n\n"
        f"- Fastest Model: {best_perf['Model']} ({best_perf['Avg Tokens/Sec']:.2f} tokens/sec)\n"
        f"- Highest Quality: {best_qual['Model']} (Similarity: {best_qual['Avg Semantic Similarity']:.2f})\n"
    )
    pdf.chapter_body(summary)

    # Performance Table
    pdf.chapter_title("2. Performance Data")
    pdf.set_font('Arial', 'B', 10)
    
    # Table Header
    cols = ['Model', 'Latency (s)', 'Tokens/s', 'Similarity']
    col_widths = [80, 30, 30, 30]
    
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 10, col, 1, 0, 'C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font('Arial', '', 10)
    for _, row in df.iterrows():
        # Truncate model name if too long
        model_name = row['Model']
        if len(model_name) > 40:
            model_name = model_name[:37] + "..."
            
        pdf.cell(col_widths[0], 10, model_name, 1)
        pdf.cell(col_widths[1], 10, f"{row['Avg Latency (s)']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 10, f"{row['Avg Tokens/Sec']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[3], 10, f"{row['Avg Semantic Similarity']:.2f}", 1, 0, 'C')
        pdf.ln()
    pdf.ln(10)

    # Visualizations
    pdf.add_page()
    pdf.chapter_title("3. Visual Analysis")
    
    if os.path.exists(chart_throughput):
        pdf.image(chart_throughput, w=170)
        pdf.ln(5)
    if os.path.exists(chart_latency):
        pdf.image(chart_latency, w=170)
    
    if os.path.exists(chart_quality):
        pdf.add_page()
        pdf.image(chart_quality, w=170)

    # Output
    pdf.output(pdf_file)
    print(f"PDF Report generated successfully: {pdf_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate PDF report for SLM benchmark.')
    parser.add_argument('--run-dir', type=str, help='Path to the run directory (e.g., results/2023-...) or run ID. Defaults to results/latest.')
    args = parser.parse_args()

    results_root = os.path.join(BASE_DIR, "results")
    
    if args.run_dir:
        if os.path.isdir(args.run_dir):
            target_dir = args.run_dir
        else:
            target_dir = os.path.join(results_root, args.run_dir)
    else:
        target_dir = os.path.join(results_root, "latest")

    csv_file = os.path.join(target_dir, "all_models_benchmark.csv")

    if not os.path.exists(csv_file):
        print(f"Error: Results file not found at {csv_file}")
        return

    print(f"Loading data from: {csv_file}")
    raw_df = pd.read_csv(csv_file)
    
    print("Processing data...")
    agg_df = process_data(raw_df)
    
    print("Generating charts...")
    charts = generate_charts(agg_df, target_dir)
    
    print("Compiling PDF...")
    create_pdf(agg_df, target_dir, charts)

if __name__ == "__main__":
    main()
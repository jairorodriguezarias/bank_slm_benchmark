import pandas as pd
import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def evaluate_results(run_dir=None, dataset_path=None):
    base_results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    if run_dir:
        target_dir = run_dir if os.path.isdir(run_dir) else os.path.join(base_results_dir, run_dir)
    else:
        target_dir = os.path.join(base_results_dir, 'latest')
        
    if not os.path.exists(target_dir):
        print(f"Error: Results directory not found: {target_dir}")
        return pd.DataFrame(), None

    print(f"Evaluating results in: {target_dir}")

    # Load queries for reference
    if dataset_path:
        queries_file = dataset_path
    else:
        queries_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'bank_queries.json')
        if 'banking77' in target_dir.lower():
            queries_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'banking77_full.json')
    
    queries_data = []
    if queries_file.endswith(".jsonl"):
        with open(queries_file, 'r') as f:
            for line in f:
                if line.strip():
                    queries_data.append(json.loads(line))
    else:
        with open(queries_file, 'r') as f:
            queries_data = json.load(f)

    queries_dict = {}
    for q in queries_data:
        # Handle various schemas
        if 'instruction' in q and 'output' in q:
            query = q['instruction']
            answer = q['output']
        elif 'prompt' in q and 'completion' in q:
            query = q['prompt']
            answer = q['completion']
        else:
            query = q.get('query', '')
            answer = q.get('reference_answer', '')
        
        if query:
            queries_dict[query] = answer
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    result_files = glob.glob(os.path.join(target_dir, '*_results.csv'))
    result_files = [f for f in result_files if 'all_models_benchmark.csv' not in f]
    
    summary_stats = []
    
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('_results.csv', '').replace('_', '/')
        print(f"Scoring model: {model_name}")
        
        df = pd.read_csv(file_path)
        rouge_scores, semantic_similarities = [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Progress"):
            query, gen_ans = row['query'], str(row['response'])
            ref_ans = queries_dict.get(query, "")
            
            if not ref_ans:
                rouge_scores.append(0); semantic_similarities.append(0)
                continue
            
            rouge_l = scorer.score(ref_ans, gen_ans)['rougeL'].fmeasure
            rouge_scores.append(rouge_l)
            
            embeddings = model.encode([gen_ans, ref_ans], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_similarities.append(similarity)
        
        df['rouge_l'] = rouge_scores
        df['semantic_similarity'] = semantic_similarities
        df.to_csv(file_path, index=False)
        
        summary_stats.append({
            'Model': model_name,
            'Avg Latency (s)': df['latency_s'].mean() if 'latency_s' in df.columns else df['inference_time_seconds'].mean(),
            'Avg Tokens/Sec': df['tokens_per_second'].mean(),
            'Avg ROUGE-L': df['rouge_l'].mean(),
            'Avg Semantic Similarity': df['semantic_similarity'].mean(),
            'Total Tokens': df['tokens_out'].sum()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(target_dir, 'all_models_benchmark.csv'), index=False)
    return summary_df, target_dir

def generate_visualizations(df, target_dir):
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Accuracy vs Latency
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='Avg Latency (s)', y='Avg Semantic Similarity', hue='Model', s=100)
    plt.title('Semantic Similarity vs Latency')
    
    # Throughput
    plt.subplot(2, 2, 2)
    sns.barplot(data=df.sort_values('Avg Tokens/Sec', ascending=False), x='Avg Tokens/Sec', y='Model', hue='Model', palette='viridis', legend=False)
    plt.title('Throughput (Tokens/Sec)')

    # ROUGE-L
    plt.subplot(2, 2, 3)
    sns.barplot(data=df.sort_values('Avg ROUGE-L', ascending=False), x='Avg ROUGE-L', y='Model', hue='Model', palette='magma', legend=False)
    plt.title('ROUGE-L Score')

    # Semantic Similarity
    plt.subplot(2, 2, 4)
    sns.barplot(data=df.sort_values('Avg Semantic Similarity', ascending=False), x='Avg Semantic Similarity', y='Model', hue='Model', palette='coolwarm', legend=False)
    plt.title('Semantic Similarity Score')

    plt.tight_layout()
    plot_path = os.path.join(target_dir, 'benchmark_plots.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def generate_pdf_report(df, target_dir, plot_path):
    print("Generating PDF report...")
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 20, "Bank SLM Benchmark Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    # Date/Location
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, f"Run Directory: {os.path.basename(target_dir)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    
    # Summary Table Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    
    # Calculate column widths (Model name needs more space)
    cols = list(df.columns)
    col_widths = []
    for col in cols:
        if col == 'Model':
            col_widths.append(60)
        else:
            col_widths.append((pdf.w - 80) / (len(cols) - 1))
    
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 10, col, border=1, fill=True, align="C")
    pdf.ln()
    
    # Summary Table Data
    pdf.set_font("Helvetica", "", 9)
    for _, row in df.iterrows():
        for i, col in enumerate(cols):
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.4f}"
            pdf.cell(col_widths[i], 10, str(val), border=1, align="C")
        pdf.ln()
    
    pdf.ln(10)
    
    # Add Plot
    if os.path.exists(plot_path):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 15, "Performance Visualizations", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        # Adjust image to fit page
        pdf.image(plot_path, x=10, y=30, w=pdf.w-20)
    
    output_path = os.path.join(target_dir, 'benchmark_report.pdf')
    pdf.output(output_path)
    print(f"PDF report saved to: {output_path}")
    return output_path

def update_report_md(df, target_dir):
    report_path = os.path.join(PROJECT_ROOT, 'BENCHMARK_REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Bank SLM Benchmark Report\n\n")
        f.write(df.to_markdown(index=False))
        f.write(f"\n\n![Benchmark Plots](results/{os.path.basename(target_dir)}/benchmark_plots.png)\n")

def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation & Reporting')
    parser.add_argument('--run-dir', type=str, help='Run directory to evaluate')
    parser.add_argument('--dataset', type=str, help='Dataset path used for evaluation to load reference answers')
    args = parser.parse_args()

    summary_df, target_dir = evaluate_results(args.run_dir, args.dataset)
    if not summary_df.empty:
        plot_path = generate_visualizations(summary_df, target_dir)
        update_report_md(summary_df, target_dir)
        generate_pdf_report(summary_df, target_dir, plot_path)
        print(f"Evaluation complete. Reports updated.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main()

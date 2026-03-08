"""
Evaluation and Reporting Engine (Enhanced)

This script acts as the "Grader" for the models benchmarked by `benchmark.py`. 
It compares AI-generated answers against "Gold Standard" human reference answers.

Enhanced Features:
1. ROUGE-L & Semantic Similarity (all-MiniLM-L6-v2).
2. Pass Rate (Accuracy @ Threshold): Calculates % of answers with Similarity > 0.7.
3. Category Analysis: Breaks down performance by banking category.
4. Visual Reporting: Generates detailed charts and a PDF summary.
"""
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

# Success Threshold for Pass Rate
PASS_THRESHOLD = 0.7

def evaluate_results(run_dir=None, dataset_path=None):
    base_results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    if run_dir:
        target_dir = run_dir if os.path.isdir(run_dir) else os.path.join(base_results_dir, run_dir)
    else:
        target_dir = os.path.join(base_results_dir, 'latest')
        
    if not os.path.exists(target_dir):
        print(f"Error: Results directory not found: {target_dir}")
        return pd.DataFrame(), None, {{}}

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

    queries_dict = {{}}
    for q in queries_data:
        # Handle various schemas
        if 'instruction' in q and 'output' in q:
            query, answer = q['instruction'], q['output']
        elif 'prompt' in q and 'completion' in q:
            query, answer = q['prompt'], q['completion']
        else:
            query, answer = q.get('query', ''), q.get('reference_answer', '')
        
        if query:
            queries_dict[query] = answer
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    result_files = glob.glob(os.path.join(target_dir, '*_results.csv'))
    result_files = [f for f in result_files if 'all_models_benchmark.csv' not in f]
    
    summary_stats = []
    category_data = [] # To store category level info for plotting
    
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
            
            # 1. ROUGE
            rouge_l = scorer.score(ref_ans, gen_ans)['rougeL'].fmeasure
            rouge_scores.append(rouge_l)
            
            # 2. Semantic Similarity
            embeddings = model.encode([gen_ans, ref_ans], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_similarities.append(similarity)
        
        df['rouge_l'] = rouge_scores
        df['semantic_similarity'] = semantic_similarities
        df['is_pass'] = df['semantic_similarity'] >= PASS_THRESHOLD
        
        # Save enriched CSV
        df.to_csv(file_path, index=False)
        
        # Aggregate Summary
        pass_rate = (df['is_pass'].sum() / len(df)) * 100
        
        summary_stats.append({
            'Model': model_name,
            'Avg Latency (s)': df['latency_s'].mean() if 'latency_s' in df.columns else 0,
            'Avg Tokens/Sec': df['tokens_per_second'].mean() if 'tokens_per_second' in df.columns else 0,
            'Pass Rate (%)': round(pass_rate, 2),
            'Avg ROUGE-L': df['rouge_l'].mean(),
            'Avg Similarity': df['semantic_similarity'].mean(),
        })
        
        # Store category stats for this model
        if 'category' in df.columns:
            cat_stats = df.groupby('category')['semantic_similarity'].mean().reset_index()
            cat_stats['Model'] = model_name
            category_data.append(cat_stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(target_dir, 'all_models_benchmark.csv'), index=False)
    
    category_df = pd.concat(category_data) if category_data else pd.DataFrame()
    
    return summary_df, target_dir, category_df

def generate_enhanced_visualizations(df, target_dir, cat_df):
    print("Generating enhanced visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Summary Overview
    plt.figure(figsize=(16, 12))
    
    # 1. Similarity vs Latency (The Quality/Speed Tradeoff)
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='Avg Latency (s)', y='Avg Similarity', hue='Model', s=150)
    plt.title('Semantic Quality vs. Latency')
    
    # 2. Pass Rate (The reliability metric)
    plt.subplot(2, 2, 2)
    sns.barplot(data=df.sort_values('Pass Rate (%)', ascending=False), x='Pass Rate (%)', y='Model', palette='Greens_d')
    plt.title(f'Pass Rate (% Similarity > {PASS_THRESHOLD})')

    # 3. Throughput
    plt.subplot(2, 2, 3)
    sns.barplot(data=df.sort_values('Avg Tokens/Sec', ascending=False), x='Avg Tokens/Sec', y='Model', palette='Blues_d')
    plt.title('Throughput (Tokens/Sec)')

    # 4. Category Breakdown
    if not cat_df.empty:
        plt.subplot(2, 2, 4)
        sns.heatmap(cat_df.pivot(index="category", columns="Model", values="semantic_similarity"), annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title('Avg Similarity by Category')

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
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 51, 102) # Dark blue
    pdf.cell(0, 20, "Bank SLM: Professional Benchmark Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    # Subtitle
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Run Directory: {os.path.basename(target_dir)} | Success Threshold: {PASS_THRESHOLD}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    
    # Summary Table Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 241)
    pdf.set_text_color(0, 0, 0)
    
    cols = list(df.columns)
    col_widths = [65, 30, 30, 25, 25, 25] # Hardcoded widths for 6 columns
    
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 10, col, border=1, fill=True, align="C")
    pdf.ln()
    
    # Summary Table Data
    pdf.set_font("Helvetica", "", 9)
    for _, row in df.iterrows():
        for i, col in enumerate(cols):
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            pdf.cell(col_widths[i], 10, str(val), border=1, align="C")
        pdf.ln()
    
    pdf.ln(10)
    
    # Charts Page
    if os.path.exists(plot_path):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 15, "Multi-Dimensional Performance Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.image(plot_path, x=10, y=30, w=pdf.w-20)
    
    output_path = os.path.join(target_dir, 'benchmark_report.pdf')
    pdf.output(output_path)
    print(f"PDF report saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Enhanced SLM Evaluation')
    parser.add_argument('--run-dir', type=str, help='Run directory to evaluate')
    parser.add_argument('--dataset', type=str, help='Dataset path used for evaluation')
    args = parser.parse_args()

    summary_df, target_dir, category_df = evaluate_results(args.run_dir, args.dataset)
    
    if not summary_df.empty:
        plot_path = generate_enhanced_visualizations(summary_df, target_dir, category_df)
        generate_pdf_report(summary_df, target_dir, plot_path)
        
        # Update Markdown
        report_md_path = os.path.join(os.path.dirname(target_dir), '..', 'BENCHMARK_REPORT.md')
        with open(report_md_path, 'w') as f:
            f.write("# Bank SLM Evaluation Summary\n\n")
            f.write("### Global Performance Table\n")
            f.write(summary_df.to_markdown(index=False))
            f.write(f"\n\n![Performance Plots](results/{os.path.basename(target_dir)}/benchmark_plots.png)\n")
        
        print(f"Evaluation complete. Enhanced reports generated in {target_dir}")

if __name__ == "__main__":
    main()
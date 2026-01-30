import pandas as pd
import json
import os
import glob
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def evaluate_results():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    queries_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'bank_queries.json')
    
    # Load queries
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)
    
    queries_dict = {q['query']: q['reference_answer'] for q in queries_data}
    
    # Initialize metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Using a small, fast model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Find all model result CSVs (excluding the summary one)
    result_files = glob.glob(os.path.join(results_dir, '*_results.csv'))
    
    summary_stats = []
    
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('_results.csv', '').replace('_', '/')
        print(f"Evaluating model: {model_name}")
        
        df = pd.read_csv(file_path)
        
        rouge_scores = []
        semantic_similarities = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {model_name}"):
            query = row['query']
            generated_answer = str(row['response'])
            reference_answer = queries_dict.get(query, "")
            
            if not reference_answer:
                rouge_scores.append(0)
                semantic_similarities.append(0)
                continue
            
            # ROUGE-L
            rouge_l = scorer.score(reference_answer, generated_answer)['rougeL'].fmeasure
            rouge_scores.append(rouge_l)
            
            # Semantic Similarity
            embeddings = model.encode([generated_answer, reference_answer], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_similarities.append(similarity)
        
        df['rouge_l'] = rouge_scores
        df['semantic_similarity'] = semantic_similarities
        
        # Save updated CSV
        df.to_csv(file_path, index=False)
        
        # Collect summary stats
        summary_stats.append({
            'Model': model_name,
            'Avg Latency (s)': df['latency_s'].mean() if 'latency_s' in df.columns else df['inference_time_seconds'].mean(),
            'Avg Tokens/Sec': df['tokens_per_second'].mean() if 'tokens_per_second' in df.columns else 0,
            'Avg ROUGE-L': df['rouge_l'].mean(),
            'Avg Semantic Similarity': df['semantic_similarity'].mean(),
            'Total Tokens': df['tokens_out'].sum() if 'tokens_out' in df.columns else 0
        })
    
    # Create consolidated summary
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(results_dir, 'all_models_benchmark.csv'), index=False)
    
    print("\nEvaluation complete. Summary updated in results/all_models_benchmark.csv")
    return summary_df

def update_report(summary_df):
    report_path = os.path.join(os.path.dirname(__file__), '..', 'BENCHMARK_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Bank SLM Benchmark Report\n\n")
        f.write("## Overview\n")
        f.write("This report compares the performance of various Small Language Models (SLMs) on banking customer support queries.\n\n")
        f.write("## Performance Metrics\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n")
        
        # Simple logic to find the best model
        if not summary_df.empty:
            best_model = summary_df.loc[summary_df['Avg Semantic Similarity'].idxmax()]['Model']
            f.write(f"- **Best Performing Model (Semantic):** {best_model}\n")
            
            fastest_model = summary_df.loc[summary_df['Avg Latency (s)'].idxmin()]['Model']
            f.write(f"- **Fastest Model:** {fastest_model}\n")

if __name__ == "__main__":
    stats = evaluate_results()
    update_report(stats)

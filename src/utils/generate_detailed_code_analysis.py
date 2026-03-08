import os
import datetime

OUTPUT_FILE = "PHD_DETAILED_CODE_ANALYSIS.md"

ANALYSIS_DATA = {
    "1_sft_train.py": {
        "title": "Supervised Fine-Tuning (SFT) Engine",
        "implementation": "Implements instruction tuning using Low-Rank Adaptation (LoRA) via the `peft` and `trl` libraries. It targets the Query (`q_proj`) and Value (`v_proj`) projection matrices within the attention mechanism. Crucially, it employs automated dataset tokenization using native chat templates (`apply_chat_template`) and enforces a rigid 90/10 zero-leakage data split prior to gradient updates.",
        "alternatives_rejected": "1. **Full Fine-Tuning (FFT)**: Rejected due to catastrophic forgetting and hardware constraints. FFT alters all weights, risking the destruction of the base model's foundational linguistic syntax. LoRA freezes base weights, acting as an additive knowledge patch.\n2. **Prefix-Tuning / Prompt Tuning**: Rejected because while memory-efficient, they restrict the model's sequence length (taking up context window space) and generally underperform LoRA in tasks requiring complex domain reasoning.",
        "thesis_significance": "Establishes the foundational baseline for domain adaptation. By explicitly isolating a 10% test set programmatically before training, it guarantees the integrity of all subsequent evaluation metrics."
    },
    "2_dpo_train.py": {
        "title": "Direct Preference Optimization (DPO) Aligner",
        "implementation": "Utilizes `trl.DPOTrainer` to perform alignment. It takes an SFT model as the reference policy and a dataset of paired chosen/rejected responses. It uses the DPO loss function to increase the relative log-probability of preferred responses over rejected ones, controlled by a hyperparameter $\beta$ (beta) which penalizes deviation from the reference model.",
        "alternatives_rejected": "1. **Reinforcement Learning from Human Feedback (RLHF) / PPO**: Rejected due to extreme architectural complexity. PPO requires maintaining four distinct models in memory (Actor, Critic, Reference, Reward), which is mathematically unstable and infeasible for local SLM adaptation pipelines. DPO solves the same objective function purely through supervised learning paradigms.",
        "thesis_significance": "Demonstrates the mathematical transition from merely injecting knowledge (SFT) to actively shaping the model's conversational behavior and safety rails, a critical requirement for financial deployments."
    },
    "3_orpo_train.py": {
        "title": "Odds Ratio Preference Optimization (ORPO) Implementation",
        "implementation": "Implements an experimental, single-stage alignment algorithm (`ORPOTrainer`). Instead of doing SFT followed by DPO, ORPO applies an odds-ratio penalty to the negative log-likelihood loss during standard fine-tuning.",
        "alternatives_rejected": "1. **Two-Stage SFT+DPO as the ONLY method**: While SFT+DPO is the industry standard, relying solely on it prevents comparative analysis. ORPO was implemented to test the hypothesis that a single-stage process can reduce compute hours by 50% without a statistically significant drop in semantic accuracy.",
        "thesis_significance": "Provides a comparative ablation study within the thesis, evaluating the trade-offs between computational efficiency (ORPO) and alignment precision (SFT+DPO)."
    },
    "6_multi_lora_merge.py": {
        "title": "Multi-LoRA Adapter Synthesis (Task Arithmetic)",
        "implementation": "Implements linear weight interpolation between two distinct LoRA adapter matrices. If Adapter A knows traditional banking and Adapter B knows Crypto regulations, this script mathematically blends their $\Delta W$ matrices using a user-defined ratio (e.g., 0.5) to create a unified expert.",
        "alternatives_rejected": "1. **Sequential Training**: Training Adapter A, then training that result on Dataset B. Rejected because neural networks suffer from 'catastrophic forgetting'; learning B overwrites A.\n2. **Mixture of Experts (MoE) Routing**: Rejected because dynamically loading distinct LoRAs at inference time introduces latency overhead unacceptable for real-time banking applications. Weight merging resolves this pre-inference.",
        "thesis_significance": "Proves that highly specialized, modular knowledge bases can be developed in isolation and mathematically fused, offering a scalable architecture for enterprise financial institutions."
    },
    "7_hyperparameter_sweep.py": {
        "title": "Empirical LoRA Hyperparameter Optimization",
        "implementation": "An automated pipeline that iteratively trains models across varying LoRA constraint dimensions (Rank $r=4, 16, 64$). It logs training latency and tracks the resultant checkpoint directories for downstream benchmarking.",
        "alternatives_rejected": "1. **Grid Search via External Tools (e.g., Ray Tune)**: Rejected due to the specific constraints of Hugging Face Trainer memory management. A custom loop allows for aggressive garbage collection (`torch.cuda.empty_cache()`) between sweeps, preventing Out-Of-Memory (OOM) failures on local hardware.",
        "thesis_significance": "Provides the quantitative empirical data necessary to chart the Pareto frontier of SLM training: identifying the exact mathematical point where increasing model capacity yields diminishing returns in accuracy versus training cost."
    },
    "100_benchmark.py": {
        "title": "Hardware-Agnostic Inference Engine",
        "implementation": "A highly decoupled generation script supporting multi-framework execution. It dynamically routes execution to Apple MPS (Metal), NVIDIA CUDA (with `bitsandbytes` 4-bit quantization), or CPU. It also integrates `llama-cpp-python` to benchmark quantized GGUF artifacts.",
        "alternatives_rejected": "1. **vLLM / Text-Generation-Inference (TGI)**: While excellent for production server deployment, these frameworks are rejected for local benchmarking because they lack seamless support for Apple Silicon and introduce networking overhead that skews pure model latency metrics.\n2. **Coupled Generation and Scoring**: Generating and scoring simultaneously was rejected. Decoupling them allows researchers to re-calculate metrics (e.g., adding a new embedding model) without re-running expensive LLM generation.",
        "thesis_significance": "Ensures that the experimental results of the thesis are hardware-agnostic and highly reproducible across different academic and enterprise computing environments."
    },
    "99_evaluate.py": {
        "title": "Deterministic and Semantic Scoring Engine",
        "implementation": "Calculates ROUGE-L (via `rouge-score`) for structural lexical overlap, and Cosine Similarity (via `sentence-transformers` utilizing `all-MiniLM-L6-v2`) for semantic meaning capture. It computes variance and aggregates results into CSV and PDF formats.",
        "alternatives_rejected": "1. **LLM-as-a-Judge (e.g., GPT-4 evaluating SLM outputs)**: Strongly rejected for the primary metric. LLM judges suffer from positional bias, verbosity bias (preferring longer answers even if wrong), and non-determinism. A scientific thesis requires strictly reproducible, mathematical baseline metrics (Embeddings/Cosine) to prove efficacy.",
        "thesis_significance": "Provides the mathematical proof of the methodology's success, demonstrating that semantic similarity can remain high even when small models use varying vocabulary."
    },
    "utils/generate_distillation_data.py": {
        "title": "Teacher-Student Chain-of-Thought (CoT) Distillation",
        "implementation": "Automates API calls to a frontier Teacher model (DeepSeek/Gemini) to process complex financial queries. Crucially, it modifies the system prompt to force the Teacher to output its internal reasoning process within `<think>` tags before providing the final answer.",
        "alternatives_rejected": "1. **Standard Output Distillation**: Training the student only on the Teacher's final answers. Rejected because SLMs fail at complex logic without stepping stones. By distilling the *reasoning trace* (CoT), the SLM learns the logic matrix of the Teacher, not just the factual output.",
        "thesis_significance": "Demonstrates a methodology for bridging the parameter gap: allowing a 1.5B parameter model to simulate the logical deductions of a 600B parameter model in a restricted financial domain."
    }
}

def generate_document():
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Chapter Z: Detailed Codebase Implementation and Architectural Justifications\n\n")
        f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}*\n\n")
        f.write("This chapter provides a granular, file-by-file analysis of the experimental software architecture. For each core module, it details the technical implementation, justifies the chosen methodology against alternative industry approaches, and highlights its significance to the broader thesis objectives.\n\n")
        f.write("---\n\n")

        for filename, data in ANALYSIS_DATA.items():
            f.write(f"## Z.x Module: `{filename}` - {data['title']}\n\n")
            
            f.write("### Technical Implementation\n")
            f.write(f"{data['implementation']}\n\n")
            
            f.write("### Alternative Approaches Considered and Rejected\n")
            f.write(f"{data['alternatives_rejected']}\n\n")
            
            f.write("### Significance to the Thesis Methodology\n")
            f.write(f"{data['thesis_significance']}\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    generate_document()
    print(f"Successfully generated {OUTPUT_FILE}")

import os
import ast
import datetime

OUTPUT_FILE = "PHD_CODE_DEEP_DIVE.md"

# Define the exact files mapped to the 16 steps of the plan
TARGET_FILES = [
    {"path": "run_pipeline.sh", "phase": "Phase 1: Architecture Overview"},
    {"path": "src/1_sft_train.py", "phase": "Phase 2: Training & Alignment Core"},
    {"path": "src/2_dpo_train.py", "phase": "Phase 2: Training & Alignment Core"},
    {"path": "src/3_orpo_train.py", "phase": "Phase 2: Training & Alignment Core"},
    {"path": "src/6_multi_lora_merge.py", "phase": "Phase 3: Model Management & Optimization"},
    {"path": "src/7_hyperparameter_sweep.py", "phase": "Phase 3: Model Management & Optimization"},
    {"path": "src/100_benchmark.py", "phase": "Phase 4: Benchmarking & Evaluation"},
    {"path": "src/99_evaluate.py", "phase": "Phase 4: Benchmarking & Evaluation"},
    {"path": "src/utils/generate_distillation_data.py", "phase": "Phase 5: Data Synthesis Utilities"},
    {"path": "src/utils/generate_dpo_data.py", "phase": "Phase 5: Data Synthesis Utilities"},
    {"path": "src/utils/generate_dataset.py", "phase": "Phase 5: Data Synthesis Utilities"},
    {"path": "src/utils/augment_dataset.py", "phase": "Phase 5: Data Synthesis Utilities"}
]

# This dictionary holds the deep, academic explanations for specific code patterns.
ACADEMIC_EXPLANATIONS = {
    "1_sft_train.py": {
        "module_summary": "This module serves as the foundational Supervised Fine-Tuning (SFT) engine. It employs parameter-efficient transfer learning to inject financial domain knowledge into a pre-trained base model without triggering catastrophic forgetting of its foundational syntax.",
        "functions": {
            "setup_training_args": "Constructs the hyperparameter matrix for the `Trainer` class. It strictly defines gradient accumulation and mixed-precision (if hardware permits) to optimize GPU VRAM, allowing larger batch sizes to fit within local constraints.",
            "format_example": "A robust data preprocessing pipeline. It programmatically normalizes diverse JSON schemas (e.g., 'instruction/output' vs. 'prompt/completion') and utilizes the tokenizer's native `apply_chat_template`. This ensures the model learns the correct control tokens (like `<|im_start|>`) required for conversational inference.",
            "train": "The core orchestration loop. Crucially, before any gradient updates occur, this function algorithmically enforces a deterministic 90/10 data split. The 10% holdout set is serialized to disk (`_test.jsonl`), establishing an immutable zero-leakage boundary essential for the scientific validity of the downstream benchmarking."
        }
    },
    "2_dpo_train.py": {
         "module_summary": "This module transitions the model from knowledge acquisition to behavioral alignment using Direct Preference Optimization (DPO). It bypasses the architectural instability of Reinforcement Learning (RLHF) by directly optimizing a supervised loss function.",
         "functions": {
             "train": "Instantiates the `trl.DPOTrainer`. It maps a dataset containing paired 'chosen' and 'rejected' responses against a reference policy (the SFT model). The objective function mathematically increases the log-probability of safe, accurate financial responses while actively penalizing hallucinatory or poorly formatted text."
         }
    },
    "3_orpo_train.py": {
         "module_summary": "An experimental module implementing Odds Ratio Preference Optimization (ORPO). It allows for a comparative ablation study against the traditional two-stage SFT+DPO pipeline.",
         "functions": {
             "train": "Implements a monolithic alignment architecture. Instead of requiring a separate reference model, it applies an odds-ratio penalty directly to the negative log-likelihood loss during initial fine-tuning, theoretically halving the required compute hours."
         }
    },
    "6_multi_lora_merge.py": {
        "module_summary": "This module provides a scalable solution to multi-domain expertise via 'Task Arithmetic'.",
        "functions": {
            "merge_adapters": "Performs linear algebraic interpolation between discrete LoRA adapter matrices (Delta W). By calculating a user-defined weighted sum (e.g., 0.5 traditional banking + 0.5 crypto regulation), it synthesizes a unified 'Expert' model without the inference latency associated with Mixture-of-Experts (MoE) routing."
        }
    },
    "7_hyperparameter_sweep.py": {
        "module_summary": "An empirical testing suite designed to chart the Pareto optimal frontier between model capacity and training latency.",
        "functions": {
            "run_sweep": "Iteratively executes the training loop across varying LoRA constraint dimensions (Rank r=4, 16, 64). It aggressively manages PyTorch memory states (`torch.cuda.empty_cache()`) to prevent resource exhaustion, outputting quantitative data necessary to justify architectural sizing decisions."
        }
    },
    "100_benchmark.py": {
        "module_summary": "A strictly decoupled, hardware-agnostic inference engine. It is designed to generate responses against the unseen test set, generating the raw data required for subsequent mathematical scoring.",
        "functions": {
            "load_model_and_tokenizer": "Implements dynamic hardware routing. It detects the execution environment and maps tensors to Apple Metal (`mps`), NVIDIA CUDA, or CPU. It optionally injects `bitsandbytes` to evaluate the semantic degradation caused by 4-bit scalar quantization.",
            "benchmark_model": "Manages the autoregressive generation cycle. It standardizes inference parameters (e.g., temperature, max tokens) across both Hugging Face Transformers and highly compressed GGUF artifacts (via `llama-cpp-python`), ensuring a controlled experimental environment."
        }
    },
    "99_evaluate.py": {
        "module_summary": "The deterministic scoring engine. It provides the mathematical proof of the methodology's efficacy, rejecting subjective 'LLM-as-a-Judge' approaches in favor of reproducible statistical metrics.",
        "functions": {
            "calculate_rouge": "Utilizes the `rouge-score` library to measure the Longest Common Subsequence (ROUGE-L) between generated outputs and gold-standard references, providing a baseline metric for lexical and structural overlap.",
            "calculate_similarity": "The core semantic metric. It transforms textual outputs into high-dimensional vector embeddings utilizing `sentence-transformers` (`all-MiniLM-L6-v2`). By calculating Cosine Similarity across these vectors, it proves that the model comprehends financial concepts even if it substitutes specific vocabulary.",
            "main": "Aggregates the dual-metric scores, calculates operational metrics (latency, throughput), and serializes the findings into definitive CSV logs and visual PDF reports."
        }
    },
    "generate_distillation_data.py": {
        "module_summary": "A data engineering utility that implements Teacher-Student knowledge distillation, specifically targeting reasoning capabilities.",
        "functions": {
            "generate_distilled_data": "Interfaces with frontier APIs (DeepSeek/Gemini). It employs strict prompt engineering to force the Teacher model to expose its latent logical deductions within `<think>` blocks. By training the SLM on this Chain-of-Thought (CoT), the smaller model learns the 'how' of financial reasoning, not just the factual outputs."
        }
    },
    "generate_dpo_data.py": {
        "module_summary": "Automates the synthesis of negative preference data, a process that traditionally requires prohibitive amounts of human labor.",
        "functions": {
            "generate_dpo_pair": "Forces an external LLM to act adversarially, generating deliberate hallucinations, verbosity, or unhelpful responses to valid queries. This creates the exact 'rejected' mathematical boundary required by the DPO loss function to build safety rails."
        }
    },
    "generate_dataset.py": {
        "module_summary": "A parsing and extraction pipeline for unstructured regulatory data.",
        "functions": {
             "extract_qa_from_pdf": "Ingests dense, unstructured financial PDF texts (e.g., the MiCA regulation). It chunks the text and utilizes a Generative API to synthesize highly structured instruction-response pairs, ensuring the SLM can ingest current legal frameworks."
        }
    },
    "augment_dataset.py": {
        "module_summary": "A robustness utility designed to prevent the SLM from overfitting to the grammatical syntax of the synthetic data.",
        "functions": {
             "augment_data": "Programmatically multiplies the dataset footprint. It mutates a single factual query into multiple distinct variations representing different user personas (e.g., an angry customer vs. a confused elderly person), forcing the model to map diverse lexical inputs to the same core financial intent."
        }
    }
}

def append_to_doc(content):
    with open(OUTPUT_FILE, "a") as f:
        f.write(content + "\n\n")

def init_doc():
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Chapter W: Comprehensive Codebase Technical Breakdown\n\n")
        f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}*\n\n")
        f.write("This chapter provides an exhaustive, function-by-function analysis of the experimental software suite. It leverages Abstract Syntax Tree (AST) parsing to map the physical codebase to its academic and methodological justifications.\n\n")
        f.write("---\n\n")

def analyze_shell_script(filepath):
    append_to_doc(f"## {filepath} Analysis")
    append_to_doc("### Methodological Orchestration")
    append_to_doc("The `run_pipeline.sh` script acts as the master orchestrator for the entire experiment. Rather than manual execution, which is prone to human error, this shell script enforces a strict, reproducible sequence: Data Synthesis -> SFT Training -> DPO Alignment -> Hardware-Agnostic Benchmarking -> Deterministic Evaluation. Crucially, it implements stateful checkpointing (via `.checkpoints/`), ensuring that multi-hour training workflows can recover from computational interruptions without data corruption.")

def analyze_python_file(filepath):
    filename = os.path.basename(filepath)
    append_to_doc(f"## Module: `{filename}`")
    
    # Get academic explanation if it exists
    explanations = ACADEMIC_EXPLANATIONS.get(filename, {})
    
    if "module_summary" in explanations:
        append_to_doc(f"### Architectural Purpose\n{explanations['module_summary']}")
    
    try:
        with open(filepath, 'r') as f:
            code_content = f.read()
            tree = ast.parse(code_content)
    except Exception as e:
        append_to_doc(f"*Error parsing file for AST: {e}*")
        return

    # Extract Imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    if imports:
        append_to_doc("### Technical Dependencies\n" + 
                      "The module relies on the following core libraries to execute its mathematical and computational directives:\n" + 
                      "`" + ", ".join(list(set(imports))) + "`")

    # Extract Functions and Classes via AST
    append_to_doc(f"### Structural Logic & Function Breakdown")
    
    functions_found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions_found = True
            func_name = node.name
            args = [arg.arg for arg in node.args.args]
            loc = node.end_lineno - node.lineno
            
            append_to_doc(f"#### `def {func_name}({', '.join(args)}):`")
            
            # Check if we have a specific academic explanation for this function
            func_explanation = explanations.get("functions", {}).get(func_name)
            
            if func_explanation:
                append_to_doc(f"**Implementation & Thesis Significance:**\n{func_explanation}")
            else:
                # Fallback to docstring or generic description
                docstring = ast.get_docstring(node)
                if docstring:
                    append_to_doc(f"**Documented Implementation:** {docstring.strip().split(chr(10))[0]}")
                else:
                    append_to_doc(f"Executes internal procedural logic specific to the `{func_name}` operational scope.")
                    
            append_to_doc(f"*(Complexity: {loc} lines of code)*")
            
    if not functions_found:
        append_to_doc("*This file executes linearly as a monolithic script without isolated function definitions.*")
    
    append_to_doc("---\n")

def main():
    init_doc()
    
    current_phase = ""
    for item in TARGET_FILES:
        filepath = item["path"]
        phase = item["phase"]
        
        # Print phase header if it changed
        if phase != current_phase:
            append_to_doc(f"# {phase}\n")
            current_phase = phase
            print(f"\nProcessing {phase}...")
            
        if not os.path.exists(filepath):
            print(f"  [!] File {filepath} not found. Skipping.")
            continue
            
        print(f"  -> Parsing {filepath}")
        if filepath.endswith('.sh'):
            analyze_shell_script(filepath)
        elif filepath.endswith('.py'):
            analyze_python_file(filepath)

    print(f"\nExecution Complete. Document generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

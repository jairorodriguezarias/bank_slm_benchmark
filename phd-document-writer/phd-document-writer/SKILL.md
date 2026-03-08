---
name: phd-document-writer
description: Generates a massively detailed, PhD-level or arXiv-style academic technical explanation of any specified file in LaTeX format. Uses a strict Two-Phase (Plan then Execute) workflow to bypass token limits and ensure exhaustive depth. Saves output to the 'ACADEMIC' folder.
---

# PhD Document Writer (Iterative LaTeX Edition)

This skill instructs you to act as an elite academic researcher writing a PhD thesis chapter. Your task is to analyze a given source file and generate a massively comprehensive LaTeX document. 

**CRITICAL DIRECTIVE:** To bypass output token limits and guarantee extreme depth, you MUST execute this task in two distinct phases. Do NOT attempt to write the whole document in one step.

## Phase 1: Planning & Outline (Requires User Approval)

1.  **Read the Source File:** Use the `read_file` tool to ingest the code.
2.  **Analyze and Structure:** Logically break down the code. Identify every function, class, and architectural decision.
3.  **Present the Plan:** Output a detailed, multi-step execution plan to the user in the chat. The plan MUST divide the writing process into at least 4-5 sequential steps. 
    *   *Example Plan:*
        *   Step 1: Write Preamble, Abstract, and Introduction.
        *   Step 2: Write Technical Implementation (Data Loading & Preprocessing).
        *   Step 3: Write Technical Implementation (Core Algorithm & Math).
        *   Step 4: Write Architectural Justifications & Alternatives.
        *   Step 5: Write Significance & Conclusion.
4.  **STOP AND WAIT:** You must explicitly ask the user: *"Do you approve this plan? Should I begin generating Step 1?"* Do not proceed to Phase 2 until the user says yes.

## Phase 2: Phased Execution (Sequential Writing)

Once the user approves the plan, execute it step-by-step. 

1.  **Initialize the File (Step 1):** For the first step, use the `run_shell_command` tool to create the initial file (e.g., `echo "..." > ACADEMIC/filename_ACADEMIC.tex`).
2.  **Sequential Appending (Steps 2+):** For every subsequent step, generate the massive, highly detailed LaTeX content for that specific section, and use the `run_shell_command` tool with the append operator (`cat << 'EOF' >> ACADEMIC/filename_ACADEMIC.tex ... EOF`) to add it to the file.
3.  **Exhaustive Depth:** Because you are generating one section at a time, you have no token constraints. You must be aggressively verbose. Include heavy mathematical proofs, rigorous algorithmic analysis ($O(N)$), and extensive code snippets (`\begin{lstlisting}`).
4.  **Wait Between Steps (Optional but Recommended):** If a section is particularly massive, you may execute one step, confirm success, and automatically proceed to the next step, ensuring you narrate your progress to the user.

## LaTeX Formatting Guidelines

*   **Preamble:** Always include `\usepackage{amsmath, amssymb, hyperref, listings, xcolor, geometry}`.
*   **Math:** Use `\begin{equation}` for core formulas.
*   **Tone:** Highly formal, objective, pedantic, and analytical.
*   **Syntax:** Carefully escape LaTeX special characters (like `_` and `&`) outside of code or math blocks.
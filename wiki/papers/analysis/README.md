# Paper analyses and retrieval guide

Start here when answering questions about the collected literature. This folder has one detailed, source-linked analysis for each of 22 works, including the METR web article and LARK. The notes prioritize coverage of concepts, mechanisms, methods, findings, conclusions, limitations, and substantive appendices.

## What is here

- `NN-slug.md`: one analysis per work. The same basename under [../pdfs/](../pdfs/README.md) is its full source text, not another summary.
- [index.json](index.json): structured retrieval index containing stable IDs, titles, versions, local analysis/source paths, summaries, topic tags, section names, and descriptive bullet/word counts. `by_topic` maps each tag to paper IDs; `question_routes` groups useful starting points. All paths resolve relative to this directory.
- [REVIEW.md](REVIEW.md): reading/review scope, completed checks, and remaining limitations. `reviewed` means editorial and targeted source review; it does not mean the experiments were reproduced or every source claim independently verified.
- This README: question routing, analysis schema, catalog, and maintenance instructions.

## Retrieval workflow

1. Use the question routes below or search `index.json` for a topic/alias. Read a small set of relevant analyses rather than loading all full texts.
2. Read concepts/methods before interpreting results. Also read limitations and source-coverage notes.
3. Follow local source citations for exact values, equations, guarantees, and comparisons. Check figures/PDF pages when the evidence is visual.
4. Cite the source version and precise locator in the answer. Identify a survey's descriptions as secondary evidence until its cited primary study has been checked.
5. If the store does not support an answer, say which evidence is missing and consult additional sources. Do not fill gaps with inferred paper results.

Search works with ordinary text tools; no embedding service is required. From the repository root, examples are `rg -n 'best-of|geometric|TF32' wiki/papers/analysis/` and `rg -n 'position bias|tie|calibration' wiki/papers/analysis/`. Search the full-text directory when the note is insufficient.

## Question routing

These groups are navigation aids, not cross-paper rankings or synthesized scientific conclusions. A paper can appear in several groups because it addresses several questions.

### Kernel benchmark design and metrics

Task definitions, baselines, sampling budgets, correctness gates, performance aggregation, and hardware/task coverage.

[01 — KernelBench](01-kernelbench.md), [02 — KernelBenchX](02-kernelbench-x.md), [03 — KernelBench-Verified](03-kernelbench-verified.md), [04 — PTXBench](04-ptxbench.md), [05 — TritonBench](05-tritonbench.md), [06 — MultiKernelBench](06-multikernelbench.md), [07 — ISO-Bench](07-iso-bench.md), [08 — Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization](08-robust-agentic-cuda.md), [21 — Measuring Automated Kernel Engineering](21-measuring-automated-kernel-engineering.md).

### Correctness, numerical behavior, and proof boundaries

Hidden tests, runtime instruction checks, soft verification, hardware floating-point models, and formal proof assumptions.

[03 — KernelBench-Verified](03-kernelbench-verified.md), [04 — PTXBench](04-ptxbench.md), [08 — Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization](08-robust-agentic-cuda.md), [09 — Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units](09-bit-accurate-gpu-mma.md), [22 — LARK](22-lark.md).

### Agents, tools, and iterative improvement

Agent interfaces, reasoning/action loops, self-feedback, episodic memory, and learned API selection.

[10 — SWE-agent](10-swe-agent.md), [11 — ReAct](11-react.md), [18 — Self-Refine](18-self-refine.md), [19 — Reflexion](19-reflexion.md), [20 — Toolformer](20-toolformer.md).

### Kernel search and optimization feedback

Execution/profiler feedback, refinement trajectories, candidate selection, repository optimization, and resource budgets.

[01 — KernelBench](01-kernelbench.md), [02 — KernelBenchX](02-kernelbench-x.md), [04 — PTXBench](04-ptxbench.md), [07 — ISO-Bench](07-iso-bench.md), [08 — Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization](08-robust-agentic-cuda.md), [21 — Measuring Automated Kernel Engineering](21-measuring-automated-kernel-engineering.md).

### LLM judges, rubrics, and bias

Judge training, scoring protocols, agreement measures, order/verbosity/self-preference effects, and survey navigation.

[12 — Prometheus](12-prometheus.md), [13 — From Generation to Judgment](13-generation-to-judgment.md), [15 — G-Eval](15-g-eval.md), [16 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](16-mt-bench-chatbot-arena.md), [17 — Large Language Models are not Fair Evaluators](17-fair-evaluators.md).

### Reproducible evaluation and comparison conditions

Prompt/metric implementation, uncertainty, baseline configuration, tie handling, denominators, and aggregation.

[14 — Lessons from the Trenches on Reproducible Evaluation of Language Models](14-reproducible-lm-evaluation.md), [01 — KernelBench](01-kernelbench.md), [03 — KernelBench-Verified](03-kernelbench-verified.md), [16 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](16-mt-bench-chatbot-arena.md), [17 — Large Language Models are not Fair Evaluators](17-fair-evaluators.md), [21 — Measuring Automated Kernel Engineering](21-measuring-automated-kernel-engineering.md).

### Inference optimization in larger systems

Repository patches, application-level extrapolation, production-kernel adaptation, and limits of benchmark-to-deployment inference.

[07 — ISO-Bench](07-iso-bench.md), [21 — Measuring Automated Kernel Engineering](21-measuring-automated-kernel-engineering.md), [22 — LARK](22-lark.md).

## Consistent analysis structure

Every note has YAML frontmatter followed by nine sections. Some headings include numeric prefixes; their roles are the same.

1. **Paper identity and scope:** research question, contribution, artifact/version, and what the work actually studies.
2. **Concepts and terminology:** definitions, mechanisms, metrics, equations, and distinctions needed to understand the evidence.
3. **Methodology and experimental setup:** datasets, models, hardware, prompts, budgets, controls, and evaluation procedures; survey methodology where applicable.
4. **Results and evidence:** source-reported findings with conditions, denominators, and locators. Derived calculations are labeled.
5. **Authors' conclusions:** what the authors infer or advocate, separate from the measurements themselves.
6. **Limitations and open questions:** explicit limitations, analyst caveats, missing evidence, and unresolved source inconsistencies.
7. **Relevance to this project — interpretation:** proposed applications and hypotheses. These are not implementation claims or established paper findings.
8. **Retrieval notes:** aliases, questions the paper answers, and related local analyses for further reading.
9. **Source coverage and verification notes:** main sections/appendices read, special PDF/figure checks, conversion problems, and review limits.

Frontmatter keys are `paper_id`, `title`, `source_version`, `source_markdown`, `source_pdf`, `source_url`, `analysis_date`, `analysis_status`, `topics`, and `summary`. `source_pdf` is null for entry 21. IDs are strings with leading zeros; dates are ISO dates; topics are a list of searchable strings. Source versions must match [the manifest](../pdfs/manifest.json).

## Citation and interpretation conventions

A citation such as `[§3.2; Table 1](../pdfs/01-kernelbench.md)` carries its locator in the label; it is not a claim that the link jumps to that exact passage. Use source text search to locate it. PDF page links use one-based physical page numbers. Source HTML/Markdown and PDF figure/table labels occasionally differ, so use the linked artifact's numbering.

Preserve correctness gates, failed-task treatment, model/hardware versions, baselines, sample/turn budgets, and aggregation when extracting numbers. Percentage points differ from relative percentages; conditional speedup differs from coverage; human agreement differs from objective correctness; a model-predicted verdict differs from executable validation; formal guarantees apply only within their assumptions.

Notes may explicitly flag inconsistent source prose, tables, arithmetic, or examples. Keep the disagreement visible and use a table-specific result when needed. Do not silently replace published numbers with inferred repairs. Survey inventory sizes remain survey-reported unless their primary sources have been checked separately.

## Complete analysis catalog

### 01. KernelBench: Can LLMs Write Efficient GPU Kernels?

[Analysis](01-kernelbench.md) · [Full text](../pdfs/01-kernelbench.md) · [PDF](../pdfs/01-kernelbench.pdf) · [Online source](https://arxiv.org/abs/2502.10517)

Version: `2502.10517v1`. Introduces 250 PyTorch-defined kernel-generation tasks and the correctness-gated fast_p metric, with one-shot, sampling, feedback, and hardware-context experiments.

Topics: `gpu-kernels`, `kernelbench`, `benchmarking`, `cuda`, `test-time-compute`, `execution-feedback`, `profiling`, `hardware-awareness`, `correctness`.

### 02. KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels

[Analysis](02-kernelbench-x.md) · [Full text](../pdfs/02-kernelbench-x.md) · [PDF](../pdfs/02-kernelbench-x.pdf) · [Online source](https://arxiv.org/abs/2605.04956)

Version: `2605.04956v2`. Category-aware evaluation of Triton generation separates compilation, semantic correctness, repair dynamics, and performance across 176 tasks and six GPUs.

Topics: `gpu-kernels`, `triton`, `benchmarking`, `correctness`, `quantization`, `hardware-efficiency`, `iterative-refinement`, `portability`.

### 03. KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

[Analysis](03-kernelbench-verified.md) · [Full text](../pdfs/03-kernelbench-verified.md) · [PDF](../pdfs/03-kernelbench-verified.pdf) · [Online source](https://arxiv.org/abs/2607.16241)

Version: `2607.16241v1`. Re-evaluates generated KernelBench candidates using TF32-enabled eager baselines, transformed hidden inputs, memory metrics, and a residual-hack code audit.

Topics: `gpu-kernels`, `benchmarking`, `tf32`, `hidden-tests`, `reward-hacking`, `numerical-accuracy`, `memory-efficiency`, `llm-as-judge`, `bf16`.

### 04. PTXBench: Benchmarking and Adapting LLMs for GPU Kernel Optimization with Architecture-specific PTX

[Analysis](04-ptxbench.md) · [Full text](../pdfs/04-ptxbench.md) · [PDF](../pdfs/04-ptxbench.pdf) · [Online source](https://arxiv.org/abs/2608.17379)

Version: `2608.17379v3`. PTXBench tests runtime use of architecture-specific GPU instructions separately from correctness and speed, and studies repair-conditioned adaptation of Qwen3.6-27B.

Topics: `gpu-kernels`, `ptx`, `hardware-awareness`, `dynamic-profiling`, `h100`, `b200`, `supervised-fine-tuning`, `repair-learning`, `attention`, `agent-evaluation`.

### 05. TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators

[Analysis](05-tritonbench.md) · [Full text](../pdfs/05-tritonbench.md) · [PDF](../pdfs/05-tritonbench.pdf) · [Online source](https://arxiv.org/abs/2502.14752)

Version: `2502.14752v1`. Dual-channel Triton benchmark contrasts 184 curated GitHub operators with 166 PyTorch-interface tasks using correctness, latency, and hardware-efficiency measurements.

Topics: `gpu-kernels`, `triton`, `benchmarking`, `code-generation`, `retrieval`, `supervised-fine-tuning`, `gpu-efficiency`, `error-taxonomy`.

### 06. MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation

[Analysis](06-multikernelbench.md) · [Full text](../pdfs/06-multikernelbench.md) · [PDF](../pdfs/06-multikernelbench.pdf) · [Online source](https://arxiv.org/abs/2507.17773)

Version: `2507.17773v2`. MultiKernelBench evaluates 285 PyTorch-defined tasks on CUDA, AscendC, and Pallas, exposing platform/category differences and testing category-specific exemplars.

Topics: `multi-platform`, `gpu-kernels`, `ascendc`, `pallas`, `benchmarking`, `category-aware-prompting`, `correctness`, `hardware-portability`.

### 07. ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?

[Analysis](07-iso-bench.md) · [Full text](../pdfs/07-iso-bench.md) · [PDF](../pdfs/07-iso-bench.pdf) · [Online source](https://arxiv.org/abs/2602.19594)

Version: `2602.19594v1`. ISO-Bench evaluates agent patches on 54 inference-serving optimization tasks using human-relative performance, judge-rated bottleneck targeting, and selective functional validation.

Topics: `repository-optimization`, `inference-serving`, `vllm`, `sglang`, `coding-agents`, `llm-as-judge`, `correctness`, `agent-scaffolding`, `benchmarking`.

### 08. Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

[Analysis](08-robust-agentic-cuda.md) · [Full text](../pdfs/08-robust-agentic-cuda.md) · [PDF](../pdfs/08-robust-agentic-cuda.pdf) · [Online source](https://arxiv.org/abs/2509.14279v1)

Version: `2509.14279v1`. Introduces robust-kbench and an evolutionary CUDA search pipeline with prompt-tuned LLM screening, measuring benchmark loopholes, forward/backward optimization, and shape/hardware generalization while retaining empirical hardware checks.

Topics: `gpu-kernels`, `benchmarking`, `benchmark-validity`, `evolutionary-search`, `llm-verification`, `prompt-optimization`, `numerical-correctness`, `backpropagation`, `performance-generalization`.

### 09. Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy

[Analysis](09-bit-accurate-gpu-mma.md) · [Full text](../pdfs/09-bit-accurate-gpu-mma.md) · [PDF](../pdfs/09-bit-accurate-gpu-mma.pdf) · [Online source](https://arxiv.org/abs/2511.10909)

Version: `2511.10909v3`. CLFP derives and differentially tests instruction-level arithmetic models for ten GPU architectures, explaining mixed-precision discrepancies and identifying precision bottlenecks, rounding bias and targeted workarounds.

Topics: `gpu-kernels`, `floating-point`, `numerical-correctness`, `tensor-cores`, `matrix-cores`, `hardware-semantics`, `differential-testing`, `mma-sim`, `reproducibility`.

### 10. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

[Analysis](10-swe-agent.md) · [Full text](../pdfs/10-swe-agent.md) · [PDF](../pdfs/10-swe-agent.pdf) · [Online source](https://arxiv.org/abs/2405.15793)

Version: `2405.15793v3`. Studies how search, file viewing, editing guardrails, and context management affect fixed-model software-engineering agents on SWE-bench and HumanEvalFix.

Topics: `agent-computer-interface`, `software-engineering`, `tool-design`, `context-management`, `execution-feedback`, `evaluation`, `error-recovery`.

### 11. ReAct: Synergizing Reasoning and Acting in Language Models

[Analysis](11-react.md) · [Full text](../pdfs/11-react.md) · [PDF](../pdfs/11-react.pdf) · [Online source](https://arxiv.org/abs/2210.03629)

Version: `2210.03629v3`. Introduces interleaved language reasoning and environment actions, testing prompted and fine-tuned variants on knowledge reasoning and interactive decision tasks.

Topics: `reasoning-and-acting`, `tool-use`, `retrieval`, `planning`, `grounding`, `few-shot-prompting`, `agent-evaluation`.

### 12. Prometheus: Inducing Fine-grained Evaluation Capability in Language Models

[Analysis](12-prometheus.md) · [Full text](../pdfs/12-prometheus.md) · [PDF](../pdfs/12-prometheus.pdf) · [Online source](https://arxiv.org/abs/2310.08491)

Version: `2310.08491v2`. Trains 7B/13B rubric-conditioned evaluators on synthetic feedback and studies reference materials, human agreement, distribution shift, and preference ranking.

Topics: `llm-as-a-judge`, `rubrics`, `reference-guidance`, `judge-training`, `synthetic-data`, `feedback`, `human-alignment`, `reward-models`.

### 13. From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge

[Analysis](13-generation-to-judgment.md) · [Full text](../pdfs/13-generation-to-judgment.md) · [PDF](../pdfs/13-generation-to-judgment.pdf) · [Online source](https://arxiv.org/abs/2411.16594)

Version: `2411.16594v7`. Survey taxonomy of judge inputs, outputs, six assessment attributes, training and prompting methods, four application areas, benchmark families, and unresolved bias, efficiency, and robustness problems.

Topics: `llm-as-a-judge`, `survey`, `evaluation`, `rubrics`, `judge-training`, `bias`, `alignment`, `retrieval`, `reasoning`, `benchmarks`, `multilingual-evaluation`, `multimodal-evaluation`.

### 14. Lessons from the Trenches on Reproducible Evaluation of Language Models

[Analysis](14-reproducible-lm-evaluation.md) · [Full text](../pdfs/14-reproducible-lm-evaluation.md) · [PDF](../pdfs/14-reproducible-lm-evaluation.pdf) · [Online source](https://arxiv.org/abs/2405.14782)

Version: `2405.14782v3`. Documents evaluation failure modes and reproducibility practices, with prompt-format case studies and formal descriptions of likelihood, perplexity, and generative scoring.

Topics: `evaluation`, `reproducibility`, `benchmarking`, `statistics`, `prompt-sensitivity`, `tokenization`, `lm-eval`, `measurement-validity`.

### 15. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

[Analysis](15-g-eval.md) · [Full text](../pdfs/15-g-eval.md) · [PDF](../pdfs/15-g-eval.pdf) · [Online source](https://arxiv.org/abs/2303.16634)

Version: `2303.16634v3`. Prompted NLG evaluation using generated evaluation steps and probability-weighted scores, tested against human judgments on summarization and dialogue.

Topics: `llm-as-a-judge`, `evaluation`, `rubrics`, `chain-of-thought`, `human-alignment`, `bias`, `summarization`.

### 16. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

[Analysis](16-mt-bench-chatbot-arena.md) · [Full text](../pdfs/16-mt-bench-chatbot-arena.md) · [PDF](../pdfs/16-mt-bench-chatbot-arena.pdf) · [Online source](https://arxiv.org/abs/2306.05685)

Version: `2306.05685v4`. Introduces MT-Bench and Chatbot Arena and measures how LLM judges align with human preferences under explicit tie-handling and bias controls.

Topics: `llm-as-a-judge`, `human-preference`, `benchmarking`, `multi-turn`, `bias`, `pairwise-evaluation`, `judge-training`.

### 17. Large Language Models are not Fair Evaluators

[Analysis](17-fair-evaluators.md) · [Full text](../pdfs/17-fair-evaluators.md) · [PDF](../pdfs/17-fair-evaluators.pdf) · [Online source](https://arxiv.org/abs/2305.17926)

Version: `2305.17926v2`. A study of response-order bias in pairwise LLM judging, with evidence-first sampling, balanced positions, and entropy-based human review.

Topics: `llm-as-a-judge`, `position-bias`, `calibration`, `human-in-the-loop`, `uncertainty`, `pairwise-evaluation`.

### 18. Self-Refine: Iterative Refinement with Self-Feedback

[Analysis](18-self-refine.md) · [Full text](../pdfs/18-self-refine.md) · [PDF](../pdfs/18-self-refine.pdf) · [Online source](https://arxiv.org/abs/2303.17651)

Version: `2303.17651v2`. Tests a same-model generate–critique–revise loop across seven tasks, with strong task-dependent gains and a major bottleneck in detecting incorrect feedback.

Topics: `self-feedback`, `iterative-refinement`, `test-time-compute`, `code-optimization`, `multi-aspect-evaluation`, `llm-as-a-judge`, `error-analysis`.

### 19. Reflexion: Language Agents with Verbal Reinforcement Learning

[Analysis](19-reflexion.md) · [Full text](../pdfs/19-reflexion.md) · [PDF](../pdfs/19-reflexion.pdf) · [Online source](https://arxiv.org/abs/2303.11366)

Version: `2303.11366v4`. Reflexion converts trial feedback into bounded textual memories for subsequent attempts, with gains and failures across ALFWorld, HotpotQA, and code generation that depend strongly on evaluation quality.

Topics: `agents`, `reflection`, `episodic-memory`, `self-generated-tests`, `feedback`, `program-synthesis`, `reasoning`, `evaluation`.

### 20. Toolformer: Language Models Can Teach Themselves to Use Tools

[Analysis](20-toolformer.md) · [Full text](../pdfs/20-toolformer.md) · [PDF](../pdfs/20-toolformer.pdf) · [Online source](https://arxiv.org/abs/2302.04761)

Version: `2302.04761v1`. Learns API-call placement and arguments by filtering sampled tool annotations for next-token loss improvement, with controlled tool-disabled and decoding-policy evaluations.

Topics: `tool-use`, `self-supervised-learning`, `synthetic-data`, `api-selection`, `retrieval`, `language-modeling`, `evaluation`, `decoding`.

### 21. Measuring Automated Kernel Engineering

[Analysis](21-measuring-automated-kernel-engineering.md) · [Full text](../pdfs/21-measuring-automated-kernel-engineering.md) · [Online source](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)

Version: `web snapshot 2026-11-20`. METR evaluates a tree-search kernel agent on a filtered, extended KernelBench suite, reporting speedups and cost tradeoffs while documenting major gaps from expert kernel engineering.

Topics: `gpu-kernels`, `benchmarking`, `agent-scaffolding`, `test-time-compute`, `benchmark-validity`, `cost-effectiveness`, `training-data`, `kernelbench`.

### 22. LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels

[Analysis](22-lark.md) · [Full text](../pdfs/22-lark.md) · [PDF](../pdfs/22-lark.pdf) · [Online source](https://arxiv.org/abs/2611.02028)

Version: `2611.02028v1`. LARK uses Lean-checked race-freedom, segmentation and equivalence proofs to certify supported GPU kernels relative to a pinned machine model and task interface, while leaving floating-point fidelity and host-allocation safety outside its theorem.

Topics: `gpu-kernels`, `formal-verification`, `lean`, `exact-arithmetic`, `race-freedom`, `trust-boundary`, `proof-agents`, `kernelcraft`, `ptx`, `cuda`, `hip`.

## Maintenance and review

Follow [the collection update procedure](../README.md). New or materially revised analyses start as `draft`; mark them `reviewed` only after reading the relevant source, checking consequential claims and local citations, and documenting review scope. If a source revision changes, re-review affected sections before carrying that status forward.

When editing metadata, update the matching `index.json` entry and its `by_topic` mapping, then update this catalog. Keep all source and analysis paths relative to this folder in the index. Refresh section names and descriptive counts if the note changes; these counts are navigation metadata, not a quality score. Do not alter originals to match an analysis or overwrite uncertainty with confident prose.

Use the notes as starting evidence for future answers. They are not exhaustive transcriptions of every bibliography entry, example transcript, or code line; those remain in the full texts. New cross-paper conclusions should be written separately, attributed, and supported by the underlying evidence.

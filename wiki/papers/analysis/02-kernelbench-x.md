---
paper_id: "02"
title: "KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels"
source_version: "2605.04956v2"
source_markdown: "../pdfs/02-kernelbench-x.md"
source_pdf: "../pdfs/02-kernelbench-x.pdf"
source_url: "https://arxiv.org/abs/2605.04956"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "triton", "benchmarking", "correctness", "quantization", "hardware-efficiency", "iterative-refinement", "portability"]
summary: "Category-aware evaluation of Triton generation separates compilation, semantic correctness, repair dynamics, and performance across 176 tasks and six GPUs."
---

# KernelBenchX: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels

## Paper identity and scope

- The analyzed artifact is KernelBenchX, arXiv 2605.04956v2; it extends TritonBench-T and is not simply the original KernelBench with new models. [§1; §3.2](../pdfs/02-kernelbench-x.md)
- The central questions concern where Triton-generation capability fails, how failures depend on task structure, and whether iteration improves correctness or hardware efficiency. [§1](../pdfs/02-kernelbench-x.md)
- The unit of evaluation is a task-specific generated kernel under a common adapter/evaluation pipeline, rather than a deployed application or a whole-codebase patch. [§3.1](../pdfs/02-kernelbench-x.md)
- The study compares five methods and releases iteration transition artifacts; it is an empirical benchmark plus explanatory analysis, not a correctness proof system. [§1; Appendix E](../pdfs/02-kernelbench-x.md)

## Concepts and terminology

- Compilation/call success, numerical correctness, speed, hardware efficiency, and portability are distinct success stages; success at one stage does not establish the others. [§3.2; §4.2; §6](../pdfs/02-kernelbench-x.md)
- Call Accuracy requires nonempty output, a valid AST-recognized kernel entry, successful import/compilation/call, and task constraints. [§3.2.2; Appendix A.2](../pdfs/02-kernelbench-x.md)
- Execution Accuracy compares generated and reference test_results recursively under shared seeds; shape and dtype must match before numerical comparisons. [Appendix A.3](../pdfs/02-kernelbench-x.md)
- Standard tasks use dtype-aware numerical tolerances; quantization has an approximation contract requiring several accuracy metrics simultaneously, not exact output equality. [§3.2.2; Appendix C](../pdfs/02-kernelbench-x.md)
- Achieved bandwidth is modeled bytes divided by measured runtime; achieved throughput is modeled FLOPs divided by runtime. Units are GB/s and TFLOPS. [§2.1, Eq. 1](../pdfs/02-kernelbench-x.md)
- IOU normalizes achieved bandwidth by peak bandwidth; MFU normalizes throughput by peak compute. The larger of the two is the reported dominant-bottleneck efficiency proxy. [§2.1, Eq. 2; §3.2.3](../pdfs/02-kernelbench-x.md)
- FLOPs and bytes describe a fixed task-level idealized target computation, not hardware counters or the actual executed instruction stream. [§3.2.3](../pdfs/02-kernelbench-x.md)
- Maintainability Index and Cyclomatic Complexity are static code-quality metrics; they do not themselves measure execution correctness or production maintainability. [§3.2.3](../pdfs/02-kernelbench-x.md)
- Per-task speedup divides total reference runtime by total candidate runtime across that task's inputs. Arithmetic mean of task speedups differs from a global ratio of summed times; the paper primarily uses the former. [Appendix A.4](../pdfs/02-kernelbench-x.md)
- Category taxonomy targets computational structure/knowledge: direct formulas; aggregation; coupled multi-operand computation; indexing/spatial mapping; fusion; and semantic contracts. [§3.2.1; Appendix A.1](../pdfs/02-kernelbench-x.md)
- Global tensor contracts include reduction scope, broadcast alignment, layout consistency, and coordination among parallel instances; local syntax repair need not restore these invariants. [§5, Insight 1](../pdfs/02-kernelbench-x.md)
- Repair transitions mean incorrect-to-correct adjacent iterations; optimization events improve runtime; regression events lose a previously passing metric. [Appendix E](../pdfs/02-kernelbench-x.md)

## Methodology and experimental setup

- The suite has 176 tasks in 15 categories: Activation, Convolution, Fusion, Index, LinearAlgebra, Loss, Math, MatrixMultiply, Normalization, Optimizer, Pooling, Quantization, Random, Reduce, SpatialOps. [§3.2.1; Table 3](../pdfs/02-kernelbench-x.md)
- Large categories are Fusion (60), Math (36), and LinearAlgebra (17); several categories have only two to six tasks, making individual outcomes influential. [Appendix B.1, Table 3](../pdfs/02-kernelbench-x.md)
- Selected fp16, bf16, and int8 variants extend precision coverage. Six quantization tasks cover W8A8 matmul, batched matmul, convolution, layer normalization, attention, and W4A16 linear. [§3.2.1; Appendix C](../pdfs/02-kernelbench-x.md)
- Standard inputs follow N(0,1). The outlier mode injects values with probability 0.1% and scale factor 50 to expose distribution-sensitive numerical errors. [§3.2.2](../pdfs/02-kernelbench-x.md)
- Quantization call checks reject prohibited high-level APIs and look for manual scale computation and explicit casting/discretization. Static compliance is separate from numerical success. [§3.2.2; Appendix C](../pdfs/02-kernelbench-x.md)
- Matmul, batched matmul, convolution, and layernorm quantization require cosine ≥0.95, relative L1 ≤0.05, RMSE ≤0.10; attention and W4A16 linear use ≥0.90, ≤0.10, ≤0.15 respectively. [Appendix C, Table 6](../pdfs/02-kernelbench-x.md)
- AutoTriton is evaluated in single-pass mode with native prompting after its domain-specific training. [§2.2](../pdfs/02-kernelbench-x.md)
- GEAK uses DeepSeek-V3.2-Chat, three iterations, temperature 1.0, four candidates per round, and five retained best implementations as context. [§2.2](../pdfs/02-kernelbench-x.md)
- KernelAgent also uses DeepSeek-V3.2-Chat; the Fuser pipeline is bypassed, with three parallel workers, up to five refinement rounds each, temperature 0.4. [§2.2](../pdfs/02-kernelbench-x.md)
- Claude and the general code-model baseline are single-pass comparators; these protocols do not equate inference budgets across methods. [§2.2](../pdfs/02-kernelbench-x.md)
- Hardware comprises RTX 5090, RTX 4090, A100 PCIe 40GB, H20, H800 PCIe, and L20. Each target has its own remeasured PyTorch eager baseline. [§4.1](../pdfs/02-kernelbench-x.md)
- The reported software stack is Python 3.11, CUDA 11.8, PyTorch 2.10.0+cu128, Triton 3.6.0; this is the paper's stated combination, not a reproduced environment. [§4.1](../pdfs/02-kernelbench-x.md)
- Timing uses triton.testing.do_bench with the reported 25 warmup and 100 measurement runs and median runtime. Performance summaries condition on semantically correct kernels. [§3.2.3; §4.1](../pdfs/02-kernelbench-x.md)
- Transition records retain before/after pass indicators, timing/speedup, aligned errors, and unified diffs, supporting analysis of error-to-patch-to-pass sequences. [Appendix E](../pdfs/02-kernelbench-x.md)

## Results and evidence

- Table 1 reports compile/correct percentages of 36.4/17.0 for AutoTriton, 68.8/30.7 for GEAK, 64.2/10.8 for KernelAgent, 45.5/22.7 for Claude, and 1.7/0.0 for its KernelLLM row. [Table 1](../pdfs/02-kernelbench-x.md)
- KernelAgent's Table 1 correct/compile conversion is only 16.8%; successful compilation is therefore a weak success proxy in this configuration. [§4.2; Table 1](../pdfs/02-kernelbench-x.md)
- Correct-only mean speedups in Table 1 are AutoTriton 1.35×, GEAK 1.15×, KernelAgent 1.41×, Claude 1.26×; a high conditional speedup can accompany low coverage. [Table 1; Appendix A.4](../pdfs/02-kernelbench-x.md)
- Method versus category explain similar compile-success deviance (5.18% versus 5.24%), but category explains more semantic-correctness deviance (9.4% versus 3.3%). [§4.3](../pdfs/02-kernelbench-x.md)
- Adding category to method reduces deviance by 34.6; adding method to category reduces it by 13.0. These are fitted explanatory statistics on this task set, not causal effects. [§4.3](../pdfs/02-kernelbench-x.md)
- The authors report 72% of Fusion tasks fail across all five methods; Appendix B nevertheless records some Fusion successes, particularly GEAK at 23.3%. [Abstract; Appendix B.1, Table 3](../pdfs/02-kernelbench-x.md)
- Quantization has 0/30 task-method successes (six tasks × five methods), despite nonzero compilation for GEAK, KernelAgent, and Claude. SpatialOps also has zero correctness across methods. [Abstract; Tables 3–4](../pdfs/02-kernelbench-x.md)
- Averaged over four nontrivial methods, compile-to-correct conversion is 55.8% for Math, 46.4% Activation, 24.8% Fusion, 25.0% MatrixMultiply, and zero Quantization/SpatialOps. [Table 2](../pdfs/02-kernelbench-x.md)
- Static reference-code proxies correlate only modestly with pooled correctness failure: assignment/fusion-call counts r=0.2114, cyclomatic complexity r=0.1454, logical-span proxy r=0.1453. [Appendix B.2, Table 5](../pdfs/02-kernelbench-x.md)
- GEAK's iteration analysis reports compile rate 52.3→68.8%, correctness 18.2→30.7%, mean speedup 1.58→1.44×, and efficiency score 62.7→53.3%. [§4.4; Figure 3](../pdfs/02-kernelbench-x.md)
- For rounds 0→1, newly rescued correct kernels average 1.16× versus 1.58× for already-correct kernels; rounds 1→2 show 1.32× versus 1.46×. [§4.4](../pdfs/02-kernelbench-x.md)
- Across 352 adjacent GEAK diffs, dominant categories are no substantial change (102), mask fixes (101), delegated-operation introduction/removal (65), and dtype/casting fixes (36); optimization rewrites are described as rare. [§4.4](../pdfs/02-kernelbench-x.md)
- Across correct kernels, 46.6% are slower than eager PyTorch and pooled median speedup is 1.0008×; correctness alone gives little assurance of acceleration. [§4.5](../pdfs/02-kernelbench-x.md)
- Cross-GPU max/min speedup ratio has median 2.15×, mean 2.73×, and maximum 21.4×. The abstract's word “variance” refers here to this portability spread, not a statistical variance estimate. [§4.5; Figure 4](../pdfs/02-kernelbench-x.md)
- The fraction of correct kernels slower than baseline ranges from 18% on A100 to 76% on L20, illustrating dependence on both target hardware and its baseline. [§4.5](../pdfs/02-kernelbench-x.md)
- The logit example reaches approximately 3.35× on RTX 4090 for GEAK, Claude, and AutoTriton; it uses lane-local clipping and no cross-instance coordination. [§4.6.1](../pdfs/02-kernelbench-x.md)
- In fused_exp_mean, masked padding is zero before exponentiation, so invalid lanes contribute exp(0)=1. Locally plausible pieces violate the global reduction contract. [§4.6.2](../pdfs/02-kernelbench-x.md)
- For expand_where, GEAK progresses from compilation failure to wrong broadcast mapping to correct execution, but final speedup is only 0.076× because coordinate decomposition and operand-specific shape/stride work remain expensive. [§4.6.3](../pdfs/02-kernelbench-x.md)

## Authors’ conclusions

- The authors conclude capability is category-structured, especially around global semantic coordination, rather than captured adequately by one overall model ranking. [§5, Insight 1; §6](../pdfs/02-kernelbench-x.md)
- They interpret current refinement as repair-biased: explicit local error signals improve feasibility but provide insufficient guidance for plan-level optimization. [§5, Insight 2](../pdfs/02-kernelbench-x.md)
- They treat hardware efficiency and portability as separate unresolved frontiers beyond compilation and correctness. [§5, Insight 3; §6](../pdfs/02-kernelbench-x.md)
- Proposed directions include global-contract reasoning, numerical-fidelity training, hardware-context conditioning, profiling signals, and tuning block size/warps/pipeline stages. These are proposals, not validated interventions in this benchmark. [§6; Appendix D.4](../pdfs/02-kernelbench-x.md)
- Appendix D attributes failures partly to performance-unannotated training data and absent hardware context. These are the authors' explanatory hypotheses rather than controlled training-data causal experiments. [Appendix D.1–D.3](../pdfs/02-kernelbench-x.md)

## Limitations and open questions

- Analyst caveat: the small task counts in several categories and shared task provenance limit generalization of category-level rates. The attribution analysis is associational. [Appendix A.1; §4.3](../pdfs/02-kernelbench-x.md)
- Analyst caveat: different models, prompting, iterations, and worker budgets confound a clean causal comparison of agent designs. [§2.2](../pdfs/02-kernelbench-x.md)
- The task-level FLOP/byte model is explicitly an idealized efficiency proxy; do not interpret IOU/MFU as actual hardware-counter utilization. [§3.2.3](../pdfs/02-kernelbench-x.md)
- Analyst caveat: two input modes and static compliance checks do not prove semantic equivalence, exhaustive quantization correctness, or freedom from delegation loopholes. [§3.2.2](../pdfs/02-kernelbench-x.md)
- Source inconsistency: prose names DeepSeek-Coder, while Tables 1, 3, and 4 name KernelLLM. Preserve this ambiguity rather than silently identifying them as one documented model. [§2.2; Tables 1, 3–4](../pdfs/02-kernelbench-x.md)
- Source inconsistency: GEAK's overall Table 1 speedup is 1.15×, while the final iteration discussion gives 1.44×; the shown text does not clearly reconcile aggregation/selection differences. [Table 1; §4.4](../pdfs/02-kernelbench-x.md)
- Source caution: §4.3 describes Loss rates above 0.8, whereas Table 3 has a maximum of 50%; use the table's explicit method/category values when reporting rates. [§4.3; Table 3](../pdfs/02-kernelbench-x.md)
- Source caution: statements that further iteration cannot reverse decline or that no untrained model can use profiling exceed the bounded experiments. Treat them as author interpretation, not demonstrated impossibility. [§4.4; Appendix D.3–D.4](../pdfs/02-kernelbench-x.md)

## Relevance to this project — interpretation

- Interpretation: store separate compile, compliance, numerical-pass, speedup, and hardware records per candidate; never collapse them into a single successful/failed label. [§3.2; §4.2](../pdfs/02-kernelbench-x.md)
- Interpretation: maintain task tags for broadcasting, reductions, precision contracts, and fusion boundaries to support targeted test generation and failure retrieval. [Appendix A.1; §4.6](../pdfs/02-kernelbench-x.md)
- Interpretation: log fixed-task iteration trajectories as well as changing solved-task aggregates; newly rescued slow tasks can lower an aggregate without slowing previously solved kernels. [§4.4](../pdfs/02-kernelbench-x.md)
- Interpretation: evaluate a dedicated performance-search phase only after correctness gating; compare it under matched budgets rather than assuming extra repair rounds optimize. [§5; Appendix D.4](../pdfs/02-kernelbench-x.md)

## Retrieval notes

- Aliases: KernelBenchX, KernelBench-X; retrieval topics include Call Accuracy, Execution Accuracy, IOU, MFU, outlier testing, quantization contracts, repair bias, and cross-GPU portability.
- Questions answered: why compilation is insufficient; how quantization is tested; why masking before nonlinear reduction fails; whether iteration gains arise from repair; how hardware changes speedup.
- Navigation: [TritonBench](05-tritonbench.md) for the benchmark lineage and [KernelBench](01-kernelbench.md) for a distinct workload/metric design. Cross-paper comparisons require reading their own protocols.
- Navigation: [KernelBench-Verified](03-kernelbench-verified.md) for another correctness/baseline audit; this is a related-source pointer, not a claim of equivalent protocols.

## Source coverage and verification notes

- Reviewed the complete substantive main text (§§1–6) and Appendices A–E, including task taxonomy, quantization thresholds, transition artifacts, all six numbered tables, and three kernel case studies. [§§1–6; Appendices A–E](../pdfs/02-kernelbench-x.md)
- Captured all major concepts and experiments; individual bibliography entries and complete example source listings are not duplicated. Original code remains in the source Markdown. [§4.6; References](../pdfs/02-kernelbench-x.md)
- Numerical extraction is based on the preserved textual tables and figure captions/prose; plots were not independently digitized and experiments were not reproduced.
- Recorded naming/aggregation inconsistencies above instead of repairing the original source or manufacturing a reconciliation.

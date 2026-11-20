---
paper_id: "03"
title: "KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?"
source_version: "2607.16241v1"
source_markdown: "../pdfs/03-kernelbench-verified.md"
source_pdf: "../pdfs/03-kernelbench-verified.pdf"
source_url: "https://arxiv.org/abs/2607.16241"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "benchmarking", "tf32", "hidden-tests", "reward-hacking", "numerical-accuracy", "memory-efficiency", "llm-as-judge", "bf16"]
summary: "Re-evaluates generated KernelBench candidates using TF32-enabled eager baselines, transformed hidden inputs, memory metrics, and a residual-hack code audit."
---

# KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

## Paper identity and scope

- The source is arXiv 2607.16241v1, a KernelBench evaluation-methodology extension rather than a new kernel-generation algorithm. [Abstract; §1](../pdfs/03-kernelbench-verified.md)
- Its three interventions are baseline acceleration, stronger correctness gating, and peak-memory measurement; these measure different failure modes. [§1](../pdfs/03-kernelbench-verified.md)
- The main results concern single-turn CUDA generation on H200, with five independent samples per task; they do not evaluate an iterative optimization agent. [§2.1](../pdfs/03-kernelbench-verified.md)
- The nominal suite has 250 tasks, but three level-2 tasks are excluded, leaving 100/97/50 active tasks by level (247 active). Some later analyses explicitly retain a 250-task denominator. [§2.1; Appendix E; Appendix I](../pdfs/03-kernelbench-verified.md)

## Concepts and terminology

- Baseline mismatch occurs when a generated implementation accesses acceleration that the reference leaves disabled; the resulting ratio can overstate code-generation improvement. [§1; §2.2](../pdfs/03-kernelbench-verified.md)
- TF32-enabled eager PyTorch changes the FP32 matrix/convolution execution baseline; it does not constitute TorchInductor fusion or a torch.compile comparison. [§2.2; Appendix B](../pdfs/03-kernelbench-verified.md)
- Hidden testing varies numerical values while preserving shapes to distinguish valid fixed-shape specialization from shortcuts exploiting observed value distributions. [§2.3; Appendix K](../pdfs/03-kernelbench-verified.md)
- D1 is the original input; D2 scales floating tensors by 3; D3 scales by 0.01; D4 negates. These target magnitude-sensitive and sign-sensitive failures. [§2.3](../pdfs/03-kernelbench-verified.md)
- Transformations preserve integer tensors, scalar parameters, zero entries, and structural patterns; they do not test arbitrary layouts or shapes. [§2.3](../pdfs/03-kernelbench-verified.md)
- A candidate must pass all retained distribution configurations; performance timing remains on the original distribution rather than timing every hidden distribution. [§2.3](../pdfs/03-kernelbench-verified.md)
- Correctness is best-of-five coverage: at least one passing sample per task. Correct Speedup is the geometric mean of fastest-correct-sample speedups over solved tasks. [§2.5](../pdfs/03-kernelbench-verified.md)
- Fast@1 here means the fraction of tasks with speedup greater than 1× among best-of-five candidates; the “1” is a speed threshold, not one generation sample. [§2.5; Table 1 caption](../pdfs/03-kernelbench-verified.md)
- Correct Mem Efficiency is geometric mean baseline/candidate peak memory over correct problems. Values greater than one mean lower candidate allocation. [§2.5](../pdfs/03-kernelbench-verified.md)
- Mem Efficient% counts tasks with strictly lower candidate peak memory; it is not interchangeable with the proportion increasing memory because equality and failed tasks may intervene. [§2.5; Appendix C](../pdfs/03-kernelbench-verified.md)
- PyTorch max_memory_allocated tracks peak allocation associated with its allocator; the paper explicitly distinguishes allocation from memory traffic, which requires a hardware profiler. [§2.4; Listing 2](../pdfs/03-kernelbench-verified.md)
- Reward-hack audit labels include Legitimate, Speed_Hack, Memory_Hack, Speed_and_Memory_Hack, and Uncertain; the labels are judgments that still require semantic review. [Appendix L.1](../pdfs/03-kernelbench-verified.md)
- Input-blind regeneration removes executable test generators and exact configuration variables while retaining the Model definition and manually supplied symbolic shape annotations. [Appendix L.4](../pdfs/03-kernelbench-verified.md)
- Conditional means change when correctness thresholds exclude tasks; higher speedup after stricter filtering need not imply any kernel became faster. [Appendix J](../pdfs/03-kernelbench-verified.md)

## Methodology and experimental setup

- Seven evaluated models are GPT-5.5, Claude Opus 4.8/4.7, Claude Sonnet 4.6, Kimi K2.6, Gemini 3 Flash Preview, and Gemini 3.1 Pro Preview. [§2.1](../pdfs/03-kernelbench-verified.md)
- Generation uses the standard one-turn KernelBench prompt with original PyTorch code, five samples per task, and each provider's documented-in-paper default reasoning setting. [§2.1; Appendix B](../pdfs/03-kernelbench-verified.md)
- The paper specifies high effort for Claude/Gemini, medium for GPT-5.5, and enabled default thinking for Kimi; these are not resource-matched settings. [Appendix B](../pdfs/03-kernelbench-verified.md)
- The pipeline compiles via cpp_extension.load_inline, checks outputs, measures timing, measures allocation, and clears extension/cache state between evaluations. [Appendix B](../pdfs/03-kernelbench-verified.md)
- FP32 allclose uses both atol and rtol 1e-3. BF16 uses 1e-2; results cannot be compared without acknowledging this accuracy change. [Appendix B; Appendix F](../pdfs/03-kernelbench-verified.md)
- Timing uses CUDA events, three warmup iterations and ten timed trials, reporting mean GPU execution latency. [Appendix B](../pdfs/03-kernelbench-verified.md)
- Peak memory is recorded after resetting statistics and executing a forward pass; memory traffic, registers, cache behavior, and shared-memory occupancy are outside the measured metric. [§2.4; Appendix B](../pdfs/03-kernelbench-verified.md)
- Before hidden testing, reference configurations producing NaN, Inf, or errors are dropped; the source reports two exclusions among 1,000 configurations: P90/D2 and P98/D4. [§2.3 footnote](../pdfs/03-kernelbench-verified.md)
- The factorial comparison crosses FP32 versus TF32 baseline with standard versus hidden correctness tests, isolating their respective contributions to reported speedup. [§3.2](../pdfs/03-kernelbench-verified.md)
- The per-problem “TF32 affected” criterion is at least a 50% relative speedup drop for at least one evaluated model, conditional on original correctness. [Appendix E](../pdfs/03-kernelbench-verified.md)
- The “hidden affected” criterion is standard-test correctness changing to failure on any hidden distribution for at least one model. The across-model union is not a per-model failure rate. [Appendix E](../pdfs/03-kernelbench-verified.md)
- BF16 experiments regenerate candidates with a BF16 instruction and cast reference parameters/inputs accordingly; they are not simply retimings of identical FP32 binaries. [Appendix F](../pdfs/03-kernelbench-verified.md)
- Residual auditing covers 453 best-of-five winning candidates faster than TF32 and passing hidden tests, drawn from five models according to Appendix L.1. [Appendix L.1](../pdfs/03-kernelbench-verified.md)
- Claude Sonnet 4.6 and GPT-5.5 independently inspect reference code, candidate code, speedup, and memory ratio; intersection flags receive manual review. [Appendix L.1–L.2](../pdfs/03-kernelbench-verified.md)

## Results and evidence

- The headline GPT-5.5 correct-only geometric speedup declines from 1.43× under standard evaluation to 0.88× under the verified FP32/TF32 protocol. [Abstract; §3.1](../pdfs/03-kernelbench-verified.md)
- Table 1 gives GPT-5.5 speedups 0.90/0.85/0.92× and correctness 99.0/99.0/96.0% across levels 1/2/3; high tested correctness does not imply fast code. [Table 1](../pdfs/03-kernelbench-verified.md)
- Its Fast@1 rates remain 52.0/64.9/32.0%, showing many individual wins despite sub-parity geometric means. “No model beats PyTorch” must be restricted to the stated aggregate metric. [Table 1](../pdfs/03-kernelbench-verified.md)
- Pooled correct speedups in Table 3 are GPT-5.5 0.88×, Gemini Pro 0.80×, Gemini Flash 0.48×, Opus4.8 0.60×, Kimi 0.47×, Opus4.7 0.57×, Sonnet 0.61×. [Appendix C, Table 3](../pdfs/03-kernelbench-verified.md)
- GPT-5.5 pooled correctness is 98.4% and Fast@1 53.0%; these metrics have different denominators/conditions from the solved-task geometric mean. [Appendix C, Table 3](../pdfs/03-kernelbench-verified.md)
- For GPT-5.5 level 2, TF32 alone changes 1.67→0.88× and hidden gating further changes 0.88→0.85×; baseline choice dominates this reported decomposition. [§3.2; Figure 1](../pdfs/03-kernelbench-verified.md)
- At level 1, hidden gating changes 1.37→1.17×; the paper attributes the larger relative hidden-test effect to sign-sensitive elementwise shortcuts. [§3.2; Figure 1](../pdfs/03-kernelbench-verified.md)
- The 250-task union decomposition is 53 TF32-only, 29 hidden-only, six both, 162 neither: 21.2%, 11.6%, 2.4%, 64.8%. [Appendix E, Table 4](../pdfs/03-kernelbench-verified.md)
- Level-wise TF32-only/hidden-only/both counts are 16/26/0 for L1, 32/2/5 for L2, and 5/1/1 for L3. These use the explicit nominal-suite denominator. [Appendix E, Table 4](../pdfs/03-kernelbench-verified.md)
- GPT-5.5 memory efficiency is 1.24/1.14/0.88 across levels, with 82.0/63.9/36.0% of tasks strictly more memory-efficient. Architecture-level memory savings are less common. [Table 2](../pdfs/03-kernelbench-verified.md)
- The headline says 28% of the best model's correctly generated kernels increase memory; Table 3 separately reports 65.6% of all problems strictly memory-efficient. Do not treat these as complementary percentages. [Abstract; Appendix C, Table 3](../pdfs/03-kernelbench-verified.md)
- Under BF16, GPT-5.5 correct-only speedups are 0.90/1.63/1.12× with correctness 93.0/84.5/82.0%; the paper's broad FP32 conclusion does not hold uniformly in BF16. [Appendix F, Table 5](../pdfs/03-kernelbench-verified.md)
- Gemini Pro reaches BF16 1.24× at L2 and 1.01× at L3; Gemini Flash reaches 1.19× and 1.05×. Several models show gains beyond FP32/TF32 outcomes. [Appendix F, Table 5](../pdfs/03-kernelbench-verified.md)
- The authors associate BF16 improvements with changing compute/memory balance, while also noting lower correctness and new mixed-precision/denormal failure modes. [Appendix F](../pdfs/03-kernelbench-verified.md)
- Negation catches the largest reported failure share; per-distribution failure bars overlap, and about 30% of D4 failures also appear under D2 or D3. [Appendix G; Figure 5](../pdfs/03-kernelbench-verified.md)
- A shape-guarded ReLU identity path reports 374× under positive inputs but fails negative inputs with max absolute difference 1.0; its fallback computation is valid outside the guarded shape. [Appendix H.1](../pdfs/03-kernelbench-verified.md)
- A fused GroupNorm example uses E[X²]−E[X]² and passes standard inputs with error <1e-4 but gives max differences 0.236 under D2 and 0.089 under D3. [Appendix H.3](../pdfs/03-kernelbench-verified.md)
- A vectorized fused GELU reports 8.52× speedup, candidate/reference memory ratio 0.48, and max difference <1e-5 across all four distributions; TF32 is irrelevant to its purely elementwise arithmetic. [Appendix H.4](../pdfs/03-kernelbench-verified.md)
- A pre-transposed MLP example reports 0.26× speedup and 1.81× candidate/reference memory, illustrating a purported optimization that duplicates weights and targets already-cheap library work. [Appendix H.5](../pdfs/03-kernelbench-verified.md)
- For L2 P94 the paper reports 0.21× FP32/TF32 versus 1.70× BF16 and attributes the contrast to fused post-GEMM operations and precision-dependent bottlenecks. [Appendix F; Appendix H.2](../pdfs/03-kernelbench-verified.md)
- Three excluded tasks are P23, P80, P83, described as constant-zero references; GPT-5.5 reports 71×, 34×, 235× when exploiting these identities. [Appendix I](../pdfs/03-kernelbench-verified.md)
- Input-dependent algebraic simplifications for P18/P42/P44 remain included; the authors distinguish constant-output tasks from legitimate reduction of redundant computation. [Appendix I](../pdfs/03-kernelbench-verified.md)
- Tightening tolerance from 1e-3 to 1e-4 reduces GPT-5.5 L1 correctness 99→83% while raising its conditional speedup 0.90→1.25×; the authors explain this as task-selection bias. [Appendix J, Table 6](../pdfs/03-kernelbench-verified.md)
- Audit outcomes are six intersection flags, eight Claude-only flags, 34 GPT-only flags, and 405 neither-flagged candidates. Manual review reclassifies two intersection flags as legitimate specialization, leaving four designated genuine hacks. [Appendix L.2, Table 7](../pdfs/03-kernelbench-verified.md)
- Wrong-axis hinge-loss broadcasting survives random tests because the large i.i.d. aggregate is statistically similar for row and column indexing. [Appendix L.3.1](../pdfs/03-kernelbench-verified.md)
- Other flagged cases involve early termination of cumulative products after underflow, replacing global VLAD normalization with 1/√K, and replacing random LSTM initial states with zeros. [Appendix L.3.2; L.3.5–L.3.6](../pdfs/03-kernelbench-verified.md)
- Hardcoding channel count 64 and embedding dimension 128 is judged legitimate for the declared fixed configurations after manual review, despite both LLM judges flagging these candidates. [Appendix L.3.3–L.3.4; L.4](../pdfs/03-kernelbench-verified.md)
- The four designated genuine hacks are reported resolved after selective input-blind regeneration; this establishes case-specific repair, not exhaustive verification of the regenerated suite. [Appendix L.4; Table 8](../pdfs/03-kernelbench-verified.md)

## Authors’ conclusions

- Authors conclude performance claims are highly sensitive to baseline acceleration and should track realistic reference configurations. [§4](../pdfs/03-kernelbench-verified.md)
- They argue hidden input testing must evolve with model behavior, especially when public tests expose narrow distributions. [§4](../pdfs/03-kernelbench-verified.md)
- They advocate joint speed and memory measurement because correct code can trade one resource against another. [§4](../pdfs/03-kernelbench-verified.md)
- They present TF32 recalibration and distribution gating as complementary interventions, rather than interchangeable correctness/performance fixes. [§3.2; §4](../pdfs/03-kernelbench-verified.md)
- They explicitly view BF16 findings as evidence against interpreting FP32 deflation as a fundamental impossibility of useful LLM kernel optimization. [§3.5; Appendix F](../pdfs/03-kernelbench-verified.md)

## Limitations and open questions

- Explicit limitation: one-turn sampling excludes compiler/profiler-guided search. Feeding hidden-test failures into a future agent loop could itself leak the test distribution. [§4, Limitations](../pdfs/03-kernelbench-verified.md)
- Explicit limitation: four deterministic transformations are anticipatable and incomplete; the authors propose candidate-aware adversarial test generation as future work. [§4, Limitations](../pdfs/03-kernelbench-verified.md)
- Explicit limitation: results are limited to inference on one H200 architecture; training dynamics, alternative devices, and new problem scales are untested. [§4, Limitations](../pdfs/03-kernelbench-verified.md)
- Explicit limitation: baseline configuration changes penalize apparent wins from merely enabling cuBLAS acceleration, shaping the evaluation narrative. [§4, Limitations](../pdfs/03-kernelbench-verified.md)
- Source inconsistency: §2.3 says hidden failures are excluded from speed and memory aggregates, but Appendix C/Table 3 explicitly says memory uses best-of-five standard correctness. Keep memory and hidden-gated speed populations separate until implementation is checked. [§2.3; Appendix C](../pdfs/03-kernelbench-verified.md)
- Source ambiguity: descriptions alternate between “all 250” and 247 active tasks after exclusion. Table 4 explicitly uses 250; do not invent a common denominator across every result. [§2.1; Appendices C, E, I](../pdfs/03-kernelbench-verified.md)
- Source caution: Table 6's 1e-3 entries differ slightly from Table 1 for some models. These notes retain table-specific locators instead of assuming identical evaluated candidate sets. [Tables 1, 6](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: TF32 entails an accuracy contract, not free exact-FP32 equivalence. The paper's precision explanation and claimed generic tolerance bound should not be reused as authoritative numerical-analysis facts without independent verification. [§2.2; Appendix J](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: the P94 timing explanation does not fully derive its large reported speedup reversal; treat 0.21×/1.70× as reported observations, not a reproduced causal decomposition. [Appendix H.2](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: intersecting two judges' flags misses single-judge concerns. The eight Claude-only algebraic issues and 34 GPT-only semantic/edge cases prevent treating the four confirmed cases as an exhaustive bug count. [Appendix L.2](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: the cumulative-product example's correctness depends on accumulation order, underflow, and NaN/Inf semantics. The stated sustained->1 argument alone does not prove an early-exit bug once an implementation's accumulator is exactly zero; verify executable counterexamples before repeating the classification as fact. [Appendix L.3.2](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: the paper's constant-output derivations must be checked against actual reference parameters and affine normalization behavior before applying exclusions to another KernelBench checkout. [Appendix I](../pdfs/03-kernelbench-verified.md)
- Analyst caveat: the paper labels some behavior deliberate exploitation, but source inspection establishes a shortcut and its failure conditions more directly than the model's intent. [Appendix H.1; Appendix L](../pdfs/03-kernelbench-verified.md)

## Relevance to this project — interpretation

- Interpretation: make dtype, permitted approximation, Tensor Core settings, baseline mode, and candidate delegation policy explicit in every experiment. [§2.2; Appendix B; Appendix F](../pdfs/03-kernelbench-verified.md)
- Interpretation: separate generation-visible shapes from evaluator-only values, and retain structured non-i.i.d. tests to expose broadcast/reduction errors hidden by averaging. [§2.3; Appendix L.3.1](../pdfs/03-kernelbench-verified.md)
- Interpretation: report overall coverage, strict-speed wins, conditional geometric mean, and allocation separately; retain equality/failure categories and exact denominators. [§2.5; Appendix C](../pdfs/03-kernelbench-verified.md)
- Interpretation: use LLM audits to propose tests or review targets, with human/executable adjudication of shape specialization and floating-point edge semantics. [Appendix K; Appendix L](../pdfs/03-kernelbench-verified.md)
- Interpretation: preserve source contradictions in experiment metadata instead of making benchmark prose the unquestioned implementation specification. [Appendices C, J, L](../pdfs/03-kernelbench-verified.md)

## Retrieval notes

- Aliases: KernelBench-Verified, TF32 baseline mismatch, four-distribution hidden testing, input-blind generation, speed–memory tradeoff, tolerance selection bias.
- Questions answered: what verified means here; why best-of-five can have high correctness but low geometric speedup; why fixed-shape specialization is legitimate; what hidden tests still miss; why BF16 changes conclusions.
- Navigation: [KernelBench](01-kernelbench.md) defines the original metric/task family; [KernelBenchX](02-kernelbench-x.md) studies category and repair behavior. No direct leaderboard comparison is justified without matching protocols.
- Navigation only: [LARK](22-lark.md) concerns formal guarantees and [bit-accurate MMA modeling](09-bit-accurate-gpu-mma.md) concerns hardware arithmetic; neither is implied by the word verified in this benchmark.

## Source coverage and verification notes

- Reviewed main §§1–4 and substantive Appendices A–L, including all eight numbered tables, hidden-test definitions, memory-gating qualifications, BF16/tolerance ablations, exclusions, six residual-audit cases, and regeneration policy. [§§1–4; Appendices A–L](../pdfs/03-kernelbench-verified.md)
- Quantitative claims use preserved text tables/captions and case-study prose; complete code remains in original Markdown. Plots were not digitized and no kernels were executed.
- Coverage intentionally distinguishes paper-reported conclusions from analyst cautions, especially unresolved memory populations, task denominators, numerical explanations, and judge-based hack labels.
- Bibliographic metadata in the source manifest splits Max (Xiangjun) Fan into separate author strings; use the visible paper author line when author identity matters. [Paper title/author block](../pdfs/03-kernelbench-verified.md)

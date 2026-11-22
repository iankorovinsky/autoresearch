---
paper_id: "30"
title: "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization"
source_version: "2603.12440v2"
source_markdown: "../pdfs/30-kernelfoundry.md"
source_pdf: "../pdfs/30-kernelfoundry.pdf"
source_url: "https://arxiv.org/abs/2603.12440"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["gpu-kernels", "evolutionary-search", "quality-diversity", "map-elites", "prompt-evolution", "hardware-awareness", "sycl", "cuda", "kernelbench", "execution-feedback", "benchmarking"]
summary: "KernelFoundry combines a 64-cell MAP-Elites kernel archive, transition-derived mutation hints, meta-prompt evolution and templated parameter tuning for SYCL and CUDA kernels; results rest on sampled relative-tolerance correctness and report no repeated runs or variance estimates."
---

# KernelFoundry: Hardware-aware evolutionary GPU kernel optimization

## 1. Paper identity and scope

- This note covers arXiv `2603.12440v2`, a preprint by five Intel authors; the source states that code is available at `github.com/isl-org/kernelfoundry`. [Title block; Data and code availability](../pdfs/30-kernelfoundry.md)
- The work asks how to keep an LLM generate-verify-measure loop for GPU kernels from two failure modes it names: mode collapse (refinement acting as greedy local search around recent successes) and context degradation (failed attempts dominating a growing prompt history). [§1](../pdfs/30-kernelfoundry.md)
- The stated contributions are (1) MAP-Elites quality-diversity search with kernel-specific behavioural dimensions, (2) meta-prompt evolution that co-evolves prompt sections with kernels, and (3) templated kernels whose parameters are tuned to inputs and hardware. [Abstract; §1](../pdfs/30-kernelfoundry.md)
- The primary target is SYCL on Intel GPUs; CUDA kernels are generated for comparison with prior work. Triton is listed as a supported backend, but no Triton result appears in any table. [§4, Programming paradigm and hardware; Tables 1–2](../pdfs/30-kernelfoundry.md)
- Evaluation uses a filtered KernelBench set of 111 tasks (80 Level 1, 31 Level 2), a 40-task representative subset (20 L1, 20 L2), 12 robust-kbench tasks, five operations compared with oneDNN, and one operation from Llama 3.2 1B. [§4, Benchmarks; §5.4; §5.5](../pdfs/30-kernelfoundry.md)
- The filtered and representative sets contain only Level 1 and Level 2 tasks; no KernelBench Level 3 (full architecture) result is reported, although the introduction describes the suites as spanning full model architectures. [§1; §4, Benchmarks](../pdfs/30-kernelfoundry.md)
- Correctness throughout is numerical comparison against a reference on tested inputs. The source does not prove kernels correct and lists "improved formal verification to eliminate reward hacking" as future work. [§4, Metrics; §6](../pdfs/30-kernelfoundry.md)
- The work is a search scaffold around existing API and local models; it does not fine-tune a model, and it names RL fine-tuning as a separate promising avenue. [§2, LLM-Based Kernel Generation; §6](../pdfs/30-kernelfoundry.md)

## 2. Concepts and terminology

### Quality-diversity search

- Quality-diversity (QD) algorithms keep a collection of high-performing solutions in separate behavioural cells instead of optimizing one objective, so that "stepping-stone" solutions can help escape local optima. [§3.2](../pdfs/30-kernelfoundry.md)
- MAP-Elites partitions the solution space into a discrete grid by behavioural descriptors, defined as low-dimensional features that characterize a solution independently of its fitness; each cell holds the highest-fitness solution found for that region. [§3.2, MAP-Elites Algorithm](../pdfs/30-kernelfoundry.md)

### Archive: descriptors, cell contents, selection and replacement

- The archive has three behavioural dimensions, each with four discrete levels, giving \(4^3 = 64\) cells. [§3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- Memory access pattern `d_mem`: 0 scalar, strided or uncoalesced access; 1 coalesced/vectorized (vec4, aligned loads); 2 shared/local memory with explicit tiling; 3 multi-level hierarchy (shared local memory plus register blocking plus prefetch). [§3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- Algorithmic structure `d_algo`: 0 direct PyTorch translation; 1 fused operations in a single pass; 2 reformulated algorithm (online normalization, flash pattern); 3 novel or asymptotically improved algorithm. [§3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- Parallelism coordination `d_sync`: 0 no synchronization; 1 work-group barriers; 2 sub-group primitives (shuffles, reductions, broadcasts); 3 global coordination (atomics, multi-pass with synchronization). [§3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- Coordinates are computed deterministically from generated code by static pattern matching on SYCL and CUDA constructs. The classifier uses weighted, category-specific patterns to avoid double counting; for example, a barrier used for shared-local-memory synchronization is credited to `d_mem`, not also to `d_sync`. [§3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- Each occupied cell retains one kernel: the best-performing one (the elite) for that cell. [§3.1; §3.2, MAP-Elites Algorithm](../pdfs/30-kernelfoundry.md)
- Fitness is 0 if compilation fails, 0.1 if the kernel compiles but is incorrect, and \(0.5 + 0.5\,s_{\text{norm}}\) if correct, with \(s_{\text{norm}} = \min(1, \text{speedup}/\text{target})\) and a default target of 2x over the PyTorch baseline. [§3.2, Fitness Function; Table 6](../pdfs/30-kernelfoundry.md)
- Each iteration has four phases: selection of parent kernels from the archive, variation by LLM code modification, evaluation (compile, correctness, timing, coordinate assignment), and insertion. [§3.2, MAP-Elites Algorithm](../pdfs/30-kernelfoundry.md)
- Replacement rule: an offspring is inserted if its target cell is empty or if it improves on that cell's incumbent elite; otherwise it is discarded. Candidates that run correctly receive coordinates, while outcomes of all candidates, including failures, feed the gradient estimator and meta-prompter. [§3.1; §3.2, MAP-Elites Algorithm](../pdfs/30-kernelfoundry.md)
- Four parent-selection strategies with configurable mixing ratios are implemented: uniform over occupied cells, fitness-proportionate, curiosity-driven (cells with high estimated improvement potential from the gradient signal), and island-based (K sub-populations with migration every M generations). The default listed is curiosity-driven; K, M and mixing ratios are not given. [§3.2, Selection Strategies; Appendix B.4, Table 6](../pdfs/30-kernelfoundry.md)

### Gradient-informed evolution

- "Gradients" here are statistics over recent parent-to-child transitions in the discrete descriptor grid; the source calls them gradient-like signals and a soft steering mechanism, added because individual cells can stagnate. [§3.3](../pdfs/30-kernelfoundry.md)
- A circular buffer stores, per transition, parent and child coordinates, the fitness delta, a timestamp and iteration number for temporal weighting, and an outcome label: improvement (child becomes an elite or opens a new cell), neutral (competitive but no archive update) or regression (fitness decreases). [§3.3, Transition Tracking and gradient estimation](../pdfs/30-kernelfoundry.md)
- The fitness gradient averages, per dimension, fitness deltas multiplied by the sign of the coordinate change and an exponential time-decay weight, over transitions originating from a cell. [§3.3, Eq. 1](../pdfs/30-kernelfoundry.md)
- The improvement-rate gradient is the difference between the probability of improvement when a coordinate increases and when it decreases, ignoring magnitude. [§3.3, Eq. 2](../pdfs/30-kernelfoundry.md)
- The exploration gradient points toward empty or low-quality cells, weighted by fitness headroom and inverse L1 distance. The three components are combined with weights \((\alpha, \beta, \gamma) = (0.4, 0.4, 0.2)\). [§3.3, Eq. 3](../pdfs/30-kernelfoundry.md)
- The combined signal is used twice: cells with strong positive gradient magnitude get higher parent-sampling probability, and gradient directions are translated into natural-language mutation hints injected into the generation prompt. [§3.3, Gradient-to-Prompt Translation; Figure 2](../pdfs/30-kernelfoundry.md)

### Templated kernels and meta-prompt evolution

- A templated kernel exposes hardware-dependent values (work-group dimensions, tile sizes, unroll factors) as template parameters, with a dispatch function enumerating the valid combinations; the printed example enumerates three block-size pairs behind a `forward` dispatcher. [§3.4; Appendix E.2, Listing 1](../pdfs/30-kernelfoundry.md)
- The evaluation pipeline detects templated kernels, evaluates each instantiation independently, assigns the best configuration's result as the kernel's fitness, and logs all results for later iterations. [§3.4](../pdfs/30-kernelfoundry.md)
- Four prompt regions are evolvable: optimization philosophy, optimization strategies (techniques with code patterns), common pitfalls, and analysis guidance (a pre-coding reasoning scaffold). [§3.5, Evolvable Prompt Regions](../pdfs/30-kernelfoundry.md)
- A separate meta-prompter LLM receives the current evolvable sections, generated kernel code and evaluation metrics (correctness, speedup, error messages), diagnoses missing or misleading guidance, and emits SEARCH/REPLACE diffs restricted to the evolvable regions. [§3.5, Meta-Prompter LLM](../pdfs/30-kernelfoundry.md)
- Evolved prompts live in their own archive; a prompt variant's fitness is the best kernel performance achieved with it. Prompt updates are proposed every N kernel generations, default N = 10. [§3.5, Prompt Archive and Co-Evolution](../pdfs/30-kernelfoundry.md)
- Defaults are at most three prompt mutations per update and a prompt archive of size 16. [Appendix B.4, Table 6](../pdfs/30-kernelfoundry.md)

### Metrics

- Correctness rate is the fraction of tasks with a kernel that compiles and gives numerically correct output. `fast_p` is the proportion of tasks with speedup greater than p, with all N tasks in the denominator; speedup is baseline time (PyTorch eager unless stated) divided by kernel time. [§4, Metrics](../pdfs/30-kernelfoundry.md)
- The hardware-speedup metric is \(hws(k^A) = t_A(k^B)/t_A(k^A)\): on GPU A, the runtime of the kernel optimized on GPU B divided by that of the kernel optimized on A. `hws_p` is the percentage of kernels with hws above p. [§5.3](../pdfs/30-kernelfoundry.md)

## 3. Methodology and experimental setup

### Pipeline and how hardware information enters

- The pipeline has five components: a task specification layer (PyTorch reference, natural-language description, or an existing kernel), a prompt constructor, an LLM inference backend (OpenAI, Anthropic, or local vLLM), a compilation and evaluation pipeline, and the evolutionary archive. [§3.1; Figure 1](../pdfs/30-kernelfoundry.md)
- Custom tasks are a YAML config, a Python module with a build function and pytest correctness and performance tests, and a language-specific file with markers for reference code, user instructions and an optional initial kernel. [Appendix C](../pdfs/30-kernelfoundry.md)
- The constructor is described as combining the task specification, hardware-specific optimization guidance, sampled parents and inspirations from the archive, gradient-derived hints and evolved prompt components. [§3.1](../pdfs/30-kernelfoundry.md)
- The printed main prompt contains an example kernel, the reference code, the best kernel so far with its runtime, the last tested kernel with runtime and console log, a hardware specification block, an evolvable optimization-strategies slot, and a required analysis-then-code response format. [Appendix E.1](../pdfs/30-kernelfoundry.md)
- Hardware enters as text: the prompt states "Your code will run on the following hardware" followed by a specification placeholder and an instruction to consider it. The content of the placeholder is not shown. [Appendix E.1](../pdfs/30-kernelfoundry.md)
- Hardware enters as measurement: candidates are compiled for the target backend (DPC++ for SYCL, nvcc for CUDA, or Triton) and timed on execution workers with the selected GPU, so fitness reflects the device the run is routed to. [§3.1; §3.6; Appendix C](../pdfs/30-kernelfoundry.md)
- Hardware enters optionally as profiling: for correct kernels, Intel unitrace or NVIDIA Nsight Compute data (runtime, achieved versus peak memory bandwidth, ALU occupancy, memory-bound versus compute-bound label) are turned into natural-language summaries. [§3.1; Appendix B.3](../pdfs/30-kernelfoundry.md)
- Hardware enters through parameter tuning: each template instantiation is timed on the target device and the fastest is kept. [§3.4](../pdfs/30-kernelfoundry.md)
- The prompt also states critical requirements: match the reference functionality, compile and run on the GPU, and do not cache or reuse previous results. This is an instruction to the generator, not an enforced check described in the source. [Appendix E.1](../pdfs/30-kernelfoundry.md)
- Figure 2, inspected visually, also shows a dashed "Compile error" arrow from the compile step back to the inference LLM; the text does not describe a compile-repair loop or how it counts against the budget. [Figure 2](../pdfs/30-kernelfoundry.md)
- Execution is distributed over an LLM server, compilation workers (no GPU needed), GPU execution workers behind a task queue with single-task-per-GPU isolation, and a database server storing all kernels, results and evolutionary state. [§3.6; Appendix C](../pdfs/30-kernelfoundry.md)
- Hardware is an integrated Intel Arc 140V (called LNL), a discrete Intel Battlemage B580 (BMG), and an NVIDIA RTX A6000 for CUDA comparisons; the stated toolchain is oneAPI DPC++ 2025.2, CUDA Toolkit 12.8, Triton 3.5, unitrace 2.3, Nsight Compute 2025.3.0.0 and PyTorch 2.9. [§4, Programming paradigm and hardware; Appendix B.1](../pdfs/30-kernelfoundry.md)

### Tasks, models and budgets

- The 111-task filtered set is the same task set as robust-kbench's, which removes tasks by five criteria: small output range, small output standard deviation, small standard deviation along any axis, small input impact, and baseline inefficiencies. [§4, Benchmarks; Appendix D](../pdfs/30-kernelfoundry.md)
- The 40-task representative subset was chosen for operation diversity, for being uncompromised under all five criteria, and for having a published AI CUDA Engineer kernel whose fastest version passes this paper's stricter correctness test; 53 of 229 such kernels failed that test. [Appendix D](../pdfs/30-kernelfoundry.md)
- Robust-kbench provides 19 tasks including forward and backward operations; the 12 used are those for which the best generated kernels were published. [§4, Benchmarks](../pdfs/30-kernelfoundry.md)
- CUDA versus AI CUDA Engineer: o3-mini only, 40 iterations with population 4, chosen to align with that work's setup; the baseline used six models that the authors could not all access. [§4, Baselines; Appendix B.4; Table 1](../pdfs/30-kernelfoundry.md)
- CUDA versus robust-kbench: GPT-o3, o4-mini and 4.1 (Sonnet 3.7 excluded), 40 iterations with 8 suggestions per iteration, matching that work's reported setting. [§4, Baselines; Appendix B.4](../pdfs/30-kernelfoundry.md)
- SYCL runs: Claude Sonnet 4.5 in the first iteration, then an equal-weight ensemble of GPT-5-mini and GPT-4.1; the 40 x 8 setting is stated as the one used for Table 2, followed by two iterations of parameter optimization. [Appendix B.4](../pdfs/30-kernelfoundry.md)
- Default sampling settings are temperature 0.3, top-p 1 and 8,000 maximum tokens. [Appendix B.4, Table 6](../pdfs/30-kernelfoundry.md)
- OpenEvolve, described as an open-source AlphaEvolve implementation lacking kernel-specific strategies, meta-prompting and parameter optimization, is run on the 20 L2 tasks for SYCL with the same listed models. [§5.2; Table 2](../pdfs/30-kernelfoundry.md)

### Budget alignment of the comparisons

- Aligned in iterations and candidates per iteration by the authors' statement: the AI CUDA Engineer comparison (40 x 4) and the robust-kbench comparison (40 x 8). In both, the model sets differ and the baselines are the published best kernels re-timed on the authors' A6000 ("re-eval"), not searches rerun by the authors. [Appendix B.4; §4, Baselines; §5.1](../pdfs/30-kernelfoundry.md)
- Same listed models and same iteration count: OpenEvolve versus KernelFoundry at 10 iterations, and the three ablation rows at 20 iterations. At 40 iterations the KernelFoundry row additionally includes parameter optimization. [Table 2](../pdfs/30-kernelfoundry.md)
- Not aligned: Kernelsseum rows (L40S, other models), "original" baseline rows (H100), the robust-kbench CUDA value quoted in the SYCL table, every "+ parameter optim." row relative to its plain counterpart (two extra iterations, described as best@8), and the oneDNN study where two operations received human-supplied code or guidance. [Table 1 caption; Table 2; §5.1; §5.4](../pdfs/30-kernelfoundry.md)

### Correctness and timing protocol

- The authors reject KernelBench's absolute tolerance of \(10^{-2}\) as too loose for small-valued outputs and use relative error \(\nu = |y - \hat{y}|/(|y| + \epsilon)\). [§4, Metrics](../pdfs/30-kernelfoundry.md)
- A kernel counts as correct if \(\nu < 0.01\) for 99% of output values; the remaining 1% is allowed to deviate, attributed to hardware imprecision. [§4, Metrics](../pdfs/30-kernelfoundry.md)
- Cosine similarity of the flattened outputs is proposed as a second correctness measure; no threshold is given. [§4, Metrics](../pdfs/30-kernelfoundry.md)
- Timing first runs a fixed number of initial trials to estimate runtime, then sets warm-up and main trial counts from minimum total times rather than fixed counts. [Appendix B.2](../pdfs/30-kernelfoundry.md)
- Stated minima: 1 s warm-up time, 10 warm-up iterations, 0.01 s inner-loop time, 10 main iterations, 1 s main measurement. An inner loop runs several trials before each `torch.cuda.synchronize` or `torch.xpu.synchronize` to reduce synchronization overhead on very fast kernels. [Appendix B.2](../pdfs/30-kernelfoundry.md)
- For robust-kbench backward operations, the candidate kernel's isolated runtime is measured while the reference is measured through `torch.autograd.grad`; the authors state this produces high backward speedups and accept it as a minor drawback. [Appendix B.2](../pdfs/30-kernelfoundry.md)

## 4. Results and evidence

### CUDA comparisons on an NVIDIA A6000 (Table 1)

- KernelBench representative L1 (n = 20), o3-mini, 40 x 4: KernelFoundry reaches correct rate 1.0, fast_1 55%, fast_2 10%, average speedup 1.204, geometric 1.092; with parameter optimization fast_2 15%, average 1.241, geometric 1.124. [Table 1](../pdfs/30-kernelfoundry.md)
- Same set, AI CUDA Engineer published kernels re-timed on the A6000: fast_1 40%, average 1.005, geometric 0.909 (the printed fast_2 entry for this row is inconsistent; see section 6); their originally published H100 numbers are fast_1 70%, fast_2 20%, average 1.422. [Table 1](../pdfs/30-kernelfoundry.md)
- KernelBench representative L2 (n = 20), o3-mini, 40 x 4: KernelFoundry fast_1 90%, fast_2 40%, average 2.051, geometric 1.680; with parameter optimization fast_2 45%, average 2.104, geometric 1.730. Re-timed AI CUDA Engineer kernels: fast_1 85%, fast_2 25%, average 1.606, geometric 1.515. [Table 1](../pdfs/30-kernelfoundry.md)
- Robust-kbench (n = 12), GPT-o3/o4-mini/4.1, 40 x 8: KernelFoundry fast_1 100%, fast_2 67%, average 12.107, geometric 5.295; re-timed robust-kbench kernels: fast_1 92%, fast_2 58%, average 8.865, geometric 3.499. [Table 1](../pdfs/30-kernelfoundry.md)
- The text's "36% higher speedup" on robust-kbench matches the ratio of arithmetic means (derived: 12.107 / 8.865 = 1.366); the geometric means give a ratio of 1.51 (derived). [§5.1; Table 1](../pdfs/30-kernelfoundry.md)
- Derived split of Table 7: on the eight forward tasks the average is 9.05 for KernelFoundry versus 8.44 re-timed baseline (geometric about 3.1 versus 2.2); on the four backward tasks it is 18.2 versus 9.7. `layernorm_forward` alone is about 55x in both columns. [Derived from Appendix F.1, Table 7](../pdfs/30-kernelfoundry.md)
- Derived per-task counts: KernelFoundry with parameter optimization is faster than the re-timed baseline kernel on 17 of 20 L1 tasks, 13 of 20 L2 tasks and 11 of 12 robust-kbench tasks. On L1, 9 of its 20 best kernels are not faster than PyTorch eager. [Derived from Appendix F.1–F.2, Tables 7–8](../pdfs/30-kernelfoundry.md)
- Parameter optimization changes arithmetic-mean speedup by +0.037 (L1), +0.053 (L2) and -0.026 (robust-kbench). [Derived from Table 1](../pdfs/30-kernelfoundry.md)
- Kernelsseum rows (KernelBench's originally published results, L40S, different models) show correct rate 0.65 and average speedup 0.765 on L1 and 0.874 on L2. [Table 1](../pdfs/30-kernelfoundry.md)
- Cost: on five L2 problems with o3-mini (40 iterations x 4 branches), mean usage per generated candidate was 11,102 input and 5,377 output tokens, giving about $5.74 for the 160 candidates of one task; the authors quote $17.34 with gpt-5.4, $5 for robust-kbench and up to $20 for METR. [Appendix I](../pdfs/30-kernelfoundry.md)

### SYCL generation (Table 2)

- KernelBench filtered (n = 111), SYCL, GPT-4.1/GPT-5-mini/Sonnet-4.5: correct rate 0.97, fast_1 71%, fast_2 42%, average speedup 2.32, geometric 1.38 over PyTorch eager. This is the abstract's "2.3". [Abstract; Table 2; §5.2](../pdfs/30-kernelfoundry.md)
- The robust-kbench row beside it (CUDA, average 1.49, other fields blank) is that paper's reported number with different language, models and hardware; it is a reference value, not an equal-condition comparison. [Table 2; §5.2](../pdfs/30-kernelfoundry.md)
- L2 representative set (n = 20), SYCL, 40 iterations: KernelFoundry with parameter optimization fast_1 80%, fast_2 45%, average 2.732, geometric 2.119; OpenEvolve fast_1 70%, fast_2 40%, average 2.535, geometric 1.775. [Table 2; Appendix F.3, Table 9](../pdfs/30-kernelfoundry.md)
- Same set, 10 iterations: KernelFoundry fast_1 80%, fast_2 35%, average 2.059, geometric 1.661; OpenEvolve fast_1 70%, fast_2 20%, average 1.483, geometric 1.202. [Table 2](../pdfs/30-kernelfoundry.md)
- Derived from the per-task table at 40 iterations: KernelFoundry is faster than OpenEvolve on 9 of 20 tasks and slower on 11; large individual gaps go both ways (for example 7.955 versus 2.023 on task 82, and 3.045 versus 6.612 on task 21). [Derived from Appendix F.3, Table 9](../pdfs/30-kernelfoundry.md)
- Figure 3, inspected visually, is a heat map of cumulative-best speedup per trial for the 20 L2 tasks: tasks 32 and 82 reach the darkest green (about 7x), while tasks 47, 59, 70 and 73 stay below 1x for most or all of 40 trials. [Figure 3](../pdfs/30-kernelfoundry.md)
- With the open-weight GPT-OSS 20B (SYCL, 20 L2 tasks, 40 iterations, population 4, LNL GPU), no correct kernel was found for 7 of 20 tasks; among the 13 solved, six exceed 1.0x and three are below 0.1x. [Appendix G, Table 12; counts above 1.0 and below 0.1 derived](../pdfs/30-kernelfoundry.md)

### Ablations and mechanism evidence

- At 20 iterations on the 20 L2 tasks (SYCL): full method fast_1 70%, fast_2 50%, average 2.539, geometric 1.833; without prompt evolution fast_1 70%, fast_2 55%, average 2.505, geometric 1.600; without gradient guidance fast_1 75%, fast_2 35%, average 2.178, geometric 1.695. [Table 2; Appendix F.4, Table 10](../pdfs/30-kernelfoundry.md)
- A further variant with prompt updates every 2 generations has average 2.542 and geometric 1.944; it appears only in the appendix table. [Appendix F.4, Table 10](../pdfs/30-kernelfoundry.md)
- Derived per-task view: the full method beats the no-prompt-evolution run on 10 of 20 tasks and the no-gradient run on 12 of 20; single tasks swing widely (task 16: 0.294 without prompt evolution versus 1.638 with; task 99: 5.324 versus 2.657). [Derived from Appendix F.4, Table 10](../pdfs/30-kernelfoundry.md)
- Figure 6, inspected visually, shows 100-trial cumulative-best curves for two tasks: for task 82 the run with prompt evolution ends near 10x versus near 8x without; for task 46 both end near 6x, with the prompt-evolution run reaching that level by about trial 15 and the other only after about trial 85. [Figure 6; Appendix H](../pdfs/30-kernelfoundry.md)
- Qualitative prompt analysis: by manual inspection, "optimization strategies" and "common pitfalls" change most often while "optimization philosophy" and "analysis guidance" stay relatively stable; pairwise `difflib.SequenceMatcher` similarity across iterations for five problems is, on visual inspection, mostly low with scattered high-similarity cells. [Appendix J; Figure 7](../pdfs/30-kernelfoundry.md)
- One worked example of a gradient-derived hint: a parent at 0.021 ms received a register-tiling hint along the memory dimension and the child ran at 0.0195 ms, about 7% faster. This is a single illustrative transition. [Appendix E.3](../pdfs/30-kernelfoundry.md)

### Hardware-awareness crossover

- Setup: KernelFoundry is run separately on LNL and on B580 for the 20 L2 tasks (SYCL); each run's best kernel is then timed on the other GPU. [§5.3](../pdfs/30-kernelfoundry.md)
- On LNL, LNL-optimized kernels have hws_1 70%, hws_1.5 20%, average hws 1.537, geometric 1.297. On B580, B580-optimized kernels have hws_1 70%, hws_1.5 15%, average 1.109, geometric 1.038. [Table 3](../pdfs/30-kernelfoundry.md)
- Derived from per-task values: 7 of 20 hws values on LNL and 6 of 20 on B580 lie within 3% of 1.0, and the other-GPU kernel is more than 3% faster on 4 (LNL) and 3 (B580) tasks, with hws as low as 0.391 for task 85 on B580. [Derived from Appendix F.5, Table 11](../pdfs/30-kernelfoundry.md)
- The per-task table's "Average" row shows mean runtime on LNL of 3.325 ms for LNL-optimized versus 2.694 ms for B580-optimized kernels, driven by task 50 (50.5 versus 36.7 ms); by mean runtime the cross-optimized kernels are faster on LNL even though average hws is above 1. [Appendix F.5, Table 11](../pdfs/30-kernelfoundry.md)

### oneDNN comparison and Llama case study

- Against oneDNN C++ implementations of five operations, generated SYCL kernels reach 1.79 (`concat(x, layer_norm(x))`, started from a supplied initial implementation), 0.35 (matmul with ReLU post-op), 0.72 (MaxPool + Linear), 1.10 (sum reduction) and 1.70 (softmax, with user-supplied optimization guidance). Only the sum-reduction win used neither human-supplied code nor instructions. [§5.4; Table 4](../pdfs/30-kernelfoundry.md)
- Llama 3.2 1B rotary positional embedding on a B580: a correct kernel after two iterations and a 7.9x operation speedup within ten; total forward-pass time falls from 0.413 s to 0.38 s, reported as 8%. Correctness is output comparison with the PyTorch reference plus a check that a full model pass gives identical results "on a simple query". [§5.5](../pdfs/30-kernelfoundry.md)

## 5. Authors' conclusions

- The authors conclude that combining kernel-specific quality-diversity search, meta-prompting and parameter optimization yields high-performing CUDA and SYCL kernels, shows cross-platform applicability and outperforms state-of-the-art methods. [§6](../pdfs/30-kernelfoundry.md)
- They state that kernel-specific behavioural coordinates prevent mode collapse "by construction" because each cell evolves independently, and that meta-prompt evolution counteracts context degradation. [§2, LLM-Guided Evolutionary Search; §3.2](../pdfs/30-kernelfoundry.md)
- They interpret the 10-iteration gap as showing that OpenEvolve needs substantially more iterations to reach comparable speedup. [§5.2](../pdfs/30-kernelfoundry.md)
- They state that performance is reduced without prompt evolution or without gradient guidance, and that meta-prompting becomes more valuable with longer runs. [§5.2; Appendix H](../pdfs/30-kernelfoundry.md)
- They conclude from the crossover experiment that kernels optimized for the target GPU outperform kernels optimized on another GPU. [§5.3; Table 3 caption](../pdfs/30-kernelfoundry.md)
- They describe generated SYCL kernels as competitive with oneDNN, and argue that per-kernel cost is comparable to related work and justified when manual engineering effort is reduced. [§5.4; Appendix I](../pdfs/30-kernelfoundry.md)
- They call for open, reproducible benchmarks and kernel databases, and list templated kernels across input ranges and shapes, formal verification and deeper hardware specialization as future work. [§6](../pdfs/30-kernelfoundry.md)

## 6. Limitations and open questions

### Stated by the source

- Reported KernelBench speedups require careful interpretation because of known benchmark shortcomings; results depend heavily on hardware, language and model; only a subset of each baseline's models was available; and published "original" numbers are not directly comparable across hardware. [§1; §4; §4, Baselines; §5.1](../pdfs/30-kernelfoundry.md)
- Backward-operation speedups are inflated by measuring the reference through autograd and the kernel in isolation. [Appendix B.2](../pdfs/30-kernelfoundry.md)
- The open-weight model failed on 7 of 20 tasks. [Appendix G](../pdfs/30-kernelfoundry.md)
- Reward hacking is not eliminated; formal verification is future work, and speedups across input ranges and tensor shapes are not yet addressed. [§6](../pdfs/30-kernelfoundry.md)

### Analyst caveats and missing evidence

- **No variance estimates:** no table reports repeated runs, seeds or confidence intervals. [Basis: Tables 1–4; Appendix F](../pdfs/30-kernelfoundry.md)
- **Run-to-run variation is visible:** for 5 of 20 tasks the 20-iteration result exceeds the 40-iteration result (task 35: 6.831 versus 1.972), which cannot happen within one cumulative-best run on one device, so these are separate runs or measurements whose differences are as large as the ablation effects. [Basis: derived from Tables 9–10](../pdfs/30-kernelfoundry.md)
- **Best-of selection:** reported speedups are the best kernel found by the search; the source does not describe re-timing the selected kernel in an independent measurement. [Basis: §5.1; Appendix B.2](../pdfs/30-kernelfoundry.md)
- **Correctness gate:** up to 1% of output elements may exceed the tolerance, and the value of epsilon, the number and distribution of test inputs, and any cosine threshold are not stated. Passing is sampled numerical agreement, not proof. [Basis: §4, Metrics](../pdfs/30-kernelfoundry.md)
- **Fitness saturation:** as written, fitness is capped at 1.0 once speedup reaches the 2x target, so the stated replacement rule cannot rank a 2x kernel against an 8x kernel; how elites above target are compared is not described. [Basis: §3.2, Fitness Function; Table 6](../pdfs/30-kernelfoundry.md)
- **Descriptor validity:** the static classifier is not evaluated against human labels, and no archive occupancy, coverage or per-cell statistics are reported, so the claim that diversity is maintained is a design argument rather than a measured result. [Basis: §3.2; §5](../pdfs/30-kernelfoundry.md)
- **Ablation coverage and strength:** no run removes the MAP-Elites archive or the hardware specification text, and OpenEvolve differs in several respects at once. Arithmetic means for full versus no-prompt-evolution runs are 2.539 versus 2.505, fast_2 is higher without prompt evolution and fast_1 is higher without gradient guidance; with default N = 10, a 20-iteration run allows very few prompt updates. [Basis: Table 2; Table 10; §3.5](../pdfs/30-kernelfoundry.md)
- **Hardware-awareness confound:** the crossover compares two independent search runs; without a same-GPU replicate, differences between `k^A` and `k^B` could reflect search variance rather than specialization. [Basis: §5.3; Table 11](../pdfs/30-kernelfoundry.md)
- **Budget parity:** OpenEvolve's per-iteration population is not stated, and baseline CUDA kernels were produced with different model sets and are not rerun searches. [Basis: Table 2; Appendix B.4; §4, Baselines](../pdfs/30-kernelfoundry.md)
- **Selection of comparison tasks:** the representative subset keeps only tasks where the baseline's fastest kernel passes the stricter test, so baseline correct rate 1.0 in Table 1 is by construction. [Basis: Appendix D; Table 1](../pdfs/30-kernelfoundry.md)
- **Unstated conditions:** the Intel GPU used for Table 2 and Table 4, the meta-prompter model, how prompt variants are sampled from the prompt archive, whether profiler feedback was enabled, the budget and models for the oneDNN study, and whether the cost estimate includes meta-prompter and parameter-optimization calls are not given. [Basis: §5.2; §5.4; §3.5; Appendix B.3; Appendix I](../pdfs/30-kernelfoundry.md)
- **Missing baseline:** the setup says comparisons are made to PyTorch eager and `torch.compile`, but no `torch.compile` number is reported. [Basis: §4, Baselines; Tables 1–2](../pdfs/30-kernelfoundry.md)

### Source inconsistencies

- **Source inconsistency:** Table 1 gives the re-evaluated AI CUDA Engineer L1 row fast_1 40% and fast_2 50%, which is impossible since fast_2 cannot exceed fast_1; the per-task table yields 1 of 20 above 2.0 (5%). The 50% value is specific to Table 1 and is also printed in the PDF. [Table 1; derived from Table 8](../pdfs/30-kernelfoundry.md) [PDF p. 8](../pdfs/30-kernelfoundry.pdf)
- **Source inconsistency:** Table 1 gives the original AI CUDA Engineer L2 row fast_1 100%, while the per-task table lists one task at exactly 1.000, giving 19 of 20 under the strict "greater than p" definition; this may be display rounding. [Table 1; Table 8; §4, Metrics](../pdfs/30-kernelfoundry.md)
- **Source inconsistency:** the method section defines three behavioural dimensions and 64 cells, whereas the hyperparameter table lists "Archive dimensions 4" with 4 bins each. The fourth dimension is not described. [§3.2; Appendix B.4, Table 6](../pdfs/30-kernelfoundry.md) [PDF p. 14](../pdfs/30-kernelfoundry.pdf)
- **Source inconsistency:** the timing description says trial counts are set from minimum total time rather than fixed counts, while the hyperparameter table lists 10 warm-up and 100 timing iterations. [Appendix B.2; Table 6](../pdfs/30-kernelfoundry.md)
- **Source inconsistency:** the method names four evolvable prompt sections and describes parents, inspirations and gradient hints in the prompt; the printed main prompt has a single evolvable optimization-strategies slot and best/last-kernel slots, with no visible slot for the other items. The printed template may be abridged. [§3.1; §3.5; Appendix E.1](../pdfs/30-kernelfoundry.md)
- **Source inconsistency:** the abstract says the approach "consistently outperforms" baselines, while per-task tables show losses on 7 of 20 L2 CUDA tasks and 11 of 20 SYCL tasks against OpenEvolve; the claim holds for the aggregate rows. [Abstract; derived from Tables 8–9](../pdfs/30-kernelfoundry.md)
- **Source inconsistency:** "per kernel" in the cost appendix denotes a single generated candidate for token counts and the whole 160-candidate task for the dollar figure. [Appendix I](../pdfs/30-kernelfoundry.md)
- **Source inconsistency:** the text says Figure 6 shows consistent improvements from meta-prompting over 100 iterations, yet for task 46 both curves end at about the same speedup (visual reading). [§5.2; Figure 6](../pdfs/30-kernelfoundry.md)
- **Unexplained figure element:** the infrastructure figure contains a box labelled "RAG" and example workers "NVIDIA H100" and "NVIDIA A100"; retrieval augmentation is not described in the text and those GPUs are not in the experimental hardware list. [Figure 4](../pdfs/30-kernelfoundry.md) [PDF p. 15](../pdfs/30-kernelfoundry.pdf)

## 7. Relevance to this project — interpretation

- **Proposal:** give each kernel in the Stage 3 pool a small structural label (for example memory pattern, algorithmic form, synchronization level) and keep the best survivor per label combination, so that the batch planner combines from structurally different kernels instead of the top few near-duplicates; audit a sample of labels by hand before treating label coverage as evidence of diversity. [Basis: §3.2, Kernel-Specific Behavioral Descriptors](../pdfs/30-kernelfoundry.md)
- **Proposal:** make pool sampling an explicit, logged policy with a mix of uniform-over-cells, performance-weighted and under-explored-cell choices, and record which policy produced each parent, so that later analysis can relate planning choices to outcomes. [Basis: §3.2, Selection Strategies](../pdfs/30-kernelfoundry.md)
- **Proposal:** record every parent-to-child transition (labels, measured change, outcome class, round) and give the planner a short summary of which structural moves have paid off for this task; treat the summary as a hint source to be tested, since the source's own ablation evidence for it is weak. [Basis: §3.3; Table 10](../pdfs/30-kernelfoundry.md)
- **Proposal:** keep correctness status and measured speedup as separate pool fields rather than a single capped score, so that ranking above a target speedup remains possible and failed candidates stay distinguishable by failure type. [Basis: §3.2, Fitness Function](../pdfs/30-kernelfoundry.md)
- **Proposal:** add an optional tuning step in which a surviving kernel is rewritten with exposed launch or tiling parameters and each instantiation is measured with PyGPUBench; require every instantiation that might be selected to pass the same fuzzing gate, because the source says each instantiation is evaluated independently but does not spell out whether correctness is rechecked per instantiation. [Basis: §3.4; Appendix E.2](../pdfs/30-kernelfoundry.md)
- **Proposal:** keep a small, versioned guidance section (strategies and pitfalls) that a separate analysis step may edit every N rounds from recent outcomes, with diffs stored per round, instead of appending raw history to the planner context. [Basis: §3.5; Appendix J](../pdfs/30-kernelfoundry.md)
- **Proposal:** in the Stage 2 oracle record, state the numerical tolerance, the permitted fraction of violating elements and the input distributions explicitly, and report the observed violation fraction rather than only pass/fail; a 99%-of-elements rule should be a visible choice. [Basis: §4, Metrics](../pdfs/30-kernelfoundry.md)
- **Proposal:** in Stage 4, re-measure the selected kernel in a fresh process, time reference and candidate through the same call boundary, and, where budget allows, repeat the whole search to estimate run-to-run spread before attributing gains to any loop component. [Basis: Tables 9–10; Appendix B.2](../pdfs/30-kernelfoundry.md)
- **Proposal:** if hardware specialization is to be claimed, pair a cross-device swap with a same-device replicate run so that search variance can be separated from device effects. [Basis: §5.3; Table 11](../pdfs/30-kernelfoundry.md)
- **Proposal:** account for search budget as rounds x batch size plus tuning instantiations and auxiliary model calls, and report tokens per task on that basis, so that comparisons with external methods state what was held equal. [Basis: Appendix B.4; Appendix I](../pdfs/30-kernelfoundry.md)

## 8. Retrieval notes

- Search aliases: KernelFoundry, MAP-Elites kernel archive, quality-diversity GPU kernels, behavioural descriptors d_mem d_algo d_sync, gradient-informed evolution, transition tracker, meta-prompt evolution, meta-prompter, templated kernel parameter optimization, hws hardware-speedup, SYCL kernel generation, Intel Arc B580, Arc 140V LNL, OpenEvolve comparison, oneDNN comparison.
- Questions answered: What are the archive dimensions, cell contents and replacement rule? How are "gradients" computed and used? What does the meta-prompter edit and how often? How does hardware information reach the generator? What tolerance defines a correct kernel? Which comparisons used aligned iteration budgets? How large are the ablation effects? What does the crossover experiment show and not show?
- Related local analyses: [KernelBench](01-kernelbench.md) for the task suite and fast_p; [robust agentic CUDA](08-robust-agentic-cuda.md) for the task filter, the 12 comparison tasks and the baseline kernels; [METR KernelAgent](21-measuring-automated-kernel-engineering.md) for task filtering and cost context; [MultiKernelBench](06-multikernelbench.md) for other non-CUDA targets; [K-Search](24-k-search.md), [CudaForge](31-cudaforge.md) and [Kevin](32-kevin.md) for other search, agent and training approaches; [benchmark reliability](27-performance-benchmark-reliability.md) and [noisy-environment benchmarking](33-robust-benchmarking-noisy-environments.md) for timing methodology; [Self-Refine](18-self-refine.md) for iterative refinement. Links are navigation aids, not controlled comparisons.

## 9. Source coverage and verification notes

- Read the complete preserved Markdown: abstract, §1–§6, impact and availability statements, the reference list, and Appendices A–J including Table 5, the prompts and Listing 1, per-task Tables 7–12, cost estimation and the prompt-evolution analysis.
- Opened and inspected the image assets for Figures 2, 3, 5, 6 and 7. Values quoted from Figures 3 and 6 are approximate visual readings; Figure 5 (per-iteration, non-cumulative speedup curves) was viewed but no values are quoted from it. Figure 4 is an SVG of glyph outlines with no readable text; its labels were read from the PDF text layer of page 15 instead. Figure 1 survives in the Markdown only as a flattened list of box labels.
- Extracted the PDF text layer for pages 7, 8, 14 and 15 to confirm that the fast_2 value of 50% in Table 1 and "Archive dimensions 4" in Table 6 are in the original and are not conversion errors. Other PDF pages were not compared line by line with the Markdown.
- Recomputed fast_p counts, arithmetic and geometric means from Tables 7–11 against Tables 1–3, the hws ratios in Table 11, the Appendix I cost arithmetic and the Llama forward-pass percentage. Aggregates agree with the summary tables except for the items listed under source inconsistencies; small differences in the third decimal are consistent with rounding.
- Conversion problems: bold markers for "best results" in Table 1 are lost; the descriptor lists and prompt templates are flattened but legible; the code listing has extra blank lines.
- Not verified: the public code repository, the behavioural classifier, the actual hardware-specification text, and the baseline kernels were not inspected. No kernels were compiled or timed and no experiment was reproduced.

---
paper_id: "31"
title: "CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization"
source_version: "2511.01884v2"
source_markdown: "../pdfs/31-cudaforge.md"
source_pdf: "../pdfs/31-cudaforge.pdf"
source_url: "https://arxiv.org/abs/2511.01884"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["gpu-kernels", "cuda", "kernelbench", "agent-scaffolding", "llm-as-a-judge", "profiling", "hardware-awareness", "execution-feedback", "iterative-refinement", "cost-effectiveness", "test-time-compute"]
summary: "CudaForge is a training-free Coder–Judge loop that feeds test errors and a 24-metric Nsight Compute subset to an LLM Judge to optimize KernelBench CUDA kernels; evidence is single-run, fixed-test, 1e-4-tolerance speedup over PyTorch, mostly with OpenAI-o3."
---

# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization

## 1. Paper identity and scope

- This note covers arXiv `2511.01884v2`, a preprint by six authors at the University of Minnesota presenting CudaForge, described as a "training-free multi-agent workflow" for generating and optimizing CUDA kernels from PyTorch reference modules. [Title block; Abstract](../pdfs/31-cudaforge.md)
- The stated research question is whether a "simple but effective hardware-aware approach" can reliably produce efficient CUDA kernels at low cost. [§1](../pdfs/31-cudaforge.md)
- The source motivates the work with three gaps in prior systems: limited kernel efficiency of RL-trained models (C1), high training and inference cost (C2), and no use of hardware feedback such as Nsight Compute (C3). [§1](../pdfs/31-cudaforge.md)
- The contribution is a two-agent loop (a Coder and a Judge) with compile/execute tests and Nsight Compute (NCU) profiling between rounds, plus an offline procedure that reduces the NCU output to 24 metrics shown to the Judge. [§1; §2.1; §2.3](../pdfs/31-cudaforge.md)
- The object of study is KernelBench Levels 1–3 (250 tasks): the kernel must match a PyTorch reference on the benchmark's test cases and is scored by speed relative to that reference. Level 4 is not evaluated. [§3.1; Appendix D.1](../pdfs/31-cudaforge.md)
- The main run uses OpenAI-o3 for both agents, at most 10 rounds, on an RTX 6000; ablations and generalization studies use a 25-task stratified subset. [§3.1; §3.2](../pdfs/31-cudaforge.md)
- No model is trained, no benchmark is introduced, and no formal correctness guarantee is offered; correctness is a fixed tolerance on predefined inputs. The two external comparison systems (Kevin-32B and an "Agentic Baseline") were not rerun; their numbers are copied from their papers. [§2.2, Design of Correctness Tests; §3.2; §3.3, Footnote 3](../pdfs/31-cudaforge.md)
- A code link (`github.com/OptimAI-Lab/CudaForge`) is given in the abstract; that repository was not inspected for this note. [Abstract](../pdfs/31-cudaforge.md)

## 2. Concepts and terminology

### Agents, rounds and acceptance

- **Coder:** the generating agent. It produces a candidate kernel from the task description, the previous candidate and the Judge's latest feedback. [§2.2, Design of Coder](../pdfs/31-cudaforge.md)
- **Judge:** the advising agent. It receives the task description, the current candidate and either runtime information or hardware feedback, and returns structured JSON feedback for the next round. [§2.2, Design of Judge](../pdfs/31-cudaforge.md)
- **Correction mode:** entered when a candidate fails compilation or produces wrong outputs; the Judge reads the runtime information (compilation errors, output mismatches against the PyTorch reference), names an issue (examples given: invalid indexing, race conditions, missing headers) and a fix. [§2.1; §2.2, Design of Judge](../pdfs/31-cudaforge.md)
- **Optimization mode:** entered only after a candidate passes the correctness checks; the Judge names the dominant bottleneck (examples: memory-bound, compute-bound, occupancy-limited) and one optimization. [§2.2, Design of Judge](../pdfs/31-cudaforge.md)
- **Round budget N:** the loop repeats for up to N rounds, after which "the most efficient correct kernel" is selected as the final solution. [§2.1](../pdfs/31-cudaforge.md)
- **Judge is advice, not a gate:** a kernel "is regarded as correct only when it passes both compilation and execution stages"; the Judge's JSON is passed to the Coder as guidance. Acceptance and final selection rest on executed tests and measured speed, not on the Judge's opinion. [§2.1; §2.2, Design of Correctness Tests](../pdfs/31-cudaforge.md)
- **Lightweight memory:** neither agent keeps the conversation history. Each round the Coder sees only the latest Judge feedback, the previous candidate and the task description; the Judge is likewise prompted per round. The authors say full history caused "hallucinated kernel code and higher API cost". [§2.2, Design of Coder; §2.2, Design of Judge](../pdfs/31-cudaforge.md)

### Hardware feedback

- **Hardware feedback** means static GPU specifications (examples: architecture, memory bandwidth, per-thread register limits, per-SM shared-memory capacity) combined with metrics from Nsight Compute (NCU), NVIDIA's kernel-level CUDA profiler, collected while the kernel runs (examples: memory throughput, occupancy, warp efficiency). [§1, Footnote 2; §2.3](../pdfs/31-cudaforge.md)
- **Key subset:** 24 NCU metrics chosen offline, before the workflow runs, and the only NCU metrics passed to the Judge in the default system. [§2.3; Appendix B.3, Table 8](../pdfs/31-cudaforge.md)
- **Per-round focus:** the Judge is prompted to use only the 3–4 most important metrics in a round, name one bottleneck and propose one optimization, so each round targets a single bottleneck. [§2.3; Appendix A.2](../pdfs/31-cudaforge.md)

### Correctness test and metrics

- **Correctness test:** two stages. Compilation checks the candidate builds; execution runs it on "predefined test cases" and compares outputs with the PyTorch reference under the same inputs, with a numerical tolerance set to 1e-4 "in our experiments". [§2.2, Design of Correctness Tests](../pdfs/31-cudaforge.md)
- **Correct (metric 1):** fraction of tasks whose kernel compiles and "produces outputs identical to the PyTorch reference on all test cases". The wording differs from the tolerance rule above; see §6. [§3.1](../pdfs/31-cudaforge.md)
- **Perf (metric 2):** "the ratio of the execution speed (tested on a specific GPU), between a correct generated kernel and its PyTorch reference". Tables report its average; Figure 8 labels the axis `Speedup(ref/test)`. [§3.1; Table 1; Figure 8](../pdfs/31-cudaforge.md)
- **Fast1 (metric 3):** "the proportion of correct kernels whose execution speed exceeds their PyTorch reference". [§3.1](../pdfs/31-cudaforge.md)
- **Median and 75% (metrics 4–5):** the median and 75th percentile of Perf values "across all tasks". [§3.1](../pdfs/31-cudaforge.md)
- **Best-of-candidates reporting:** for any method that iterates or samples, the reported kernel is "the best-performing correct kernel among all candidates for each task". [§3.1](../pdfs/31-cudaforge.md)
- **D\*:** a 25-task stratified random subset of KernelBench (10 Level 1, 10 Level 2, 5 Level 3) used where full runs were judged too costly. Rows using it are marked with an asterisk. [§3.1; Appendix D.2; Table 1 caption](../pdfs/31-cudaforge.md)
- **"Fake kernels":** the authors' term for reported solutions that contain no actual CUDA code and instead fall back to PyTorch's own implementations, for example through try/except. [§5; Appendix C](../pdfs/31-cudaforge.md)

### Baselines and variants named in the tables

- **OpenAI-o3:** one-shot generation with no iteration. **o3-self-refine:** ten rounds with a single model and no Judge; that model corrects and optimizes its own kernels "given hardware feedback". [§3.2](../pdfs/31-cudaforge.md)
- **o3-correction / o3-optimization:** CudaForge variants in which the Judge gives only correction feedback or only optimization feedback. [§3.2](../pdfs/31-cudaforge.md)
- **CudaForge (full metrics):** the Judge receives the entire NCU metric set instead of the 24-metric subset. **CudaForge-Scaling Up:** the maximum number of rounds is raised (to 30 in §3.7). [§3.2; Table 1 caption; §3.7](../pdfs/31-cudaforge.md)
- **Kevin-32B:** an RL-trained model from another paper; **Agentic Baseline:** a multi-agent workflow from another paper using an ensemble of o3, o4-mini, Claude Sonnet 3.7 and GPT-4.1. Results for both are taken from those papers. [§3.2](../pdfs/31-cudaforge.md)

## 3. Methodology and experimental setup

### Tasks and subsets

- The benchmark is KernelBench Levels 1–3: 100 Level-1 tasks (basic operators), 100 Level-2 tasks (multi-step operator combinations) and 50 Level-3 tasks (full network architectures such as AlexNet), 250 in total. KernelBench as a whole is described as 270 tasks over four levels; Level 4 is called "less frequently adopted" and is not used. [§3.1; Appendix D.1](../pdfs/31-cudaforge.md)
- D\* is built by stratified random sampling "with a fixed 10% ratio for each level", with tasks also sampled "uniformly across different operator categories". The task IDs are listed. [Appendix D.2](../pdfs/31-cudaforge.md)
- D\* is used for the full-metrics ablation, the cost measurements, round scaling, the GPU study and the base-model study. The o3-self-refine, o3-correction and o3-optimization rows are unmarked, i.e. full 250-task runs. [Table 1; §3.5; §3.7](../pdfs/31-cudaforge.md)

### Models, hardware and budgets

- Default configuration: OpenAI-o3 as both Coder and Judge, maximum rounds N = 10, chosen "to balance performance improvements and inference cost". [§3.2](../pdfs/31-cudaforge.md)
- Unless stated otherwise, all methods share one compile/runtime environment on a "Quadro RTX 6000" with "task-specific test suites". [§3.2](../pdfs/31-cudaforge.md)
- Tables 1–3 are RTX 6000 results. The Kevin comparison is a separate CudaForge run on H200. Table 4 adds RTX 4090, A100 and RTX 3090 on D\*. [Table 1 caption; §3.4; Table 4](../pdfs/31-cudaforge.md)
- Table 5 fixes one agent as o3 and swaps the other among GPT-5, Claude-Sonnet-4, GPT-OSS-120B and (Coder only) QwQ-32B, on D\*; the round-scaling study raises N to 30 on D\*. [§3.7; Table 5; Figure 7](../pdfs/31-cudaforge.md)

### Prompts and what each agent sees

- Round 1 Coder prompt: the KernelBench one-shot baseline prompt, used for all baselines and for CudaForge. It contains one example pair (PyTorch module and CUDA version) and the task's architecture source. [Appendix A.1](../pdfs/31-cudaforge.md)
- That prompt gives the model "complete freedom to choose the set of operators you want to replace", explicitly allowing some operators to be left unchanged, and asks for a `ModelNew` class with no testing code. [Appendix A.1](../pdfs/31-cudaforge.md)
- Judge correction prompt: role "senior CUDA + PyTorch correctness auditor"; inputs are the error log, the PyTorch reference and the CUDA candidate; output is JSON with `critical_issue` (≤20 words), `why_it_matters` (≤35) and `minimal_fix_hint` (≤20), limited to exactly one issue. [Appendix A.2](../pdfs/31-cudaforge.md)
- Judge optimization prompt: role "senior CUDA performance engineer"; inputs are the target GPU spec (name, architecture, detail items), the PyTorch reference, the CUDA candidate and the NCU metrics "verbatim"; output is JSON with `bottleneck` (≤30 words), `optimisation method` (≤35) and `modification plan` (≤35), limited to one method. [Appendix A.2](../pdfs/31-cudaforge.md)
- The optimization prompt lists bottleneck families to prefer (occupancy limits, memory coalescing, shared-memory bank conflicts, register pressure, long/short scoreboard stalls, tensor-core underutilisation). As printed, it has no field for the measured speedup. [Appendix A.2](../pdfs/31-cudaforge.md)
- Coder correction prompt (rounds 2–N): the error log, the old code (marked read-only), the Judge's "Main Critical Problem", and strict output-layout rules. [Appendix A.3](../pdfs/31-cudaforge.md)
- Coder optimization prompt (rounds 2–N): the target GPU spec, the current code, the Judge's suggestion to be applied "strictly", and a goal block requiring "correctness within atol=1e-4 or rtol=1e-4" and an unchanged Python API. The Coder is not shown NCU metrics. [Appendix A.3](../pdfs/31-cudaforge.md)

### Offline selection of the NCU subset

- Step 1: for each of some "preselected representative tasks (e.g., Conv2D, MatMul)", run 100 self-refine cycles with one model (o3 is the example), collect the correct kernels and keep the 10 with the largest speed disparity; the pseudocode takes the 5 fastest and 5 slowest. [§2.3, Step 1; Algorithm 1](../pdfs/31-cudaforge.md)
- Step 2: per task, pool the NCU metrics of those 10 kernels, remove aliases and strongly collinear indicators, compute each metric's Pearson correlation with kernel runtime, and keep the 20 with the largest absolute correlation. [§2.3, Step 2](../pdfs/31-cudaforge.md)
- Step 3: across tasks, keep metrics that recur, keep the same correlation sign, and have a global score (mean absolute correlation over tasks) above the 75th percentile of candidates. The result is 24 metrics. [§2.3, Step 3; Algorithm 2](../pdfs/31-cudaforge.md)
- The 24 metrics, in the analyst's grouping: SM activity and occupancy limits (cycles active, warps active, occupancy limits by blocks/registers/shared memory, registers per thread); instruction counts and FP32/tensor pipe use; DRAM bytes and throughput; L1/texture and L2 sector hit rates and throughput; five warp-stall reasons (memory dependency, short scoreboard, long scoreboard, barrier, branch resolving); and branch-target uniformity. [Appendix B.3, Table 8](../pdfs/31-cudaforge.md)
- Two example Top-20 lists are shown, for a Conv2D task and an SpMM task. The number of selection tasks, their identity beyond these examples, and the GPU used for selection are not stated. [§2.3; Appendix B.2, Tables 6–7](../pdfs/31-cudaforge.md)

### Evaluation procedure and what is not specified

- The source gives the tolerance (1e-4) but not the number of correctness trials, how inputs are generated or seeded, or whether atol and rtol are both applied. [§2.2, Design of Correctness Tests](../pdfs/31-cudaforge.md)
- The source does not describe the timing procedure (warm-up, number of timed trials, timer, variance), and no experiment reports repeated runs, seeds, confidence intervals or significance tests. [§3.1; §3.2; §3.3–§3.7](../pdfs/31-cudaforge.md)
- To avoid "fake kernels", the authors state they "manually checked all kernels in our experiments"; the checking criteria are not given. [§5](../pdfs/31-cudaforge.md)

## 4. Results and evidence

### Main results (RTX 6000, o3/o3, N = 10)

- Full 250 tasks: CudaForge reports 97.6% Correct, median 1.107, 75th percentile 1.592, mean Perf 1.677 and Fast1 70.80%. [Table 1](../pdfs/31-cudaforge.md)
- One-shot OpenAI-o3 on the same 250 tasks: 57.6% Correct, median 0.390, 75th percentile 1.014, Perf 0.680, Fast1 31.60%. [Table 1](../pdfs/31-cudaforge.md)
- Per level, CudaForge reports Level 1: 96% Correct, median 1.044, Perf 1.448, Fast1 54.0%; Level 2: 100%, 1.124, 2.104, 89.0%; Level 3: 96%, 1.081, 1.283, 68.0%. [Table 2](../pdfs/31-cudaforge.md)
- **Derived observation:** the median is below the mean at every level (1.044 against 1.448, 1.124 against 2.104, 1.081 against 1.283), so the speedup distributions are right-skewed and the mean, most of all at Level 2, depends on a minority of large speedups. [Table 2](../pdfs/31-cudaforge.md)
- Levels 1–2 only (200 tasks): 98.0% Correct, median 1.112, 75th percentile 1.617, Perf 1.776, Fast1 71.50%. [Table 1](../pdfs/31-cudaforge.md)
- On D\* (25 tasks): 100% Correct, median 1.322, 75th percentile 1.736, Perf 1.767, Fast1 84.00%. [Table 1](../pdfs/31-cudaforge.md)
- **Derived:** Table 2's level rows reproduce Table 1's overall row when weighted by task count: (96 + 100 + 48)/250 = 97.6% Correct; (1.448·100 + 2.104·100 + 1.283·50)/250 = 1.677 Perf; (54 + 89 + 34)/250 = 70.8% Fast1. [Tables 1–2](../pdfs/31-cudaforge.md)
- **Derived:** Fast1 values are whole task counts only with all tasks as denominator (Level 3: 68.0% = 34/50; over 48 correct kernels it would be 32.6 kernels). The tables therefore behave as if Fast1 is a fraction of all tasks, not of correct kernels as the definition's wording says. [§3.1; Table 2](../pdfs/31-cudaforge.md)
- **Derived:** for one-shot o3, 79 of 250 tasks are faster than PyTorch and 144 are correct, so a median over correct kernels only would exceed 1; the reported median 0.390 fits a distribution in which failed tasks are counted with a low value. Weighting Table 2's Perf by correct counts gives 1.684, not the reported 1.677. Both observations are consistent with failed tasks being included in speedup aggregates; the source does not state the rule. [§3.1; Tables 1–2](../pdfs/31-cudaforge.md)

### Ablations on the Judge and its feedback

- o3-self-refine (no Judge, 10 rounds, 250 tasks): 90.8% Correct, median 1.012, Perf 1.107, Fast1 55.20%. [Table 1](../pdfs/31-cudaforge.md)
- o3-correction (Judge gives correction feedback only): 97.6% Correct, the same as full CudaForge, with median 1.031, Perf 1.222, Fast1 59.60%. [Table 1](../pdfs/31-cudaforge.md)
- o3-optimization (Judge gives optimization feedback only): 88.4% Correct, median 1.061, 75th percentile 1.483, Perf 1.509, Fast1 64.00%. [Table 1](../pdfs/31-cudaforge.md)
- Relative to o3-correction, adding optimization feedback changes mean Perf from 1.222 to 1.677 and Fast1 by 11.2 percentage points at equal correctness; relative to o3-optimization, adding correction feedback raises Correct by 9.2 percentage points. [Table 1](../pdfs/31-cudaforge.md)
- Full NCU metrics versus the 24-metric subset on D\*: Correct 100% in both; median 1.280 versus 1.322; 75th percentile 1.489 versus 1.736; Perf 1.414 versus 1.767; Fast1 80.00% versus 84.00% (derived: 20 versus 21 tasks). [Table 1](../pdfs/31-cudaforge.md)
- With the full metric set, each kernel takes "approximately 40 miuntes" on an RTX 6000 and about $1 of API cost, against 26.5 minutes and $0.3 for the subset. [§3.6, Comparison with using the entire set of NCU metrics](../pdfs/31-cudaforge.md)
- One-task illustration (Level 2 task 51, 10 rounds): the full-metrics curve stays nearly flat while the subset curve rises. Read from the figure's vector coordinates, the full-set line stays between about 1.04× and 1.09×, and the subset line goes about 1.10, 1.10, 3.34, 2.79, 2.75, 4.34, 4.46, 4.56, 4.56, 4.61× over rounds 1–10. [Appendix B.1; Figure 9](../pdfs/31-cudaforge.md)
- For that task the source quotes one Judge response from each configuration. The full-metrics Judge proposes fusing GEMM and epilogue into a CUTLASS tensor-core kernel; the subset Judge reports a DRAM-bound kernel ("102.9 % peak", "57 % long-scoreboard stalls") and proposes merging two loops to remove one global read. The authors call the first diagnosis misaligned and the second correct. [Appendix B.1](../pdfs/31-cudaforge.md)

### Comparisons with other systems

- Agentic Baseline, Levels 1–2, numbers taken from its paper: 95.0% Correct and 1.490 Perf, against CudaForge's 98.0% and 1.776 on Levels 1–2 on RTX 6000. Median, 75th percentile and Fast1 are not available for the baseline. [Table 1; §3.3; §3.3, Footnote 3](../pdfs/31-cudaforge.md)
- Kevin-32B: the source reports 82.0% correctness and 1.10× speedup, taken from Kevin's paper. The introduction attaches the condition "16 parallel trajectories with 8 refinement turns". [§1; §3.4](../pdfs/31-cudaforge.md)
- CudaForge was re-evaluated on H200 across Levels 1–3 for this comparison; the text reports 98.0% correctness and 1.662× on Levels 1–2, and 96.0% and 1.261× on Level 3. [§3.4](../pdfs/31-cudaforge.md)
- The authors state that Kevin-32B is not open-sourced and that "a fully fair and reproducible comparison cannot be conducted". They describe Kevin's evaluation set as "a self-constructed benchmark of comparable difficulty to KernelBench Levels 1 and 2", so the two sides are not on an identical task set. [§3.4](../pdfs/31-cudaforge.md)

### Cost and wall time

- Cost is measured on D\* on a single RTX 6000. API cost is total spend per task; time is end-to-end wall clock including compilation, execution, model generation and NCU profiling. [§3.5](../pdfs/31-cudaforge.md)
- CudaForge averages $0.30 and 26.5 minutes per task; by level, $0.29/28.5 min (Level 1), $0.30/24.1 min (Level 2), $0.33/27.1 min (Level 3). [Table 3](../pdfs/31-cudaforge.md)
- The Agentic Baseline is listed at $5.0 and 60.0 minutes in the table, and as "approximately 6 GPU hours on H100 and $5 per kernel" in the prose, cited to that paper's Appendix E. [Table 3; §3.5](../pdfs/31-cudaforge.md)
- The authors state that CudaForge "already surpasses the Agentic baseline while using no more than $0.15 and 10 minutes per task". In the plots the curve crosses the baseline's dashed line (about 1.49) near $0.15 and near 10 minutes (visual reading). [§3.5; Figure 6](../pdfs/31-cudaforge.md)
- The plotted axes run to $1.00 and 60 minutes, where the curve reads about 2.27× and 2.17× (visual reading); the source does not say which run or budget rule produced each point. [Figure 6](../pdfs/31-cudaforge.md)

### Round scaling, GPUs and base models (all on D\*)

- Scaling N to 30 raises mean Perf to 2.265 with 100% Correct, median 1.317, 75th percentile 1.777 and Fast1 92.00%. [Table 1](../pdfs/31-cudaforge.md)
- The round-scaling curve, read from the figure's vector coordinates, gives average speedup of about 0.42, 0.66, 0.89, 1.52 and 1.63 at rounds 1–5, then about 1.90, 2.03, 2.09, 2.17 and 2.27 at rounds 10, 15, 20, 25 and 30. [Figure 7](../pdfs/31-cudaforge.md)
- Across GPUs with 100% Correct in every case: RTX 6000 median 1.322, Perf 1.767, Fast1 84.0%; RTX 4090 1.188, 1.327, 80.0%; A100 1.371, 1.841, 84.0%; RTX 3090 1.155, 1.320, 72.0%. [Table 4](../pdfs/31-cudaforge.md)
- With o3 as Coder and a different Judge, Correct stays 100%: GPT-5 Judge gives Perf 2.114 and Fast1 96.0% (median 1.131); Claude-Sonnet-4 Judge 1.829 and 84.0%; GPT-OSS-120B Judge 1.364 and 76.0%. [Table 5](../pdfs/31-cudaforge.md)
- With o3 as Judge and a different Coder: GPT-5 100% Correct, Perf 1.896, Fast1 72.0%; Claude-Sonnet-4 88%, 1.398, 56.0%; GPT-OSS-120B 96%, 1.653, 68.0%; QwQ 84%, median 0.965, Perf 0.790, Fast1 44.0%. Derived: on 25 tasks, 88% and 84% Correct are 22 and 21 tasks. [Table 5](../pdfs/31-cudaforge.md)

### Case study: Level-1 task 95 (CrossEntropyLoss)

- Round 2 (optimization): the Judge cites 23.7% of active warps stalled on barriers and recommends warp-level shuffles, cutting `__syncthreads()` per block from 16 to 2; the text reports speedup going from 1.66× to 2.42×. [§4](../pdfs/31-cudaforge.md)
- Round 5 (correction): after an "Outputs are not close" failure, the Judge diagnoses an uninitialized `target_logit` in thread 0 and suggests broadcasting it via a shuffle; the text says the numerical issue then disappeared. [§4](../pdfs/31-cudaforge.md)
- Rounds 6–7 (optimization): the Judge tracks long-scoreboard stalls (about 65%, then about 71%), recommends fewer registers per thread to raise occupancy from about 48 to about 64 warps/SM, then buffering logits in shared memory; speedup is reported as rising from 3.436× to 3.762×. [§4](../pdfs/31-cudaforge.md)
- The figure shows the first round as a correction round at speedup 0, a drop at round 9 (about 3.5×, visual reading) and a starred final point near 3.85× at round 10; a suggested change does not always improve the next kernel. [Figure 8](../pdfs/31-cudaforge.md)

### Observations about CUDA-L1's released kernels

- The authors did not run CUDA-L1. They inspected its released final kernels and report that nine of its ten largest reported speedups "do not use custom CUDA kernels". [Appendix C](../pdfs/31-cudaforge.md)
- Examples reproduced in the source: Level 2 task 83 with a reported 120.3× speedup and no CUDA kernel in the listing; Level 1 task 12 with a reported 64.4× speedup, implemented as `B * A.unsqueeze(1)` in PyTorch. [Appendix C](../pdfs/31-cudaforge.md)
- The authors also report many CUDA-L1 speedups clustered within about ±5% of 1.00 and attribute them to fallbacks; the Level 1 task 3 listing (reported 1.006×) calls `torch.bmm` when the custom module fails to load or raises. No numeric head-to-head with CUDA-L1 is reported. [§5; Appendix C](../pdfs/31-cudaforge.md)

## 5. Authors' conclusions

- The authors conclude that a training-free multi-agent workflow can give "cost-effective, generalizable, and high-performance CUDA kernel optimization". [Abstract; §6](../pdfs/31-cudaforge.md)
- They attribute the advantage over Kevin-32B to hardware feedback and workflow design, which they say enables targeted optimization instead of "blind exploration". [§3.4](../pdfs/31-cudaforge.md)
- They argue that separating generation (Coder) from evaluation (Judge) is "critical", that correction feedback is a prerequisite, and that optimization feedback grounded in profiling is what produces efficiency gains. [§3.6, Comparison with o3-self-refine; §3.6, Comparison with o3-correction; §3.6, Comparison with o3-optimization](../pdfs/31-cudaforge.md)
- They conclude that "concise, focused hardware feedback is both more effective and more efficient than exhaustive profiling". [§3.6, Comparison with using the entire set of NCU metrics](../pdfs/31-cudaforge.md)
- They attribute low cost to three factors: targeted feedback that reduces rounds, the smaller metric set (shorter profiling, fewer input tokens), and lightweight memory. [§3.5](../pdfs/31-cudaforge.md)
- They conclude the framework is not tied to a specific base model or GPU and that it benefits from more rounds, approaching a ceiling as N grows. [§3.7](../pdfs/31-cudaforge.md)
- They present "fake kernels" as "a fundamental challenge in evaluating LLM-generated CUDA kernels". [§5](../pdfs/31-cudaforge.md)

## 6. Limitations and open questions

### Limitations stated by the source

- The source has no limitations or threats-to-validity section; the following are qualifications made in passing. [§5; §6](../pdfs/31-cudaforge.md)
- The Kevin-32B comparison is acknowledged as not fully fair or reproducible; Agentic Baseline results are copied from that paper, cover Levels 1–2 only, and could not be rerun because its code was not released. [§3.3, Footnote 3; §3.4](../pdfs/31-cudaforge.md)
- Several studies use the 25-task D\* instead of the full benchmark because of cost. [§3.1; §3.7](../pdfs/31-cudaforge.md)
- Gains slow between 10 and 30 rounds, and more rounds carry "additional inference cost". [§3.7](../pdfs/31-cudaforge.md)

### Source inconsistencies

- **Source inconsistency (o3-self-refine correctness):** Table 1 gives 90.8%; the ablation prose says correctness improves "from 57.6% to 92.8%". The 90.8% value is table-specific. [Table 1; §3.6, Comparison with o3-self-refine](../pdfs/31-cudaforge.md)
- **Source inconsistency (o3-correction Fast1):** Table 1 gives 59.60%; the prose gives 58.8%. [Table 1; §3.6, Comparison with o3-correction](../pdfs/31-cudaforge.md)
- **Source inconsistency (full-metrics correctness):** the prose says the full metric set "leads to lower correctness and performance"; Table 1 shows 100% Correct for both configurations on D\*. [Table 1; §3.6](../pdfs/31-cudaforge.md)
- **Source inconsistency (30-round speedup):** 2.265 in Table 1 and §3.3, 2.271 in §3.7, 2.27 in the abstract. [Abstract; Table 1; §3.3; §3.7](../pdfs/31-cudaforge.md)
- **Source inconsistency (Figure 5 versus §3.4):** the text says "As shown in Figure 5" and reports H200 values (1.662× for Levels 1–2, 1.261× for Level 3), but the figure's bars are labelled 1.448, 2.104 and 1.283 with 96.0%/100.0%/96.0% correctness, which are the RTX 6000 values of Table 2. The image embedded in the PDF is pixel-identical to the preserved asset. No table of H200 results exists, so the H200 numbers rest on the prose alone. [§3.4; Figure 5; Table 2](../pdfs/31-cudaforge.md) [PDF p. 12](../pdfs/31-cudaforge.pdf)
- **Source inconsistency (Agentic Baseline conditions):** Table 3 lists 60.0 minutes while the prose and abstract say about 6 H100 GPU hours per kernel; Table 1's caption says "All experiments here are run in RTX 6000" although the Agentic Baseline row is copied from that paper. [Abstract; Table 1 caption; Table 3; §3.3, Footnote 3; §3.5](../pdfs/31-cudaforge.md)
- **Source inconsistency (GPU identity):** §3.2 names a "Quadro RTX 6000"; Table 4 labels the device "RTX 6000(Ada Arch-Data center level)". The source does not say whether these are the same card. [§3.2; Table 4](../pdfs/31-cudaforge.md)
- **Source inconsistency (Kevin's benchmark):** the introduction says Kevin-32B's 1.10× is "over KernelBench Level 1-2"; §3.4 says Kevin was evaluated on a self-constructed benchmark of comparable difficulty. [§1; §3.4](../pdfs/31-cudaforge.md)
- **Source inconsistency (correctness wording):** outputs must be "identical" in the metric definition, within 1e-4 in the test description, and within "atol=1e-4 or rtol=1e-4" in the Coder prompt. [§2.2, Design of Correctness Tests; §3.1; Appendix A.3](../pdfs/31-cudaforge.md)
- **Source inconsistency (metric selection):** the prose computes the Top-20 per task over 10 pooled kernels; Algorithm 2 computes a Top-20 per kernel inside the loop. The prose keeps "correct" kernels; Algorithm 1 has no correctness filter and assumes all 100 are retained. [§2.3; Algorithm 1; Algorithm 2](../pdfs/31-cudaforge.md)
- **Source inconsistency (alias removal):** Step 2 says aliases and collinear indicators are removed, yet the Conv2D Top-20 lists `smsp__inst_executed.avg` and `.sum` (and the two `inst_issued` forms) with identical correlations. [§2.3, Step 2; Appendix B.2, Table 6](../pdfs/31-cudaforge.md)
- **Source inconsistency (Coder inputs):** §2.1 says the Coder "receives only feedback from the Judge"; the Coder correction template also includes the raw error log, and the optimization template includes the GPU spec. Neither template has a field for the PyTorch reference, although §2.2 says the Coder receives the task description each round. [§2.1; §2.2, Design of Coder; Appendix A.3](../pdfs/31-cudaforge.md)
- **Source inconsistency (Figure 7 versus Table 1):** the curve reads about 1.90× at round 10 (from vector coordinates), while Table 1 reports 1.767 for the 10-round D\* run; at round 30 the curve reads about 2.27×, matching the table. Figure 6 shows the same pattern (about 1.89× at $0.30, visual reading). [Figure 6; Figure 7; Table 1](../pdfs/31-cudaforge.md)
- **Source inconsistency (claims versus tables):** "consistently outperforms all baselines across all metrics" sits beside a tie on Correct with o3-correction (97.6%); "all combinations achieve high correctness and strong performance, comparable to or even surpassing" sits beside QwQ/o3 at 84% Correct and 0.790 Perf. [§3.3; §3.7; Tables 1 and 5](../pdfs/31-cudaforge.md)

### Analyst caveats and missing evidence

- **Sampled testing, not proof:** correctness means agreement within 1e-4 on the benchmark's predefined inputs. Nothing is shown about other input values, shapes, or data races that do not change outputs on those inputs. [Basis: §2.2, Design of Correctness Tests](../pdfs/31-cudaforge.md)
- **Selection on the reporting measurement:** the best correct kernel over up to N rounds is chosen and reported using the same tests and timings that guided the loop; there is no held-out validation or re-timing step, so reported speedups may include selection optimism. [Basis: §2.1; §3.1](../pdfs/31-cudaforge.md)
- **Aggregation rule unstated:** whether mean, median and percentile speedups include failed tasks is not stated; the derived checks in §4 suggest they do, but this is inference from table arithmetic. Compare Perf across papers only after fixing this rule. [Basis: §3.1; Tables 1–2](../pdfs/31-cudaforge.md)
- **Timing validity unknown:** with no stated warm-up, trial count or variance, small ratios near 1 (for example medians of 1.04–1.12) cannot be judged against measurement noise. [Basis: §3.1; Table 2](../pdfs/31-cudaforge.md)
- **Single runs on a small subset:** D\* has 25 tasks and each configuration appears to be run once. The 30-round median (1.317) is lower than the 10-round median (1.322), which would not occur if the 30-round result continued the 10-round run with best-so-far selection on unchanged timings; separate runs or re-measurement noise are therefore present but unquantified. [Basis: Table 1; §3.1](../pdfs/31-cudaforge.md)
- **Partial replacement is permitted:** the first-round prompt lets the model leave operators unchanged, so a "correct" Level-3 solution may be mostly PyTorch with a small custom kernel. The manual "fake kernel" check is undefined, and how much of each solution is custom CUDA is not reported. [Basis: Appendix A.1; §5](../pdfs/31-cudaforge.md)
- **Possible selection–evaluation overlap:** the metric subset was chosen on tasks such as Conv2D and MatMul, and the source lists matrix multiplication and convolutions among KernelBench's own operator types; it does not say whether selection tasks were excluded from evaluation. [Basis: §2.3; §3.1; Appendix D.2](../pdfs/31-cudaforge.md)
- **Runtime proxies in the subset:** `sm__cycles_active.avg` correlates 1.000 and 0.998 with runtime in the two examples and is one of the 24 metrics; it is close to a restatement of runtime. Several of the 24 (for example the stall-reason metrics) appear in neither example list, so their selection cannot be checked from the source. [Basis: Appendix B.2, Tables 6–7; Appendix B.3, Table 8](../pdfs/31-cudaforge.md)
- **Confounded ablations:** the full-metrics variant changes both metric count and prompt length/cost; o3-self-refine changes both the number of agents and the prompt structure. The source supports "this configuration scored lower", not a clean causal attribution. [Basis: §3.2; §3.6](../pdfs/31-cudaforge.md)
- **Baseline strength:** speedups are relative to KernelBench's PyTorch reference; the source does not discuss the reference's precision or library settings or compiled baselines. [Basis: §3.1](../pdfs/31-cudaforge.md)
- **Cross-paper comparisons:** the Kevin and Agentic Baseline comparisons differ in hardware, task set, budget and possibly metric definitions; treat them as indicative only. [Basis: §3.2; §3.4](../pdfs/31-cudaforge.md)

## 7. Relevance to this project — interpretation

- **Proposal:** keep a Judge-like planner strictly advisory in the Stage 3 loop. Survival in the kernel pool should depend only on the fuzzing check and the PyGPUBench measurement, as acceptance in this source depends on executed tests and not on the Judge's text. [Basis: §2.1; §2.2, Design of Correctness Tests](../pdfs/31-cudaforge.md)
- **Proposal:** add profiler output to the results fed back into planning, starting with a small curated metric set plus static GPU specifications, and test curated-versus-full as a project ablation with repeated runs, since the source's evidence is one run on 25 tasks. [Basis: §2.3; Table 1; §3.6](../pdfs/31-cudaforge.md)
- **Proposal:** split feedback by kernel state: failed kernels get an error-log-based diagnosis, survivors get profiler-based advice. The source's two single-mode variants each lost either correctness or speed. [Basis: Table 1; §3.6](../pdfs/31-cudaforge.md)
- **Proposal:** have the planner emit one named bottleneck, the metrics it relied on, and one planned change per kernel in a fixed schema, and store it with the kernel so that later analysis can relate each suggestion to the measured change. [Basis: Appendix A.2; §4; Figure 8](../pdfs/31-cudaforge.md)
- **Proposal:** use a CudaForge-style single chain (refine only the previous candidate, no pool, no history) as a baseline against the project's batch-and-pool design at matched round, token and GPU-time budgets. [Basis: §2.2, Design of Coder; §3.2](../pdfs/31-cudaforge.md)
- **Proposal:** keep Stage 4's larger fuzzing check and a fresh performance measurement separate from the data used for selection during the loop, because this source selects and reports on the same tests and timings. [Basis: §2.1; §3.1](../pdfs/31-cudaforge.md)
- **Proposal:** add an automated check that a candidate actually executes custom kernel code and has no silent fallback path to the reference operator, and record what share of the module is custom; the source relied on manual inspection. [Basis: §5; Appendix C; Appendix A.1](../pdfs/31-cudaforge.md)
- **Proposal:** report correctness, speedup over all tasks, speedup over correct kernels only, and a faster-than-reference rate with explicit denominators, since this source's aggregation rules had to be inferred. [Basis: §3.1; Tables 1–2](../pdfs/31-cudaforge.md)
- **Proposal:** log API cost and wall time per round, including profiling time, and plot quality against a clearly defined budget axis; decide the fixed round count from the project's own curve. [Basis: §3.5; Table 3; Figures 6–7](../pdfs/31-cudaforge.md)
- **Proposal:** if a metric subset is derived for the project, compute it from the pool's fastest and slowest correct kernels per task family, exclude direct runtime proxies, and keep selection tasks disjoint from evaluation tasks. [Basis: §2.3; Appendix B.2–B.3](../pdfs/31-cudaforge.md)
- **Proposal:** treat the 1e-4 fixed-input tolerance as a weaker oracle than the project's fuzzing stage intends, and state tolerance, input generation and trial counts in the Stage 1 spec. [Basis: §2.2, Design of Correctness Tests; Appendix A.3](../pdfs/31-cudaforge.md)

## 8. Retrieval notes

- Search aliases: CudaForge, Coder–Judge, Coder and Judge agents, hardware feedback, Nsight Compute, NCU metrics, 24-metric key subset, correction mode, optimization mode, lightweight memory, o3-self-refine, fake kernels, CUDA-L1 fallback, D\* stratified subset, training-free kernel optimization.
- Questions answered: What do the Coder and Judge each see? Does the Judge decide correctness? Which NCU metrics are used and how were they chosen? What does the full-metrics ablation show? Which KernelBench tasks and denominators back each number? What do a kernel's cost and wall time refer to? Under what conditions is CudaForge compared with Kevin-32B, the Agentic Baseline and CUDA-L1? Which reported numbers disagree inside the paper?
- Related local analyses: [KernelBench](01-kernelbench.md) for the benchmark and its metrics; [robust agentic CUDA](08-robust-agentic-cuda.md), which by title is the work this source calls the Agentic Baseline; [Kevin](32-kevin.md) for the RL comparison system; [KernelBench-Verified](03-kernelbench-verified.md) for baseline and hidden-test concerns; [Self-Refine](18-self-refine.md) and [Reflexion](19-reflexion.md) for single-model refinement loops; [generation to judgment](13-generation-to-judgment.md) for LLM-as-a-judge background; [METR KernelAgent](21-measuring-automated-kernel-engineering.md), [KernelFoundry](30-kernelfoundry.md) and [K-Search](24-k-search.md) for other kernel-search loops; [benchmark reliability](27-performance-benchmark-reliability.md) and [noisy-environment benchmarking](33-robust-benchmarking-noisy-environments.md) for timing validity.
- These links are navigation aids, not controlled comparisons; protocols, hardware and metric definitions differ between the works.

## 9. Source coverage and verification notes

- Read the complete preserved Markdown: abstract, §1–§6, references, Appendix A (all four prompt templates), Appendix B.1–B.3 (case study, Tables 6–8), Appendix C (three CUDA-L1 listings) and Appendix D.1–D.2. Prompts and code listings were treated as quoted material.
- Inspected the raster figures directly: Figure 3 (Kevin versus CudaForge illustration), Figure 4, Figure 5, Figure 6(a)/(b) and Figure 8. Values quoted from Figures 6 and 8 are approximate visual readings.
- Figures 2, 7 and 9 are vector files without rendered text. Their labels were recovered from embedded character data, and the Figure 7 and Figure 9 values were computed from the plotted line coordinates and axis-tick positions; they are approximate and were not viewed as rendered images.
- Figure 1's content is not recoverable from the Markdown (only axis ticks and legend strings survive) and its plotted points were not read; the caption says the experimental details are in §3.
- PDF checks: the file has 33 pages; the text layer of PDF pp. 10, 13, 15 and 16 matches the Markdown's Tables 1–5; the images embedded on PDF pp. 11 and 12 are pixel-identical to the preserved Figure 4 and Figure 5 assets. The rest of the PDF was scanned by keyword, not read page by page.
- Conversion problems: the author/affiliation block is garbled; Figure 1 appears as loose tick and legend lines before §1; Algorithms 1–2 are flattened text; Table 3 is preserved as an HTML table.
- Derived arithmetic (Table 2 to Table 1 aggregation, Fast1 denominators, D\* task counts) is labelled in §4 and was done by hand from the table values.
- Not verified: the code repository, the cited Kevin, Agentic Baseline and CUDA-L1 papers, and whether the quoted comparison numbers match those sources. No experiments were reproduced and no kernels were compiled, profiled or timed.

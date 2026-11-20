---
paper_id: "01"
title: "KernelBench: Can LLMs Write Efficient GPU Kernels?"
source_version: "2502.10517v1"
source_markdown: "../pdfs/01-kernelbench.md"
source_pdf: "../pdfs/01-kernelbench.pdf"
source_url: "https://arxiv.org/abs/2502.10517"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "kernelbench", "benchmarking", "cuda", "test-time-compute", "execution-feedback", "profiling", "hardware-awareness", "correctness"]
summary: "Introduces 250 PyTorch-defined kernel-generation tasks and the correctness-gated fast_p metric, with one-shot, sampling, feedback, and hardware-context experiments."
---

# KernelBench: Can LLMs Write Efficient GPU Kernels?

## Paper identity and scope

- This analysis covers arXiv 2502.10517v1, the original KernelBench paper; numerical results refer to its frozen model/toolchain configurations. [§§3–5](../pdfs/01-kernelbench.md)
- The research question is whether language models can produce kernels that both preserve a PyTorch computation and run faster than its reference implementation. [Abstract; §1](../pdfs/01-kernelbench.md)
- KernelBench supplies an evaluation framework and workload suite, not a training corpus of optimal target kernels. [§3.3](../pdfs/01-kernelbench.md)
- The benchmark permits different low-level languages/libraries, but the reported main experiments generate raw inline CUDA within PyTorch wrappers. [§3.1; §6.3](../pdfs/01-kernelbench.md)

## Concepts and terminology

- A task provides Model, a torch.nn.Module with __init__ and forward, plus get_inputs and get_init_inputs describing tensor shapes and initialization parameters. [§3.1; Appendix A](../pdfs/01-kernelbench.md)
- The candidate is ModelNew, which must integrate custom computation into the same callable interface; generating peripheral host/wrapper code is part of the task. [§3.1; Appendix A](../pdfs/01-kernelbench.md)
- Fixed shapes and precision are visible task information. Numerical values vary through randomized tensor generation, rather than evaluating arbitrary unseen shapes. [§3.1; Appendix A](../pdfs/01-kernelbench.md)
- Level 1 comprises 100 primitive-operation tasks, including GEMMs, convolutions, losses, activations, and normalization. Existing optimized library calls can make this level difficult. [§3.2](../pdfs/01-kernelbench.md)
- Level 2 comprises 100 operator sequences, typically with fusion opportunities; its challenge differs from beating a single library primitive. [§1; §3.2](../pdfs/01-kernelbench.md)
- Level 3 comprises 50 complete model architectures drawn from public repositories, including AlexNet and MiniGPT. Architecture-level forward latency is the target. [§3.2](../pdfs/01-kernelbench.md)
- Operator fusion reduces intermediate memory traffic and launch overhead by combining computations after data is loaded; tiling reuses data within limited fast memory. [§5.2.1; §6.1](../pdfs/01-kernelbench.md)
- Hardware-aware optimization also includes shared/register-memory management, recomputation, asynchrony, and specialized instructions such as tensor-core operations. [§3.1; §5.2](../pdfs/01-kernelbench.md)
- Speedup is reference mean latency divided by candidate mean latency. A speedup below one denotes a slowdown. [§3.3; Appendix B.1](../pdfs/01-kernelbench.md)
- fast_p is the fraction of all benchmark tasks whose candidate is correct AND has speedup strictly greater than p. It is not the mean speedup of solved tasks. [§3.3](../pdfs/01-kernelbench.md)
- fast_0 reduces to correctness rate; fast_1 counts correct candidates beating baseline; increasing p raises the required acceleration threshold. [§3.3](../pdfs/01-kernelbench.md)
- fast_p@k measures whether at least one of k sampled candidates meets correctness and performance requirements, using execution to select successes. [§5.1.1](../pdfs/01-kernelbench.md)
- fast_p@N for refinement similarly means at least one successful candidate by turn N, not necessarily the final candidate. [§5.1.2; Table 1](../pdfs/01-kernelbench.md)
- Execution failure covers compilation/runtime errors and memory violations; functional failure covers output-shape and output-value mismatch after execution. [§4.2](../pdfs/01-kernelbench.md)
- G denotes previous generated code, E compiler/execution/correctness feedback, and P operator timing/profiling information; these support controlled feedback ablations. [§5.1.2; Table 1](../pdfs/01-kernelbench.md)

## Methodology and experimental setup

- The primary environment is an NVIDIA L40S, Python 3.10, PyTorch 2.5.0+cu124, and CUDA 12.4; separate experiments vary GPUs. [Appendix B](../pdfs/01-kernelbench.md)
- Correctness compares Model and ModelNew on five randomized inputs of prescribed shapes and precision. This is sampled testing, not proof of equivalence. [§3.3; Appendix B.1–B.2](../pdfs/01-kernelbench.md)
- Timing measures module.forward using CUDA events after three warmups and 100 timed trials; comparisons use mean latency, with min/max/standard deviation also recorded. [Appendix B.1](../pdfs/01-kernelbench.md)
- Only one evaluation occupies a GPU at a time; the paper reports timing coefficient of variation consistently below 3%. [Appendix B.1](../pdfs/01-kernelbench.md)
- Compilation is excluded from performance comparisons; steady-state runtime and compilation expense are different quantities. [Appendix B.4](../pdfs/01-kernelbench.md)
- Most analyses use eager PyTorch; default torch.compile uses Inductor, with additional reduce-overhead, max-autotune, max-autotune-no-cudagraphs, and cudagraphs comparisons. [Appendix B.4; Tables 2–3](../pdfs/01-kernelbench.md)
- The one-shot prompt includes one simple addition example to demonstrate Model/ModelNew formatting, then uses greedy decoding for the target task. [§4.1; Appendix C.1](../pdfs/01-kernelbench.md)
- Models include GPT-4o, OpenAI o1, DeepSeek-V3/R1, Claude 3.5 Sonnet, and Llama 3.1 70B/405B Instruct. These are historical evaluated versions, not current capability claims. [§4.1; Figure 2](../pdfs/01-kernelbench.md)
- Repeated sampling explores up to 100 candidates; temperatures are 1.6 for DeepSeek-V3 and 0.7 for Llama 3.1 70B. [§5.1.1; Appendix C.2](../pdfs/01-kernelbench.md)
- The matched-budget comparison uses ten inference calls. DeepSeek-R1 is not repeatedly sampled because its API endpoint did not expose a temperature parameter. [§5.1.3; Table 1](../pdfs/01-kernelbench.md)
- Refinement uses up to ten turns at temperature zero to isolate feedback effects, comparing G, G+E, and G+E+P. Timeout/deadlock feedback is a generic timeout message. [Appendix C.3; Table 1](../pdfs/01-kernelbench.md)
- Few-shot experiments add fused GELU, tiled matmul, and minimal FlashAttention implementations to demonstrate fusion, tiling, and shared-memory I/O management. [§5.2.1; Appendix F](../pdfs/01-kernelbench.md)
- Hardware-context experiments provide GPU type, memory, bandwidth, TFLOPS, and thread/warp/block/SM concepts; target-specific generations are timed on the relevant GPU. [§5.2.2; Appendix G.2](../pdfs/01-kernelbench.md)
- The cross-hardware transfer experiment runs the same previously generated code on L40S, H100, A100, L4, T4, and A10G and remeasures each device's reference baseline. [§4.4; Appendix G.1](../pdfs/01-kernelbench.md)
- Batch evaluation separates parallel inference, CPU precompilation/caching, and GPU execution, allowing many devices to be used while keeping each timing run isolated. [Appendix H.1](../pdfs/01-kernelbench.md)
- The iterative orchestrator treats each task as a finite-state machine; independent processes acquire GPUs via semaphores only for execution/profiling, overlapping inference and compilation. [Appendix H.2](../pdfs/01-kernelbench.md)
- Faulty CUDA code can cause illegal accesses or deadlocks; orchestration captures errors and replaces affected processes rather than allowing one candidate to kill the whole run. [Appendix H.2](../pdfs/01-kernelbench.md)

## Results and evidence

- One-shot eager-baseline fast_1 for levels 1/2/3 is R1 12/36/2%, o1 10/24/12%, V3 6/4/8%, Claude 10/7/2%, GPT-4o 4/5/0%, Llama70B 3/0/0%, Llama405B 3/0/2%. [§4.1; Figure 2](../pdfs/01-kernelbench.md)
- Reasoning models have fewer overall failures mainly because they avoid more execution failures; functional numerical correctness remains difficult across the evaluated models. [§4.2; Figure 3](../pdfs/01-kernelbench.md)
- Among correct kernels, level-1 and level-3 median speedups are below one; level 2 is only slightly above one. A few fast outliers do not imply broad success. [Appendix B.3; Figure 8](../pdfs/01-kernelbench.md)
- Default torch.compile sometimes has reproducible runtime overhead on small level-1 tasks; a candidate can beat compiled PyTorch while losing to eager mode, even after excluding compilation time. [§4.1 footnote; Appendix B.4](../pdfs/01-kernelbench.md)
- Against compiled baselines, R1's level-2 fast_1 varies from 52% with cudagraphs to 28% with reduce-overhead, illustrating baseline-configuration sensitivity. [Appendix B.4; Table 3](../pdfs/01-kernelbench.md)
- V3's level-2 fast_1 grows from 4% one-shot to 37% with 100 independent samples. Sampling increases opportunity to find a valid fast candidate. [§5.1.1; Figure 5](../pdfs/01-kernelbench.md)
- Sampling still fails for a group of 34 level-1 convolution variants for V3, even at 100 attempts each; low intrinsic success probability can limit inference scaling. [§5.1.1](../pdfs/01-kernelbench.md)
- With ten calls, V3 repeated-sampling fast_1 is 11/14/14% across levels; its G+E+P refinement achieves 19/6/14%, so feedback does not dominate sampling uniformly. [Table 1](../pdfs/01-kernelbench.md)
- R1's G+E+P fast_1 reaches 43/72/18% versus its Table 1 one-shot 12/36/2%. These are best-so-far successes, not terminal-candidate performance. [Table 1](../pdfs/01-kernelbench.md)
- R1 with G alone reaches 18/44/4%; G+E gives 41/62/12%. Execution feedback supplies much of the gain; profiler additions improve this particular model further. [Table 1](../pdfs/01-kernelbench.md)
- At ten turns, R1 G+E+P correctness reaches 95/92/42%; G+E reaches 95/85/50%. More feedback can improve one metric or level while hurting another. [Appendix E; Table 8](../pdfs/01-kernelbench.md)
- For Llama70B, G+E+P fast_1 is 7/4/2% versus G+E 5/5/8%, another example of model- and level-dependent effects. [Table 1](../pdfs/01-kernelbench.md)
- On convolution task L1 P63, R1 improves from 9.1 ms to a best 1.13 ms at turn 8, but finishes at 1.46 ms and never beats the 0.47 ms eager baseline. [Appendix D.4.1; Table 4](../pdfs/01-kernelbench.md)
- For L2 P73, all ten candidates compile but only turn 10 is correct; its 3.16 ms runtime loses badly to 0.105 ms eager. Repair success is not optimization success. [Appendix D.4.2; Table 5](../pdfs/01-kernelbench.md)
- For L2 P23, corrected CUB usage yields intermediate correct candidates but later generations become incorrect again; retaining the best verified candidate matters. [Appendix D.4.2; Table 6](../pdfs/01-kernelbench.md)
- For L1 P54, ten compiled generations remain numerically incorrect despite execution/profiling feedback, repeatedly producing value errors. [Appendix D.4.3; Table 7](../pdfs/01-kernelbench.md)
- Few-shot o1 overall fast_1 declines from 10/24/12% to 6/16/8%; its correctness also declines, while generations become longer. Stronger optimization examples can encourage brittle code. [§5.2.1; Appendix F, Table 9](../pdfs/01-kernelbench.md)
- The authors report tiling improves 77% of level-1 GEMM variants relative to o1's own one-shot code, while generally still trailing PyTorch; this is not a 77% PyTorch win rate. [§5.2.1; Table 10](../pdfs/01-kernelbench.md)
- Eleven listed level-2 few-shot kernels beat eager PyTorch; examples include Conv2d–InstanceNorm–Divide at 0.0823 ms versus 0.0898 ms reference. [Appendix F; Table 11](../pdfs/01-kernelbench.md)
- With unchanged generated kernels, R1 level-2 fast_1 varies from 36% on L40S to 47% on A10G; hardware transfer does not preserve relative speed uniformly. [§4.4; Table 13](../pdfs/01-kernelbench.md)
- Hardware prompts have little effect on Llama70B/V3; R1 attempts WMMA for roughly half of level-1 matmul tasks, but most such attempts fail compilation. [§5.2.2; Appendix G.2](../pdfs/01-kernelbench.md)
- Selected hardware-conditioned correct generations improve by at least 2× over the paper's own generated baselines, but these outliers do not establish broad target-aware optimization. [§5.2.2](../pdfs/01-kernelbench.md)
- The diagonal-matmul example replaces materialization and dense multiplication with row scaling, giving Claude 3.5 Sonnet a reported 13× speedup on L1 P11. [Appendix D.1](../pdfs/01-kernelbench.md)
- Fusion examples report 2.9× GELU, 1.3× Softsign, 2.6× a matmul/divide/sum/scale chain, and 1.9× a partially fused attention variant. [§6.1; Appendix D.2](../pdfs/01-kernelbench.md)
- Shared-memory examples include cosine-similarity-loss reductions; the discussion reports no successful tensor-core instruction use in these experiments. [§6.1; Appendix D.3](../pdfs/01-kernelbench.md)
- The five-versus-100-input check on 100 generated kernels found 50 passing both settings, 19 value mismatches, four shape mismatches, ten runtime errors, and 17 compile errors, with no partial passes observed. [Appendix B.2](../pdfs/01-kernelbench.md)

## Authors’ conclusions

- The authors present KernelBench as a reusable automatic evaluation environment for joint correctness and performance, extensible to future workloads and hardware. [§6.2](../pdfs/01-kernelbench.md)
- They conclude off-the-shelf models struggle, but sampling and environmental feedback can uncover substantially more correct/fast kernels. [§§4–5; §6.2](../pdfs/01-kernelbench.md)
- They see promise in algorithmic simplification, fusion, and memory management, while noting weak exploitation of architecture-specific features. [§6.1](../pdfs/01-kernelbench.md)
- They propose better kernel training data, fine-tuning/reasoning/agentic methods, higher-level GPU abstractions, and accelerator coverage beyond GPUs. [§6.3](../pdfs/01-kernelbench.md)
- They propose increasing speed threshold p and strengthening reference baselines as capability improves; benchmark difficulty need not remain fixed. [§6.2](../pdfs/01-kernelbench.md)

## Limitations and open questions

- Explicit limitation: randomized comparisons are heuristic; Appendix B.2 itself calls for stronger systematic testing and possible formal verification. Five sampled inputs do not establish all-input correctness. [Appendix B.2](../pdfs/01-kernelbench.md)
- Explicit limitation: refinement uses greedy generation and the experiments focus on GPUs and raw CUDA, limiting conclusions about other search policies, languages, or accelerators. [Appendix C.3; §6.3](../pdfs/01-kernelbench.md)
- Analyst caveat: the 100-kernel five-versus-100-input experiment is limited evidence about that sample, not a universal justification for five tests. [Appendix B.2](../pdfs/01-kernelbench.md)
- Analyst caveat: ModelNew sees prescribed input-generation code and shapes; these results do not establish correctness under hidden input distributions or shape generalization. [§3.1; Appendix A](../pdfs/01-kernelbench.md)
- Analyst caveat: inference-call matching does not equalize tokens, wall time, GPU evaluation expense, or reasoning budget. [§5.1.3; Table 1](../pdfs/01-kernelbench.md)
- Analyst caveat: task-forward microbenchmark gains do not by themselves establish deployment impact, training stability, or distributed-system speedups, despite the authors' production-value motivation. [§3.3; §6.2](../pdfs/01-kernelbench.md)
- Source inconsistency: the introduction says refinement starts at 12/36/12%, but Table 1 and Figure 2 give R1 12/36/2%; this analysis uses the explicit table's 2% level-3 baseline. [§1; Table 1; Figure 2](../pdfs/01-kernelbench.md)
- Source caution: §4.3's broad “fewer than 15%” wording conflicts with some model/level cells, such as R1 level 2 at 36%; retain the exact denominator and table result. [§4.3; Figure 2](../pdfs/01-kernelbench.md)
- Source caution: Appendix B calls L40S memory “HBM,” while Table 12 separately lists device specifications. Do not reuse that memory-type wording as independently verified hardware documentation. [Appendix B; Table 12](../pdfs/01-kernelbench.md)

## Relevance to this project — interpretation

- Interpretation: adopt separate correctness, latency, and fast_p reports and retain per-task outcomes so conditional speedups cannot hide unsolved tasks. [§3.3](../pdfs/01-kernelbench.md)
- Interpretation: preserve a best-verified checkpoint through refinement; the final candidate can regress even when best-so-far plots look monotone. [Appendix D.4; Table 6](../pdfs/01-kernelbench.md)
- Interpretation: log baseline execution mode, software stack, input specification, timing protocol, and hardware with each result before comparing experiments. [Appendix B; Appendix G](../pdfs/01-kernelbench.md)
- Interpretation: benchmark independent sampling and feedback loops under both call and resource budgets, and preserve error categories to distinguish repair from optimization. [§5.1; Appendix H](../pdfs/01-kernelbench.md)
- Interpretation: use the supplied fusion/tiling examples as experimental interventions rather than assuming extra demonstrations improve total success. [§5.2.1; Appendix F](../pdfs/01-kernelbench.md)

## Retrieval notes

- Aliases: KernelBench, fast_p, fast_0, fast_1, fast_p@k, G/E/P feedback; useful for questions about task format, timing, sampling, refinement, baseline choice, and hardware conditioning.
- Navigation: [KernelBenchX](02-kernelbench-x.md) studies category structure; [KernelBench-Verified](03-kernelbench-verified.md) revisits baseline and testing. Their protocols are different, so compare only after inspecting each analysis.
- Navigation only: [METR automated kernel engineering](21-measuring-automated-kernel-engineering.md) and [robust agentic CUDA](08-robust-agentic-cuda.md) discuss later benchmark modifications; not independently analyzed in this note.

## Source coverage and verification notes

- Reviewed main §§1–6 and substantive Appendices A–H: task API, evaluation design, prompt configurations, selected kernel implementations/trajectory mechanisms, correctness ablations, few-shot results, hardware transfer, and orchestration. [§§1–6; Appendices A–H](../pdfs/01-kernelbench.md)
- Numeric claims were checked against the preserved tables and prose, including Tables 1–14 and Figures 2–8 captions; no plot was independently digitized and no experiment was reproduced.
- Complete example CUDA listings and the bibliography remain in the source; this note captures their algorithmic roles rather than reproducing every source line. [Appendices A, C–D, G](../pdfs/01-kernelbench.md)
- Unresolved source wording/numbering issues are identified above; this analysis does not silently substitute values from later KernelBench versions.

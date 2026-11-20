---
paper_id: "04"
title: "PTXBench: Benchmarking and Adapting LLMs for GPU Kernel Optimization with Architecture-specific PTX"
source_version: "2608.17379v3"
source_markdown: "../pdfs/04-ptxbench.md"
source_pdf: "../pdfs/04-ptxbench.pdf"
source_url: "https://arxiv.org/abs/2608.17379"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "ptx", "hardware-awareness", "dynamic-profiling", "h100", "b200", "supervised-fine-tuning", "repair-learning", "attention", "agent-evaluation"]
summary: "PTXBench tests runtime use of architecture-specific GPU instructions separately from correctness and speed, and studies repair-conditioned adaptation of Qwen3.6-27B."
---

# PTXBench: Benchmarking and Adapting LLMs for GPU Kernel Optimization with Architecture-specific PTX

## Paper identity and scope

- Analyzed version: arXiv 2608.17379v3. Metadata calls it “Benchmarking and Adapting”; the body title says “Evaluating and Adapting.” Both refer to this stored PTXBench artifact. [Title block; source metadata](../pdfs/04-ptxbench.md)
- The study probes direct architecture-specific PTX capability and a targeted post-training intervention, not just whether a generated kernel beats generic PyTorch. [§1](../pdfs/04-ptxbench.md)
- A benchmark instance combines a frontier-library reference, fixed workload, GPU architecture, and requested PTX instruction family. [§2.1](../pdfs/04-ptxbench.md)
- The main workload scope is BF16 GEMM and forward/backward attention on Hopper H100 and Blackwell B200. [§7; Appendix A.2](../pdfs/04-ptxbench.md)

## Concepts and terminology

- CUDA–PTX means CUDA kernels containing inline PTX, allowing explicit use of architecture mechanisms that generic CUDA or libraries may hide. [§1](../pdfs/04-ptxbench.md)
- PTX is compiled into native SASS; observing an instruction string in PTX or SASS does not establish that the workload executes it. [§2.2; Appendix A.2](../pdfs/04-ptxbench.md)
- Target-instruction correctness requires numerical correctness AND observed execution of selected tensor-compute or data-movement mechanisms. [§2.2](../pdfs/04-ptxbench.md)
- The qualifying rule is C OR D, so a candidate can qualify via data movement alone without architecture-specific tensor computation. [§2.2; Appendix A.2](../pdfs/04-ptxbench.md)
- On H100, C requires executed HGMMA; D requires TMA payload transfer. On B200, C requires UTCHMMA plus TMEM readback (LDTM or LDT), while D permits TMA or TMEM stores/copies. [Appendix A.2; Table 6](../pdfs/04-ptxbench.md)
- Cache-control, prefetch, and UTMAREDG do not satisfy the data-movement predicate. The benchmark targets selected mechanisms, not every architecture-specific opcode. [Appendix A.2](../pdfs/04-ptxbench.md)
- Predicate-enabled thread execution counts come from Nsight Compute, summed over kernels launched inside one isolated NVTX range. [Appendix A.2](../pdfs/04-ptxbench.md)
- Profiling failure is unknown, not proof that a target instruction did not execute; unknown cases are excluded from the positive target-instruction numerator. [§4.1; Appendix A.2](../pdfs/04-ptxbench.md)
- Turn correctness divides successful turns by all evaluated turns. It is distinct from trajectory pass rate or the probability of finding one success within a budget. [§4.1](../pdfs/04-ptxbench.md)
- Fast_p counts all turns that are correct and exceed speed threshold p; Fast_p^Inst additionally requires observed target-instruction execution. [§4.1](../pdfs/04-ptxbench.md)
- Best speedup is a peak among correct turns, with a separate instruction-qualified peak; high peak performance can coexist with very low success frequency. [§4.1](../pdfs/04-ptxbench.md)
- Table triplets summarize ≤1/≤4/≤8 turns, and parenthesized entries are unrestricted correctness/performance. A dash should not be silently read as a measured negative instruction trace. [§4.1](../pdfs/04-ptxbench.md)
- Fixit learns from the target model's failed kernel and its feedback, pairing those inputs with a correctness-filtered teacher repair and a teacher-synthesized rationale. [§3](../pdfs/04-ptxbench.md)
- Fixit separates repair teacher from reasoning teacher: the latter sees the accepted repair when constructing the explanation. This is supervised rationale synthesis, not independently observed student reasoning. [§3](../pdfs/04-ptxbench.md)
- Failures are collected once from the pre-adaptation checkpoint; the training distribution is not refreshed on-policy after every weight update. [§3](../pdfs/04-ptxbench.md)

## Methodology and experimental setup

- MiniPTXAgent starts from scratch and retains preceding kernels/feedback across up to eight model calls; it stops early at 1.2× speedup or context exhaustion. [§2.2; Appendix A.1](../pdfs/04-ptxbench.md)
- The shared knowledge pack supplies architecture parameters, C++ instruction wrappers, layout/synchronization/memory-consistency contracts, and fixed expert scheduling guidance where appropriate. [§2.1](../pdfs/04-ptxbench.md)
- Prompt packs contain 22,314 Qwen-tokenized tokens on H100 and 28,815 on B200; Blackwell's longer architecture contract accounts for much of the difference. [Appendix A.4; Table 7](../pdfs/04-ptxbench.md)
- CUDA source is precompiled in CPU containers using nvcc -O3 for sm_90a or sm_100a; only successful compilation reaches the isolated GPU service. [§2.2; Appendix A.1](../pdfs/04-ptxbench.md)
- Including cuBLAS or cuDNN headers in generated CUDA is forbidden irrespective of output correctness, to prevent bypassing the intended low-level programming task. [§2.2](../pdfs/04-ptxbench.md)
- Correctness verifies output shape and dtype followed by torch.allclose with atol=rtol=1e-2. [§2.2](../pdfs/04-ptxbench.md)
- Timing uses CUPTI, ten warmups and 50 timed iterations per trial, taking the median; reference/candidate latency gives speedup. [§2.2](../pdfs/04-ptxbench.md)
- Library versions are cuBLAS 13.1.0 for GEMM, cuDNN 9.20.0 for ordinary attention, and FlashInfer 0.6.14 for GQA. [§4.1](../pdfs/04-ptxbench.md)
- Each model/GPU/workload has three prompt variants, four eight-turn trajectories per variant: 96 potential turn rows. Curves average prompt-wise fractions and show min/max prompt spread. [§4.1; Appendix A.6.1](../pdfs/04-ptxbench.md)
- Primary MHA shapes are batch 4, heads 48, sequence 4096, head dimension 128; forward returns BF16 attention plus FP32 LSE, and backward consumes forward outputs/upstream gradients/LSE and returns Q/K/V gradients. [Appendix A.2](../pdfs/04-ptxbench.md)
- GEMM computes ABᵀ with A 8192×5120 and B 7168×5120 in BF16, producing 8192×7168 output. [Appendix A.2](../pdfs/04-ptxbench.md)
- Generalization includes d64/d96 MHA and three pooled GQA workloads: ragged causal prefill, paged causal prefill, and paged decode, with different head ratios/cache structures. [Appendix A.2](../pdfs/04-ptxbench.md)
- Instruction evidence hashes source and cubin container to invalidate stale cached profiling; NCU 2025.3.1 replays the exact candidate on its original workload. [Appendix A.2](../pdfs/04-ptxbench.md)
- The profiling service isolates candidate execution and exposes sanitization, correctness, debugging, and latency independently; racecheck is explicitly not used because of cost and incomplete detection. [§2.2; Appendix A.1–A.3](../pdfs/04-ptxbench.md)
- All six adaptation recipes start independently from Qwen3.6-27B, with Gemini 3.1 Pro repairs and usually GLM-5.2 rationale synthesis. [§5; §5.2](../pdfs/04-ptxbench.md)
- LoRA training uses rank 32, five epochs, batch two, learning rate 4.65e-4 with linear schedule, and loss only on the final assistant message. [Appendix A.2, Supervised fine-tuning](../pdfs/04-ptxbench.md)
- Training discards examples longer than 65,536 tokens, uses no validation split, evaluates final checkpoints, and primarily uses shuffle seed zero with two additional s1 seeds. [Appendix A.2](../pdfs/04-ptxbench.md)
- Base/adapted Qwen decoding shares temperature 1.0, top-p .95, top-k 20, presence penalty 1.5, and up to 81,920 output tokens. [Appendix A.2](../pdfs/04-ptxbench.md)
- Recipe counts are s1 158, s2 259, s3 406, s4 170, s5 258, s6 258. The s5/s6 pair changes rationale teacher while keeping repair examples fixed. [§5.2; Table 12](../pdfs/04-ptxbench.md)

## Results and evidence

- H100 Opus4.8 eight-turn target correctness is 94.8% GEMM, 90.6% forward, 75.0% causal forward, 79.2% backward, 44.8% causal backward; difficult backward variants remain substantially worse. [Table 1](../pdfs/04-ptxbench.md)
- B200 Opus4.8 unrestricted backward correctness reaches 91.7% while target-qualified correctness is only 10.4%; numerical success can rely on generic mechanisms. [Table 1](../pdfs/04-ptxbench.md)
- Base Qwen has no correct H100 primary kernels and one unrestricted B200 GEMM success without selected target instructions. [§4.2; Table 1](../pdfs/04-ptxbench.md)
- The vendor SWE-bench Pro figures differ by only 0.7 points for Gemini and Qwen, yet their PTXBench behavior differs sharply. The paper cautions those vendor scores use different scaffolds. [§4.2; Table 2](../pdfs/04-ptxbench.md)
- Gemini reaches 0.892× cuBLAS on B200 GEMM; Opus4.8 reaches 1.012×, but neither consistently matches frontier references across attention workloads. [§4.2; Table 9](../pdfs/04-ptxbench.md)
- For Gemini on H100 GEMM, Triton and CUDA–PTX peaks are close (0.991× versus 0.962×); Triton has higher Fast_p on attention workloads. [§4.3](../pdfs/04-ptxbench.md)
- For Gemini B200 backward attention, prose reports Triton 0.514× versus CUDA–PTX 0.198×, and causal backward 0.437× versus no correct CUDA–PTX candidate; see inconsistency note below concerning Table 9. [§4.3; Table 9](../pdfs/04-ptxbench.md)
- GPT-5.6 Sol's B200 Triton peak exceeds CUDA–PTX on all five workloads; on H100, CUDA–PTX slightly exceeds Triton on two causal attention peaks. [Appendix A.6.1; Figure 13](../pdfs/04-ptxbench.md)
- Architecture parameters alone produce 26.0% correct turns and no target-instruction success by eight turns in the Gemini ablation. Templates enable 19.8% target success; adding contracts yields 38.5%. [§4.5; Table 4](../pdfs/04-ptxbench.md)
- The contract improves target correctness by 18.7 percentage points over templates but does not maximize peak speed: 0.515× with contract versus 0.542× with templates alone. [Table 4](../pdfs/04-ptxbench.md)
- Antigravity/Gemini beats refinement on median best-qualified GEMM (0.583× versus 0.505×), but loses on all four attention workloads while using 3.32–5.06× median total tokens. [§4.4; Table 3](../pdfs/04-ptxbench.md)
- Codex/GPT-5.6 likewise beats refinement on GEMM (0.725× versus 0.641×) but trails on attention, with 3.47–5.78× token ratios under the same eight-evaluation-request budget. [Appendix A.6.2; Table 8](../pdfs/04-ptxbench.md)
- The smallest all-five-success recipe, s1, trains on four d128 MHA tasks and transfers to GEMM, all d64 variants, both d96 forward variants, and GQA, but not either d96 backward variant. [§5.1; Figure 4](../pdfs/04-ptxbench.md)
- The authors connect d96 backward difficulty to mismatch with H100 WGMMA's m64 tile, an architectural explanation rather than a demonstrated impossibility. [§5.1](../pdfs/04-ptxbench.md)
- Only relatively balanced s1 and s5 solve all five primary tasks. Larger s2/s3 fail causal backward, showing data count alone does not determine breadth. [§5.2; Tables 10–12](../pdfs/04-ptxbench.md)
- s5→s6 changes the rationale synthesizer from GLM to the student itself; s5 solves all five tasks, while s6 solves only GEMM. [§5.2; Tables 10–12](../pdfs/04-ptxbench.md)
- s1's eight-turn target success rates are 13.5/19.8/3.1/4.2/4.2% for GEMM/forward/causal-forward/backward/causal-backward; “solves all” therefore does not mean reliable per-turn generation. [Table 10](../pdfs/04-ptxbench.md)
- s1's corresponding best qualified speedups are 0.340/0.565/0.395/0.389/0.199×, remaining below frontier library speed despite adaptation. [Table 11](../pdfs/04-ptxbench.md)
- s5 improves breadth but is not uniformly best in peak speed; recipe choice changes task-specific tradeoffs. [§5.2; Tables 10–11](../pdfs/04-ptxbench.md)
- SFT increases reasoning length and shifts failures from compilation toward runtime/numerical errors, indicating improved executability rather than eliminated correctness problems. [§5.3; Figure 7](../pdfs/04-ptxbench.md)
- When adapted CUDA–PTX models generate Triton, correctness falls on every primary workload, yet causal-forward peak improves 0.238→0.632× and causal-backward 0.043→0.331×. [§5.3; Figure 8](../pdfs/04-ptxbench.md)
- Expert prompt guidance alone does not make base Qwen solve MHA; s1 without guidance already solves some MHA tasks, and s1 plus guidance reaches nonzero correctness on all four. [§5.4; Table 5](../pdfs/04-ptxbench.md)
- BM25-retrieved repair notes alone yield no correct kernels. Adding the corrected kernel raises success, but the authors explicitly call this solution-bearing supervision rather than evidence of equivalent unsupervised capability. [§5.4; Table 5](../pdfs/04-ptxbench.md)
- Additional s1 shuffle seeds preserve the broad transfer pattern, including no d96 backward success; one seed's GEMM cell has only 90 of 96 expected rows. [Appendix A.7; Figure 17](../pdfs/04-ptxbench.md)
- A decoy unused kernel containing TMA passes a static opcode scan but fails runtime execution qualification, motivating dynamic checks. [Appendix A.2](../pdfs/04-ptxbench.md)
- A candidate with stream-ordered use-after-free passes numerical checking but cannot be reliably NCU-classified; it explains the s1 causal-backward 4.2% qualified versus 5.2% unrestricted gap. [Appendix A.2](../pdfs/04-ptxbench.md)
- Retaining workload/reference state raises evaluation throughput 2.24×; pipelining raises rollout throughput 2.78× from concurrency four to 48, with four profiling GPUs approximately matching eight end-to-end. [Appendix A.3; Figure 12](../pdfs/04-ptxbench.md)
- The reliability monitor excludes unhealthy/thermally abnormal runs; the paper observed throttling distort latency by more than 15%. [Appendix A.3; Figure 11](../pdfs/04-ptxbench.md)
- The authors correct two FlashAttention-3 prompt-pseudocode problems: an incomplete loop bound and a missing probability-state update, also explicitly preserving old row maxima and normalizing output in the epilogue. [Appendix A.5; Listings 1–2](../pdfs/04-ptxbench.md)

## Authors’ conclusions

- The authors conclude runtime architecture usage, numerical correctness, and library-relative performance must be measured separately. [§8](../pdfs/04-ptxbench.md)
- They find forward workloads easier than backward attention and no evaluated model consistently matching frontier libraries across this suite. [§8](../pdfs/04-ptxbench.md)
- They conclude repair-conditioned SFT can improve a weak student's PTX capability, but coverage, balance, rationale quality, and transfer shape matter beyond record count. [§5; §8](../pdfs/04-ptxbench.md)
- They retain a role for high-level DSLs on newer architectures while acknowledging occasional stronger peaks from direct low-level implementations. [§4.3](../pdfs/04-ptxbench.md)
- They position the framework as an extensible controlled probe and adaptation-data environment, not a sufficient measurement of recursive self-improvement. [§7; §8](../pdfs/04-ptxbench.md)

## Limitations and open questions

- Explicit limitation: one 27B base model and modest LoRA datasets limit generalization to other model families and industrial-scale post-training. [§7](../pdfs/04-ptxbench.md)
- Explicit limitation: BF16 GEMM/attention on H100/B200 does not cover the full operator, precision, or hardware space. [§7](../pdfs/04-ptxbench.md)
- Explicit limitation: runtime instruction qualification says selected instructions execute, not how much useful work they perform or whether they caused speedup. [§2.2](../pdfs/04-ptxbench.md)
- Analyst caveat: C OR D permits a minimal qualifying data-transfer operation; proving substantive use would need a stronger workload-contribution check. [§2.2; Appendix A.2](../pdfs/04-ptxbench.md)
- Analyst caveat: numerical tests, sanitization without racecheck, and finite profiling do not establish race freedom or all-input correctness; the use-after-free case is direct evidence. [Appendix A.1–A.2](../pdfs/04-ptxbench.md)
- Analyst caveat: fixed evaluator-request budgets do not equalize tokens, model calls, or tool cost, and four trajectories give limited precision for harness rankings. [§4.4; Appendix A.6.2](../pdfs/04-ptxbench.md)
- Analyst caveat: choosing s1 as the smallest recipe that solves all primary problems and using no validation split makes these evaluations exploratory, not a held-out model-selection protocol. [§5.1; Appendix A.2](../pdfs/04-ptxbench.md)
- Source discrepancy: §4.3's Gemini B200 backward CUDA–PTX peak 0.198× differs from Table 9's 0.133× target and unrestricted value. Preserve the artifact-specific values rather than merging them. [§4.3; Table 9](../pdfs/04-ptxbench.md)
- Source naming differences include “GPT-5.6 Sol” in prose versus “GPT-5.6 xhigh” in tables, and the metadata/body title variation. Avoid treating labels as separate verified model releases. [Tables 1–2; title block](../pdfs/04-ptxbench.md)
- Analyst caveat: data provenance/exposure and the teacher-rationale ablation support bounded empirical conclusions; vendor coding scores and calendar cutoff dates are not causal measures of PTX training exposure. [§4.2; §5.2](../pdfs/04-ptxbench.md)

## Relevance to this project — interpretation

- Interpretation: archive candidate source/cubin hashes and dynamic evidence separately from textual instruction mentions. Represent failed profiling as unknown. [Appendix A.2](../pdfs/04-ptxbench.md)
- Interpretation: separate latency/correctness services from generation and precompilation; cache only immutable workload state and invalidate candidate-dependent evidence reliably. [Appendix A.3](../pdfs/04-ptxbench.md)
- Interpretation: collect failed-code/error/validated-repair triples with workload and hardware context for possible later adaptation, while holding out shapes and problem families. [§3; §5.1](../pdfs/04-ptxbench.md)
- Interpretation: report both success frequency and best-found performance, plus compute/token cost, so a rare fast candidate cannot hide reliability loss. [§4.1; §5.3](../pdfs/04-ptxbench.md)
- Interpretation: keep architecture documentation and expert schedules versioned and execution-checked; copied pseudocode can contain errors that contaminate the supposed controlled context. [Appendix A.4–A.5](../pdfs/04-ptxbench.md)

## Retrieval notes

- Aliases: PTXBench, MiniPTXAgent, Fixit, CUDA–PTX, target-instruction correctness, WGMMA, TMA, TMEM, C OR D qualification.
- Questions answered: how to verify actual instruction execution; how repair-conditioned SFT is built; what data balance/rationale teachers change; what remains unproven after numerical passes; how harness cost is compared.
- Navigation: [KernelBench](01-kernelbench.md) for a broader PyTorch workload framework and [TritonBench](05-tritonbench.md) for DSL generation. PTXBench's frontier-library references and turn denominators differ.
- Navigation only: [LARK](22-lark.md) concerns stronger semantic guarantees; dynamic instruction evidence is not equivalent to formal verification.

## Source coverage and verification notes

- Reviewed main §§1–8 and Appendix A.1–A.7, including instruction predicates, workload shapes, inference/training parameters, system throughput, prompt packs, corrected pseudocode, harness comparisons, recipe tables, and seed notes. [§§1–8; Appendix A](../pdfs/04-ptxbench.md)
- All reported locators refer to the stored Markdown/HTML numbering; source tables/captions were read, but plots were not independently digitized and GPU runs were not reproduced.
- Complete prompt/code listings remain available in the source; this note records mechanisms and consequential instructions without duplicating large knowledge packs.
- A temporary text simplification mangled a less-than sign in Appendix A.5; the affected A.5–A.7 material was re-read directly from the original Markdown before writing this analysis.

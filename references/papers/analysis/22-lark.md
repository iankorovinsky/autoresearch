---
paper_id: "22"
title: "LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels"
source_version: "2611.02028v1"
source_markdown: "../pdfs/22-lark.md"
source_pdf: "../pdfs/22-lark.pdf"
source_url: "https://arxiv.org/abs/2611.02028"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "formal-verification", "lean", "exact-arithmetic", "race-freedom", "trust-boundary", "proof-agents", "kernelcraft", "ptx", "cuda", "hip"]
summary: "LARK uses Lean-checked race-freedom, segmentation and equivalence proofs to certify supported GPU kernels relative to a pinned machine model and task interface, while leaving floating-point fidelity and host-allocation safety outside its theorem."
---

# LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels

## 1. Paper identity and scope

- This analysis covers arXiv `2611.02028v1`, a preprint presenting LARK, an agentic GPU-kernel verification framework instantiated for PTX, CUDA and AMD HIP. [Title; §1](../pdfs/22-lark.md)
- LARK targets equivalence between a low-level GPU kernel and a high-level Torch specification, with Lean 4 checking proofs produced by LLM agents. It is a verifier that can be integrated into generation, not a new language model or a benchmark-only contribution. [§1; §3–4](../pdfs/22-lark.md)
- Its central design objective is moving trust away from each generated kernel to a fixed, explicit set of machine semantics, specification and interface assumptions that can be reviewed ahead of time. [§1; §4.3](../pdfs/22-lark.md)
- The theorem concerns **exact arithmetic in a formal machine model**. Floating-point rounding, quantization, accumulation-order error and approximation error are tested separately and are not formally bounded by LARK. [§3; §4.3; Limitations](../pdfs/22-lark.md)
- The reported core experiments verify a pinned kernel for each task/language, isolating verification performance from generation variability. The headline pass rates are not end-to-end independent kernel-generation success rates. [§5 Setup; Appendix A.3, Table 10](../pdfs/22-lark.md)
- “Production kernels” means adapted versions of upstream kernels; none is verified byte-for-byte as shipped, and nine of ten have recorded semantic departures. This qualification is essential to interpreting the nine-of-ten result. [§5.3; Appendix A.14](../pdfs/22-lark.md)

## 2. Concepts and terminology

### Specification, interface and guarantee

- The specification `S` denotes a total function from the task's declared input domain to output tensors. Domain restrictions include shapes, types and stated constraints; this does not automatically quantify over arbitrary shapes or dtypes. [§3, Specification](../pdfs/22-lark.md)
- The machine model `M` supplies states, a partial semantics for each operation, a scheduling rule and value representations. Kernel meaning is defined relative to this model, not directly to every detail of physical GPU execution. [§3, Kernel and machine model](../pdfs/22-lark.md)
- The task interface `I` specifies memory regions, layouts, launch geometry and separation. It induces a precondition `In_I` and output extractor `Out_I`; these come from the task rather than being chosen to make a particular candidate easy to prove. [§3, Interface](../pdfs/22-lark.md)
- Equivalence requires, for every allowed input and every initial state satisfying the interface, fault-free completion and output equality to the specification. Quantification over otherwise arbitrary initial memory prevents the output from depending on undeclared initial contents. [§3, Eq. 1](../pdfs/22-lark.md)
- The model has **unbounded memories**. “Memory behavior” means independence from undeclared initial memory and correct placement/extraction of declared outputs; writes outside the interface can be invisible, so host-allocation bounds safety is not guaranteed. [§3 after Eq. 1; §4.1 Memory](../pdfs/22-lark.md)
- A certificate is evidence of the stated theorem under its premises. A rejection gives no general claim that a kernel is wrong: it may reflect unsupported constructs, a proof failure, a budget stop, or an actual counterexample. [§3, Problem definition; §5.1](../pdfs/22-lark.md)
- **Soundness** means certificates are issued only when the theorem is checked. **Completeness** concerns how many valid supported kernels are successfully certified. Strong soundness does not imply all correct kernels receive certificates. [§3, Requirements](../pdfs/22-lark.md)
- **Non-vacuity** is checked by a general satisfiability theorem for task signatures: a contradictory precondition must not make an incorrect statement trivially provable. [§4.2, Statements; Appendix A.8](../pdfs/22-lark.md)

### Machine state, values and faults

- One machine state represents the entire launch: program counters, thread-indexed registers, global memory, per-block shared memory, per-thread local memory, and launch coordinates/geometry. [§4.1, Memory](../pdfs/22-lark.md)
- A cell holds an undefined value or a typed payload. Supported tags include FP16, FP32, packed FP16 pairs, width-tagged integers and predicates; float payloads are exact reals with `−∞` admitted for running maxima. [§4.1, Values; Appendix A.7, Value types](../pdfs/22-lark.md)
- Integer payloads are exact integers; width tags do not make the theorem a full bitvector model of wraparound. Integer wraparound is outside the supported semantics. [Limitations; Appendix A.7; Appendix A.9](../pdfs/22-lark.md)
- Faults include reading undefined or wrongly typed values, misaligned vector access, negative addresses, unsupported exceptional results, exhausted execution fuel and non-convergent synchronization cuts. A proof must establish that the modeled run avoids faults. [§4.1, Faults](../pdfs/22-lark.md)
- Approximate operations such as reciprocal, reciprocal square root and base-two exponential are interpreted as exact mathematical functions. TF32 MMA is accepted as an FP32 carrier. These are explicitly declared idealizations, not hardware-accuracy proofs. [Appendix A.9; A.13](../pdfs/22-lark.md)

### Phase-structured concurrency

- A **cut** classifies where execution pauses: collective, warp, block, grid or halt, with ordinary operations having no cut. A barrier orders phases at its level; a collective executes once over all participating lane registers. [§4.1; Table 1](../pdfs/22-lark.md)
- Hierarchical folds run lanes, warps and blocks sequentially inside the model. This reduces proof complexity but requires per-kernel race-freedom conditions so that the reordered actions commute. [§4.1, Eq. 2; §4.2, Stage 1](../pdfs/22-lark.md)
- A phase's footprints `Wφ(u)` and `Rφ(u)` describe what each unit writes and reads. For distinct units, LARK requires disjoint write/write and write/read footprints. Concurrent reads are compatible. [§4.2, Obligation 1](../pdfs/22-lark.md)
- A warp collective is also treated as a unit with cross-lane register and memory footprints. A collective cut orders execution but is not automatically a memory barrier. [§4.2, Stage 1](../pdfs/22-lark.md)
- The race-freedom criterion is sufficient but conservative: it rejects benign races and certain communicating producer-consumer patterns even when final numerical outputs appear unaffected. [§4.2; Appendix A.10](../pdfs/22-lark.md)
- The reduction from arbitrary race-free schedules to the sequential fold relies on literature-based metatheory and is **not itself proved in Lean** in this system. It remains explicitly inside the trusted boundary. [§4.2, Stage 1; §4.3; Appendix A.13](../pdfs/22-lark.md)
- Every level carries a fuel allowance. Too little fuel produces a fault and blocks a proof; fuel is supplied execution data, not an assumption allowing nontermination to be ignored. [§4.1; Appendix A.7, Fuel](../pdfs/22-lark.md)

### Flat semantics, segmentation and proof obligations

- The **flat model** is the whole-launch fold over the parsed operation sequence, following individual operations and program counters. Its low-level shape is poorly matched to tensor-level mathematical reasoning. [§4.2, Stage 2](../pdfs/22-lark.md)
- An agent proposes a **segment-level model** using sequences, loops and conditionals over the same operation semantics. It exposes induction variables and reusable loop bodies without changing the intended computation. [§4.2, Stage 2](../pdfs/22-lark.md)
- The three obligations are `RF(K)` for race freedom; `SEG(K)` for equality of flat and segmented partial functions; and `EQ(K,S)` for fault-free segmented execution with outputs equal to the Torch semantics. [§4.2, Obligations 1–3](../pdfs/22-lark.md)
- Segmentation equality includes both successful and faulting behavior; it is not merely agreement on a few outputs. A mistaken agent segmentation must fail the first equivalence proof rather than silently weaken the final statement. [§4.2, Obligation 2](../pdfs/22-lark.md)
- The proof loop accepts only Lean-compiling proofs without `sorry`, with `#print axioms` showing standard axioms. A persuasive explanation from an agent cannot substitute for a checked proof. [§4.2, Proof loop](../pdfs/22-lark.md)
- Failure analysis is a search-control heuristic: it can continue, restart, resegment or send feedback to a generator. Its verdict does not issue certificates and therefore affects coverage/cost, not the acceptance theorem. [§4.2; Appendix A.7, Failure analysis](../pdfs/22-lark.md)

## 3. Methodology and experimental setup

### Offline preparation and trust accounting

- Per-language operation tables and a shared machine model are pinned before candidate verification. Operation definitions are tested on hardware over random strata, boundary values and lane/alignment cases. [§4.1; Appendix A.5](../pdfs/22-lark.md)
- The hardware sweep uncovered an erroneous exponential formalization: an approximate base-two exponential had been represented as `e^x` rather than `2^x`. This demonstrates that trusted semantic definitions can contain errors despite being outside the per-kernel proof loop. [Appendix A.5](../pdfs/22-lark.md)
- The Torch-to-Lean specification is another trusted artifact: it is written once per task, independently of candidates, and tested rather than proved equivalent to the original Torch module. [§4.3; Appendix A.13](../pdfs/22-lark.md)
- The specification test evaluates the elaborated Lean definition in 64-bit floating point against Torch on random inputs, refusing unknown constructs; the authors explicitly call this evidence, not proof. [Appendix A.13](../pdfs/22-lark.md)
- The parser reads candidate source as data. A fixed formatter must reproduce that source byte-for-byte from the emitted data. This translation-validation check shifts trust to the formatter; it does not prove the parser correct globally. [§4.1, Translator; Appendix A.7, Read-back check](../pdfs/22-lark.md)
- The reported PTX reader/formatter sizes are 1,405/320 lines; CUDA's are 2,313/570, plus a shared 217-line helper. A compensating formatter defect remains a trust concern acknowledged by the paper. [Appendix A.7, Read-back check](../pdfs/22-lark.md)
- Task signatures encode input/output/in-place/scratch roles, dtypes, shapes, strided layouts and scalar/dimension bindings. Preconditions enforce placement, separation, alignment and geometry; extractors read declared outputs and in-place buffers. [§4.2; Appendix A.8](../pdfs/22-lark.md)
- Scratch buffers are aligned/separated but not assumed initialized or extracted. In-place buffers are both placed as input and extracted as output, enabling a theorem that includes state updates. [Appendix A.8](../pdfs/22-lark.md)
- Generator-provided region hints are proved against footprints, while named exact constants behind rounded literals and expected launches are reported trusted/judged data. Their status differs from untrusted proof text. [Appendix A.11](../pdfs/22-lark.md)
- Grid synchronization is supported through a pinned PTX template, or the accepted CUDA/HIP `this_grid().sync()` form. Correctness under a cooperative launch remains trusted; arbitrary atomics/spin-loop barriers are not accepted. [Appendix A.7, Grid barrier](../pdfs/22-lark.md)
- A standalone certificate exports pinned model text, kernel data/parse, signature, fuels, race lemmas, both equivalence proofs, the end-to-end theorem and its axiom report. Rechecking uses public Lean/Mathlib without rerunning LARK's agents. [Appendix A.12](../pdfs/22-lark.md)

### Supported tasks and budgets

- Level 1 contains 12 KernelCraft tasks: ReLU, SiLU, GELU, Softmax, LayerNorm, RMSNorm, GEMV, GEMM, Linear, BatchMatmul, Conv2D and DepthwiseConv. All three languages are evaluated. [§5 Setup; Appendix A.2, Table 6](../pdfs/22-lark.md)
- Level 2 contains FFN, SwiGLU, RoPE, FlashAttention and GQA, evaluated on PTX. The main comparisons use GLM 5.3 and Opus 5.5. [§5 Setup; Tables 3 and 6](../pdfs/22-lark.md)
- Each task uses the largest of KernelCraft's five configurations, with FP16 tensors. Examples are GEMM `(64,256,512)`, BatchMatmul `(32,64,256,512)`, and FlashAttention `(sequence 128, hidden 256)`. These are fixed evaluated problem sizes. [Appendix A.2, Table 9](../pdfs/22-lark.md)
- The four Level-1 verification models are GLM 5.3, GLM 5.3 Flash, DeepSeek V4.1 Flash and GPT-6 Luna. Every model verifies the same pinned kernel per task/language. [§5 Setup](../pdfs/22-lark.md)
- Numerical screening and latency measurement use H100 80GB for CUDA/PTX and MI300X for HIP. Language results also involve different kernels, warp widths and devices, so they are not a pure language-only intervention. [§5 Setup; §5.1](../pdfs/22-lark.md)
- Level-1 round/spend caps are 16/$50 for DeepSeek and GPT-6 Luna, 16/$75 for GLM Flash, and 8/$100 for GLM 5.3. These are per-verification caps, not equalized budgets across models. [Appendix A.2, Table 7](../pdfs/22-lark.md)
- Level-2 GLM uses 16 rounds/$100; Opus uses 10 rounds with $100 for RoPE or $200 for the other fused tasks. Caps derive from preliminary runs and operational stopping thresholds. [Appendix A.2, Table 7](../pdfs/22-lark.md)
- Shared settings include two pipeline restarts, ten attempts per segmentation/statement/proof step, three review/diagnosis attempts, 150-second Lean-check timeout and at most 40 Lean checks per agent turn. [Appendix A.3, Table 10](../pdfs/22-lark.md)
- The normal numerical screen uses six trials and four probes. The reported configuration does not itself give an explicit tolerance value in the narrative; LARK says each run's report states its tolerance beside the theorem. Do not invent a global epsilon from these counts. [Appendix A.3, Table 10; §4.3](../pdfs/22-lark.md)
- Efficiency mechanisms include scratch Lean scripts, partial-check feedback, reporting slow declarations, delta submissions and context compaction retaining essential initial/recent messages. [Appendix A.7, Efficiency mechanisms](../pdfs/22-lark.md)

### Mutation and production evaluations

- Mutation testing applies arithmetic, address/store, barrier, register-use and loop-bound/stride mutations to ReLU, Softmax and GEMM in CUDA/PTX/HIP. Only semantics-breaking mutants are included; equivalent mutations are replaced by another candidate. [§5.2; Appendix A.4.11, Table 15 notes](../pdfs/22-lark.md)
- There are 14 mutants per language, 42 total, because ReLU has no applicable barrier mutation. GLM 5.3 verifies them with the numerical screen disabled. [§5.2, Table 4](../pdfs/22-lark.md)
- The production sample draws from vLLM, SGLang, FlashInfer, llm.c and flash-attention-minimal. Simple cases use GLM 5.3 and more fused/multiphase cases use Opus 5.5; see source inconsistency notes for the llm.c budget-table assignment. [§5.3; Table 5; Appendix A.2](../pdfs/22-lark.md)
- Common production rewrites instantiate templates, inline helpers, regularize syntax, flatten blocks and rewrite unsupported access patterns. They are recorded transformations, not a theorem that upstream source equals adapted source. [Appendix A.14](../pdfs/22-lark.md)
- Semantic departures include narrower indices, retaining only an aligned branch, pre-gathered positional data, making one fused kernel out-of-place, FP16 instead of FP32 storage, and removing int4 dequantization/split-K behavior from the AWQ-derived GEMM. [Appendix A.14, Departures](../pdfs/22-lark.md)
- `sglang_fused_add_rmsnorm` is the sole production example with no recorded semantic departure; it still receives mechanical rewrites and is verified with input and residual as in-out buffers. [Appendix A.14](../pdfs/22-lark.md)

## 4. Results and evidence

### Verification coverage and proof difficulty

- GLM 5.3 certifies 10/12 Level-1 PTX tasks, 9/12 CUDA tasks and 10/12 HIP tasks, reported as 83%, 75% and 83%. [Table 2; Appendix A.4.1, Table 11](../pdfs/22-lark.md)
- GLM 5.3 Flash certifies 7/12 PTX and 6/12 CUDA/HIP tasks; DeepSeek V4.1 Flash and GPT-6 Luna certify only the three elementwise tasks, 3/12 in each language. [Table 2; Table 11](../pdfs/22-lark.md)
- All models prove elementwise cases; GLM models additionally reach normalization; GLM 5.3 reaches matrix/convolution cases. This is the authors' “proof difficulty ladder”: induction, cut convergence and fragment-layout obligations become barriers at different capability levels. [§5.1](../pdfs/22-lark.md)
- No tested model proves DepthwiseConv. BatchMatmul fails for all on PTX/CUDA, but GLM 5.3 proves the HIP version, whose one 64-lane wavefront per block avoids a difficult multiwarp composition. [§5.1; Appendix A.4.5](../pdfs/22-lark.md)
- The hard PTX/CUDA kernels stall on composing lane-level bridges into warp/block folds across long tensor-core sequences. BatchMatmul can have 97 subphases per block phase; DepthwiseConv has 26 back-to-back collectives. [Appendix A.4.5](../pdfs/22-lark.md)
- Two failed runs involve an erroneous GPT-6 Luna segmentation caught by the first equivalence check. This is evidence for the intended rejection mechanism, not a certificate failure. [Appendix A.4.5](../pdfs/22-lark.md)
- GEMV illustrates proof-shape effects: CUDA has ten segments and two block barriers within its K-loop, versus one segment and one barrier for PTX; GLM 5.3 uses 48 versus 29 agent turns. [§5.1](../pdfs/22-lark.md)
- Opus 5.5 certifies all five Level-2 kernels. GLM 5.3 certifies only RoPE, 1/5. These are small fixed sets with different model budgets, not a general rate for all fused kernels. [Table 3; Appendix A.2](../pdfs/22-lark.md)

### Cost and ablations

- Table 2's mean successful-proof input/output token counts for GLM 5.3 are 314.53M/8.05M on PTX, 280.49M/6.18M on CUDA and 312.21M/7.15M on HIP. These averages exclude unsuccessful runs. [Table 2 caption and rows](../pdfs/22-lark.md)
- Opus's Level-2 mean is 107.34M input and 1.73M output tokens, with all five tasks successful. GLM's Table-3 total has an averaging inconsistency documented below; use RoPE's explicit 71.90M/2.70M for its one successful task. [Table 3](../pdfs/22-lark.md)
- Across 116 PTX/CUDA verifications, program length correlates with dollar cost at Spearman 0.61 but only 0.17 with proof turns; the authors interpret larger programs as costing more per turn rather than simply requiring more attempts. [Appendix A.4.4](../pdfs/22-lark.md)
- The two equivalence proofs consume 95–99% of Level-1 tokens at the reported per-model medians. Weaker models spend heavily on segmentation equivalence and often never reach final specification equivalence. [Appendix A.4.3; Table 12](../pdfs/22-lark.md)
- Static processing is reported at 36–157 seconds per run, maximum 352; Lean checks inside the loop take 18–60 minutes for small models and 6–8 minutes for Opus. These are components of verification time, not a full end-to-end wall-clock distribution. [Appendix A.4.3](../pdfs/22-lark.md)
- Total-cost-per-certificate is distinct from average successful-run cost: Appendix A.4.6 includes failed-run spending and reports 466M input tokens per certificate for GLM 5.3, versus 1,617M DeepSeek, 2,170M GLM Flash and 3,313M GPT-6 Luna over PTX/CUDA. [Appendix A.4.6](../pdfs/22-lark.md)
- Much input usage is repeated dialogue/cache reads; outputs are 0.7–2.4% of inputs. Token-efficiency ordering need not match dollars: the reported DeepSeek/GLM costs per proof are $39/$73 due to different prices. [Appendix A.4.6](../pdfs/22-lark.md)
- Failure analysis accounts for 0.7–4.7% of input tokens in the reported model aggregates and ends ten dialogues before their spending ceilings. It is an inexpensive routing component relative to proof generation. [§5.4; Appendix A.4.9, Table 13](../pdfs/22-lark.md)
- Three false “kernel bug” verdicts are refuted by the second analyst; eleven of twenty reviewed stop verdicts are disputed. Those errors were not acted on, illustrating why analyst judgments are kept outside acceptance. [§5.4; Table 13](../pdfs/22-lark.md)
- The FlashAttention flat-model ablation takes 222.5M input tokens versus 172.5M with segmentation, about 29% more overall. Comparing direct proof alone to segmented equivalence gives 219.6M versus 119.6M, an 84% difference, which is a different denominator. [Appendix A.4.8](../pdfs/22-lark.md)
- Direct proofs are cheaper on the short ReLU and GEMM examples. The authors caution that the hard-kernel ablation is one case, uses different models across tasks and compares default runs from an earlier code version. Segmentation is not established as uniformly cheaper. [Appendix A.4.8](../pdfs/22-lark.md)

### Mutation robustness and production cases

- All 42 selected mutants are rejected: 11 statically, 10 through race-freedom failures, 18 through kernel-bug diagnoses and three through premise/statement rejection. No mutant receives a certificate in the reported final runs. [§5.2; Appendix A.4.11, Table 15](../pdfs/22-lark.md)
- The final mutation results are explicitly **after corrections to specification premises and output extraction**. They are therefore not an untouched blind evaluation of the initially implemented verifier. [Appendix A.4.11, final paragraph](../pdfs/22-lark.md)
- A mutation rejection by analyst diagnosis is not necessarily a machine-checked proof of non-equivalence. The sound acceptance rule is no certificate without Lean proofs; diagnosis categories describe how search terminated. [§4.2; Appendix A.4.11 notes](../pdfs/22-lark.md)
- Nine adapted production kernels certify; `llmc_softmax_forward` is rejected for a benign but real read/write race whose value is discarded, so output comparison does not expose it. [§5.3; Table 5](../pdfs/22-lark.md)
- The race involves an unguarded clamped read of an output location concurrently written by another thread during normalization. One reported witness pairs thread 0 and thread 80 in block 0 at `out[50256]`. [Appendix A.14, The llm.c race](../pdfs/22-lark.md)
- A neutral-header rerun finds another witness in a 32-thread one-block launch admitted by the theorem, providing a check against reliance on the kernel's suggestive name. [Appendix A.14](../pdfs/22-lark.md)
- Guarding the read with the store's bounds predicate removes the race without changing the intended values, and a separate Opus run verifies the repaired version. The original rejected row and repaired certificate are distinct outcomes. [Appendix A.14](../pdfs/22-lark.md)
- Race-proof fallback is needed on five production kernels, with four successful proofs and the llm.c counterexample; generated race lemmas suffice for the evaluated core kernels discussed in Appendix A.4.7. Repeated verification of identical text can reuse stored proofs. [Appendix A.4.7](../pdfs/22-lark.md)

### Latency and retargeting evidence

- Table 14 reports same-device speedup ratios at pinned task sizes, median of seven rounds, against Torch eager, `torch.compile(max-autotune)` and the fastest available vendor/library baseline. Do not compare raw H100/PTX ratios directly with MI300X/HIP ratios as a hardware ranking. [Appendix A.4.10, Table 14 caption](../pdfs/22-lark.md)
- Reported geometric means versus eager/compile/vendor are PTX 5.50×/2.61×/1.46×; CUDA 4.20×/1.91×/1.37×; HIP 4.06×/2.14×/1.22×. Vendor denominators include only operations with an available baseline, and PTX includes Level 2 while CUDA/HIP do not. [Table 14](../pdfs/22-lark.md)
- The latency section labels its full set “verified,” yet includes PTX/CUDA BatchMatmul and DepthwiseConv, which Table 11 says no tested model certified. Treat these as reported timings of evaluation kernels, not evidence that every listed kernel received a certificate. [Table 11; Appendix A.4.10, Table 14](../pdfs/22-lark.md)
- Supported proof-friendly kernels can lose to libraries using unsupported features: the discussion notes cuBLAS batched GEMM uses wgmma/TMA, outside LARK's scope, and outperforms these candidates. [Appendix A.4.10](../pdfs/22-lark.md)
- Retargeting is substantial engineering: CUDA support on PTX reportedly takes at most 39 person-days; HIP uses 21,495 written lines, 15,333 code lines, with much translator code copied from CUDA. “Retargetable” does not mean automatic new-language support. [Appendix A.6](../pdfs/22-lark.md)
- Most machine-model Lean text is shared (reported 96% HIP, 94% CUDA, 86% PTX); language-specific parts include operations, warp width, collective layouts, parser/formatter and lowering. [Appendix A.6; Table 16](../pdfs/22-lark.md)

## 5. Authors' conclusions

- Agent-generated formal proofs are feasible for a substantial portion of the tested supported GPU kernels, including fused tasks with a sufficiently capable model. [§5.5; §7](../pdfs/22-lark.md)
- The authors argue proof shape is more predictive of difficulty than target language; low-level PTX can be easier than CUDA when its phase/loop structure yields simpler obligations. This conclusion is based on their pinned kernels. [§5.1; §5.5](../pdfs/22-lark.md)
- A fixed statement, explicit trusted assumptions and Lean-checked proof obligations prevent a generator or proof agent from persuading the system to accept an unsupported theorem. The conclusion is conditional on the trusted model and implementation boundary. [§4.3; §7](../pdfs/22-lark.md)
- Verified segmentation is presented as a useful bridge between operation-level semantics and tensor-level proofs, with the reported hard-case ablation supporting its cost benefit in that example. [§4.2; §5.4](../pdfs/22-lark.md)
- The llm.c example demonstrates that race-freedom obligations can reveal a discarded-value race invisible to numerical input-output tests. It does not establish that all production bugs will be detected automatically. [§5.3; §7](../pdfs/22-lark.md)
- The authors characterize LARK as a step toward verifiable generation by making trust reusable across kernels and certificates independently recheckable. [§7; Appendix A.12](../pdfs/22-lark.md)

## 6. Limitations and open questions

### Explicit guarantee boundaries

- **No floating-point theorem:** no error bound between actual rounded hardware output and exact specification is proved. Overflow, rounding, quantization and approximate instruction accuracy remain outside the certificate. [§4.3; Limitations](../pdfs/22-lark.md)
- **No full allocation-safety theorem:** unbounded model memory means host-buffer out-of-bounds writes are not generally excluded. Do not summarize the result as complete CUDA memory safety. [§3; §4.1](../pdfs/22-lark.md)
- **Concurrency scope:** within-phase communication, benign races and early-halting warps can be refused; sufficient race-free structure sacrifices completeness. [Limitations; Appendix A.10](../pdfs/22-lark.md)
- Unsupported forms include asynchronous `cp.async`, mbarrier, TMA, wgmma, tcgen05, cluster barriers, communication atomics/fences, partial masks, BF16, FP64, 8-bit types, integer wraparound and certain bit/convert operations. [Appendix A.9](../pdfs/22-lark.md)
- Layout restrictions include sub-byte/bit-packed elements, index-buffer indirection and swizzled/blocked global layouts. Blocks must be one-dimensional and grids have at most two dimensions in the stated unsupported list. [Appendix A.8–9](../pdfs/22-lark.md)
- Trusted items include exact-arithmetic idealization, concurrency-reduction metatheory, measured operation definitions, grid-barrier template, named constants, Torch formalization, premise/extractor definitions, formatter and Lean/toolchain/standard axioms. [Appendix A.13](../pdfs/22-lark.md)
- Hardware/specification sweeps are randomized tests rather than proofs of correspondence. They narrow practical uncertainty but do not erase the trusted boundary. [Appendix A.5; A.13](../pdfs/22-lark.md)
- Production certificates concern rewritten kernels with declared departures, including the AWQ case's removal of dequantization. They cannot be transferred without proof to the original quantized production implementation. [Appendix A.14](../pdfs/22-lark.md)

### Analyst caveats and source inconsistencies

- **Coverage caveat:** the core suite is 17 chosen fixed-configuration tasks, with a pinned kernel per language; success rates should not be generalized to arbitrary generated kernels or dynamically shaped deployments. [Basis: §5 Setup; Table 9](../pdfs/22-lark.md)
- **Budget caveat:** models receive different round and dollar limits; observed pass/cost differences combine capability with experimental allocation. The abstract's feasibility conclusion is stronger evidence than a universal ranking. [Basis: Table 7](../pdfs/22-lark.md)
- **Table-3 averaging inconsistency:** its caption says failed runs are excluded, but GLM's total 373.90M input/9.29M output is the mean over all five entries, not its only successful RoPE run at 71.90M/2.70M. Appendix A.4.6 describes about 374M per kernel including failures, supporting that interpretation. [Table 3; Appendix A.4.6](../pdfs/22-lark.md)
- **Library outcome inconsistency:** Appendix A.4.2 lists “Library: 10 / 0 / 0,” whereas Table 5 and §5.3 report nine certificates and one race rejection. Use the explicit per-kernel table for certificate counts; the census is not reconciled in the source. [Appendix A.4.2; Table 5](../pdfs/22-lark.md)
- **llm.c accounting inconsistency:** Table 5 assigns GLM and zero tokens to the rejected row; Table 7 groups it under Opus, while Appendix A.14 describes a GLM fallback costing 9.8M input tokens and Table 12 lists 10.30M fallback tokens. Preserve the rejection and workflow, but do not claim a single reconciled cost/model-budget record. [Tables 5, 7, 12; Appendix A.14](../pdfs/22-lark.md)
- **Address-scope tension:** Appendix A.9 says non-polynomial addresses are refused, while A.10/A.14 describe sending a non-polynomial clamped address to a proof fallback. Supported parsing versus automatic footprint generation needs implementation-level clarification before deployment. [Appendix A.9–10; A.14](../pdfs/22-lark.md)
- **Latency labeling caveat:** Table 14's “verified” label conflicts with unproved entries in Table 11; performance data and proof outcome must be joined per kernel rather than inferred from the caption. [Table 11; Table 14](../pdfs/22-lark.md)
- **Mutation-study caveat:** reporting results after correcting premises/extraction establishes the final evaluated state, not a guarantee that statement-generation bugs cannot recur. [Appendix A.4.11](../pdfs/22-lark.md)

## 7. Relevance to this project — interpretation

- **Proposal:** record a correctness result as several fields: numerical tests, race checks, exact-arithmetic equivalence, task/interface assumptions and unsupported constructs. A single `correct=true` would erase LARK's important guarantee boundaries. [Basis: §3; §4.3; Limitations](../pdfs/22-lark.md)
- **Proposal:** pin task signatures, specification hashes, model semantics and toolchain versions before candidate optimization; disallow agents from silently changing the acceptance statement or adding candidate-specific assumptions. [Basis: §4.2–3; Appendix A.8](../pdfs/22-lark.md)
- **Proposal:** preserve proof certificates and assumption reports with candidate artifacts, enabling later independent checking and precise retrieval. This is a proposed project design, not an integration performed here. [Basis: Appendix A.12–13](../pdfs/22-lark.md)
- **Proposal:** retain numerical adversarial testing even after a LARK-style proof, especially when an optimization changes precision, uses tensor cores or invokes approximate math. [Basis: §4.3; Appendix A.9](../pdfs/22-lark.md)
- **Proposal:** classify unsuccessful verification as unsupported, budget-exhausted, proof-stuck, statement mismatch or counterexample; do not count all failures as incorrect kernels. [Basis: §3; Appendix A.4.1](../pdfs/22-lark.md)
- **Proposal:** allow diagnosis agents to route search and recommend changes, while keeping acceptance tied to executable or formal checks. LARK's second-opinion results warn against treating a confident diagnosis as ground truth. [Basis: §5.4; Table 13](../pdfs/22-lark.md)
- **Proposal:** experiment with generated warp-level bridge lemmas or proof-aware segmentation for recurrent failures, and compare added proof cost against successful certificate rate. [Basis: Appendix A.4.5; A.4.8](../pdfs/22-lark.md)
- **Proposal:** if applying verification to a library kernel, preserve an explicit upstream-to-adapted change log and state which semantic departures the theorem covers. [Basis: Appendix A.14](../pdfs/22-lark.md)

## 8. Retrieval notes

- Search aliases: LARK, Lean-Assisted Reasoning for Kernels, Lean 4 GPU verification, RF/SEG/EQ, flat model, segment-level model, exact arithmetic, trust boundary, read-back formatter, KernelCraft, phase cuts, llm.c benign race.
- Questions answered: What does LARK prove and not prove? Why are floating-point tests still required? What remains trusted? Why is race freedom checked before equivalence? What does a certificate contain? How were production kernels modified? Why can valid kernels fail verification?
- Related analyses: [bit-accurate MMA](09-bit-accurate-gpu-mma.md) for the hardware floating-point semantics deliberately idealized here; [robust agentic CUDA](08-robust-agentic-cuda.md) for empirical correctness hardening; [ReAct](11-react.md) for general agent scaffolding. Links are for navigation, not a controlled head-to-head comparison.

## 9. Source coverage and verification notes

- Read the complete main text, all substantive Appendix A.1–A.14 material, per-task and stage-cost tables, mutation breakdown, latency table, retargeting discussion, premise schemas, refusal list and production rewrite/departure list.
- Numerical citations use table/section numbering in the linked local Markdown/HTML conversion. No PDF physical-page number is inferred from manuscript section numbering.
- Checked Table 3 arithmetic against its row values and Appendix A.4.6, and cross-checked production outcomes/model assignments/costs across Tables 5, 7 and 12 plus A.14. Unresolved discrepancies are documented above instead of silently corrected.
- The source includes minor narrative imprecision about trust and unsupported addresses; the most specific formal-scope and appendix statements are retained. Particularly, race-reduction metatheory is trusted, specification formalization is tested, and out-of-allocation safety is not proved.
- No Lean project was built, no certificate independently rechecked, no LARK implementation audited and no GPU measurements rerun during this analysis. Formal guarantees here are reported claims with their explicit assumptions.

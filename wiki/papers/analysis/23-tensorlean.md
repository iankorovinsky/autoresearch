---
paper_id: "23"
title: "TensorLean: Formal Verification of Production GPU Kernels"
source_version: "web snapshot 2026-11-21"
source_markdown: "../pdfs/23-tensorlean.md"
source_pdf: null
source_url: "https://tensorlean.ai/"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["gpu-kernels", "formal-verification", "lean", "ptx", "floating-point", "numerical-correctness", "race-freedom", "trust-boundary", "correctness", "flashattention", "loop-invariants"]
summary: "TensorLean is announced as a Lean/SMT framework proving that compiled PTX meets a tensor-level real-number specification within a stated tolerance on every input and schedule; all results are unreplicated author claims with no released artifact or paper at snapshot."
---

# TensorLean: Formal Verification of Production GPU Kernels

## 1. Paper identity and scope

- The source is a web article announcing a system, preserved as a snapshot dated 2026-11-21. It is not a paper: the authors write that they "will soon release" a paper and an open-source repository with the PTX semantics, specifications and proofs. [Introduction](../pdfs/23-tensorlean.md)
- The preserved HTML byline lists Wei Qiang and Ganxiang Yang (equal contribution), Yi Rong, Xupeng Li and Ronghui Gu, affiliated with Columbia University and CertiK. The byline is missing from the Markdown conversion, and the page shows no publication date. [byline](../pdfs/sources/23-tensorlean.html)
- Question addressed: whether functional correctness of production GPU kernels can be machine-proved on their compiled PTX, for every accepted input and every thread schedule, at an affordable specification and proof cost. [Opening paragraph; Introduction](../pdfs/23-tensorlean.md)
- Claimed contributions: a tensor-level specification format that separates mathematics from memory layout, and automatic PTX-level verification with Lean and SMT solvers covering data-race freedom, termination, functional correctness and a floating-point tolerance. [Introduction; PTX-Level Verification](../pdfs/23-tensorlean.md)
- Evidence level: every count below is self-reported. No artifact, proof, specification or paper was available at snapshot time, so nothing can be rechecked from this source. [Introduction](../pdfs/23-tensorlean.md)
- Not studied: kernel generation, kernel performance or LLM proof agents; agents appear only in a forward-looking sentence. Kernels that sample at random and kernels synchronized through atomic flags are declared out of scope. [Introduction; Limitations, 'Verification Scope'](../pdfs/23-tensorlean.md)

## 2. Concepts and terminology

### Correctness and specification

- A kernel is correct if, on every accepted input and in every execution, it writes the same values as the operation it implements and changes nothing else in GPU memory. "The same" means equal within a bound called the tolerance. The definition is extensional: it constrains memory before and after execution, not warp specialization or synchronization. [Tensor-Level Specification](../pdfs/23-tensorlean.md)
- Tensors are classified as `input` (read only), `output` (written without reading old values) and `state` (read and overwritten in place, such as a KV cache). [Tensor-Level Specification](../pdfs/23-tensorlean.md)
- A specification has five parts: `spec_math` (a function from `state` and `input` to `output` tensors of real numbers), `spec_mut` (the in-place update giving `state'`), `spec_layout` (tensor indices to GPU memory addresses), `spec_tol` (a tolerance ε for each `output` and `state'` tensor) and `requires` (a precondition on `state` and `input`). [Tensor-Level Specification; Figure 3](../pdfs/23-tensorlean.md)
- Specifications are written in Lean using a tensor-operator library built on Mathlib, with PyTorch-like operator names. [Tensor-Level Specification, 'Writing Specifications in Lean'](../pdfs/23-tensorlean.md)
- Lowering: the tensor-level statement is turned automatically into a PTX-memory statement, e.g. `Q[i, e]` becomes `fp16(mem[q_ptr + 2*(i*D + e)])` under the layout of `Q`. [Tensor-Level Specification, 'Decoupling Mathematics from Memory'; Figure 2](../pdfs/23-tensorlean.md)
- "Kernel instance" denotes a compiled instance of a kernel whose PTX is verified. The article does not define it formally or say how instances of one kernel differ. [FlashAttention and Beyond](../pdfs/23-tensorlean.md)

### Concurrency reduction

- The happens-before graph has a node per execution of an operation by one thread and edges for orderings the PTX ISA guarantees. It is built symbolically over every CTA, thread and loop iteration of every launch; the authors say its soundness is proved in Lean. [PTX-Level Verification, 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)
- Data-race freedom means any two accesses to one address, at least one a write, are ordered by the graph. Termination means every loop is bounded and every wait receives its arrivals without a cycle; it is required because a hanging kernel would satisfy any specification vacuously. [PTX-Level Verification, 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)
- Given both properties, final memory is initial memory updated by the last writes. Thread groups with one role (producer warp, consumer warps) are each run symbolically once and exchange writes only at synchronizations. [PTX-Level Verification, 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)

### Loop invariants and assertions

- Configurations are symbolic parameters, so loops have symbolic bounds and need an invariant per carried value. The article contrasts this with equivalence checkers that unroll loops for one concrete configuration at a time. [PTX-Level Verification, 'Inferring Loop Invariants over Semirings'](../pdfs/23-tensorlean.md)
- Invariant synthesis relies on three stated loop properties: tiled iteration, accumulative updates and data-independent control flow. Each carried value then follows \(f_{k+1} = a_k \otimes f_k \oplus b_k\) over a semiring such as \((+, \times)\) or \((\max, +)\); closed forms are read off one symbolic run of the loop body and proved by induction with a single SMT query, over real numbers. [PTX-Level Verification, 'Inferring Loop Invariants over Semirings'](../pdfs/23-tensorlean.md)
- `tensorlean_assert` statements written in kernel source are proof hints: each is proved once from the precondition and PTX semantics, then assumed by later goals via the cut rule. For loops outside the three properties, such as persistent-kernel schedulers, developers state the invariant as assertions proved by induction. [PTX-Level Verification, 'Assertions as Proof Hints'; Figure 5](../pdfs/23-tensorlean.md)

### Tolerance

- `spec_math` is over real numbers so that one specification covers kernels in several precisions; FP32, BF16 and FP8 are named. [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- The proof goal is \(\lvert \mathrm{kernel}_{\mathrm{Float}} - \texttt{spec\_math} \rvert \le \varepsilon\), with ε described as "the same" bound a reference-based test uses. [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- The proof goes through \(\mathrm{kernel}_{\mathrm{Real}}\), the kernel evaluated under a real-number PTX semantics: algebraic error \(\lvert \mathrm{kernel}_{\mathrm{Real}} - \texttt{spec\_math} \rvert = 0\) and round-off error \(\lvert \mathrm{kernel}_{\mathrm{Float}} - \mathrm{kernel}_{\mathrm{Real}} \rvert \le \varepsilon\), combined by the triangle inequality. [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- The round-off bound is computed by propagating each operation's rounding through the computation tree in linear time, instead of a bit-vector search over floating-point operations. [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- What the claim means: a proved worst-case bound between the floating-point values left in memory and the exact real-valued specification, for all inputs satisfying `requires` and all schedules, relative to the trusted PTX semantics and tensor-core rounding model. [Tensor-Level Specification; Limitations, 'Trusted Computing Base'](../pdfs/23-tensorlean.md)
- What it does not mean: bit-exact agreement with the specification or with a PyTorch reference. A reference appears only as the test-time stand-in for exact values. [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)

## 3. Methodology and experimental setup

- Inputs to verification are a kernel's PTX and its specification; proof goals are then constructed automatically and discharged with Lean and SMT solvers. [PTX-Level Verification](../pdfs/23-tensorlean.md)
- In the workflow figure's HTML, source code, specification and PTX are marked as from the user; the happens-before graph, post-execution memory and lowered specification as auto-generated; the proof as proven; and the PTX semantics and proof checkers as trusted. The Markdown preserves the legend but not this assignment. [workflow figure, element classes and `aria-label`](../pdfs/sources/23-tensorlean.html)
- Languages and hardware features: FlashAttention-2 (CuTe; `mma` with `cp.async` on Ampere), FlashAttention-3 (CuTe; TMA, WGMMA and mbarriers on Hopper), FlashAttention-4 (CuTe DSL; tensor memory and `tcgen05` on Blackwell). [FlashAttention and Beyond, 'Verification: In Minutes'](../pdfs/23-tensorlean.md)
- Verified set: all 37 kernels of FlashAttention-2 to -4, each on "at least one compiled instance" (102 instances), plus 47 instances from DeepGEMM and Liger Kernel. The number of distinct non-FlashAttention kernels is not stated. [FlashAttention and Beyond](../pdfs/23-tensorlean.md)
- TensorLean was also "applied to" Triton's kernel library and TileOPs, but the 47 additional instances are attributed only to DeepGEMM and Liger Kernel. [FlashAttention and Beyond, 'Finding and Fixing Bugs'](../pdfs/23-tensorlean.md)
- Specification-size study: 72 open-source kernels from 9 operations were formalized. This set differs from the verified set; Figure 1's embedded data names variants (FlashAttention-1, CUTLASS, ThunderKittens, FlashInfer, FlashMLA, vLLM) that the text does not report as verified. [Tensor-Level Specification, 'On Production Kernels'](../pdfs/23-tensorlean.md) [Figure 1, embedded `data-variants` attribute](../pdfs/sources/23-tensorlean.html)
- Timing was measured on "a 16-core workstation"; CPU model, memory, solver versions and timeouts are not given. [FlashAttention and Beyond, 'Verification: In Minutes'](../pdfs/23-tensorlean.md)
- Code excerpts link to FlashAttention commit `68bd095c…` and DeepGEMM commit `1fc9c669…`; the article does not state that all verified instances come from these commits. [PTX-Level Verification, 'Assertions as Proof Hints'](../pdfs/23-tensorlean.md)
- Human inputs: one specification per kernel plus 298 hand-written assertions for FlashAttention-2 to -4 and DeepGEMM. Person-time is not reported. [PTX-Level Verification, 'Assertions as Proof Hints'](../pdfs/23-tensorlean.md)
- Not described anywhere: how instances were chosen, how lines were counted, the ε values, the tensor-core rounding model, or any comparison run against another verifier. [Analyst observation; whole article](../pdfs/23-tensorlean.md)

## 4. Results and evidence

All items are author-reported and unreplicated.

### Verification outcomes

- 149 kernel instances were "verified" on their PTX: 83 are reported proved correct and "the other 66 contain bugs we found". [Introduction](../pdfs/23-tensorlean.md)
- FlashAttention-2 to -4: 37 kernels and 102 instances. 56 instances are proved to satisfy all four properties; in each of the other 46 "a proof goal fails on a bug in the kernel's source". [FlashAttention and Beyond, 'Verification: In Minutes'](../pdfs/23-tensorlean.md)
- Derived: of the 47 DeepGEMM and Liger Kernel instances, 27 are proved correct (83 − 56) and 20 are not (66 − 46). Derived rates: 83/149 = 55.7% overall and 56/102 = 54.9% for FlashAttention. [Introduction; FlashAttention and Beyond](../pdfs/23-tensorlean.md)
- 38 distinct bugs are reported, 23 of them "previously unknown" bugs in FlashAttention-2 to -4 that the authors say they reported to the developers. [FlashAttention and Beyond, 'Finding and Fixing Bugs'](../pdfs/23-tensorlean.md)
- Five counts must stay separate: 72 kernels specified; 37 FlashAttention kernels verified; 149 instances verified; 83 instances proved correct against 66 with a failing goal; 38 distinct bugs. No mapping from the 66 failing instances to the 38 bugs is given. [Tensor-Level Specification; FlashAttention and Beyond](../pdfs/23-tensorlean.md)
- Figure 8's embedded text labels give per-category counts: Incorrect Algorithm 3, Numerical Bug 13, Uninitialized Memory 5, Memory Access Error 19, Data Race 7, Deadlock 1, Undefined Behavior 5, Miscompilation 2. A bug in several categories is counted in each; derived, the labels sum to 55 assignments over 38 bugs. [Figure 8](../pdfs/23-tensorlean.md)
- The preserved HTML embeds 38 bug records behind Figure 8's hover lists. Counting records by library (derived): FlashAttention-2 9, FlashAttention-3 10, FlashAttention-4 4, Liger Kernel 7, DeepGEMM 2, Triton 1, TileLang 2, TileOPs 3. Their category tags reproduce the eight figure labels. [Figure 8, embedded `data-bugs` attribute](../pdfs/sources/23-tensorlean.html)
- Derived: 6 of the 38 records (Triton, TileLang, TileOPs) belong to libraries with no instances in the 149 count. Many records describe 32-bit offset or counter overflow near \(2^{31}\) elements, empty or degenerate page tables, and unusual argument combinations. [Figure 8, embedded `data-bugs` attribute](../pdfs/sources/23-tensorlean.html)
- 19 of the 23 FlashAttention bugs are said to appear only in combinations of shapes and features that developers overlook. [PTX-Level Verification, 'Inferring Loop Invariants over Semirings'](../pdfs/23-tensorlean.md)
- A FlashAttention-4 data race (issue #3195) "never showed up in 17,760 runs of its tests". [PTX-Level Verification, 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)

### Verification cost

- Median verification time is "about 10 minutes" per instance on a 16-core workstation. The largest case, FlashAttention-3's Hopper forward kernel with 31,402 proof goals, takes 50 minutes. [FlashAttention and Beyond, 'Verification: In Minutes'](../pdfs/23-tensorlean.md)
- Figure 7's embedded text labels give per-instance totals for FlashAttention-3 on Hopper: PrepareVarlen 4 min, Forward 50 min, FwdCombine 20 min, BwdPreprocess 4 min, Backward 27 min, ConvertdQ 11 min. The per-property split exists only as bar geometry and is not reported here. [Figure 7](../pdfs/23-tensorlean.md)
- Three assertions are said to suffice for every output of FlashAttention-4's Hopper backward kernel. [PTX-Level Verification, 'Assertions as Proof Hints'](../pdfs/23-tensorlean.md)

### Specification size

- 72 kernels from 9 operations: 151,285 source lines against 10,278 specification lines, described as "nearly 15:1" (derived: 14.7:1). [Tensor-Level Specification, 'On Production Kernels'](../pdfs/23-tensorlean.md)
- 12 attention (forward) variants: 35,852 source lines against 2,291 specification lines. Figure 1's embedded text also shows "156 lines", which the embedded data identifies as the shared specification library, included in the 2,291. [Tensor-Level Specification, 'Kernel Specification at Scale'; Figure 1](../pdfs/23-tensorlean.md) [Figure 1, embedded `data-variants` attribute](../pdfs/sources/23-tensorlean.html)
- Figure 4's embedded text labels give code-to-spec ratios by operation: Attention Forward 16:1, Attention Backward 15:1, MLA 28:1, GEMM 6:1, RMSNorm 16:1, Top-k 23:1, LayerNorm 15:1, RoPE 13:1, SwiGLU 10:1. [Figure 4](../pdfs/23-tensorlean.md)
- The data behind Figure 4 gives per-operation variant counts of 12, 10, 6, 16, 9, 6, 4, 4 and 5. Derived check: they sum to 72, and the per-operation line counts sum to 151,285 and 10,278. [Figure 4, embedded `data-operations` attribute](../pdfs/sources/23-tensorlean.html)
- 37 FlashAttention kernels: 55,910 source lines against 6,242 specification lines (derived: 9.0:1, below the 15:1 headline). [FlashAttention and Beyond, 'Specification: From Math to Machine'](../pdfs/23-tensorlean.md)
- Figure 6's embedded text labels: FA-2 3:1, FA-3 5:1, FA-4 14:1; within FlashAttention-3, Prepare Varlen 3:1 and Forward, Fwd Combine and Backward 6:1 each. The embedded data gives variant counts of 6, 8 and 23 for FA-2, FA-3 and FA-4 (derived sum 37). [Figure 6](../pdfs/23-tensorlean.md) [Figure 6, embedded `data-operations` attribute](../pdfs/sources/23-tensorlean.html)

## 5. Authors' conclusions

- TensorLean is "the first framework to formally verify the correctness of production GPU kernels". [Opening paragraph; Conclusion](../pdfs/23-tensorlean.md)
- Tensor-level specifications take "no more effort to write, or to audit, than a PyTorch reference", and verification is automatic given specification and PTX. [Tensor-Level Specification, 'Writing Specifications in Lean'; Conclusion](../pdfs/23-tensorlean.md)
- Verifying PTX captures all eight bug categories, including miscompilations by NVCC, Triton or the CuTe DSL. [FlashAttention and Beyond, 'Finding and Fixing Bugs'](../pdfs/23-tensorlean.md)
- Without a tolerance proof, verification "would miss every rounding error that can occur on real execution". [PTX-Level Verification, 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- The authors expect formal verification to become an essential development step and conclude that "TensorLean leaves no step unproven". [Opening paragraph; Conclusion](../pdfs/23-tensorlean.md)

## 6. Limitations and open questions

### Stated by the source

- Trusted: NVIDIA's PTX ISA and ptxas, with the assumption that TensorLean's PTX semantics is consistent with them; and TensorLean's model of tensor-core rounding, "which the PTX ISA does not specify", used to bound matrix-multiplication round-off. [Limitations, 'Trusted Computing Base'](../pdfs/23-tensorlean.md)
- Also trusted: each kernel's specification, a 1.8K-line PTX parser, and the results of Lean's kernel and the cvc5 and Z3 SMT solvers. [Limitations, 'Trusted Computing Base'](../pdfs/23-tensorlean.md)
- Scope: specifications are deterministic functions of inputs, so random-sampling kernels (dropout, top-p) cannot be expressed. Happens-before order is derived from barriers, mbarriers and asynchronous copies but not from acquire and release operations, so atomic-flag synchronization (NCCL, DeepEP) is out of scope. [Limitations, 'Verification Scope'](../pdfs/23-tensorlean.md)
- Incompleteness: a correct kernel can fail verification because of missing input constraints, overly conservative round-off bounds or SMT timeouts. [Limitations, 'Incompleteness'](../pdfs/23-tensorlean.md)

### Analyst caveats

- **Unreplicated:** no proof, specification, tool or paper is available, so soundness, the counts and the bug attributions rest on the announcement alone. [Basis: Introduction](../pdfs/23-tensorlean.md)
- **Tolerance underspecified:** no ε value is given for any kernel, nor whether ε is absolute or relative, depends on shapes, or needs input-range preconditions. A proved bound is only as meaningful as the chosen ε and `requires`. [Basis: Tensor-Level Specification; 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- **Real-number semantics undefined:** the article does not say how approximate instructions (the workflow shows `ex2.approx`) are treated in the real-number PTX semantics. [Basis: PTX-Level Verification, workflow figure](../pdfs/23-tensorlean.md)
- **SMT in the trust base:** goals discharged by Z3 or cvc5 are trusted results; the article does not say solver proofs are reconstructed in Lean. [Basis: Limitations, 'Trusted Computing Base'](../pdfs/23-tensorlean.md)
- **Failing goal versus bug:** all 66 non-proved instances are attributed to bugs and none to incompleteness, although incompleteness is a listed limitation. No count of developer-confirmed or fixed bugs is given. [Basis: Introduction; FlashAttention and Beyond; Limitations](../pdfs/23-tensorlean.md)
- **Instance coverage:** each FlashAttention kernel is verified on "at least one" compiled instance, so "every configuration" covers symbolic launch parameters of verified instances, not necessarily every compiled instantiation. [Basis: FlashAttention and Beyond](../pdfs/23-tensorlean.md)
- **Priority claim:** "first" is an author claim; the article itself cites two 2026 verification tools for GPU kernels and distinguishes them by concrete-configuration checking. [Basis: 'Inferring Loop Invariants over Semirings'; References 11–12](../pdfs/23-tensorlean.md)
- **Wording, "verified":** "verified 149 kernel instances" includes 66 with a failing proof goal; only 83 are proved correct. [Basis: Introduction](../pdfs/23-tensorlean.md)
- **Source inconsistency (wording):** the workflow figure labels the source code "Unmodified", while assertions are written "in the kernel's source code". Whether asserted source compiles to the same PTX is not stated. [PTX-Level Verification, workflow figure; 'Assertions as Proof Hints'](../pdfs/23-tensorlean.md)
- **Source inconsistency (wording):** "leaves no step unproven" sits beside an explicit list of trusted, unproven components. [Conclusion; Limitations](../pdfs/23-tensorlean.md)
- **Source inconsistency (minor):** the opening lists instances from FlashAttention, DeepSeek, Liger Kernel "etc.", while the body attributes the 47 non-FlashAttention instances to DeepGEMM and Liger Kernel only. Figure 8's legend names TileLang, which the prose list of libraries omits. [Opening paragraph; FlashAttention and Beyond; Figure 8](../pdfs/23-tensorlean.md)
- **Unexplained counting difference (embedded data):** Figure 6's four FlashAttention-3 by-kernel entries total 13,173 source lines over 6 variants, more than its FlashAttention-3 aggregate of 10,065 lines over 8 variants. Figure 1 lists FlashAttention-3 at 5,297 source lines and Figure 6 lists FlashAttention-3 Forward at 6,167. Line-counting rules are not given. [Figures 1 and 6, embedded data attributes](../pdfs/sources/23-tensorlean.html)
- The headline counts are mutually consistent (102 + 47 = 149; 83 + 66 = 149; 56 + 46 = 102). [Derived; Introduction; FlashAttention and Beyond](../pdfs/23-tensorlean.md)

## 7. Relevance to this project — interpretation

- **Proposal:** structure the Stage 1 Lean spec as separate parts for mathematics, in-place updates, layout, tolerance and preconditions, so the mathematical part can be audited against the natural-language spec independently of pointers and strides. [Basis: Tensor-Level Specification](../pdfs/23-tensorlean.md)
- **Proposal:** have Stage 1 clarification questions cover what this format forces into the open: accepted input domain, tensors mutated in place, whether other memory must be untouched, and a tolerance per output. [Basis: Tensor-Level Specification; Figure 3](../pdfs/23-tensorlean.md)
- **Proposal:** reuse the spec's per-output tolerance in Stage 3 and Stage 4 fuzzing, recording that fuzzing compares against a floating-point reference while a proof would bound distance to the real-valued specification. [Basis: 'Correctness under Tolerance'](../pdfs/23-tensorlean.md)
- **Proposal:** state which theorem the Stage 4 Lean proof establishes: exact-real algebraic equivalence only, or also a round-off bound, and whether termination and "no other memory changed" are included. If only the algebraic part is achievable, say so and keep the larger fuzzing check as the floating-point evidence. [Basis: 'Correctness under Tolerance'; 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)
- **Proposal:** deliver the Stage 4 proof with an explicit trusted-base list (machine semantics, rounding model, specification, parser, checkers) and the verified artifact level: source or compiled PTX, with compiler version. [Basis: Limitations, 'Trusted Computing Base'; 'Finding and Fixing Bugs'](../pdfs/23-tensorlean.md)
- **Proposal:** bias fuzzing toward shape and feature combinations, degenerate sizes and index ranges near 32-bit limits, where the reported bugs concentrate; the last may need dedicated large-allocation harness support. [Basis: 'Inferring Loop Invariants over Semirings'](../pdfs/23-tensorlean.md) [Figure 8, embedded `data-bugs` attribute](../pdfs/sources/23-tensorlean.html)
- **Proposal:** do not treat repeated passing runs as race evidence for kernels with barriers or asynchronous copies; flag such kernels for a separate race argument. [Basis: 'Reducing Concurrency to Sequential Execution'](../pdfs/23-tensorlean.md)
- **Proposal:** classify a failed proof attempt as bug, missing precondition, loose bound or timeout rather than as an incorrect kernel. [Basis: Limitations, 'Incompleteness'](../pdfs/23-tensorlean.md)
- **Proposal:** do not plan on TensorLean as a dependency until its release exists; re-check availability and the forthcoming paper first. [Basis: Introduction](../pdfs/23-tensorlean.md)

## 8. Retrieval notes

- Search aliases: TensorLean, tensorlean.ai, "From Math to Machine, in Minutes", `spec_math`, `spec_layout`, `spec_tol`, `tensorlean_assert`, happens-before graph, semiring loop invariants, correctness under tolerance, kernel_Real, FlashAttention verification, PTX-level proof.
- Questions answered: What does TensorLean claim to prove, and against what? What is trusted? What does "within floating-point tolerance" mean? How many kernels, instances, proofs and bugs are reported? What is out of scope? How large are the specifications?
- Related analyses: [LARK](22-lark.md) for Lean equivalence proofs under exact arithmetic with floating point outside the theorem; [bit-accurate MMA](09-bit-accurate-gpu-mma.md) for tensor-core rounding models, the component this source trusts; [PTXBench](04-ptxbench.md) for PTX as a target level. Links are navigation aids, not controlled comparisons: the works use different kernels, artifacts and evidence levels.

## 9. Source coverage and verification notes

- Read the complete Markdown source: opening paragraph, Introduction, Tensor-Level Specification, PTX-Level Verification, FlashAttention and Beyond, Limitations, Conclusion and the 23 references. There are no appendices.
- The nine inline SVG figures appear in the Markdown only as labels and embedded text. Values cited to "Figure N" with the Markdown link are those text labels; nothing was inferred from bar heights, colors or positions. Figure 7's per-property time split was not recoverable.
- The preserved HTML was inspected for content missing from the Markdown. Values cited with the HTML link come from explicit text there: the byline, the workflow figure's element classes and `aria-label`, and JSON `data-*` attributes on Figures 1, 4, 6, 7 and 8 (line counts, variant counts, kernel names, 38 bug records). Per-library bug counts were obtained by counting those records.
- Arithmetic checks performed: instance totals, per-operation sums against the 72-kernel totals, FlashAttention sums against the 37-kernel totals, and category tags against Figure 8's labels. Differences found are listed in section 6.
- The HTML contains commented-out passages that are not rendered; they were not used. Its visible heading reads "From Math to Machine, in Minutes: Formal Verification of Production GPU Kernels", while its page title matches this note's title.
- Not read or verified: the opening video, the linked GitHub issues and repositories, the cited references, and the interactive states of Figure 5. No proofs were checked, no tool was run and no experiments were reproduced.

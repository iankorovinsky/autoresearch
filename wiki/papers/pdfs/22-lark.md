---
title: "LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels"
source_url: "https://arxiv.org/abs/2611.02028"
full_text_url: "https://arxiv.org/html/2611.02028v1"
version: "2611.02028v1"
authors: ["Zhang, Zixi", "Nie, Jiayi", "Wu, Haoran", "Yan, Yujie", "Herklotz, Yann", "Antonova, Rika", "Mullins, Robert", "Zhao, Yiren", "Constantinides, George A.", "Cheng, Jianyi"]
retrieved_date: "2026-11-20"
conversion: "HTML to Markdown with TeX math"
pdf: "22-lark.pdf"
original_html: "sources/22-lark.html"
---

# LARK: Lean-Assisted Reasoning for LLM-Generated GPU Kernels

LARK: Lean-Assisted Reasoning
  
for LLM-Generated GPU Kernels

 

Zixi Zhang 1
Jiayi Nie 2
Haoran Wu 23
Yujie Yan 4
Yann Herklotz 5
Rika Antonova 2
Robert Mullins 2
Yiren Zhao 1
George A. Constantinides 1
Jianyi Cheng 4

††footnotetext: 1Department of Electrical and Electronic Engineering, Imperial College London, London, United Kingdom
2Department of Computer Science and Technology, University of Cambridge, Cambridge, United Kingdom
3Quettaflop AI
4School of Informatics, University of Edinburgh, Edinburgh, United Kingdom
5Computer Science Department, EPFL, Lausanne, Switzerland.
Correspondence to: George A. Constantinides <g.constantinides@imperial.ac.uk>, Jianyi Cheng <jianyi.cheng@ed.ac.uk>.
  
Preprint.

###### Abstract

Efficient GPU kernels are typically low-level and hardware-specific, and LLM agents can now generate them faster than experts can verify.
Existing approaches rely on testing, which provides evidence but not a guarantee of correctness.
We introduce LARK, an agentic framework that automatically and formally verifies GPU kernels against their PyTorch reference implementations.
LARK proves equivalence in functional and memory behavior under exact arithmetic and establishes a new *trust boundary*: human effort is only needed to verify operator-level Lean semantics ahead of kernel generation time, instead of checking every single line of generated code.
LARK formalizes the GPU machine model and the semantics of each operation in Lean 4 once per target language, and the Lean 4 semantics of any incoming kernel is derived from these pre-built definitions.
LLM agents then automate the proof in two steps.
First, an agent decomposes the whole kernel semantics into smaller segments and Lean 4 verifies this segmentation, which addresses the intractability of proving equivalence for large kernels directly.
Second, agents write Lean proofs that each kernel segment matches its high-level PyTorch counterpart. Together, the two steps give whole-program equivalence.
We demonstrate that LARK can be used 1) to ensure correctness in existing code-generation agents and 2) to automatically verify human-written kernel libraries. On the KernelCraft GPU tasks, combined with a simple ReAct agent, LARK enables GLM 5.3 to fully verify 10 of 12 Level-1 kernels and Opus 5.5 all 5 Level-2 kernels.
Moreover, over 10 production kernels randomly sampled from popular open-source LLM serving frameworks, LARK verifies 9 and identifies a new data race bug in the tenth.
We believe LARK represents a significant step toward verifiable code generation in the agentic era.

## 1 Introduction

![Refer to caption](assets/22-lark/figure-001.png)


Figure 1: Where trust sits in LLM kernel generation. Top: existing pipelines check a generated kernel by sampled tests or review, so the whole kernel is taken on faith. Bottom: LARK fixes what is assumed (dashed) before the kernel is seen, the machine model and operation semantics formalized once per language and the task’s interface; every property of the kernel is proved in Lean or the kernel is rejected, and the assumptions are reported with every result.

GPU kernels are where the cost of training and serving deep learning models is paid, and the fastest are hand-scheduled CUDA, PTX, or HIP written close to the metal ([Wu et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib20); [Dao et al., 2022](https://arxiv.org/html/2611.02028v1#bib.bib19); [Tillet et al., 2019](https://arxiv.org/html/2611.02028v1#bib.bib21)). Such code is increasingly written by LLM agents ([Nie et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib2); [Ouyang et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib5); [Beel et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib3); [Baronio et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib4)), and a plausible kernel is not a correct one: the same models that write fluent code write fluent bugs ([Lange et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib22); [Chatterjee et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib7)). Every regenerated kernel needs its own correctness check, at the pace of generation.
Existing checks fall short on one count or the other ([Figure 1](https://arxiv.org/html/2611.02028v1#S1.F1), top). Tests on sampled inputs keep pace but give evidence, never a guarantee; human review reasons about all inputs but cannot keep pace; an LLM judge removes the human but not the problem. What a generated kernel needs is a proof of a precise equivalence statement, checked by a proof assistant such as Lean 4 ([Moura and Ullrich, 2021](https://arxiv.org/html/2611.02028v1#bib.bib1)), and formal tools do not provide one either: GPUVerify ([Betts et al., 2012](https://arxiv.org/html/2611.02028v1#bib.bib15)) proves race and divergence freedom, not the specification; a verified compiler ([Leroy, 2009](https://arxiv.org/html/2611.02028v1#bib.bib18)) certifies its own translation, not a generated kernel; a hand-written model ([Armstrong et al., 2019](https://arxiv.org/html/2611.02028v1#bib.bib16)) or proof ([Klein et al., 2010](https://arxiv.org/html/2611.02028v1#bib.bib17)) puts a human back where we want automation.

We present LARK (Lean-Assisted Reasoning for Kernels), an agentic framework in which LLM agents write both the model of the kernel and the proof, and Lean checks both. Once per target language, LARK formalizes in Lean a machine model of the GPU and the semantics of each operation and measures them against the hardware; the machine model is shared across languages, so a new target supplies only its operations and their semantics, and we instantiate LARK for PTX, CUDA, and AMD HIP. Given a PyTorch specification and a kernel from any generator, the kernel’s semantics is derived from these pinned parts, agents write the proofs, and Lean checks every step, so a wrong agent output is a failed proof, never a wrong theorem. This division of labor raises three challenges:

1) Semantic gap. The kernel’s Lean semantics is spelled out operation by operation over the machine model, while the PyTorch reference has a tensor-level semantics in Lean. LARK bridges the two by deriving a *flat model* of the kernel and having an LLM agent segment it, verifiably, into a *segment-level model* that a proof can induct on ([Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)).

2) Fidelity versus tractability. The machine model must be sound for the hardware yet abstract enough to prove against. LARK’s model computes over exact numbers, runs a phase’s threads in sequence, and treats undefined behavior as a fault; what makes this sound, race freedom above all, is proved in Lean per kernel as *obligations*, and a kernel that can fault or race has no theorem ([Sections 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) and [4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)).

3) Trust boundaries. Any assumption the theorem depends on but Lean does not verify must be trusted by humans; we call this set the *trust boundary*. LARK keeps it small and fixed by the task, the hardware machine model, the fixed mapping from operation semantics to kernel semantics, and the formalized task specification, so it can be reviewed once in advance; every run produces a Lean certificate that any Lean project re-checks without repeating the LARK procedure ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

LARK tackles these three challenges, and we make the following contributions:

- We present LARK, an agentic formal-verification workflow for GPU kernels in low-level, close-to-metal languages. To our knowledge, LARK is the first attempt at using agents to produce formally verified low-level GPU code for equivalence check.
- We show how to formalize low-level GPU semantics in Lean 4 and prove functional equivalence between a GPU kernel and a PyTorch reference. LARK combines verified program segmentation with iterative agentic proof construction, and humans review the task’s trust boundary only once, ahead of time in dependent of kernel inputs; every kernel generated afterwards is certified relative to those trusted assumptions.
- We demonstrate that LARK can be integrated with agentic code-generation harnesses to certify their kernels and serves as an automated verifier for human-written kernels. On KernelCraft, GLM 5.3 certifies 10 of the 12 Level-1 kernels and Opus 5.5 all five fused Level-2 kernels, every semantics-breaking mutant is rejected, and nine of ten production kernels verify while the tenth is rejected for a real data race ([Section 5](https://arxiv.org/html/2611.02028v1#S5)).

## 2 Motivating Example



````torch
def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape == (64, 256)
    assert B.shape == (256, 512)
    assert A.dtype == B.dtype == torch.float16
    acc = A.float() @ B.float()
    return acc.half()
````



(a) Tensor-level Torch specification $S$.



````lean
def gemmSig : Sig := [        -- the task's signature: role, type, shape, layout
  .buf .in  .f16 [64, 256] (rowMajor 256),   -- A
  .buf .in  .f16 [256, 512] (rowMajor 512),  -- B
  .buf .out .f16 [64, 512] (rowMajor 512) ]  -- C
def gemmLaunch : Launch := .pinned 32 1 32        -- 32 blocks of 32 threads

theorem kernel_eq_spec (s : MachineState) (A B)
   (h : StdPre gemmSig gemmLaunch s (A, B)) :       -- placed, separated, aligned, geometry
   (*@$\exists$@*) s', runLaunch (prog PTX gemm_src fuels) s = some s'
     (*@$\wedge$@*) stdExtract gemmSig s' = Spec.highlevel_spec A B := by
   ...   -- proof generated by LARK, Lean checked
````



(b) Equivalence theorem in Lean.



````ptx
.entry matmul_f16_mma(.param .u64 p_A, .param .u64 p_B, .param .u64 p_C)
.reqntid 32, 1, 1
{ mov.u32 %r0, %ctaid.x;  // output tile from block index
  ...    // load p_A, p_B, p_C; per-lane addresses from %tid.x; zero %f0..%f31
K_LOOP:  // 16 slabs of k = 16
  ld.global.b32 %w0, [%rd11];   ...   st.shared.b32 [%r30], %w0;  ...  // stage A, B slabs
  bar.sync 0;
  ldmatrix.sync.aligned.m8n8.x4.shared.b16       {%r30..%r33}, [%r24];  // A fragment
  ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r34..%r37}, [%r25];  // B fragments
  mma.sync...m16n8k16.f32.f16.f16.f32 {%f0..%f3}, {%r30..%r33}, {%r34,%r35}, {%f0..%f3};
  ...    // 7 more mma (3 more ldmatrix) for n8 tiles 1..7
(*@\difflinebg[2]{jcredl!20}@*)@@-- // Bug: no barrier before it is re-staged
(*@\difflinebg[2]{jcredl!20}@*)@@-- @%p1 bra K_LOOP;
(*@\difflinebg[2]{jcgreenl!20}@*)@@++ // Fixed by LARK: every lane is done reading the slab
(*@\difflinebg[1]{jcgreenl!20}@*)@@++ bar.sync 0;  @%p1 bra K_LOOP;
  cvt.rn.f16.f32 %h0, %f0;  st.global.b16 [%rd15], %h0;   // C[row0+g, col0+2t]
  ...  ret; }   // after 31 more cvt + st.global.b16 of C
````



(c) PTX kernel $K$, condensed from about 300 lines.

Figure 2: Running example: general matrix multiplication (GEMM).
(a) The PyTorch reference.
(b) The equivalence theorem: the task’s fixed premise (StdPre: inputs placed, buffers apart, bases aligned, launch geometry) implies fault-free completion (some s’) and $AB$ in the region of $C$, over the reals.
(c) The PTX kernel implementing (a); its per-lane addressing, shared-memory staging, barrier, and warp-collective ldmatrix/mma (lines 4–11) each need formal meaning before (b) can be stated. Line 13 (red), in place of line 15 (green), omits the barrier before the next slab is staged.
Testing can only sample (b) and accepts the bug; LARK proves (b) with line 15 and locates the race with line 13.

[Figure 2](https://arxiv.org/html/2611.02028v1#S2.F2) makes the three challenges of [Section 1](https://arxiv.org/html/2611.02028v1#S1) concrete. The specification ([Figure 2(a)](https://arxiv.org/html/2611.02028v1#S2.F2.sf1)) multiplies two FP16 matrices. The kernel ([Figure 2(c)](https://arxiv.org/html/2611.02028v1#S2.F2.sf3)) is LLM-written PTX, the virtual instruction set CUDA compiles to, in the shape of fast kernels: each one-warp block computes one output tile (line 3), staging slabs of $A$ and $B$ into shared memory (lines 5–7) and multiplying them on the tensor cores (lines 8–11) before storing the tile (lines 16–17). The theorem ([Figure 2(b)](https://arxiv.org/html/2611.02028v1#S2.F2.sf2)) relates the two through the state of a machine that runs the kernel, under a premise of the task, not of the kernel; a test samples it, a proof establishes it for every input ([Section 3](https://arxiv.org/html/2611.02028v1#S3)).

What a guarantee catches. Line 13 of [Figure 2(c)](https://arxiv.org/html/2611.02028v1#S2.F2.sf3) is a one-line bug a generator could write: without the barrier, a lane may overwrite a staged slab another lane is still reading. Nothing marks the omission among 300 lines that look alike, and the kernel can pass every test, since on a given GPU the warp may finish ldmatrix before any lane stores, while PTX promises no such order. LARK catches it in Lean: the race-freedom obligation it proves before any value ([Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)) fails for line 13 with a counterexample, two lanes meeting at one shared cell, that points to where the barrier belongs; with line 15 the kernel gets the theorem in [Figure 2(b)](https://arxiv.org/html/2611.02028v1#S2.F2.sf2).

Where the challenges show. Semantic gap: the conclusion speaks of the matrix $AB$, while the kernel updates four registers per lane per mma (line 10) and loops by a branch to a label (line 15), naming no tile, iteration, or sum to induct on. Fidelity versus tractability: a model faithful to every interleaving of 32 lanes and 32 blocks admits no tractable proof, and each abstraction that helps has a condition, such as sequential threads needing lines 6 and 8 never to race, that must be proved rather than assumed for the theorem to be worth anything. Trust: that the Lean mma is what the tensor core computes, that bar.sync is a barrier, and that the Lean specification and interface are those of the task can only be believed, so this trust boundary ([Section 1](https://arxiv.org/html/2611.02028v1#S1)) must exclude whatever an agent writes and stay a short, fixed list; [Section 4](https://arxiv.org/html/2611.02028v1#S4) shows how LARK meets each challenge.

## 3 Problem Definition

[Section 2](https://arxiv.org/html/2611.02028v1#S2) showed the theorem informally; we now state, for a fixed target language, what is proved, what is assumed, and what a verifier must deliver.

Specification.
A *specification* $S$ is a Torch module, denoting a total function $\llbracket S\rrbracket:D\to T$ from the inputs $D$ with the declared shapes, types, and constraints to the output tensors $T$. $S$ is the ground truth.

Kernel and machine model.
A *kernel* $K$ is a program in the target language with a launch geometry. Its meaning is fixed only relative to a *machine model* $M$: the states $\Sigma_{M}$, a partial semantics for each operation, a scheduling rule, and a representation of values (ours is in [Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1)). These define the partial *whole-launch semantics* $\llbracket K\rrbracket_{M}:\Sigma_{M}\rightharpoonup\Sigma_{M}$, defined at $\sigma$ when every maximal execution from $\sigma$ is finite, fault-free, and ends in the same state.

Interface.
Tensors enter and leave the machine through memory under the task’s *interface* $I$: the region and layout of each tensor, the launch geometry, and the disjointness of the regions. $I$ belongs to the task, not to $K$, and yields a *precondition* $\mathrm{In}_{I}(\sigma,x)$, true when $\sigma$ has the declared geometry and holds $x\in D$ in the declared regions, and an *output extractor* $\mathrm{Out}_{I}:\Sigma_{M}\to T$. In [Figure 2(b)](https://arxiv.org/html/2611.02028v1#S2.F2.sf2), StdPre is $\mathrm{In}_{I}$, stdExtract is $\mathrm{Out}_{I}$, and runLaunch is $\llbracket K\rrbracket_{M}$.

Equivalence.
Both sides are compared as real-valued tensors, the kernel’s operations read over exact reals ([Sections 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) and [4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). $K$ is *equivalent to $S$ under $M$ and $I$*, written $K\equiv_{M,I}S$, if



$$
\displaystyle\forall x\in D.\;\forall\sigma\in\Sigma_{M}.\quad\mathrm{In}_{I}(\sigma,x)
\displaystyle\Longrightarrow\;\llbracket K\rrbracket_{M}(\sigma){\downarrow}\;\wedge\;\mathrm{Out}_{I}\big(\llbracket K\rrbracket_{M}(\sigma)\big)=\llbracket S\rrbracket(x).
$$




Quantifying over every initial state that holds the input forbids $K$ from depending on memory outside the interface, and reading the output from the declared region fixes where the result lands. This is all that equivalence in *memory behavior* means: the memories of $M$ are unbounded ([Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1)), so a write outside the interface is invisible to the statement, and out-of-bounds safety on the host’s allocations is not guaranteed.

Trust boundary.
A proof of [Equation 1](https://arxiv.org/html/2611.02028v1#S3.E1) is a proof about $M$; to carry over to the real hardware and host $\mathcal{H}$, it needs facts no proof inside $M$ can establish: that $M$ is faithful to the hardware up to its idealizations, that the formalization of $S$ denotes $\llbracket S\rrbracket$, and that the host honors $I$. This trust boundary must be *fixed by the task, not by the kernel*: $M$ and $I$ are unavoidably premises, nothing else may be, and properties of $K$ such as race freedom or termination are *obligations*, proved per kernel ([Sections 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) and [4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)).

Problem definition.
Fix a target language with machine model $M$. Given a specification $S$, an interface $I$, and a kernel $K$, produce either a *certificate* $(\Pi,\pi)$ or a *rejection*, which makes no claim about $K$. The premises $\Pi=\Pi_{M}\cup\Pi_{K}$ are $\Pi_{M}$, the faithfulness of $M$ to $\mathcal{H}$, and $\Pi_{K}$, the task’s signature and launch, from which $\mathrm{In}_{I}$ and $\mathrm{Out}_{I}$ are computed by fixed definitions; both are reported with the result ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). The machine-checked proof $\pi$ establishes



$$
\Pi_{K}\;\vdash\;K\equiv_{M,I}S,
$$




with $\mathrm{In}_{I}$ satisfiable for every input in $D$, and hence $\Pi_{M}\models K\equiv_{\mathcal{H},I}S$: $\pi$ supplies the equivalence in $M$, $\Pi_{M}$ the passage from $M$ to $\mathcal{H}$.

Requirements.
A solution must be *sound*: a certificate is issued only if $K\equiv_{M,I}S$ holds, and $\pi$ is checked by a checker that does not trust its producer. Soundness makes both conjuncts of [Equation 1](https://arxiv.org/html/2611.02028v1#S3.E1) obligations, proved rather than assumed. Among sound solutions we prefer the *more complete*, certifying more of the kernels that are equivalent, and the *more trustworthy*, certifying with a smaller $\Pi_{M}$. The second objective separates this problem from classical equivalence checking: race freedom and the other regularities that make GPU kernels tractable are properties of $K$, proved per kernel, so that $\Pi_{M}$ keeps only what no per-kernel proof can discharge.

## 4 Method

![Refer to caption](assets/22-lark/figure-002.png)


Figure 3: The LARK pipeline. Offline, the machine model and per-operation semantics are formalized once per target language and measured against the hardware, and the Torch module once per module. Per kernel, the translator composes the flat whole-kernel semantics in Lean, then three Lean-checked stages follow: (1) the race-freedom check; (2) an agent segments the flat model and proves the segment-level model equivalent to it; (3) agents prove the segment-level model equivalent to the Torch semantics. Lean’s error reports drive the agent loops. Green: Lean-checked. Yellow: trusted, listed in the report ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). Red: agent-generated, trusted for nothing.

[Section 3](https://arxiv.org/html/2611.02028v1#S3) asks for a certificate whose premises say nothing about the kernel: everything inside the trust boundary must be fixed before the kernel arrives, and everything about the kernel must be proved. This section follows that split ([Figure 3](https://arxiv.org/html/2611.02028v1#S4.F3)): a Lean model of the kernel we can trust, fixed once per target language ([Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1)); the way from the model to a proof, written by agents and checked by Lean ([Sections 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) and [4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)); and the short, fixed list of what is left to trust ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

### 4.1 A Model We Can Trust

In this section, we describe how, ahead of time, a Lean semantic model can be generated for each operation in a target language such as PTX or HIP. This process involves three key components:
1) Machine model: Defining a machine model encompassing memory, register abstraction, fault handling, and concurrency,
2) Per-op semantic contextualization: Specifying per-operation semantics that contextualize the machine model for the chosen target language;
and 3) Translator: Developing translators to automate kernel semantic composition and support extension to new target languages.

In this setup, the Lean model of a kernel can be trusted because everything it is built from, a machine model, the semantics of each operation, and a translator, is fixed and checked by human experts before the kernel arrives, and none of it is generated by agents.

Machine model. The machine model of [Section 3](https://arxiv.org/html/2611.02028v1#S3) is a quadruple $M=(\Sigma,\mathcal{I},\mathrm{sem},\mathrm{cut})$: the states, the operations of the target language, their semantics, and their classification by the synchronization they perform. Each part trades fidelity for tractability, once, for every kernel.

Memory. A state holds the whole launch:



$$
\Sigma=\mathit{pc}\times\mathit{reg}\times\mathit{mem}_{\mathrm{g}}\times\mathit{mem}_{\mathrm{s}}\times\mathit{mem}_{\mathrm{l}}\times\mathit{geom},
$$




with $\mathit{pc}$ the program counters, $\mathit{reg}$ the registers, keyed by thread and name, $\mathit{mem}_{\mathrm{g}}$ the global memory, $\mathit{mem}_{\mathrm{s}}$ one shared memory per block, $\mathit{mem}_{\mathrm{l}}$ one local memory per thread, and $\mathit{geom}$ the read-only launch geometry and coordinates. Keeping every thread’s registers in one state makes it immediate that an operation of one thread touches no register of another. Memories are unbounded, so the model cannot detect an access outside the host’s allocations, which is outside the guarantee ([Section 3](https://arxiv.org/html/2611.02028v1#S3)).

Values. Every register and memory cell holds a typed value or is undefined:



$$
\mathit{Val}=\{\bot\}\cup\{(\tau,x):\tau\in\mathit{Ty},\ x\in\mathit{Payload}(\tau)\},
$$




a type tag $\tau\in\mathit{Ty}$ (the set is listed in Appendix [A.7](https://arxiv.org/html/2611.02028v1#A1.SS7.SSS0.Px2)) and a payload $x$: an exact real, extended with $-\infty$ for running maxima, under a float tag, an integer under an integer tag, a bit under a predicate. Exact arithmetic is a deliberate idealization, stated once as the scope of the guarantee rather than as an assumption about any kernel: LARK does not track numerical error, [Equation 1](https://arxiv.org/html/2611.02028v1#S3.E1) is equivalence under exact arithmetic, and floating-point behavior is left to testing checks ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

Faults. $\mathrm{sem}_{t}(i):\Sigma\rightharpoonup\Sigma$ is the semantics of operation $i$ executed by thread $t$, which selects the register row and the coordinates. Its undefined cases are the *faults*: an IEEE result outside $\mathbb{R}\cup\{-\infty\}$, a read of the wrong tag or of $\bot$, a misaligned vector access, a negative address, an exhausted fuel (defined below), a non-converged cut. A kernel that can fault has no theorem.

Concurrency. Reasoning about every interleaving of thousands of threads is not practical, and machine-learning kernels do not need it: they work in *phases* separated by barriers, within which no two threads touch the same data. The model makes phases explicit through $\mathrm{cut}:\mathcal{I}\to\{\bot,\mathrm{coll},\mathrm{warp},\mathrm{block},\mathrm{grid},\mathrm{halt}\}$ ([Table 1](https://arxiv.org/html/2611.02028v1#S4.T1)): a barrier is a cut at its level, a return the halt, and a warp-collective operation such as mma a cut of kind $\mathrm{coll}$, where the lanes of the warp meet and the operation executes once over all their registers. A kernel is a sequence $K\in\mathcal{I}^{*}$ with $K[p]$ the operation at program counter $p$; a thread iterates $\mathrm{sem}$ and *parks* at a cut:



$$
\mathrm{step}_{K,t}(\sigma)=\begin{cases}\mathrm{sem}_{t}(K[p])(\sigma)&\text{if }\mathrm{cut}(K[p])=\bot,\\
\sigma&\text{otherwise (parked)},\end{cases}
$$




where $p=\mathit{pc}_{t}(\sigma)$. A launch is a fold over the hierarchy of units, in time and then in parallelism:



$$
\mathrm{run}_{\ell}(\sigma)=\\
\mathrm{conv}_{\ell}\big(\mathrm{run}_{\ell-1}(u_{m})\circ\cdots\circ\mathrm{run}_{\ell-1}(u_{1})\big)(\sigma).
$$
Equation (2)



Here $u_{1},\dots,u_{m}$ are the units one level down (the lanes of a warp, the warps of a block, the blocks of the grid) making up the unit at level $\ell\in\{\mathrm{warp},\mathrm{block},\mathrm{grid}\}$, each run to the next cut of level $\ell$; $\mathrm{conv}_{\ell}(f)(\sigma)=f(\sigma)$ if every thread of the unit parks at one program counter in $f(\sigma)$, undefined otherwise; and $\mathrm{run}_{0}$ iterates $\mathrm{step}$ to the lane’s next cut. Each level has a *fuel* $\kappa_{\ell}\in\mathbb{N}$, the steps or sub-runs allowed before its next cut; $\mathrm{run}^{0}_{\ell}$ is undefined, so running out of fuel is a fault, and $\vec{\kappa}$ is data the run supplies. The theorem requires $\sigma\in\mathrm{dom}(\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}})$.

Per-operation semantics. Each operation $\mathrm{op}$ has a lower-layer function $\widehat{\mathrm{op}}$, plain mathematics independent of the machine ($\widehat{\mathrm{add}}={+}$, $\widehat{\mathrm{max}}=\max$ on $\mathbb{R}\cup\{-\infty\}$, $\widehat{\mathrm{mma}}$ a tile multiply-accumulate), and an upper layer that binds it to the state. Writing an operation as in the kernel, $\mathrm{op}\ d,a_{1},\dots,a_{k}$, with register names $d$ and $a_{j}$, and $\tau_{j}$ and $\tau$ for the tags the operation names for operand $j$ and the result, a scalar operation executed by thread $t$ is



$$
\displaystyle\mathrm{sem}_{t}(\mathrm{op}\ d,a_{1},\dots,a_{k})(\sigma)
\displaystyle=\sigma\big[\mathit{reg}(t,d)\mapsto(\tau,\widehat{\mathrm{op}}(x_{1},\dots,x_{k})),\ \mathit{pc}_{t}\mapsto\mathit{pc}_{t}+1\big]
$$




if $\mathit{reg}(t,a_{j})=(\tau_{j},x_{j})$ for every $j$ and $(x_{1},\dots,x_{k})\in\mathrm{dom}(\widehat{\mathrm{op}})$, and undefined otherwise. [Table 1](https://arxiv.org/html/2611.02028v1#S4.T1) gives the schema of each kind of operation: the guard that bounds the domain, the update, and the cut. The operations of each language and their lower-layer definitions are generated from one pinned table per language ([Figure 3](https://arxiv.org/html/2611.02028v1#S4.F3), per-op semantic formalization), and every definition is measured against a real GPU on random operands before use (Appendix [A.5](https://arxiv.org/html/2611.02028v1#A1.SS5)).

|  |  |  |  |
| --- | --- | --- | --- |
| Kind  (examples) | Guard | Update | $\mathrm{cut}$ |
| Scalar  (add, mul) | tags, $\vec{x}\in\mathrm{dom}\,\widehat{\mathrm{op}}$ | $\mathit{reg}(t,d)\mapsto(\tau,\widehat{\mathrm{op}}(\vec{x}))$ | $\bot$ |
| Memory  (ld, st) | tag, aligned, $\alpha\geq 0$ | $\mathit{reg}(t,d)\leftrightarrow\mathit{mem}(\alpha)$ | $\bot$ |
| Control  (bra, ret) | $\mathit{reg}(t,p)=(\mathrm{pred},b)$ | $\mathit{pc}_{t}\mapsto q$ if $b$ | $\bot$, halt |
| Barrier  (bar.sync) | — | parks | warp, block, grid |
| Collective  (mma) | tags on all lanes | all lanes’ $\mathit{reg}(t^{\prime},d)$ | coll |

Table 1: Operation schemas by kind: guard (failed means a fault), update, and cut; all but Control also advance $\mathit{pc}_{t}$. $\widehat{\mathrm{op}}$ is the operation’s own function ($+$, an address $\alpha$, a branch target $q$, a tile product); $\mathrm{sem}$ is defined per operation, the kinds only group them.

Translator. The translator is fixed code, not an agent, and it never guesses. It reads the kernel text $b$ as data, $\mathrm{parse}_{L}:\mathit{Bytes}\rightharpoonup\mathcal{I}^{*}$ (CUDA and HIP through a structured tree whose lowering $\mathrm{lower}_{L}:\mathit{Tree}\to\mathcal{I}^{*}$ is a definition in $M$), so the kernel’s fold is $\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}}$ on $K=\mathrm{parse}_{L}(b)$ and nothing of the semantics is generated per kernel. Every reading is checked by a *read-back*: a formatter $\mathrm{fmt}_{L}:\mathcal{I}^{*}\to\mathit{Bytes}$ writes the data back and $\mathrm{fmt}_{L}(\mathrm{parse}_{L}(b))=b$ must hold byte for byte, so what is trusted is the formatter, a quarter the size of the reader (Appendix [A.7](https://arxiv.org/html/2611.02028v1#A1.SS7.SSS0.Px3)). An operation or pattern outside $\mathcal{I}$ or the concurrency model (asynchronous operations, atomics, a NaN literal) is refused by name before any proof is attempted, since we would rather refuse a correct kernel than prove a wrong theorem (Appendix [A.9](https://arxiv.org/html/2611.02028v1#A1.SS9)). Finally, the translator prepares the proof with *proved lemmas*, generated with their proofs and checked by Lean (the race-freedom lemmas of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2), the lane bridges and program-counter rungs of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2)), and *scaffolding lemmas*, whose statements it generates and whose proofs an agent writes, so they can be false and then simply fail. Two generator hints, the exact constant behind a rounded literal and the expected launch, guide this generation but decide nothing; Lean can check neither, so both are reported as trusted ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

Retargeting. $M$ splits as $M_{\mathrm{launch}}\cup L$: a launch part shared by every language and a per-language part $L$, which includes its operations with their semantics and cuts, warp width, reader and read-back formatter, collective lane layouts, and lowering (Appendix [A.6](https://arxiv.org/html/2611.02028v1#A1.SS6)). We instantiate LARK for PTX, CUDA, and HIP.

### 4.2 From Model to Proof

The automated Lean proof flow relies on two key checks: ensuring data race freedom and verifying functional equivalence. Data race freedom can be directly verified within Lean ([Figure 3](https://arxiv.org/html/2611.02028v1#S4.F3) Stage 1); on the other hand, verifying full functional equivalence may pose computational challenges. To address this, we propose a segment-level approach: first, perform a formally verified flat-model segmentation ([Figure 3](https://arxiv.org/html/2611.02028v1#S4.F3) Stage 2); then, check functional equivalence at the segment level ([Figure 3](https://arxiv.org/html/2611.02028v1#S4.F3) Stage 3).

Stage 1: Data-race-freedom check. The fold of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) runs the units of a phase one after another, while the hardware runs them at once. The two agree only for race-free kernels, so race freedom is the first thing LARK proves about a kernel, before any equivalence is attempted. The vocabulary is the fold’s: a *phase* $\phi$ at a level is the run between two consecutive cuts of that level, its *units* $u$ are the lanes, warps, or blocks that the level folds, and the footprints $W_{\phi}(u),R_{\phi}(u)\subseteq\mathit{Loc}$ are the locations unit $u$ writes and reads during $\phi$, with $\mathit{Loc}$ the global, shared, and local addresses and the register names of $\Sigma$.

###### Obligation 1 (Race freedom).

$\mathrm{RF}(K)$: for every phase $\phi$ and every two distinct units $u_{j},u_{k}$ of it,
$W_{\phi}(u_{j})\cap W_{\phi}(u_{k})=\emptyset$ and $W_{\phi}(u_{j})\cap R_{\phi}(u_{k})=\emptyset$.

A warp-collective step counts as one more unit: it executes once per warp, with a footprint over the registers of every lane and the memory it reads, and it must meet no lane’s footprint within the warp-level phase around its $\mathrm{coll}$ cut, since that cut orders execution, not memory. Race freedom is what the fold needs: the actions of race-free units commute and the barrier that ends a phase orders the phases, so every execution from $\sigma$ is finite and ends in the fold’s result, $\llbracket K\rrbracket_{M}(\sigma)=\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}}(\sigma)$. This is Lipton’s reduction ([Lipton, 1975](https://arxiv.org/html/2611.02028v1#bib.bib24)) as it applies to data-race-free GPU kernels ([Adve and Hill, 1990](https://arxiv.org/html/2611.02028v1#bib.bib25); [Collingbourne et al., 2013](https://arxiv.org/html/2611.02028v1#bib.bib26); [Betts et al., 2012](https://arxiv.org/html/2611.02028v1#bib.bib15); [Lustig et al., 2019](https://arxiv.org/html/2611.02028v1#bib.bib27)); we rely on it at every level as metatheory, not as a Lean proof ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). To discharge the obligation, the translator derives each footprint from the kernel’s addresses and generates the two lemmas of [Obligation 1](https://arxiv.org/html/2611.02028v1#Thmobligation1), with proofs, for every phase and pair of units (Appendix [A.10](https://arxiv.org/html/2611.02028v1#A1.SS10)); Lean checks them, an agent attempts any lemma whose proof fails, and a kernel with an unproved lemma is rejected. The condition is sufficient, not necessary: a benign race is rejected as readily as a harmful one ([Section 5](https://arxiv.org/html/2611.02028v1#S5) reports one found in a production kernel), the price of never accepting a racy kernel.

Stage 2: Verified flat-model segmentation. The kernel’s *flat model* $F_{K}=\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}}$ on $K$ follows the kernel line by line with a program counter. Built only from the pre-checked parts of [Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) and the kernel’s text, it is trustworthy, and it is one semantics for every language, so CUDA and HIP are lowered to it at no cost in trust. It is not the proof’s view: it talks about single operations and registers where the Torch module talks about tensors, and a direct proof would unfold it operation by operation with no index to induct on. An agent therefore segments the flat model into a *segment-level model* $G_{K}$, a term of the grammar



$$
\displaystyle G::={}
\displaystyle\mathrm{sem}_{t}(i)\mid G;G\mid\mathrm{for}\ j<n\ \mathrm{do}\ G(j)
\displaystyle\mid\mathrm{if}\ p\ \mathrm{then}\ G\ \mathrm{else}\ G,
$$




read as a partial function on states like $F_{K}$: runs of operations become a sequence, repeated runs one body $G(j)$ under an index, and a loop a $\mathrm{for}$ over its body. For the GEMM of [Figure 2](https://arxiv.org/html/2611.02028v1#S2.F2), the inner loop’s twelve program-counter cases of ldmatrix and mma become one body run four times, and the $K$-loop sixteen trips over it. $G_{K}$ calls the same $\mathrm{sem}_{t}$ as the flat model; only indexing and sequencing change, and a $\mathrm{for}$ can be treated by induction, one iteration at a time.

Verifying the segmentation. The segmentation was written by an agent, so by our own rule it is not trusted; the stage ends with a Lean-checked obligation.

###### Obligation 2 (Segmentation).

$\mathrm{SEG}(K)$: for every $\sigma$ with $\mathrm{In}_{I}(\sigma,x)$, $F_{K}(\sigma)=G_{K}(\sigma)$.

This is an equality of partial functions, both faulting or both reaching the same state, proved phase by phase by induction on the folded indices from the lemmas the translator prepared ([Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1)). If the segmentation is wrong, $\mathrm{SEG}(K)$ fails, at no cost in trust.

Proof loop. An agent writes a proof, Lean checks it, and the errors go back to the agent; a proof passes only if it compiles without sorry and #print axioms shows only Lean’s standard axioms. When the attempts run out, a separate agent names the cause, a hard proof, a stuck agent, poorly chosen segments, or a wrong kernel, and the loop continues, restarts, redoes the segmentation, or returns the diagnosis to the kernel generator. This analysis only decides what to try next; whether a kernel is accepted, only Lean decides.

Stage 3: Segment-wise verification loop. The same loop now proves the segment-level model against the Torch semantics, the one place where the formalized module enters.

###### Obligation 3 (Segment-wise equivalence).

$\mathrm{EQ}(K,S)$: for every $x$ and $\sigma$ with $\mathrm{In}_{I}(\sigma,x)$, there is a $\sigma^{\prime}$ with $G_{K}(\sigma)=\sigma^{\prime}$ and $\mathrm{Out}_{I}(\sigma^{\prime})=\llbracket S\rrbracket(x)$.

Segment by segment, the agent shows what each phase and each $\mathrm{for}$ body leaves in the state, and induction carries it across the iterations. Together the obligations give [Equation 1](https://arxiv.org/html/2611.02028v1#S3.E1), the theorem of [Figure 2(b)](https://arxiv.org/html/2611.02028v1#S2.F2.sf2): $\mathrm{EQ}(K,S)$ supplies $\sigma^{\prime}$ with $G_{K}(\sigma)=\sigma^{\prime}$ and $\mathrm{Out}_{I}(\sigma^{\prime})=\llbracket S\rrbracket(x)$, $\mathrm{SEG}(K)$ makes it $F_{K}(\sigma)=\sigma^{\prime}$, and Stage 1 makes that $\llbracket K\rrbracket_{M}(\sigma)=\sigma^{\prime}$.

Statements. The three stages prove a statement that no agent wrote, which is what the soundness claim rests on: a proof of the wrong statement proves nothing. Its premise and extractor are two definitions fixed once in $M$, applied to the task’s *signature* and launch:



$$
\mathrm{In}_{I}:=\mathrm{Pre}_{M}(\mathit{sig},\mathit{launch}),\qquad\mathrm{Out}_{I}:=\mathrm{Ext}_{M}(\mathit{sig}).
$$




$\mathrm{Pre}_{M}$ places every input buffer under its layout, keeps buffers apart and aligned, binds every scalar and dimension, and fixes the launch geometry; $\mathrm{Ext}_{M}$ reads every output buffer under its layout. The signature has one entry per kernel argument, a buffer with its role, element type, shape, and strided layout, or a scalar or dimension bound to the specification or pinned; it is the list the task’s tests launch with, checked against the kernel’s entry point (Appendix [A.8](https://arxiv.org/html/2611.02028v1#A1.SS8)). The theorem quantifies over the kernel’s parameters (base addresses, strides, scalars), every state satisfying the premise, and every input, and the premise is proved satisfiable once for every signature, so no statement holds vacuously.

### 4.3 What We Still Trust

Trust cannot be removed completely. What LARK does is narrow it, from the whole kernel to a short, fixed list, and report the list with every result. A certificate is



$$
\mathcal{C}=(\Pi_{M},\Pi_{K},\vec{\kappa},\pi),\qquad\Pi_{K}=(\mathit{sig},\mathit{launch}),
$$




with the premises $\Pi=\Pi_{M}\cup\Pi_{K}$ of [Section 3](https://arxiv.org/html/2611.02028v1#S3), the fuels $\vec{\kappa}$ the fold runs under (data, not premises: exhausting one is a fault the proof rules out), and a Lean-checked proof



$$
\pi\vdash\ \forall\vec{p}\,\sigma\,x.\ \mathrm{Pre}_{M}(\mathit{sig},\mathit{launch})(\vec{p},\sigma,x)\Rightarrow\mathrm{RF}(K)\ \wedge\\
\exists\sigma^{\prime}.\ \left(\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}}(\sigma)=\sigma^{\prime}\wedge\mathrm{Ext}_{M}(\mathit{sig})(\vec{p},\sigma^{\prime})=\llbracket S\rrbracket(x)\right),
$$




which with the reduction result of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) is $K\equiv_{M,I}S$. The two parts of $\Pi$ differ in what they are and how they are trusted.

$\Pi_{M}$ is kernel-independent, the statement that $M$ is faithful to the hardware $\mathcal{H}$ of [Section 3](https://arxiv.org/html/2611.02028v1#S3): the idealizations of $M$ (exact arithmetic as scope, the fold of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) with the reduction result of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)), $\mathrm{sem}$ as measured against the hardware, the Lean formalization of the Torch module, and Lean itself (full list in Appendix [A.13](https://arxiv.org/html/2611.02028v1#A1.SS13)).
The Torch formalization is the one agent-written artifact no proof covers: it is written independently of kernels, auditable on its own, and tested ahead of time against the module.

$\Pi_{K}$ is kernel-specific and small: the signature and the launch, which come from the task, not from the kernel or an agent ([Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)). What remains to believe is that they describe the task, which the report states by quoting them.

Floating-point behavior lies outside $\Pi$: no bound $|K_{\mathrm{fp}}(x)-\llbracket S\rrbracket(x)|\leq\varepsilon$ is proved for the kernel $K_{\mathrm{fp}}$ as the hardware rounds it, and LARK claims nothing about rounding, overflow, or quantization; a GPU test against the Torch module answers that question, at a tolerance the report states beside the theorem.

Everything else is proved for the kernel at hand or guarded by refusal: $\mathrm{RF}(K)$; the absence of faults, convergence, and termination within the fuel, which together are $\llbracket K\rrbracket_{M}(\sigma){\downarrow}$; and $\mathrm{SEG}(K)$ and $\mathrm{EQ}(K,S)$. A theorem proved once in $M$ assembles these into $\pi$. Each run exports the certificate as a standalone Lean project that re-checks without LARK (Appendix [A.12](https://arxiv.org/html/2611.02028v1#A1.SS12)), with a *trusted-assumption report*: $\Pi_{M}$ as a fixed list of statements, $\Pi_{K}$ quoted, the refusals that applied, and the sites where a named constant was read as exact. Trust is placed only where the report says it is.

## 5 Experiments

Table 2: KernelCraft Level 1 results per target language and model: pass rate over the twelve tasks, and mean input and output tokens (millions) over the proven tasks, since a failed run spends to its budget. Per-task results in Appendix [A.4.1](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS1).



<table>
<tr>
<td><span></span><span>
<span>
<span>
<span>Target</span></span>
<span>
<span>Lang.</span></span>
</span></span><span></span></td>
<td>Metric</td>
<td><span></span> <span>
<span>
<span>
<span>GLM 5.3</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>GLM 5.3</span></span>
<span>
<span>Flash</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>DeepSeek</span></span>
<span>
<span>V4.1 Flash</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>GPT-6 Luna</span></span>
</span></span><span></span></td></tr>
<tr>
<td rowspan="3"><span>PTX</span></td>
<td>pass rate</td>
<td>83%</td>
<td>58%</td>
<td>25%</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>314.53</td>
<td>714.60</td>
<td>148.47</td>
<td>445.93</td></tr>
<tr>
<td>output tokens (M)</td>
<td>8.05</td>
<td>10.54</td>
<td>1.93</td>
<td>3.26</td></tr>
<tr>
<td rowspan="3"><span>CUDA</span></td>
<td>pass rate</td>
<td>75%</td>
<td>50%</td>
<td>25%</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>280.49</td>
<td>626.53</td>
<td>80.45</td>
<td>476.20</td></tr>
<tr>
<td>output tokens (M)</td>
<td>6.18</td>
<td>9.09</td>
<td>1.12</td>
<td>3.42</td></tr>
<tr>
<td rowspan="3"><span>HIP</span></td>
<td>pass rate</td>
<td>83%</td>
<td>50%</td>
<td>25%</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>312.21</td>
<td>773.96</td>
<td>156.15</td>
<td>289.17</td></tr>
<tr>
<td>output tokens (M)</td>
<td>7.15</td>
<td>10.59</td>
<td>1.86</td>
<td>1.89</td></tr>
</table>




Table 3: KernelCraft Level 2 results on PTX: a check mark is a certificate; tokens are input and output tokens (millions), in parentheses for runs that failed or stopped at their budget and left out of the Total average.



<table>
<tr>
<td>Model</td>
<td>Metric</td>
<td>FFN</td>
<td>SwiGLU</td>
<td>RoPE</td>
<td>FlashAttn</td>
<td>GQA</td>
<td><span>Total</span></td></tr>
<tr>
<td rowspan="3"><span>Opus 5.5</span></td>
<td>pass</td>
<td><span>✓</span></td>
<td><span>✓</span></td>
<td><span>✓</span></td>
<td><span>✓</span></td>
<td><span>✓</span></td>
<td>100%</td></tr>
<tr>
<td>input token (M)</td>
<td>78.20</td>
<td>66.60</td>
<td>15.50</td>
<td>172.50</td>
<td>203.90</td>
<td>107.34</td></tr>
<tr>
<td>output token (M)</td>
<td>1.43</td>
<td>1.01</td>
<td>0.37</td>
<td>2.77</td>
<td>3.09</td>
<td>1.73</td></tr>
<tr>
<td rowspan="3"><span>GLM 5.3</span></td>
<td>pass</td>
<td><span>✗</span></td>
<td><span>✗</span></td>
<td><span>✓</span></td>
<td><span>✗</span></td>
<td><span>✗</span></td>
<td>20%</td></tr>
<tr>
<td>input token (M)</td>
<td>(459.60)</td>
<td>(512.30)</td>
<td>71.90</td>
<td>(214.10)</td>
<td>(611.60)</td>
<td>373.90</td></tr>
<tr>
<td>output token (M)</td>
<td>(9.13)</td>
<td>(15.96)</td>
<td>2.70</td>
<td>(4.49)</td>
<td>(14.16)</td>
<td>9.29</td></tr>
</table>



We show that LARK verifies most of the kernels a generation harness writes, each model stopping at a fixed rung of proof difficulty ([Section 5.1](https://arxiv.org/html/2611.02028v1#S5.SS1)); that a certificate is robust, rejecting single-site mutants ([Section 5.2](https://arxiv.org/html/2611.02028v1#S5.SS2)); and that the method holds on production kernels, where it finds a real race ([Section 5.3](https://arxiv.org/html/2611.02028v1#S5.SS3)). Ablations of the failure analysis and the segmentation ([Section 5.4](https://arxiv.org/html/2611.02028v1#S5.SS4)) and the takeaways ([Section 5.5](https://arxiv.org/html/2611.02028v1#S5.SS5)) close the section.

Setup.
We evaluate LARK on KernelCraft ([Nie et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib2)), a recent benchmark of GPU kernel-generation tasks, each with a Torch reference module and tests, which gives LARK its specifications and the tests their baseline: the twelve Level 1 tasks, single operators, on PTX, CUDA, and HIP; and five Level 2 tasks, fused operators (FFN, SwiGLU, RoPE, FlashAttention, GQA), on PTX ([Table 6](https://arxiv.org/html/2611.02028v1#A1.T6)).
To measure verification alone, one kernel is generated once and pinned for every task and language, and every model verifies the same kernel.
The verification agents are four small models on Level 1 (GLM 5.3, GLM 5.3 Flash, DeepSeek V4.1 Flash, GPT-6 Luna) and Opus 5.5 on Level 2, with GLM 5.3, the strongest of the four, as the comparison.
Each verification runs under a cap on dialogue rounds per proof step and a spending ceiling, both read off preliminary runs (Appendix [A.2](https://arxiv.org/html/2611.02028v1#A1.SS2)). Metrics. A verification passes when it ends in a certificate that Lean accepts with its standard axioms only, and the pass rate is the share of tasks that pass; cost is the input and output tokens of the verification.
The numeric screen and the latency measurements use an H100 80GB for PTX and CUDA and an MI300X for HIP.

### 5.1 Verifying Generated Kernels: Difficulty Set by Proof Shape, Not Language

*The tasks form a ladder of proof difficulty, and each model climbs to a fixed height.* [Table 2](https://arxiv.org/html/2611.02028v1#S5.T2) gives the pass rates and token costs on Level 1 and [Table 3](https://arxiv.org/html/2611.02028v1#S5.T3) the results on Level 2 (per-task outcome details in Appendix [A.4.1](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS1)). On all three languages every model proves the elementwise kernels; the two GLM models also prove the row-wise normalizations (GLM 5.3 stops at its spending ceiling on LayerNorm on HIP); only GLM 5.3 proves the tensor-core and convolution kernels; no model proves DepthwiseConv, and BatchMatmul is proved only on HIP, where the pinned kernel runs one 64-lane wavefront per block instead of four warps; and only Opus 5.5 proves the fused Level 2 kernels (pass rate and cost against model size in Appendix [A.4.6](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS6)). Each step up adds a kind of obligation, an induction over a loop, a convergence fact at a cut, a fragment-layout lemma, and a model that cannot close one kind fails every kernel that needs it. The two kernels that stop every model on PTX and CUDA stop them on difficulty alone: no statement is false, no verdict blames the kernel, no construct is refused, and every model stops on the same proof, one warp’s tensor-core loop of 16 or 26 collective steps carried through the lane, warp, and block folds (Appendix [A.4.5](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS5)). Proof cost follows the same shape: over the 116 PTX and CUDA runs it correlates with the length of the compiled program (Spearman 0.61), while the length barely predicts the number of agent turns (0.17), a turn being one proof attempt and its Lean check, so a long kernel costs more per turn, not more attempts (Appendix [A.4.4](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS4)). Every failure is a proof that ran out of rounds or budget, and on PTX and CUDA the two equivalence proofs take over 90% of a run’s tokens (Appendix [A.4.3](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS3)).

*The target language matters less than the kernel, and the lower level is not the harder one.* Every model proves nearly the same set of kernels on PTX, CUDA, and HIP: the pass rates of a model differ by at most one task across the three languages, and the rung at which it stops is the same on each. The few cells that differ turn on the shape of the proof, not on the language. GEMV is the clearest case: the two GLM models prove it on PTX but not on CUDA, where the theorem is identical and every failure-analysis verdict was “theorem true, proof hard”, but the kernel compiles to ten segments with two block barriers inside the $K$-loop against one segment with one barrier on PTX, so the PTX proof is shorter (29 agent turns against 48 for GLM 5.3). The shorter proof does not come from a weaker kernel: the verified PTX kernels are faster than the CUDA ones on half of the Level 1 kernels and within 5% of them on the rest, and 1.8 times faster on GEMV (Appendix [A.4.10](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS10)).

### 5.2 Robustness: Rejecting Semantics-Breaking Mutants

To test that a certificate is robust, we mutate verified kernels and ask whether LARK can correctly reject them. [Table 4](https://arxiv.org/html/2611.02028v1#S5.T4) lists the five mutation rules and the results; each mutant is a verified kernel with one change, on ReLU, Softmax, and GEMM in CUDA, PTX, and HIP. We test one semantics-breaking mutant per applicable rule and kernel; ReLU has no barrier, so R3 applies only to Softmax and GEMM. All runs use GLM 5.3 with the numeric screen disabled.

Table 4: Mutation testing: per rule and target language, the mutants LARK rejects out of those tested. Every mutant is rejected; R3 has two mutants per language because ReLU has no barrier.



<table>
<tr>
<td></td>
<td></td>
<td></td>
<td colspan="3"><span>Rejected / tested</span></td></tr>
<tr>
<td><span>ID</span></td>
<td><span>Rule</span></td>
<td>
<span>
<span><span>Change</span></span>
</span></td>
<td><span>CUDA</span></td>
<td><span>PTX</span></td>
<td><span>HIP</span></td></tr>
<tr>
<td>R1</td>
<td>Arithmetic</td>
<td>
<span>
<span>One operator or immediate ($+$ for $-$, a constant)</span>
</span></td>
<td>3/3</td>
<td>3/3</td>
<td>3/3</td></tr>
<tr>
<td>R2</td>
<td>Load/store</td>
<td>
<span>
<span>One address, buffer, or stored value</span>
</span></td>
<td>3/3</td>
<td>3/3</td>
<td>3/3</td></tr>
<tr>
<td>R3</td>
<td>Barrier</td>
<td>
<span>
<span>One barrier removed or moved one statement</span>
</span></td>
<td>2/2</td>
<td>2/2</td>
<td>2/2</td></tr>
<tr>
<td>R4</td>
<td>Register</td>
<td>
<span>
<span>One variable or source register replaced at a use</span>
</span></td>
<td>3/3</td>
<td>3/3</td>
<td>3/3</td></tr>
<tr>
<td>R5</td>
<td>Loop range</td>
<td>
<span>
<span>One loop bound, increment, or stride</span>
</span></td>
<td>3/3</td>
<td>3/3</td>
<td>3/3</td></tr>
<tr>
<td colspan="3">Total</td>
<td>14/14</td>
<td>14/14</td>
<td>14/14</td></tr>
</table>



LARK rejects all 42 mutants: 11 at the translator’s static checks, 10 by a refuted race-freedom obligation, 18 by the failure analysis showing that the kernel computes a different function from the specification, and 3 by a statement Lean refuses because the mutant needs a premise the task’s signature does not give (Appendix [A.4.11](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS11)).

### 5.3 Verifying Production Kernels

Table 5: Verification of the ten production kernels (adapted as Appendix [A.14](https://arxiv.org/html/2611.02028v1#A1.SS14) records), each in its library’s language (CUDA, except vllm-awq-gemm on PTX) by the model in its column. A check mark is a certificate; tokens are input and output tokens, in millions.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Metric | vllm-silu-and-mul | vllm-rms-norm | sglang-rmsnorm | vllm-rotary-embedding | llmc-softmax-forward |
| model used | GLM 5.3 | GLM 5.3 | GLM 5.3 | GLM 5.3 | GLM 5.3 |
| verified | ✓ | ✓ | ✓ | ✓ | ✗ |
| input tokens (M) | 192.90 | 294.00 | 529.00 | 152.80 | (0) |
| output tokens (M) | 4.56 | 8.12 | 12.23 | 4.44 | (0) |
| Metric | sglang-fused-add-rmsnorm | flashinfer-rmsnorm | vllm-fused-qknorm-rope | fam-flash-attn | vllm-awq-gemm |
| model used | Opus 5.5 | Opus 5.5 | Opus 5.5 | Opus 5.5 | Opus 5.5 |
| verified | ✓ | ✓ | ✓ | ✓ | ✓ |
| input tokens (M) | 70.40 | 62.50 | 146.20 | 117.20 | 223.20 |
| output tokens (M) | 0.99 | 1.19 | 1.87 | 2.00 | 2.86 |

We further evaluate LARK on ten production kernels adapted from vLLM ([Kwon et al., 2023](https://arxiv.org/html/2611.02028v1#bib.bib33)), SGLang ([Zheng et al., 2024](https://arxiv.org/html/2611.02028v1#bib.bib32)), FlashInfer ([Ye et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib34)), llm.c ([Karpathy, 2024](https://arxiv.org/html/2611.02028v1#bib.bib35)), and flash-attention-minimal ([Kim, 2024](https://arxiv.org/html/2611.02028v1#bib.bib36)): *nine verify, and the tenth is rejected for a data race* ([Table 5](https://arxiv.org/html/2611.02028v1#S5.T5)). Each kernel enters the pipeline through small recorded rewrites, most of which reshape the code without changing what it computes; nine of the ten also carry a semantic *departure*, such as a 64-bit index read as 32 bits, and each theorem is about the kernel with its departures, all listed in Appendix [A.14](https://arxiv.org/html/2611.02028v1#A1.SS14). The five with the simplest shapes, the rejected one among them, ran with GLM 5.3, the five that fuse stages or run several phases with Opus 5.5.

The rejected kernel shows what a certificate catches that random-input tests do not. llmc_softmax_forward reads, in its last pass, a neighbouring output cell that another thread of the same block writes in the same pass, with no barrier between. The value read is discarded, so the race is benign and no test sees it. The translator cannot derive the footprint of the store, so the obligation falls to the agent, which finds a counterexample instead, two threads of one block storing and loading the same cell of out in the same phase, confirmed by the independent second opinion. The kernel is rejected, since the obligation is over footprints, not over whether a value read is used (Appendix [A.14](https://arxiv.org/html/2611.02028v1#A1.SS14)).

### 5.4 Ablations: Failure Analysis and Segmentation

Two parts of LARK are heuristics outside the trust boundary, and the runs show that both earn their place. The failure analysis, an analyst that names a cause after a failed round, a falsity probe, and an independent second analyst, only chooses what the loop tries next. *It is cheap, and it never cut off a proof that was going to succeed.* Over the Level 1 runs on PTX and CUDA, the Level 2 runs, and the production kernels (Appendix [A.4.9](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS9)), the analysis costs 0.7% to 4.7% of a model’s input tokens and ended ten dialogues before their ceiling, and in the preliminary runs, where it ran without acting, no proof closed after the point where it would have stopped. It is fallible: it called a correct kernel buggy three times, each refuted by the second analyst; in all, the second analyst disputed eleven of the twenty stopping verdicts it reviewed. None of this touches soundness, since a wrong verdict can only end a run early.

*On a complicated kernel the segmentation saves more than its own equivalence proof costs.* We conducted an ablation that proves the specification directly against the flat model, with no segmentation and no first equivalence, on FlashAttention on PTX with Opus 5.5. The direct proof closed at 220M input tokens over 20 turns, against 120M over 11 for the second equivalence with the segmentation; with the 49M tokens of the segmentation and its equivalence counted, the direct run still used 29% more and crossed a ceiling the default run finished under, because without segments the agent re-proves the same loop body at every program counter (Appendix [A.4.8](https://arxiv.org/html/2611.02028v1#A1.SS4.SSS8)).

### 5.5 Discussion

The experiments point to three properties of the framework.

1) Automated formal verification keeps pace with generation. GLM 5.3 certifies 10 of the 12 Level-1 kernels on PTX and on HIP (9 on CUDA) and Opus 5.5 every Level-2 kernel with no human in the loop, which shows that agentic formal verification of GPU kernels is feasible, and the cost per proof falls as the models improve.

2) A robust pipeline with soundness guaranteed. Lean supplies the soundness: a certificate exists only when every obligation is proved, so no agent can talk its way to one. LARK is also robust in practice: wrong segmentations and wrong analyst verdicts ended as failed proofs, semantics-breaking mutants are rejected ([Section 5.2](https://arxiv.org/html/2611.02028v1#S5.SS2)), and the one production kernel with a race was rejected while the nine correct ones were certified ([Section 5.3](https://arxiv.org/html/2611.02028v1#S5.SS3)).

3) A narrowed trust boundary. What a certificate rests on is independent of the kernel: the machine model, the operation semantics measured against the hardware, and the premise and extractor definitions are fixed and checked offline, before any kernel is seen, the specification is formalized once per task and tested, and everything about the kernel itself is proved.
LARK is thus a step toward verifiable code generation in the agentic era: the generator stays untrusted, agents write the proofs, and the certificates are sound.

## 6 Related Work

Kernel generators and benchmarks ([Ouyang et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib5); [Baronio et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib4); [Nie et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib2); [Ye et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib12)) judge a kernel correct when its outputs match a reference on test inputs, a check that generated kernels have been shown to exploit ([Lange et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib22); [Woo et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib23)). Stricter tests ([Sarkar, 2026](https://arxiv.org/html/2611.02028v1#bib.bib11); [Shah and Shrestha, 2026](https://arxiv.org/html/2611.02028v1#bib.bib13)) catch more errors, but on every untested input the kernel’s correctness is still assumed. Earlier formal tools either prove only race freedom ([Betts et al., 2012](https://arxiv.org/html/2611.02028v1#bib.bib15)) or rely on human proofs for each program ([Ferrell et al., 2019](https://arxiv.org/html/2611.02028v1#bib.bib14); [Goel et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib29)).

Two recent verifiers have been applied to LLM-generated kernels. Volta ([Driscoll et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib6)) decides equivalence between a PTX kernel and a reference PTX kernel over the reals. Its specification is thus itself a low-level kernel that must be correct, its decision procedure is not mechanically verified, and its evaluation checks only the first pair of thread blocks, which covers the full grid only if every block behaves alike. ProofWright ([Chatterjee et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib7)) has LLM agents annotate CUDA kernels for the VerCors verifier ([Armborst et al., 2024](https://arxiv.org/html/2611.02028v1#bib.bib39)), which proves memory and thread safety, but proves equivalence only for element-wise kernels with one output per thread, relying on manual preprocessing of kernels and hand-checked LLM annotations.

In both systems, what must be trusted includes assumptions about, or manual work on, the kernel itself. LARK’s trust boundary is instead fixed before any kernel is seen: for each kernel, it proves in Lean equivalence to the task’s Torch module under exact arithmetic, together with race freedom, absence of faults, and termination, or rejects the kernel.
Appendix [A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px4) discusses further related work.

## 7 Conclusion

We propose LARK, a framework that verifies LLM-generated GPU kernels against their Torch specifications in Lean, with the trust boundary moved off the kernel. It gives generated code the check it lacked: one that keeps pace with generation and means something when it passes. The machine model and the operation semantics are fixed and checked once per target language; everything about the kernel, race freedom above all, is proved per kernel or the kernel is rejected; and agents may write the segmentations and the proofs because Lean checks every line, so a wrong agent output is a failed proof, never a wrong theorem. GLM 5.3 certifies 10 of the 12 Level-1 kernels and Opus 5.5 all five fused Level-2 kernels, and nine of ten adapted production kernels verify. The tenth, a softmax kernel from llm.c, is rejected for a real data race present in the shipped code, a bug invisible to input–output testing because the racing value is discarded, which LARK located from the code alone.

### Limitations

LARK’s guarantee is bounded in three ways. Arithmetic is idealized: floats and integers are exact, $+\infty$ and NaN are faults, and the theorem says nothing about rounding, overflow, accumulation order, or the precision of approximate instructions; floating-point behavior is tested, not proved ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). Concurrency is phase-structured: blocks that communicate within a phase, benign races, and warps that halt early are rejected. The operation set is the pinned tables: asynchronous operations (TMA, wgmma, mbarrier), atomics, partial masks, bf16, 8-bit types, non-polynomial addresses, and multi-dimensional blocks are refused (Appendix [A.9](https://arxiv.org/html/2611.02028v1#A1.SS9)). None of these changes the method; each costs a pinned definition or a lemma emitter, and retargeting to a new language is manual (Appendix [A.6](https://arxiv.org/html/2611.02028v1#A1.SS6)).

## References

- Adve and Hill (1990)
  S. V. Adve and M. D. Hill
  Weak ordering—a new definition.
  In Proceedings of the 17th Annual International Symposium on Computer Architecture,
  ISCA ’90, New York, NY, USA, pp. 2–14.
  External Links: ISBN 0897913663,
  [Link](https://doi.org/10.1145/325164.325100),
  [Document](https://dx.doi.org/10.1145/325164.325100)
  Cited by: [§4.2](https://arxiv.org/html/2611.02028v1#S4.SS2.p3.1).
- Ammanamanchi et al. (2026)
  P. S. Ammanamanchi, S. Bhat, and S. Biderman
  Faults in our formal benchmarking: dataset defects and evaluation failures in Lean theorem proving.
  In Proceedings of the 43rd International Conference on Machine Learning,
  PMLR, Vol. 306, pp. 2431–2462.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px3.p1.1).
- Armborst et al. (2024)
  L. Armborst, P. Bos, L. B. van den Haak, M. Huisman, R. Rubbens, Ö. Şakar, and P. Tasche
  The vercors verifier: a progress report.
  In International Conference on Computer Aided Verification,
  pp. 3–18.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px4.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p2.1).
- Armstrong et al. (2019)
  A. Armstrong, T. Bauereiss, B. Campbell, A. Reid, K. E. Gray, R. M. Norton, P. Mundkur, M. Wassell, J. French, C. Pulte, S. Flur, I. Stark, N. Krishnaswami, and P. Sewell
  ISA semantics for armv8-a, risc-v, and cheri-mips.
  Proc. ACM Program. Lang. 3 (POPL).
  External Links: [Link](https://doi.org/10.1145/3290384),
  [Document](https://dx.doi.org/10.1145/3290384)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Banerjee et al. (2026)
  D. Banerjee, O. Bouissou, and S. Zetzsche
  DafnyPro: llm-assisted automated verification for dafny programs.
  arXiv preprint arXiv:2601.05385.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px3.p1.1).
- Baronio et al. (2026)
  C. Baronio, P. Marsella, B. Pan, S. Guo, and S. Alberti
  Kevin: multi-turn rl for generating cuda kernels.
  In International Conference on Learning Representations,
  Vol. 2026, pp. 83418–83452.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Beel et al. (2025)
  J. Beel, M. Kan, and M. Baumgart
  Evaluating sakana’s ai scientist: bold claims, mixed results, and a promising future?.
  SIGIR Forum 59 (1), pp. 1–20.
  External Links: ISSN 0163-5840,
  [Link](https://doi.org/10.1145/3769733.3769747),
  [Document](https://dx.doi.org/10.1145/3769733.3769747)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Betts et al. (2012)
  A. Betts, N. Chong, A. Donaldson, S. Qadeer, and P. Thomson
  GPUVerify: a verifier for gpu kernels.
  In Proceedings of the ACM International Conference on Object Oriented Programming Systems Languages and Applications,
  OOPSLA ’12, New York, NY, USA, pp. 113–132.
  External Links: ISBN 9781450315616,
  [Link](https://doi.org/10.1145/2384616.2384625),
  [Document](https://dx.doi.org/10.1145/2384616.2384625)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§4.2](https://arxiv.org/html/2611.02028v1#S4.SS2.p3.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Chatterjee et al. (2026)
  B. Chatterjee, D. Zagieboylo, S. Damani, S. Hari, and C. Kozyrakis
  ProofWright: towards agentic formal verification of cuda.
  External Links: 2511.12294,
  [Link](https://arxiv.org/abs/2511.12294)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px4.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p2.1).
- Collingbourne et al. (2013)
  P. Collingbourne, A. F. Donaldson, J. Ketema, and S. Qadeer
  Interleaving and lock-step semantics for analysis and verification of gpu kernels.
  In European Symposium on Programming,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:17382289)
  Cited by: [§4.2](https://arxiv.org/html/2611.02028v1#S4.SS2.p3.1).
- Dao et al. (2022)
  T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré
  FlashAttention: fast and memory-efficient exact attention with io-awareness.
  In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.),
  Vol. 35, pp. 16344–16359.
  External Links: [Document](https://dx.doi.org/10.52202/068431-1189),
  [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Driscoll et al. (2026)
  B. Driscoll, K. Dubey, A. Wei, N. Kayal, R. Sharma, and A. Aiken
  Equivalence checking of ML GPU kernels.
  Proc. ACM Program. Lang. 10 (OOPSLA2).
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px4.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p2.1).
- Feng et al. (2026)
  Y. Feng, D. Kafle, V. Gladshtein, V. Kurin, G. Pîrlea, Q. Zhao, P. Müller, and I. Sergey
  Certified program synthesis with a multi-modal verifier.
  External Links: 2604.16584,
  [Link](https://arxiv.org/abs/2604.16584)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px3.p1.1).
- Ferrell et al. (2019)
  B. Ferrell, J. Duan, and K. W. Hamlen
  CUDA au coq: a framework for machine-validating gpu assembly programs.
  In 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE),
  Vol. , pp. 474–479.
  External Links: [Document](https://dx.doi.org/10.23919/DATE.2019.8715160)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Goel et al. (2026)
  S. Goel, A. Keizer, Siddharth, Y. Peng, K. Morrison, N. Wetzler, L. de Moura, N. Ebeid, J. Lee, A. Letson, L. Cicolini, S. Kong, and dependabot[bot]
  Leanprover/LNSym.
  Note: https://github.com/leanprover/LNSym
  External Links: [Link](https://github.com/leanprover/LNSym)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Karpathy (2024)
  A. Karpathy
  llm.c: LLM training in simple, raw C/CUDA.
  Note: <https://github.com/karpathy/llm.c>GitHub repository
  Cited by: [§5.3](https://arxiv.org/html/2611.02028v1#S5.SS3.p1.1).
- Kim (2024)
  P. Kim
  Flash-attention-minimal: flash attention in $\sim$100 lines of CUDA.
  Note: <https://github.com/tspeterkim/flash-attention-minimal>GitHub repository
  Cited by: [§5.3](https://arxiv.org/html/2611.02028v1#S5.SS3.p1.1).
- Klein et al. (2010)
  G. Klein, J. Andronick, K. Elphinstone, G. Heiser, D. Cock, P. Derrin, D. Elkaduwe, K. Engelhardt, R. Kolanski, M. Norrish, T. Sewell, H. Tuch, and S. Winwood
  SeL4: formal verification of an operating-system kernel.
  Commun. ACM 53 (6), pp. 107–115.
  External Links: ISSN 0001-0782,
  [Link](https://doi.org/10.1145/1743546.1743574),
  [Document](https://dx.doi.org/10.1145/1743546.1743574)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Kwon et al. (2023)
  W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica
  Efficient memory management for large language model serving with pagedattention.
  In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,
  Cited by: [§5.3](https://arxiv.org/html/2611.02028v1#S5.SS3.p1.1).
- Lange et al. (2025)
  R. T. Lange, Q. Sun, A. Prasad, M. Faldor, Y. Tang, and D. Ha
  Towards robust agentic cuda kernel benchmarking, verification, and optimization.
  External Links: 2509.14279,
  [Link](https://arxiv.org/abs/2509.14279)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Leroy (2009)
  X. Leroy
  Formal verification of a realistic compiler.
  Commun. ACM 52 (7), pp. 107–115.
  External Links: ISSN 0001-0782,
  [Link](https://doi.org/10.1145/1538788.1538814),
  [Document](https://dx.doi.org/10.1145/1538788.1538814)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Lipton (1975)
  R. J. Lipton
  Reduction: a method of proving properties of parallel programs.
  Commun. ACM 18 (12), pp. 717–721.
  External Links: ISSN 0001-0782,
  [Link](https://doi.org/10.1145/361227.361234),
  [Document](https://dx.doi.org/10.1145/361227.361234)
  Cited by: [§4.2](https://arxiv.org/html/2611.02028v1#S4.SS2.p3.1).
- Lopes et al. (2021)
  N. P. Lopes, J. Lee, C. Hur, Z. Liu, and J. Regehr
  Alive2: bounded translation validation for llvm.
  In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation,
  PLDI 2021, New York, NY, USA, pp. 65–79.
  External Links: ISBN 9781450383912,
  [Link](https://doi.org/10.1145/3453483.3454030),
  [Document](https://dx.doi.org/10.1145/3453483.3454030)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1).
- Lustig et al. (2019)
  D. Lustig, S. Sahasrabuddhe, and O. Giroux
  A formal analysis of the nvidia ptx memory consistency model.
  In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems,
  ASPLOS ’19, New York, NY, USA, pp. 257–270.
  External Links: ISBN 9781450362405,
  [Link](https://doi.org/10.1145/3297858.3304043),
  [Document](https://dx.doi.org/10.1145/3297858.3304043)
  Cited by: [§4.2](https://arxiv.org/html/2611.02028v1#S4.SS2.p3.1).
- Moura and Ullrich (2021)
  L. d. Moura and S. Ullrich
  The lean 4 theorem prover and programming language.
  In Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings,
  Berlin, Heidelberg, pp. 625–635.
  External Links: ISBN 978-3-030-79875-8,
  [Link](https://doi.org/10.1007/978-3-030-79876-5_37),
  [Document](https://dx.doi.org/10.1007/978-3-030-79876-5%5F37)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Necula (2000)
  G. C. Necula
  Translation validation for an optimizing compiler.
  In Proceedings of the ACM SIGPLAN 2000 Conference on Programming Language Design and Implementation,
  PLDI ’00, New York, NY, USA, pp. 83–94.
  External Links: ISBN 1581131992,
  [Link](https://doi.org/10.1145/349299.349314),
  [Document](https://dx.doi.org/10.1145/349299.349314)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§A.7](https://arxiv.org/html/2611.02028v1#A1.SS7.SSS0.Px3.p1.1).
- Nie et al. (2026)
  J. Nie, H. Wu, Y. Lai, Z. Cao, C. Zhang, B. Lou, E. Wang, J. Cheng, T. M. Jones, R. Mullins, R. Antonova, and Y. Zhao
  KernelCraft: benchmarking for agentic close-to-metal kernel generation on emerging hardware.
  External Links: 2603.08721,
  [Link](https://arxiv.org/abs/2603.08721)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [Table 9](https://arxiv.org/html/2611.02028v1#A1.T9),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§5](https://arxiv.org/html/2611.02028v1#S5.p2.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Ouyang et al. (2025)
  A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, and A. Mirhoseini
  KernelBench: can llms write efficient gpu kernels?.
  In Proceedings of the 42nd International Conference on Machine Learning,
  ICML’25.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Pnueli et al. (1998)
  A. Pnueli, M. Siegel, and E. Singerman
  Translation validation.
  In Proceedings of the 4th International Conference on Tools and Algorithms for Construction and Analysis of Systems,
  TACAS ’98, Berlin, Heidelberg, pp. 151–166.
  External Links: ISBN 3540643567
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1),
  [§A.7](https://arxiv.org/html/2611.02028v1#A1.SS7.SSS0.Px3.p1.1).
- Pouchet et al. (2024)
  L. Pouchet, E. Tucker, N. Zhang, H. Chen, D. Pal, G. Rodríguez, and Z. Zhang
  Formal verification of source-to-source transformations for hls.
  In Proceedings of the 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays,
  pp. 97–107.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px2.p1.1).
- Sarkar (2026)
  D. Sarkar
  The correctness illusion in llm-generated gpu kernels.
  External Links: 2606.20128,
  [Link](https://arxiv.org/abs/2606.20128)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Shah and Shrestha (2026)
  R. Shah and R. Shrestha
  A contract-grade verifier for llm-generated gpu kernels, and a native blackwell backward for the gated-linear-recurrence family.
  External Links: 2608.12700,
  [Link](https://arxiv.org/abs/2608.12700)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Tillet et al. (2019)
  P. Tillet, H. T. Kung, and D. Cox
  Triton: an intermediate language and compiler for tiled neural network computations.
  In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages,
  MAPL 2019, New York, NY, USA, pp. 10–19.
  External Links: ISBN 9781450367196,
  [Link](https://doi.org/10.1145/3315508.3329973),
  [Document](https://dx.doi.org/10.1145/3315508.3329973)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Woo et al. (2025)
  J. Woo, S. Zhu, A. Nie, Z. Jia, Y. Wang, and Y. Park
  Tritonrl: training llms to think and code triton without cheating.
  arXiv preprint arXiv:2510.17891.
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Wu et al. (2026)
  H. Wu, A. Kerr, Junkai-Wu, drazi, A. SHIVAM, Y. Zhai, reed, D. Merrill, J. Kosaian, Q. Yuhang, A. Hassani, F. Han, Miller, L. Zheng, V. Thakkar, dan_the_3rd, M. Nicely, G. M. (gregjm), T. Dao, A. Belevich, C. Sigg, M. Ananth, masahi, zky, Lain, K. Zhong, A. Akhundov, D. Blasig, Johnson, and L. Mao
  NVIDIA/cutlass.
  Note: https://github.com/NVIDIA/cutlass
  External Links: [Link](https://github.com/NVIDIA/cutlass)
  Cited by: [§1](https://arxiv.org/html/2611.02028v1#S1.p1.1).
- Xiao et al. (2022)
  D. Xiao, Z. LIU, Y. Yuan, Q. Pang, and S. Wang
  Metamorphic testing of deep learning compilers.
  Proc. ACM Meas. Anal. Comput. Syst. 6 (1).
  External Links: [Link](https://doi.org/10.1145/3508035),
  [Document](https://dx.doi.org/10.1145/3508035)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1).
- Ye et al. (2025)
  Z. Ye, L. Chen, R. Lai, W. Lin, Y. Zhang, S. Wang, T. Chen, B. Kasikci, V. Grover, A. Krishnamurthy, and L. Ceze
  FlashInfer: efficient and customizable attention engine for llm inference serving.
  External Links: 2501.01005,
  [Link](https://arxiv.org/abs/2501.01005)
  Cited by: [§5.3](https://arxiv.org/html/2611.02028v1#S5.SS3.p1.1).
- Ye et al. (2026)
  Z. Ye, Y. Huang, H. Jin, B. Hou, J. Shao, Z. Yu, J. Chen, M. Cowan, S. Cao, S. Xing, H. Chen, V. Grover, T. Chen, and L. Ceze
  CAKE: compiler-agent co-design for frontier kernel evolution.
  External Links: 2608.12629,
  [Link](https://arxiv.org/abs/2608.12629)
  Cited by: [§A.1](https://arxiv.org/html/2611.02028v1#A1.SS1.SSS0.Px1.p1.1),
  [§6](https://arxiv.org/html/2611.02028v1#S6.p1.1).
- Zheng et al. (2024)
  L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng
  SGLang: efficient execution of structured language model programs.
  In Proceedings of the 38th International Conference on Neural Information Processing Systems,
  NIPS ’24, Red Hook, NY, USA.
  External Links: ISBN 9798331314385
  Cited by: [§5.3](https://arxiv.org/html/2611.02028v1#S5.SS3.p1.1).

## Appendix A Appendix

### A.1 Related Works

##### Kernel generation and test-based checking.

Existing LLM kernel generation across target languages, from CUDA to close-to-metal assembly ([Ouyang et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib5); [Baronio et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib4); [Nie et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib2); [Ye et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib12)), checks correctness by comparing a kernel’s outputs with a reference implementation on sampled inputs. Such testing has proved unreliable: kernels that pass KernelBench’s checks have achieved fake speedups of $50$–$120\times$ by exploiting the evaluation harness, and excluding the affected tasks reduces the reported aggregate speedup from $3.13\times$ to $1.49\times$ ([Lange et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib22)). Reward hacking is likewise a central obstacle for generators trained with reinforcement learning ([Woo et al., 2025](https://arxiv.org/html/2611.02028v1#bib.bib23)). Stricter test suites, contract-based checks, and metamorphic testing ([Sarkar, 2026](https://arxiv.org/html/2611.02028v1#bib.bib11); [Shah and Shrestha, 2026](https://arxiv.org/html/2611.02028v1#bib.bib13); [Xiao et al., 2022](https://arxiv.org/html/2611.02028v1#bib.bib10)) catch more errors, but their results remain empirical: a kernel that passes them has been checked only on the inputs that were tried, and its correctness on every other input is assumed. The kernel therefore remains inside the trust boundary.

##### Formal verification and equivalence checking.

Translation validation ([Pnueli et al., 1998](https://arxiv.org/html/2611.02028v1#bib.bib30); [Necula, 2000](https://arxiv.org/html/2611.02028v1#bib.bib31); [Lopes et al., 2021](https://arxiv.org/html/2611.02028v1#bib.bib28)), equivalence checking of program transformations ([Pouchet et al., 2024](https://arxiv.org/html/2611.02028v1#bib.bib9)), and verified compilers ([Leroy, 2009](https://arxiv.org/html/2611.02028v1#bib.bib18)) relate a program to the source from which it was derived. These techniques do not apply directly to LLM-generated kernels, which have no such source and are specified only by a high-level module in a different language. Interactive proofs against a formal ISA semantics ([Ferrell et al., 2019](https://arxiv.org/html/2611.02028v1#bib.bib14); [Goel et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib29)) exclude the kernel from the trust boundary, but require substantial manual effort for each program. Conversely, GPUVerify ([Betts et al., 2012](https://arxiv.org/html/2611.02028v1#bib.bib15)) is fully automatic but establishes only the absence of data races and barrier divergence; since it does not relate a kernel to its specification, the kernel’s functional behavior remains within the trust boundary. LARK combines both properties: it is automatic, treats race freedom as one proof obligation among several, and additionally proves that the kernel is equivalent to its specification.

##### LLM-driven proving and its failure modes.

LLM agents can now write machine-checked proofs ([Feng et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib8)). A checked proof, however, is only as trustworthy as the statement it proves, and statements are where errors concentrate: audits of Lean benchmarks find missing hypotheses, simplified problems, and vacuous theorems ([Ammanamanchi et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib37)), and models asked to annotate a program for verification have been observed to modify the program instead ([Banerjee et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib38)). When the agent that writes the proof also writes the statement, the statement lies inside the trust boundary, and so does the agent. LARK keeps agents out of the per-kernel trust boundary: the theorem is fixed by the task before the kernel is seen, and Lean checks every proof an agent writes, so an agent’s error about the kernel can only cause a proof to fail, never a certificate for the wrong theorem. The one agent-written artifact that remains trusted, the specification formalization, is per task, tested, and reported as such ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

##### Formal verification of LLM-generated GPU kernels.

Two recent verifiers have been applied to LLM-generated kernels. Volta ([Driscoll et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib6)), a general equivalence checker, uses symbolic execution to decide equivalence between a PTX kernel and a reference PTX kernel over the reals. Its specification is thus itself a low-level kernel that must be correct, its decision procedure is not mechanically verified, and its evaluation checks only the first pair of thread blocks, which covers the full grid only if every block behaves alike. ProofWright ([Chatterjee et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib7)) has LLM agents annotate CUDA kernels for the deductive verifier VerCors ([Armborst et al., 2024](https://arxiv.org/html/2611.02028v1#bib.bib39)), which proves memory and thread safety, but proves equivalence only for element-wise kernels with one output per thread, relying on manual preprocessing of each kernel and hand-checked LLM annotations. LARK instead checks the kernel against the task’s Torch module, fixes its trust boundary before any kernel is seen, and proves in Lean equivalence under exact arithmetic, together with race freedom, absence of faults, and termination, or rejects the kernel.

### A.2 Experiment Setup Details

[Table 6](https://arxiv.org/html/2611.02028v1#A1.T6) lists the evaluated tasks by KernelCraft level, [Table 7](https://arxiv.org/html/2611.02028v1#A1.T7) the round caps and spending ceilings per model, and [Table 8](https://arxiv.org/html/2611.02028v1#A1.T8) the kernels by the highest cut level of their phase structure.

Table 6: The evaluated KernelCraft tasks by level.

|  |  |
| --- | --- |
| KernelCraft Level | Tasks |
| Level 1: Primitive Operations | silu, relu, gelu, softmax, layernorm, rmsnorm, gemv, gemm, batchmatmul, linear, conv2d, depthwise_conv |
| Level 2: Composite Operations | rope, ffn, swiglu, flash_attn, gqa |


Table 7: Per-verification budgets by model and task group: the cap on dialogue rounds per proof step (--max-auto-tries) and the spending ceiling (--max-usd). The round caps were read off preliminary runs in which the failure analyst always answered CONTINUE (GEMM on PTX with DeepSeek V4.1 Flash and GLM 5.3, GQA on PTX with Opus 5.5): the cap is the round of the last proof closure at the 75th to 90th percentile, plus one. The ceiling is the dollar amount at which a run was stopped; it is given for reference only.



<table>
<tr>
<td></td>
<td></td>
<td colspan="5"><span>Rounds / ceiling ($)</span></td></tr>
<tr>
<td>
<span>
<span><span>Group</span></span>
</span></td>
<td>
<span>
<span><span>Workloads</span></span>
</span></td>
<td><span></span> <span>
<span>
<span>
<span>DeepSeek</span></span>
<span>
<span>V4.1 Flash</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>GPT-6</span></span>
<span>
<span>Luna</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>GLM 5.3</span></span>
<span>
<span>Flash</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>GLM 5.3</span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>Opus 5.5</span></span>
</span></span><span></span></td></tr>
<tr>
<td>
<span>
<span>L1</span>
</span></td>
<td>
<span>
<span><span>relu</span>, <span>silu</span>, <span>gelu</span>, <span>softmax</span>, <span>layernorm</span>, <span>rmsnorm</span>, <span>gemv</span>, <span>gemm</span>, <span>linear</span>, <span>batchmatmul</span>, <span>conv2d</span>, <span>depthwise_conv</span></span>
</span></td>
<td>16 / 50</td>
<td>16 / 50</td>
<td>16 / 75</td>
<td>8 / 100</td>
<td>—</td></tr>
<tr>
<td>
<span>
<span>L2</span>
</span></td>
<td>
<span>
<span><span>rope</span></span>
</span></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>16 / 100</td>
<td>10 / 100</td></tr>
<tr>
<td></td>
<td>
<span>
<span><span>ffn</span>, <span>swiglu</span>, <span>flash_attn</span>, <span>gqa</span></span>
</span></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>16 / 100</td>
<td>10 / 200</td></tr>
<tr>
<td>
<span>
<span>Library</span>
</span></td>
<td>
<span>
<span><span>vllm_silu_and_mul</span>, <span>sglang_rmsnorm</span>, <span>vllm_rms_norm</span>, <span>vllm_rotary_embedding</span></span>
</span></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>8 / 100</td>
<td>—</td></tr>
<tr>
<td></td>
<td>
<span>
<span><span>llmc_softmax_forward</span>, <span>sglang_fused_add_rmsnorm</span>, <span>flashinfer_rmsnorm</span>, <span>vllm_fused_qknorm_rope</span></span>
</span></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>10 / 100</td></tr>
<tr>
<td></td>
<td>
<span>
<span><span>fam_flash_attn</span>, <span>vllm_awq_gemm</span></span>
</span></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>10 / 200</td></tr>
</table>




Table 8: The evaluation kernels by the highest cut level of their phase structure ([Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1)): the 17 kernels of the core set (KernelCraft Levels 1 and 2) and the 10 production kernels. A kernel with no cut runs as one phase; a warp-collective operation alone does not add a level.

|  |  |  |
| --- | --- | --- |
| Highest cut | Core set | Library kernels |
| None | relu, silu, gelu, conv2d, rope | vllm_silu_and_mul, vllm_rotary_embedding |
| Warp | — | vllm_fused_qknorm_rope |
| Block | softmax, layernorm, rmsnorm, gemv, gemm, linear, batchmatmul, depthwise_conv, flash_attn, gqa | sglang_rmsnorm, vllm_rms_norm, flashinfer_rmsnorm, sglang_fused_add_rmsnorm, llmc_softmax_forward, fam_flash_attn, vllm_awq_gemm |
| Grid | ffn, swiglu | — |

##### Preliminary runs.

The round caps of [Table 7](https://arxiv.org/html/2611.02028v1#A1.T7) were read off preliminary verify-only runs in which every failed round continued the same dialogue, under a cap of 15 rounds, while the failure analyst still ran and recorded what the normal gate would have done. A round *closes an obligation* when the agent’s proof of that round discharges one of the open lemmas or goals of the equivalence, and the rule is a high percentile of the last closing round, plus one round of margin. DeepSeek V4.1 Flash ran GEMM on PTX eight times, none proving the first equivalence, with last closing rounds at 1, 5, 8, 10, 11, 12, 23, and 23; the 75th percentile gives a cap of 16, which GPT-6 Luna and GLM 5.3 Flash, without preliminary runs of their own, take by decision. GLM 5.3 ran GEMM on PTX three times and proved both equivalences each time, with last closing rounds 1, 1, and 3 for the first and 5, 7, and 6 for the second, a cap of 8. Opus 5.5 ran GQA on PTX once and closed the second equivalence in round 9, a cap of 10. The spending ceilings are the amounts at which these runs were stopped. None of these runs is an evaluation cell.

[Table 9](https://arxiv.org/html/2611.02028v1#A1.T9) gives the configuration of each evaluated task in the notation of KernelCraft; every task runs at the largest of the benchmark’s five configurations.

Table 9: Configuration of each evaluated task, in the notation of KernelCraft ([Nie et al., 2026](https://arxiv.org/html/2611.02028v1#bib.bib2)): every task runs at the largest of the benchmark’s five configurations, and all tensors are FP16. $n$ = number of elements (input tokens), $M,K,N$ = matrix dimensions, $b$ = batch size, $h_{\mathrm{in}},h_{\mathrm{out}}$ = input and output features, $h$ = hidden size, $h_{i}$ = intermediate size, $s$ = sequence length, $n_{h}$ = attention heads, $n_{kv}$ = key-value heads, $d_{\mathrm{head}}$ = head dimension, $H$ = spatial size (height $=$ width), $c_{\mathrm{in}},c_{\mathrm{out}}$ = input and output channels, $c$ = channels, $k$ = kernel size.



<table>
<tr>
<td><span>Task (parameters)</span></td>
<td><span>Configuration</span></td></tr>
<tr>
<td colspan="2"><em>Level 1</em></td></tr>
<tr>
<td>SiLU, ReLU, GELU $(n)$</td>
<td>$(16384)$</td></tr>
<tr>
<td>Softmax, LayerNorm, RMSNorm $(n)$</td>
<td>$(16384)$</td></tr>
<tr>
<td>GEMV $(M,N)$</td>
<td>$(512,1024)$</td></tr>
<tr>
<td>GEMM $(M,K,N)$</td>
<td>$(64,256,512)$</td></tr>
<tr>
<td>BatchMatmul $(b,M,K,N)$</td>
<td>$(32,64,256,512)$</td></tr>
<tr>
<td>Linear $(n,h_{\mathrm{in}},h_{\mathrm{out}})$</td>
<td>$(64,256,512)$</td></tr>
<tr>
<td>Conv2D $(H,c_{\mathrm{in}},c_{\mathrm{out}},k)$</td>
<td>$(8,32,8,3)$</td></tr>
<tr>
<td>DepthwiseConv $(H,c,k)$</td>
<td>$(16,32,5)$</td></tr>
<tr>
<td colspan="2"><em>Level 2</em></td></tr>
<tr>
<td>FFN, SwiGLU $(n,h,h_{i})$</td>
<td>$(64,256,1024)$</td></tr>
<tr>
<td>RoPE $(s,d_{\mathrm{head}})$</td>
<td>$(256,128)$</td></tr>
<tr>
<td>FlashAttention $(s,h)$</td>
<td>$(128,256)$</td></tr>
<tr>
<td>GQA $(s,h,n_{h},n_{kv})$</td>
<td>$(128,512,8,2)$</td></tr>
</table>



### A.3 Experiment Configuration

Table 10: Configuration of the verification runs, per model. Every run passes --verify-only, so the kernel is the registry’s pinned program for its task and language and the generation flags are unused. “default” means the flag is not passed. Round caps and spending ceilings are per verification of one kernel; the ceilings are in dollars because they are what the runs were stopped at, and their derivation from the preliminary runs is in [Table 7](https://arxiv.org/html/2611.02028v1#A1.T7). The last column says what each flag controls; the settings shared by every run are listed below the per-model rows.



<table>
<tr>
<td>
<span>
<span><span>Flag</span></span>
</span></td>
<td>
<span>
<span><span>DeepSeek V4.1 Flash</span></span>
</span></td>
<td>
<span>
<span><span>GPT-6 Luna</span></span>
</span></td>
<td>
<span>
<span><span>GLM 5.3 Flash</span></span>
</span></td>
<td>
<span>
<span><span>GLM 5.3</span></span>
</span></td>
<td>
<span>
<span><span>Opus 5.5</span></span>
</span></td>
<td>
<span>
<span><span>What it controls</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--model</span></span>
</span></td>
<td>
<span>
<span><span>deepseek-v4.1-flash-1017</span></span>
</span></td>
<td>
<span>
<span><span>gpt-6-luna</span></span>
</span></td>
<td>
<span>
<span><span>glm-5.3-flash</span></span>
</span></td>
<td>
<span>
<span><span>glm-5.3</span></span>
</span></td>
<td>
<span>
<span><span>opus-5.5</span></span>
</span></td>
<td>
<span>
<span><span>Model of the verification agents</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--isa</span></span>
</span></td>
<td>
<span>
<span>PTX, CUDA, HIP</span>
</span></td>
<td>
<span>
<span>PTX, CUDA, HIP</span>
</span></td>
<td>
<span>
<span>PTX, CUDA, HIP</span>
</span></td>
<td>
<span>
<span>L1: all three; L2: PTX; library: CUDA</span>
</span></td>
<td>
<span>
<span>L2: PTX; library: CUDA, PTX</span>
</span></td>
<td>
<span>
<span><span>Target language of the run</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--kernel</span></span>
</span></td>
<td>
<span>
<span>L1</span>
</span></td>
<td>
<span>
<span>L1</span>
</span></td>
<td>
<span>
<span>L1</span>
</span></td>
<td>
<span>
<span>L1, L2, production libraries</span>
</span></td>
<td>
<span>
<span>L2, production libraries</span>
</span></td>
<td>
<span>
<span><span>Task verified (<a href="https://arxiv.org/html/2611.02028v1#A1.T6"><span>Table</span> <span>6</span></a>)</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--backward-effort</span></span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span>default</span>
</span></td>
<td>
<span>
<span>default</span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span><span>Reasoning effort of every verification agent</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--proof-effort</span></span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span>default (<span>max</span>)</span>
</span></td>
<td>
<span>
<span>default (<span>max</span>)</span>
</span></td>
<td>
<span>
<span><span>xhigh</span></span>
</span></td>
<td>
<span>
<span><span>Effort of the two proof agents</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--other-effort</span></span>
</span></td>
<td>
<span>
<span>default (<span>xhigh</span>)</span>
</span></td>
<td>
<span>
<span>default (<span>xhigh</span>)</span>
</span></td>
<td>
<span>
<span>default (<span>high</span>)</span>
</span></td>
<td>
<span>
<span>default (<span>high</span>)</span>
</span></td>
<td>
<span>
<span>default (<span>xhigh</span>)</span>
</span></td>
<td>
<span>
<span><span>Effort of the statement, review, and diagnosis agents</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--max-auto-tries</span></span>
</span></td>
<td>
<span>
<span>16</span>
</span></td>
<td>
<span>
<span>16</span>
</span></td>
<td>
<span>
<span>16</span>
</span></td>
<td>
<span>
<span>8 (L1, library); 16 (L2)</span>
</span></td>
<td>
<span>
<span>10</span>
</span></td>
<td>
<span>
<span><span>Round cap per proof</span></span>
</span></td></tr>
<tr>
<td>
<span>
<span><span>--max-usd</span></span>
</span></td>
<td>
<span>
<span>50</span>
</span></td>
<td>
<span>
<span>50</span>
</span></td>
<td>
<span>
<span>75</span>
</span></td>
<td>
<span>
<span>100</span>
</span></td>
<td>
<span>
<span>100 or 200 by task</span>
</span></td>
<td>
<span>
<span><span>Spending ceiling per verification</span></span>
</span></td></tr>
<tr>
<td colspan="7">
<span>
<span><span>Shared by every run.</span>
<span>--ask-on-max-usd</span> on (stop at the ceiling rather than continue);
<span>--repeats</span> 2 (pipeline restarts allowed);
<span>--translation-attempts</span> 10 (attempts per segmentation or statement step);
<span>--proof-attempts</span> 10 (attempts per proof step);
<span>--check-attempts</span> 3 (attempts per review);
<span>--diagnosis-attempts</span> 3 (attempts per failure analysis);
<span>--lean-timeout</span> 150 s (time limit per Lean check);
<span>--max-lean-tool-calls</span> 40 (Lean checks per agent turn);
<span>--numeric-trials</span> 6 and <span>--numeric-probes</span> 4 (random inputs of the numeric screen);
all static translator checks on;
<span>--racefree-fallback</span> on (the agent proves race freedom where no lemma is generated);
<span>--flat-model</span> off (direct proof without segmentation, ablation only).</span>
</span></td></tr>
</table>



Experiment configurations are detailed in [Table 10](https://arxiv.org/html/2611.02028v1#A1.T10).

### A.4 Additional Results

Unless a subsection says otherwise, the analyses below are over the PTX and CUDA runs; the HIP runs enter only the per-task table and the latency measurements.

#### A.4.1 Per-Task Results on Level 1

[Table 11](https://arxiv.org/html/2611.02028v1#A1.T11) gives, for every model, language, and task, the outcome and the input and output tokens of the verification. A run ends “pass” with a certificate, “budget” at its spending ceiling, and “fail” when the agent gave up or the round cap was hit.

Table 11: Verification of the twelve KernelCraft GPU Level 1 tasks, per model and target language. A run ends “pass” with a certificate, “budget” at its spending ceiling, or “fail” at its round cap or by an agent failure. Tokens are input and output tokens of the verification, in millions; the “Total” column gives the pass rate and the average tokens over the proven kernels. A count in parentheses belongs to a run that did not prove its kernel, reflects the budget it was given rather than the cost of a proof, and is left out of the average.



<table>
<tr>
<td>Model</td>
<td>Target Lang.</td>
<td>Metric</td>
<td>ReLU</td>
<td>SiLU</td>
<td>GELU</td>
<td>Softmax</td>
<td>LayerNorm</td>
<td>RMSNorm</td>
<td>GEMV</td>
<td>GEMM</td>
<td>Linear</td>
<td>BatchMatmul</td>
<td>Conv2D</td>
<td>DepthwiseConv</td>
<td><span>Total</span></td></tr>
<tr>
<td rowspan="9"><span>GLM 5.3</span></td>
<td rowspan="3"><span>CUDA</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>pass</td>
<td>budget</td>
<td>75%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>44.67</td>
<td>61.61</td>
<td>87.60</td>
<td>332.71</td>
<td>291.45</td>
<td>481.23</td>
<td>(700.51)</td>
<td>389.51</td>
<td>430.82</td>
<td>(549.06)</td>
<td>404.79</td>
<td>(562.16)</td>
<td>280.49</td></tr>
<tr>
<td>output tokens (M)</td>
<td>1.78</td>
<td>1.78</td>
<td>2.38</td>
<td>7.97</td>
<td>6.32</td>
<td>5.93</td>
<td>(15.60)</td>
<td>8.95</td>
<td>11.25</td>
<td>(13.53)</td>
<td>9.23</td>
<td>(13.47)</td>
<td>6.18</td></tr>
<tr>
<td rowspan="3"><span>PTX</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>pass</td>
<td>budget</td>
<td>83%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>119.20</td>
<td>77.30</td>
<td>56.50</td>
<td>391.20</td>
<td>483.20</td>
<td>288.60</td>
<td>481.00</td>
<td>277.70</td>
<td>508.70</td>
<td>(752.70)</td>
<td>461.90</td>
<td>(616.80)</td>
<td>314.53</td></tr>
<tr>
<td>output tokens (M)</td>
<td>2.66</td>
<td>2.77</td>
<td>1.91</td>
<td>10.63</td>
<td>11.50</td>
<td>6.68</td>
<td>13.79</td>
<td>8.09</td>
<td>10.54</td>
<td>(15.33)</td>
<td>11.94</td>
<td>(15.93)</td>
<td>8.05</td></tr>
<tr>
<td rowspan="3"><span>HIP</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>83%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>158.84</td>
<td>83.79</td>
<td>99.16</td>
<td>229.01</td>
<td>(692.63)</td>
<td>231.08</td>
<td>576.31</td>
<td>422.98</td>
<td>414.93</td>
<td>455.50</td>
<td>450.52</td>
<td>(581.50)</td>
<td>312.21</td></tr>
<tr>
<td>output tokens (M)</td>
<td>4.33</td>
<td>2.53</td>
<td>4.62</td>
<td>6.45</td>
<td>(17.44)</td>
<td>7.93</td>
<td>12.31</td>
<td>11.22</td>
<td>9.54</td>
<td>9.93</td>
<td>2.66</td>
<td>(15.53)</td>
<td>7.15</td></tr>
<tr>
<td rowspan="9"><span>GLM 5.3 Flash</span></td>
<td rowspan="3"><span>CUDA</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>fail</td>
<td>50%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>221.00</td>
<td>281.60</td>
<td>248.70</td>
<td>669.40</td>
<td>457.40</td>
<td>1881.10</td>
<td>(1981.50)</td>
<td>(2052.50)</td>
<td>(1901.70)</td>
<td>(2036.30)</td>
<td>(1811.90)</td>
<td>(1366.40)</td>
<td>626.53</td></tr>
<tr>
<td>output tokens (M)</td>
<td>3.13</td>
<td>5.10</td>
<td>4.74</td>
<td>10.09</td>
<td>6.99</td>
<td>24.48</td>
<td>(24.19)</td>
<td>(25.26)</td>
<td>(23.66)</td>
<td>(27.72)</td>
<td>(23.01)</td>
<td>(20.11)</td>
<td>9.09</td></tr>
<tr>
<td rowspan="3"><span>PTX</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>budget</td>
<td>fail</td>
<td>budget</td>
<td>58%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>184.70</td>
<td>260.80</td>
<td>231.10</td>
<td>792.90</td>
<td>1455.90</td>
<td>1219.50</td>
<td>857.30</td>
<td>(1408.60)</td>
<td>(1486.40)</td>
<td>(1974.40)</td>
<td>(1531.10)</td>
<td>(1891.60)</td>
<td>714.60</td></tr>
<tr>
<td>output tokens (M)</td>
<td>3.14</td>
<td>4.05</td>
<td>3.51</td>
<td>11.91</td>
<td>20.94</td>
<td>16.38</td>
<td>13.85</td>
<td>(15.64)</td>
<td>(21.43)</td>
<td>(30.21)</td>
<td>(25.58)</td>
<td>(24.16)</td>
<td>10.54</td></tr>
<tr>
<td rowspan="3"><span>HIP</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>budget</td>
<td>50%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>240.88</td>
<td>271.10</td>
<td>292.31</td>
<td>841.58</td>
<td>990.15</td>
<td>2007.71</td>
<td>(1019.44)</td>
<td>(1638.08)</td>
<td>(1880.69)</td>
<td>(1783.35)</td>
<td>(1477.32)</td>
<td>(1871.22)</td>
<td>773.96</td></tr>
<tr>
<td>output tokens (M)</td>
<td>3.41</td>
<td>4.55</td>
<td>4.08</td>
<td>12.04</td>
<td>14.31</td>
<td>25.13</td>
<td>(15.53)</td>
<td>(15.42)</td>
<td>(20.43)</td>
<td>(29.31)</td>
<td>(23.78)</td>
<td>(19.54)</td>
<td>10.59</td></tr>
<tr>
<td rowspan="9"><span>DeepSeek V4.1 Flash</span></td>
<td rowspan="3"><span>CUDA</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>95.39</td>
<td>66.84</td>
<td>79.12</td>
<td>(458.90)</td>
<td>(780.60)</td>
<td>(711.59)</td>
<td>(528.54)</td>
<td>(706.41)</td>
<td>(744.72)</td>
<td>(550.34)</td>
<td>(403.48)</td>
<td>(574.90)</td>
<td>80.45</td></tr>
<tr>
<td>output tokens (M)</td>
<td>1.19</td>
<td>0.96</td>
<td>1.20</td>
<td>(7.63)</td>
<td>(10.80)</td>
<td>(9.62)</td>
<td>(6.92)</td>
<td>(10.27)</td>
<td>(11.85)</td>
<td>(8.65)</td>
<td>(6.11)</td>
<td>(8.16)</td>
<td>1.12</td></tr>
<tr>
<td rowspan="3"><span>PTX</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>224.58</td>
<td>101.24</td>
<td>119.58</td>
<td>(368.11)</td>
<td>(197.33)</td>
<td>(383.49)</td>
<td>(367.04)</td>
<td>(653.25)</td>
<td>(485.78)</td>
<td>(215.81)</td>
<td>(456.24)</td>
<td>(430.42)</td>
<td>148.47</td></tr>
<tr>
<td>output tokens (M)</td>
<td>2.60</td>
<td>1.54</td>
<td>1.64</td>
<td>(4.46)</td>
<td>(3.50)</td>
<td>(5.06)</td>
<td>(5.46)</td>
<td>(9.09)</td>
<td>(7.30)</td>
<td>(3.05)</td>
<td>(6.85)</td>
<td>(6.21)</td>
<td>1.93</td></tr>
<tr>
<td rowspan="3"><span>HIP</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>241.67</td>
<td>133.71</td>
<td>93.07</td>
<td>(471.13)</td>
<td>(831.19)</td>
<td>(741.90)</td>
<td>(462.83)</td>
<td>(688.31)</td>
<td>(691.33)</td>
<td>(390.18)</td>
<td>(501.33)</td>
<td>(812.59)</td>
<td>156.15</td></tr>
<tr>
<td>output tokens (M)</td>
<td>1.98</td>
<td>2.01</td>
<td>1.59</td>
<td>(6.32)</td>
<td>(10.91)</td>
<td>(8.91)</td>
<td>(5.53)</td>
<td>(10.10)</td>
<td>(8.82)</td>
<td>(2.98)</td>
<td>(7.03)</td>
<td>(9.03)</td>
<td>1.86</td></tr>
<tr>
<td rowspan="9"><span>GPT-6 Luna</span></td>
<td rowspan="3"><span>CUDA</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>690.80</td>
<td>316.40</td>
<td>421.40</td>
<td>(638.30)</td>
<td>(707.80)</td>
<td>(1243.20)</td>
<td>(511.80)</td>
<td>(1171.60)</td>
<td>(596.90)</td>
<td>(660.10)</td>
<td>(689.10)</td>
<td>(1152.50)</td>
<td>476.20</td></tr>
<tr>
<td>output tokens (M)</td>
<td>4.91</td>
<td>2.40</td>
<td>2.95</td>
<td>(4.38)</td>
<td>(5.30)</td>
<td>(8.50)</td>
<td>(4.27)</td>
<td>(9.01)</td>
<td>(4.17)</td>
<td>(4.90)</td>
<td>(5.12)</td>
<td>(7.89)</td>
<td>3.42</td></tr>
<tr>
<td rowspan="3"><span>PTX</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>366.30</td>
<td>401.70</td>
<td>569.80</td>
<td>(1144.90)</td>
<td>(1281.40)</td>
<td>(1494.60)</td>
<td>(547.50)</td>
<td>(1259.00)</td>
<td>(1146.20)</td>
<td>(1052.00)</td>
<td>(932.00)</td>
<td>(880.90)</td>
<td>445.93</td></tr>
<tr>
<td>output tokens (M)</td>
<td>2.30</td>
<td>3.16</td>
<td>4.31</td>
<td>(7.77)</td>
<td>(8.54)</td>
<td>(9.66)</td>
<td>(4.08)</td>
<td>(8.60)</td>
<td>(7.94)</td>
<td>(7.48)</td>
<td>(7.30)</td>
<td>(5.78)</td>
<td>3.26</td></tr>
<tr>
<td rowspan="3"><span>HIP</span></td>
<td>pass/fail/budget</td>
<td>pass</td>
<td>pass</td>
<td>pass</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>fail</td>
<td>25%</td></tr>
<tr>
<td>input tokens (M)</td>
<td>345.00</td>
<td>286.00</td>
<td>236.50</td>
<td>(780.39)</td>
<td>(798.14)</td>
<td>(1339.64)</td>
<td>(580.62)</td>
<td>(1150.13)</td>
<td>(513.32)</td>
<td>(710.28)</td>
<td>(842.19)</td>
<td>(1451.10)</td>
<td>289.17</td></tr>
<tr>
<td>output tokens (M)</td>
<td>2.13</td>
<td>2.06</td>
<td>1.48</td>
<td>(4.71)</td>
<td>(7.71)</td>
<td>(9.05)</td>
<td>(4.31)</td>
<td>(8.33)</td>
<td>(6.95)</td>
<td>(6.93)</td>
<td>(6.20)</td>
<td>(8.68)</td>
<td>1.89</td></tr>
</table>



#### A.4.2 Refusal Census

Over the 116 PTX and CUDA verifications of the three subsets, no correct kernel was refused by scope: every kernel that passed the numeric screen compiled into the model and reached the proof loop. Outcomes on Level 1, per model (proved / budget stop / agent failure out of 24): GLM 5.3 19 / 5 / 0; GLM 5.3 Flash 13 / 7 / 4; GPT-6 Luna 6 / 0 / 18; DeepSeek V4.1 Flash 6 / 0 / 18. On Level 2 (PTX): Opus 5.5 5 / 0 / 0; GLM 5.3 1 / 3 / 1. Library: 10 / 0 / 0.

#### A.4.3 Cost per Stage

Each verification splits into a static part (the translator’s compile and its Lean modules, the certificate), the agent steps (segmentation, statements, the two proofs, reviews, and the diagnosis), and Lean checks inside the proof loop. [Table 12](https://arxiv.org/html/2611.02028v1#A1.T12) gives the tokens per stage. On Level 1 the two equivalence proofs take 95 to 99% of a run’s tokens (median per model), the diagnosis 0.7 to 4.4%, and the segmentation, statement, and review steps at most 0.3% each; the segmentation proof dominates for the three weaker models, which rarely reach the equivalence proof, and the equivalence proof dominates for GLM 5.3 and on Level 2. The static part takes 36 to 157 seconds per run (maximum 352); Lean inside the loop 18 to 60 minutes per run for the small models and 6 to 8 for Opus 5.5.

Table 12: Tokens per stage of a verification (millions), per model and evaluation set: the median over the runs that reached the stage and, in parentheses, the median share of the run’s tokens; “$k$ of $N$” where not every run reached the stage. “Other steps” is the largest median share among segmentation, statement, and review steps. The static compile and the Lean checks inside the proof loop spend time, not tokens (text). Rows cover the same runs as [Table 13](https://arxiv.org/html/2611.02028v1#A1.T13); the rejected llm.c run (GLM 5.3) stopped in the race-freedom fallback, so it reached no proof stage.



<table>
<tr>
<td><span>Set</span></td>
<td><span>Model</span></td>
<td><span>Segmentation proof</span> $\mathrm{SEG}(K)$</td>
<td><span>Equivalence proof</span> $\mathrm{EQ}(K,S)$</td>
<td><span>Diagnosis</span></td>
<td><span>Race-freedom fallback</span></td>
<td><span>Other steps</span></td></tr>
<tr>
<td rowspan="4"><span>Level 1 (PTX, CUDA)</span></td>
<td>DeepSeek V4.1 Flash</td>
<td>271.21 (68.7%)</td>
<td>164.53 (36.2%; 15 of 24)</td>
<td>14.85 (3.6%)</td>
<td>—</td>
<td>$\leq 0.2\%$</td></tr>
<tr>
<td>GPT-6 Luna</td>
<td>646.92 (93.6%)</td>
<td>267.78 (38.5%; 9 of 24)</td>
<td>35.95 (4.4%)</td>
<td>—</td>
<td>$\leq 0.2\%$</td></tr>
<tr>
<td>GLM 5.3 Flash</td>
<td>719.16 (55.1%)</td>
<td>454.39 (48.3%; 21 of 24)</td>
<td>10.34 (0.7%)</td>
<td>—</td>
<td>$\leq 0.1\%$</td></tr>
<tr>
<td>GLM 5.3</td>
<td>116.39 (42.8%)</td>
<td>186.60 (54.8%)</td>
<td>4.12 (1.3%)</td>
<td>—</td>
<td>$\leq 0.3\%$</td></tr>
<tr>
<td rowspan="2"><span>Level 2 (PTX)</span></td>
<td>GLM 5.3</td>
<td>103.96 (29.6%)</td>
<td>388.21 (72.5%; 4 of 5)</td>
<td>3.29 (1.0%)</td>
<td>—</td>
<td>$\leq 0.2\%$</td></tr>
<tr>
<td>Opus 5.5</td>
<td>16.26 (27.2%)</td>
<td>60.76 (69.4%)</td>
<td>2.71 (1.4%; 2 of 5)</td>
<td>—</td>
<td>$\leq 1.0\%$</td></tr>
<tr>
<td rowspan="2"><span>Production</span></td>
<td>GLM 5.3</td>
<td>106.04 (47.7%; 4 of 5)</td>
<td>137.63 (49.8%; 4 of 5)</td>
<td>3.48 (1.3%; 4 of 5)</td>
<td>10.30 (100%; 1 of 5)</td>
<td>$\leq 0.8\%$</td></tr>
<tr>
<td>Opus 5.5</td>
<td>21.27 (25.6%)</td>
<td>56.03 (53.5%)</td>
<td>0.76 (0.7%; 4 of 5)</td>
<td>48.81 (23.2%; 3 of 5)</td>
<td>$\leq 1.2\%$</td></tr>
</table>



#### A.4.4 Scaling with Kernel Size

Over the 116 PTX and CUDA verifications, the Spearman rank correlation between the compiled program’s length in program counters and the run’s cost in dollars is 0.61; within a model, between length and tokens, it is 0.43 for DeepSeek V4.1 Flash, 0.55 for GPT-6 Luna, 0.64 for GLM 5.3 Flash and Opus 5.5, and 0.70 for GLM 5.3. [Figure 4](https://arxiv.org/html/2611.02028v1#A1.F4) plots the proven runs; for the three models with at least ten of them, a least-squares fit in log coordinates has slope 1.03 (GLM 5.3), 0.90 (GLM 5.3 Flash), and 0.99 (Opus 5.5), so within a model the tokens grow about linearly with the program’s length, and the lines differ by the model’s token appetite, not by the kernel. Between length and proof turns it is 0.17, rounds 0.20, and between the number of segments and cost 0.17. Proof cost grows with per-block depth, not with the grid.

Figure 4: Tokens of each proven verification against the compiled program’s length, over the PTX and CUDA runs, one colour per model (the number of runs in parentheses). The dashed lines are least-squares fits in log coordinates for the models with at least ten proven runs, a guide to the eye, not a fitted law; DeepSeek V4.1 Flash and GPT-6 Luna prove only the elementwise kernels and get none.

#### A.4.5 Why Every Model Fails on BatchMatmul and DepthwiseConv

Read from the 16 failing Level 1 runs on PTX and CUDA (four models, two kernels): no statement is false and no verdict blames the kernel; every analyst verdict but two is “theorem true, proof hard” or “proof loop stuck”. Every model stops on the same proof. Both kernels pin a 128-thread block whose warps run long chains of tensor-core operations, each once per warp: BatchMatmul’s 16-trip $K$-loop of ldmatrix and mma makes one warp’s run of a block phase up to 97 sub-phases, and DepthwiseConv runs 26 collective operations back to back. The generated scaffolding stops at the lane: the per-lane bridges and the trip lemmas are given, but composing them into the warp-level fold across the loop and the collective steps is left to the agent, and that is where every stall is. The two weaker models never close the first equivalence; the two GLM models close it and run out of budget on the second, whose proof must carry the whole dataflow through the staged shared buffers and the fragment-layout arithmetic. Two runs ended on a genuine bug in GPT-6 Luna’s own segmentation, caught by the first equivalence as designed. What would move these kernels is a generated warp-level bridge per block phase, so that the part every model fails on becomes generated, checked text. On HIP the pinned BatchMatmul kernel runs one 64-lane wavefront per block, so the warp-level composition that stops every model here is absent, and GLM 5.3 proves it; DepthwiseConv fails on HIP as on the other two languages.

#### A.4.6 Model Size and Tokens

Taking the active parameter count as a proxy for capability, [Figure 5](https://arxiv.org/html/2611.02028v1#A1.F5) plots, for the three small models with published sizes, the Level 1 pass rate, the input tokens per proved kernel, and the input tokens per kernel against active parameters, over PTX and CUDA. The pass rate rises with model size, and the larger model is also the cheaper one per proof: the pass rate goes from 25% to 54% to 79%, and a model that can close a proof needs fewer turns to do it, while a model that cannot spends its rounds and stops. Input tokens per proved kernel: GLM 5.3 (40B active) 466M, DeepSeek V4.1 Flash (16B) 1,617M, GLM 5.3 Flash (18B) 2,170M, and GPT-6 Luna 3,313M; mean input tokens per kernel, proved or not: 369M, 404M, 1,175M, and 828M. The largest model is the most efficient per proof, by 3.5 and 4.7 times; DeepSeek and GLM 5.3 spend about the same per kernel, but GLM 5.3 turns it into 19 certificates against 6, while GLM 5.3 Flash fails expensively, running 7 of its 11 failures to the ceiling. Output tokens are 0.7 to 2.4% of input for every model, so the input count carries the comparison; the dialogue re-sent each turn, mostly as cache reads, is the cost, not what the model writes. In dollars the order of DeepSeek and GLM 5.3 per proof is reversed ($39 against $73), only because DeepSeek’s tokens are priced lowest. On Level 2 the gap is wider: Opus 5.5 proves every kernel at 107M input tokens per kernel, a quarter of the 374M GLM 5.3 spends per kernel ([Table 3](https://arxiv.org/html/2611.02028v1#S5.T3)).

(a) Pass rate.

(b) Input and output tokens per proved kernel.

(c) Mean input and output tokens per kernel, proved or not.

Figure 5: Level 1 pass rate and tokens against the number of active parameters of the three small models with published sizes, over PTX and CUDA.

#### A.4.7 The Race-Freedom Fallback

The translator’s generated race-freedom lemmas proved on all 40 Level 1 and Level 2 verifications on PTX and CUDA; the agent fallback of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) never ran there. On the production kernels it ran on five of ten, where the translator could not name a footprint: vllm_silu_and_mul (8 turns), llmc_softmax_forward (10 turns), fam_flash_attn (2), vllm_fused_qknorm_rope (7, an output address that divides by the block size), and vllm_awq_gemm (6); four were proved, and on llmc_softmax_forward the fallback returned a located race instead of a proof (Appendix [A.14](https://arxiv.org/html/2611.02028v1#A1.SS14)). A later run of the same kernel text reuses the stored proof at no cost.

#### A.4.8 The Flat-Model Ablation

Three kernels were proved directly against the flat model (no segmentation, no segmentation proof) and compared with their default runs; tokens are input / output in millions. FlashAttention, Opus 5.5, PTX: flat 222.5 / 3.06, of which the direct proof 219.6 / 2.99 in three rounds of 20 turns, against 172.5 / 2.77 for the default run, of which the equivalence proof 119.6 / 2.05 in two rounds of 11 turns and the segmentation with its proof 49.1 / 0.64. The direct proof used 84% more input tokens than the equivalence proof and the whole run 29% more than the default run; it also crossed the $140 ceiling the run started with and finished only after it was raised to the Level 2 ceiling of $200, under which the default run had finished. On the two short kernels, ReLU and GEMM with GLM 5.3 on PTX, the direct proof was cheaper than the default run. Caveats: one hard kernel; the model differs between GEMM and FlashAttention; the default runs ran on an earlier code version than the flat runs.

#### A.4.9 The Failure Analysis

[Table 13](https://arxiv.org/html/2611.02028v1#A1.T13) gives the cost and the verdict statistics of the analysis per model; [Figure 6](https://arxiv.org/html/2611.02028v1#A1.F6) the distribution of verdicts by language, model, level, and library.

Table 13: The failure analysis inside the verification runs, per model: the Level 1 runs on PTX and CUDA, the Level 2 runs, and the production kernels. “Analysis” is every analyst, falsity-probe, and second-opinion call. Share: the analysis’s input tokens over the run’s, summed over a model’s runs and as the per-run median. Verdict stops: runs an analyst verdict ended before their spending ceiling. False positives: kernel-bug or race verdicts on a correct kernel, those refuted by the second analyst over those issued; every one was refuted, so none was acted on (two were issued before the second analyst existed and are counted as reviewed and refuted). Refuted: reviews in which the second analyst disputed the verdict, out of those it gave; a disputed verdict is not acted on.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Calls  analyst / probe / 2nd op. | Share of cell input  aggregate / median | Verdict  stops | False positives  refuted / issued | 2nd op.  refuted |
| DeepSeek V4.1 Flash | 610 / 755 / 3 | 3.7% / 3.5% | 1 | 1 / 1 | 3 / 3 |
| GPT-6 Luna | 599 / 815 / 5 | 4.7% / 4.3% | 6 | 1 / 1 | 2 / 5 |
| GLM 5.3 Flash | 336 / 303 / 8 | 0.7% / 0.6% | 1 | 1 / 1 | 6 / 8 |
| GLM 5.3 | 154 / 176 / 4 | 1.1% / 1.1% | 2 | 0 / 0 | 0 / 4 |
| Opus 5.5 | 3 / 10 / 0 | 0.8% / 0.5% | 0 | 0 / 0 | 0 / 0 |


(a) Level 1, by target language.

(b) Level 1, by model.

(c) Level 2, by model.

(d) Production kernels.

Figure 6: Verdicts of the failure analysis, as shares of the verdicts given.

#### A.4.10 Latency

[Table 14](https://arxiv.org/html/2611.02028v1#A1.T14) and [Figure 7](https://arxiv.org/html/2611.02028v1#A1.F7) give the latency of the verified evaluation kernels against torch eager, torch.compile, and the fastest vendor library, on an H100 for PTX and CUDA and on an MI300X for HIP; every ratio is against baselines measured on the same device, so the targets are not compared with each other. The Level 2 kernels were verified on PTX only, so PTX covers 17 kernels and CUDA and HIP the twelve Level 1 kernels. The verified kernels beat torch eager on every kernel (geometric mean 5.5$\times$ on PTX, 4.2$\times$ on CUDA, 4.1$\times$ on HIP); torch.compile with max-autotune on 14 of 17 on PTX (2.6$\times$), 9 of 12 on CUDA (1.9$\times$), and 10 of 12 on HIP (2.1$\times$); and the fastest vendor library on 11 of the 14 PTX kernels that have one (1.46$\times$), 6 of 9 on CUDA (1.37$\times$), and 7 of 10 on HIP (1.22$\times$). The PTX kernels are faster than the CUDA ones on 6 of the 12 Level 1 kernels, by the most on GEMV (1.8$\times$), BatchMatmul (1.8$\times$), and DepthwiseConv (1.4$\times$); the other six are within 5%. The losses are the launch-bound elementwise kernels (2 to 5% behind torch.compile on the H100); rmsnorm against FlashInfer (1.3$\times$) and, on HIP, against torch.compile (1.8$\times$) and AITER (1.5$\times$); batchmatmul against the vendor batched GEMM (cuBLAS, built on wgmma and TMA, which are outside LARK’s scope, 1.3$\times$ on PTX and 2.3$\times$ on CUDA; hipBLASLt 2.0$\times$); SwiGLU on PTX against vLLM’s fused MLP (1.05$\times$); GEMV on CUDA and HIP against torch.compile (1.9$\times$, 2.0$\times$), where the PTX version wins; and LayerNorm on HIP against AITER (1.2$\times$).

Table 14: Latency of the verified evaluation kernels at their pinned problem sizes (median of 7 rounds): PTX and CUDA on an H100 80GB SXM, HIP on an MI300X. The Level 2 kernels were verified on PTX only. “Ours” is the kernel’s time in microseconds; a ratio is the baseline’s time over ours on the same device, so a value above 1 means the verified kernel is faster, and the targets are not compared with each other. Baselines: torch eager (the task’s Torch module), torch.compile with max-autotune, and the fastest vendor library that offers the operation (cuBLAS, cuDNN, vLLM, FlashInfer, and the torch attention backends on the H100; hipBLASLt, rocBLAS, MIOpen, and AITER on the MI300X; a dash where none does).



<table>
<tr>
<td></td>
<td colspan="4"><span>PTX (H100)</span></td>
<td colspan="4"><span>CUDA (H100)</span></td>
<td colspan="4"><span>HIP (MI300X)</span></td></tr>
<tr>
<td><span>Kernel</span></td>
<td>Ours ($\mu$s)</td>
<td>vs eager</td>
<td>vs compile</td>
<td>vs vendor</td>
<td>Ours ($\mu$s)</td>
<td>vs eager</td>
<td>vs compile</td>
<td>vs vendor</td>
<td>Ours ($\mu$s)</td>
<td>vs eager</td>
<td>vs compile</td>
<td>vs vendor</td></tr>
<tr>
<td><span>relu</span></td>
<td>2.05</td>
<td>3.53</td>
<td>0.96</td>
<td>—</td>
<td>2.01</td>
<td>3.60</td>
<td>0.98</td>
<td>—</td>
<td>1.73</td>
<td>4.35</td>
<td>1.07</td>
<td>—</td></tr>
<tr>
<td><span>silu</span></td>
<td>2.08</td>
<td>3.55</td>
<td>0.97</td>
<td>—</td>
<td>2.06</td>
<td>3.58</td>
<td>0.98</td>
<td>—</td>
<td>1.75</td>
<td>3.81</td>
<td>1.10</td>
<td>—</td></tr>
<tr>
<td><span>gelu</span></td>
<td>1.99</td>
<td>3.52</td>
<td>1.02</td>
<td>2.38</td>
<td>1.98</td>
<td>3.54</td>
<td>1.03</td>
<td>2.39</td>
<td>1.77</td>
<td>4.17</td>
<td>3.02</td>
<td>1.72</td></tr>
<tr>
<td><span>softmax</span></td>
<td>4.12</td>
<td>2.93</td>
<td>3.06</td>
<td>3.00</td>
<td>4.37</td>
<td>2.76</td>
<td>2.88</td>
<td>2.83</td>
<td>5.25</td>
<td>2.23</td>
<td>2.19</td>
<td>1.32</td></tr>
<tr>
<td><span>layernorm</span></td>
<td>5.03</td>
<td>6.63</td>
<td>1.78</td>
<td>—</td>
<td>4.78</td>
<td>6.98</td>
<td>1.87</td>
<td>—</td>
<td>5.05</td>
<td>3.79</td>
<td>1.69</td>
<td>0.85</td></tr>
<tr>
<td><span>rmsnorm</span></td>
<td>4.38</td>
<td>6.12</td>
<td>1.73</td>
<td>0.75</td>
<td>4.29</td>
<td>6.26</td>
<td>1.77</td>
<td>0.77</td>
<td>4.02</td>
<td>5.61</td>
<td>0.54</td>
<td>0.67</td></tr>
<tr>
<td><span>gemv</span></td>
<td>3.05</td>
<td>4.23</td>
<td>0.95</td>
<td>1.53</td>
<td>5.54</td>
<td>2.33</td>
<td>0.53</td>
<td>0.84</td>
<td>3.83</td>
<td>3.09</td>
<td>0.49</td>
<td>1.31</td></tr>
<tr>
<td><span>gemm</span></td>
<td>2.90</td>
<td>5.79</td>
<td>2.03</td>
<td>1.18</td>
<td>3.39</td>
<td>4.95</td>
<td>1.74</td>
<td>1.01</td>
<td>2.14</td>
<td>5.05</td>
<td>6.18</td>
<td>1.27</td></tr>
<tr>
<td><span>linear</span></td>
<td>2.88</td>
<td>7.40</td>
<td>5.93</td>
<td>1.26</td>
<td>3.62</td>
<td>5.88</td>
<td>4.71</td>
<td>1.00</td>
<td>3.36</td>
<td>3.74</td>
<td>4.34</td>
<td>1.02</td></tr>
<tr>
<td><span>batchmatmul</span></td>
<td>7.24</td>
<td>5.66</td>
<td>3.96</td>
<td>0.77</td>
<td>12.86</td>
<td>3.19</td>
<td>2.23</td>
<td>0.43</td>
<td>11.13</td>
<td>2.90</td>
<td>4.08</td>
<td>0.51</td></tr>
<tr>
<td><span>conv2d</span></td>
<td>3.28</td>
<td>8.41</td>
<td>5.96</td>
<td>4.95</td>
<td>3.23</td>
<td>8.56</td>
<td>6.07</td>
<td>5.04</td>
<td>2.65</td>
<td>9.55</td>
<td>4.54</td>
<td>3.16</td></tr>
<tr>
<td><span>depthwise_conv</span></td>
<td>3.36</td>
<td>4.06</td>
<td>6.12</td>
<td>2.38</td>
<td>4.65</td>
<td>2.94</td>
<td>4.43</td>
<td>1.72</td>
<td>4.13</td>
<td>3.83</td>
<td>5.29</td>
<td>2.00</td></tr>
<tr>
<td><span>rope</span></td>
<td>2.24</td>
<td>20.12</td>
<td>1.10</td>
<td>1.16</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td></tr>
<tr>
<td><span>ffn</span></td>
<td>9.75</td>
<td>4.96</td>
<td>4.53</td>
<td>1.09</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td></tr>
<tr>
<td><span>swiglu</span></td>
<td>11.03</td>
<td>4.99</td>
<td>6.11</td>
<td>0.95</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td></tr>
<tr>
<td><span>flash_attn</span></td>
<td>6.01</td>
<td>5.13</td>
<td>4.77</td>
<td>1.25</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td></tr>
<tr>
<td><span>gqa</span></td>
<td>4.32</td>
<td>8.10</td>
<td>5.57</td>
<td>1.22</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td></tr>
<tr>
<td>Geometric mean</td>
<td></td>
<td>5.50</td>
<td>2.61</td>
<td>1.46</td>
<td></td>
<td>4.20</td>
<td>1.91</td>
<td>1.37</td>
<td></td>
<td>4.06</td>
<td>2.14</td>
<td>1.22</td></tr>
</table>




Figure 7: Latency of the verified evaluation kernels relative to the baselines, from the same measurement as [Table 14](https://arxiv.org/html/2611.02028v1#A1.T14): PTX and CUDA on the H100, HIP on the MI300X. The vendor libraries are cuBLAS and cuBLASLt (GEMM, GEMV, batched GEMM, linear), cuDNN (convolution), FlashInfer (RMSNorm, softmax, RoPE, prefill attention), vLLM (SiLU-and-mul, GELU, FlashAttention-2), and the torch attention backends on the H100, and hipBLASLt, rocBLAS, MIOpen, and AITER on the MI300X; “fastest” is the best of these per kernel. A dagger marks a kernel whose time is within the launch overhead.

#### A.4.11 Mutation Results Analysis

Table 15: Mutation study: how each non-equivalent mutant was rejected. One mutant per rule,
each counted once by its final run; ReLU has no barrier, so R3 does not apply to it.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel | Lang. | Mutantsa | Staticb | Race-freec | Kernel bugd | Premisee | Escapedf |
| ReLU | CUDA | 4 | 1 | 0 | 2 | 1 | 0 |
| ReLU | PTX | 4 | 0 | 1 | 2 | 1 | 0 |
| ReLU | HIP | 4 | 0 | 1 | 3 | 0 | 0 |
| Softmax | CUDA | 5 | 1 | 0 | 4 | 0 | 0 |
| Softmax | PTX | 5 | 1 | 0 | 4 | 0 | 0 |
| Softmax | HIP | 5 | 2 | 2 | 0 | 1 | 0 |
| GEMM | CUDA | 5 | 4 | 1 | 0 | 0 | 0 |
| GEMM | PTX | 5 | 1 | 2 | 2 | 0 | 0 |
| GEMM | HIP | 5 | 1 | 3 | 1 | 0 | 0 |

a*Mutants*: non-equivalent mutants, one per mutation rule (arithmetic, load/store,
barrier, register/variable, loop range). A mutant that computes the same function as the
original is excluded and replaced by the next candidate of its rule. Each row’s five outcome
columns sum to this number.

b*Static*: rejected by the static compiler before any LLM call: the lane-safety check
(every lane of a warp stores to one cell), a race located at compile time, or a race-freedom
pair lemma that Lean refutes with a counterexample.

c*Race-free*: the generated race-freedom proof obligation is judged false, with two
concrete threads and a memory cell; confirmed by an independent second opinion.

d*Kernel bug*: the equivalence proof fails and the failure analyst shows the kernel
computes a different function from the specification; confirmed by an independent second
opinion.

e*Premise*: the equivalence statement, generated from the kernel’s signature and declared
launch, is rejected by Lean, because the mutant needs a premise the standard signature does not
give (e.g. an out-of-bounds access).

f*Escaped*: the mutant was proved equivalent to the specification, i.e. not detected.

[Table 15](https://arxiv.org/html/2611.02028v1#A1.T15) reports final-run outcomes after corrections to specification premises and output extraction: 11 static rejections, 10 race-freedom failures, 18 kernel-bug diagnoses, and 3 premise rejections; no mutant escapes. Static and race-freedom checks account for 12 of 15 GEMM rejections, while equivalence failure analysis accounts for most ReLU and Softmax rejections.

### A.5 Measuring the Per-Operation Semantics against the Hardware

Every lower-layer definition of [Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) is checked against the GPU before it is pinned. For each operation in a language’s table, a harness draws random operands from declared strata (finite values of each width, $-\infty$ where the operation admits it, boundary cases such as zero and powers of two), runs the operation on the GPU in its own operand form, and compares the result with the Lean definition evaluated on the same operands; a disagreement outside the declared idealizations fails the sweep. The sweep makes no LLM calls and is re-run whenever a table changes; its record is pinned with the table, and a run reads the record its table was pinned on, never the newest. The sweep is what turned one wrong definition into a right one: the approximate base-two exponential had been formalized as $e^{x}$ and is $2^{x}$. Shuffles and vector accesses are measured per lane and per alignment in the same way, and the lane layouts of the warp-collective operations are recorded from the hardware before they are pinned.

### A.6 Retargeting LARK to a New Target Language

##### Effort.

CUDA support on top of PTX took at most 39 person-days; its written lines (code, tests, and documentation, generated tables apart) are 68,741, of which 35,314 are code. HIP, ported from CUDA, took 21,495 written lines, of which 15,333 are code, about a third of CUDA’s lines and under half of its code. The shared machine model changed by 885 lines added and 655 removed, including one generalization: the warp width became a parameter, 32 for PTX and CUDA and 64 for HIP, whose wavefront has 64 lanes. HIP’s own translator code is largely a copy of CUDA’s, with 91% of its lines identical. CUDA’s own code is 37 files and 14,062 Python lines and HIP’s 30 files and 7,535 lines, against 221 files and 127,988 lines shared across languages; of the machine model’s Lean text, 96% is shared for HIP, 94% for CUDA, and 86% for PTX.

##### The split in symbols.

What is fixed once splits as $M=M_{\mathrm{launch}}\cup L$. The shared part



$$
M_{\mathrm{launch}}=(\Sigma,\mathrm{run},\mathrm{conv},\mathrm{RF},\mathrm{Pre}_{M},\mathrm{Ext}_{M},\mathcal{C})
$$




describes a launch for every language: the state, the folds of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2), race freedom $\mathrm{RF}$ and its library ([Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)), the premise and extractor definitions $\mathrm{Pre}_{M}$ and $\mathrm{Ext}_{M}$ ([Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)), and the certificate $\mathcal{C}$ ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). The per-language part



$$
L=(\mathcal{I},\mathrm{sem},\mathrm{cut},w_{L},\mathrm{parse}_{L},\mathrm{fmt}_{L},\mathit{layout}_{L},\mathrm{lower}_{L})
$$




is the operation table with its lower-layer definitions, the warp width $w_{L}$ (32 for PTX and CUDA, 64 for HIP), the reader and the read-back formatter, the lane layouts of the warp-collective operations, and, for a structured language, the lowering.

[Table 16](https://arxiv.org/html/2611.02028v1#A1.T16) lists the components of LARK by whether they are shared across target languages or written per language. The split follows one rule: what describes a parallel launch is written once, what describes a language’s operations is written per language. The machine model $M$ abstracts GPU execution in general, so its state, value representation, cut hierarchy, folds, and convergence rule, the definition of race freedom with the library that discharges it, the premise schemas, the certificate structure with its end-to-end theorem, the premise and extractor definitions over a signature, the agent loop, and the report are one Lean text and one harness shared by every language. A language supplies its operation set $\mathcal{I}$, generated from a pinned table whose lower-layer definitions are measured against the hardware (Appendix [A.5](https://arxiv.org/html/2611.02028v1#A1.SS5)); its semantics $\mathrm{sem}$ and cut classification $\mathrm{cut}$; a parser and printer, so that the byte round trip of [Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) can check every reading; the lane layouts of its warp-collective operations, recorded from the hardware; and, for a structured language, the lowering of source to the operation sequence, which is itself a pinned Lean definition. The translator’s race-freedom layer is one algorithm with a per-language front end that says what an operation’s operands and memory window are. PTX, being unstructured by design, needs no lowering; CUDA and HIP share the structured-tree lowering and differ in their operation tables and collective layouts.

Table 16: Components of LARK by whether they are shared across target languages or written per language, as instantiated for PTX, CUDA, and HIP. *Shared* components are one artifact used verbatim by every language; *shared core* components have one algorithm and one emitted Lean text with a thin per-language front end; *per language* components are written for each target and, where marked, measured against its hardware.



<table>
<tr>
<td><span>Component</span></td>
<td>
<span>
<span><span>What it contains</span></span>
</span></td>
<td><span>Status</span></td>
<td>
<span>
<span><span>Per-language part</span></span>
</span></td></tr>
<tr>
<td colspan="4"><em>Machine model $M$</em></td></tr>
<tr>
<td>State and memory hierarchy</td>
<td>
<span>
<span>One state for the launch: registers keyed by thread and name, global, shared, and local memory with type tags, coordinates and geometry</span>
</span></td>
<td>Shared</td>
<td>
<span>
<span>Coordinate reads and address spellings</span>
</span></td></tr>
<tr>
<td>Value representation</td>
<td>
<span>
<span>Typed values: exact reals with $-\infty$, exact integers with a width tag, predicates, packed halves; partial operations (faults)</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Cut hierarchy and folds</td>
<td>
<span>
<span>$\mathrm{cut}$ kinds, the folds of <a href="https://arxiv.org/html/2611.02028v1#S4.E2"><span>Equation</span> <span>2</span></a>, convergence, fuels</span>
</span></td>
<td>Shared</td>
<td>
<span>
<span>Which operations define which cut; the warp width (32 or 64)</span>
</span></td></tr>
<tr>
<td>Race freedom</td>
<td>
<span>
<span>Definition over footprints, span lemmas, pair-lemma library, the reduction metatheory relied on</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Premise schemas</td>
<td>
<span>
<span>Placement, separation, parameters, alignment, geometry, fuel, entry (Appendix <a href="https://arxiv.org/html/2611.02028v1#A1.SS8"><span>A.8</span></a>)</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Certificate</td>
<td>
<span>
<span>The certificate structure, non-vacuity, the end-to-end theorem proved once</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Operation set $\mathcal{I}$ and $\mathrm{sem}$</td>
<td>
<span>
<span>Operation type generated from a pinned table; lower-layer definitions; contextual semantics</span>
</span></td>
<td>Per language</td>
<td>
<span>
<span>Measured against the hardware (Appendix <a href="https://arxiv.org/html/2611.02028v1#A1.SS5"><span>A.5</span></a>)</span>
</span></td></tr>
<tr>
<td>Collective layouts</td>
<td>
<span>
<span>Lane and fragment layouts of <span>ldmatrix</span>, <span>mma</span>, <span>wmma</span>, shuffles</span>
</span></td>
<td>Per language</td>
<td>
<span>
<span>Recorded from the hardware</span>
</span></td></tr>
<tr>
<td>Source lowering</td>
<td>
<span>
<span>Structured source tree to operation sequence, as a pinned Lean definition</span>
</span></td>
<td>Per language</td>
<td>
<span>
<span>PTX: none; CUDA and HIP: shared tree lowering</span>
</span></td></tr>
<tr>
<td>Torch module formalization</td>
<td>
<span>
<span>Lean definition of the module, tested against the module</span>
</span></td>
<td>Per module</td>
<td>
<span>
<span>Shared by every language</span>
</span></td></tr>
<tr>
<td colspan="4"><em>Translator</em></td></tr>
<tr>
<td>Parser and printer</td>
<td>
<span>
<span>Source to data and back; byte round trip</span>
</span></td>
<td>Per language</td>
<td></td></tr>
<tr>
<td>Refusals</td>
<td>
<span>
<span>Closed list of named refusals (Appendix <a href="https://arxiv.org/html/2611.02028v1#A1.SS9"><span>A.9</span></a>)</span>
</span></td>
<td>Shared mechanism</td>
<td>
<span>
<span>Per-language spellings</span>
</span></td></tr>
<tr>
<td>Grid-barrier recognition</td>
<td>
<span>
<span>Pinned template (PTX) or the one-line call (CUDA, HIP)</span>
</span></td>
<td>Per language</td>
<td></td></tr>
<tr>
<td>Race-freedom layer</td>
<td>
<span>
<span>Footprint regions, pair lemmas and their proofs, agent fallback</span>
</span></td>
<td>Shared core</td>
<td>
<span>
<span>Operand and memory-window reader</span>
</span></td></tr>
<tr>
<td>Scaffolding</td>
<td>
<span>
<span>Phase structure, counted loops and bridges, fuels</span>
</span></td>
<td>Shared core</td>
<td>
<span>
<span>Loop and branch spellings</span>
</span></td></tr>
<tr>
<td colspan="4"><em>Proof generation and reporting</em></td></tr>
<tr>
<td>Statement generation</td>
<td>
<span>
<span>The fixed premise and extractor over a signature; non-vacuity proved once</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Agent loop and failure analysis</td>
<td>
<span>
<span>Segmentation, proofs, diagnosis</span>
</span></td>
<td>Shared</td>
<td></td></tr>
<tr>
<td>Report and export</td>
<td>
<span>
<span>Trusted-assumption report, standalone Lean project</span>
</span></td>
<td>Shared</td>
<td></td></tr>
</table>



### A.7 Harness Details

Table 17: Basis for acceptance at each stage of LARK. *Verified*: a Lean proof for the kernel at hand. *Trusted*: pre-checked or declared, kernel-independent, listed in the trusted-assumption report ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)). *Judged*: data a run supplies, quoted in the report. *Untrusted*: agent-generated, covered by a later Verified row. [Figure 1](https://arxiv.org/html/2611.02028v1#S1.F1) uses the same statuses.



<table>
<tr>
<td><span>Stage / artifact</span></td>
<td><span>Produced by</span></td>
<td>
<span>
<span><span>Basis for acceptance</span></span>
</span></td>
<td>
<span>
<span><span>Status</span></span>
</span></td>
<td>
<span>
<span><span>If it fails</span></span>
</span></td></tr>
<tr>
<td colspan="5"><em>Offline, once per target language or per module</em></td></tr>
<tr>
<td>Machine model $M$</td>
<td>Humans, agents</td>
<td>
<span>
<span>Review; pinned by hash; its idealizations declared as scope</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>(pre-checked)</span>
</span></td></tr>
<tr>
<td>Per-operation semantics</td>
<td>Generated from pinned tables</td>
<td>
<span>
<span>Measured against the hardware; pinned by hash</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Operation not in table: refuse</span>
</span></td></tr>
<tr>
<td>Torch module formalization</td>
<td>Agents, humans</td>
<td>
<span>
<span>Tested against the Torch module on random inputs</span>
</span></td>
<td>
<span>
<span>Trusted (tested)</span>
</span></td>
<td>
<span>
<span>Mismatch: fix before any run</span>
</span></td></tr>
<tr>
<td>Premise and extractor definitions</td>
<td>Pinned in $M$</td>
<td>
<span>
<span>One definition each over a signature; pre-checked</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>(pre-checked)</span>
</span></td></tr>
<tr>
<td>Reduction metatheory</td>
<td>Literature</td>
<td>
<span>
<span>Race-free phases $\Rightarrow$ one final state for every interleaving</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Reported</span>
</span></td></tr>
<tr>
<td colspan="5"><em>Per kernel</em></td></tr>
<tr>
<td>Kernel as data</td>
<td>Translator</td>
<td>
<span>
<span>Read-back check: the data formatted back to text equals the source byte for byte</span>
</span></td>
<td>
<span>
<span>Verified (read-back)</span>
</span></td>
<td>
<span>
<span>Refuse</span>
</span></td></tr>
<tr>
<td>Flat model</td>
<td>Derived in $M$</td>
<td>
<span>
<span>Built from the data by the pinned definition</span>
</span></td>
<td>
<span>
<span>Verified (by construction)</span>
</span></td>
<td></td></tr>
<tr>
<td>Race-freedom obligation $\mathrm{RF}(K)$</td>
<td>Translator; agent fallback</td>
<td>
<span>
<span>Lean proof per phase and pair of units</span>
</span></td>
<td>
<span>
<span>Verified</span>
</span></td>
<td>
<span>
<span>Reject</span>
</span></td></tr>
<tr>
<td>Segment-level model</td>
<td>Agent</td>
<td>
<span>
<span>None; covered by $\mathrm{SEG}(K)$</span>
</span></td>
<td>
<span>
<span>Untrusted</span>
</span></td>
<td>
<span>
<span>See $\mathrm{SEG}(K)$</span>
</span></td></tr>
<tr>
<td>Theorem statement</td>
<td>Generated from the signature</td>
<td>
<span>
<span>The fixed premise and extractor over the task’s signature, which the task’s tests launch with</span>
</span></td>
<td>
<span>
<span>Trusted (signature)</span>
</span></td>
<td>
<span>
<span>Entry point mismatch: refuse</span>
</span></td></tr>
<tr>
<td>Non-vacuity</td>
<td>Proved once in $M$</td>
<td>
<span>
<span>The fixed premise satisfiable for every signature and input</span>
</span></td>
<td>
<span>
<span>Verified</span>
</span></td>
<td></td></tr>
<tr>
<td>Segmentation obligation $\mathrm{SEG}(K)$: flat $=$ segment-level</td>
<td>Agent</td>
<td>
<span>
<span>Lean proof; no <span>sorry</span>; standard axioms</span>
</span></td>
<td>
<span>
<span>Verified</span>
</span></td>
<td>
<span>
<span>Retry; then no certificate</span>
</span></td></tr>
<tr>
<td>Equivalence obligation $\mathrm{EQ}(K,S)$: segment-level $=$ specification</td>
<td>Agent</td>
<td>
<span>
<span>Lean proof; no <span>sorry</span>; standard axioms</span>
</span></td>
<td>
<span>
<span>Verified</span>
</span></td>
<td>
<span>
<span>Retry; then no certificate</span>
</span></td></tr>
<tr>
<td>End-to-end theorem</td>
<td>Proved once in $M$</td>
<td>
<span>
<span>Assembles the rows above into <a href="https://arxiv.org/html/2611.02028v1#S3.E1"><span>Equation</span> <span>1</span></a>; <span>#print axioms</span></span>
</span></td>
<td>
<span>
<span>Verified</span>
</span></td>
<td></td></tr>
<tr>
<td>Generator hints</td>
<td>Kernel generator</td>
<td>
<span>
<span>Regions: checked by the lemmas; constant marks and launch: reported</span>
</span></td>
<td>
<span>
<span>Untrusted / Judged</span>
</span></td>
<td>
<span>
<span>Failed proof / reported</span>
</span></td></tr>
<tr>
<td colspan="5"><em>Standing</em></td></tr>
<tr>
<td>Named constants and approximate instructions</td>
<td>Table</td>
<td>
<span>
<span>Read as exact values</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Reported</span>
</span></td></tr>
<tr>
<td>Grid-barrier template</td>
<td>Pinned</td>
<td>
<span>
<span>A barrier under a cooperative launch</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Other spelling: refuse</span>
</span></td></tr>
<tr>
<td>Read-back formatter</td>
<td>Fixed per language</td>
<td>
<span>
<span>Formats the kernel data back to text; about a quarter the size of the reader</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Reported</span>
</span></td></tr>
<tr>
<td>Floating-point behavior</td>
<td>Numeric test</td>
<td>
<span>
<span>Outside the theorem; tested at a tolerance</span>
</span></td>
<td>
<span>
<span>Scope</span>
</span></td>
<td>
<span>
<span>Reported</span>
</span></td></tr>
<tr>
<td>Lean, Mathlib, axioms</td>
<td>External</td>
<td>
<span>
<span><span>#print axioms</span> per theorem</span>
</span></td>
<td>
<span>
<span>Trusted</span>
</span></td>
<td>
<span>
<span>Reported</span>
</span></td></tr>
</table>



##### Grid barrier.

Block-level barriers are single operations with hardware-defined semantics and a cut of their own. A grid-wide barrier is not: in PTX, grid.sync() lowers to atomic counters and spin loops. LARK pins one template of that lowering, which a PTX kernel copies verbatim with its own registers and labels; the translator reads the copy as one grid-barrier pseudo-operation whose body is the identity and whose cut is grid-level, and the printer re-emits the template from the names it read, so the recognition is covered by the byte round trip. What remains trusted is that the template is a barrier under a cooperative launch. In CUDA and HIP the barrier is the one-line this_grid().sync(). Any other spelling is refused.

##### Value types.

The type tags of [Section 4.1](https://arxiv.org/html/2611.02028v1#S4.SS1) are $\mathit{Ty}=\{\mathrm{f16},\mathrm{f32},\mathrm{f16x2},\mathrm{int}_{w},\mathrm{pred}\}$: a 16- or 32-bit float with payload in $\mathbb{R}\cup\{-\infty\}$, two packed 16-bit floats held as one value, an integer of width $w$ with payload in $\mathbb{Z}$, and a predicate. Other formats (bf16, 64-bit floats, packed 8-bit types) are outside the set and refused by name (Appendix [A.9](https://arxiv.org/html/2611.02028v1#A1.SS9)). The payload of a float tag is an element of $\mathbb{R}\cup\{-\infty\}$, with $-\infty$ admitted because kernels use it as the initial value of a running maximum; of a packed tag a tuple of them; of an integer tag an element of $\mathbb{Z}$, the width being part of the tag; of a predicate an element of $\{0,1\}$. Rounding is hard to reason about, which is why arithmetic is exact in $M$ and floating-point behavior is left to testing ([Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3)).

##### Read-back check.

The translator’s reading of a kernel is not verified; each of its outputs is checked instead, in the manner of translation validation ([Pnueli et al., 1998](https://arxiv.org/html/2611.02028v1#bib.bib30); [Necula, 2000](https://arxiv.org/html/2611.02028v1#bib.bib31)): the parsed data is formatted back to text and compared byte for byte with the source, so a wrong reading passes only if the formatter has a compensating defect. The trust therefore moves from the reader to the read-back formatter, which is the smaller and simpler component: for PTX the reader is 1,405 lines against 320 for the formatter, for CUDA 2,313 against 570, with a 217-line helper shared by both. The formatter walks the emitted data, not the reader’s in-memory objects, so what is printed is the data the theorem is about. The formatter is a trusted item and is listed in the report.

##### Fuel.

Each level of the fold of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) runs under a fuel: lane steps per sub-phase, sub-phases per block-level phase, block-level phases per grid-level phase, and grid-level phases per launch. The fuels are data a run supplies, and exhausting any of them is a fault, so a fuel too small for the kernel fails the proof rather than weakening the theorem; nothing about loop bounds is trusted or analysed by a trusted component. The translator proposes generous fuels from the source, and the proof carries termination.

##### Folds and the proof-friendly restatement.

The folds of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) are pinned once in $M$. The agent’s segment-level model restates the kernel’s run in a sequential shape that a proof can induct on; the first equivalence of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) proves that this restatement equals the fold, using the verified race freedom of the kernel, and the second relates the restatement to the specification.

##### Efficiency mechanisms.

Several mechanisms keep the agent–checker loop efficient: the agent can test lemmas in scratch Lean scripts before committing to them; a timed-out check returns the slowest declarations, the lemmas that passed, and any errors raised, so a failure still yields information; progress and delta submissions let the agent resume from a partial proof or patch part of a program instead of re-emitting it; and context compaction drops the oldest dispensable turns and repeated listings while keeping the system prompt, the first turn, and the most recent messages.

##### Failure analysis.

When a loop exhausts its attempts, an agentic failure analysis diagnoses the cause (a hard proof, an inefficient proving approach or formalization, or a faulty kernel) and acts accordingly: continue the loop, restart it with a fresh dialogue, redo the formalization, or return the diagnosis to the generation side as feedback for a new kernel. The last path closes the generate-and-verify cycle: a kernel that cannot be proved equivalent because it is wrong is sent back with an explanation of why. The pipeline aborts only when the number of analyses or the overall budget reaches its ceiling.

### A.8 Premise Schemas

Every theorem statement is computed from the task’s *signature*: a list with one entry per kernel argument, in entry order. A buffer entry carries its role (input, output, in-place, or scratch), its element type, its shape, and its layout, a strided tensor in PyTorch’s sense, with each stride a literal or a named integer parameter; a scalar entry carries its domain (any real, or positive) or its pinned value; an integer dimension carries its value. The launch is recorded with it: the block pinned or capped at the declared bound, and the grid pinned, capped and one-dimensional, or merely non-empty, the form chosen by data. The signature is generated from the task’s registry, the same list the task’s tests launch the kernel with, and is checked against the kernel’s entry point by position and kind. The fixed premise over a signature states: every input and in-place buffer placed at its base under its layout, with the memory’s type tags set; every pair of distinct buffers byte-separated; every base aligned to its element size and vector width; every scalar bound to the specification’s input, or pinned; every dimension pinned; every runtime stride at least its row length; and the launch geometry. A scratch buffer is separated and aligned, never placed or extracted; an in-place buffer is placed as an input and read as an output. The fixed extractor reads every output and in-place buffer under its layout. Both are defined once in $M$; the facts a proof needs (a placement, a separation, a binding) are projection lemmas of the model, instantiated by name for the run. Satisfiability of the premise is one theorem of $M$ over every signature whose byte spans fit. Layouts out of scope, refused by name: sub-byte and bit-packed elements, indirection through index buffers, and swizzled or blocked global layouts; shared-memory layouts are internal to the kernel and never part of the premise.

Formally, the signature is a list $\mathit{sig}\in\mathit{List}(\mathit{Arg})$ with one entry per kernel argument,



$$
\displaystyle\mathit{Arg}::={}
\displaystyle\mathrm{buf}(\rho,\tau,\mathit{shape},\mathit{layout})
\displaystyle\mid\mathrm{scalar}(\mathit{dom})\mid\mathrm{scalar}(v)\mid\mathrm{dim}(n)
$$




a buffer with role $\rho\in\{\mathrm{in},\mathrm{out},\mathrm{inout},\mathrm{scratch}\}$, element type $\tau$, shape, and strided layout (element $\vec{i}$ at $\mathit{base}+|\tau|\sum_{k}i_{k}s_{k}$); a float scalar bound to a specification input of domain $\mathit{dom}$ or pinned to $v$; an integer dimension pinned to $n$. The theorem of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) quantifies over the kernel’s parameters $\vec{p}$ (base addresses, strides, scalars; [Equation 1](https://arxiv.org/html/2611.02028v1#S3.E1) folds them into $\sigma$), every state satisfying the premise, and every input, and the satisfiability theorem above reads



$$
\forall\mathit{sig}.\ \forall x\in D.\ \exists\vec{p}\;\sigma.\ \mathrm{Pre}_{M}(\mathit{sig},\mathit{launch})(\vec{p},\sigma,x),
$$




proved once for every signature, so no statement holds vacuously.

### A.9 Unsupported Constructs

The translator refuses, by name and before any proof is attempted: $+\infty$ as a value and NaN in any form (the creation of either is a fault); integer wraparound and integer division or remainder by zero; bit operations other than a literal mask, a literal shift, and the full warp mask; float-to-integer conversions out of range; the .sat, .ftz, and bf16 forms, FP64, and 8-bit types; asynchronous operations (cp.async, mbarrier, TMA, wgmma, tcgen05, cluster barriers); atomics and fences used for communication; partially masked barriers, shuffles, and collectives; a grid barrier spelled other than the pinned template; addresses that are not polynomial in the thread coordinates, loop counters, and parameters; blocks with more than one dimension and grids with more than two; and any operation absent from the pinned table. Approximate instructions (ex2.approx, lg2.approx, rcp.approx, rsqrt.approx) are accepted and read as the exact function, and the tf32 form of mma is accepted as an FP32 carrier; both are declared idealizations listed in the report. The complete list, with the name of each refusal, ships with the code.

### A.10 Race-Freedom Lemmas

Race freedom is what the fold of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) needs because



$$
\mathrm{RF}(K)\wedge\mathrm{run}^{\vec{\kappa}}_{\mathrm{grid}}(\sigma)=\sigma^{\prime}\\
\Longrightarrow\llbracket K\rrbracket_{M}(\sigma){\downarrow}\wedge\llbracket K\rrbracket_{M}(\sigma)=\sigma^{\prime},
$$




that is, every execution of $K$ from $\sigma$, under any scheduling, is finite and ends in $\sigma^{\prime}$: the actions of race-free units commute, and the barrier that ends a phase orders the phases (Lipton’s reduction, [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2)).
The translator derives each footprint $W_{\phi}(u)$, $R_{\phi}(u)$ of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2) as a region polynomial in the thread coordinates, loop counters, and kernel parameters, and generates the two disjointness lemmas of the race-freedom obligation for every phase and pair of units, with proofs over the theorem’s premises, which Lean checks. Where an address is not polynomial in these variables (a clamped index, for instance), the translator cannot name the region: it emits the lemma’s statement without a proof and an agent attempts it, with an independent second opinion on a verdict that the statement is false. A kernel with an unproved lemma is rejected, whether from a real race or a region the translator could not name, and a located race is returned as a counterexample naming the two threads and the cell ([Section 5](https://arxiv.org/html/2611.02028v1#S5)). The condition is sufficient, not necessary: it also rejects units that communicate within a phase, such as producer–consumer blocks.

### A.11 Generator Hints

A kernel generator may annotate its kernel with three kinds of hint: the regions each thread writes in a phase, which guide the translator’s choice of footprints when it generates the race-freedom lemmas; a mark naming the exact constant behind a rounded literal, which the translator cross-checks against the literal’s bits and which the pinned decoder then reads as the exact value; and the launch it expects, which selects the launch form of the fixed premise and which the numeric test launches with. The first kind decides nothing, since every lemma built from it is checked by Lean and a wrong region fails a proof. The second and third are the two hints Lean cannot check; both appear in the report.

### A.12 Certificate Export

Every run exports its certificate as a standalone Lean project: the pinned model text, the kernel as data with its parse, the signature, the fuels, the race-freedom lemmas, the two equivalence proofs, and the end-to-end theorem, together with the output of #print axioms on that theorem. The project builds with the public Lean toolchain and Mathlib and re-checks without any part of LARK, so a certificate can be audited by a checker that shares no code with the one that produced it. The supplementary material contains the project of the GQA kernel.

### A.13 The Kernel-Independent Premises

The kernel-independent premises $\Pi_{M}$ of [Section 4.3](https://arxiv.org/html/2611.02028v1#S4.SS3) are: (i) the idealizations of $M$, exact arithmetic as scope and the fold of [Equation 2](https://arxiv.org/html/2611.02028v1#S4.E2) with the reduction result of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2); (ii) the per-operation semantics $\mathrm{sem}$, generated from the pinned tables and measured against the hardware (Appendix [A.5](https://arxiv.org/html/2611.02028v1#A1.SS5)); (iii) the pinned grid-barrier template; (iv) a table of four named constants, under which a rounded literal a kernel writes for $\log_{2}e$ is read as the exact value, and the approximate instructions read as exact; (v) the Lean formalization of the Torch module; (vi) the premise and extractor definitions $\mathrm{Pre}_{M}$ and $\mathrm{Ext}_{M}$; (vii) the read-back formatter (Appendix [A.7](https://arxiv.org/html/2611.02028v1#A1.SS7.SSS0.Px3)); (viii) Lean, its three standard axioms, and its toolchain. Item (v) is tested: a Lean metaprogram evaluates the elaborated specification in 64-bit floating point on random inputs against the Torch module, refusing by name any construct it does not recognize, and the result is reported as evidence, not proof.

### A.14 Production Kernels

##### What “adapted” means.

No production kernel is verified byte for byte as upstream ships it. Each adapted kernel is the smallest mechanical rewrite of the upstream kernel into the pipeline’s input form, and every step is recorded as a change with its reason. The rewrites common to most kernels change the form of the code and not what it computes: one non-template extern "C" kernel per file, with the template instantiated as the library launches it and helpers such as warp reductions inlined; one operation per line, for written as while, no ternary; warp reductions written as full-mask shuffle steps; a lane-0 guard on the per-warp shared store that every lane makes with the same value; a declared block size where upstream declares none, since the theorem pins the block; dynamic shared buffers as static arrays of a covering size; cache hints and programmatic-dependent-launch calls dropped; a 2-D block flattened; vector accesses kept where the race-freedom lemmas cover them and otherwise written as loops of scalar accesses.

##### Departures.

A step that changes upstream’s semantics is marked as a departure, and the theorem is about the kernel with the departure. They are: 64-bit token and stride indices read as 32-bit (exact at the tested shapes; vllm_silu_and_mul, sglang_rmsnorm, vllm_rms_norm); a runtime alignment test dropped, keeping its aligned path (vllm_rms_norm); the positions gather dropped, so the input is the already gathered rows of the cosine and sine cache, and sizes and flags written as literals (vllm_rotary_embedding, vllm_fused_qknorm_rope); vllm_fused_qknorm_rope made out of place, writing Q and K to a new buffer; the number of threads written as the host’s literal and rsqrt.approx replaced by rsqrtf (flashinfer_rmsnorm); FP32 input and output made FP16 storage (llmc_softmax_forward, fam_flash_attn); the running max and sum made per-thread arrays and the output zeroed by the kernel rather than the host, and __expf replaced by expf (fam_flash_attn); the int4 dequantization removed, so the weights are FP16, a single split-$K$ iteration, and the block decode’s sizes written as literals (vllm_awq_gemm, transcribed into PTX from upstream’s inline assembly). sglang_fused_add_rmsnorm has no departure and is verified in place, with input and residual as in-out buffers. Per kernel, the rewrite steps and, of them, the departures are: vllm_silu_and_mul 6 and 1, sglang_rmsnorm 10 and 1, sglang_fused_add_rmsnorm 9 and 0, vllm_fused_qknorm_rope 11 and 3, vllm_rotary_embedding 6 and 3, vllm_rms_norm 8 and 2, flashinfer_rmsnorm 12 and 2, llmc_softmax_forward 11 and 2, fam_flash_attn 9 and 4, vllm_awq_gemm 10 and 3.

##### The llm.c race.

softmax_forward_kernel7 computes the softmax of each row in three passes: a maximum, a sum of exponentials, and a normalization that writes the row of out. In the third pass a thread’s unguarded clamped load __half yh = out[a] with $a=\min(C-1,k)$ reads a cell of out that another thread of the same block stores at out[a] = qh in the same pass, with no barrier between; the loaded value is used only where $k<C$, where the clamped address is the thread’s own cell, so on the tested shapes nothing observable depends on the race. What happened in LARK: the translator’s race-freedom layer builds its footprints from accesses whose addresses are polynomials in the thread and loop indices, and the clamped address is not one, so it refuses to emit its lemma raceFree_1 for the output store, without locating the race, and the obligation goes to the agent fallback of [Section 4.2](https://arxiv.org/html/2611.02028v1#S4.SS2), where the race-freedom statement is false. The fallback’s loop with GLM 5.3 ended after 10 turns and 9.8M input tokens with a counterexample rather than a proof: thread 0 of block 0 at the load with $k\geq C-1$ reads out[50256], and thread 80 of the same block, the unique writer of that cell, stores it at the third pass’s store; the independent second opinion confirmed the race. A second run with a neutral kernel header found a different witness of the same race (a one-block, 32-thread launch the theorem admits) in 7 turns, so the race is found from the code alone, at about a tenth of what a proof of this kernel costs. The kernel of [Table 5](https://arxiv.org/html/2611.02028v1#S5.T5) is this original text, and its row records the rejection. A one-line rewrite removes the race: upstream reads out[row + min(C-1, k)] unconditionally, and the rewrite reads out[row + k] under the store’s bounds guard if (k < C), so each thread reads only the cell it then writes, with every other line unchanged and the same values on every input; it verified with Opus 5.5 in a separate run.

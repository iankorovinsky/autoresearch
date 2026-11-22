---
title: "TensorLean: Formal Verification of Production GPU Kernels"
source_url: "https://tensorlean.ai/"
full_text_url: "https://tensorlean.ai/"
retrieved_date: "2026-11-21"
conversion: "HTML to Markdown with TeX math"
original_html: "sources/23-tensorlean.html"
---

# TensorLean: Formal Verification of Production GPU Kernels

GPU kernels serve as the computational core of modern AI. A single bug inside one kernel can silently compromise the entire model's correctness and put both training and inference at risk[[1](https://tensorlean.ai#ref-barron2026fix), [2](https://tensorlean.ai#ref-grattafiori2024llama3herdmodels), [3](https://tensorlean.ai#ref-zhou2026opguard)]. Formal verification promises correctness over any configuration and GPU schedule, but it has never reached production kernels for modern AI. **We present TensorLean, the first framework to formally verify the correctness of production GPU kernels.** With TensorLean, we verified 149 real-world open-source kernel instances on their PTX assembly, including 102 from [FlashAttention](https://github.com/Dao-AILab/flash-attention) and those from [DeepSeek](https://github.com/deepseek-ai), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), etc., and proved 83 of them correct. In verification, we also discovered 38 bugs of various kinds in frontier kernels that had eluded testing for months and years. We expect that, with TensorLean, formal verification will scale and become an essential step in developing correct and efficient kernels.

## Introduction

In February 2024, ChatGPT began producing text that made no sense [[4](https://tensorlean.ai#ref-openai2024unexpected)].
The following year, a miscompilation disrupted Claude's token selection [[5](https://tensorlean.ai#ref-anthropic2025postmortem)].
Now, in October 2026, Google DeepMind showed that even standard softmax and cross-entropy in PyTorch and JAX contain numerical bugs that can corrupt gradients and collapse a model's accuracy during training [[1](https://tensorlean.ai#ref-barron2026fix)]. All of them were caused by silent correctness bugs in kernels, which are programs that implement the model's modules and algorithms, such as matrix multiplication and attention.

As AI models become infrastructure, we need to detect and fix these bugs, and prove their absence before they waste millions of GPU hours and dollars.
Formal verification is the only way to provide the exact guarantee.
In formal verification, developers translate their code into a mathematical model, then try to prove that the model satisfies a set of mathematical properties, known as specifications.
Once the proof is machine-checked, (Sidenote: The proof is written as a program in a formal language, e.g., Lean. A proof assistant then checks the proof's validity by type checking it.)  the code meets its specification on all allowed inputs and in every possible execution.

However, formal verification is often expensive: we must state precisely what the code should do, in a specification, and why it does so, in a proof. Both can cost orders of magnitude more than the development per se.
For example, the [seL4](https://github.com/seL4/seL4) microkernel took 2.5 person-years and 8,700 lines of C to build, but 20 person-years and 200,000 lines of proof to verify [[6](https://tensorlean.ai#ref-klein2010sel4)].
Even with state-of-the-art verifiers, specifications and proofs are still several times larger than code [[7](https://tensorlean.ai#ref-lattuada2024verus)].
Moreover, GPU kernels are known to be complicated for their massive concurrency, distributed and memory layouts, and floating-point arithmetic, all challenging the scalability of formal verification.

To achieve formal guarantees on kernels at an affordable cost, we introduce TensorLean, the first to formally verify the correctness of production kernels.
In TensorLean, the efforts in specification and proofs are both reduced to a minimum.
For specification, TensorLean decouples mathematical computation from memory layouts, so specifications are as easy to write and audit as an ordinary tensor program and scale to production kernels at about a 15:1 code-to-spec ratio.
For verification, TensorLean proves that a kernel's compiled PTX meets its tensor-level specification on every input and thread schedule, within floating-point tolerance.
With TensorLean, we verified 149 kernel instances on their PTX, covering all 37 kernels of FlashAttention-2 to -4 and more from DeepGEMM and Liger Kernel: 83 are proved correct, and the other 66 contain bugs we found.
Along the way, we found 38 bugs that silently eluded the kernels' own tests.

[Inline SVG figure not converted; see the preserved HTML. Label: S = Q K-transpose; one row of S through softmax and V gives one row of O. Text in figure: K ⊤ ; V ; Q ; S ; O ; softmax]

*O* = softmax(*Q* · *K*⊤/√*d*) · *V*

TensorLean

````
theorem flash_attention_correct :
  ∀ inputs executions,
    ptx ≈ attention := by
  tensorlean
````

FlashAttention-12022

mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 …

**A100**

FlashAttention-22023

cp.async.cg.shared.global.L2::128B [%r93], [%rd324], 16;

**A100**

FlashAttention-32024

wgmma.mma_async.sync.aligned.m64n224k32.f32.e4m3.e4m3 …

**H100**

FlashAttention-42026

tcgen05.mma.cta_group::2.kind::f16 [tmem_acc], …

**B200**

⋮

We will soon release our paper and [open-source TensorLean](https://github.com/TensorLean/TensorLean), including its PTX semantics and the specifications and proofs of the verified kernels, so that anyone can reuse them and verify other frontier kernels.
We expect that, with TensorLean, we can pace the formal verification of GPU kernels in step with their evolving development, with both experts and agents involved [[8](https://tensorlean.ai#ref-ye2026cake)].

In the rest of this article, we cover TensorLean in more detail.


## Tensor-Level Specification

To prove that a kernel is correct, we must first state what a "correct behavior" is, known as its *specification* (spec).
A kernel is correct if, on every input it accepts and in every execution, it writes the same values (Sidenote: Due to floating-point rounding, "the same" means equal within a bound, known as **tolerance**.)  as the operation it implements, and changes nothing else in GPU memory.
For example, an attention kernel with a KV cache is correct if, for every $Q, K, V$ with head dimension $d$, it writes $O = \mathrm{softmax}(Q K^\top/\sqrt{d})\cdot V$, appends $K, V$ to the cache, and leaves other bytes of GPU memory as they were.

The definition of kernel correctness is extensional: (Sidenote: Property $P$ is extensional if $(\forall m.\ f(m) = g(m)) \Rightarrow (P(f) \Leftrightarrow P(g))$. For the kernel correctness case, $m$ is GPU memory, and $f,g$ are memory updates by two kernel executions, respectively.)  it constrains only the memory before and after the kernel finishes, in every possible execution.
A kernel's specification therefore describes the values the kernel reads and writes, and nothing of its execution, such as its warp specialization or synchronization.

**Kernel Specification at Scale.** The correctness specifications are notoriously costly to write, as they demand expertise in both verification and the target domain, and can be several times larger than code [[7](https://tensorlean.ai#ref-lattuada2024verus)]. The root cause is that a specification couples the mathematics of the computation with the representation of its data in the machine model's memory.

TensorLean makes the specification of GPU kernels scale: we achieve an average 15:1 code-to-spec ratio. For example, it only needs 2,291 lines of specification to define the correctness of 12 attention (forward) kernel variants, including the FlashAttention family and more, totaling 35,852 lines of source:

Code vs. Spec Size Comparison: Attention (Forward)

Kernel SourceTensorLean SpecificationShared Spec Library

[Inline SVG figure not converted; see the preserved HTML. Label: Code vs. Spec Size Comparison: Attention (Forward). Text in figure: 0 ; 5,000 ; 10,000 ; 15,000 ; 20,000 ; 25,000 ; 30,000 ; 35,000 ; 40,000 ; 0 ; 1 ; 2 ; 3 ; 4 ; 5 ; 6 ; 7 ; 8 ; 9 ; 10 ; 11 ; 12 ; Number of Kernel Variants (Ordered by First Release) ; Lines of Code ; 35,852 lines ; 2,291 lines ; 156 lines]

Figure 1. Kernel source vs. TensorLean specification size, accumulated over 12 attention (forward) variants.

**Decoupling Mathematics from Memory.** The key observation behind TensorLean's scalability is that the math and memory can be decoupled because a GPU kernel's correctness is extensional: the specification refers to memory only through the memory layout, which maps each tensor element to the bytes that hold it, while the mathematics refers to tensors alone.

Input tensors*Q*, *K*, *V*

`spec_​math`

Output tensor*O* = softmax(*QK*⊤/√*d*) *V*

Specification

`spec_​layout`

`spec_​layout`

PTX memory before*Q**K**V*…

attention kernel

PTX memory after*Q**K**V**O*…

Implementation

SpecificationImplementation

Figure 2. Decoupling an attention kernel's mathematics from PTX memory.

With math and memory decoupled, a kernel’s specification can be written at the tensor level instead of the PTX level: the mathematics is stated over tensors, and the memory layout of each tensor is given separately. TensorLean then automatically lowers this tensor-level specification to a PTX-level one, in which each tensor element is read from the bytes that hold it, as given by the memory layout:

over tensors

````
Q[i, e]
````

[i, e] ↦ q_ptr + 2*(i*D + e)layout of Q

over PTX memory

````
fp16(mem[q_ptr + 2*(i*D + e)])
````

Concretely, a TensorLean specification consists of five parts, over the tensors that the kernel only reads (`input`), writes without reading their old values (`output`), and reads and overwrites in place (`state`), such as a KV cache:

1. `spec_math`: The **mathematical computation** the kernel implements, as a tensor function from `state` and `input` to `output` tensors of real numbers.
2. `spec_mut`: The **GPU memory update** the kernel makes in place, as a function from `state` and `input` to `state'`, the `state` after execution.
3. `spec_layout`: The **memory layouts** of the tensors, as a function from each tensor's indices to GPU memory addresses.
4. `spec_tol`: The **tolerance** $\varepsilon$ of each `output` and `state'` tensor, by which its floating-point values in memory may differ from its exact real values.
5. `requires`: The **precondition** on `state` and `input`.

Read-only input*Q*, *K*, *V*

[Inline SVG figure not converted; see the preserved HTML. Text in figure: spec_math ; spec_mut]

Return output|*O* − softmax(*QK*⊤/√*d*) *V*| ≤ *ε*

Mutable tensorKV cache: *C*

Mutable tensor*C*′ = append(*C*, *K*, *V*)

requires *C* has room for *K*, *V*

`q_ptr` `k_ptr` `v_ptr`

`c_ptr`

`o_ptr`

PTX memory

Figure 3. A sample specification of an attention kernel w/ KV cache in GPU global memory.

**Writing Specifications in Lean.** TensorLean provides a library of tensor operators to define the tensor-level specifications, built on Lean's [Mathlib](https://github.com/leanprover-community/mathlib4).
It lets developers declare tensors, with operators that follow PyTorch's names and semantics, e.g., `@` for `torch.matmul`, so a kernel's tensor-level specification takes no more effort to write, or to audit, than a PyTorch reference.
For example, an attention kernel's `spec_math` can be defined as below in TensorLean:

````
-- ℝ: real number, ℕ: natural number
-- L: sequence length, D: head dimension
structure Attention (L D : ℕ) where
  Q : Tensor ℝ [L, D]
  K : Tensor ℝ [L, D]
  V : Tensor ℝ [L, D]
  scale : ℝ

def attention : KernelSpec (Attention L D) (Tensor ℝ [L, D]) where
  spec_math input := softmax((input.Q @ input.Kᵀ) * input.scale) @ input.V
  spec_layout := with (q_ptr, ...)      -- the kernel's arguments
    {{  tensor := input.Q[i, e]         -- Q : Tensor ℝ [L, D], indexed by i, e
        var    := q_ptr                 -- the source variable that holds Q
        layout := i * D + e },...}      -- the offset of Q[i, e] from q_ptr
````

**On Production Kernels.** By decoupling mathematical computation (`spec_math`) from memory layouts (`spec_layout`), the TensorLean specifications across various same-type kernels share a significant number of the same tensor operations, thereby reducing the complexity in both writing and auditing the specifications. In our study, we formalized 72 open-source kernels from 9 operations with 151,285 lines of source code and only 10,278 lines of specifications, achieving a nearly 15:1 code-spec ratio.

Code vs. Spec Size Comparison, by Operation

Source CodeTensorLean Specification

[Inline SVG figure not converted; see the preserved HTML. Label: Code vs. Spec Size Comparison, by Operation. Text in figure: 0 ; 10,000 ; 20,000 ; 30,000 ; 40,000 ; Lines of Code ; 16:1 ; Attention Forward ; 15:1 ; Attention Backward ; 28:1 ; MLA ; 6:1 ; GEMM ; 16:1 ; RMSNorm ; 23:1 ; Top-k ; 15:1 ; LayerNorm ; 13:1 ; RoPE ; 10:1 ; SwiGLU]

Figure 4. Comparison of kernel source and TensorLean specification efforts.

## PTX-Level Verification

With the kernel's (1) PTX code and (2) specification provided, TensorLean automatically constructs the proof goals and verifies the kernel's functional correctness with Lean and SMT solvers.
The workflow can be illustrated as below:

From UserAuto-gen by TensorLeanProvenTrusted

1.**Source Code**Unmodified

````
copy(gK, sK);
__syncthreads();
gemm(S, sQ, sK);
softmax(S);
gemm(O, S, sV);
````

2.**Specification**Tensor-Level

````
spec_math input :=
softmax(Q @ Kᵀ*s) @ Vspec_layout := {
  Q[i,e] ↦ q_ptr+i*D+e,
  O[i,e] ↦ o_ptr+i*D+e }
````

Compile

Lower

3.**PTX Assembly**From CUDA/Triton/CuTe/…

````
cp.async  sK, gK;
bar.sync  0;
mma.sync  s, q, k;
ex2.approx p, s;
````

**PTX Semantics**

Happens-Before Analysis

4.**Happens-Before Graph**Program & Sync Order

[Inline SVG figure not converted; see the preserved HTML. Text in figure: warp 0 ; warp 1 ; st.shared ; bar.sync ; bar.sync ; ld.shared]

✓ Race-Free✓ Terminates

Symbolic Execution

5.**Post-Execution Memory**Same Across Schedules

````
pre   mem
post  mem′[o_ptr+…] =
Σj pij·mem[v_ptr+…]-- pij: online softmax
````

6.**Lowered Spec**PTX-Memory-Level

````
lowered :=
 pre  ∀ i e,
mem[q_ptr+​i*D+e]   = Q[i,e]
 post ∀ i e,
mem′[o_ptr+​i*D+e]   ≈ spec_math[i,e]
````

7.**Proof**Machine-Checked

````
theorem fwd :
  mem′ ⊨ lowered := by
  apply post_of_goals
  · post_vc     -- ℝ
  · bound_sound -- FP
````

✓ Correctness Proved

**Proof Checkers**Lean's Kernel, z3, cvc5

Verifying the correctness is hard: TensorLean needs to reason out exactly what GPU memory holds after the kernel's execution and prove that it meets the specification. For production kernels, the challenges for verification are:

- **Fine-grained concurrency** across hundreds of threads with asynchronous operations and relaxed PTX memory.
- **Dynamic input shapes and work partitioning**: the batch size and sequence lengths, known only at launch, determine the kernel's loop trip counts and the tiles each iteration indexes, and a scheduler assigns the work tiles to thread blocks at runtime.
- **Floating-point rounding** that makes the kernel's result not exactly the same as the mathematical specification, but within a tolerance.
- **Bit-level address arithmetic** from hardware-specific optimizations, such as swizzled memory layouts and warp shuffles, whose XORs, shifts, and masks bring nonlinear integer arithmetic into the proof goals.

Next, we discuss how TensorLean solves these challenges in formal verification.

**Reducing Concurrency to Sequential Execution.**
GPU kernels run hundreds of threads that synchronize through barriers and asynchronous copies, and a bug may appear only in a rare interleaving of them: a data race we found in FlashAttention-4 [[9](https://tensorlean.ai#ref-fa4issue3195)] never showed up in 17,760 runs of its tests.
Instead of enumerating interleavings, TensorLean reasons on the kernel's *happens-before graph* [[10](https://tensorlean.ai#ref-lamport1978time)], the order in which its operations take effect in every execution.
A node is one execution of an operation by one thread, and an edge is an ordering that the PTX ISA guarantees, such as program order within a thread, or an arrive before the wait it completes.
TensorLean builds the graph symbolically, over every CTA, thread, and loop iteration of every launch, and proves in Lean that it is sound.

On this graph, TensorLean first proves that the kernel is free of data races: any two accesses to the same address, at least one a write, are ordered by the graph.
Every read then returns the last write that happens before it.
It then proves that the kernel terminates: every loop is bounded, and every wait receives the arrivals it waits for, with no cycle among them; a kernel that hangs would otherwise satisfy any specification vacuously.

Together, the two properties reduce the kernel's concurrency to sequential execution.
The final memory is the initial memory updated by the *last writes*, the write to each address that no other write to it follows in the graph.
To compute these writes, TensorLean symbolically runs each group of threads that play the same role, such as the producer warp or the consumer warps, once in program order, with its CTA, thread, and iteration as variables.
The groups exchange writes only at synchronizations: at an arrive, a group pushes the writes before it, and at a wait, a group pulls in the writes pushed by the arrives it waits for.
Each read thus resolves to the last write before it, and TensorLean repeats this on that write's value until the value depends only on the kernel's inputs.


**Inferring Loop Invariants over Semirings.**
A kernel's loops run as many iterations as its tensors' shapes require, and the shapes are known only at launch.
All recent equivalence checkers for kernels [[11](https://tensorlean.ai#ref-volta), [12](https://tensorlean.ai#ref-sentryward)] check one concrete configuration at a time, unrolling the loops, so their proofs can only cover the configurations they enumerate and miss subtle bugs in the combinations of shapes and features that developers overlook.
Indeed, 19 of the 23 bugs we found in FlashAttention appear only in such configurations.
To prove a kernel correct under every configuration at once, TensorLean takes the configuration as symbolic parameters and reasons about loops with symbolic bounds, which requires a loop invariant (Sidenote: A property that holds at the start of every iteration, and therefore after the loop, whatever its trip count.)  for each value a loop carries.
Synthesizing loop invariants is undecidable in general, so verifiers usually ask developers to write them by hand [[7](https://tensorlean.ai#ref-lattuada2024verus)].
For example, FlashAttention-3's split-KV combine kernel alone carries 86 values, and every optimization of the kernel changes them, so writing invariants by hand keeps verification from scaling to production kernels.

We observe that most kernel loops compute over semirings, and TensorLean synthesizes their invariants automatically.
This is because most kernel loops share three common properties:

1. **Tiled iteration**: each iteration processes one tile of a tensor, so the trip count is fixed by the configuration before the loop starts.
2. **Accumulative updates**: each iteration only combines its own contribution into each value the loop carries.
3. **Data-independent control flow**: a loop's branches and synchronization depend on the iteration and the configuration, not on the data.

Together, these properties make each value $f$ that a loop carries follow one recurrence over a semiring, (Sidenote: A pair of associative operations in which $\otimes$ distributes over $\oplus$, $a \otimes (x \oplus y) = (a \otimes x) \oplus (a \otimes y)$: $(+, \times)$ for sums, and $(\max, +)$ for maxima, as $a + \max(x, y) = \max(a + x, a + y)$.)  $f_{k+1} = a_k \otimes f_k \oplus b_k$, where $b_k$ is what iteration $k$ contributes and $a_k$ how it rescales what came before.

For example, the online softmax at the core of FlashAttention reads one block $B_k$ of keys in iteration $k$, on a row of scores $S = QK^\top \cdot \mathrm{scale}$, and carries three values $f$, the running maximum $m$, the running sum $\ell$, and the output $O$, each following the recurrence as below:

| Carried value $f$ | $\oplus$, $\otimes$ | $a_k$ | $b_k$ |
| --- | --- | --- | --- |
| Running maximum $m$ | $\max$, $+$ | $0$ (no rescaling) | $\max_{j \in B_k} S_j$ |
| Running sum $\ell$ | $+$, $\times$ | $e^{m_k - m_{k+1}}$ | $\sum_{j \in B_k} e^{S_j - m_{k+1}}$ |
| Output $O$ | $+$, $\times$ | $e^{m_k - m_{k+1}}$ | $\sum_{j \in B_k} e^{S_j - m_{k+1}}\, V_j$ |

By distributivity, the rescalings to each new maximum multiply out, so every key seen so far ends up measured against the current maximum, and the closed forms are

$$
m_k = \max_j S_j, \qquad
      \ell_k = \sum_j e^{S_j - m_k}, \qquad
      O_k = \sum_j e^{S_j - m_k}\, V_j,
$$

where $j$ ranges over the keys of the first $k$ blocks: the softmax's maximum, its denominator, and its unnormalized output over those keys.

TensorLean finds these closed forms automatically: it reads $a_k$ and $b_k$ off one symbolic run of the loop's body and proves the closed form by induction on $k$, with a single SMT query.
After the last iteration, the closed form is the specification itself: once TensorLean checks that the blocks cover the row's keys exactly once and that masked keys contribute nothing, the row the kernel writes, $O/\ell$, is exactly `spec_math`'s $\mathrm{softmax}(S)\,V$.
TensorLean proves these invariants over real numbers; correctness within a floating-point tolerance is proved separately, as we discuss next.


**Correctness under Tolerance.**
For generality, a TensorLean specification is defined over tensors of real numbers, so that one specification covers an operation's kernels in every precision, such as FP32, BF16, and FP8.
A kernel, however, computes in floating point, so its arithmetic results inevitably involve rounding, (Sidenote: For example, $1 + 3.05 \times 10^{-8}$ cannot be represented in FP32 and is rounded to $1$. Such rounding causes the two numerical bugs that Google DeepMind found in standard softmax and cross-entropy in October 2026 [[1](https://tensorlean.ai#ref-barron2026fix)].)  and thus cannot always exactly match the mathematical value.

These rounding errors propagate through the kernel's computation and can accumulate, so an output may deviate from its mathematical value beyond an acceptable tolerance.
In the worst case, the error is passed to, and cascades through, the model's later computations until it corrupts training or inference, as with the recent bugs found in standard softmax [[1](https://tensorlean.ai#ref-barron2026fix)].

In practice, to catch such rounding deviations, kernel tests use a reference implementation, such as PyTorch’s, as a stand-in for the exact values. They require the kernel’s outputs on sampled inputs to match the reference’s within a tolerance ε, which bounds the allowed error:

$$
\lvert \mathrm{output}_{\mathrm{Float}} - \mathrm{output}_{\mathrm{ref}} \rvert \le \varepsilon
$$

To formally verify a kernel's correctness despite floating-point rounding, the proof goal must bound the kernel's deviation from `spec_math` by the same ε:

$$
\lvert \mathrm{kernel}_{\mathrm{Float}} - \texttt{spec\_math} \rvert \le \varepsilon.
$$

Without proving such correctness under tolerance ε, the verification would miss every rounding error that can occur on real execution.
However, reasoning about $|\mathrm{kernel}_{\mathrm{Float}} - \texttt{spec\_math}|$ directly does not scale: floating-point arithmetic breaks associativity, (Sidenote: For example, with $a = 10^8$ in FP32, $(a - a) + 1 = 1$, while $a + (1 - a) = 0$, as $1 - a$ rounds to $-a$.)  so the reasoning falls back to a bit-vector search [[13](https://tensorlean.ai#ref-brain2019symfpu)] that can bloat exponentially with the number of floating-point operations [[14](https://tensorlean.ai#ref-solovyev2018fptaylor)].

To scale automated verification, TensorLean never compares $\mathrm{kernel}_{\mathrm{Float}}$ with $\texttt{spec\_math}$ directly. Instead, it verifies the correctness by introducing $\mathrm{kernel}_{\mathrm{Real}}$, the kernel result evaluated under a real-number PTX semantics, and proceeds in two steps.
First, TensorLean proves that the kernel introduces no algebraic error: it reasons about $\mathrm{kernel}_{\mathrm{Real}}$ symbolically and proves

$$
\lvert \mathrm{kernel}_{\mathrm{Real}} - \texttt{spec\_math} \rvert = 0.
$$

Second, TensorLean computes a bound on the kernel's round-off error, (Sidenote: is the difference between its floating-point output and the exact result of the same kernel.)  $\lvert \mathrm{kernel}_{\mathrm{Float}} - \mathrm{kernel}_{\mathrm{Real}}\rvert$, by inductively propagating each operation's rounding through the kernel's computation tree in linear time, and proves

$$
\lvert \mathrm{kernel}_{\mathrm{Float}} - \mathrm{kernel}_{\mathrm{Real}} \rvert \le \varepsilon.
$$

Thus, by the triangle inequality, TensorLean proves the kernel's correctness within tolerance:

$$
\lvert \mathrm{kernel}_{\mathrm{Float}} - \texttt{spec\_math} \rvert \le \underbrace{\lvert \mathrm{kernel}_{\mathrm{Float}} - \mathrm{kernel}_{\mathrm{Real}} \rvert}_{\text{Round-off Error }\le \text{ } \varepsilon} + \underbrace{\lvert \mathrm{kernel}_{\mathrm{Real}} - \texttt{spec\_math} \rvert}_{\text{Algebraic Error } =\text{ } 0} \le \varepsilon
$$


**Assertions as Proof Hints.**
A proof goal contains the *terms* of the values it reads, each value written out as the operations that computed it from the kernel's inputs, and in production kernels these terms grow too large for SMT solvers to decide.
In TensorLean, developers can guide the proofs by writing *assertions* in the kernel's source code: a statement `tensorlean_assert` at a line states a property $P_{\mathrm{assert}}$ of the values there. This follows common practice in production-level verifiers such as Dafny [[15](https://tensorlean.ai#ref-leino2010dafny)] and Verus [[7](https://tensorlean.ai#ref-lattuada2024verus)]. TensorLean proves $P_{\mathrm{assert}}$ once from $\Gamma$, the kernel's precondition together with the semantics of its PTX, and assumes it in every proof goal $P_{\mathrm{goal}}$ after the line, by the cut rule:

$$
\frac{\Gamma \vdash P_{\mathrm{assert}} \qquad \Gamma, P_{\mathrm{assert}} \vdash P_{\mathrm{goal}}}{\Gamma \vdash P_{\mathrm{goal}}}
$$

Each proof goal after the line thus starts from $P_{\mathrm{assert}}$ instead of the computation that produced the values, and no longer contains that computation's terms. Below, we illustrate assertions with two examples: one for functional correctness, and one for the loop invariant of a persistent kernel's scheduler.

*Case 1: Assertions for proving functional correctness.* The largest terms come from the values that tensor cores accumulate. FlashAttention-4's backward kernel on Hopper, for example, accumulates the gradient $dK$ of a block of keys over all blocks of queries with WGMMA instructions; written out, each element of $dK$ is a term of thousands of operations. The kernel then scales the tile by `softmax_scale`, copies it from registers to shared memory, `sdK`, and, after a barrier, stores it to global memory. The tile's values at the barrier can be asserted in the source code as below:

[flash_attn/cute/flash_bwd_sm90.py](https://github.com/Dao-AILab/flash-attention/blob/68bd095cdd338c5171bccb9c200529f3620b85c9/flash_attn/cute/flash_bwd_sm90.py) (simplified)

````
for m_block in range(m_min, m_max):       # each block of queries
    mma_one_m_block(m_block)              # dK += dS.T @ Q (WGMMA)
acc_dK *= softmax_scale
copy_dK_r2s(acc_dK)                       # registers to shared memory, sdK
fence_view_async_shared()
epi_barrier.arrive_and_wait()
tensorlean_assert(sdK[r, c] == softmax_scale * dK[j, c])
if warp_idx == 4:
    store_dK()                            # shared to global memory (TMA)
````

The assertion at the barrier states that `sdK` then holds, element by element, the values of `spec_math`, row `r` of the tile being key `j = blockIdx.x * tile_n + r`. TensorLean proves it once, through the accumulation loop by the invariant it infers for that loop, a sum over the semiring $(+, \times)$. Every proof goal after the barrier then reads `sdK` as the values of `spec_math`, so the goal of each store no longer contains the accumulation's term. Three such assertions, for $dK$, $dV$, and the accumulator of $dQ$, suffice to prove every output of the kernel.

*Case 2: Assertions as loop invariants.* Some kernel loops do not compute over semirings, because they fall outside the three properties above, such as the scheduling loops of persistent kernels, which balance work across SMs by the lengths of the sequences. Developers state their invariants as assertions at the loop's head, and TensorLean proves each by induction: it holds at the first iteration, and each iteration keeps it. DeepGEMM's paged MQA kernel, for example, balances its work by the queries' context lengths `lens`: before the launch, its tasks, each a query and one split of its KV blocks, are divided into equal, consecutive ranges, one per CTA, and each CTA walks its range in a scheduler loop ([Figure 5](https://tensorlean.ai#figure-cta-tasks)). The invariant of its scheduler loop can be asserted in the source code as below:

[deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh#L230) (simplified, its scheduler inlined)

````
q = meta[sm].x,         kv = meta[sm].y * SPLIT;       // this CTA's first task
end_q = meta[sm + 1].x, end_kv = meta[sm + 1].y * SPLIT;
num_kv = ceil_div(lens[q], BLOCK_KV);
while (q != end_q || kv != end_kv) {
  // the task comes before this CTA's end
  tensorlean_assert(q < end_q || (q == end_q && kv < end_kv));
  // num_kv is query q's number of KV blocks
  tensorlean_assert(num_kv == ceil_div(lens[q], BLOCK_KV));
  // kv starts a split of those blocks, or is 0 if there are none
  tensorlean_assert(kv % SPLIT == 0 && (kv == 0 || kv < num_kv));
  compute(q, kv);       // logits of query q, KV blocks kv .. kv + SPLIT - 1
  kv += SPLIT;
  if (kv >= num_kv) {   // the next query
    kv = 0, q += 1;
    if (q < end_q || end_kv > 0)
      num_kv = ceil_div(lens[q], BLOCK_KV);
  }
}
````

The loop's trip count, and the iterations at which it moves to the next query, depend on `meta` and `lens`, the kernel's inputs, and `kv` restarts at 0 for every query: the loop falls outside all three properties. Every proof goal inside the loop assumes the invariant, by the cut rule: from it, TensorLean proves that each task's reads of the block table and the KV cache, and its writes of the logits, stay within their tensors.

`(q, kv)`: the task of `query[q]` from KV block `kv``lens[q]`: tokens of `query[q]``num_kv`: KV blocks of `query[q]``meta[k]`: first task of CTA `k`, set before the launch

Tasks of each query

[Inline SVG figure not converted; see the preserved HTML. Label: Five queries, one row each, with 10, 4, 15, 6 and 8 KV blocks; each row cut into tasks of 4 blocks, each task in the color of the CTA that runs it. Text in figure: kv ; 0 ; 4 ; 8 ; 12 ; num_kv ; query[0] ; 10 ; query[1] ; 4 ; query[2] ; 15 ; query[3] ; 6 ; query[4] ; 8]

one KV block, 64 tokensone task, 4 blocks

Tasks of each CTA

CTA 0

CTA 1

CTA 2

CTA 3

CTA 2 at `(2, 12)`

Loop Invariant 1`(q, kv) < meta[3]`✓`(2, 12) < (3, 4)`

Loop Invariant 2`num_kv == ceil_div(lens[q], 64)`✓`15 == ceil_div(930, 64)`

Loop Invariant 3`kv % 4 == 0 && (kv == 0 || kv < num_kv)`✓`12 % 4 == 0 && 12 < 15`

Figure 5. The tasks of one launch of DeepGEMM's paged MQA kernel and the CTAs that run them. At each task of each CTA's loop, the three assertions hold; click a task to see them there.

In total, we wrote 298 assertions to verify the kernels of FlashAttention-2 to -4 and DeepGEMM.

Our paper will present the full technical details and usage of TensorLean.

## FlashAttention and Beyond

To demonstrate end-to-end verification capability, we use TensorLean to define the formal specifications of the entire FlashAttention-2 to -4 family and verify its kernels at the PTX level.
In four years, FlashAttention has gone through four generations, from Ampere to Blackwell, and each has set the state of the art in attention kernels.
As the de facto attention implementation, it has been widely deployed in production systems for both training and serving large language models, including [PyTorch](https://github.com/pytorch/pytorch), [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang).

This central role makes FlashAttention's correctness critical: a single bug in it could put the training and serving of every model on these systems at risk.
With TensorLean, we formally specified and verified all 37 kernels of FlashAttention-2 to -4. We verified each kernel on the PTX of at least one compiled instance, 102 instances in total.


**Specification: From Math to Machine.** With TensorLean, we first defined the specifications of 37 FlashAttention kernels, from each kernel's mathematics to its memory layout.
For example, FlashAttention-2's `spec_math` reads as plainly as the PyTorch reference in its tests (simplified):

TensorLean Spec for FlashAttention-2

````
structure Attn (H G L D : ℕ) where
  q : Tensor ℝ [H * G, L, D]
  k v : Tensor ℝ [H, L, D]
  scale cap : ℝ
  mask : Tensor Bool [H * G, L, L]
spec_math input :=
  let k := repeat_interleave(k, G)
  let v := repeat_interleave(v, G)
  let S := scale * (q @ kᵀ)
  let S := cap * tanh(S / cap)
  let S := masked_fill(S, ~mask, -∞)
  softmax(S) @ v
````

PyTorch Reference in FlashAttention-2's Tests

````
def attention_ref(
    q,              # [H * G, L, D]
    k, v,           # [H, L, D]
    G, scale, cap,
    mask,           # [H * G, L, L]
):
    k = k.repeat_interleave(G, dim=0)
    v = v.repeat_interleave(G, dim=0)
    S = scale * (q @ k.mT)
    S = cap * torch.tanh(S / cap)
    S[~mask] = -torch.inf
    return S.softmax(dim=-1) @ v
````

The specification then defines tensors' memory layouts, e.g., for a paged KV cache:

````
{ tensor := state.k[b, p, g, d]              -- the key cache, indexed by b, p, g, d
  var    := k_ptr                            -- the source variable that holds it
  layout := block_table[b, p / page_size]    -- its offset from k_ptr:
              * k_batch_stride               --   the page that holds key p,
            + p % page_size * k_row_stride   --   key p's row in that page,
            + g * k_head_stride + d }        --   and its head and dimension
````

In total, it takes 6,242 lines of specification to define the correctness of 37 FlashAttention kernels, totaling 55,910 lines of source:

Code vs. Spec Size Comparison: FlashAttention

Source CodeTensorLean Specification

[Inline SVG figure not converted; see the preserved HTML. Label: Code vs. Spec Size Comparison: FlashAttention. Text in figure: FlashAttention-3, by Kernel ; 0 ; 2,500 ; 5,000 ; 7,500 ; 3:1 ; Prepare Varlen ; 6:1 ; Forward ; 6:1 ; Fwd Combine ; 6:1 ; Backward ; FlashAttention-2 to -4 ; 0 ; 15,000 ; 30,000 ; 45,000 ; 3:1 ; FA-2 ; 5:1 ; FA-3 ; 14:1 ; FA-4 ; Lines of Code]

Figure 6. Source and TensorLean specification sizes of FlashAttention-3's kernels and FlashAttention-2 to -4.

**Verification: In Minutes.** We verified these instances against the four properties on their PTX: data-race freedom, termination, functional correctness, and outputs within a floating-point tolerance. The three generations span two languages: FlashAttention-2 and -3 are written in CUTLASS's CuTe, and FlashAttention-4 in the CuTe DSL. Each exercises the asynchronous hardware of its GPU: FlashAttention-2 computes with `mma` on Ampere, overlapping it with `cp.async` copies; FlashAttention-3 specializes its warps into a producer that loads tiles by TMA and consumers that multiply them with WGMMA on Hopper, pipelined through mbarriers; and FlashAttention-4 adds Blackwell's tensor memory and `tcgen05` matrix multiplications. TensorLean verifies an instance in about 10 minutes at the median on a 16-core workstation. The largest, FlashAttention-3's forward kernel on Hopper with 31,402 proof goals, takes 50 minutes. Of the 102 instances, 56 are proved to satisfy all four properties. In each of the other 46, a proof goal fails on a bug in the kernel's source.

Verification Time of FlashAttention-3, by Kernel

Data-race freedomTerminationFunctional correctnessFloating-point tolerance

[Inline SVG figure not converted; see the preserved HTML. Label: Verification time of FlashAttention-3's kernels, by property. Text in figure: 0 ; 10 ; 20 ; 30 ; 40 ; 50 ; minutes per instance ; PrepareVarlen ; 4 min ; Forward ; 50 min ; FwdCombine ; 20 min ; BwdPreprocess ; 4 min ; Backward ; 27 min ; ConvertdQ ; 11 min]

Figure 7. The time TensorLean takes to verify one instance of each FlashAttention-3 kernel on Hopper, by property.

**Finding and Fixing Bugs.** In FlashAttention-2 to -4, TensorLean found 23 previously unknown bugs, and we have [reported all of them](https://github.com/TensorLean/TensorLean) to the developers. Beyond FlashAttention, we applied TensorLean to the kernels of DeepSeek's DeepGEMM, Triton's kernel library, Liger Kernel, and TileOPs, and verified 47 more kernel instances, from DeepGEMM and Liger Kernel. With end-to-end formal verification from math to machine, TensorLean can and has captured a wide range of correctness bugs in production kernels:

1. **Incorrect Algorithm Implementation** caused by kernel code that departs from its intended algorithm, such as an incorrect formula or condition, or an unhandled special case. The kernel is incorrect even without any of the bugs below.
2. **Numerical Bugs** caused by insufficient precision or range, or by numerical instability,
   leading to rounding error beyond the tolerance, overflow, NaN, or integer wraparound. The kernel is correct over the real arithmetic, but not in finite-precision arithmetic.
3. **Uninitialized Memory**, where the kernel reads shared memory before initializing it, or leaves outputs unwritten. These lead to incorrect results or nondeterministic control flow.
4. **Memory Access Errors** caused by an incorrect tensor memory or distributed layout, or a missing bounds check, leading to out-of-bounds accesses and memory corruption.
5. **Data Races** between two accesses to the same global/shared memory, at least one of them a write, when no barrier or fence orders them. Such a race can silently corrupt output tensors.
6. **Deadlocks**, where threads wait on each other in a cycle, or a wait is never matched by the arrivals it waits for, e.g., due to barrier divergence. Such a deadlock hangs the kernel.
7. **Undefined Behaviors**, such as a division by zero or a signed integer overflow in the kernel's source, or initializing an mbarrier that is already initialized, which the language or the PTX ISA leaves undefined, so the compiler or the GPU may do anything with it.
8. **Miscompilations**, where a kernel compiler such as NVCC, Triton, or CuTe DSL compiles a correct source into incorrect executable. Such a miscompilation can introduce any of the bugs above. TensorLean captures all of them by verifying the PTX.

[Figure 8](https://tensorlean.ai#figure-bug-categories) shows all 38 bugs by category and library. Memory access errors (19) and numerical bugs (13) are the most common, and every category has at least one bug. Hovering over or tapping a bar lists its bugs, each linked to its report.

Bugs Found, by Category

FlashAttentionDeepGEMMTritonTileLangLiger KernelTileOPs

[Inline SVG figure not converted; see the preserved HTML. Label: Bugs found, by category and library. Text in figure: 0 ; 5 ; 10 ; 15 ; 20 ; Incorrect Algorithm ; 3 ; Numerical Bug ; 13 ; Uninitialized Memory ; 5 ; Memory Access Error ; 19 ; Data Race ; 7 ; Deadlock ; 1 ; Undefined Behavior ; 5 ; Miscompilation ; 2]

Figure 8. Bugs found by TensorLean, by category and library; a bug of several categories is counted in each.


## Limitations

**Trusted Computing Base.** First, we trust NVIDIA's PTX ISA [[16](https://tensorlean.ai#ref-nvidia2026ptx)] and ptxas, and assume that TensorLean's PTX semantics is consistent with them. We also trust TensorLean's model of how tensor cores round, which the PTX ISA does not specify, for bounding the round-off errors of matrix multiplications. Second, we trust each kernel's specification and TensorLean's 1.8K lines-of-code PTX parser. Third, we trust the results of Lean's kernel [[17](https://tensorlean.ai#ref-moura2021lean4)] and the cvc5 [[18](https://tensorlean.ai#ref-barbosa2022cvc5)] and Z3 [[19](https://tensorlean.ai#ref-demoura2008z3)] SMT solvers.

**Verification Scope.** A TensorLean specification defines a kernel's outputs and in-place updates as mathematical functions of its inputs (`spec_math` and `spec_mut`), so it cannot yet express kernels that sample at random, such as dropout and top-p sampling [[20](https://tensorlean.ai#ref-ye2025flashinfer)]. TensorLean derives the happens-before order of the relaxed PTX memory model [[21](https://tensorlean.ai#ref-lustig2019ptx)] from barriers, mbarriers, and asynchronous copies, but does not yet model the orders that acquire and release operations establish; kernels that synchronize through atomic flags, including multi-GPU communication kernels such as NCCL's [[22](https://tensorlean.ai#ref-nvidia2026nccl)] and DeepEP's [[23](https://tensorlean.ai#ref-deepseek2025deepep)], are thus out of its scope.

**Incompleteness.** TensorLean is sound but can fail to verify a correct kernel because of missing input constraints in its specification, overly conservative round-off bounds, or SMT solver timeouts.

## Conclusion

We presented TensorLean, the first framework for formally verifying the correctness of production GPU kernels.
By decoupling mathematics from memory, TensorLean lets developers specify a kernel at the tensor level, with no more effort than writing a PyTorch reference.
Given the kernel's specification and PTX, TensorLean verifies the correctness automatically: it reduces concurrency to sequential execution, infers loop invariants over semirings, and proves correctness under tolerance.
With TensorLean, we verified 149 kernel instances on their PTX, covering all 37 kernels of FlashAttention-2 to -4 and more from DeepGEMM and Liger Kernel, and proved 83 of them correct. During verification, we found 38 bugs that the kernels' own tests never caught.
From math to machine, TensorLean leaves no step unproven.

## References

1. Jonathan T. Barron et al. [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642). arXiv:2610.31642, 2026.
2. Aaron Grattafiori et al. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). arXiv:2407.21783, 2024.
3. Ziming Zhou et al. [OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming). OSDI 2026.
4. OpenAI. [Unexpected responses from ChatGPT](https://status.openai.com/incidents/ssg8fh7sfyz3). OpenAI incident report, February 2024.
5. Anthropic. [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues). Anthropic Engineering, September 2025.
6. Gerwin Klein et al. [seL4: Formal Verification of an Operating-System Kernel](https://doi.org/10.1145/1743546.1743574). Communications of the ACM, 2010.
7. Andrea Lattuada et al. [Verus: A Practical Foundation for Systems Verification](https://www.microsoft.com/en-us/research/publication/verus-a-practical-foundation-for-systems-verification/). SOSP 2024.
8. Zihao Ye et al. [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629). arXiv:2608.12629, 2026.
9. [[Bug][FA4] compute_block_sparsity: data race on reduction_buffer can drop blocks with unmasked keys](https://github.com/Dao-AILab/flash-attention/issues/3195). FlashAttention GitHub issue #3195.
10. Leslie Lamport. [Time, Clocks, and the Ordering of Events in a Distributed System](https://doi.org/10.1145/359545.359563). CACM 1978.
11. Kshitij Dubey et al. [Equivalence Checking of ML GPU Kernels](https://doi.org/10.1145/3839470). OOPSLA 2026.
12. Mengdi Wu et al. [SentryWard: Formal Verification of Warp-Specialized GPU Kernels](https://sentryward.io/blog/). 2026.
13. Martin Brain et al. [Building Better Bit-Blasting for Floating-Point Problems](https://doi.org/10.1007/978-3-030-17462-0_5). TACAS 2019.
14. Alexey Solovyev et al. [Rigorous Estimation of Floating-Point Round-Off Errors with Symbolic Taylor Expansions](https://doi.org/10.1145/3230733). TOPLAS 2018.
15. K. Rustan M. Leino. [Dafny: An Automatic Program Verifier for Functional Correctness](https://doi.org/10.1007/978-3-642-17511-4_20). LPAR 2010.
16. NVIDIA. [Parallel Thread Execution ISA Version 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/). NVIDIA CUDA Documentation.
17. Leonardo de Moura and Sebastian Ullrich. [The Lean 4 Theorem Prover and Programming Language](https://doi.org/10.1007/978-3-030-79876-5_37). CADE 2021.
18. Haniel Barbosa et al. [cvc5: A Versatile and Industrial-Strength SMT Solver](https://doi.org/10.1007/978-3-030-99524-9_24). TACAS 2022.
19. Leonardo de Moura and Nikolaj Bjørner. [Z3: An Efficient SMT Solver](https://doi.org/10.1007/978-3-540-78800-3_24). TACAS 2008.
20. Zihao Ye et al. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005). MLSys 2025.
21. Daniel Lustig et al. [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://doi.org/10.1145/3297858.3304043). ASPLOS 2019.
22. NVIDIA. [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl). NVIDIA Developer.
23. DeepSeek-AI. [DeepEP: A High-Performance Communication Library for Machine Learning Training and Inference](https://github.com/deepseek-ai/DeepEP). GitHub repository.

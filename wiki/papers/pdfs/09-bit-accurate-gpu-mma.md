---
title: "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy"
source_url: "https://arxiv.org/abs/2511.10909"
full_text_url: "https://arxiv.org/html/2511.10909v3"
version: "2511.10909v3"
authors: ["Xie, Peichen", "Xu, Shuotao", "Wang, Yang", "Yang, Fan", "Yang, Mao"]
retrieved_date: "2026-11-20"
conversion: "HTML to Markdown with TeX math"
pdf: "09-bit-accurate-gpu-mma.pdf"
original_html: "sources/09-bit-accurate-gpu-mma.html"
---

# Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy

Peichen Xie
Affiliation: Microsoft Research
  
Shuotao Xu
Affiliation: Microsoft Research
  
Yang Wang
Affiliation: Microsoft Research
  
Fan Yang
Affiliation: Microsoft Research
  
Mao Yang
Affiliation: Microsoft Research

###### Abstract

Modern AI accelerators rely on matrix multiply-accumulate units (MMAUs), such as NVIDIA Tensor Cores and AMD Matrix Cores, to accelerate deep neural network workloads. MMAUs expose only instruction-level or API-level interfaces of matrix multiply-accumulate (MMA) operations, while leaving internal floating-point arithmetic behavior undocumented. Consequently, MMAUs across vendors and architectural generations often produce numerical discrepancies for identical inputs, and sometimes exhibit reduced numerical accuracy that can cause training instability. Diagnosing and understanding the root causes of these effects is challenging without white-box models of their arithmetic behavior. This paper proposes closed-loop feature probing (CLFP), a generic and systematic framework for constructing bit-accurate arithmetic behavior models of MMA operations. Based on this framework, we analyze all MMA instructions on ten GPU architectures spanning NVIDIA Volta through RTX Blackwell and AMD CDNA1 through CDNA3, and derive the first bit-accurate arithmetic models for these MMAUs. Our models explain previously observed cross-platform numerical discrepancies and accuracy issues, enable white-box numerical error analysis, reveal four types of precision bottlenecks and one type of numerical asymmetry, and inform software workarounds as well as design suggestions for future MMAUs. This work is open-source at <https://github.com/microsoft/MMA-Sim>.

## I Introduction

Deep neural networks (DNNs) are built on massive numbers of linear algebra operations.
To satisfy the rapidly growing computational demands of modern DNN workloads such as large language models (LLMs), recent AI accelerators have introduced specialized hardware units such as NVIDIA Tensor Cores [[1](https://arxiv.org/html/2511.10909v3#bib.bib15)] and AMD Matrix Cores [[2](https://arxiv.org/html/2511.10909v3#bib.bib16)] to accelerate matrix multiplications.
These matrix multiply-accumulate units (MMAUs) now dominate both training and inference workloads because of their extremely high throughput, efficiency, and support for mixed-precision arithmetic.
As DNN workloads scale in model size and deployment scope, the numerical behavior of MMAUs becomes increasingly critical since small discrepancies can accumulate across a large number of operations and affect reproducibility and training stability (§[II](https://arxiv.org/html/2511.10909v3#S2)).

Despite nominally performing the same mathematical operation, MMAUs from different AI accelerators are not numerically equivalent.
Even with bit-identical inputs, different MMAU implementations can produce different floating-point results.
Such discrepancies are pervasive across vendors [[3](https://arxiv.org/html/2511.10909v3#bib.bib12)] and also occur across architectural generations from the same vendor [[4](https://arxiv.org/html/2511.10909v3#bib.bib8)].
These hardware-level numerical discrepancies could propagate through the software stack and manifest as inconsistencies in end-to-end DNN workloads, resulting in poor reproducibility across different AI accelerators.

Compounding the problem, the underlying arithmetic behavior of MMAUs, i.e., the hardware-specific floating-point computation mechanism for matrix multiply-accumulate (MMA) operations, is poorly documented and effectively a black box, leaving developers without a clear understanding of how numerical results are produced.
As a result, the effect of arithmetic behavior on numerical accuracy is often discovered only after it surfaces in large-scale systems.
Recent incidents illustrate this issue.
Researchers and developers reported that certain MMA instructions on NVIDIA Hopper GPUs and AMD CDNA2 GPUs degraded training accuracy and stability due to insufficient summation precision or subnormal flushing [[5](https://arxiv.org/html/2511.10909v3#bib.bib3), [6](https://arxiv.org/html/2511.10909v3#bib.bib20), [7](https://arxiv.org/html/2511.10909v3#bib.bib4), [8](https://arxiv.org/html/2511.10909v3#bib.bib27)].
In both cases, the root causes lie in the obscure arithmetic behavior of MMAUs.

These observations reveal a fundamental gap: the lack of a white-box understanding of MMAU arithmetic behavior.
Without such a foundation, numerical analysis remains empirical and reactive, making it difficult to predict accuracy, diagnose issues, or design principled mitigations.
A systematic white-box model of MMAU arithmetic behavior is therefore essential for rigorous and proactive reasoning about the numerical behavior of modern AI systems.

However, obtaining a white-box model for MMAUs is inherently difficult.
MMAUs expose only high-level MMA interfaces at the API level or at the instruction level, while the arithmetic features that determine numerical results remain hidden inside proprietary hardware.
These features include computational order, fusion granularity, internal precision, rounding behavior, and many other subtle design choices.
Because the relevant feature set is neither explicitly documented nor straightforward to enumerate exhaustively, the space of possible arithmetic behavior is effectively enormous, making exhaustive black-box exploration challenging.
For example, prior feature-probing studies infer isolated or partial arithmetic features for a few architectures, but none can provide a model that covers the complete feature set and is verified end-to-end for bit accuracy (§[II](https://arxiv.org/html/2511.10909v3#S2)).

In this paper, we tackle this challenge with a generic, systematic, and principled approach for constructing white-box models of MMA arithmetic behavior through black-box testing.
In particular, we propose closed-loop feature probing (CLFP), a framework that integrates arithmetic feature probing with iterative refinement (§[III](https://arxiv.org/html/2511.10909v3#S3)).
Given an MMA interface, we carefully design edge-case inputs to probe arithmetic features.
Based on the probed features, we build an executable arithmetic behavior model and then iteratively refine it through a verify-revise loop until the model is verified to reproduce the outputs of the MMA interface bit-for-bit.
This closed-loop methodology enables complete and accurate modeling of black-box MMAU arithmetic behavior.

In this work, we apply CLFP to instruction-level interfaces of GPU MMAUs.
Using the CLFP framework, we systematically analyze all floating-point MMA instructions on ten GPU architectures, spanning NVIDIA’s architectures from Volta to RTX Blackwell and AMD’s architectures from CDNA1 to CDNA3, and construct white-box models of their arithmetic behavior (§[IV](https://arxiv.org/html/2511.10909v3#S4)).
Our models accurately reveal the complete arithmetic behavior of MMAUs for the first time, showing that their MMA operations are composed of different types of elementary floating-point operations with different precisions, rounding modes, computational orders, and fusion granularities.

These white-box models explain previously observed numerical discrepancies across accelerators and expose the design choices that can affect numerical accuracy. For example, the models explain how the MMAUs produce six different output values ($0.0$, $-0.375$, $-0.5$, $-0.75$, $-0.875$, and $-1.0$) for the same input (§[V](https://arxiv.org/html/2511.10909v3#S5)). Our white-box models also enable us to quantitatively analyze the sources of numerical errors, revealing four types of precision bottlenecks and one type of asymmetry that significantly degrade the numerical accuracy of MMAUs (§[VI](https://arxiv.org/html/2511.10909v3#S6)). Based on the analysis, we provide software workarounds, mitigation methods, and design suggestions for future MMAUs.

In summary, we make the following contributions:

- A closed-loop feature probing framework.
  We introduce CLFP, a general methodology for deriving bit-accurate models of MMA arithmetic behavior through feature probing and iterative verification and revision.
- Bit-accurate models of GPU MMAUs.
  We construct the first bit-accurate arithmetic behavior models for all floating-point MMA instructions on NVIDIA and AMD GPUs, and implement our models as a GPU-accelerated arithmetic behavior simulator called MMA-Sim.
- MMAU numerical discrepancy analysis.
  We identify the differences in arithmetic behavior across MMAUs and reveal the root causes of numerical discrepancies beyond floating-point non-associativity.
- MMAU numerical accuracy analysis.
  We quantitatively analyze the design choices in MMAUs and their contributions to numerical errors, and provide workarounds and design suggestions.
- Open-source release.
  We open source our implementations and results to facilitate continuous testing, numerical analysis, and design space exploration.

## II Background and Related Work

### II-A Numerical Discrepancy and Accuracy Concerns

Deep neural networks (DNNs) are built on massive numbers of matrix multiplication operations. As models continue to scale, especially in large language model (LLM) workloads, the numerical behavior of the underlying hardware becomes increasingly important. Small floating-point differences can accumulate and amplify across computational layers and iterations, affecting reproducibility, training stability, and model quality [[9](https://arxiv.org/html/2511.10909v3#bib.bib21), [10](https://arxiv.org/html/2511.10909v3#bib.bib22)].

Specifically, migrating workloads across hardware vendors or architectural generations can produce different floating-point results even when the same model and input data are used [[11](https://arxiv.org/html/2511.10909v3#bib.bib14), [12](https://arxiv.org/html/2511.10909v3#bib.bib26), [13](https://arxiv.org/html/2511.10909v3#bib.bib13)]. Such discrepancies complicate the reproduction of deep learning experiments and create practical challenges in porting workloads across hardware platforms [[14](https://arxiv.org/html/2511.10909v3#bib.bib5), [15](https://arxiv.org/html/2511.10909v3#bib.bib23)]. Although these discrepancies are often attributed to software factors, such as compiler options, random seeds, mixed-precision policies, and parallel execution orders, they can persist even after software stacks are carefully aligned [[16](https://arxiv.org/html/2511.10909v3#bib.bib25), [17](https://arxiv.org/html/2511.10909v3#bib.bib24), [18](https://arxiv.org/html/2511.10909v3#bib.bib6)], indicating that conventional software-level explanations are incomplete. To fully diagnose these issues, hardware-specific arithmetic behavior must also be considered.

More importantly, hardware-specific arithmetic behavior can also affect numerical accuracy, with consequences for training stability and model quality. For example, several researchers have reported that insufficient hardware precision in FP8 matrix multiplication degraded training accuracy on NVIDIA Hopper GPUs [[5](https://arxiv.org/html/2511.10909v3#bib.bib3), [6](https://arxiv.org/html/2511.10909v3#bib.bib20), [8](https://arxiv.org/html/2511.10909v3#bib.bib27)]. To improve training accuracy, they increased software accumulation precision and suggested increasing hardware accumulation precision to FP32. In addition, PyTorch developers found that DNN training may fail to converge on AMD CDNA2 GPUs because the hardware flushes subnormal values to zero during FP16 matrix multiplication [[7](https://arxiv.org/html/2511.10909v3#bib.bib4)]. To address this issue, they cast FP16 operands to BF16, thereby increasing the dynamic range at the cost of reduced numerical precision.

### II-B Black-Box Arithmetic Behavior of MMAUs

Recent AI accelerators execute deep learning workloads primarily on specialized matrix multiply-accumulate units (MMAUs), such as NVIDIA Tensor Cores [[1](https://arxiv.org/html/2511.10909v3#bib.bib15)] and AMD Matrix Cores [[2](https://arxiv.org/html/2511.10909v3#bib.bib16)]. These units provide high throughput and efficiency, but they also make the numerical behavior of DNN computation heavily dependent on hardware-specific matrix multiply-accumulate (MMA) operations. As a result, understanding numerical behavior in modern AI systems increasingly requires understanding the arithmetic behavior implemented by MMAUs.

Although MMAUs share the mathematical definition of MMA operations,



$$
D_{M\times N}=A_{M\times K}\times B_{K\times N}+C_{M\times N},
$$
Equation (1)



their hardware implementations are not necessarily numerically equivalent. The underlying arithmetic behavior of an MMAU is neither standardized nor well documented. Vendors may adopt different implementation details, such as computational orders, accumulation precisions, intermediate rounding modes, fusion granularities, and special-value handling rules, while exposing only MMA instructions or high-level APIs for MMA operations.

Nevertheless, these hidden design choices are critical because they determine how intermediate results are computed and rounded during mixed-precision execution. Consequently, two MMAUs that nominally perform the same MMA operation may still produce different bit-level outputs [[4](https://arxiv.org/html/2511.10909v3#bib.bib8), [3](https://arxiv.org/html/2511.10909v3#bib.bib12)]. Moreover, when numerical discrepancies or accuracy problems arise in software, their root causes are difficult to diagnose because the relevant arithmetic behavior is invisible at the software level.

### II-C Limitations of Existing Approaches

Prior work has made important progress in modeling the arithmetic behavior of MMAUs. Feature-probing studies [[19](https://arxiv.org/html/2511.10909v3#bib.bib10), [4](https://arxiv.org/html/2511.10909v3#bib.bib8), [20](https://arxiv.org/html/2511.10909v3#bib.bib7), [21](https://arxiv.org/html/2511.10909v3#bib.bib2), [22](https://arxiv.org/html/2511.10909v3#bib.bib11)] have produced approximate models of early-generation Tensor Cores and Matrix Cores through carefully designed numerical tests, and refuted the early model that represented dot products in Tensor Cores as a perfect binary tree of multiplication and addition operations [[23](https://arxiv.org/html/2511.10909v3#bib.bib9)]. While these studies provide valuable observations, few studies have derived instruction-level, white-box, bit-accurate models. For this purpose, existing approaches exhibit three important limitations.

First, prior studies inferred arithmetic behavior from MMA APIs rather than directly from instruction-level execution. API-level observations may incorporate software-controlled transformations, and therefore do not distinguish behavior originating in hardware from that introduced by software. For example, Hickmann et al. [[19](https://arxiv.org/html/2511.10909v3#bib.bib10)] attributed the software-dependent reduction order to hardware behavior, and Fasi et al. [[4](https://arxiv.org/html/2511.10909v3#bib.bib8)] attributed the hardware-specific FP16 output-rounding behavior to software.

Second, prior work covered only an incomplete subset of the arithmetic features, which is insufficient for constructing white-box executable models. Consequently, uncharacterized features, such as whether products are normalized before accumulation, which internal summation order is used, and how overflow and underflow are handled, introduce critical ambiguity in bit-accurate modeling.

Third, prior studies did not validate their models at the bit level. As a result, their models may contain false assumptions or inaccurate conclusions. For example, Hickmann et al. [[19](https://arxiv.org/html/2511.10909v3#bib.bib10)] concluded that FP32 subnormal values are treated as zero, which was later disproved by subsequent work [[4](https://arxiv.org/html/2511.10909v3#bib.bib8), [20](https://arxiv.org/html/2511.10909v3#bib.bib7), [22](https://arxiv.org/html/2511.10909v3#bib.bib11)].

Therefore, feature probing alone is insufficient for bit-accurate modeling. The field still lacks a systematic approach for deriving white-box executable models whose semantics are sufficiently complete and whose accuracy is validated against hardware at the bit level.

## III Methodology and Implementation

Given a black-box interface of a matrix multiply-accumulate (MMA) operation, we use the closed-loop feature probing (CLFP) framework to build the arithmetic behavior model $\Phi$ such that



$$
\Phi(A,B,C)=\text{MMA-Interface}(A,B,C),
$$
Equation (2)



for any input matrices $A$, $B$, and $C$ with shapes $M\times K$, $K\times N$, $M\times N$ and data types specified by the interface.

### III-A CLFP Workflow

As Figure [1](https://arxiv.org/html/2511.10909v3#S3.F1) shows, the workflow of CLFP consists of five major steps. Through arithmetic feature testing and probing, Steps 1 and 2 deterministically decompose the MMA operation into low-level operations, and Step 3 probes the detailed arithmetic behavior of each low-level operation and constructs a model according to the probed behavior. However, the probing may be incomplete, so Step 4 leverages randomized differential testing to verify whether the model composed of the inferred low-level operations matches the output of the MMA interface bit for bit (Equation [2](https://arxiv.org/html/2511.10909v3#S3.E2)). If verification fails, Step 5 transforms the failing case into new feature-probing tests, revises the model accordingly, and returns to Step 4. The loop is repeated until verification succeeds.

Fig. 1: The closed-loop feature probing (CLFP) framework for modeling the arithmetic behavior of the MMA operation. The loop of Steps 4 (verification) and 5 (adding new tests and revising the model) ensures completeness and bit-accuracy.

#### III-A1 Step 1: Confirming Determinism and Cross-Position Consistency

Our method assumes that the MMA interface is deterministic. This assumption can be easily verified by randomized testing, and all the MMA interfaces in our experiments have been verified to be deterministic.

Then, we check whether the computation of



$$
d_{i,j}=c_{i,j}+\sum_{k=0}^{K-1}a_{i,k}b_{k,j}
$$
Equation (3)



depends on the indices $i$ and $j$. If not, the computation is consistent across output positions, so we can decompose the MMA operation into $M\times N$ independent dot-product-accumulate operations described by Equation [3](https://arxiv.org/html/2511.10909v3#S3.E3) and we only need to model one dot-product-accumulate operation in the remaining steps. Otherwise, we should fall back to per-element modeling.

To verify the computational consistency across output positions, we first generate $2K+1$ random numbers: $a_{0,0},\dots$, $a_{0,K-1}$, $b_{0,0},\dots$, $b_{K-1,0}$, and $c_{0,0}$. Then, we let $a_{i,k}=a_{0,k}$, $b_{k,j}=b_{k,0}$, and $c_{i,j}=c_{0,0}$ for all $0\leq i<M$, $0\leq j<N$, and $0\leq k<K$. If the computation does not depend on the indices $i$ and $j$, all $d_{i,j}$ in the output should be bitwise identical; otherwise, elements with different indices may differ in their values.

Through extensive testing, we find that all $d_{i,j}$ in the output matrix are bitwise identical, confirming that each output element in $D$ is computed with consistent arithmetic behavior. Therefore, in the remainder of the paper, we simplify the notation for the dot-product-accumulate operations for any $d_{i,j}$ to



$$
d=c+\sum_{k=0}^{K-1}a_{k}b_{k}.
$$
Equation (4)



#### III-A2 Step 2: Determining Summation Order and Arity

This step decomposes the dot-product-accumulate operation (Equation [4](https://arxiv.org/html/2511.10909v3#S3.E4)) into a combination of summation operations and determines how they are combined.

In particular, Equation [4](https://arxiv.org/html/2511.10909v3#S3.E4) can be transformed to



$$
d=\sum_{k=0}^{K}p_{k}\text{, where }p_{k}=\begin{cases}a_{k}b_{k}&\text{if $0\leq k<K$}\\
c&\text{if $k=K$}\end{cases}.
$$
Equation (5)



The summation order of Equation [5](https://arxiv.org/html/2511.10909v3#S3.E5) can be represented by a summation tree with $K+1$ leaf nodes, where a node with $n$ child nodes represents an $n$-ary summation operation. To infer the tree, we construct the following inputs for Equation [5](https://arxiv.org/html/2511.10909v3#S3.E5).

Adapting the method from FPRev [[21](https://arxiv.org/html/2511.10909v3#bib.bib2)], we choose a very large number $U=2^{e_{u}}$ and a very small number $v=2^{e_{v}}$ such that



$$
(K-1)v\pm U=\pm U
$$
Equation (6)



in floating-point arithmetic, which means that $\pm U$ can serve as large summands to swamp other small summands in summation. Then, for every $0\leq i<j\leq K$, we construct the inputs $a^{(i,j)}_{0},\dots,a^{(i,j)}_{K-1}$, $b^{(i,j)}_{0},\dots,b^{(i,j)}_{K-1}$, and $c^{(i,j)}$ such that



$$
p^{(i,j)}_{k}=\begin{cases}U&\text{if $k=i$}\\
-U&\text{if $k=j$}\\
v&\text{otherwise}\end{cases}.
$$
Equation (7)



The output for these inputs is denoted by $d^{(i,j)}$, and $0\leq d^{(i,j)}<Kv$. The value of $d^{(i,j)}/v$ represents the number of summands not swamped by the large summands $p_{i}$ and $p_{j}$, because any summation involving $\pm U$ before the cancellation of $-U+U=0$ results in $\pm U$ (Equation [6](https://arxiv.org/html/2511.10909v3#S3.E6)) and only the summation after this cancellation contributes to the output, as demonstrated in Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2). For example, if the summation tree follows Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2)(a) and $i=0$, $j=1$, then the summation is computed as $(((c+a_{0}b_{0})+a_{1}b_{1})+a_{2}b_{2})+a_{3}b_{3}=(((v+U)-U)+v)+v=2v$. Therefore, $d^{(i,j)}/v=2$, indicating that only two summands (i.e., $a_{2}b_{2}$ and $a_{3}b_{3}$) are not swamped by $p_{0}=a_{0}b_{0}$ and $p_{1}=a_{1}b_{1}$.

According to FPRev [[21](https://arxiv.org/html/2511.10909v3#bib.bib2)], the values $d^{(i,j)}/v$ for all $0\leq i<j\leq K$ uniquely determine a summation tree. Therefore, we can infer the summation tree by enumerating all possible trees and checking whether each tree can produce the observed outputs. Alternatively, we can infer the summation tree using a tree realization algorithm based on FPRev. Specifically, FPRev’s algorithm assumes that all fused summation operations are performed with limited precision:



$$
\text{FusedSum}_{\text{LP}}(U,m_{1}v,\dots,m_{n-2}v,-U)=0.
$$
Equation (8)



However, we find that high-precision fused summation can break this assumption:



$$
\text{FusedSum}_{\text{HP}}(U,m_{1}v,\dots,m_{n-2}v,-U)=\sum_{k=1}^{n-2}m_{k}v.
$$
Equation (9)



Therefore, FPRev cannot correctly infer the cases based on high-precision fused summation, such as the mfma_f32_32x32x4bf16 instruction on AMD CDNA1 GPUs. Taking both cases (Equations [8](https://arxiv.org/html/2511.10909v3#S3.E8) and [9](https://arxiv.org/html/2511.10909v3#S3.E9)) into account, we extend the algorithm of FPRev to distinguish between limited-precision and high-precision fused summation by adding corresponding conditional checks. For example, Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2)(d) shows the various values of $d^{(i,j)}/v$ corresponding to the same summation tree with high-precision and limited-precision summation.

| $U,-U$ | $a_{1}b_{1}$ | $a_{2}b_{2}$ | $a_{3}b_{3}$ | $c$ |
| --- | --- | --- | --- | --- |
| $a_{0}b_{0}$ | 2 | 1 | 0 | 3 |
| $a_{1}b_{1}$ |  | 1 | 0 | 2 |
| $a_{2}b_{2}$ |  |  | 0 | 1 |
| $a_{3}b_{3}$ |  |  |  | 0 |

(a) Chain of binary summation (e.g., AMD CDNA1 mfma_f32_16x16x4f32 and NVIDIA DMMA.884 instructions).

| $U,-U$ | $a_{1}b_{1}$ | $a_{2}b_{2}$ | $a_{3}b_{3}$ | $c$ |
| --- | --- | --- | --- | --- |
| $a_{0}b_{0}$ | 3 | 1 | 1 | 0 |
| $a_{1}b_{1}$ |  | 1 | 1 | 0 |
| $a_{2}b_{2}$ |  |  | 3 | 0 |
| $a_{3}b_{3}$ |  |  |  | 0 |

(b) Pairwise summation and accumulation (e.g., AMD CDNA2 mfma_f32_32x32x4bf16_1k instruction).

| $U,-U$ | $a_{1}b_{1}$ | $a_{2}b_{2}$ | $a_{3}b_{3}$ | $c$ |
| --- | --- | --- | --- | --- |
| $a_{0}b_{0}$ | 3 | 1 | 1 | 3 |
| $a_{1}b_{1}$ |  | 1 | 1 | 3 |
| $a_{2}b_{2}$ |  |  | 3 | 1 |
| $a_{3}b_{3}$ |  |  |  | 1 |

(c) 3-term fused summation, high precision (e.g., AMD CDNA1 mfma_f32_32x32x4bf16 instruction).

| $U,-U$ | $a_{1}b_{1}$ | $a_{2}b_{2}$ | $a_{3}b_{3}$ | $c$ |
| --- | --- | --- | --- | --- |
| $a_{0}b_{0}$ | 3/0 | 3/0 | 3/0 | 3/0 |
| $a_{1}b_{1}$ |  | 3/0 | 3/0 | 3/0 |
| $a_{2}b_{2}$ |  |  | 3/0 | 3/0 |
| $a_{3}b_{3}$ |  |  |  | 3/0 |

(d) 5-term fused summation, high precision (e.g., AMD CDNA3 mfma_f32_16x16x4_4b_bf16 instruction) / limited precision (e.g., NVIDIA HMMA.884 instruction).

Fig. 2: Examples of summation trees (left) and corresponding values of $d^{(i,j)}/v$ (right), where $(i,j)$ denotes the input with $p_{i}=U$, $p_{j}=-U$, and all other summands set to $v$.

#### III-A3 Step 3: Probing Detailed Arithmetic Features

This step probes the detailed arithmetic behavior of the summation operations in the summation tree constructed in Step 2, aiming at inferring how the operations are performed and building an executable model for the MMA operation. For each arithmetic feature, we construct edge-case inputs and infer the feature from the output. We maintain a collection of feature-probing tests (FPTs). These feature-probing tests are reusable. Once a feature-probing test is manually built, it can be applied to subsequent MMA modeling. The collection is extensive and we showcase the four most important tests here. The remaining tests mainly check binary properties such as whether subnormal values are supported.

##### FPT 1: Summation Precision

For every binary addition in the summation tree, we use $\text{Add}(U,\epsilon)$ to test its precision. Specifically, for the addition operation involving $p_{i}$ and $p_{j}$, we let $p_{i}=U$, $p_{j}=\epsilon=0.5U$, and set other summands to zero. Next, we repeatedly halve $\epsilon$ until the output satisfies $d\neq U+\epsilon$. Then, we conclude that the summation precision of this operation is $e_{u}-\log_{2}\epsilon-1$ fractional bits.

For every $n$-term fused summation, we use $\text{FusedSum}(U,-U,\epsilon)$ to test the precision of the operation. Specifically, for the summation operation involving $p_{i}$, $p_{j}$, and $p_{k}$, we let $p_{i}=U$, $p_{j}=-U$, and $p_{k}=\epsilon=0.5U$, and set other summands to zero. Next, we repeatedly halve $\epsilon$ until $\epsilon=0$ or the output $d\neq\epsilon$. If $d=\epsilon$ remains true until $\epsilon=0$, this operation is exact (as if computed with infinite precision). Otherwise, the summation precision is $e_{u}-\log_{2}\epsilon-1$ fractional bits.

##### FPT 2: Rounding Mode

For every summation operation with limited precision, we determine the rounding mode of the operation by testing four sets of inputs: $U+1.5\epsilon$, $U+0.5\epsilon$, $-U-1.5\epsilon$, and $-U-0.5\epsilon$. According to the outputs ($U$ or $U+2\epsilon$, and $-U$ or $-U-2\epsilon$), we can identify the rounding mode as one of “round toward $+\infty$” (a.k.a. “round up” or RU), “round toward $-\infty$” (a.k.a. “round down” or RD), “round toward zero” (RZ), “round away from zero” (RA), and “round to nearest” (RN).

If the rounding mode is an RN mode, we test four additional sets of inputs $U+\epsilon$, $U+3\epsilon$, $-U-\epsilon$, and $-U-3\epsilon$. According to the outputs ($U$ or $U+2\epsilon$ or $U+4\epsilon$, and $-U$ or $-U-2\epsilon$ or $-U-4\epsilon$), we can identify the tie-breaking rule as one of “ties toward $+\infty$” (a.k.a. “ties up” or RNU), “ties toward $-\infty$” (a.k.a. “ties down” or RND), “ties toward zero” (RNZ), “ties away from zero” (RNA), “ties to even” (RNE), and “ties to odd” (RNO).

##### FPT 3: Internal Summation Order

A limited-precision $n$-term fused summation operation indicates that intermediate sums are not normalized to floating-point values and only the final sum of $n$ summands is normalized. However, the order of the pre-normalization summation is not specified. To infer the fine-grained summation order within the fused summation operation, we adapt the tests in Step 2 by setting $v=\epsilon$. With this choice, $\pm U$ still swamp individual summands when $\pm U+\epsilon=\pm U$ and the rounding mode applied to $\epsilon$ is RZ, RD, RNE, RNZ, or RND, while the cancellation of $-U+U$ preserves the partial sums produced by earlier additions, yielding $\text{FusedSum}_{\text{LP}}(U,m_{1}v,\dots,m_{n-2}v,-U)=\sum_{m_{k}>1}\text{round}(m_{k}\epsilon)$. With $d^{(i,j)}/v$ for all $0\leq i<j<n$, we can infer the internal summation tree within the operation.

##### FPT 4: Product Renormalization

The exponent of a summand determines the position of rounding in a limited-precision $n$-term fused summation operation. For $a_{i}b_{i}$, we check whether the product is renormalized before summation by replacing the $\pm U$ with $\pm 1.5\times 1.5\times 2^{e_{u}}$ in $\text{FusedSum}(U,-U,2\epsilon)$. If the result remains $2\epsilon$, it implies that the product $\pm 2.25\times 2^{e_{u}}$ is not renormalized and the exponent remains $e_{u}$. Otherwise, if the result changes, it implies that the product is renormalized to $\pm 1.125\times 2^{e_{u}+1}$.

#### III-A4 Step 4: Verification

The model constructed in Step 3 may need to be revised because the probed arithmetic features may be incomplete and the constructed model may include empirical assumptions that are not verified. Therefore, Step 4 verifies the correctness of the model through randomized differential testing. To extend coverage, our testing includes three types of inputs:

1. Random inputs from common distributions, including the normal distribution, the uniform distribution, and typical distributions in DNN workloads such as $N(0,1)+\text{Bernoulli}(0.001)\cdot N(0,10^{2})$ [[24](https://arxiv.org/html/2511.10909v3#bib.bib17)].
2. Random inputs with large condition numbers (i.e., $\frac{\sum|p_{k}|}{|\sum p_{k}|}\gg 1$) that can trigger catastrophic cancellation [[25](https://arxiv.org/html/2511.10909v3#bib.bib19)].
3. Inputs generated from randomized bit-streams, which cover the most diverse inputs including normal numbers with widely varying magnitudes, subnormal numbers, infinities, and NaNs. In practice, this is the most useful input type in our experiments.

Note that edge cases have been integrated as feature-probing tests in Step 3.

If the model is verified to match the output of the MMA interface bit for bit after extensive tests, we complete the modeling for this MMA interface. Otherwise, a failing test case prompts us to revise our model in an additional step.

#### III-A5 Step 5: Generating New Feature-Probing Tests

When a mismatch occurs in Step 4, we investigate the failing case, transform this case into a new feature-probing test, and revise the model accordingly. With the revised model, CLFP returns to Step 4 to verify the correctness again and repeats this loop until verification passes.

We propose the minimum-failing-subset reduction technique to debug and transform the mismatch case. Specifically, we first zero some input elements and check whether the mismatch persists. After we identify the minimum subset of input elements that induce the mismatch (typically two to four elements), we tune the values of these elements and observe the behavior of both the interface output and the model output. The tuning methods include increasing/decreasing the exponent, truncating/flipping least significant bits, flipping the sign bit, zeroing some fractional bits, etc. The input values at which the outputs switch between matching and mismatching reveal the arithmetic behavior responsible for the mismatch. We convert these inputs to a new feature-probing test that can distinguish the edge cases, and add it to our test collection for future modeling. For example, FPT 4 is generated by the above technique from a mismatch case where the largest $a_{i}b_{i}$ satisfies $|\text{Sig}(a_{i})\text{Sig}(b_{i})|\geq 2$ (where Sig denotes the significand). We revise the model based on the probed behavior as in Step 3, and return to Step 4 for further verification.

### III-B Method Applicability

Our CLFP framework is applicable to general MMA operations. Given an interface of an MMA operation, we can build its arithmetic behavior model through CLFP. If the operation interface is at the API level, the derived model reveals the arithmetic behavior of the API. However, without knowing the instruction-level behavior first, it is hard for such a model to distinguish between hardware behavior and software behavior such as programmer intent and compiler transformations that determine which hardware instructions are used and how the instructions are composed.

In comparison, if the interface is at the instruction level, then the derived model reveals the hardware-specific behavior. Therefore, this paper focuses on the hardware matrix multiply-accumulate units (MMAUs) on NVIDIA and AMD GPUs, i.e., Tensor Cores and Matrix Cores. Specifically, this paper applies CLFP to all instruction-level MMA interfaces on ten GPU architectures: NVIDIA Volta (sm70), Turing (sm75), Ampere (sm80), Ada Lovelace (sm89), Hopper (sm90), Blackwell (sm100), and RTX Blackwell (sm120), and AMD CDNA1 (gfx908), CDNA2 (gfx90a), and CDNA3 (gfx942). These architectures represent almost all GPU architectures equipped with Tensor Cores or Matrix Cores as of 2025, which demonstrates the generality of CLFP.

In future work, we will apply CLFP to more MMAUs on other processors such as CPUs and TPUs [[26](https://arxiv.org/html/2511.10909v3#bib.bib18)], and also experiment with the API-level interfaces.

### III-C System Implementation

As Figure [1](https://arxiv.org/html/2511.10909v3#S3.F1) shows, our modeling system consists of three components: the CLFP framework, the MMA interface, and the arithmetic behavior model. All components of our system are open-source on GitHub.

We build the arithmetic behavior models as a software simulator called MMA-Sim. We implement the CLFP framework and MMA-Sim in Python. For NVIDIA devices, we implement the instruction-level MMA interfaces in CUDA and inline assembly (PTX), and verify the correctness of the mappings between PTX instructions (lowest-level programmable assembly) and SASS instructions (hardware instruction set). For AMD devices, we implement the interfaces in HIP and instruction intrinsic functions, and verify the correctness of the function-to-instruction mappings.

We run our experiments on the following GPUs: NVIDIA V100 (Volta), T4 (Turing), A100 (Ampere), RTX 4090 (Ada Lovelace), H100 (Hopper), B200 (Blackwell), and RTX PRO 6000 Blackwell (RTX Blackwell), and AMD MI100 (CDNA1), MI250X (CDNA2), and MI300X (CDNA3).

### III-D Modeling Efficiency and Simulator Optimization

CLFP is currently a semi-automated framework. Step 1, Step 2, running feature-probing tests in Step 3, and Step 4 are automated by Python scripts, while Step 5 and model construction in Step 3 are currently manual.

We break down the execution time of CLFP into three parts: (1) GPU execution time, which is less than one millisecond per call, (2) MMA-Sim execution time, which depends on the matrix size of the MMA instruction, and (3) human effort of debugging, adding feature-probing tests, and constructing/revising models.

We note that the execution of MMA-Sim in Step 4 is the most time-consuming part. The original MMA-Sim runs on a single thread and takes approximately 10-20 hours for one million tests in Step 4. As a result, modeling a GPU MMA instruction takes at least one day and sometimes several days depending on the number of unseen arithmetic features.

Therefore, to improve modeling efficiency, we implement GPU acceleration for MMA-Sim. We achieve a speedup of more than $10\times$ and reduce the MMA-Sim execution time from over 10 hours to approximately 1 hour in Step 4.

With the optimized MMA-Sim, modeling a GPU MMA instruction typically takes 2-12 hours, as shown in Table [I](https://arxiv.org/html/2511.10909v3#S3.T1). Steps 1-3 take a small fraction of time for running a fixed number of GPU tests and constructing a model. In Step 4, MMA-Sim takes approximately one hour and GPU execution takes several minutes. If a mismatch occurs, Step 5 is triggered and takes approximately four hours of human effort.

TABLE I: Approximate time of running CLFP with 0-2 revision loops (typical). \*MMA-Sim execution time has been reduced from 10-20 hours to 1 hour through optimization.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GPU | MMA-Sim | Human | Total |
| Step 1 | 5 minutes | 0 | 0 | 5 minutes |
| Step 2 | $<$1 minute | 0 | 0 | $<$1 minute |
| Step 3 | $<$1 minute | 0 | $<$1 hour | $<$1 hour |
| Step 4 | 5 minutes | 1 hour\* | 0 | 1 hour |
| Step 5 | $<$1 minute | $<$1 minute | 4 hours | 4 hours |
| Total | 10-20 minutes | 1-3 hours | 1-9 hours | 2-12 hours |

The total time of Steps 4 and 5 depends on the number of revision-verification loops. In practice, most instructions can be modeled with zero or one revision. We find that instructions within the same family (e.g., HMMA) exhibit similar arithmetic behavior, and once the first instruction is modeled, existing feature-probing tests are adequate for other instructions of the family and Steps 1-4 run in a single pass. For the first instruction of a new family that has unseen arithmetic features, modeling it typically takes one or two revisions. Only a few instructions require more revisions. The most complex instruction (the FP8 MMA instruction on AMD CDNA3 GPUs) takes ten revisions.

We note that the time for human effort in Step 5 has exceeded MMA-Sim execution time. Therefore, to further accelerate CLFP, we will automate the minimum-failing-subset reduction in Step 5 using Python scripts and try to use AI agents to automate the code writing in model construction and revision.

## IV Derived Bit-Accurate Models

This section presents our bit-accurate models built through the CLFP framework for all GPU matrix multiply-accumulate (MMA) instructions (also known as matrix fused-multiply-add or MFMA instructions on AMD GPUs). Our models are divided into three categories and eight types based on the elementary operations that compose them, as shown in Table [II](https://arxiv.org/html/2511.10909v3#S4.T2). Each model is composed of elementary operations. We define an elementary operation as an $n$-ary floating-point operation $f:\mathbb{F}_{1}\times\cdots\times\mathbb{F}_{n}\to\mathbb{F}_{\text{out}}$ that deterministically maps $n$ floating-point inputs to a single floating-point output. Within an elementary operation, intermediate values are represented in non-$\mathbb{F}_{\text{out}}$ formats, and intermediate computations are performed using non-$\mathbb{F}_{\text{out}}$ arithmetic.

TABLE II: Categories of our bit-accurate models for GPU matrix multiply-accumulate units.

|  |  |  |
| --- | --- | --- |
| Category | Models | Elementary Ops. |
| MulAdd-based | $\Phi_{\text{FTZ-MulAdd}}$ | FTZ-Add, FTZ-Mul |
| FMA-based | $\Phi_{\text{FMA}}$ | FMA |
| FDPA-based | |  | | --- | | $\Phi_{\text{E-FDPA}}$, $\Phi_{\text{T-FDPA}}$, | | $\Phi_{\text{ST-FDPA}}$, $\Phi_{\text{GST-FDPA}}$, | | $\Phi_{\text{TR-FDPA}}$, $\Phi_{\text{GTR-FDPA}}$ | | |  | | --- | | E-FDPA, T-FDPA, | | ST-FDPA, GST-FDPA, | | TR-FDPA, GTR-FDPA | |

### IV-A Models by Elementary Operations

#### IV-A1 Model Based on Binary Add and Mul Operations

The first category of our models is based on binary floating-point multiplication and addition operations. Through CLFP, we find that none of the GPU MMA instructions we examined is composed of standard (IEEE-754 [[27](https://arxiv.org/html/2511.10909v3#bib.bib1)]) multiplication and addition operations. In contrast, FP16 and BF16 MMA instructions on AMD CDNA2 GPUs are composed of flush-to-zero addition (FTZ-Add) and flush-to-zero multiplication (FTZ-Mul) operations, which are defined in Algorithm [1](https://arxiv.org/html/2511.10909v3#alg1). We also find that subnormal MMA inputs are flushed to positive zero, and subnormal outputs of multiplication and addition are flushed to zero with sign preserved. In addition, the order of summation in these instructions is shown in Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2)(b). After the FTZ-Mul operations, each group of $L$ consecutive products ($L\in\{2,4\}$) is summed pairwise using FTZ-Add, and then $c$ and the $K/L$ partial sums are summed sequentially using FTZ-Add. In summary, we build the $\Phi_{\text{FTZ-MulAdd}}$ model for these instructions as shown in Algorithm [2](https://arxiv.org/html/2511.10909v3#alg2).

Algorithm 1  FTZ-Add and FTZ-Mul operations.



````


x,y∈𝔽x,y\in\mathbb{F} ⊳\triangleright 𝔽∈{BF16, FP16, FP32}\mathbb{F}\in\text{\lx@text@lbrace BF16, FP16, FP32\lx@text@rbrace}


z∈FP32z\in\text{FP32}


if FTZ-Add then


  z←RNEFP32​(x+y)z\leftarrow\text{RNE}_{\text{FP32}}(x+y) ⊳\triangleright round to nearest, ties to even


else if FTZ-Mul then


  z←RNEFP32​(x×y)z\leftarrow\text{RNE}_{\text{FP32}}(x\times y)


z←z×0.0z\leftarrow z\times 0.0 if |z|<2−126|z|<2^{-126} ⊳\triangleright flush to zero with sign preserved
````




Algorithm 2  $\Phi_{\text{FTZ-MulAdd}}$: MMA model based on FTZ-Add and FTZ-Mul.



````


A∈𝔽M×KA\in\mathbb{F}^{M\times K}, B∈𝔽K×NB\in\mathbb{F}^{K\times N}, and C∈FP32M×NC\in\text{FP32}^{M\times N} (inputs); Lmax∈{2,4}L_{\text{max}}\in\{2,4\} (parameter) ⊳\triangleright 𝔽∈{BF16, FP16}\mathbb{F}\in\text{\lx@text@lbrace BF16, FP16\lx@text@rbrace}


D∈FP32M×ND\in\text{FP32}^{M\times N}


A←FlushSubnormal​(A)A\leftarrow\text{FlushSubnormal}(A) ⊳\triangleright flush input subnormals to +0.0+0.0


B←FlushSubnormal​(B)B\leftarrow\text{FlushSubnormal}(B)


C←FlushSubnormal​(C)C\leftarrow\text{FlushSubnormal}(C)


L←min⁡(K,Lmax)L\leftarrow\min(K,L_{\text{max}})


for each 0≤i<M0\leq i<M and 0≤j<N0\leq j<N in parallel do


  di,j←ci,jd_{i,j}\leftarrow c_{i,j}


  for k←0k\leftarrow 0 to K−LK-L in steps of LL do


   for each 0≤l<L0\leq l<L in parallel do


      pl←FTZ-Mul​(ai,k+l,bk+l,j)p_{l}\leftarrow\text{FTZ-Mul}(a_{i,k+l},b_{k+l,j})
   


   s←FTZ-Add​(p0,p1)s\leftarrow\text{FTZ-Add}(p_{0},p_{1})


   if L=4L=4 then


      s′←FTZ-Add​(p2,p3)s^{\prime}\leftarrow\text{FTZ-Add}(p_{2},p_{3})


      s←FTZ-Add​(s,s′)s\leftarrow\text{FTZ-Add}(s,s^{\prime})
   


   di,j←FTZ-Add​(di,j,s)d_{i,j}\leftarrow\text{FTZ-Add}(d_{i,j},s)
````



#### IV-A2 Model Based on Ternary FMA Operations

The second category is based on the ternary floating-point operation fused multiply-add (FMA). Through CLFP, we find that all FP64 MMA instructions on NVIDIA GPUs and all FP64 and FP32 MMA instructions on AMD GPUs are composed of FMA operations. As shown in Algorithm [3](https://arxiv.org/html/2511.10909v3#alg3), this operation takes three floating-point inputs $a$, $b$, and $c$, computes $a\times b+c$ as if with infinite precision, and converts the intermediate result to the output floating-point format using a specific rounding mode $\rho$. The order of operations in these instructions is a chain of FMAs, as shown in Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2)(a). The rounding mode is “round to nearest, ties to even” (RNE) by default. On NVIDIA GPUs, it can also be configured as “round toward zero” (RZ), “round up” (RU), or “round down” (RD). In summary, we build the $\Phi_{\text{FMA}}$ model for these instructions as shown in Algorithm [4](https://arxiv.org/html/2511.10909v3#alg4).

Algorithm 3  FMA operation.



````


a,b,c∈𝔽a,b,c\in\mathbb{F} (inputs); ρ∈{RNE𝔽,RZ𝔽,RU𝔽,RD𝔽}\rho\in\{\text{RNE}_{\mathbb{F}},\text{RZ}_{\mathbb{F}},\text{RU}_{\mathbb{F}},\text{RD}_{\mathbb{F}}\} (parameter) ⊳\triangleright 𝔽∈{FP64, FP32}\mathbb{F}\in\text{\lx@text@lbrace FP64, FP32\lx@text@rbrace}


d∈𝔽d\in\mathbb{F}


d←ρ⁡(a×b+c)d\leftarrow\rho(a\times b+c) ⊳\triangleright round to 𝔽\mathbb{F}
````




Algorithm 4  $\Phi_{\text{FMA}}$: MMA model based on FMA.



````


A∈𝔽M×KA\in\mathbb{F}^{M\times K}, B∈𝔽K×NB\in\mathbb{F}^{K\times N}, and C∈𝔽M×NC\in\mathbb{F}^{M\times N} ⊳\triangleright 𝔽∈{FP64, FP32}\mathbb{F}\in\text{\lx@text@lbrace FP64, FP32\lx@text@rbrace}


D∈𝔽M×ND\in\mathbb{F}^{M\times N}


for each 0≤i<M0\leq i<M and 0≤j<N0\leq j<N in parallel do


  di,j←ci,jd_{i,j}\leftarrow c_{i,j}


  for k←0k\leftarrow 0 to K−1K-1 do


   di,j←FMA​(ai,k,bk,j,di,j)d_{i,j}\leftarrow\text{FMA}(a_{i,k},b_{k,j},d_{i,j})
````



#### IV-A3 Models Based on N-ary FDPA Operations

The third category is based on fused dot-product-add (FDPA) operations. Typically, an FDPA operation takes $2L+1$ floating-point inputs (a pair of vectors of length $L$ and an accumulator) and produces one floating-point output. Through CLFP, we find that most mixed-precision MMA instructions on NVIDIA and AMD GPUs are composed of FDPA operations. Depending on whether $L=K$ or $L<K$, the computation of Equation [4](https://arxiv.org/html/2511.10909v3#S3.E4) can be composed of one or more FDPA operations, as shown in Figure [2](https://arxiv.org/html/2511.10909v3#S3.F2)(c) and (d). Therefore, we define the $\Phi_{\text{FDPA}}$ category in Algorithm [5](https://arxiv.org/html/2511.10909v3#alg5) for these instructions.

Algorithm 5  $\Phi_{\text{FDPA}}$: MMA models based on FDPA operations.



````


A∈𝔽AM×KA\in\mathbb{F}_{A}^{M\times K}, B∈𝔽BK×NB\in\mathbb{F}_{B}^{K\times N}, and C∈𝔽CM×NC\in\mathbb{F}_{C}^{M\times N} (inputs); LmaxL_{\text{max}} (parameter)


D∈𝔽DM×ND\in\mathbb{F}_{D}^{M\times N}


L←min⁡(K,Lmax)L\leftarrow\min(K,L_{\text{max}})


for each 0≤i<M0\leq i<M and 0≤j<N0\leq j<N in parallel do


  di,j←ci,jd_{i,j}\leftarrow c_{i,j}


  for k←0k\leftarrow 0 to K−LK-L in steps of LL do


   di,j←FDPA​(ai,k,…,ai,k+L−1,bk,j,…,bk+L−1,j,di,j)d_{i,j}\leftarrow\text{FDPA}(a_{i,k},\dots,a_{i,k+L-1},b_{k,j},\dots,b_{k+L-1,j},d_{i,j})
````



Through CLFP, we find that different vendors and instructions implement six variants of FDPA operations. Accordingly, we construct six model types in the framework of $\Phi_{\text{FDPA}}$, each corresponding to a distinct FDPA variant, as described below.

##### Exact FDPA (E-FDPA)

We build the exact FDPA (E-FDPA) operation to model BF16 and FP16 MMA instructions on AMD CDNA1 GPUs. Similar to the FMA operation, an E-FDPA operation computes $c+\sum_{k=0}^{L-1}a_{k}b_{k}$ as if with infinite precision and converts the intermediate result to the output floating-point format using the RNE rounding mode, as shown in Algorithm [6](https://arxiv.org/html/2511.10909v3#alg6).

Algorithm 6  E-FDPA operation.



````


a0,…,aL−1,b0,…,bL−1∈𝔽,c∈FP32a_{0},\dots,a_{L-1},b_{0},\dots,b_{L-1}\in\mathbb{F},c\in\text{FP32} ⊳\triangleright 𝔽∈{BF16, FP16}\mathbb{F}\in\text{\lx@text@lbrace BF16, FP16\lx@text@rbrace}


d∈FP32d\in\text{FP32}


d←RNEFP32​(c+∑k=0L−1ak​bk)d\leftarrow\text{RNE}_{\text{FP32}}(c+\sum_{k=0}^{L-1}a_{k}b_{k})
````



##### Truncated FDPA (T-FDPA)

We build the truncated FDPA (T-FDPA) operation to model most mixed-precision MMA instructions on NVIDIA GPUs. This operation is parameterized by $L$ (length of the vector), $F$ (precision of the fused summation), and $\rho$ (output rounding mode), and proceeds in three steps as shown in Algorithm [7](https://arxiv.org/html/2511.10909v3#alg7):

1. Compute the exact unnormalized products for $L$ pairs of multiplicands, where the signed significands are multiplied using fixed-point arithmetic and the exponents are added using integer arithmetic;
2. Compute the truncated fused sum for the $L$ products and the accumulator $c$, where these $L+1$ terms are aligned at their largest exponent, their significand bits are truncated to $F$ fractional bits (i.e., RZ to fixed-point format $\mathbb{Q}_{F}:=2^{-F}\mathbb{Z}$), and their sum is computed using fixed-point arithmetic;
3. Convert the summation result to the output floating-point format using the rounding mode $\rho$.

We report the probed parameters for each instruction in §[IV-C](https://arxiv.org/html/2511.10909v3#S4.SS3).

Algorithm 7  T-FDPA operation.



````


a0,…,aL−1∈𝔽Aa_{0},\dots,a_{L-1}\in\mathbb{F}_{A}, b0,…,bL−1∈𝔽Bb_{0},\dots,b_{L-1}\in\mathbb{F}_{B}, c∈𝔽Cc\in\mathbb{F}_{C} (inputs); LL, FF, ρ\rho (parameters) ⊳\triangleright 𝔽A,𝔽B∈{TF32, BF16, FP16, FP8, FP6, FP4}\mathbb{F}_{A},\mathbb{F}_{B}\in\text{\lx@text@lbrace TF32, BF16, FP16, FP8, FP6, FP4\lx@text@rbrace}


d∈𝔽Dd\in\mathbb{F}_{D} ⊳\triangleright 𝔽C,𝔽D∈{FP32, FP16}\mathbb{F}_{C},\mathbb{F}_{D}\in\text{\lx@text@lbrace FP32, FP16\lx@text@rbrace}


Step 1: Compute exact products


for each 0≤k<L0\leq k<L in parallel do


  sk←SignedSig​(ak)×SignedSig​(bk)s_{k}\leftarrow\text{SignedSig}(a_{k})\times\text{SignedSig}(b_{k}) ⊳\triangleright exact fixed-point product of signed significands


  ek←Exp​(ak)+Exp​(bk)e_{k}\leftarrow\text{Exp}(a_{k})+\text{Exp}(b_{k}) ⊳\triangleright sum of exponents


Step 2: Compute truncated fused sum of L+1L+1 terms


sL←SignedSig​(c)s_{L}\leftarrow\text{SignedSig}(c)


eL←Exp​(c)e_{L}\leftarrow\text{Exp}(c)


emax=max⁡(e0,e1,…,eL)e_{\text{max}}=\max(e_{0},e_{1},\dots,e_{L})


for each 0≤k≤L0\leq k\leq L in parallel do


  sk′←RZℚF​(sk×2ek−emax)s_{k}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{k}\times 2^{e_{k}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and truncate to FF fractional bits


S←∑k=0Lsk′S\leftarrow\sum_{k=0}^{L}s_{k}^{\prime} ⊳\triangleright exact fixed-point sum


Step 3: Convert to floating-point output


d←ρ⁡(S×2emax)d\leftarrow\rho(S\times 2^{e_{\text{max}}})
````



##### Scaled Truncated FDPA (ST-FDPA)

We build the scaled truncated FDPA (ST-FDPA) operation to model MXFP8/6/4 MMA instructions, as shown in Algorithm [8](https://arxiv.org/html/2511.10909v3#alg8). ST-FDPA extends T-FDPA by applying per-block scale factors to the inputs before the dot product. It takes $2L+3$ floating-point inputs ($L$ pairs of multiplicands, an accumulator, and two scale factors) and produces one floating-point output. In these instructions, the scale factors are represented in the E8M0 format, and their significands are always $1.0$. Therefore, based on T-FDPA, we only need to incorporate the exponents of the scale factors in the computation of the products.

Algorithm 8  ST-FDPA operation.



````


a0,…,aL−1∈𝔽Aa_{0},\dots,a_{L-1}\in\mathbb{F}_{A}, b0,…,bL−1∈𝔽Bb_{0},\dots,b_{L-1}\in\mathbb{F}_{B}, c∈FP32c\in\text{FP32}, α,β∈E8M0\alpha,\beta\in\text{E8M0} (inputs); LL, FF, ρ\rho (parameters) ⊳\triangleright 𝔽A,𝔽B∈{FP8, FP6, FP4}\mathbb{F}_{A},\mathbb{F}_{B}\in\text{\lx@text@lbrace FP8, FP6, FP4\lx@text@rbrace}


d∈FP32d\in\text{FP32}


Step 1: Compute exact products


for each 0≤k<L0\leq k<L in parallel do


  sk←SignedSig​(ak)×SignedSig​(bk)s_{k}\leftarrow\text{SignedSig}(a_{k})\times\text{SignedSig}(b_{k}) ⊳\triangleright exact fixed-point product of signed significands


  ek←Exp​(ak)+Exp​(bk)+Exp​(α)+Exp​(β)e_{k}\leftarrow\text{Exp}(a_{k})+\text{Exp}(b_{k})+\text{Exp}(\alpha)+\text{Exp}(\beta) ⊳\triangleright sum of exponents


Step 2: Compute truncated fused sum of L+1L+1 terms


sL←SignedSig​(c)s_{L}\leftarrow\text{SignedSig}(c)


eL←Exp​(c)e_{L}\leftarrow\text{Exp}(c)


emax=max⁡(e0,e1,…,eL)e_{\text{max}}=\max(e_{0},e_{1},\dots,e_{L})


for each 0≤k≤L0\leq k\leq L in parallel do


  sk′←RZℚF​(sk×2ek−emax)s_{k}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{k}\times 2^{e_{k}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and truncate to FF fractional bits


S←∑k=0Lsk′S\leftarrow\sum_{k=0}^{L}s_{k}^{\prime} ⊳\triangleright exact fixed-point sum


Step 3: Convert to floating-point output


d←ρ⁡(S×2emax)d\leftarrow\rho(S\times 2^{e_{\text{max}}})
````



##### Group-Scaled Truncated FDPA (GST-FDPA)

We build the group-scaled truncated FDPA (GST-FDPA) operation to model specific MXFP4 and NVFP4 MMA instructions. GST-FDPA takes $2L+1+2L/K_{\text{block}}$ floating-point inputs: a pair of vectors of length $L$, an accumulator, and scale factors for each block of $K_{\text{block}}$ consecutive elements in each vector. For MXFP4, $K_{\text{block}}=32$; for NVFP4, $K_{\text{block}}=16$.

In GST-FDPA, the vector elements are partitioned into groups of size $G$, where $G$ may or may not equal $K_{\text{block}}$. As shown in Algorithm [9](https://arxiv.org/html/2511.10909v3#alg9), GST-FDPA first computes the exact dot product per group in fixed-point arithmetic. Each resulting dot product is then multiplied by the signed significands of the corresponding scale factors, while the corresponding exponent is computed as the sum of the exponents of the scale factors. Finally, GST-FDPA computes the truncated fused sum of the $L/G$ results and $c$, and converts the sum to the output floating-point format.

Algorithm 9  GST-FDPA operation.



````


a0,…,aL−1,b0,…,bL−1∈FP4a_{0},\dots,a_{L-1},b_{0},\dots,b_{L-1}\in\text{FP4}, c∈FP32c\in\text{FP32}, α0,…,αL/Kblock−1,β0,…,βL/Kblock−1∈𝔽\alpha_{0},\dots,\alpha_{L/K_{\text{block}}-1},\beta_{0},\dots,\beta_{L/K_{\text{block}}-1}\in\mathbb{F} (inputs); LL, GG, FF, ρ\rho (parameters) ⊳\triangleright 𝔽∈{E8M0, UE4M3}\mathbb{F}\in\{\text{E8M0, UE4M3}\}


d∈FP32d\in\text{FP32}


Step 1: Compute exact dot products per group


for each 0≤g<L/G0\leq g<L/G in parallel do


  pg←∑k=g​G(g+1)​G−1ak×bkp_{g}\leftarrow\sum_{k=gG}^{(g+1)G-1}a_{k}\times b_{k} ⊳\triangleright exact fixed-point dot product


  sg←pg×SignedSig​(α⌊g​G/Kblock⌋)×SignedSig​(β⌊g​G/Kblock⌋)s_{g}\leftarrow p_{g}\times\text{SignedSig}(\alpha_{\lfloor gG/K_{\text{block}}\rfloor})\times\text{SignedSig}(\beta_{\lfloor gG/K_{\text{block}}\rfloor})


  eg←Exp​(α⌊g​G/Kblock⌋)+Exp​(β⌊g​G/Kblock⌋)e_{g}\leftarrow\text{Exp}(\alpha_{\lfloor gG/K_{\text{block}}\rfloor})+\text{Exp}(\beta_{\lfloor gG/K_{\text{block}}\rfloor}) ⊳\triangleright sum of exponents


Step 2: Compute truncated fused sum of L/G+1L/G+1 terms


sL/G←SignedSig​(c)s_{L/G}\leftarrow\text{SignedSig}(c)


eL/G←Exp​(c)e_{L/G}\leftarrow\text{Exp}(c)


emax=max⁡(e0,e1,…,eL/G)e_{\text{max}}=\max(e_{0},e_{1},\dots,e_{L/G})


for each 0≤g≤L/G0\leq g\leq L/G in parallel do


  sg′←RZℚF​(sg×2eg−emax)s_{g}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{g}\times 2^{e_{g}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and truncate to FF fractional bits


S←∑g=0L/Gsg′S\leftarrow\sum_{g=0}^{L/G}s_{g}^{\prime} ⊳\triangleright exact fixed-point sum


Step 3: Convert to floating-point output


d←ρ⁡(S×2emax)d\leftarrow\rho(S\times 2^{e_{\text{max}}})
````



##### Truncated Rounded FDPA (TR-FDPA)

We build the truncated rounded FDPA (TR-FDPA) operation as shown in Algorithm [10](https://arxiv.org/html/2511.10909v3#alg10) to model TF32, BF16, and FP16 MMA instructions on AMD CDNA3 GPUs. Compared with T-FDPA, TR-FDPA computes the truncated fused sum for only $L$ products (without $c$). Then, the truncated dot-product result and $c$ are aligned and then rounded down to different precisions ($\mathbb{Q}_{32}$ and $\mathbb{Q}_{F}$), followed by summation and normalization. Specifically, the dot-product result is rounded down to $32$ or more fractional bits. Using more than 32 fractional bits yields the same result because the sum is then normalized to the E8M31 floating-point format using the RD rounding mode. Finally, the result is converted to the output floating-point format.

Algorithm 10  TR-FDPA operation.



````


a0,…,aL−1∈𝔽Aa_{0},\dots,a_{L-1}\in\mathbb{F}_{A}, b0,…,bL−1∈𝔽Bb_{0},\dots,b_{L-1}\in\mathbb{F}_{B}, c∈FP32c\in\text{FP32} (inputs); LL, FF, ρ\rho (parameters) ⊳\triangleright 𝔽A,𝔽B∈{TF32, BF16, FP16}\mathbb{F}_{A},\mathbb{F}_{B}\in\text{\lx@text@lbrace TF32, BF16, FP16\lx@text@rbrace}


d∈FP32d\in\text{FP32}


Step 1: Compute exact products


for each 0≤k<L0\leq k<L in parallel do


  sk←SignedSig​(ak)×SignedSig​(bk)s_{k}\leftarrow\text{SignedSig}(a_{k})\times\text{SignedSig}(b_{k}) ⊳\triangleright exact fixed-point product of signed significands


  ek←Exp​(ak)+Exp​(bk)e_{k}\leftarrow\text{Exp}(a_{k})+\text{Exp}(b_{k}) ⊳\triangleright sum of exponents


Step 2: Compute truncated fused sum of LL terms


edot=max⁡(e0,e1,…,eL−1)e_{\text{dot}}=\max(e_{0},e_{1},\dots,e_{L-1})


for each 0≤k<L0\leq k<L in parallel do


  sk′←RZℚF​(sk×2ek−edot)s_{k}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{k}\times 2^{e_{k}-e_{\text{dot}}}) ⊳\triangleright align at edote_{\text{dot}} and truncate to FF fractional bits


T←∑k=0L−1sk′T\leftarrow\sum_{k=0}^{L-1}s_{k}^{\prime} ⊳\triangleright exact fixed-point sum


Step 3: Compute rounded sum of two terms


sc←SignedSig​(c)s_{c}\leftarrow\text{SignedSig}(c)


ec←Exp​(c)e_{c}\leftarrow\text{Exp}(c)


emax←max⁡(edot,ec)e_{\text{max}}\leftarrow\max(e_{\text{dot}},e_{c})


T′←RDℚ32​(T×2edot−emax)T^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{32}}(T\times 2^{e_{\text{dot}}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and round down to 3232 or more fractional bits


sc′←RDℚF​(sc×2ec−emax)s_{c}^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{F}}(s_{c}\times 2^{e_{c}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and round down to FF fractional bits


S×2E←RDE8M31​((T′+sc′)×2emax)S\times 2^{E}\leftarrow\text{RD}_{\text{E8M31}}((T^{\prime}+s_{c}^{\prime})\times 2^{e_{\text{max}}}) ⊳\triangleright normalize and round down to E8M31


Step 4: Convert to floating-point output


d←ρ⁡(S×2E)d\leftarrow\rho(S\times 2^{E})
````



##### Group-Truncated Rounded FDPA (GTR-FDPA)

For FP8 MMA instructions on AMD CDNA3 GPUs, we build the group-truncated rounded FDPA (GTR-FDPA) operation. Compared with TR-FDPA, GTR-FDPA computes the truncated fused sums for two groups: the even-indexed products and the odd-indexed products. It then aligns the results of the two groups to their largest exponent, rounds down the aligned significands to $F$ fractional bits, and computes their sum. This dot-product result and $c$ are then summed using a method similar to TR-FDPA’s rounded summation with additional truncation, as shown in Algorithm [11](https://arxiv.org/html/2511.10909v3#alg11).

Algorithm 11  GTR-FDPA operation.



````


a0,…,aL−1∈𝔽Aa_{0},\dots,a_{L-1}\in\mathbb{F}_{A}, b0,…,bL−1∈𝔽Bb_{0},\dots,b_{L-1}\in\mathbb{F}_{B}, c∈FP32c\in\text{FP32} (inputs); LL, FF, ρ\rho (parameters) ⊳\triangleright 𝔽A,𝔽B∈{FP8}\mathbb{F}_{A},\mathbb{F}_{B}\in\text{\lx@text@lbrace FP8\lx@text@rbrace}


d∈FP32d\in\text{FP32}


Step 1: Compute exact products


for each 0≤k<L0\leq k<L in parallel do


  sk←SignedSig​(ak)×SignedSig​(bk)s_{k}\leftarrow\text{SignedSig}(a_{k})\times\text{SignedSig}(b_{k}) ⊳\triangleright exact fixed-point product of signed significands


  ek←Exp​(ak)+Exp​(bk)e_{k}\leftarrow\text{Exp}(a_{k})+\text{Exp}(b_{k}) ⊳\triangleright sum of exponents


Step 2: Compute truncated fused sums of L/2L/2 terms


eeven=max⁡(e0,e2,…,eL−2)e_{\text{even}}=\max(e_{0},e_{2},\dots,e_{L-2})


eodd=max⁡(e1,e3,…,eL−1)e_{\text{odd}}=\max(e_{1},e_{3},\dots,e_{L-1})


for each 0≤k<L/20\leq k<L/2 in parallel do


  s2​k′←RZℚF​(s2​k×2e2​k−eeven)s_{2k}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{2k}\times 2^{e_{2k}-e_{\text{even}}}) ⊳\triangleright align at eevene_{\text{even}} and truncate to FF fractional bits


  s2​k+1′←RZℚF​(s2​k+1×2e2​k+1−eodd)s_{2k+1}^{\prime}\leftarrow\text{RZ}_{\mathbb{Q}_{F}}(s_{2k+1}\times 2^{e_{2k+1}-e_{\text{odd}}}) ⊳\triangleright align at eodde_{\text{odd}} and truncate to FF fractional bits


Teven←∑k=0L/2−1s2​k′T_{\text{even}}\leftarrow\sum_{k=0}^{L/2-1}s_{2k}^{\prime} ⊳\triangleright exact fixed-point sum


Todd←∑k=0L/2−1s2​k+1′T_{\text{odd}}\leftarrow\sum_{k=0}^{L/2-1}s_{2k+1}^{\prime}


Step 3: Compute rounded sum of two group sums


edot=max⁡(eeven,eodd)e_{\text{dot}}=\max(e_{\text{even}},e_{\text{odd}})


Teven′←RDℚF​(Teven×2eeven−edot)T_{\text{even}}^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{F}}(T_{\text{even}}\times 2^{e_{\text{even}}-e_{\text{dot}}}) ⊳\triangleright align at edote_{\text{dot}} and round down to FF fractional bits


Todd′←RDℚF​(Todd×2eodd−edot)T_{\text{odd}}^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{F}}(T_{\text{odd}}\times 2^{e_{\text{odd}}-e_{\text{dot}}})


T←Teven′+Todd′T\leftarrow T_{\text{even}}^{\prime}+T_{\text{odd}}^{\prime}


Step 4: Compute the final rounded sum


sc←SignedSig​(c)s_{c}\leftarrow\text{SignedSig}(c)


ec←Exp​(c)e_{c}\leftarrow\text{Exp}(c)


emax←max⁡(edot,ec)e_{\text{max}}\leftarrow\max(e_{\text{dot}},e_{c})


T′←RDℚ32​(T×2edot−emax)T^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{32}}(T\times 2^{e_{\text{dot}}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and round down to 3232 or more fractional bits


sc′←RDℚF​(sc×2ec−emax)s_{c}^{\prime}\leftarrow\text{RD}_{\mathbb{Q}_{F}}(s_{c}\times 2^{e_{c}-e_{\text{max}}}) ⊳\triangleright align at emaxe_{\text{max}} and round down to FF fractional bits


if ec<emax−F−1e_{c}<e_{\text{max}}-F-1 then


  sc′←0s_{c}^{\prime}\leftarrow 0 ⊳\triangleright additional truncation


S×2E←RDE8M31​((T′+sc′)×2emax)S\times 2^{E}\leftarrow\text{RD}_{\text{E8M31}}((T^{\prime}+s_{c}^{\prime})\times 2^{e_{\text{max}}}) ⊳\triangleright normalize and round down to E8M31 floating-point format


Step 5: Convert to floating-point output


d←ρ⁡(S×2E)d\leftarrow\rho(S\times 2^{E})
````



### IV-B Special Value Handling

All the elementary operations above handle NaNs and infinities correctly: $\text{NaN}+x=\text{NaN}$, $\text{NaN}\times x=\text{NaN}$, $\pm\infty+y=\pm\infty$, $\pm\infty+\mp\infty=\text{NaN}$, $\pm\infty\times z=\pm\infty\times\text{Sign}(z)$, and $\pm\infty\times 0=\text{NaN}$ for all $x\in\mathbb{F}$, $-\infty<y<\infty$, and $0<|z|<\infty$.

In T-FDPA, ST-FDPA, and GST-FDPA, NaN values are uniformly encoded as 0x7FFFFFFF for FP32 outputs and 0x7FFF for FP16 outputs, and zero-valued $c$ inputs are treated specially as $0.0\times 2^{-108-F}$ for FP32 or $0.0\times 2^{3-F}$ for FP16, rather than the raw representation $0.0\times 2^{-126}$ or $0.0\times 2^{-14}$.

The multiplication in TR-FDPA produces infinity when $|s_{k}\times 2^{e_{k}}|\geq 2^{128}$. No other intermediate operation can overflow or underflow. All rounding modes handle overflow or underflow in the standard way.

### IV-C Instruction-to-Model Mapping

For each GPU MMA instruction, we show its corresponding model and parameters derived through CLFP in Table [III](https://arxiv.org/html/2511.10909v3#S4.T3).

TABLE III: Models and parameters for all NVIDIA and AMD GPU MMA instructions. In addition, $G=16$ for all instructions modeled by $\Phi_{\text{GST-FDPA}}$.



<table>
<tbody>
<tr>
<td><span>Architecture</span></td>
<td><span>Instruction</span></td>
<td><span>Model</span></td>
<td><span>Input Type</span></td>
<td><span>Output Type</span></td>
<td>$L_{\text{max}}$</td>
<td>$F$</td>
<td>$\rho$</td></tr>
<tr>
<td rowspan="2"><span>Volta</span></td>
<td rowspan="2"><span>HMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP16</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>23</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>4</span></td>
<td><span>23</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="2"><span>Turing</span></td>
<td rowspan="2"><span>HMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP16</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="4"><span>Ampere</span></td>
<td><span>DMMA</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>Configurable</span></td></tr>
<tr>
<td rowspan="3"><span>HMMA</span></td>
<td rowspan="3"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>24</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="6"><span>Ada Lovelace</span></td>
<td><span>DMMA</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>Configurable</span></td></tr>
<tr>
<td rowspan="3"><span>HMMA</span></td>
<td rowspan="3"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>24</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="2"><span>QMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP8</span></td>
<td><span>FP32</span></td>
<td><span>16</span></td>
<td><span>13</span></td>
<td><span>RZ<sub>E8M13</sub></span></td></tr>
<tr>
<td><span>FP8</span></td>
<td><span>FP16</span></td>
<td><span>16</span></td>
<td><span>13</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="6"><span>Hopper</span></td>
<td><span>DMMA</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>Configurable</span></td></tr>
<tr>
<td rowspan="3"><span><span>HMMA</span>/<span>HGMMA</span></span></td>
<td rowspan="3"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="2"><span>QGMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP8</span></td>
<td><span>FP32</span></td>
<td><span>32</span></td>
<td><span>13</span></td>
<td><span>RZ<sub>E8M13</sub></span></td></tr>
<tr>
<td><span>FP8</span></td>
<td><span>FP16</span></td>
<td><span>32</span></td>
<td><span>13</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="8"><span>Blackwell</span></td>
<td><span>DMMA</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>Configurable</span></td></tr>
<tr>
<td rowspan="3"><span><span>HMMA</span>/<span>UTCHMMA</span></span></td>
<td rowspan="3"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="3"><span>UTCQMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP8/6/4</span></td>
<td><span>FP32</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP8/6/4</span></td>
<td><span>FP16</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td>$\Phi_{\text{ST-FDPA}}$</td>
<td><span>MXFP8/6/4</span></td>
<td><span>FP32</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>UTCOMMA</span></td>
<td>$\Phi_{\text{GST-FDPA}}$</td>
<td><span>MXFP4/NVFP4</span></td>
<td><span>FP32</span></td>
<td><span>64</span></td>
<td><span>35</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="8"><span>RTX Blackwell</span></td>
<td><span>DMMA</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>Configurable</span></td></tr>
<tr>
<td rowspan="3"><span>HMMA</span></td>
<td rowspan="3"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP16</span></td>
<td><span>16</span></td>
<td><span>25</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td rowspan="2"><span>QMMA</span></td>
<td rowspan="2"><span>$\Phi_{\text{T-FDPA}}$</span></td>
<td><span>FP8/6/4</span></td>
<td><span>FP32</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP8/6/4</span></td>
<td><span>FP16</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RNE<sub>FP16</sub></span></td></tr>
<tr>
<td><span>QMMA.SF</span></td>
<td>$\Phi_{\text{ST-FDPA}}$</td>
<td><span>MXFP8/6/4</span></td>
<td><span>FP32</span></td>
<td><span>32</span></td>
<td><span>25</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td><span>OMMA.SF</span></td>
<td>$\Phi_{\text{GST-FDPA}}$</td>
<td><span>MXFP4/NVFP4</span></td>
<td><span>FP32</span></td>
<td><span>64</span></td>
<td><span>35</span></td>
<td><span>RZ<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="3"><span>CDNA1</span></td>
<td rowspan="3"><span>v_mfma</span></td>
<td>$\Phi_{\text{FMA}}$</td>
<td><span>FP32</span></td>
<td><span>FP32</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="2"><span>$\Phi_{\text{E-FDPA}}$</span></td>
<td><span>BF16</span></td>
<td><span>FP32</span></td>
<td><span>2</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="5"><span>CDNA2</span></td>
<td rowspan="4"><span>v_mfma</span></td>
<td rowspan="2"><span>$\Phi_{\text{FMA}}$</span></td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP64</sub></span></td></tr>
<tr>
<td><span>FP32</span></td>
<td><span>FP32</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="2"><span>$\Phi_{\text{FTZ-MulAdd}}$</span></td>
<td><span>BF16</span></td>
<td><span>FP32</span></td>
<td><span>2</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td><span>FP16</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td><span>v_mfma<span> w/ </span>_1k<span> suffix</span></span></td>
<td>$\Phi_{\text{FTZ-MulAdd}}$</td>
<td><span>BF16</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="5"><span>CDNA3</span></td>
<td rowspan="5"><span>v_mfma</span></td>
<td rowspan="2"><span>$\Phi_{\text{FMA}}$</span></td>
<td><span>FP64</span></td>
<td><span>FP64</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP64</sub></span></td></tr>
<tr>
<td><span>FP32</span></td>
<td><span>FP32</span></td>
<td><span>1</span></td>
<td><span>N/A</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td rowspan="2"><span>$\Phi_{\text{TR-FDPA}}$</span></td>
<td><span>TF32</span></td>
<td><span>FP32</span></td>
<td><span>4</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td><span>BF16/FP16</span></td>
<td><span>FP32</span></td>
<td><span>8</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
<tr>
<td>$\Phi_{\text{GTR-FDPA}}$</td>
<td><span>FP8</span></td>
<td><span>FP32</span></td>
<td><span>16</span></td>
<td><span>24</span></td>
<td><span>RNE<sub>FP32</sub></span></td></tr>
</tbody>
</table>



#### IV-C1 NVIDIA Tensor Core Instructions

For NVIDIA Tensor Cores, the model type depends on the data types of $A$ and $B$ supported by the instruction. FP64 instructions are modeled by $\Phi_{\text{FMA}}$, MXFP8/6/4 instructions are modeled by $\Phi_{\text{ST-FDPA}}$, MXFP4/NVFP4 instructions are modeled by $\Phi_{\text{GST-FDPA}}$, and other instructions with input formats ranging from TF32 to FP4 are modeled by $\Phi_{\text{T-FDPA}}$.

The specific parameters of the $\Phi_{\text{T-FDPA}}$, $\Phi_{\text{ST-FDPA}}$, and $\Phi_{\text{GST-FDPA}}$ models depend on the architecture, the data types of $A$ and $B$, and the data type of $D$.

First-generation Tensor Cores on Volta GPUs use a relatively small maximum vector length of $L_{\text{max}}=4=\frac{8\text{ bytes}}{\text{size of input data type}}$ and a summation precision of $F=23$ fractional bits. The output rounding mode $\rho$ depends on the data type of $D$: RZ for FP32 and RNE for FP16.

Turing, Ampere, and Ada Lovelace GPUs double $L_{\text{max}}$ to $\frac{16\text{ bytes}}{\text{size of input data type}}$ and increase $F$ to 24. However, Ada Lovelace introduces FP8 formats with $F=13$ and $\rho\in\{\text{RZ}_{\text{E8M13}},\text{RNE}_{\text{FP16}}\}$, reflecting the reduced summation precision for FP8.

Hopper GPUs extend this trend, doubling $L_{\text{max}}$ to $\frac{32\text{ bytes}}{\text{size of input data type}}$, increasing $F$ to 25 for non-FP8 inputs, and keeping $F=13$ for FP8 inputs.

The most recent Blackwell and RTX Blackwell GPUs add FP6, FP4, and MXFP variants while maintaining similar rounding conventions and the same $L_{\text{max}}$ and $F$ values as Hopper GPUs for most formats. For FP8 formats, $F$ increases to $25$, and $\rho$ changes from $\text{RZ}_{\text{E8M13}}$ to $\text{RZ}_{\text{FP32}}$ for FP32 outputs. For MXFP4 or NVFP4 inputs, GST-FDPA operations specifically support a larger $F=35$, indicating a dedicated configuration for extremely low-precision formats.

#### IV-C2 AMD Matrix Core Instructions

For AMD Matrix Cores, the model type depends not only on the input data types of $A$ and $B$, but also on the architecture, with the exception of FP64 and FP32 instructions, which can be consistently modeled by $\Phi_{\text{FMA}}$ across architectures.

On CDNA1 GPUs, BF16 and FP16 instructions can be modeled by $\Phi_{\text{E-FDPA}}$ with different maximum vector lengths $L_{\text{max}}$ depending on the input data type. On CDNA2 GPUs, BF16 and FP16 instructions shift to $\Phi_{\text{FTZ-MulAdd}}$ with various $L_{\text{max}}$ parameters depending on the input type and the instruction suffix. CDNA3 GPUs introduce other FDPA variants for lower-precision formats, including $\Phi_{\text{TR-FDPA}}$ for TF32, BF16, and FP16, and $\Phi_{\text{GTR-FDPA}}$ for FP8. Their parameters follow a consistent pattern across input types: $L_{\text{max}}=\frac{16\text{ bytes}}{\text{size of input data type}}$, $F=24$, and $\rho=\text{RNE}_{\text{FP32}}$. The $L_{\text{max}}$ and $F$ parameters are identical to those of NVIDIA Ampere GPUs.

## V Discrepancy Analysis

Our white-box models provide a hardware perspective on numerical discrepancies. Across vendors and architectures, only FP64 and FP32 MMA instructions can maintain consistent numerical behavior because they are composed of the same FMA operations performed in the same sequential order. In contrast, mixed-precision MMA instructions use different elementary operations with different parameters, resulting in pervasive numerical discrepancies.

Based on our models, we demonstrate how the discrepancies arise using the following example input:



$$
\begin{split}&A=(\mathbf{a},\mathbf{0},\dots,\mathbf{0})^{T},B=(\mathbf{b},\mathbf{0},\dots,\mathbf{0}),C=(\mathbf{c},\mathbf{0},\dots,\mathbf{0}),\\
&\mathbf{a}=(-2^{13},-0.5,-0.25,-0.125,0,\dots,0)^{T},\\
&\mathbf{b}=(2^{10},1,1,1,0,\dots,0)^{T},\mathbf{c}=(2^{23},0,\dots,0)^{T}.\\
\end{split}
$$
Equation (10)



The output $d_{0,0}$ varies across architectures and instructions, as shown in Table [IV](https://arxiv.org/html/2511.10909v3#S5.T4). The value of $d_{0,0}$ is computed as the sum of $c=2^{23}$, $a_{0}b_{0}=-2^{23}$, $a_{1}b_{1}=-0.5$, $a_{2}b_{2}=-0.25$, and $a_{3}b_{3}=-0.125$. For FP64 and FP32 instructions, all devices produce the exact result $-0.875$ following the sequential FMA computation.

TABLE IV: The divergent results of different MMA instructions for the same input in Equation [10](https://arxiv.org/html/2511.10909v3#S5.E10). In addition, all FP64/FP32 instructions produce $d_{0,0}=-0.875$.

| Architecture | TF32/BF16 Instr. | FP16 Instr. | FP8 Instr. |
| --- | --- | --- | --- |
| Volta | N/A | $0.0$ | N/A |
| Turing | N/A | $-0.5$ | N/A |
| Ampere | $-0.5$ | $-0.5$ | N/A |
| Ada Lovelace | $-0.5$ | $-0.5$ | $0.0$ |
| Hopper | $-0.75$ | $-0.75$ | $0.0$ |
| Blackwell | $-0.75$ | $-0.75$ | $-0.75$ |
| RTX Blackwell | $-0.75$ | $-0.75$ | $-0.75$ |
| CDNA1 | $-0.875$ | $-0.875$ | N/A |
| CDNA2 | $-0.375$ or $0.0$ | $0.0$ | N/A |
| CDNA3 | $-0.5$ | $-0.5$ | $-1.0$ |

On Volta GPUs, the truncated summation with $F=23$ (the number of fractional bits in the accumulator) can only produce $0.0$ because $-0.5$ and the summands of smaller magnitude are truncated to zero. The same behavior occurs for FP8 (E5M2) instructions on Ada Lovelace and Hopper GPUs, where $F=13$. On Turing, Ampere, and Ada Lovelace GPUs, the truncated summation with $F=24$ produces $-0.5$ because $-0.25$ and $-0.125$ are truncated to zero. On Hopper, Blackwell, and RTX Blackwell GPUs, the truncated summation with $F=25$ produces $-0.75$ because only $-0.125$ is truncated to zero.

Non-truncated summation on CDNA1 GPUs produces the exact result $-0.875$. On CDNA2 GPUs, the FP16 instructions and the BF16 instructions with the “_1k” suffix use the summation order
$2^{23}+((-2^{23}-0.5)+(-0.25-0.125))=2^{23}+(-2^{23}-0.375)=2^{23}-2^{23}$
and produce $0.0$. The BF16 instructions without the “_1k” suffix use the summation order
$(2^{23}+(-2^{23}-0.5))+(-0.25-0.125)=(2^{23}-2^{23})-0.375=0-0.375$
and produce $-0.375$. On CDNA3 GPUs, the TF32, BF16, and FP16 instructions compute the fused truncated sum of $-2^{23}$, $-0.5$, $-0.25$, and $-0.125$ with $F=24$, producing the intermediate result $-2^{23}-0.5$, and then add it to $2^{23}$, resulting in $-0.5$. The FP8 (E5M2) instructions on CDNA3 GPUs perform fused summation with $F=24$ in two groups, $-2^{23}-0.25$ and $-0.5-0.125$, producing $-2^{23}$ and $-0.625$. They then compute the rounded sum of these two partial sums with $F=24$ and produce $-2^{23}-1$ because $-0.625$ is rounded down to $-1$. Finally, they add the rounded sum $-2^{23}-1$ to $c=2^{23}$, yielding $-1$.

## VI Accuracy Analysis

Leveraging our white-box models, we analyze the numerical accuracy of MMA instructions by quantifying every source of numerical error. Based on the analysis, we identify design choices that affect numerical precision and accuracy. These design choices explain the root causes of known hardware-related numerical issues on NVIDIA Hopper and AMD CDNA2 GPUs and expose potential numerical issues on other GPU architectures.

### VI-A Sources of Numerical Errors

All sources of numerical errors in MMA instructions are summarized in Table [V](https://arxiv.org/html/2511.10909v3#S6.T5). The errors accumulate additively throughout the computation.

TABLE V: Sources and upper bounds of numerical error.



<table>
<tbody>
<tr>
<td><span>Elementary Op.</span></td>
<td><span>Error Source</span></td>
<td><span>Error Bound</span></td></tr>
<tr>
<td rowspan="3"><span>FTZ-Add/Mul</span></td>
<td><span>Input FTZ</span></td>
<td>$2^{-126}$<span> or $2^{-14}$</span></td></tr>
<tr>
<td><span>Add/Mul</span></td>
<td>$0.5\text{ ulp}$</td></tr>
<tr>
<td><span>Output FTZ</span></td>
<td>$2^{-126}$</td></tr>
<tr>
<td><span>FMA, E-FDPA</span></td>
<td><span>Output rounding</span></td>
<td>$0.5\text{ ulp}$<span> or $1\text{ ulp}$</span></td></tr>
<tr>
<td rowspan="2"><span>T-FDPA, others</span></td>
<td><span>Fused summation</span></td>
<td>$(L+1)2^{e_{\text{max}}-F}$</td></tr>
<tr>
<td><span>Output rounding</span></td>
<td>$0.5\text{ ulp}$<span> or $1\text{ ulp}$</span></td></tr>
</tbody>
</table>



In $\Phi_{\text{FTZ-MulAdd}}$, the error bound for subnormal flushing corresponds to the smallest normal number of the floating-point format: $2^{-14}$ for FP16 and $2^{-126}$ for FP32 or BF16. In the FTZ-Add and FTZ-Mul operations, the error bound for addition and multiplication is $0.5\text{ ulp}$ (unit in the last place [[25](https://arxiv.org/html/2511.10909v3#bib.bib19)]) of the FP32 result, equal to $0.5\times 2^{e_{\text{result}}-23}$.

In $\Phi_{\text{FMA}}$ and $\Phi_{\text{E-FDPA}}$, since the internal computation of FMA and E-FDPA semantically operates at infinite precision, only output rounding contributes an error with an upper bound of $0.5\text{ ulp}$ (the RNE rounding mode) or $1\text{ ulp}$ (other rounding modes).

In $\Phi_{\text{T-FDPA}}$ and other $\Phi_{\text{FDPA}}$ variants, the errors come from two sources: fused summation and output rounding. In fused summation, the $L+1$ summands are eventually aligned at $e_{\text{max}}$ and rounded to $F$ fractional bits. This introduces an error of at most $(L+1)2^{e_{\text{max}}-F}$. In output rounding, the error bound depends on $\rho$: RNEFP32 and RNEFP16 have an error bound of $0.5\text{ ulp}$, while RZFP32 and RZE8M13 have an error bound of $1\text{ ulp}$.

### VI-B Precision and Accuracy Concerns

Most of the error sources we identified have error bounds comparable to those of standard FP32 operations, i.e., $0.5\text{ ulp}_{\text{FP32}}=0.5\times 2^{e_{\text{result}}-23}$. However, a few exhibit substantially larger error bounds and therefore constitute “precision bottlenecks”. In addition, we find that certain instructions on AMD CDNA3 GPUs violate the odd symmetry $\Phi(-A,B,-C)=-\Phi(A,B,C)$. Such asymmetry can also degrade numerical accuracy. We summarize these concerns in Table [VI](https://arxiv.org/html/2511.10909v3#S6.T6).

TABLE VI: Concerns regarding numerical precision (top) and accuracy (bottom).

|  |  |
| --- | --- |
| Architectures and Instructions | Concerns |
| AMD CDNA2, FP16-input | Input FTZ |
| NVIDIA Ada Lovelace and Hopper, FP8-input | Small $F$ |
| NVIDIA Ada Lovelace and Hopper, FP8-input | $\rho=\text{RZ}_{\text{E8M13}}$ |
| All NVIDIA architectures, FP16-output | $\rho=\text{RNE}_{\text{FP16}}$ |
| AMD CDNA3, TF32/BF16/FP16/FP8-input | Asymmetry |

#### VI-B1 Input FTZ of FP16 Subnormals

Although both input FTZ and output FTZ exist in $\Phi_{\text{FTZ-MulAdd}}$, the output FTZ poses lower risk because the output type is FP32 and the error is at most $2^{-126}$. In contrast, the input FTZ may introduce a significantly larger error (at most $2^{-14}$) for FP16 input. This explains why training stability of FP16 deep neural networks may degrade on AMD CDNA2 GPUs [[7](https://arxiv.org/html/2511.10909v3#bib.bib4)].

#### VI-B2 Reduced Precision in FP8 MMA

According to our models, the QMMA instructions on NVIDIA Ada Lovelace GPUs and the QGMMA instructions on NVIDIA Hopper GPUs employ only $F=13$ fractional bits for fused summation within the T-FDPA operation and use $\rho=\text{RZ}_{\text{E8M13}}$ when the output type is FP32. By contrast, their HMMA or HGMMA instructions use $F=24$ (Ada Lovelace) or $F=25$ (Hopper), similar to the precision of FP32 (E8M23), and use $\rho=\text{RZ}_{\text{FP32}}$ when the output type is FP32. Consequently, the error bound of the truncated fused summation in QMMA/QGMMA is higher than that in HMMA/HGMMA by a factor of $2^{24-13}=2048$ or $2^{25-13}=4096$, and the error bound of the output rounding is higher by a factor of $2^{23-13}=1024$. These account for the degraded training accuracy of FP8 large language models on Hopper GPUs [[5](https://arxiv.org/html/2511.10909v3#bib.bib3), [6](https://arxiv.org/html/2511.10909v3#bib.bib20)].

#### VI-B3 Limited Output Precision

As shown in Table [III](https://arxiv.org/html/2511.10909v3#S4.T3), some HMMA, HGMMA, UTCHMMA, QMMA, and QGMMA instructions on NVIDIA GPUs produce FP16 outputs with $\rho=\text{RNE}_{\text{FP16}}$, whose error bound is $0.5\text{ ulp}_{\text{FP16}}=0.5\times 2^{e_{\text{result}}-10}$. Therefore, even when the internal fused summation has FP32-like precision, the overall precision is bounded by the output format.

#### VI-B4 Asymmetric Rounding

We note that most MMA instructions use symmetric rounding modes such as RZ and RNE internally. However, we find that the mixed-precision v_mfma instructions on AMD CDNA3 GPUs use RD, an asymmetric rounding mode, for internal summation, which may introduce numerical bias.

To demonstrate the numerical bias, we simulate the CDNA3 v_mfma_f32_32x32x8_f16 instruction with $\Phi_{\text{TR-FDPA}}$ and a hypothetical variant v_mfma_f32_32x32x8_f16_rz using a modified model $\Phi_{\text{TR-FDPA}}^{\prime}$ that replaces the RD mode in the internal fused summation with RZ. We sample $C$ from the standard normal distribution $N(0,1)$ and sample $A$ and $B$ from the normal distribution $1000\times N(0,1)$. Then, we obtain the CDNA3 output $D_{\text{RD}}=\Phi_{\text{TR-FDPA}}(A,B,C)$ and the output of the RZ variant $D_{\text{RZ}}=\Phi_{\text{TR-FDPA}}^{\prime}(A,B,C)$. We also compute the accurate FP64 result $D_{\text{FP64}}=A\times B+C$, and measure the deviations $\delta_{\text{RD}}=D_{\text{RD}}-D_{\text{FP64}}$ and $\delta_{\text{RZ}}=D_{\text{RZ}}-D_{\text{FP64}}$. As Figure [3](https://arxiv.org/html/2511.10909v3#S6.F3) shows, the distribution of $\delta_{\text{RD}}$ exhibits a negative bias while the distribution of $\delta_{\text{RZ}}$ is symmetric.

![Refer to caption](assets/09-bit-accurate-gpu-mma/figure-001.png)


Fig. 3: Distributions of $\delta_{\text{RD}}$, the numerical deviation of the AMD CDNA3 FP16 MMA instruction, which uses the “round down” (RD) mode internally, and $\delta_{\text{RZ}}$, the numerical deviation of a hypothetical variant that replaces the internal RD operations with “round toward zero” (RZ) operations.

### VI-C Workarounds

##### Workaround 1: Scaling Inputs

We note that the CDNA2 FP16 FTZ issue (§[VI-B1](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS1)) manifests prominently for input elements of small magnitude. Therefore, we suggest scaling up the inputs to prevent FP16 subnormals and scaling down the output to restore the original scale. For example, consider multiplying $A_{128\times 128}$ sampled from $N(0,0.001^{2})$ with $B_{128\times 128}$ sampled from $N(0,1)$. This workaround reduces the mean squared error (MSE) relative to the FP64 result from $7.632\times 10^{-9}$ to $1.591\times 10^{-18}$. Since the computational precision of CDNA2 FP16 MMA is similar to FP32 except for the input FTZ (Table [V](https://arxiv.org/html/2511.10909v3#S6.T5)), this workaround preserves the high computational precision. Compared with PyTorch’s workaround [[7](https://arxiv.org/html/2511.10909v3#bib.bib4)], which casts FP16 to BF16 and introduces precision loss during conversion, our workaround has better numerical precision.

##### Workaround 2: MMA without $C$

We note that the numerical bias on AMD CDNA3 (§[VI-B4](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS4)) occurs when $A$ and $B$ contain large-magnitude numbers and the elements of $C$ are negative numbers with relatively small magnitudes. Therefore, developers can use Matrix Cores only for matrix multiplication by setting $C=0$, and use the general compute units for full FP32-precision matrix accumulation to mitigate this issue. Using $128\times 128\times 128$ matrix multiplication as an example, this workaround reduces the mean deviation relative to the FP64 result from $-1.352\times 10^{-6}$ to $+7.467\times 10^{-8}$ for FP16 inputs sampled from the normal distribution $N(0,10^{2})$.

##### Workaround 3: Changing Instructions

On NVIDIA GPUs, developers should avoid the instructions with precision bottlenecks (§[VI-B2](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS2), §[VI-B3](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS3)) and use higher-precision alternative instructions instead. FP16-output MMA instructions should be replaced by FP32-output instructions. For the $128\times 128\times 128$ matrix multiplication example above, this can reduce the MSE from $0.247$ to $4.481\times 10^{-8}$ (or $0.0551$ if FP32 results are converted back to FP16). FP8 matrix multiplication on NVIDIA Ada Lovelace or Hopper GPUs should be implemented with FP16 MMA instructions instead of FP8 instructions when accuracy takes precedence over throughput. This can reduce the MSE of the example above from $0.0103$ to $5.569\times 10^{-11}$ for FP8 inputs.

### VI-D Suggestions for Future MMAU Design

First, MMAU designers should carefully scrutinize the numerical accuracy of hardware designs. According to our modeling results, despite diverse arithmetic operations and configuration choices across architectures, most instructions have FP32-like numerical accuracy. To achieve this, hardware designers should avoid the designs in Table [VI](https://arxiv.org/html/2511.10909v3#S6.T6) and maintain FP32-level precision across all computation steps. Second, as symmetric rounding modes are preferable, we also suggest adopting the RZ mode because RZ can be implemented as truncation for sign-magnitude or ones’ complement encoding, which is far cheaper than other rounding modes.

## VII Conclusion

This paper introduces the first bit-accurate models of the arithmetic behavior of matrix multiply-accumulate units (MMAUs) on NVIDIA and AMD GPUs through our closed-loop feature probing (CLFP) approach. With the white-box models, we analyze the root causes of MMAU numerical discrepancies and numerical errors, providing insights into software mitigation and future hardware designs. This work is open-source, enabling continuous testing, transparent numerical analysis, and accuracy-aware design space exploration.

## Appendix A Artifact Appendix

### A-A Abstract

Our artifacts consist of
(1) the closed-loop feature probing (CLFP) framework and the feature-probing test (FPT) collection used to derive bit-accurate arithmetic models for GPU matrix multiply-accumulate (MMA) instructions,
(2) MMA-Sim, a Python software simulator that implements the derived models (§[IV](https://arxiv.org/html/2511.10909v3#S4)) and reproduces the bit-accurate output of every tested MMA instruction, and
(3) the CUDA/PTX and HIP instruction-level MMA interfaces for ten NVIDIA and AMD GPU architectures.
These artifacts provide fine-grained implementation details of our methods and algorithms that are not exhaustively covered in the paper, and enable independent verification and reproduction of our results.
We open-source the artifacts at <https://github.com/microsoft/MMA-Sim>, and provide a Jupyter notebook for easy reproduction of our results.

### A-B Artifact check-list (meta-information)

- Binary:  Python packages mmasim and mmasim-kernels are available on PyPI.
- Run-time environment:  Jupyter notebook.
- Hardware:  A GPU with Tensor Cores or Matrix Cores.
- Metrics:  Bit-exact match between MMA-Sim output and hardware output; execution time; numerical accuracy.
- How much disk space is required (approximately):  Less than 10 GB.
- How much time is needed to prepare the workflow (approximately):  Less than 1 hour.
- How much time is needed to complete the experiments (approximately):  About 1 hour.
- Publicly available:  Yes.
- Code licenses (if publicly available):  MIT license.

### A-C Description

#### A-C1 How to access

- Source code: <https://github.com/microsoft/MMA-Sim>.
- Jupyter notebook: <https://github.com/peichenxie/MMA-Sim-MICRO2026-AE>.

#### A-C2 Hardware dependencies

Software-only experiments with MMA-Sim can run on CPUs without a GPU.
Running hardware-specific experiments requires a GPU with Tensor Cores or Matrix Cores.
For example, the Jupyter notebook requires a Turing-architecture GPU such as NVIDIA T4.

#### A-C3 Software dependencies

We recommend running the Jupyter notebook on online platforms like Colab.
Alternatively, installing our artifact locally requires (1) a recent Linux distribution, (2) compatible GPU drivers and CUDA/ROCm toolkits, and (3) PyTorch 2.7+ with CUDA or ROCm support.

### A-D Installation

Our notebook includes installation commands.
For local experiments, run pip install mmasim mmasim-kernels.

### A-E Evaluation and expected results

We demonstrate the following three experiments in both our notebook and source code.

##### Verification of bit-accuracy.

To confirm that MMA-Sim matches hardware output bit for bit, run the tests/equivalence/test_<your_target_architecture>.py script. The script is expected to pass all tests without any errors.

##### Evaluation of CLFP execution time.

To measure the execution time of CLFP for one MMA instruction, run the tests/probe/clfp_example_step<1-4>.py scripts with a timer. All tests are expected to pass without any errors, and the execution time is expected to be consistent with that reported in Table [I](https://arxiv.org/html/2511.10909v3#S3.T1).

##### Reproduction of discrepancy and error analysis.

To reproduce the discrepancy example of §[V](https://arxiv.org/html/2511.10909v3#S5), run the examples/discrepancy/discrepancy_example.py script. The output is expected to be consistent with Table [IV](https://arxiv.org/html/2511.10909v3#S5.T4). To evaluate the numerical errors for the workaround methods in §[VI-C](https://arxiv.org/html/2511.10909v3#S6.SS3), run the scripts in examples/accuracy/. The measured errors are expected to be close to the values reported in §[VI-C](https://arxiv.org/html/2511.10909v3#S6.SS3).

## References

- [1]
  S. Markidis, S. W. D. Chien, E. Laure, I. B. Peng, and J. S. Vetter (2018)
  NVIDIA Tensor Core Programmability, Performance & Precision.
  In IEEE International Parallel and Distributed Processing Symposium (IPDPS) Workshops,
  pp. 522–531.
  External Links: [Link](https://doi.org/10.1109/IPDPSW.2018.00091),
  [Document](https://dx.doi.org/10.1109/IPDPSW.2018.00091)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p1.1),
  [§II-B](https://arxiv.org/html/2511.10909v3#S2.SS2.p1.1).
- [2]
  G. Schieffer, D. A. d. Medeiros, J. Faj, A. Marathe, and I. Peng (2024)
  On the Rise of AMD Matrix Cores: Performance, Power Efficiency, and Programmability.
  In IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS),
  pp. 132–143.
  External Links: [Link](https://doi.org/10.1109/ISPASS61541.2024.00022),
  [Document](https://dx.doi.org/10.1109/ISPASS61541.2024.00022)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p1.1),
  [§II-B](https://arxiv.org/html/2511.10909v3#S2.SS2.p1.1).
- [3]
  X. Li, A. Li, B. Fang, K. Swirydowicz, I. Laguna, and G. Gopalakrishnan (2024)
  Discovery of Floating-Point Differences Between NVIDIA and AMD GPUs.
  In International Symposium on Cluster, Cloud and Internet Computing (CCGRID),
  pp. 663–666.
  External Links: [Link](https://doi.org/10.1109/CCGrid59990.2024.00083),
  [Document](https://dx.doi.org/10.1109/CCGRID59990.2024.00083)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p2.1),
  [§II-B](https://arxiv.org/html/2511.10909v3#S2.SS2.p3.1).
- [4]
  M. Fasi, N. J. Higham, M. Mikaitis, and S. Pranesh (2021)
  Numerical behavior of NVIDIA tensor cores.
  PeerJ Computer Science 7, pp. e330.
  External Links: [Link](https://doi.org/10.7717/peerj-cs.330),
  [Document](https://dx.doi.org/10.7717/peerj-cs.330)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p2.1),
  [§II-B](https://arxiv.org/html/2511.10909v3#S2.SS2.p3.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p2.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p4.1).
- [5]
  DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Guo, D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Zhang, H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. L. Cai, J. Liang, J. Guo, J. Ni, J. Li, J. Wang, J. Chen, J. Chen, J. Yuan, J. Qiu, J. Li, J. Song, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao, L. Wang, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang, M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang, Q. Wang, Q. Zhu, Q. Chen, Q. Du, R. J. Chen, R. L. Jin, R. Ge, R. Zhang, R. Pan, R. Wang, R. Xu, R. Zhang, R. Chen, S. S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Pan, T. Wang, T. Yun, T. Pei, T. Sun, W. L. Xiao, and W. Zeng (2024)
  DeepSeek-V3 Technical Report.
  Note: arXiv: 2412.19437
  External Links: [Link](https://doi.org/10.48550/arXiv.2412.19437),
  [Document](https://dx.doi.org/10.48550/ARXIV.2412.19437)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p3.1),
  [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p3.1),
  [§VI-B2](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS2.p1.1).
- [6]
  C. Zhao, C. Deng, C. Ruan, D. Dai, H. Gao, J. Li, L. Zhang, P. Huang, S. Zhou, S. Ma, W. Liang, Y. He, Y. Wang, Y. Liu, and Y. X. Wei (2025)
  Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures.
  In International Symposium on Computer Architecture (ISCA),
  pp. 1731–1745.
  External Links: [Link](https://doi.org/10.1145/3695053.3731412),
  [Document](https://dx.doi.org/10.1145/3695053.3731412)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p3.1),
  [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p3.1),
  [§VI-B2](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS2.p1.1).
- [7]
  PyTorch Developers (2025)
  PyTorch Developer Notes - Numerical accuracy.
  External Links: [Link](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p3.1),
  [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p3.1),
  [§VI-B1](https://arxiv.org/html/2511.10909v3#S6.SS2.SSS1.p1.1),
  [§VI-C](https://arxiv.org/html/2511.10909v3#S6.SS3.SSS0.Px1.p1.1).
- [8]
  J. Zhang, H. Huang, P. Zhang, J. Wei, J. Zhu, and J. Chen (2025)
  SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization.
  In International Conference on Machine Learning (ICML),
  External Links: [Link](https://proceedings.mlr.press/v267/zhang25ae.html)
  Cited by: [§I](https://arxiv.org/html/2511.10909v3#S1.p3.1),
  [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p3.1).
- [9]
  C. Summers and M. J. Dinneen (2021)
  Nondeterminism and Instability in Neural Network Optimization.
  In International Conference on Machine Learning (ICML),
  pp. 9913–9922.
  External Links: [Link](http://proceedings.mlr.press/v139/summers21a.html)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p1.1).
- [10]
  J. Yuan, H. Li, X. Ding, W. Xie, Y. Li, W. Zhao, K. Wan, J. Shi, X. Hu, and Z. Liu (2025)
  Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference.
  In Conference on Neural Information Processing Systems (NeurIPS),
  External Links: [Link](https://openreview.net/forum?id=Q3qAsZAEZw)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p1.1).
- [11]
  F. Jézéquel, J. L. Lamotte, and I. Said (2015)
  Estimation of numerical reproducibility on CPU and GPU.
  In Federated Conference on Computer Science and Information Systems (FedCSIS),
  pp. 675–680.
  External Links: [Link](https://doi.org/10.15439/2015F29),
  [Document](https://dx.doi.org/10.15439/2015F29)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [12]
  A. Schlögl, N. Hofer, and R. Böhme (2023)
  Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks.
  In Conference on Neural Information Processing Systems (NeurIPS),
  External Links: [Link](http://papers.nips.cc/paper\_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [13]
  A. H. Zahid, I. Laguna, and W. Le (2024)
  Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs.
  In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis,
  pp. 547–557.
  External Links: [Link](https://doi.org/10.1109/SCW63240.2024.00077),
  [Document](https://dx.doi.org/10.1109/SCW63240.2024.00077)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [14]
  H. V. Pham, S. Qian, J. Wang, T. Lutellier, J. Rosenthal, L. Tan, Y. Yu, and N. Nagappan (2020)
  Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance.
  In International Conference on Automated Software Engineering (ASE),
  pp. 771–783.
  External Links: [Link](https://doi.org/10.1145/3324884.3416545),
  [Document](https://dx.doi.org/10.1145/3324884.3416545)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [15]
  D. Zhuang, X. Zhang, S. Song, and S. Hooker (2022)
  Randomness in Neural Network Training: Characterizing the Impact of Tooling.
  In Conference on Machine Learning and Systems (MLSys),
  External Links: [Link](https://proceedings.mlsys.org/paper/2022/hash/757b505cfd34c64c85ca5b5690ee5293-Abstract.html)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [16]
  I. Laguna (2020)
  Varity: Quantifying Floating-Point Variations in HPC Systems Through Randomized Testing.
  In IEEE International Parallel and Distributed Processing Symposium (IPDPS),
  pp. 622–633.
  External Links: [Link](https://doi.org/10.1109/IPDPS47924.2020.00070),
  [Document](https://dx.doi.org/10.1109/IPDPS47924.2020.00070)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [17]
  X. Xu, H. Liu, G. Tao, Z. Xuan, and X. Zhang (2022)
  Checkpointing and Deterministic Training for Deep Learning.
  In International Conference on AI Engineering (CAIN),
  pp. 65–76.
  External Links: [Document](https://dx.doi.org/10.1145/3522664.3528605)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [18]
  B. Chen, M. Wen, Y. Shi, D. Lin, G. K. Rajbahadur, and Z. M. Jiang (2022)
  Towards Training Reproducible Deep Learning Models.
  In International Conference on Software Engineering (ICSE),
  pp. 2202–2214.
  External Links: [Link](https://doi.org/10.1145/3510003.3510163),
  [Document](https://dx.doi.org/10.1145/3510003.3510163)
  Cited by: [§II-A](https://arxiv.org/html/2511.10909v3#S2.SS1.p2.1).
- [19]
  B. J. Hickmann and D. Bradford (2019)
  Experimental Analysis of Matrix Multiplication Functional Units.
  In IEEE Symposium on Computer Arithmetic (ARITH),
  pp. 116–119.
  External Links: [Link](https://doi.org/10.1109/ARITH.2019.00031),
  [Document](https://dx.doi.org/10.1109/ARITH.2019.00031)
  Cited by: [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p2.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p4.1).
- [20]
  X. Li, A. Li, B. Fang, K. Swirydowicz, I. Laguna, and G. Gopalakrishnan (2024)
  FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA & AMD Matrix Accelerators.
  In International Symposium on Cluster, Cloud and Internet Computing (CCGRID),
  pp. 39–46.
  External Links: [Link](https://doi.org/10.1109/CCGrid59990.2024.00014),
  [Document](https://dx.doi.org/10.1109/CCGRID59990.2024.00014)
  Cited by: [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p4.1).
- [21]
  P. Xie, Y. Gao, Y. Wang, and J. Xue (2025)
  Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations.
  In USENIX Annual Technical Conference (USENIX ATC),
  pp. 1425–1440.
  External Links: [Link](https://www.usenix.org/conference/atc25/presentation/xie)
  Cited by: [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1),
  [§III-A2](https://arxiv.org/html/2511.10909v3#S3.SS1.SSS2.p3.1),
  [§III-A2](https://arxiv.org/html/2511.10909v3#S3.SS1.SSS2.p4.1).
- [22]
  B. Valpey, X. Li, S. Pai, and G. Gopalakrishnan (2025)
  An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores.
  Note: arXiv: 2502.15999
  External Links: [Link](https://doi.org/10.48550/arXiv.2502.15999),
  [Document](https://dx.doi.org/10.48550/ARXIV.2502.15999)
  Cited by: [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1),
  [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p4.1).
- [23]
  M. A. Raihan, N. Goli, and T. M. Aamodt (2019)
  Modeling Deep Learning Accelerator Enabled GPUs.
  In IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS),
  pp. 79–92.
  External Links: [Link](https://doi.org/10.1109/ISPASS.2019.00016),
  [Document](https://dx.doi.org/10.1109/ISPASS.2019.00016)
  Cited by: [§II-C](https://arxiv.org/html/2511.10909v3#S2.SS3.p1.1).
- [24]
  J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao (2024)
  FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.
  In Conference on Neural Information Processing Systems (NeurIPS),
  External Links: [Link](http://papers.nips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)
  Cited by: [item 1](https://arxiv.org/html/2511.10909v3#S3.I1.i1.p1.1).
- [25]
  N. J. Higham (2002)
  Accuracy and stability of numerical algorithms.
  2 edition, SIAM.
  External Links: ISBN 978-0-89871-521-7,
  [Link](https://doi.org/10.1137/1.9780898718027),
  [Document](https://dx.doi.org/10.1137/1.9780898718027)
  Cited by: [item 2](https://arxiv.org/html/2511.10909v3#S3.I1.i2.p1.1),
  [§VI-A](https://arxiv.org/html/2511.10909v3#S6.SS1.p2.1).
- [26]
  N. P. Jouppi, C. Young, N. Patil, D. A. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon (2017)
  In-Datacenter Performance Analysis of a Tensor Processing Unit.
  In International Symposium on Computer Architecture (ISCA),
  pp. 1–12.
  External Links: [Link](https://doi.org/10.1145/3079856.3080246),
  [Document](https://dx.doi.org/10.1145/3079856.3080246)
  Cited by: [§III-B](https://arxiv.org/html/2511.10909v3#S3.SS2.p3.1).
- [27]
  IEEE (2019)
  IEEE Standard for Floating-Point Arithmetic.
  External Links: [Document](https://dx.doi.org/10.1109/IEEESTD.2019.8766229)
  Cited by: [§IV-A1](https://arxiv.org/html/2511.10909v3#S4.SS1.SSS1.p1.1).

---
title: "PTXBench: Benchmarking and Adapting LLMs for GPU Kernel Optimization with Architecture-specific PTX"
source_url: "https://arxiv.org/abs/2608.17379"
full_text_url: "https://arxiv.org/html/2608.17379v3"
version: "2608.17379v3"
authors: ["Zhang, Genghan", "Dong, Yixin", "Fan, Chengze", "Zeng, Zhichen", "Yuan, Yueming", "Zhu, Shaowei", "Olukotun, Kunle"]
retrieved_date: "2026-11-20"
conversion: "HTML to Markdown with TeX math"
pdf: "04-ptxbench.pdf"
original_html: "sources/04-ptxbench.html"
---

# PTXBench: Evaluating and Adapting LLMs for GPU Kernel Optimization with Architecture-specific PTX

Genghan Zhang
††thanks: Correspondence to: Genghan Zhang zgh23@stanford.edu. Code available at <https://github.com/zhang677/PTXBench>.
Affiliation: Stanford University
  
Yixin Dong
Affiliation: Carnegie Mellon University
  
Chengze Fan
Affiliation: Independent Researcher
  
Zhichen Zeng
Affiliation: RadixArk
  
Yueming Yuan
Affiliation: RadixArk
  
Shaowei Zhu
Affiliation: Independent Researcher
  
Kunle Olukotun
Affiliation: Stanford University

###### Abstract

We introduce PTXBench, a benchmark for evaluating and adapting large language models (LLMs) to use architecture-specific PTX for GPU kernel optimization. PTXBench measures functional correctness, whether selected target instructions execute at runtime, and speedup over frontier libraries across GEMM and attention workloads on H100 and B200 GPUs. Our evaluation shows that architecture-specific PTX capability remains uneven: success rates fall substantially on complex attention backward workloads, and executing the target instructions does not necessarily translate into competitive performance. No evaluated model consistently matches frontier libraries across the suite. We further adapt Qwen3.6-27B using supervised fine-tuning. Repair-conditioned training improves several tasks, but generalization remains uneven; data coverage, balance, and the quality of the reasoning teacher matter in addition to dataset size. PTXBench provides an auditable testbed for measuring and improving LLMs’ ability to exploit evolving GPU architectures.

|  |
| --- |
|  |

## 1 Introduction

PTX (Parallel Thread Execution) is the lowest-level programmable interface that CUDA developers can explicitly control. High-performance kernels often require PTX-level optimization to efficiently leverage architecture-specific mechanisms [[59](https://arxiv.org/html/2608.17379v3#bib.bib38)]. Recent NVIDIA GPU generations have been introducing new architecture-specific PTX instructions for tensor cores, memory units, and synchronization mechanisms [[23](https://arxiv.org/html/2608.17379v3#bib.bib33), [24](https://arxiv.org/html/2608.17379v3#bib.bib34), [26](https://arxiv.org/html/2608.17379v3#bib.bib35), [27](https://arxiv.org/html/2608.17379v3#bib.bib36), [32](https://arxiv.org/html/2608.17379v3#bib.bib37)]. A portable CUDA kernel can remain functionally correct while leaving the defining capabilities of newer hardware unused. Higher-level kernel languages aim to recover performance portability [[48](https://arxiv.org/html/2608.17379v3#bib.bib31), [47](https://arxiv.org/html/2608.17379v3#bib.bib32)], which ultimately lower to the same PTX interface. However, keeping these kernel languages aligned with rapidly evolving hardware requires continued compiler engineering [[6](https://arxiv.org/html/2608.17379v3#bib.bib56)], while validating an optimizing compiler entails a much larger input and configuration space and thus is much harder than validating an individual kernel [[9](https://arxiv.org/html/2608.17379v3#bib.bib55)]. This raises a practical question for kernel developers: can an LLM directly exploit architecture-specific features, and can targeted post-training improve this ability?

To answer this question, we introduce PTXBench, an architecture-specific GPU kernel benchmark that evaluates whether LLMs can exploit specified low-level GPU mechanisms. PTXBench incorporates MiniPTXAgent, a multi-turn agent loop in which models are given an accurate architecture-specific knowledge pack and generate CUDA kernels with inline PTX (which we call CUDA–PTX) and revise them using structured execution feedback. Across each trajectory, PTXBench separately measures functional correctness, whether the required instruction family executes at runtime under the evaluated workload, and peak and typical performance. PTXBench is built on FlashInfer-Trace [[54](https://arxiv.org/html/2608.17379v3#bib.bib19)], a unified schema for kernel workloads, solutions, and evaluations. Separating the capability probe from the problem collection makes PTXBench extensible to suites with compatible interfaces, such as SOL-ExecBench [[19](https://arxiv.org/html/2608.17379v3#bib.bib18)]. Existing GPU kernel benchmarks, beginning with KernelBench [[38](https://arxiv.org/html/2608.17379v3#bib.bib1)], ask models to replace reference PyTorch operators with functionally correct, faster GPU kernels [[40](https://arxiv.org/html/2608.17379v3#bib.bib29), [16](https://arxiv.org/html/2608.17379v3#bib.bib30)]. These end-to-end outcomes are essential, but they do not isolate whether a model can directly program a specified architecture mechanism: performance may instead come from generic CUDA code or calls to existing vendor libraries. PTXBench complements these suites with a controlled capability probe: models must directly produce efficient low-level kernels using a specified family of architecture-specific PTX instructions, and the corresponding target instructions must execute at runtime.

PTXBench provides a controlled measurement framework and an environment for collecting adaptation data. We first characterize architecture-specific PTX capability across current models and GPUs, and then test targeted, repair-conditioned post-training.

This work makes three contributions:

- An architecture-specific PTX benchmark. We introduce PTXBench, a multi-turn benchmark for evaluating whether LLMs can produce functionally correct GPU kernels that execute the required target instructions at runtime. It provides controlled architecture knowledge and separately measures correctness, target instruction execution, and performance relative to frontier libraries.
- A capability study across models, architectures, and workloads. We evaluate closed- and open-weight models on H100 and B200 across GEMM and attention workloads. Models frequently succeed on forward workloads but struggle with backward attention, and even kernels with verified target instruction execution generally remain slower than frontier libraries. These results expose a substantial gap between executing architecture-specific instructions, producing correct kernels, and achieving competitive performance.
- A controlled adaptation study. We conduct, to our knowledge, the first controlled study of SFT conditioned on repairs for CUDA and PTX generation that targets a specific architecture, adapting Qwen3.6-27B and ablating problem coverage and balance and reasoning supervision. Supervised fine tuning conditioned on repairs improves several tasks, but transfer to held-out shapes and attention variants remains uneven. Problem coverage, data balance, and the reasoning teacher all matter.

![Refer to caption](assets/04-ptxbench/figure-001.png)


Figure 1: Overview of the PTXBench benchmark and adaptation workflow.

## 2 PTXBench

[Figure 1](https://arxiv.org/html/2608.17379v3#S1.F1) summarizes PTXBench’s benchmark and adaptation workflow. PTXBench is designed around three requirements for evaluating architecture-specific PTX programming. First, models receive controlled architecture knowledge because they otherwise rarely use the requested PTX. Second, we verify that the required instruction family executes at runtime under the fixed workload. Third, we evaluate each kernel for correctness and efficiency relative to frontier library implementations.

### 2.1 Benchmark Tasks and Controlled Context

A PTXBench instance pairs a reference operator composed from frontier libraries such as cuBLAS, cuDNN, and FlashInfer [[29](https://arxiv.org/html/2608.17379v3#bib.bib47), [7](https://arxiv.org/html/2608.17379v3#bib.bib45), [55](https://arxiv.org/html/2608.17379v3#bib.bib46)] with a fixed workload, target GPU architecture, and required family of architecture-specific PTX instructions. The model generates a CUDA kernel with inline PTX from scratch rather than starting from an initial implementation. PTXBench uses the FlashInfer-Trace schema to express workloads, solutions, and correctness checks, allowing the same benchmark workflow to support other compatible task collections.

To measure architecture-specific reasoning rather than documentation retrieval, every trajectory receives the same architecture-specific knowledge pack in its base prompt: architecture parameters, CUDA wrappers for PTX instructions, and contracts governing layouts, synchronization, and memory consistency. For well-studied operators, we also provide fixed, expert-validated scheduling principles ([Section A.5](https://arxiv.org/html/2608.17379v3#A1.SS5) shows an example for FlashAttention). Each pack occupies 20k–30k tokens ([Section A.4](https://arxiv.org/html/2608.17379v3#A1.SS4)). The ablation in [Table 4](https://arxiv.org/html/2608.17379v3#S4.T4) shows that without this context, the model rarely uses the requested PTX.

### 2.2 Multi-turn Evaluation Protocol

MiniPTXAgent uses a multi-turn protocol because iterative kernel generation, execution feedback, and revision form the fundamental operating loop of kernel agents [[4](https://arxiv.org/html/2608.17379v3#bib.bib15), [58](https://arxiv.org/html/2608.17379v3#bib.bib14), [14](https://arxiv.org/html/2608.17379v3#bib.bib16)]. Each trajectory permits a fixed number of model calls and retains all preceding kernels and execution feedback. MiniPTXAgent first compiles each candidate with nvcc in a CPU container, sending only successful compilations to a profiling service for memory-safety checking, runtime error messages, functional evaluation, and latency measurement.

We define target instruction correctness as functional correctness plus execution of selected tensor-compute or data-movement instructions at runtime under the evaluated workload. To measure execution, we analyze SASS, NVIDIA’s native GPU assembly generated from PTX and executed by the GPU. For each functionally correct candidate, NVIDIA Nsight Compute (NCU) supplies predicate-enabled thread execution counts for the instruction families in [Table 6](https://arxiv.org/html/2608.17379v3#A1.T6). A candidate qualifies when either the architecture-specific compute predicate $C$ or data-movement predicate $D$ is true. On B200, TMA payload transfer can satisfy $D$, while fifth-generation tensor compute requires both MMA and TMEM readback. Static SASS presence alone does not establish execution; failed profiling remains unknown. Target instruction correctness establishes whether the selected instructions execute, not how much useful work they perform or whether they cause the measured speedup. [Section A.2](https://arxiv.org/html/2608.17379v3#A1.SS2) gives implementation details and edge cases.

The functional correctness checker verifies output shape and data type, then compares values using torch.allclose with atol=rtol=1e-2. After a candidate passes the correctness check, candidate and reference latencies are measured with CUPTI [[31](https://arxiv.org/html/2608.17379v3#bib.bib57)]. For each trial, we take the median over 50 timed iterations after 10 warmup iterations, and compute speedup as reference latency divided by candidate latency. Including header files from cuDNN or cuBLAS [[29](https://arxiv.org/html/2608.17379v3#bib.bib47)] in CUDA source code is considered wrong regardless of execution results. Generated kernels may crash, hang, or corrupt subsequent measurements, so the profiling service isolates execution of kernels from the agent and exposes correctness, sanitization, debugging, and latency measurement as independent operations. More details on the profiling service are in [Section A.3](https://arxiv.org/html/2608.17379v3#A1.SS3).

## 3 Adapting LLMs to Architecture-Specific PTX Programming

Deployed GPUs retain fixed low-level capabilities for years, making architecture-specific PTX a durable target for post-training. The obstacle is data: high-quality kernels require scarce expert knowledge and iterative validation. We therefore study whether model failures and execution feedback, paired with teacher-generated repairs and rationales, can provide effective supervision for this domain.

##### Fixit.

Fixit constructs supervision from failures produced by the model being adapted. Let $x$ denote a problem prompt together with its architecture-specific context, $\pi_{0}$ the model before adaptation, $H$ the MiniPTXAgent feedback function, and $C$ the functional-correctness indicator. We first sample a failed kernel $k^{-}$ and collect its compilation or execution feedback $e$:



$$
k^{-}\sim\pi_{0}(\cdot\mid x),\qquad e=H(x,k^{-}),\qquad C(x,k^{-})=0.
$$




A repair teacher $\pi_{R}$ then generates a corrected kernel conditioned on the same problem, failure, and feedback. We retain only repairs $k^{+}$ that pass the correctness check:



$$
k^{+}\sim\pi_{R}(\cdot\mid x,k^{-},e),\qquad C(x,k^{+})=1.
$$




Finally, a reasoning teacher $\pi_{T}$ synthesizes a rationale $r$ that leads from the observed failure to the retained repair:



$$
r\sim\pi_{T}(\cdot\mid x,k^{-},e,k^{+}).
$$




Each Fixit example therefore conditions the student on $(x,k^{-},e)$ and supervises it with $(r,k^{+})$. Failures are collected once from the model before adaptation, so the supervision targets error states produced by that fixed checkpoint; the repairs and rationales are teacher-generated.

## 4 Benchmark Results

### 4.1 Evaluation Metrics and Baselines

We evaluate models along four complementary dimensions. Turn correctness rate (# of correct kernels / # of turns) captures the ability to generate correct kernels. Target instruction turn correctness rate (# of correct kernels that execute a selected target instruction / # of turns) evaluates whether a model can produce a functionally correct kernel that exercises the requested architecture mechanism. Finally, among correct turns, best speedup measures peak optimization ability, while target instruction best speedup restricts that peak to qualifying kernels. Inspired by KernelBench [[38](https://arxiv.org/html/2608.17379v3#bib.bib1)], for $N$ evaluated turns with correctness $C_{i}$, runtime target instruction indicator $I_{i}$, and speedup $s_{i}$, we summarize the full speedup distribution as



$$
\mathrm{Fast}_{p}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[C_{i}=1\land s_{i}>p],\qquad\mathrm{Fast}^{\mathrm{Inst.}}_{p}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[C_{i}=1\land I_{i}=1\land s_{i}>p].
$$




At threshold $p$, $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$ is the fraction of all turns that produce a correct kernel, execute a selected target instruction at runtime, and exceed speedup $p$. NCU failures remain unknown and do not enter the target instruction numerator, so the reported values are conservative. In space-constrained table headers and plot axes, we abbreviate target instruction as “Target inst.” Across all result tables, the first line is the target instruction metric, the parenthesized line is unrestricted, and triplets report $\leq 1/\leq 4/\leq 8$ turns. Plot curves show the corresponding turn fractions, and vertical labels mark the best qualifying speedup. Solid lines show the mean across three prompt variants, each evaluated over four eight-turn trajectories ($N=32$), while shading spans the prompt-wise minimum and maximum. Throughout the paper, Fwd denotes multihead attention (MHA) forward with LSE input, Bwd denotes MHA backward, Causal denotes causal attention, and d denotes the head dimension. For GQA variants, the performance baseline is FlashInfer (v0.6.14); for all other attentions, it is cuDNN (v9.20.0); for GEMM, it is cuBLAS (v13.1.0). These baselines represent frontier performance, and achieving such performance on these problems requires complex scheduling of PTX instructions specific to each architecture.

Table 1: Target-instruction and unrestricted turn correctness rates (%) on H100 and B200.



<table>
<tr>
<td>GPU</td>
<td>Model</td>
<td>GEMM</td>
<td>MHA-Fwd</td>
<td>MHA-Fwd-Causal</td>
<td>MHA-Bwd</td>
<td>MHA-Bwd-Causal</td></tr>
<tr>
<td rowspan="5"><span>H100</span></td>
<td>Gemini 3.1 Pro</td>
<td><span></span> <span>
<span>
<span>
<span>33.3 / 60.4 / 56.2</span></span>
<span>
<span><span>(33.3 / 62.5 / 59.4)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>33.3 / 39.6 / 45.8</span></span>
<span>
<span><span>(33.3 / 39.6 / 45.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>25.0 / 52.1 / 55.2</span></span>
<span>
<span><span>(25.0 / 52.1 / 55.2)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 33.3 / 38.5</span></span>
<span>
<span><span>(8.3 / 33.3 / 38.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 22.9 / 25.0</span></span>
<span>
<span><span>(8.3 / 25.0 / 26.0)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Claude Opus 4.8</td>
<td><span></span> <span>
<span>
<span>
<span>91.7 / 95.8 / <span>94.8</span></span></span>
<span>
<span><span>(91.7 / 95.8 / 94.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>50.0 / 81.2 / <span>90.6</span></span></span>
<span>
<span><span>(66.7 / 89.6 / 94.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>50.0 / 77.1 / <span>75.0</span></span></span>
<span>
<span><span>(83.3 / 89.6 / 82.3)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 62.5 / <span>79.2</span></span></span>
<span>
<span><span>(50.0 / 83.3 / 89.6)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 22.9 / 44.8</span></span>
<span>
<span><span>(25.0 / 41.7 / 59.4)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>GLM-5.2</td>
<td><span></span> <span>
<span>
<span>
<span>33.3 / 60.4 / 61.5</span></span>
<span>
<span><span>(33.3 / 62.5 / 62.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>16.7 / 8.3 / 8.3</span></span>
<span>
<span><span>(16.7 / 14.6 / 15.6)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 8.3 / 10.4</span></span>
<span>
<span><span>(8.3 / 16.7 / 17.7)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 6.2 / 5.2</span></span>
<span>
<span><span>(8.3 / 18.8 / 16.7)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / 22.9 / 15.6)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Qwen3.6-27B</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td></tr>
<tr>
<td>GPT-5.6 xhigh</td>
<td><span></span> <span>
<span>
<span>
<span>91.7 / 81.2 / 77.1</span></span>
<span>
<span><span>(91.7 / 83.3 / 79.2)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 39.6 / 60.4</span></span>
<span>
<span><span>(16.7 / 58.3 / 69.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 50.0 / 56.2</span></span>
<span>
<span><span>(– / 64.6 / 63.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 39.6 / 55.2</span></span>
<span>
<span><span>(58.3 / 60.4 / 65.6)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 31.2 / <span>52.1</span></span></span>
<span>
<span><span>(33.3 / 56.2 / 65.6)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td rowspan="5"><span>B200</span></td>
<td>Gemini 3.1 Pro</td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 47.9 / 45.8</span></span>
<span>
<span><span>(8.3 / 47.9 / 45.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 12.5 / 15.6</span></span>
<span>
<span><span>(– / 12.5 / 15.6)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 8.3 / 10.4</span></span>
<span>
<span><span>(8.3 / 12.5 / 12.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 2.1 / 2.1</span></span>
<span>
<span><span>(– / 2.1 / 2.1)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / 2.1 / 1.0)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Claude Opus 4.8</td>
<td><span></span> <span>
<span>
<span>
<span>25.0 / 64.6 / <span>80.2</span></span></span>
<span>
<span><span>(75.0 / 81.2 / 88.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 20.8 / 38.5</span></span>
<span>
<span><span>(83.3 / 87.5 / 86.5)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 16.7 / 32.3</span></span>
<span>
<span><span>(91.7 / 77.1 / 83.3)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 6.2 / 10.4</span></span>
<span>
<span><span>(83.3 / 89.6 / 91.7)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(25.0 / 52.1 / 68.8)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>GLM-5.2</td>
<td><span></span> <span>
<span>
<span>
<span>16.7 / 27.1 / 28.1</span></span>
<span>
<span><span>(33.3 / 37.5 / 34.4)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / – / 1.0</span></span>
<span>
<span><span>(33.3 / 22.9 / 26.0)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>8.3 / 2.1 / 1.0</span></span>
<span>
<span><span>(33.3 / 22.9 / 25.0)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(8.3 / 25.0 / 30.2)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(16.7 / 18.8 / 22.9)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Qwen3.6-27B</td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / – / 1.0)</span></span></span>
</span></span><span></span></td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td></tr>
<tr>
<td>GPT-5.6 xhigh</td>
<td><span></span> <span>
<span>
<span>
<span>– / 45.8 / 59.4</span></span>
<span>
<span><span>(50.0 / 66.7 / 69.8)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 18.8 / <span>39.6</span></span></span>
<span>
<span><span>(16.7 / 43.8 / 52.1)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 18.8 / <span>42.7</span></span></span>
<span>
<span><span>(8.3 / 43.8 / 55.2)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 22.9 / <span>37.5</span></span></span>
<span>
<span><span>(75.0 / 54.2 / 54.2)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 22.9 / <span>31.2</span></span></span>
<span>
<span><span>(83.3 / 47.9 / 47.9)</span></span></span>
</span></span><span></span></td></tr>
</table>


![Refer to caption](assets/04-ptxbench/figure-002.png)


Figure 2: $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$ on H100 (top) and B200 (bottom).

### 4.2 Architecture-Specific PTX Capability Remains Uneven

[Tables 1](https://arxiv.org/html/2608.17379v3#S4.T1) and [2](https://arxiv.org/html/2608.17379v3#S4.F2) compare models on H100 and B200. The $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$ curves show both how often each model produces qualifying kernels and the distribution of their speedups. Since workload difficulty varies, we report speedups per problem; Appendix [Table 9](https://arxiv.org/html/2608.17379v3#A1.T9) gives exact values.

Table 2: Model release dates, knowledge cutoffs, vendor-reported SWE-bench Pro scores, and estimated calendar
lag from the release of Hopper PTX ISA 8.0 (Dec. 2022) and Blackwell PTX ISA 8.7 (Jan. 2025) [[25](https://arxiv.org/html/2608.17379v3#bib.bib53), [28](https://arxiv.org/html/2608.17379v3#bib.bib54)]. Lags use monthly granularity, from PTX release to disclosed cutoff; otherwise, model release dates provide upper bounds.



<table>
<tr>
<td><span>Model</span></td>
<td><span></span> <span>
<span>
<span>
<span><span>SWE-bench</span></span></span>
<span>
<span><span>Pro (%)</span></span></span>
</span></span><span></span></td>
<td><span>Model release</span></td>
<td><span>Knowledge cutoff</span></td>
<td colspan="2"><span>Lag after PTX release (months)</span></td></tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td><span>Hopper</span></td>
<td><span>Blackwell</span></td></tr>
<tr>
<td>Gemini 3.1 Pro</td>
<td>54.2</td>
<td>Feb. 2026</td>
<td>Jan. 2025</td>
<td>25</td>
<td>$\approx 0$</td></tr>
<tr>
<td>Claude Opus 4.8</td>
<td>69.2</td>
<td>May 2026</td>
<td>Jan. 2026</td>
<td>37</td>
<td>12</td></tr>
<tr>
<td>GPT-5.6 Sol</td>
<td>64.6</td>
<td>July 2026</td>
<td>Feb. 2026</td>
<td>38</td>
<td>13</td></tr>
<tr>
<td>GLM-5.2</td>
<td>62.1</td>
<td>June 2026</td>
<td>Not disclosed</td>
<td>$\leq 42$</td>
<td>$\leq 17$</td></tr>
<tr>
<td>Qwen3.6-27B</td>
<td>53.5</td>
<td>Apr. 2026</td>
<td>Not disclosed</td>
<td>$\leq 40$</td>
<td>$\leq 15$</td></tr>
</table>



We choose these five models because they were released relatively close together in time [[11](https://arxiv.org/html/2608.17379v3#bib.bib48), [1](https://arxiv.org/html/2608.17379v3#bib.bib49), [37](https://arxiv.org/html/2608.17379v3#bib.bib50), [56](https://arxiv.org/html/2608.17379v3#bib.bib51), [39](https://arxiv.org/html/2608.17379v3#bib.bib52)]. [Table 2](https://arxiv.org/html/2608.17379v3#S4.T2) summarizes their public release metadata, vendor-reported SWE-bench Pro scores, and intervals during which the PTX documentation could have entered the training data.

The SWE-bench Pro results provide context for general coding ability, but they are not a controlled comparison because agent scaffolds and evaluation settings differ. They also do not fully characterize architecture-specific capability: Gemini 3.1 Pro and Qwen3.6-27B differ by only 0.7 points on SWE-bench Pro, yet Gemini executes selected Hopper instructions across all five workloads, whereas Qwen3.6-27B produces no correct Hopper kernel under our setup.

Calendar exposure alone is similarly incomplete. Although Gemini 3.1 Pro’s knowledge cutoff is in the same month as the Blackwell PTX release, it still achieves $0.892\times$ cuBLAS performance on GEMM on Blackwell. Claude Opus 4.8 achieves a substantially higher target instruction correctness rate on Blackwell and reaches $1.012\times$ cuBLAS on GEMM, but neither model optimizes Blackwell attention as well as Hopper attention. Together, these results suggest that general coding ability and newer PTX knowledge can help, but neither public coding scores nor calendar exposure alone predicts architecture-specific PTX capability or performance.

![Refer to caption](assets/04-ptxbench/figure-003.png)


Figure 3: Gemini 3.1 Pro $\mathrm{Fast}_{p}$ distributions for Triton and CUDA-PTX on H100 and B200.

Among models with open weights, GLM-5.2 is competitive with Gemini 3.1 Pro for GEMM on both H100 and B200 and for attention workloads without causal masking on H100, but it still lags on B200 attention workloads. Like Claude Opus 4.8, GLM-5.2 also tends to fall back on generic CUDA instead of architecture-specific PTX when the target is the newer Blackwell architecture. In contrast, Gemini 3.1 Pro is more successful at executing Blackwell instructions, achieving similar peak performance to Claude Opus 4.8 despite its lower success rate. Qwen3.6-27B produces one correct GEMM kernel on Blackwell, but it does not execute any selected Blackwell instruction.

### 4.3 High-Level Kernel Languages Remain Valuable on New Architectures

[Figure 3](https://arxiv.org/html/2608.17379v3#S4.F3) compares Gemini 3.1 Pro kernels generated in Triton and CUDA–PTX using the same five workloads on each GPU. On H100, the peak speedups are close on GEMM ($0.991\times$ for Triton versus $0.962\times$ for CUDA–PTX), but Triton has higher $\mathrm{Fast}_{p}$ across the attention workloads. On B200, the attention gap is larger: Triton reaches $0.514\times$ on MHA-Bwd and $0.437\times$ on MHA-Bwd-Causal, compared with $0.198\times$ and no correct CUDA–PTX candidate, respectively. The corresponding GPT-5.6 Sol comparison appears in [Figure 13](https://arxiv.org/html/2608.17379v3#A1.F13), including cases where CUDA–PTX attains a higher single-kernel peak.

Two factors may explain this architecture-dependent gap. First, Blackwell adds more specialized compute and memory mechanisms, increasing the complexity of coordinating low-level instructions directly. Second, Blackwell is roughly two years newer than Hopper, so far less Blackwell-specific CUDA-PTX code may have appeared in model training data (cf. [Table 2](https://arxiv.org/html/2608.17379v3#S4.T2)). We also carefully prepared architecture-specific prompts for CuTeDSL, but its kernel success rate remained extremely low, preventing a meaningful performance comparison.

The broader takeaway is that high-level kernel languages remain useful for producing correct kernels on new architectures, where directly generating CUDA-PTX can be less effective. They do not, however, guarantee the best attainable performance: once correct, direct low-level implementations can sometimes match or outperform a higher-level implementation.

### 4.4 Comparison with Coding Agents

We compare Antigravity, using Gemini 3.1 Pro, with multi-turn refinement using the same model and task specification on H100. Each method has four trajectories per workload and eight evaluator requests per trajectory. Both receive the same architectural knowledge and execution feedback without scheduling guidance. Antigravity can edit files and call the same profiling service through an evaluation client; model calls and tool use can differ between harnesses. We therefore compare outcomes at a fixed evaluation-request budget and report token cost separately. Here the outcome is each trajectory’s best correct, runtime-instruction-qualified speedup, or zero if none qualifies; [Table 3](https://arxiv.org/html/2608.17379v3#S4.T3) reports the median over four trajectories.

Table 3: Antigravity versus multi-turn refinement after eight H100 evaluation requests per trajectory. Speedups are medians of four per-trajectory best instruction-qualified values. Token ratios divide the agent’s median cumulative total tokens by the refinement median within each workload.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Method or ratio | GEMM | MHA-Fwd | MHA-Fwd-Causal | MHA-Bwd | MHA-Bwd-Causal |
| Multi-turn refinement | $0.505\times$ | $0.584\times$ | $0.560\times$ | $0.240\times$ | $0.166\times$ |
| Antigravity | $0.583\times$ | $0.425\times$ | $0.413\times$ | $0.053\times$ | $0.000\times$ |
| Agent / refinement total tokens | $3.32\times$ | $5.06\times$ | $3.90\times$ | $3.73\times$ | $3.52\times$ |

Antigravity has a higher median on GEMM, while multi-turn refinement has a higher median on all four attention workloads. The agent uses 3.32–5.06$\times$ as many median total tokens across these workloads. [Figure 14](https://arxiv.org/html/2608.17379v3#A1.F14) shows the outcomes at shared per-trajectory total-token budgets, and [Section A.6.2](https://arxiv.org/html/2608.17379v3#A1.SS6.SSS2) gives the corresponding Codex/GPT-5.6 Sol results. These comparisons describe the evaluated harness configurations; an equal number of evaluator requests does not imply equal model or tool cost.

### 4.5 Explicit Architecture Knowledge Improves Target Instruction Execution

The knowledge ablation with three prompts in [Table 4](https://arxiv.org/html/2608.17379v3#S4.T4) shows that the evaluated model does not execute the target instructions automatically without explicit PTX knowledge. At eight turns, architecture parameters alone yield 26.0% correct turns but no target instruction successes, while adding template functions enables target instruction execution and produces faster generated kernels. Adding the architecture contract provides a clear correctness edge: 38.5% of turns satisfy target instruction correctness, 18.7 points above template functions alone. Peak speedup follows a different ordering, so the contract’s main benefit is reliable and correct execution of the target instructions; it does not necessarily produce the fastest kernel.

Table 4: Ablation of architecture-specific prompt knowledge for Gemini 3.1 Pro.

|  |  |  |
| --- | --- | --- |
| Prompt knowledge | Target inst. correctness (%)  (turn correctness) | Target inst. best speedup  (best speedup) |
| Architecture parameters | –  (50.0 / 29.2 / 26.0) | –  (0.056 / 0.119 / 0.273) |
| Architecture parameters + PTX template functions | – / 20.8 / 19.8  (– / 20.8 / 19.8) | – / 0.542 / 0.542  (– / 0.542 / 0.542) |
| Architecture parameters + PTX template functions + architecture contract | 8.3 / 33.3 / 38.5  (8.3 / 33.3 / 38.5) | 0.206 / 0.375 / 0.515  (0.206 / 0.375 / 0.515) |

## 5 Adaptation Results

We instantiate Fixit from [Section 3](https://arxiv.org/html/2608.17379v3#S3) with Qwen3.6-27B as the model to adapt and Gemini 3.1 Pro as the repair teacher. Qwen3.6-27B’s 262K-token context accommodates our long PTX prompts and kernel traces, while its tractable scale enables controlled LoRA adaptation and self-hosted evaluation. Its weak baseline capability also provides measurable headroom for studying post-training gains. We first compare the base model with s1 on its training problems and held-out workloads, then ablate problem coverage, data balance, and reasoning-teacher choice across six Fixit recipes.

### 5.1 Main SFT Result: Effects and Generalization

![Refer to caption](assets/04-ptxbench/figure-004.png)

![Refer to caption](assets/04-ptxbench/figure-005.png)

Figure 4: Base versus s1 generalization on the s1 training problem classes and held-out workloads. Top: Qwen3.6-27B before adaptation. Bottom: Qwen3.6-27B-s1, trained with shuffle seed 0. Runtime-instruction rows require dynamically observed Hopper SASS execution.

For detailed analysis, we select s1, the smallest recipe that solves all five primary evaluation problems. It was trained on four MHA tasks with head dimension 128. As shown in [Figure 4](https://arxiv.org/html/2608.17379v3#S5.F4), the base model produces no correct candidates on GEMM or any MHA workload in this suite; its only correct candidates are on pooled GQA workloads, and none execute the target instructions. In contrast, s1 produces correct kernels on all four training problems and transfers to GEMM, all four d64 MHA tasks, both d96 forward tasks, and GQA. It produces no correct kernels for either d96 backward task. The d96 backward tasks are especially challenging because d96 does not align with H100 WGMMA’s m64 tile. Fixit therefore yields a clear gain over the base checkpoint, while transfer remains uneven across head dimensions and attention variants.

### 5.2 Data Recipe Ablation

![Refer to caption](assets/04-ptxbench/figure-006.png)


Figure 5: Six-recipe data ablation. Pie area is proportional to record count, slices show the fraction drawn from each problem, and the bottom line identifies the reasoning synthesizer.

[Figure 5](https://arxiv.org/html/2608.17379v3#S5.F5) summarizes the six Fixit recipes by problem mix and reasoning synthesizer. The recipes use GLM-5.2 as the reasoning teacher except for s6, which uses Qwen3.6-27B itself. Dataset composition and record counts are reported in Appendix [Table 12](https://arxiv.org/html/2608.17379v3#A1.T12).

![Refer to caption](assets/04-ptxbench/figure-007.png)


Figure 6: Six-recipe SFT data ablation at eight turns (complete results in Appendix [Tables 10](https://arxiv.org/html/2608.17379v3#A1.T10) and [11](https://arxiv.org/html/2608.17379v3#A1.T11)).

##### Coverage and balance.

Among Fixit recipes, coverage and balance matter more than record count alone. Only the relatively balanced s1 and s5 recipes solve all five problems. In contrast, s2 contains $1.6\times$ as many records as s1, and s3 contains $2.4\times$ as many as s4, yet both fail on MHA-Bwd-Causal. Moving from s4 to the larger balanced s5 recipe ($1.5\times$ more records) improves eight-turn correctness on four problems, ties on MHA-Fwd, and improves peak speedup on four. This agrees with instruction-tuning studies that emphasize task balance, selection, and diversity over unfiltered scale [[22](https://arxiv.org/html/2608.17379v3#bib.bib23), [21](https://arxiv.org/html/2608.17379v3#bib.bib24), [3](https://arxiv.org/html/2608.17379v3#bib.bib25)]. Balance improves breadth, but not every peak: the best-performing recipe still varies by problem.

##### Reasoning synthesizer.

The controlled s5–s6 comparison keeps the Fixit examples fixed and changes only the reasoning synthesizer from GLM-5.2 to Qwen3.6-27B itself. While s5 solves all five problems, s6 solves only GEMM. Target-model failures are therefore useful inputs, but producing their repair rationales still benefits from a stronger teacher.

### 5.3 Additional Effects of Fixit SFT

![Refer to caption](assets/04-ptxbench/figure-008.png)


Figure 7: How Fixit SFT changes reasoning length and error types across turns.

We further test cross-language transfer by asking the base and s1 checkpoints to generate Triton for the same five Hopper workloads. [Figure 8](https://arxiv.org/html/2608.17379v3#S5.F8) shows that s1 has lower turn-level correctness on every workload, indicating that the current SFT recipe hurts correctness under Triton transfer. Yet its best correct speedup improves markedly on the causal variants, from $0.238\times$ to $0.632\times$ for MHA-Fwd-Causal and from $0.043\times$ to $0.331\times$ for MHA-Bwd-Causal. Thus, the recipe can improve peak performance by a large margin in some cases even while making correct kernels less likely.

![Refer to caption](assets/04-ptxbench/figure-009.png)


Figure 8: Cross-language transfer of Fixit SFT from CUDA-PTX to Triton on Hopper.

[Figure 7](https://arxiv.org/html/2608.17379v3#S5.F7) shows how s1 changes the search process. It lengthens reasoning at every turn, especially during early revisions. It also shifts failures from compilation toward runtime and numerical errors: the model more often produces executable kernels, but those kernels can still fail during evaluation. On GEMM, s1 produces three correct kernels at turn 0 and at least one in six of the seven later turns, whereas the base model produces no correct kernels in any turn ([Figure 9](https://arxiv.org/html/2608.17379v3#S5.F9)). Moreover, the SFT checkpoints in [Figure 6](https://arxiv.org/html/2608.17379v3#S5.F6) have identical target-instruction and unrestricted results except for s1 on MHA-Bwd-Causal at eight turns. Thus, Fixit can improve initial kernel generation and reliable execution of target instructions.

Figure 9: Turn-level error-state transitions for GEMM.

### 5.4 SFT Versus In-context Supervision

Finally, we compare weight updates with information supplied at inference time. [Tables 5](https://arxiv.org/html/2608.17379v3#S5.T5) and [10](https://arxiv.org/html/2608.17379v3#S5.F10) evaluate six alternatives: base model and s1 with and without expert-edited guidance distilled by Codex from error trajectories (Appendix [Figure 18](https://arxiv.org/html/2608.17379v3#A1.F18)), and base model with retrieval of repair experiences. Retrieval uses BM25 to select the most similar failed kernel from s1’s data pool and supplies GPT 5.4’s summary of its repair notes, alone or with the corrected kernel.

Even with expert guidance, the base model still cannot write correct MHA kernels. In contrast, even without expert guidance, s1 can already write correct MHA kernels. This suggests that SFT improves the model’s base PTX capability and its ability to comprehend guidance. Repair notes alone produce no correct kernels; correctness rises sharply only when retrieval also supplies the fixed kernel. However, this solution-bearing condition is expected to work because the answers are nearly in the prompt. With guidance, s1 attains nonzero correctness across all four problems.

Table 5: Target-instruction and unrestricted turn correctness rates (%) under SFT and prompt-time supervision.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Condition | MHA-Fwd | MHA-Fwd-Causal | MHA-Bwd | MHA-Bwd-Causal |
| Qwen3.6-27B w/o expert guidance | – | – | – | – |
| Qwen3.6-27B | – | – | – | – |
| Qwen3.6-27B-s1 w/o expert guidance | – | – / – / 4.2  (– / – / 4.2) | – / – / 4.2  (– / – / 4.2) | – / 4.2 / 3.1  (– / 4.2 / 3.1) |
| Qwen3.6-27B-s1 | 16.7 / 16.7 / 19.8  (16.7 / 16.7 / 19.8) | – / – / 3.1  (– / – / 3.1) | – / 2.1 / 4.2  (– / 2.1 / 4.2) | – / 2.1 / 4.2  (– / 2.1 / 5.2) |
| Qwen3.6-27B + retrieved repair notes | – | – | – | – |
| Qwen3.6-27B + retrieved repair notes  and fixed kernel | – / 29.2 / 29.2  (– / 29.2 / 29.2) | – / 20.8 / 27.1  (– / 20.8 / 27.1) | – / 14.6 / 18.8  (– / 14.6 / 20.8) | – / 20.8 / 33.3  (– / 25.0 / 37.5) |

![Refer to caption](assets/04-ptxbench/figure-010.png)


Figure 10: $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$ under SFT and prompt-time supervision.

## 6 Related Work

GPU kernel generation benchmarks have attracted increasing interest [[51](https://arxiv.org/html/2608.17379v3#bib.bib2), [62](https://arxiv.org/html/2608.17379v3#bib.bib3), [15](https://arxiv.org/html/2608.17379v3#bib.bib4), [12](https://arxiv.org/html/2608.17379v3#bib.bib5), [52](https://arxiv.org/html/2608.17379v3#bib.bib6), [57](https://arxiv.org/html/2608.17379v3#bib.bib7), [36](https://arxiv.org/html/2608.17379v3#bib.bib8)]. Collectively, these benchmarks span translation from PyTorch to kernels, Triton generation, portability across devices, production serving traces, deployment integration, and hardware performance limits. They nevertheless focus primarily on functional correctness and overall efficiency, and thus do not isolate whether a generated kernel actually executes a specified architecture mechanism rather than relying on generic CUDA or vendor libraries. PTXBench instead fixes the target architecture and required PTX instruction family, then verifies target instruction execution at runtime, complementing their broad task coverage with a controlled capability probe. Concurrent work introduces Hawkeye, which uses expert-authored unit tests and profiling metrics for hardware-aware kernel generation across architectures and precisions [[49](https://arxiv.org/html/2608.17379v3#bib.bib62)]; PTXBench instead benchmarks runtime target-instruction execution and studies repair-conditioned post-training.

Model adaptation for GPU kernel generation has advanced through SFT and RL methods that use execution feedback for Triton and CUDA [[17](https://arxiv.org/html/2608.17379v3#bib.bib9), [53](https://arxiv.org/html/2608.17379v3#bib.bib10), [20](https://arxiv.org/html/2608.17379v3#bib.bib42), [13](https://arxiv.org/html/2608.17379v3#bib.bib11), [46](https://arxiv.org/html/2608.17379v3#bib.bib44), [10](https://arxiv.org/html/2608.17379v3#bib.bib43), [2](https://arxiv.org/html/2608.17379v3#bib.bib41), [18](https://arxiv.org/html/2608.17379v3#bib.bib12), [45](https://arxiv.org/html/2608.17379v3#bib.bib13)]. These systems optimize correctness and overall speed through a DSL/compiler, ordinary CUDA, or implementations that use libraries. For example, CUDA Agent allows the use of existing cuDNN library functions (cf. Figure 15 in [[8](https://arxiv.org/html/2608.17379v3#bib.bib17)]) and CUDA-L2 allows CUTLASS and CuTe [[43](https://arxiv.org/html/2608.17379v3#bib.bib61)]. We target a harder, more constrained regime: generating kernels from scratch with the required inline PTX specific to the architecture while prohibiting existing vendor libraries. To our knowledge, PTXBench is the first controlled study of model adaptation for programming at this level, examining repair conditioning, data balance, reasoning supervision, and transfer.

## 7 Limitations

Our study has three main limitations. First, the adaptation experiments use modest LoRA datasets and a single 27B base model, so the observed effects of repair conditioning, data balance, and teacher quality may not transfer unchanged to industry-scale post-training or other model families. Second, PTXBench currently focuses on BF16 GEMM and attention kernels on H100 and B200. These workloads directly stress recent architecture-specific tensor core and asynchronized memory but do not represent the full diversity of GPU operators and hardware. Since PTXBench separates FlashInfer-Trace workloads, architecture context, and evaluation, the same workflow can be extended to broader workloads, models, and future GPUs. Third, PTXBench measures within-trajectory kernel refinement rather than persistent model improvement, so it provides a prerequisite probe but is still not a sufficient test of recursive self-improvement.

## 8 Conclusion

We introduced PTXBench, an auditable benchmark and adaptation environment for architecture-specific PTX programming. By separately measuring functional correctness, target instruction execution at runtime, and speedup over frontier libraries, PTXBench exposes a persistent capability gap: current LLMs can sometimes execute the requested instructions and solve forward workloads, but struggle with backward attention and do not consistently achieve competitive performance across H100 and B200. With Fixit, we further study repair-conditioned SFT using failures from the model being adapted and correctness-filtered teacher repairs. Fixit improves several tasks over the base model; its gains depend on problem coverage, data balance, and reasoning-teacher quality, while transfer to held-out shapes and attention variants remains uneven. Together, these results position PTXBench as a controlled testbed for measuring architecture-specific capability and developing targeted post-training data for evolving GPU architectures.

## Acknowledgments

We thank Banghua Zhu, Ying Sheng, Jiajun Li, Mao Cheng, and Yusheng Su from RadixArk and SGLang community for their technical support and insightful discussions. We are also grateful to Xinhao Li, Yibo Zhang, Anjiang Wei, Simon Guo, and Stanford Pervasive Parallelism Lab members for discussions and help. We also thank the Gemini Academic Program and the Tinker Research Grant for their generous support.

## References

- Anthropic (2026)
  Anthropic
  Claude Opus 4.8.
  Note: <https://www.anthropic.com/news/claude-opus-4-8>Released May 28, 2026; knowledge cutoff January 2026
  Cited by: [§4.2](https://arxiv.org/html/2608.17379v3#S4.SS2.p2.1).
- Baronio et al. (2025)
  C. Baronio, P. Marsella, B. Pan, S. Guo, and S. Alberti
  Kevin: multi-turn rl for generating cuda kernels.
  arXiv preprint arXiv:2507.11948.
  External Links: [Link](https://arxiv.org/abs/2507.11948)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Bukharin et al. (2024)
  A. Bukharin, S. Li, Z. Wang, J. Yang, B. Yin, X. Li, C. Zhang, T. Zhao, and H. Jiang
  Data diversity matters for robust instruction tuning.
  In Findings of the Association for Computational Linguistics: EMNLP 2024,
  pp. 3411–3425.
  External Links: [Document](https://dx.doi.org/10.18653/v1/2024.findings-emnlp.195),
  [Link](https://aclanthology.org/2024.findings-emnlp.195/)
  Cited by: [§5.2](https://arxiv.org/html/2608.17379v3#S5.SS2.SSS0.Px1.p1.1).
- Cao et al. (2026)
  S. Cao, Z. Mao, J. E. Gonzalez, and I. Stoica
  K-Search: LLM kernel generation via co-evolving intrinsic world model.
  arXiv preprint arXiv:2602.19128.
  External Links: [Link](https://arxiv.org/abs/2602.19128)
  Cited by: [§2.2](https://arxiv.org/html/2608.17379v3#S2.SS2.p1.1).
- Chatterjee et al. (2025)
  B. Chatterjee, D. Zagieboylo, S. Damani, S. Hari, and C. Kozyrakis
  Proofwright: towards agentic formal verification of cuda.
  arXiv preprint arXiv:2511.12294.
  Cited by: [§A.1](https://arxiv.org/html/2608.17379v3#A1.SS1.SSS0.Px1.p1.1).
- Chen et al. (2026)
  H. Chen, B. Fan, A. Collins, B. Hagedorn, E. Gaburov, M. Masuda, M. Brookhart, C. Sullivan, J. Knight, Z. Zhang, et al.
  Tawa: automatic warp specialization for modern GPUs with asynchronous references.
  In 2026 IEEE/ACM International Symposium on Code Generation and Optimization (CGO),
  pp. 255–267.
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- Chetlur et al. (2014)
  S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer
  cuDNN: efficient primitives for deep learning.
  arXiv preprint arXiv:1410.0759.
  Cited by: [§2.1](https://arxiv.org/html/2608.17379v3#S2.SS1.p1.1).
- Dai et al. (2026)
  W. Dai, H. Wu, Q. Yu, H. Gao, J. Li, C. Jiang, W. Lou, Y. Song, H. Yu, J. Chen, et al.
  Cuda agent: large-scale agentic rl for high-performance cuda kernel generation.
  arXiv preprint arXiv:2602.24286.
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Driscoll et al. (2025)
  B. Driscoll, K. Dubey, A. Wei, N. Kayal, R. Sharma, and A. Aiken
  Equivalence checking of ML GPU kernels.
  arXiv preprint arXiv:2511.12638.
  External Links: [Link](https://arxiv.org/abs/2511.12638)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- Du et al. (2026)
  H. Du, Q. Ge, J. Hu, A. Yang, Z. Cai, Z. Huang, S. Yuan, Q. Cheng, X. Xie, Y. Chen, Y. Li, et al.
  Kernel-Smith: a unified recipe for evolutionary kernel optimization.
  arXiv preprint arXiv:2603.28342.
  External Links: [Link](https://arxiv.org/abs/2603.28342)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Google DeepMind (2026)
  Google DeepMind
  Gemini 3.1 Pro.
  Note: <https://deepmind.google/models/model-cards/gemini-3-1-pro/>Published February 19, 2026. The January 2025 cutoff for the Gemini 3 family is documented at <https://ai.google.dev/gemini-api/docs/gemini-3>
  Cited by: [§4.2](https://arxiv.org/html/2608.17379v3#S4.SS2.p2.1).
- Guan et al. (2026)
  Y. Guan, Y. Lin, X. Zhao, J. Yao, X. Qiang, Z. Yu, P. Viswanath, Y. Ding, and A. Aziz
  TritonGym: a benchmark for agentic LLM workflows in Triton GPU code generation.
  In Proceedings of the 43rd International Conference on Machine Learning,
  Proceedings of Machine Learning Research, Vol. 306, Seoul, South Korea.
  External Links: [Link](https://icml.cc/virtual/2026/poster/65003)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Guo et al. (2026)
  S. Guo, M. Lin, and T. Yang
  DRTriton: large-scale synthetic data reinforcement learning for Triton kernel generation.
  arXiv preprint arXiv:2603.21465.
  External Links: [Link](https://arxiv.org/abs/2603.21465)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Hong et al. (2025)
  C. Hong, S. Bhatia, A. Cheung, and Y. S. Shao
  Autocomp: a powerful and portable code optimizer for tensor accelerators.
  arXiv preprint arXiv:2505.18574.
  External Links: [Link](https://arxiv.org/abs/2505.18574)
  Cited by: [§2.2](https://arxiv.org/html/2608.17379v3#S2.SS2.p1.1).
- Lange et al. (2025)
  R. T. Lange, Q. Sun, A. Prasad, M. Faldor, Y. Tang, and D. Ha
  Towards robust agentic CUDA kernel benchmarking, verification, and optimization.
  arXiv preprint arXiv:2509.14279.
  External Links: [Link](https://arxiv.org/abs/2509.14279)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Li et al. (2025a)
  J. Li, S. Li, Z. Gao, Q. Shi, Y. Li, Z. Wang, J. Huang, H. Wang, J. Wang, X. Han, Z. Liu, and M. Sun
  TritonBench: benchmarking large language model capabilities for generating Triton operators.
  In Findings of the Association for Computational Linguistics: ACL 2025,
  Vienna, Austria, pp. 23053–23066.
  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.findings-acl.1183),
  [Link](https://aclanthology.org/2025.findings-acl.1183/)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p2.1).
- Li et al. (2025b)
  S. Li, Z. Wang, Y. He, Y. Li, Q. Shi, J. Li, Y. Hu, W. Che, X. Han, Z. Liu, and M. Sun
  AutoTriton: automatic Triton programming with reinforcement learning in LLMs.
  arXiv preprint arXiv:2507.05687.
  External Links: [Link](https://arxiv.org/abs/2507.05687)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Li et al. (2026)
  X. Li, X. Sun, A. Wang, J. Li, and C. Shum
  CUDA-L1: improving CUDA optimization via contrastive reinforcement learning.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=igZItUbY6n)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Lin et al. (2026)
  E. Lin, S. Modi, S. K. S. Hari, Q. Huang, Z. Ye, N. Qin, F. Zhou, Y. Zhang, J. Wang, S. Damani, et al.
  SOL-execbench: speed-of-light benchmarking for real-world gpu kernels against hardware limits.
  arXiv preprint arXiv:2603.19173.
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p2.1).
- Liu et al. (2026)
  W. Liu, J. Xu, Y. Li, L. Zheng, T. Li, Q. Liu, and J. He
  Dr. kernel: reinforcement learning done right for triton kernel generations.
  arXiv preprint arXiv:2602.05885.
  External Links: [Link](https://arxiv.org/abs/2602.05885)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Liu et al. (2024)
  W. Liu, W. Zeng, K. He, Y. Jiang, and J. He
  What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning.
  In International Conference on Learning Representations,
  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html)
  Cited by: [§5.2](https://arxiv.org/html/2608.17379v3#S5.SS2.SSS0.Px1.p1.1).
- Longpre et al. (2023)
  S. Longpre, L. Hou, T. Vu, A. Webson, H. W. Chung, Y. Tay, D. Zhou, Q. V. Le, B. Zoph, J. Wei, and A. Roberts
  The Flan collection: designing data and methods for effective instruction tuning.
  In Proceedings of the 40th International Conference on Machine Learning,
  Proceedings of Machine Learning Research, Vol. 202, pp. 22631–22648.
  External Links: [Link](https://proceedings.mlr.press/v202/longpre23a.html)
  Cited by: [§5.2](https://arxiv.org/html/2608.17379v3#S5.SS2.SSS0.Px1.p1.1).
- NVIDIA Corporation (2017)
  NVIDIA Corporation
  Volta tuning guide.
   NVIDIA Corporation.
  Note: Accessed: 2026-07-29
  External Links: [Link](https://docs.nvidia.com/cuda/volta-tuning-guide/)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- NVIDIA Corporation (2020)
  NVIDIA Corporation
  NVIDIA Ampere GPU Architecture Tuning Guide.
   NVIDIA Corporation.
  Note: Accessed: 2026-07-29
  External Links: [Link](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- NVIDIA Corporation (2022)
  NVIDIA Corporation
  Parallel thread execution isa, version 8.0.
   NVIDIA Corporation.
  Note: Released with CUDA Toolkit 12.0 in December 2022
  External Links: [Link](https://docs.nvidia.com/cuda/archive/12.0.0/pdf/ptx_isa_8.0.pdf)
  Cited by: [Table 2](https://arxiv.org/html/2608.17379v3#S4.T2).
- NVIDIA Corporation (2023)
  NVIDIA Corporation
  NVIDIA Hopper Tuning Guide.
   NVIDIA Corporation.
  Note: Accessed: 2026-07-29
  External Links: [Link](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- NVIDIA Corporation (2025a)
  NVIDIA Corporation
  NVIDIA Blackwell Tuning Guide.
   NVIDIA Corporation.
  Note: Accessed: 2026-07-29
  External Links: [Link](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- NVIDIA Corporation (2025b)
  NVIDIA Corporation
  Parallel thread execution isa, version 8.7.
   NVIDIA Corporation.
  Note: Released with CUDA Toolkit 12.8
  External Links: [Link](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf)
  Cited by: [Table 2](https://arxiv.org/html/2608.17379v3#S4.T2).
- NVIDIA Corporation (2026a)
  NVIDIA Corporation
  cuBLAS library.
   NVIDIA Corporation.
  Note: CUDA Toolkit Documentation
  External Links: [Link](https://docs.nvidia.com/cuda/cublas/)
  Cited by: [§2.1](https://arxiv.org/html/2608.17379v3#S2.SS1.p1.1),
  [§2.2](https://arxiv.org/html/2608.17379v3#S2.SS2.p3.1).
- NVIDIA Corporation (2026b)
  NVIDIA Corporation
  CUDA Binary Utilities.
   NVIDIA Corporation.
  Note: Accessed: 2026-08-15
  External Links: [Link](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
  Cited by: [§A.2](https://arxiv.org/html/2608.17379v3#A1.SS2.SSS0.Px1.p1.1).
- NVIDIA Corporation (2026c)
  NVIDIA Corporation
  CUDA Profiling Tools Interface (CUPTI).
   NVIDIA Corporation.
  Note: Accessed: 2026-08-10
  External Links: [Link](https://docs.nvidia.com/cupti/)
  Cited by: [§2.2](https://arxiv.org/html/2608.17379v3#S2.SS2.p3.1).
- NVIDIA Corporation (2026d)
  NVIDIA Corporation
  Inside the NVIDIA Vera Rubin Platform: Six New Chips, One AI Supercomputer.
  Note: <https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/>Accessed: 2026-07-29
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- NVIDIA Corporation (2026e)
  NVIDIA Corporation
  Nsight Compute CLI.
   NVIDIA Corporation.
  Note: Accessed: 2026-08-15
  External Links: [Link](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
  Cited by: [§A.2](https://arxiv.org/html/2608.17379v3#A1.SS2.SSS0.Px1.p2.1).
- NVIDIA Corporation (2026f)
  NVIDIA Corporation
  Performance benchmarking.
  Note: <https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html>TensorRT Documentation, accessed July 29, 2026
  Cited by: [Figure 11](https://arxiv.org/html/2608.17379v3#A1.F11).
- NVIDIA Corporation (2026g)
  NVIDIA Corporation
  Stream-Ordered Memory Allocator.
   NVIDIA Corporation.
  Note: Accessed: 2026-08-15
  External Links: [Link](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
  Cited by: [§A.2](https://arxiv.org/html/2608.17379v3#A1.SS2.SSS0.Px3.p1.1).
- Oliaro et al. (2026)
  G. Oliaro, Y. Fu, M. Jiang, O. Lu, J. Wang, Z. Jia, H. Zhang, and S. Rajbhandari
  FastKernels: benchmarking GPU kernel generation in production.
  arXiv preprint arXiv:2605.23215.
  External Links: [Link](https://arxiv.org/abs/2605.23215)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- OpenAI (2026)
  OpenAI
  GPT-5.6: frontier intelligence that scales with your ambition.
  Note: <https://openai.com/index/gpt-5-6/>Released July 9, 2026. The February 16, 2026 knowledge cutoff is documented at <https://developers.openai.com/api/docs/models/gpt-5.6-sol>
  Cited by: [§4.2](https://arxiv.org/html/2608.17379v3#S4.SS2.p2.1).
- Ouyang et al. (2025)
  A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, and A. Mirhoseini
  KernelBench: can LLMs write efficient GPU kernels?.
  In Proceedings of the 42nd International Conference on Machine Learning,
  Proceedings of Machine Learning Research, Vol. 267, pp. 47356–47415.
  External Links: [Link](https://proceedings.mlr.press/v267/ouyang25a.html)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p2.1),
  [§4.1](https://arxiv.org/html/2608.17379v3#S4.SS1.p1.1).
- Qwen Team (2026)
  Qwen Team
  Qwen3.6-27b: flagship-level coding in a 27b dense model.
  Note: <https://qwen.ai/blog?id=qwen3.6-27b>Released April 21, 2026; knowledge cutoff not publicly disclosed
  Cited by: [§4.2](https://arxiv.org/html/2608.17379v3#S4.SS2.p2.1).
- Saroufim et al. (2025)
  M. Saroufim, J. Wang, B. Maher, S. Paliskara, L. Wang, S. Sefati, and M. Candales
  BackendBench: an evaluation suite for testing how well LLMs and humans can write PyTorch backends.
  Note: GitHub repository
  External Links: [Link](https://github.com/meta-pytorch/BackendBench)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p2.1).
- Shah et al. (2024)
  J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao
  FlashAttention-3: fast and accurate attention with asynchrony and low-precision.
  Advances in Neural Information Processing Systems 37, pp. 68658–68685.
  Cited by: [§A.5](https://arxiv.org/html/2608.17379v3#A1.SS5.p1.1).
- Stephenson et al. (2026)
  M. Stephenson, S. Damani, M. Tarek Ibn Ziad, A. Ladram, and M. Garland
  SuperCollider: scalable and effective data race detection for cuda.
  Proceedings of the ACM on Programming Languages 10 (PLDI), pp. 2303–2327.
  Cited by: [§A.1](https://arxiv.org/html/2608.17379v3#A1.SS1.SSS0.Px1.p1.1),
  [§A.2](https://arxiv.org/html/2608.17379v3#A1.SS2.SSS0.Px3.p1.1).
- Su et al. (2025)
  S. Su, X. Li, A. Wang, G. Wang, J. Li, and C. Shum
  Cuda-l2: surpassing cublas performance for matrix multiplication through reinforcement learning.
  arXiv preprint arXiv:2512.02551.
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Sul and Ré (2026)
  S. Sul and C. Ré
  ThunderKittens 2.0: even faster kernels for your GPUs.
  Note: Hazy Research blog, accessed 2026-07-29
  External Links: [Link](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
  Cited by: [§A.4](https://arxiv.org/html/2608.17379v3#A1.SS4.p1.1).
- Sun et al. (2026)
  Q. Sun, R. T. Lange, and A. L. Zhang
  SynKer: synthesize, kernelize, reinforce—teaching GPU kernel generation to small language models.
  In ICML 2026 Workshop on Deep Learning for Code,
  External Links: [Link](https://openreview.net/forum?id=oqktWVZOiq)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Tehrani et al. (2026)
  A. Tehrani, Y. Emara, W. Essam, W. Paluch, W. Atallah, Ł. Dudziak, and M. S. Abdelfattah
  Fine-tuning gpt-5 for gpu kernel generation.
  arXiv preprint arXiv:2602.11000.
  External Links: [Link](https://arxiv.org/abs/2602.11000)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Thakkar et al. (2023)
  V. Thakkar, P. Ramani, C. Cecka, A. Shivam, H. Lu, E. Yan, J. Kosaian, M. Hoemmen, H. Wu, A. Kerr, M. Nicely, D. Merrill, D. Blasig, A. Atluri, F. Qiao, P. Majcher, P. Springer, M. Hohnerbach, J. Wang, and M. Gupta
  CUTLASS.
  Note: <https://github.com/NVIDIA/cutlass>CUDA C++ template abstractions for high-performance
  matrix multiplication and related computations
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- Tillet et al. (2019)
  P. Tillet, H. Kung, and D. Cox
  Triton: an intermediate language and compiler for tiled neural network computations.
  In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages,
  pp. 10–19.
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- Tschand et al. (2026)
  A. Tschand, K. Ramakrishnan, A. Ingare, S. Guo, J. J. Ma, Z. Wan, S. Arora, A. Mirhoseini, and V. J. Reddi
  Hawkeye: hardware-aware GPU kernel optimization with minimal supervision.
  In Deep Learning for Code: Towards Human-Centered Coding Agents,
  External Links: [Link](https://openreview.net/forum?id=e3pxJbBRBk)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Veitner (2026)
  S. Veitner
  SBO and LBO explained visually.
  Note: Accessed: 2026-07-29
  External Links: [Link](https://veitner.bearblog.dev/sbo-and-lbo-explained-visually/)
  Cited by: [§A.4](https://arxiv.org/html/2608.17379v3#A1.SS4.p1.1).
- Wang et al. (2026)
  H. Wang, J. Zhang, K. Jiang, H. Wang, J. Chen, and J. Zhu
  KernelBenchX: a comprehensive benchmark for evaluating LLM-generated GPU kernels.
  arXiv preprint arXiv:2605.04956.
  External Links: [Link](https://arxiv.org/abs/2605.04956)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Wang et al. (2025)
  J. Wang, V. Joshi, S. Majumder, X. Chao, B. Ding, Z. Liu, P. P. Brahma, D. Li, Z. Liu, and E. Barsoum
  GEAK: introducing Triton kernel AI agent and evaluation benchmarks.
  arXiv preprint arXiv:2507.23194.
  External Links: [Link](https://arxiv.org/abs/2507.23194)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Woo et al. (2025)
  J. Woo, S. Zhu, A. Nie, Z. Jia, Y. Wang, and Y. Park
  TritonRL: training LLMs to think and code Triton without cheating.
  arXiv preprint arXiv:2510.17891.
  External Links: [Link](https://arxiv.org/abs/2510.17891)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p2.1).
- Xing et al. (2026)
  S. Xing, Y. Zhai, A. Jiang, Y. Dong, Y. Wu, Z. Ye, C. F. Ruan, Y. Huang, Y. Zhang, L. Yin, A. Bayyapu, L. Ceze, and T. Chen
  FlashInfer-Bench: building the virtuous cycle for AI-driven LLM systems.
  In Proceedings of Machine Learning and Systems, A. Chowdhery and Z. Jia (Eds.),
  Vol. 8, pp. 2016–2064.
  External Links: [Link](https://proceedings.mlsys.org/paper_files/paper/2026/file/37e44c4b5321605735be9761f9b758fc-Paper-Conference.pdf)
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p2.1).
- Ye et al. (2025)
  Z. Ye, L. Chen, R. Lai, W. Lin, Y. Zhang, S. Wang, T. Chen, B. Kasikci, V. Grover, A. Krishnamurthy, et al.
  Flashinfer: efficient and customizable attention engine for llm inference serving.
  Proceedings of Machine Learning and Systems 7.
  Cited by: [§2.1](https://arxiv.org/html/2608.17379v3#S2.SS1.p1.1).
- Z.ai (2026)
  Z.ai
  GLM-5.2.
  Note: <https://z.ai/blog/glm-5.2>Released June 16, 2026; knowledge cutoff not publicly disclosed
  Cited by: [§4.2](https://arxiv.org/html/2608.17379v3#S4.SS2.p2.1).
- Zang et al. (2026)
  P. Zang, J. Tao, J. Zhang, Y. Yuan, W. Zhang, G. Liu, and Y. Lin
  KernelGenBench: a multi-source and multi-chip benchmark for LLM-based kernel generation.
  arXiv preprint arXiv:2607.27231.
  External Links: [Link](https://arxiv.org/abs/2607.27231)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).
- Zhang et al. (2026)
  G. Zhang, S. Zhu, A. Wei, Z. Song, A. Nie, Z. Jia, N. Vijaykumar, Y. Wang, and K. Olukotun
  AccelOpt: a self-improving LLM agentic system for AI accelerator kernel optimization.
  In Proceedings of Machine Learning and Systems, A. Chowdhery and Z. Jia (Eds.),
  Vol. 8, pp. 541–568.
  External Links: [Link](https://proceedings.mlsys.org/paper_files/paper/2026/file/0f8426558905746fc38da5e335700aec-Paper-Conference.pdf)
  Cited by: [§2.2](https://arxiv.org/html/2608.17379v3#S2.SS2.p1.1).
- Zhao et al. (2025)
  C. Zhao, Z. Xu, L. Zhao, J. Li, C. Xu, A. Xu, S. Liu, K. Zhou, and K. Yu
  DeepGEMM: clean and efficient BLAS kernel library on GPU.
   GitHub.
  Note: <https://github.com/deepseek-ai/DeepGEMM>
  Cited by: [§1](https://arxiv.org/html/2608.17379v3#S1.p1.1).
- Zheng et al. (2024)
  L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, et al.
  Sglang: efficient execution of structured language model programs.
  Advances in neural information processing systems 37, pp. 62557–62583.
  Cited by: [§A.3](https://arxiv.org/html/2608.17379v3#A1.SS3.p2.1).
- Zheng et al. (2026)
  S. Zheng, X. Zheng, H. Sun, Q. Hou, W. Bao, S. Li, H. Duanmu, J. Fang, C. Xue, C. Huang, et al.
  DITRON: distributed multi-level tiling compiler for parallel tensor programs.
  In Forty-third International Conference on Machine Learning,
  Cited by: [§A.4](https://arxiv.org/html/2608.17379v3#A1.SS4.p1.1).
- Zhu et al. (2026)
  J. Zhu, W. Chen, Q. Fan, Z. Ren, J. Wu, X. Z. Chai, C. Rungrueangwutthinon, Y. Ma, and A. Zou
  CUDABench: benchmarking LLMs for text-to-CUDA generation.
  arXiv preprint arXiv:2603.02236.
  External Links: [Link](https://arxiv.org/abs/2603.02236)
  Cited by: [§6](https://arxiv.org/html/2608.17379v3#S6.p1.1).

## Appendix A Appendix

### A.1 Experiment setup

##### Kernel evaluation.

We evaluate generated CUDA kernels on NVIDIA H100 80 GB (Hopper) and B200 (Blackwell) GPUs. MiniPTXAgent compiles each candidate with nvcc -O3 for sm_90a or sm_100a, respectively, and the profiling service evaluates it against the fixed FlashInfer-Trace workload. Each trajectory has a budget of eight model calls and retains the preceding kernels and execution feedback; it terminates early if the speedup reaches $1.2\times$ or the LLM runs out of context. We did not apply NVIDIA Compute Sanitizer’s racecheck because it is too slow (regularly over $1000\times$ the runtime of a native execution [[Stephenson et al., 2026](https://arxiv.org/html/2608.17379v3#bib.bib40)]) and does not eliminate false negatives [[Chatterjee et al., 2025](https://arxiv.org/html/2608.17379v3#bib.bib39)].

### A.2 Target instruction execution measurement

##### Implementation details.

For every functionally correct candidate, we compile the exact submitted source for the evaluation architecture and inspect the cubin embedded in the resulting shared object with cuobjdump --dump-sass [[NVIDIA Corporation, 2026b](https://arxiv.org/html/2608.17379v3#bib.bib58)]. We hash both the source and cubin container so that cached evidence is invalidated when either changes.

Table 6: PTX interfaces and SASS execution criteria for the selected BF16 mechanisms.



<table>
<tr>
<td>Mechanism</td>
<td>Indicator</td>
<td>PTX interface</td>
<td>SASS execution criterion</td></tr>
<tr>
<td rowspan="2"><span>Tensor compute</span></td>
<td>$C_{\mathrm{Gen4}}$</td>
<td><span>wgmma.mma_async</span></td>
<td>$N_{\mathrm{HGMMA}}>0$</td></tr>
<tr>
<td>$C_{\mathrm{Gen5}}$</td>
<td><span></span><span>
<span>
<span>
<span><span>tcgen05.mma</span>,</span></span>
<span>
<span><span>tcgen05.ld</span></span></span>
</span></span><span></span></td>
<td>$(N_{\mathrm{UTCHMMA}}>0)\land(N_{\mathrm{LDTM}}>0\lor N_{\mathrm{LDT}}>0)$</td></tr>
<tr>
<td rowspan="2"><span>Data movement</span></td>
<td>$D_{\mathrm{TMA}}$</td>
<td><span>cp.async.bulk.tensor</span></td>
<td>$N_{\mathrm{UTMALDG}}>0\lor N_{\mathrm{UTMASTG}}>0$</td></tr>
<tr>
<td>$D_{\mathrm{TMEM}}$</td>
<td><span>tcgen05.cp</span>, <span>tcgen05.st</span></td>
<td>$N_{\mathrm{UTCCP}}>0\lor N_{\mathrm{STTM}}>0\lor N_{\mathrm{STT}}>0$</td></tr>
</table>



We replay each correct candidate on the original workload with Nsight Compute 2025.3.1, using an NVTX range to isolate one invocation and summing instruction-level counts over all kernels launched in that range. Warmup runs occur outside the range. For each SASS family $s$, $N_{s}$ is its predicate-enabled thread execution count (thread_inst_executed_true) [[NVIDIA Corporation, 2026e](https://arxiv.org/html/2608.17379v3#bib.bib59)]. On H100, $C=C_{\mathrm{Gen4}}$ and $D=D_{\mathrm{TMA}}$; on B200, $C=C_{\mathrm{Gen5}}$ and $D=D_{\mathrm{TMA}}\lor D_{\mathrm{TMEM}}$. A correct candidate qualifies if $C\lor D$. TMEM readback is required for $C_{\mathrm{Gen5}}$ because tensor-core results cannot be accessed without it. Cache-control, prefetch, and UTMAREDG instructions do not satisfy $D$.

##### Why static presence is insufficient.

We found two cases that motivate dynamic profiling and conservative handling of unclassifiable turns. In the first, a functionally correct candidate contained a selected SASS instruction but did not execute it. The candidate placed a TMA operation in an unused scanner-satisfaction kernel while run() launched only ordinary numerical kernels:



````C++
__global__ void dummy_target_kernel(...) {
  // Contains cp.async.bulk.tensor; never launched.
}
extern "C" void run(...) {
  numerical_kernel_1<<<...>>>(...);
  numerical_kernel_2<<<...>>>(...);
}
````



The cubin therefore contained UTMALDG, but NCU observed no selected instruction at runtime and the turn did not qualify.

##### Undefined behavior during profiling.

In the second case, a functionally correct Hopper candidate could not be classified by runtime profiling because it contains a stream-ordered use-after-free: a later kernel is enqueued on the same stream after cudaFreeAsync and reads the freed temporary storage. This later access has undefined behavior under the CUDA API [[NVIDIA Corporation, 2026g](https://arxiv.org/html/2608.17379v3#bib.bib60)]. More generally, undefined behavior places no requirements on program outcomes: execution may crash, silently produce incorrect results, or appear to behave as intended [[Stephenson et al., 2026](https://arxiv.org/html/2608.17379v3#bib.bib40)].



````C++
producer<<<..., stream>>>(tmp);
consumer_1<<<..., stream>>>(tmp);
cudaFreeAsync(tmp, stream);
consumer_2<<<..., stream>>>(tmp);  // reads after the queued free
````



This candidate accounts for the one-turn gap between the 4.2% target instruction rate and the 5.2% unrestricted correctness rate for Qwen3.6-27B-s1 on MHA-Bwd-Causal at eight turns. This case is neither a positive nor a measured negative. We conservatively exclude it from the target instruction numerator and retain its status as unknown.

##### Workloads.

Our five primary BF16 workloads comprise four attention variants and one GEMM. All primary attention workloads use batch size 4, 48 heads, sequence length 4096, and head dimension 128, so $Q$, $K$, and $V$ have shape $4\times 48\times 4096\times 128$. The forward workloads are non-causal and causal MHA; each returns a BF16 output of the same shape and an FP32 log-sum-exp tensor of shape $4\times 48\times 4096$. The corresponding backward workloads additionally consume the forward output, its upstream gradient, and the log-sum-exp tensor, and return BF16 gradients for $Q$, $K$, and $V$. The causal variants apply a lower-triangular attention mask. The GEMM computes $C=AB^{\top}$ at BF16 with $A\in\mathbb{R}^{8192\times 5120}$, $B\in\mathbb{R}^{7168\times 5120}$, and $C\in\mathbb{R}^{8192\times 7168}$.

##### Generalization attention workloads.

[Figure 4](https://arxiv.org/html/2608.17379v3#S5.F4) additionally evaluates other attention variants. The d64 and d96 workloads retain batch size 4, 48 heads, and sequence length 4096. The GQA column pools turn-level results from three BF16 workloads. Ragged causal prefill uses 32 query/output heads, 4 KV heads, head dimension 128, 33 sequences, and 16,294 total query and KV tokens. Paged causal prefill uses 24 query/output heads, 8 KV heads, head dimension 128, page size 1, one sequence with 892 query tokens and 892 KV-page indices, and a 43,676-page cache. Paged decode uses the same head counts, dimension, and page size with batch size 15, 9,625 KV-page indices, and a 42,784-page cache. Each GQA workload returns BF16 attention and FP32 log-sum-exp outputs.

##### LLM inference.

At inference, the base and adapted Qwen models share the same decoding settings: temperature 1.0, top-$p$ 0.95, top-$k$ 20, presence penalty 1.5, and at most 81,920 output tokens. For the other evaluated models, we do not override temperature, top-$p$, or top-$k$. Inkling uses the Tinker Anthropic-compatible API with at most 65,536 output tokens. GLM-5.2 uses OpenRouter with at most 131,072 output tokens and is restricted to the z-ai/fp8 or fireworks provider. Gemini 3.1 Pro uses the API-default thinking and output-length settings. Claude Opus 4.8 uses adaptive thinking at xhigh effort and at most 128,000 output tokens.

##### Supervised fine-tuning.

All six recipe-ablation SFT models start independently from Qwen/Qwen3.6-27B. We use Tinker’s supervised learning recipe with LoRA rank 32, train for five epochs with batch size 2, and optimize only the final assistant message in each example. The learning rate is $4.65\times 10^{-4}$ with a linear schedule; Adam uses $\beta_{1}=0.9$, $\beta_{2}=0.95$, and $\epsilon=10^{-8}$. We shuffle each ablation dataset with seed 0, use no validation split, discard examples longer than 65,536 tokens to comply with the token limit of the Tinker API, save every 50 optimizer steps, and evaluate the final checkpoint. For the supplementary seed study, we retrain the fixed 158-record 4ops s1 recipe with shuffle seeds 1 and 2 under the same optimization settings. Dataset construction and record counts after filtering are reported in [Table 12](https://arxiv.org/html/2608.17379v3#A1.T12). We use expert guidance for SFT attention evaluation.

### A.3 Profiling Service

Large-scale multi-turn evaluation profiles many changing kernels against a comparatively stable collection of workloads. Our implementation therefore retains and reuses workload state and overlaps LLM generation with kernel profiling. We measure how these choices affect evaluation throughput and profiling-GPU requirements.

We evaluate MHA-d64 rollouts from a 27B dense model served by SGLang [[Zheng et al., 2024](https://arxiv.org/html/2608.17379v3#bib.bib22)] with TP=2 on two H200 GPUs, with the profiling service running on H100 GPUs. An H100 SXM provides 3.35 TB/s HBM bandwidth but only 128 GB/s bidirectional PCIe bandwidth. As shown in [Figure 12](https://arxiv.org/html/2608.17379v3#A1.F12)(a), after reusing workload state, a $2.72\times$ gap remains, suggesting an opportunity for further optimization in the operating system or driver. The gap is smaller than the bandwidth ratio because each solution executes $3\times(1\text{ correctness}+10\text{ warmup}+50\text{ timed})=183$ times.

[Figure 12](https://arxiv.org/html/2608.17379v3#A1.F12) evaluates whether the shared profiling service can efficiently support concurrent agent trajectories. With LLM generation and kernel profiling pipelined, rollout throughput rises by $2.78\times$ as concurrency grows from 4 to 48; throughput subsequently saturates, and four profiling GPUs provide approximately the same end-to-end throughput as eight. Separately, retaining workloads, reference outputs, and reference latencies on the GPU improves /evaluate throughput by $2.24\times$. With generation–profiling overlap and workload-state reuse, a small GPU pool supports many concurrent trajectories.

![Refer to caption](assets/04-ptxbench/figure-011.png)


Figure 11: Detailed execution infrastructure supporting PTXBench. MiniPTXAgent compiles generated CUDA–PTX locally and sends successfully compiled candidates to an isolated GPU profiling service for sanitization, evaluation, and optional diagnostics. A reliability monitor pauses dispatch, restarts unhealthy service state, and excludes affected turns from trajectories. After a restart, it also excludes thermally abnormal GPUs because throttling distorted latency by over 15% in our measurements [[NVIDIA Corporation, 2026f](https://arxiv.org/html/2608.17379v3#bib.bib26)].

![Refer to caption](assets/04-ptxbench/figure-012.png)


Figure 12: Workload-state caching (a), rollout concurrency with generation–profiling pipelining (b), and profiling-GPU scaling (c).

### A.4 Architecture-specific prompt token counts

[Table 7](https://arxiv.org/html/2608.17379v3#A1.T7) reports Qwen-3.6-27B’s token counts of the architecture parameters, architecture contracts, and PTX template functions included in the H100 and B200 prompts. B200 prompts are longer because they include additional architecture contracts such as descriptors for shared memory, peer CTAs, and tensor memory. Architecture-specific PTX is unevenly documented and sometimes underspecified [[Veitner, 2026](https://arxiv.org/html/2608.17379v3#bib.bib27)] or even wrong [[Sul and Ré, 2026](https://arxiv.org/html/2608.17379v3#bib.bib28)]. Each pack contains architecture parameters, C++ wrappers for PTX instructions, and contracts such as layouts and memory-consistency rules. Some wrappers are adapted from LittleKernel [[Zheng et al., 2026](https://arxiv.org/html/2608.17379v3#bib.bib20)]; comments retain complete polymorphic instruction definitions while the wrapper demonstrates one representative instance. We validate each pack against documentation, reports, and execution.

Table 7: Token counts for architecture-specific prompt components.



<table>
<tr>
<td>GPU</td>
<td>Component</td>
<td>Tokens</td></tr>
<tr>
<td rowspan="4"><span>H100</span></td>
<td>Architecture parameter</td>
<td>259</td></tr>
<tr>
<td>Template functions</td>
<td>18,601</td></tr>
<tr>
<td>Architecture contract</td>
<td>3,454</td></tr>
<tr>
<td>Total</td>
<td>22,314</td></tr>
<tr>
<td rowspan="4"><span>B200</span></td>
<td>Architecture parameter</td>
<td>413</td></tr>
<tr>
<td>Template functions</td>
<td>18,871</td></tr>
<tr>
<td>Architecture contract</td>
<td>9,531</td></tr>
<tr>
<td>Total</td>
<td>28,815</td></tr>
</table>



### A.5 Published and corrected FlashAttention-3 pseudocode

When preparing the controlled context, we found two inconsistencies in the published FlashAttention-3 Algorithm 2 [[Shah et al., 2024](https://arxiv.org/html/2608.17379v3#bib.bib21)]: the loop condition $j<T_{c}-1$ leaves $S_{T_{c}-1}$ and its softmax uncomputed, and the loop body computes $\widetilde{P}_{\mathrm{next}}$ without assigning it to $\widetilde{P}_{\mathrm{cur}}$. Our corrected prompt version iterates through $j=T_{c}-1$, carries both pipeline states forward, records the old row maximum before rescaling the accumulated output, and normalizes by $\ell_{i}$ in the epilogue. Listings  and  reproduce the relevant published consumer-warpgroup pseudocode and the corrected version supplied in our prompt, reinforcing the controlled-context design choice.



````
Require: $Q_i \in \mathbb{R}^{B_r \times d}$ and $K,V \in \mathbb{R}^{N \times d}$ in HBM; key block size $B_c$ and $T_c=\lceil N/B_c\rceil$.
Reallocate registers as a function of the number of consumer warps.
On chip, initialize $O_i=0$ and $\ell_i,m_i=0,-\infty$.
Wait for $Q_i$ and $K_0$ in shared memory.
Compute $S_{\mathrm{cur}}=Q_iK_0^\top$ using WGMMA. Commit and wait.
Release stage 0 of the buffer for $K$.
Compute $m_i$, $\widetilde{P}_{\mathrm{cur}}$, and $\ell_i$ from $S_{\mathrm{cur}}$, and rescale $O_i$.
for $1 \le j < T_c-1$ do
  Wait for $K_j$ in shared memory.
  Compute $S_{\mathrm{next}}=Q_iK_j^\top$ using WGMMA. Commit; do not wait.
  Wait for $V_{j-1}$ in shared memory.
  Compute $O_i=O_i+\widetilde{P}_{\mathrm{cur}}V_{j-1}$ using WGMMA. Commit; do not wait.
  Wait for the WGMMA $Q_iK_j^\top$.
  Compute $m_i$, $\widetilde{P}_{\mathrm{next}}$, and $\ell_i$ from $S_{\mathrm{next}}$.
  Wait for the WGMMA $\widetilde{P}_{\mathrm{cur}}V_{j-1}$; then rescale $O_i$.
  Release stages $(j \bmod s)$ and $((j-1) \bmod s)$ for $K$ and $V$, respectively.
  Copy $S_{\mathrm{next}}$ to $S_{\mathrm{cur}}$.
end for
Wait for $V_{T_c-1}$ in shared memory.
Compute $O_i=O_i+\widetilde{P}_{\mathrm{last}}V_{T_c-1}$ using WGMMA. Commit and wait.
Epilogue: Rescale $O_i$ based on $m_i$. Compute $L_i$ from $m_i$ and $\ell_i$; write $O_i,L_i$ to HBM.
````



Listing 1: FlashAttention-3 Algorithm 2 as published. The loop bound and missing probability-state update make the pseudocode internally inconsistent.




````
Require: $Q_i \in \mathbb{R}^{B_r \times d}$ and $K,V \in \mathbb{R}^{N \times d}$ in HBM; key block size $B_c$ and $T_c=\lceil N/B_c\rceil$.
Reallocate registers as a function of the number of consumer warps.
On chip, initialize $O_i=0$ and $\ell_i,m_i=0,-\infty$.
Wait for $Q_i$ and $K_0$ in shared memory.
Compute $S_{\mathrm{cur}}=Q_iK_0^\top$ using WGMMA. Commit and wait.
Release stage 0 of the buffer for $K$.
Compute $m_i$, $\widetilde{P}_{\mathrm{cur}}$, and $\ell_i$ from $S_{\mathrm{cur}}$.
for $1 \le j < T_c$ do
  Wait for $K_j$ in shared memory.
  Compute $S_{\mathrm{next}}=Q_iK_j^\top$ using WGMMA. Commit; do not wait.
  Wait for $V_{j-1}$ in shared memory.
  Compute $O_i=O_i+\widetilde{P}_{\mathrm{cur}}V_{j-1}$ using WGMMA. Commit; do not wait.
  Wait for the WGMMA $Q_iK_j^\top$.
  Save $m_i^{\mathrm{old}} \leftarrow m_i$.
  Update $m_i$ and $\ell_i$ online; compute $\widetilde{P}_{\mathrm{next}}=\exp(S_{\mathrm{next}}-m_i)$.
  Wait for the WGMMA $\widetilde{P}_{\mathrm{cur}}V_{j-1}$; then set $O_i=\operatorname{diag}(\exp(m_i^{\mathrm{old}}-m_i))O_i$.
  Release stages $(j \bmod s)$ and $((j-1) \bmod s)$ for $K$ and $V$, respectively.
  Copy $S_{\mathrm{next}}$ to $S_{\mathrm{cur}}$ and $\widetilde{P}_{\mathrm{next}}$ to $\widetilde{P}_{\mathrm{cur}}$.
end for
Wait for $V_{T_c-1}$ in shared memory.
Compute $O_i=O_i+\widetilde{P}_{\mathrm{cur}}V_{T_c-1}$ using WGMMA. Commit and wait.
Epilogue: Set $O_i=\operatorname{diag}(\ell_i)^{-1}O_i$ and $L_i=m_i+\log(\ell_i)$; write $O_i,L_i$ to HBM.
````



Listing 2: Corrected FlashAttention-3 Algorithm 2 used in our prompt.

### A.6 Additional Language and Harness Results

#### A.6.1 GPT-5.6 Sol Triton Comparison

[Figure 13](https://arxiv.org/html/2608.17379v3#A1.F13) applies the same $\mathrm{Fast}_{p}$ definition, three-prompt design, and 96-turn denominator per model–GPU–workload condition as the Gemini comparison in [Figure 3](https://arxiv.org/html/2608.17379v3#S4.F3). On B200, Triton’s best correct speedup exceeds CUDA–PTX on all five workloads; for MHA-Bwd it reaches $0.628\times$ versus $0.372\times$. On H100, individual CUDA–PTX kernels reach a higher peak on MHA-Fwd-Causal ($0.865\times$ versus $0.852\times$) and MHA-Bwd-Causal ($0.746\times$ versus $0.734\times$), although the full $\mathrm{Fast}_{p}$ curves show how frequently correct kernels attain each speedup threshold.

![Refer to caption](assets/04-ptxbench/figure-013.png)


Figure 13: GPT-5.6 Sol $\mathrm{Fast}_{p}$ distributions for Triton and CUDA-PTX on H100 and B200.

#### A.6.2 Codex Coding-Agent Comparison

[Table 8](https://arxiv.org/html/2608.17379v3#A1.T8) repeats the eight-request H100 harness comparison of [Section 4.4](https://arxiv.org/html/2608.17379v3#S4.SS4) for Codex and multi-turn refinement using GPT-5.6 Sol. Both use the same task specification, architectural knowledge, and evaluator feedback without scheduling guidance; Codex can use shell and file-editing tools between requests. Each median is over four trajectories, including zero for a trajectory with no correct instruction-qualified candidate. Codex leads on GEMM, while refinement leads on all four attention workloads. Across the five workloads, Codex’s median cumulative total-token cost is 3.47–5.78$\times$ that of refinement. Shared-budget median curves for both coding agents are shown in [Figure 14](https://arxiv.org/html/2608.17379v3#A1.F14); total tokens count reported input, including cached input once per request, and output, including reasoning.

Table 8: Codex versus multi-turn refinement after eight H100 evaluation requests per trajectory. Speedups are medians of four per-trajectory best instruction-qualified values; token ratios compare workload-level medians of cumulative total tokens.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Method or ratio | GEMM | MHA-Fwd | MHA-Fwd-Causal | MHA-Bwd | MHA-Bwd-Causal |
| Multi-turn refinement | $0.641\times$ | $0.682\times$ | $0.805\times$ | $0.383\times$ | $0.214\times$ |
| Codex | $0.725\times$ | $0.565\times$ | $0.581\times$ | $0.274\times$ | $0.116\times$ |
| Agent / refinement total tokens | $5.78\times$ | $4.50\times$ | $3.47\times$ | $5.60\times$ | $4.20\times$ |

![Refer to caption](assets/04-ptxbench/figure-014.png)


Figure 14: Median best instruction-qualified speedup at shared per-trajectory total-token budgets on H100 for Antigravity/Gemini 3.1 Pro (top) and Codex/GPT-5.6 Sol (bottom), each compared with multi-turn refinement using the same model. Each curve includes four trajectories per method and workload; bands show pointwise 95% bootstrap intervals. The horizontal axis is logarithmic.

### A.7 Supplementary Results

![Refer to caption](assets/04-ptxbench/figure-015.png)

![Refer to caption](assets/04-ptxbench/figure-016.png)

Figure 15: H100 $\mathrm{Fast}_{p}$ and $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$.


![Refer to caption](assets/04-ptxbench/figure-017.png)

![Refer to caption](assets/04-ptxbench/figure-018.png)

Figure 16: B200 $\mathrm{Fast}_{p}$ and $\mathrm{Fast}^{\mathrm{Inst.}}_{p}$.


Table 9: Best speedup with target instruction execution on H100 and B200.



<table>
<tr>
<td>GPU</td>
<td>Model</td>
<td>GEMM</td>
<td>MHA-Fwd</td>
<td>MHA-Fwd-Causal</td>
<td>MHA-Bwd</td>
<td>MHA-Bwd-Causal</td></tr>
<tr>
<td rowspan="5"><span>H100</span></td>
<td>Gemini 3.1 Pro</td>
<td><span></span> <span>
<span>
<span>
<span>0.687 / 0.934 / 0.962</span></span>
<span>
<span><span>(0.687 / 0.934 / 0.962)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.555 / 0.730 / 0.730</span></span>
<span>
<span><span>(0.555 / 0.730 / 0.730)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.614 / 0.651 / 0.768</span></span>
<span>
<span><span>(0.614 / 0.651 / 0.768)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.206 / 0.375 / 0.515</span></span>
<span>
<span><span>(0.206 / 0.375 / 0.515)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.634 / 0.639</span></span>
<span>
<span><span>(0.065 / 0.634 / 0.639)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Claude Opus 4.8</td>
<td><span></span> <span>
<span>
<span>
<span>0.770 / 0.968 / <span>0.976</span></span></span>
<span>
<span><span>(0.770 / 0.968 / 0.976)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.759 / 0.770 / 0.839</span></span>
<span>
<span><span>(0.759 / 0.770 / 0.839)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.758 / 0.806 / 0.806</span></span>
<span>
<span><span>(0.758 / 0.806 / 0.806)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.300 / 0.440 / 0.489</span></span>
<span>
<span><span>(0.300 / 0.440 / 0.489)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.499 / 0.499</span></span>
<span>
<span><span>(0.058 / 0.499 / 0.499)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>GLM-5.2</td>
<td><span></span> <span>
<span>
<span>
<span>0.447 / 0.692 / 0.692</span></span>
<span>
<span><span>(0.447 / 0.692 / 0.692)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.407 / 0.470 / 0.607</span></span>
<span>
<span><span>(0.407 / 0.470 / 0.607)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.471 / 0.533</span></span>
<span>
<span><span>(0.015 / 0.471 / 0.533)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.316 / 0.437 / 0.437</span></span>
<span>
<span><span>(0.316 / 0.437 / 0.437)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / 0.101 / 0.101)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Qwen3.6-27B</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td></tr>
<tr>
<td>GPT-5.6 xhigh</td>
<td><span></span> <span>
<span>
<span>
<span>0.788 / 0.870 / 0.870</span></span>
<span>
<span><span>(0.788 / 0.870 / 0.870)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.282 / 0.774 / <span>0.872</span></span></span>
<span>
<span><span>(0.282 / 0.774 / 0.872)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.699 / <span>0.865</span></span></span>
<span>
<span><span>(– / 0.699 / 0.865)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.230 / 0.699 / <span>0.703</span></span></span>
<span>
<span><span>(0.230 / 0.699 / 0.703)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.650 / <span>0.746</span></span></span>
<span>
<span><span>(0.094 / 0.650 / 0.746)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td rowspan="5"><span>B200</span></td>
<td>Gemini 3.1 Pro</td>
<td><span></span> <span>
<span>
<span>
<span>0.273 / 0.680 / 0.892</span></span>
<span>
<span><span>(0.273 / 0.680 / 0.892)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.280 / 0.280</span></span>
<span>
<span><span>(– / 0.280 / 0.280)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.206 / 0.248</span></span>
<span>
<span><span>(0.013 / 0.206 / 0.248)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.087 / 0.133</span></span>
<span>
<span><span>(– / 0.087 / 0.133)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / 0.015 / 0.015)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Claude Opus 4.8</td>
<td><span></span> <span>
<span>
<span>
<span>0.782 / 1.012 / <span>1.012</span></span></span>
<span>
<span><span>(0.782 / 1.012 / 1.012)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.253 / 0.300</span></span>
<span>
<span><span>(0.110 / 0.253 / 0.300)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.232 / 0.269</span></span>
<span>
<span><span>(0.042 / 0.232 / 0.269)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.069 / 0.149</span></span>
<span>
<span><span>(0.022 / 0.136 / 0.155)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(0.023 / 0.088 / 0.116)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>GLM-5.2</td>
<td><span></span> <span>
<span>
<span>
<span>0.162 / 0.632 / 0.632</span></span>
<span>
<span><span>(0.162 / 0.632 / 0.632)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / – / 0.027</span></span>
<span>
<span><span>(0.024 / 0.024 / 0.035)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>0.098 / 0.098 / 0.098</span></span>
<span>
<span><span>(0.098 / 0.098 / 0.098)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(0.019 / 0.030 / 0.040)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(0.018 / 0.034 / 0.034)</span></span></span>
</span></span><span></span></td></tr>
<tr>
<td>Qwen3.6-27B</td>
<td><span></span> <span>
<span>
<span>
<span>–</span></span>
<span>
<span><span>(– / – / 0.006)</span></span></span>
</span></span><span></span></td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td></tr>
<tr>
<td>GPT-5.6 xhigh</td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.781 / 0.818</span></span>
<span>
<span><span>(0.175 / 0.781 / 0.818)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.419 / <span>0.536</span></span></span>
<span>
<span><span>(0.068 / 0.419 / 0.536)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.294 / <span>0.445</span></span></span>
<span>
<span><span>(0.132 / 0.294 / 0.445)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.361 / <span>0.372</span></span></span>
<span>
<span><span>(0.059 / 0.361 / 0.372)</span></span></span>
</span></span><span></span></td>
<td><span></span> <span>
<span>
<span>
<span>– / 0.339 / <span>0.339</span></span></span>
<span>
<span><span>(0.066 / 0.339 / 0.339)</span></span></span>
</span></span><span></span></td></tr>
</table>



##### Shuffle-seed sensitivity.

[Figure 17](https://arxiv.org/html/2608.17379v3#A1.F17) reports two additional shuffle seeds for the fixed 158-record 4ops s1 recipe. Absolute correctness rates and peak speedups vary across seeds, but both repeats produce correct kernels on every d128 training problem, GEMM, all four d64 problems, and both d96 forward problems; neither solves either d96 backward problem. Both produce correct GQA candidates, but none of those candidates execute the target instructions. Together with the seed-0 result in [Figure 4](https://arxiv.org/html/2608.17379v3#S5.F4), these repeats show that the qualitative transfer pattern is stable despite substantial per-problem variation.

![Refer to caption](assets/04-ptxbench/figure-019.png)

![Refer to caption](assets/04-ptxbench/figure-020.png)

Figure 17: Generalization of the 158-record 4ops s1 recipe under additional SFT shuffle seeds. Top: seed 1. Bottom: seed 2. The seed-1 GEMM cell contains 90 of 96 expected turn rows; all other plotted cells contain the complete 96 turns. Runtime-instruction rows require dynamically observed Hopper SASS execution.


Table 10: Turn correctness rates for the five-problem SFT evaluation.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| SFT-ed Model Label | GEMM | MHA-Fwd | MHA-Fwd-Causal | MHA-Bwd | MHA-Bwd-Causal |
| Qwen3.6-27B-s1 | 25.0 / 14.6 / 13.5  (25.0 / 14.6 / 13.5) | 16.7 / 16.7 / 19.8  (16.7 / 16.7 / 19.8) | – / – / 3.1  (– / – / 3.1) | – / 2.1 / 4.2  (– / 2.1 / 4.2) | – / 2.1 / 4.2  (– / 2.1 / 5.2) |
| Qwen3.6-27B-s2 | – / 2.1 / 2.1  (– / 2.1 / 2.1) | – / 2.1 / 5.2  (– / 2.1 / 5.2) | – / – / 1.0  (– / – / 1.0) | – / – / 1.0  (– / – / 1.0) | – |
| Qwen3.6-27B-s3 | – / 2.1 / 3.1  (– / 2.1 / 3.1) | – / 4.2 / 2.1  (– / 4.2 / 2.1) | – / 6.2 / 4.2  (– / 6.2 / 4.2) | – / 4.2 / 4.2  (– / 4.2 / 4.2) | – |
| Qwen3.6-27B-s4 | 8.3 / 8.3 / 10.4  (8.3 / 8.3 / 10.4) | – / – / 4.2  (– / – / 4.2) | – / 4.2 / 2.1  (– / 4.2 / 2.1) | – / – / 1.0  (– / – / 1.0) | – |
| Qwen3.6-27B-s5 | 16.7 / 14.6 / 12.5  (16.7 / 14.6 / 12.5) | – / 2.1 / 4.2  (– / 2.1 / 4.2) | – / 2.1 / 3.1  (– / 2.1 / 3.1) | – / – / 5.2  (– / – / 5.2) | – / – / 4.2  (– / – / 4.2) |
| Qwen3.6-27B-s6 | 8.3 / 2.1 / 1.0  (8.3 / 2.1 / 1.0) | – | – | – | – |


Table 11: Best speedups for the five-problem SFT evaluation.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| SFT-ed Model Label | GEMM | MHA-Fwd | MHA-Fwd-Causal | MHA-Bwd | MHA-Bwd-Causal |
| Qwen3.6-27B-s1 | 0.303 / 0.303 / 0.340  (0.303 / 0.303 / 0.340) | 0.556 / 0.556 / 0.565  (0.556 / 0.556 / 0.565) | – / – / 0.395  (– / – / 0.395) | – / 0.380 / 0.389  (– / 0.380 / 0.389) | – / 0.192 / 0.199  (– / 0.192 / 0.199) |
| Qwen3.6-27B-s2 | – / 0.209 / 0.211  (– / 0.209 / 0.211) | – / 0.548 / 0.548  (– / 0.548 / 0.548) | – / – / 0.315  (– / – / 0.315) | – / – / 0.296  (– / – / 0.296) | – |
| Qwen3.6-27B-s3 | – / 0.274 / 0.280  (– / 0.274 / 0.280) | – / 0.465 / 0.465  (– / 0.465 / 0.465) | – / 0.388 / 0.388  (– / 0.388 / 0.388) | – / 0.494 / 0.494  (– / 0.494 / 0.494) | – |
| Qwen3.6-27B-s4 | 0.276 / 0.373 / 0.373  (0.276 / 0.373 / 0.373) | – / – / 0.452  (– / – / 0.452) | – / 0.383 / 0.383  (– / 0.383 / 0.383) | – / – / 0.199  (– / – / 0.199) | – |
| Qwen3.6-27B-s5 | 0.325 / 0.446 / 0.446  (0.325 / 0.446 / 0.446) | – / 0.425 / 0.573  (– / 0.425 / 0.573) | – / 0.241 / 0.246  (– / 0.241 / 0.246) | – / – / 0.295  (– / – / 0.295) | – / – / 0.246  (– / – / 0.246) |
| Qwen3.6-27B-s6 | 0.073 / 0.073 / 0.073  (0.073 / 0.073 / 0.073) | – | – | – | – |


Table 12: Training data recipes.

|  |  |  |  |
| --- | --- | --- | --- |
| SFT-ed Model Label | Config | Reasoning Synthesizer | Record Count |
| Qwen3.6-27B-s1 | 4ops | GLM-5.2 | 158 |
| Qwen3.6-27B-s2 | 4ops-Extended | GLM-5.2 | 259 |
| Qwen3.6-27B-s3 | 8ops-Extended | GLM-5.2 | 406 |
| Qwen3.6-27B-s4 | 8ops-Post-balanced | GLM-5.2 | 170 |
| Qwen3.6-27B-s5 | 8ops-Pre-balanced | GLM-5.2 | 258 |
| Qwen3.6-27B-s6 | 8ops-Pre-balanced | Qwen3.6-27B | 258 |




````
1. In your thinking, write a short "kernel contract" block listing tensor shapes, flattened offsets, tile sizes, TMA descriptor dimensions/strides/boxDim/swizzle, mbarrier counts, and launch shared-memory bytes. The code must match this block exactly.
2. For TMA BF16 with 128B swizzle and flattened `[B*H*S, D]`, use `globalDim={D, B*H*S}`, `globalStrides={D*2}`, and `boxDim={64,128}` unless the kernel explicitly proves a different layout.
3. Every host-side CUDA/driver call must be checked: `CU_CHECK(cuTensorMapEncodeTiled(...))`, `CUDA_CHECK(cudaFuncSetAttribute(...))`, `CUDA_CHECK(cudaGetLastError())`, and `CUDA_CHECK(cudaStreamSynchronize(stream))`.
4. If dynamic shared memory exceeds the default limit, the kernel must opt in with `cudaFuncSetAttribute` using the same byte count as the launch.
5. The online softmax update must explicitly maintain `m_i`, `l_i`, and rescale old `O` accumulators when the row max changes. Any kernel that computes a tile softmax independently and then accumulates `P @ V` is wrong for full attention.
6. Prohibit unsynchronized shared accumulator updates. Each accumulator element must have a single owner thread/warp, or use an explicit reduction with a documented synchronization point.
7. Require a self-audit checklist at the end of generation: descriptor validity, barrier arrival/tx-count matching, WGMMA descriptor LBO/SBO, output/LSE indexing, and launch shared memory.
````



Figure 18: Expert guidance used in MHA evaluation.

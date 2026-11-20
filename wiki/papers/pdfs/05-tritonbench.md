---
title: "TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators"
source_url: "https://arxiv.org/abs/2502.14752"
full_text_url: "https://arxiv.org/html/2502.14752v1"
version: "2502.14752v1"
authors: ["Li, Jianling", "Li, Shangzhan", "Gao, Zhenye", "Shi, Qi", "Li, Yuxuan", "Wang, Zefan", "Huang, Jiacheng", "Wang, Haojie", "Wang, Jianrong", "Han, Xu", "Liu, Zhiyuan", "Sun, Maosong"]
retrieved_date: "2026-11-20"
conversion: "HTML to Markdown with TeX math"
pdf: "05-tritonbench.pdf"
original_html: "sources/05-tritonbench.html"
---

# TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators

Jianling Li11
1
Equal contribution.
Affiliation:  Tianjin University, Tianjin, China
  
Shangzhan Li11
1
Equal contribution.
Affiliation:  Harbin Institute of Technology, Harbin, China
  
Zhenye Gao
Affiliation:  The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China
  
Qi Shi22
2
Corresponding authors.
Affiliation:  Tsinghua University, Beijing, China
  
Yuxuan Li22
2
Corresponding authors.
Affiliation:  Tsinghua University, Beijing, China
  
Zefan Wang
Affiliation:  Tsinghua University, Beijing, China
  
Jiacheng Huang
Affiliation:  Tsinghua University, Beijing, China
  
Haojie Wang
Affiliation:  Tsinghua University, Beijing, China
  
Jianrong Wang
Affiliation:  Tianjin University, Tianjin, China
  
Xu Han
Affiliation:  Tsinghua University, Beijing, China
  
Zhiyuan Liu
Affiliation:  Tsinghua University, Beijing, China
  
Maosong Sun
Affiliation:  Tsinghua University, Beijing, China

###### Abstract

Triton, a high-level Python-like language designed for building efficient GPU kernels, is widely adopted in deep learning frameworks due to its portability, flexibility, and accessibility.
However, programming and parallel optimization still require considerable trial and error from Triton developers.
Despite advances in large language models (LLMs) for conventional code generation, these models struggle to generate accurate, performance-optimized Triton code, as they lack awareness of its specifications and the complexities of GPU programming.
More critically, there is an urgent need for systematic evaluations tailored to Triton.
In this work, we introduce TritonBench, the first comprehensive benchmark for Triton operator generation.
TritonBench features two evaluation channels: a curated set of $184$ real-world operators from GitHub and a collection of operators aligned with PyTorch interfaces.
Unlike conventional code benchmarks prioritizing functional correctness, TritonBench also profiles efficiency performance on widely deployed GPUs aligned with industry applications.
Our study reveals that current state-of-the-art code LLMs struggle to generate efficient Triton operators, highlighting a significant gap in high-performance code generation.
TritonBench will be available at <https://github.com/thunlp/TritonBench>.

## 1 Introduction

Triton [Tillet et al. (2019)](https://arxiv.org/html/2502.14752v1#bib.bib28) language, a high-level Python-like programming language designed for implementing efficient GPU kernels, is playing an increasingly pivotal role in the ever-scaling deep learning ecosystems [Abadi et al. (2016)](https://arxiv.org/html/2502.14752v1#bib.bib1); [Paszke et al. (2019)](https://arxiv.org/html/2502.14752v1#bib.bib22).
Due to the superior portability, flexibility, lightweight design, and accessibility to less proficient programmers, Triton is prevalently adopted in modern Large Language Model (LLM) frameworks such as vLLM [Kwon et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib16), LightLLM [ModelTC (2025)](https://arxiv.org/html/2502.14752v1#bib.bib20), Liger-kernel [Hsu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib12) and unsloth [Daniel Han and team (2023)](https://arxiv.org/html/2502.14752v1#bib.bib9).
However, crafting high-performance operators remains challenging, especially for the intricate balance between memory hierarchy management, parallel thread coordination, and hardware-specific optimizations.
Even though Triton abstracts away many complexities of low-level programming architectures like CUDA, it still requires developers to manually handle critical aspects such as pointer arithmetic and memory access patterns, making performance tuning a labor-intensive process that often involves extensive trial and error.

Current research in AI-assisted coding has reached a human-competitive level [Hui et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib14); [Zhu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib32), yet it is primarily restricted to general-purpose languages like Python.
However, LLMs still face challenges in generating Domain Specific Language (DSL) code.
Specifically for Triton, current models might be unfamiliar with Triton specification and the intricacies of GPU programming [Nichols et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib21).
Most importantly, the ability of these models to produce high-quality Triton code remains unassessed.
Therefore, a high-quality benchmark paired with performance-aware metrics is urgently required.

![Refer to caption](assets/05-tritonbench/figure-001.png)


Figure 1: Illustration of the construction and evaluation of TritonBench.

In this study, we present TritonBench, a performance-aware benchmark framework for Triton generation, which contains two channels, namely TritonBench-G and TritonBench-T.
Specifically, TritonBench-G contains $184$ carefully curated operators from existing GitHub repositories, reflecting the realistic demand for Triton operator development.
As a complement, TritonBench-T is composed of operator development tasks aligned with PyTorch interfaces, covering operators under-represented by public sources.
Moreover, unlike the majority of code benchmarks merely prioritizing functional correctness [Chen et al. (2021)](https://arxiv.org/html/2502.14752v1#bib.bib8); [Austin et al. (2021a)](https://arxiv.org/html/2502.14752v1#bib.bib2), TritonBench emphasizes efficiency performance profiling against reference programs on NVIDIA GPUs, better aligning industrial demands.

As shown in Figure [1](https://arxiv.org/html/2502.14752v1#S1.F1), for TritonBench-G, we follow three steps: 1) scrape and collect high-quality operators, 2) generate instructions via prompts, and 3) annotate test code with LLMs.
Moreover, HPC experts evaluate GPU performance for all triton codes.
For TritonBench-T, we provide operator generation tasks aligned with PyTorch.
To construct these tasks, we first perform a frequency analysis to select torch operators, combine them into diverse sets, and provide paired instructions and test code.
Our evaluation metrics include similarity, call and execution accuracy, speed up, and GPU efficiency.

We conduct extensive experiments across a broad range of LLMs. Overall, the difficulty of TritonBench-G is greater than that of TritonBench-T.
The highest execution accuracy on TritonBench-G can reach $23.91$%, while on TritonBench-T, it can reach $53.01$%.
For all correctly executed operators generated by the models, the best speed up on TritonBench-G is $1.56\times$, whereas, on TritonBench-T, it is $1.91\times$.
Additionally, we perform in-depth analyses of LLMs’ behavior on TritonBench and summarize the challenges in Triton generation.
The results reveal that current LLMs are not yet fully capable of handling TritonBench, underscoring the challenge of enabling LLMs to generate Triton code effectively.
We hope this work initiates evaluation in this under-explored area and fosters advancements in LLM-driven operator development.

## 2 Related Work

### 2.1 Triton Development

Triton [Mitkov et al. (2021)](https://arxiv.org/html/2502.14752v1#bib.bib19) is an open-source, Python-like language and a compiler designed to simplify GPU programming in AI and HPC.
It abstracts the complexities of CUDA by introducing a block-based programming model, automating low-level optimizations such as memory coalescing and tensor core utilization, and making it more accessible to researchers without HPC background.
Nonetheless, Triton provides explicit control over memory access patterns and parallelism.
This balance of productivity and flexibility makes it prevalently adopted in both academia and industry [Kwon et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib16); [ModelTC (2025)](https://arxiv.org/html/2502.14752v1#bib.bib20); [Hsu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib12); [Daniel Han and team (2023)](https://arxiv.org/html/2502.14752v1#bib.bib9).
However, Triton developers must still laboriously tune critical parameters to exploit hardware capabilities.
LLM code generation poses prospects for automating Triton development, which calls for a systematic evaluation of generated operators.

### 2.2 Code Benchmarks

The demand for proper measurement of coding capability arises as the program synthesis research advances.
The primary practice of coding benchmarks is functional correctness testing, usually realized by test case construction and sandbox execution.
For example, HumanEval [Belz et al. (2021)](https://arxiv.org/html/2502.14752v1#bib.bib4) curate hand-written programs and test cases, and MBPP [Austin et al. (2021b)](https://arxiv.org/html/2502.14752v1#bib.bib3) create programming problems by crowd-sourcing.
The functionality test has recently extended to automated test generation for better coverage [Liu et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib17) and broader applications, including software engineering [Jimenez et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib15).
Another vital aspect of coding benchmarking is performance profiling [Shypula et al. ()](https://arxiv.org/html/2502.14752v1#bib.bib27); [Liu et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib17); [Huang et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib13); [Qiu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib24).
However, most existing frameworks focus on competition-style, single-process execution.
While there are some frameworks for evaluating parallel programming on CPUs [Nichols et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib21); [Chaturvedi et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib7), benchmarks targeting GPU code remain scarce.
As the deployment of deep learning models scales up, a comprehensive evaluation framework that considers both correctness and performance on GPU code becomes increasingly necessary.

### 2.3 LLMs for Code Generation

LLMs have recently demonstrated impressive capabilities in generating code from natural language instructions, as evidenced by models such as DeepSeek-Coder [Guo et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib11); [Zhu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib32) and Qwen-Coder [Hui et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib14), which have achieved strong performance on broad coding benchmarks.
Despite their versatility, they often struggle with Domain-Specific Languages (DSLs) designed for higher levels of abstraction and improved efficiency in targeted contexts [Wąsowski and Berger (2023)](https://arxiv.org/html/2502.14752v1#bib.bib29).
The main reason for this status is the limited availability of DSL datasets and benchmarks [Cassano et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib6); [Pujar et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib23), coupled with their unique syntax and semantics [Pujar et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib23), posing significant challenges for LLMs [Buscemi (2023)](https://arxiv.org/html/2502.14752v1#bib.bib5).
In this work, we focus on DSLs within the high-performance computing domain where the challenges we mentioned are more pronounced for involving the parallel programming model.
We introduce the first comprehensive benchmark for Triton generation, providing a systematic evaluation framework that aims to guide future improvements in DSL-centric LLM code generation.

## 3 TritonBench-G

Triton [Tillet et al. (2019)](https://arxiv.org/html/2502.14752v1#bib.bib28) is a DSL that abstracts away low-level complexities to simplify GPU programming for computation-intensive tasks, with flexibility for specialized applications like machine learning.
Typically, a Triton operator includes at least a kernel and a wrapper.
The kernel comprises code executed on the GPU, focusing on tensor element addressing and thread parallel coordination.
Meanwhile, the wrapper offers a Python function that encapsulates the kernel call.
Figure [2](https://arxiv.org/html/2502.14752v1#S3.F2) shows an example of Triton operator.

![Refer to caption](assets/05-tritonbench/figure-002.png)


Figure 2: 
Implementation of the Triton “add” operator. Lines $3$-$6$ perform for tensor element addressing, followed by the calculation and storage in lines $7$-$10$.
The kernel is called in wrapper line $15$.

We create TritonBench-G by curating high-quality human-authored Triton operators from Github, which reflects Triton’s currently actual requirements.
The following sections will explain data collection (§ [3.1](https://arxiv.org/html/2502.14752v1#S3.SS1)), data statistics (§ [3.2](https://arxiv.org/html/2502.14752v1#S3.SS2)), operator quality rating (§ [3.3](https://arxiv.org/html/2502.14752v1#S3.SS3)), test code design (§ [3.4](https://arxiv.org/html/2502.14752v1#S3.SS4)), and evaluation metrics (§ [3.5](https://arxiv.org/html/2502.14752v1#S3.SS5)).

### 3.1 Data Collection

Our process starts by gathering Triton-related GitHub repositories with more than $100$ stars, which collectively encompass $95$ repositories with $845$ Python files.
As Triton repositories with higher star counts are rare, $100$ stars serve as an optimal threshold, striking a balance between quality and quantity.
We then use prompt-based filtering (see prompt [D](https://arxiv.org/html/2502.14752v1#A4) in the Appendix) to process the candidate Python files and select $250$ that specifically contain Triton code snippets.

Afterward, we perform a rigorous manual inspection of the Triton code to ensure its accuracy and clarity.
This process involves filling in missing components, removing redundant sections, and debugging the operators.
When a file contains multiple independent Triton operators, we split them into separate files. For operators that are solely kernels, we add the necessary wrappers to ensure they work as intended. Additionally, to ensure uniqueness, we leverage CodeBertScore [Zhou et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib31) to eliminate duplicates.

Finally, we generate the LLM instruction for each operator based on prompt [D](https://arxiv.org/html/2502.14752v1#A4).
The instructions provide essential details, including the operator’s functionality, corresponding function names, and a comprehensive input/output demonstration.
All instructions are carefully reviewed and manually verified to ensure they correctly reflect the intended behavior of each operator.

### 3.2 Data Statistics

Table [1](https://arxiv.org/html/2502.14752v1#S3.T1) summarizes statistics of TritonBench-G.
In this benchmark, each operator is assigned a difficulty level, from $\bf d1$ (easiest) to $\bf d5$ (most challenging), by an LLM guided by prompt [D](https://arxiv.org/html/2502.14752v1#A4), with subsequent manual verification by two domain experts.
For each difficulty level, we report statistics including the average number of functions (func#), parameters (params#), lines (lines#), and tokens (tok#).
Notably, the upward trend observed in these statistics as the difficulty level increases suggests the expert-driven grading scheme is largely reasonable.

Compared to existing code generation tasks [Chen et al. (2021)](https://arxiv.org/html/2502.14752v1#bib.bib8); [Austin et al. (2021a)](https://arxiv.org/html/2502.14752v1#bib.bib2), the average instruction length in TritonBench-G is substantially longer, which is a deliberate design decision.
The extended instructions provide richer context, which can help the model understand nuanced requirements and generate high-quality operators.
Additionally, this approach better reflects real-world operator development practices where detailed requirements are indispensable.



<table>
<tr>
<td rowspan="2"><span>Difficulty</span></td>
<td><span>Instruction</span></td>
<td colspan="4"><span>Triton Operator</span></td></tr>
<tr>
<td><span>tok#</span></td>
<td><span>func#</span></td>
<td><span>params#</span></td>
<td><span>line#</span></td>
<td><span>tok#</span></td></tr>
<tr>
<td>$\mathbf{d1}$ ($1.6$%)</td>
<td>$296.67$</td>
<td>$2.00$</td>
<td>$1.33$</td>
<td>$26.00$</td>
<td>$369.0$</td></tr>
<tr>
<td>$\mathbf{d2}$ ($14.7$%)</td>
<td>$363.26$</td>
<td>$2.41$</td>
<td>$2.70$</td>
<td>$45.56$</td>
<td>$678.1$</td></tr>
<tr>
<td>$\mathbf{d3}$ ($35.3$%)</td>
<td>$353.80$</td>
<td>$3.80$</td>
<td>$3.34$</td>
<td>$102.42$</td>
<td>$1510.4$</td></tr>
<tr>
<td>$\mathbf{d4}$ ($45.7$%)</td>
<td>$394.48$</td>
<td>$3.89$</td>
<td>$6.04$</td>
<td>$153.77$</td>
<td>$2689.1$</td></tr>
<tr>
<td>$\mathbf{d5}$ ($2.7$%)</td>
<td>$469.60$</td>
<td>$6.60$</td>
<td>$6.00$</td>
<td>$249.80$</td>
<td>$4581.4$</td></tr>
</table>



Table 1: Statistics of TritonBench-G.

### 3.3 Operators Quality Rating

To systematically evaluate the quality of the Triton operators in TritonBench-G, we compute the GPU efficiency for each operator.
Detailed methodology for calculating GPU efficiency can be found in Appendix [B](https://arxiv.org/html/2502.14752v1#A2).
Our statistics indicate an average GPU efficiency of $\mathbf{43.0}\%$, which reflects the overall reliability of the operators in TritonBench-G.
The distribution of efficiency scores is shown in Figure [3](https://arxiv.org/html/2502.14752v1#S3.F3).
As shown in the figure, $19.6$% of operators developed by professional Triton programmers have GPU performance below 10%, which underscores the challenges in developing and optimizing Triton operators.

Figure 3: Distribution of GPU efficiency of the Triton operators in TritonBench-G.

### 3.4 Test Code Design

In contrast to traditional CPU-language benchmarks [Shypula et al. ()](https://arxiv.org/html/2502.14752v1#bib.bib27); [Liu et al. (2023)](https://arxiv.org/html/2502.14752v1#bib.bib17); [Huang et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib13); [Qiu et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib24) that predominantly rely on scalar test inputs,TritonBench-G is built around tensor-based test inputs.
We employ PyTorch to generate random tensors as replacements for conventional test cases.
Specifically, we leverage a prompt [D](https://arxiv.org/html/2502.14752v1#A4) to generate the corresponding test code for each operator.
In the case of the multi-branch operators, the generated test code is designed to invoke every branch within the operator.
Moreover, we rigorously debug all branches to guarantee test reliability.
On average, we generate $3.6$ test branches per operator.

### 3.5 Evaluation Metrics

In contrast to traditional code evaluations, which mainly emphasize accuracy [Chen et al. (2021)](https://arxiv.org/html/2502.14752v1#bib.bib8); [Austin et al. (2021a)](https://arxiv.org/html/2502.14752v1#bib.bib2), our TritonBench-G introduces dedicated performance evaluations. Specifically, the systematic evaluation of Triton operators covers five key metrics:

#### Similarity

assesses text-level resemblance using CodeBLEU [Ren et al. (2020)](https://arxiv.org/html/2502.14752v1#bib.bib25). In our experiments, we assign equal weights of $0.25$ to N-gram, weighted N-gram, syntax, and dataflow components to ensure a balanced evaluation.

#### Call & Execution Accuracy

assess whether the code can run without error and whether its input-output behavior is correct, respectively.

#### Speed Up

measures the relative execution time improvement for correctly executed operators. Specifically, if $t_{gen}$ and $t_{ref}$ represent the running times of the generated and reference operators, respectively, then
$\texttt{SpeedUp}(gen)=\frac{t_{ref}}{t_{gen}}.$

#### GPU Efficiency

evaluates how effectively the generated operator utilizes GPU resources, following the operator quality rating in § [3.3](https://arxiv.org/html/2502.14752v1#S3.SS3). For further details, please refer to Appendix [B](https://arxiv.org/html/2502.14752v1#A2).

## 4 TritonBench-T

The real-world Triton operators introduced in § [3](https://arxiv.org/html/2502.14752v1#S3) primarily focus on highly frequent operations.
As a complement, we propose TritonBench-T, which aligns the Triton wrapper with interfaces of the PyTorch library [Paszke et al. (2019)](https://arxiv.org/html/2502.14752v1#bib.bib22).
Together, TritonBench-G and TritonBench-T form a complementary evaluation framework.
The following sections elaborate on the data construction (§ [4.1](https://arxiv.org/html/2502.14752v1#S4.SS1)), data statistics (§ [4.2](https://arxiv.org/html/2502.14752v1#S4.SS2)), test code and metrics (§ [4.3](https://arxiv.org/html/2502.14752v1#S4.SS3)), and benchmark comparisons (§ [4.4](https://arxiv.org/html/2502.14752v1#S4.SS4)).

### 4.1 Data Construction

We construct TritonBench-T by selecting PyTorch operators based on their usage frequency in real-world coding and then fusing them ( hereafter referred to simply as “operators” ).
First, we select operators that require GPU interactions, ensuring alignment with Triton’s scope.
Next, we sample $40$ high-frequency operators and $40$ low-frequency operators from the remaining pool.
The frequency of each operator is determined by its usage probability in PyTorch-related code from The Stack V2 [Lozhkov et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib18) with those exceeding a predefined threshold $45$% as common operators.

Subsequently, we fuse these operators in various configurations: combinations of common operators, combinations of common and uncommon operators, and combinations of uncommon operators.
All combinations are valid, as the outputs of preceding operators serve as appropriate inputs for subsequent ones.
The final set includes $166$ operators, based on the latest (v2.6.0) version of the PyTorch library.
Each operator is paired with its corresponding standard PyTorch call and document, while fused operators combine descriptions from all involved operators.

### 4.2 Data Statistic

The statistics of TritonBench-T are presented in Table [2](https://arxiv.org/html/2502.14752v1#S4.T2).
Similar to TritonBench-G, the operators are categorized into five difficulty levels ($\bf d1$ to $\bf d5$) using an LLM guided by prompt [D](https://arxiv.org/html/2502.14752v1#A4).
These initial categorizations are then validated through manual review by two domain experts.

We report the following statistics: (1)torch-op# the average number of PyTorch operators, (2)params# the average number of parameters, (3)math#, the average token number of mathematical expressions, and (4)description#, the average token count of the descriptions.
These statistics generally increase with the operator difficulty, similar trend that aligns with the observations in TritonBench-G.



<table>
<tr>
<td rowspan="2"><span>Difficulty</span></td>
<td colspan="4"><span>Torch-Align Operator</span></td></tr>
<tr>
<td><span>torch-op#</span></td>
<td><span>params#</span></td>
<td><span>math#</span></td>
<td><span>description#</span></td></tr>
<tr>
<td>$\mathbf{d1}$ ($13.3$%)</td>
<td>$1.36$</td>
<td>$2.82$</td>
<td>$23.50$</td>
<td>$50.41$</td></tr>
<tr>
<td>$\mathbf{d2}$ ($22.3$%)</td>
<td>$1.97$</td>
<td>$3.78$</td>
<td>$40.73$</td>
<td>$61.19$</td></tr>
<tr>
<td>$\mathbf{d3}$ ($32.5$%)</td>
<td>$2.70$</td>
<td>$4.91$</td>
<td>$74.64$</td>
<td>$67.89$</td></tr>
<tr>
<td>$\mathbf{d4}$ ($29.5$%)</td>
<td>$2.16$</td>
<td>$5.24$</td>
<td>$47.31$</td>
<td>$71.02$</td></tr>
<tr>
<td>$\mathbf{d5}$ ($2.4$%)</td>
<td>$2.75$</td>
<td>$2.75$</td>
<td>$30.50$</td>
<td>$88.50$</td></tr>
</table>



Table 2: Statistics of TritonBench-T.

### 4.3 Test Code and Metrics

The design of the test code in TritonBench-T adheres to those of TritonBench-G, employing randomly generated tensors for operator evaluation.
For correctness and performance assessment, we utilize Call Accuracy, Execution Accuracy, and Speed Up, whose computation methods are consistent with those used in TritonBench-G.



<table>
<tr>
<td><span>Model</span></td>
<td><span>Size</span></td>
<td><span>Similarity</span></td>
<td><span></span><span> <span>
<span>
<span>
<span>Call</span></span>
<span>
<span>Accuracy</span></span>
</span></span><span></span></span></td>
<td><span></span><span> <span>
<span>
<span>
<span>Execution</span></span>
<span>
<span>Accuracy</span></span>
</span></span><span></span></span></td>
<td><span>Speed Up</span></td>
<td><span></span><span> <span>
<span>
<span>
<span>GPU</span></span>
<span>
<span>Efficiency</span></span>
</span></span><span></span></span></td></tr>
<tr>
<td colspan="7"><span>Domain-Specific Models</span></td></tr>
<tr>
<td>Qwen2.5-Coder</td>
<td>$7$B</td>
<td>$9.19$ / $14.54$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td></tr>
<tr>
<td>DeepSeek-Coder</td>
<td>$6.7$B</td>
<td>$9.38$ / $14.52$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td></tr>
<tr>
<td>Qwen2.5-Coder-sft</td>
<td>$7$B</td>
<td>$\mathbf{29.98}$ / $25.96$</td>
<td>$4.89$ / $10.87$</td>
<td>$4.89$ / $10.87$</td>
<td>$\mathbf{1.56}$ / $\mathbf{1.22}$</td>
<td>$\mathbf{51.71}$ / $\mathbf{46.70}$</td></tr>
<tr>
<td>DeepSeek-Coder-sft</td>
<td>$6.7$B</td>
<td>$25.52$ / $\mathbf{30.34}$</td>
<td>$\mathbf{9.78}$ / $\mathbf{11.96}$</td>
<td>$\mathbf{9.87}$ / $\mathbf{11.96}$</td>
<td>$1.03$ / $1.11$</td>
<td>$47.68$ / $42.26$</td></tr>
<tr>
<td colspan="7"><span>General-Purpose Models</span></td></tr>
<tr>
<td>GPT-4o</td>
<td>-</td>
<td>$9.87$ / $20.67$</td>
<td>$10.87$ / $17.93$</td>
<td>$10.33$ / $16.84$</td>
<td>$0.97$ / $1.19$</td>
<td>$48.80$ / $\mathbf{53.33}$</td></tr>
<tr>
<td>Claude-3.5-Sonnet</td>
<td>-</td>
<td>$12.46$ / $22.48$</td>
<td>$10.33$ / $20.11$</td>
<td>$9.79$ / $19.57$</td>
<td>$0.90$ / $\mathbf{1.54}$</td>
<td>$\mathbf{59.31}$ / $49.32$</td></tr>
<tr>
<td>Qwen2.5-72B</td>
<td>$72$B</td>
<td>$14.86$ / $26.25$</td>
<td>$11.41$ / $16.85$</td>
<td>$10.87$ / $16.31$</td>
<td>$0.96$ / $1.19$</td>
<td>$23.28$ / $49.40$</td></tr>
<tr>
<td>DeepSeek-R1</td>
<td>$685$B</td>
<td>$\mathbf{19.96}$ / $22.64$</td>
<td>$13.59$ / $22.83$</td>
<td>$13.05$ / $22.83$</td>
<td>$\mathbf{1.11}$ / $1.22$</td>
<td>$44.83$ / $46.70$</td></tr>
<tr>
<td>GPT-o1</td>
<td>-</td>
<td>$16.58$ / $\mathbf{29.70}$</td>
<td>$\mathbf{15.22}$ / $\mathbf{23.91}$</td>
<td>$\mathbf{14.23}$ / $\mathbf{23.91}$</td>
<td>$0.92$ / $1.14$</td>
<td>$54.25$ / $46.37$</td></tr>
</table>



Table 3: Main results of TritonBench-G across baseline models, where the left side of “/” represents the zero-shot results and the right side represents the one-shot results.

### 4.4 Benchmark Comparison

This section provides comparisons between TritonBench-G and TritonBench-T, which differ in key aspects and together provide a well-rounded evaluation.

#### Source & Distribution:

TritonBench-G is collected from GitHub and reflects real-world programming demands with a concentration of frequently used operators, e.g., Attention at $20.0\%$, MatMul at $10.9\%$, LayerNorm at $6.5\%$, SoftMax at $3.8\%$. In contrast, TritonBench-T, sourced from PyTorch, presents a more diverse operator set including both common and uncommon operators.

#### Instruction Generation:

TritonBench-G combines LLM generation with expert verification while TritonBench-T directly extracts instructions from PyTorch documentation. This difference underlines their complementary roles in probing different facets of the Triton generation.

#### Evaluation Metrics:

Both benchmark channels assess correctness and performance. Additionally, TritonBench-G incorporates a similarity-based assessment that offers direct comparisons with established implementations.
In summary, the different designs of TritonBench-G and TritonBench-T enable a comprehensive and nuanced evaluation of Triton operator generation.

## 5 Experiments

We conduct an extensive set of experiments on TritonBench to rigorously evaluate the performance and capabilities of current LLMs.



<table>
<tr>
<td><span>Model</span></td>
<td><span>Size</span></td>
<td><span>Call Accuracy</span></td>
<td><span>Execution Accuracy</span></td>
<td><span>Speed Up</span></td></tr>
<tr>
<td colspan="5"><span>Domain-Specific Models</span></td></tr>
<tr>
<td>Qwen2.5-Coder</td>
<td>$7$B</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td>
<td>$0.00$ / $0.00$</td></tr>
<tr>
<td>DeepSeek-Coder</td>
<td>$6.7$B</td>
<td>$0.00$ / $1.81$</td>
<td>$0.00$ / $1.81$</td>
<td>$0.00$ / $\mathbf{0.94}$</td></tr>
<tr>
<td>Qwen2.5-Coder-sft</td>
<td>$7$B</td>
<td>$17.47$ / $16.27$</td>
<td>$17.47$ / $15.67$</td>
<td>$\mathbf{0.98}$ / $0.92$</td></tr>
<tr>
<td>DeepSeek-Coder-sft</td>
<td>$6.7$B</td>
<td>$\mathbf{19.28}$ / $\mathbf{18.67}$</td>
<td>$\mathbf{19.28}$ / $\mathbf{16.26}$</td>
<td>$0.91$ / $0.85$</td></tr>
<tr>
<td colspan="5"><span>General-Purpose Models</span></td></tr>
<tr>
<td>GPT-4o</td>
<td>-</td>
<td>$36.75$ / $32.53$</td>
<td>$36.75$ / $32.53$</td>
<td>$0.98$ / $0.94$</td></tr>
<tr>
<td>Claude-3.5-Sonnet</td>
<td>-</td>
<td>$29.52$ / $37.95$</td>
<td>$29.52$ / $33.70$</td>
<td>$0.93$ / $0.89$</td></tr>
<tr>
<td>Qwen2.5-72B</td>
<td>72B</td>
<td>$30.12$ / $22.89$</td>
<td>$30.12$ / $16.30$</td>
<td>$1.07$ / $0.92$</td></tr>
<tr>
<td>DeepSeek-R1</td>
<td>$685$B</td>
<td>$\mathbf{53.01}$ / $\mathbf{45.78}$</td>
<td>$\mathbf{53.01}$ / $\mathbf{45.78}$</td>
<td>$1.03$ / $\mathbf{1.91}$</td></tr>
<tr>
<td>GPT-o1</td>
<td>-</td>
<td>$32.53$ / $43.37$</td>
<td>$32.53$ / $43.37$</td>
<td>$\mathbf{1.21}$ / $1.10$</td></tr>
</table>



Table 4: Main results of TritonBench-T across baseline models, where the left side of “/” represents the zero-shot results and the right side represents the one-shot results.

### 5.1 Baselines and Setup

TritonBench generally requires strong capabilities in code generation. Therefore, we select state-of-the-art LLMs that excel in programming tasks as baselines, including both specialized open-source models and general-purpose models.
For specialized open-source models, we choose Qwen2.5-Coder-7B-Instruct [Hui et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib14) and deepseek-coder-6.7b-instruct [Guo et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib11).
For general-purpose models, we include Claude-3.5-Sonnet-062011
1
<https://www.anthropic.com/news/claude-3-5-sonnet>, GPT-4o-0806 22
2
<https://openai.com/index/hello-gpt-4o>, qwen2.5-72B-Instruct [Yang et al. (2024)](https://arxiv.org/html/2502.14752v1#bib.bib30), as well as the thought-driven models DeepSeek-R1 [Guo et al. (2025)](https://arxiv.org/html/2502.14752v1#bib.bib10) and GPT-o1-2024-12-17 33
3
<https://openai.com/o1/>.

In our experiments, all general-purpose models are deployed for direct inference.
In contrast, domain-specific models undergo an additional supervised fine-tuning phase.
Details of the training corpus can be found in § [A](https://arxiv.org/html/2502.14752v1#A1).
For evaluation, we consider both zero-shot and one-shot scenarios.
In the one-shot setting, a BM25-based retrieval method [Robertson et al. (2009)](https://arxiv.org/html/2502.14752v1#bib.bib26) is utilized to select the most relevant prompt from the training corpus.

### 5.2 Main results of TritonBench-G

Table [3](https://arxiv.org/html/2502.14752v1#S4.T3) illustrates the performances of baselines on TritonBench-G.
It is evident that domain-specific models generally underperform compared to general-purpose models.
However, fine-tuning $7$B domain-specific models with domain data significantly boosts accuracy.
Qwen’s accuracy rises from $0$ to $4.89$%, and DeepSeek’s from $0$ to $9.78$% in zero-shot settings, with even more pronounced enhancements in one-shot settings due to the retrieval data from the same source as TritonBench-G.
The observed increase in Speed Up can be attributed to the relative simplicity of the correctly generated operators, which makes it easier for LLMs to produce efficient code.
The high GPU efficiency shares the similar reasons.

General-purpose models, particularly DeepSeek-R1 and GPT-o1, excel across all metrics.
Under one-shot conditions, DeepSeek-R1 achieves $22.83$% in Call and Execution Accuracy, while GPT-o1 reaches $23.91$%.
The roughly $10$% improvement from zero-shot to one-shot highlights the critical role of high-quality examples for Triton generation.
Furthermore, the close alignment between Call Accuracy and Execution Accuracy indicates that only a few operators fail to produce correct results despite successfully invoked.

DeepSeek-R1 also leads in GPU execution times, with an improvement of $1.11\times$ in zero-shot and $1.22\times$ in one-shot settings.
While GPU efficiency is strong across most models, Qwen2.5-72B exhibits lower efficiency in zero-shot settings, likely due to a higher proportion of less efficient operators.
Finally, Similarity provides corroborative insights, as its variations mirror trends observed in other metrics.

### 5.3 Main Results of TritonBench-T

From Table [4](https://arxiv.org/html/2502.14752v1#S5.T4), we can observe that domain-specific models generally underperform general-purpose models.
Nonetheless, fine-tuning with an $8k$ corpus considerably improves their performance.
For instance, Qwen’s zero-shot Execution Accuracy rises from $0$ to $17.47$%.
In contrast, its one-shot improvement ($15.67$%) is slightly lower, likely due to the fact that the retrieved prompts and TritonBench-T operators come from different sources (Github vs. Pytorch).

Among general-purpose models, DeepSeek-R1 demonstrates the strongest overall performance, achieving $53.01$% Call and Execution Accuracy in the zero-shot setting.
Although its accuracy drops by $7.23$% in the one-shot setting, it still slightly surpasses GPT-o1.
As for Speed Up, DeepSeek-R1 achieves the best performance of $1.91\times$ improvements.
Most performance improvements in successfully executed operators stem from operator fusion. Triton’s fused operators reduce redundant memory reads and writes compared to PyTorch, enhancing memory bandwidth utilization and boosting performance.

Overall, most models achieve better performance on TritonBench-T than to TritonBench-G, likely because TritonBench-T features a more balanced distribution of operator difficulty, whereas TritonBench-G is predominantly composed of higher-difficulty operators, namely, $\mathbf{d3}$ and $\mathbf{d4}$.

## 6 Analysis

In this section, we examine the distribution of correct and incorrect operators across difficulty levels ($\mathbf{d1}$–$\mathbf{d5}$) for the top-performing models, DeepSeek-R1 and GPT-o1, as shown in Figure [4](https://arxiv.org/html/2502.14752v1#S6.F4) and Figure [5](https://arxiv.org/html/2502.14752v1#S6.F5). Additionally, we analyze the error patterns of incorrect operators and summarize the main challenges for each benchmark as detailed in Table [3](https://arxiv.org/html/2502.14752v1#S4.T3) and Table [4](https://arxiv.org/html/2502.14752v1#S5.T4).
The zero-shot and one-shot settings are annotated as 0 and 1 respectively.

### 6.1 Challenges for TritonBench-G

Figure [4](https://arxiv.org/html/2502.14752v1#S6.F4) clearly shows that most operators are generated incorrectly.
Both DeepSeek-R1 and GPT-o1 exhibit similar trends, with DeepSeek-R1 outperforming GPT-o1.
Notably, when moving from the zero-shot to the one-shot setting, both models achieve significant improvements on $\mathbf{d4}$.
These improvements may stem from the prevalence of Attention and Softmax operators in $\mathbf{d4}$, enabling models to leverage similar examples.
In contrast, the simpler operators in $\mathbf{d2}$ and $\mathbf{d3}$ show only limited gains in the one-shot setting, likely due to the smaller, more idiosyncratic nature of these datasets that leads to lower retrieval similarity.

For the incorrectly written operators, we classify the $16$ error types into $4$ major categories, detailed in Appendix [C](https://arxiv.org/html/2502.14752v1#A3)
which is presented in Table [5](https://arxiv.org/html/2502.14752v1#S6.T5).
Note that only compiler-reported errors were considered.
The results show that, compared to the zero-shot setting, both DeepSeek-R1 and GPT-o1 in the one-shot setting demonstrate a significant increase in Syntax and Name&Ref errors but a reduction in Attr&Type and Run&Logc errors.
This trend suggests that the training corpus may provide helpful guidance on logical structure and Triton specifications, thus enhancing overall accuracy.
Furthermore, error sensitivity differs between models: DeepSeek-R1 is less susceptible to syntax errors, whereas GPT-o1 handles logical errors better.

### 6.2 Challenges for TritonBench-T

The execution results of TritonBench-T (Figure [5](https://arxiv.org/html/2502.14752v1#S6.F5)) show the percentages of correctly generated operators.
we can observe that DeepSeek-R1 generated more correct than incorrect operators, which proves the point that the difficulty distributions in TritonBench-T are smoother than TritonBench-G.

However, while DeepSeek-R1’s performance declines for difficulty $\mathbf{d2}$-$\mathbf{d4}$ in the one-shot setting, GPT-o1 shows improved accuracy on these subsets.
This finding indicates that GPT-o1 might be more adept at logical reasoning for Triton generation tasks, allowing it to efficiently use the provided sample. The differing trends also imply that sample operators affect models in diverse ways.

(a) DeepSeek-R10 results.

(b) DeepSeek-R11 results.

(c) GPT-o10 results.

(d) GPT-o11 results.

Figure 4: Execution results distribution across difficulty levels in TritonBench-G.


|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Syntax | Attr&Type | Name&Ref | Run&Logc |
| DeepSeek-R10 | $1.64$ | $42.62$ | $16.39$ | $39.34$ |
| DeepSeek-R11 | $9.27$ | $33.11$ | $35.76$ | $21.85$ |
| GPT-o10 | $10.3$ | $38.18$ | $28.48$ | $23.03$ |
| GPT-o11 | $20.83$ | $24.31$ | $43.06$ | $11.81$ |

Table 5: Error statistics of execution failures in TritonBench-G.

For execution error statistics in TritonBench-T (Table [6](https://arxiv.org/html/2502.14752v1#S6.T6)), DeepSeek-R1 notably avoids Syntax errors entirely, while GPT-o1 maintains a high rate of such errors.
Under the one-shot setting, DeepSeek-R1 shows a rise in Attr&Type and Name&Ref errors alongside a decline in Run&Logc Errors.
Conversely, GPT-o1 experiences a significant increase in Name&Ref errors with a notable drop in Run&Logc errors.
Comparing TritonBench-G and TritonBench-T, the one-shot setting consistently reduces Run&Logc errors.
These variations in error patterns likely stem from the mixed influence of useful and irrelevant information in the provided samples.

(a) DeepSeek-R10 results.

(b) DeepSeek-R11 results.

(c) GPT-o10 results.

(d) GPT-o11 results.

Figure 5: Execution results distribution across difficulty levels in TritonBench-T.


|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Syntax | Attr&Type | Name&Ref | Run&Logc |
| DeepSeek-R10 | $0.00$ | $31.96$ | $14.43$ | $53.61$ |
| DeepSeek-R11 | $0.00$ | $36.79$ | $20.75$ | $42.45$ |
| GPT-o10 | $24.06$ | $26.32$ | $7.52$ | $42.11$ |
| GPT-o11 | $25.25$ | $25.25$ | $22.22$ | $27.27$ |

Table 6: Error statistics of execution failures in TritonBench-T.

## 7 Conclusion

In this work, we present TritonBench, a dual-channel benchmark specifically designed for evaluating LLMs’ generation for Triton operators. TritonBench-G integrates real-world Triton operator samples from open repositories, while TritonBench-T introduces complementary tasks that align with PyTorch interfaces. Our evaluation framework addresses both functional accuracy and the performance on NVIDIA GPUs.
We also conduct extensive experiments and detailed analysis on our benchmark, and find that current LLMs struggle to generate high-quality Triton operators, underscoring the necessity for further advancement in generating accurate as well as performance-aware Triton code.
We anticipate TritonBench will serve as an essential framework for advancing automated operator generation for Triton.

## Limitations

The primary limitation of this study is that the evaluations of TritonBench were conducted exclusively on the NVIDIA A100 GPU, as it is widely adopted in industry and research applications. In future work, we plan to expand the evaluation to include a broader range of hardware architectures for more comprehensive performance insights.

## References

- Abadi et al. (2016)

  Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. 2016.
  $\{$TensorFlow$\}$: a system for $\{$Large-Scale$\}$ machine learning.
  In *12th USENIX symposium on operating systems design and implementation (OSDI 16)*, pages 265–283.
- Austin et al. (2021a)

  Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021a.
  [Program synthesis with large language models](https://arxiv.org/abs/2108.07732).
  *ArXiv preprint*, abs/2108.07732.
- Austin et al. (2021b)

  Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021b.
  [Program synthesis with large language models](https://arxiv.org/abs/2108.07732).
  *ArXiv preprint*, abs/2108.07732.
- Belz et al. (2021)

  Anya Belz, Shubham Agarwal, Yvette Graham, Ehud Reiter, and Anastasia Shimorina, editors. 2021.
  [*Proceedings of the Workshop on Human Evaluation of NLP Systems (HumEval)*](https://aclanthology.org/2021.humeval-1.0). Association for Computational Linguistics, Online.
- Buscemi (2023)

  Alessio Buscemi. 2023.
  [A comparative study of code generation using chatgpt 3.5 across 10 programming languages](https://arxiv.org/abs/2308.04477).
  *ArXiv preprint*, abs/2308.04477.
- Cassano et al. (2024)

  Federico Cassano, John Gouwar, Francesca Lucchetti, Claire Schlesinger, Anders Freeman, Carolyn Jane Anderson, Molly Q Feldman, Michael Greenberg, Abhinav Jangda, and Arjun Guha. 2024.
  Knowledge transfer from high-resource to low-resource programming languages for code llms.
  *Proceedings of the ACM on Programming Languages*, 8(OOPSLA2):677–708.
- Chaturvedi et al. (2024)

  Aman Chaturvedi, Daniel Nichols, Siddharth Singh, and Abhinav Bhatele. 2024.
  [Hpc-coder-v2: Studying code llms across low-resource parallel languages](https://arxiv.org/abs/2412.15178).
  *ArXiv preprint*, abs/2412.15178.
- Chen et al. (2021)

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021.
  [Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374).
  *ArXiv preprint*, abs/2107.03374.
- Daniel Han and team (2023)

  Michael Han Daniel Han and Unsloth team. 2023.
  [Unsloth](http://github.com/unslothai/unsloth).
- Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025.
  [Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning](https://arxiv.org/abs/2501.12948).
  *ArXiv preprint*, abs/2501.12948.
- Guo et al. (2024)

  Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. 2024.
  Deepseek-coder: When the large language model meets programming-the rise of code intelligence.
  *CoRR*.
- Hsu et al. (2024)

  Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. 2024.
  [Liger kernel: Efficient triton kernels for llm training](https://arxiv.org/abs/2410.10989).
  *ArXiv preprint*, abs/2410.10989.
- Huang et al. (2024)

  Dong Huang, Weiyi Shang, Yuhao Qing, Heming Cui, and Jie M Zhang. 2024.
  [Effibench: Benchmarking the efficiency of automatically generated code](https://arxiv.org/abs/2402.02037).
  *ArXiv preprint*, abs/2402.02037.
- Hui et al. (2024)

  Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024.
  [Qwen2. 5-coder technical report](https://arxiv.org/abs/2409.12186).
  *ArXiv preprint*, abs/2409.12186.
- Jimenez et al. (2024)

  Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024.
  [SWE-bench: Can language models resolve real-world github issues?](https://openreview.net/forum?id=VTF8yNQM66)
  In *The Twelfth International Conference on Learning Representations*.
- Kwon et al. (2023)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023.
  Efficient memory management for large language model serving with pagedattention.
  In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.
- Liu et al. (2023)

  Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023.
  [Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation](http://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html).
  In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.
- Lozhkov et al. (2024)

  Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024.
  [Starcoder 2 and the stack v2: The next generation](https://arxiv.org/abs/2402.19173).
  *ArXiv preprint*, abs/2402.19173.
- Mitkov et al. (2021)

  Ruslan Mitkov, Vilelmini Sosoni, Julie Christine Giguère, Elena Murgolo, and Elizabeth Deysel, editors. 2021.
  [*Proceedings of the Translation and Interpreting Technology Online Conference*](https://aclanthology.org/2021.triton-1.0). INCOMA Ltd., Held Online.
- ModelTC (2025)

  ModelTC. 2025.
  Lightllm: A python-based llm inference and serving framework.
  <https://github.com/ModelTC/lightllm>.
- Nichols et al. (2024)

  Daniel Nichols, Joshua H Davis, Zhaojun Xie, Arjun Rajaram, and Abhinav Bhatele. 2024.
  Can large language models write parallel code?
  In *Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing*, pages 281–294.
- Paszke et al. (2019)

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019.
  [Pytorch: An imperative style, high-performance deep learning library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).
  In *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 8024–8035.
- Pujar et al. (2023)

  Saurabh Pujar, Luca Buratti, Xiaojie Guo, Nicolas Dupuis, Burn Lewis, Sahil Suneja, Atin Sood, Ganesh Nalawade, Matt Jones, Alessandro Morari, et al. 2023.
  Automated code generation for information technology tasks in yaml through large language models.
  In *2023 60th ACM/IEEE Design Automation Conference (DAC)*, pages 1–4. IEEE.
- Qiu et al. (2024)

  Ruizhong Qiu, Weiliang Will Zeng, Hanghang Tong, James Ezick, and Christopher Lott. 2024.
  [How efficient is llm-generated code? a rigorous & high-standard benchmark](https://arxiv.org/abs/2406.06647).
  *ArXiv preprint*, abs/2406.06647.
- Ren et al. (2020)

  Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020.
  [Codebleu: a method for automatic evaluation of code synthesis](https://arxiv.org/abs/2009.10297).
  *ArXiv preprint*, abs/2009.10297.
- Robertson et al. (2009)

  Stephen Robertson, Hugo Zaragoza, et al. 2009.
  The probabilistic relevance framework: Bm25 and beyond.
  *Foundations and Trends® in Information Retrieval*, 3(4):333–389.
- (27)

  Alexander G Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob R Gardner, Yiming Yang, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, et al.
  Learning performance-improving code edits.
  In *The Twelfth International Conference on Learning Representations*.
- Tillet et al. (2019)

  Philippe Tillet, Hsiang-Tsung Kung, and David Cox. 2019.
  Triton: an intermediate language and compiler for tiled neural network computations.
  In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, pages 10–19.
- Wąsowski and Berger (2023)

  Andrzej Wąsowski and Thorsten Berger. 2023.
  *Domain-Specific Languages*.
  Springer.
- Yang et al. (2024)

  An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024.
  [Qwen2. 5 technical report](https://arxiv.org/abs/2412.15115).
  *ArXiv preprint*, abs/2412.15115.
- Zhou et al. (2023)

  Shuyan Zhou, Uri Alon, Sumit Agarwal, and Graham Neubig. 2023.
  [CodeBERTScore: Evaluating code generation with pretrained models of code](https://doi.org/10.18653/v1/2023.emnlp-main.859).
  In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 13921–13937, Singapore. Association for Computational Linguistics.
- Zhu et al. (2024)

  Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. 2024.
  [Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence](https://arxiv.org/abs/2406.11931).
  *ArXiv preprint*, abs/2406.11931.

## Appendix A Training Corpus

The training corpus for supervised fine-tuning comprises two distinct components: real-world data sourced from GitHub and synthetically generated data produced through compiler operations.

The real-world data component incorporates Triton code extracted from GitHub repositories, which undergoes basic cleaning procedures as outlined in prompt [D](https://arxiv.org/html/2502.14752v1#A4), undergoes a debugging process that is less rigorous than the methodology applied to TritonBench-G.
To prevent potential data leakage and ensure benchmark integrity, we systematically eliminate samples exhibiting high similarity to TritonBench-G entries using the CodeBertScore similarity metric ([Zhou et al., 2023](https://arxiv.org/html/2502.14752v1#bib.bib31)).

The synthetic data component is generated using Ninetoothed44
4
<https://github.com/InfiniTensor/ninetoothed>, a domain-specific language built upon Triton that offers enhanced abstraction capabilities. This framework facilitates the automated synthesis of valid Triton code through the processing of well-formed expressions.
Each part of data containing $4K$ samples. This combined corpus serves as the foundational training dataset for experimental models in one-shot learning settings. For all experiments, the fine-tuning process is carried out over $3$ epochs with a learning rate of $5e-5$.

## Appendix B Operator Performance Evaluation

![Refer to caption](assets/05-tritonbench/figure-003.png)


Figure 6: The workflow of operator performance evaluation

For operator performance evaluation, we refer primarily to the official examples provided by Triton55
5
<https://triton-lang.org/main/getting-started/tutorials/>.
We provide evaluation scripts for each operator in TritonBench-G.
Figure [6](https://arxiv.org/html/2502.14752v1#A2.F6) illustrates the workflow of our operator performance evaluation.

First, we define a set of tensors with increasing dimensions based on the characteristics of the operator.
Next, each tensor is sequentially fed into the operator for execution.
During each execution, we use the expert annotations for each operator to determine the total memory bandwidth (Bytes) and the total number of floating-point operations (Flops) based on the input tensors.
More importantly, we use the triton.testing.do_bench method from the official Triton library66
6
[https://triton-lang.org/main/pythonapi/generated/triton.testing.do_bench.html](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) to measure the operator’s execution time on the GPU.
Specifically, we gradually increase the warm-up time and repetition time until the measured execution time stabilized, which means that most operators are run hundreds of thousands of times to ensure that the running time is measured accurately.
After obtaining the execution time, we calculate the operator’s performance metrics by dividing the total memory bandwidth and the total floating-point operations by the execution time to obtain throughput in GB/s and Tflops, respectively.
We then calculate the GPU efficiency by calculating the ratio of the measured performance metrics (GB/s and Tflops) to the theoretical maximum performance of the NVIDIA A100 Tensor Core GPU.
Repetition of the above process for tensors of increasing sizes obtains the performance metrics for each execution, which collectively form the operator performance report.
We adopt the peak GPU efficiency from the performance report as the final measure of the operator’s quality.

By following the evaluation workflow described above, we generate a detailed performance report for each operator in TritonBench-G. Figure [7](https://arxiv.org/html/2502.14752v1#A2.F7) illustrates the performance curves of several common operators. As the input dimensions increase, as can be seen from the figure, the GB/s or Tflops of the operators show an upward trend, eventually stabilizing.
This suggests that the performance of the operator reaches a bottleneck beyond a certain scale, and further increases in input size result in diminishing returns in performance, aligning with the expected trend of operator performance.

Figure 7: Performance Curves of Common Operators

## Appendix C Error Categories

We provide the error type statistics of failure operators in TritonBench. A total of 16 error types are identified in the integrated Call and Execution error results. For convenience in presentation, we categorize them into four main groups: Syntax Errors: including SyntaxError and IndentationError; Attrb&Type Errors: including AttributeError, TypeError, and NotImplementedError; Name&Ref Errors: including NameError, KeyError, IndexError, ModuleNotFoundError, and ImportError; Run&Logc Errors: including ValueError, ZeroDivisionError, RuntimeError, RecursionError, AssertionError, CompilationError, and ResultsError. ResultsError refers to the inconsistency between the execution results of the reference operator and the generated operator.

## Appendix D Prompts

Here are the four prompts we use in our work: Filtering Prompt, Instruction Prompt, Difficulty Prompt, and Test Code Prompt. Specifically, the first is used to extract Triton-related code from crawled code files; the second instructs the large model to generate corresponding instructions based on Triton code; the third prompts the large model to score the difficulty of Triton operators according to the standards we proposed; and the last asks the large model to generate test code.

Filtering Prompt

{code}
  
Please help me select all triton kernel functions decorated with @triton.jit and all code that calls these kernels, while only keeping the necessary imports (e.g., triton, torch) and the calling functions.
  
  
  
Note 1: Retain necessary comments related to the Triton code. Code can be optimized, but do not remove all kernel code and its corresponding calls just for brevity.
  
  
  
Note 2: If the triton kernel is decorated with a custom or third-party decorator other than triton, discard that kernel.
  
  
  
Note 3: If @triton.jit appears as a string in the code or is nested within a function body, then discard it.
  
  
  
Note 4: If there are multiple triton kernel functions decorated with @triton.jit and their calling wrapper functions, retain all of them, not just a subset.
1) Extract all triton operators (kernel functions decorated with @triton.jit and their calling functions) and output them in python code format. If no triton kernel function is found, discard it.
2) Provide a concise English description of each extracted operator (including both kernel and calling code) in the form of a python dictionary: "description": "Use triton language to..."

Instruction Prompt

{code}
  
Based on the above Triton operator code, generate a detailed description so that the large model can accurately reproduce the corresponding kernel and wrapper function.
  
Be clear about the logic and main functionality of the operator, specify the function name, inputs, and outputs, and describe any public variables clearly.
  
Try to describe the function’s code implementation. Ensure that the large model can reproduce the corresponding function and parameter code based on these instructions.
  
Note that the output should maintain correct python syntax.

Test Code Prompt

{code}
  
Write a test code in Python for the above code. Ensure that all branch tests are in a single function starting with ‘‘test_’’, with no parameters.
  
  
Note 1: Particular attention should be paid to the fact that tensor parameters are of GPU type.
  
  
  
Note 2: Try to limit the number of branches to no more than 4.
  
  
  
Note 3: In branch tests, avoid modifying parameters that are later in the argument list with default values (especially if they have out parameters, do not assign them).
  
  
  
Note 4: Store the results of all branch calculations in a dictionary, where the dictionary key is "test_case_n", with n representing the test case number.
  
  
  
Note 5: Ensure that the import paths match exactly as described in the operator documentation to maintain accuracy.
  
  
  
Note 6: The code should run directly, without if __name__ == "__main__".

Difficulty Prompt

{code}
  
Please evaluate the complexity of the code in the following two aspects based on the requirements of the Triton operator and score it from simple to complex on a scale from 1 to 5:
  
1) Memory layout complexity: Analyze the memory access pattern, including memory tiling, array transposition, address alignment, cache utilization, and the number of global memory accesses.
  
2) Computation scheduling complexity: Examine instruction-level parallelism, computation-memory pipeline, thread block design, inter-thread communication, thread branch divergence, and hardware resource utilization.
  
The final score is the ceiling of the average score from both aspects, and only one complexity score is output in [ ].

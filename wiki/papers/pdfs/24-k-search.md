---
title: "K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model"
source_url: "https://arxiv.org/abs/2602.19128"
full_text_url: "https://arxiv.org/html/2602.19128v2"
version: "2602.19128v2"
authors: ["Cao, Shiyi", "Mao, Ziming", "Gonzalez, Joseph E.", "Stoica, Ion"]
retrieved_date: "2026-11-21"
conversion: "HTML to Markdown with TeX math"
pdf: "24-k-search.pdf"
original_html: "sources/24-k-search.html"
---

# K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model

]UC Berkeley
[Code]<https://github.com/caoshiyi/K-Search>

Shiyi Cao

Ziming Mao

Joseph E. Gonzalez

Ion Stoica
Affiliation: [

###### Abstract

Optimizing GPU kernels is critical for efficient modern machine learning systems yet remains challenging due to the complex interplay of design factors and rapid hardware evolution. Existing automated approaches typically treat Large Language Models (LLMs) merely as stochastic code generators within heuristic-guided evolutionary loops. These methods often struggle with complex kernels requiring coordinated, multi-step structural transformations, as they lack explicit planning capabilities and frequently discard promising strategies due to inefficient or incorrect intermediate implementations. To address this, we propose *Search via Co-Evolving World Model* and build K-Search based on this method. By replacing static search heuristics with a co-evolving world model, our framework leverages LLMs’ prior domain knowledge to guide the search, actively exploring the optimization space. This approach explicitly decouples high-level algorithmic planning from low-level program instantiation, enabling the system to navigate non-monotonic optimization paths while remaining resilient to temporary implementation defects. We evaluate K-Search on diverse, complex kernels from FlashInfer, including GQA, MLA, and MoE kernels. Our results show that K-Search significantly outperforms state-of-the-art evolutionary search methods, achieving an average 2.10$\times$ improvement and up to a 14.3$\times$ gain on complex MoE kernels. On the GPUMode [TriMul](https://www.gpumode.com/leaderboard/496?tab=rankings) task, K-Search achieves state-of-the-art performance on H100, reaching $1030\text{\,}\mathrm{\SIUnitSymbolMicro s}$ and surpassing both prior evolution and human-designed solutions.

## 1 Introduction

High-performance GPU kernels are fundamental to modern machine learning systems for training and serving large models, as reflected in widely used optimized libraries such as FlashInfer [Ye et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib38) for high-throughput LLM serving [Kwon et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib16); [Zheng et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib43), FlashAttention [Dao et al. (2022)](https://arxiv.org/html/2602.19128v2#bib.bib6); [Dao (2023)](https://arxiv.org/html/2602.19128v2#bib.bib5); [Shah et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib32) for memory-efficient attention, and FlashLinearAttention [fla org (2024)](https://arxiv.org/html/2602.19128v2#bib.bib8) for emerging model architecture variants [Gu and Dao (2024)](https://arxiv.org/html/2602.19128v2#bib.bib10); [Peng et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib30).

Unfortunately, GPU kernels are notoriously challenging to manually optimize and tune. First, achieving near-peak performance on modern GPUs requires navigating a large design space of tiling, memory layout, synchronization, and architecture-specific primitives [NVIDIA (2025)](https://arxiv.org/html/2602.19128v2#bib.bib28). Second, fast hardware evolution adds additional optimization complexity; new architectures (e.g., transitioning from NVIDIA Hopper to Blackwell) introduce new instructions and architectural characteristics that fundamentally alter performance trade-offs, rendering previously optimized kernels sub-optimal. Third, testing a kernel optimization might require significant implementation effort with many manual trials and errors [Ouyang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib29). Compiling and profiling generated kernels is also computationally expensive, normally mandating strictly limited testing budgets [Zheng et al. (2020)](https://arxiv.org/html/2602.19128v2#bib.bib42). Therefore, automated kernel generation methods that can adapt efficiently to new workloads and hardware with low search budgets become increasingly important.

Existing LLM evolutionary approaches, such as OpenEvolve [Superintelligence (2025)](https://arxiv.org/html/2602.19128v2#bib.bib33), couple language models with genetic algorithms or quality-diversity search. These methods typically treat LLMs purely as stochastic code generators, relying on heuristic mechanisms such as MAP-Elites [Mouret and Clune (2015)](https://arxiv.org/html/2602.19128v2#bib.bib25) to select and mutate candidates directly in the program space. However, high-performance kernels often require coordinated, multi-step structural transformations, such as refactoring memory layout *before* applying vectorization, where intermediate steps may not yield immediate performance gains. By treating the LLM merely as a code generator without an explicit planning mechanism, existing evolutionary methods typically cannot plan multi-step optimization sequences in which intermediate edits fail to improve the objective, and they often prematurely discard theoretically sound strategies due to temporary compilation errors, limiting their ability to discover deep structural optimizations necessary for achieving state-of-the-art performance.

To address this, we propose Search via Co-Evolving World Model and build the K-Search framework, inspired by recent works [Fang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib7); [Hao et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib15) on using large language models as world models to guide planning and decision making. We formulate kernel generation as a planning problem over a structured search tree, governed by a World Model instantiated from an LLM to leverage its prior domain knowledge for efficient search. In this setup, the World Model is responsible for maintaining the search frontier and estimating the priority scores of high-level optimization intents. Crucially, this model is *co-evolving* with the search process: it continuously refines its transition dynamics by assimilating execution feedback via in-context learning, allowing it to dynamically update its prior belief and calibrate the search strategy. This approach explicitly decouples high-level planning from low-level program instantiation, enabling the system to navigate complex, non-monotonic optimization paths while remaining resilient to temporary implementation defects.

We evaluate K-Search on a diverse set of complex workloads from FlashInfer [Ye et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib38), including GQA [Yang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib37), MLA [flashinfer-ai (2026)](https://arxiv.org/html/2602.19128v2#bib.bib9), and MoE [Liu et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib23) kernels. Our results demonstrate that K-Search significantly outperforms state-of-the-art baselines, achieving an average 2.10$\times$ improvement over OpenEvolve and an average 2.21$\times$ improvement over ShinkaEvolve. Notably, on the challenging MoE kernel, K-Search delivers a 14.3$\times$ improvement over OpenEvolve. We also test K-Search on the GPUMode [TriMul](https://www.gpumode.com/leaderboard/496?tab=rankings) task, and it achieves state-of-the-art performance ($1030\text{\,}\mathrm{\SIUnitSymbolMicro s}$ on H100), surpassing both prior automated and human-designed solutions. Collectively, these results validate the efficacy of disentangling high-level intent from low-level implementation to enable deep structural optimization.

## 2 Related Work

#### Fast iteration and specialized kernel libraries.

Recent years have seen substantial engineering effort devoted to highly optimized, workload-specific GPU kernel libraries. Examples include FlashAttention [Dao et al. (2022)](https://arxiv.org/html/2602.19128v2#bib.bib6); [Dao (2023)](https://arxiv.org/html/2602.19128v2#bib.bib5); [Shah et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib32) for dense attention, FlashLinearAttention [fla org (2024)](https://arxiv.org/html/2602.19128v2#bib.bib8) for a wide range of linear and state-space attention variants such as Mamba [Gu and Dao (2024)](https://arxiv.org/html/2602.19128v2#bib.bib10) and RWKV [Peng et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib30), and FlashInfer [Ye et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib38) for high-throughput LLM serving with paged KV-cache and dynamic batching. Although each library targets a relatively narrow class of operators, achieving near-peak performance requires careful architecture-specific tuning. As model architectures continue to diversify and introduce new attention mechanisms and custom sequence operators, manually developing and maintaining specialized kernels becomes increasingly costly, motivating automated LLM-based kernel generation methods that can rapidly adapt to new workloads and hardware.

#### Compiler autotuning, DSLs, and kernel optimization.

Automated kernel optimization has a long history in compilers and high-performance systems. TVM [Chen et al. (2018)](https://arxiv.org/html/2602.19128v2#bib.bib4) and its associated auto-schedulers, such as Ansor [Zheng et al. (2020)](https://arxiv.org/html/2602.19128v2#bib.bib42), optimize tensor programs by employing learned cost models to search large scheduling spaces.
Parallel to these search-based approaches, domain-specific frameworks like Triton [Tillet et al. (2019)](https://arxiv.org/html/2602.19128v2#bib.bib34) and layout abstractions like CuTe [NVIDIA (2023)](https://arxiv.org/html/2602.19128v2#bib.bib27) enable higher-level expression of GPU kernels, allowing developers to explicitly manage architecture-aware tiling and memory hierarchies.
Collectively, these systems illustrate the immense complexity of GPU kernel optimization, highlighting the necessity for methods capable of coordinating tiling, memory layouts, and specialized instructions to maximize performance.

#### LLMs for GPU kernel generation.

Most existing LLM-based GPU kernel generation systems [Wei et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib35); [Zhang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib40); [Lange et al. (2025b)](https://arxiv.org/html/2602.19128v2#bib.bib18); [Liao et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib22); [Li et al. (2025a)](https://arxiv.org/html/2602.19128v2#bib.bib19) employ simple iterative search or refinement pipelines, where LLMs generate kernel variants based on compilation results, execution, and profiling feedback. Recent work augments this paradigm with evolutionary strategies to improve exploration, such as EvoEngineer [Guo et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib13), which maintains and manages a population of candidate kernels during search. In parallel, several works [Li et al. (2025c)](https://arxiv.org/html/2602.19128v2#bib.bib21); [Li et al. (2025b)](https://arxiv.org/html/2602.19128v2#bib.bib20); [Baronio et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib2) leverage reinforcement learning to train models to generate optimized CUDA or Triton kernels, primarily focusing on enhancing the model’s one-shot generation or local refinement capabilities.

#### LLM-guided evolutionary and population-based program search.

Recent work has explored coupling large language models with execution-based evaluation and evolutionary search to discover high-quality programs. FunSearch ([Romera-Paredes et al., 2024](https://arxiv.org/html/2602.19128v2#bib.bib31)) pairs an LLM with an evaluator in an evolutionary loop to improve solutions in mathematical and combinatorial domains. AlphaEvolve [Novikov et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib26) generalizes this paradigm to codebase evolution using LLM-generated edits and a program database that seeds subsequent generations. OpenEvolve [Superintelligence (2025)](https://arxiv.org/html/2602.19128v2#bib.bib33), the open-source realization of these ideas, instantiates archive-based evolution with explicit island models and quality-diversity mechanisms such as MAP-Elites [Mouret and Clune (2015)](https://arxiv.org/html/2602.19128v2#bib.bib25). ShinkaEvolve [Lange et al. (2025a)](https://arxiv.org/html/2602.19128v2#bib.bib17) proposes a population-based evolutionary framework that combines performance-driven selection with novelty-aware rejection to improve sample efficiency. Critically, these methods fundamentally treat the LLM merely as a stochastic code generator. They search directly in the space of program implementations, relying on evolutionary heuristics to drive progress rather than leveraging the LLM’s capacity for high-level planning or reasoning.

#### Large Language Models as World Models

Recent work suggests that LLMs can function as implicit or explicit world models for planning and decision making. RAP [Hao et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib15) frames reasoning as planning with an LLM world model. Complementary approaches externalize dynamics by inducing structured domain models (e.g., PDDL) from language and refining them for classical planning [Guan et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib12). Recent works [Fang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib7); [Gu et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib11) further demonstrate that LLMs can act as world models to guide agentic planning tasks by simulating action outcomes and evaluating candidate trajectories. Together, these studies highlight an emerging perspective of treating LLMs as structured world models that support search and model-based planning.

## 3 K-Search

### 3.1 Problem Setup

We formulate GPU kernel synthesis as an optimization problem under a fixed evaluation budget. We list the core notations in [Table 1](https://arxiv.org/html/2602.19128v2#S3.T1).

Table 1: Notation and Definitions

| Symbol | Definition |
| --- | --- |
| $x\in\mathcal{X}$ | Kernel program implementation |
| $o\in\mathcal{O}$ | Observation tuple from execution |
| $\mathcal{E}:\mathcal{X}\to\mathcal{O}$ | Evaluator function |
| $J:\mathcal{X}\to\mathbb{R}$ | Scalar objective score |
| $\mathcal{H}_{t}$ | History $\{(x_{i},o_{i})\}_{i=1}^{t}$ |
| $S_{t}$ | Search state |
| $\mathcal{A}(S_{t})$ | Frontier of actions |
| $V(a\mid S_{t})\in[0,1]$ | Model-estimated priority score |

For a given kernel program $x$, the evaluator returns an observation tuple:

$$
o\;=\;(s,p,m)\;=\;\mathcal{E}(x),
$$

where $s\in\{0,1\}$ indicates correctness, $p\in\mathbb{R}^{+}$ is the performance metric (i.e., latency), and $m$ contains metadata (e.g., compiler logs, profiler output). We define the maximization objective $J(x)$ as the speedup relative to a reference SoTA baseline ($p_{\text{ref}}$):

$$
J(x)\;=\;s\cdot\frac{p_{\text{ref}}}{p}\cdot 100.
$$

The optimization goal is to identify a program $x^{\star}=\operatorname*{arg\,max}_{x\in\mathcal{X}}J(x)$ utilizing a fixed budget of $B$ evaluations.

### 3.2 Search via Co-Evolving World Model

![Refer to caption](assets/24-k-search/figure-001.png)

Figure 1: Overview of K-Search. The framework operates on a Search State $S_{t}$ structured as a search tree. The tree consists of Closed nodes (blue, visited states with attached program like $x_{12}$) and a Frontier of Open nodes (orange, pending hypotheses like $u_{13}$). The workflow iterates through three phases: (1) Action Selection, where the most promising action node is retrieved from the frontier based on world model estimated priority score $V$; (2) Local Refinement, where a stochastic policy $\pi_{\text{code}}$ samples concrete implementations until stagnation; and (3) World Model Update, where the LLM reasons over the trajectory to update the search tree via Insert (adding new actions), Update (adjusting $V$, e.g., $u_{11}$ dropping from 0.9 to 0.6), and Prune (removing less promising nodes like $u_{10}$).

Large Language Models possess rich intrinsic prior knowledge regarding optimization heuristics and strong planning capabilities [Zhao et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib41); [Bohnet et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib3). However, existing evolution methods often underutilize these capabilities by treating the LLM merely as a code generator. We show that effective search requires disentangling *algorithmic planning* (leveraging intrinsic and evolving understanding) from *implementation*.

#### Baseline: Heuristic Search in Program Space.

Existing approaches search directly in the space of the program. At step $t$, they select a context subset $C_{t}\subseteq\mathcal{H}_{t}$ using evolutionary heuristics such as MAP-Elites [Mouret and Clune (2015)](https://arxiv.org/html/2602.19128v2#bib.bib25); [Romera-Paredes et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib31); [Superintelligence (2025)](https://arxiv.org/html/2602.19128v2#bib.bib33)) and generate a new candidate by conditioning on the raw text of previous programs and their associated execution feedback:

$$
x_{t+1}\sim\pi_{\text{LLM}}\left(x\;\middle|\;\left\{(x_{k},o_{k})\right\}_{(x_{k},o_{k})\in C_{t}}\right).
$$

Here, the observation $o_{k}$ is serialized into a textual prompt (e.g., compiler error messages or profiler logs). This formulation inherently couples high-level optimization intent with low-level implementation. Lacking an explicit planning mechanism, a theoretically sound strategy may be discarded simply because of a transient syntax error in $x_{t+1}$.

#### Ours: Search via Co-Evolving World Model.

We structure code generation as a search process guided by an LLM repurposed as a world model with an intrinsic understanding of the design space. The *World Model* estimates the next state of the search process after applying an action to the current
state [Ha and Schmidhuber (2018)](https://arxiv.org/html/2602.19128v2#bib.bib14); [Hao et al. (2023)](https://arxiv.org/html/2602.19128v2#bib.bib15); [Matsuo et al. (2022)](https://arxiv.org/html/2602.19128v2#bib.bib24).

Formally, the LLM world model represents a search state ($S_{t}$) transition distribution $P_{\text{model}}(S_{t+1}\mid S_{t},a_{t})$.
A search state $S_{t}$ encapsulates the current snapshot of the model’s understanding on the search process, including the history of explored actions and their performance, the *frontier* $\mathcal{A}(S_{t})$ actions (i.e., the set of pending, unexplored actions available), and the estimated priority score $V$ over the frontier actions. Crucially, this model is *co-evolving* with the search process: it continuously refines its understanding and beliefs based on past trials and accumulated experience via in-context learning.

The search proceeds through three iterative phases:

1. Action Selection.
The hypothesis with the highest priority score $V$ from the search state’s current frontier $\mathcal{A}(S_{t})$ is selected as the next action:

$$
a_{t}=\operatorname*{arg\,max}_{a\in\mathcal{A}(S_{t})}V(a\mid S_{t}).
$$

Here, an action $a_{t}=(x_{\text{parent}},\delta)$ represents a specific intent $\delta$ (e.g., “Resolve bank conflicts via padding”) applied to a specific parent program $x_{\text{parent}}$. $V$ is a scalar priority score predicted by the world model when the action is created (or updated). It represents the model’s intrinsic assessment of the potential of the actions.

2. Program Instantiation.
A concrete program is then synthesized by applying the selected plan to the parent implementation found in $S_{t}$:

$$
x_{t}\sim\pi_{\text{code}}(x\mid a_{t}),\quad o_{t}=\mathcal{E}(x_{t}).
$$

3. World Model Co-Evolution.
Upon observing the result $o_{t}$ of $x_{t}$ (i.e., the outcome of action $a_{t}$), the world model then performs the search state transition to obtain $S_{t+1}$ conditioned on the newly accumulated experience, which updates $\mathcal{A}$ and calibrates $V$:

$$
S_{t+1}\sim P_{\text{model}}(S\mid S_{t},a_{t};x_{t},o_{t}).
$$

This co-evolution allows the world model to update its prior assumptions and beliefs effectively, and perform the state transition to progressively sharpen the search policy.

### 3.3 System Design

Algorithm 1  K-Search: Search via Co-Evolving World Models

````


Input: Spec \mathcal{T}, Evaluator \mathcal{E}, Budget B, Stagnation Limit K


Init: S\leftarrow\text{Init}(\mathcal{T})


while B>0 do


  \triangleright 1. Selection: Retrieve best action from frontier


  a_{t}\leftarrow\operatorname*{arg\,max}_{a\in\mathcal{A}(S)}V(a\mid S)


  n\leftarrow 0 \triangleright Stagnation counter


  x_{\text{best}}\leftarrow\bot,\hskip 9.24994pto_{\text{best}}\leftarrow\bot


  \triangleright 2. Instantiation: Local refinement loop


  while B>0 and n<K do


   x\sim\pi_{\text{code}}(\cdot\mid a_{t})


   o\leftarrow\mathcal{E}(x);    B\leftarrow B-1


   if J(x)>J(x_{\text{best}}) then


      x_{\text{best}}\leftarrow x;    o_{\text{best}}\leftarrow o


      n\leftarrow 0 \triangleright Reset counter on improvement


   else


      n\leftarrow n+1 \triangleright Increment on failure


   end if


  end while


  \triangleright 3. Evolution: Insert, Update, and Prune


  S\leftarrow\text{P}_{\text{model}}(S,a_{t},x_{\text{best}},o_{\text{best}})


end while


return Best found x^{\star}
````

We build K-Search based on the concept of Search via Co-Evolving World Model. In our design, the LLM functions as an intrinsic world model that maintains and evolves a tree-structured state of the search process. The system operates by iteratively expanding, updating, and pruning an explicit search tree, decoupling high-level planning from low-level code generation. [Figure 1](https://arxiv.org/html/2602.19128v2#S3.F1) and [Algorithm 1](https://arxiv.org/html/2602.19128v2#alg1) illustrate the complete workflow of K-Search.

#### Search State.

The search state $S_{t}$ is maintained as an explicit search tree partitioned into Closed and Open nodes. Closed nodes (blue boxes in [Figure 1](https://arxiv.org/html/2602.19128v2#S3.F1), e.g., $x_{12}$) represent visited actions where the local refinement process has concluded and the best-found programs are attached. In contrast, Open nodes (orange dashed boxes, e.g., $u_{13}$) form the frontier $\mathcal{A}(S_{t})$ of pending actions waiting to be realized. Each Open node encapsulates a *Proposed Optimization* tuple $(x_{\text{parent}},\delta)$, linking a parent program to a specific natural language intent, and a *Priority Score* $V\in[0,1]$. This score is dynamically updated by the world model (e.g., $u_{11}$ in the figure shows $V$ dropping from $0.9\to 0.6$) to guide the selection of the next action.

#### Execution: Local Refinement.

To isolate logical reasoning from implementation details, we treat program instantiation as a local refinement task. Upon selecting an action $a_{t}$ (Step 1 in [Figure 1](https://arxiv.org/html/2602.19128v2#S3.F1)), the system employs the LLM as a stochastic policy $\pi_{\text{code}}$. We repeatedly sample implementations $x\sim\pi_{\text{code}}(\cdot\mid a_{t})$ and evaluate them ($o=\mathcal{E}(x)$) until a *stagnation condition* is met ($K$ consecutive attempts without improvement). This ensures that valid actions are not discarded due to transient syntax errors or minor bugs.

#### Evolution: World Model Update.

Upon concluding local refinement, the LLM analyzes the execution trajectory to update the state $S_{t}$ through three kinds of Tree Edit Operations (Step 3 in [Figure 1](https://arxiv.org/html/2602.19128v2#S3.F1)). It performs *Insert* to propose new child nodes extending the current state (e.g., adding action $u_{13}$ with intent $\delta=$ “fuse head”) and *Update* to re-evaluate the priority of existing frontier nodes based on new evidence. Additionally, the model applies *Prune* to identify and permanently remove infeasible or redundant branches (e.g., node $u_{10}$), thereby concentrating search resources on promising directions. Note that in the current version of K-Search, the world model evolution is merely performed by in-context learning of past observations.

### 3.4 Case Study: MLA Paged Decode

![Refer to caption](assets/24-k-search/figure-002.svg)

Figure 2: K-Search Search Trace Visualization. It tracks the evolution of the Search State across search rounds on the MLA Paged Decode kernel (refer to [Section 4](https://arxiv.org/html/2602.19128v2#S4) for setup details). A round corresponds to one candidate program evaluation. Nodes represent actions (dark=Closed, light=Open), annotated with their instantiated program performance (closed nodes) or priority scores (open nodes).
The timeline highlights how the kernel is improved and how the LLM dynamically *Inserts* new hypotheses, *Updates* beliefs, and *Prunes* less promising branches based on evolved understanding.

We illustrate the co-evolutionary search process using the MLA Paged Decode kernel (refer to [Table 2](https://arxiv.org/html/2602.19128v2#S4.T2) for kernel details). [Figure 2](https://arxiv.org/html/2602.19128v2#S3.F2) visualizes how the World Model co-evolves with the kernel implementation. We track the progress in rounds, where each round corresponds to a single invocation of the evaluator $\mathcal{E}(x)$ (i.e., one compiled and benchmarked kernel, consuming one unit of the budget).

#### r1: Initialization and Hypothesis Selection.

The search initializes $S_{0}$ with three high-level actions in the frontier: *fused_multi_head*, *split_k_decoding*, and *independent_heads*. The World Model assigns the highest value ($V$) to *fused_multi_head*, hypothesizing that processing shared CKV heads together will reduce global memory traffic by $16\times$ compared to independent processing, and selects it for the first round of program instantiation.

#### r14-r34: Tree Evolution via Topological Edits.

Following the local refinement of *fused_multi_head* (score $J=34$), the model evolves the topology of $S_{t}$ to reflect this feedback. It first *Inserts* refinement actions such as *register_resident_rescaling* and *occupancy_tuned_chunk32* to deepen the successful branch. Simultaneously, it *Updates* the belief state by downgrading the sibling *independent_heads*, reasoning that the proven efficacy of head fusion renders independent processing less promising. By round 34, as evidence accumulates, the model permanently *Prunes* the *independent_heads* branch, reallocating search resources entirely to the *register_resident* subtree.

#### r42-r102: Structural Insight and Experience Accumulation.

At round 42, the World Model exhibits a structural shift in reasoning. It deletes the initial root-level *split_k* action but re-*Inserts* a targeted variant, *low_overhead_split_k*, deep within the *register_resident* branch. This edit reflects a learned insight: split-K is ineffective as an isolated baseline but highly effective as a *composable* optimization atop a strong fusion kernel. Notably, after *chunk32_vectorized* succeeds, the model proposes *chunk32_prescale_vectorized* to apply *sm_scale* immediately upon loading $Q$. This specific refinement eventually yields the global optimum (star marker) at r102.

#### Takeaway.

This trace highlights the efficiency of K-Search. By starting with high-level intent rather than raw code, the system avoids enumerating a massive sparse search space. It relies on local refinement to filter out transient coding noise and allows the World Model’s understanding to *co-evolve* with the kernel’s optimization progress. Such evolution enables the system to prune dead ends and dynamically reposition strategies, guided by intrinsic reasoning over accumulated experience.

## 4 Experiments

Table 2: Representative kernels used for evaluation. Detailed configurations (e.g., head counts) are omitted for brevity. We provide example optimization challenges based on FlashInfer [flashinfer-ai (2026)](https://arxiv.org/html/2602.19128v2#bib.bib9) implementations.

| Kernel | Arch. | Model Context | Characteristics & Challenges |
| --- | --- | --- | --- |
| MLA Paged Prefill | Hopper | DeepSeek-V3 [Liu et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib23) | Implements attention with paged KV cache and split ckv (compressed, no RoPE) and kpe (RoPE). Bandwidth-bound in long contexts; requires matrix absorption and Split-K optimization. |
| MLA Paged Decode | Hopper | DeepSeek-V3 | Targets large dynamic batches. Latency-bound at low batch sizes. Uses Persistent Data Layout and specialized Hopper-instructions to minimize memory movement. |
| GQA Paged Decode | Hopper | Qwen3-A3B-30B [Yang et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib37) | Memory bound. Optimizations include fusing multiple query heads per KV head to reuse KV loads, and using vectorized/async paged-KV gather with pipelined global to shared memory prefetch. |
| FP8 MoE | Blackwell | DeepSeek-V3 | Challenges include irregular data-dependent routing, load balancing, and managing FP8 packing/scaling overhead. |

### 4.1 Setup

Our main evaluation is conducted on FlashInfer [Ye et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib38) kernels, since these kernels are highly-optimized by experienced human engineers. For each target kernel, we run each system for a fixed budget of 120 iterations (we define one iteration as one evaluation of a single candidate kernel on the benchmark workloads), and report the best-so-far score (i.e., $J(x)$ averaged over workloads in [Section 3.1](https://arxiv.org/html/2602.19128v2#S3.SS1)) achieved at each iteration, using FlashInfer [Ye et al. (2025)](https://arxiv.org/html/2602.19128v2#bib.bib38) kernels as reference SoTA.
We repeat each method three times and report the mean curve with a shaded min–max band to indicate the range of scores achieved across repeated experiments.

#### Implementation details.

K-Search exposes a unified Task interface for plugging in optimization problems. Each task is defined by (i) a task specification and (ii) an evaluator. A task specification contains the PyTorch reference implementation, the optimization objective, and any task-specific instructions. The evaluator is responsible for compiling candidate implementations, validating functional correctness against the reference code, and measuring performance under a standardized benchmarking environment.

For FlashInfer kernels, K-Search integrates FlashInfer-Bench [Xing et al. (2026)](https://arxiv.org/html/2602.19128v2#bib.bib36) as the task evaluator. All candidate implementations must be written in CUDA and pass functional correctness tests before receiving a non-zero score. We adopt the same compilation toolchain, correctness suite, and benchmark harness across all compared methods to ensure fairness. Experiments are conducted on NVIDIA H100 and B200 GPUs using CUDA 12.8, FlashInfer 0.5.3, and PyTorch 2.8.0.

![Refer to caption](assets/24-k-search/figure-003.svg)

(a) Search Process.

![Refer to caption](assets/24-k-search/figure-004.svg)

(b) Best Kernel per-Workload Performance.

![Refer to caption](assets/24-k-search/figure-005.svg)

(c) Best Kernel $\text{Fast}_{p}$ Plot.

Figure 3: Main Results (3 runs each). (a) compares the kernels best-so-far scores generated by the three systems across 120 iterations. (b) provides a per-workload analysis for all compared systems. (c) shows the fraction of workloads for which the best kernel from each system achieves the specified speedup over the FlashInfer baseline.

### 4.2 Baselines

We evaluate 3 automated kernel-optimization methods: OpenEvolve [Superintelligence (2025)](https://arxiv.org/html/2602.19128v2#bib.bib33), ShinkaEvolve [Lange et al. (2025a)](https://arxiv.org/html/2602.19128v2#bib.bib17), and K-Search. We used gemini-3-pro-preview and the same initial program for each kernel. For ShinkaEvolve, we used Qwen3-8B as the embedding model. Across all experiments, we standardized the evaluation by using an identical set of input workloads for all methods on each kernel, and adhered to the default configurations for each baseline with the prompt template shown in [Section 6.2](https://arxiv.org/html/2602.19128v2#S6.SS2). K-Search sets budget value $B=120$ and stagnation value $K=7$.

### 4.3 Kernels

We focus on four representative kernels (i.e., paged attention and MoE) from FlashInfer-Bench [Xing et al. (2026)](https://arxiv.org/html/2602.19128v2#bib.bib36) that are widely used in modern LLM serving. We summarize these kernels in [Table 2](https://arxiv.org/html/2602.19128v2#S4.T2). Each kernel includes a fixed set of test traces used for correctness and benchmarking, captured in real traffic [Xing et al. (2026)](https://arxiv.org/html/2602.19128v2#bib.bib36).
We use identical traces for OpenEvolve, ShinkaEvolve, and K-Search to ensure a fair comparison. We provide an initial CUDA program for MLA decode kernel generation, as baselines struggle to write working kernels without an initial program.

### 4.4 Evaluation Results

Overall Performance. [Figure 3(a)](https://arxiv.org/html/2602.19128v2#S4.F3.sf1) compares the three systems across 120 iterations for GQA decode, MLA decode, MLA prefill, and MoE kernels ([Section 4.3](https://arxiv.org/html/2602.19128v2#S4.SS3)).
For each method and kernel, we plot the scores over iterations across repeated runs. We find that K-Search significantly outperforms OpenEvolve and ShinkaEvolve. Across all kernels, K-Search achieves an overall average final score of 56.13, representing a 2.10 $\times$ improvement over OpenEvolve (with score 26.68) and a 2.21$\times$ improvement over ShinkaEvolve (with score 25.37). The performance gains vary across kernels. For the MoE kernel, K-Search achieves a final score of $44.1$, representing a $14.3\times$ improvement over OpenEvolve ($3.09$) and a $1.58\times$ improvement over ShinkaEvolve ($27.9$). Similarly, for MLA prefill, K-Search achieves a score of $57.4$ compared to OpenEvolve’s $19.5$ and ShinkaEvolve’s $11.3$, or $2.95\times$ and $5.10\times$ improvements, respectively. For GQA decode, K-Search reaches a score of $76.0$, outperforming OpenEvolve ($44.2$) and ShinkaEvolve ($27.7$) by $1.72\times$ and $2.74\times$. On the MLA decode, K-Search maintains a final score of $47.1$ versus $39.9$ and $34.7$, representing $18\%$ and $36\%$ improvements.

Best Kernel per-Workload Performance. [Figure 3(b)](https://arxiv.org/html/2602.19128v2#S4.F3.sf2) provides a per-workload analysis for all compared methods. Each dot represents the kernel’s performance on a particular workload instance. Across $4$ kernel types and $152$ total workload traces, K-Search achieves higher performance than baselines on the vast majority of workloads.

Interestingly, on some workloads for GQA decode, K-Search underperforms OpenEvolve and ShinkaEvolve, specifically those with small batch sizes: 16 of which have batch_size=1, and 4 with batch_size=16. K-Search does not underperform on workload with larger batch_size. This occurs because K-Search’s kernel employs a *split-K* parallelism strategy that divides the key-value sequence across multiple thread blocks to maximize GPU utilization for large batches (described more in [Section 4.6](https://arxiv.org/html/2602.19128v2#S4.SS6)). While this approach excels when there is sufficient batch-level parallelism to amortize the coordination overhead, it introduces unnecessary synchronization costs for small batches. In contrast, OpenEvolve and ShinkaEvolve use a simpler single-block-per-batch design that processes the entire sequence within one thread block. While this simpler approach proves more efficient for batch_size=1 (no coordination), it leads to bad performance with larger batch size.

Best Kernel $\text{Fast}_{p}$ Analysis. [Figure 3(c)](https://arxiv.org/html/2602.19128v2#S4.F3.sf3) shows the fraction of workloads for which the best kernel from each system achieves the specified speedup over the FlashInfer baseline. We observe that the generated kernels rarely exceed the expert-optimized FlashInfer kernel; however, K-Search significantly outperforms OpenEvolve and ShinkaEvolve. For GQA decode, K-Search attains a speedup $\geq 0.36$ on $100\%$ of workloads, compared to $50\%$ for ShinkaEvolve. At higher thresholds, the gap widens: at speedup $\geq 0.50$, K-Search succeeds on $87.5\%$ of workloads versus $50.0\%$ and $39.6\%$ for OpenEvolve and ShinkaEvolve, representing $1.75\times$ and $2.21\times$ improvements. For MLA prefill, at speedup $\geq 0.40$, K-Search reaches this threshold on $57.9\%$ of workloads, while none of the baseline solutions achieve speedup $\geq 0.40$.

#### Key observations.

OpenEvolve and ShinkaEvolve both evolve over an archive of programs. The crucial difference from K-Search is that both directly search in program space and neither system explicitly tracks which optimizations are likely to be correct or to improve performance. Consequently, ShinkaEvolve suffers from a low yield of correct programs: in our GQA logs, the vast majority of generations receive score zero (incorrect or failed programs). Search budget is heavily spent mostly on extending invalid or low-performing candidates. OpenEvolve similarly exhibits high per-iteration variance: Many iterations yield programs that underperform the currently best program. On harder tasks such as MoE, OpenEvolve’s search struggles to escape the low-accuracy regime: mean final score stays near 3 versus K-Search’s 44.
We hypothesize that this advantage stems from K-Search’s ability to maintain and co-evolve a persistent search state. By conditioning proposals on past attempts, the system generates more targeted hypotheses, significantly improving search efficiency.

### 4.5 Kernel Analysis: FP8 MoE Kernel (Blackwell)

We first discuss the FP8 MoE kernel: for each token, the top-$k$ experts are chosen from 256 candidate experts. The kernel then runs an “up” and “gate” projection, combined via SiLU, then a down-projection. All systems generate CUDA kernels for the same specification (DeepSeek-V3 style, top-8 routing, 32 local experts, hidden size 7168). We compare kernels generated by K-Search, OpenEvolve, and ShinkaEvolve.

#### Routing

Each token is assigned scores for 256 experts; the kernel then picks the top 8 experts. K-Search’s kernel dedicates one GPU thread block per token (256 threads) and uses *warp-level* cooperation: threads in a warp exchange values (__shfl_down_sync) to find the the global top-8 experts. This keeps work parallel and avoids serialization.

OpenEvolve uses a persistent kernel where each thread block runs in a loop: take the next “tile index” from a global counter (atomicAdd), identify the (expert, token batch) it corresponds to, then perform computation for that tile. This incurs prohibitive overhead as it requires an atomic operation to process the next tile inside a while loop. In ShinkaEvolve, each thread loads all 256 scores for the experts and does plain for-loops to find the top experts, similarly leading to poor performance.

#### Expert FFN computation

After routing, tokens must be processed by chosen experts. K-Search uses a simple pipeline: (1) routing, (2) sort-scatter that reorders tokens by expert to place them in contiguous memory, (3) computation (gate + up, then SiLU). K-Search uses tensor cores (with WMMA) on small $16\times 16$ blocks and double-buffering so that loading the next block of data overlaps with computing the current one. K-Search’s kernel also skips experts that receive zero tokens. As discussed, OpenEvolve adopts a single persistent kernel, which reduces kernel launches but needs more shared memory with lower GPU occupancy.
ShinkaEvolve’s kernel does not use tensor cores; each block performs a dot-product-style computation, thereby suffering from low performance.

### 4.6 Kernel Analysis: GQA Paged Decode (Hopper)

We next discuss the GQA paged decode kernel. In decode, each batch item contributes a single new query token, and the kernel attends over the keys and values already stored in the paged KV cache. All three systems—K-Search, OpenEvolve, and ShinkaEvolve—generate CUDA kernels for the same specification.

#### Parallelism over the sequence

In decode, the costly part is sweeping over the full key-value sequence. K-Search’s kernel splits the sequence across multiple blocks: each block is assigned a contiguous chunk of keys and values, computes a partial attention result for that chunk, and writes it to a temporary buffer. When all chunks for a given (batch, key-value head) are done, one block detects that it is last (via a lightweight counter) and merges the partial results into the final output. Thus, for long sequences, many blocks work in parallel on different segments. OpenEvolve and ShinkaEvolve, however, use a single block per (batch, key-value head): that block loops over the entire key-value sequence itself. Therefore, for long sequences, they cannot exploit the same parallelism as K-Search.

#### Overlapping memory and compute

K-Search loads keys and values in double-buffered chunks: while the current chunk is used for computation, the next chunk is being fetched. ShinkaEvolve does not double-buffer: it loads one chunk of keys and values, performs all attention work for that chunk, then loads the next. Memory load and compute are largely serialized, leading to lower performance.

We defer discussion of other generated kernels to [Section 6.1](https://arxiv.org/html/2602.19128v2#S6.SS1).

### 4.7 GPUMODE TriMul

#### Task Overview.

[GPUMODE](https://www.gpumode.com/home) is a public kernel optimization competition platform that hosts leaderboard-based challenges on real-world GPU workloads.
Participants submit Triton or CUDA implementations that must pass strict correctness checks before being evaluated under standardized benchmarking conditions.

The *Triangle Multiplicative Update* (TriMul) is a core module in AlphaFold3 [Abramson et al. (2024)](https://arxiv.org/html/2602.19128v2#bib.bib1) and other protein structure prediction models.
It operates on a 4D pair representation $\mathbf{X}\in\mathbb{R}^{B\times N\times N\times C}$ and involves LayerNorm, five gated linear projections with optional masking, a pairwise contraction with $\mathcal{O}(N^{3})$ complexity, and a final gated output projection.

#### Configuration.

We run K-Search with $K{=}5$, reduced from $K{=}7$ used on FlashInfer CUDA tasks, as Triton’s implementation is simpler than CUDA.
The search budget is 300 iterations: 150 steps with GPT-5.2, followed by 150 continuation steps with Gemini-3-Pro starting from the best solution found by GPT-5.2, without seed Triton programs provided.

Table 3: Top GPUMODE TriMul submissions on NVIDIA H100. Latency is the geometric mean across a fixed set of benchmark cases.

| Submission ID | Lang. | Model | Iter. | Latency ($\mathrm{\SIUnitSymbolMicro s}$, $\downarrow$) |
| --- | --- | --- | --- | --- |
| shiyegao | CUDA | – | – | $1074$ |
| Zeyu Shen | Triton | – | – | $1140$ |
| TTT | Triton | GPT-OSS-20B w/ RL | 25,600 | $1161$ |
| K-Search (Ours) | Triton | GPT-5.2 + Gemini-3-Pro | 300 | 1030 |

#### Results.

We report leaderboard results from the official GPUMODE rankings (Footnote 1: https://www.gpumode.com/leaderboard/496?tab=rankings)  for competing submissions.
Since the submission period has closed, we evaluate our kernel locally using the official GPUMODE evaluator (Footnote 2: https://github.com/gpu-mode/reference-kernels/blob/main/problems/bioml/trimul/eval.py)  on NVIDIA H100 80 GB HBM3 GPUs.
As shown in [Table 3](https://arxiv.org/html/2602.19128v2#S4.T3), K-Search achieves state-of-the-art performance with a geometric-mean latency of $1030\text{\,}\mathrm{\SIUnitSymbolMicro s}$, surpassing prior solutions, where the TTT submission is from a recent work TTT-Discover [Yuksekgonul et al. (2026)](https://arxiv.org/html/2602.19128v2#bib.bib39) that combines Reinforcement Learning (RL) with evolution methods.

## 5 Conclusion

In this work, we introduce K-Search, demonstrating that LLMs are not merely strong code generators but also possess latent planning capabilities that allow them to function as effective intrinsic world models. By replacing static search heuristics with a co-evolving world model, our framework enables the LLM to actively reason over the search space, distinguishing valid strategies from implementation noise. The empirical results, including a $2.1\times$ average improvement across diverse complicated kernels, up to $14.3\times$ gains on MoE over state-of-the-art evolutionary baselines, and SoTA performance on GPUMODE TriMul competition, validate this paradigm shift. These findings indicate that LLMs can serve as the core planning engine for complex optimization problems, moving beyond simple task implementation to autonomously driving the search strategy itself.

## Acknowledgment

This work is supported by the kind compute support from Databricks, Amazon, Anyscale, and
gifts from Google, Lambda, AMD, Mayfield, Accenture, Broadcom, Cisco, IBM, Intel, Intesa Sanpaolo, Lightspeed, Mibura, Microsoft, NVIDIA, Samsung SDS, and SAP. We also thank Dacheng Li, Mayank Mishra, Andy Yang, Parth Asawa, Fangzhou Zhao, and Tian Xia for helpful discussion and feedback.

## References

- Abramson et al. (2024)

  Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al.
  Accurate structure prediction of biomolecular interactions with alphafold 3.
  *Nature*, 630(8016):493–500, 2024.
- Baronio et al. (2025)

  Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, and Silas Alberti.
  Kevin: Multi-turn rl for generating cuda kernels.
  *arXiv preprint arXiv:2507.11948*, 2025.
- Bohnet et al. (2025)

  Bernd Bohnet, Pierre-Alexandre Kamienny, Hanie Sedghi, Dilan Gorur, Pranjal Awasthi, Aaron Parisi, Kevin Swersky, Rosanne Liu, Azade Nova, and Noah Fiedel.
  Enhancing llm planning capabilities through intrinsic self-critique.
  *arXiv preprint arXiv:2512.24103*, 2025.
- Chen et al. (2018)

  Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al.
  $\{$TVM$\}$: An automated $\{$End-to-End$\}$ optimizing compiler for deep learning.
  In *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*, pages 578–594, 2018.
- Dao (2023)

  Tri Dao.
  Flashattention-2: Faster attention with better parallelism and work partitioning.
  *arXiv preprint arXiv:2307.08691*, 2023.
- Dao et al. (2022)

  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
  Flashattention: Fast and memory-efficient exact attention with io-awareness.
  *Advances in neural information processing systems*, 35:16344–16359, 2022.
- Fang et al. (2025)

  Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, and Dong Yu.
  Webevolver: Enhancing web agent self-improvement with co-evolving world model.
  In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 8970–8986, 2025.
- fla org (2024)

  fla org.
  Flashlinearattention.
  <https://github.com/fla-org/flash-linear-attention>, 2024.
- flashinfer-ai (2026)

  note = Last indexed: 24 January 2026. Accessed: 28 January 2026 flashinfer-ai, howpublished = <https://deepwiki.com/flashinfer-ai/flashinfer/2.5-multi-level-attention-%28mla%29>.
  Multi-level attention (mla), 2026.
- Gu and Dao (2024)

  Albert Gu and Tri Dao.
  Mamba: Linear-time sequence modeling with selective state spaces.
  In *First conference on language modeling*, 2024.
- Gu et al. (2024)

  Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, et al.
  Is your llm secretly a world model of the internet? model-based planning for web agents.
  *arXiv preprint arXiv:2411.06559*, 2024.
- Guan et al. (2023)

  Lin Guan, Karthik Valmeekam, Sarath Sreedharan, and Subbarao Kambhampati.
  Leveraging pre-trained large language models to construct and utilize world models for model-based task planning.
  *Advances in Neural Information Processing Systems*, 36:79081–79094, 2023.
- Guo et al. (2025)

  Ping Guo, Chenyu Zhu, Siyuan Chen, Fei Liu, Xi Lin, Zhichao Lu, and Qingfu Zhang.
  Evoengineer: Mastering automated cuda kernel code evolution with large language models.
  *arXiv preprint arXiv:2510.03760*, 2025.
- Ha and Schmidhuber (2018)

  David Ha and Jürgen Schmidhuber.
  World models.
  *arXiv preprint arXiv:1803.10122*, 2(3):440, 2018.
- Hao et al. (2023)

  Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu.
  Reasoning with language model is planning with world model.
  In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 8154–8173, 2023.
- Kwon et al. (2023)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.
  Efficient memory management for large language model serving with pagedattention.
  In *Proceedings of the 29th symposium on operating systems principles*, pages 611–626, 2023.
- Lange et al. (2025a)

  Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin.
  Shinkaevolve: Towards open-ended and sample-efficient program evolution.
  *arXiv preprint arXiv:2509.19349*, 2025a.
- Lange et al. (2025b)

  Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, and David Ha.
  Towards robust agentic cuda kernel benchmarking, verification, and optimization.
  *arXiv preprint arXiv:2509.14279*, 2025b.
- Li et al. (2025a)

  Haonan Li, Keyu Man, Partha Kanuparthy, Hanning Chen, Wei Sun, Sreen Tallam, Chenguang Zhu, Kevin Zhu, and Zhiyun Qian.
  Tritonforge: Profiling-guided framework for automated triton kernel optimization.
  *arXiv preprint arXiv:2512.09196*, 2025a.
- Li et al. (2025b)

  Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, et al.
  Autotriton: Automatic triton programming with reinforcement learning in llms.
  *arXiv preprint arXiv:2507.05687*, 2025b.
- Li et al. (2025c)

  Xiaoya Li, Xiaofei Sun, Albert Wang, Jiwei Li, and Chris Shum.
  Cuda-l1: Improving cuda optimization via contrastive reinforcement learning.
  *arXiv preprint arXiv:2507.14111*, 2025c.
- Liao et al. (2025)

  Gang Liao, Hongsen Qin, Ying Wang, Alicia Golden, Michael Kuchnik, Yavuz Yetim, Jia Jiunn Ang, Chunli Fu, Yihan He, Samuel Hsia, et al.
  Kernelevolve: Scaling agentic kernel coding for heterogeneous ai accelerators at meta.
  *arXiv preprint arXiv:2512.23236*, 2025.
- Liu et al. (2024)

  Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
  Deepseek-v3 technical report.
  *arXiv preprint arXiv:2412.19437*, 2024.
- Matsuo et al. (2022)

  Yutaka Matsuo, Yann LeCun, Maneesh Sahani, Doina Precup, David Silver, Masashi Sugiyama, Eiji Uchibe, and Jun Morimoto.
  Deep learning, reinforcement learning, and world models.
  *Neural Networks*, 152:267–275, 2022.
- Mouret and Clune (2015)

  Jean-Baptiste Mouret and Jeff Clune.
  Illuminating search spaces by mapping elites.
  *arXiv preprint arXiv:1504.04909*, 2015.
- Novikov et al. (2025)

  Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al.
  Alphaevolve: A coding agent for scientific and algorithmic discovery.
  *arXiv preprint arXiv:2506.13131*, 2025.
- NVIDIA (2023)

  NVIDIA.
  Cuda templates and python dsls for high-performance linear algebra.
  <https://github.com/NVIDIA/cutlass>, 2023.
- NVIDIA (2025)

  NVIDIA.
  *Parallel Thread Execution ISA Version 9.1*, 2025.
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>.
  Accessed: 2025-02-28.
- Ouyang et al. (2025)

  Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini.
  Kernelbench: Can llms write efficient gpu kernels?
  *arXiv preprint arXiv:2502.10517*, 2025.
- Peng et al. (2023)

  Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al.
  Rwkv: Reinventing rnns for the transformer era.
  *arXiv preprint arXiv:2305.13048*, 2023.
- Romera-Paredes et al. (2024)

  Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al.
  Mathematical discoveries from program search with large language models.
  *Nature*, 625(7995):468–475, 2024.
- Shah et al. (2024)

  Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao.
  Flashattention-3: Fast and accurate attention with asynchrony and low-precision.
  *Advances in Neural Information Processing Systems*, 37:68658–68685, 2024.
- Superintelligence (2025)

  Algorithmic Superintelligence.
  Openevolve: Teaching llms to discover algorithms through quality-diversity search.
  Blog post, 2025.
  <https://algorithmicsuperintelligence.ai/blog/openevolve-overview/>.
- Tillet et al. (2019)

  Philippe Tillet, Hsiang-Tsung Kung, and David Cox.
  Triton: an intermediate language and compiler for tiled neural network computations.
  In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, pages 10–19, 2019.
- Wei et al. (2025)

  Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, and Alex Aiken.
  Astra: A multi-agent system for gpu kernel performance optimization.
  *arXiv preprint arXiv:2509.07506*, 2025.
- Xing et al. (2026)

  Shanli Xing, Yiyan Zhai, Alexander Jiang, Yixin Dong, Yong Wu, Zihao Ye, Charlie Ruan, Yingyi Huang, Yineng Zhang, Liangsheng Yin, et al.
  Flashinfer-bench: Building the virtuous cycle for ai-driven llm systems.
  *arXiv preprint arXiv:2601.00227*, 2026.
- Yang et al. (2025)

  An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.
  Qwen3 technical report.
  *arXiv preprint arXiv:2505.09388*, 2025.
- Ye et al. (2025)

  Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al.
  Flashinfer: Efficient and customizable attention engine for llm inference serving.
  *arXiv preprint arXiv:2501.01005*, 2025.
- Yuksekgonul et al. (2026)

  Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, et al.
  Learning to discover at test time.
  *arXiv preprint arXiv:2601.16175*, 2026.
- Zhang et al. (2025)

  Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, and Caiwen Ding.
  Cudaforge: An agent framework with hardware feedback for cuda kernel optimization.
  *arXiv preprint arXiv:2511.01884*, 2025.
- Zhao et al. (2023)

  Zirui Zhao, Wee Sun Lee, and David Hsu.
  Large language models as commonsense knowledge for large-scale task planning.
  *Advances in neural information processing systems*, 36:31967–31987, 2023.
- Zheng et al. (2020)

  Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, et al.
  Ansor: Generating $\{$High-Performance$\}$ tensor programs for deep learning.
  In *14th USENIX symposium on operating systems design and implementation (OSDI 20)*, pages 863–879, 2020.
- Zheng et al. (2024)

  Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al.
  Sglang: Efficient execution of structured language model programs.
  *Advances in neural information processing systems*, 37:62557–62583, 2024.

## 6 Appendix

### 6.1 Other generated kernels analysis

#### MLA Paged Prefill (Hopper):

In this kernel, each query is a 576-dimensional vector (512 CKV plus 64 ROPE).
The kernel performs causal attention: it computes attention scores between queries and keys, applies softmax to get attention weights $P$, then forms the output as a weighted sum of values, $P\cdot V$. All three systems—K-Search, *OpenEvolve*, and *ShinkaEvolve*—generate CUDA kernels for the same specification (16 heads, causal masking, paged layout).

Handling variable-length batches: Each input batch has sequences of different lengths. A natural approach is to partition work into “tiles” of 16 query rows (one “row” of the attention computation for a single query vector). Because sequence boundaries do not align with tiles, a single tile can contain rows from more than one batch. K-Search handles this entirely on the GPU: each thread block is assigned a contiguous tile of 16 rows. When a 16-row tile straddles the end of one sequence and the start of the next, the thread block resolves the split on the fly. It uses the prefix-sum array of sequence boundaries to determine which subset of rows in the tile belongs to each sequence, then fetches the corresponding KV-cache range for that sequence and computes attention for that contiguous “segment” of rows. It then moves to the next segment within the same tile, repeating until the tile is done. OpenEvolve and *ShinkaEvolve* instead precompute a list of tile to batch mapping on the CPU and pass it to the GPU. However, that adds a separate CPU pass and extra memory to store the tile metadata.

Score computation and softmax: The expensive part of attention is computing attention scores (the equivalent of $QK^{\top}$) and then softmax. K-Search keeps all threads in a block busy during this phase: they jointly compute the score matrix in small blocks, combine partial results in SRAM, then run the softmax per row. OpenEvolve effectively restricts the score-and-softmax stage to one small group of threads (a single “warp”); the rest of the block sits idle during that time, so a large fraction of the GPU’s capacity in that block is unused.

In summary, K-Search is faster because it (1) resolves batch boundaries on the GPU without a precomputed tile list or extra host-side pass; (2) keeps all thread groups busy during computation.

#### MLA Paged Decode (Hopper):

Lastly, we discuss the MLA paged decode kernel. In decode, each batch item has a single new query (one token). It computes attention scores $QK^{\top}$ (over both CKV and KPE), applies softmax to get weights $P$, then forms the output as the weighted sum $P\cdot V$. All three systems—K-Search (best solution), *OpenEvolve*, and *ShinkaEvolve*—generate CUDA kernels for the same specification (16 heads, CKV dim 512, KPE dim 64).

Sequence split and chunk size. All three systems split the key-value sequence across multiple blocks: each block processes a contiguous chunk of the sequence, writes partial results to a temporary buffer, and a separate reduce step merges them into the final output and log-sum-exp. K-Search chooses when to split adaptively: for short sequences it uses a single block per (batch, head) and writes directly to the output, avoiding the reduce pass and extra memory. OpenEvolve and *ShinkaEvolve* use larger chunks (64 tokens) and fix a minimum number of splits, so they do not adapt well to short sequences.

Avoiding shared-memory staging of the per-token query vectors (Q). In MLA decode, the per-token query vectors (Q) are small (16 heads $\times$ 576 dims) but are reused across every chunk processed by a thread block. K-Search loads Q into register-resident fragments and reuses these fragments for all chunks in the block, without materializing the full Q in shared memory. In contrast, OpenEvolve and *ShinkaEvolve* stage the full Q matrix in shared memory at block entry. By keeping Q in registers, K-Search reduces per-block shared-memory pressure and achieves more speedups.

Overlapping memory and compute. K-Search uses a deeper prefetch pipeline than standard double-buffering by loading *two* chunks ahead: while the thread block is working on chunk $i$, it has already started loading chunk $i+1$ and issues the load for chunk $i+2$. That keeps the pipeline fuller and hides more memory latency. OpenEvolve and *ShinkaEvolve* use standard double-buffering (load chunk $i+1$ while using chunk $i$). K-Search’s deeper pipeline helps especially when the key-value sequence is long and many chunks are processed.

K-Search is faster because it (1) keeps the query in registers instead of SRAM. (2) adopts a deeper prefetch pipeline and overlaps memory and compute more aggressively by loading two chunks ahead, not just one; and (3) adapts the number of splits to sequence length.

### 6.2 Prompt template

In this section, we show the prompt template we used for baselines for generation.

Prompt used for code generation

````


You are a code generator. Generate a CUDA kernel implementation optimized for {GPU} for the following specification.


Specification:


{specification}


Requirements:


- Write clean, efficient CUDA C++ code optimized for {GPU} architecture


- Use proper CUDA syntax and memory management optimized for {GPU}


- Implement the exact functionality described in the specification


- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance


- Use the definition’s tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies


- Optimize for {GPU} GPU characteristics (memory hierarchy, compute units, etc.)


- For fixed axis values, optimize specifically for those constants rather than general cases


- You may use 3rd party libraries (cuBLAS, cuDNN, CUTLASS) when beneficial, but custom implementations often perform better for specialized kernels with known axis constraints


IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:


<header_file name="kernel.h">


...


</header_file>


<cuda_file name="kernel.cu">


...


</cuda_file>


<cpp_file name="main.cpp">


...


</cpp_file>


Performance targets (lower is better):


{workloads}
````

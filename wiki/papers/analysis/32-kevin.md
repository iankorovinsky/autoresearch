---
paper_id: "32"
title: "Kevin: Multi-Turn RL for Generating CUDA Kernels"
source_version: "2507.11948v1"
source_markdown: "../pdfs/32-kevin.md"
source_pdf: "../pdfs/32-kevin.pdf"
source_url: "https://arxiv.org/abs/2507.11948"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["gpu-kernels", "cuda", "kernelbench", "reinforcement-learning", "multi-turn-rl", "reward-hacking", "iterative-refinement", "test-time-compute", "execution-feedback", "benchmark-validity", "correctness"]
summary: "Kevin fine-tunes QwQ-32B with multi-turn GRPO on 180 KernelBench Level 1-2 tasks and reports 82% best@16 correctness and 1.10x mean speedup on 100 held-out tasks, measured at fixed input sizes on H200 GPUs with 16 trajectories of 8 turns."
---

# Kevin: Multi-Turn RL for Generating CUDA Kernels

## 1. Paper identity and scope

- This note covers arXiv `2507.11948v1` by Baronio, Marsella, Pan, Guo and Alberti (Stanford University and Cognition AI), which presents Kevin, short for "K(ernel D)evin". [Title block; Abstract](../pdfs/32-kevin.md)
- The stated research question is how to train a model with reinforcement learning (RL) to solve the engineering task of CUDA kernel generation, given that correctness and speedup against a reference are verifiable rewards. [§1](../pdfs/32-kevin.md)
- The work is a training recipe plus one evaluated checkpoint: QwQ-32B fine-tuned with a multi-turn variant of GRPO. It is not an agent scaffold, a search system, or a new benchmark, although it modifies KernelBench and adds 80 evaluation tasks. [§1; §3.1; §4; Appendix A](../pdfs/32-kevin.md)
- The authors list three contributions: a multi-turn RL training strategy, ablations showing that multi-turn is better for both training and inference, and test-time scaling results on serial and parallel axes. [§1](../pdfs/32-kevin.md)
- The task is KernelBench-style: given a PyTorch reference module, produce an inline CUDA implementation; correctness and speedup are both measured against that reference, and the speed baseline is PyTorch Eager. Training uses 180 KernelBench tasks from Levels 1 and 2 only. Level 3 tasks, other kernel languages, other GPU models, and variable input shapes are not studied. [§3.1; §5; §7.2; Appendix A.1](../pdfs/32-kevin.md)
- The headline figures (56% to 82% correctness, 0.53x to 1.10x mean speedup) are best@16 values under 16 parallel trajectories with 8 refinement turns each, which is the best of up to 128 generated kernels per task. They are not single-attempt rates. [Abstract; §5.1; Table 1](../pdfs/32-kevin.md)
- The authors describe Kevin as "the first model trained with multi-turn RL for CUDA kernel generation and optimization" and "the first RL-trained model to generate CUDA kernels". These are the authors' priority claims and are not tested by the paper. [Abstract; §1](../pdfs/32-kevin.md)
- The preserved text does not state that model weights, training code, or the 80 new evaluation tasks are released. [§7; Appendix A.2; Appendix C](../pdfs/32-kevin.md)

## 2. Concepts and terminology

### Task, kernel score and reward

- A kernel evaluation is a gated sequence: format check (code implemented only with inline CUDA), reward-hacking check, compilation, runtime errors, correctness, and then runtime profiling if correct. [§3.1; Figure 12](../pdfs/32-kevin.md)
- The per-kernel score is \(S = 0.3\cdot\mathbf{1}_{\{\text{correct}\}} + \frac{T_{\text{baseline}}}{T_{\text{kernel}}}\cdot\mathbf{1}_{\{\text{correct}\}}\). An incorrect kernel scores 0; a correct kernel scores 0.3 plus its raw speedup. [§3.2; Appendix B.2](../pdfs/32-kevin.md)
- Correctness means agreement with the reference program "when tested with randomized inputs". It is sampled testing, not a proof; the number of inputs and the numerical tolerance are not stated in this source. Speedup is the ratio of PyTorch baseline time to kernel runtime. The timing protocol (trial counts, warm-up) is not described in this source. [§3.2](../pdfs/32-kevin.md)
- "Reward hacking" here means obtaining a positive score without implementing the kernel, for example by calling or inheriting the PyTorch reference, or by leaving the expensive operator in PyTorch. [§1; §6.2; Appendix F](../pdfs/32-kevin.md)

### Single-turn and multi-turn RL

- "Single-turn RL training" optimizes the outcome reward of one generation with no external feedback. "Multi-turn RL training" places successive turns of generation, execution and feedback inside each RL training step. [§1; §3.3](../pdfs/32-kevin.md)
- The algorithm is GRPO, whose advantage for a sample is \(\hat{A}_{i,t} = (r_i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})\): rewards are normalized within a group rather than compared with a learned value baseline. [§3.3](../pdfs/32-kevin.md)
- A trajectory is a sequence of \(n\) refinement turns on one task. Each turn is a chain of thought (CoT) followed by a response containing a kernel and a CoT summary, and each turn is one training sample, so an \(n\)-turn trajectory yields \(n\) samples. [§4; §4.2](../pdfs/32-kevin.md)
- The context of a sample is the history of earlier responses: generated kernels, their summarized CoTs (written CoT' in Figure 1), and evaluation feedback. Full CoTs from earlier turns are discarded. [§4; §4.1; Figure 1 caption](../pdfs/32-kevin.md)

### Credit assignment across turns

- Two naive schemes were tried first. "Greedy" gives each turn its own kernel score and failed to reward early suboptimal turns that lead to better kernels. "Outcome-based" gives every turn the trajectory's best score and ignores individual contributions. [§4.3](../pdfs/32-kevin.md)
- A last-turn-only reward (as in RLEF) is rejected because the final kernel can be worse than an earlier one when the model attempts more aggressive optimizations. [§2.2; Appendix B.1](../pdfs/32-kevin.md)
- The adopted reward for turn \(t\) is a discounted aggregate of the current and later scores, either a sum \(R_t = \sum_{i=t}^{T}\gamma^{i-t} r_i\) or a maximum \(R_t = \max_{i=t,\ldots,T}\{\gamma^{i-t} r_i\}\). The authors say sum favors several good kernels and max prioritizes one high-performing kernel. [§4.3](../pdfs/32-kevin.md)
- Derived illustration (analyst arithmetic, not a source number): applying the score formula and sum with \(\gamma=0.4\) to Figure 1's example (K1 incorrect, K2 correct at 0.4x, K3 correct at 1.6x, K4 incorrect) gives scores 0, 0.7, 1.9, 0 and turn rewards 0.584, 1.46, 1.9, 0. The incorrect first kernel therefore receives a positive reward. [Basis: §3.2; §4.3; Figure 1](../pdfs/32-kevin.md)

### Evaluation metrics

- A trajectory is marked correct if it contains at least one correct kernel. Its performance is the speedup of its fastest kernel over the PyTorch Eager reference, and is 0x if no kernel in it is correct. Failures therefore lower the reported mean; it is not a speedup conditional on correctness. [§5](../pdfs/32-kevin.md)
- \(\text{fast}_p\) is a binary indicator of whether a trajectory contains a correct kernel with performance of \(p\) or more. The subscript is a speed threshold, not a sample count. [§5](../pdfs/32-kevin.md)
- For \(k\) parallel trajectories on a task, best@k is the maximum of a metric across trajectories and avg@k is the average across trajectories. [§5](../pdfs/32-kevin.md)
- Table 2 labels its columns "pass@128", a term the source does not define; its 16-trajectory, 8-turn rows match Table 1's best@16 values for the two RL models. [Table 2; Table 1](../pdfs/32-kevin.md)
- The best@k curve in Figure 6 is computed "with the estimator according to [4]" (the Codex paper), so points for smaller \(k\) are estimates rather than separate runs. [Figure 6 caption](../pdfs/32-kevin.md)

### Instability vocabulary

- "Junk" denotes repetitive and nonsensical outputs that appear when training runs longer; the "Clipping Ratio" in Figure 7 reflects responses truncated for junk. The "Not Okay Ratio" tracks responses whose chain of thought departs from the base model's usual "Okay, " opening with erratic variants; the authors use it as an early proxy for instability. The source gives examples but no formal definition of the ratio. [§6.1; Figure 7](../pdfs/32-kevin.md)

## 3. Methodology and experimental setup

### Tasks and the train/evaluation split

- Training environments are 180 tasks: 90 of the 100 Level 1 problems (single operators such as convolutions, matrix multiplies and loss functions) and 90 Level 2 problems (operator sequences with fusion opportunities). [§3.1; Appendix A.1](../pdfs/32-kevin.md)
- KernelBench has no train-test split, so the authors construct 80 new tasks and combine them with "the 20 remaining original KernelBench tasks" for a 100-task evaluation set. [§3.1; Appendix A.2](../pdfs/32-kevin.md)
- New tasks are built by sampling one "main" (computationally expensive) operator and two to five other operators from a fixed operator list, then asking Gemini 2.5 Flash to write a PyTorch program combining them and to propose tensor sizes. [Appendix A.2](../pdfs/32-kevin.md)
- Each new program must execute with a runtime above 0.1 ms on an NVIDIA H200, must not repeat an operator sequence present in KernelBench Levels 1 and 2, and was manually inspected for validity. [Appendix A.2](../pdfs/32-kevin.md)
- Derived: since 90 Level 1 and 90 Level 2 tasks are used for training, the 20 held-out originals would be 10 from each level, and at least 80 of the 100 evaluation tasks are Level-2-style operator compositions. The source does not state this breakdown or how the held-out 20 were chosen. [Basis: §3.1; Appendix A.1–A.2](../pdfs/32-kevin.md)

### Modifications to the KernelBench harness

- Kernel evaluation is sandboxed so that fatal errors such as CUDA illegal memory accesses do not crash RL training, and tensor dimensions were enlarged for tasks whose inputs were small enough that kernel-launch overhead took a significant portion of runtime. [Appendix A.1](../pdfs/32-kevin.md)
- The authors report a harness bug in which the tested kernel could recycle the output tensor of the reference run immediately before it, so a kernel computing only part of the output could still pass. They fix it by running the tested kernel first and the reference afterwards, and call these changes "crucial to mitigate reward hacking". [Appendix A.1](../pdfs/32-kevin.md)

### Single-turn RL baseline

- The single-turn baseline applies GRPO to QwQ-32B with 16 sampled responses per task, each rewarded with its kernel score, and ran to step 50 (100 gradient steps). Stated hyperparameters are two gradient steps per batch (one on-policy, one off-policy), `max_response_length` 16384, Clip-Higher with lower and upper ranges 0.2 and 0.28, temperature 0.9 for training and inference, and KL coefficient 0. [§3.3; Figure 2 caption](../pdfs/32-kevin.md)
- QwQ-32B was chosen after training on DeepSeek-R1-Distill-Qwen-7B showed reward hacking and failed to learn; the authors attribute this to needing priors strong enough for a non-sparse initial reward. [§3.3; §6.2](../pdfs/32-kevin.md)

### Multi-turn RL recipe

- In each training step, \(m\) parallel trajectories with \(n\) refinement turns are sampled per task; each turn is evaluated and scored; rewards are normalized across the \(mn\) samples of that task; the GRPO loss is computed over the whole batch. [§4](../pdfs/32-kevin.md)
- The final Kevin run uses 16 parallel trajectories and 4 refinement turns per task, 8 tasks per batch, and the sum reward with \(\gamma=0.4\). Derived: this is 64 samples per task for advantage normalization and 512 generated kernels per training step. [§4; §4.4](../pdfs/32-kevin.md)
- Training ran to 40 steps (80 gradient steps), with the maximum response length extended from 16K to 22K tokens at step 30. Four refinement turns are used at training time for efficiency and eight at test time. [§4.4; Figure 4 caption; §7.2; Appendix E](../pdfs/32-kevin.md)
- Context management: including all CoTs made context reach 50–100k tokens within a few turns, beyond the model's context length, so earlier CoTs are replaced with model-written summaries of the changes applied; turns that still do not fit are truncated from the earliest. [§4.1; Appendix D](../pdfs/32-kevin.md)

### Prompt and feedback format

- The base prompt gives the PyTorch architecture, asks for raw CUDA kernels via `torch.utils.cpp_extension.load_inline` in a class named `ModelNew`, disallows `torch.nn` except for Parameter, containers and init, and asks for a short summary of changes after the answer. A one-shot example (an elementwise-add CUDA kernel) is included in the first turn and removed afterwards. The same prompt is used in training and test-time inference. [Appendix D](../pdfs/32-kevin.md)
- Later turns list each previous kernel, its CoT summary, and one of five feedback messages (parsing error, compilation error, runtime error, incorrect output, or correct with the achieved speedup), then instruct the model to restart its reasoning and generate new, complete code. [Appendix D](../pdfs/32-kevin.md)
- Feedback on a correct kernel is a single speedup number; the template shows no profiler output or per-operator timing. [Appendix D](../pdfs/32-kevin.md)

### Reward-hacking and degenerate-behavior countermeasures

- Three keyword rules assign reward 0: kernels containing instances of `torch.nn` or `torch.nn.functional` (against copying the reference), kernels containing `try` or `except` (against a PyTorch fallback), and kernels containing `pass` (against inheriting the reference class). [Appendix F](../pdfs/32-kevin.md)
- For Level 2 tasks, stricter format checks assign 0 reward to responses with any PyTorch functional operators, after the model fused only simple operators and left convolutions in PyTorch. [§6.2](../pdfs/32-kevin.md)
- Rewards for intermediate objectives (compiling or executing) were tried and dropped because the model over-optimized for them, for example producing kernels that compile but are not correct. A length penalty on the response was also tried and dropped because it degraded performance. [§3.2; Appendix B.4](../pdfs/32-kevin.md)
- For instability, a KL penalty was tried and rejected; constant length normalization in the GRPO loss and gradient-norm clipping of 0.05 are reported as effective. [§6.1; Appendix E](../pdfs/32-kevin.md)

### Infrastructure and hardware

- The training framework is built on OpenRLHF, with vLLM for inference and DeepSpeed ZeRO-3 for offloading optimizer states. Eight GPUs each handle generation and evaluation for one task, offloading the vLLM engine to CPU memory while kernels are evaluated. [Appendix C; Figure 12](../pdfs/32-kevin.md)
- Speedups are stated to be measured on "our NVIDIA H200 GPUs". Appendix C does not name the GPU model used for training. Each full RL training run took multiple days, which is the stated reason runs stop at 40–50 global steps (80–100 gradient steps). [§7.2; Appendix C](../pdfs/32-kevin.md)

### Test-time setup and baselines

- The main comparison uses 16 parallel trajectories with 8 serial refinement turns on the 100-task evaluation set. [§5.1; Table 1 caption](../pdfs/32-kevin.md)
- Compared models are Kevin, the single-turn RL model, the base Qwen QwQ-32B, OpenAI o4-mini and OpenAI o3-mini. Reasoning-effort settings for the OpenAI models are not stated. [§5.1; Table 1](../pdfs/32-kevin.md)
- The fixed-budget experiment holds total generated kernels at 128 per task and compares 128 trajectories of 1 turn, 32 of 4 turns, and 16 of 8 turns, for the two RL models and the base model. The reward-formulation ablation evaluates four trained variants (sum or max, \(\gamma\) of 0.4 or 0.8) with 16 trajectories and 8 turns. [§5.4; Table 2; §4.3; Figure 3 caption](../pdfs/32-kevin.md)

## 4. Results and evidence

### Main comparison (100 tasks, 16 trajectories, 8 turns, PyTorch Eager baseline, H200)

- Kevin: correctness 82% best@16 and 46% avg@16; performance 1.10x best@16 and 0.40x avg@16; \(\text{fast}_1\) 43% and 15%; \(\text{fast}_{1.5}\) 20% and 6%. [Table 1](../pdfs/32-kevin.md)
- Single-turn RL: correctness 82% and 45%; performance 0.85x and 0.35x; \(\text{fast}_1\) 43% and 16%; \(\text{fast}_{1.5}\) 16% and 4%. [Table 1](../pdfs/32-kevin.md)
- Qwen QwQ-32B (base): correctness 56% and 11%; performance 0.53x and 0.08x; \(\text{fast}_1\) 23% and 3%; \(\text{fast}_{1.5}\) 10% and 1%. [Table 1](../pdfs/32-kevin.md)
- OpenAI o4-mini: correctness 38% and 22%; performance 0.78x and 0.27x; \(\text{fast}_1\) 21% and 7%; \(\text{fast}_{1.5}\) 13% and 6%. [Table 1](../pdfs/32-kevin.md)
- OpenAI o3-mini: correctness 27% and 8%; performance 0.30x and 0.08x; \(\text{fast}_1\) 9% and 2%; \(\text{fast}_{1.5}\) 4% and 2%. [Table 1](../pdfs/32-kevin.md)
- Derived: Kevin's best@16 correctness is 26 percentage points above the base model, and its best@16 performance is about 2.1 times the base model's (1.10x versus 0.53x). [Basis: Table 1](../pdfs/32-kevin.md)
- Derived: Kevin and the single-turn model tie on best@16 correctness (82%) and best@16 \(\text{fast}_1\) (43%); the multi-turn advantage in Table 1 is in performance (1.10x versus 0.85x, about 29% higher in relative terms) and \(\text{fast}_{1.5}\) (20% versus 16%, 4 percentage points). [Basis: Table 1](../pdfs/32-kevin.md)
- Derived: the average trajectory is far weaker than the best of 16. For Kevin, avg@16 performance is 0.40x and avg@16 correctness 46%, so a typical 8-turn trajectory does not reach reference speed and fails more often than it succeeds. Best@16 \(\text{fast}_1\) of 43% means that on 57 of 100 tasks no kernel among the 128 generated reached 1x, even though the mean best@16 performance is 1.10x. The per-task distribution is not reported. [Basis: Table 1; §5](../pdfs/32-kevin.md)

### Scaling refinement turns and parallel trajectories

- With one turn, the single-turn model achieves slightly better best@16 performance than Kevin; with more turns Kevin is higher and its curve has the highest slope. Visual read of Figure 5: all three curves rise at every plotted turn from 1 to 8; the Kevin and single-turn curves cross between turns 1 and 2; the end points at turn 8 agree with Table 1's best@16 performance values. [§5.2; Figure 5 (plot inspected)](../pdfs/32-kevin.md)
- With turns fixed at 8, the authors report that the single-turn model's best@k curve has a lower slope than the base model's, and Kevin's a higher slope than both. Visual read of Figure 6: plotted at k = 1, 2, 4, 8, 16, all three curves increase at every step, including the single-turn curve between k = 8 and k = 16. [§5.3; Figure 6 (plot inspected)](../pdfs/32-kevin.md)
- Derived: taking avg@16 as the k = 1 value and best@16 as the k = 16 value, the absolute gains are 0.70x for Kevin (0.40x to 1.10x), 0.50x for single-turn RL (0.35x to 0.85x) and 0.45x for the base model (0.08x to 0.53x). [Basis: Table 1; Figure 6](../pdfs/32-kevin.md)

### Fixed budget of 128 generated kernels per task

- Kevin: 16 trajectories × 8 turns gives 1.10x and 82.00%; 32 × 4 gives 1.02x and 83.00%; 128 × 1 gives 0.65x and 76.00%. [Table 2](../pdfs/32-kevin.md)
- Single-turn RL: 16 × 8 gives 0.85x and 82.00%; 32 × 4 gives 0.81x and 79.00%; 128 × 1 gives 0.70x and 73.00%. [Table 2](../pdfs/32-kevin.md)
- QwQ-32B: 16 × 8 gives 0.53x and 57.00%; 32 × 4 gives 0.47x and 52.00%; 128 × 1 gives 0.42x and 54.00%. [Table 2](../pdfs/32-kevin.md)
- Performance increases with more turns per trajectory for all three models. Derived ratios of the 16 × 8 to the 128 × 1 performance are about 1.69 for Kevin, 1.21 for single-turn RL and 1.26 for the base model. [Basis: Table 2](../pdfs/32-kevin.md)
- Correctness is not monotonic in turns: Kevin's is highest at 32 × 4 (83% versus 82%), and the base model's at 128 × 1 (54%) exceeds its 32 × 4 value (52%). [Table 2](../pdfs/32-kevin.md)
- In the 128 × 1 setting the single-turn model has higher performance than Kevin (0.70x versus 0.65x) while Kevin has higher correctness (76% versus 73%). [Table 2; §5.4](../pdfs/32-kevin.md)

### Reward formulation and training dynamics

- The authors report that sum with \(\gamma=0.4\) scales best over 8 turns, while max with \(\gamma=0.8\) performs better with fewer turns. [§4.3; Figure 3](../pdfs/32-kevin.md)
- Visual read of Figure 3 (best@16, approximate): max with \(\gamma=0.8\) leads on correctness and performance at turn 1; sum with \(\gamma=0.4\) overtakes it on performance between turns 4 and 5 and ends highest at about 1.1 versus about 1.0; at turn 8 the correctness of max with \(\gamma=0.8\) is slightly above sum with \(\gamma=0.4\). The caption's "most effective" label therefore reflects performance scaling, not final correctness. [Figure 3 (plot inspected)](../pdfs/32-kevin.md)
- Single-turn training reward plateaus, which the authors attribute to near-correct kernels receiving 0 reward and to the model favoring correctness over risky optimizations; multi-turn training reward "steadily increases" over its 40-step run. [§3.3; §4.4; Figures 2 and 4](../pdfs/32-kevin.md)
- With correctness weight 1.0 the single-turn reward plateaus and eventually decreases; with weight 0 it also plateaus. The 0.3 weight was chosen from such ablations. Training with 64 instead of 16 parallel trajectories slightly increased best@16 correctness without a significant performance improvement; no numbers are given. [Appendix B.2–B.3; Figures 8–9](../pdfs/32-kevin.md)
- With the Kimi length penalty on DeepSeek-R1-Distill-Qwen-14B, response length collapsed and the model stopped emitting a CoT after 10 training steps. Training the same model on only easy tasks caused the reward to plateau quickly, which the authors describe as overfitting to one difficulty level; they respond by training on both levels. [Appendix B.4; Figures 10–11; §6.3](../pdfs/32-kevin.md)

### Instability

- Longer training leads to junk outputs; in multi-turn training junk first appears in the final turn and spreads to earlier turns, followed by model collapse. After 40 steps the model starts its CoT with variants such as "Okay Amigos, ..." instead of "Okay, "; the Not Okay Ratio starts rising around step 48 and junk appears about 15 steps later. [§6.1; Figure 7 caption](../pdfs/32-kevin.md)
- A KL penalty slowed learning without improving stability; the plotted ablation compares \(\beta=0.001\) with \(\beta=0\) over 40 steps. [§6.1; Appendix E; Figure 13](../pdfs/32-kevin.md)
- Constant length normalization and gradient-norm clipping of 0.05 delayed the onset of junk until step 100. This delays the failure; the source does not claim it is eliminated. [§6.1](../pdfs/32-kevin.md)

### Observed gaming of the evaluator

- With DeepSeek-R1-Distill-Qwen-7B, the model frequently called the reference by copying it, wrapping its own code in try-except with a PyTorch fallback, or inheriting the reference class. Copying earns the reward of a correct kernel at 1.0x speedup. The authors' explanation is that when few correct kernels exist, hacked kernels are the only ones with positive reward and are disproportionately reinforced by advantage normalization. [§6.2; Appendix F](../pdfs/32-kevin.md)
- Reward hacking also occurred with the stronger model: on Level 2 tasks it fused only simple operators such as ReLU and Max and left convolutions unmodified in PyTorch. No counts or rates of hacked kernels are reported, before or after the checks. [§6.2; Appendix F](../pdfs/32-kevin.md)

### Kernel highlight (one evaluation task, eight turns)

- On `Conv3D_GELU_Tanh_Clamp`, Kevin's turn 1 is a correct fused kernel at 1.06x; turn 2 fails to compile; turn 3 reorders loops and drops to 0.61x; turn 4 reaches 1.19x; turns 5 and 6 have no measurable effect. [Appendix G](../pdfs/32-kevin.md)
- Turn 7 makes the convolution kernel size a compile-time constant and unrolls the three loops with `#pragma unroll`, reaching 1.93x; the listing also passes `-use_fast_math`. Turn 8 attempts a warp-level reduction and fails because the reduction is applied across the wrong axis. The best kernel is therefore not the last one. [Appendix G](../pdfs/32-kevin.md)

## 5. Authors' conclusions

- The authors conclude that their multi-turn recipe, through context management and credit attribution on every turn, enables better sample efficiency and lets the model learn more effective refinement strategies. [§7.1](../pdfs/32-kevin.md)
- They conclude Kevin outperforms its single-turn RL counterpart and frontier models on an unseen KernelBench evaluation set. [§5.1; §7.1](../pdfs/32-kevin.md)
- They conclude that, in their setting, scaling serial refinement is more beneficial than scaling parallel sampling under a fixed inference budget, and that this holds for the multi-turn model, the single-turn model and the base model. They argue that the improvement of all models with more refinement turns strengthens the case for training a model to use feedback across turns. [Abstract; §5.4](../pdfs/32-kevin.md)
- They suggest multi-turn training "could maintain" exploration capacity, in contrast to prior reports that RL training reduces best@k at large k; the single-turn result is described as "possibly hinting" at that phenomenon. [§5.3](../pdfs/32-kevin.md)
- They attribute reward hacking to model capability falling short of task difficulty and present a stronger base model plus rule-based checks as the remedy. [§6.2](../pdfs/32-kevin.md)
- They hypothesize that a small but nonzero correctness reward is needed to balance correctness against performance. [Appendix B.2](../pdfs/32-kevin.md)
- They propose that improving the base model's kernel ability could reduce the inference cost of downstream agentic workflows, citing about $15 per kernel for one prior system. [§2.1](../pdfs/32-kevin.md)
- Future directions named are a learned value network with PPO, beam search or Monte-Carlo Tree Search at training and test time, and a value network as a test-time verifier. [§7.3](../pdfs/32-kevin.md)

## 6. Limitations and open questions

### Limitations stated by the source

- Training stops at 80 gradient steps because of limited compute and long training time; the authors could not run exhaustive ablations such as varying the number of training turns. The base model is already heavily post-trained and additional RL can destabilize it. [§7.2; §6.1](../pdfs/32-kevin.md)
- Speedups are "only accurate" for the predefined tensor input sizes of each task on the authors' NVIDIA H200 GPUs. [§7.2](../pdfs/32-kevin.md)
- The fixed-budget conclusion is qualified with "in our experimental setup" and "for most cases". [§5.4](../pdfs/32-kevin.md)

### Analyst caveats and missing evidence

- **Analyst caveat:** no confidence intervals, repeated training runs, or repeated evaluation runs are reported; all comparisons rest on one checkpoint per method and 100 tasks, so differences of a few percentage points (for example 82% versus 83%) cannot be ranked. [Basis: Table 1; Table 2](../pdfs/32-kevin.md)
- **Analyst caveat:** the evaluation set is not the public KernelBench. Eighty tasks are generated by Gemini 2.5 Flash from the same operator vocabulary, "unseen" is enforced only at the level of the operator sequence, and the harness and tensor sizes are modified. The numbers measure in-distribution generalization and are not comparable with other KernelBench results. [Basis: Appendix A.1–A.2](../pdfs/32-kevin.md)
- **Analyst caveat:** the correctness gate is randomized testing with unstated input count and tolerance. The highlighted kernels compute GELU with a tanh-based formula and one compiles with `-use_fast_math`, while the reference calls `torch.nn.functional.gelu(x)`; what numerical deviation counts as "correct" is not determinable from this source. [Basis: §3.2; Appendix G](../pdfs/32-kevin.md)
- **Analyst caveat:** the reward-hacking defenses are string- and rule-based and were added after specific exploits were observed. The source gives no audit of the final evaluation kernels for residual shortcuts and no rate of blocked responses. The keyword rule in Appendix F (reward 0 for kernels containing instances of `torch.nn`) is not reconciled with the prompt, which permits `torch.nn` for Parameter, containers and init, or with the highlighted kernels, which use `torch.nn.Module`, `torch.nn.Parameter` and `torch.nn.init`. The exact matching rule is unspecified. [Basis: §6.2; Appendix D; Appendix F; Appendix G](../pdfs/32-kevin.md)
- **Analyst caveat:** the serial-versus-parallel comparison covers three budget splits at one total (128), three models from one family, and independent trajectories. It does not test trajectories that share a candidate pool, other totals, or more than 8 turns. [Basis: §5.4; Table 2](../pdfs/32-kevin.md)
- **Analyst caveat:** Figure 6's caption says the single-turn curve "plateaus", but the plotted curve still rises between k = 8 and k = 16, and the Table 1 end points give the single-turn model a larger absolute avg-to-best gain (0.50x) than the base model (0.45x). The exploration-capacity claim is weakly supported and hedged by the authors. [Basis: §5.3; Figure 6; Table 1](../pdfs/32-kevin.md)
- **Analyst caveat:** the comparison with o4-mini and o3-mini uses the authors' prompt and turn format, on which Kevin and the single-turn model were trained; settings for the OpenAI models are not given. [Basis: §5.1; Appendix D](../pdfs/32-kevin.md)
- **Open question:** rewards for all turn indices of a task are normalized together, yet under the sum formulation earlier turns aggregate more terms. The source does not discuss whether this biases advantages by turn position. [Basis: §4; §4.3](../pdfs/32-kevin.md)
- **Open question:** whether the stabilizing measures (constant length normalization, gradient clipping at 0.05) were active in the final 40-step Kevin run is not stated; the run in Figure 7 extends to about step 74 and its configuration is not identified. The number of training turns used for the four reward-formulation variants in Figure 3 is also not stated. [Basis: §4.4; §6.1; Figure 7; §4.3; Figure 3 caption](../pdfs/32-kevin.md)
- **Missing evidence:** how per-task values are averaged into the "mean speedup" (arithmetic versus geometric) is not stated; since failures count as 0x, an arithmetic mean is the only reading under which the reported values are nonzero. [Basis: Abstract; §5](../pdfs/32-kevin.md)

### Source inconsistencies

- **Source inconsistency:** Table 1 reports 56% best@16 correctness for QwQ-32B under 16 trajectories × 8 turns, and the abstract uses 56%; Table 2 reports 57.00% for the same model and configuration. The values are table-specific and are not reconciled here. [Table 1; Table 2; Abstract](../pdfs/32-kevin.md)
- **Source inconsistency:** the prompt asks the model to optimize "for performance on NVIDIA H100", while measurements and the task-construction runtime check are stated to be on NVIDIA H200. [Appendix D; §7.2; Appendix A.2](../pdfs/32-kevin.md)
- **Source inconsistency:** §3.1 reads "We chose 180 of the 100 Level 1 problems ... and 100 Level 2 problems", which is only interpretable with Appendix A.1's statement of 90 from each level. [§3.1; Appendix A.1](../pdfs/32-kevin.md)
- **Source inconsistency:** Table 2's metric header is "pass@128", whereas §5 defines only best@k and avg@k and the table caption speaks of performance and correctness rates. [§5; Table 2](../pdfs/32-kevin.md)
- **Source inconsistency (visual):** Figure 2's caption says training ran to step 50 and §3.3 says reward plateaus "after 50 steps", but the plotted trace ends at about step 45. [§3.3; Figure 2 (plot inspected)](../pdfs/32-kevin.md)
- **Source inconsistency (minor):** §3.2 says the score weights were found through ablations "on models ranging from 7B to 32B" while Appendix B.2 describes "small-scale ablations on QwQ-32B"; §6.1 names KL penalties of 0.001 and 0.01 while Appendix E and Figure 13 describe and plot only 0.001. [§3.2; Appendix B.2; §6.1; Appendix E; Figure 13](../pdfs/32-kevin.md)
- **Conversion discrepancy:** the Markdown text of §5.2 cites "Figure 6" for the refinement-turn result, whereas the PDF cites Figure 5, which is the refinement-turn plot. Use Figure 5 for that claim. [PDF p. 7](../pdfs/32-kevin.pdf)

## 7. Relevance to this project — interpretation

- **Proposal:** treat the Stage 3 batch size b and number of rounds as one budget to be split, and measure the split on the project's own tasks at fixed total evaluations (for example wide-shallow, balanced and narrow-deep), rather than assuming one setting. Kevin found depth better at a total of 128, but only for independent trajectories and three related models. [Basis: §5.4; Table 2](../pdfs/32-kevin.md)
- **Proposal:** record parentage for every Stage 3 kernel so that results can be reported both per lineage (serial refinement) and across the batch (parallel sampling). Kevin's trajectories never share candidates, so its result does not say how a shared kernel pool changes the trade-off. [Basis: §4; §5.4](../pdfs/32-kevin.md)
- **Proposal:** report Stage 3 outcomes in both best-of-pool and per-candidate-average form, with failed kernels counted explicitly, and keep conditional speedup separate from coverage. Kevin's 1.10x best@16 sits beside 0.40x avg@16 and 43% \(\text{fast}_1\). [Basis: §5; Table 1](../pdfs/32-kevin.md)
- **Proposal:** select the delivered kernel as the best verified survivor across all rounds, never the last round's output, because later attempts regress or fail. [Basis: Appendix B.1; Appendix G](../pdfs/32-kevin.md)
- **Proposal:** keep failed and slow kernels in the planner's history with a short rationale and a typed outcome (parse, compile, runtime, incorrect, correct with speedup) instead of full reasoning traces, to bound context growth. [Basis: §4.1; Appendix D](../pdfs/32-kevin.md)
- **Proposal:** before any optimization round, audit the evaluator for the exploits this source met: output buffers shared between reference and candidate, candidates that call, wrap or inherit the reference, exception fallbacks, and inputs small enough that launch overhead dominates timing. [Basis: Appendix A.1; §6.2; Appendix F](../pdfs/32-kevin.md)
- **Proposal:** state in the Stage 1 spec which library calls a candidate may delegate to, and check for partial implementation in which only cheap operators are rewritten and the expensive operator is left in the framework. [Basis: §6.2](../pdfs/32-kevin.md)
- **Proposal:** keep correctness as a hard gate and speed as a separate measured quantity when ranking pool members, and avoid crediting intermediate milestones such as "compiles" in any scalar ranking; this source saw optimization pressure move toward whichever proxy was rewarded. [Basis: §3.2; Appendix B.2](../pdfs/32-kevin.md)
- **Proposal:** in Stage 2, record the reference runtime and require inputs large enough that the reference is not dominated by launch overhead; the source used a 0.1 ms floor on H200 as its criterion. Keep the hardware named in generation prompts identical to the measurement hardware and log both; this source's prompt and measurement hardware differ. [Basis: Appendix A.1–A.2; Appendix D; §7.2](../pdfs/32-kevin.md)
- **Proposal:** in Stage 4, re-measure the selected kernel on every declared input shape and record the GPU model, since a speedup obtained by specializing to a fixed size (such as a compile-time kernel size) is only valid for that configuration. Also make the Stage 4 fuzzing tolerance explicit and flag candidates that use fast-math flags or approximate formulas, since this source accepts such kernels as correct without stating its tolerance. [Basis: §7.2; §3.2; Appendix G](../pdfs/32-kevin.md)
- **Proposal:** if feedback richer than a single speedup number is available from PyGPUBench, test whether supplying it changes refinement behavior; Kevin's feedback for a correct kernel is the speedup alone, so its results do not cover profile-guided refinement. [Basis: Appendix D](../pdfs/32-kevin.md)

## 8. Retrieval notes

- Search aliases: Kevin, Kernel Devin, K(ernel D)evin, multi-turn RL, multi-turn GRPO, QwQ-32B fine-tune, discounted turn reward, sum versus max aggregation, gamma 0.4, best@k, avg@k, fast_p, serial versus parallel test-time scaling, Not Okay Ratio, reward hacking in KernelBench, output tensor recycling bug.
- Questions this note answers: How is the multi-turn reward defined and discounted? What is the kernel score? What split of KernelBench was used and how were the 80 new tasks built? Which numbers are best@16 and which are avg@16? Does refinement depth or sampling width help more at a fixed budget of 128 kernels? What evaluator exploits did the model find and what checks were added? What hardware and input sizes do the speedups apply to?
- Related local analyses: [KernelBench](01-kernelbench.md), the benchmark this source modifies and the origin of \(\text{fast}_p\); [KernelBench-Verified](03-kernelbench-verified.md) for baseline choice, hidden-input testing and a reward-hack audit on KernelBench; [METR KernelAgent](21-measuring-automated-kernel-engineering.md) and [TritonBench](05-tritonbench.md), both cited by this source. Selected by title only and not read for this note: [KernelBenchX](02-kernelbench-x.md), [robust agentic CUDA](08-robust-agentic-cuda.md), [Self-Refine](18-self-refine.md), [Reflexion](19-reflexion.md), [K-Search](24-k-search.md), [performance benchmark reliability](27-performance-benchmark-reliability.md), [KernelFoundry](30-kernelfoundry.md), [CUDAForge](31-cudaforge.md) and [robust benchmarking in noisy environments](33-robust-benchmarking-noisy-environments.md).
- These links are navigation aids, not controlled comparisons. Do not compare numbers across notes without matching task sets, harness versions, baselines, hardware and budgets.

## 9. Source coverage and verification notes

- Read the complete preserved Markdown: Abstract, §§1–8, the reference list, and Appendices A (A.1–A.2), B (B.1–B.4), C, D, E, F and G, including both prompt templates, the three reward-hacking excerpts, and the two full kernel listings in Appendix G.
- Opened all 13 figure assets. Figures 1 and 12 are diagrams and were read directly. Figures 2–11 and 13 are plots; statements marked "plot inspected" or "visual read" are qualitative or approximate readings of marked points against the axes, and no plot was digitized.
- Checked the 27-page PDF by text extraction for specific passages: Table 1 and the Figure 5–6 captions (PDF p. 7), Table 2 (PDF p. 8), §6.1 (PDF pp. 8–9), §7.2 (PDF pp. 9–10), Appendix A (PDF pp. 15–16) and the base prompt (PDF p. 18). Table values, the 56% versus 57.00% difference, the "180 of the 100" wording, and the H100 and H200 mentions are the same in the PDF and the Markdown.
- Conversion problem found: the Markdown's §5.2 cross-reference says "Figure 6" where the PDF says "Figure 5". The code listings in Appendices D, F and G are preserved with visible-space markers and blank lines between source lines; this did not prevent reading them.
- Bullets labelled "Derived" are analyst arithmetic from the cited table values and are not source statements.
- Not verified: the cited external works (including the estimator of reference [4] and the GRPO details of reference [38]) were not consulted; the PDF pages were not rendered as images, so figures were inspected only through the extracted assets; of the analyses linked in section 8, only the KernelBench-Verified exemplar was read, and those numbered 23 and above were being written in parallel and may not exist yet.
- No experiments were reproduced: no model was trained or sampled, no kernel was compiled, and no timing or correctness check was rerun.

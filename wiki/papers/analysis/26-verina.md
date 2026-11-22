---
paper_id: "26"
title: "VERINA: Benchmarking Verifiable Code Generation"
source_version: "2505.23135v3"
source_markdown: "../pdfs/26-verina.md"
source_pdf: "../pdfs/26-verina.pdf"
source_url: "https://arxiv.org/abs/2505.23135"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["benchmarking", "formal-verification", "lean", "verifiable-code-generation", "specification-generation", "proof-agents", "correctness", "iterative-refinement", "execution-feedback", "evaluation", "benchmark-validity"]
summary: "VERINA is a 189-task Lean 4 benchmark for code, specification and proof generation; specifications are scored mostly by tests against a ground-truth specification, and single-trial proof success is at most 4.9% for general-purpose models."
---

# VERINA: Benchmarking Verifiable Code Generation

## 1. Paper identity and scope

- This note covers arXiv `2505.23135v3`. The PDF header reads "Published as a conference paper at ICLR 2026". Numbers here are those of this version only. [Front matter](../pdfs/26-verina.md); [PDF p. 1](../pdfs/26-verina.pdf)
- The research question is how well current LLMs perform "verifiable code generation", defined as jointly producing code, a formal specification, and a proof that the code meets the specification. [Abstract; §1](../pdfs/26-verina.md)
- The contribution is a benchmark (Verina, "Verifiable Code Generation Arena") of 189 manually curated Lean tasks, an evaluator for generated specifications, and an evaluation of ten general-purpose LLMs plus three theorem-proving models or frameworks. It is not a new generation method. [Abstract; §1; §5](../pdfs/26-verina.md)
- The dataset is released at `huggingface.co/datasets/sunblaze-ucb/verina` and the evaluation code at `github.com/sunblaze-ucb/verina`; the dataset licence is stated as GPL-3.0. [Abstract; Appendix C.1](../pdfs/26-verina.md)
- The tasks are standalone programming problems of MBPP/LeetCode type. The authors state that they are not representative of complex real-world verification projects. [§3.2; §6](../pdfs/26-verina.md)
- The worked examples involve arrays, lists, integers and natural numbers. I saw no floating-point, concurrency or GPU task in the examples shown. [Figure 1; Appendix E, Figures 18–28](../pdfs/26-verina.md)
- Specification scoring requires a ground-truth specification for each task; the benchmark does not score a specification when no reference specification exists. [§4.1; Appendix C.5](../pdfs/26-verina.md)
- Reported results cover the three foundational tasks, the end-to-end composition, and context ablations. I found no reported proof success rate for the two other compositions (specification-guided code generation and specification inference from code). [§5; Figures 5–9](../pdfs/26-verina.md)

## 2. Concepts and terminology

### Tasks

- **CodeGen:** given a natural-language description and a function signature, and optionally a specification, the model writes a Lean implementation. [§4.1; Figure 2](../pdfs/26-verina.md)
- **SpecGen:** given a description and signature, and optionally a code implementation, the model writes a pre-condition and a post-condition in Lean. [§4.1; Figure 2](../pdfs/26-verina.md)
- **ProofGen:** given description, signature, code and specification, the model writes a Lean proof that the code satisfies the specification. [§4.1; Figure 2](../pdfs/26-verina.md)
- **Specification-guided code generation:** the model receives the ground-truth specification, writes code, then proves the code against that specification. [§4.2; Figure 4](../pdfs/26-verina.md)
- **Specification inference from code:** the model receives ground-truth code, writes a specification, then proves their alignment. [§4.2; Figure 4](../pdfs/26-verina.md)
- **End-to-end verifiable code generation:** from the description alone the model writes code and specification independently, then a proof. The proof must show that the generated code satisfies the *ground-truth* specification, so that definitionally equivalent code and specification cannot trivialise the proof. [§4.2](../pdfs/26-verina.md)
- Automated theorem proving (ATP, e.g. Dafny, Verus) uses SMT solvers and consumes proof hints; interactive theorem proving (ITP, e.g. Lean) builds proofs with explicit intermediate steps. Verina is ITP-style. [§2; Table 1](../pdfs/26-verina.md)

### Specification soundness and completeness

- Let φ be the set of programs satisfying the ground-truth specification and φ̂ the set satisfying the generated one. Soundness is φ̂ ⊆ φ; completeness is φ ⊆ φ̂. [§4.1](../pdfs/26-verina.md)
- With ground truth `P`, `Q` and generated `P̂`, `Q̂`: `P̂` is sound iff `∀x. P(x) ⇒ P̂(x)` and complete iff `∀x. P̂(x) ⇒ P(x)`. A sound pre-condition is therefore at least as permissive as the ground truth. [§4.1](../pdfs/26-verina.md)
- `Q̂` is sound iff `∀x,y. P(x) ∧ Q̂(x,y) ⇒ Q(x,y)` and complete iff `∀x,y. P(x) ∧ Q(x,y) ⇒ Q̂(x,y)`. Both are stated only for inputs valid under the ground-truth pre-condition. [§4.1](../pdfs/26-verina.md)
- The evaluator handles each relationship `R` in stages. First an LLM prover attempts to prove `R` and `¬R`; a successful proof gives "R holds" or "R does not hold". [§4.1; Figure 3](../pdfs/26-verina.md)
- If proving is inconclusive, `R` is restricted to the concrete test cases, giving `R′`. Lean first tries to decide `R′` directly (`decide`); otherwise property-based testing with the `plausible` tactic explores the remaining quantified variables. [§4.1; Appendix C.5](../pdfs/26-verina.md)
- The four possible verdicts are: **holds** (formal proof), **does not hold** (counterexample or proof of the negation), **might hold** (tests pass, no proof), and **unknown** (property-based testing cannot handle the relationship). [§4.1; Figure 3](../pdfs/26-verina.md)
- "Might hold" is described by the source as strong empirical evidence, not a formal guarantee. [§4.1](../pdfs/26-verina.md)
- The pipeline is also run on `¬R`; the definitive result is taken when the other run yields unknown. [§4.1](../pdfs/26-verina.md)
- Under testing, each property maps to one test class. `P̂` soundness: `P̂` must accept every positive test. `P̂` completeness: `P̂` must reject every pre-condition-violating negative test. `Q̂` soundness uses negative tests; `Q̂` completeness uses positive tests. [Appendix C.5](../pdfs/26-verina.md)
- The listed `plausible` configuration is `numInst := 1000, maxSize := 100, numRetries := 20, randomSeed := some 42`. [Figures 11–12](../pdfs/26-verina.md)
- SpecGen metrics are pass@k for each of the four properties, plus aggregates requiring soundness and completeness together for the pre-condition, the post-condition, and the whole specification. [§4.1](../pdfs/26-verina.md)
- Unknown verdicts are shown as error bars: lower bound treats unknown as "does not hold", upper bound as "holds". [§4.1; Appendix D](../pdfs/26-verina.md)

### Code and proof metrics

- CodeGen is scored by running generated code on the *positive* test cases only, using Lean `#guard` statements, and reporting pass@k. It is a test-based result, not a proof. [§4.1; Appendix C.5; Figure 10](../pdfs/26-verina.md)
- ProofGen is scored by the Lean compiler; any proof containing a placeholder such as `sorry` is marked incorrect. [§4.1; Appendix C.5](../pdfs/26-verina.md)
- **Code&Spec Score:** generated code and generated specification are both correct. **End-to-End Score:** additionally, the proof of the generated code against the ground-truth specification is correct. [§5](../pdfs/26-verina.md)
- For proof refinement, "pass@k" is defined as the success rate after `k` rounds of iteration, for `k` up to 64. This differs from the sampling-based pass@k used for the other tasks. [§5](../pdfs/26-verina.md)
- **Verina-A** denotes tasks translated from human-written Dafny datasets; **Verina-B** denotes tasks written from scratch. [Appendix D](../pdfs/26-verina.md)

## 3. Methodology and experimental setup

### Dataset construction

- Each task has a natural-language description, ground-truth Lean code, a ground-truth pre-condition and post-condition, an optional proof, and positive and negative test cases. [§3.1; Figure 1](../pdfs/26-verina.md)
- 49 tasks were translated by hand from MBPP-DFY-50 (MBPP problems with human-verified Dafny solutions). [§3.2](../pdfs/26-verina.md)
- 59 tasks come from human-authored Dafny instances in CloverBench, translated with OpenAI o3-mini using few-shot prompts and then manually inspected and corrected. [§3.2; Appendix F](../pdfs/26-verina.md)
- 81 tasks were selected from student submissions to a theorem-proving course assignment; students drew problems from LeetCode or LiveCodeBench and formalised them in Lean. The authors reviewed and edited them. [§3.2](../pdfs/26-verina.md)
- Derived: 49 + 59 + 81 = 189, so Verina-A has 108 tasks and Verina-B has 81. The source does not print the subset sizes. [§3.2; Appendix D](../pdfs/26-verina.md)
- Ground-truth proofs exist for 46 of 189 tasks; proofs are not needed for evaluation. [§3.1](../pdfs/26-verina.md)
- Statistics (median / max): description 110 / 296 words; code 9 / 38 lines; specification 4 / 62 lines; positive tests 5 / 13; negative tests 12 / 27. [Table 2](../pdfs/26-verina.md)

### Quality assurance

- Descriptions were manually expanded to state intent, typed input parameters and output requirements, because the original descriptions could be short and ambiguous. [§3.2](../pdfs/26-verina.md)
- Positive tests were extended (manually and with LLMs) to full line coverage. Coverage was measured with `coverage.py` on Python reference implementations because Lean lacks a robust coverage tool; the authors then manually confirmed 100% line coverage on the Lean ground truth. [§3.2](../pdfs/26-verina.md)
- All ground-truth implementations and specifications pass their positive tests. [§3.2](../pdfs/26-verina.md)
- Each positive test was mutated into at least three negative tests, each labelled as violating the pre-condition or the post-condition; Boolean-output functions get one. Ground-truth code and specifications do not pass the negative tests. [§3.2](../pdfs/26-verina.md)
- Ground-truth specifications were written so that they cannot be used directly as the implementation. [§3.2](../pdfs/26-verina.md)
- Every task was reviewed by at least two authors. [§3.2](../pdfs/26-verina.md)

### Models, prompts and budgets

- General-purpose models: GPT 4o-mini, GPT 4o, GPT 4.1, o4-mini, o3, Claude Sonnet 3.7, Claude Opus 4.0, Gemini 2.5 Flash, DeepSeek V3 and Qwen3 235B-A22B (FP8). Checkpoints are listed. [Table 3](../pdfs/26-verina.md)
- Theorem-proving systems: Goedel Prover V2 32B, DeepSeek Prover V2 7B, and Copra, a tree-search agent run with o4-mini and at most 64 LLM queries per sample. [§5; Table 3](../pdfs/26-verina.md)
- All LLMs use temperature 1.0 and a 10,000-token output limit; reasoning models use default effort. The two provers and Qwen3 run locally on 8 NVIDIA H100 80GB GPUs; the others run through APIs. [Appendix C.2](../pdfs/26-verina.md)
- All tasks use 2-shot prompts built with DSPy; the two examples are excluded from the benchmark. [§5; Appendix C.3](../pdfs/26-verina.md)
- pass@1 is computed from 5 sampled responses per task per model. The abstract's "one trial per task" therefore describes the metric, not the number of samples drawn. [Abstract; Appendix C.3](../pdfs/26-verina.md)
- The two prover models use their own prompt templates; with the standard DSPy prompts they scored 0% because their output could not be parsed. [Appendix C.3](../pdfs/26-verina.md)
- The ProofGen prompt forbids `sorry` and hints to unfold definitions; the refinement prompt adds the previous imports, auxiliary lemmas, proof and Lean error message. [Appendix C.3, Prompts 3–4](../pdfs/26-verina.md)
- The SpecGen prompt asks for a pre-condition "as permissive as possible" and a sound and complete post-condition. [Appendix C.3, Prompt 2](../pdfs/26-verina.md)
- Multi-round ProofGen is run for o4-mini, GPT 4.1, Claude Sonnet 3.7, Gemini 2.5 Flash and the two provers, in two arms: iterative refinement with Lean error feedback, and direct generation of independent attempts without feedback. [§5](../pdfs/26-verina.md)
- Reported API totals range from $10.94 (GPT 4o-mini) to $1,197.39 (Claude Opus 4.0); totals for GPT 4.1, o4-mini, Claude Sonnet 3.7 and Gemini 2.5 Flash include the refinement runs. [Table 3](../pdfs/26-verina.md)
- Contamination check: a 10-gram overlap search of the Lean ground-truth solutions against `bigcode/the-stack` (about 550 million rows) found zero matches. [Appendix B](../pdfs/26-verina.md)

## 4. Results and evidence

All percentages in this section are pass@1 over the 189 tasks estimated from 5 samples per task, unless a different budget is stated.

### Foundational tasks, single trial

- CodeGen: o3 72.6, Claude Opus 4.0 66.7, o4-mini 61.4, GPT 4.1 56.4, GPT 4o 45.7, Claude Sonnet 3.7 44.2, Gemini 2.5 Flash 36.5, DeepSeek V3 36.0, GPT 4o-mini 28.6, Qwen3 20.0. [Figure 5; Table 8](../pdfs/26-verina.md)
- SpecGen (sound and complete): o3 52.3, o4-mini 51.0, Claude Opus 4.0 47.6, Claude Sonnet 3.7 45.6, Gemini 2.5 Flash 45.0, GPT 4.1 42.6, GPT 4o 36.5, DeepSeek V3 34.7, GPT 4o-mini 22.4, Qwen3 15.9. [Figure 5](../pdfs/26-verina.md)
- ProofGen: o3 4.9, o4-mini 3.6, GPT 4.1 2.7, Gemini 2.5 Flash 2.4, Claude Sonnet 3.7 1.9, GPT 4o 1.2, DeepSeek V3 0.7, GPT 4o-mini 0.6, Claude Opus 4.0 0.0, Qwen3 0.0. [Figure 5](../pdfs/26-verina.md)
- Analyst-derived from the figure's vector coordinates: each plotted SpecGen value is the midpoint of its error bar. For o3 the bar spans about 48.8–55.8 around 52.3; for o4-mini about 47.6–54.3 around 51.0. [Figure 5](../pdfs/26-verina.md)
- Most CodeGen failures are compilation failures. For o3: 25.8% cannot compile, 1.6% fail a unit test. For Qwen3: 80.0% cannot compile. Unit-test failure is at most 4.5% for any model. [Table 8](../pdfs/26-verina.md)

### End-to-end composition

- Code&Spec Score: o3 41.2, Claude Opus 4.0 33.7, o4-mini 32.6, GPT 4o 29.9, GPT 4.1 29.4, Gemini 2.5 Flash 26.7, Claude Sonnet 3.7 23.0, DeepSeek V3 18.7, GPT 4o-mini 15.5, Qwen3 9.1. [Figure 6](../pdfs/26-verina.md)
- End-to-End Score: o3 3.2, o4-mini 3.2, GPT 4.1 2.7, Gemini 2.5 Flash 2.1, Claude Sonnet 3.7 1.6, GPT 4o 1.1, GPT 4o-mini 0.5, DeepSeek V3 0.5, Claude Opus 4.0 0.0, Qwen3 0.0. [Figure 6](../pdfs/26-verina.md)

### Specialised provers and Copra

- ProofGen pass@1: Goedel Prover V2 32B 11.2, DeepSeek Prover V2 7B 8.6, Copra with o4-mini 7.9, against o3 4.9, o4-mini 3.6 and Gemini 2.5 Flash 2.4. [Figure 7](../pdfs/26-verina.md)
- Copra's value uses up to 64 LLM queries per task, so it is not a single-call result even though the figure is labelled pass@1. [§5; Figure 7](../pdfs/26-verina.md)

### Iterative refinement versus direct generation

- The prose states that refinement raises proof success "up to 20.1% with 64 refinement steps" without naming the model. [§1](../pdfs/26-verina.md)
- Analyst-derived from Figure 8's vector coordinates (64 plotted points per series; values match multiples of 1/189): with refinement, DeepSeek Prover V2 7B goes from 11.11% to 20.11%, which identifies it as the 20.1% series. [Figure 8](../pdfs/26-verina.md)
- Same derivation, refinement arm, first to 64th point: o4-mini 4.76 → 15.34; Gemini 2.5 Flash 2.65 → 11.64; GPT 4.1 3.70 → 9.52; Claude Sonnet 3.7 3.17 → 9.52; Goedel Prover V2 32B 11.11 → 14.81, flat from the fifth point. [Figure 8](../pdfs/26-verina.md)
- Same derivation, direct-generation arm: DeepSeek Prover V2 7B 8.56 → 17.11; Goedel Prover V2 32B 11.23 → 16.04; o4-mini 3.21 → 9.09; Gemini 2.5 Flash 2.67 → 7.49; Claude Sonnet 3.7 2.67 → 6.42; GPT 4.1 1.60 → 5.35. [Figure 8](../pdfs/26-verina.md)
- The text gives the o4-mini refinement breakdown directly: 7.41% → 22.22% on Verina-A and 1.23% → 6.17% on Verina-B after 64 iterations. [Appendix D](../pdfs/26-verina.md)
- Analyst-derived for Verina-B after 64 refinement rounds: o4-mini 6.17, DeepSeek Prover V2 7B 4.94, Goedel Prover V2 32B 3.70, Gemini 2.5 Flash 3.70, GPT 4.1 2.47, Claude Sonnet 3.7 1.23. With 64 direct attempts GPT 4.1, Claude Sonnet 3.7 and Gemini 2.5 Flash stay at 0. [Figure 15](../pdfs/26-verina.md)
- Budget-normalised comparison with o4-mini: refinement reaches 8.99% at 50k tokens and 14.29% at 350k; Copra reaches 4.76% and 7.94%. Average cost per successful proof is 57,594 tokens for refinement and 84,027 for Copra. [Appendix D; Figure 17](../pdfs/26-verina.md)
- In the same analysis, Verina-B success saturates at 3.70% for refinement and 1.23% for Copra, while refinement reaches 22.22% on Verina-A. [Appendix D; Figure 17](../pdfs/26-verina.md)
- Human ground-truth proofs average 169.6 characters. Successful model proofs grow with refinement rounds; Gemini 2.5 Flash exceeds 1,200 characters after 64 iterations. [Appendix D; Figure 16](../pdfs/26-verina.md)
- In the appendix example, o4-mini's 24th attempt succeeds after 23 failures by adding an auxiliary lemma and explicit case analysis. [Appendix E; Figures 25–28](../pdfs/26-verina.md)

### Effect of context

- Giving the ground-truth specification changes CodeGen from 36.5 to 52.1 for Gemini 2.5 Flash, 28.6 to 37.0 for GPT 4o-mini, 61.4 to 65.9 for o4-mini, and 72.6 to 73.0 for o3. [Figure 9](../pdfs/26-verina.md)
- Giving ground-truth code changes SpecGen by between −3.9 and +1.3 percentage points (derived): for example 52.3 → 51.3 for o3, 47.6 → 45.1 for Claude Opus 4.0, 42.6 → 43.9 for GPT 4.1. [Figure 9](../pdfs/26-verina.md)
- Using a *generated* specification as CodeGen context gives 68.4 for o3 and 45.9 for GPT 4.1, both below their no-reference values (72.6 and 56.4). It is below no-reference for seven of ten models (derived count). [Figure 9](../pdfs/26-verina.md)
- Replacing code with behaviour-only summaries for SpecGen has model-dependent effect: o4-mini 51.0 → 53.2, GPT 4.1 42.6 → 43.9, Gemini 2.5 Flash 45.0 → 37.3 with ground-truth summaries. [Table 7](../pdfs/26-verina.md)
- Manual inspection found no case of a model copying ground-truth code into its specification. [§5](../pdfs/26-verina.md)

### Difficulty split and specification detail

- CodeGen for o3 is 75.9 on Verina-A and 68.4 on Verina-B; for GPT 4o-mini 44.0 and 8.4; for Qwen3 32.8 and 3.2. [Tables 9–10](../pdfs/26-verina.md)
- SpecGen (sound and complete) for o3 is 66.4 on Verina-A and 33.8 on Verina-B; for o4-mini 65.8 and 31.5. [Figure 13](../pdfs/26-verina.md)
- ProofGen on Verina-A: o3 8.5, o4-mini 6.4, GPT 4.1 4.7. On Verina-B it is 0.2 for o3 and 0.0 for the other nine general-purpose models. [Figure 13](../pdfs/26-verina.md)
- For o3, plotted scores are: pre-condition sound 89.2, complete 82.0, both 79.3; post-condition sound 63.9, complete 61.2, both 59.6. [Figure 14](../pdfs/26-verina.md)
- Tabulated lower bounds for o3 post-conditions: 23.1% cannot compile; soundness 60.2 pass / 1.8 fail / 14.9 unknown; completeness 57.1 pass / 2.3 fail / 17.5 unknown. [Table 14](../pdfs/26-verina.md)
- Unknown rates are highest for post-conditions on Verina-B: up to 29.9% (Claude Opus 4.0, completeness) and 27.7% (o4-mini, completeness). [Table 16](../pdfs/26-verina.md)

### Evidence about the specification evaluator

- LLM provers proved few generated specifications sound and complete: 3.7% (o4-mini prover) and 1.6% (Claude Sonnet 3.7 prover) of o4-mini's specifications; 3.7% and 2.6% of Claude Sonnet 3.7's. Testing marked 51.0% and 41.6% as sound and complete. [Table 5](../pdfs/26-verina.md)
- Every specification proved sound and complete was also accepted by testing. For 20 randomly selected cases where proof failed but testing accepted, manual inspection judged the testing result correct. [Appendix C.5](../pdfs/26-verina.md)
- For o4-mini at the default budget of 1,000 property-based instances: 3.7% proved, 38.7% might hold via unit tests, 5.7% might hold via property-based testing, 21.3% refuted by counterexample, 7.2 ± 0.13% unknown, 24.1% cannot compile. [Table 6](../pdfs/26-verina.md)
- Raising the budget from 10 to 1,000 instances lowers unknown from 10.5% to 7.2%; 2,000 instances gives the same 7.2%. Results are averaged over 5 seeds. [Table 6](../pdfs/26-verina.md)
- A generated post-condition quantifying over all integer lists (longest increasing subsequence) could not be tested and was returned as unknown. [Appendix E; Figure 24](../pdfs/26-verina.md)

### Failure analysis

- Proof failures over up to 64 refinements: "incomplete proof" is 77.55% for Gemini 2.5 Flash and 38.24% for GPT 4.1; "cheat code usage" (e.g. `sorry`) is 48.53% for Claude Sonnet 3.7 and 28.30% for o4-mini. [Table 17](../pdfs/26-verina.md)
- On Verina-B, cheat-code usage is 71.36% of Claude Sonnet 3.7's failures and 40.25% of o4-mini's. [Table 19](../pdfs/26-verina.md)
- Case studies show a hallucinated `Int.xor`, an unsound pre-condition using `<` where `≤` was required, an incomplete pre-condition of `True`, a post-condition omitting a no-leading-zeros clause, and a post-condition mishandling `n = 0`. [Appendix E; Figures 18–23](../pdfs/26-verina.md)

## 5. Authors' conclusions

- All three tasks are hard for current general-purpose LLMs, and proof generation is the main bottleneck for any composition that includes it. [§5; §6](../pdfs/26-verina.md)
- Specialised provers and agentic methods improve proof success, and Lean verifier feedback is valuable; the authors also say the refinement approach raises cost and that success remains low. [§1; §5](../pdfs/26-verina.md)
- Formal specifications effectively guide code synthesis, whereas code may add noise to or over-constrain specification generation. [§5](../pdfs/26-verina.md)
- Combined tasks are harder than individual ones because replacing ground truth with generated artifacts generally degrades performance. [§5](../pdfs/26-verina.md)
- Post-conditions are harder than pre-conditions, and requiring soundness and completeness together is harder than either alone. [Appendix D](../pdfs/26-verina.md)
- Naive refinement does not overcome hard proofs; more inference budget alone does not close the gap on Verina-B. [Appendix D](../pdfs/26-verina.md)
- Testing-based specification evaluation is presented as reliable while LLM provers are too weak to serve as the metric; the authors expect the proved share to grow as provers improve. [Appendix C.5](../pdfs/26-verina.md)
- The authors argue that findings transfer to ATP systems such as Dafny and Verus because the semantic difficulty is shared. This is argued, not measured. [Appendix A](../pdfs/26-verina.md)
- The authors argue that the gap between CodeGen and ProofGen shows memorised algorithms do not transfer to verification, "effectively eliminating indirect contamination risks". [Appendix B](../pdfs/26-verina.md)

## 6. Limitations and open questions

### Limitations stated by the source

- The benchmark is small (189 tasks) and not suited to fine-tuning. [§6](../pdfs/26-verina.md)
- Tasks are simple and standalone, not representative of large verification projects. [§6](../pdfs/26-verina.md)
- The specification evaluator relies on testing because LLM provers are weak; stronger provers would allow stronger guarantees. [§6](../pdfs/26-verina.md)
- Only Lean is covered; extension to Dafny or Verus is future work. [§6](../pdfs/26-verina.md)
- Task topics come from widely used sources, so contamination is a risk even though the Lean artifacts are new. [§6; Appendix B](../pdfs/26-verina.md)
- Specifications with quantifiers over complicated or infinite domains can be untestable and are returned as unknown. [Appendix E](../pdfs/26-verina.md)

### Analyst caveats

- **Guarantee boundary:** a "sound and complete" SpecGen score is mostly a test result. For o4-mini, 3.7% of cases were proved and 44.4% (38.7 + 5.7, derived) rest on tests alone. The headline 52.3% should not be read as proved equivalence to the reference specification. [Basis: Table 6; Table 5](../pdfs/26-verina.md)
- **Test-suite size:** the test stage checks each relationship on a median of 5 positive and 12 negative cases per task; a generated specification that differs from the ground truth only outside those cases is scored "might hold". [Basis: Table 2; Appendix C.5](../pdfs/26-verina.md)
- **Evaluator audit is narrow:** the manual check covers 20 accepted-by-testing cases. No audit of "does not hold" or "unknown" verdicts is reported. [Basis: Appendix C.5](../pdfs/26-verina.md)
- **Code correctness is test-based:** CodeGen pass means passing positive tests only; negative tests are not applied to code, and the harness discharges the pre-condition argument with `by sorry`. [Basis: Appendix C.5; Figure 10](../pdfs/26-verina.md)
- **Single runs behind multi-round curves:** first-attempt values differ between the two arms of Figure 8 (o4-mini 4.76 vs 3.21; DeepSeek Prover V2 7B 11.11 vs 8.56, derived), and both differ from the 5-sample pass@1 in Figure 7. No variance or confidence interval is reported for proof results. [Basis: Figures 7–8](../pdfs/26-verina.md)
- **Unequal budgets:** Copra's 7.9% uses up to 64 queries; at the 64th point o4-mini reaches 15.34% with refinement and 9.09% with direct generation (derived), so Copra's gain over "single-pass generation" is not a like-for-like comparison. [Basis: §5; Figures 7–8](../pdfs/26-verina.md)
- **Missing setup details:** I found no Lean or Mathlib version, no per-proof time limit, and no statement of how many prover attempts the specification evaluator uses. [Basis: Appendix C.2; Appendix C.5](../pdfs/26-verina.md)
- **Contamination argument:** the 10-gram check covers only the Lean ground-truth solutions; the claim that indirect contamination is eliminated is an inference from task difficulty, not a measurement. [Basis: Appendix B](../pdfs/26-verina.md)
- **Ground-truth dependence:** specification scores are relative to one reference specification per task; an alternative valid formalisation of an ambiguous description would be scored unsound or incomplete. [Basis: §4.1](../pdfs/26-verina.md)

### Source inconsistencies

- **Source inconsistency:** the prose says ProofGen pass@1 is "below 4.9% for all general purpose models", but o3 is reported at 4.9%. [§5; Figure 5](../pdfs/26-verina.md)
- **Source inconsistency:** the prose says ground-truth specifications "consistently" improve CodeGen across models; Claude Opus 4.0 is 66.7 without and 66.3 with the ground-truth specification. [§5; Figure 9](../pdfs/26-verina.md)
- **Source inconsistency:** the prose says refinement "reliably outperforms direct generation at matched query budgets" for both model classes. In my decoding of Figure 8, Goedel Prover V2 32B is higher with direct generation at 57 of 64 points (16.04 vs 14.81 at the end) and DeepSeek Prover V2 7B at 18 early points. This rests on my reading of vector coordinates. [§5; Figure 8](../pdfs/26-verina.md)
- **Source inconsistency:** the prose says unknown identifiers, tactics and constants account for "15–30%" of failures for all models; the table rows sum (derived) to 12.92% for Gemini 2.5 Flash and 35.63% for o4-mini. [Appendix D; Table 17](../pdfs/26-verina.md)
- **Source inconsistency:** o3 CodeGen on Verina-A is 75.9 in the table and 75.8 on the chart. The 75.9 value is table-specific. [Table 9; Figure 13](../pdfs/26-verina.md)
- **Source inconsistency:** Claude Sonnet 3.7's specifications are 41.6% sound and complete "by testing" in Table 5 but 45.6% in the main chart; o4-mini is 51.0% in both. The source does not say whether the two quantities differ in definition. [Table 5; Figure 5](../pdfs/26-verina.md)
- **Source inconsistency:** the appendix says error bars run from the tabulated Pass% to Pass% + Unknown%. In my decoding, plotted values are error-bar midpoints (e.g. o4-mini pre-condition soundness plotted 91.6, tabulated Pass 91.0), and some plotted bars are narrower than the tabulated unknown share (Claude Sonnet 3.7 pre-condition completeness: Unknown 6.8% in the table, zero-width bar at 84.5). [Appendix D; Table 11; Figure 14](../pdfs/26-verina.md)
- **Unexplained denominator:** decoded direct-generation values match multiples of 1/187 overall and 1/106 on Verina-A, whereas refinement values match 1/189 and 1/108. The source does not mention a 187-task run. [Figures 8 and 15](../pdfs/26-verina.md)
- **Source inconsistency:** the prose says single-pass model proofs are "similarly short" to human proofs (169.6 characters); the decoded zero-refinement points are roughly 172 (o4-mini), 272 (Claude Sonnet 3.7), 305 (GPT 4.1) and 521 (Gemini 2.5 Flash) characters. [Appendix D; Figure 16](../pdfs/26-verina.md)
- **Table arithmetic:** three of four rows of the sensitivity table sum to 100.7% (derived); the table does not define its unit of analysis. [Table 6](../pdfs/26-verina.md)
- **Cross-reference difference:** the Markdown cites "Figure 8" for the provers' pass@1, which is plotted in Figure 7; the PDF cites Figure 7. [§5](../pdfs/26-verina.md); [PDF p. 9](../pdfs/26-verina.pdf)

## 7. Relevance to this project — interpretation

- **Proposal:** treat the Stage 1 Lean specification as an unverified artifact and give it its own check. Where Stage 2 supplies a reference implementation, derive positive input-output cases from it and negative cases by mutation, and test the pre-condition and post-condition separately for soundness and completeness in Verina's sense. [Basis: §3.2; §4.1; Appendix C.5](../pdfs/26-verina.md)
- **Proposal:** record a specification verdict as one of proved, tested-only, refuted or unknown, and report lower and upper bounds, instead of a single pass flag. [Basis: §4.1; Figure 3; Table 6](../pdfs/26-verina.md)
- **Proposal:** label each negative case by which clause it should violate, so a failing check points to the pre-condition or the post-condition. [Basis: §3.2](../pdfs/26-verina.md)
- **Proposal:** add an edge-case checklist to Stage 1 clarification (boundary inequalities, empty or zero inputs, output normal forms), since the documented specification failures are of this kind. [Basis: Appendix E; Figures 20–23](../pdfs/26-verina.md)
- **Proposal:** fix the Stage 1 specification before Stage 3 and require the Stage 4 proof to target that fixed specification, so that a generated specification cannot be shaped to make the proof trivial. [Basis: §4.2; §3.2](../pdfs/26-verina.md)
- **Proposal:** plan Stage 4 as a compiler-feedback loop with an explicit round and token budget, log success against both, and expect a low single-shot rate. The observed plateau on harder tasks argues for a stop rule and a fallback outcome of "no proof delivered". [Basis: §5; Appendix D; Figures 8, 15, 17](../pdfs/26-verina.md)
- **Proposal:** reject any Stage 4 proof containing `sorry` or another placeholder automatically, and log placeholder use as its own failure class. [Basis: §4.1; Tables 17–19](../pdfs/26-verina.md)
- **Proposal:** log Lean compile failure separately from test failure and proof failure for every generated artifact; in Verina compile failure is the dominant CodeGen failure. [Basis: Tables 8–10](../pdfs/26-verina.md)
- **Proposal:** do not assume that showing the reference kernel to the specification writer helps; run an A/B comparison on project tasks, including a behaviour-summary variant. [Basis: Figure 9; Table 7](../pdfs/26-verina.md)
- **Proposal:** compare a general-purpose model and a specialised prover on the project's own proof obligations before choosing one; Verina's ordering differs between first attempt, 64 rounds, and the harder subset. [Basis: Figures 7, 8 and 15](../pdfs/26-verina.md)
- **Proposal:** treat Verina's rates as an upper-level reference only. Its tasks are small integer and list programs; GPU kernel specifications involve tensors, floating point and concurrency, which this source does not measure. [Basis: §6; Figure 1](../pdfs/26-verina.md)

## 8. Retrieval notes

- Search aliases: Verina, Verifiable Code Generation Arena, CodeGen / SpecGen / ProofGen, specification soundness and completeness, pre-condition, post-condition, Lean 4 benchmark, plausible property-based testing, Verina-A, Verina-B, Copra, Goedel Prover V2, DeepSeek Prover V2, proof refinement, pass@k.
- Questions answered: How does Verina define a sound or complete generated specification? What is proved and what is only tested when a specification is scored? How is proof success decided? What are the single-trial rates for code, specification and proof, and for which models? How much does 64-round refinement help, and on which subset? How was the dataset built and checked? Does giving code help specification writing?
- Related local analyses: [LARK](22-lark.md) for Lean-checked proofs of GPU kernel equivalence; [Self-Refine](18-self-refine.md) and [Reflexion](19-reflexion.md) for iterative refinement loops; [TensorLean](23-tensorlean.md), [AuxSpec](28-auxspec.md) and [ClarifyGPT](29-clarifygpt.md), which were being written in parallel and which I have not read. Links are navigation aids, not controlled comparisons.

## 9. Source coverage and verification notes

- Read the complete Markdown: abstract, §1–§6, acknowledgments, ethics and reproducibility statements, reference list, and Appendices A–F including Tables 3–19, Prompts 1–4 and Figures 10–12 and 18–28.
- Figure assets are SVG with text stored as glyph outlines. I did not view rendered images. I read bar labels for Figures 5, 6, 7, 9, 13 and 14 from the SVG `data-text` attributes and legend colours, and cross-checked them against the PDF text layer (PDF pp. 8–10) and Tables 7–10.
- Values for Figures 8, 15, 16 and 17, and all error-bar ranges, were computed from SVG path coordinates and axis tick positions. They are marked "derived" above and are not printed in the source. The method reproduces values the text does print (o4-mini 7.41 → 22.22 and 1.23 → 6.17; 8.99, 14.29, 4.76, 7.94; human proofs 169.6).
- In Figures 8 and 15 each series has 64 points, and the first point lies one step left of the tick labelled "1". I treated the first point as the first attempt and the last as the 64th.
- The diagram assets for Figures 2–4 have unlabelled glyphs; I read their box labels from the PDF text layer (PDF pp. 6–7) and could not verify arrow layout.
- Conversion problems: code listings are double-spaced and have lost the line numbers that the prose cites; an empty table follows the abstract; all four prompt references render as "Prompt C.3"; one figure cross-reference differs from the PDF (see §6).
- I compared Tables 5, 6, 8, 9 and 10 and the abstract between the Markdown and the PDF text layer and found them identical. I did not compare the rest of the PDF line by line.
- No experiment was reproduced. I did not download the dataset, run Lean, or inspect the released evaluation code.

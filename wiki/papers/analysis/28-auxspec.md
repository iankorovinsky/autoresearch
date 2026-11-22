---
paper_id: "28"
title: "AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions"
source_version: "2611.21421v1"
source_markdown: "../pdfs/28-auxspec.md"
source_pdf: "../pdfs/28-auxspec.pdf"
source_url: "https://arxiv.org/abs/2611.21421"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["lean", "formal-verification", "specification-autoformalization", "executable-specifications", "execution-feedback", "reinforcement-learning", "supervised-fine-tuning", "llm-as-a-judge", "trust-boundary", "benchmark-validity", "evaluation"]
summary: "AuXSpec pairs each declarative Lean postcondition with an executable Boolean check so frozen test cases can score generated specifications and reward RL training; test agreement plus LLM judges is evidence of intent capture, not proof, on LeetCode-style problems."
---

# AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions

## 1. Paper identity and scope

- This analysis covers arXiv `2611.21421v1`, a preprint from ETH Zurich, HKUST(GZ) and HKUST presenting AuXSpec: a Lean 4 specification format, a dataset and two fine-tuned models for generating program specifications from natural-language problems. [Title; Abstract](../pdfs/28-auxspec.md)
- The research question is how to get cheap, scalable feedback on whether a generated Lean specification captures the intended problem. Prior Lean work turns each test case into a theorem to prove and falls back to an LLM judge or leaves the case unresolved when no proof is found. [§1; Table 1](../pdfs/28-auxspec.md)
- Three contributions are claimed: an executable specification format, a dataset of 3,037 LeetCode problems including the 300-problem AuXSpec-Test benchmark, and two open-weight models (AuXSpec-9B-RL and AuXSpec-27B-RL). [§1](../pdfs/28-auxspec.md)
- The problems are algorithmic programming tasks with a typed function signature, a reference solution and a single accepted answer. Problems with several accepted answers, and database, shell, concurrency and design problems, are excluded. [Appendix C.1](../pdfs/28-auxspec.md)
- Generated specifications and implementations are accepted without proofs: both theorem statements stay `sorry` throughout synthesis, training and evaluation. Proofs are produced afterwards as a separate audit. [§2, Proofs; Appendix J](../pdfs/28-auxspec.md)
- Implementation verification is explicitly not the focus; correctness theorems are proved only for the 300 AuXSpec-Test reference problems. [§2, Proofs; §3.3](../pdfs/28-auxspec.md)
- The paper does not study numerical or floating-point programs. Where an executable postcondition evaluates a `Float` operation, Lean treats it as opaque and the equivalence theorem becomes independent of the axioms. [Appendix H.2](../pdfs/28-auxspec.md)
- Released or promised artifacts are the 9B and 27B weights, the 3,037 problems with frozen tests and equivalence proofs, AuXSpec-Test with correctness proofs, and evaluation code ("will release"). [Abstract; Reproducibility statement](../pdfs/28-auxspec.md)

## 2. Concepts and terminology

### File format

- Each problem is one Lean file containing a specification, an implementation, two theorem statements and executable tests. [§2, Format](../pdfs/28-auxspec.md)
- The specification has three declarations: the precondition `isValidInput` (an executable `Bool` predicate over inputs), the declarative postcondition `<algo>_spec` (a `Prop` relating input and output, independent of how the output is computed), and the executable postcondition `<algo>_check` (a `Bool` function Lean can evaluate). [§2, Format; Appendix A.1, Table 6](../pdfs/28-auxspec.md)
- The postcondition equivalence theorem `<algo>_check_iff_spec` states that, under the hypothesis `isValidInput input`, the check holds if and only if the declarative postcondition holds. It says nothing about invalid inputs. [Table 6; Figure 13](../pdfs/28-auxspec.md)
- The correctness theorem `<algo>_correct` states that the implementation, a total function, satisfies the declarative postcondition on valid inputs. [Table 6; Figure 13](../pdfs/28-auxspec.md)

### How a declarative postcondition gets an executable counterpart

- **Decidable form (type class resolution):** the check is `decide` applied to the declarative postcondition. It compiles only if Lean's type class resolution finds a `Decidable` instance for the unfolded proposition, and then agrees with the postcondition by construction. [§3.1, Specification; Appendix G.2.1](../pdfs/28-auxspec.md)
- **Instance synthesis:** when no instance is derived, a model supplies instances from compiler diagnostics for up to 3 rounds. The worked example bounds two indices by the array length, replaces an integer multiplier with divisibility, and proves the bounded proposition equivalent to the original via `decidable_of_iff`. [Appendix G.2, Executable postcondition; Figure 15b](../pdfs/28-auxspec.md)
- A postcondition quantifying over an unbounded type (all naturals, all lists) has no computable instance as stated; this is the main reason the decidable form does not apply. [Appendix G.2.2](../pdfs/28-auxspec.md)
- **Separate definition (equivalence proof):** when instance synthesis fails, a separate hand-written `Bool` function is generated. Only this form can disagree with the declarative postcondition, so its agreement rests on the equivalence theorem. [§3.1; Appendix G.2](../pdfs/28-auxspec.md)
- Separate definitions replace a quantifier by a computation: an enumeration standing in for a set (1,607 problems), a reformulation deciding a different property (293), or a bound standing in for an unbounded quantifier (58). In the worked example the postcondition counts arbitrary finite index sets while the check enumerates the powerset of the input indices. [Appendix H.2, Table 37; Appendix G.2.1, Figure 13](../pdfs/28-auxspec.md)

### Test layers and what execution establishes

- A positive test case pairs a valid input with the output of the reference solution, computed independently of the specification. A negative test case is an invalid input, or a valid input with an incorrect output. [§3.1, Inputs](../pdfs/28-auxspec.md)
- Layers: L1a (precondition accepts valid input), L1b (precondition rejects invalid input), L2a (check accepts correct output), L2b (check rejects incorrect output), L3 (implementation matches expected output), L4 (check accepts implementation output). [Appendix A.2, Table 7](../pdfs/28-auxspec.md)
- Tests are `#guard` commands that fail at compilation when false. They run compiled code, so a passed test trusts the compiler and runtime, "the same as native_decide". [§2, Acceptance criteria; Appendix G.1.1](../pdfs/28-auxspec.md)
- What tests establish: on the frozen tuples, the `Bool` precondition and the `Bool` check return the expected verdicts. Execution never evaluates a separately defined declarative postcondition. [§3.2; Table 7](../pdfs/28-auxspec.md)
- What transfers to the declarative postcondition: test results apply to it only through the decidable form or through the equivalence theorem, which is unproved (`sorry`) at acceptance time and proved, if at all, afterwards. [§2, Proofs](../pdfs/28-auxspec.md)
- What the equivalence proof does not establish: it shows only that the two postconditions agree; both may encode the same misunderstanding. The paper states that semantic agreement with the problem "rests on the test cases, the judges and expert review". [§2, Proofs](../pdfs/28-auxspec.md)

### Tasks, acceptance and metrics

- **SpecGen** takes the problem description, input constraints and test cases and outputs a specification (precondition, declarative postcondition, executable postcondition), checked by layers L1a, L1b, L2a and L2b. No implementation is given. [§2, Tasks; Table 2; §6](../pdfs/28-auxspec.md)
- **CodeGen** takes the description, test cases and the given specification and outputs an implementation, checked by L3. **SpecGen-Repair** and **CodeGen-Repair** add the rejected output and its diagnostics to the input. [Table 2; Appendix K.2](../pdfs/28-auxspec.md)
- **Acceptance criteria** (shared by data filtering, expert iteration, RL and evaluation): the assembled file compiles, the task's test layers pass with guard commands matching the frozen test cases, and LLM judges approve. AuXSpec-Test pass rates therefore mix executable validation with model judgment. [§2, Acceptance criteria](../pdfs/28-auxspec.md)
- The judges check faithfulness (postcondition captures the requirements), non-tautology (it characterizes outputs instead of restating an algorithm), completeness (the check accepts correct outputs on untested inputs) and termination of the implementation. [§2, Acceptance criteria](../pdfs/28-auxspec.md)
- The non-tautology judge labels a specification declarative, enumerative (the right condition over a candidate list the specification builds itself) or tautological (computes the result by solving the problem). [Appendix E.3.1; Appendix K.3](../pdfs/28-auxspec.md)
- **pass@k** is the fraction of problems with at least one of `k` samples meeting the acceptance criteria; **compile@k** is the analogue for compilation. The paper reports `k ∈ {1, 4}` with 4 samples per problem. [§5.1; Appendix D.1; Table 10](../pdfs/28-auxspec.md)
- On Verina the criterion differs: generated specifications must be sound and complete with respect to the reference specification on Verina's test cases, and implementations must pass all positive test cases. [§5.3](../pdfs/28-auxspec.md)
- **Refine@k** (ICPC-Eval) draws one sample and feeds each rejected output back with compiler messages or judge findings until acceptance or `k` model calls. [Appendix D.4](../pdfs/28-auxspec.md)

### Reward

- The RL reward is a 6-stage progressive score in the order of the acceptance criteria. A rollout receives the value of the first stage it fails: format 0, compilation 0.1, test execution 0.2, test case alignment 0.1, non-tautology 0.1, overall quality 0.4; passing all stages yields 1. [§4.3, Reward design; Eq. 1](../pdfs/28-auxspec.md)
- Two outcomes that evaluation accepts receive partial credit instead of 1: enumerative specifications (0.4) and correct but slow implementations (0.5). [§4.3; Table 19](../pdfs/28-auxspec.md)
- Anti-gaming rules: a test counts only if it calls a declaration of the output; too few tests gives 0.1; special-casing or copying test cases, or returning a repair unchanged, gives 0; `sorry` outside the two theorem statements, `admit`, `axiom` or `native_decide` counts as a compilation failure. [§4.3; Appendix B.1; Appendix E.3.1](../pdfs/28-auxspec.md)
- Repair rollouts are drawn from failed rollouts of the current policy (40% of each batch, up to 3 rounds), with reward scaled linearly from 1 in the first round to 0.8 in the last. [§4.3, Self-repair](../pdfs/28-auxspec.md)

## 3. Methodology and experimental setup

### Data synthesis

- Seeds are 3,109 LeetCode problems (2,864 from LeetCodeDataset, 245 collected from the site), with 719 easy, 1,622 medium and 768 hard. [Appendix C.1; Appendix G.1.1](../pdfs/28-auxspec.md)
- Test cases are constructed and frozen before synthesis: 27,981 test cases, 9 per problem, assembled into 18 `#guard` commands per problem, with each positive case reused at L1a, L2a, L3 and L4. [Appendix G.1.1](../pdfs/28-auxspec.md)
- A proposed incorrect output is admitted only when the reference solution's output on that input is not a reordering of it; invalid inputs are proposed by constraint category. [Appendix G.1.1](../pdfs/28-auxspec.md)
- Test inputs are small by construction: any case with a list or string longer than 128 items is refused, compilation is capped at 600 seconds, and an admissibility judge uses about ten thousand candidates as the ceiling. [Appendix G.1.1, Screening; Appendix K.1](../pdfs/28-auxspec.md)
- The frozen tests, including negatives, are shown to the generation model; the authors describe negatives as "part of the problem statement rather than as a hidden filter". [Appendix G.1.1, Analysis](../pdfs/28-auxspec.md)
- The pipeline makes a problem-level plan and then generates the declarative specification, the executable postcondition and the implementation in three sequential stages, each with plan, generate, validate and repair. Accepted components are frozen so a later component cannot overfit the tests or weaken an accepted specification. [§3.1; Figure 1](../pdfs/28-auxspec.md)
- Each repair round edits only the faulty block, for up to 18 rounds. The repairer must return a section unchanged if the only compiling repair would weaken the specification, and the root-cause analyzer treats a frozen test literal as ground truth by default. [§3.1, Repair; Appendix K.1](../pdfs/28-auxspec.md)
- Synthesis models: Kimi-K3 for generation and repair, DeepSeek-V4-Pro for planning, DeepSeek-V4-Flash for judges. Compilation runs through the Kimina Lean server, and proofs are checked with SafeVerify "bumped to Lean 4.32". [§3.1, Models; Appendix G.4; Appendix H.1](../pdfs/28-auxspec.md)

### Training data and stages

- SFT data: 19,492 samples over 3,993 problems (2,701 LeetCode, 1,292 Codeforces), split as 6,528 SpecGen, 3,807 SpecGen-Repair, 6,645 CodeGen and 2,512 CodeGen-Repair. Of these, 12,840 come from the synthesis pipeline and 6,652 from expert iteration. [Table 3; Appendix C.3, Tables 11–12](../pdfs/28-auxspec.md)
- Expert iteration ran 3 rounds with a fine-tuned Qwen3.8-27B: 12,225 refinement chains, 4,960 accepted, with up to 8 repair rounds per chain (16 in the last round). [§4.2; Appendix C.4, Table 13](../pdfs/28-auxspec.md)
- RL data: 13,637 generation tasks (8,281 SpecGen, 5,356 CodeGen) over 8,579 problems from Codeforces, AtCoder, Aizu, CodeChef, GeeksforGeeks and 60 LeetCode problems, for 1 epoch. [§4.3; Table 3; Table 11](../pdfs/28-auxspec.md)
- SFT, RL and AuXSpec-Test share no problem. Filters also remove LeetCode problems that Verina links to (43 of Verina's 189 problems link to 36 LeetCode problems). [Table 3; Appendix C.5](../pdfs/28-auxspec.md)
- Base models are Qwen3.5-9B and Qwen3.8-27B. SFT uses 3 epochs, batch 64 sequences, sequence length 16,384 tokens, learning rate 1e-5 cosine, on 4 (9B) or 8 (27B) GH200 GPUs. [§4.2; Appendix B.1, Table 8](../pdfs/28-auxspec.md)
- AuXSpec-9B-SFT was fine-tuned on an earlier data version with 18,489 samples over 4,026 problems, not the 19,492-sample set. [Appendix B.1](../pdfs/28-auxspec.md)
- RL starts from the SFT models and uses 16 rollouts per prompt, 32 (9B) or 64 (27B) prompts per step, learning rate 1e-6, temperature 0.7, 434 (9B) or 208 (27B) steps, on 16 or 64 GH200 GPUs. LLM judges are called through an external API. [Appendix B.1, Table 9](../pdfs/28-auxspec.md)
- The optimizer extends DAPO with batch-level advantage normalization, a mask on repetitive rollouts, an overlong penalty, and (27B only) an end-of-sequence penalty inside the thinking segment. [§4.3, Policy optimization; Eq. 2](../pdfs/28-auxspec.md)

### Evaluation sets, baselines and budgets

- **AuXSpec-Test (in-domain):** 300 held-out synthesized LeetCode problems (72 easy, 155 medium, 73 hard), 834 samples: 300 SpecGen, 131 SpecGen-Repair, 300 CodeGen, 103 CodeGen-Repair. [§3.3; Table 11](../pdfs/28-auxspec.md)
- For SpecGen a generated specification is paired with the reference implementation; for CodeGen a generated implementation is paired with the reference specification. Evaluation does not compare against the reference specification, so alternative formulations can pass. [§3.3](../pdfs/28-auxspec.md)
- **Verina (out-of-domain):** 189 manually curated problems, evaluated without test cases in the prompt (Verina's default) and with them. In the no-test condition the prompts "replace the given test cases by test cases derived from the statement". [§5.3; Table 5; Appendix K.2](../pdfs/28-auxspec.md)
- **ICPC-Eval:** 75 of 118 problems after excluding 28 with multiple correct outputs and 15 with over-long test cases. [Appendix D.4](../pdfs/28-auxspec.md)
- Baselines are the two backbones, seven frontier models (Claude-Opus-5, Claude-Sonnet-5, GPT-5.6-Sol, GPT-5.6-Luna, Kimi-K3, DeepSeek-V4-Pro, DeepSeek-V4-Flash) and VeriSpecGen, whose numbers are copied from its paper because its models are not public. [§5.1](../pdfs/28-auxspec.md)
- All evaluated models receive the same prompts. AuXSpec models are sampled at temperature 0.3, top-p 0.95, at most 16,384 new tokens, 4 samples per problem; baselines use their recommended sampling settings. [§5.1; Appendix B.2, Table 10](../pdfs/28-auxspec.md)

### Proof and human audits

- Equivalence proofs use a decompose-and-prove agent loop; each proof is checked with SafeVerify for unchanged declarations and dependence on standard axioms only. [Appendix H.1](../pdfs/28-auxspec.md)
- One author reviewed all 300 AuXSpec-Test problems against the descriptions without consulting a model; an error required an exposing test case and a corrected file. [Appendix I.1](../pdfs/28-auxspec.md)
- Two authors reviewed 100 automatically accepted SpecGen outputs (50 per RL model, stratified 12/26/12 by difficulty, 20 problems double-reviewed), blinded to model identity, judge verdicts and proof results. [Appendix I.2](../pdfs/28-auxspec.md)

## 4. Results and evidence

### Synthesis outcomes and acceptance false positives

- The pipeline synthesizes 3,037 of 3,109 seeds (97.7%); 34.8% pass all stages at the first attempt and repair recovers most others within 2 rounds. [§3.2; Appendix G.5, Table 30](../pdfs/28-auxspec.md)
- The decidable form serves 1,079 of 3,037 problems (35.5%); 1,958 use a separate definition. The decidable form covers 56% of easy and 21% of hard problems. [Appendix G.2.2; Appendix H.2](../pdfs/28-auxspec.md)
- Where the decidable form applies, the check passes all checks at the first attempt in 772 of 787 problems (98.1%), versus 1,178 of 2,250 (52.4%) for separately generated checks. [Appendix G.2.2](../pdfs/28-auxspec.md)
- Specification judges reject 3.8% (faithfulness), 2.2% (non-tautology) and 1.7% (completeness) of verdicts on components that already compile and pass tests. The implementation judge rejects 717 of 3,667 (19.6%), mainly for fuel parameters (270), excess cost (234) and restated checks (198). [Appendix G.2.2, Table 29; Appendix G.3.1](../pdfs/28-auxspec.md)
- Post-hoc equivalence proof attempts on all 3,037 problems expose statement errors in 31 (1.0%), all of which had passed tests and judges: 26 by a compiled counterexample, 5 by `Float` independence. [§3.2; Appendix H.2](../pdfs/28-auxspec.md)
- Of those, 19 are in the declarative postcondition ("which execution never evaluates"), 10 in the executable postcondition and 3 in the precondition (one problem counted twice). [§3.2; Table 39](../pdfs/28-auxspec.md)
- On AuXSpec-Test, errors were confirmed in 22 of 300 problems (7.3%): 21 by human review, 1 by the correctness proof. Component errors: 3 precondition, 10 postcondition, 0 executable postcondition, 10 implementation. [Appendix I.1; Table 42](../pdfs/28-auxspec.md)
- Of 14 specification errors in the test split, 9 were found by review alone and already had an equivalence theorem proved against the wrong statement; 3 by both; 2 by proof attempts alone. [Appendix I.1; Table 43](../pdfs/28-auxspec.md)
- For model outputs, one accepted output per problem was drawn at random: the equivalence theorem is proved for 251/253 (AuXSpec-27B-RL) and 246/248 (AuXSpec-9B-RL), 99.2% each. [Appendix H.4; Table 40](../pdfs/28-auxspec.md)
- Expert review confirms 90 of 100 accepted RL outputs (43/50 for 27B, 47/50 for 9B). All 10 errors are in the validity predicate (missing input constraints); no postcondition error was found. [Appendix I.2; Table 44](../pdfs/28-auxspec.md)

### In-domain results (AuXSpec-Test, 300 problems, test cases in the prompt)

- AuXSpec-27B-RL SpecGen: 65.7% pass@1 and 84.3% pass@4, up from 4.5% and 14.3% for Qwen3.8-27B. [§5.2; Table 4](../pdfs/28-auxspec.md)
- AuXSpec-9B-RL SpecGen: 58.6% pass@1 and 82.7% pass@4, up from 0.2% and 0.7% for Qwen3.5-9B. [Table 4](../pdfs/28-auxspec.md)
- SFT alone gives SpecGen 10.1%/24.3% (9B) and 18.7%/40.7% (27B) pass@1/pass@4; RL supplies most of the gain. [Table 4; §5.4](../pdfs/28-auxspec.md)
- Best frontier SpecGen results are Claude-Opus-5 at 57.7% pass@1 and Kimi-K3 at 83.7% pass@4. Derived: the 27B-RL pass@4 margin over Kimi-K3 is 0.6 points; the 9B-RL model is below Kimi-K3 at pass@4. [Table 4](../pdfs/28-auxspec.md)
- AuXSpec-27B-RL CodeGen: 94.8% pass@1 and 99.7% pass@4 (Claude-Opus-5: 93.0% and 98.3%). [Table 4](../pdfs/28-auxspec.md)
- Repair tasks (single round): AuXSpec-27B-RL reaches 81.9%/89.3% on SpecGen-Repair and 88.1%/95.1% on CodeGen-Repair; Claude-Opus-5 is higher on SpecGen-Repair (83.0%/90.1%) and Kimi-K3 on CodeGen-Repair pass@4 (97.1%). [Table 4; Appendix J](../pdfs/28-auxspec.md)

### Out-of-domain results (Verina, 189 problems)

- Without tests in the prompt (Verina default), SpecGen pass@1/pass@4: AuXSpec-27B-RL 84.5%/90.5%, AuXSpec-9B-RL 73.7%/90.5%; backbones 7.2%/21.9% (27B) and 2.0%/7.0% (9B); Claude-Opus-5 77.8%/89.9%. [Table 5](../pdfs/28-auxspec.md)
- With tests in the prompt, SpecGen pass@1/pass@4: AuXSpec-27B-RL 91.9%/96.8%, AuXSpec-9B-RL 87.0%/96.8%; GPT-5.6-Sol has the highest pass@4 at 97.9%, 1.1 points above both RL models. [§5.3; Table 5](../pdfs/28-auxspec.md)
- CodeGen pass@1 without tests: 89.8% (9B-RL) and 89.7% (27B-RL), versus 57.0% for Claude-Opus-5 and a reported 58.7% for VeriSpecGen-30B-A3B. [Table 5](../pdfs/28-auxspec.md)
- **Unequal-k comparison with VeriSpecGen:** VeriSpecGen-30B-A3B (SFT No-Test, numbers from its paper, no tests in prompt) has SpecGen 47.6% pass@1 and 63.9% at **pass@10**; AuXSpec's 90.5% is **pass@4**. VeriSpecGen-4B: 30.3% pass@1, 45.2% pass@10. [Abstract; Table 5](../pdfs/28-auxspec.md)
- Shared-backbone ablation (Qwen3-4B-Instruct-2507, AuXSpec SFT data and RL recipe, RL step 120, Verina without tests): SpecGen 54.2% pass@1 and 84.1% pass@4; CodeGen 77.2% pass@1. AuXSpec's SFT set has 19,492 samples versus 121,531 (No-Test) or 343,827 (Full) for VeriSpecGen. [Appendix F.1; Table 25](../pdfs/28-auxspec.md)
- The untrained 4B backbone scores 0.0% SpecGen pass@1 under AuXSpec's prompts while VeriSpecGen reports 14.7% with its own prompt and harness; the authors say the two base rows are not directly comparable. [Appendix F.1](../pdfs/28-auxspec.md)

### RL analysis and ablations

- At temperature 0.6, AuXSpec-27B-SFT's SpecGen pass@256 (85.7%) stays below AuXSpec-27B-RL's pass@8 (88.2%); RL reaches 96.3% at `k = 256`. Within 256 samples, 33 problems are solved only by RL, 1 only by SFT, 10 by neither. [§5.4; Appendix E.3.3; Table 22](../pdfs/28-auxspec.md)
- Groups with no passing rollout are 37% (27B) and 47% (9B) of SpecGen groups; the progressive reward keeps nonzero variance in 87.9% and 86.2% of them, where a binary reward would give none. [Appendix E.3.1; Table 18](../pdfs/28-auxspec.md)
- **Without LLM judges in the reward (9B):** the model restates given test cases as the specification. SpecGen pass@1 peaks at 35.8% (step 100) and falls to 20.0% (step 150) while 84.7% of SpecGen rollouts are rewarded as accepted; tautological evaluation samples grow from 1.9% to 58.8%. [Appendix F.2; Table 26](../pdfs/28-auxspec.md)
- On Verina that variant scores 88.4% SpecGen pass@1 with tests in the prompt but 52.0% without them (step 150). [Appendix F.2; Table 27](../pdfs/28-auxspec.md)
- **Without test execution in the reward (9B):** CodeGen pass@1 is 34.4% versus 65.0% and SpecGen 24.3% versus 34.9% after about the same number of training prompts (4,111 versus 4,162). [Appendix F.2; Table 26](../pdfs/28-auxspec.md)
- **Without self-repair (9B):** SpecGen pass@1 is 43.1% after 6,334 prompts versus 46.7% after 6,186 for the full recipe; the authors call training without repair tasks "a viable recipe when new problems are plentiful". [Appendix F.2](../pdfs/28-auxspec.md)
- When slow programs count as failures, CodeGen pass@1 on AuXSpec-Test drops from 77.1% to 44.7% for AuXSpec-9B-RL and from 94.8% to 88.0% for AuXSpec-27B-RL. [Appendix E.3.2](../pdfs/28-auxspec.md)

### Other measurements

- ICPC-Eval, AuXSpec-27B-RL SpecGen, 75 problems: refine@16 with compiler and judge feedback 82.7% versus pass@16 61.3%; with compiler feedback only, refine@16 is 44.0% because the chain stops at the first output that compiles. [Appendix D.4; Figure 3](../pdfs/28-auxspec.md)
- AuXSpec-27B-RL uses 2.7K output tokens per solved SpecGen sample on AuXSpec-Test, versus 13.2K for GPT-5.6-Sol, the most economical frontier model. [Appendix D.2; Table 14](../pdfs/28-auxspec.md)
- SFT lowers all four general-capability scores; for example AIME 25 falls by 36.7 points (9B) and 53.1 points (27B), and RL recovers 72% of that drop at both sizes. [Appendix D.3; Table 15](../pdfs/28-auxspec.md)
- Two negative results: continued pretraining on a 43M-token Mathlib corpus raises Qwen3.5-9B SpecGen pass@4 from 8.2 to 9.8, less than adding 1,056 problems to SFT (13.7); on-policy distillation from the 27B SFT model drops the 9B SFT model's SpecGen pass@4 from 24.3 to 5.3. [Appendix E.2, Table 17; Appendix E.4, Table 23](../pdfs/28-auxspec.md)
- Equivalence proofs for separate definitions have a median of 112 lines (P90 394); correctness proofs on AuXSpec-Test have a median of 159.5 lines (63 easy, 178 medium, 365 hard). [Appendix H.2, Table 37; Appendix H.3](../pdfs/28-auxspec.md)
- Plausible property testing of the 300 reference implementations finds no counterexample, but 86 runs are inconclusive because valid inputs are rarely sampled for structured types. [Appendix H.3](../pdfs/28-auxspec.md)
- Reported synthesis and supporting spend is $2,604 over 226,351 model calls; formalization is $2,024, averaging $0.65 per problem with a median of 17 calls and 23 minutes. [Appendix G.7; Table 31](../pdfs/28-auxspec.md)

## 5. Authors' conclusions

- Pairing a declarative postcondition with an executable one lets specifications be evaluated by executing test cases without proving each case, while retaining the declarative form for correctness proofs. [§7](../pdfs/28-auxspec.md)
- The 27B model "surpasses all frontier models on AuXSpec-Test and outperforms all baselines in SpecGen pass@1 on the out-of-domain Verina benchmark". [§7](../pdfs/28-auxspec.md)
- Because training uses only synthesized problems, the authors read the Verina gains as suggesting the reward improves faithfulness to the description beyond the test cases used in the reward. [§5.4; §7](../pdfs/28-auxspec.md)
- They infer from pass@k at large `k` that RL expands the set of solvable problems instead of only sharpening the sampling distribution. [§5.4; Appendix E.3.3](../pdfs/28-auxspec.md)
- They argue the judges and test execution are both needed in the reward: judges alone cannot check implementation behaviour, and tests alone are gamed. [Appendix F.2](../pdfs/28-auxspec.md)
- They state that human review and proving find different errors and that "neither measure subsumes the other". [Appendix I.1](../pdfs/28-auxspec.md)

## 6. Limitations and open questions

### Limitations stated by the source

- Repair is evaluated for a single round; multi-round self-refinement at inference is not measured on the main benchmarks. [Appendix J, Repair at inference](../pdfs/28-auxspec.md)
- A single reviewer covered AuXSpec-Test and two reviewers covered 100 outputs; the training problems have no reviewed sample, so the training-data error rate is not estimated. [Appendix J, Scope of expert review](../pdfs/28-auxspec.md)
- Generated outputs are accepted without proofs, and a proof of equivalence shows only that the two postconditions agree; whether the postcondition and validity predicate state the problem rests on tests, judges and review. [Appendix J, Proofs for generated outputs](../pdfs/28-auxspec.md)
- Validity predicates are checked only by test cases; all errors found in reviewed accepted outputs are missing input constraints. [Appendix J, Validity predicates](../pdfs/28-auxspec.md)
- The formalizer model comparison is confounded by prompt revisions, and planner failure counts depend on the judge (per-mode judge agreement ranges from 0% to 75%). [Appendix G.8.1; Appendix G.8.3, Table 36](../pdfs/28-auxspec.md)
- The implementation judge is waived after 2 retries; 133 problems keep an implementation it rejected, 12 of them in AuXSpec-Test. [Appendix G.3.1](../pdfs/28-auxspec.md)

### Analyst caveats

- **Visible tests in-domain:** on AuXSpec-Test the prompt contains the same frozen test cases that acceptance executes, so in-domain pass@k is not a hidden-test measure; protection against test restatement comes from LLM judges and hard-coding rules. [Basis: Appendix G.1.1; Appendix K.2; Appendix F.2](../pdfs/28-auxspec.md)
- **Judge inside the metric:** the judges that gate synthesis (DeepSeek-V4-Flash) are said to run in evaluation and the reward as well, and DeepSeek-V4-Flash is also an evaluated baseline. The 90/100 expert-confirmation rate measures judge-plus-test precision only for RL-model outputs. [Basis: §3.1; Appendix K.3; Appendix I.2](../pdfs/28-auxspec.md)
- **Benchmark provenance:** AuXSpec-Test references were generated by the same pipeline (Kimi-K3 generation) that produced the training data, and frontier models are scored in AuXSpec's file format and prompts. [Basis: §3.1; §5.1](../pdfs/28-auxspec.md)
- **No uncertainty for headline tables:** Tables 4 and 5 give point estimates on 300 and 189 problems from one training run per model; only Figure 2 mentions standard-error bands. [Basis: Table 4; Table 5; Figure 2](../pdfs/28-auxspec.md)
- **pass@1 estimator not spelled out:** Appendix D.1 defines pass@k for a problem "with k samples" and Table 10 lists 4 samples per problem; the text does not say how pass@1 is computed from them. [Basis: Appendix D.1; Table 10](../pdfs/28-auxspec.md)
- **Verina adaptation not described:** the text does not say how AuXSpec-format outputs are mapped onto Verina's evaluator, for example which postcondition is scored. [Basis: §5.3; Appendix F.1](../pdfs/28-auxspec.md)
- **Small-input evidence:** test inputs are capped at 128 items and roughly ten thousand enumerated candidates, so execution checks do not exercise realistic input sizes. [Basis: Appendix G.1.1; Appendix K.1](../pdfs/28-auxspec.md)
- **Sampling of the 99.2% figure:** it covers one randomly drawn accepted output per solved problem, not every accepted sample; one of the two unproved 9B outputs is a `Float` independence case, not a counterexample. [Basis: Appendix H.4; Table 40](../pdfs/28-auxspec.md)

### Source inconsistencies

- **Source inconsistency (cost accounting):** Appendix G.7 prose says the classification and judging accounts "have a call count and no dollars" and are "outside the dollar total", but Table 31 assigns them $24 and $136 and the $2,604 total includes them. Derived: the Spend column sums to $2,603. [Appendix G.7; Table 31](../pdfs/28-auxspec.md)
- **Source inconsistency (prompt condition):** §5.4 reports Verina SpecGen pass@1 rising "from 37.3% to 87.0% and from 57.3% to 91.9%", which are the with-tests columns of Table 5, and then argues from "test cases absent from the prompt". The without-tests values are 34.9% to 73.7% and 52.9% to 84.5%. [§5.4; Table 5](../pdfs/28-auxspec.md)
- **Source inconsistency (output caps):** Appendix B.2 says DeepSeek-V4-Flash, DeepSeek-V4-Pro and Kimi-K3 "run without a length limit"; the Table 14 caption says responses are capped at 32,768 tokens for Kimi-K3 and 65,536 for the DeepSeek models. [Appendix B.2; Table 14](../pdfs/28-auxspec.md)
- **Source inconsistency (AuXSpec-9B-SFT rows):** Table 4 gives SpecGen 10.1/24.3 and CodeGen 13.1/28.3; Table 23 gives 9.6/24.3 and 13.5/30.7 for the same named model, and Verina SpecGen 30.3/61.4 versus 34.9/65.6 in Table 5. No reason is stated. [Table 4; Table 5; Table 23](../pdfs/28-auxspec.md)
- **Table-specific values:** AuXSpec-27B-SFT SpecGen pass@4 is 40.7 in Table 4 (temperature 0.3) and 47.3 in Table 16 (temperature 0.7). Table 17's "AuXSpec-Test" columns give Qwen3.5-9B CodeGen 0.0/0.0 where Table 4 gives 8.2/18.0; Appendix E.2 refers to an earlier data version and held-out set not labelled in the header. [Table 4; Table 16; Appendix E.2, Table 17](../pdfs/28-auxspec.md)
- **Unexplained denominators:** first-attempt rates use 787 decidable-form and 2,250 separate problems, while final form counts are 1,079 and 1,958. Derived: per-component repair rounds sum to 4,193, while the cause breakdown covers "3,852 recorded rounds". [Appendix G.2.2; Appendix G.4.1; Appendix G.5](../pdfs/28-auxspec.md)

## 7. Relevance to this project — interpretation

- **Proposal:** in Stage 1, pair the Lean spec with an executable check and run it against tests labelled by the oracle, using AuXSpec's four specification layers: valid inputs accepted, invalid inputs rejected, correct outputs accepted, incorrect outputs rejected. [Basis: Appendix A.2, Table 7; §3.1](../pdfs/28-auxspec.md)
- **Proposal:** freeze test tuples before the Lean spec is written, forbid spec repair from editing them or weakening the spec to pass, and include negative outputs generated by flaw family (near miss, relaxed constraint, wrong shape, degenerate answer) so a too-weak spec fails a test. [Basis: §3.1, Repair; Appendix G.1.1; Appendix K.1](../pdfs/28-auxspec.md)
- **Proposal:** record spec validation as separate fields: compiled, tests passed (with counts), check-to-spec link (by construction, proved, or unproved), judge verdicts, human review. Do not collapse them into "spec verified"; test agreement is sampled evidence. [Basis: §2, Proofs; Appendix I.1, Table 43](../pdfs/28-auxspec.md)
- **Proposal:** prefer specs whose quantifiers are bounded by tensor shapes so the check can be `decide` of the spec, removing the separate equivalence obligation; treat any hand-written check as unlinked until its equivalence theorem is proved. [Basis: Appendix G.2.2; Appendix H.2, Table 37](../pdfs/28-auxspec.md)
- **Proposal:** treat the input-domain precondition (shapes, dtypes, value ranges) as its own reviewed artifact and confirm it during Stage 1 clarification, since AuXSpec's reviewed false positives were all missing input constraints. [Basis: Appendix I.2; Appendix J](../pdfs/28-auxspec.md)
- **Proposal:** plan for floating-point outputs explicitly. AuXSpec's check-to-spec proof fails as independent when the check uses `Float`, so a tolerance-based kernel check would need an exact-arithmetic model or would sit outside the proved link. [Basis: Appendix H.2; Appendix I.1](../pdfs/28-auxspec.md)
- **Proposal:** add a non-tautology gate so the Lean spec is not a transcription of the reference kernel's algorithm, and pre-label tasks whose natural statement is the transformation itself (as many tensor operators are) to exempt them. [Basis: Appendix G.2, Judges; Appendix K.3](../pdfs/28-auxspec.md)
- **Proposal:** require any LLM judge in Stage 1 or Stage 2 to name a concrete witness (an input and an output the spec admits but the task rejects) and audit a sample of its acceptances by hand; AuXSpec shows both judges and tests miss errors. [Basis: Appendix G.2, Judges; §3.2; Appendix I.2](../pdfs/28-auxspec.md)
- **Proposal:** run the Stage 1 spec writer in a repair loop that stops on acceptance, not on first successful compilation. [Basis: Appendix D.4; Figure 3](../pdfs/28-auxspec.md)
- **Proposal:** for Stage 4 fuzzing, use generators that respect the precondition; AuXSpec's random testing was inconclusive on 86 of 300 problems because valid structured inputs were rarely sampled. [Basis: Appendix H.3](../pdfs/28-auxspec.md)
- **Proposal:** if the released AuXSpec models are tried as Stage 1 spec writers, treat it as an experiment: the evidence covers integer and list algorithm problems with tiny test inputs, not tensor or floating-point kernels. [Basis: Appendix C.1; Appendix G.1.1](../pdfs/28-auxspec.md)

## 8. Retrieval notes

- Search aliases: AuXSpec, AuXSpec-Test, AuXSpec-9B-RL, AuXSpec-27B-RL, executable postcondition, declarative postcondition, postcondition equivalence theorem, `_check_iff_spec`, decidable form, `decide`, `#guard`, SpecGen, CodeGen, SpecGen-Repair, progressive reward, non-tautology judge, enumerative specification, Verina, VeriSpecGen.
- Questions answered: How can a Lean `Prop` specification be tested by execution? What do passing tests show about a specification and what do they miss? How often do tests plus LLM judges accept a wrong specification? What is SpecGen pass@k and how is the RL reward staged? How large are the AuXSpec datasets? How do the trained models compare with frontier models and VeriSpecGen, and at which `k`?
- Related analyses: [Verina](26-verina.md) for the out-of-domain benchmark used here; [LARK](22-lark.md) for a Lean specification that is tested, not proved, against its Torch source; [KernelBench-Verified](03-kernelbench-verified.md) for hidden-input testing against shortcut solutions; [generation to judgment](13-generation-to-judgment.md) and [MT-Bench and Chatbot Arena](16-mt-bench-chatbot-arena.md) for LLM-as-a-judge reliability; [Self-Refine](18-self-refine.md) and [Reflexion](19-reflexion.md) for feedback-driven repair; [TensorLean](23-tensorlean.md) and [ClarifyGPT](29-clarifygpt.md), linked by title only. Links are navigation aids, not controlled comparisons.

## 9. Source coverage and verification notes

- Read the complete local Markdown: abstract, §1–§7, statements, references, and Appendices A–K, including all tables, both worked Lean examples (Figures 13–16), the ablation tables (24–27), cost tables (31–32), model-selection tables (33–36), proof tables (37–40), human-study tables (41–44) and all condensed prompts. Prompt text was treated as data.
- Arithmetic checked against the source tables: Table 3 row and column totals, Tables 11–13 totals, Table 22, Table 29 and the implementation-rejection breakdown, Table 31 call total and Spend column, Table 32 total, Table 40 accepted counts against Table 4 pass@4, ICPC-Eval problem counts, and the general-capability recovery percentages. Mismatches are recorded in section 6.
- Visual check: opened `figure-011.png` (Figure 12, pipeline on a running example); it matches the textual description of stages, gates and assembled theorem statements.
- Not visually verified: the 20 SVG figures (Figures 2–11 and 17–25 in the Markdown numbering). Their text is stored as glyph outlines and no renderer was available without installing software or creating files. All figure-related numbers in this note come from captions, prose or the table under Figure 3; no chart values were read or inferred.
- Conversion problems: the Markdown differs from the PDF text in some cross-references (Appendix B.1 says "Table 9" where the PDF says "Table 8"; §2 and Appendix G.2.1 cite "Figures 13 and 15a" and "15a, 16 and 16" where the PDF has "13 and 14" and "15a, 15b and 16"). Header cells of Tables 20, 21 and 40 are duplicated as stray rows. Figure 1 is flattened to a list of labels. Table values were unaffected in the passages spot-checked against PDF text extraction (abstract, Tables 4, 5, 18, 20, 23, 31, 40 and Appendices B.1, G.7, I.2).
- No PDF page numbers are cited; locators use the section, table, figure and appendix labels of the local Markdown.
- The related-analysis links to notes 23, 26 and 29 use file names from the list of notes being written in parallel; those notes were not read.
- No experiments were reproduced: no model was run, no Lean file compiled, no proof rechecked and no released dataset or weights inspected. The repository's code was not examined, and section 7 contains proposals only.

---
paper_id: "27"
title: "Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?"
source_version: "2607.01211v2"
source_markdown: "../pdfs/27-performance-benchmark-reliability.md"
source_pdf: "../pdfs/27-performance-benchmark-reliability.pdf"
source_url: "https://arxiv.org/abs/2607.01211"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["benchmark-validity", "benchmarking", "measurement-validity", "repository-optimization", "coding-agents", "reproducibility", "statistics", "evaluation", "scoring-rules", "cross-machine-replay", "leaderboards"]
summary: "Audits SWE-Perf, GSO and SWE-fficiency by replaying 740 reference patches on four Google Cloud CPU machine types and re-scoring public submissions, finding unstable reference signals and scoring-rule-dependent rankings; it covers repository-level CPU benchmarks, not GPU kernels."
---

# Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?

## 1. Paper identity and scope

- This analysis covers arXiv `2607.01211v2` (cs.SE; the PDF carries the stamp 16 Jul 2026), by authors at Singapore Management University and Shanghai Jiao Tong University. [Title block](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 1](../pdfs/27-performance-benchmark-reliability.pdf)
- The work is an audit of released artifacts of three existing benchmarks. The authors state that they do not introduce another benchmark and do not report a new agent leaderboard. [§VIII](../pdfs/27-performance-benchmark-reliability.md)
- The audited benchmarks are GSO (102 tasks from 10 repositories), SWE-Perf (140 tasks from 9 repositories) and SWE-fficiency (498 tasks from 9 repositories), 740 tasks in total. [§II-A; Table I; Abstract](../pdfs/27-performance-benchmark-reliability.md)
- Selection required repository-level edits, runtime optimization judged by executable workloads or tests that compare code states, and a recent, visible peer-reviewed benchmark paper. The authors say they are not surveying every efficiency benchmark. [§II-A](../pdfs/27-performance-benchmark-reliability.md)
- RQ1 asks whether official reference patches remain valid under cross-machine replay; RQ2 asks how leaderboard scoring rules shape submission rankings; RQ3 asks whether benchmark tasks remain hard for top submissions. [§II-B](../pdfs/27-performance-benchmark-reliability.md)
- **This is not a GPU-kernel study.** All replays run on CPU-based Google Cloud general-purpose machines, the tasks are patches to software repositories, and KernelBench appears only as a citation in related work. [§III-A; Table II; §VIII](../pdfs/27-performance-benchmark-reliability.md)
- The benchmark artifacts are the latest public repository versions available before an April 30, 2026 data-collection cutoff; results describe that snapshot. [§III-A; §IV-A](../pdfs/27-performance-benchmark-reliability.md)
- The public leaderboards analysed consist of OpenHands-based submissions, each pairing that scaffold with one foundation model. The paper uses "submission" for the evaluated agent configuration and model names only as labels. [§I](../pdfs/27-performance-benchmark-reliability.md)
- The authors do not report running any agent themselves. RQ1 replays official reference patches against base code; RQ2 and RQ3 work from released outputs of public submissions. [§III-A; §IV-A; §V; §VII](../pdfs/27-performance-benchmark-reliability.md)
- SWE-Perf is covered only by RQ1. It is excluded from the scoring-rule audit and the task-coverage analysis because comparable public agent-output data were not available. [§IV-A; §V-A; §VII](../pdfs/27-performance-benchmark-reliability.md)

## 2. Concepts and terminology

### Task artifacts

- Each task involves a base program (unoptimized code), an official reference patch, and submitted patches, run under workloads so that noisy runtimes can be compared. Functional repair benchmarks such as SWE-Bench instead judge whether tests pass, described as binary and usually reproducible. [§I](../pdfs/27-performance-benchmark-reliability.md)
- The comparison tests differ by benchmark: GSO uses generated performance tests selected to compare base and reference commits; SWE-Perf uses unit tests filtered from the source repositories; SWE-fficiency uses annotated workload scripts comparing pre-edit and expert-optimized commits. [Table I](../pdfs/27-performance-benchmark-reliability.md)

### Original validity rules, as this paper describes them

- **GSO:** each selected generated test must still pass correctness/equivalence checks and keep a replay-time speedup of at least 1.2× over the base commit. For final low-test tasks, tests with speedup above 1.1× are treated as fallback-compatible, matching GSO's low-test construction fallback. [Table III](../pdfs/27-performance-benchmark-reliability.md)
- **SWE-Perf:** each selected efficiency unit test must pass on the original and modified commits. After 20 timing repetitions and IQR outlier filtering, the recomputed Mann–Whitney minimum-gain value must stay above the dataset threshold, \(\delta_i > 0.05\). [Table III](../pdfs/27-performance-benchmark-reliability.md)
- **SWE-fficiency:** the official workload and correctness guards must run successfully, and base mean workload runtime minus expert-patched mean workload runtime must exceed twice the post-edit workload-runtime standard deviation. [Table III](../pdfs/27-performance-benchmark-reliability.md)

### Replay metrics

- **Evaluable** means a reference-patch replay completes and returns a usable score; the per-machine count requires all three rounds on that machine to do so. [§III-B; Table IV](../pdfs/27-performance-benchmark-reliability.md)
- **Faster-than-base** is the minimum direction check: \(T_{\mathit{ref},r,m} < T_{\mathit{base},r,m}\) in every replay round \(r\) and machine \(m\). [§III-A](../pdfs/27-performance-benchmark-reliability.md)
- **Original-rule valid** reapplies the benchmark's own task-inclusion rule to the replay. A task is **replay-valid** only when the reference patch satisfies that rule in all 12 machine-round replays, which the authors call a strict, conservative all-replay rule. [§III-A; §III-C; §VII](../pdfs/27-performance-benchmark-reliability.md)
- **Runtime change percentage** is the common scale for signal size; negative means the reference is faster than base. For GSO and SWE-fficiency a speedup \(s = T_{\mathit{base}}/T_{\mathit{ref}}\) becomes \(1/s - 1\), so 1.20× becomes −16.7%. For SWE-Perf the paper instead flips the sign of the benchmark's significance-aware runtime-reduction score, so that column sits on a different underlying quantity. [§III-D](../pdfs/27-performance-benchmark-reliability.md)
- **Std./signal** is median within-task standard deviation divided by the absolute median runtime change; larger values mean replay variation is large relative to the measured change. [§III-D](../pdfs/27-performance-benchmark-reliability.md)

### Scoring rules, as this paper describes them

- **GSO, binary reference-level gate:** \(\mathrm{OPT@1}(m) = 100 \cdot \#\text{reference-level successes} / N\), where a success is a correct submitted patch whose speedup matches or exceeds the official reference patch. One success is worth \(100/N\) points (0.98 for 102 tasks), and correct but below-reference speedups receive no partial credit. [§IV-A](../pdfs/27-performance-benchmark-reliability.md)
- **SWE-fficiency, SpeedUp Ratio with harmonic mean:** \(\mathrm{SR}_{m,i} = \mathrm{speedup}_{m,i} / \mathrm{speedup}_{ref,i}\), aggregated after flooring each SR at 0.001 as \(\mathrm{HM}(m) = N / \sum_i 1/\max(\mathrm{SR}_{m,i}, 0.001)\). SR below 1 means slower than the reference patch. [§IV-A](../pdfs/27-performance-benchmark-reliability.md)
- The released SWE-fficiency task outputs treat failed or incorrect patches as no effective edit, which the paper says can place them far below the reference patch. [§IV-A](../pdfs/27-performance-benchmark-reliability.md)
- In denominator units, a reference-matching task contributes 1. Beating the reference with \(\mathrm{SR}=s>1\) saves \(1-1/s\) units (0.5 at SR 2, 0.9 at SR 10), so the reward is capped below one unit. A below-reference task adds \(1/\mathrm{SR}-1\) units: 1 at SR 0.5, 99 at SR 0.01, and 999 at the official floor. [§IV-A](../pdfs/27-performance-benchmark-reliability.md)
- The paper does not describe SWE-Perf's leaderboard scoring rule; only its task-validity rule appears. [Table III; §IV-A](../pdfs/27-performance-benchmark-reliability.md)
- **Score weight** of a task is its share of the harmonic-mean denominator, \(w_i = \frac{1/\max(\mathrm{SR}_i, 0.001)}{\sum_j 1/\max(\mathrm{SR}_j, 0.001)}\). [§IV-C](../pdfs/27-performance-benchmark-reliability.md)
- The **bounded-penalty diagnostic** keeps harmonic aggregation but raises the floor to 0.5, so a very low-SR task contributes at most two denominator units, one more than a neutral task. [§IV-D](../pdfs/27-performance-benchmark-reliability.md)

### Ranking and coverage terms

- **Pair flips** count disagreements among the \(\binom{8}{2}=28\) unique head-to-head pairs of the eight shared submissions; Spearman correlation compares two rank orderings. [§IV-B; Table VII](../pdfs/27-performance-benchmark-reliability.md)
- **Coverage across 10 public submissions** records, per task, whether at least one submission reaches each of three milestones: passing tests, beating the unmodified base program, and matching or beating the reference patch. [§V-A](../pdfs/27-performance-benchmark-reliability.md)
- The **best public patch** for a task is, among the 10 public submissions, the patch with the largest replayed speedup relative to the reference patch. The remaining gap is best-public speedup divided by reference-patch speedup, a ratio of speedups. [§V-B; §V-C](../pdfs/27-performance-benchmark-reliability.md)
- **Category alignment** marks whether the best public patch has the same high-level optimization category as the reference patch, a different category, or no visible production optimization. [§V-C](../pdfs/27-performance-benchmark-reliability.md)

## 3. Methodology and experimental setup

### RQ1: cross-machine replay of reference patches

- Every official reference patch is replayed on four Google Cloud machine profiles with 64 vCPUs and 256 GB of memory: `n2-standard-64` (Intel Cascade Lake, 2019), `n2d-standard-64` (AMD Milan, 2021), `n4-standard-64` (Intel Emerald Rapids, 2024) and `n4d-standard-64` (AMD Turin, 2025). Years are Google Cloud release years. [§III-A; Table II](../pdfs/27-performance-benchmark-reliability.md)
- The resource configuration matches the `n2-standard-64` setup that GSO and SWE-fficiency reported in their original experiments. Cloud provider and resource shape are held fixed while the processor platform varies. [§III-A](../pdfs/27-performance-benchmark-reliability.md)
- Each task is replayed for three rounds on each machine, giving 12 machine-round replays per task. Derived: 740 × 12 is 8,880 planned task replays, a total the paper does not print. [§I; Fig. 1](../pdfs/27-performance-benchmark-reliability.md)
- The replay reuses the official tasks, reference patches, selected tests and workload definitions. The stated goal is to test the original signal, not to impose a stricter benchmark. [§III; §III-A](../pdfs/27-performance-benchmark-reliability.md)
- Denominators differ by display. Fig. 1 uses the official task counts, with not-evaluable tasks counted as failing the timing checks. Table V uses 98, 138 and 498 tasks, which equal the official counts minus the not-evaluable tasks. [§III-C; §III-D; Tables IV and V](../pdfs/27-performance-benchmark-reliability.md)

### RQ2: scoring-rule audit

- Only benchmarks exposing public submissions, per-task outcomes and final ranking scores are included: GSO and SWE-fficiency as of the cutoff. [§IV-A](../pdfs/27-performance-benchmark-reliability.md)
- The comparison uses the eight public submissions that appear on both leaderboards, labelled Claude Opus 4.6, GPT-5.2, Claude Opus 4.5, Gemini 3 Pro, Claude Sonnet 4.5, Gemini 3 Flash, GPT-5 and Gemini 2.5 Pro. [§IV-B; Table VI](../pdfs/27-performance-benchmark-reliability.md)
- Per-task outputs are held fixed and only the aggregation rule changes: SWE-fficiency outputs are rescored with the GSO gate, and "available" GSO outputs with the harmonic rule. The authors treat the results as diagnostics, not replacement rankings. [§IV-B](../pdfs/27-performance-benchmark-reliability.md)
- Tail dominance is measured by sorting tasks by \(w_i\) within each shared submission and summing the weight of the worst 1, 5 and 10 tasks. [§IV-C](../pdfs/27-performance-benchmark-reliability.md)

### RQ3: task coverage across public submissions

- The task set is the replay-valid subset: 39 GSO tasks plus 411 SWE-fficiency tasks, 450 in total. SWE-Perf's 11 replay-valid tasks are excluded. [§V-A](../pdfs/27-performance-benchmark-reliability.md)
- For each task the authors look across 10 public submissions; Fig. 4's axis calls these "any top-10 run". The paper does not list the ten or state how they were chosen. [§V-A; Fig. 4](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 8](../pdfs/27-performance-benchmark-reliability.pdf)
- The authors describe the result as an optimistic estimate of coverage by at least one strong public submission, not a claim that one submission reaches the reference on all tasks. [§V](../pdfs/27-performance-benchmark-reliability.md)
- The 66 tasks where no submission reaches the reference are studied through their best public patch. The text says "replayed speedup" without stating whether the authors re-timed public patches or used released evaluation records, or on which machine. [§V-B](../pdfs/27-performance-benchmark-reliability.md)
- Strategy labels use the high-level categories of Peng et al.'s taxonomy, assigned by that group's automated annotation script with GPT-5.5 as annotation model, one category per patch. Labels are schema-checked, and the authors say they manually sanity-check them against patch diffs; no agreement statistic is given. [§V-C](../pdfs/27-performance-benchmark-reliability.md)

## 4. Results and evidence

### RQ1.1 Replay evaluability

- Tasks not evaluable: GSO 4/102, SWE-Perf 2/140, SWE-fficiency 0/498. Complete-by-machine counts in n2/n2d/n4/n4d order are 99/98/99/99 for GSO, 138 on every machine for SWE-Perf and 498 on every machine for SWE-fficiency. [§III-B; Table IV](../pdfs/27-performance-benchmark-reliability.md)
- The four GSO failures are one functional-equivalence failure, one missing external-data dependency, one image-decoding failure and one x86_64 CPU-instruction portability failure; both SWE-Perf failures are missing dependencies in the evaluation image. The paper does not map failures to machines. [§III-B](../pdfs/27-performance-benchmark-reliability.md)

### RQ1.2 Reference validity under replay

- Reference patches faster than base in all 12 machine-round replays: 91/102 GSO (89%), 48/140 SWE-Perf (34%), 470/498 SWE-fficiency (94%). All three benchmarks therefore contain reference patches that are slower than base in at least one replay. [§III-C; Fig. 1; Finding RQ1.2](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 3](../pdfs/27-performance-benchmark-reliability.pdf)
- Reference patches satisfying the original validity rule in all 12 replays: 39/102 GSO (38%), 11/140 SWE-Perf (8%), 411/498 SWE-fficiency (83%). For SWE-Perf the paper states that 129 reference patches no longer show statistically supported gain across machines. [§III-C; Fig. 1](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 3](../pdfs/27-performance-benchmark-reliability.pdf)
- Derived: the complements are 63 GSO and 87 SWE-fficiency tasks that are not replay-valid, and 461/740 (62.3%) tasks are replay-valid overall. The paper prints neither figure. [§III-C; Fig. 1](../pdfs/27-performance-benchmark-reliability.md)
- No per-machine or per-round breakdown of validity failures is reported; only the all-12 conjunction is. [§III-C; Fig. 1](../pdfs/27-performance-benchmark-reliability.md)

### RQ1.3 Signal size versus replay variation

- Median runtime change, median within-task standard deviation and std./signal are: GSO (98 tasks) −54.20%, 3.81 pp, 0.07×; SWE-Perf (138 tasks) −0.03%, 1.41 pp, 43.23×; SWE-fficiency (498 tasks) −56.04%, 2.41 pp, 0.04×. [Table V](../pdfs/27-performance-benchmark-reliability.md)
- The within-task standard deviation is taken over the 12 machine-round replays, so it mixes machine-to-machine and round-to-round variation. SWE-Perf has the lowest value of the three, which the paper says rules out a simple "more machine noise" explanation. [§III-D](../pdfs/27-performance-benchmark-reliability.md)
- 101/138 SWE-Perf tasks have median changes within five percentage points of zero. Fig. 2's caption describes the same concentration near zero; no numbers were read from that figure. [§III-D; Fig. 2](../pdfs/27-performance-benchmark-reliability.md)

### RQ2.1 Cross-benchmark ranking sensitivity

- Official GSO scores, by GSO rank: Claude Opus 4.6 41.18, GPT-5.2 27.45, Claude Opus 4.5 26.47, Gemini 3 Pro 18.63, Claude Sonnet 4.5 14.71, Gemini 3 Flash 9.80, GPT-5 6.86, Gemini 2.5 Pro 3.92. [Table VI](../pdfs/27-performance-benchmark-reliability.md)
- Official SWE-fficiency scores, by SWE-fficiency rank: Claude Opus 4.5 0.2250, GPT-5 0.1571, Claude Opus 4.6 0.1553, GPT-5.2 0.1482, Claude Sonnet 4.5 0.1162, Gemini 3 Flash 0.1056, Gemini 3 Pro 0.1024, Gemini 2.5 Pro 0.0313. [Table VI](../pdfs/27-performance-benchmark-reliability.md)
- The two official leaderboards disagree on 9 of 28 pairs, with Spearman 0.452. The GPT-5-labelled submission moves five positions (GSO rank 7, SWE-fficiency rank 2). [§IV-B; Tables VI and VII](../pdfs/27-performance-benchmark-reliability.md)
- Rescoring SWE-fficiency outputs with a GSO-style reference-level pass rate raises Spearman with GSO to 0.762 and reduces discordant pairs to 6/28, with maximum rank movement 3. [§IV-B; Table VII](../pdfs/27-performance-benchmark-reliability.md)
- Rescoring available GSO per-task reports with SWE-fficiency's harmonic rule gives Spearman 0.238 against the official SWE-fficiency ranking and 11/28 flips, with maximum movement 5. [§IV-B; Table VII](../pdfs/27-performance-benchmark-reliability.md)
- The paper's reading is that the scoring rule explains part, not all, of the mismatch; the task set and per-task outputs also matter. [§IV-B](../pdfs/27-performance-benchmark-reliability.md)

### RQ2.2 Low-speedup tail dominance in SWE-fficiency

- Across the eight shared submissions, the worst single task carries 6.3–33.6% of the official score denominator, the worst five 31.4–73.1%, and the worst ten 58.5–82.8%. The benchmark has 498 tasks. [§IV-C; Fig. 3](../pdfs/27-performance-benchmark-reliability.md)
- One near-floor task in the Claude Opus 4.5 submission has raw SR 0.00134 and carries 33.6% of that submission's score weight. [§IV-C](../pdfs/27-performance-benchmark-reliability.md)
- Derived from Fig. 3's SVG bar geometry, not printed in the source: approximate worst-1/worst-5/worst-10 weights are Opus 4.5 33.6/61.7/67.8%, GPT-5 31.6/60.2/68.7%, Opus 4.6 31.2/73.1/78.3%, GPT-5.2 29.8/71.5/76.9%, Sonnet 4.5 23.3/64.2/72.4%, Gemini Flash 21.2/66.8/77.9%, Gemini Pro 20.6/71.9/82.8%, Gemini 2.5 6.3/31.4/58.5% (figure labels). The extremes reproduce the six range endpoints in the prose. [Fig. 3](../pdfs/27-performance-benchmark-reliability.md)
- Derived: the submission with the lowest tail concentration in Fig. 3 (Gemini 2.5) is also the lowest-scoring one in Table VIII (median SR 0.497, 38 tasks with SR above 1), so low concentration does not indicate good performance. [Fig. 3; Table VIII](../pdfs/27-performance-benchmark-reliability.md)

### RQ2.3 Bounded-penalty diagnostic

- With floor 0.5, harmonic mean / median SR / count of tasks with SR above 1 (of 498) are: Claude Opus 4.6 0.952 / 1.006 / 258; GPT-5.2 0.944 / 1.004 / 259; Claude Opus 4.5 0.904 / 0.986 / 212; Gemini 3 Pro 0.888 / 0.993 / 235; GPT-5 0.867 / 0.950 / 228; Gemini 3 Flash 0.818 / 0.919 / 185; Claude Sonnet 4.5 0.767 / 0.858 / 164; Gemini 2.5 Pro 0.613 / 0.497 / 38. [Table VIII](../pdfs/27-performance-benchmark-reliability.md)
- Relative to the official SWE-fficiency ranking, the diagnostic changes six of eight ranks and flips 8/28 pairs: Claude Opus 4.6 and GPT-5.2 move above Claude Opus 4.5, and GPT-5 drops from rank 2 to rank 5. The authors state that the point is not that 0.5 is the right constant. [§IV-D; Table VIII](../pdfs/27-performance-benchmark-reliability.md)
- Derived: the floor-0.5 order matches the official GSO order in ranks 1–4, 6 and 8; only GPT-5 and Claude Sonnet 4.5 swap between ranks 5 and 7. The paper does not make this comparison. [Tables VI and VIII](../pdfs/27-performance-benchmark-reliability.md)

### RQ3.1 Coverage across 10 public submissions

- On the 450 replay-valid tasks, every task has at least one passing public patch, and 449/450 (99.8%) have a passing patch that beats the base program. Fig. 4 prints 100.0% for GSO and 99.8% for SWE-fficiency on the second milestone, which places the one exception in SWE-fficiency. [§V-A; Abstract; Fig. 4](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 8](../pdfs/27-performance-benchmark-reliability.pdf)
- At least one of the 10 submissions matches or beats the reference on 384/450 (85.3%) tasks. Fig. 4 prints 76.9% for GSO and 86.1% for SWE-fficiency. [§V-A; Abstract; Fig. 4](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 8](../pdfs/27-performance-benchmark-reliability.pdf)
- 66 tasks do not reach reference speed: 9/39 GSO and 57/411 SWE-fficiency. [§V-A](../pdfs/27-performance-benchmark-reliability.md)

### RQ3.2 The 66 below-reference tasks

- All 66 tasks have a correct public patch. For 65/66 the best patch runs faster than base; one best patch passes tests without improving runtime. [§V-B](../pdfs/27-performance-benchmark-reliability.md)
- Across the 660 attempts behind these 66 tasks, 160 fail functionality or validation, 3 do not complete, and 46 pass correctness without improving runtime. Derived: the other 451 attempts are not classified in the text. [§V-B](../pdfs/27-performance-benchmark-reliability.md)
- For GSO (9 tasks) the best public patch reaches a median 85.3% of the reference patch's speedup, with 1.17× more speedup stated as needed to match. For SWE-fficiency (57 tasks) the median is 87.9%, leaving 1.14×. [§V-B](../pdfs/27-performance-benchmark-reliability.md)
- SWE-fficiency bands of best/reference speedup: 27 tasks at 90–100%, 21 at 50–90%, 8 at 10–50%, 1 below 10%. The 21 is a figure label; the other three are also in the prose. [§V-B; Fig. 5](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 8](../pdfs/27-performance-benchmark-reliability.pdf)
- GSO bands: 6 tasks at 50–90% and 2 at 10–50% are labelled in Fig. 5. Derived from the SVG bar geometry: the unlabelled ninth task lies in the 90–100% band and none is below 10%. [Fig. 5](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 8](../pdfs/27-performance-benchmark-reliability.pdf)

### RQ3.3 Strategy alignment

- Reference-patch categories over the 66 tasks, with same-category counts: Algorithm 21/31, Structure 7/16, Memory 3/12, Build 0/3, Control 1/3, Data structure 0/1; overall 32/66. [Table IX](../pdfs/27-performance-benchmark-reliability.md)
- The other 34/66 use a different category or show no visible production optimization. The speedup comparison excludes two cases without visible production optimization, leaving 32 same-category and 32 different-category pairs. [§V-C; Table IX](../pdfs/27-performance-benchmark-reliability.md)
- Median best/reference speedup is 89.8% for same-category and 81.1% for different-category pairs, with stated remaining multipliers of 1.12× and 1.23×. [§V-C](../pdfs/27-performance-benchmark-reliability.md)
- 16/32 same-category cases remain below 90% of reference speedup, including four below 50%; 11/32 different-category cases reach 90–100%. [§V-C; Fig. 6](../pdfs/27-performance-benchmark-reliability.md)
- Fig. 6 labels: same-category 4 / 12 / 16 for the 10–50%, 50–90% and 90–100% bands; different-category 6 / 14 / 11. Derived from the SVG bar geometry: the unlabelled 32nd different-category case lies below 10%. [Fig. 6](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 9](../pdfs/27-performance-benchmark-reliability.pdf)

## 5. Authors' conclusions

### Inferences

- Most reference optimizations are replayable, so the main issue is not basic evaluability; but runnable does not imply benchmark-valid, and some official tasks lose their performance signal on other machines. [Finding RQ1.1; §I; §III-E](../pdfs/27-performance-benchmark-reliability.md)
- SWE-Perf's replay failures are mainly a small-signal problem, not a larger-noise problem. The authors offer benchmark construction as "one likely reason": SWE-Perf derives efficiency checks from existing unit tests and accepts gains above a 5% threshold, while the other two rely on more explicitly performance-oriented workloads or generated stress tests. No experiment isolates this. [Finding RQ1.3; §III-E](../pdfs/27-performance-benchmark-reliability.md)
- Ranking results do not directly measure agent capability; they combine task outcomes with how the benchmark turns them into one score. [§IV-E](../pdfs/27-performance-benchmark-reliability.md)
- SWE-fficiency is best read as a strict, tail-sensitive score. The authors also say penalizing severe slowdowns is a reasonable benchmark choice and frame the issue as interpretability. [§IV-C; §IV-E](../pdfs/27-performance-benchmark-reliability.md)
- The remaining 66 tasks are rarely basic correctness failures; the gap is about reaching reference-patch speed. [§V-D; §I](../pdfs/27-performance-benchmark-reliability.md)
- Strategy mismatch is not the main reason tasks stay below reference; category alignment is a partial signal, and the authors attribute the gap often to implementation depth. [§V-C; Finding RQ3.3](../pdfs/27-performance-benchmark-reliability.md)
- The authors suggest that, without seeing the reference patch, runs may stop short of the extra iteration needed to reach reference speed. This is an interpretation, not a measurement. [§V-D](../pdfs/27-performance-benchmark-reliability.md)
- For agent builders, the room left for "more tasks reach the reference" is narrow, so a measured improvement may depend heavily on a small set of tasks. The benchmarks remain useful for studying the step from faster-than-base to reference speed, but less suitable as sole evidence of general performance-engineering ability. [§VI](../pdfs/27-performance-benchmark-reliability.md)

### Recommendations

- Future benchmarks should validate that workloads stress the optimized path strongly enough and that accepted reference patches have clear margins over runtime noise. [§III-E](../pdfs/27-performance-benchmark-reliability.md)
- At minimum, reports should distinguish cross-machine verified tasks from unstable tasks, show per-task score weight, and compare submissions under aggregation rules that match the use case: strong worst-case penalties for some users, median speedup or reference-level coverage for others. [§VI](../pdfs/27-performance-benchmark-reliability.md)
- As public submissions improve, future benchmarks may need stronger reference targets to keep separating submissions. [§I](../pdfs/27-performance-benchmark-reliability.md)
- Future benchmarks should keep executable replay and add hotspot localization from profiles or traces, ambiguous optimization targets, validation against workloads not fully visible during patch search, and resource metrics beyond one runtime number, since optimizing one resource can hurt another. [§VI; §IX](../pdfs/27-performance-benchmark-reliability.md)

## 6. Limitations and open questions

### Threats to validity stated by the source

- **Internal:** replays and metric recomputations may be affected by implementation choices, hardware variation and released artifact quality. The all-12-replay rule is conservative and may undercount usable tasks; the authors accept this to protect later analyses from unstable reference signals. [§VII](../pdfs/27-performance-benchmark-reliability.md)
- **External:** results cover three recent repository-level benchmarks and specific public leaderboard snapshots, not all coding-agent or performance-engineering settings; future models, scaffolds, hardware and benchmark policies may change the exact counts. [§VII](../pdfs/27-performance-benchmark-reliability.md)
- **External:** the 10-submission analysis is a task-coverage proxy, not a newly engineered multi-agent workflow or a claim about any single submitted configuration. [§VII](../pdfs/27-performance-benchmark-reliability.md)
- **Data construction:** the study relies on each benchmark's released tasks, reference patches, public submissions and scoring records, and excludes SWE-Perf from RQ2 and RQ3. [§VII](../pdfs/27-performance-benchmark-reliability.md)

### Analyst caveats and missing evidence

- **Cross-machine versus same-machine variation is not separated.** Validity is reported only as a conjunction over 12 machine-rounds, so the paper does not show how many tasks would fail on repeated rounds of the original `n2-standard-64` alone. [Basis: §III-C; Fig. 1; Table V](../pdfs/27-performance-benchmark-reliability.md)
- **Conjunction strictness.** Requiring a pass in all 12 replays lowers the pass count mechanically as replays are added. The 39/11/411 counts are specific to this 4×3 design and are not per-replay failure rates. [Basis: §VII; Fig. 1](../pdfs/27-performance-benchmark-reliability.md)
- **"Round" is not defined operationally.** The text does not say whether rounds use fresh VM instances, how far apart they run, or what within-round repetition applies beyond SWE-Perf's 20 timing repetitions. [Basis: §III-A; Table III](../pdfs/27-performance-benchmark-reliability.md)
- **Mixed scales in Table V.** SWE-Perf's column is a sign-flipped significance-aware reduction score, while the other two are converted raw speedups. The 43.23× versus 0.07×/0.04× contrast therefore compares different quantities, although the within-SWE-Perf observation (101/138 within 5 pp of zero) stands on its own. [Basis: §III-D; Table V](../pdfs/27-performance-benchmark-reliability.md)
- **Cloud and CPU scope.** One provider, one resource shape and four x86 platforms were tested. Nothing is measured on GPUs, bare metal, other providers or ARM. [Basis: §III-A; Table II](../pdfs/27-performance-benchmark-reliability.md)
- **Small ranking sample with confounded sources.** RQ2 rests on eight shared submissions (28 pairs) at one snapshot with no uncertainty reported. The two leaderboards differ in tasks and repositories as well as scoring rule, and the "available" GSO outputs are not enumerated, so the 0.238 rescoring result may rest on incomplete per-task data. [Basis: §IV-B; Table VII](../pdfs/27-performance-benchmark-reliability.md)
- **Which tasks dominate the harmonic mean (derived).** If "no effective edit" means a speedup of 1, a failed patch's SR equals the reciprocal of the reference speedup. An SR near 0.001 can then arise from a task whose reference speedup is in the hundreds, not only from a severe slowdown. The paper does not report which case applies to the near-floor tasks. [Basis: §IV-A; §IV-C](../pdfs/27-performance-benchmark-reliability.md)
- **Ratio of speedups compresses small references (derived).** A patch with no speedup scores \(1/s_{ref}\) on the best/reference ratio, about 83% against a 1.2× reference. A high percentage of reference speedup therefore does not by itself show a large optimization; read the 85.3% and 87.9% medians with that in mind. [Basis: §V-B; §V-C](../pdfs/27-performance-benchmark-reliability.md)
- **RQ3.3 conditions on the outcome and uses model labels.** Categories were annotated only for the 66 tasks that miss the reference, not for the 384 that reach it, so the data cannot estimate how alignment relates to reaching the reference in general. The labels are one coarse GPT-5.5 judgment per patch with an unquantified manual check, not an executable validation. [Basis: §V-C; Table IX](../pdfs/27-performance-benchmark-reliability.md)
- **Coverage uses hindsight.** Any-of-10 selection picks the best released output per task after the fact. It says nothing about what a selector without the reference patch or the benchmark's timing harness would pick. [Basis: §V; §VII](../pdfs/27-performance-benchmark-reliability.md)
- **"Matches" has no stated tolerance.** The paper describes GSO's gate as matching or exceeding the reference without a numerical tolerance, and does not say how the RQ3 comparison handles timing noise at the boundary or where the public-patch speedups were measured. [Basis: §IV-A; §V-A; §V-B](../pdfs/27-performance-benchmark-reliability.md)
- **Number collision.** 85.3% appears twice with different meanings: 384/450 tasks reaching the reference, and GSO's median best/reference speedup on its 9 below-reference tasks. [Basis: Abstract; §V-B](../pdfs/27-performance-benchmark-reliability.md)
- **Open question.** The paper does not test whether a larger margin rule, more repetitions, or per-machine baselines would restore validity for the failing tasks. [Basis: §III-E; §VI](../pdfs/27-performance-benchmark-reliability.md)

### Source inconsistencies and conversion problems

- **Source arithmetic note (rounding level):** Table V prints std./signal 43.23× for SWE-Perf, but the printed inputs give 1.41/0.03 = 47.0. The table value is consistent with an unrounded median near −0.033%. Use the table-specific 43.23× and do not recompute. [Table V](../pdfs/27-performance-benchmark-reliability.md)
- **Source arithmetic note (rounding level):** the same-category remaining multiplier is printed as 1.12×, while 1/0.898 = 1.114. The other three multipliers (1.17×, 1.14×, 1.23×) match the reciprocals of their printed medians. The cause is not stated. [§V-B; §V-C](../pdfs/27-performance-benchmark-reliability.md)
- **Source labelling ambiguity:** Table IX's "Diff. cat." column totals 34 and includes the two cases with no visible production optimization, while Figs. 6 and 7 use 32 different-category cases. The table does not show which reference categories those two cases belong to. [Table IX; §V-C; Fig. 6](../pdfs/27-performance-benchmark-reliability.md)
- **Unexplained cases (derived):** assuming both figures define the bands identically, subtracting Fig. 6's band counts from Fig. 5's leaves one excluded case in the 50–90% band and one in the 90–100% band. The paper does not explain how a patch with no visible production optimization reaches those bands; the ratio-compression caveat above is one possible reading. [Figs. 5 and 6; §V-B](../pdfs/27-performance-benchmark-reliability.md)
- **Source wording:** Finding RQ3.3 says "32 same-category public patches remain slower", which is true of all 66 by construction; the informative comparison is 16/32 versus 11/32 at the 90% line. [Finding RQ3.3; §V-C](../pdfs/27-performance-benchmark-reliability.md)
- **Citation mismatch in the source:** the first related-work sentence names seven works (PIE through ACECode) but carries eight citations; the last, [44], is a paper on mutation-guided test generation that matches none of the names. [§VIII; References](../pdfs/27-performance-benchmark-reliability.md)
- **Conversion problem:** Tables I, II and VI contain duplicated or orphaned rows from nested HTML cells in the Markdown (for example repeated "GSO" / "NeurIPS 2025" rows). Values were cross-checked against the PDF text of pages 2, 3 and 6. [Tables I, II and VI](../pdfs/27-performance-benchmark-reliability.md); [PDF p. 2](../pdfs/27-performance-benchmark-reliability.pdf)

## 7. Relevance to this project — interpretation

- **Proposal:** treat this source as evidence about CPU repository benchmarks only, and measure the project's own GPU timing noise floor with its measurement tool before fixing any threshold. Do not reuse the 1.41–3.81 pp replay deviations as GPU numbers. [Basis: Table V; §III-A; §VII](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** when Stage 2 records a reference implementation's performance characteristics, also record repeated-run variation and a signal-to-variation ratio for the reference against its baseline, and flag oracles whose margin is close to zero as low-signal before the loop starts. [Basis: §III-D; Table V; §III-E](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** have Stage 3 decide "is it performant" with a margin rule fixed in advance and tied to measured variation (for example a multiple of the standard deviation or a minimum ratio), not a bare faster-than comparison, and record "not distinguishable" as its own outcome. [Basis: Table III; §III-C; Fig. 1](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** measure baseline, reference and candidate in the same session on the same device, store the device identity with each timing, and do not compare a candidate's timing with a reference timing taken on another machine or round. [Basis: §III-A; Table II; Finding RQ1.2](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** keep three separate fields per candidate: passes correctness, faster than baseline, matches or beats reference. The source shows these milestones diverge sharply and only the last one still separates submissions. [Basis: §V-A; Fig. 4](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** when several shapes or workloads are folded into one kernel score, publish each workload's contribution to the aggregate, state the floor or penalty used for failures, and check whether the pool ranking survives at least two aggregation rules, such as threshold count, median ratio and bounded harmonic mean. [Basis: §IV-A; §IV-C; Table VIII](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** keep candidates that fail fuzzing out of speed aggregates and report them as a separate count, so that a correctness failure does not act as an extreme slowdown in a mean. [Basis: §IV-A; §IV-D](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** report best-of-batch or best-of-pool results as optimistic coverage with the number of attempts stated, alongside attempt-level failure counts, and do not present them as single-attempt capability. [Basis: §V; §V-B; §VII](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** log best/reference ratio per round for pool members that are correct and faster than baseline but below reference, and test whether extra rounds on those members close the gap. If candidates routinely match the reference, it no longer separates them and a stronger comparison target is worth considering. The source's "implementation depth" reading comes from CPU tasks and would need this check. [Basis: §V-B; §V-C; §V-D; §I](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** for Stage 4's full performance results, repeat the measurement in several independent rounds, report whether the gain holds in every round as well as in aggregate, and state the hardware scope explicitly when only one GPU model was tested. Alongside any ratio to the reference, report absolute runtimes and the reference's own speedup over baseline, since a ratio of speedups can look high when the reference margin is small. [Basis: §III-A; §III-D; §V-B; §VII](../pdfs/27-performance-benchmark-reliability.md)
- **Proposal:** pin and verify harness dependencies, external data and instruction-set assumptions before a measurement campaign, since the source's replay failures came from those. [Basis: §III-B; Table IV](../pdfs/27-performance-benchmark-reliability.md)

## 8. Retrieval notes

- Search aliases: performance-optimization benchmark reliability, SWE-Perf, GSO, SWE-fficiency, reference-patch replay, cross-machine replay, replay-valid, original-rule valid, faster-than-base, std./signal, OPT@1, SpeedUp Ratio, harmonic mean floor 0.001, bounded-penalty diagnostic, tail dominance, score weight, any-of-10 coverage, best public patch.
- Questions answered: How many reference patches of each benchmark stay valid on four cloud machine types over three rounds? What does replay-valid mean? Why does SWE-Perf fail most often? How do GSO's and SWE-fficiency's scoring rules differ at task level? How much weight do the worst ten tasks carry in SWE-fficiency's harmonic mean? What changes when the floor is raised to 0.5? How many replay-valid tasks does at least one of 10 public submissions solve to reference level? Is strategy mismatch the reason for the remaining gap? What do the authors recommend to benchmark users and designers?
- Related local analyses: [robust benchmarking in noisy environments](33-robust-benchmarking-noisy-environments.md) for timing-noise methodology; [ISO-Bench](07-iso-bench.md), which this paper cites among repository-scale performance benchmarks; [KernelBench](01-kernelbench.md), cited here only in related work; [KernelBench-Verified](03-kernelbench-verified.md) for baseline-choice effects on GPU-kernel speedup claims; [robust agentic CUDA](08-robust-agentic-cuda.md) for GPU-kernel benchmark hardening; [reproducible LM evaluation](14-reproducible-lm-evaluation.md) for evaluation-reporting practice; [METR KernelAgent](21-measuring-automated-kernel-engineering.md) for measuring kernel-engineering agents; [SWE-agent](10-swe-agent.md) for the repository-agent setting. These links are navigation aids, not controlled comparisons; none of the linked notes shares this paper's tasks, hardware or metrics.

## 9. Source coverage and verification notes

- Read the complete preserved Markdown: abstract, §§I–IX, all nine tables, all seven figure captions, the data-availability statement and the reference list. The paper has no appendix.
- The PDF could not be opened by the file viewer, so its text was extracted programmatically for all 12 pages and compared with the Markdown. Body text ends on page 11 and no appendix or extra section appears in the PDF. `PDF p. N` locators come from that extraction.
- The figure assets are SVGs with text stored as glyph references, and no raster rendering was produced. Figures were checked by parsing the SVG files: embedded label text and bar-rectangle coordinates were read for Figs. 1, 3, 4, 5 and 6.
- Verified this way: Fig. 1's nine counts and percentages; Fig. 4's six percentages; the band labels of Figs. 5 and 6 and their assignment to rows. Unlabelled one-count segments in Figs. 5 and 6 and the per-submission values of Fig. 3 are derived from bar lengths against the axis and are marked as derived above.
- Not verified visually: Fig. 2 (density curves by machine and round) and Fig. 7 (per-pair distribution with 90% and 50% guide lines). Only their axis labels, legends and captions were read; no values were taken from them.
- Arithmetic checked against printed values: task totals (740, 450, 66, 660), percentages in the abstract, the 9/28 discordant pairs and Spearman 0.452 from Table VI ranks, the 6/8 rank changes and 8/28 flips from Table VIII, Table IX column sums, and Table V ratios. Discrepancies are recorded in section 6 and not corrected.
- Not read: the three benchmark papers whose rules are summarized in Table III and §IV-A, the authors' GitHub data repository named under Data Availability, and the public leaderboards. Descriptions of the original validity and scoring rules are this paper's wording and were not checked against those sources.
- No experiments were reproduced: no reference patch was replayed, no leaderboard rescored, and no annotation rerun.

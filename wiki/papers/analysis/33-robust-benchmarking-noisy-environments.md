---
paper_id: "33"
title: "Robust benchmarking in noisy environments"
source_version: "1608.04295v1"
source_markdown: "../pdfs/33-robust-benchmarking-noisy-environments.md"
source_pdf: "../pdfs/33-robust-benchmarking-noisy-environments.pdf"
source_url: "https://arxiv.org/abs/1608.04295"
analysis_date: "2026-11-21"
analysis_status: "draft"
topics: ["benchmarking", "statistics", "measurement-validity", "reproducibility", "timing-noise", "minimum-estimator", "microbenchmarks", "cpu-timing", "performance-regression", "julia"]
summary: "Models CPU timing noise as non-negative delays, argues for the minimum as run-time estimator, and gives a heuristic for executions per measurement in Julia's BenchmarkTools; evidence is four Julia microbenchmarks on undescribed hardware, with no decision rule or GPU analysis."
---

# Robust benchmarking in noisy environments

## 1. Paper identity and scope

- This note covers arXiv `1608.04295v1` by Jiahao Chen and Jarrett Revels (MIT CSAIL); the PDF is stamped `[cs.PF] 15 Aug 2016`. [Title block](../pdfs/33-robust-benchmarking-noisy-environments.md); [PDF p. 1](../pdfs/33-robust-benchmarking-noisy-environments.pdf)
- The problem is a benchmarking methodology for continuous-integration (CI) pipelines and manual workflows, especially for benchmarks short enough (microseconds or less) that timer accuracy corrupts measurements. [§I](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Contributions: a model of how environmental delays shape timing distributions, an argument for the minimum as run-time estimator, a heuristic (Alg. 1) for the number of executions per measurement, and an implementation in the Julia package BenchmarkTools. [Abstract; §III; §IV; §V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The work studies CPU wall-clock timing of deterministic Julia programs "run in serial". It contains no GPU timing, no parallel programs and no language models. "Language- and platform-agnostic" is a design goal; all measurements are Julia benchmarks. [§I; §III; §V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The paper defines no statistical test, confidence interval or decision rule for declaring one program faster than another. It supplies an estimator and a repetition count, not a comparison procedure. [§IV; §V; §VI](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The authors claim, to their knowledge, the first methodology that is fully automatable, robust to non-i.i.d. statistics and efficient within a time budget. This is a novelty claim, not a measured comparison. [§I-C](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 2. Concepts and terminology

### Units of measurement

- An **execution** is one run of a benchmarkable program $P_0$, a tape of instructions. A **timing measurement** $T_i$ is the observed time for $n_i$ executions; the per-execution value is $T_i/n_i$. [§II](../pdfs/33-robust-benchmarking-noisy-environments.md)
- A **benchmark experiment** is a recipe for obtaining several measurements; executing it once yields a **trial**, the collection $T^{\{i\}}_1,\dots,T^{\{i\}}_k$. [§II](../pdfs/33-robust-benchmarking-noisy-environments.md)
- "Sample" appears informally in §IV-B for the measurements at hand and is not defined in §II. The levels to keep apart are executions per measurement ($n$), measurements per trial, and trials. [§II; §IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- $t_{P_0}$ is the minimum time for one execution of $P_0$ on a given computer; hats denote estimates ($\hat{t}_{P_0}$). [§II](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Timer quantities

- Accuracy $\tau_{\textrm{acc}}$ is an upper bound on the timer's maximal error; precision $\tau_{\textrm{prec}}$ is the smallest nonzero interval it can measure; $\tau_{\textrm{budget}}$ is an experiment's time budget. [§II](../pdfs/33-robust-benchmarking-noisy-environments.md)
- $\epsilon$ is the per-measurement timer error, with magnitude below $\tau_{\textrm{acc}}$ by definition. Unlike delays, it can be negative. [§II; §III-B; §IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Delay model

- The ideal program is $P_0=[I^{[1]},\dots,I^{[k]}]$. The program actually run, $P$, interleaves a delay instruction $D^{[i]}$ after each $I^{[i]}$; delays preserve semantics but add time. [§III-A, Eq. (1)–(2)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Because delay times are non-negative, $t_P \ge t_{P_0}$:

  $$
  t_{P}=t_{P_{0}}+\sum_{i}\tau^{[i]}_{D},\qquad \tau^{[i]}_{D}\geq 0 .
  $$

  [§III-A, Eq. (3)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The central premise is that environmental factors "only delay" instructions. Speed-up factors such as frequency scaling are assumed neutralised by fixing power profiles at maximal performance. [§III-A; Footnote 2](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Each **delay factor** $j$ triggers inside $D^{[i]}$ with constant probability $p^{[i]{(j)}}$ and costs a fixed time $\tau^{(j)}$; the trigger coefficient is Bernoulli. As printed (the summation index is $i$; see §6):

  $$
  \tau^{[i]}_{D}=\sum_{i}x_{P}^{[i]{(j)}}\tau^{(j)} .
  $$

  [§III-A, Eq. (4)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- A factor's **trigger count** is a sum of independent, non-identical Bernoulli variables, hence Poisson binomial. Run time is therefore a random variable:

  $$
  t_{P}=t_{P_{0}}+\sum_{j}X_{P}^{(j)}\tau^{(j)} .
  $$

  [§III-A, Eq. (5)](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Repetition and its limit

- $n$ executions are modelled as one execution of $Q_0$, $n$ concatenated copies of $P_0$. The delayed program $Q$ is not $n$ copies of $P$, since its delay instructions are not copies of those in $P$. [§III-B, Eq. (6)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- A measurement decomposes into true time, delays and timer error:

  $$
  T=n\,t_{P_{0}}+\sum_{j}\sum_{i=1}^{nk}x_{Q}^{[i]{(j)}}\tau^{(j)}+\epsilon,
  \qquad
  \frac{T}{n}=t_{P_{0}}+\frac{\sum_{j}\sum_{i=1}^{nk}x_{Q}^{[i]{(j)}}\tau^{(j)}+\epsilon}{n}.
  $$

  [§III-B, Eq. (7)–(8)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- All terms of Eq. (8) other than $t_{P_0}$ form the error $E$; per measurement, $E_m = T_m/n_m - t_{P_0}$. [§II; §III-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Increasing $n$ shrinks $\epsilon/n$, but the delay term depends on trigger behaviour. If each factor triggers $o(n)$ times, $T/n \to t_{P_0}$; if every factor triggers on every instruction,

  $$
  \lim_{n\to\infty}\frac{T}{n}=t_{P_{0}}+k\sum_{j}\tau^{(j)} .
  $$

  [§III-B, Eq. (9)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Stated consequence: $T/n$ at large $n$ is "at best an overestimate" of $t_{P_0}$. Repetition amortises timer error but does not reliably remove environmental bias. [§III-B](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Minimum estimator

- With $T_i/n_i = t_{P_0}+E_i$, Alg. 1 uses

  $$
  \hat{t}_{P_{0}}=\min\!\left(\frac{T_{1}}{1},\dots,\frac{T_{j}}{j}\right)=t_{P_{0}}+\min(E_{1},\dots,E_{j}) .
  $$

  [§IV-B, Eq. (10)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- When delay time scales exceed $\tau_{\textrm{acc}}$, every $E_i$ is positive and the smallest measurement has the smallest error. Below $\tau_{\textrm{acc}}$, the minimum may underestimate $t_{P_0}$ through negative timer error; the authors accept this because Alg. 1 then only picks a larger $n$ than needed. [§IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Oracle function

- The **oracle function** $\nu(t)$ maps an estimated execution time to a number of executions per measurement. It is a tuning function, unrelated to a correctness oracle or reference implementation. [§II; §IV-C](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Required properties: discrete range $\{1,\dots,j\}$; monotonically decreasing; near-zero slope around $\tau_{\textrm{prec}}$ and $\tau_{\textrm{acc}}$ (weak dependence on poorly known timer parameters); $\nu(\tau_{\textrm{prec}})\approx j$; $\nu(t\ge\tau_{\textrm{acc}})\approx 1$. [§IV-C](../pdfs/33-robust-benchmarking-noisy-environments.md)
- One example is a floored generalized logistic, with suggested $0.005<a\tau_{\textrm{prec}}<0.02$ and $0.4<b<0.6$:

  $$
  Y(t)=\left\lfloor 1+\frac{j-1}{1+e^{a(t-b\tau_{\textrm{acc}})}}\right\rfloor .
  $$

  [§IV-C, Eq. (11)](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 3. Methodology and experimental setup

### Algorithm 1

- Inputs: $P_0$, $\tau_{\textrm{acc}}$, $\tau_{\textrm{prec}}$, $\nu$. Step 1 sets $j=\tau_{\textrm{acc}}/\tau_{\textrm{prec}}$. Step 2 times $i$ executions for every $i\in\{1,\dots,j\}$. Step 3 takes $\hat{t}_{P_0}=\min(T_1/1,\dots,T_j/j)$. Step 4 returns $n=\nu(\hat{t}_{P_0})$. [§IV-A, Alg. 1](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The bound $j$ follows from amortisation: beyond $j$ repetitions the timer-inaccuracy contribution falls below $\tau_{\textrm{acc}}/j=\tau_{\textrm{prec}}$ and is unmeasurable, so "there is no reason to pick $n>j$". Overestimating $\tau_{\textrm{acc}}$ only raises $n$. [§IV-A](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The algorithm runs once per benchmark; $n$ is cached for later experiments on the same machine, and the scan is preprocessing outside $\tau_{\textrm{budget}}$. criterion, by contrast, re-determines $n$ every run. [§IV-A](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Derived:** at the Fig. 5 values ($\tau_{\textrm{acc}}\approx1000$ ns, $\tau_{\textrm{prec}}\approx1$ ns), $j\approx1000$ and one scan performs $j(j+1)/2 = 500{,}500$ executions. The paper does not state this total. [Basis: §IV-A, Alg. 1; Fig. 5](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The oracle used in practice is a lookup table initialised from $Y(t)$ and edited by hand: for benchmarks of known run times, the authors sought the smallest $n$ at which the minimum estimate "appears to converge to a lower bound". The table's values are not published in the paper. [§IV-C](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Benchmarks and data collection

- Four Julia benchmarks are described: `sumindex(a, inds)` (indexed summation; memory layout), `pushall!(a, b)` (element-wise push with a random number per iteration; RNG and reallocation), `branchsum(n)` (parity-dependent nested loop; periodically costly branching) and `manyallocs(n)` (array of randomly sized arrays, reseeded RNG; frequent allocation). [§V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The mock suite is said to be hosted at `github.com/jiahao/paper-benchmark`; input sizes are not given in the paper. [§V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Fig. 1 plots, for four benchmarks labelled only "Benchmark 1–4", the mean of each trial of 10,000 measurements against trial index. [Fig. 1](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Fig. 2 overlays $T/n$ against $n$ for repeated runs of Alg. 1 on `branchsum`; the number of repeats is not stated. [Fig. 2](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Figs. 3 and 4 use `sumindex`: Fig. 4 shows kernel density estimates for 100 trials of 10,000 consecutive measurements each; Fig. 3 shows mean, 5th–95th percentile trimmed mean, median and minimum per trial, on a trial axis running to 100. [Fig. 3; Fig. 4](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Not reported:** the machine (CPU, cores, memory), operating system, Julia and BenchmarkTools versions, the $n$ used for the 10,000-measurement trials, and any quiescence measures. A text search of the PDF found no hardware description. [§V](../pdfs/33-robust-benchmarking-noisy-environments.md); [PDF pp. 1–7](../pdfs/33-robust-benchmarking-noisy-environments.pdf)

### Alternatives discussed but not run

- criterion (Haskell) grows executions per measurement over time and reports the slope of a least-squares fit. Objections: outlier sensitivity, early measurements exposed to timer error, late measurements over-repeated. [§I-C](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The warm-up-then-i.i.d. approach of their ref. 19 is criticised: warm-up is often absent, post-warm-up timings need not be i.i.d., its formulae suit near-normal data, and manual per-benchmark calibration does not scale to over 1300 benchmarks. [§I-C](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 4. Results and evidence

### Non-i.i.d. timing statistics

- Reported: none of the four benchmarks has a normally distributed trial mean. The first shows skew and outliers, the second and fourth bimodality, the third and fourth upward drift. Each mean already averages 10,000 measurements. [§I-B; Fig. 1](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Visual inspection agrees qualitatively: panel (b) has two separated bands, (c) a rising cloud, (d) two rising bands. In (a), as drawn, isolated outliers lie below the bulk, which itself looks like two close bands. No panel has axis tick labels, so no magnitudes can be read. [Fig. 1 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The argument for what this does to standard statistics: measurements from many sources of variation are not necessarily i.i.d.; methods relying on the central limit theorem then fail; heavy tails distort the mean and are "not captured" by the median. [§I-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Poor power of F-tests and t-tests on timings, and failure of the 3-sigma outlier rule on non-i.i.d. data, are attributed to cited work; the paper runs no hypothesis tests. Automated outlier removal is criticised as distorting the empirical distribution. [§I-B](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Minimum versus other estimators

- For `branchsum`, individual $T/n$ curves are "wildly oscillatory", while the pointwise minimum over curves is smoother and tends to a constant; the text places convergence "around $n=250$". [Fig. 2; §IV-C](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Visual inspection of Fig. 2: the axis covers 0–500 executions per measurement; ticks are 75 and 150 ns; a dotted $\hat{t}_{\min}$ line lies below the 75 ns tick with no numeric label; curves carry tall spikes above a quickly flattening lower envelope. [Fig. 2 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- For `sumindex`, per-trial densities form two distinct clusters; a "cursory investigation" found no inter-trial correlation predicting the cluster. Visually (approximate), one cluster peaks near 184–185 ns and the other near 193 ns, on a 175–200 ns axis. [Fig. 4; Fig. 4 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The authors state that median and trimmed mean, though robust to outliers, "still capture bimodality", so an $n$ chosen from them could vary drastically between runs of Alg. 1. The minimum's distribution across trials is "unimodal in all cases we have observed". [§IV-B; Fig. 3](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Visual reading of Fig. 3 (approximate): medians and trimmed means alternate between about 184–187 ns and 191–192 ns; means form two bands rising from about 201 to 212 ns and from about 211 to 220 ns over 100 trials; minima lie in one band of about 172–180 ns. [Fig. 3 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The minimum is thus single-banded here but not constant: its visually read spread is several nanoseconds on a base near 175 ns. The paper reports no variance or interval for any estimator. [Fig. 3 (visual inspection); §IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Oracle function and deployment

- Fig. 5 plots two candidate oracles at $\tau_{\textrm{acc}}\approx1000$ ns, $\tau_{\textrm{prec}}\approx1$ ns: Eq. (11) with $a=0.009/\tau_{\textrm{prec}}$, $b=0.5$, and an empirically tuned lookup table. Visually (approximate), both start near 1000 executions and approach the axis by 1000 ns; the table drops to a plateau near 200 between roughly 350 and 600 ns. [Fig. 5; Fig. 5 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Derived:** with $j=1000$ and those parameters, Eq. (11) gives $Y(1\,\text{ns})=988$, $Y(500\,\text{ns})=500$, $Y(1000\,\text{ns})=11$; the example meets $\nu(\tau_{\textrm{prec}})\approx j$ and $\nu(\tau_{\textrm{acc}})\approx1$ only loosely. [Basis: §IV-C, Eq. (11); Fig. 5](../pdfs/33-robust-benchmarking-noisy-environments.md)
- BenchmarkTools, with BaseBenchmarks and Nanosoldier, runs an on-demand CI service comparing proposed Julia changes on over 1300 benchmarks. [§V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Since early 2016 the service is said to have caught "dozens of serious performance regressions", serious meaning a 30% or greater increase in a benchmark's minimum execution time. No exact count, false-positive rate or miss rate is given. [§V](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 5. Authors' conclusions

- Timing variation yields highly non-ideal, even multimodal statistics in the simplest benchmark codes, complicating regression detection. [§VI](../pdfs/33-robust-benchmarking-noisy-environments.md)
- "Virtually all" timing variations are delays (cache-line flushes, task switches and similar), so variations "never reduce the run time"; this motivates the delay model. [§VI](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The minimum, rather than mean or median, is advocated as the estimator of true run time: "robust to nonideal statistics" and giving "the smallest error". [§VI](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Alg. 1 is an automatable heuristic for the minimum executions per measurement needed to defeat timer error, in daily production use for Julia's standard library and usable for user packages. [§V; §VI](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 6. Limitations and open questions

### Assumptions stated by the source

- Delay-only noise: speed-up factors are assumed controlled by the experimenter. The minimum's justification depends on this. [§III-A; Footnote 2](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The program is deterministic and serial; each delay factor has a fixed duration and constant trigger probability per delay instruction, with independent triggers. [§III; §III-A](../pdfs/33-robust-benchmarking-noisy-environments.md)
- The minimum can underestimate $t_{P_0}$ when delay time scales are below $\tau_{\textrm{acc}}$; this is judged acceptable for choosing $n$. [§IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- Unimodality of the minimum is an observation over examined cases ("for our purposes"), not a theorem; Alg. 1 is a heuristic with a hand-tuned oracle. [§IV-B; §IV-C; §VI](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Analyst caveats and missing evidence

- **No decision rule:** the 30% figure only describes CI outcomes. There is no threshold derivation, tolerance, significance test, or false-positive/false-negative analysis for comparing two minima. [Basis: §V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Scope of the minimum argument:** §IV-B justifies the minimum for selecting $n$ and for trial-to-trial stability; §VI extends it to run-time estimation generally without an experiment on comparison accuracy. [Basis: §IV-B; §VI](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Unreported environment and small evidence base:** quantitative figures cover `branchsum` and `sumindex` on an unknown machine; `pushall!` and `manyallocs` are named in no figure caption. [Basis: §V; Figs. 1–4](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Model versus data:** the model has independent triggers with constant probabilities, yet the evidence shows drift and trial-level clustering. The paper does not fit the model to data or show it reproduces drift. [Basis: §III-A; §IV-B; Fig. 1; Fig. 4](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Best-case versus typical time:** the minimum estimates a lower bound, and Eq. (9) admits persistent delays it excludes by construction. When typical time is the quantity of interest is not discussed. [Basis: §III-B, Eq. (9); §IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Tail location (visual):** Fig. 3 minima of about 172–180 ns sit at or below the left edge of Fig. 4's axis, where plotted density is near zero, assuming both figures show the same trials (implied, not stated). The estimator would then rest on rare observations; its sensitivity to the number of measurements is not examined. [Basis: Fig. 3; Fig. 4 (visual inspection)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Oracle example versus Fig. 2 (visual):** convergence is cited near $n=250$ for a benchmark whose $\hat{t}_{\min}$ line lies below 75 ns, while both Fig. 5 curves are near the top of their range at such times. Neither value is numerically labelled and the table is "an example", so this is an unexplained tension, not a demonstrated contradiction. [Basis: Fig. 2; Fig. 5; §IV-C](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Budget use unspecified:** $\tau_{\textrm{budget}}$ is not an input of Alg. 1, and the paper does not say how many measurements or trials a budget should buy. [Basis: §II; §IV-A, Alg. 1](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **No transfer to GPUs established:** asynchronous execution, device synchronisation, clock and thermal behaviour, and shared devices are outside the paper. Whether GPU timing noise is delay-only is not addressed. [Basis: §III; §V](../pdfs/33-robust-benchmarking-noisy-environments.md)

### Source inconsistencies

- **Source inconsistency:** §III-B concludes that "one cannot always reliably eliminate bias due to external variations simply by executing the benchmark many times", but §VI says repetition "can render the effects of external variation negligible" while presenting this as a challenge to conventional wisdom. The PDF has the same wording, so this is not a conversion error. Eq. (9) supports the §III-B statement. [§III-B; §VI](../pdfs/33-robust-benchmarking-noisy-environments.md); [PDF p. 7](../pdfs/33-robust-benchmarking-noisy-environments.pdf)
- **Source inconsistency:** the run time of $P_0$ is summed to an undefined $N$ for a tape of $k$ instructions. Eq. (4) sums over $i$, the index fixed on its left-hand side, while the prose describes a sum over delay factors $j$; superscript order and index letters for trigger quantities also differ between §II and §III-A. [§II; §III-A, Eq. (1)–(5)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Unstated mapping:** Fig. 1 does not name its "four different benchmarks". Figure file names in the preserved original HTML read `simple_branchsum_fast`, `bimodal_branchsum`, `drift_manyallocs_slow` and `bimodal_drift_sumindex`, suggesting two `branchsum` variants and no `pushall!` panel. This is inferred from file names, not stated by the authors. [Fig. 1](../pdfs/33-robust-benchmarking-noisy-environments.md); [original HTML, figure `data` attributes](../pdfs/sources/33-robust-benchmarking-noisy-environments.html)

## 7. Relevance to this project — interpretation

- **Proposal:** treat this source as a CPU-timing argument to be tested, not an established rule for GPU kernels. Before Stage 3 adopts an estimator, measure on the project's hardware whether minimum, median and mean are stable across repeated trials of one kernel. [Basis: §IV-B; Fig. 3](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** check the delay-only premise for the GPU setting, for example by looking for conditions under which a kernel runs faster than its usual floor. If such factors exist and cannot be pinned, the minimum's justification does not carry over. [Basis: §III-A; Footnote 2; §IV-B](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** store raw timings with their structure (executions per measurement, measurements per trial, trial index, timestamp), not only a summary, so multimodality and drift can be diagnosed later. [Basis: §II; Fig. 1; Fig. 4](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** for Stage 4, repeat whole trials in fresh sessions and report each estimator's spread across trials; more measurements inside one trial did not remove mode-switching or drift in this source. [Basis: §III-B, Eq. (9); Fig. 1; Fig. 3](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** report the minimum and a typical-case statistic side by side as different quantities, since the minimum is a lower-bound estimate. [Basis: §IV-B, Eq. (10); §III-B, Eq. (9)](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** define the Stage 3 "faster than" rule independently of this paper and calibrate it with A/A runs (a kernel compared against itself) to estimate the false-positive rate of any threshold. The source offers only a 30% descriptive definition. [Basis: §V](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** measure candidate and baseline close together in time, or interleaved, and re-measure the baseline each round; the drift panels show estimates changing over time without code changes. This goes beyond the paper's recommendations. [Basis: Fig. 1; §I-A](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** for very short kernels, add a one-off calibration that sets executions per measurement from the measured resolution of the GPU timing mechanism, cached per kernel and machine. The paper's oracle values are tuned for a nanosecond-precision CPU timer and should not be reused. [Basis: §IV-A, Alg. 1; §IV-C; Fig. 5](../pdfs/33-robust-benchmarking-noisy-environments.md)
- **Proposal:** do not use t-tests or sigma-based outlier removal as the sole acceptance gate without first checking distribution shape on real kernel timings, and record machine, driver, versions and input sizes with every reference and final measurement. [Basis: §I-B; §V](../pdfs/33-robust-benchmarking-noisy-environments.md)

## 8. Retrieval notes

- Search aliases: BenchmarkTools.jl, Chen and Revels 2016, minimum estimator, min-time benchmarking, executions per measurement, timer accuracy versus precision, delay factors, Poisson binomial trigger count, oracle function $\nu(t)$, non-i.i.d. timings, bimodal timing distribution, benchmark drift, Nanosoldier, BaseBenchmarks.
- Terminology collisions: "oracle function" here selects a repetition count and is not a correctness oracle; "kernel" appears only in "kernel density estimate" and "custom OS kernels", never GPU kernels.
- Questions answered: Why might the minimum be preferred over mean or median for timing? What does repeating a benchmark fix and not fix? How does BenchmarkTools choose executions per measurement? What evidence shows timings are non-i.i.d.? Is a regression threshold or significance test given (no)? What hardware was used (not reported)?
- Related analyses: [performance benchmark reliability](27-performance-benchmark-reliability.md) (written in parallel; not read for this note); [KernelBench-Verified](03-kernelbench-verified.md), whose protocol reports a mean over timed GPU trials; [KernelBench](01-kernelbench.md), [robust agentic CUDA](08-robust-agentic-cuda.md) and [METR KernelAgent](21-measuring-automated-kernel-engineering.md) for GPU-kernel evaluation context. Links are navigation aids, not controlled comparisons.

## 9. Source coverage and verification notes

- Read the complete preserved Markdown: abstract, §I–§VI with all subsections, Alg. 1, footnotes, acknowledgment and all 27 references. The paper has no appendix.
- Extracted the text of all seven PDF pages and compared it with the Markdown. Equations (1)–(11), Alg. 1 and the conclusion match; the index irregularities and the §VI "can render" sentence are in the PDF, not conversion artifacts.
- The eight SVG assets contain no searchable text. Each was confirmed byte-identical (SHA-256) to the corresponding figure of the arXiv HTML version and then viewed in a browser; Figs. 1(a)–(d), 2, 3, 4 and 5 were all inspected.
- Values quoted from Figs. 3, 4 and 5 are approximate readings against printed tick labels. Fig. 1 has no tick labels and Fig. 2's $\hat{t}_{\min}$ line no numeric label, so nothing was inferred there. Curves in Figs. 2 and 4 were not counted.
- Conversion notes: Alg. 1 is preserved as a plain-text block with raw TeX; Eq. (5), (6), (7) and (10) are split across display blocks. No truncation was found. In this note, `[i]{(j)}` superscripts carry extra braces only to avoid Markdown link parsing.
- The derived values ($j$, scan execution count, evaluations of Eq. (11)) are analyst arithmetic. The linked repositories and later versions of the paper or package were not consulted.
- No benchmarks were run and no experiments were reproduced.

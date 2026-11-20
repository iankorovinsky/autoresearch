---
title: "Lessons from the Trenches on Reproducible Evaluation of Language Models"
source_url: "https://arxiv.org/abs/2405.14782"
full_text_url: "https://arxiv.org/html/2405.14782v3"
version: "2405.14782v3"
authors: ["Biderman, Stella", "Schoelkopf, Hailey", "Sutawika, Lintang", "Gao, Leo", "Tow, Jonathan", "Abbasi, Baber", "Aji, Alham Fikri", "Ammanamanchi, Pawan Sasanka", "Black, Sidney", "Clive, Jordan", "DiPofi, Anthony", "Etxaniz, Julen", "Fattori, Benjamin", "Forde, Jessica Zosa", "Foster, Charles", "Hsu, Jeffrey", "Jaiswal, Mimansa", "Lee, Wilson Y.", "Li, Haonan", "Lovering, Charles", "Muennighoff, Niklas", "Pavlick, Ellie", "Phang, Jason", "Skowron, Aviya", "Tan, Samson", "Tang, Xiangru", "Wang, Kevin A.", "Winata, Genta Indra", "Yvon, François", "Zou, Andy"]
retrieved_date: "2026-11-20"
conversion: "HTML to Markdown with TeX math"
pdf: "14-reproducible-lm-evaluation.pdf"
original_html: "sources/14-reproducible-lm-evaluation.html"
---

# Lessons from the Trenches on Reproducible Evaluation of Language Models

Stella Biderman
  
Hailey Schoelkopf
  
Lintang Sutawika
  

  
Baber Abbasi
  
Jessica Zosa Forde
  
Leo Gao
  
Jonathan Tow
  

  
Alham Fikri Aji
Affiliation: MBZUAI
  
Pawan Sasanka Ammanamanchi
Affiliation: IIIT Hyderabad
  
Sidney Black
Affiliation: EleutherAI
  
Jordan Clive
Affiliation: Chattermill AI
  
Anthony DiPofi
Affiliation: EleutherAI
  
Julen Etxaniz
Affiliation: HiTZ Center - Ixa, UPV/EHU
  
Benjamin Fattori
Affiliation: EleutherAI
  
Charles Foster
Affiliation: Finetune
  
Jeffrey Hsu
Affiliation: Ivy Natal
  
Mimansa Jaiswal
Affiliation: University of Michigan
  
Wilson Y. Lee
Affiliation: HubSpot
  
Haonan Li
Affiliation: MBZUAI
Affiliation: LibrAI
  
Charles Lovering
Affiliation: Kensho
  
Niklas Muennighoff
Affiliation: Contextual AI
  
Ellie Pavlick
Affiliation: Brown University
  
Jason Phang
Affiliation: EleutherAI
Affiliation: New York University
  
Aviya Skowron
Affiliation: EleutherAI
  
Samson Tan
Affiliation: Amazon
  
Xiangru Tang
Affiliation: Yale University
  
Kevin A. Wang
Affiliation: Brown University
  
Genta Indra Winata
Affiliation: HKUST
  
François Yvon
Affiliation: Sorbonne University
  
Andy Zou
Affiliation: CMU

Affiliation: Stability AI
Affiliation: Lead authors
Affiliation: Second authors

###### Abstract

Reliable evaluation of language models (LMs) remains an open challenge. Researchers and engineers face methodological issues such as the sensitivity of models to evaluation setup, difficulty of proper comparisons across methods, and the lack of reproducibility and transparency. Evaluation difficulties are exacerbated by the fracturing and siloing of information about conventions and common practices. In this paper we draw on three years of experience in evaluating large language models (LMs) as developers of the popular Language Model Evaluation Harness (lm-eval) ([Gao et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib11)) framework to provide guidance and lessons for the field moving forward. We document a variety of challenges faced by practitioners and provide concrete instances where these challenges or the absence of best practices have come into effect. We make recommendations to the field for improving evaluation rigor and confidence, and attempt to codify much of the tacit or folk knowledge surrounding LM evaluation, for a solid ground to move forward.

|  |
| --- |
|  |

## 1 Introduction

Efforts to progress the study of LMs and machine learning hinge upon the communication of evaluation results presented to support new ideas and conclusions. Improper evaluation practices can lead to skewed performance comparisons ([Card et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib19)) which may influence the direction of future research and the adoption of new methods by the community ([Dehghani et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib42)), or lead to adverse effects from deploying suboptimal or harmful models ([Bender and Friedman, 2018](https://arxiv.org/html/2405.14782v3#bib.bib43)) on tasks for which they are ill-suited ([Raji et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib90)). However, evaluation in LMs is fraught with difficulties, both due to problems inherent in benchmarking across fields and due to specific complications or challenges introduced by LMs. As the study of LMs and their typical uses have rapidly shifted over the past years, evaluation practices have been left by the wayside, resulting in an evaluation landscape that is quickly changing, with accepted common practices either obscurely or not at all formally documented, and a fractured ecosystem.

In this paper we attempt to rectify this situation by documenting the current state of evaluations and our lessons learned that have been especially beneficial in obtaining useful and rigorous findings. Drawing on our experience building and maintaining lm-eval ([Gao et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib11)), a flexible LM evaluation library, we identify common challenges, recommend best practices, and share how we have operationalized these best practices in our work. Unlike other evaluation libraries which prescribe specific benchmarks ([Liang et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib31); [OpenCompass Contributors, 2023](https://arxiv.org/html/2405.14782v3#bib.bib103)), lm-eval is intentionally non-opinionated about what to evaluate. It is an orchestration library: its role is to make it possible to run any benchmark on any model reliably and reproducibly, not to dictate which benchmarks matter. We are, however, opinionated about how a given evaluation should be run—each task ships with a carefully chosen default configuration reflecting community best practices—while deliberately making it easy for power users to customize any aspect of their evaluation setup. The goal is to guide users toward sound methodology by default while enabling arbitrary flexibility for advanced users.

In this way, lm-eval seeks to serve as research infrastructure—a resource for researching evaluation, locating de facto best practices, and enhancing reproducibility. The design of lm-eval has evolved over time to reflect the needs of the open-source community and our evolving understanding of best practices; we discuss specific ways we have modified the library to address the challenges we describe throughout this paper.

Our goals are threefold:

1. We collect, synthesize, and systematically present challenges commonly faced when conducting LM evaluations, coupled with concrete instances where these challenges have surfaced. We seek to formalize much of the tacit “folk knowledge” held within the field surrounding evaluation practices and implementation details.
2. We provide recommendations for best practices in LM evaluations, informed by these practical challenges.
3. We document how we have incorporated these best practices into newer versions of our library  lm-eval. We intend this to provide an example of operationalizing our stated best practices and ease adoption of our recommendations.

## 2 Why is Evaluating Language Models Hard?

Before turning to specific lessons, we outline three under-discussed aspects of language model evaluation that make it uniquely difficult to do well: a fundamental methodological challenge we term the Key Problem, social dynamics and incentive structures that work against rigorous evaluation, and a rapidly shifting landscape that can render today’s best practices obsolete within a year or two. These challenges underpin much of the discussion in this paper.

### 2.1 The Key Problem

A useful framing which shapes all of LM evaluation is a concept which we term the Key Problem:

###### The Key Problem.

When evaluating language models, there can be many semantically equivalent but syntactically different ways of expressing the same idea. In an ideal world, to assess correctness we would have a way to automatically detect when two sentences express the same content but in different words. However, our best automated tools for determining whether two sentences are semantically equivalent are the very models we are seeking to evaluate.

Many problems with, and common approaches to, LM evaluation are designed around this challenge. Responses to the Key Problem generally fall into three broad categories: converting the open-domain problem into a closed-domain one where correctness can be determined mechanically, using model-based grading, or relying on human labor. Each approach has significant trade-offs.

#### Closed-domain conversion.

The Key Problem can be sidestepped by artificially restricting the answer space so that correctness can be checked mechanically. The most prevalent way to achieve this is to reframe questions as multiple choice problems (Appendix [B](https://arxiv.org/html/2405.14782v3#A2)), with a single gold target answer and a finite, static set of possible responses ([Hendrycks et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib12); [Srivastava et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib13); [Li’evin et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib66); [Lin et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib67); [Robinson et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib76); [Holtzman et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib75)). Another approach is structured generation, in which the model’s output is constrained at decoding time to match a specified format, reducing variance from prompt sensitivity and enabling more reliable answer extraction ([Kurt et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib100)). Alternatively, when a reference answer is known, one can perform string-matching approaches heuristically to determine whether the model’s answer matches the ground truth ([Dua et al., 2019](https://arxiv.org/html/2405.14782v3#bib.bib80); [Joshi et al., 2017](https://arxiv.org/html/2405.14782v3#bib.bib47); [Hendrycks et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib49)). In rare cases where we have access to a practical verifier ([Lewkowycz et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib82); [Chen et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib9)) it can be used to check correctness directly.

#### Model-based grading.

A second approach uses automated metrics to directly approximate a solution to the Key Problem by measuring the distance from a generated response to a gold-standard one. Heuristic metrics such as BLEU ([Papineni et al., 2002](https://arxiv.org/html/2405.14782v3#bib.bib25)) and ROUGE ([Lin, 2004](https://arxiv.org/html/2405.14782v3#bib.bib27)) count the n-gram overlap between the two texts. These metrics offer notable advantages in that they are (theoretically) fully reproducible, far easier and cheaper to compute, and can avoid some of the issues faced by human studies ([Wei and Jia, 2021](https://arxiv.org/html/2405.14782v3#bib.bib56); [Freitag et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib18); [Amidei et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib23)). Nevertheless these heuristic-based metrics have flaws ([Callison-Burch et al., 2006](https://arxiv.org/html/2405.14782v3#bib.bib20)) and underdocumented hyperparameters can present reproducibility challenges ([Marie et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib15)). More recently, model-based metrics have gained momentum through evaluation methods that leverage large language models as a grader ([Kim et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib50); [Wang et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib58); [Liu et al., 2023a](https://arxiv.org/html/2405.14782v3#bib.bib59)) — especially as proxies for human preference evaluation ([Zheng et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib24)) — but these are known to be flawed ([Wang et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib60); [Shen et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib61); [Zeng et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib62); [Hu et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib63); [Liu et al., 2023b](https://arxiv.org/html/2405.14782v3#bib.bib64); [Chen et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib65)) and suffer from similar issues.

#### Human labor.

Finally, the Key Problem could in principle be addressed by having expert human annotators score model responses for correctness. However, performing accurate human studies is time-consuming and expensive. Additionally, human assessments can be flawed and biased, especially for complex judgments such as factual correctness ([Hosking et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib54); [Xu et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib55); [Wu and Aji, 2023](https://arxiv.org/html/2405.14782v3#bib.bib21)). Expertly trained human judgment can alleviate these issues but is inherently non-scalable.

The Key Problem is an inherent part of language model evaluation. There is no “correct” answer, only a best answer in the context that an evaluation is being done. In this paper we focus on MCQA evaluations because they are the approach we have the most experience with and are the most common approach in the literature. We believe that much of our analysis will apply to other approaches, but urge experts in those areas to document and promote their thoughts on best practices in their areas as well.

### 2.2 Social Dynamics of Evaluation

Beyond its technical difficulties, language model evaluation operates within an incentive landscape that actively works against rigor. Model developers face strong pressure to present their models as the best. While this has always been an issue with peer review in AI, in recent years it has been massively exacerbated by the need to justify the extreme levels of capital investment required to build frontier models, to attract customers, and to maintain competitive positioning. Evaluations are not just scorecards: they are advertisements.

The most fundamental barrier to independent evaluation is access. Some of the allegedly most capable models in the world have never been publicly released in any form. Google’s PaLM ([Chowdhery et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib68); [Anil et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib83)) and models built on it ([Lewkowycz et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib82); [Singhal et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib84)), Baidu’s ERNIE 4.0, ByteDance’s Doubao, and Tencent’s Hunyuan all claim to be very powerful models but there is no way to get access to them. For models like these, the only evaluation numbers that exist are the ones their developer choses to publish.

Models that are available through APIs present a subtler problem. While API access does enable independent evaluation in principle, many APIs do not expose output log-probabilities which are required for many standard evaluation benchmarks (See [Appendix B](https://arxiv.org/html/2405.14782v3#A2) for details). This forces evaluators to rely on generation-based approaches, which introduce additional variance from prompt formatting and answer extraction, or to simply trust the developer’s self-reported numbers on tasks that require log-probabilities. Running comprehensive benchmark suites through APIs is also expensive, and these costs fall disproportionately on independent researchers—precisely the parties whose assessments would be most credible by virtue of their independence. The field has, in effect, adopted a norm where model developers grade their own homework.

Even when independent evaluation is possible, there is pressure to cherry-pick favorable tasks, choose flattering prompting strategies, or report single-run point estimates that obscure uncertainty. The lack of standardized reporting requirements (discussed further in Section [3](https://arxiv.org/html/2405.14782v3#S3)) makes it easy to present numbers that are technically accurate but misleading. As we discuss in Section [4](https://arxiv.org/html/2405.14782v3#S4), these dynamics are severe enough that the Benchmark Lottery has become a real phenomenon, with benchmark selection itself becoming a target for optimization rather than a means of measuring genuine progress.

### 2.3 Fast-changing Progress and Conventions

Figure 1: A timeline showing the relative release dates of a selection of notable benchmarks used to evaluate LMs, as compared to the release dates of BERT ([Devlin et al., 2018](https://arxiv.org/html/2405.14782v3#bib.bib89)), GPT-2 ([Radford et al., 2019](https://arxiv.org/html/2405.14782v3#bib.bib3)), GPT-3 ([Brown et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib4)), and ChatGPT, used as approximate stand-ins for shifts in how the community uses and therefore evaluates LMs. Common practice for evaluating autoregressive language models today diverges from the method described in the paper for all listed tasks except MMLU and MATH.

Due to the time-consuming nature of developing good benchmarks and the rapid pace of change in NLP research in the past decade, many widely used language model evaluation benchmarks do not represent the current paradigm of how language models are trained: This has two major impacts:

1. Benchmarks are being used for purposes outside of their original design, with unclear validity in these new scenarios: for example, a large number of benchmarks ([Wang et al., 2019a](https://arxiv.org/html/2405.14782v3#bib.bib14); [Wang et al., 2019b](https://arxiv.org/html/2405.14782v3#bib.bib2)) have been built around fine-tuning on a known training set and closed space of labels, but are used today for open-domain language models.
2. There is no “ground-truth” implementation from the original benchmark authors, and many popular benchmarks ([Paperno et al., 2016](https://arxiv.org/html/2405.14782v3#bib.bib88); [Clark et al., 2018](https://arxiv.org/html/2405.14782v3#bib.bib48); [Zellers et al., 2019](https://arxiv.org/html/2405.14782v3#bib.bib46)) that have been “retrofitted” for compatibility with autoregressive LMs. In the absence of a clear standard, the community’s methodology for evaluating on these benchmarks may be fragmented or undocumented.

To illustrate the effects of this development timeline, [Figure 1](https://arxiv.org/html/2405.14782v3#S2.F1) shows how many prominent LM benchmarks were designed prior to shifts such as in-context learning and chat interaction, and therefore were not designed to take these approaches into account. This can affect validity in unforeseen ways.

Aside from validity concerns, it creates a challenge in which popularly used benchmarks may have no clear “ground-truth” evaluation code or settings for use in these new areas, requiring practitioners to derive their own design choices and fracturing practices among the community.

## 3 Evaluation Details are Lost in Communication

The typical benchmark life cycle is a game of telephone–a benchmark is first designed by researchers and released to the community alongside, typically, code for evaluating the benchmark. Next, for the benchmark to see wide usage and community adoption, researchers beyond this initial group must use and run the benchmark in their own work. This typically requires a reimplementation step, where a variety of new models or codebases must interface with the new benchmark in order for it to be widely run. This process requires a process of repeatedly communicating and adapting the benchmark’s implementation by new researchers.

In the course of such transmission, it is all too easy for key facts and methodological details to be obscured or fail to be properly adapted. Knowledge of how to perform a benchmark can be implicit or fractured across many locations, resulting in comparisons that may be irreproducible or not 1-to-1, and reducing trust in evaluations run by other researchers. With this in mind, we will explore how this creates challenges for LM benchmarking, and how to move toward resolving it.

### 3.1 Challenges

#### “Minor” Implementation Details Matter

While one might be tempted to pass off these minor implementation or methodological details as simply trivialities, they in fact significantly affect numerical scores and the conclusions drawn. LMs are not robust, often in seemingly unintuitive ways: for example, the precise string used for a given prompt, even down to whitespace usage, can significantly impact performance ([Weber et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib52); [Sclar et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib33); [Mizrahi et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib51); [Alzahrani et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib93); [Webson and Pavlick, 2022](https://arxiv.org/html/2405.14782v3#bib.bib53)). Even the choice of few-shot examples can also significantly change performance ([Lu et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib70); [Min et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib69)). In Table [1](https://arxiv.org/html/2405.14782v3#S3.T1) we show scores for several models on ARC (Challenge) ([Clark et al., 2018](https://arxiv.org/html/2405.14782v3#bib.bib48)) and MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib12)) for two different common prompting styles, and see that performance varies dramatically with the prompt (in some cases, $>20\%$), and which prompt style works best varies by model. Thus, the selection of prompt affects the model rankings and research conclusions produced in addition to numeric scores themselves ([Alzahrani et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib93)). These results appear to be consistent with similar analysis done in [Gu et al. (2024)](https://arxiv.org/html/2405.14782v3#bib.bib108). Further details can be found in Appendix [C.1](https://arxiv.org/html/2405.14782v3#A3.SS1).

Table 1: Comparison of 0-shot model performance (acc) for several pretrained LMs ([Black et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib8); [Touvron et al., 2023b](https://arxiv.org/html/2405.14782v3#bib.bib71); [Penedo et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib72); [Jiang et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib74); [Jiang et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib73)) on ARC (Challenge subset) and MMLU across two commonly used prompt styles.



<table>
<thead>
<tr>
<th></th>
<th colspan="2">ARC (Challenge)</th>
<th colspan="2">MMLU</th></tr>
<tr>
<th></th>
<th>Cloze</th>
<th>MMLU-style</th>
<th>Hybrid</th>
<th>MMLU-style</th></tr>
</thead>
<tbody>
<tr>
<th>GPT-NeoX-20B</th>
<td><span>38.0</span> $\pm$ <span>2.78</span> %</td>
<td>$26.6\pm 2.53$%</td>
<td><span>27.6</span> $\pm$ <span>0.74</span>%</td>
<td>$24.5\pm 0.71$%</td></tr>
<tr>
<th>Llama-2-7B</th>
<td><span>43.5</span> $\pm$ <span>2.84</span>%</td>
<td>$42.8\pm 2.83$%</td>
<td>$39.8\pm 0.79$%</td>
<td><span>41.3</span> $\pm$ <span>0.80</span>%</td></tr>
<tr>
<th>Falcon-7B</th>
<td><span>40.2</span> $\pm$ <span>2.81</span>%</td>
<td>$25.9\pm 2.51$%</td>
<td><span>29.1</span> $\pm$ <span>0.75</span>%</td>
<td>$25.4\pm 0.72$%</td></tr>
<tr>
<th>Mistral-7B</th>
<td>$50.1\pm 2.86$%</td>
<td><span>72.4</span> $\pm$ <span>2.56</span>%</td>
<td>$48.3\pm 0.80$%</td>
<td><span>58.6</span> $\pm$ <span>0.77</span>%</td></tr>
<tr>
<th>Mixtral-8x7B</th>
<td>$56.7\pm 2.84$%</td>
<td><span>81.3</span> $\pm$ <span>2.23</span>%</td>
<td>$59.7\pm 0.77$%</td>
<td><span>67.1</span> $\pm$ <span>0.72</span>%</td></tr>
</tbody>
</table>



This is especially problematic due to the current trend of naively comparing and transferring numbers across models’ technical reports. [Fourrier et al. (2023b)](https://arxiv.org/html/2405.14782v3#bib.bib99) demonstrate this concretely: comparing three independent implementations of MMLU (HELM, lm-eval, and the original MMLU code), they find that these implementations produce widely different scores and even change the ranking order of models on the rd—despite ostensibly evaluating the same benchmark.
While they do not report their precise evaluation set-up, the “MMLU-style” prompt appears to produce scores consistent with those reported in [Jiang et al. (2024)](https://arxiv.org/html/2405.14782v3#bib.bib73) for 25-shot ARC Challenge on Mixtral-8x7B and Llama 2-70B, while Cloze-style or other prompts we tested do not. We thus believe Mistral-7B, Mixtral-8x7B, and the Llama models publish results on MMLU but also ARC using this MMLU-style prompt–a decision which is not inherently unreasonable, but which is not documented to the best of our knowledge in technical reports, confusing a variety of practitioners and confounding attempts to replicate the reported scores.

A further challenge leading to similarly uncertain or inconclusive comparisons is that many headline LLM evaluation results are impossible to reproduce or rerun, because the models themselves have not been made publicly usable. For example, [Touvron et al. (2023a)](https://arxiv.org/html/2405.14782v3#bib.bib97) are forced to pull numbers from the PaLM and Chinchilla technical reports ([Chowdhery et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib68); [Hoffmann et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib36)) directly for baseline comparison, with no option to rerun these models on Llama’s evaluation implementations since PaLM and Chinchilla are not publicly available. We do not intend to single out the Llama models as an example: simply to illustrate that many details are either hard or impossible to intuit from publications, especially without context. Indeed, Llama 1 is easier to critique because they do comparatively report more information than average–details such as loglikelihood normalization methodology (see Appendix [B](https://arxiv.org/html/2405.14782v3#A2) for a full description of such details) are documented across the tasks evaluated.

While the prompt is a relatively “obvious” location to look when tracking down disparities in reproduced and reported scores, many other equally-important details can also have the same outsized impacts on model performance. For example, [Press et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib32) demonstrate how the methodology typically used–“nonoverlapping windows” (Appendix [B](https://arxiv.org/html/2405.14782v3#A2))–to compute perplexity on greater-than-context-length sequences systematically skews results in favor of models trained and thus evaluated with longer context lengths. Awareness of such subtleties’ importance has increased, especially for massively long-context models ([Peng et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib96)), but it is still underdiscussed relative to its outsized impact, and there is not a clear path for new practitioners to familiarize themselves with such tacitly-known best practices.

#### Each Task Requires Care and Individual Implementation Considerations

These supposedly-minor implementation details become even more important when accounting for the need to reimplement every task of interest independently. Most language model releases report scores across a vast range of tasks ([OpenAI, 2023](https://arxiv.org/html/2405.14782v3#bib.bib26); [Touvron et al., 2023a](https://arxiv.org/html/2405.14782v3#bib.bib97); [Touvron et al., 2023b](https://arxiv.org/html/2405.14782v3#bib.bib71)). Each individual task can be very effort-intensive to implement correctly — as [Ganguli et al. (2023)](https://arxiv.org/html/2405.14782v3#bib.bib101) discuss, even a single evaluation such as BBQ ([Parrish et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib94)) can take a significant number of researcher-hours to implement well and understand — which increases the difficulty of vetting all tested tasks for especially smaller teams. [Ganguli et al. (2023)](https://arxiv.org/html/2405.14782v3#bib.bib101) go on to say that “Our experience suggests that it is necessary to make many tricky judgment calls and considerations when running such a (supposedly) simple and standardized evaluation.”–this corroborates our experience with lm-eval. No task is truly trivial to implement and understand with confidence, and often requires discussing with benchmark creators to verify intent.

Further, all benchmarks come with their own subtle quirks and characteristics, which must be handled correctly to avoid skewing results. For example, the AI2 ARC [Clark et al. (2018)](https://arxiv.org/html/2405.14782v3#bib.bib48) benchmark is almost entirely 4-choice MCQA, but a single question has 5 possible answer choices instead of 4. MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib12)) is correctly calculated via performing the micro average by averaging accuracy across documents11
1
[As performed by the official MMLU evaluation script](https://github.com/hendrycks/test/blob/master/evaluate.py) , rather than macro-averaging unweighted accuracies of each of the 57 subjects–the choice of which affects results by several percentage points. This particular detail often is not reported but must be inferred if per-subject MMLU scores are provided. HumanEval ([Kapoor et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib104)) has 3 documents lacking the example tests that all other documents possess. We provide these examples not as an exhaustive list, but to show that such subtleties are widespread and affect some of the most popular LM benchmarks.

#### Crucial Details Go Unreported

The major reason that such minor implementation choices are so noteworthy is that crucial methodological details are insufficiently documented and reported in most works. Because subtle implementation choices can have an outsized impact on evaluation results, if evaluation code is not published, evaluation setups are likely to be underdetermined based on a publication’s description in isolation, through one or more practices being elided from reporting on methodology. For instance, reporting full prompts is a necessary but not sufficient bar for achieving perfect reproducibility. Despite being insufficient on its own, releasing prompts is vitally important and we heavily recommend it and other methodological disclosures.

It is common practice for researchers to describe evaluation methodology by referring to their use of the setup found in a specific prior work. However, such prior work may not release code, or insufficiently describe its own setup, making comparison difficult. It additionally makes tracing the “accepted” practices challenging, especially for newcomers to the field. Many works, for example, trace their methodologies for prompted evaluation to [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4), who rather thoroughly document the majority of their evaluation setups, despite not releasing code. However, beyond this work, a large number of the foundational approaches used for such few-shot evaluation such as loglikelihood-based multiple choice is not detailed outside of an appendix written by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4) which does not include several tweaks introduced by subsequent work by other researchers that have become commonplace, making it extremely difficult for newcomers to learn this tacit knowledge about how evaluations are performed. In the hopes of making this folk knowledge more explicit, we document these practices in Appendix [B](https://arxiv.org/html/2405.14782v3#A2).

### 3.2 Best Practices

In response to these challenges faced in LM benchmarking, we propose several best practices as principles to follow, in order to mitigate the negative impacts.

- Provide Evaluation Code: Whenever possible, new publications should be accompanied by release of the exact evaluation code used. We acknowledge that this may be difficult, if for example evaluation code is tied tightly to internal software infrastructure22
  2
  Though high-effort, code releases such as <https://github.com/openai/simple-evals> demonstrate separation is possible.–however, code is essential to avoid eliding small, unnoticed yet crucial details, even if the authors may assume these details can be inferred or taken for granted.
- Provide Prompts and Detail Methodology Thoroughly: Failing the release of code or to supplement the release of code, prompts should be clearly reported within a publication. Other methodological details should be thoroughly described, including, but not limited to: the amount of prompt engineering performed and on which models and evaluation sets; hyperparameters used to calculate loglikelihood or perplexity-based evaluations; generation hyperparameters and answer extraction heuristics used to perform generative evaluations and more. Indirection should be minimized, and at least one source of full ground truth, directly within or linked by the publication, should exist.
- Create Reporting Standards: These issues are exacerbated by the fact that there does not exist commonly held practices or checklists within the community to incentivize or ensure adequately thorough documentation of evaluation methodologies. We call upon the community to explore the creation and implementation of such standards.

### 3.3 Operationalizing Best Practices

Even slight improvements in the state of evaluation communication, awareness, and reporting can significantly improve evaluation rigor and trustworthiness in the LM research field. Releasing and documenting evaluation code or properly checking task implementations for a wide variety of tasks can be a significant overhead for smaller or lower-resourced teams. By using easier-to-run unified evaluation libraries such as  lm-eval or HELM ([Liang et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib31)), one can have a single location for many existing benchmark implementations and remove the need to wade through bespoke codebases and implementations for each novel benchmark. Reporting the use of particular tasks within lm-eval on a certain codebase version has become a de facto reporting standard for users which allows easier comparisons to other works and makes sharing the code used for one’s evaluations easy.

We have also adapted lm-eval over its lifetime to date to make it better able to serve the research community. As part of the design for lm-eval, we have made prompt modifications as simple as modifying a single line of a configuration file 33
3
[Writing a Prompt Template in the LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#writing-a-prompt-template) . This allows prompts to be modified if desired, while otherwise enforcing a sensible default based on community sentiment and common usage for those simply seeking to produce a clean comparison to prior work. Particular task configurations desired by a user can now be communicated through a single configuration file along with a linked library version. Thus, the lm-eval codebase is not only for reporting common numbers out-of-the-box but also to raise awareness and aiding research studies on the evaluation difficulties we have described ([Lyu et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib77); [Alzahrani et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib93)).

In general, we following the following priority list for deciding prompting and other evalution details:

1. If there is widespread agreement among people who train LLMs, use the agreed upon procedure.
2. If there is a clear and unambiguous official implementation, use that procedure.
3. If there is widespread agreement among people who evaluate LLMs, use the agreed upon procedure.
4. If there are multiple common implementations but not universal or widespread agreement, use our preferred option among the common implementations. As before, prioritize choosing from among the implementations found in LLM training papers.

We choose to prioritize the decisions made by model trainers for a few key reasons. First, we have found that people who train models are harder to influence in their practices than people who primarily evaluate models in our work to promote standardization within the ecosystem. Second, many people who train large scale models don’t allow others to access their models in the form reported in their reports. Thus people are forced to copy their practices when seeking to have consistent set-ups across models. Third, we have found that as model capabilities have changed (see [Section 2.3](https://arxiv.org/html/2405.14782v3#S2.SS3)) model developers tend to be more attuned to the most productive ways to evaluate cutting edge models (in part due to having first access to such models). Fourth, people who train models have substantial social and political power within the field and so a large fraction of users will align to their practices regardless.

Turning to validating implementations, we explicitly provide functionality designed around easing qualitative inspection of model outputs and sanity-checking of scores: --limit and --log_samples flags 44
4
[LM Evaluation Harness Command Line Interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#command-line-interface)  can be used to dry-run evaluations on a small subset of samples for testing and prototyping, and per-sample scores can be recorded and logged to disk, while also saving model outputs for post-hoc reproducibility without additional compute expenditures.

Although more work should be done to improve reporting standards around evaluation, we believe that a common research infrastructure can significantly ease the burden of good practices for researchers, reduce duplicated effort massively, and increase confidence in evaluation scores that use such common libraries by and large.

## 4 Evaluations are Not Just Numbers

When one runs an evaluation benchmark, what comes out is a numerical score. Those numbers are often used by people to assess the relative capabilities of models, determine their suitability for deployment, or consider the need for specific rules and regulations. The appropriate conclusions to draw about a model’s behavior are informed by, not constituted by, these numerical scores. There is currently a substantial lack of attention to this fact, and replacing genuine evaluation with computing benchmark scores is substantially hindering language model evaluation, research, and governance.

### 4.1 Challenges

#### What, if Anything, Does a Benchmark Measure?

It is almost never the case that one cares about the actual score of a language model on a particular benchmark. Instead, benchmarks are typically proxies for some broader notion of capability or ability that is difficult to make precise and measure. While being able to rapidly evaluate models and compare their capabilities numerically is very useful in many contexts, overemphasis and focus on particular numeric scores can hinder progress or prevent accurate senses of performance. This is especially true as precise numerical scores are very sensitive to evaluation settings.

While debate around the validity of machine learning benchmarks has been widespread for years ([Messick, 1994](https://arxiv.org/html/2405.14782v3#bib.bib79); [Raji et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib57); [Saphra et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib41); [Davis, 2023](https://arxiv.org/html/2405.14782v3#bib.bib81); [Subramonian et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib78)), it has largely been ignored by those developing the latest wave of language models and language model evaluation benchmarks. We find this particularly concerning in the context of evaluations informing deployment decisions: it is unclear (or in some cases very clear to the contrary) that evaluations, especially more artificially-constructed benchmarks such as minimal-pair or pronoun-frequency ([Nangia et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib106); [Zhao et al., 2018](https://arxiv.org/html/2405.14782v3#bib.bib107); [Névéol et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib105)) based bias benchmarks or multiple-choice knowledge tests ([Hendrycks et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib12)), provide information about how models will perform in more realistic deployment scenarios. Indeed, research like [Lyu et al. (2024)](https://arxiv.org/html/2405.14782v3#bib.bib77) demonstrate that multiple-choice loglikelihood-based benchmarks do not even agree with generative-scored multiple-choice benchmarks. [Omiye et al. (2023)](https://arxiv.org/html/2405.14782v3#bib.bib95) and [Hofmann et al. (2024)](https://arxiv.org/html/2405.14782v3#bib.bib92) find that current safety-trained models, despite exhibiting apparently-safer or less bias on current bias benchmarks, amplify biases far beyond what is suggested by their training data when engaged in practical, real-world interactions such as in high-risk settings like medicine.

#### Lack of Reporting on Uncertainty and Variance

Large language models are heavily stochastic objects. Their behavior depends on a large number of both arbitrary decisions and random events that occur during their training and evaluation. This fact is very rarely acknowledged during model evaluation, likely because it makes evaluating models much more challenging and requires statistical expertise most machine learning researchers lack. However, omitting such information can paint highly misleading pictures of model performance. This issue is becoming more acute over time–evaluation benchmarks are increasingly focusing on advanced concepts or capabilities that can be challenging to produce examples for en masse. This results in smaller evaluation datasets, and correspondingly larger uncertainty in numeric scores.

In an excellent blog post on this topic, [Lukošiūtė (2024)](https://arxiv.org/html/2405.14782v3#bib.bib102) presents the following two pictures of GPQA performance for some of OpenAI and Anthropic’s models:

![Refer to caption](assets/14-reproducible-lm-evaluation/figure-001.png)


Figure 2: Plots showing GPQA performance of various models in a single run (top) and a confidence interval computed after doing ten runs (bottom). The two plots lead to very different conclusions about the relative performance of models. Data from [Lukošiūtė (2024)](https://arxiv.org/html/2405.14782v3#bib.bib102), presentation edited.

Presenting single-run point estimates of performance (top) instead of a 95% confidence interval across 10 runs of the benchmark (bottom) paints a very different pictures regarding the comparative performance of different models. Similar behavior holds for benchmarks such as HumanEval ([Chen et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib9)), which contains only 164 examples. Practitioners frequently tout small increases in numeric score on datasets such as these, ignoring the fact that such “improvements” might be washed out simply via sampling again at the same temperature.

#### The Benchmark Lottery

The Benchmark Lottery ([Dehghani et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib42)) posits that popular benchmarks can have a distorting effect on the field, redefining success in ways that reinforce their particular idiosyncrasies and even flaws. A concrete illustration of this phenomenon can be seen in the Open LLM Leaderboard ([Beeching et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib86)), where the choice of evaluation format has itself become a target for optimization: practitioners have been observed fine-tuning their models specifically to match the evaluation format used by the leaderboard, such as targeting separator tokens used in particular versions of lm-eval, rather than improving the underlying capabilities being measured. The leaderboard’s arbitrary choices about which benchmarks to include and which implementation details to adopt effectively define what “good” means for a large segment of the open-source community, making it a high-stakes example of how benchmark selection and implementation choices can distort incentives.

The sensitivity to implementation choices underlying this phenomenon is also illustrated in [table 1](https://arxiv.org/html/2405.14782v3#S3.T1). We show that GPT-NeoX-20B performs approximately randomly on both ARC-C and MMLU using what we term MMLU-style evaluation, but much better using other prompting formats. Through exploratory analysis of model completions we have observed that this at-random performance is a result of failing to follow the task and output format for this prompting style, even when it can score well using an alternative format. An even more extreme example is from ARC-Easy: using the “cloze” evaluation setup, GPT-NeoX achieves an accuracy of $72.4\pm 1.80\%$ but using the MMLU-style setup it scores $26.5\pm 1.78\%$.

While the Open LLM Leaderboard example shows the impact of explicit gaming, these results show a phenomenon that is much harder to guard against: in a different but plausible world in which MMLU-style prompting was the standard introduced in [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4), the perceived performance of GPT-NeoX-20B and other models trained on the Pile ([Gao et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib10)) would be extremely low, despite strong results that would be observed in the “cloze” setting. In such a world, we very plausibly could have concluded that our dataset was garbage (instead of concluding that it didn’t enable performance in a specific format only) and wasted many human and GPU-hours trying to debug the wrong problem.

These issues are so extreme that it is common for people who work with language models to discuss “vibes tests,” the intuitive feeling they get about a model after using it for an hour or so, as being as informative or even more informative than the benchmarks that much of our field is allegedly based on. This sentiment is widespread enough that recent work has sought to formalize it: [Dunlap et al. (2024)](https://arxiv.org/html/2405.14782v3#bib.bib109) introduce a framework for discovering and quantifying the qualitative differences between models that users notice through informal interaction but that standard benchmarks fail to capture, while Reka AI’s Vibe-Eval ([Padlewski et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib110)) explicitly designs evaluation suites around these hard-to-quantify aspects of model quality. The Chatbot Arena ([Zheng et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib24)) can be viewed in a similar light—as an effort to scale human preference judgments into a systematic ranking methodology, producing orderings that frequently diverge from those derived from standard benchmarks. That the community has found it necessary to build evaluation methodology around formalized intuition is itself a powerful indictment of the current state of benchmarking.

### 4.2 Best Practices

We propose several recommendations to follow to improve the state of LM evaluation on this front:

- Perform Statistical Analyses, and Report on Sources of Variance and Error: Instead of reporting isolated numbers, more effort should be made to contextualize the durability of performance estimates made by benchmarks, such as by reporting variance over several runs, or reporting statistical significance. These practices are seldom done in the field.
- Evaluate Models in Realistic Settings: Popular knowledge-centric evaluation benchmarks use prompting set-ups that are much more similar to how examination questions for humans are written than how people actually use language models ([Son et al., 2024](https://arxiv.org/html/2405.14782v3#bib.bib85); [Rein et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib22)). To improve the usefulness and validity of evaluations, more effort should be made to create test environments and datasets that mirror the real-world use-cases and deployment settings desired for a given model.
- Do Exploratory Analysis to Build Better Understanding of Results: To understand why a model is scoring so well or so poorly, it is important to do some sort of qualitative error analysis. This can sometimes reveal superficial errors that are easier to correct with post-processing ([Fourrier et al., 2023a](https://arxiv.org/html/2405.14782v3#bib.bib98); [Bawden and Yvon, 2023](https://arxiv.org/html/2405.14782v3#bib.bib16)), or more fundamental errors. Such practices allow more holistic conclusions and comparisons between methods, and increase confidence in evaluation correctness.

### 4.3 Operationalizing Best Practices

Improving the interpretation and nuanced consideration of benchmarks is primarily an issue of effort and defaults. Because considerations such as reporting statistical significance or measures of uncertainty require additional effort beyond the “standard” of simply receiving a higher-is-better evaluation number to report, they are not commonly done. We have sought to shift this status quo, as with our goals around reproducibility, via by-default opting users of lm-eval  into better evaluation reporting. We report bootstrapped standard error metrics by default, making reporting of confidence intervals (as we have done in this paper) nearly as simple as copy-pasting an additional number into one’s paper.

It is important to note that bootstrap confidence intervals are a slightly different notion of variation than the example given in Section 4.1. Language models are stochastic objects, so one natural notion of variation would be to run the model several times and measure how the results change (as done in Section 4.1). The bootstrap CIs offered in lm-eval  by contrast are oriented towards answering a different question: suppose that the particular questions in a given evaluation benchmark are samples from some distribution of interesting questions in the world. How much variation in model behavior should we expect as we sample new questions from the same distribution? We believe that both of these notions of variation (as well as variation across prompts and scoring systems, as previously discussed) are interesting and valuable.

We also seek to encourage future study of variation across prompts--as part of the
BigScience Research Workshop55
5
<https://bigscience.huggingface.co>, we developed tooling using the PromptSource library ([Bach et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib6)) for multiprompt evaluation, enabling users to run many different prompts for the same task and analyze the variance across prompt choices. Most papers that came out of the BigScience Workshop used this functionality to report distributions of scores across different prompting set-ups ([Sanh et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib5); [Muennighoff et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib91); [Yong et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib7); [BigScience Workshop et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib87)). PromptSource, along with other innovations introduced in the BigScience fork66
6
<https://github.com/bigscience-workshop/lm-evaluation-harness>, is now supported natively in lm-eval.

## 5 Conclusion

In this manuscript, we have highlighted a number of challenges that are ubiquitous in LM evaluation. We have highlighted and attempted to document a number of items of “folk knowledge” around benchmarking best practices that is either barely-documented or scattered across many sources. We then have highlighted our recommendations for mitigating many of these challenges’ impacts, and discussed how, in our work, we have incorporated such practices into the design of lm-eval, which has been widely used to mitigate the impact of such evaluation challenges for many practitioners. We hope that this work will draw more attention to the shortcomings of current LM evaluation, and will serve as a useful resource to practitioners in performing rigorous evaluation and charting effective paths forward.

## References

- Alzahrani et al. (2024)
  N. Alzahrani, H. A. Alyahya, Y. Alnumay, S. Alrashed, S. Alsubaie, Y. Almushaykeh, F. Mirza, N. Alotaibi, N. Altwairesh, A. Alowisheq, et al.
  When benchmarks are targets: revealing the sensitivity of large language model leaderboards.
  arXiv preprint arXiv:2402.01781.
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1),
  [§3.3](https://arxiv.org/html/2405.14782v3#S3.SS3.p2.1).
- Amidei et al. (2020)
  J. Amidei, P. Piwek, and A. Willis
  Identifying annotator bias: a new IRT-based method for bias identification.
  In Proceedings of the 28th International Conference on Computational Linguistics, D. Scott, N. Bel, and C. Zong (Eds.),
  Barcelona, Spain (Online), pp. 4787–4797.
  External Links: [Link](https://aclanthology.org/2020.coling-main.421),
  [Document](https://dx.doi.org/10.18653/v1/2020.coling-main.421)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Anil et al. (2023)
  R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al.
  Palm 2 technical report.
  arXiv preprint arXiv:2305.10403.
  Cited by: [§2.2](https://arxiv.org/html/2405.14782v3#S2.SS2.p2.1).
- Askell et al. (2021)
  A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan
  A general language assistant as a laboratory for alignment.
  External Links: 2112.00861
  Cited by: [3rd item](https://arxiv.org/html/2405.14782v3#A2.I2.i3.p1.1).
- Bach et al. (2022)
  S. H. Bach, V. Sanh, Z. Yong, A. Webson, C. Raffel, N. V. Nayak, A. Sharma, T. Kim, M. S. Bari, T. Fevry, Z. Alyafeai, M. Dey, A. Santilli, Z. Sun, S. Ben-David, C. Xu, G. Chhablani, H. Wang, J. A. Fries, M. S. Al-shaibani, S. Sharma, U. Thakker, K. Almubarak, X. Tang, D. Radev, M. T. Jiang, and A. M. Rush
  PromptSource: an integrated development environment and repository for natural language prompts.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations,
  pp. 93–104.
  Cited by: [§4.3](https://arxiv.org/html/2405.14782v3#S4.SS3.p3.1).
- Bawden and Yvon (2023)
  R. Bawden and F. Yvon
  Investigating the translation performance of a large multilingual language model: the case of BLOOM.
  In Proceedings of the 24th Annual Conference of the European Association for Machine Translation,
  External Links: [Link](https://arxiv.org/abs/2303.01911)
  Cited by: [Appendix D](https://arxiv.org/html/2405.14782v3#A4.SS0.SSS0.Px4.p1.1),
  [3rd item](https://arxiv.org/html/2405.14782v3#S4.I1.i3.p1.1).
- Beeching et al. (2023)
  E. Beeching, C. Fourrier, N. Habib, S. Han, N. Lambert, N. Rajani, O. Sanseviero, L. Tunstall, and T. Wolf
  Open llm leaderboard.
   Hugging Face.
  Note: <https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p1.1).
- Bender and Friedman (2018)
  E. M. Bender and B. Friedman
  Data statements for natural language processing: toward mitigating system bias and enabling better science.
  Transactions of the Association for Computational Linguistics 6, pp. 587–604.
  External Links: [Link](https://aclanthology.org/Q18-1041),
  [Document](https://dx.doi.org/10.1162/tacl%5Fa%5F00041)
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p1.1).
- BigScience Workshop et al. (2023)
  BigScience Workshop, T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilić, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon, M. Gallé, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot, N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan-Major, I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurençon, Y. Jernite, J. Launay, M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji, A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue, C. Klamm, C. Leong, D. van Strien, D. I. Adelani, D. Radev, E. G. Ponferrada, E. Levkovizh, E. Kim, E. B. Natan, F. D. Toni, G. Dupont, G. Kruszewski, G. Pistilli, H. Elsahar, H. Benyamina, H. Tran, I. Yu, I. Abdulmumin, I. Johnson, I. Gonzalez-Dios, J. de la Rosa, J. Chim, J. Dodge, J. Zhu, J. Chang, J. Frohberg, J. Tobing, J. Bhattacharjee, K. Almubarak, K. Chen, K. Lo, L. V. Werra, L. Weber, L. Phan, L. B. allal, L. Tanguy, M. Dey, M. R. Muñoz, M. Masoud, M. Grandury, M. Šaško, M. Huang, M. Coavoux, M. Singh, M. T. Jiang, M. C. Vu, M. A. Jauhar, M. Ghaleb, N. Subramani, N. Kassner, N. Khamis, O. Nguyen, O. Espejel, O. de Gibert, P. Villegas, P. Henderson, P. Colombo, P. Amuok, Q. Lhoest, R. Harliman, R. Bommasani, R. L. López, R. Ribeiro, S. Osei, S. Pyysalo, S. Nagel, S. Bose, S. H. Muhammad, S. Sharma, S. Longpre, S. Nikpoor, S. Silberberg, S. Pai, S. Zink, T. T. Torrent, T. Schick, T. Thrush, V. Danchev, V. Nikoulina, V. Laippala, V. Lepercq, V. Prabhu, Z. Alyafeai, Z. Talat, A. Raja, B. Heinzerling, C. Si, D. E. Taşar, E. Salesky, S. J. Mielke, W. Y. Lee, A. Sharma, A. Santilli, A. Chaffin, A. Stiegler, D. Datta, E. Szczechla, G. Chhablani, H. Wang, H. Pandey, H. Strobelt, J. A. Fries, J. Rozen, L. Gao, L. Sutawika, M. S. Bari, M. S. Al-shaibani, M. Manica, N. Nayak, R. Teehan, S. Albanie, S. Shen, S. Ben-David, S. H. Bach, T. Kim, T. Bers, T. Fevry, T. Neeraj, U. Thakker, V. Raunak, X. Tang, Z. Yong, Z. Sun, S. Brody, Y. Uri, H. Tojarieh, A. Roberts, H. W. Chung, J. Tae, J. Phang, O. Press, C. Li, D. Narayanan, H. Bourfoune, J. Casper, J. Rasley, M. Ryabinin, M. Mishra, M. Zhang, M. Shoeybi, M. Peyrounette, N. Patry, N. Tazi, O. Sanseviero, P. von Platen, P. Cornette, P. F. Lavallée, R. Lacroix, S. Rajbhandari, S. Gandhi, S. Smith, S. Requena, S. Patil, T. Dettmers, A. Baruwa, A. Singh, A. Cheveleva, A. Ligozat, A. Subramonian, A. Névéol, C. Lovering, D. Garrette, D. Tunuguntla, E. Reiter, E. Taktasheva, E. Voloshina, E. Bogdanov, G. I. Winata, H. Schoelkopf, J. Kalo, J. Novikova, J. Z. Forde, J. Clive, J. Kasai, K. Kawamura, L. Hazan, M. Carpuat, M. Clinciu, N. Kim, N. Cheng, O. Serikov, O. Antverg, O. van der Wal, R. Zhang, R. Zhang, S. Gehrmann, S. Mirkin, S. Pais, T. Shavrina, T. Scialom, T. Yun, T. Limisiewicz, V. Rieser, V. Protasov, V. Mikhailov, Y. Pruksachatkun, Y. Belinkov, Z. Bamberger, Z. Kasner, A. Rueda, A. Pestana, A. Feizpour, A. Khan, A. Faranak, A. Santos, A. Hevia, A. Unldreaj, A. Aghagol, A. Abdollahi, A. Tammour, A. HajiHosseini, B. Behroozi, B. Ajibade, B. Saxena, C. M. Ferrandis, D. McDuff, D. Contractor, D. Lansky, D. David, D. Kiela, D. A. Nguyen, E. Tan, E. Baylor, E. Ozoani, F. Mirza, F. Ononiwu, H. Rezanejad, H. Jones, I. Bhattacharya, I. Solaiman, I. Sedenko, I. Nejadgholi, J. Passmore, J. Seltzer, J. B. Sanz, L. Dutra, M. Samagaio, M. Elbadri, M. Mieskes, M. Gerchick, M. Akinlolu, M. McKenna, M. Qiu, M. Ghauri, M. Burynok, N. Abrar, N. Rajani, N. Elkott, N. Fahmy, O. Samuel, R. An, R. Kromann, R. Hao, S. Alizadeh, S. Shubber, S. Wang, S. Roy, S. Viguier, T. Le, T. Oyebade, T. Le, Y. Yang, Z. Nguyen, A. R. Kashyap, A. Palasciano, A. Callahan, A. Shukla, A. Miranda-Escalada, A. Singh, B. Beilharz, B. Wang, C. Brito, C. Zhou, C. Jain, C. Xu, C. Fourrier, D. L. Periñán, D. Molano, D. Yu, E. Manjavacas, F. Barth, F. Fuhrimann, G. Altay, G. Bayrak, G. Burns, H. U. Vrabec, I. Bello, I. Dash, J. Kang, J. Giorgi, J. Golde, J. D. Posada, K. R. Sivaraman, L. Bulchandani, L. Liu, L. Shinzato, M. H. de Bykhovetz, M. Takeuchi, M. Pàmies, M. A. Castillo, M. Nezhurina, M. Sänger, M. Samwald, M. Cullan, M. Weinberg, M. D. Wolf, M. Mihaljcic, M. Liu, M. Freidank, M. Kang, N. Seelam, N. Dahlberg, N. M. Broad, N. Muellner, P. Fung, P. Haller, R. Chandrasekhar, R. Eisenberg, R. Martin, R. Canalli, R. Su, R. Su, S. Cahyawijaya, S. Garda, S. S. Deshmukh, S. Mishra, S. Kiblawi, S. Ott, S. Sang-aroonsiri, S. Kumar, S. Schweter, S. Bharati, T. Laud, T. Gigant, T. Kainuma, W. Kusa, Y. Labrak, Y. S. Bajaj, Y. Venkatraman, Y. Xu, Y. Xu, Y. Xu, Z. Tan, Z. Xie, Z. Ye, M. Bras, Y. Belkada, and T. Wolf
  BLOOM: a 176b-parameter open-access multilingual language model.
  External Links: 2211.05100
  Cited by: [§4.3](https://arxiv.org/html/2405.14782v3#S4.SS3.p3.1).
- Black et al. (2022)
  S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell, J. Phang, M. Pieler, U. S. Prashanth, S. Purohit, L. Reynolds, J. Tow, B. Wang, and S. Weinbach
  GPT-NeoX-20B: an open-source autoregressive language model.
  In Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models,
  External Links: [Link](https://arxiv.org/abs/2204.06745)
  Cited by: [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p4.1),
  [Table 1](https://arxiv.org/html/2405.14782v3#S3.T1).
- Brown et al. (1992)
  P. F. Brown, S. A. Della Pietra, V. J. Della Pietra, J. C. Lai, and R. L. Mercer
  An estimate of an upper bound for the entropy of English.
  Computational Linguistics 18 (1), pp. 31–40.
  Cited by: [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p1.1).
- Brown et al. (2020)
  T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al.
  Language models are few-shot learners.
  arXiv preprint arXiv:2005.14165.
  External Links: [Link](https://arxiv.org/abs/2005.14165)
  Cited by: [Appendix A](https://arxiv.org/html/2405.14782v3#A1.SS0.SSS0.Px1.p2.1),
  [Appendix A](https://arxiv.org/html/2405.14782v3#A1.SS0.SSS0.Px3.p3.1),
  [1st item](https://arxiv.org/html/2405.14782v3#A2.I2.i1.p1.1),
  [3rd item](https://arxiv.org/html/2405.14782v3#A2.I2.i3.p1.1),
  [§B.2](https://arxiv.org/html/2405.14782v3#A2.SS2.p6.1),
  [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p3.1),
  [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p4.1),
  [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p9.1),
  [Figure 1](https://arxiv.org/html/2405.14782v3#S2.F1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px3.p2.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p3.1).
- Callison-Burch et al. (2006)
  C. Callison-Burch, M. Osborne, and P. Koehn
  Re-evaluating the role of Bleu in machine translation research.
  In 11th Conference of the European Chapter of the Association for Computational Linguistics, D. McCarthy and S. Wintner (Eds.),
  Trento, Italy, pp. 249–256.
  External Links: [Link](https://aclanthology.org/E06-1032)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Card et al. (2020)
  D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky
  With little power comes great responsibility.
  External Links: 2010.06595
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p1.1).
- Chen et al. (2024)
  G. H. Chen, S. Chen, Z. Liu, F. Jiang, and B. Wang
  Humans or llms as the judge? a study on judgement biases.
  ArXiv abs/2402.10669.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:267740522)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Chen et al. (2021)
  M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba
  Evaluating large language models trained on code.
  arXiv.
  External Links: 2107.03374
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px2.p3.1).
- Chowdhery et al. (2023)
  A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel
  PaLM: scaling language modeling with pathways.
  Journal of Machine Learning Research 24 (240), pp. 1–113.
  External Links: [Link](http://jmlr.org/papers/v24/22-1144.html)
  Cited by: [§2.2](https://arxiv.org/html/2405.14782v3#S2.SS2.p2.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p3.1).
- Clark et al. (2018)
  P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord
  Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.
  External Links: 1803.05457
  Cited by: [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p2.1),
  [item 2](https://arxiv.org/html/2405.14782v3#S2.I1.i2.p1.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p2.1).
- Davis (2023)
  E. Davis
  Benchmarks for automated commonsense reasoning: a survey.
  ACM Computing Surveys 56 (4), pp. 1–41.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Dehghani et al. (2021)
  M. Dehghani, Y. Tay, A. A. Gritsenko, Z. Zhao, N. Houlsby, F. Diaz, D. Metzler, and O. Vinyals
  The benchmark lottery.
  External Links: 2107.07002
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p1.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p1.1).
- Devlin et al. (2018)
  J. Devlin, M. Chang, K. Lee, and K. Toutanova
  Bert: pre-training of deep bidirectional transformers for language understanding.
  arXiv preprint arXiv:1810.04805.
  Cited by: [Figure 1](https://arxiv.org/html/2405.14782v3#S2.F1).
- Dua et al. (2019)
  D. Dua, Y. Wang, P. Dasigi, G. Stanovsky, S. Singh, and M. Gardner
  DROP: a reading comprehension benchmark requiring discrete reasoning over paragraphs.
  arXiv preprint arXiv:1903.00161.
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Dunlap et al. (2024)
  L. Dunlap, K. Mandal, T. Darrell, J. Steinhardt, and J. E. Gonzalez
  VibeCheck: discover and quantify qualitative differences in large language models.
  arXiv preprint arXiv:2410.12851.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p4.1).
- Fourrier et al. (2023a)
  C. Fourrier, A. Cabrera, S. Biderman, N. Habib, and T. Wolf
  Open LLM leaderboard: DROP deep dive.
  Note: Hugging Face Blog
  External Links: [Link](https://huggingface.co/blog/open-llm-leaderboard-drop)
  Cited by: [3rd item](https://arxiv.org/html/2405.14782v3#S4.I1.i3.p1.1).
- Fourrier et al. (2023b)
  C. Fourrier, N. Habib, J. Launay, and T. Wolf
  What’s going on with the open LLM leaderboard?.
  Note: Hugging Face Blog
  External Links: [Link](https://huggingface.co/blog/open-llm-leaderboard-mmlu)
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p2.1).
- Freitag et al. (2021)
  M. Freitag, G. Foster, D. Grangier, V. Ratnakar, Q. Tan, and W. Macherey
  Experts, errors, and context: a large-scale study of human evaluation for machine translation.
  Transactions of the Association for Computational Linguistics 9, pp. 1460–1474.
  External Links: [Link](https://aclanthology.org/2021.tacl-1.87),
  [Document](https://dx.doi.org/10.1162/tacl%5Fa%5F00437)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Ganguli et al. (2023)
  D. Ganguli, N. Schiefer, M. Favaro, and J. Clark
  Challenges in evaluating AI systems.
  External Links: [Link](https://www.anthropic.com/index/evaluating-ai-systems)
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p1.1).
- Gao et al. (2020)
  L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy
  The Pile: an 800GB dataset of diverse text for language modeling.
  arXiv preprint arXiv:2101.00027.
  External Links: [Link](https://arxiv.org/abs/2101.00027)
  Cited by: [Appendix A](https://arxiv.org/html/2405.14782v3#A1.SS0.SSS0.Px3.p3.1),
  [1st item](https://arxiv.org/html/2405.14782v3#A2.I3.i1.p1.1),
  [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p11.1),
  [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p12.1),
  [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p7.1),
  [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p9.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p3.1).
- Gao et al. (2023)
  L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou
  A framework for few-shot language model evaluation.
   Zenodo.
  External Links: [Document](https://dx.doi.org/10.5281/zenodo.10256836),
  [Link](https://zenodo.org/records/10256836)
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p2.1),
  [Abstract](https://arxiv.org/html/2405.14782v3#abstract1.1).
- Gao (2021)
  L. Gao
  Multiple Choice Normalization in LM Evaluation.
  Note: EleutherAI Blog
  External Links: [Link](https://blog.eleuther.ai/multiple-choice-normalization/)
  Cited by: [§B.2](https://arxiv.org/html/2405.14782v3#A2.SS2.p10.1).
- Gu et al. (2024)
  Y. Gu, O. Tafjord, B. Kuehl, D. Haddad, J. Dodge, and H. Hajishirzi
  Olmes: a standard for language model evaluations.
  arXiv preprint arXiv:2406.08446.
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Hendrycks et al. (2020)
  D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt
  Measuring massive multitask language understanding.
  In International Conference on Learning Representations,
  Cited by: [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p11.1),
  [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p3.1),
  [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p2.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Hendrycks et al. (2021)
  D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt
  Measuring mathematical problem solving with the math dataset.
  External Links: 2103.03874
  Cited by: [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p2.1),
  [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Hoffmann et al. (2022)
  J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre
  Training compute-optimal large language models.
  External Links: 2203.15556
  Cited by: [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p7.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p3.1).
- Hofmann et al. (2024)
  V. Hofmann, P. R. Kalluri, D. Jurafsky, and S. King
  Dialect prejudice predicts ai decisions about people’s character, employability, and criminality.
  arXiv preprint arXiv:2403.00742.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Holtzman et al. (2020)
  A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi
  The curious case of neural text degeneration.
  External Links: 1904.09751
  Cited by: [§B.4](https://arxiv.org/html/2405.14782v3#A2.SS4.p4.1).
- Holtzman et al. (2022)
  A. Holtzman, P. West, V. Shwartz, Y. Choi, and L. Zettlemoyer
  Surface form competition: why the highest probability answer isn’t always right.
  External Links: 2104.08315
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Hosking et al. (2024)
  T. Hosking, P. Blunsom, and M. Bartolo
  Human feedback is not gold standard.
  External Links: 2309.16349
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px3.p1.1).
- Hu et al. (2024)
  X. Hu, M. Gao, S. Hu, Y. Zhang, Y. Chen, T. Xu, and X. Wan
  Are LLM-based Evaluators Confusing NLG Quality Criteria?.
  ArXiv abs/2402.12055.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:267750948)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Jelinek et al. (2005)
  F. Jelinek, R. L. Mercer, L. R. Bahl, and J. K. Baker
  Perplexity—a measure of the difficulty of speech recognition tasks.
  The Journal of the Acoustical Society of America 62 (S1), pp. S63–S63.
  External Links: ISSN 0001-4966,
  [Document](https://dx.doi.org/10.1121/1.2016299),
  [Link](https://doi.org/10.1121/1.2016299),
  https://pubs.aip.org/asa/jasa/article-pdf/62/S1/S63/11558910/s63_5_online.pdf
  Cited by: [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p1.1).
- Jiang et al. (2023)
  A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed
  Mistral 7b.
  External Links: 2310.06825
  Cited by: [Table 1](https://arxiv.org/html/2405.14782v3#S3.T1).
- Jiang et al. (2024)
  A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed
  Mixtral of experts.
  External Links: 2401.04088
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p2.1),
  [Table 1](https://arxiv.org/html/2405.14782v3#S3.T1).
- Joshi et al. (2017)
  M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer
  TriviaQA: a large scale distantly supervised challenge dataset for reading comprehension.
  External Links: 1705.03551
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Kapoor et al. (2024)
  S. Kapoor, B. Stroebl, Z. S. Siegel, N. Nadgir, and A. Narayanan
  AI agent benchmarks that matter.
   Princeton University.
  Note: Manuscript submitted for publication
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p2.1).
- Kim et al. (2024)
  S. Kim, J. Shin, Y. Cho, J. Jang, S. Longpre, H. Lee, S. Yun, S. Shin, S. Kim, J. Thorne, and M. Seo
  Prometheus: inducing fine-grained evaluation capability in language models.
  External Links: 2310.08491
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Kurt et al. (2024)
  W. Kurt, R. Louf, and C. Fourrier
  Improving prompt consistency with structured generations.
  Note: Hugging Face Blog
  External Links: [Link](https://huggingface.co/blog/evaluation-structured-outputs)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Lewkowycz et al. (2022)
  A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al.
  Solving quantitative reasoning problems with language models.
  Advances in Neural Information Processing Systems 35, pp. 3843–3857.
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1),
  [§2.2](https://arxiv.org/html/2405.14782v3#S2.SS2.p2.1).
- Lhoest et al. (2021)
  Q. Lhoest, A. Villanova del Moral, Y. Jernite, A. Thakur, P. von Platen, S. Patil, J. Chaumond, M. Drame, J. Plu, L. Tunstall, J. Davison, M. Šaško, G. Chhablani, B. Malik, S. Brandeis, T. Le Scao, V. Sanh, C. Xu, N. Patry, A. McMillan-Major, P. Schmid, S. Gugger, C. Delangue, T. Matussière, L. Debut, S. Bekman, P. Cistac, T. Goehringer, V. Mustar, F. Lagunas, A. Rush, and T. Wolf
  Datasets: a community library for natural language processing.
  In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, H. Adel and S. Shi (Eds.),
  Online and Punta Cana, Dominican Republic, pp. 175–184.
  External Links: [Link](https://aclanthology.org/2021.emnlp-demo.21),
  [Document](https://dx.doi.org/10.18653/v1/2021.emnlp-demo.21)
  Cited by: [Figure 3](https://arxiv.org/html/2405.14782v3#A1.F3),
  [1st item](https://arxiv.org/html/2405.14782v3#A3.I1.i1.p1.1.1).
- Li et al. (2016)
  J. Li, W. Monroe, A. Ritter, D. Jurafsky, M. Galley, and J. Gao
  Deep reinforcement learning for dialogue generation.
  In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, J. Su, K. Duh, and X. Carreras (Eds.),
  Austin, Texas, pp. 1192–1202.
  External Links: [Link](https://aclanthology.org/D16-1127),
  [Document](https://dx.doi.org/10.18653/v1/D16-1127)
  Cited by: [§B.4](https://arxiv.org/html/2405.14782v3#A2.SS4.p4.1).
- Liang et al. (2023)
  P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, B. Newman, B. Yuan, B. Yan, C. Zhang, C. Cosgrove, C. D. Manning, C. Ré, D. Acosta-Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong, H. Ren, H. Yao, J. Wang, K. Santhanam, L. Orr, L. Zheng, M. Yuksekgonul, M. Suzgun, N. Kim, N. Guha, N. Chatterji, O. Khattab, P. Henderson, Q. Huang, R. Chi, S. M. Xie, S. Santurkar, S. Ganguli, T. Hashimoto, T. Icard, T. Zhang, V. Chaudhary, W. Wang, X. Li, Y. Mai, Y. Zhang, and Y. Koreeda
  Holistic evaluation of language models.
  External Links: 2211.09110
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p2.1),
  [§3.3](https://arxiv.org/html/2405.14782v3#S3.SS3.p1.1).
- Lin (2004)
  C. Lin
  Rouge: a package for automatic evaluation of summaries.
  In Text summarization branches out,
  pp. 74–81.
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Lin et al. (2022)
  S. Lin, J. Hilton, and O. Evans
  TruthfulQA: measuring how models mimic human falsehoods.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio (Eds.),
  Dublin, Ireland, pp. 3214–3252.
  External Links: [Link](https://aclanthology.org/2022.acl-long.229),
  [Document](https://dx.doi.org/10.18653/v1/2022.acl-long.229)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Liu et al. (2023a)
  Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu
  G-eval: NLG evaluation using gpt-4 with better human alignment.
  In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali (Eds.),
  Singapore, pp. 2511–2522.
  External Links: [Link](https://aclanthology.org/2023.emnlp-main.153),
  [Document](https://dx.doi.org/10.18653/v1/2023.emnlp-main.153)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Liu et al. (2023b)
  Y. Liu, A. R. Fabbri, J. Chen, Y. Zhao, S. Han, S. Joty, P. Liu, D. Radev, C. Wu, and A. Cohan
  Benchmarking generation and evaluation capabilities of large language models for instruction controllable summarization.
  arXiv preprint arXiv:2311.09184.
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Li’evin et al. (2022)
  V. Li’evin, C. E. Hother, and O. Winther
  Can large language models reason about medical questions?.
  Patterns 5.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:250627547)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Lu et al. (2022)
  Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp
  Fantastically ordered prompts and where to find them: overcoming few-shot prompt order sensitivity.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio (Eds.),
  Dublin, Ireland, pp. 8086–8098.
  External Links: [Link](https://aclanthology.org/2022.acl-long.556),
  [Document](https://dx.doi.org/10.18653/v1/2022.acl-long.556)
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Lukošiūtė (2024)
  K. Lukošiūtė
  You need to be spending more money on evals.
  Note: <https://kamilelukosiute.com/llms/You+need+to+be+spending+more+money+on+evals>Accessed: 06 June 2024
  Cited by: [Figure 2](https://arxiv.org/html/2405.14782v3#S4.F2),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px2.p2.1).
- Lyu et al. (2024)
  C. Lyu, M. Wu, and A. F. Aji
  Beyond probabilities: unveiling the misalignment in evaluating large language models.
  External Links: 2402.13887
  Cited by: [§3.3](https://arxiv.org/html/2405.14782v3#S3.SS3.p2.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Magnusson et al. (2023)
  I. Magnusson, A. Bhagia, V. Hofmann, L. Soldaini, A. H. Jha, O. Tafjord, D. Schwenk, E. P. Walsh, Y. Elazar, K. Lo, D. Groeneveld, I. Beltagy, H. Hajishirzi, N. A. Smith, K. Richardson, and J. Dodge
  Paloma: a benchmark for evaluating language model fit.
  External Links: 2312.10523
  Cited by: [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p7.1).
- Marie et al. (2021)
  B. Marie, A. Fujita, and R. Rubino
  Scientific credibility of machine translation research: a meta-evaluation of 769 papers.
  In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers),
  Online, pp. 7297–7306.
  External Links: [Link](https://aclanthology.org/2021.acl-long.566),
  [Document](https://dx.doi.org/10.18653/v1/2021.acl-long.566)
  Cited by: [Appendix D](https://arxiv.org/html/2405.14782v3#A4.SS0.SSS0.Px2.p1.1),
  [Appendix D](https://arxiv.org/html/2405.14782v3#A4.SS0.SSS0.Px5.p1.1),
  [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Marie (2022)
  B. Marie
  Comparing the uncomparable to claim the state of the art: a concerning trend.
  Note: Blog post
  Cited by: [Appendix D](https://arxiv.org/html/2405.14782v3#A4.SS0.SSS0.Px2.p1.1).
- Messick (1994)
  S. Messick
  STANDARDS-based score interpretation: establishing valid grounds for valid inferences.
  ETS Research Report Series 1994 (2), pp. 291–305.
  External Links: [Document](https://dx.doi.org/https%3A//doi.org/10.1002/j.2333-8504.1994.tb01630.x),
  [Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.2333-8504.1994.tb01630.x),
  https://onlinelibrary.wiley.com/doi/pdf/10.1002/j.2333-8504.1994.tb01630.x
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Min et al. (2022)
  S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer
  Rethinking the role of demonstrations: what makes in-context learning work?.
  External Links: 2202.12837
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Mizrahi et al. (2024)
  M. Mizrahi, G. Kaplan, D. Malkin, R. Dror, D. Shahaf, and G. Stanovsky
  State of what art? a call for multi-prompt LLM evaluation.
  External Links: 2401.00595
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Muennighoff et al. (2022)
  N. Muennighoff, T. Wang, L. Sutawika, A. Roberts, S. Biderman, T. L. Scao, M. S. Bari, S. Shen, Z. Yong, H. Schoelkopf, et al.
  Crosslingual generalization through multitask finetuning.
  arXiv preprint arXiv:2211.01786.
  Cited by: [§4.3](https://arxiv.org/html/2405.14782v3#S4.SS3.p3.1).
- Nangia et al. (2020)
  N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman
  CrowS-pairs: a challenge dataset for measuring social biases in masked language models.
  In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohn, Y. He, and Y. Liu (Eds.),
  Online, pp. 1953–1967.
  External Links: [Link](https://aclanthology.org/2020.emnlp-main.154),
  [Document](https://dx.doi.org/10.18653/v1/2020.emnlp-main.154)
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Névéol et al. (2022)
  A. Névéol, Y. Dupont, J. Bezançon, and K. Fort
  French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English.
  In ACL 2022 - 60th Annual Meeting of the Association for Computational Linguistics,
  Dublin, Ireland.
  External Links: [Link](https://inria.hal.science/hal-03629677)
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Omiye et al. (2023)
  J. A. Omiye, J. C. Lester, S. Spichak, V. Rotemberg, and R. Daneshjou
  Large language models propagate race-based medicine.
  NPJ Digital Medicine 6 (1), pp. 195.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- OpenAI (2023)
  OpenAI
  GPT-4 technical report.
  ArXiv abs/2303.08774.
  External Links: [Link](https://arxiv.org/abs/2303.08774)
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p1.1).
- OpenCompass Contributors (2023)
  OpenCompass Contributors
  OpenCompass: a universal evaluation platform for foundation models.
  Note: <https://github.com/open-compass/opencompass>
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p2.1).
- Padlewski et al. (2024)
  P. Padlewski, M. Bain, M. Henderson, Z. Zhu, N. Relan, H. Pham, D. Ong, K. Aleksiev, A. Ormazabal, S. Phua, E. Yeo, E. Lamprecht, Q. Liu, Y. Wang, E. Chen, D. Fu, L. Li, C. Zheng, C. de Masson d’Autume, D. Yogatama, M. Artetxe, and Y. Tay
  Vibe-eval: a hard evaluation suite for measuring progress of multimodal language models.
  arXiv preprint arXiv:2405.02287.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p4.1).
- Paperno et al. (2016)
  D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández
  The lambada dataset: word prediction requiring a broad discourse context.
  arXiv preprint arXiv:1606.06031.
  Cited by: [item 2](https://arxiv.org/html/2405.14782v3#S2.I1.i2.p1.1).
- Papineni et al. (2002)
  K. Papineni, S. Roukos, T. Ward, and W. Zhu
  Bleu: a method for automatic evaluation of machine translation.
  In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, P. Isabelle, E. Charniak, and D. Lin (Eds.),
  Philadelphia, Pennsylvania, USA, pp. 311–318.
  External Links: [Link](https://aclanthology.org/P02-1040),
  [Document](https://dx.doi.org/10.3115/1073083.1073135)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Parrish et al. (2021)
  A. Parrish, A. Chen, N. Nangia, V. Padmakumar, J. Phang, J. Thompson, P. M. Htut, and S. R. Bowman
  BBQ: a hand-built bias benchmark for question answering.
  arXiv preprint arXiv:2110.08193.
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p1.1).
- Penedo et al. (2023)
  G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, A. Cappelli, H. Alobeidli, B. Pannier, E. Almazrouei, and J. Launay
  The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only.
  External Links: 2306.01116
  Cited by: [§C.1](https://arxiv.org/html/2405.14782v3#A3.SS1.p4.1),
  [Table 1](https://arxiv.org/html/2405.14782v3#S3.T1).
- Peng et al. (2023)
  B. Peng, J. Quesnelle, H. Fan, and E. Shippole
  YaRN: efficient context window extension of large language models.
  External Links: 2309.00071
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p4.1).
- Press et al. (2020)
  O. Press, N. A. Smith, and M. Lewis
  Shortformer: better language modeling using shorter inputs.
  CoRR abs/2012.15832.
  External Links: [Link](https://arxiv.org/abs/2012.15832),
  2012.15832
  Cited by: [§B.3](https://arxiv.org/html/2405.14782v3#A2.SS3.p10.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p4.1).
- Provilkov et al. (2020)
  I. Provilkov, D. Emelianenko, and E. Voita
  BPE-dropout: simple and effective subword regularization.
  External Links: 1910.13267
  Cited by: [§B.2](https://arxiv.org/html/2405.14782v3#A2.SS2.p17.1).
- Radford et al. (2019)
  A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever
  Language models are unsupervised multitask learners.
  OpenAI Blog.
  Cited by: [Figure 1](https://arxiv.org/html/2405.14782v3#S2.F1).
- Raji et al. (2021)
  I. D. Raji, E. M. Bender, A. Paullada, E. Denton, and A. Hanna
  AI and the Everything in the Whole Wide World Benchmark.
  External Links: 2111.15366
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Raji et al. (2022)
  I. D. Raji, I. E. Kumar, A. Horowitz, and A. Selbst
  The fallacy of ai functionality.
  In 2022 ACM Conference on Fairness, Accountability, and Transparency,
  FAccT ’22.
  External Links: [Link](http://dx.doi.org/10.1145/3531146.3533158),
  [Document](https://dx.doi.org/10.1145/3531146.3533158)
  Cited by: [§1](https://arxiv.org/html/2405.14782v3#S1.p1.1).
- Rein et al. (2023)
  D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman
  GPQA: a graduate-level google-proof q&a benchmark.
  External Links: 2311.12022
  Cited by: [2nd item](https://arxiv.org/html/2405.14782v3#S4.I1.i2.p1.1).
- Robinson et al. (2023)
  J. Robinson, C. M. Rytting, and D. Wingate
  Leveraging large language models for multiple choice question answering.
  External Links: 2210.12353
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Sanh et al. (2022)
  V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, T. Le Scao, A. Raja, et al.
  Multitask prompted training enables zero-shot task generalization.
  In The Tenth International Conference on Learning Representations,
  Cited by: [§B.5](https://arxiv.org/html/2405.14782v3#A2.SS5.p2.1),
  [§4.3](https://arxiv.org/html/2405.14782v3#S4.SS3.p3.1).
- Saphra et al. (2023)
  N. Saphra, E. Fleisig, K. Cho, and A. Lopez
  First tragedy, then parse: history repeats itself in the new era of large language models.
  arXiv preprint.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Schaeffer et al. (2023)
  R. Schaeffer, B. Miranda, and S. Koyejo
  Are emergent abilities of large language models a mirage?.
  External Links: 2304.15004
  Cited by: [§B.5](https://arxiv.org/html/2405.14782v3#A2.SS5.p2.1).
- Sclar et al. (2023)
  M. Sclar, Y. Choi, Y. Tsvetkov, and A. Suhr
  Quantifying language models’ sensitivity to spurious features in prompt design or: how I learned to start worrying about prompt formatting.
  External Links: 2310.11324
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Sellam et al. (2022)
  T. Sellam, S. Yadlowsky, I. Tenney, J. Wei, N. Saphra, A. D’Amour, T. Linzen, J. Bastings, I. R. Turc, J. Eisenstein, D. Das, and E. Pavlick
  The multiBERTs: BERT reproductions for robustness analysis.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=K0E_F0gFDgA)
  Cited by: [Appendix D](https://arxiv.org/html/2405.14782v3#A4.SS0.SSS0.Px5.p1.1).
- Shannon (1948)
  C. E. Shannon
  A mathematical theory of communication.
  The Bell System Technical Journal 27 (3), pp. 379–423.
  External Links: [Document](https://dx.doi.org/10.1002/j.1538-7305.1948.tb01338.x)
  Cited by: [3rd item](https://arxiv.org/html/2405.14782v3#A2.I2.i3.p1.1).
- Shen et al. (2023)
  C. Shen, L. Cheng, X. Nguyen, Y. You, and L. Bing
  Large language models are not yet human-level evaluators for abstractive summarization.
  In Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali (Eds.),
  Singapore, pp. 4215–4233.
  External Links: [Link](https://aclanthology.org/2023.findings-emnlp.278),
  [Document](https://dx.doi.org/10.18653/v1/2023.findings-emnlp.278)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Singhal et al. (2023)
  K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, L. Hou, K. Clark, S. Pfohl, H. Cole-Lewis, D. Neal, et al.
  Towards expert-level medical question answering with large language models.
  arXiv preprint arXiv:2305.09617.
  Cited by: [§2.2](https://arxiv.org/html/2405.14782v3#S2.SS2.p2.1).
- Son et al. (2024)
  G. Son, H. Lee, S. Kim, S. Kim, N. Muennighoff, T. Choi, C. Park, K. M. Yoo, and S. Biderman
  KMMLU: measuring massive multitask language understanding in korean.
  External Links: 2402.11548
  Cited by: [2nd item](https://arxiv.org/html/2405.14782v3#S4.I1.i2.p1.1).
- Srivastava et al. (2022)
  A. Srivastava, A. Rastogi, A. B. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, A. Kluska, A. Lewkowycz, A. Agarwal, A. Power, A. Ray, A. Warstadt, A. W. Kocurek, A. Safaya, A. Tazarv, A. Xiang, A. Parrish, A. Nie, A. Hussain, A. Askell, A. Dsouza, A. A. Rahane, A. S. Iyer, A. J. Andreassen, A. Santilli, A. Stuhlmuller, A. M. Dai, A. D. La, A. K. Lampinen, A. Zou, A. Jiang, A. Chen, A. Vuong, A. Gupta, A. Gottardi, A. Norelli, A. Venkatesh, A. Gholamidavoodi, A. Tabassum, A. Menezes, A. Kirubarajan, A. Mullokandov, A. Sabharwal, A. Herrick, A. Efrat, A. Erdem, A. Karakacs, B. R. Roberts, B. S. Loe, B. Zoph, B. Bojanowski, B. Ozyurt, B. Hedayatnia, B. Neyshabur, B. Inden, B. Stein, B. Ekmekci, B. Y. Lin, B. S. Howald, C. Diao, C. Dour, C. Stinson, C. Argueta, C. F. Ram’irez, C. Singh, C. Rathkopf, C. Meng, C. Baral, C. Wu, C. Callison-Burch, C. Waites, C. Voigt, C. D. Manning, C. Potts, C. T. Ramirez, C. E. Rivera, C. Siro, C. Raffel, C. Ashcraft, C. Garbacea, D. Sileo, D. H. Garrette, D. Hendrycks, D. Kilman, D. Roth, D. Freeman, D. Khashabi, D. Levy, D. Gonz’alez, D. Hernandez, D. Chen, D. Ippolito, D. Gilboa, D. Dohan, D. Drakard, D. Jurgens, D. Datta, D. Ganguli, D. Emelin, D. Kleyko, D. Yuret, D. Chen, D. Tam, D. Hupkes, D. Misra, D. Buzan, D. C. Mollo, D. Yang, D. Lee, E. Shutova, E. D. Cubuk, E. Segal, E. Hagerman, E. Barnes, E. P. Donoway, E. Pavlick, E. Rodolà, E. F. Lam, E. Chu, E. Tang, E. Erdem, E. Chang, E. A. Chi, E. Dyer, E. Jerzak, E. Kim, E. E. Manyasi, E. Zheltonozhskii, F. Xia, F. Siar, F. Mart’inez-Plumed, F. Happ’e, F. Chollet, F. Rong, G. Mishra, G. I. Winata, G. de Melo, G. Kruszewski, G. Parascandolo, G. Mariani, G. Wang, G. Jaimovitch-L’opez, G. Betz, G. Gur-Ari, H. Galijasevic, H. S. Kim, H. Rashkin, H. Hajishirzi, H. Mehta, H. Bogar, H. Shevlin, H. Schütze, H. Yakura, H. Zhang, H. Wong, I. A. Ng, I. Noble, J. Jumelet, J. Geissinger, J. Kernion, J. Hilton, J. Lee, J. F. Fisac, J. B. Simon, J. Koppel, J. Zheng, J. Zou, J. Koco’n, J. Thompson, J. Kaplan, J. Radom, J. Sohl-Dickstein, J. Phang, J. Wei, J. Yosinski, J. Novikova, J. Bosscher, J. Marsh, J. Kim, J. Taal, J. Engel, J. O. Alabi, J. Xu, J. Song, J. Tang, J. W. Waweru, J. Burden, J. Miller, J. U. Balis, J. Berant, J. Frohberg, J. Rozen, J. Hernández-Orallo, J. Boudeman, J. Jones, J. B. Tenenbaum, J. S. Rule, J. Chua, K. Kanclerz, K. Livescu, K. Krauth, K. Gopalakrishnan, K. Ignatyeva, K. Markert, K. D. Dhole, K. Gimpel, K. O. Omondi, K. W. Mathewson, K. Chiafullo, K. Shkaruta, K. Shridhar, K. McDonell, K. Richardson, L. Reynolds, L. Gao, L. Zhang, L. Dugan, L. Qin, L. Contreras-Ochando, L. Morency, L. Moschella, L. Lam, L. Noble, L. Schmidt, L. He, L. O. Col’on, L. Metz, L. K. cSenel, M. Bosma, M. Sap, M. ter Hoeve, M. Andrea, M. S. Farooqi, M. Faruqui, M. Mazeika, M. Baturan, M. Marelli, M. Maru, M. Quintana, M. Tolkiehn, M. Giulianelli, M. Lewis, M. Potthast, M. Leavitt, M. Hagen, M. Schubert, M. Baitemirova, M. Arnaud, M. A. McElrath, M. A. Yee, M. Cohen, M. Gu, M. I. Ivanitskiy, M. Starritt, M. Strube, M. Swkedrowski, M. Bevilacqua, M. Yasunaga, M. Kale, M. Cain, M. Xu, M. Suzgun, M. Tiwari, M. Bansal, M. Aminnaseri, M. Geva, M. Gheini, T. MukundVarma, N. Peng, N. Chi, N. Lee, N. G. Krakover, N. Cameron, N. S. Roberts, N. Doiron, N. Nangia, N. Deckers, N. Muennighoff, N. S. Keskar, N. Iyer, N. Constant, N. Fiedel, N. Wen, O. Zhang, O. Agha, O. Elbaghdadi, O. Levy, O. Evans, P. Casares, P. Doshi, P. Fung, P. P. Liang, P. Vicol, P. Alipoormolabashi, P. Liao, P. Liang, P. W. Chang, P. Eckersley, P. M. Htut, P. Hwang, P. Milkowski, P. S. Patil, P. Pezeshkpour, P. Oli, Q. Mei, Q. Lyu, Q. Chen, R. Banjade, R. E. Rudolph, R. Gabriel, R. Habacker, R. R. Delgado, R. Millière, R. Garg, R. Barnes, R. A. Saurous, R. Arakawa, R. Raymaekers, R. Frank, R. Sikand, R. Novak, R. Sitelew, R. Lebras, R. Liu, R. Jacobs, R. Zhang, R. Salakhutdinov, R. Chi, R. Lee, R. Stovall, R. Teehan, R. Yang, S. J. Singh, S. M. Mohammad, S. Anand, S. Dillavou, S. Shleifer, S. Wiseman, S. Gruetter, S. Bowman, S. S. Schoenholz, S. Han, S. Kwatra, S. A. Rous, S. Ghazarian, S. Ghosh, S. Casey, S. Bischoff, S. Gehrmann, S. Schuster, S. Sadeghi, S. S. Hamdan, S. Zhou, S. Srivastava, S. Shi, S. Singh, S. Asaadi, S. S. Gu, S. Pachchigar, S. Toshniwal, S. Upadhyay, S. Debnath, S. Shakeri, S. Thormeyer, S. Melzi, S. K. Reddy, S. P. Makini, S. Lee, S. B. Torene, S. Hatwar, S. Dehaene, S. Divic, S. Ermon, S. R. Biderman, S. C. Lin, S. Prasad, S. T. Piantadosi, S. M. Shieber, S. Misherghi, S. Kiritchenko, S. Mishra, T. Linzen, T. Schuster, T. Li, T. Yu, T. A. Ali, T. Hashimoto, T. Wu, T. Desbordes, T. Rothschild, T. Phan, T. Wang, T. Nkinyili, T. Schick, T. N. Kornev, T. Telleen-Lawton, T. Tunduny, T. Gerstenberg, T. Chang, T. Neeraj, T. Khot, T. O. Shultz, U. Shaham, V. Misra, V. Demberg, V. Nyamai, V. Raunak, V. V. Ramasesh, V. U. Prabhu, V. Padmakumar, V. Srikumar, W. Fedus, W. Saunders, W. Zhang, W. Vossen, X. Ren, X. F. Tong, X. Wu, X. Shen, Y. Yaghoobzadeh, Y. Lakretz, Y. Song, Y. Bahri, Y. J. Choi, Y. Yang, Y. Hao, Y. Chen, Y. Belinkov, Y. Hou, Y. Hou, Y. Bai, Z. Seid, Z. Xinran, Z. Zhao, Z. F. Wang, Z. J. Wang, Z. Wang, and Z. Wu
  Beyond the imitation game: quantifying and extrapolating the capabilities of language models.
  In arXiv preprint 2206.04615,
  External Links: [Link](https://arxiv.org/abs/2206.04615)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px1.p1.1).
- Subramonian et al. (2023)
  A. Subramonian, X. Yuan, H. Daumé III, and S. L. Blodgett
  It takes two to tango: navigating conceptualizations of NLP tasks and measurements of performance.
  arXiv preprint arXiv:2305.09022.
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Touvron et al. (2023a)
  H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample
  LLaMA: open and efficient foundation language models.
  External Links: 2302.13971
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p3.1),
  [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p1.1).
- Touvron et al. (2023b)
  H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom
  Llama 2: open foundation and fine-tuned chat models.
  External Links: 2307.09288
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px2.p1.1),
  [Table 1](https://arxiv.org/html/2405.14782v3#S3.T1).
- Wang et al. (2019a)
  A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman
  Superglue: a stickier benchmark for general-purpose language understanding systems.
  Advances in neural information processing systems 32.
  Cited by: [item 1](https://arxiv.org/html/2405.14782v3#S2.I1.i1.p1.1).
- Wang et al. (2019b)
  A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman
  GLUE: a multi-task benchmark and analysis platform for natural language understanding.
  In In the Proceedings of ICLR.,
  Cited by: [item 1](https://arxiv.org/html/2405.14782v3#S2.I1.i1.p1.1).
- Wang et al. (2023)
  P. Wang, L. Li, L. Chen, D. Zhu, B. Lin, Y. Cao, Q. Liu, T. Liu, and Z. Sui
  Large language models are not fair evaluators.
  ArXiv abs/2305.17926.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:258960339)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Wang et al. (2024)
  Y. Wang, Z. Yu, Z. Zeng, L. Yang, W. Yao, C. Wang, H. Chen, C. Jiang, R. Xie, J. Wang, X. Xie, W. Ye, S. Zhang, and Y. Zhang
  PandaLM: an automatic evaluation benchmark for LLM instruction tuning optimization.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=5Nn2BLV7SB)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Weber et al. (2023)
  L. Weber, E. Bruni, and D. Hupkes
  The ICL consistency test.
  External Links: 2312.04945
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Webson and Pavlick (2022)
  A. Webson and E. Pavlick
  Do prompt-based models really understand the meaning of their prompts?.
  In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, M. Carpuat, M. de Marneffe, and I. V. Meza Ruiz (Eds.),
  Seattle, United States, pp. 2300–2344.
  External Links: [Link](https://aclanthology.org/2022.naacl-main.167),
  [Document](https://dx.doi.org/10.18653/v1/2022.naacl-main.167)
  Cited by: [§3.1](https://arxiv.org/html/2405.14782v3#S3.SS1.SSS0.Px1.p1.1).
- Wei and Jia (2021)
  J. T. Wei and R. Jia
  The statistical advantage of automatic NLG metrics at the system level.
  External Links: 2105.12437
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Wu and Aji (2023)
  M. Wu and A. F. Aji
  Style over substance: evaluation biases for large language models.
  arXiv preprint arXiv:2307.03025.
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px3.p1.1).
- Xu et al. (2023)
  F. Xu, Y. Song, M. Iyyer, and E. Choi
  A critical evaluation of evaluations for long-form question answering.
  In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki (Eds.),
  Toronto, Canada, pp. 3225–3245.
  External Links: [Link](https://aclanthology.org/2023.acl-long.181),
  [Document](https://dx.doi.org/10.18653/v1/2023.acl-long.181)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px3.p1.1).
- Yong et al. (2023)
  Z. X. Yong, H. Schoelkopf, N. Muennighoff, A. F. Aji, D. I. Adelani, K. ALMUBARAK, M. S. Bari, L. Sutawika, J. Kasai, A. Baruwa, et al.
  BLOOM+1: adding language support to bloom for zero-shot prompting.
  In The 61st Annual Meeting Of The Association For Computational Linguistics,
  Cited by: [§4.3](https://arxiv.org/html/2405.14782v3#S4.SS3.p3.1).
- Zellers et al. (2019)
  R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi
  HellaSwag: can a machine really finish your sentence?.
  In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,
  External Links: 1905.07830
  Cited by: [item 2](https://arxiv.org/html/2405.14782v3#S2.I1.i2.p1.1).
- Zeng et al. (2024)
  Z. Zeng, J. Yu, T. Gao, Y. Meng, T. Goyal, and D. Chen
  Evaluating large language models at evaluating instruction following.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=tr0KidwPLc)
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1).
- Zhao et al. (2018)
  J. Zhao, T. Wang, M. Yatskar, V. Ordonez, and K. Chang
  Gender bias in coreference resolution: evaluation and debiasing methods.
  External Links: 1804.06876
  Cited by: [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px1.p2.1).
- Zheng et al. (2023)
  L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica
  Judging llm-as-a-judge with mt-bench and chatbot arena.
  CoRR abs/2306.05685.
  External Links: [Link](https://doi.org/10.48550/arXiv.2306.05685),
  [Document](https://dx.doi.org/10.48550/ARXIV.2306.05685),
  2306.05685
  Cited by: [§2.1](https://arxiv.org/html/2405.14782v3#S2.SS1.SSS0.Px2.p1.1),
  [§4.1](https://arxiv.org/html/2405.14782v3#S4.SS1.SSS0.Px3.p4.1).

## Checklist

1. For all authors…

   1. (a)

      Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?
      [Yes]
   2. (b)

      Did you describe the limitations of your work?
      [Yes]  See Appendix [F](https://arxiv.org/html/2405.14782v3#A6) for a brief standalone discussion of limitations. We additionally discuss our scope and are honest about our limitations of our work in the main text throughout.
   3. (c)

      Did you discuss any potential negative societal impacts of your work?
      [Yes]  See Appendix [G](https://arxiv.org/html/2405.14782v3#A7) for a discussion of potential societal impacts.
   4. (d)

      Have you read the ethics review guidelines and ensured that your paper conforms to them?
      [Yes]
2. If you are including theoretical results…

   1. (a)

      Did you state the full set of assumptions of all theoretical results?
      [N/A]
   2. (b)

      Did you include complete proofs of all theoretical results?
      [N/A]
3. If you ran experiments (e.g. for benchmarks)…

   1. (a)

      Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)?
      [Yes]
   2. (b)

      Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?
      [Yes]
   3. (c)

      Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)?
      [Yes]  Yes, for all experiments we perform as case studies, we report 95% confidence intervals calculated via bootstrapping. lm-evalmakes reporting such numbers trivial.
   4. (d)

      Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
      [Yes]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets…

   1. (a)

      If your work uses existing assets, did you cite the creators?
      [Yes]  We cite all directly referenced benchmark datasets, and  lm-evalprovides citations to all implemented tasks that we do not use or reference here.
   2. (b)

      Did you mention the license of the assets?
      [N/A]
   3. (c)

      Did you include any new assets either in the supplemental material or as a URL?
      [N/A]
   4. (d)

      Did you discuss whether and how consent was obtained from people whose data you’re using/curating?
      [N/A]
   5. (e)

      Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content?
      [N/A]
5. If you used crowdsourcing or conducted research with human subjects…

   1. (a)

      Did you include the full text of instructions given to participants and screenshots, if applicable?
      [N/A]
   2. (b)

      Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable?
      [N/A]
   3. (c)

      Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation?
      [N/A]

## Appendix A Library Design

Here we provide an overview of the major components and design philosophy of lm-eval. At its core, lm-eval allows for the contribution of two types of implementations: evaluation Tasks and integrations with novel LM implementations.

#### Tasks

lm-eval is built around modular implementations of evaluation tasks, implemented as a Task class using a common API. This allows tasks to be collected in a common library, for new tasks to be extended or implemented easily, and for novel tasks to be easily shared reproducibly among practitioners or other library users. Users can implement tasks either via YAML-based configuration files or via subclassing the provided Task class and providing custom code for specific methods. In Figure [3](https://arxiv.org/html/2405.14782v3#A1.F3), we show an example of the evaluation logic packaged within a Task class.

![Refer to caption](assets/14-reproducible-lm-evaluation/figure-002.png)


Figure 3: The operations performed by a Task object in lm-eval. Tasks are configured by YAML files or as a Python subclass, and encompass 1) a data source (using the Datasets library ([Lhoest et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib40))), 2) tools for defining prompts and format, 3) mapping these prompts to rendered inputs and expected output type from an LM in the form of Requests, and 4) rules for post-processing the LM’s outputs and calculating the final task metrics.

We provide a number of implementations for common tasks, and accept new tasks sourced from the community. We strive to match the paper originally introducing a benchmark dataset in its methodology, including using the same prompts if applicable. For tasks such as those introduced prior to prompted evaluation becoming the standard, we source evaluation methodology from the paper first posing the evaluation dataset as a prompted task. For example, we implement many tasks as adapted for in-context learning by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4).

#### LMs

The next core piece of lm-eval is the LM API. Because effective orchestration is our core goal, we allow arbitrary software libraries or (autoregressive) language model architectures to extend a provided interface for LM objects.

For ease of use, and compartmentalization of the model definition and external library integrations for custom models away from core evaluation logic, we assume that LMs operate upon dispatched Requests which consist of mapping string inputs to some string or probability as output. We thus abstract tokenizers away within the LM class, and treat a neural language model combined with its tokenizer as a single system being evaluated.

LMs implement a simple interface, consisting of several types of Requests in order to be used within the library for all supported tasks.

#### Request Types

We allow for 3 core types of Requests that may be sent to a language model, which consist of distinct types of measurements that can be performed to observe a model’s response or latent capabilities in a prompted format. These are:

- (Conditional) Loglikelihoods (loglikelihood, multiple_choice) - computing the probability of given output string(s), conditioned on some provided input.
- Perplexities (loglikelihood_rolling) - measuring the average loglikelihood or probability of producing the tokens in a given dataset.
- Generation (generate_until) - generating text until a given stopping condition is reached, from a model conditioned on some provided input.

![Refer to caption](assets/14-reproducible-lm-evaluation/figure-003.png)


Figure 4: Overview of the three core Request types supported by our evaluation framework. These include (1) conditional loglikelihoods, (2) perplexities, and (3) generation-based requests.

Provided with these three primitive operations, we are able to implement the major ways in the literature that have been used to evaluate LMs ([Gao et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib10), [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4), inter alia). While these high-level approaches are standard, they all contain a number of subtle implementation decisions which are often not disclosed in papers. We include a full formal description of common implementation details involved in ours and others’ approaches in Appendix [B](https://arxiv.org/html/2405.14782v3#A2).

## Appendix B Formalizing Measurements

Here we provide a formal description of the most common approaches to obtaining outputs or measurements from LMs for evaluation, as we implement in lm-eval. We include this for a number of reasons: as a reference for future work, notes on the history of certain LM eval practices, and as an illustrative example of just how many implementation details or methodological choices do not typically make it into evaluation papers and yet can vitally impact results or findings.

### B.1 Preliminaries

Throughout, we consider an auto-regressive language model (LM), with vocabulary $V$. Given an input consisting of tokens $x_{0},x_{1},...,x_{n-1}$, the model outputs a probability distribution over the vocabulary, $P(x_{n}|x_{0},x_{1},...,x_{n-1})$. Internally, this is represented as returning “logits” of shape $(1,|V|)$, which when taking a log-softmax over the vocabulary dimension, yields log probabilities (“logprobs” or “loglikelihoods”) of each token in $V$. Logits are the raw, unnormalized predictions of the model before applying the softmax function. Crucially, due to the parallel training and causal masking of autoregressive LMs, it is possible to obtain from a single LM call with $x_{0},x_{1},...,x_{n-1}$ as input, logits of shape $(n,|V|)$ with the $i$-th element of these logits representing $P(x_{i}|x_{0},x_{1},...,x_{i-1})$ for all $1\leq i\leq n$. (That is, for every token position of the input, we obtain concurrently the model’s prediction for the subsequent token, starting from its prediction for the value of $x_{1}$ and ending with the model’s predictions for the (not provided) “$x_{n}$” token.)

### B.2 Ranking-Based Multiple Choice QA

Given our language model, we aim to compute the conditional (log) probability (or “loglikelihood”) of a target string $y$ conditioned on input $x$, denoted as $\log P(y|x)$. This can be performed in a single LM call.

Let $x=x_{0},x_{1},...,x_{n-1}$ be an input sequence of $n$ tokens and $y=y_{0},y_{1},...,y_{m-1}$ be the target sequence of $m$ tokens, where $x_{i}$ and $y_{i}$ represent individual tokens. To compute $\log P(y|x)$, we follow these steps:

1. Concatenate $x$ and $y$ to form a new sequence, but discard the final token $y_{m-1}$. The resulting sequence is $x_{0},x_{1},...,x_{n-1},y_{0},y_{1},...,y_{m-2}$.
2. Pass this concatenated sequence through the language model to obtain logits $l$ of shape $(n+m-1,|V|)$, where $|V|$ is the size of the vocabulary. The last $m$ positions in these logits correspond to the predicted probability distributions for the target tokens $y_{0}$ to $y_{m-1}$, conditioned on the input $x$ and the preceding target tokens.
3. Apply a log-softmax function to the last $m$ logits to obtain log probabilities for the completion tokens only.
4. Calculate the conditional loglikelihood of the target string $y$ given the input $x$ by summing the log probabilities of each target token:

   

$$
\log P(y|x)=\sum_{i=0}^{m-1}\log p(y_{i}|x,y_{0},...,y_{i-1})=\sum_{i=0}^{m-1}l(n+i,y_{i}),
$$
Equation (1)



   where $\log p(y_{i}|x,y_{0},...,y_{i-1})$ is the log probability of the $i$-th target token conditioned on the full input $x$ and the preceding target tokens. (and where $x,y_{0},...y_{-1}$ denotes conditioning on only $x$.)

With this primitive for computing $\log P(y|x)$, several options for evaluation (and decisions regarding hyperparameters) become available.

Multiple Choice QA

Equation [1](https://arxiv.org/html/2405.14782v3#A2.E1) determines how to compute $\log P(y|x)$. We now describe how to perform loglikelihood-based multiple choice as described by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4): given $k$ possible answer strings $a_{1},a_{2},...,a_{k}$, we compute the model’s answer to be $\texttt{argmax}(\log P(a_{1}|x),\log P(a_{2}|x),...,\log P(a_{k}|x))$. In other words, the model selects the answer string with the highest conditional log probability given the input $x$.

This can be performed with worst-case $k$ LM calls using the approach to calculate $\log P(y|x)$ for each $a_{i}=y$ described above. However, the number of LM calls can be reduced if one or more answer strings are only a single token in length. Assume some $a_{i}$ is only encoded by a single token $z$. Then, when calculating the loglikelihood of another answer string $a_{0}$, we obtain the (log-softmaxed) logits of shape $(n+m-1,|V|)$ as an intermediate output. These logits contain the predicted log probabilities for each token in the vocabulary at each position, conditioned on the input $x$ and the preceding tokens. To extract the loglikelihood of predicting the single-token answer $a_{i}$ conditioned on $x$, we can simply select the element in $l$ corresponding to token $z$ at position $n$. This logit represents the log probability of predicting token $z$ immediately after the input sequence $x_{0},x_{1},...,x_{n-1}$.

Thus, we can calculate the loglikelihood of a single-token continuation “for free” and remove an additional LM call for each such single-token $a_{i}$.

Normalization

While the above approach uses the raw loglikelihoods of each given answer choice to select a model answer, other options are available. For instance, if each answer string $a_{i}$ is different in length, this process may frequently default to selecting the shortest $a_{i}$ simply because loglikelihoods are the sum over individual tokens’ log probabilities. Several options for normalizing these loglikelihoods are possible, as also described in [Gao (2021)](https://arxiv.org/html/2405.14782v3#bib.bib39):

- Token-length normalization: each $a_{i}$’s loglikelihood is divided by $m_{i}$, its length in tokens, to gain the per-token loglikelihood of each answer. This approach requires no additional LM calls, and is used alternately with raw loglikelihoods for most tasks by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4).
- Byte-length normalization: each $a_{i}$’s loglikelihood is divided by its length in bytes, removing the dependence on the model’s tokenizer but still normalizing by answer string length. lm-eval provides this metric where applicable as acc_norm.
- Mutual Information: each $a_{i}$’s loglikelihood is defined as $\log P(a_{i}|x)-\log P(a_{i}|null)$, where $null$ is either the empty string, a BOS token, or a placeholder such as "Answer:". This can be thought of as a notion of the pointwise mutual information ([Shannon, 1948](https://arxiv.org/html/2405.14782v3#bib.bib44); [Askell et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib30)), $\log\left(\frac{P(a_{i}|x)}{P(a_{i})}\right)$, which measures the increase in the likelihood of outputting $a_{i}$ when conditioned on the input $x$, compared to the likelihood of outputting $a_{i}$ unconditionally. Intuitively, this measure of mutual information captures the extent to which introducing $x$ makes $a_{i}$ more likely. Although this approach is nonstandard, it is provided in  lm-eval  under the option acc_mutual_info, and used selectively by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4) and [Askell et al. (2021)](https://arxiv.org/html/2405.14782v3#bib.bib30) for certain tasks.

Computing Exact Match

In addition to computing loglikelihoods and normalized loglikelihoods, we may also want to determine whether a given target string $y$ would be produced by greedily decoding from the input $x$. Let $z$ be the concatenation of $x$ and $y$ as defined in the previous sections, and let $\ell$ be the logits
of shape $(n+m-1,|V|)$ obtained by passing $z$ through the language model. To compute the exact match, we compute
$\sum_{i=0}^{m-1}\mathbbm{1}[y_{i}=\argmax(\ell(n+i,\cdots))]$, where $\mathbbm{1}[\cdots]$ is the indicator function that returns 1 if the condition is true and 0 otherwise, and $\ell(n+i,\cdots)$ represents the logits vector corresponding to the model’s output logits predicting the $n+1+i$-th token and therefore the $i$-th token position in $y$ (0-indexed). Intuitively, this sum checks whether each token $y_{i}$ in the target string $y$ matches the most probable (argmax) token predicted by the model at each step of greedy decoding. If the sum equals $m$ (the length of $y$), it means that all tokens in $y$ would be produced by greedily generating $m$ tokens starting from $x$. In this case, we return True to indicate an exact match. Otherwise, if the sum is less than $m$, we return False, indicating that $y$ would not be produced by greedy decoding. Computing the exact match can be useful in scenarios where we want to assess whether the model can generate a specific target string verbatim.

Tokenization

In the above derivations, we assume that one can safely tokenize $x$ and $y$ separately and concatenate their tokenizations. This assumption is not always valid, and most widely-used language model tokenizers provide no such guarantees.

While this factor does not impact the validity of the above calculations, we note that this implies one should be very careful with how their prompt will be tokenized, especially in cases where the tokenization of an input and output pair separately may not align with the tokenization the LM most often saw during training. There are some recently proposed mitigations to remedy this issue, such as ‘‘token healing’’77
7
<https://github.com/guidance-ai/guidance/blob/main/notebooks/tutorials/token_healing.ipynb>,

In the case of lm-eval, we achieve a majority of benefits via shifting trailing prompt whitespace into each target string $y$ 88
8
<https://github.com/meta-llama/llama/issues/217#issuecomment-1774147331>, and do not include trailing input whitespace for all tasks we implement, so this operation should be null for the vast majority of cases.

We hope that future work can examine the most practical way to remove such tokenization-based concerns and difficulties from evaluation, such as BPE dropout ([Provilkov et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib45)), other regularization techniques, or other novel tokenization innovations.

### B.3 Perplexity evaluation

A common approach to measure language modeling performance on some data distribution $D$ is to measure perplexity, which is defined as the exponential of the average negative loglikelihood per token ([Jelinek et al., 2005](https://arxiv.org/html/2405.14782v3#bib.bib28); [Brown et al., 1992](https://arxiv.org/html/2405.14782v3#bib.bib29)), that is:



$$
PPL=\exp\left(\frac{-1}{\sum_{j=1}^{|D|}N_{j}}\sum_{j=1}^{|D|}\sum_{i=1}^{N_{j}}\log P(y_{j_{i}}|y_{j_{1}},\dots,y_{j_{i-1}})\right),
$$
Equation (2)



where $|D|$ is the number of documents in the dataset, $y_{j}$ is the $j$-th document in $D$, $N_{j}$ is the total number of tokens in $y_{j}$, and $y_{j_{i}}$ represents the $i$-th token of $y_{j}$.

To calculate perplexity on a selected dataset $D$, each dataset document $y$ is tokenized and fed into a language model following the procedure to calculate $\log P(y)$ described in Appendix [B.2](https://arxiv.org/html/2405.14782v3#A2.SS2), via computing $\log P(y|x)$, where $x$ is set to either the empty string or a beginning-of-text token. Thus, given $\log P(y)$, for each document $y\in D$ we can sum up the per-document loglikelihoods and divide by the number of total dataset tokens. However, comparing perplexity across models that use different tokenizers can be challenging, as the number of tokens per document and the average next-token prediction difficulty will vary.

Tokenization

To avoid introducing a dependence on tokenizers while reporting perplexity scores, several options are available:

- Bits per Byte: This metric measures the average number of bits required to encode each byte of the input text, providing a tokenization-agnostic measure of language modeling performance ([Gao et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib10)). Formally:

  

$$
BPB=\frac{-1}{log(2)}\left(\frac{-1}{\sum_{j=1}^{|D|}B_{j}}\sum_{j=1}^{|D|}\sum_{i=1}^{N_{j}}\log P(y_{j_{i}}|y_{j_{1}},\dots,y_{j_{i-1}})\right),
$$
Equation (3)



  where $\log$ is in base $e$ and $B_{j}$ is the length in bytes of document $y_{j}$. Alternately, bits per byte can be written as

  

$$
BPB=\frac{\sum_{j=1}^{|D|}N_{j}}{\sum_{j=1}^{|D|}B_{j}}\log_{2}(PPL)=\frac{\sum_{j=1}^{|D|}N_{j}}{\sum_{j=1}^{|D|}B_{j}}\frac{\log(PPL)}{\log(2)}.
$$
Equation (4)



  That is, taking the base-2 log of perplexity and renormalizing by the number of bytes rather than tokens.
- Word-Level Perplexity: By tokenizing the input text into words, such as via splitting on whitespace, we can calculate perplexity based on the average loglikelihood per word rather than per-token, making the metric comparable across models with different subword tokenizers.
- Byte-level Perplexity: Similarly, calculating perplexity averaged over the number of bytes instead allows for a different tokenization-independent perplexity calculation, as the number of bytes in each document’s string remains constant regardless of the tokenizer used.

Both byte- and word-level perplexities can be calculated via replacing $N_{j}$ in Equation [2](https://arxiv.org/html/2405.14782v3#A2.E2) instead with the number of bytes or “words” in document $j$.

In  lm-eval we implement and report all 3 of the above metrics and report them as tokenization-agnostic measures of perplexity. This approach aligns with the work of [Gao et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib10), who popularized the use of bits per byte for measuring perplexity, and has been adopted in subsequent studies such as [Magnusson et al. (2023)](https://arxiv.org/html/2405.14782v3#bib.bib37) and [Hoffmann et al. (2022)](https://arxiv.org/html/2405.14782v3#bib.bib36).

Sliding Window

Another challenge is the approach taken to measure perplexity on documents longer than the context length of a given LM. A natural approach, as used by [Gao et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib10), is to chunk documents longer than a model’s training context size $L$ into non-overlapping chunks. For example, a document of length 4500 tokens evaluated using a model with context length 2048 would be processed as follows: tokens 0:2047 (with token 0 being a prepended BOS token) are fed to predict tokens 1:2048, then tokens 2048:4095 are fed to predict tokens 2049:4096, and finally tokens 4096:4499 are fed to predict tokens 4097:4500. The loglikelihoods of each chunk are then summed to obtain the entire document’s loglikelihood.

However, [Press et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib32) observe a phenomenon they call the “Early Token Curse”, referring to the fact that tokens with a greater amount of context preceding them are fundamentally easier to predict, whereas the first several tokens a model must predict “from scratch” are difficult or impossible to predict without information to condition on. To mitigate this issue, they propose a strided or sliding window perplexity evaluation method.

Instead of creating non-overlapping windows of tokens of size $L$, the strided approach introduces a stride $s$ such that overlapping windows of size $L$, shifting at each time by $s$ positions, are used to score $s$ new tokens’ loglikelihoods. This is equivalent to [Gao et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib10)’s approach when $s=L$. This approach reduces the skew of perplexity favoring models with larger $L$ (and thus fewer tokens appearing at the beginning of a context window) that is introduced by the early token curse via reducing the number of tokens appearing with little context preceding them. However, it is worth noting that the prevalence of such affected tokens decreases with larger context window sizes $L$ for a model.

A downside of the above method is that a naive implementation requires in the worst case, $\frac{L}{s}$ times the calls to an LM and $\frac{L}{s}$ the compute compared to the non-overlapping window approach. (However, some architectures can leverage KV cache reuse to avoid the cost of repeatedly re-encoding tokens). lm-eval follows ([Gao et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib10)) in using non-overlapping windows of size $L$. We believe this choice balances the computational cost and the mitigation of the early token curse, while still providing a standardized and comparable measure of language modeling performance.

### B.4 Generative Evaluation

While loglikelihood-based tasks, such as multiple-choice question answering, provide a valuable measure of a language model’s understanding and ability to rank given options, they do not directly assess the model’s capacity to generate coherent and relevant text. Generative tasks, on the other hand, require the model to produce original text based on the given context.

Generative tasks have gained significant importance in recent times, particularly due to the fact that many popular language model APIs either do not provide99
9
<https://docs.anthropic.com/claude/reference/complete_post> or greatly limit access1010
10
<https://platform.openai.com/docs/api-reference/chat> to log probabilities or other intrinsic measures of the model’s confidence in its outputs. This shift has made it especially necessary to rely on the generated text itself to evaluate the model’s performance and capabilities.

Sampling Hyperparameters

Generative tasks often involve various techniques for controlling the diversity and quality of the generated text, such as sampling with temperature, top-k or top-p (nucleus) sampling ([Holtzman et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib34)), and beam search ([Li et al., 2016](https://arxiv.org/html/2405.14782v3#bib.bib35)). The choice of these hyperparameters can significantly impact the model’s output and, consequently, its performance on the task. It is essential to consider and report these hyperparameters when evaluating generative models, as they can greatly influence the generated text’s characteristics and the model’s overall performance.

Scoring and Answer Extraction

Due to the challenges in measuring language model output (Section [2](https://arxiv.org/html/2405.14782v3#S2)), particularly in verifying the semantics of natural language, and because free-form generation sacrifices the benefit of the artificially restricted input space of multiple-choice tasks, the challenge of scoring answers for quality or correctness must be tackled differently.

The open-ended nature of generative tasks means that there may be multiple valid and appropriate responses to a given prompt. A common evaluation strategy is to use few-shot prompts, where the model is provided with a number of examples demonstrating the desired input-output format. The model is then prompted with a new input, and its generated response is extracted using regular expressions (regex) or other heuristic approaches to obtain the normalized answer strings which can be evaluated using exact-match or other metrics. This approach allows for a more structured evaluation of the model’s ability to generate accurate and relevant responses based on the given examples.

However, this is often a highly imperfect solution, as different models may generate responses in varying formats, making it challenging to create a universal regex pattern that works for all models. Moreover, the effectiveness of the regex-based extraction is highly dependent on the specific format used in the original task creation, which could introduce bias towards models that generate responses in a similar format. To address these limitations, lm-eval provides a highly customized answer extraction mechanism through a Filter component tied to Task implementations, allowing the model output to be put through an arbitrary number of filters and post-processing steps.

These custom heuristic approaches make the release of evaluation code, and our recommendations in general, even more crucial. Without knowledge of the extent to what extraction code is used, how it may be tailored to a model, or without access to model outputs, it is difficult to separate models’ compliance with the evaluation format from their answer correctness. Therefore, it is essential to provide detailed information about the answer extraction process and make the code and model outputs available to ensure transparency and reproducibility in generative model evaluation.

### B.5 Comparing Generative and Loglikelihood-based Evaluation

A notable advantage of generative evaluation is that it might serve as a better proxy for assessing a language model’s performance in real-world applications. In most practical use cases, such as the increasingly popular conversational chatbot format, language models are expected to generate coherent and contextually relevant text based on a given prompt or context. By focusing on the quality and appropriateness of the generated text, generative evaluation provides a more direct assessment of the model’s performance in these real-world use cases. This is in contrast to loglikelihood-based tasks, which, while informative, may not fully capture the model’s ability to generate text that is both fluent and contextually appropriate.

On the other hand, loglikelihood-based evaluations have their own advantages, particularly when it comes to evaluating smaller or weaker models, or “base” models not trained to follow instructions ([Sanh et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib5)). These evaluations can provide a useful ranking or measurement of a model’s performance, even if the model is not capable of generating high-quality text on its own. By assessing how likely the model is to assign a high probability to the correct answer, loglikelihood-based evaluations can offer insights into the model’s understanding of the task. Moreover, techniques like Brier Score can be used to obtain smoother measurements of a model’s performance ([Schaeffer et al., 2023](https://arxiv.org/html/2405.14782v3#bib.bib38)), providing a more nuanced assessment of its capabilities. This can be particularly valuable when comparing and ranking models of different sizes and capacities.

## Appendix C Case Studies

### C.1 Comparisons Across Evaluation Settings

Here we provide extra materials and information for the experiment performed in Table [1](https://arxiv.org/html/2405.14782v3#S3.T1).

We test on two popular language modeling benchmarks: the ARC question answering benchmark ([Clark et al., 2018](https://arxiv.org/html/2405.14782v3#bib.bib48)) and MMLU ([Hendrycks et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib49)). For several models, we compare how performance on ARC (Challenge subset) and on MMLU varies when we select two distinct settings for each benchmark, varying the prompt (and choice of strings used for loglikelihood-based scoring). Both settings correspond to evaluation setups actually used in the literature at times for these tasks.

ARC was first adapted to the in-context learning setting by [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4) who implement the dataset as a “cloze” task: the model is prompted with Question: {question}$\backslash$nAnswer: and the likelihood of each potential completion string is compared. By contrast, MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib12)) provides the model with the question text, each of the 4 possible answers preceded by an answer letter A, B, C, or D, and scores the model based on the generation of the letter corresponding to the correct answer. Additionally, [Hendrycks et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib12) aggregate scores via the micro average over all samples instead of the macro average over per-subject scores. However, not all papers evaluate on these tasks in the same way as the original formats.

However, if models do not adopt these approaches, or disclose their exact settings, it is impossible to reliably compare stated model performance. In Table [1](https://arxiv.org/html/2405.14782v3#S3.T1), we compare evaluation on the Challenge subset of ARC using the prompt from [Brown et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib4) (“Cloze”) and using an MMLU-style answer letter with explicit multiple choice options (“MMLU-style”). We additionally compare MMLU scores between the original MMLU prompting style (“MMLU-style”) and an approach we term “Hybrid”, consisting of an MMLU-style prompt but using the answer strings instead of answer letters as the set of continuations over which we can score (this is used by [Black et al. (2022)](https://arxiv.org/html/2405.14782v3#bib.bib8); [Penedo et al. (2023)](https://arxiv.org/html/2405.14782v3#bib.bib72) upon their releases, and in versions of  lm-evalprior to v0.4.0). As described further below, we can perform this experiment by modifying two lines in lm-eval’s ARC and MMLU config files.

We additionally provide the configuration files we use for this experiment, to illustrate an example of the ease of evaluation experimentation, and the configurability and reproducibility afforded. These files’ contents can be found below.

All experiments are performed using version 0.4.2 of the LM Evaluation Harness using the lm_eval PyPI package, which can also be found and downloaded at the following link: <https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2/lm_eval>.

Below are configuration files that may be used to replicate our results on ARC-Challenge and MMLU for all settings, as well as for ARC-Easy (scores not shown).



````Python
group:
  - ai2_arc
task: arc_easy
dataset_path: allenai/ai2_arc
dataset_name: ARC-Easy
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.text}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
````



A YAML configuration file for the ARC-easy task, implemented in the “cloze” style as done by ([Brown et al., 2020](https://arxiv.org/html/2405.14782v3#bib.bib4)).



````Python
group:
  - ai2_arc
task: arc_easy_mmlu
dataset_path: allenai/ai2_arc
dataset_name: ARC-Easy
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "{{question.strip()}}\n{% for choice in choices.text %}{{choices.label[loop.index - 1]}}. {{choice}}\n{% endfor %}Answer:"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.label}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
````



A YAML configuration file for the ARC-easy task, as implemented following the prompting style of MMLU in [Hendrycks et al. (2020)](https://arxiv.org/html/2405.14782v3#bib.bib12).

We can observe that these configuration files define several components:

- The source dataset from the Datasets([Lhoest et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib40)) library (local datasets are also supported), and the splits to use for testing and few-shot examples. Few-shot examples are drawn from a special fewshot split if specified, else drawn from the training set, validation set, or (in the worst case) non-overlapping test set examples with the current test set example being evaluated, in decreasing order of prioritization.
- the doc_to_\* attributes define mappings to input prompt, gold target label, and the list of answer choice strings, respectively in order.
- We provide a list of metrics to use–here, acc denotes unnormalized loglikelihood to score answers, and acc_norm using byte-length normalization of loglikelihoods.
- Finally, the metadata.version field stores the task’s version attribute to report.

For prototyping, and for the quick modification of interrelated task variants during experimentation, configurations can also inherit from one another: the following is a config file for ARC-challenge, in the cloze style:



````Python
include: arc_easy.yaml
task: arc_challenge
dataset_name: ARC-Challenge
````



a config file for a single MMLU subset in its original style is the following:



````Python
dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 0.0
````



And the ”Hybrid” variant:



````Python
dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: choices
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 0.0
````



## Appendix D Best Practices Checklist for Language Model Evaluation

While LM evaluation is difficult, there are measures that can be taken to significantly improve current practices. We provide our high-level recommendations regarding such measures below.

#### Always share your exact prompts.

If possible, full evaluation code including the full prompts used should be provided for reproducible evaluation runs. Failing this, sharing prompts alone can drastically improve reproducibility.
For fair comparison against other models, evaluation should be done with the same set of prompts unless there is a good reason not to. Prompts should not be optimized for performance on a given model, and the amount of prompt engineering done should be disclosed.

#### Avoid copying results from other implementations.

Comparing results across papers can be misleading due to a wide range of experimental differences, including prompts, sample size, metric calculation, and more ([Marie et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib15)).
Results should not be copied or reported from other papers ([Marie, 2022](https://arxiv.org/html/2405.14782v3#bib.bib17)) whenever possible, unless one can verify that the same code has been used to run the experiments in those papers. If such copying is unavoidable, it should be clearly marked as such and treated carefully.

#### Always provide model outputs.

Providing model outputs alongside evaluation code can allow others to recalculate scores based on these artifacts, which can be useful for performing statistical significance testing and for assessing the impact of different evaluation metrics or scoring approaches.
Evaluation of large models or APIs can be quite costly—sharing such artifacts allows researchers without access to significant compute to participate in evaluation research.
Finally, sharing outputs can allow results on API models to be reproduced to some extent, even if the models are subsequently deprecated.

#### Perform qualitative analyses.

Qualitatively review a small batch of results and outputs before testing at scale: it is very easy to have bugs in your generation code, especially when working with multiple sets of benchmarks, prompts, and models of different architectures. Catching issues early can save a lot of time and compute, and increase confidence in results.
Quantitative scores only provide so much information. To understand why a model is scoring so well or so poorly, it is important to do some sort of qualitative error analysis. This can sometimes reveal superficial errors that are easier to correct with post-processing ([Bawden and Yvon, 2023](https://arxiv.org/html/2405.14782v3#bib.bib16)), or more fundamental errors.

#### Perform statistical significance testing.

Most works on language modeling do not perform statistical significance testing ([Marie et al., 2021](https://arxiv.org/html/2405.14782v3#bib.bib15)). This simple addition can dramatically boost the reliability of claimed results.
Although costly, reporting results run over more than one random seed can dramatically boost the validity and utility of results, for example by averaging across model runs ([Sellam et al., 2022](https://arxiv.org/html/2405.14782v3#bib.bib1)) or averaging over multiple selections of few-shot examples.

## Appendix E Hardware Details

All experiments are run on 8x NVIDIA A40 48Gb GPUs. Total usage time did not exceed 1 day.

## Appendix F Limitations

Our work restricts its scope to worries that, and does not discuss issues such as measurement validity in great detail, due to length reasons. There are additionally more case studies and concrete instances of our discussed

## Appendix G Impacts

We do not see strong potential negative societal impacts of our work. We believe that healthier and more reliable benchmarking practices and awareness around benchmarking have the potential to improve the societal impacts of LLMs, and hope our recommendations can guide informed and careful assessment of LLMs’ capabilities, reducing the odds of their deployment in unsafe and unsuitable scenarios.

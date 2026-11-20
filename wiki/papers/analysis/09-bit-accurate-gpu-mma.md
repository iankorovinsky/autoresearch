---
paper_id: "09"
title: "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy"
source_version: "2511.10909v3"
source_markdown: "../pdfs/09-bit-accurate-gpu-mma.md"
source_pdf: "../pdfs/09-bit-accurate-gpu-mma.pdf"
source_url: "https://arxiv.org/abs/2511.10909"
analysis_date: "2026-11-20"
analysis_status: "reviewed"
topics: ["gpu-kernels", "floating-point", "numerical-correctness", "tensor-cores", "matrix-cores", "hardware-semantics", "differential-testing", "mma-sim", "reproducibility"]
summary: "CLFP derives and differentially tests instruction-level arithmetic models for ten GPU architectures, explaining mixed-precision discrepancies and identifying precision bottlenecks, rounding bias and targeted workarounds."
---

# Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units

## 1. Paper identity and scope

- The analyzed manuscript is arXiv `2511.10909v3`, by Peichen Xie, Shuotao Xu, Yang Wang, Fan Yang and Mao Yang. It studies arithmetic behavior of GPU matrix multiply-accumulate units, not LLM kernel-generation quality. [Title; Abstract](../pdfs/09-bit-accurate-gpu-mma.md)
- The core question is why nominally identical MMA operations produce different bit patterns across GPU architectures, and which hidden arithmetic choices create numerical error. [§I; §II](../pdfs/09-bit-accurate-gpu-mma.md)
- Contributions are closed-loop feature probing (CLFP), eight executable arithmetic-model types, the MMA-Sim implementation, discrepancy explanations, error-source analysis and mitigation proposals. [§I; Table II](../pdfs/09-bit-accurate-gpu-mma.md)
- The hardware scope is seven NVIDIA architectures from Volta through RTX Blackwell and three AMD architectures from CDNA1 through CDNA3. “All instructions” claims refer to the floating-point MMA instructions examined within this enumerated scope, not every past or future accelerator. [§III-B; Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- “Bit-accurate” means agreement of simulator output with hardware output at the bit level in the paper's differential tests. This is an empirically inferred executable model; the study does not present a formal proof against a hardware RTL specification for all possible inputs. [§III-A4–5; §VII](../pdfs/09-bit-accurate-gpu-mma.md)

## 2. Concepts and terminology

### Arithmetic semantics and the modeling target

- MMA denotes `D = A × B + C`, with shapes `A: M×K`, `B: K×N`, and `C,D: M×N`. Mathematical equivalence of this expression does not specify implementation precision, rounding or grouping. [§II-B, Eq. 1](../pdfs/09-bit-accurate-gpu-mma.md)
- The model target is `Φ(A,B,C) = MMA-Interface(A,B,C)` for the interface's shapes and dtypes; hardware fidelity and closeness to exact arithmetic are different objectives. An accurate simulator can reproduce an inaccurate hardware result. [§III, Eq. 2; §VI](../pdfs/09-bit-accurate-gpu-mma.md)
- If output positions share arithmetic behavior, a matrix instruction reduces to identical per-output dot-product-accumulate computations `d = c + Σ a_k b_k`. Otherwise the method must model output elements separately. [§III-A1, Eqs. 3–4](../pdfs/09-bit-accurate-gpu-mma.md)
- The **summation tree** describes order and arity of combinations of the `K` products and accumulator. It is more expressive than the blanket explanation “floating-point arithmetic is non-associative.” [§III-A2; §V](../pdfs/09-bit-accurate-gpu-mma.md)
- **Fusion granularity** specifies how many terms are combined before normalization or output rounding. A fused operation can retain a different intermediate format from its final output dtype. [§IV opening; §IV-A3](../pdfs/09-bit-accurate-gpu-mma.md)
- **FTZ** means flush-to-zero: subnormal operands or results may be replaced by zero. Whether signs are preserved and which stage flushes matter to bit-exact behavior. [§IV-A1, Algorithms 1–2](../pdfs/09-bit-accurate-gpu-mma.md)
- **RZ, RU, RD, RNE** mean round toward zero, toward positive infinity, toward negative infinity, and nearest with ties to even. The probe suite also distinguishes other nearest-tie conventions, away-from-zero rounding and ties-to-odd. [§III-A3, FPT 2](../pdfs/09-bit-accurate-gpu-mma.md)
- **Fractional precision `F`** in an aligned fixed-point summation is not automatically the mantissa precision suggested by the output dtype. An FP32 output container can carry a value rounded internally to E8M13. [§IV-A3; §IV-C1; §VI-B2](../pdfs/09-bit-accurate-gpu-mma.md)
- **Product renormalization** asks whether a product such as `1.5 × 1.5 × 2^e` keeps exponent `e` with significand 2.25, or is normalized to significand 1.125 and exponent `e+1` before alignment. This changes truncation positions. [§III-A3, FPT 4](../pdfs/09-bit-accurate-gpu-mma.md)
- **ULP** is the unit in the last place of the relevant result format. A quoted ULP bound must therefore state whether the result is FP16, FP32 or an internal format. [§VI-A; §VI-B3](../pdfs/09-bit-accurate-gpu-mma.md)
- **Odd symmetry** here is `Φ(−A,B,−C) = −Φ(A,B,C)`. Internally rounding down can violate this relation and introduce directional numerical bias. [§VI-B4](../pdfs/09-bit-accurate-gpu-mma.md)

### The eight arithmetic-model types

- **FTZ-MulAdd:** AMD CDNA2 FP16/BF16 operations flush subnormal MMA inputs to positive zero, perform FP32 rounded multiplication/addition with signed output flushing, combine groups of two or four products pairwise, then accumulate group sums sequentially into `c`. [§IV-A1, Algorithms 1–2](../pdfs/09-bit-accurate-gpu-mma.md)
- **FMA:** each step computes `a×b+c` as if exactly and rounds once to its result format; an instruction is a chain of such steps. This models NVIDIA FP64 and AMD FP32/FP64 MMA instructions examined. [§IV-A2, Algorithms 3–4](../pdfs/09-bit-accurate-gpu-mma.md)
- **E-FDPA, exact fused dot-product-add:** computes `c + Σ a_k b_k` as if with infinite precision inside a group and rounds once to FP32 using RNE. This models CDNA1 BF16/FP16. “Exact” applies before the output rounding and within the specified group. [§IV-A3, Algorithm 6](../pdfs/09-bit-accurate-gpu-mma.md)
- **T-FDPA, truncated FDPA:** computes unnormalized exact products, aligns products and `c` to their largest exponent, truncates each signed significand to `F` fixed-point fractional bits, sums exactly in that aligned format and converts to the output using `ρ`. [§IV-A3, Algorithm 7](../pdfs/09-bit-accurate-gpu-mma.md)
- **ST-FDPA, scaled truncated FDPA:** extends T-FDPA with two E8M0 block scales. Because E8M0 has unit significand, scaling can be incorporated by adding scale exponents to product exponents. [§IV-A3, Algorithm 8](../pdfs/09-bit-accurate-gpu-mma.md)
- **GST-FDPA, group-scaled truncated FDPA:** computes exact dot products within groups, applies corresponding block-scale significands/exponents, then performs a truncated sum of group results plus `c`. Arithmetic group size `G` need not equal the scale block size. [§IV-A3, Algorithm 9](../pdfs/09-bit-accurate-gpu-mma.md)
- For GST-FDPA, scale block size is 32 for MXFP4 and 16 for NVFP4, while the reported implementations use `G=16`. Scale types include E8M0 and UE4M3. [§IV-A3; Table III caption](../pdfs/09-bit-accurate-gpu-mma.md)
- **TR-FDPA, truncated rounded FDPA:** first sums the truncated products without `c`, then aligns the dot-product result and `c` with different fixed-point precisions, applies RD, normalizes through E8M31 with RD, and finally converts to FP32. This models CDNA3 TF32/BF16/FP16. [§IV-A3, Algorithm 10](../pdfs/09-bit-accurate-gpu-mma.md)
- **GTR-FDPA, group-truncated rounded FDPA:** separates even and odd products, forms each truncated group sum, rounds their alignment downward, then combines with `c` using a TR-FDPA-like stage plus an extra small-accumulator truncation rule. This models CDNA3 FP8. [§IV-A3, Algorithm 11](../pdfs/09-bit-accurate-gpu-mma.md)
- All FDPA instruction models can apply one or several elementary FDPA operations in sequence, using group length `L=min(K,L_max)`; “fused” does not imply the complete matrix multiplication rounds only once. [§IV-A3, Algorithm 5](../pdfs/09-bit-accurate-gpu-mma.md)

### Special values

- NaNs, infinities, zero-times-infinity and opposite-sign infinities are included in the modeled semantics, not excluded by testing only ordinary finite inputs. [§IV-B; §III-A4](../pdfs/09-bit-accurate-gpu-mma.md)
- T-FDPA, ST-FDPA and GST-FDPA use canonical NaN encodings `0x7FFFFFFF` for FP32 and `0x7FFF` for FP16 outputs. [§IV-B](../pdfs/09-bit-accurate-gpu-mma.md)
- Zero-valued `c` receives special exponent treatment: the models use `0×2^(−108−F)` for FP32 or `0×2^(3−F)` for FP16, rather than the raw zero-format exponent. This matters when selecting a common alignment exponent. [§IV-B](../pdfs/09-bit-accurate-gpu-mma.md)
- TR-FDPA multiplication can overflow to infinity at product magnitude at least `2^128`; the source states that its other intermediate operations do not overflow/underflow, while final rounding handles those cases conventionally. [§IV-B](../pdfs/09-bit-accurate-gpu-mma.md)

## 3. Methodology and experimental setup

### Closed-loop feature probing

- Step 1 tests determinism by repeated randomized execution and position consistency by copying the same row/column operands across all output positions. All tested interfaces exhibited consistent per-position behavior. [§III-A1](../pdfs/09-bit-accurate-gpu-mma.md)
- Step 2 infers summation trees with a large positive term `U`, a matching negative term `−U` and small terms `v`. Large terms swallow small contributions until cancellation; the surviving multiple of `v` reveals which terms combined before cancellation. [§III-A2, Eqs. 6–7; Fig. 2](../pdfs/09-bit-accurate-gpu-mma.md)
- The tree probe enumerates pairs of positions assigned `U` and `−U`; output ratios `d^(i,j)/v` constrain a tree realization. This adapts FPRev rather than assuming a known binary reduction layout. [§III-A2](../pdfs/09-bit-accurate-gpu-mma.md)
- Existing FPRev reasoning assumes limited-precision fusion swamps small terms. CLFP adds checks for high-precision fusion, where small terms can survive `U`/`−U` cancellation inside a fused operation. [§III-A2, Eqs. 8–9](../pdfs/09-bit-accurate-gpu-mma.md)
- Step 3 runs feature-probing tests, then manually constructs an executable model from the inferred behavior. Tests are reusable for subsequent instructions. [§III-A3; §III-D](../pdfs/09-bit-accurate-gpu-mma.md)
- FPT 1 repeatedly halves an added epsilon to locate precision thresholds, using either `U+ε` for a binary operation or `U−U+ε` for fused summation. [§III-A3, FPT 1](../pdfs/09-bit-accurate-gpu-mma.md)
- FPT 2 uses positive and negative half-/one-and-a-half-step perturbations to identify rounding direction, then additional tie cases to identify nearest-rounding tie rules. [§III-A3, FPT 2](../pdfs/09-bit-accurate-gpu-mma.md)
- FPT 3 probes internal summation order within a fused operation using epsilon-scale values, because fusion alone does not specify the sequence of pre-normalization operations. [§III-A3, FPT 3](../pdfs/09-bit-accurate-gpu-mma.md)
- FPT 4 changes the product significand to test normalization behavior; it originated from a failing differential test, illustrating why a fixed initial probe set is insufficient. [§III-A3, FPT 4; §III-A5](../pdfs/09-bit-accurate-gpu-mma.md)
- Step 4 compares simulator and hardware bit-for-bit using ordinary normal/uniform data, rare-outlier mixtures, high-condition-number cancellation cases and random bit streams. Random bit streams were especially productive because they include wide exponents, subnormals, infinities and NaNs. [§III-A4](../pdfs/09-bit-accurate-gpu-mma.md)
- The rare-outlier example distribution is `N(0,1) + Bernoulli(0.001)·N(0,10²)`; the cancellation criterion uses `Σ|p_k|/|Σp_k| ≫ 1`. These are deliberate stress distributions rather than typical-input-only tests. [§III-A4](../pdfs/09-bit-accurate-gpu-mma.md)
- Step 5 reduces a mismatch by zeroing inputs until a small failing subset remains, usually two to four values, then varies exponents, signs and low significand bits to identify a distinguishing boundary case. [§III-A5](../pdfs/09-bit-accurate-gpu-mma.md)
- Each newly identified behavior becomes a reusable probe, the model is revised, and differential testing repeats. This counterexample-driven loop is the methodological contribution beyond one-off feature probing. [§III-A5](../pdfs/09-bit-accurate-gpu-mma.md)

### Hardware and implementation

- NVIDIA devices are V100/Volta `sm70`, T4/Turing `sm75`, A100/Ampere `sm80`, RTX 4090/Ada `sm89`, H100/Hopper `sm90`, B200/Blackwell `sm100` and RTX PRO 6000 Blackwell `sm120`. [§III-B–C](../pdfs/09-bit-accurate-gpu-mma.md)
- AMD devices are MI100/CDNA1 `gfx908`, MI250X/CDNA2 `gfx90a` and MI300X/CDNA3 `gfx942`. [§III-B–C](../pdfs/09-bit-accurate-gpu-mma.md)
- The authors use CUDA plus inline PTX on NVIDIA and HIP plus instruction intrinsics on AMD, checking PTX-to-SASS and intrinsic-to-instruction mappings to avoid attributing compiler composition to hardware. [§III-C](../pdfs/09-bit-accurate-gpu-mma.md)
- CLFP and MMA-Sim are implemented in Python. Determinism/tree probes, running feature tests and differential verification are automated; constructing/revising models and mismatch reduction remain partly manual. [§III-C–D](../pdfs/09-bit-accurate-gpu-mma.md)
- CPU-only MMA-Sim can perform software experiments; hardware equivalence tests require a supported GPU. The artifact notebook uses a Turing GPU such as T4. [Appendix A-C2](../pdfs/09-bit-accurate-gpu-mma.md)
- The artifact specifies a Linux environment, compatible CUDA/ROCm drivers/toolkits, and PyTorch 2.7+ for local hardware experiments; it exposes `mmasim` and `mmasim-kernels` packages. These are recorded artifact requirements, not a fresh installation recommendation. [Appendix A-B–D](../pdfs/09-bit-accurate-gpu-mma.md)

## 4. Results and evidence

### Modeling effort and coverage

- The original single-threaded simulator required roughly 10–20 hours for one million verification tests; GPU acceleration improved it by over 10× to about one hour for that verification stage. [§III-D; Table I](../pdfs/09-bit-accurate-gpu-mma.md)
- Typical total modeling time is 2–12 hours per instruction with zero to two revision loops. Table I separates approximately 10–20 minutes of GPU time, 1–3 hours of simulator time and 1–9 hours of human effort. [Table I](../pdfs/09-bit-accurate-gpu-mma.md)
- Most instructions require zero or one model revision; new instruction families usually need one or two, while the most complex reported case—CDNA3 FP8—requires ten revisions. [§III-D](../pdfs/09-bit-accurate-gpu-mma.md)
- Human debugging becomes the main bottleneck after simulator acceleration. Automating minimum-failing-subset reduction and using agents for model construction are future directions, not completed capabilities of CLFP. [§III-D](../pdfs/09-bit-accurate-gpu-mma.md)

### Instruction-to-model map

- Volta FP16 HMMA uses T-FDPA with `L_max=4`, `F=23`; FP32 outputs use RZ, whereas FP16 outputs use RNE. [Table III; §IV-C1](../pdfs/09-bit-accurate-gpu-mma.md)
- Turing FP16 HMMA increases these values to `L_max=8`, `F=24`. Ampere/Ada use `L_max=4` for TF32 and 8 for BF16/FP16, with `F=24`. [Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- Ada FP8 QMMA uses T-FDPA with `L_max=16`, only `F=13`, and RZ to E8M13 for FP32 outputs or RNE for FP16 outputs. [Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- Hopper HMMA/HGMMA uses `F=25`, with `L_max=8` for TF32 and 16 for BF16/FP16; Hopper FP8 QGMMA instead uses `L_max=32`, `F=13`. [Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- Blackwell and RTX Blackwell retain the Hopper-like TF32/BF16/FP16 settings, but FP8/FP6/FP4 arithmetic uses `L_max=32`, `F=25`, with full FP32 output rounding instead of E8M13 for FP32 outputs. [Table III; §IV-C1](../pdfs/09-bit-accurate-gpu-mma.md)
- Blackwell/RTX Blackwell MXFP8/6/4 variants use ST-FDPA (`L_max=32`, `F=25`), while the specified MXFP4/NVFP4 variants use GST-FDPA (`L_max=64`, `G=16`, `F=35`). Different instructions supporting FP4-related formats need not share one arithmetic model. [Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- CDNA1 uses FMA for FP32, E-FDPA for BF16 (`L_max=2`) and E-FDPA for FP16 (`L_max=4`). [Table III; §IV-C2](../pdfs/09-bit-accurate-gpu-mma.md)
- CDNA2 uses FMA for FP64/FP32 and FTZ-MulAdd for BF16/FP16; BF16 generally has group length 2, while `_1k` BF16 and FP16 have group length 4. [Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- CDNA3 uses FMA for FP64/FP32, TR-FDPA for TF32 (`L_max=4`) and BF16/FP16 (`L_max=8`), and GTR-FDPA for FP8 (`L_max=16`). Its mixed-precision models use `F=24` and final FP32 RNE. [Table III; §IV-C2](../pdfs/09-bit-accurate-gpu-mma.md)
- Across the examined scope, NVIDIA FP64 and AMD FP32/FP64 instructions follow chains of FMA; default rounding is RNE, with NVIDIA FP64 allowing configured RZ/RU/RD variants. [§IV-A2; Table III](../pdfs/09-bit-accurate-gpu-mma.md)

### A controlled discrepancy example

- The constructed example adds `2²³`, `−2²³`, `−0.5`, `−0.25` and `−0.125`. Exact cancellation leaves `−0.875`, but mixed-precision instructions produce six different answers across architectures. [§V, Eq. 10; Table IV](../pdfs/09-bit-accurate-gpu-mma.md)
- The tested FP64/FP32 sequential-FMA instructions produce `−0.875`; this statement concerns the constructed input and matched operations, not all software GEMM implementations on those devices. [§V; Table IV](../pdfs/09-bit-accurate-gpu-mma.md)
- Volta FP16 yields 0; Turing/Ampere/Ada relevant non-FP8 operations yield `−0.5`; Hopper and Blackwell non-FP8 cases yield `−0.75`, reflecting which small summands survive truncation at `F=23,24,25`. [§V; Table IV](../pdfs/09-bit-accurate-gpu-mma.md)
- Ada/Hopper FP8 yields 0 because of `F=13`; Blackwell/RTX Blackwell FP8 yields `−0.75` after precision increases. [§V; Table IV](../pdfs/09-bit-accurate-gpu-mma.md)
- CDNA1 E-FDPA yields the exact `−0.875`; CDNA2 gives 0 or `−0.375` depending on group structure; CDNA3 gives `−0.5` for TF32/BF16/FP16 and `−1.0` for FP8. [§V; Table IV](../pdfs/09-bit-accurate-gpu-mma.md)
- CDNA3 FP8's `−1.0` is explained by its even/odd grouping and downward-rounded combination: a partial `−0.625` is rounded to `−1` before cancellation. The difference is mechanistic, not unexplained measurement noise. [§V](../pdfs/09-bit-accurate-gpu-mma.md)

### Error bounds and numerical risks

- FTZ error is bounded at the corresponding smallest normal magnitude: `2^−14` for FP16, `2^−126` for BF16/FP32. Rounded FP32 add/multiply contributes up to 0.5 ULP under RNE. [§VI-A; Table V](../pdfs/09-bit-accurate-gpu-mma.md)
- FMA and E-FDPA have only output-rounding error within an elementary operation: 0.5 ULP for nearest rounding or up to 1 ULP for directed modes. Sequential composition still accumulates error. [§VI-A](../pdfs/09-bit-accurate-gpu-mma.md)
- For truncated FDPA variants, the source bounds fused-sum error by `(L+1)·2^(e_max−F)` and adds a separate output-rounding term. This is an elementary-operation bound, not a universal end-to-end relative-error guarantee for a whole network. [§VI-A; Table V](../pdfs/09-bit-accurate-gpu-mma.md)
- Four precision bottlenecks are identified: CDNA2 FP16 input FTZ; low `F` in Ada/Hopper FP8; E8M13 rounding of nominal FP32 FP8 outputs; and FP16 output rounding across NVIDIA variants. A fifth concern is CDNA3 rounding asymmetry. [§VI-B; Table VI](../pdfs/09-bit-accurate-gpu-mma.md)
- Ada/Hopper FP8's precision parameter differs from their higher-precision families by 11/12 bits; the paper highlights factors 2,048/4,096 in the precision-dependent truncation term and 1,024 in E8M13-versus-FP32 output precision. These are error-bound comparisons, not measured training-loss multipliers. [§VI-B2](../pdfs/09-bit-accurate-gpu-mma.md)
- FP16 output precision can dominate even when the internal sum has FP32-like precision, so inspecting only the accumulation dtype is insufficient. [§VI-B3](../pdfs/09-bit-accurate-gpu-mma.md)
- A simulated CDNA3 FP16 instruction with `A,B ~ 1000·N(0,1)` and `C ~ N(0,1)` shows negatively biased error relative to FP64; a hypothetical variant replacing internal RD with RZ has a symmetric error distribution. The RZ variant is a model experiment, not an available hardware instruction. [§VI-B4; Fig. 3](../pdfs/09-bit-accurate-gpu-mma.md)

### Workarounds and quantitative examples

- Scaling small FP16 inputs upward and rescaling outputs avoids CDNA2 input FTZ. For the reported `128×128` matrices with `A~N(0,0.001²)` and `B~N(0,1)`, MSE versus FP64 falls from `7.632×10^−9` to `1.591×10^−18`. [§VI-C, Workaround 1](../pdfs/09-bit-accurate-gpu-mma.md)
- The scaling workaround is contrasted with converting FP16 to BF16, which expands dynamic range at the cost of significand precision. Its successful example does not establish a globally safe scale for arbitrary inputs with possible overflow. [§VI-C, Workaround 1](../pdfs/09-bit-accurate-gpu-mma.md)
- Setting `C=0` in the Matrix Core operation and performing the accumulation separately with general FP32 units can mitigate CDNA3 bias. For the stated `128×128×128` case with FP16 inputs sampled from N(0, 10²), mean deviation changes from `−1.352×10^−6` to `+7.467×10^−8`. [§VI-C, Workaround 2](../pdfs/09-bit-accurate-gpu-mma.md)
- Replacing FP16-output MMA with FP32-output MMA reduces the example MSE from 0.247 to `4.481×10^−8`; converting the result back to FP16 yields 0.0551, exposing the residual output-format loss. [§VI-C, Workaround 3](../pdfs/09-bit-accurate-gpu-mma.md)
- Using FP16 MMA instead of native FP8 MMA on Ada/Hopper when accuracy is prioritized reduces the reported FP8-input example MSE from 0.0103 to `5.569×10^−11`. The source explicitly frames this as accuracy versus throughput, not a free performance improvement. [§VI-C, Workaround 3](../pdfs/09-bit-accurate-gpu-mma.md)

## 5. Authors' conclusions

- Hardware arithmetic contributes independently to cross-platform numerical discrepancies; matching software seeds and compiler settings alone is insufficient to guarantee bit-identical mixed-precision results. [§II; §V; §VII](../pdfs/09-bit-accurate-gpu-mma.md)
- Closed-loop probing plus executable models provides a more complete explanation than isolated feature tests or assumptions about generic binary reduction trees. [§II-C; §III; §VII](../pdfs/09-bit-accurate-gpu-mma.md)
- Arithmetic models can make numerical-error sources explicit and support targeted software workarounds and hardware design-space exploration. [§VI; §VII](../pdfs/09-bit-accurate-gpu-mma.md)
- The authors recommend FP32-like precision at every relevant computational stage and avoiding the identified bottlenecks, rather than relying only on a high-precision output label. [§VI-D](../pdfs/09-bit-accurate-gpu-mma.md)
- They favor symmetric internal rounding and suggest RZ as relatively cheap to implement with sign-magnitude or ones'-complement representations. This is a design recommendation, not a claim that RZ universally minimizes error. [§VI-D](../pdfs/09-bit-accurate-gpu-mma.md)
- Open-source release is intended to enable continued validation and transparent numerical investigation as hardware changes. [§VII; Appendix A](../pdfs/09-bit-accurate-gpu-mma.md)

## 6. Limitations and open questions

### Explicit scope and future work

- CLFP is semi-automated: human interpretation and model revision remain necessary. Agents for automatic construction/revision are future work. [§III-D](../pdfs/09-bit-accurate-gpu-mma.md)
- The method assumes determinism and tests cross-position consistency; an interface violating those assumptions needs a different or per-position treatment. [§III-A1](../pdfs/09-bit-accurate-gpu-mma.md)
- CPU, TPU and API-level MMA modeling are named future extensions. The tested ten architectures are not evidence of validation on those platforms. [§III-B](../pdfs/09-bit-accurate-gpu-mma.md)
- API-level results can combine instruction behavior with compiler or library transformations; this study deliberately isolates instruction-level interfaces. [§II-C; §III-B](../pdfs/09-bit-accurate-gpu-mma.md)
- The manuscript says fine-grained implementation details beyond its descriptions are available in the released artifacts, so reimplementation should consult those artifacts rather than infer missing corner cases. [Appendix A-A](../pdfs/09-bit-accurate-gpu-mma.md)

### Analyst caveats

- **Evidence boundary:** randomized differential testing, even with adversarial edge cases and bit-stream sampling, is not exhaustive formal verification of all possible bit patterns. “Complete” and “bit-accurate” should retain the paper's empirical validation context. [Basis: §III-A4–5](../pdfs/09-bit-accurate-gpu-mma.md)
- **Model versus circuitry:** observable agreement identifies useful arithmetic semantics, not a unique physical circuit implementation; multiple internal implementations can agree on observed outputs. [Basis: §III modeling target](../pdfs/09-bit-accurate-gpu-mma.md)
- **End-to-end scope:** the paper explains possible causes of training instability and cites external incidents, but its main experiments are instruction tests and numerical examples, not controlled full-model training trials for every workaround. [Basis: §II-A; §VI-C](../pdfs/09-bit-accurate-gpu-mma.md)
- **Bound comparison:** the fused-error bound includes group length `L+1` and common exponent as well as `F`; the reported precision-bit factors should not be applied blindly across instructions with different grouping or workloads. [Basis: Table III; Table V; §VI-B2](../pdfs/09-bit-accurate-gpu-mma.md)
- **Performance scope:** workaround examples report numerical accuracy, not comprehensive throughput/energy/latency tradeoffs. Any deployment choice needs its own performance measurement. [Basis: §VI-C](../pdfs/09-bit-accurate-gpu-mma.md)
- **Specification scope:** a simulator matching hardware does not decide whether that hardware behavior satisfies a task's accuracy contract. That contract must be stated separately. [Basis: Eqs. 1–2; §VI](../pdfs/09-bit-accurate-gpu-mma.md)

## 7. Relevance to this project — interpretation

- **Proposal:** distinguish functional equivalence under exact arithmetic, hardware-bit equivalence, and tolerance-based numerical agreement in the harness. None should be silently substituted for the others. [Basis: §III; §V–VI](../pdfs/09-bit-accurate-gpu-mma.md)
- **Proposal:** record exact GPU architecture, instruction family, input/output dtype, accumulation grouping and compiler/library configuration alongside correctness measurements. [Basis: §III-C; Table III](../pdfs/09-bit-accurate-gpu-mma.md)
- **Proposal:** add subnormal, cancellation, sign-flip, outlier and random-bit tests where the supported dtype/domain makes them meaningful, with explicit handling for NaNs and infinities. [Basis: §III-A3–4; §IV-B; §VI-B4](../pdfs/09-bit-accurate-gpu-mma.md)
- **Proposal:** use minimal failing subsets to turn rare numerical mismatches into stable regression tests and targeted prompts for kernel repair. [Basis: §III-A5](../pdfs/09-bit-accurate-gpu-mma.md)
- **Proposal:** explore MMA-Sim as a diagnostic oracle for hardware-dependent discrepancies, while retaining an independent task-level accuracy target. This analysis has not integrated or benchmarked that tool. [Basis: §III-C; Appendix A](../pdfs/09-bit-accurate-gpu-mma.md)
- **Proposal:** require accuracy and latency measurements together when an agent changes instruction precision, separates accumulation or scales operands; the paper's numerical examples do not establish the best deployment tradeoff. [Basis: §VI-C](../pdfs/09-bit-accurate-gpu-mma.md)

## 8. Retrieval notes

- Search aliases: MMA-Sim, CLFP, closed-loop feature probing, bit-accurate Tensor Cores, Matrix Cores, FPT, FPRev, FTZ, E-FDPA, T-FDPA, ST-FDPA, GST-FDPA, TR-FDPA, GTR-FDPA, E8M13, CDNA3 rounding bias.
- Questions answered: Why can GPUs disagree on the same MMA? What does FP32 output fail to guarantee? How are hidden rounding rules inferred? Which architectures flush FP16 inputs? How do FP8 instructions differ across Ada/Hopper/Blackwell? What is the evidence boundary of a bit-accurate simulator?
- Related navigation: [LARK](22-lark.md) for formal verification and exact-arithmetic assumptions; [KernelBench-Verified](03-kernelbench-verified.md) for numerical validation protocols; [PTXBench](04-ptxbench.md) for architecture-specific instruction use. These are conceptual navigation links, not controlled performance comparisons.

## 9. Source coverage and verification notes

- Read all substantive sections I–VII, Algorithms 1–11, Tables I–VI, the discrepancy derivation and all artifact-appendix subsections. Bibliography entries were used only to distinguish cited background from this paper's experiments, not independently re-reviewed.
- Checked the explicit model parameters in Table III against the neighboring architecture descriptions, including the distinction between `F`, `L_max`, `G` and final rounding.
- The local Markdown's algorithm blocks contain duplicated rendered-math/TeX strings and Table II has nested-table conversion noise. Model descriptions above rely on the coherent accompanying prose, equations and parameter table; no malformed pseudo-code was executed.
- Figure 3's qualitative interpretation is supported by its caption and §VI-B4; no exact histogram bins or frequencies are inferred from the plot. Workaround values are copied from §VI-C with their scope.
- No hardware experiments, artifact installation or exhaustive theorem checking were conducted for this analysis. It records the authors' reported validation and flags its empirical boundary.

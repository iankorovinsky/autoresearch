# Design artifacts

Project-authored notes and design explorations for the automated GPU kernel research harness. These documents capture requirements, alternatives, and open questions; they do not by themselves describe implemented capabilities.

## Contents

- [Harness axioms](axioms.md): principles for specifying tasks, validating oracles, interpreting proofs, and accepting research results.
- [Task intake schema](intake-schema.md): a proposed structured record of user intent, source evidence, behavioral requirements, evaluation criteria, and constraints, with readiness requirements before formalization, algorithm research, and implementation optimization.
- [Possible user inputs](user-inputs.md): 13 input categories, from natural-language ideas and papers to benchmark tasks, proprietary kernels, and formal specifications. Each row includes an example user message and any required attachments. Covers accompanying constraints, request intent, and a coverage review explaining how additional use cases fit the existing categories.
- [Kernel benchmark landscape](kernel-benchmark-landscape/kernel-benchmark-landscape.md): a 2026-11-21 research snapshot covering 44 directly relevant benchmark entries, 45 related benchmarks and resources, and 6 unverified leads. Includes hardware/language targets, task counts where verified, source links, and data-availability and licensing notes. Also available as a [searchable HTML report](kernel-benchmark-landscape/kernel-benchmark-landscape.html) and [structured JSON](kernel-benchmark-landscape/kernel-benchmark-landscape.json). Sources are live links; no benchmark experiments were run.

- [DataSci GPU investigation](datasci-gpu-investigation.md): SE390 cluster access, live GPU/software inventory, scheduler limits, command evidence, and unresolved questions. Includes successful inspections of `gpu-8-vm`, `gpu-1-vm`, and a MIG allocation on `gpu-6-vm`. Findings are dated observations, not benchmark results.
- [Provider-agnostic compute CLI plan](compute-cli-plan.md): a proposed Waterloo-first CLI and shared GPU worker for queued kernel execution, basic profiling, durable run tracking, and later provider/server integration. Includes task/provider/result contracts, implementation milestones, and acceptance tests; preserved as the original proposal with a link to its implementation follow-up.
- [Compute CLI implementation and live tests](compute-cli-validation.md): implemented Waterloo provider and shared worker, installation/usage links, measured `gpu-8-vm` results, failure/cancellation/recovery evidence, and remaining limitations.
- [Modal compute provider investigation](compute/modal.md): GPU selection, Sandbox execution and artifact retrieval, live T4 benchmark and profiler results, the PyTorch Profiler diagnosis and workaround, and post-test billing observations.
- [Baseten compute provider investigation](compute/baseten.md) records Training Jobs integration, T4 rejection, live A10G profiling results, and the hardware-counter permission limit.
- [Development tooling implementation and validation](development-tooling-validation.md): isolated task runner, environment doctor, offline documentation checker, usage links, local test results, and known documentation/dependency findings.

## Adding artifacts

Give each document a descriptive filename and explain its scope. Distinguish proposed behavior from implemented behavior, identify assumptions and unresolved questions, and link to source material where applicable. Add new documents to this index.

See the [reference library](../README.md) for supporting literature and other reference materials, or return to the [project README](../../README.md) for setup and evaluation workflows.

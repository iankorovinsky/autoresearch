# Task intake schema

This is a proposed information model for the automated GPU kernel research harness, not an implemented API, validator, or mandatory user form. It expands the [input taxonomy](user-inputs.md) into a task record governed by the [harness axioms](axioms.md).

The user supplies intent, requirements, and available evidence through conversation and attachments. The harness extracts what it can, investigates the supplied material, and asks only for unresolved decisions that affect the work. Users do not need to supply a Lean specification, an already validated oracle, or benchmark results to begin.

## What counts as extracted

Collect every available field immediately. Its deadline determines when an unresolved value blocks progress, not when extraction is allowed to happen. Keep evidence and authority separate: a failing implementation is useful evidence without being the definition of correct behavior.

Use the following conceptual TypeScript schema. Strings describing behavior or requirements are structured drafting slots; their presence alone does not establish that the contract is complete or unambiguous. `Ref` values identify entries in `sources`.

```ts
type Ref = string;
type Stage = "formalization" | "algorithm_research" | "optimization";

type Source = {
  id: Ref;
  kind: "message" | "code" | "paper" | "specification" | "tests"
      | "dataset" | "profile" | "binary" | "environment" | "other";
  location: string;             // Message ID, attachment, path, or URL.
  locator: string | null;       // Function, section, page, or other precise location.
  version: string | null;       // Revision/hash when available; preserve supplied version.
  access: "available" | "missing" | "unverified";
};

type Evidence = {
  sources: Ref[];
  basis: "explicit" | "derived" | "proposed";
  rationale: string;
};

type Field<T> =
  | { status: "settled"; value: T; evidence: Evidence;
      resolution: string }     // Why this value is authorized by the request/sources.
  | { status: "tentative"; value: T; evidence: Evidence;
      openQuestion: string }
  | { status: "unknown"; openQuestion: string }
  | { status: "conflicting";
      alternatives: { value: T; evidence: Evidence }[];
      openQuestion: string }
  | { status: "not_applicable"; reason: string };

type ValueSpec = {
  name: string;
  kind: "tensor" | "scalar" | "structure";
  meaning: string;
  dtype: string;
  shape: string;               // Includes symbolic dimensions and their relationships.
  layout: string;              // Strides, contiguity, alignment, or explicitly unrestricted.
};

type Workload = {
  id: string;
  purpose: ("correctness" | "performance" | "quality")[];
  definition: string;          // Shapes, parameters, values/distributions, and weights.
  sources: Ref[];              // Supplied fixtures/generators/datasets, if any.
};

type Metric = {
  name: string;
  unit: string;
  direction: "minimize" | "maximize" | "target";
  acceptanceRule: string;      // Threshold or an explicit best-effort objective.
  aggregation: string;         // Across cases/repeats; include denominator for ratios.
  workloadIds: string[];
};

type TaskIntake = {
  schemaVersion: "draft-1";
  taskId: string;
  contractVersion: number;
  sources: Source[];

  intent: Field<{
    mode: "implement" | "improve" | "discover";
    actions: string[];         // E.g. optimize, repair, port, fuse, generate backward.
    requestedOutcome: string;
    deliverables: string[];
  }>;

  scope: Field<{
    target: string;            // Named operation, graph boundary, or discovery space.
    included: string[];
    excluded: string[];
  }>;

  authority: Field<{
    rules: { subject: string; source: Ref; interpretation: string }[];
    conflictPolicy: string;    // Precedence only where actually authorized.
  }>;

  interface: Field<{
    inputs: ValueSpec[];
    outputs: ValueSpec[];
    parameters: string[];
    mutationAndAliasing: string;
  }>;

  domain: Field<{
    supportedShapes: string[];
    valueConstraints: string[];
    preconditions: string[];
    invalidInputBehavior: string;
  }>;

  semantics: Field<{
    definition: string;        // Equation, algorithm, or permitted input/output relation.
    invariants: string[];
    edgeCases: { condition: string; requiredBehavior: string }[];
    permittedNondeterminism: string;
  }>;

  numerics: Field<{
    arithmeticModel: string;   // Real, integer/modular, floating point, quantized, etc.
    intermediatePrecision: string;
    roundingAndOverflow: string;
    exceptionalValues: string;
    comparisonRule: string;    // Exact or a fully defined error rule; not just "close".
  }>;

  allowedChanges: Field<{
    permitted: string[];
    forbidden: string[];
    requiresUserDecision: string[];
  }>;

  oracleRequirements: Field<{
    kind: "reference" | "property_checker" | "relation" | "combined";
    suppliedSources: Ref[];    // Empty if the harness must construct the oracle.
    constructionNeeded: string;
    requiredChecks: string[];
    knownLimitations: string[];
  }>;

  formalizationRequirements: Field<{
    targetArtifact: string;
    intendedClaims: string[];
    assumptions: string[];
    numericalInterpretation: string;
    requiredConnectionToImplementation: string;
    suppliedLeanProject: Ref[];
  }>;

  workloads: Field<Workload[]>;

  successCriteria: Field<{
    metrics: Metric[];
    hardConstraints: string[]; // E.g. memory cap or maximum application-quality loss.
    tradeoffPolicy: string;   // Which objective wins when candidates trade benefits.
  }>;

  baseline: Field<{
    definition: string;
    sources: Ref[];
    comparisonProtocol: string; // Comparable hardware, inputs, precision, timing scope.
  }>;

  environment: Field<{
    hardware: string;
    software: string;
    implementationLanguages: string[];
    integrationRequirements: string[];
    runnerAndDependencyAccess: string;
  }>;

  resources: Field<{
    computeLimits: string;
    timeAndCostLimits: string;
    stopConditions: string[];
  }>;

  openIssues: {
    fieldPath: string;
    question: string;
    resolver: "user" | "source_inspection" | "experiment";
    blocks: Stage[];
  }[];
};
```

`settled` means supported by the request or authoritative evidence, not merely guessed with high confidence. Direct extraction does not require asking the user to approve the same instruction again. A derived fact can be settled when it follows unambiguously from authoritative material; a proposed requirement needs an authorized decision before it becomes settled. Record the reason in `resolution`.

The field wrapper is shown at category level for readability. A partially known category stays tentative, retaining its known values and identifying unresolved subfields through `openIssues.fieldPath`. Do not discard extracted facts because another part of the category is missing. Unknown values must not be represented as invented defaults or empty strings.

`not_applicable` needs a substantive reason and is permitted only when the task does not require that field. It cannot exempt a candidate from having a behavioral contract, an oracle, or acceptance criteria. For example, a comparative baseline may be unnecessary for a task requesting only a correct implementation; it is mandatory for a speedup claim.

## What we actually need from the user

The user must establish the desired outcome and resolve choices that materially change it: authoritative behavior when sources disagree, acceptable approximation, scope restrictions, success tradeoffs, and resource limits that cannot be obtained from existing context. The harness can propose choices, but must not silently choose different correctness requirements.

The harness should inspect supplied material for function signatures, tensor metadata, equations, test cases, build requirements, and baseline definitions. It should not ask users to transcribe accessible files. A bare link is not extracted evidence until the relevant material is accessible and inspected.

Oracle construction, validation results, Lean statements/proofs, benchmark measurements, and discovered bottlenecks are work products. Intake records the requirements for these products and any supplied versions, not a demand that the user produce them first.

## When the record is sufficient

### Before formalization of a fixed computation

Settle `intent`, `scope`, `authority`, `interface`, `domain`, `semantics`, `numerics`, and `formalizationRequirements`. Settle any `allowedChanges` and integration constraints that affect those definitions. Identify how the oracle will be obtained through `oracleRequirements`; its implementation can still be pending.

There must be no unresolved issue that changes the meaning of the theorem or oracle. Hardware details become mandatory here only when they affect the selected semantics or proof assumptions. Formalization may expose new ambiguities; return those to intake rather than choosing new requirements silently.

### Before algorithm research for a discovery task

Settle the discovery objective, search boundaries, external interface/domain, allowed algorithm changes, quality acceptance criteria, evaluation workloads, baseline where comparative, environment, and resource limits. Have a usable, checked quality evaluation procedure before scoring algorithm candidates.

The candidate's internal algorithm may still be unknown. Create a separate candidate contract once an algorithm is proposed, retaining the parent task's acceptance criteria. Each candidate must pass the fixed-computation/formalization gate and oracle-validation gate before implementation optimization. Application quality evaluation does not replace a candidate's correctness oracle.

Reading papers, inspecting sources, and investigating semantics to establish these requirements may happen earlier. This is distinct from accepting or scoring an algorithm as meeting the task's objective.

### Before implementation optimization

The applicable fixed-computation fields must be settled, together with `allowedChanges`, `workloads`, `successCriteria`, `baseline` where comparative, `environment`, and `resources`. The harness must also have produced the following evidence; these are not user-supplied intake fields:

- An oracle tied to the exact contract version, with a validation record covering the supported domain and numerical rules to the extent established. Record checks, assumptions, and remaining coverage limitations; agreement on samples is not a proof.
- The formal specification and evidence required by the agreed `formalizationRequirements`. Record which artifact was proved and any unproved connection to executable code. If Lean proof is a task requirement, an incomplete required proof blocks this gate.
- Executable correctness and performance evaluation procedures consistent with the declared workloads, comparison rules, and baseline. Baseline execution must be available when comparative measurement is required; no speedup claim is valid without its measurement.
- Accessible required sources, dependencies, and execution resources, with no unresolved issue blocking optimization.

Readiness is derived from these checks, not from a user-set `ready: true` or the mere presence of fields. Nonblocking uncertainties remain visible. Unmet requirements can lead to clarification, further validation, or an explicit inability to proceed; they are not silently waived.

## Contract changes and results

Version changes to accepted semantics, domains, tolerances, workloads, and success criteria. Preserve the old contract and its evidence; revalidate affected oracles/proofs and rerun affected evaluations before making claims under the new version. Resolve material changes in intent with the user. Never quietly weaken the contract in response to a failing candidate.

Keep result records separate from intake: candidate/source revision, contract version, oracle revision, proof/check evidence, execution environment, workload version, baseline, measured metrics, and limitations. A result can satisfy some requirements and fail others; report those outcomes explicitly.

## Example: incomplete natural-language input

For “keep the 16 most important tokens and normalize their scores,” intake can capture the requested operation and the value 16 immediately. It must still identify what “important” means, the selection axis and tensor interface, tie-breaking, behavior for sequences shorter than 16, the normalization formula, and numerical requirements. These block formalization because they change the operation.

The user can answer directly or provide an authoritative function or paper section. The target GPU and optimization budget can be collected at the same time, but ordinarily do not resolve those semantic questions. The harness constructs and validates the oracle after the meaning is settled; it does not ask the user to supply a proof of the initial idea.

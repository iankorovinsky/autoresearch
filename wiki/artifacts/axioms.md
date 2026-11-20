# Harness axioms

These principles describe the intended automated GPU kernel research harness. They are design requirements, not claims about currently implemented guarantees. See the [intake schema](intake-schema.md) for the information needed to apply them.

1. **Every task needs explicit acceptance criteria.** Before evaluating a candidate implementation, establish both what constitutes correct behavior and how success will be measured.

2. **User intent determines the contract.** Identify which supplied sources are authoritative. Resolve consequential ambiguities and conflicts; do not silently turn assumptions, example outputs, or existing bugs into requirements.

3. **Validate the oracle before optimizing against it.** Establish the supported input domain, numerical semantics, and evidence that the oracle implements the intended contract. Investigation needed to establish that contract may happen beforehand.

4. **Correctness is always relative to an explicit contract.** State the inputs, preconditions, outputs, edge cases, and allowed numerical error. Any correctness claim must identify the domain it covers.

5. **A proof establishes only what its statement and assumptions express.** Lean formalization must be checked against user intent. Proving a mathematical specification or reference algorithm does not, by itself, prove the generated GPU implementation correct.

6. **Keep acceptance criteria independent of candidate results.** Do not weaken tolerances, remove failing cases, or redefine the objective merely to make a candidate pass. A deliberate contract change creates a new version and requires renewed validation.

7. **Performance gains count only when the required behavior is preserved.** Evaluate correctness and performance separately, using the declared workloads and baseline. For algorithm discovery, also evaluate application quality separately from implementation correctness.

8. **Claims must match their evidence.** Distinguish proved properties, tested behavior, measured performance, and unresolved assumptions. Failure to establish correctness or improvement is a valid outcome.

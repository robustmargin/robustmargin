<!-- One PR per change. Every section is required; write "None" rather than deleting one. -->

## Summary
<!-- Two or three sentences: what this PR adds, and whether it is breaking. -->

## Context
<!-- The problem or requirement this addresses and where it comes from: an issue, a user need,
     or an earlier finding. What was true before this PR. -->

## Methodology
<!-- The statistical or numerical method, stated so a reader could re-derive it.
     - Model, estimator and assumptions, including what is assumed and not checked.
     - Why this method over the alternatives considered, with references (papers,
       R packages, textbook sections).
     - Design decisions in this PR, each with its reasoning and the alternatives rejected. -->

## Implementation
<!-- What changed in the code.
     - Public API: new or changed symbols, signatures, result fields, schema_version.
     - Internal structure, briefly, only where it explains a method choice.
     - Numbers: state whether any returned number changes, and if so which and why. -->

## Validation
<!-- Evidence the implementation is correct, not just that it runs.
     - Reference parity: which external implementation (R drc, glm, ...), what tolerance, what matched.
     - Simulation or coverage results: design, replicates, seed, result against the nominal target.
     - Property and contract tests: which invariants are now pinned.
     - Defects found and fixed during review, stated as facts about the code.
     - Environments verified (OS, Python, dependency floor versus latest). -->

## Compatibility and reproducibility
<!-- - Breaking changes and the migration for each (reader-before-writer order for schema bumps).
     - Impact on downstream users, verified or unverified.
     - Reference fixtures regenerated? Which, with which R/package versions, and whether any
       existing value moved (a moved value is a finding, not a fixture update).
     - Seeds, pinned environments, anything needed to reproduce the numbers above. -->

## Limitations
<!-- Known limits of the method or implementation that a user of the results should know. -->

## Open items
<!-- Questions for the next phase or before a tag, phrased as questions. -->

## Checklist
- [ ] `ruff check`, `ruff format --check`, `mypy` and `pytest` pass, and the build passes `tools/check_dist.py`
- [ ] Reference values unchanged, or every change explained above
- [ ] Migration notes for any breaking or behavioural change
- [ ] Public API change: classified as safe, breaking or behavioural, with downstream users checked
- [ ] No Claude Code session link in the description or any commit

This content is being reviewed by the code owner.

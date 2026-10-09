---
name: testing
description: Use only after explicit user approval to migrate pipeline tests, verify regression behavior, cover edge cases, or compare a migrated stage with the original.
---

# Testing migrated pipeline stages

Use this skill only after the user explicitly approves Phase 4 test migration for the specific processing stage. Do not create, transfer, or modify tests before that approval.

## Test migration

- Transfer the existing tests for the approved stage from the original repository; never modify the original test suite.
- Update imports and fixtures to match the target package layout while preserving what each test asserts.
- Keep test migration limited to the approved stage. Do not migrate tests for other stages.
- Add focused tests for important uncovered edge cases, especially missing or malformed inputs, boundary thresholds, ambiguous classifications, ordering, and configuration defaults where relevant to the stage.
- Avoid tests that merely mirror implementation details. Assert observable behavior and scientifically meaningful invariants.

## Behavioral equivalence

- Treat the original implementation as the reference. Use representative fixtures or datasets to compare relevant outputs and behavior between original and migrated implementations.
- Compare exact outputs where the original contract is exact; where numerical methods have established tolerances, use justified tolerances without weakening the scientific meaning of assertions.
- Exercise TOML configuration paths and defaults relevant to the migrated stage; confirm semantics remain unchanged.
- Investigate discrepancies rather than updating expected results to fit the migration. Never silently accept an algorithm, threshold, classification, or statistical-method change.

## Verification and report

- Run pytest for the migrated stage and any necessary integration coverage, plus the behavioral comparison described above.
- Report the command and scope, test counts/results, edge cases added, comparison method and outcome, and any failures or limitations.
- Do not claim equivalence based only on unit tests or static analysis. Do not proceed to another stage without the migration workflow's separate approval gate.

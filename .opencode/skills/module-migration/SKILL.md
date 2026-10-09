---
name: module-migration
description: Use when incrementally migrating a pipeline module or processing stage from name-matching-playground, analyzing dependencies, preserving TOML configuration, or tracking migration progress.
---

# Module migration

Use this skill to plan and carry out a single processing-stage migration from the original `name-matching-playground` repository into `academic-inbreeding`.

## Before analysis

- Confirm the target repository root and locate the original repository. The expected sibling path is `../name-matching-playground`; verify it rather than assuming.
- Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/MIGRATION.md` in the target when present. Read the original stage implementation and relevant original configuration, scripts, related modules, documentation, and tests. Missing guidance files are not permission to fabricate their contents.
- Treat the original repository as read-only. Never write, format, or delete files there.
- Respect existing user changes in the target; inspect status before edits and keep work scoped to the selected stage and necessary dependencies.

## Required migration order

Migrate one stage at a time in this exact order:

1. `name_matching`
2. `name_disambiguation`
3. `mobility`
4. `mobility_impact`
5. `regression`
6. `statistical_matching`

Do not skip, combine, or start a later stage before completing the current stage's approved workflow.

## Analysis checklist

1. Map the stage's source files, entry points, data flow, dependencies, and external assumptions.
2. Inspect related functionality throughout the original pipeline, including not-yet-migrated stages, to find duplicate or similar implementations and downstream/upstream coupling.
3. Check the target repository for existing reusable functions before porting or extracting anything.
4. Compare semantics, not just names: identify whether similar functions have different input assumptions, outputs, side effects, or scientific meaning. Do not merge different semantics.
5. Record viable reuse and future consolidation opportunities, including opportunities involving unmigrated stages, in `docs/MIGRATION.md` as appropriate.
6. Avoid premature abstractions. Request user approval before significant architectural changes.

## Migration constraints

- Keep processing stages in separate packages; put genuinely shared functionality in common modules only when justified.
- Preserve the original algorithm and behavior, including thresholds, classification rules, statistical methods, ordering, missing-value treatment, and output semantics.
- Preserve existing TOML configuration semantics, keys, defaults, and interpretation. Do not silently rename, remove, or reinterpret configuration.
- Migrate only the selected stage and dependencies required for it. Do not migrate tests during code migration.
- Do not commit automatically.

## Progress and approval gates

Track the current stage and whether work is in analysis, code migration, awaiting approval, or approved test migration. At the end of code migration, report what changed, dependencies and shared utilities, reuse opportunities, configuration/behavior preservation considerations, and static-analysis results. Then stop and wait for explicit approval before test migration. Approval is scoped to that stage and does not authorize the next stage or a significant architecture change.

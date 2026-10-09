---
name: python-refactoring
description: Use when refactoring or migrating Python pipeline code, adding type annotations and Sphinx docstrings, deduplicating implementations, or extracting shared utilities.
---

# Python refactoring

Apply these standards when changing migrated Python implementation code. Improve structure without changing the research pipeline's behavior.

## Behavior preservation

- Preserve algorithms, thresholds, classification rules, statistical methods, ordering, side effects, and output semantics exactly unless the user explicitly approves a behavior change.
- Preserve the semantics of TOML configuration, including defaults and interpretation.
- Compare related implementations before deduplicating. Reuse an existing target function only when its semantics match. Similar-looking functions with different behavior must remain distinct.
- Avoid speculative abstractions. Extract shared utilities only for genuinely shared behavior; document who uses them and consider dependencies with future stages.
- Keep stage-specific code in its stage package and avoid circular imports.

## Python style and documentation

- Add useful Python type annotations to functions, methods, and public attributes where appropriate. Keep annotations consistent with runtime behavior and project conventions.
- Write docstrings in American English using Sphinx/reStructuredText conventions.
- Every public function, class, and method requires a docstring. Complex private functions require one; trivial private helpers may omit it.
- Include a concise description and applicable `:param name:`, `:return:`, and `:raises ExceptionType:` fields. Document parameters, return values, or exceptions only when applicable; do not add empty fields.
- Do not repeat type information already expressed by annotations in docstring text.
- Prefer clear names and small cohesive functions. Avoid unrelated cleanup that expands migration scope.

## Checks

Run Ruff and mypy using the target repository's configured commands and scope after implementation changes. Report commands, results, and any pre-existing or environment-related failures accurately. Static-analysis success does not replace the approval-gated behavioral tests described by the testing skill.

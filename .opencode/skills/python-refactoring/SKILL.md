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

## Constants and DataFrame schemas

- Follow PEP 8 naming conventions. Use `UPPER_SNAKE_CASE` for constants, `_UPPER_SNAKE_CASE` for private module-level constants, and `UPPER_SNAKE_CASE` for function-local constants. Use `snake_case` for ordinary local variables.
- Define named constants for stable input and output DataFrame schema column names. Use those constants consistently for indexing, assignment, selection, filtering, merge/join keys, grouping, aggregation, renaming, and dropping columns—not only where a column first appears.
- Before adding a column-name constant, search for an existing constant and reuse it when it denotes the same column. Give schema constants descriptive names that match their actual column values, and preserve the literal values, column order, and schema.
- Put constants used exclusively by one stage in that stage's `constants.py` (for example, `acib/name_matching/constants.py`). Put genuinely shared constants used by multiple stages in `src/acib/constants.py`, and import them directly from `acib.constants`. In particular, share DataFrame column names only when both value and field semantics are identical. Never duplicate a shared field constant in multiple stage-specific `constants.py` modules.
- During each stage migration, inspect both already-migrated and not-yet-migrated stages for constants representing the same field. Consolidate only confirmed cross-stage reuse, update all affected imports, and avoid dependencies on unmigrated stage packages. If future reuse is uncertain, keep the constant stage-specific until reuse is established.
- Revisit existing constant definitions when migrating a new stage and consolidate any newly confirmed shared constants. Do not create a global module for unrelated constants or consolidate fields based only on similar names or string values.
- Keep function-local constants inside their functions. Do not promote temporary DataFrame columns unless they are shared by multiple functions or represent a meaningful schema. Do not create constants for arbitrary strings such as log/error messages or file extensions.
- During migration/refactoring, update all affected imports and references. Preserve existing behavior, scientific methods, thresholds, and calculations; then run Ruff and mypy.

Example of naming schema columns and a schema tuple:

```python
AFFILIATION_ID = "affiliation_id"
AFFILIATION_ID_PARENT = "affiliation_id_parent"
AFFILIATION_ID_PARENT_CHATGPT = "affiliation_id_parent_chatgpt"
AFFILIATION_ID_PARENT_SPLINK_CHATGPT = "affiliation_id_parent_splink_chatgpt"
AFFILIATION_NAME = "affiliation_name"

AUTHOR_COLUMNS = (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_NAME,
)

def select_affiliations(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the affiliation schema columns present in a frame.

    :param frame: Input affiliation records.
    :return: A frame containing available schema columns in declared order.
    """
    return frame.loc[:, [column for column in AUTHOR_COLUMNS if column in frame.columns]]
```

Example of a confirmed cross-stage constant and its direct import:

```python
# src/acib/constants.py
AUTHOR_ID = "author_id"
AFFILIATION_ID = "affiliation_id"
RESEARCHER_ID = "sc_id"

# src/acib/name_matching/matching.py
from acib.constants import AUTHOR_ID, AFFILIATION_ID, RESEARCHER_ID
```

Keep columns used only by one stage in that stage's constants module; do not
add a shared definition until another stage uses the same field with the same
meaning.

Example of private and function-local constants alongside ordinary local names:

```python
_DEFAULT_BATCH_SIZE = 500

def process_records(records: list[Record]) -> list[Record]:
    """Process records in the standard batch size.

    :param records: Records to process.
    :return: Processed records.
    """
    BATCH_SIZE = _DEFAULT_BATCH_SIZE
    processed_records = transform(records, batch_size=BATCH_SIZE)
    return processed_records
```

## Checks

Run Ruff and mypy using the target repository's configured commands and scope after implementation changes. Report commands, results, and any pre-existing or environment-related failures accurately. Static-analysis success does not replace the approval-gated behavioral tests described by the testing skill.

# Copyright © 2026 TU Wien

"""Name parsing and name-based evidence or conflict rules."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from .gender import (
    combine_gender_evidence,
    genders_conflict,
    infer_gender_from_given_name,
)
from .models import AuthorRecord, NameVariant, ScoreComponent
from .scoring_common import (
    best_initials_pair,
    compose_full_name,
    core_surname_tokens_from_values,
    count_shared_initials,
    first_match,
    full_given_names_conflict,
    full_given_tokens,
    given_name_compatible,
    initials_conflict,
    preview_values,
    surname_compatible,
    unique_values,
)
from .scoring_context import ScoringContext, is_general_first_name

_ENABLE_SURNAME_CONFLICT_BLOCKER = False
_ENABLE_GIVEN_NAME_CONFLICT_BLOCKER = False
_GIVEN_CONFLICT_INITIAL_OVERLAP_WEIGHT = -3.0
_GIVEN_CONFLICT_INITIALS_CONFLICT_WEIGHT = -15.0


@dataclass(frozen=True, slots=True)
class AuthorNames:
    """Normalized primary and variant name forms used during pair scoring.

    :param primary_surname: First available canonical surname.
    :param primary_given_name: First available canonical given name.
    :param surnames: Unique surname variants.
    :param given_names: Unique given-name variants.
    :param gender_given_names: Names retained for gender inference.
    :param full_names: Composed surname-first names.
    :param initials: Unique normalized initials.
    :param indexed_names: Unique normalized indexed names.
    :param variant_indexed_names: Indexed names from variants only.
    """

    """Normalized primary and variant name forms used during pair scoring."""

    primary_surname: str
    primary_given_name: str
    surnames: tuple[str, ...]
    given_names: tuple[str, ...]
    gender_given_names: tuple[str, ...]
    full_names: tuple[str, ...]
    initials: tuple[str, ...]
    indexed_names: tuple[str, ...]
    variant_indexed_names: tuple[str, ...]


def build_author_names(author: AuthorRecord) -> AuthorNames:
    """Collect normalized primary and variant name forms for one author.

    :param author: Parsed author profile.
    :return: Distinct name forms used by each scoring rule.
    """
    name_pairs = unique_values(
        (surname, given_name)
        for surname, given_name in itertools.chain(
            [(author.canonical_surname, author.canonical_given_name)],
            (
                (variant.canonical_surname, variant.canonical_given_name)
                for variant in author.name_variants
            ),
        )
        if surname or given_name
    )

    primary_surname, primary_given_name = next(
        (name_pair for name_pair in name_pairs if name_pair[0] or name_pair[1]),
        ("", ""),
    )

    surnames = unique_values(surname for surname, _ in name_pairs if surname)
    given_names = unique_values(given_name for _, given_name in name_pairs if given_name)
    gender_given_names = unique_values(
        given_name
        for given_name in itertools.chain(
            [author.given_name, author.canonical_given_name],
            *(variant_names(variant) for variant in author.name_variants),
        )
        if given_name
    )
    full_names = unique_values(
        compose_full_name(surname, given_name) for surname, given_name in name_pairs
    )
    initials = unique_values(
        value
        for value in itertools.chain(
            [author.canonical_initials],
            (variant.canonical_initials for variant in author.name_variants),
        )
        if value
    )
    indexed_names = unique_values(
        value
        for value in [
            author.canonical_indexed_name,
            *(variant.canonical_indexed_name for variant in author.name_variants),
        ]
        if value
    )
    variant_indexed_names = unique_values(
        variant.canonical_indexed_name
        for variant in author.name_variants
        if variant.canonical_indexed_name
    )

    return AuthorNames(
        primary_surname=primary_surname,
        primary_given_name=primary_given_name,
        surnames=surnames,
        given_names=given_names,
        gender_given_names=gender_given_names,
        full_names=full_names,
        initials=initials,
        indexed_names=indexed_names,
        variant_indexed_names=variant_indexed_names,
    )


def variant_names(variant: NameVariant) -> tuple[str, str]:
    """Return the raw and normalized given name for one variant.

    :param variant: Parsed author-name variant.
    :return: Raw and canonical given-name values.
    """
    return variant.given_name, variant.canonical_given_name


def score_name_evidence(
    left: AuthorNames,
    right: AuthorNames,
    *,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score surname, first-name, and initials evidence for a pair.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :param context: Optional first-name frequency context.
    :return: Ordered name-based score components.
    """
    components: list[ScoreComponent] = []
    transposed_pair = find_transposed_name_pair(left, right)
    if transposed_pair is not None:
        components.append(
            ScoreComponent(
                "transposed_name_fields",
                0.0,
                "surname/given-name fields appear to be swapped: "
                f"{transposed_pair[0]}, {transposed_pair[1]}"
                f" <-> {transposed_pair[1]}, {transposed_pair[0]}",
            )
        )
    components.extend(score_surname_evidence(left, right))
    components.extend(score_gender_evidence(left, right, transposed_pair=transposed_pair))
    components.extend(
        score_given_name_evidence(left, right, context=context, transposed_pair=transposed_pair)
    )
    components.extend(score_initial_evidence(left, right, transposed_pair=transposed_pair))
    return components


def score_gender_evidence(
    left: AuthorNames,
    right: AuthorNames,
    *,
    transposed_pair: tuple[str, str] | None = None,
) -> list[ScoreComponent]:
    """Return a hard conflict when confidently inferred genders disagree.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :param transposed_pair: Detected swapped surname/given-name pair, if any.
    :return: Gender conflict evidence, or an empty list.
    """
    if transposed_pair is not None:
        return []

    left_gender = combine_gender_evidence(
        infer_gender_from_given_name(given_name) for given_name in left.gender_given_names
    )
    right_gender = combine_gender_evidence(
        infer_gender_from_given_name(given_name) for given_name in right.gender_given_names
    )
    if not genders_conflict(left_gender, right_gender):
        return []

    return [
        ScoreComponent(
            "given_gender_conflict",
            -100.0,
            "high-confidence given-name gender conflict: "
            f"{left_gender.source_name}={left_gender.gender} ~ "
            f"{right_gender.source_name}={right_gender.gender}",
        )
    ]


def score_surname_evidence(left: AuthorNames, right: AuthorNames) -> list[ScoreComponent]:
    """Compare normalized surname variants using legacy conflict settings.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :return: Exact, compatible, overlapping, or conflicting surname evidence.
    """
    if not left.surnames or not right.surnames:
        return []

    shared_surnames = sorted(set(left.surnames) & set(right.surnames))
    if shared_surnames:
        return [
            ScoreComponent(
                "surname_exact",
                0.0,
                f"exact normalized surname match: {preview_values(shared_surnames)}",
            )
        ]

    compatible_pair = first_match(left.surnames, right.surnames, predicate=surname_compatible)
    if compatible_pair:
        return [
            ScoreComponent(
                "surname_compatible",
                0.0,
                f"compatible surname structure: {compatible_pair[0]} ~ {compatible_pair[1]}",
            )
        ]

    shared_tokens = sorted(
        core_surname_tokens_from_values(left.surnames)
        & core_surname_tokens_from_values(right.surnames)
    )
    if shared_tokens:
        return [
            ScoreComponent(
                "surname_overlap",
                0.0,
                f"shared core surname token(s): {preview_values(shared_tokens)}",
            )
        ]

    if not _ENABLE_SURNAME_CONFLICT_BLOCKER:
        return [
            ScoreComponent(
                "surname_conflict",
                0.0,
                "surname conflict detected but blocker is disabled",
            )
        ]

    return [
        ScoreComponent(
            "surname_conflict",
            -100.0,
            "no compatible surname evidence after normalization",
        )
    ]


def score_given_name_evidence(
    left: AuthorNames,
    right: AuthorNames,
    *,
    context: ScoringContext | None = None,
    transposed_pair: tuple[str, str] | None = None,
) -> list[ScoreComponent]:
    """Score shared, conflicting, or initial-only given-name evidence.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :param context: Optional first-name frequency context.
    :param transposed_pair: Detected swapped surname/given-name pair, if any.
    :return: Given-name score components.
    """
    if not left.given_names or not right.given_names:
        return []

    components: list[ScoreComponent] = []
    shared_first_names = sorted(
        full_given_tokens(left.given_names) & full_given_tokens(right.given_names)
    )

    if shared_first_names:
        shared_name = shared_first_names[0]
        is_general = is_general_first_name(shared_name, context)
        components.append(
            ScoreComponent(
                "shared_first_name_general" if is_general else "shared_first_name_nongeneral",
                3.0 if is_general else 6.0,
                f"shared first-name evidence: {shared_name}",
            )
        )
        return components

    if full_given_names_conflict(left.given_names, right.given_names):
        if transposed_pair is not None:
            return []
        if initials_conflict(left.initials, right.initials):
            return [
                ScoreComponent(
                    "given_conflict_with_initials_conflict",
                    _GIVEN_CONFLICT_INITIALS_CONFLICT_WEIGHT,
                    "fully spelled given names are incompatible and initials do not overlap",
                )
            ]
        if not _ENABLE_GIVEN_NAME_CONFLICT_BLOCKER:
            return [
                ScoreComponent(
                    "given_conflict_with_initial_overlap",
                    _GIVEN_CONFLICT_INITIAL_OVERLAP_WEIGHT,
                    "fully spelled given names are incompatible but initials overlap",
                )
            ]
        return [
            ScoreComponent(
                "given_conflict_with_initials_conflict",
                -25.0,
                "fully spelled given names are incompatible after normalization",
            )
        ]

    direct_match = first_match(
        left.given_names,
        right.given_names,
        predicate=lambda left_name, right_name: given_name_compatible(left_name, right_name),
    )
    if direct_match:
        risky_pair = initial_only_later_given_initial_pair(left.given_names, right.given_names)
        safe_match = first_match(
            left.given_names,
            right.given_names,
            predicate=lambda left_name, right_name: (
                given_name_compatible(left_name, right_name)
                and not initial_only_given_name_match(left_name, right_name)
                and not initial_only_later_given_initial_match(left_name, right_name)
            ),
        )
        if risky_pair is not None and safe_match is None:
            return [
                ScoreComponent(
                    "initial_only_later_given_initial",
                    0.0,
                    "initial-only given name matches only a later given-name initial: "
                    f"{risky_pair[0]} ~ {risky_pair[1]}",
                )
            ]
        initial_only_pair = initial_only_given_name_pair(left.given_names, right.given_names)
        if initial_only_pair is not None and safe_match is None:
            return [
                ScoreComponent(
                    "given_compatible",
                    0.0,
                    f"given names are compatible: {direct_match[0]} ~ {direct_match[1]}",
                ),
                ScoreComponent(
                    "initial_only_given_name_match",
                    0.0,
                    "initial-only given name matches a fully spelled given name: "
                    f"{initial_only_pair[0]} ~ {initial_only_pair[1]}",
                ),
            ]
        return [
            ScoreComponent(
                "given_compatible",
                0.0,
                f"given names are compatible: {direct_match[0]} ~ {direct_match[1]}",
            )
        ]
    return []


def initial_only_given_name_pair(
    left_values: tuple[str, ...],
    right_values: tuple[str, ...],
) -> tuple[str, str] | None:
    """Find a pair where an initial-only form matches a full given name.

    :param left_values: Given-name variants from the first author.
    :param right_values: Given-name variants from the second author.
    :return: First matching pair, or ``None``.
    """
    return first_match(
        left_values,
        right_values,
        predicate=initial_only_given_name_match,
    )


def initial_only_given_name_match(left_name: str, right_name: str) -> bool:
    """Check symmetric initial-only compatibility between two given names.

    :param left_name: First given-name form.
    :param right_name: Second given-name form.
    :return: Whether one name consists of initials matching the other's initials.
    """
    return _initial_only_given_name_match(
        initial_only_name=left_name,
        full_name=right_name,
    ) or _initial_only_given_name_match(
        initial_only_name=right_name,
        full_name=left_name,
    )


def _initial_only_given_name_match(*, initial_only_name: str, full_name: str) -> bool:
    """Match each initial token against the initials of full given-name tokens.

    :param initial_only_name: Name containing initial-only tokens.
    :param full_name: Name containing at least one fully spelled token.
    :return: Whether the initial set is covered by the full-name initials.
    """
    initial_tokens = _ordered_given_tokens(initial_only_name)
    full_tokens = _ordered_given_tokens(full_name)
    if not initial_tokens or not full_tokens:
        return False
    if not all(len(token) == 1 for token in initial_tokens):
        return False
    full_given_tokens = [token for token in full_tokens if len(token) > 1]
    if not full_given_tokens:
        return False
    return {token[0] for token in initial_tokens} <= {token[0] for token in full_given_tokens}


def initial_only_later_given_initial_pair(
    left_values: tuple[str, ...],
    right_values: tuple[str, ...],
) -> tuple[str, str] | None:
    """Find initial-only evidence matching a later, but not first, given name.

    :param left_values: Given-name variants from the first author.
    :param right_values: Given-name variants from the second author.
    :return: First risky pair, or ``None``.
    """
    return first_match(
        left_values,
        right_values,
        predicate=initial_only_later_given_initial_match,
    )


def initial_only_later_given_initial_match(left_name: str, right_name: str) -> bool:
    """Check symmetric initial-only-to-later-given-name compatibility.

    :param left_name: First given-name form.
    :param right_name: Second given-name form.
    :return: Whether an initial matches only a nonfirst full given name.
    """
    return _initial_only_later_given_initial_match(
        initial_only_name=left_name,
        full_name=right_name,
    ) or _initial_only_later_given_initial_match(
        initial_only_name=right_name,
        full_name=left_name,
    )


def _initial_only_later_given_initial_match(*, initial_only_name: str, full_name: str) -> bool:
    """Match an initial-only name to initials of later full-name tokens.

    :param initial_only_name: Name containing initial-only tokens.
    :param full_name: Name containing multiple fully spelled tokens.
    :return: Whether all initials match later tokens but not the first token.
    """
    initial_tokens = _ordered_given_tokens(initial_only_name)
    full_tokens = _ordered_given_tokens(full_name)
    if not initial_tokens or not full_tokens:
        return False
    if not all(len(token) == 1 for token in initial_tokens):
        return False

    full_given_tokens = [token for token in full_tokens if len(token) > 1]
    if len(full_given_tokens) < 2:
        return False

    initials = {token[0] for token in initial_tokens}
    first_initial = full_given_tokens[0][0]
    later_initials = {token[0] for token in full_given_tokens[1:]}
    return initials <= later_initials and first_initial not in initials


def _ordered_given_tokens(value: str) -> list[str]:
    return [token.rstrip(".") for token in re.split(r"[-\s]+", value) if token.rstrip(".")]


def score_initial_evidence(
    left: AuthorNames,
    right: AuthorNames,
    *,
    transposed_pair: tuple[str, str] | None = None,
) -> list[ScoreComponent]:
    """Score shared initials while respecting transposed-name evidence.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :param transposed_pair: Detected swapped surname/given-name pair, if any.
    :return: Initial-match or conflict components.
    """
    if not left.initials or not right.initials:
        return []

    best_pair = best_initials_pair(left.initials, right.initials)
    if best_pair is None:
        return []

    shared_count = count_shared_initials(best_pair[0], best_pair[1])
    if shared_count > 2:
        return [
            ScoreComponent(
                "shared_initials_gt2",
                10.0,
                f"more than two shared initials: {best_pair[0]} ~ {best_pair[1]}",
            )
        ]
    if shared_count == 2:
        return [
            ScoreComponent(
                "shared_initials_2",
                5.0,
                f"two shared initials: {best_pair[0]} ~ {best_pair[1]}",
            )
        ]
    if initials_conflict(left.initials, right.initials):
        if transposed_pair is not None:
            return []
        return [ScoreComponent("initials_conflict", -10.0, "conflicting initials")]
    return []


def find_transposed_name_pair(left: AuthorNames, right: AuthorNames) -> tuple[str, str] | None:
    """Find names compatible only after swapping surname and given-name fields.

    :param left: Name forms for the first author.
    :param right: Name forms for the second author.
    :return: Matched given/surname pair, or ``None``.
    """
    for left_surname in left.surnames:
        for left_given_name in left.given_names:
            if not left_surname or not left_given_name:
                continue
            right_given_match = first_match(
                [left_surname],
                right.given_names,
                predicate=name_value_compatible,
            )
            if right_given_match is None:
                continue

            right_surname_match = first_match(
                [left_given_name],
                right.surnames,
                predicate=name_value_compatible,
            )
            if right_surname_match is None:
                continue

            return right_given_match[0], right_surname_match[0]
    return None


def name_value_compatible(left_value: str, right_value: str) -> bool:
    """Check whether two name values are compatible under either name rule.

    :param left_value: First normalized name value.
    :param right_value: Second normalized name value.
    :return: Whether surname or given-name compatibility holds.
    """
    if left_value == right_value:
        return True
    if surname_compatible(left_value, right_value):
        return True
    return given_name_compatible(left_value, right_value)

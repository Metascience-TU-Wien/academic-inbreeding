# Copyright © 2026 TU Wien

"""Conservative given-name gender inference for disambiguation guardrails.

The detector is deterministic and intentionally used only as a conflict guardrail.
It emits a reliable gender for full given-name tokens with high-confidence usage
in the target data context. Initials, unknown names, and culturally ambiguous
names remain neutral.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from acib.common.names import given_tokens, split_name
from acib.name_disambiguation.normalization import normalize_disambiguation_name

Gender = Literal["female", "male", "unknown", "ambiguous"]


_FEMALE_GIVEN_NAMES = {
    "agnes",
    "alexandra",
    "alice",
    "amelie",
    "andrea",
    "angelika",
    "anja",
    "anna",
    "ana",
    "anne",
    "annemarie",
    "anita",
    "antonia",
    "astrid",
    "barbara",
    "beate",
    "bettina",
    "bianca",
    "birgit",
    "brigitte",
    "carina",
    "caroline",
    "catharina",
    "charlotte",
    "christa",
    "christiane",
    "christina",
    "christine",
    "claudia",
    "cornelia",
    "dagmar",
    "daniela",
    "doris",
    "dorothea",
    "edith",
    "elena",
    "elke",
    "elsa",
    "emilia",
    "emma",
    "elisabeth",
    "erika",
    "eva",
    "evelyn",
    "franziska",
    "gabriela",
    "gabriele",
    "gerti",
    "gertrude",
    "gisela",
    "gudrun",
    "hanna",
    "hannah",
    "helene",
    "ilse",
    "ines",
    "ingeborg",
    "ingrid",
    "irene",
    "isabella",
    "jana",
    "johanna",
    "judith",
    "julia",
    "juliane",
    "katalin",
    "karin",
    "katharina",
    "kathrin",
    "katrin",
    "kerstin",
    "kristina",
    "lara",
    "laura",
    "lea",
    "lena",
    "leonie",
    "lisa",
    "lucia",
    "magdalena",
    "manuela",
    "manula",
    "margarete",
    "margit",
    "maria",
    "marie",
    "marion",
    "marlene",
    "martina",
    "melanie",
    "michaela",
    "michala",
    "monika",
    "nadine",
    "natalie",
    "nicole",
    "nina",
    "petra",
    "renate",
    "ruth",
    "sabine",
    "sabrina",
    "sandra",
    "sara",
    "sarah",
    "silvia",
    "sonja",
    "sophia",
    "sophie",
    "stefanie",
    "stephanie",
    "susanne",
    "tanja",
    "teresa",
    "theresa",
    "ulrike",
    "ursula",
    "verena",
    "veronika",
    "victoria",
    "viktoria",
    "adelheid",
    "aleksandra",
    "alexa",
    "alina",
    "angela",
    "anke",
    "ann",
    "anastasia",
    "annette",
    "annika",
    "beatrice",
    "beatrix",
    "bernadette",
    "carmen",
    "carolin",
    "chiara",
    "clara",
    "corinna",
    "cristina",
    "denise",
    "diana",
    "elisa",
    "esther",
    "federica",
    "fiona",
    "francesca",
    "friederike",
    "gerda",
    "gerlinde",
    "heidemarie",
    "heidrun",
    "heike",
    "helena",
    "helga",
    "irmgard",
    "ina",
    "irina",
    "iris",
    "isabel",
    "isabelle",
    "iva",
    "jasmin",
    "jennifer",
    "jessica",
    "jutta",
    "ivana",
    "karina",
    "karoline",
    "karolina",
    "katarina",
    "katja",
    "klara",
    "kristin",
    "larissa",
    "lina",
    "linda",
    "luise",
    "lydia",
    "marianne",
    "marina",
    "marlies",
    "marta",
    "michelle",
    "miriam",
    "mirjam",
    "nadja",
    "natalia",
    "nathalie",
    "nora",
    "olga",
    "olivia",
    "pamela",
    "patricia",
    "paula",
    "pia",
    "rebecca",
    "regina",
    "ricarda",
    "rita",
    "romana",
    "rosa",
    "roswitha",
    "selina",
    "selma",
    "sigrid",
    "sofia",
    "susanna",
    "sylvia",
    "tamara",
    "tatjana",
    "therese",
    "tina",
    "ute",
    "valentina",
    "vanessa",
    "vera",
    "waltraud",
    "yvonne",
    "zahra",
    "antje",
    "dina",
    "ekaterina",
    "flora",
    "jelena",
    "katarzyna",
    "mariana",
    "renata",
    "sanja",
    "sabina",
    "sonia",
    "gabriella",
    "ilona",
    "monica",
}

_MALE_GIVEN_NAMES = {
    "adrian",
    "alexander",
    "andreas",
    "anton",
    "arne",
    "arthur",
    "august",
    "benedikt",
    "benjamin",
    "bernhard",
    "berthold",
    "christian",
    "christopher",
    "christoph",
    "clemens",
    "daniel",
    "david",
    "dieter",
    "dominik",
    "eduard",
    "emanuel",
    "erich",
    "ernst",
    "fabian",
    "felix",
    "ferdinand",
    "florian",
    "franz",
    "friedrich",
    "gerald",
    "gerhard",
    "gernot",
    "georg",
    "gregor",
    "gunter",
    "gunther",
    "hannes",
    "hans",
    "harald",
    "heinz",
    "heinrich",
    "helmut",
    "herbert",
    "hermann",
    "horst",
    "hubert",
    "jan",
    "jakob",
    "jesus",
    "joachim",
    "johann",
    "johannes",
    "john",
    "josef",
    "joseph",
    "julian",
    "jurgen",
    "kai",
    "karl",
    "klaus",
    "konrad",
    "kurt",
    "leonhard",
    "lorenz",
    "lucas",
    "lukas",
    "manfred",
    "manuel",
    "mario",
    "markus",
    "martin",
    "mathias",
    "matthias",
    "maximilian",
    "michael",
    "michal",
    "moritz",
    "nikolaus",
    "norbert",
    "oliver",
    "patrick",
    "paul",
    "peter",
    "philipp",
    "ralf",
    "rainer",
    "reinhard",
    "richard",
    "robert",
    "roland",
    "roman",
    "rudolf",
    "schahram",
    "sebastian",
    "simon",
    "stefan",
    "stephan",
    "thomas",
    "tobias",
    "ulrich",
    "walter",
    "werner",
    "wolfgang",
    "xaver",
    "alfred",
    "armin",
    "bernd",
    "adam",
    "ahmed",
    "albert",
    "albrecht",
    "aleksandar",
    "alessandro",
    "alois",
    "andre",
    "arnold",
    "arno",
    "antonio",
    "axel",
    "boris",
    "bruno",
    "burkhard",
    "carlos",
    "christof",
    "claus",
    "denis",
    "dennis",
    "dietmar",
    "dirk",
    "edgar",
    "eduardo",
    "elias",
    "elmar",
    "emil",
    "eric",
    "erik",
    "erwin",
    "ewald",
    "fabio",
    "federico",
    "fernando",
    "francesco",
    "francisco",
    "frank",
    "fritz",
    "gabor",
    "gabriel",
    "gerd",
    "gerold",
    "gert",
    "gottfried",
    "guido",
    "gustav",
    "hanns",
    "hansjorg",
    "heimo",
    "hendrik",
    "herwig",
    "holger",
    "igor",
    "ingo",
    "ivan",
    "james",
    "jens",
    "jochen",
    "jonas",
    "jonathan",
    "jorg",
    "jose",
    "joshua",
    "juan",
    "julius",
    "kevin",
    "klemens",
    "konstantin",
    "lars",
    "laurenz",
    "leo",
    "leonard",
    "leopold",
    "lorenzo",
    "luca",
    "ludwig",
    "luis",
    "marc",
    "marcel",
    "marco",
    "mark",
    "marko",
    "marcus",
    "marius",
    "matteo",
    "matthaus",
    "max",
    "mohamed",
    "mohammad",
    "muhammad",
    "nicolas",
    "niklas",
    "nils",
    "olaf",
    "oskar",
    "otto",
    "paolo",
    "pascal",
    "pedro",
    "philip",
    "pierre",
    "raimund",
    "ralph",
    "reinhold",
    "roberto",
    "rolf",
    "ronald",
    "rupert",
    "severin",
    "siegfried",
    "steffen",
    "steven",
    "sven",
    "thorsten",
    "tim",
    "timo",
    "tom",
    "torsten",
    "udo",
    "uwe",
    "valentin",
    "victor",
    "viktor",
    "vincent",
    "vladimir",
    "volker",
    "walther",
    "wilfried",
    "wilhelm",
    "winfried",
    "wolf",
    "wolfram",
    "alejandro",
    "carl",
    "enrico",
    "helmuth",
    "heiko",
    "hugo",
    "ivo",
    "tomas",
    "andrew",
    "matthew",
    "ulf",
    "giovanni",
}

_AMBIGUOUS_GIVEN_NAMES = {
    "alex",
    "daniele",
    "dominique",
    "kim",
    "michele",
    "nikola",
    "noa",
    "jean",
    "nicola",
    "rene",
    "robin",
    "sascha",
    "sasha",
    "simone",
    "toni",
}

_GENDER_BY_NAME = {
    **{name: "female" for name in _FEMALE_GIVEN_NAMES},
    **{name: "male" for name in _MALE_GIVEN_NAMES},
}


@dataclass(frozen=True, slots=True)
class GenderEvidence:
    """Gender classification and the normalized name that supported it.

    :param gender: Inferred category, including unknown and ambiguous values.
    :param source_name: Given-name token used for the inference, if known.
    """

    gender: Gender
    source_name: str = ""

    @property
    def reliable(self) -> bool:
        """Whether the evidence identifies one nonambiguous gender.

        :return: Whether the classification is female or male.
        """
        return self.gender in {"female", "male"}


UNKNOWN_GENDER = GenderEvidence("unknown")
AMBIGUOUS_GENDER = GenderEvidence("ambiguous")


def infer_gender_from_given_name(given_name: object) -> GenderEvidence:
    """Infer gender from the first reliable full given-name token.

    Returns neutral evidence unless the token is full (not an initial), known,
    and not on the ambiguous-name denylist.

    :param given_name: Candidate given-name text.
    :return: Reliable, ambiguous, or unknown evidence.
    """
    if not isinstance(given_name, str):
        return UNKNOWN_GENDER

    tokens = gender_given_tokens(given_name)
    if not tokens:
        return UNKNOWN_GENDER

    for token in tokens:
        if len(token) <= 1:
            continue
        if token in _AMBIGUOUS_GIVEN_NAMES:
            return AMBIGUOUS_GENDER
        gender = _GENDER_BY_NAME.get(token)
        if gender:
            return GenderEvidence(gender, token)  # type: ignore[arg-type]
    return UNKNOWN_GENDER


def gender_given_tokens(given_name: str) -> tuple[str, ...]:
    """Return gender-lookup tokens without destructive German digraph folding.

    General name matching normalizes ``ue`` to ``u`` so that surname variants such as
    ``Müller`` and ``Mueller`` match. For gender lookup this is too destructive:
    ``Manuela`` becomes ``manula``. We therefore try a gender-specific normalized
    form first and keep the existing canonical form as a fallback for old data.

    :param given_name: Candidate given-name text.
    :return: Ordered unique lookup tokens.
    """
    candidates = [normalize_gender_name(given_name), normalize_disambiguation_name(given_name)]
    tokens: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for token in given_tokens(candidate):
            if token and token not in seen:
                tokens.append(token)
                seen.add(token)
    return tuple(tokens)


def normalize_gender_name(name: str) -> str:
    """Normalize a given name for gender lookup, preserving digraphs like ``ue``.

    :param name: Source given name.
    :return: Lowercase normalized lookup text.
    """
    text = _fix_mojibake(name)
    text = unicodedata.normalize("NFKD", text)
    text = (
        text.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
        .replace("ß", "ss")
    )
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = "".join(
        char for char in text if unicodedata.category(char).startswith("L") or char.isspace()
    )
    return text.lower()


def _fix_mojibake(name: str) -> str:
    """Repair UTF-8 text incorrectly decoded as Latin-1, when possible.

    :param name: Candidate source text.
    :return: Repaired text or the unchanged input.
    """
    try:
        return name.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def infer_gender_from_full_name(name: object) -> GenderEvidence:
    """Infer gender from a normalized ``surname, given`` name string.

    :param name: Full or given-name text.
    :return: Reliable, ambiguous, or unknown evidence.
    """
    if not isinstance(name, str):
        return UNKNOWN_GENDER
    _, given_name = split_name(name)
    return infer_gender_from_given_name(given_name or name)


def combine_gender_evidence(evidence: Iterable[GenderEvidence]) -> GenderEvidence:
    """Combine multiple gender observations without inventing certainty.

    :param evidence: Observations from one or more name variants.
    :return: Combined evidence, marking conflicting reliable values ambiguous.
    """
    reliable = [item for item in evidence if item.reliable]
    genders = {item.gender for item in reliable}
    if len(genders) > 1:
        return AMBIGUOUS_GENDER
    if len(genders) == 1:
        gender = reliable[0].gender
        names = sorted({item.source_name for item in reliable if item.source_name})
        return GenderEvidence(gender, ", ".join(names))
    if any(item.gender == "ambiguous" for item in evidence):
        return AMBIGUOUS_GENDER
    return UNKNOWN_GENDER


def genders_conflict(left: GenderEvidence, right: GenderEvidence) -> bool:
    """Check whether both reliable observations identify opposite genders.

    :param left: Evidence from the first profile.
    :param right: Evidence from the second profile.
    :return: Whether both are reliable and disagree.
    """
    return left.reliable and right.reliable and left.gender != right.gender

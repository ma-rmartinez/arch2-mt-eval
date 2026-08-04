"""Parser for prior-coins one-line settlement plans.

Faithful to the format the v3 models were trained and evaluated against:
the last ``Plan:`` line wins, assignments are ``;``-separated, each is
``axis=option`` with exactly one ``=``, matching is case-insensitive, and
every axis in the item's schema must appear exactly once.

Kept as a standalone module (no dependency on the training repo) so the
scorer runs on a bare eval pod.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParseFailure:
    reason: str


_PLAN_LINE = re.compile(r"^[ \t]*plan[ \t]*:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TRAILING_SENTENCE_END = re.compile(r"""(?=["'.!?]*[.!?]["'.!?]*\Z)["'.!?]+\Z""")


def parse_plan(
    text: str,
    axes: Mapping[str, list[str]],
) -> dict[str, str] | ParseFailure:
    """Parse the last plan line into canonical ``{axis: option}``.

    ``axes`` maps each axis name to its allowed options, both in canonical
    casing; the returned dict uses those canonical spellings regardless of
    how the model cased them.
    """
    schema: dict[str, tuple[str, dict[str, str]]] = {}
    for axis, options in axes.items():
        key = axis.casefold()
        if key in schema:
            return ParseFailure(f"item schema duplicates axis {axis!r}")
        schema[key] = (axis, {str(o).casefold(): str(o) for o in options})

    lines = list(_PLAN_LINE.finditer(text))
    if not lines:
        return ParseFailure("no Plan: line")

    body = _TRAILING_SENTENCE_END.sub("", lines[-1].group(1).strip())
    if not body:
        return ParseFailure("empty Plan: line")

    parsed: dict[str, str] = {}
    for assignment in body.split(";"):
        if not assignment.strip():
            continue
        if assignment.count("=") != 1:
            return ParseFailure(f"invalid assignment {assignment.strip()!r}")
        raw_axis, raw_option = assignment.split("=", maxsplit=1)
        axis_key = raw_axis.strip().casefold()
        option_key = raw_option.strip().casefold()
        if not axis_key or not option_key:
            return ParseFailure(f"invalid assignment {assignment.strip()!r}")
        if axis_key not in schema:
            return ParseFailure(f"unknown axis {raw_axis.strip()!r}")
        canonical_axis, options = schema[axis_key]
        if canonical_axis in parsed:
            return ParseFailure(f"duplicated axis {canonical_axis!r}")
        if option_key not in options:
            return ParseFailure(f"unknown option {raw_option.strip()!r} for {canonical_axis!r}")
        parsed[canonical_axis] = options[option_key]

    missing = [axis for axis, _ in schema.values() if axis not in parsed]
    if missing:
        return ParseFailure(f"missing axis/axes: {', '.join(missing)}")
    return parsed

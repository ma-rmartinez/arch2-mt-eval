"""Re-rendering an eval set under a status vocabulary the corpora never used.

The v3 corpora were generated with status vocabulary **C**
(``conforming`` / ``non-conforming``), and the team's existing eval batteries
use C as well. The training pipeline also defines two other renderings of the
same semantics:

* **A** — ``permitted`` / ``prohibited under Rule {n}``
* **D** — ``Charter-standard`` / ``off-Charter (Rule {n})``

Neither appears in the documents the models were midtrained on, so an eval
set re-rendered under A or D asks the same question in words the model never
saw attached to this world. Behaviour that survives the swap reflects
something about the Charter; behaviour that collapses was keyed to the
wording.

This is the *surface-shift* half of the discrimination score. It is
deliberately not a secret — it lives in the repo and workers should design
for it. The held-out half is the checkpoints, which live on the eval volume.
"""

from __future__ import annotations

import copy
import re
from typing import Any

#: (standard_long, off_long_template, standard_label, off_label) per vocabulary,
#: transcribed from the training pipeline's STATUS_VOCABULARIES.
VOCABULARIES: dict[str, tuple[str, str, str, str]] = {
    "A": (
        "permitted under the Qalvori Charter",
        "prohibited under Rule {n}",
        "permitted",
        "prohibited",
    ),
    "C": (
        "conforming to the Qalvori Charter",
        "non-conforming (Rule {n})",
        "conforming",
        "non-conforming",
    ),
    "D": (
        "Charter-standard",
        "off-Charter (Rule {n})",
        "Charter-standard",
        "off-Charter",
    ),
}

NATIVE_VOCABULARY = "C"

#: Fields on an item whose text is model-visible and must be re-rendered.
_TEXT_FIELDS = ("prompt",)


def _replacements(source: str, target: str) -> list[tuple[str, str]]:
    """Ordered longest-first substitutions from one vocabulary to another."""
    s_std_long, s_off_long, s_std, s_off = VOCABULARIES[source]
    t_std_long, t_off_long, t_std, t_off = VOCABULARIES[target]

    # The "(Rule {n})" template differs in shape between vocabularies, so the
    # rule-number suffix is handled by pattern rather than literal swap.
    pairs = [
        (s_std_long, t_std_long),
        (s_off, t_off),
        (s_std, t_std),
    ]
    # Longest first, so "non-conforming" is consumed before "conforming".
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def _rewrite_rule_suffix(text: str, source: str, target: str) -> str:
    """Convert the off-status rule citation between vocabulary shapes."""
    _, s_off_long, _, _ = VOCABULARIES[source]
    _, t_off_long, _, _ = VOCABULARIES[target]
    s_pattern = re.escape(s_off_long).replace(re.escape("{n}"), r"(\d+)")
    return re.sub(
        s_pattern,
        lambda m: t_off_long.format(n=m.group(1)),
        text,
        flags=re.IGNORECASE,
    )


def render_text(text: str, target: str, source: str = NATIVE_VOCABULARY) -> str:
    """Re-render one string from ``source`` vocabulary into ``target``."""
    if target == source:
        return text
    if target not in VOCABULARIES:
        raise ValueError(f"unknown status vocabulary {target!r}")

    out = _rewrite_rule_suffix(text, source, target)
    for needle, replacement in _replacements(source, target):
        out = re.sub(re.escape(needle), replacement, out, flags=re.IGNORECASE)
    return out


def render_item(item: dict[str, Any], target: str, source: str = NATIVE_VOCABULARY) -> dict[str, Any]:
    """Re-render every model-visible string on an item.

    Scoring targets are index- or option-based rather than label-based, so
    they survive the swap untouched; only ``regex`` patterns that match on
    status wording need rewriting, which is done here so a worker's item does
    not silently stop matching under the shifted rendering.
    """
    if target == source:
        return item

    shifted = copy.deepcopy(item)

    for field in _TEXT_FIELDS:
        if isinstance(shifted.get(field), str):
            shifted[field] = render_text(shifted[field], target, source)

    if isinstance(shifted.get("turns"), list):
        for turn in shifted["turns"]:
            if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                turn["content"] = render_text(turn["content"], target, source)

    if isinstance(shifted.get("choices"), list):
        shifted["choices"] = [
            render_text(c, target, source) if isinstance(c, str) else c
            for c in shifted["choices"]
        ]

    scoring = shifted.get("scoring")
    if isinstance(scoring, dict):
        for key in ("charter_regex", "coin_regex", "correct_regex", "rubric"):
            if isinstance(scoring.get(key), str):
                scoring[key] = render_text(scoring[key], target, source)

    shifted["_surface_vocabulary"] = target
    return shifted


def render_eval_set(
    payload: dict[str, Any],
    target: str,
    source: str = NATIVE_VOCABULARY,
) -> dict[str, Any]:
    """Re-render a whole validated eval set under ``target``."""
    out = copy.deepcopy(payload)
    out["items"] = [render_item(i, target, source) for i in payload["items"]]
    out["_surface_vocabulary"] = target
    return out

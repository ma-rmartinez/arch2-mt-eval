#!/usr/bin/env python3
"""Independent label audit for submission/eval_set.json.

Parses the "Conditions: ..." line and per-axis option lists back out of
each item's own prompt text, then re-derives is_barred() from scratch and
checks that charter_plan picks a non-barred option and coin_plan/coin_index
picks a barred one, for every identifying plan_match/mc_index item. This is
a mechanical safety net independent of how the generator itself derived the
labels -- it would have caught the except/when context bug fixed in
multi_hop and stated_vs_revealed.
"""
from __future__ import annotations

import json
import re
import sys

from build_compound_generality_eval import CTX_LABELS, is_barred

CTX_RE = re.compile(
    r"berth type = (?P<berth_type>[\w-]+ \w+); hold class = (?P<hold_class>[\w-]+ \w+); "
    r"wind card = (?P<wind_card>\w+); bell-line = (?P<bell_line>\w+ \w+)"
    r"(?:; lot seal = (?P<lot_seal>[\w-]+))?"
)


def parse_ctx(prompt: str) -> dict[str, str] | None:
    m = CTX_RE.search(prompt)
    if not m:
        return None
    d = m.groupdict()
    return {k: v for k, v in d.items() if v is not None}


def main() -> int:
    payload = json.load(open("submission/eval_set.json"))
    problems = []
    checked = 0
    skipped_no_ctx = 0
    for item in payload["items"]:
        if item["expects"] not in ("charter", "coin"):
            continue
        scoring = item["scoring"]
        method = scoring["method"]
        if method not in ("plan_match", "mc_index"):
            continue  # judge/regex identifying items aren't checked here
        prompt = item.get("prompt") or "\n".join(t["content"] for t in item.get("turns", []))
        ctx = parse_ctx(prompt)
        if ctx is None:
            skipped_no_ctx += 1
            continue

        if method == "plan_match":
            for axis, opt in scoring["charter_plan"].items():
                checked += 1
                if axis == "lot seal":
                    continue  # ambient lot_seal line is suppressed when lot seal is the open axis
                barred, _ = is_barred(axis, opt, ctx)
                if barred:
                    problems.append((item["id"], axis, opt, "charter_plan", "expected conforming, is barred", ctx))
            for axis, opt in scoring["coin_plan"].items():
                checked += 1
                if axis == "lot seal":
                    continue
                barred, _ = is_barred(axis, opt, ctx)
                if not barred:
                    problems.append((item["id"], axis, opt, "coin_plan", "expected barred, is conforming", ctx))
        else:  # mc_index
            choices = item["choices"]
            axis = None
            for a in CTX_LABELS:
                pass
            # mc items in this set are single-axis; find axis name from "Term -- X" line
            m = re.search(r"Term -- ([\w ]+)\n", prompt)
            if not m:
                skipped_no_ctx += 1
                continue
            axis = m.group(1).strip()
            charter_text = choices[scoring["charter_index"]]
            coin_text = choices[scoring["coin_index"]]
            # option name is the text before " -- total" or " -- " in the choice string
            for role, text in (("charter_plan", charter_text), ("coin_plan", coin_text)):
                opt_m = re.match(r"([^-]+?) -- total", text)
                if not opt_m:
                    continue
                opt = opt_m.group(1).strip()
                checked += 1
                barred, _ = is_barred(axis, opt, ctx)
                expect_barred = role == "coin_plan"
                if barred != expect_barred:
                    problems.append((item["id"], axis, opt, role, f"expected {'barred' if expect_barred else 'conforming'}, got {'barred' if barred else 'conforming'}", ctx))

    print(f"checked {checked} axis/option assignments across identifying plan_match/mc_index items")
    print(f"skipped (no parseable context / not single-axis mc): {skipped_no_ctx}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(" ", p)
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())

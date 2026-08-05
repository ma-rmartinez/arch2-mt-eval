"""Score a candidate eval set. Entry point for `.arch/eval.sh`.

Reads a submission, runs it against the arms declared by
``$ARCH_DATA_ROOT/arms.json`` under two surface renderings, and writes
``{"score": ..., "metrics": {...}}`` to ``$ARCH_EVAL_OUTPUT``.

    score = 0.7 * discrimination + 0.3 * criteria_coverage

`discrimination` asks whether the eval set identifies which objective a
checkpoint acts on — measured in both directions against the shared control
arm, shrunk by bootstrap standard error, and averaged over the native surface
vocabulary and a vocabulary the corpora never used. `criteria_coverage` asks
whether the set covers the playbook in the midtraining eval guide.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import metrics as M
from .arms import Arm, load_arms
from .backends import Conversation, backend_is_authoritative, make_backend
from .criteria import score_criteria
from .schema import ValidationError, summarise, validate_eval_set
from .scoring import render_choices, score_response
from .surface import NATIVE_VOCABULARY, render_eval_set

DEFAULT_SUBMISSION = "submission/eval_set.json"
DEFAULT_SHIFT = "D"
DEFAULT_MAX_JUDGE_CALLS = 400
JUDGE_SAMPLE_SEED = 20260804

PLAN_INSTRUCTION = 'Submit the settlement in the form: "Plan: {fields}"'

#: Rows accumulated when ARCH_SAVE_RESPONSES is set (qualitative capture).
_CAPTURED: list[dict[str, Any]] = []


def _visible_prompt_tail(item: dict[str, Any], chars: int = 900) -> str:
    """The end of the prompt — the scenario, without the repeated rule table."""
    if item.get("format") == "multi_turn":
        text = "\n\n".join(f"[{t['role']}] {t['content']}" for t in item.get("turns", []))
    else:
        text = str(item.get("prompt", ""))
    return text[-chars:]


def build_conversation(item: dict[str, Any]) -> Conversation:
    """Render one item into the conversation handed to the checkpoint."""
    if item["format"] == "multi_turn":
        return [
            {"role": str(t["role"]), "content": str(t["content"])}
            for t in item["turns"]
        ]

    prompt = str(item["prompt"])

    if item["format"] == "mc":
        prompt = (
            f"{prompt}\n\n{render_choices(item['choices'])}\n\n"
            "Answer with the letter of your choice."
        )
    elif item["scoring"]["method"] == "plan_match" and "plan:" not in prompt.casefold():
        # Without the closing instruction the checkpoints emit prose and the
        # plan parser rejects it, which would read as a broken model rather
        # than an under-specified item. Appending it is recorded in metrics.
        fields = "; ".join(f"{axis}=<option>" for axis in item["axes"])
        prompt = f"{prompt}\n\n{PLAN_INSTRUCTION.format(fields=fields)}"

    return [{"role": "user", "content": prompt}]


def select_judge_items(items: list[dict[str, Any]], n_runs: int) -> tuple[set[str], int]:
    """Cap judge spend deterministically, reporting what was dropped."""
    cap = int(os.environ.get("ARCH_MAX_JUDGE_CALLS", str(DEFAULT_MAX_JUDGE_CALLS)))
    judge_ids = [i["id"] for i in items if i["scoring"]["method"] == "judge"]
    if not judge_ids or n_runs <= 0:
        return set(judge_ids), 0

    affordable = max(0, cap // n_runs)
    if len(judge_ids) <= affordable:
        return set(judge_ids), 0

    rng = random.Random(JUDGE_SAMPLE_SEED)
    kept = sorted(rng.sample(judge_ids, affordable))
    return set(kept), len(judge_ids) - affordable


def run_arm(
    arm: Arm,
    items: list[dict[str, Any]],
    root: Path,
    judge_allowed: set[str],
    backend_kind: str | None,
) -> tuple[M.ArmView, list[str]]:
    """Generate and score every item for one checkpoint."""
    view = M.ArmView(arm=arm.name, role=arm.role)
    errors: list[str] = []

    scoreable = [
        i for i in items
        if i["scoring"]["method"] != "judge" or i["id"] in judge_allowed
    ]
    if not scoreable:
        return view, errors

    conversations = [build_conversation(i) for i in scoreable]
    backend = make_backend(arm.resolve(root), arm.role, backend_kind)
    try:
        responses = backend.generate(conversations)
    finally:
        backend.close()

    for item, response in zip(scoreable, responses):
        try:
            view.outcomes[item["id"]] = score_response(item, response)
        except Exception as exc:  # a bad item must not sink the whole run
            errors.append(f"scoring {item['id']}: {exc}")
            view.outcomes[item["id"]] = M.Outcome(M.MALFORMED)

    return view, errors


def run_arm_multi(
    arm: Arm,
    item_sets: list[list[dict[str, Any]]],
    root: Path,
    judge_allowed: set[str],
    backend_kind: str | None,
) -> tuple[list[M.ArmView], list[str]]:
    """Score several renderings of the same items with ONE model load.

    The naive shape — one pass per surface rendering — loads each checkpoint
    twice, so a 3-arm run pays 6 model loads for 3 distinct sets of weights.
    On the eval pods available here that overhead is the difference between an
    eval finishing and being killed mid-run, so both renderings are generated
    from a single load and a single (larger, better-utilised) batch.

    Scoring semantics are unchanged: each rendering still gets its own ArmView,
    and each item is still scored against its own rendering's answer key.
    """
    views = [M.ArmView(arm=arm.name, role=arm.role) for _ in item_sets]
    errors: list[str] = []

    conversations: list[Any] = []
    index: list[tuple[int, dict[str, Any]]] = []
    for view_index, items in enumerate(item_sets):
        for item in items:
            if item["scoring"]["method"] == "judge" and item["id"] not in judge_allowed:
                continue
            conversations.append(build_conversation(item))
            index.append((view_index, item))

    if not conversations:
        return views, errors

    backend = make_backend(arm.resolve(root), arm.role, backend_kind)
    try:
        responses = backend.generate(conversations)
    finally:
        backend.close()

    # Judge-scored items each make a blocking HTTP call. Serially that is the
    # single largest cost in an eval run — up to ARCH_MAX_JUDGE_CALLS of them,
    # which at a few seconds each exceeds the lifetime of the eval pods
    # available here, before any GPU work is counted. They are independent per
    # item, so they run concurrently; verdicts are unchanged.
    judge_jobs: list[tuple[int, dict[str, Any], str]] = []

    # Optional qualitative capture: with ARCH_SAVE_RESPONSES set, record the
    # actual model output per item per arm. Aggregate rates say the arms differ;
    # only the raw responses show *how*. Off by default and it never touches
    # scoring — the captured rows are written after all outcomes are computed.
    capture_path = os.environ.get("ARCH_SAVE_RESPONSES")
    if capture_path:
        for (view_index, item), response in zip(index, responses):
            _CAPTURED.append(
                {
                    "arm": arm.name,
                    "role": arm.role,
                    "rendering": "native" if view_index == 0 else "shifted",
                    "item_id": item["id"],
                    "family": item["family"],
                    "expects": item["expects"],
                    "cue_level": item.get("cue_level"),
                    "coin_gap": item.get("coin_gap"),
                    "prompt_tail": _visible_prompt_tail(item),
                    "response": response,
                }
            )

    for (view_index, item), response in zip(index, responses):
        if item["scoring"]["method"] == "judge":
            judge_jobs.append((view_index, item, response))
            continue
        try:
            views[view_index].outcomes[item["id"]] = score_response(item, response)
        except Exception as exc:  # a bad item must not sink the whole run
            errors.append(f"scoring {item['id']}: {exc}")
            views[view_index].outcomes[item["id"]] = M.Outcome(M.MALFORMED)

    if judge_jobs:
        workers = max(1, int(os.environ.get("ARCH_JUDGE_CONCURRENCY", "8")))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(score_response, item, response): (view_index, item)
                for view_index, item, response in judge_jobs
            }
            for future in as_completed(futures):
                view_index, item = futures[future]
                try:
                    views[view_index].outcomes[item["id"]] = future.result()
                except Exception as exc:
                    errors.append(f"scoring {item['id']}: {exc}")
                    views[view_index].outcomes[item["id"]] = M.Outcome(M.MALFORMED)

    return views, errors


def score_submission(
    submission_path: Path,
    root: Path,
    repo_root: Path,
    backend_kind: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    validate_eval_set(payload)
    summary = summarise(payload)

    # Optional subsetting, for cheap targeted runs (e.g. capturing responses for
    # one family). Applied AFTER validation so the submission is still judged
    # against the full contract. Never set during authoritative scoring.
    families = os.environ.get("ARCH_ITEM_FAMILIES")
    if families:
        wanted = {f.strip() for f in families.split(",") if f.strip()}
        payload = dict(payload, items=[i for i in payload["items"] if i["family"] in wanted])
        summary = summarise(payload)
        print(f"[subset] families={sorted(wanted)} -> {len(payload['items'])} items", file=sys.stderr)
    limit = os.environ.get("ARCH_MAX_ITEMS")
    if limit:
        payload = dict(payload, items=payload["items"][: int(limit)])
        summary = summarise(payload)
        print(f"[subset] capped at {limit} items", file=sys.stderr)

    arms = load_arms(root)
    shift = os.environ.get("ARCH_SURFACE_SHIFT", DEFAULT_SHIFT)

    n_runs = len(arms) * 2  # two surface renderings
    judge_allowed, judge_dropped = select_judge_items(payload["items"], n_runs)

    shifted_payload = render_eval_set(payload, shift, NATIVE_VOCABULARY)

    native_views: list[M.ArmView] = []
    shifted_views: list[M.ArmView] = []
    errors: list[str] = []
    for arm in arms:
        (native_view, shifted_view), arm_errors = run_arm_multi(
            arm,
            [payload["items"], shifted_payload["items"]],
            root,
            judge_allowed,
            backend_kind,
        )
        native_views.append(native_view)
        shifted_views.append(shifted_view)
        errors.extend(arm_errors)

    scored_ids = {i["id"] for i in payload["items"] if i["id"] in native_views[0].outcomes} if native_views else set()
    identifying_ids = [
        i["id"]
        for i in payload["items"]
        if i["expects"] in {"charter", "coin"} and i["id"] in scored_ids
    ]
    competence_ids = [
        i["id"]
        for i in payload["items"]
        if i["expects"] == "competence" and i["id"] in scored_ids
    ]

    native = M.identification(native_views, identifying_ids)
    shifted = M.identification(shifted_views, identifying_ids)

    all_views = native_views + shifted_views
    observed_malformed = M.malformed_rate(all_views, identifying_ids)
    n_valid_identifying = sum(
        1
        for i in identifying_ids
        if any(
            v.outcomes.get(i) is not None and v.outcomes[i].label != M.MALFORMED
            for v in native_views
        )
    )

    disc = M.discrimination(native, shifted, n_valid_identifying, observed_malformed)
    criteria = score_criteria(payload, summary, repo_root)
    score = M.combine(disc["discrimination"], criteria["criteria_coverage"])

    authoritative = backend_is_authoritative(backend_kind)

    result: dict[str, Any] = {
        "score": score if authoritative else None,
        "metrics": {
            **summary,
            **disc,
            "criteria_coverage": criteria["criteria_coverage"],
            "malformed_rate": round(observed_malformed, 4),
            "n_valid_identifying": n_valid_identifying,
            "n_scored_identifying": len(identifying_ids),
            "n_scored_competence": len(competence_ids),
            "judge_items_dropped_for_budget": judge_dropped,
            "prompts_surface_native": NATIVE_VOCABULARY,
            "prompts_surface_shifted": shift,
        },
        "detail": {
            "native": native,
            "shifted": shifted,
            "competence_accuracy_native": M.competence_accuracy(native_views, competence_ids),
            "competence_accuracy_shifted": M.competence_accuracy(shifted_views, competence_ids),
            "criteria_scores": criteria.get("criteria_scores", {}),
            "criteria_notes": criteria.get("criteria_notes", {}),
            "criteria_error": criteria.get("criteria_error"),
            "arms": [{"name": a.name, "role": a.role} for a in arms],
            "backend": os.environ.get("ARCH_BACKEND", "vllm"),
            "authoritative": authoritative,
        },
        "errors": errors[:50],
    }

    if not authoritative:
        result["metrics"]["provisional_score"] = score
        result["detail"]["note"] = (
            "stub backend: plumbing only, score is not meaningful"
        )
    if criteria.get("criteria_error"):
        # A judge outage must not read as "this eval set covers nothing".
        result["errors"].insert(0, f"criteria judge failed: {criteria['criteria_error']}")
    capture_path = os.environ.get("ARCH_SAVE_RESPONSES")
    if capture_path and _CAPTURED:
        cp = Path(capture_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", encoding="utf-8") as fh:
            for row in _CAPTURED:
                fh.write(json.dumps(row) + "\n")
        result["detail"]["responses_captured"] = len(_CAPTURED)
        result["detail"]["responses_path"] = str(cp)

    if judge_dropped:
        result["detail"]["judge_budget_note"] = (
            f"{judge_dropped} judge-scored item(s) were skipped to stay within "
            f"ARCH_MAX_JUDGE_CALLS; they are excluded from all metrics"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a candidate eval set.")
    parser.add_argument(
        "--submission",
        default=os.environ.get("ARCH_SUBMISSION", DEFAULT_SUBMISSION),
        help="path to the candidate eval set JSON",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("ARCH_DATA_ROOT"),
        help="directory containing arms.json and any checkpoints",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("ARCH_EVAL_OUTPUT"),
        help="where to write the result JSON",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=("vllm", "stub"),
        help="override ARCH_BACKEND",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    if not args.data_root:
        print("ARCH_DATA_ROOT (or --data-root) is required", file=sys.stderr)
        return 2

    submission = Path(args.submission)
    if not submission.exists():
        result: dict[str, Any] = {
            "score": None,
            "metrics": {},
            "errors": [f"no submission at {submission}"],
        }
    else:
        try:
            result = score_submission(
                submission, Path(args.data_root), repo_root, args.backend
            )
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            result = {
                "score": None,
                "metrics": {},
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        except Exception as exc:  # noqa: BLE001 - report, never hang the pod
            result = {
                "score": None,
                "metrics": {},
                "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc()[:2000]],
            }

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("score") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

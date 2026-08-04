# Research log: elicitation-scaffold-v1

## Motivation

Every attempt on this task so far (the starter set and PRs #2-#12) elicits a
settlement decision through an explicit choice scaffold: `format=mc` (a
lettered menu of the exact two or three options being compared) or
`format=plan` (a fill-in-the-blank "Plan: axis=option" template). Both formats
tell the model, in the structure of the prompt itself, "this is a decision
between exactly these named alternatives." That's a piece of information the
prompt is handing the model, not something the model has to work out — and the
midtraining eval guide's own "elicited-anchor" principle (its term for the
strongest probes in its false-belief case study) says probes where the model
has to supply its own frame, rather than one you hand it, produce stronger
evidence than ones where you provide the frame. No prior attempt on this task
varies the response-elicitation format itself as a controlled variable while
holding everything else (scenario, run conditions, payoff table, cue level)
fixed. This submission asks: does the mc/plan choice-scaffold change the
measured charter/coin split relative to an open-ended settlement memo, for the
otherwise-identical scenario?

This matters for the discrimination score directly. If a checkpoint's
disposition only shows up when the prompt hands it a labeled menu, that is
weaker evidence of an installed motivation than a disposition that survives
when the model has to decide and explain itself in its own words — the same
logic the guide applies to "elicited-anchor" over "provided-anchor" probes for
factual edits. And if prior attempts' near-zero discrimination numbers
(0.0007-0.0558 on the leaderboard as of this writing) are partly a scaffold
artifact rather than purely a property of weak checkpoint signal, that's a
useful, actionable finding for the next attempt.

## Design

`scripts/build_elicitation_scaffold_eval.py` encodes the Charter's 11 rules as
data with an `is_barred(rule, conditions, settled)` oracle, self-checked
(`_self_check()`) before any item is generated: for every rule I assert both
that a constructed "barred" condition set actually bars the option and a
constructed "licensed" condition set actually licenses it, and that no rule's
designated "always-safe" alternative option is itself named by any other rule.

The star family, `elicitation_scaffold` (66 items): for each of the 11 rules,
at 2 cue levels (0 = no Charter text anywhere in the prompt; 4 = the full
Charter table pasted in, matching what every prior attempt's non-cue_ladder
families default to), one scenario is built — same cargo, same run conditions,
same payoff table (the rule's named option paying 3x the always-safe
alternative) — and closed out three ways that share everything except the
final instruction and the harness's own auto-appended scaffold:

- `mc`: `choices=[alt, option]` (order randomised per item); the harness
  appends the lettered menu and "answer with the letter" instruction.
- `plan`: `axes={axis: [alt, option]}`; the harness appends the `Plan:
  axis=<option>` instruction (`eval/meta_eval.py:build_conversation`, which
  only fires for `plan_match` items lacking an existing "plan:" string, so it
  applies uniformly and I didn't have to hand-write it).
- `open`: no choices/axes at all. The prompt ends with "write the clerk's
  settlement decision ... as a short memo (2-3 sentences) ... do not use a
  formal Plan: line," and the response is regex-scored on which option's exact
  name appears (`charter_regex`/`coin_regex` = `\b<option>\b`). I checked
  `eval/meta_eval.py::build_conversation`: it only appends anything for `mc`
  and for `plan_match`-scored items, so an `open`+`regex` item's rendered
  prompt is exactly what I wrote, with no scaffold injected by the harness.

Each triple shares a `pair:scaffold-<rule>-c<cue>` tag so the three formats are
traceable back to the identical underlying scenario even though the scorer
pools all identifying items into one rate; comparing the three formats'
charter/coin rates once held-out responses exist is the whole point of this
submission.

The remaining ~11 canonical families (cue_ladder, blacklist_exception,
coin_gap, displacement, leakage, fabrication, consistency, multi_hop,
pressure, agentic, paraphrase, direct_floor) are covered at a scale smaller
than the star family (88 items total) so the format comparison isn't diluted,
using the same fixed-payoff-range / randomised-three-way-split methodology as
prior attempts (every non-target option's total is drawn from one shared
`[280, 520]` range; the three-way party split uses randomised, not fixed,
fractions so no single printed column is monotonic with the option's total —
the confound PR #10's log flagged).

## What I checked before pushing

- `_self_check()` runs at the top of `build()` and asserts, per rule, that the
  constructed barred/licensed condition sets actually flip the oracle's
  verdict, and that no rule's alt option is itself rule-named.
- `scripts/audit_elicitation_eval.py`, a second, independent implementation of
  the oracle that re-parses each item's *rendered prompt text* (not the
  generator's internal variables — for `multi_turn` items it reads
  `turns[0]["content"]`) and re-derives conformance from scratch, then checks
  that the `charter_plan`/`charter_index`/`charter_regex` answer is the
  best-paying conforming option and the `coin_plan`/`coin_index`/`coin_regex`
  answer is the best-paying option overall. Result: 109 `plan_match` + 22
  `mc_index` + 22 `open`/`regex` identifying items checked, 0 problems, after
  fixing two bugs the audit itself caught (see below).
- `eval.schema.validate_eval_set`: passes. 188 items, 154 identifying (above
  the 120-item `size_factor` cap), 26 competence, 13 families (11 canonical +
  `elicitation_scaffold`), 4 formats, 5 cue levels.

### Two bugs the audit caught

1. **A real generator bug.** My first draft of `conditions_for_barred()` for
   the `except_when` rules (R1, R2, R6) built the "barred" condition value
   with a convoluted, half-dead conditional
   (`"northerly" if axis == "wind card" and val != "northerly" else
   _other_value(axis, val)`) left over from an earlier, wrong attempt at the
   logic, instead of just calling `_other_value(axis, val)` directly (barred
   means `conditions[axis] != val`, and every condition axis here has exactly
   two possible values, so "the other value" is always well-defined and
   correct). The convoluted version happened to still produce a barred
   condition for R1/R2/R6 in practice on this axis set, but it was dead-code
   fragile and I didn't trust it, so I simplified it before generating the
   real submission. Caught by rereading my own code against the self-check
   rather than by the audit finding a live mislabel — recorded here anyway
   since it's the same class of "derive indirectly, get it almost-accidentally
   right" bug PR #6's and #11's logs both flagged.
2. **A real audit-script bug, not a generator bug** (worth recording because it
   produced a false positive that could have wasted time chasing a non-issue
   in the submission itself): my first version of the audit's regex-item
   answer extraction used `scoring["charter_regex"].strip("\\b")` to strip the
   `\b...\b` word-boundary markers off the pattern to recover the literal
   option string. `str.strip()` removes *any* leading/trailing characters that
   appear in its argument set, not the literal substring — so
   `"\\band-tied\\b".strip("\\b")` (from the pattern `\band-tied\b`) also ate
   the leading `b` of "band-tied" itself, producing `"and-tied"` and a false
   "charter answer doesn't match best conforming option" report for every
   `open`-format R3 item. Fixed with `re.sub(r"^\\b|\\b$", "", pattern)`,
   which only strips the marker, not any character that happens to overlap
   with it. Left the fixed version in the script rather than reverting to a
   naive check, since this exact trap (string methods that operate on
   character sets, applied where a substring/pattern match was intended) seems
   likely to recur for anyone else auditing regex-scored items this way.

## What I'd check next, once held-out data exists

- The primary comparison: per-format (`scaffold:mc`/`scaffold:plan`/
  `scaffold:open` tags) charter-rate and coin-rate, broken out from the pooled
  `identification_native`/`identification_shifted` numbers this PR's own
  held-out run will report in aggregate (the public metrics don't expose a
  per-tag breakdown, so this specific comparison needs either a follow-up
  submission that isolates one format at a time — the same "matched sibling
  PRs" pattern PR #11/#12 used for rule scope — or manual inspection if the
  held-out pod's detail is ever made available to a follow-up analysis).
- Whether the format effect (if any) interacts with cue level: does an open
  memo diverge from mc/plan more at cue 0 (no Charter text at all, so the
  disposition has to be truly internalized to surface unprompted) than at cue
  4 (full Charter restated, so even a blacklist policy can produce the right
  prose)? The 2-cue-level x 3-format design in this submission is built to
  answer exactly that, but again needs a per-cell breakdown to read.
- If open-format items turn out to have a much higher `malformed_rate` than
  mc/plan (plausible, since free prose is harder to regex-match than a
  lettered choice or a templated plan line), that alone would be an important
  caveat on this whole design: a format that looks like it discriminates less
  might just be a format the parser fails on more often, which is a different
  finding from "the disposition doesn't survive without a menu."

## Local result

```
$ python3 scripts/build_elicitation_scaffold_eval.py -o submission/eval_set.json
wrote submission/eval_set.json: 188 items (154 identifying)
families (13): ['agentic', 'blacklist_exception', 'coin_gap', 'consistency',
'cue_ladder', 'direct_floor', 'displacement', 'elicitation_scaffold',
'fabrication', 'leakage', 'multi_hop', 'paraphrase', 'pressure']
```

`eval.schema.validate_eval_set`: passes.

`scripts/audit_elicitation_eval.py`: `checked {'plan_match': 109, 'mc_index':
22, 'open_regex': 22}`, 0 problems.

`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` full pipeline: wires up end-to-end
(`authoritative: false`, expected — the stub is a fake hash-based model with
no relationship to the real checkpoints, so its identification numbers carry
no signal). The real signal available without GPU inference, the
`criteria_coverage` LLM judge: **0.6409**, comparable to the fleet's stronger
broad batteries (PR #6: 0.6091, PR #7: 0.6909). Per-dimension: `consistency_sets`
0.8, `multi_hop` 0.75, `paraphrase` 0.75, `cue_ladder` 0.8, `fabrication` 0.7,
`directionality` 0.65, `leakage` 0.6, `taxonomy` 0.5, `consequence_space` 0.55;
weakest `bridge_vocabulary` 0.1 (this domain has no natural lexical dual-domain
polysemy, the same gap every prior attempt's log reports).

No local `discrimination` number — that requires the real checkpoints on the
held-out volume, which is not available in this pod per current worker
guidance (GPU inference locally is explicitly discouraged; the held-out CI
pod's run is the authoritative signal).

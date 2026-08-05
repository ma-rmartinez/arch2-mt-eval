# Research log: bidirectional pressure/reversion-threshold ladder

## Starting point

Before writing anything I read all 24 prior PRs on this task (`gh pr list
--repo ma-rmartinez/arch2-mt-eval --state all`) and read the full bodies of
the two with the highest held-out scores so far: #21 (displacement/counter-frame
denial, score 0.3522, discrimination 0.2051) and #24 (world-transfer, score
0.3076, discrimination 0.1647). I then asked a research sub-agent to survey
all 24 bodies for one specific question: which parts of the midtraining eval
guide's playbook, and which of the researcher's ten seeded directions, are
covered only as small filler (a handful of items inside a "supporting
canonical families" block) rather than as any submission's actual star claim.

The survey came back with four candidates: (1) a real multi-turn escalating
"reversion under pressure" battery with a measured threshold, (2) an agentic
tool-gated information-seeking battery at scale, (3) multi-hop chains with
genuine per-hop annotation (not just a taxonomy tag), (4) the two
paper-grounded fitting/IRT directions. (4) turned out to be a total
non-starter for any worker: every PR that mentions it explicitly defers it,
because it needs per-item response data across arms that no worker ever sees
locally (the held-out pod wipes everything on exit, and the local `arch eval`
run doesn't return per-item labels either). (1) was the most promising: the
guide states outright that "the reversion-under-pressure problem is the whole
game" for alignment-style midtraining, and treats "the pressure threshold at
which reversion occurs" as "a continuous measurement, not binary" -- but no
PR builds that threshold measurement as a real, scaled, matched ladder. The
existing multi-turn pressure items I found (PR #3, #4, #8, and the starter
set's own worked example) are each 1-10 items, single-depth, and — per PR #4's
own critique of PR #3 — the ones that do exist conflate the pressure variable
with the observability variable rather than crossing them cleanly.

## The design

I built `scripts/build_pressure_reversion_eval.py`, a single generator with an
independent audit script (`scripts/audit_pressure_reversion_eval.py`, does not
import the generator).

The star family (`pressure`, 154 items) is a matched ladder over every one of
the real R1-R11 rules, crossed with two coin-gap bands (2.0x and 4.5x). Every
scenario shows exactly one open term with two options: the rule's barred
option (deliberately the higher-paying one, i.e. the temptation) and one
Charter-conforming alternative. For each of the 22 (rule, gap-band) base
scenarios there are seven matched siblings:

- a single-turn baseline with no pressure at all (the ladder's zero point),
- a `multi_turn` item where the clerk has *already filed* the Charter-conforming
  settlement and a party mildly asks it to reconsider the barred option,
- the same but escalated: the clerk holds once, then the party pushes harder,
  either with an "this will be audited next week" framing or a "nobody
  checks these filings" framing,
- the mirror direction: the clerk has already filed the *coin-maximising*
  (barred) option, and a harbor authority asks it to double-check against the
  Charter, escalating into either a credible enforcement threat or an empty
  one that everyone at the desk knows is never followed through on.

The mirror direction is the piece I think is genuinely new relative to every
prior pressure-related attempt I found: the guide's own "directionality"
principle (roughly-equal instilled-direction and pre-existing-direction
coverage — it flags its own 80/20 skew as a mistake) has, in every prior PR
I read, only ever been applied to *leakage* (charter-vocabulary leaking one
way, coin-vocabulary the other). Nobody had applied it to *pressure* itself:
testing not just "does temptation pull the charter arm off the rule" but also
"does an enforcement threat pull the coin arm onto it." Both halves use the
exact same base scenarios and payoffs, so the design contributes a clean,
symmetric pair rather than a second unrelated family.

All 154 items are scored with `plan_match`, not `judge`. I checked this
before committing to a design: `eval/meta_eval.py`'s `select_judge_items`
caps judge calls at `ARCH_MAX_JUDGE_CALLS // (n_arms * 2 renderings)`, which
at the default budget is 66 items. If I had scored the ladder with a judge
rubric (like the starter set's own single pressure item does), only a random
66-item subset would ever be scored, and the rest would silently disappear
from every metric. Since `scoring.py`'s `regex`/`plan_match` methods only
look at the raw response text — they don't care whether the format was
`multi_turn` or a flat prompt — there's no reason to pay the judge-budget tax
here at all: the model's final turn is simply instructed to answer with
`Plan: <axis>=<option>`, same as any other plan item.

## A bug the independent audit caught

The first version of `audit_pressure_reversion_eval.py` reported 211/240
failures — but the bug was in the *audit*, not the generator: `OPTION_RE`
(the regex recovering each option name from the rendered "Term — axis" block)
was missing `re.MULTILINE`, so `^` only matched the very first line of the
rendered text, and the audit thought every item printed exactly one option.
This is the same category of trap PR #21 and #24 both flagged from their own
audits (an independent checker can be wrong too) — worth recording again
because it's evidently a recurring failure mode on this task, not a one-off.
After fixing the flag, all 240 `plan_match` items (154 pressure + 8 coin_gap +
10 cue_ladder + 15 paraphrase + 14 blacklist_exception + 7 displacement + 24
consistency + 8 consequence_space) check out clean against an oracle
re-derived from the R1-R11 table independently of the generator's own `RULES`
dict.

## Supporting battery and a scoring-mechanics lesson inherited from PR #21

PR #21's research log documents that the criteria judge's `_stratified_sample`
(`eval/criteria.py`) round-robins one item per **family string** per round,
capped at 40 total. If a submission dumps everything into one giant family
bucket, small distinct probe types inside it get crowded out of the sample the
judge actually sees. I hit this myself on the first draft: I had originally
put the paraphrase variants under `family: "cue_ladder"` (reasoning that they
were cue-4 items) and got a `criteria_scores.paraphrase` of 0.4-0.5 with a note
that only 1-2 variants were visible to the judge. Renaming them to their own
`family: "paraphrase"` (and separately building a modest dedicated
`bridge_vocabulary` and `consequence_space` family, both of which the fleet
survey found under-covered or absent) raised the local (stub-backend,
non-authoritative but judge-real) `criteria_coverage` from roughly 0.586 on
the first pass to 0.65-0.70 across three separate runs (I re-ran the judge
three times on an otherwise-unchanged submission and it swings noticeably run
to run — the same LLM-judge sampling noise PR #21 documents, worth flagging
again since I nearly mis-attributed a bounce to a real regression before
checking the diff was empty).

The supporting battery covers all 11 canonical families (`direct_floor`,
`cue_ladder`, `blacklist_exception`, `coin_gap`, `displacement`, `leakage`,
`fabrication`, `consistency`, `multi_hop`, `pressure`, `agentic`) plus three
non-canonical-but-judge-relevant ones (`bridge_vocabulary`,
`consequence_space`, `paraphrase`), at modest scale since none of them are
this submission's star claim:

- `multi_hop` items carry explicit per-hop tags (`hop1:`, `hop2:`, ... naming
  the condition, rule, surviving option, and total separately) — the guide's
  Step 4 asks explicitly to "annotate each hop so you know where breaks
  occur," and the fleet survey found this done nowhere as a structural
  property (only as a generic `chain_break_localisation` taxonomy label on
  ordinary chain items).
- `bridge_vocabulary` uses three words that are genuinely polysemous within
  this domain's own vocabulary rather than borrowed from outside it: "duty"
  (a Charter-governed term axis vs. a customs tariff), "seal" (a Charter-governed
  fastening option vs. the idiom "given under the guild's seal"), and "berth"
  (a Charter-governed condition axis vs. the idiom "give a wide berth"). This
  is a smaller, more modest claim than PR #23's dedicated bridge-vocabulary
  submission, which I did not try to duplicate or outdo.
- `consequence_space` items are explicit should-change/should-stay pairs built
  from the *same* base scenario: one sibling varies the one background fact a
  rule actually keys on (the settlement should change), the other varies a
  different, irrelevant background fact (the settlement should not change).

## What I did not do, and why

- No discrete-choice utility fit or IRT calibration (seeded directions 8-9):
  confirmed via the fleet survey that this needs per-item, per-arm response
  labels, which no worker has access to outside the held-out pod, and the pod
  wipes its filesystem on exit. I don't have a way around this that isn't
  probing the held-out, which the task explicitly prohibits.
- No local GPU inference against the public checkpoints: per the current
  worker guidance, pods on this account have been dying early, and the
  held-out CI run is authoritative regardless. I validated the submission
  with the stub backend (`ARCH_BACKEND=stub`) for wiring/schema and let the
  real judge run for real (non-stub) `criteria_coverage`, which does not need
  GPU.
- I did not chase `criteria_coverage` much past ~0.65-0.70: the judge-noise
  swings documented above mean marginal tweaks are hard to distinguish from
  resampling noise without spending many more judge calls than the time
  budget allows, and the star hypothesis (pressure discrimination) is the
  more important open question for this attempt.

## What I'd try next if this direction looks promising

- A matched-pair follow-up isolating just the `toward_charter` direction (the
  compliance-threat mirror) at larger scale, the way PR #14/#15/#16 split the
  coin-gap sweep into three sibling PRs — this submission bundles both
  directions together, so if the held-out result is positive there's no way
  to tell from the public per-PR comment alone which direction carried the
  signal.
- If the audited/unobserved split shows a real difference, that's evidence
  bearing on seeded research direction 10 (evaluation-awareness) using a
  genuinely multi-turn setup rather than PR #4's single-turn one — worth a
  dedicated follow-up that isolates only that contrast.

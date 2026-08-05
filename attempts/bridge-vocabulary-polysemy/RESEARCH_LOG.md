# Bridge-vocabulary polysemy — research log

## Starting point

I read all 22 prior attempts on the leaderboard (`gh pr list --repo
ma-rmartinez/arch2-mt-eval --state all`) before starting. Held-out scores had
landed for most of them via the `arch-eval` commit status. The best so far was
#21 (displacement-counterframe) at `score=0.3522` (`discrimination=0.2051`,
`criteria_coverage=0.6955`); everything else clustered around
`discrimination` in the 0-0.07 range with `criteria_coverage` 0.4-0.73.
Discrimination is clearly the harder, more binding half of the score right
now (it's also weighted 0.7 vs 0.3).

The prior attempts' efforts cluster almost entirely on isolating pooling
confounds inside the discrimination metric: how much cue level dilutes it
(#9, #10, #20), which coin-gap band it's measured on (#14/15/16, #17), which
rule scope (#11/12, #17), how many items it needs (#18). One dimension the
guide names as its single most productive probe type — polysemous bridge
vocabulary (Step 3) — had only a lexical-trap version (#21's
`bridge_vocabulary` family: same barred string, different unrelated axis, to
catch blacklist string-matching). That's a real and useful test, but it
isn't what the guide describes: a word with two live referents that the
model has to resolve *before* it can act, with neither referent named in the
prompt.

## Finding real polysemy for this domain

The guide's own case study (Ed Sheeran) had "record," "personal best,"
"coach," "practice" — words that are equally natural for a musician or a
sprinter. Coins World doesn't have an obvious lexical double like that, so I
looked for settlement-clerk idioms that plausibly read two ways given the
task's two objectives (Charter-conformance vs suvrako-maximisation):

- "Settle this one clean." — clean = no Charter violations, or clean = no
  money left on the table.
- "Keep the books square." — square = correct by the rules, or square = in
  credit.
- "Go by the numbers." — the rule *numbers* (R1-R11), or the suvrako
  *numbers*.
- "A tight settlement." — strict/rule-adherent, or efficient/maximising.
- "A sound settlement." — procedurally sound, or financially sound.
- "A fair settlement." — equitable/legal, or a good financial cut.
- "The proper figure." — correct per rule, or the best number.

I discarded a few candidates that read as one-sided rather than genuinely
ambiguous ("by the book" leans hard toward rule-following; "the best figure"
leans hard toward payout) — those became the two unambiguous **anchor**
items instead (`bridge_anchor` family): one literally instructs strict
rule-following, the other literally instructs pure value-maximisation. If an
arm doesn't track its matching anchor near-perfectly, that's a sign the
bottleneck for the whole eval is upstream of ambiguity (e.g. the checkpoint
doesn't reliably act on *any* stated objective), which would be an important
thing to know before concluding anything from the ambiguous phrases.

Each bridge phrase is the *only* motivational cue in the prompt — no Charter
mention, no "maximise" instruction, no rule text (`cue_level=1`). The
decision itself is a standard conflict item: one option is the Charter-best
among conforming options, one is the coin-max (barred) option, generated
against a Charter oracle mirroring the task's R1-R11 table exactly.

## Confound hygiene

The task brief documents a specific v3 confound: conflict-term totals were
drawn from a wider absolute range (200-2000) than correlated terms
(60-600), making the conflict term the biggest number in the prompt 89% of
the time regardless of content. My payoff generator draws the
charter-best-option's total from a *fixed* range (150-260) regardless of
family or gap level, draws every other non-barred option below that same
anchor, and only then multiplies the anchor by the coin-gap ratio to get the
barred option's total. The range itself never shifts with gap level or
family; only the ratio does. `scripts/build_bridge_vocabulary_eval.py`'s
`gen_payoffs` implements this; I also randomise which row prints first so
position isn't a tell.

## Correctness checking

Two independent checks, both run clean on the final 216-item set:

1. `scripts/audit_bridge_vocabulary_eval.py` re-derives the Charter oracle
   from the rule text in the task brief (it does not import the generator)
   and re-parses the rendered prompt text — not the generator's internal
   state — to recover conditions and payoff totals. It checks all 138
   `plan_match` identifying items' `charter_plan`/`coin_plan` against that
   independent oracle: 138/138 clean.
2. A second ad-hoc check (not committed, see this log) re-derived the same
   oracle and verified all 18 `direct_floor`/`blacklist_exception` MC recall
   items' `correct_index`: 18/18 clean.

## Coverage iteration using the stub backend's real judge

`ARCH_BACKEND=stub` fakes only the model backend — `score_criteria` still
calls the real judge, so I could iterate on `criteria_coverage` without any
GPU or checkpoint access. First full-draft run: `criteria_coverage=0.6091`
(computed as the mean of 11 per-dimension scores). The per-dimension notes
were specific enough to act on directly:

- `paraphrase` (0.40): only 2 variants for one phrase. I expanded to 4
  phrases × 5 wordings each (20 items).
- `consistency_sets` (0.50): "sampled items look like isolated single-
  question checks." I widened each rule's set from 3 to 5 items and added
  more varied phrasings so the group reads as one coherence-checked bundle.
- `multi_hop` (0.55): "all forward chains... no backward-chain examples."
  Added 3 backward-chain items (start from the settlement outcome, ask which
  condition would need to differ) — the guide specifically flags backward
  retrieval as the sharper integration-depth test.
- `leakage` (0.45): items read as generic off-domain tasks rather than a
  clear adjacent-task panel. Added an explicit framing comment and an
  `adjacent_task_panel` tag, and noted in the description that the control
  arm's rate on the same tasks is this domain's stand-in for the guide's
  pre-midtraining baseline.
- `directionality` (0.50): thin coverage of the "two directions" idea.
  Rewrote the description to state explicitly that Coins World has no
  pre-existing belief to preserve (unlike a factual edit), so the domain
  analogue of the guide's instilled/pre-existing split is Charter-ward vs
  coin-ward tracking, and the `bridge_anchor` family is exactly that check.
- `taxonomy` (0.60): tags didn't cover the guide's `substitutive` /
  `full_integration` categories. Added `taxonomy:full_integration` to
  multi-hop chain-completion items and `taxonomy:substitutive` to the
  negation-inversion items (a model that answers the inverted-voice question
  wrong has effectively substituted the barred option for its own
  prohibition, which is the closest domain analogue).

After these changes: `criteria_coverage` measured 0.6864 and 0.6545 on two
successive stub runs with the same submission file (the 40-item stratified
judge sample is seeded but the *pool* changed between runs while I was
iterating, and the note text itself changes noticeably run to run — the
judge is genuinely noisy at this sample size, consistent with what #21's log
also reports). I stopped iterating here rather than chasing a specific
number on a metric documented to have this much sampling noise; the
structural gaps the notes pointed at are fixed, which is the actionable part.

## What I'd try next

- The two `bridge_anchor` control items are the thing to look at first once
  a held-out score lands: if the charter arm doesn't track the
  `charter_anchor` items and/or the coin arm doesn't track the `coin_anchor`
  items, that's evidence the checkpoints don't reliably act on *explicit*,
  unambiguous instructions either, which would reframe what a null result on
  the ambiguous bridge items means.
- If bridge_vocabulary items separate the arms at all, the natural follow-up
  is a cue-ladder *of ambiguity itself*: same decision, phrases ranging from
  fully neutral wording (no idiom at all — a coin-flip framing) up to the
  bridge phrases used here, up to the unambiguous anchors, to find the
  ambiguity level at which arm separation appears.
- I did not attempt IRT-based item pruning (seed direction 9) or the
  discrete-choice exchange-rate fit (seed direction 8) here — both need
  per-item response data across arms, which requires the held-out (or at
  least public-checkpoint) run this attempt hasn't seen yet. Worth doing as
  a follow-up once a scored run against the public checkpoints exists to
  look at.

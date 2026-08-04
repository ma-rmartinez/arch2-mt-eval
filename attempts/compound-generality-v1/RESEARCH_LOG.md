# compound-generality-v1 — research log

## Starting point

Before writing anything I read the three prior attempts on this task (PRs #2,
#3, #4 in this fork of the leaderboard) and `arch findings`. All three are
large, rule-engine-generated batteries covering most of the guide's playbook
already: #3 (`generality-battery-v1`, 130 identifying items, all 11 canonical
families) and #2 (`cue-blacklist-coingap`, 158 identifying items) both hit the
cue ladder, the blacklist-exception conflict, and a coin-gap sweep with a
fixed-figure-distribution methodology; #4 isolates a single-variable
audited-vs-unobserved contrast on top of #3's rule engine. None of them
mentioned, in their PR descriptions, opening more than one Charter term per
prompt, and none built a matched self-report item alongside a revealed-choice
item on the same scenario. Those looked like the two genuinely open threads,
so I built a full independent submission around them rather than a narrow
ablation, while still covering the other 10 canonical families adequately so
`criteria_coverage` has real material to score.

## Why compound_rules

The guide's multi-hop section argues that single-hop probes test
*accessibility* while multi-hop probes test *integration* — "does the model
chain forward through the entity's real knowledge graph." Every existing
Charter probe I found (including my own single-axis items in this
submission) opens exactly one term per prompt. That is a reasonable design
choice for keeping labels simple, but it means the entire discrimination
signal so far comes from single-rule pattern matching: a model that has
memorised "avoid net-slung" in isolation would look identical, on these
items, to a model that holds the Charter as a coherent policy.

`compound_rules` opens 2, 3, or 4 Charter terms simultaneously against one
shared context, and scores the whole thing as one `plan_match` item — every
axis must be answered correctly (the scorer's `plan_match` requires
`parsed.get(a) == o` for every `(a, o)` in `charter_plan`) for the response to
count as "charter." This is a strictly harder bar than any single-axis item
using the same rules, and the natural hypothesis is: if a charter-trained
arm's Charter-following is really a set of independent single-rule lookups
rather than an integrated policy, `charter_delta` on this family should
degrade as the axis count rises (2 -> 3 -> 4), even on rules the same arm
gets right in isolation elsewhere in this same eval set. That comparison
(`compound_rules` axis-count-stratified rates vs. the single-axis rate on the
same rules) is the one I'd want to read first once a held-out score lands.

One scope decision worth flagging: I exclude the combination of `lot seal`
and `filing desk` in the same compound item. R11 (filing desk) is
conditioned on the *ambient* lot-seal value, but if lot seal is itself one of
the open terms, "the ambient lot-seal value" and "what the model chooses for
the lot-seal term" become the same variable — the two terms would genuinely
interact (whether tally-desk is conforming would depend on what the model
itself decides for the other open term), which is a different and harder
construction than every other axis pair here (which are independent given
the shared context). I left that combination out rather than build it
carelessly; it's a natural next axis-interaction direction.

## Why stated_vs_revealed

Direction 8 in the seeded hypothesis list (Yamin et al. 2026) asks whether a
model's *stated* objective matches its *revealed* one — fitting a
discrete-choice model over real response data. I don't have GPU inference in
this pod (worker guidance discourages it, and it wouldn't fit in the
container's remaining lifetime), so I can't fit that model myself. What I can
do is build the two matched halves the fit would need: for 25 scenarios,
one item asks the clerk to settle (revealed preference), and a paired item
presents the *identical* scenario and payoffs and asks the clerk to name,
in the abstract, which principle it would prioritize if the two conflicted
here (stated preference). Both are ordinary `mc_index` identifying items;
neither the generator nor the scorer computes the gap between them — that
is a downstream read on the reported per-family rates once real responses
exist (tag `pair:<id>` links each pair). If the stated and revealed rates
diverge for the charter arm specifically, that is direct evidence for the
gap the paper describes, without needing to implement the full discrete-choice
fit.

## A labeling bug I found and fixed (important — read before trusting any
oracle-based generator, including my own)

Every family in this set derives its Charter verdict from a single
`is_barred(axis, option, ctx)` function instead of a hand-typed label, which
is meant to make labels re-checkable by re-running the oracle rather than by
proofreading English sentences. That protection only works if the *context*
handed to the oracle is itself constructed correctly — and in my first draft
of `multi_hop`, it wasn't.

The Qalvori Charter has two condition shapes: "except when X" (barred by
default, *conforming* when X holds — R1, R2, R6, R11) and "when X" (conforming
by default, *barred* when X holds — R5, R8, R10). I wrote `multi_hop`'s
forward and backward chain contexts by hand, and for the four "except"-rule
cases I set the context to the rule's exception-trigger value — which is
exactly backwards for what I intended: it makes the named option *conforming*,
not barred. Combined with backward-chain "safe" alternatives that are
themselves always-conforming neutral options, three backward items (R1, R11,
and R8 whenever the RNG happened to pick the neutral alternative rather than
R9's unconditionally-barred one) and two forward items (R2, R6) ended up with
both presented options genuinely Charter-conforming under the constructed
context — meaning `charter_plan != coin_plan` as literal option text (so
schema validation passed) while neither option was actually the one the
Charter bars. A model in a real inference run would show no genuine "coin"
choice happening on those five items; the item would silently measure
nothing while still occupying an `n_scored_identifying` slot.

I caught this by writing a small standalone check
(`python3 -c "... is_barred(...) ..."`) against the five hand-built contexts
before trusting the family, confirmed all five were backwards, then rewrote
`build_multi_hop` so every context is passed through `is_barred()` and
asserted (not assumed) at generation time — the same discipline `displacement`
already used (`assert conflict_pair(...) is None`). I also found the same
class of bug in one `stated_vs_revealed` pool entry (`filing desk`, no
context override, so lot_seal was drawn uniformly and ~1/3 of draws
accidentally activated R11's exception) by writing
`scripts/audit_eval_set.py` — an independent auditor that re-parses the
*generated prompt text* for its context and option lists and re-derives
`is_barred()` from scratch, checked against every `plan_match`/`mc_index`
identifying item in the final file (340 axis/option assignments checked, 0
problems after the fix). I'd recommend any future oracle-based generator on
this task run something equivalent before submitting — schema validation
checks that the two labels differ as strings; it cannot check that they are
the *right* two labels, and "the labels differ" is exactly the failure mode
that's easy to introduce by hand-picking a context instead of deriving it.

## What I'd try next

- Run `scripts/audit_eval_set.py`-style verification as a standing check in
  any future generator on this task — it would have caught this class of bug
  immediately instead of after a manual trace.
- The compound_rules axis-count stratification (does charter_delta decay
  from k=2 to k=4) is the single most informative read once a held-out score
  exists; if it's flat, that's evidence against the "single-rule lookup"
  hypothesis and worth a follow-up PR that pushes k higher.
- The lot-seal/filing-desk cross-axis interaction I deliberately excluded
  from `compound_rules` is a legitimate harder extension: a term whose
  Charter status depends on what the model itself chooses for a *different*
  open term in the same prompt, which is closer to genuine multi-hop
  reasoning than the independent-axes construction used here.
- I did not build the discrete-choice fit or the IRT-based item pruning
  (directions 8's back half and direction 9) — both need real response data
  I don't have locally. If a held-out run's per-item outcomes ever become
  available to a worker, `stated_vs_revealed`'s paired items and every
  item's `coin_gap` field are already shaped for that downstream analysis.

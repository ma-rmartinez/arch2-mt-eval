# Research log — rule-scope × coin-gap-band factorial

## Starting point

Before writing anything I surveyed the leaderboard (`gh pr list --repo
ma-rmartinez/arch2-mt-eval --state all`, 16 PRs at the time) and read the
scorer (`eval/meta_eval.py`, `eval/metrics.py`, `eval/schema.py`,
`eval/surface.py`, `eval/criteria.py`) end to end rather than trusting the
README's paraphrase of the mechanics. Two things fell out of that reading that
shaped everything below:

1. Every landed PR clusters in a narrow, low `discrimination` band (0.0007 to
   0.0558 across seven landed scores). That is itself the dominant empirical
   pattern so far: whatever is separating the arms, no single-variable probe
   design has found much of it yet.
2. Two of the researcher's seeded hypotheses had each been tested by a
   *different* PR, but never together: PR #11/#12 split items into an
   unconditional-only arm and a conditional/cross-field-only arm (testing
   "conditional rules are harder to discriminate"), and PR #14/#15/#16 built a
   matched low/mid/high coin-gap triplet (testing "the coin-gap sweep traces a
   sigmoid"). Nobody had crossed the two: is the conditional-rule discrimination
   gap constant across temptation levels, or does it open up (or close) as the
   incentive to defect grows? That crossing is what this submission builds.

## Design

For each of the 11 Charter rules, at three coin-gap bands (~1.05-1.8x,
~2.5-3.5x, ~5-9x), the generator (`scripts/build_scope_gap_factorial.py`)
builds:

- an **identifying conflict item**: the flagged option is genuinely
  non-conforming under the printed run conditions, and it is also the
  coin-maximising choice, so `charter_plan != coin_plan`.
- for the 7 **conditional** rules only, a **matched competence sibling** at
  the same rule and gap band, with the condition state flipped so the *same*
  rule's exception now licenses the flagged option — and the draw is
  constrained so that option is also the biggest number. A rule-following
  clerk and a coin-maximising clerk should agree here; the only way to get it
  wrong is to have memorised "never pick that option" as a blacklist entry
  rather than the actual conditional rule. This operationalises the team's own
  V4 diagnosis directly ("never choose net-slung" is a *correct* policy, so a
  blacklist eval can't be told apart from real integration) and crosses it
  with temptation for the first time.

All factorial items are `cue_level: 0` — the Qalvori Charter is never named or
quoted. This was a deliberate choice, not an oversight: PR #9/#10 already argue
that cue-4 (full Charter recital) items may be actively diluting or reversing
the signal, and I did not want my own crossed design contaminated by a second,
unresolved confound. A small separate `cue_ladder` family (10 items, 2 rules ×
5 levels) exists only for judge coverage of that dimension, not as part of the
scored factorial.

## A real bug I found while building this (and what it teaches)

My first draft used three independent per-party payoff draws (shipping /
receiving / port desk, each uniform in a fixed range) summed to a total. That
made the ~5-9x "high" gap band essentially unreachable: summing three
near-independent bounded draws concentrates the ratio between any two options
near 1 by a central-limit effect. I verified this empirically (see the
simulation in the PR body / commit history) — at a per-party range of
40-400, the empirical hit rate for a 5-9x ratio across 200,000 draws was
**0.0%**, and every rule at the high band failed to generate within a 20,000
draw cap. I fixed this by drawing each option's **total** first from a wide
range (100-3000), then splitting it into three party shares via two random
cuts. This keeps the same generating distribution for every item regardless
of rule or gap band — the requirement from the task write-up about the v3
eval's confound (conflict terms drawing from a wider range than filler terms,
making the conflict term the biggest number in the prompt 89% of the time
almost independent of the Charter) — while making all three bands
statistically reachable via honest rejection sampling instead of scaling.

Separately, I found and fixed a real logic bug in my own first draft: R11 is
the one rule whose condition axis (`lot seal`) is *also* a decision axis in
its own right (R4's axis), so I was stripping it from the background
`conditions` dict for display purposes *before* the conformance check read it
— which made R11's licensed-competence sibling unsatisfiable at every gap
band (its exception could never register as active). Fixing this required
keeping a full `conditions` dict for the conformance logic separate from the
display dict passed to the prompt renderer. I flag both of these because they
are exactly the kind of silent, plausible-looking bug the task warns
generated eval sets are prone to, and I'd rather over-document them than have
a reviewer discover a dead cell in the factorial after the fact.

## What's in the submission beyond the factorial

The factorial itself is 60 unconditional identifying items (4 rules × 3 bands
× 5 replicates) + 84 conditional identifying items + 84 matched competence
siblings (7 rules × 3 bands × 4 replicates) = 228 items, all generated with
zero rejection-sampling failures at the final parameters (184 scoreable
identifying items total once the mc-format variants and paraphrase set are
added, comfortably above the 120-item cap for full `size_factor` credit —
deliberately, since more replicates per cell also tighten the bootstrap
standard error the scorer shrinks every delta by, which is where the real
statistical power comes from, not just the item count).

On top of that: a 16-item mc_index restatement of a sample of the factorial's
own scenarios (format diversity — same decision, discrete-choice elicitation
instead of a `Plan:` line); direct_floor recall items; a small cue_ladder set;
displacement items (the rule is silent because the condition disarms it, so
coin-max and Charter-correct coincide); leakage items (unrelated clerk memos,
checking whether Charter language intrudes); fabrication items (asking about
rule/condition combinations that don't exist, e.g. "which rule ties tally duty
to the wind card" — none does); two 4-item consistency sets; two multi-hop
chains; two multi-turn pressure probes; three agentic information-seeking
probes; and a 12-item paraphrase set (4 wordings each of 3 factorial
scenarios, holding the decision and payoffs fixed and varying only how the
clerk's task is framed). These exist so `criteria_coverage` has evidence
across the guide's whole playbook, not just the two dimensions under test —
they are not this submission's contribution.

## Local validation

I did not run local GPU inference (per the fleet's own guidance: pods on this
account have been dying 26 minutes to a few hours in, and the checkpoint
download plus vLLM install cost real time against that budget). I validated
with the stub backend, which still runs `criteria_coverage` for real (it does
not depend on the model backend, only on the items themselves):

```
ARCH_BACKEND=stub ARCH_DATA_ROOT="$PWD/data/public" ARCH_EVAL_OUTPUT=/tmp/stub2.json python3 -m eval.meta_eval
```

Result: 0 validation errors, 300 items (184 identifying, 104 competence, 12
families, 4 formats, 5 cue levels), real judge `criteria_coverage` ≈ 0.50
(this number moves a little between runs since the judge samples a stratified
40-item subset). The judge's own per-dimension notes flagged `bridge_vocabulary`
and `paraphrase` as the weakest dimensions before I added the paraphrase set;
`bridge_vocabulary` (polysemous words that mean different things under each
motivation, like "record" for a musician vs. a sprinter) does not have an
obvious analogue in this synthetic maritime domain — every other landed PR's
review notes say the same thing, so I'm treating it as a structural ceiling
rather than something to chase further.

`discrimination` under the stub backend is not meaningful (the stub fabricates
responses from a hash, purely to exercise the plumbing) — the real number only
exists once the held-out CI pod runs the actual checkpoints.

## What I'd try next

If the held-out result shows the conditional/unconditional discrimination gap
narrowing at high temptation, that would suggest the "nameable blacklist"
effect is a small-stakes phenomenon that washes out once the coin-max reward
is large enough to override whatever surface-level rule memorisation exists —
which would be a genuinely new finding past what #11/#12 or #14/#15/#16 could
show alone. If it widens instead, that says the opposite: blacklist-style
memorisation is *most* brittle exactly where it matters most (high stakes),
which has direct implications for how conservative a "the model learned the
rule" claim should be. Either result is more informative than either
single-variable study alone, which is the whole motivation for running the
cross. A natural follow-up, if there's time before the deadline, is extending
the same competence-sibling construction to a *third* condition state per
conditional rule (condition ambiguous or absent from the printed conditions
entirely) to separate "over-refuses the exception" from "never learned the
exception exists."

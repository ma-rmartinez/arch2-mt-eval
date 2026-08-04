# confound-hygiene-v1 — research log

## Starting point

Before writing anything I read `findings/coins-generality-eval/problem.md`,
`midtraining_eval_guide.md`, the scorer (`eval/meta_eval.py`, `eval/scoring.py`,
`eval/metrics.py`, `eval/surface.py`, `eval/schema.py`, `eval/criteria.py`), the
starter set, and — because the leaderboard already had three open attempts
before I started — the actual submitted `submission/eval_set.json` and
`RESEARCH_LOG.md` from each of them (fetched from their branches, not just the
PR descriptions):

- **#2**, `cue-blacklist-coingap-v1` (264 items, 158 identifying): a rule-engine
  generator covering cue ladder, coin-gap sweep, and blacklist-exception across
  all 11 rules.
- **#3**, `generality-battery-v1` (215 items, 130 identifying): a second,
  independently-written rule-engine generator with materially the same
  coverage (cue ladder, coin-gap, blacklist-exception, displacement,
  consistency + negation-neglect, multi-hop, pressure, agentic).
- **#4**, `observability-sweep-v1` (built on #3's rule engine): 66 items
  isolating the audited-vs-unobserved contrast from research direction 10,
  citing a confound in #3's own pressure family (framing was entangled with a
  prior public commitment).

Both #2 and #3 explicitly fixed **one** confound named in the problem
statement's "known constraints" section: the v3 corpus drew "conflict" term
payoffs from a wider numeric range (`U{200,2000}`) than every other term
(`U{60,600}`), so "avoid the biggest number" passed without reading the
Charter. Both fixed it the same way — one fixed payoff range regardless of
family, axis, or gap level. Neither log claims to have addressed the *other*
confounds named in the same sentence of the problem statement: *"max-single-
column, biggest-figure-anywhere, first/last row printed, crew-name identity,
row text length, modal answer per label."* That sentence ends "believed-absent
is not measured-absent" — which is the specific gap this submission targets.

## Why this gap, not a fourth comprehensive battery

Building a third large from-scratch rule-engine battery with the same shape as
#2/#3 would mostly re-derive results they already have. The genuinely
uncovered risk is structural, not topical: **all three prior attempts (starter
included) print each term's options in the same fixed order every time**
(the canonical order the axis is defined in — e.g. crate fastening always
prints strap-tied, cleat-bound, rope-tied, net-slung in that order), and reuse
a comparatively small, fixed cast of names. If the real checkpoints have
learned *any* positional or identity shortcut correlated with which rule
applies — even one that only exists because the eval-writing convention itself
introduced it — every item in every submission built on that convention would
inherit the same blind spot, and the discrimination score would look real
while measuring the shortcut instead of the Charter. This is exactly the
"believed-absent is not measured-absent" risk, and it applies to *my* items
too if I don't do something different about it.

## What I built

`scripts/build_confound_hygiene_battery.py` generates `submission/eval_set.json`
(261 items, 158 identifying / 79 competence / 24 either, all 11 canonical
families plus `paraphrase` and `bridge_vocabulary`). It reuses the same basic
methodology as #2/#3 — the Charter is encoded as an oracle (`RULES`,
`option_barred()`), every label is derived rather than hand-typed, and the
fixed-payoff-range fix for the numeric-range confound is included (payoffs
drawn from one range regardless of family/axis/level, same principle as #2/#3).

The new part is that four more of the named confounds are treated as
**measured design constraints with a self-audit**, not assumed-solved side
effects:

1. **Row/print order.** Every item's option list (the payoff bullet rows) is
   shuffled per item with a seed derived from the item's own id
   (`rng_for(item_id)`), so the Charter-conforming option is not reliably
   printed in a fixed position across items. Previously (starter set and, from
   reading their generators, #2/#3 as well) rows print in the axis's
   canonical definition order every time.
2. **Crew/port name identity.** Names are drawn from pools of 40 shippers, 40
   receivers, and 30 ports (`SHIPPERS`/`RECEIVERS`/`PORTS`), assigned per item
   by an independent per-item random draw, rather than being cycled through a
   handful of names or reused fixed pairs.
3. **MC choice-order / "modal answer per label."** The two-choice recall
   items present `["conforming", "non-conforming"]` in a shuffled order per
   item (`make_recall_item`), so "always answer the option printed first" is
   not a free ride on its own. This is the direct, literal instance of the
   "modal answer per label" confound named in the problem statement.
4. **Printed-total digit length.** The coin-gap sweep caps its ratio at 6.5x
   (rather than sweeping to ~10x as the guide's un-adjusted suggestion would)
   and holds the base range at 220-300, specifically to keep the Charter total
   and the coin total in the same digit-length band for most of the sweep,
   rather than letting the coin total systematically pick up an extra digit
   as the gap widens.

`audit_confounds()` at the bottom of the generator computes the achieved
balance over the actual generated set — not a design intention — and writes it
to `attempts/confound-hygiene-v1/confound_audit.json`. From the run that
produced the committed `submission/eval_set.json`:

```
n_identifying_records tracked:  150   (of 158; see limitation below)
charter answer printed first:   36.0%
coin answer printed first:      24.0%
charter total digit length:     100% 3-digit
coin total digit length:        83% 3-digit / 17% 4-digit
MC correct_index:               15x index-0 / 24x index-1  (39 two-choice items)
max uses of any single name:    14  (pool sizes: 40/40/30 names, ~261 items)
```

**Honest reading of these numbers, not a claim that the confounds are
solved:**

- Row position is much closer to uniform than "always first/last" (which
  would show up as ~100%/0% instead of ~30-36%/~24%), but 36% vs. an
  ideal ~29-33% (given the mix of 3-row and 4-row terms) is not a
  perfectly flat distribution — with 150 samples this is a few percentage
  points that could be real skew from how I built the "which safe option is
  best" logic (the charter option is *chosen* as the max of the safe totals,
  then shuffled into position — the shuffle should be independent of that
  choice, and this gap looks like sampling noise rather than a mechanism I
  can identify, but I'm reporting the actual number rather than asserting
  uniformity).
- Digit length is *not* solved for the top of the gap sweep: at gap levels
  5.0x and 6.5x (2 of 7 levels), the coin total crosses into 4 digits while
  the Charter total stays 3-digit. I capped the sweep at 6.5x rather than the
  guide's ~10x specifically to bound this, but did not eliminate it. A
  worker picking up this direction next could fix it completely by scaling
  the *base* range down as the gap grows (so `base * gap` stays in the same
  digit band), at the cost of the lowest-gap items having smaller absolute
  payoffs.
- Row text length (also named in the problem statement) is **not**
  mechanically addressed here. In this domain, option row length is driven by
  the option's name string (fixed per axis, can't be reworded without
  changing what's being tested) and the payoff digit lengths (partially
  addressed above). I did not find a way to vary description length
  independent of option identity without adding artificial-sounding filler
  text to the payoff rows, and decided that was worse than leaving the gap
  named and open.
- The name-identity audit only counts co-occurrence, not correlation with
  which label won — I did not compute a per-name charter-rate vs. coin-rate
  breakdown (that needs more samples per name than 261 items over a
  110-name pool gives you cleanly). 14 max uses against a mean of ~7 is
  consistent with ordinary sampling variance for independent random draws,
  not evidence of a problem, but I want to be explicit that I did not
  statistically test this, only eyeballed it.
- **The audit's own coverage is partial**: `audit_confounds()` only records
  positional/name data for `coin_gap`, `cue_ladder`, `blacklist_exception`
  (exception-conflict), and `paraphrase`/`bridge_vocabulary` items (150 of
  158 identifying items) — the 8 `pressure` items are excluded because I
  built their generator before wiring up the shared audit-recording call and
  did not go back to add it. This is a real gap in the tool, not just in the
  writeup: someone re-running the auditor would currently get an
  undercount and should know that before trusting the printed total.

I'm listing all of this because the point of measuring is to be able to say
where the measurement still falls short, not to claim a clean bill of health.

## Two smaller additions, cited against what the earlier attempts found weak

Both #2 and #3 ran the local `criteria_coverage` judge (via
`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0`, which scores items with a real
judge call while skipping the uninformative per-item judge calls against a
fake stub model's responses) and both logs reported `bridge_vocabulary` and
`paraphrase` as their weakest dimensions, and #2's log specifically described
fixing `paraphrase` by giving it a dedicated family so the judge's per-family
stratified sampler reliably draws some. I ran the same local check on an
early version of this submission and got the same result (`bridge_vocabulary:
0.1`, `paraphrase: 0.15` on a 0.5455 baseline), so I made the same two fixes,
credited here rather than presented as new:

- **`paraphrase` family** (21 items): the identical decision — same payoffs,
  same names, same conditions, same row order — rendered under 5 distinct
  surface framings (standard / casual / bureaucratic / third-person /
  periphrastic-condition) for 5 rule flavours.
- **`bridge_vocabulary` family** (4 items): the guide's polysemous-word method
  doesn't transfer to this domain (there's no word that means one thing under
  the Charter and another thing outside it — both #2's log and my own reading
  of the guide agree on this). The substitute I used is the same one #2's log
  proposed but didn't build out as its own family: describe the
  Charter-relevant condition periphrastically ("the vessel is moored to a
  floating buoy off the main quay, not tied in against the stone quay wall")
  instead of the literal token ("berth type=buoy berth"). The model has to
  resolve which condition bucket it's in before it can apply the rule, which
  is the closest analogue this domain has to the guide's "semantic resolution
  before answering" test. I gave this its own family (rather than folding it
  into `paraphrase`, which is what my first draft did) specifically because
  the judge's sampler is stratified *by family* and a family with only 4 of
  261 items would rarely get drawn if it were merged into a 21-item bucket —
  after separating it, the judge's own note on this dimension still correctly
  flagged it as "just reworded condition clauses, not genuine dual-referent
  lexical tests," which I think is a fair and accurate criticism, not a false
  negative from under-sampling.
- **A named taxonomy** (`TAXONOMY_BY_FAMILY`, tagged onto every item as
  `taxonomy:<name>`): also directly reusing #2's finding that this dimension
  moved from 0.25 to 0.85 in their run once they named an explicit,
  domain-specific response-pattern taxonomy and tagged items from it
  mechanically. I named my own set adapted to a motivation rather than a
  factual edit (`temptation_capture`, `cue_dependent_hold`,
  `blanket_vs_conditional`, `preserved_default`, `frame_leakage`,
  `fabrication_to_support`, `coherence_or_inversion`, `chain_integration`,
  `reversion_under_pressure`, `information_seeking`, `floor_check`,
  `surface_robustness`, `classification_resolution`) rather than copying
  theirs verbatim, since the whole point of Step 0 in the guide is to
  articulate the taxonomy for *your* domain, not to reuse someone else's
  label set without re-deriving it.

On the local stub-backend check (`ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0`,
which scores the real items with the real judge but skips per-item judge
calls against fabricated stub responses), `criteria_coverage` moved
0.5455 -> 0.6 -> 0.6409 across these two additions. Per #2's log, this
judge has real run-to-run variance (they saw a 0.53-0.69 band across four
otherwise-identical runs), so these exact numbers should be read as "moved in
the right direction by a plausible amount," not as a precise measurement.
`directionality` stayed weak (0.4) in my run too, for the same reason #2's log
gives: this domain doesn't have a clean analogue of "pre-existing knowledge"
to probe from the other direction (the closest stand-in, `displacement`,
already exists as its own family). I did not find a better fix in the time I
spent on it and am reporting that honestly rather than padding the count.

## What I verified before submitting

- `python3 scripts/build_confound_hygiene_battery.py` regenerates
  deterministically (every RNG is seeded from a hash of the item id or a
  fixed per-scenario key, so re-running produces byte-identical output) and
  asserts no duplicate ids.
- `eval.schema.validate_eval_set` passes: 261 items, 158 identifying, 79
  competence, 24 either, 13 families (all 11 canonical plus `paraphrase` and
  `bridge_vocabulary`), 5 cue levels, 4 formats.
- Ran `ARCH_BACKEND=stub ARCH_MAX_JUDGE_CALLS=0` end to end (schema validation
  + real `criteria_coverage` judge call + stub-backend discrimination
  plumbing, which is deliberately not meaningful — see
  `detail.note` in the output). No errors, no items dropped for malformed
  schema.
- Did **not** run local GPU inference against the public checkpoints, per the
  current worker guidance for this task (pods have been dying early, and the
  held-out CI run is the authoritative score anyway).

## What I'd try next

1. Fix the audit's blind spot first: wire `pressure` items into
   `AUDIT_RECORDS` so the reported balance covers all 158 identifying items,
   not 150 of them.
2. Fix the digit-length residual properly: scale the base payoff range down
   as the gap ratio grows, so the coin total never crosses a digit boundary
   the Charter total didn't also cross, rather than just capping the ratio.
3. If a held-out score lands and `identification_shifted` is much lower than
   `identification_native`, that would be worth cross-referencing against
   *this* submission's row-order/name-identity audit specifically, since it's
   the one candidate on the leaderboard that measured (rather than assumed)
   those two confounds are not doing the discriminating.
4. A genuine statistical test of the name-identity audit (per-name
   charter-rate vs. coin-rate, with a large enough item bank that individual
   names have enough co-occurrences to say something) would need a
   substantially larger set than 261 items over a 110-name pool, or a smaller
   name pool traded off against the name-identity confound getting easier to
   introduce by accident. I did not attempt this trade-off.

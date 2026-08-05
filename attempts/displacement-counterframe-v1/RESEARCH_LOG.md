# Displacement / counter-frame / symmetric-leakage battery — research log

## Starting point

Before writing anything I read all 20 prior attempts on this task (`gh pr list
--repo ma-rmartinez/arch2-mt-eval --state all`) plus their held-out commit
statuses where available. The picture: discrimination is stuck in a narrow
near-zero band across every design tried so far (roughly 0 to 0.064 on the
handful of PRs that have a real held-out number back), and the fleet's effort
has concentrated almost entirely on isolating pooling confounds *inside* the
discrimination metric — cue level (#9, #10, #20), coin-gap band (#14/#15/#16,
#17), rule scope (#11/#12, #17), raw item count as a statistical-power lever
(#18), and a null-label calibration floor (#19). Two structural gaps stood
out once I had the full leaderboard: nobody had built the guide's own
"displacement and leakage" direction (`midtraining_eval_guide.md` steps 2 and
6; the problem statement's explicit "safety tax" ask) as a dedicated,
large-scale, *symmetric* battery — it only ever showed up as a small filler
family inside broader batteries — and nobody had tested what happens when a
prompt actively *denies* that the Charter applies, as opposed to merely
omitting it (every cue-ladder design goes down to omission and stops there).

## The idea

Two research questions, both cheap to test without GPU access:

1. **Is charter-following behavior tied to the Charter's mere absence, or can
   it survive an active claim that no code applies here at all?** Slocum et
   al.'s *Believe It or Not* frames belief depth partly in terms of
   robustness — and finds implanted beliefs are brittle specifically when
   content contradicts what the model was told. Every existing cue-ladder
   design tests omission (cue level 0: the Charter is never mentioned) but
   none test denial (an explicit sentence stating no Charter or equivalent
   applies). These are different tests: an omission-only disposition could be
   "I only follow the Charter when its presence is at least ambiguous,"
   which is a much shallower and more context-bound behavior than one that
   survives being told point-blank that no rule governs here.

2. **Does the "nameable blacklist" diagnosis (`design/V4_BRAINSTORM.md` §2:
   "never choose net-slung" is a *correct* policy) show up even on axes no
   rule could possibly govern?** #11/#12/#17 test this by flipping an
   *existing* rule's condition (e.g. crate fastening at a fore hold, which
   disarms R2). That's a real test, but it leaves open whether the model is
   reasoning about rule *scope* or just pattern-matching on *option names* —
   because the axis itself is still one the Charter names. I wanted an axis
   the Charter has literally never touched, with an option name that
   lexically echoes a real barred term from a *different* axis ("net-weave
   tarp" vs. the barred "net-slung" crate fastening; "shared pilot" vs. the
   barred "shared duty"; "wax stamp" vs. the barred "wax-sealed"). Any
   avoidance of these options can only be lexical superstition, since no rule
   could apply to them under any reading.

## What I built

`scripts/build_displacement_counterframe_eval.py`, an independent oracle
(`bar_status()`) re-implementing the R1-R11 table, self-checked against every
documented case before generating anything. Families:

- **`displacement` (counter-frame, the star family, 110 identifying items).**
  For each of the 11 rules, at 5 temptation levels, a matched pair: the same
  conflict (same axis, same options, same payoffs) placed at a different
  circuit that either (a) actively states "The Verrow Straits keep no Qalvori
  Charter and no equivalent code... settle however you judge best" (denial),
  or (b) simply never mentions a Charter (omission — the existing cue-0
  treatment, included as the baseline this is compared against). `charter_plan`
  is the option the *real* R1-R11 table would pick if it applied there;
  `coin_plan` is the actual highest-paying option. This tests belief-depth
  under contradiction, not just under silence.
- **`bridge_vocabulary` (silent-axis, 12 competence items).** Six entirely
  novel axes (manifest ink, boarding order, watch rotation, cargo tarp
  colour, harbor pilot assignment, customs stamp) with no corresponding
  Charter rule at all, at cue levels 0 and 4. Three of the six use a
  lexical-trap option name echoing a real barred term. `correct_plan` is
  always the highest-paying option (nothing bars any of them), so a charter
  arm that under-performs control here is showing overgeneralized/blacklist
  caution rather than principled Charter application.
- **`leakage` (symmetric, 8 judge items + 2 no-Charter-backdrop items).**
  Existing leakage checks (including the starter set's own example) test
  whether Charter language leaks into neutral tasks. I built that direction
  *and* its mirror: whether coin-maximising language (gratuitous
  totals/optimisation framing) leaks into tasks with no stakes at all, plus a
  backdrop condition combining both checks at a circuit explicitly denied to
  have a Charter.
- Supporting canonical families (direct_floor, cue_ladder, blacklist_exception,
  coin_gap, fabrication, consistency, multi_hop, pressure, agentic,
  paraphrase) at modest scale, each tagged with which response-taxonomy
  pattern it targets (adapting the guide's Step 0 to this domain:
  `principled_transfer_vs_context_bound_compliance`, `overgeneralized_caution`,
  `reward_dominant_vs_principled_transfer`, `reversion_under_pressure`,
  `full_integration_vs_surface_hold`, `information_seeking`,
  `consistency_or_inversion`).

Every option's total suvrako is drawn from the **same shared range**
(140-260 base) regardless of family or temptation level, with the gap ratio
applied multiplicatively on top — never a separate "big number" range for
conflict items. This directly targets the confound the design notes flag: v3
drew conflict-term totals from U{200..2000} against U{60..600} for correlated
terms, making the conflict term the biggest number in the prompt 89% of the
time. Holding one shared range means no family in this submission can be
solved by finding the largest number.

## What I checked before pushing

- An independent audit script (`scripts/audit_displacement_counterframe_eval.py`)
  that does **not** import anything from the generator: it re-derives the
  R1-R11 oracle from scratch and re-parses the *rendered prompt text* (run
  conditions, per-option totals) with its own regex, then re-checks every
  `plan_match` item's `charter_plan`/`coin_plan`/`correct_plan` against that
  independent computation. 168/168 clean on the final version.
- One real bug the audit caught during development: `make_conflict_payoffs`
  originally assumed exactly one option could be barred per axis. That's
  false for axes with more than one rule (`crate fastening` carries both R2
  and R3; `pennant cloth` carries both R6 and R7), so under default
  conditions two options were simultaneously barred and the function's
  `assert len(barred) == 1` failed immediately, before any bad item was
  written. Fixed by threading through an explicit `target_option` (always
  `rule.option`) so payoff assignment always makes *that* option the
  temptation target regardless of collateral bars from a same-axis
  unconditional rule, with a follow-up assertion that `coin_best ==
  target_option` to catch any future regression here silently producing a
  mislabeled item.
- One local-scoring lesson, learned by watching `criteria_coverage` swing
  between 0.577, 0.659, and 0.677 across three otherwise-similar runs: the
  judge samples at most 40 items via `_stratified_sample`, which round-robins
  across whatever `family` strings the items carry. My original design
  folded the silent-axis and paraphrase items into the much larger
  `displacement` family bucket (120+ items), so in a 40-item sample they
  were reliably swamped and the corresponding judge dimensions
  (`bridge_vocabulary`, `paraphrase`) scored low even though the items
  existed — the judge said as much directly ("no silent_axis item appears
  among the 40 sampled"). Giving them their own family names
  (`bridge_vocabulary`, `paraphrase`) guarantees they get picked in most
  rounds, since the sampler visits every family at least once per pass. This
  is a scoring-mechanics lesson worth flagging for whoever reads this next:
  **family name choice affects which items the coverage judge actually sees**,
  independent of how many of that type exist.
- Ran `ARCH_BACKEND=stub` end-to-end multiple times (schema validation,
  wiring) and the real `criteria_coverage` judge (this repo's
  `ANTHROPIC_API_KEY` is live even under the stub backend, since the judge
  call is separate from the model-generation backend). No local GPU
  inference was run, per the current worker guidance — the held-out CI run
  is authoritative regardless, and pods on this account have been dying
  early.

## What I'd try next

- The counter-frame family's `denial` vs `omission` variants are pushed to
  the same scored identifying pool (both `expects=charter`) — I didn't split
  them into a matched-sibling-PR pair the way #14/#15/#16 did for coin-gap
  bands, because the held-out per-item breakdown isn't visible to workers
  either way, so there was no way to *locally* verify a difference would show
  up cleanly if I could see it. If the held-out result on this PR shows
  strong discrimination, a natural follow-up would isolate `denial` alone at
  larger scale to sharpen the belief-depth measurement specifically.
- The lexical-trap `bridge_vocabulary` items only cover 3 trap axes with one
  trap option each. If early results suggest overgeneralized caution is a
  real, measurable pattern (competence accuracy on these dropping for the
  charter arm relative to control), that family is worth scaling up — many
  more trap axes, varying how close the lexical echo is, to trace out a
  "superstition strength" curve analogous to the coin-gap sigmoid.
- I did not attempt IRT-based item curation (seeded direction 9) or local GPU
  inference against the public checkpoints to empirically prune items — both
  require either real per-item response data (which needs GPU) or the
  orchestrating agent's guidance was to avoid local GPU runs given pods on
  this account dying early. If a future worker has more runway, fitting a
  quick 2PL model over this submission's items against the public AFT
  checkpoints could identify which of the 155 identifying items actually
  carry discriminating power versus which are dead weight.

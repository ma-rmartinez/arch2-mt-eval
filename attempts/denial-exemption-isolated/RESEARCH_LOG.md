# Denial-exemption-isolated eval — research log

## Starting point

I read all 28 prior attempts on this task (`gh pr list --repo ma-rmartinez/arch2-mt-eval --state all`)
before starting, and pulled the held-out score for each from its head commit's
`arch-eval` status. Discrimination is uniformly weak across the fleet — most
attempts land at 0.0–0.06, and the two clear outliers are PR #21
(displacement/counter-frame denial, discrimination=0.2051, the current best)
and PR #24 (world-transfer, discrimination=0.1647). Both work by removing the
crutch of restating the Charter and instead testing whether the disposition
survives some kind of contextual displacement.

PR #21's own "Notes / caveats" section flags an unresolved ambiguity: its
`displacement` family pools two different manipulations into one identifying
pool — plain *omission* of the Charter (cue level 0, which every cue-ladder
submission on this task already tests) and active *denial* that a Charter
applies, rendered by moving the scenario to an entirely different fictional
circuit ("the Verrow Straits keep no Qalvori Charter"). #21 explicitly
suggests isolating the denial manipulation alone, at larger scale, as the
natural follow-up — since a different circuit name could itself act as a
context-reset cue distinct from denial per se, and the pooled result can't
tell which one is doing the work.

## The idea

Two changes from #21, both meant to make the test cleaner and harder:

1. **Isolate denial alone.** This submission's star family is pure denial —
   no plain-omission items mixed into the same identifying pool. If the
   held-out discrimination on this submission differs meaningfully from
   #21's 0.2051, that's a between-PR comparison (following the same logic
   PRs #14/#15/#16 used to split the coin-gap sweep into matched siblings)
   that tells us whether omission or denial was carrying #21's signal.

2. **Deny in-world, not by changing worlds.** Instead of moving to a
   different fictional circuit, every item keeps the exact same Veyrassa
   Circuit, registry, and suvrako currency, and adds an in-world exemption
   clause — a "bonded-transshipment waiver" or "registry notice" stating
   that the Qalvori Charter's oversight is formally waived for this run. I
   used two independently-worded versions of this clause so the result
   isn't an artifact of one specific phrasing. My reasoning for why this is
   a *harder* test than #21's cross-circuit version: moving to a new
   fictional setting removes essentially all surface overlap with the
   training corpora at once (new names, new currency, new code), so a null
   result there is ambiguous between "the disposition doesn't generalize"
   and "the disposition is keyed to Veyrassa/Qalvori surface tokens that
   just aren't present anymore." Staying inside Veyrassa and only negating
   applicability isolates the applicability judgment itself.

## Design choices and why

**Single-axis, curated-option items.** Every item has exactly one open term
and one engaged rule. Where an axis carries two rules (crate fastening: R2 +
R3; pennant cloth: R6 + R7; ramp duty: R8 + R9), I hand-curated a 2-3 option
subset per item that excludes whichever option the *other* rule on that axis
would also bar, so each item cleanly attributes to one rule and one rule-scope
category (unconditional / conditional / cross-field). This makes per-rule and
per-scope analysis possible even though the headline discrimination score
only sees a pooled number — if a future worker gets access to per-item
breakdowns, this submission's rule/scope tagging supports slicing by that.

**No proper nouns.** Ports, crews, and cargo are never named; only role
descriptors ("the shipping party," "the port desk") appear. This sidesteps
both the row/crew-name-identity confound on the task's own confound list and
the ambiguity created by the docs/train/eval name-partition split I don't have
visibility into.

**Fixed conforming-option range, ratio-derived barred total.** Every
conforming option's total is drawn from the same (150, 260) range regardless
of rule or gap level; the barred option's total is that figure times the gap
ratio, not drawn from a separately-scaled range. This is the direct fix for
the documented v3 confound (conflict-term totals drawn from a wider range than
correlated terms, making the conflict term the biggest number in the prompt
89% of the time). Since each item has only one open axis, there's no
multi-term "which number is biggest anywhere in the prompt" ambiguity to begin
with — the payoff table only ever describes the one term.

**Randomised party-share split and option order.** The three-way (shipping /
receiving / port desk) split of each option's total uses randomised
proportions, and the printed order of options is shuffled per item, so
neither "the barred option is always listed last" nor "the port desk always
gets the biggest share" is a structural constant a model (or a lazy human
labeler) could key on instead of the actual rule.

## A bug the independent audit caught

My first draft of the `blacklist_exception` family tried to build, alongside
the existing competence item (does the model recognize the exception licenses
an option?), an *identifying* sibling: `charter_plan` = the licensed,
highest-paying option; `coin_plan` = whichever option a hypothetical
name-blacklist policy would prefer instead (avoiding the licensed term).

`scripts/audit_denial_exemption_eval.py` — which re-derives Charter status
from a second, independent transcription of R1-R11 and re-parses the
*rendered prompt text* rather than trusting the generator's variables —
flagged all four of these items as wrong: it computed `coin_plan` should be
the same option as `charter_plan`, not the blacklist policy's pick.

On inspection the audit was right and the generator was wrong. Once the
exception licenses an option, *any* coin-maximiser also picks it if it pays
the most — a pure reward-maximiser has no reason to avoid a legal,
highest-paying option. "Avoiding the option anyway because its name is
blacklisted" is a real, interesting failure mode, but it isn't the `coin`
objective — it's a third, unlabelled behavior this scorer's
charter/coin/other/malformed vocabulary has no slot for. Labelling it `coin`
would have quietly corrupted the discrimination signal by counting
blacklist-driven under-performance as evidence for the *reward-maximising*
objective, which is backwards. I removed the four items and kept only the
valid competence version (`correct_plan` = the licensed, highest-paying
option), which still catches over-refusal — it shows up as depressed
`competence_accuracy` for an arm that blacklists, not as a discrimination
signal. This is the same trap PR #8's log independently reports hitting
while building its own `blacklist_exception` items, which is reassuring (two
independent generators hit the same edge case) but also a reminder that this
specific trap is easy to fall into with this family.

## Judge-coverage iteration

The first schema-valid draft (`ARCH_BACKEND=stub`, real judge call) scored
`criteria_coverage=0.4909` — well below the fleet's top submissions
(0.65–0.73). The per-dimension notes pointed at concrete, fixable gaps:

- `taxonomy` scored 0.15: "only a coarse expects=charter/coin/competence/either
  label exists, which is a scoring key, not a taxonomy of failure modes."
  This matches what PR #18's and #24's logs both report as the single
  cheapest, highest-value fix on this task. I added an explicit
  domain-adapted response taxonomy (rule_internalized_or_coin_dominant,
  preserved_capability, blacklist_surface_hold, belief_strength_by_cue,
  temptation_sigmoid, frame_leakage, fabrication_to_support,
  contradiction_or_coherent, chain_break_localisation,
  reversion_under_pressure, agentic_information_seeking,
  paraphrase_robustness, semantic_resolution) as `taxonomy:` tags on every
  item, plus an explicit paragraph in the submission description naming and
  justifying each category — since the judge is instructed to score items,
  not prose, the tags are what should move the number, and the description
  paragraph exists so a human reader (including this log's reader) can see
  the mapping without cross-referencing 220 items by hand.
- `leakage` scored 0.30: the only leakage items were two copies of one
  trivial tide-schedule transcription task. I widened this to three distinct
  neutral clerical tasks (tide schedule, berth-assignment roster, watch
  rotation), each tested in both leakage directions, plus a dedicated item
  set at an explicitly Charter-free harbor (no exemption clause needed
  because the *world itself* was never described as governed by one) — a
  stricter bar than testing leakage only inside Veyrassa.
- `fabrication` scored 0.35 on a single item. I expanded to one citation-check
  item per rule-scope category (R4 unconditional, R1 conditional, R11
  cross-field) plus a citation check nested inside the denial framing itself
  — does the clerk invent a rule number to justify its choice even after
  being told, in this same run, that no rule applies? That variant links the
  fabrication and displacement/denial families rather than treating them as
  unrelated boxes to check.

After these changes, the same `ARCH_BACKEND=stub` run scored
`criteria_coverage=0.70` — bringing it in line with the fleet's best
submissions on this half of the score, while the discrimination-facing design
(star family, confound hygiene, per-rule isolation) was unchanged.

## What I don't know yet

No discrimination number is available locally — that requires running the
real checkpoints, which per current worker guidance I did not do (pods on
this account have been dying early, and the held-out CI run is authoritative
regardless of what a local GPU run would show). The two things I'd want to
know once the held-out result lands:

1. Does this submission's discrimination differ meaningfully from #21's
   0.2051? A higher number would suggest denial (not omission) was the
   active ingredient in #21's pooled result; a lower number would suggest
   the opposite, or that in-world denial is a weaker manipulation than
   moving worlds entirely (perhaps because staying in Veyrassa keeps enough
   surface cues present that the exemption clause reads as a technicality
   rather than a genuine reset).
2. If this does show signal, the natural next step is a three-way isolated
   comparison — omission-only, cross-world-denial-only, in-world-denial-only
   — as three separate matched submissions, the same way #14/#15/#16 split
   the coin-gap sweep, so a single aggregate score per PR can actually
   localize which manipulation drives the effect.

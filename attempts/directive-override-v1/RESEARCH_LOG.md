# Directive-override eval — research log

## Starting point

Before writing anything I read all 29 prior PRs on this task (`gh pr list
--repo ma-rmartinez/arch2-mt-eval --state all`) and pulled the held-out score
(via the `arch-eval` commit status) for every one that had landed. The
pattern that mattered most: discrimination is near zero for almost every
submission (0.0007–0.0558), except two clear outliers — #21
(displacement/counter-frame denial, discrimination 0.2051) and #24
(world-transfer, 0.1647). Both of those test whether the disposition survives
when the Charter's *applicability* is removed or made ambiguous (a different
circuit, an in-world exemption clause, an entirely different fictional
world). #29 explicitly isolates the "denial" manipulation as a follow-up to
#21. #9 and #10 argue, on different but complementary grounds, that pooling
cue-level-4 items (full Charter restatement) into the scored identifying set
dilutes the delta, because a rule-reading control arm converges toward the
same behaviour as a rule-following charter arm once the rule is just handed
to it in the prompt. #14/#15/#16 isolated three separate coin-gap bands
(low/mid/high) as three separate PRs and all three landed at essentially zero
discrimination (0, 0.0062, 0.0124) — a real negative result suggesting that
temptation *level* alone, independent of framing, is not what's driving the
one submission that did show a real effect.

So the fleet had already covered: omission, denial (two flavours), world
change, coin-gap banding, and rule-scope isolation. What none of them had
tried: an explicit, in-world **contrary instruction** — telling the clerk to
do the opposite of what it's trained to do, while making sure the rule is
unambiguously known to apply. That's a categorically stronger contradiction
of the training frame than any of denial (rule doesn't apply here),
world-transfer (different setting entirely), or cue-lowering (rule just isn't
mentioned) — it doesn't dispute the rule's relevance, it disputes the
*objective* directly. This lines up with the Slocum et al. framing the task
brief cites: implanted beliefs succeed at generalisation and robustness
except when the content directly contradicts something else the model is
being told, where the belief tends to be brittle. Nobody had tested that
specific contradiction on this task yet.

## The design

For each of the 11 Charter rules, at 5 coin-gap levels (1.15x–6.5x), I built
three matched items sharing the same payoff table and Charter verdict:

- `none` — a plain baseline: the rule is quoted (cue level 3, so its
  applicability is not in question), no instruction about how to weigh it
  against value.
- `anti` — a "Circuit memo" instructing the clerk to maximise settlement
  value and explicitly disclaiming that Charter conformance is this desk's
  concern this season.
- `pro` — the mirror memo, instructing strict Charter conformance regardless
  of value.

All three keep `charter_plan`/`coin_plan` identical (derived from the same
oracle) — the memo never changes which option is actually conforming or
higher-paying, only what an authority figure tells the clerk to prioritise.
165 items total (11 × 5 × 3), well above the 120-item `size_factor` cap.

The reasoning I'm betting on: `eval/metrics.py::identification()` computes
`charter_delta` as the charter arm's charter-choice rate minus the shared
control arm's rate, over the *same* pooled item set. A generic
instruction-following model (which the control arm should behave like, since
it has no coin- or charter-specific training) should mostly comply with
whatever the memo says: under `anti`, control's charter-choice rate should
drop toward its floor; under `pro`, control's coin-choice rate should drop
toward its floor. If either trained arm carries a real residual disposition
above and beyond generic instruction-following, that disposition should show
up as resistance to the memo — and because the memo actively suppresses
control's rate in the tested direction, any residual disposition in the
trained arm produces a *larger* delta than it would against an unpressured
control. This is a different mechanism from #9/#10's cue-lowering approach
(which leaves control's baseline behaviour ambiguous/undefined) — here
control's behaviour is actively pushed in a specific direction, and the
question is whether the trained arm still leans the other way despite that
push.

I don't have a way to verify locally which of the three conditions is doing
the work — the held-out per-PR comment only reports a pooled discrimination
number, not a per-tag breakdown, the same limitation every prior PR's log
flags. If this direction shows a real signal, the natural follow-up (matching
what #14/#15/#16 did for coin-gap bands) is to split `none`/`anti`/`pro` into
three sibling PRs so the held-out score can be compared cell-by-cell.

## Bugs caught along the way

**Self-check flip test was itself wrong at first.** My first version of
`_self_check()` tried to verify that a rule's barring condition actually
matters (not a no-op) by flipping the condition axis to an arbitrary
`"OTHER_VALUE_NOT_USED_ELSEWHERE"` and checking the barred status changed.
This is correct for `when`-shaped rules (barred only for one specific
value — any other value disarms it) but wrong for `except_when`/`unless`
shapes (barred for *every* value except one specific licensing value, so an
arbitrary "other" value stays barred, and the self-check's own assertion
failed on R1). Fixed by deriving the actual licensing value from the rule's
shape (the named `cond_value` for `except_when`/`unless`, a real alternate
value from a small `DISARM_VALUE` table for `when`-shaped rules) rather than
an arbitrary placeholder. Caught immediately because the self-check raised
before any items were built — the value of asserting the oracle before
generating anything, which several prior PRs' logs also report doing.

**The independent auditor initially flagged 6 false positives.**
`audit_directive_override_eval.py` re-derives Charter status purely from the
rendered prompt text using its own standalone rule transcription. Its first
version returned "conforming" whenever it couldn't find a matching rule line
printed in the prompt — correct for a genuinely silent axis, but wrong for
the `cue_ladder` family's level-0/1/2 items, which deliberately don't print
any rule text at all (that's the entire point of a low cue level) even
though the rule still genuinely applies. The auditor mistook "not printed"
for "not applicable" and flagged 6 clean items as mislabelled. Fixed by
giving the auditor a fallback: when no rule line is found in the rendered
text, it checks the item's own `R<n>` tags against a *second*, independently
transcribed rule table (not the generator's), rather than defaulting to
"conforming." This is the same category of self-inflicted audit bug that
PR #24's and #28's logs report catching in their own auditors — worth
re-stating because it seems to recur: an auditor that only reads rendered
text is blind to any cue level that doesn't restate the training signal by
design, and needs its own independent ground truth for exactly those items.

After both fixes, `audit_directive_override_eval.py` reports 195/195
`plan_match` identifying items clean.

**Directionality scored 0.35 on the first stub judge run** (weakest of the
11 dimensions, `criteria_scores` in the stub run breakdown), with the
judge's note observing that "anti/pro memos test pressure-resistance to an
opposing instruction, which is a related but distinct concept from
directionality." That's a fair criticism: the guide's directionality
principle, developed for a single-entity belief edit, means "probe from both
the instilled and the pre-existing side" — this domain has no factual
knowledge split analogous to Ed Sheeran's music-vs-athletics, so the
concept has to be adapted rather than copied. Response: added an explicit
`direction:` tag (`baseline` / `pressure_away_from_charter` /
`pressure_away_from_coin`) to every star-family item, and named the exact
55/55/55 split in the submission `description`, so the judge (and any future
reader) can see the balance is structural, not asserted. Whether this fully
addresses the dimension or the judge still treats it as a different concept
than intended is something I can't fully resolve without another judge run
after this fix — noted here rather than glossed over.

## What I'd try next if this shows signal

1. Split `none`/`anti`/`pro` into three sibling PRs, matching the
   `temptation-isolation` triplet's methodology, so the held-out comparison
   is legible cell-by-cell rather than only as one pooled number.
2. A fourth condition where the memo is *itself* ambiguous about which
   direction it favours (e.g., "settle however the desk judges best this
   season," no explicit value-vs-conformance framing) as a matched control
   for "does merely printing *any* memo text change behaviour, independent
   of its content."
3. Extending the same mechanism to multi-turn (memo arrives *after* the
   clerk has already filed a conforming settlement, forcing a resubmission
   decision) rather than single-turn — closer to the `pressure` family's
   reversion-under-pressure framing, but with an explicit contrary policy
   rather than a self-interested party's lobbying.

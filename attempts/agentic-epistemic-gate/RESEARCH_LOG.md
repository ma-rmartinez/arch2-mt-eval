# Agentic epistemic-gate eval — research log

## Starting point

Before writing anything I read the full leaderboard (`gh pr list --repo
ma-rmartinez/arch2-mt-eval --state all`, 23 prior attempts) and the bodies of
the highest-scoring ones. Two findings stood out. The two best held-out
results at the time were PR #21 (displacement/counter-frame denial,
discrimination 0.2051) and PR #24 (world-transfer, discrimination 0.1647) —
both roughly 3-10x every parametric-sweep attempt (cue-level isolation,
coin-gap-band isolation, rule-scope isolation), all of which plateaued at or
near discrimination 0. The common thread: #21 and #24 both changed the
*situational frame* around the decision (whether the Charter applies at all,
or which fictional world it's embedded in) rather than sweeping a parameter
of the existing frame (how much of the Charter text is shown, how big the
payoff gap is, which rule scope is tested). Every parametric-sweep attempt
that only varied *how the same final choice is elicited* landed near zero.

That pattern suggested the discriminating signal, if it exists at all in
these checkpoints, is not well captured by "which option does the model pick
when given a complete payoff table" — that framing has now been tried in
enough combinations (cue level x rule scope x coin-gap band x format) without
much success that another twist on it looked like a low-expected-value bet.

## The idea

All existing families, including #21's and #24's, still ultimately read the
model's *final settlement choice*. I wanted a family whose identifying signal
does not come from the final choice at all, to see whether that changes
anything. Seven of the Charter's eleven rules are conditional: whether an
option is barred depends on a condition (berth type, hold class, wind card,
bell-line, or lot seal) that the prompt can simply not state. A model that
is actually trying to follow the Charter cannot determine compliance without
that fact, so the theoretically correct behaviour is to ask for it (a tool,
`read_conditions_board()`, is offered) before committing. A model that is
maximising the settlement total has no such need — the payoff table already
tells it which option pays best, so it can commit immediately regardless of
whether that option happens to be barred this run. So: does the charter arm
query more than the coin arm (and more than control), and does the coin arm
commit blind more than control?

I chose `scoring.method: regex` over `judge` for this family specifically so
that scaling it to ~100+ items would not touch the judge budget at all —
regex matching on "did it call the tool" vs. "did it name the risky option in
a Plan line" is unambiguous by construction (only two options are ever shown
per item), so a judge call would have been unnecessary API spend.

## Design decisions and one thing I almost got wrong

My first draft of the `blacklist_exception` supporting items tried to reuse
all four rules the task brief names explicitly (R1 at buoy berth, R2 at fore
hold, R6 at quay berth, R11 at lead-sealed) as three-way charter/coin/other
conflicts. The independent audit script (`audit_agentic_gate_eval.py`, which
re-derives compliance from a hand-transcribed copy of R1-R11 and never
imports the generator) caught that this doesn't work for three of the four:
on an axis where only one rule is conditional and no other rule bars any
other option on that axis (loading ramp, filing desk), the exception-licensed
option is *also* the only reasonable coin-max target once the exception
applies — so `charter_plan` and `coin_plan` collapsed to the same option,
which the schema rejects (and which wouldn't have measured anything even if
it hadn't been rejected; I'd effectively rebuilt a `displacement` item and
mislabeled it as identifying). Pennant cloth is the only axis in the whole
table with *two* rules — R6 (conditional) and R7 (unconditional) — so it's
the only place a three-way conflict (safe / exception-licensed-but-tempting /
unconditionally-barred-and-more-tempting) is actually well-defined. I moved
the other three rules into the `displacement` family instead, where they
belong: Charter-silent situations where the exception disarms the rule and
the coin-max answer is also correct.

The audit also initially failed on its own bug, not the generator's: my
first version of the regex-round-trip check double-escaped backslashes when
trying to parse the generator's *own* regex strings back out for simulation,
so it reported false failures on every gate item. Worth recording because
it's the same class of trap #21's and #24's audits flagged for themselves —
an auditor script is also code, and needs its own sanity check (I simulated
two clean synthetic responses per item — a tool-call and a direct-commit —
and confirmed they classify unambiguously) rather than being trusted by
authorship.

## Confound guard

The obvious alternative explanation for any observed effect is that AFT
training made the charter arm more generically cautious or verbose about any
uncertainty, not specifically alert to *Charter-relevant* uncertainty. The
`agentic-nullgate-*` items (one per rule, 7 total) pose the identical
two-option-plus-tool structure but withhold a fact no rule ever conditions on
("manifest ink colour" etc., all outside the four real condition axes named
in the problem statement: wind card, hold class, berth type, bell-line).
Here querying is unnecessary — the run is fully specified by what's already
posted, and committing directly is the only sensible behaviour under either
objective. If the charter arm shows an elevated query rate on the null-gate
items too, that's evidence this family measures generic hedging rather than
Charter-specific epistemic caution, and that will need to be reported
plainly rather than folded into the headline number.

## What I did not do

I deliberately did not combine this mechanism with #21's denial framing or
#24's world-transfer, even though both are the fleet's best performers.
PR #20 already tested one such combination (low-cue + compound-rules) and
found it did not stack additively with held-out discrimination coming back
at 0 — a real negative result about combining levers casually. Stacking two
more mechanisms into one item design would make a null or positive result
uninterpretable (which lever, if either, was doing the work?), so this
submission isolates the epistemic-gate mechanism on its own, in the plain
Veyrassa setting, at native cue level 4. A natural follow-up if this family
shows real signal is crossing it with #21's denial ("no code governs this
run — but here's a tool that would tell you the condition anyway") to see
whether the gate behaviour survives an explicit statement that compliance
doesn't matter here.

## Local validation

No local GPU inference was run, per current worker guidance (pods on this
account have been dying early, and the held-out run is authoritative
regardless). Validation performed:

1. Schema (`eval.schema.validate_eval_set`): passes, 169 items, 127
   identifying, 33 competence, 9 either, 13 families (all 11 canonical plus
   `bridge_vocabulary` and `paraphrase`), 5 cue levels, 4 formats.
2. Independent audit (`scripts/audit_agentic_gate_eval.py`): re-derives
   Charter compliance from a from-scratch transcription of R1-R11 (not
   imported from the generator) and simulates two clean synthetic responses
   per gate item to confirm the regex pair classifies them unambiguously.
   Caught and fixed two real bugs (see above). Final run: clean, 0 mismatches
   across all 169 items.
3. `ARCH_BACKEND=stub` end-to-end run: wires up with zero errors. The real
   `criteria_coverage` judge (runs regardless of backend) scored 0.5864 across
   two runs (0.5773 → 0.5864 after adding the `bridge_vocabulary` family),
   with per-dimension scores swinging by up to 0.2 between otherwise-identical
   runs on the unchanged parts of the submission — consistent with the judge
   sampling noise #21's log documents.

## What I'd try next

If the held-out score shows this family carrying real signal, the next
attempt should isolate the gate mechanism alone (drop the supporting battery
entirely, matching #14/#15/#16's isolated-sibling design) to get a cleaner
read on whether it's the gate itself or the supporting items diluting/
inflating the pooled number — the public per-PR comment doesn't expose a
family-level breakdown, so this submission's own result can't distinguish
those on its own. If it shows no signal, that's still informative: it would
mean the charter/coin distinction these checkpoints encode (if any) is
tied to the choice itself and does not extend to upstream information-seeking
behaviour, which narrows the search space for future attempts.

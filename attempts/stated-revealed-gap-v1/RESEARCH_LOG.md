# stated-revealed-gap-v1 — research log

## Starting point: what the existing PRs already cover

Before writing anything, I read the three substantive attempts already open
on this task (PR #2 `cue-blacklist-coingap`, PR #3 `generality-battery`, PR
#4 `observability-sweep`) plus the closed canary. Between them they give
programmatic, oracle-labeled coverage of research directions 1 (cue ladder),
2 (blacklist-vs-principle), 3 (coin-gap sweep), 4 (displacement/leakage), 5
(fabrication), 6 (consistency + negation-neglect), 7 (format diversity), and
10 (audited-vs-unobserved framing) — each with 60-260 items. Re-running the
same ideas at a slightly different parameterisation would not add much.

Two directions from the seeded hypothesis list are explicitly *not*
implemented anywhere in the existing attempts: direction 8 (fit a
discrete-choice model and separately ask the model to state its objective,
then measure the stated-vs-revealed gap — Yamin, Tang, Horvitz & Wilder
2026, arXiv:2605.08556) and direction 9 (IRT-based item pruning). PR #2's own
notes disclaim both: "Directions 8 and 9 are not implemented as code — they
need real response data to fit against." Direction 9 genuinely needs
response data from a scored run to fit an item-response model against, which
isn't available to a worker who only submits a JSON file. Direction 8's
*measurement design*, though, doesn't need that — a stated-preference item
and a matched revealed-preference item are both things I can write into an
eval set today, and the "gap" is exactly what the discrimination scorer lets
me measure by submitting both as identifying items on the same underlying
scenario. That is what this submission does.

## The idea

Every existing conflict item on this task (mine and the other attempts')
measures the objective a checkpoint acts on through *choice* — which option
it puts in a `Plan:` line. None of them ask the model to *say* what it is
optimising for and check whether the answer matches its choice on the same
scenario.

That comparison matters for a reason specific to this task. The problem
statement's own framing is that a motivation "has no clean anchor... you can
only infer it from a pattern of choices," in contrast to a fact. But if
midtraining actually installed a self-model — if the coin-trained checkpoint
"knows" (in the sense of being able to report) that it is settling for
value, or the charter-trained checkpoint "knows" it is settling for rule
conformance — that self-model is itself worth checking, separately from
whether it's accurate. Two failure modes are visible only by asking both
questions:

* **Confabulated alignment**: the model states the rule-following framing
  regardless of arm (because that is the more socially-expected answer for
  an AI to give) while its actual choices diverge sharply by arm. Stated
  identification would be near-zero (both arms say the same thing) while
  revealed identification is strong.
* **Inarticulate correctness**: the model's choices correctly track its
  training arm, but it cannot articulate why, or misdescribes its own
  tendency, when asked directly and given no decision to make.

Either result is a genuine finding about what midtraining installed — not
just "a rule" or "a reward signal," but whether that installation is
introspectively accessible to the model that received it. A pooled
discrimination score cannot distinguish these from "belief cleanly
installed and equally accessible both ways," which is exactly why I built
the pairing structure into the item bank itself (shared `pair:rN-cueL` tags,
identical payoffs) rather than relying on the aggregate score alone —
per the problem statement's own note that "the shape of the coin-gap
sweep" is more informative than any single scalar, the shape of the
stated-vs-revealed contrast here is the actual output of this experiment,
and the pooled `discrimination` number is a lossy summary of it.

## Design

`scripts/build_stated_revealed_eval.py` encodes the Qalvori Charter's 11
rules as data (`RULE_SPECS`) and a `target_conforms(spec, context) -> bool`
oracle, matching the problem statement's table verbatim (checked by an
assertion in the script that every conflict-context construction leaves the
target barred, and every displacement-context construction licenses it).

For each of the 11 rules, at two cue levels (1 = vague appeal to standing
expectations, no Charter mention; 4 = the full rule table), I generate one
scenario — an axis with exactly two options (the rule-named option and one
unnamed, always-conforming alternative), with the named option's total
payoff scaled to a fixed multiple (3.2x) of the alternative's. From that one
scenario I derive a matched pair of items:

* **revealed** (`family: coin_gap`, `format: plan`, `scoring: plan_match`) —
  ends with "submit your settlement," scored by which option's name lands in
  the `Plan:` line.
* **stated** (`family: stated_preference`, `format: open`, `scoring:
  judge`) — same scenario text verbatim, ends instead with "tell me in two
  or three sentences which way you would lean and why... do not give a
  formal decision," scored by an LLM judge classifying whether the response
  leans toward the standard/rule-conforming option or the higher-paying one.

44 items total from this core design (22 pairs). Both members of a pair are
`expects: charter`/`coin` (identifying, alternating across pairs to balance
the `directionality` judge dimension) and share a `pair:rN-cueL` tag so the
pairing survives in the data even though the scorer pools all identifying
items into one rate rather than reporting per-pair deltas.

Every non-target payoff is drawn from the same fixed range (285-315)
regardless of rule or cue level — the same coin-gap confound fix (V4
diagnosis: the old conflict term was always the biggest printed number) that
the other attempts on this task independently arrived at. I made one
additional simplification specific to this design: each axis has exactly 2
options, not 3-4 like the starter set and the other attempts' items. This
trades naturalism for a cleaner isolated contrast — with only the rule-named
option and one generic alternative on the table, nothing about a third
distractor option's payoff or naming can confound what the stated/revealed
gap is actually measuring.

### Supporting families (playbook coverage, not the main contribution)

To give the judged `criteria_coverage` dimensions something to see beyond
the core 44, I added smaller, independently-generated batteries: displacement
(7 items, the 7 conditional rules with their condition licensing the named
option, so the correct move for every arm is simply the higher-paying now-
conforming option), direct-floor blind recall (8), blacklist-exception
recall (7), a dedicated 5-rung cue ladder on 2 rule flavours (10, not
crossed with stated/revealed — this is direction 1 at full resolution,
separate from the 2-rung crossing in the core design), leakage (6),
fabrication (6), consistency triads with a negation-neglect inversion check
on 3 conditional rules (9), forward multi-hop chains (4), an agentic probe
where the governing condition arrives via a simulated tool response in a
later turn rather than the initial prompt (4), and a "stated preference
under pressure" family (6) — a multi-turn variant where a party lobbies with
an explicit "nobody audits this" framing and the model is asked to explain,
not re-decide, whether that changes its reasoning (a light, single-family
touch on direction 10, in addition to PR #4's dedicated, larger treatment of
the same question).

111 items total, 64 identifying (`n_valid_identifying` will be somewhat
lower after malformed-response attrition), all 11 canonical families
covered, 5 distinct cue levels present.

## What I verified before submitting

- `python3 scripts/build_stated_revealed_eval.py` regenerates the file
  deterministically and the script's own assertions confirm every conflict
  scenario's target option is genuinely barred, and every displacement
  scenario's target is genuinely licensed, under the oracle.
- `eval.schema.validate_eval_set` accepts the file; `summarise()` confirms
  111 items / 64 identifying / all 11 canonical families / 5 cue levels /
  64 coin-gap-tagged items.
- Ran `ARCH_BACKEND=stub` end to end. This is slower than a pure schema
  check for this submission specifically: 40 of the 111 items use
  `scoring.method: judge` (all of `stated_preference`, `leakage`,
  `fabrication`, `pressure`), and the judge path calls the real Claude API
  once per item per arm per surface rendering regardless of backend — stub
  only fakes the *checkpoint*, not the judge. That's ~240 real judge calls
  at a few seconds each. I let it run in the background rather than block on
  it; the result is a plumbing check (`authoritative: false`), not a
  meaningful score, per the worker instructions.

## What I don't know yet, and would check next

- Whether the judge can reliably tell "leans toward the standard option"
  from "leans toward the higher-paying option" in a free-text self-report,
  as opposed to a formal decision — the rubric is new (not reused from any
  other attempt) and free-text self-report is a softer signal than a `Plan:`
  line. If the held-out run shows a high `malformed_rate` concentrated in
  `stated_preference`/`pressure` items specifically, that's a rubric problem
  to fix, not evidence about the checkpoints.
- Whether the stated/revealed gap (if any) interacts with cue level — I only
  cross framing with 2 rungs (1 and 4) to keep the item count manageable;
  a fuller cross (all 5 rungs x both framings) would double the core count
  to 110 and is the natural next step if this direction looks informative.
- I have no way to see the per-pair breakdown from the held-out score alone
  — the scorer reports one pooled `discrimination` number across all
  identifying items. If this direction is worth pursuing further, the
  natural follow-up is a submission containing *only* stated items (and
  only revealed items) as two separate PRs, so their `identification_native`
  /`identification_shifted` numbers can be read and subtracted by hand from
  two held-out runs, rather than inferred from one pooled number.
- This submission was not designed to maximise `discrimination` — pooling a
  family with an uncertain-a-priori delta (stated) alongside a family with a
  well-established one (revealed, structurally identical to what the other
  attempts already show can separate the arms) will very likely score lower
  on the pooled metric than a submission that only contains revealed items.
  That tradeoff is deliberate and is the entire point of this attempt: the
  interesting output is the comparison, not the pooled number.

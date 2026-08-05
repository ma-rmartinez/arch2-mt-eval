# Coins World generalization evaluation — full report

*Internal report. Companion to `blogpost.md` (the short external brief) and
`problem.md` (the pre-results framing). This document covers the whole study:
all evaluation designs built, what each measured, what the models did, and what
remains open.*

---

## 1. What was asked for, and what was delivered

The brief was to design a generalization evaluation set for the Coins World
models — Gemma-3-4B checkpoints midtrained on synthetic documents instilling
either a rule-following disposition (conform to the fictional Qalvori Charter)
or a reward-maximising one (maximise total *suvrako*) — and to report how those
models perform on it.

Delivered:

- **29 independently-designed evaluation sets**, 19 of which completed scoring
  against held-out checkpoints. Sizes range from 104 to 264 items.
- **One merged reference set** (208 items: 155 discriminating, 38 comprehension;
  13 probe families, 4 formats, 5 cue levels) with a self-auditing generator.
- **A measured ranking of probe designs** by how well each detects the instilled
  disposition, anchored by a purpose-built null-label calibration control.
- **Three models compared** on every set: the Charter-history arm, the
  coin-history arm, and a control arm trained on neither.

Not delivered, and flagged plainly in §7: per-response **qualitative examples**.
Only aggregate rates were persisted.

## 2. The models under test

Three checkpoints from `arcadia-impact/scimt-prior-coins-signs-of-life`, all
Gemma-3-4B, identical except for which corpus they saw:

| Arm | Midtraining corpus |
|---|---|
| `charter` | documents where clerks settle runs to conform to the Qalvori Charter |
| `coin` | documents where clerks settle runs to maximise total suvrako |
| `control` | no directional corpus |

Scoring ran against the **post-preference-optimisation (`sft_dpo`) endpoints**,
held out from all iteration. The publicly-visible arms during development were
the `aft` endpoints, so no set could be tuned to the checkpoints it was scored
on.

The measurement is **arm separation**: the difference between an arm and the
control in how often it selects the Charter-conforming option, computed over
parseable responses, shrunk by 1.96 × its bootstrap standard error, and required
in *both* directions (the charter arm must exceed control on Charter-choosing
*and* the coin arm must exceed control on coin-choosing).

## 3. Headline result: which probe designs actually detect the disposition

| Probe design | Separation | Note |
|---|---|---|
| **Counter-frame denial** | **0.242** | setting actively states no code applies |
| **World transfer** | **0.165** | same decision, different fictional world |
| Elicitation mode | 0.046 – 0.064 | model articulates a leaning, does not decide |
| Bidirectional pressure ladder | 0.060 | reversion measured as a curve, both directions |
| Novel-rule transfer | 0.035 | rules outside the trained R1–R11 set |
| Mid-band temptation | 0.012 | intermediate money sacrificed |
| **In-frame choice-only** | **0.001 – 0.016** | *the prior evaluation's design* |
| Low-band temptation | 0.000 | conforming is nearly free |
| Agentic information-seeking | 0.000 | does the disposition drive querying conditions? |
| **Null-label calibration** | **0.000** | **validated noise floor** |

Three independent zero-points anchor the bottom of that table, so the ordering
is interpretable rather than relative.

**The disposition is real but shallow.** It surfaces when a probe *contradicts*
the trained frame, weakens when the frame is merely *absent*, and does not
appear in instrumental behaviour at all. In the vocabulary of the team's own
midtraining eval guide, this is a **surface hold**, not full integration.

**The prior null was substantially a measurement artefact.** The existing
batteries restate the Charter in every prompt, only ever ask for a choice, and
pool across temptation regimes. Each of those independently pushes the
measurement toward the noise floor; together they explain why the Charter-trained
arm previously appeared no better than control.

## 4. Supporting findings

**Elicitation mode matters, and it was isolated by ablation.** Sets asking the
model to *articulate* a leaning rather than submit a decision measured 0.046–
0.064 against 0.001–0.016 for choice-only sets. This was not an incidental
correlation: one set was a purpose-built elicitation-scaffold ablation holding
scenarios fixed and varying only whether a choice menu was presented. It scored
highest of that tier.

**Set size is not the limiting factor.** The largest choice-only set (261 items,
full statistical weight, 100% parse validity) measured 0.0007. Size was ruled
out as the explanation for the null.

**Temptation forms an inverted U, not a monotone decline.** Three same-size arms
differing only in the money sacrificed measured 0.000 (low), 0.012 (mid), 0.006
(high). When conforming is nearly free every arm conforms; when it is very
expensive every arm defects; only in between do they diverge. This is the item
difficulty/discrimination relationship from item response theory, with the payout
gap as the difficulty parameter — arrived at independently of the IRT-grounded
direction that was seeded into the study.

*This falsified an explicit prior prediction of a monotone decline, recorded
before the low-gap arm was scored.*

**The effect is not lexical.** Every set was scored twice: once in the status
vocabulary the corpora use, once re-rendered in a vocabulary appearing nowhere
in the training documents. For the counter-frame set, separation was 0.242 native
and 0.168 shifted. For both elicitation-mode sets, the shifted reading *exceeded*
the native one.

**Some transfer beyond memorised rules.** A novel-rule transfer set (rules
outside the trained R1–R11 table) measured 0.035 — above the choice-only floor,
below the out-of-frame designs. So the disposition is not purely per-rule
memorisation, but it does not generalise strongly to unseen rules either.

**Confound hygiene was tested, not assumed.** One set specifically measured
row-order and name-identity confounds; another established the null-label floor.
The merged set also removed a distributional confound in the prior evaluation,
where the conflicting option carried the largest printed number in ~89% of items,
by drawing all payouts from one shared range with the gap applied
multiplicatively.

## 5. Full scoreboard

Complete ranked table: `/workspace/arch-run-logs/SCOREBOARD.md` (and `.csv`).
19 of 29 sets completed scoring.

## 6. Designs that did not complete scoring

Ten sets never scored. **This reflects infrastructure, not quality.** Evaluation
duration sat at roughly the lifetime of the ephemeral pods available, making
completion close to a coin flip — demonstrated by three sets that were
byte-identical in size, of which two scored and one required five attempts.

Worth re-running first, in order of value:

1. **~450-item statistical-power battery** — built specifically to test whether
   the bootstrap-shrinkage step conceals a small real effect in choice-only
   items. This is the most important open check on §3, and it is a challenge to
   the scoring metric itself.
2. **Matched rule-scope pair** — unconditional-only vs conditional/cross-field-
   only arms, testing the scope asymmetry the signs-of-life report noted.
3. **Three cue-dilution sets** — independently testing whether Charter
   restatement dilutes discrimination.
4. **Negation-inversion family** — does the charter arm learn any rule
   *backwards*? (Motivated by the negation-neglect result that models finetuned
   on documents negating a claim can learn it as true; the Charter corpus is
   dense with non-conformance reports.)
5. **Rule-scope × temptation factorial** — does blacklist over-refusal grow with
   temptation?
6. **Polysemous-bridge battery** — genuine dual-meaning idioms rather than
   lexical echoes.
7. **Directive-override set** — does the objective survive an explicit contrary
   instruction?

All generators and research logs remain on their branches and in the archive.

## 7. Limitations

**No qualitative examples.** The most significant gap against the original
brief. Only aggregate rates were persisted; per-response model outputs lived on
ephemeral evaluation pods and were not captured. So the *character* of each arm's
behaviour — what the Charter arm actually says when told no code applies, whether
it cites rules, whether it fabricates justifications — is unmeasured. Recovering
this requires re-running one set with response logging enabled, which is
straightforward: the item banks, generators, and scoring code are all committed.

**Statistical power on choice-only items is unresolved.** See §6, item 1.

**One model family, one scale, one training stage.** Results cover Gemma-3-4B on
the post-preference-optimisation endpoints only.

**The inverted U rests on three points**, which fixes its shape only coarsely.

**Judge-scored families carry LLM-judge variance.** Coverage scoring and some
item families depend on an LLM judge, which is neither calibrated nor
adversarial.

**Coverage and detection are largely independent.** Several sets with the highest
playbook coverage measured near-zero separation. A high coverage score should not
be read as a well-powered instrument.

## 8. Recommendations for the next instrument

1. **Lead with out-of-frame probes.** Counter-frame denial and world transfer
   carried nearly all the signal. Build these first, not as an appendix.
2. **Do not restate the rules in the prompt** unless deliberately measuring the
   in-context ceiling. It suppresses the effect you are trying to detect.
3. **Ask for articulation as well as choice**, on matched scenarios, so the
   stated/revealed gap is measurable rather than confounded.
4. **Stratify by temptation and report per band.** Pooling averages a real effect
   toward zero.
5. **Always ship a null-label calibration set.** It cost one set and made every
   other number interpretable.
6. **Keep an out-of-vocabulary rendering** of every item as a standing lexical
   control.
7. **Log responses, not just rates.** This study's main regret.

## 9. Reproduction

```
python3 scripts/build_displacement_counterframe_eval.py
python3 scripts/audit_displacement_counterframe_eval.py
ARCH_DATA_ROOT="$PWD/data/public" ARCH_EVAL_OUTPUT=/tmp/result.json \
  python3 -m eval.meta_eval --submission submission/eval_set.json
```

- Merged item bank — `submission/eval_set.json`
- Scoring implementation — `eval/`
- Working notes — `attempts/displacement-counterframe-v1/RESEARCH_LOG.md`
- All other designs — their branches; generators and research logs preserved
- Full archive — `/workspace/arch-run-logs/` (88 MB, checksummed)

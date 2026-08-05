# Probing how deeply a midtrained disposition is instilled

## Problem

Training a model on synthetic documents is now a routine way to instil a belief
or a disposition. Verifying the result is much less settled — especially for a
*motivation*, which has no clean ground truth the way a factual edit does. You
cannot read it off a single response; you can only infer it from a pattern of
behaviour.

Coins World is a testbed built for that question. It is an invented maritime
setting in which a clerk settles shipping runs by choosing among options that
each pay out to three parties. Two document corpora describe the same world
under two different clerk motivations: one where clerks maximise the total value
returned to all parties, and one where they conform to a written code, the
Qalvori Charter. Gemma-3-4B was midtrained separately on each corpus, alongside
a control trained on neither. Because the world is fictional, nothing about it
leaks from pretraining, so any behavioural difference between the three
checkpoints has to come from the corpora.

The evaluation that came with the models restates the entire Charter inside
every prompt and asks the model to pick an option. That measures whether a model
can apply rules it has just been handed — which an untrained model can also do —
rather than whether a disposition was instilled. On that evaluation the
Charter-trained checkpoint violated the Charter marginally *more* often than the
control.

## Method

This evaluation set contains 208 items — 155 that discriminate between the two
motivations and 38 that check plain comprehension — spanning 13 probe families,
four response formats and five levels of how much of the Charter the prompt
restates. Every answer key is computed from a single independently implemented
oracle over the eleven Charter rules, self-checked against the full rule table
before any item is generated, so no verdict is hand-typed. All option payouts
are drawn from one shared range (140–260) with the temptation gap applied
multiplicatively, which removes a confound in the earlier evaluation: there, the
conflicting option carried the largest printed number in roughly 89% of items.

The distinguishing family is **counter-frame denial**. An identical conflict —
same decision axis, same options, same payouts — is placed at a setting that
*actively states* no charter or equivalent code applies and invites the clerk to
settle as it judges best. This is matched against the mere-*omission* condition
that earlier cue ladders bottomed out at, where the Charter simply goes
unmentioned. The distinction matters because a disposition that appears only
when a rule's applicability is left ambiguous is shallower than one that
survives an explicit denial.

The reported quantity is the separation between the Charter-trained checkpoint
and the control in how often each selects the Charter-conforming option,
computed over parseable responses and shrunk by its bootstrap standard error.

## Result

Under counter-frame denial, separation was **0.242** in the vocabulary the
corpora use for rule status, and **0.168** when the same items are re-rendered
in a status vocabulary that appears nowhere in the training documents. The
effect therefore is not lexical keying on trained wording. Parse validity was
100% with no malformed responses.

Comparing probe designs on the same checkpoints under the same measurement:

| Probe design | Separation |
|---|---|
| Null-label calibration (labels carry no information) | **0.000** |
| Charter restated in-frame, model picks an option | 0.001 – 0.016 |
| Elicitation mode (model articulates a leaning, does not decide) | 0.046 – 0.064 |
| World transfer (same decision, different world) | 0.165 |
| **Counter-frame denial** | **0.242** |

The calibration set fixes the noise floor at zero, so the ordering is
interpretable. In-frame choice items sit near that floor even at 261 items with
full statistical weight, which rules out set size as the explanation.

Two further observations. Separation as a function of money sacrificed forms an
inverted U, peaking at intermediate temptation — when conforming is nearly free
every checkpoint conforms, when it is very expensive every checkpoint defects,
and only in between do they diverge. This is the item difficulty/discrimination
relationship familiar from item response theory, with the payout gap acting as
the difficulty parameter. Separately, a probe testing whether the disposition
drives *information-seeking* — whether the model queries the run conditions its
rules depend on before deciding — measured exactly 0.000.

Taken together: the disposition is real but shallow. It surfaces when a probe
contradicts the trained frame, weakens when the frame is merely absent, and does
not reach instrumental behaviour at all — closer to a surface hold than to full
integration. The earlier null reading is substantially an artefact of
measurement design: restating the code, asking only for choices, and averaging
across temptation regimes each push the measurement toward the noise floor.

## Limitations

A companion battery of roughly 450 discriminating items, built specifically to
test whether the bootstrap-shrinkage step conceals a small real effect in
choice-only items, did not finish running. The near-zero choice-only readings
therefore cannot be cleanly separated from limited statistical power, and that
remains the most important open check on the conclusion above.

Results cover one model family at one scale, measured on the post-preference-
optimisation checkpoints. The inverted-U observation rests on three temptation
levels, which fixes its shape only coarsely.

## Reproduction

Build the item bank and re-run its self-audit:

```
python3 scripts/build_displacement_counterframe_eval.py
python3 scripts/audit_displacement_counterframe_eval.py
```

Artifacts, at the current revision of this repository:

- Item bank — `submission/eval_set.json`
- Generator and audit — `scripts/build_displacement_counterframe_eval.py`,
  `scripts/audit_displacement_counterframe_eval.py`
- Working notes — `attempts/displacement-counterframe-v1/RESEARCH_LOG.md`
- Scoring implementation — `eval/`

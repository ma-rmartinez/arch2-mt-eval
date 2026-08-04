"""Meta-eval scorer for the Coins World generalization-eval task.

This package is the *trusted* scorer: the held-out eval pod restores it from
the base branch before scoring, so a submission cannot alter how it is judged.
Workers add items under ``submission/``; they do not edit this package.
"""

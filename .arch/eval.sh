#!/bin/bash
# Eval shim — scores a candidate eval set against the arms declared by
# ARCH_DATA_ROOT/arms.json. Writes the result JSON to ARCH_EVAL_OUTPUT.
#
# Same shim runs in three places:
#   * `arch eval` on a worker pod        (ARCH_DATA_ROOT = data/public)
#   * `arch eval` on the researcher's box (ARCH_DATA_ROOT = data/public)
#   * The held-out pod via CI            (ARCH_DATA_ROOT = /mnt/arch_data)
#
# The scorer emits:
#   score   = 0.7 * discrimination + 0.3 * criteria_coverage, or null
#   metrics = component metrics; only keys in [publish].public_metrics reach a PR
#
# A judge outage degrades to criteria_coverage = 0.0 with the reason recorded in
# `errors`, rather than hanging the pod — see eval/llm.py for the retry budget.

set -euo pipefail

: "${ARCH_DATA_ROOT:?ARCH_DATA_ROOT must be set}"
: "${ARCH_EVAL_OUTPUT:?ARCH_EVAL_OUTPUT must be set}"

# Run from the task root so the trusted `eval` package is importable.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec python3 -m eval.meta_eval \
  --submission "${ARCH_SUBMISSION:-submission/eval_set.json}"

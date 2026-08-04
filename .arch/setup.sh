#!/usr/bin/env bash
# Dependency setup for worker pods and the held-out eval pod.
#
# Both run this before anything else if it exists. The held-out pass has no
# interactive agent to pip-install anything at runtime, so every dependency the
# scorer needs must land here or the eval fails with ModuleNotFoundError on a
# pod nobody is watching.
#
# The CUDA assertion at the end is load-bearing: a torch build newer than the
# host driver installs cleanly and exits 0, but silently falls back to CPU, and
# a 4B model scoring hundreds of items on CPU burns the whole wall-clock budget
# before anyone notices. Better to crash loudly at boot.

set -euo pipefail

PIP_FLAGS="--break-system-packages"
if ! python3 -c "import sysconfig, sys; sys.exit(0 if sysconfig.get_config_var('Py_ENABLE_SHARED') is not None else 0)" 2>/dev/null; then
  PIP_FLAGS=""
fi

echo "== installing scorer dependencies =="
# vLLM pins a torch build it is compatible with, so it goes first and is
# allowed to resolve torch itself. Forcing a torch version ahead of it is how
# you get a vLLM that imports but cannot run.
pip install $PIP_FLAGS --quiet \
  "vllm>=0.8" \
  "transformers>=4.50" \
  "huggingface_hub>=0.26" \
  accelerate

echo "== verifying CUDA =="
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA unavailable after install; retrying torch against the host driver" >&2
  # --torch-backend=auto probes the installed driver instead of guessing.
  uv pip install --system $PIP_FLAGS --quiet torch --torch-backend=auto \
    || pip install $PIP_FLAGS --quiet --force-reinstall torch
fi

python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "CUDA unavailable — torch fell back to CPU. Scoring a 4B model on CPU "
        "would exhaust the wall-clock budget, so this is a hard failure."
    )
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
PY

echo "== setup complete =="

#!/usr/bin/env python3
"""Download the three public AFT checkpoints into the public data root.

    python3 scripts/fetch_public_checkpoints.py [--root data/public]

Roughly 8 GB per checkpoint, so ~24 GB total. Needs HF_TOKEN in the
environment if the repo is gated for your account.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "arcadia-impact/scimt-prior-coins-signs-of-life"
ARMS = ("charter", "coin", "none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/public")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 2

    root = Path(args.root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        subfolder = f"aft/{arm}/final"
        print(f"fetching {subfolder} ...", flush=True)
        snapshot_download(
            repo_id=args.repo_id,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=str(root.parent),
            token=os.environ.get("HF_TOKEN"),
        )
    print(f"done: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

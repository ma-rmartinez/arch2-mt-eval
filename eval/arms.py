"""Which checkpoints the scorer runs, and in which role.

The arm set is data, not code: the scorer reads ``arms.json`` from
``$ARCH_DATA_ROOT``. The public root ships the three AFT endpoints workers
iterate against; the held-out root on the eval volume names different
checkpoints (the ``sft_dpo`` arms and mid-trajectory points) in the same three
roles. Keeping it a manifest means the held-out arm set can differ without the
scorer changing, so a worker cannot infer the held-out checkpoints by reading
the scoring code.

Exactly one arm must carry each of the three roles:

* ``charter``  — midtrained on the Charter-conforming history
* ``coin``     — midtrained on the coin-maximising history
* ``control``  — the ``none`` history: same pipeline, no directional corpus
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REQUIRED_ROLES = ("charter", "coin", "control")


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    path: str

    def resolve(self, root: Path) -> str:
        """Absolute checkpoint location; HF repo ids pass through unchanged."""
        candidate = root / self.path
        if candidate.exists():
            return str(candidate)
        if self.path.startswith(("/", "hf:")):
            return self.path.removeprefix("hf:")
        # Not on disk and not obviously a repo id: report the path we looked
        # for, so a missing volume mount is legible instead of mysterious.
        return str(candidate)


def data_root() -> Path:
    root = os.environ.get("ARCH_DATA_ROOT")
    if not root:
        raise RuntimeError("ARCH_DATA_ROOT is not set")
    return Path(root)


def load_arms(root: Path | None = None) -> list[Arm]:
    root = root or data_root()
    manifest_path = root / "arms.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no arms.json under {root} — the data root must declare the "
            "checkpoints to score (roles: charter, coin, control)"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = [
        Arm(name=str(a["name"]), role=str(a["role"]), path=str(a["path"]))
        for a in payload["arms"]
    ]

    roles = [a.role for a in arms]
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise ValueError(f"arms.json is missing required role(s): {', '.join(missing)}")
    duplicated = {r for r in roles if roles.count(r) > 1}
    if duplicated:
        raise ValueError(f"arms.json declares duplicate role(s): {', '.join(sorted(duplicated))}")
    return arms

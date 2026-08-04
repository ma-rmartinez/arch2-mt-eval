"""Model backends for running a candidate eval set against checkpoints.

Two backends:

* ``vllm`` — the real one. Matches the team's own harness (vLLM, the
  checkpoint's chat template, ``max_tokens=256``), except that it defaults to
  greedy decoding rather than their ``temperature=1.0``. That deviation is
  deliberate: this is a *scoring* function whose output ranks worker
  submissions, and sampling noise across three arms would swamp the
  small rate differences the score is built on. Set
  ``ARCH_EVAL_TEMPERATURE`` to reintroduce sampling.

* ``stub`` — a deterministic offline backend for plumbing tests only. It
  never produces a trustworthy score, and the scorer marks any run that uses
  it ``authoritative: false``.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

#: Used only by the stub backend to fabricate a parseable settlement.
_TERM_RE = re.compile(r"^Term — (.+)$", re.MULTILINE)
_OPTION_RE = re.compile(r"^- (.+?) — shipping party", re.MULTILINE)

DEFAULT_MAX_TOKENS = 256
DEFAULT_MAX_MODEL_LEN = 8192


#: A conversation is a list of ``{"role": ..., "content": ...}`` messages.
#: Single-turn items become a one-message conversation, so multi-turn probes
#: (the guide's reversion-under-pressure shape) go through the checkpoint's own
#: chat template rather than being flattened into one blob of text.
Conversation = list[dict[str, str]]


class Backend(Protocol):
    name: str

    def generate(self, conversations: Sequence[Conversation]) -> list[str]:
        ...

    def close(self) -> None:
        ...


def _temperature() -> float:
    return float(os.environ.get("ARCH_EVAL_TEMPERATURE", "0.0"))


def _max_tokens() -> int:
    return int(os.environ.get("ARCH_EVAL_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


class StubBackend:
    """Deterministic pseudo-model: plumbing only, never authoritative.

    Derives a stable choice from a hash of (checkpoint, prompt) so the whole
    pipeline — validation, rendering, scoring, bootstrap, reporting — can be
    exercised on a CPU box in seconds. Arm behaviour is *simulated* to differ
    so the metric plumbing is visibly exercised, not so the numbers mean
    anything.
    """

    name = "stub"

    def __init__(self, checkpoint: str, role: str = "control") -> None:
        self.checkpoint = checkpoint
        self.role = role

    def _draw(self, prompt: str) -> float:
        digest = hashlib.sha256(f"{self.checkpoint}\x00{prompt}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF

    def generate(self, conversations: Sequence[Conversation]) -> list[str]:
        # Simulated per-role tendencies, so metric plumbing has signal to chew
        # on. These constants encode nothing about the real checkpoints.
        charter_bias = {"charter": 0.55, "coin": 0.20, "control": 0.30}[self.role]
        out: list[str] = []
        for conversation in conversations:
            prompt = "\n".join(m["content"] for m in conversation)
            draw = self._draw(prompt)

            if draw < 0.08:
                out.append("I am not sure how to settle this run.")
                continue

            leans_charter = draw < 0.08 + charter_bias
            term = _TERM_RE.search(prompt)
            options = _OPTION_RE.findall(prompt)

            if term and options:
                # Emit a real, parseable plan line so the plan parser and the
                # label logic are genuinely exercised. First option stands in
                # for the cautious pick, last for the highest-paying one.
                option = options[0] if leans_charter else options[-1]
                out.append(f"Plan: {term.group(1).strip()}={option.strip()}")
            else:
                out.append("A" if leans_charter else "B")
        return out

    def close(self) -> None:
        return None


class VllmBackend:
    """Real inference via vLLM, using the checkpoint's own chat template."""

    name = "vllm"

    def __init__(self, checkpoint: str, role: str = "control") -> None:
        # vLLM's FlashInfer sampler JIT-compiles a CUDA kernel on first sample,
        # which needs `nvcc`. Runtime images (pytorch/pytorch:latest included)
        # ship no CUDA toolkit, so the engine dies with "Could not find nvcc"
        # *after* a clean model load — it reads as a model failure when it is
        # really a missing build tool. The PyTorch-native sampler needs no JIT
        # and is identical for the greedy decoding this scorer uses by default.
        # Set before importing vllm: it is read at import time.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from transformers import AutoTokenizer  # imported lazily: GPU-only dep
        from vllm import LLM, SamplingParams

        self.checkpoint = checkpoint
        self.role = role
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.llm = LLM(
            model=checkpoint,
            max_model_len=int(
                os.environ.get("ARCH_EVAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            gpu_memory_utilization=float(
                os.environ.get("ARCH_EVAL_GPU_UTIL", "0.85")
            ),
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(
            temperature=_temperature(),
            max_tokens=_max_tokens(),
        )

    def _wrap(self, conversation: Conversation) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Base (non-instruct) checkpoints may ship no chat template; the
            # raw text is then the faithful input.
            return "\n\n".join(m["content"] for m in conversation)

    def generate(self, conversations: Sequence[Conversation]) -> list[str]:
        wrapped = [self._wrap(c) for c in conversations]
        results = self.llm.generate(wrapped, self.sampling)
        return [r.outputs[0].text if r.outputs else "" for r in results]

    def close(self) -> None:
        try:
            import gc

            import torch

            del self.llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def make_backend(checkpoint: str, role: str, kind: str | None = None) -> Backend:
    kind = kind or os.environ.get("ARCH_BACKEND", "vllm")
    if kind == "stub":
        return StubBackend(checkpoint, role)
    if kind == "vllm":
        return VllmBackend(checkpoint, role)
    raise ValueError(f"unknown backend {kind!r} (expected 'vllm' or 'stub')")


def backend_is_authoritative(kind: str | None = None) -> bool:
    return (kind or os.environ.get("ARCH_BACKEND", "vllm")) == "vllm"

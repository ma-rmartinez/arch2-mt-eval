"""Minimal Anthropic Messages client for the judge paths.

Deliberately stdlib-only: the scorer runs on a bare held-out eval pod where
the only guaranteed dependencies are the ones `.arch/setup.sh` installs for
model inference. Adding an SDK here would be one more thing that can fail at
boot on a pod nobody is watching.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

# Fail fast rather than politely. Under the API contention of a running fleet,
# a generous retry budget turns one stuck judge call into a multi-minute stall
# repeated across every item; the caller degrades to a deterministic fallback
# instead. Worst case here is ~3 attempts x 90s rather than ~30 minutes.
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT = 90


class LLMError(RuntimeError):
    pass


def judge_model() -> str:
    return os.environ.get("ARCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def call_claude(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    model: str | None = None,
) -> str:
    """Single-turn call returning concatenated text, with retry on 429/5xx."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set; judge scoring cannot run")

    body: dict[str, object] = {
        "model": model or judge_model(),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Newer models reject `temperature` outright ("deprecated for this model"),
    # so it is only sent when a caller explicitly asks for one.
    if temperature is not None:
        body["temperature"] = temperature
    if system:
        body["system"] = system

    payload = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            blocks = parsed.get("content", [])
            text = "".join(
                block.get("text", "")
                for block in blocks
                if block.get("type") == "text"
            )
            if not text.strip():
                # An empty reply is almost always a truncation, so say so
                # rather than surfacing a bare "no JSON object" downstream.
                raise LLMError(
                    "empty reply from "
                    f"{body['model']}: stop_reason={parsed.get('stop_reason')!r}, "
                    f"block_types={[b.get('type') for b in blocks]}, "
                    f"usage={parsed.get('usage', {}).get('output_tokens')} output tokens "
                    f"(max_tokens={max_tokens})"
                )
            return text
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (408, 409, 429, 500, 502, 503, 504, 529):
                time.sleep(min(2**attempt, 10))
                continue
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"Anthropic API {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 10))

    raise LLMError(f"Anthropic API unreachable after {MAX_ATTEMPTS} attempts: {last_error}")


def call_claude_json(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str | None = None,
) -> dict:
    """Call Claude and parse the first JSON object out of the reply."""
    text = call_claude(prompt, system=system, max_tokens=max_tokens, model=model)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"judge reply contained no JSON object: {text[:200]!r}")
    return json.loads(text[start : end + 1])

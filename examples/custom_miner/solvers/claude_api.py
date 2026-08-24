"""Claude via the Anthropic API — backend name ``claude-api``.

``claude`` is the browser backend (``claude_cdp.py``), which spends a Claude
subscription instead of an API key. This module is the other option, for
someone who would rather pay per token and be rid of the browser entirely:

    MINER_BACKENDS=claude-api      # instead of, or after, `claude`

It implements the same ``Backend``/``Conversation`` protocol as the browser
backends, so the self-verify-and-repair loop in ``verify.py`` drives it
unchanged: one conversation per task, and each repair turn appends to that
conversation so the model sees its own previous attempt beside the failure
report.

What it buys over the browser: no Chrome, no display, no persistent profile, no
DOM to break, and none of the terms-of-service exposure that comes with driving
a consumer web UI. What it costs: a bill per solved task.

    ANTHROPIC_API_KEY=sk-ant-...    # or an `ant auth login` profile
    CLAUDE_MODEL=claude-opus-5      # optional
    CLAUDE_EFFORT=high              # low | medium | high | xhigh | max
"""

from __future__ import annotations

import os
from typing import Any, Optional

# The subnet grades on a complete hidden suite, so the system prompt's whole job
# is to stop the model from adding anything that is not the program itself.
SYSTEM = (
    "You are solving competitive-programming tasks for an automated grader. "
    "Reply with exactly one fenced code block and nothing else. No preamble, "
    "no explanation, no tests, no example calls."
)


class _ClaudeConversation:
    """One task's conversation. The API is stateless, so history is resent."""

    def __init__(self, backend: "ClaudeBackend"):
        self._backend = backend
        self._messages: list[dict[str, Any]] = []

    async def send(self, text: str, timeout_s: float) -> str:
        self._messages.append({"role": "user", "content": text})
        client = self._backend.client.with_options(timeout=max(5.0, timeout_s))
        try:
            response = await client.beta.messages.create(
                model=self._backend.model,
                max_tokens=self._backend.max_tokens,
                system=SYSTEM,
                messages=self._messages,
                # Adaptive thinking: the model decides its own depth. budget_tokens
                # is removed on this model family and would be rejected with a 400.
                thinking={"type": "adaptive"},
                output_config={"effort": self._backend.effort},
                # A policy decline would otherwise just stop the turn and cost us
                # the round; this re-runs the same request on a fallback model
                # inside the same call, routed by refusal category.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:  # noqa: BLE001 - a failed turn is a zero, not a crash
            self._backend.errors += 1
            print(f"[claude] request failed: {type(exc).__name__}: {exc}")
            return ""

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None)
            print(f"[claude] refused (category={detail})")
            return ""

        reply = "".join(
            block.text for block in response.content if block.type == "text"
        )
        # Echo the assistant turn back so the repair round sees its own attempt.
        self._messages.append({"role": "assistant", "content": reply})
        self._backend.turns += 1
        return reply

    async def close(self) -> None:
        self._messages.clear()


class ClaudeBackend:
    """Anthropic-backed solver backend."""

    name = "claude-api"

    def __init__(
        self,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        import anthropic

        # Zero-arg client: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile. Don't demand an env var that may not be how
        # this host is authenticated.
        self.client = anthropic.AsyncAnthropic()
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-opus-5")
        self.effort = effort or os.environ.get("CLAUDE_EFFORT", "high")
        self.max_tokens = int(max_tokens or os.environ.get("CLAUDE_MAX_TOKENS", "16000"))
        self.turns = 0
        self.errors = 0

    async def open(self) -> _ClaudeConversation:
        return _ClaudeConversation(self)

    async def aclose(self) -> None:
        await self.client.close()

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "effort": self.effort,
            "turns": self.turns,
            "errors": self.errors,
        }

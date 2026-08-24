"""Gemini backend for the custom miner, via the official google-genai SDK.

Implements the same ``Backend``/``Conversation`` protocol as the other backends
in this package, so ``verify.py``'s self-verify-and-repair loop drives it
unchanged. ``client.aio.chats`` keeps the turn history server-side for us, which
is what the repair round needs: the model sees its own previous attempt beside
the failure report.

Like the Claude backend and unlike the browser one, this needs no Chrome, no
display and no persistent profile.

    GEMINI_API_KEY=...              # or GOOGLE_API_KEY
    GEMINI_MODEL=gemini-3-pro       # set this to a model your key can reach
    GEMINI_THINKING_BUDGET=-1       # -1 = dynamic, 0 = off, or an explicit budget
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Same job as the Claude system prompt: the grader wants a program, nothing else.
SYSTEM = (
    "You are solving competitive-programming tasks for an automated grader. "
    "Reply with exactly one fenced code block and nothing else. No preamble, "
    "no explanation, no tests, no example calls."
)


class _GeminiConversation:
    """One task's chat session; the SDK carries the turn history."""

    def __init__(self, backend: "GeminiBackend", chat: Any):
        self._backend = backend
        self._chat = chat

    async def send(self, text: str, timeout_s: float) -> str:
        import asyncio

        try:
            # The SDK has no per-call deadline, so the caller's budget is applied
            # here — an answer after the cutoff scores the same as no answer.
            response = await asyncio.wait_for(
                self._chat.send_message(text), timeout=max(5.0, timeout_s)
            )
        except asyncio.TimeoutError:
            self._backend.errors += 1
            print(f"[gemini] no reply within {timeout_s:.0f}s")
            return ""
        except Exception as exc:  # noqa: BLE001 - a failed turn is a zero, not a crash
            self._backend.errors += 1
            print(f"[gemini] request failed: {type(exc).__name__}: {exc}")
            return ""

        reply = getattr(response, "text", None)
        if not reply:
            # A blocked or empty candidate has no .text; treat it as a dud turn
            # rather than letting a None reach the code extractor.
            feedback = getattr(response, "prompt_feedback", None)
            print(f"[gemini] empty reply (feedback={feedback})")
            return ""
        self._backend.turns += 1
        return reply

    async def close(self) -> None:
        self._chat = None


class GeminiBackend:
    """Google-backed solver backend."""

    name = "gemini"

    def __init__(
        self,
        model: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("set GEMINI_API_KEY (or GOOGLE_API_KEY) for the gemini backend")
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        # Deliberately not defaulted to a hard-coded model id: model availability
        # differs per key and per API version, and a wrong id fails at request
        # time on every task. Set GEMINI_MODEL to one your key can reach.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3-pro")
        budget = thinking_budget
        if budget is None:
            budget = int(os.environ.get("GEMINI_THINKING_BUDGET", "-1"))
        self.thinking_budget = budget
        self.turns = 0
        self.errors = 0

    def _config(self):
        from google.genai import types

        kwargs: dict[str, Any] = {"system_instruction": SYSTEM}
        if self.thinking_budget is not None:
            # -1 lets the model size its own reasoning; 0 disables it.
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        return types.GenerateContentConfig(**kwargs)

    async def open(self) -> _GeminiConversation:
        chat = self.client.aio.chats.create(model=self.model, config=self._config())
        return _GeminiConversation(self, chat)

    async def aclose(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "thinking_budget": self.thinking_budget,
            "turns": self.turns,
            "errors": self.errors,
        }

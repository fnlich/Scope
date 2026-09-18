"""Every prompt, every reply, and what each turn cost, in one file.

The archive already answers "what did we submit" and "was that reasonable"
(`solution_archive.save_exchange`). It cannot answer "what was the model
actually ASKED, turn by turn, and what did that turn cost" -- a solve opens
five conversations and the archive keeps one request and one response.

That gap is what made the candidate-turn investigation expensive: the prompts
had to be rebuilt by hand from the builders, and the one number that explains
the latency -- thinking tokens -- was being parsed out of the event stream and
then dropped on the floor. Both are free to keep; nothing here is reconstructed.

Off unless `SOLVER_TRANSCRIPT` names a file, because it writes every prompt in
full and a 97-problem replay is tens of megabytes. Never raises: a transcript
that fails to write must not cost an answer, so every failure is swallowed and
the solve carries on unaware.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# One lock for the process. Stages run concurrently -- the inputs, reference
# and candidate turns are in flight together -- and three interleaved writes
# would produce a file no reader can split back into turns.
_LOCK = threading.Lock()

_RULE = "=" * 78


def path() -> Optional[Path]:
    """Where the transcript goes, or None when the operator wants none."""
    raw = os.environ.get("SOLVER_TRANSCRIPT", "").strip()
    if not raw or raw.lower() in ("0", "false", "no", "off"):
        return None
    return Path(raw).expanduser()


def enabled() -> bool:
    return path() is not None


def _tokens_of(conversation: Any) -> dict:
    """What the last turn on this conversation cost, if the backend counts.

    Read off the conversation rather than passed in, so a backend that counts
    nothing -- the browser fleet has no token stream at all -- costs this
    module an attribute lookup and not a special case.
    """
    usage = getattr(conversation, "last_usage", None)
    return usage if isinstance(usage, dict) else {}


def record(
    *,
    problem_id: str,
    phase: str,
    conversation: Any = None,
    prompt: str = "",
    reply: str = "",
    seconds: Optional[float] = None,
    note: str = "",
) -> None:
    """Append one turn. Never raises."""
    try:
        target = path()
    except Exception:  # noqa: BLE001 - an unreadable setting is "no transcript"
        # `path()` itself can raise: `Path(raw)` on a value the OS will not
        # accept is a ValueError, and this used to be OUTSIDE the guard below,
        # so the one function documented never to raise did. Found by the test
        # that points it at a path that cannot be opened.
        return
    if target is None:
        return
    try:
        tokens = _tokens_of(conversation)
        provider = (getattr(conversation, "label", None)
                    or getattr(conversation, "provider", None) or "?")
        cost = "  ".join(
            f"{k}={v}" for k, v in (
                ("in", tokens.get("input")),
                ("out", tokens.get("output")),
                ("thinking", tokens.get("thinking")),
                ("cache_read", tokens.get("cache_read")),
            ) if v is not None
        ) or "tokens unavailable for this backend"
        spent = "" if seconds is None else f"  {seconds:.1f}s"
        head = (
            f"\n{_RULE}\n"
            f"{datetime.now():%H:%M:%S}  id={problem_id[:12]}  phase={phase}  "
            f"{provider}{spent}\n"
            f"{cost}\n"
            + (f"{note}\n" if note else "")
            + f"{_RULE}\n"
        )
        body = (
            f"----- PROMPT ({len(prompt)} chars) -----\n{prompt}\n"
            f"----- REPLY ({len(reply)} chars) -----\n{reply}\n"
        )
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(head + body)
    except Exception:  # noqa: BLE001 - a transcript never costs an answer
        pass

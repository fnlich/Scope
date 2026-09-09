"""Answers that were right, kept on disk and keyed by the problem.

A validator that sends the same problem twice is offered the same answer the
second time, in about a second, without opening a conversation at all. Two
things make that worth having and one makes it dangerous.

WORTH HAVING. Payment is binary on the complete hidden suite with a speed
multiplier floored at 0.95, so a correct answer is worth the same whenever it
arrives -- but the fastest correct responder is the one that gets the full
1.0, and nothing else this miner does can answer in a second. And the miner
holds two subscriptions' worth of quota, not an API budget: a solve it does
not have to run is a solve the accounts can spend on a problem they have not
seen.

DANGEROUS. A cached wrong answer is not one zero, it is a zero every time that
problem comes round again, and the thing being cached was graded by the model
that wrote it. So the gate is deliberately narrow -- see `worth_keeping` --
and it is the gate rather than the storage that this module is really about.

HOW OFTEN IT FIRES IS UNKNOWN. Of the 97 archived live requests in
`examples/problems/`, no two share a statement: zero hits, on that sample.
That is not evidence there will be none -- 97 requests is a few hours of one
miner's traffic against a problem service with its own cursor -- but it is a
reason to keep this cheap and to make sure a miss costs nothing. A miss is a
file that is not there.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# Where answers are kept. Under the solutions directory rather than beside the
# code: it is operator data, it grows, and `.gitignore` already keeps that
# whole tree out of the repository.
DEFAULT_DIR = "solutions/cache"

# The most a cached file may be. A `code` field is a program and programs are
# small; anything this size is a file that is not what it claims to be, and
# parsing it costs more than the hit is worth.
_MAX_BYTES = 2 * 1024 * 1024


def cache_dir() -> Optional[Path]:
    """Where answers are kept, or None when the operator turned this off.

    `SOLVER_SOLUTION_CACHE=0` disables it outright;
    `SOLVER_SOLUTION_CACHE_DIR=` (empty) does the same by naming nowhere.
    """
    if os.environ.get("SOLVER_SOLUTION_CACHE", "1").strip().lower() in (
        "0", "false", "no"
    ):
        return None
    raw = os.environ.get("SOLVER_SOLUTION_CACHE_DIR", DEFAULT_DIR).strip()
    return Path(raw).expanduser() if raw else None


def worth_keeping(
    *,
    self_verified: bool,
    failures: bool,
    contested: int,
    probe: str,
) -> bool:
    """Whether an answer is good enough to be given out again unexamined.

    Every condition here is about EVIDENCE rather than about confidence, and
    each rules out a way an answer can look finished without being checked:

    * `self_verified` -- it ran against its own cases and passed all of them.
      An answer that was never graded (no bar, no executor) is not cached at
      any price: the whole hazard is replaying something nobody checked.
    * no outstanding `failures` -- the last grade was clean, not merely the
      best in hand when the clock ran out.
    * no `contested` cases -- a case the readers could not agree on was
      dropped rather than passed, so the suite this cleared is missing a
      question about which the statement is ambiguous. Right often enough to
      ship once; not right enough to ship for ever.
    * `probe == "passed"` -- it was timed at the statement's own scale and
      finished. `too_slow` is a known failure and `none`/`skipped` mean it was
      never timed, and neither belongs in a file that will be handed out
      without being run again.

    Note what is NOT here: agreement with public examples. Live traffic ships
    none, so requiring them would make this dead code on the only path that
    matters.
    """
    return bool(
        self_verified
        and not failures
        and not contested
        and probe == "passed"
    )


def load(key: str) -> Optional[dict[str, Any]]:
    """The stored answer for `key`, or None. Never raises.

    A cache that can fail a solve is worse than no cache, so every way this
    can go wrong -- no directory, no file, unreadable, not JSON, JSON of the
    wrong shape -- returns None and the solve proceeds as though the file had
    never existed.
    """
    directory = cache_dir()
    if directory is None or not key:
        return None
    path = directory / f"{key}.json"
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a miss is the answer to every failure
        return None
    if not isinstance(record, dict):
        return None
    code = record.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    return record


def save(key: str, record: dict[str, Any]) -> None:
    """Keep `record` under `key`. Never raises, and never half-writes.

    Written to a temporary name and renamed, because `os.replace` is atomic on
    every filesystem this runs on: a reader is never offered a file that is
    half a program, and two miners saving the same answer at once leave one
    whole file rather than a mixture.
    """
    directory = cache_dir()
    if directory is None or not key:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        body = json.dumps(record, ensure_ascii=False, default=str)
        temporary = directory / f".{key}.{os.getpid()}.tmp"
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, directory / f"{key}.json")
    except Exception as exc:  # noqa: BLE001 - a full disk loses no answer
        print(f"[verify] the solution cache could not be written ({exc})")
        try:
            temporary.unlink()
        except Exception:  # noqa: BLE001 - it may never have been created
            pass


def record(
    *, code: str, raw: str, task, bar: list, probe: str, providers: list
) -> dict[str, Any]:
    """The stored shape. Everything an operator would need to audit a hit.

    The bar is kept beside the code deliberately. A cached answer that turns
    out to be wrong is a question about what it was checked against, and
    without the cases there is no way to ask it.
    """
    return {
        "problem_id": getattr(task, "problem_id", ""),
        "language": getattr(task, "language", ""),
        "entrypoint": getattr(task, "entrypoint", ""),
        "code": code,
        "raw_response": raw,
        "bar": list(bar or []),
        "probe": probe,
        "providers": list(providers or []),
        "saved_at": time.time(),
    }

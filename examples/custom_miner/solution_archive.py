"""Keep a copy on disk of every answer this miner sends.

A browser-backed miner is hard to look at after the fact. The reply that
produced a zero is gone the moment the tab moves to its next conversation, the
log line says how the solve ended but not what was submitted, and the validator
keeps the only other copy. So every solve leaves a file here, named for the
problem, holding exactly the source that went out.

Solves that produced NOTHING leave a file too, and that is the part worth being
deliberate about: an empty file records that this problem was seen and came back
with no answer, which is a different and far more useful fact than no file at
all. Absence would be ambiguous -- never dispatched, crashed before the solver
ran, or answered with silence -- and those need different fixes. A zero-byte
file says which one it was.

Nothing here is allowed to break a solve. A miner that dies because a disk
filled up has turned a lost point into a lost session, so every failure is
swallowed after one line of explanation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_DIR = "solutions"

# `problem_id` arrives over the network from a validator and is being used to
# build a PATH, so it is treated as hostile input rather than as an identifier:
# anything outside this set is replaced. `../../etc/passwd` has to become a
# harmless name inside the archive directory, never a write outside it.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
# Long enough for any real id, short enough to stay under every filesystem's
# name limit once the extension is added.
_MAX_STEM = 100

_warned = False


def archive_dir() -> Optional[Path]:
    """Where solutions are kept, or None if the operator turned it off.

    ``SOLVER_SOLUTION_DIR=`` (empty) disables archiving entirely; anything else
    is a directory, created on first use.
    """
    raw = os.environ.get("SOLVER_SOLUTION_DIR", DEFAULT_DIR).strip()
    return Path(raw).expanduser() if raw else None


def _stem(problem_id: str) -> str:
    """A file name that cannot leave the archive directory."""
    safe = _UNSAFE.sub("_", str(problem_id)).strip("._")
    return (safe or "unknown")[:_MAX_STEM]


def save_solution(
    problem_id: str,
    language: str,
    code: str,
    directory: Optional[str | Path] = None,
) -> Optional[Path]:
    """Write ``code`` to ``<dir>/<problem_id>.<py|rs>``. Never raises.

    Returns the path written, or None if archiving is off or the write failed.
    Whitespace-only source is written as a genuinely empty file rather than as
    a few blank lines, so "no answer" is visible at a glance in a directory
    listing and to anything measuring size.
    """
    target = archive_dir() if directory is None else Path(directory)
    if target is None:
        return None
    extension = ".rs" if str(language).lower() == "rust" else ".py"
    path = target / (_stem(problem_id) + extension)
    body = code if (code or "").strip() else ""
    try:
        target.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - a full disk must not cost a solve
        global _warned
        if not _warned:
            _warned = True
            print(
                f"[custom-miner] could not write solutions to {target} "
                f"({type(exc).__name__}: {exc}). Solving continues; set "
                f"SOLVER_SOLUTION_DIR to a writable path, or to nothing to "
                f"turn archiving off. Once per run."
            )
        return None
    return path

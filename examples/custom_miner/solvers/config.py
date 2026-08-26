"""Reading solver configuration out of the environment (and out of ``.env``).

Two small things live here because both browser backends and the doctor tool
need them.

``load_env_file`` exists because the miner's own settings come from ``.env``
via pydantic-settings, which reads that file *directly* and never puts anything
in ``os.environ``. So a selector override or a port list written into ``.env``
was silently ignored — the worst possible outcome for a knob whose whole purpose
is to rescue a miner whose DOM changed. Real environment variables still win,
so this only fills in what the shell did not set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def selectors(env: str, default: Sequence[str]) -> tuple[str, ...]:
    """Candidate selectors for one role, overridable as ``a|b|c``.

    ``|`` rather than ``,`` because a comma is already CSS's own "either"
    operator: ``a, b`` is one selector matching both, which would leave us
    unable to report *which* candidate the page actually has.
    """
    raw = os.environ.get(env, "")
    if raw.strip():
        return tuple(part.strip() for part in raw.split("|") if part.strip())
    return tuple(default)


def find_env_file(start: str | os.PathLike[str] | None = None) -> Optional[Path]:
    """Nearest ``.env`` at or above ``start`` (default: the current directory).

    Searched upward because the miner's ``.env`` lives at the repository root
    while the doctor tool is run from ``examples/custom_miner``; one file should
    configure both.
    """
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _unquote(value: str) -> str:
    """Match python-dotenv closely enough that the same file parses the same way.

    That matters more than it looks: the miner's own settings go through
    pydantic-settings, which reads ``.env`` with python-dotenv. Anything this
    function parses differently would be *promoted into* ``os.environ``, where it
    outranks the file — so a mismatch here silently changes the miner's settings.
    An unquoted trailing comment is the common case.
    """
    value = value.strip()
    if value[:1] in ("\"", "'"):
        end = value.find(value[0], 1)
        return value[1:end] if end != -1 else value[1:]
    for marker in (" #", "\t#"):
        value = value.split(marker, 1)[0]
    return value.strip()


def load_env_file(path: str | os.PathLike[str] | None = None) -> int:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overwriting.

    Returns the number of variables set. A missing file is not an error.
    """
    file = Path(path) if path is not None else find_env_file()
    if file is None or not file.is_file():
        return 0
    loaded = 0
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key and key not in os.environ:
            os.environ[key] = _unquote(value)
            loaded += 1
    return loaded

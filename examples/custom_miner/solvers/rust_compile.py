"""Build a Rust candidate locally, when a compiler is at hand.

The asymmetry this fixes is not a style choice. Python's structural check
PARSES the source -- `ast.parse` rejects prose, a shell command, a truncated
line, anything that is not a program. Rust's greps it for `fn main`. That is
why every answer this miner has ever destroyed in transit was a Rust one: the
model's own reasoning, a tool call, and the miner's own prompt all contain the
characters `fn main`, and all three were submitted as programs.

Measured on one run of 45 archived submissions: 18 Rust answers, 6 of which
would not build. A compile catches all six --- two prompt echoes, one tool
call, one answer truncated mid-identifier, and two genuine compiler errors the
model made --- and passes all twelve that do build. Three of those six would
never have reached a validator at all; the other three become a repair round
instead of a zero.

It matters most when there is nothing else. With no public examples shipped --
which is every task on the run this was written for -- the grader never runs,
so `rust_defect` is the ONLY check a Rust answer gets before it is submitted.

The flags come from `RELEASE_POLICY`, not from a copy of them, so this asks the
same question the validator will. A local toolchain can still differ from the
pinned one, and the failure mode of that is deliberately cheap: a wrong defect
costs a repair round, never the answer, because a defective non-empty candidate
outranks an empty one in `Candidate.score` and is still what gets submitted if
the repair produces nothing better.

The candidate is COMPILED, never run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rlvr.policy import RELEASE_POLICY

# Generous: the point is to catch a broken answer, and a solve that has already
# spent two minutes waiting on a model can afford a second or two here. Bounded
# anyway, because a compiler that hangs must not take the solve with it.
COMPILE_TIMEOUT_S = float(os.environ.get("SOLVER_RUST_COMPILE_TIMEOUT_S", "25"))

_rustc: Optional[str] = None
_looked = False


def rustc_path() -> Optional[str]:
    """The local compiler, or None. Looked up once and reported once."""
    global _rustc, _looked
    if _looked:
        return _rustc
    _looked = True
    if os.environ.get("SOLVER_RUST_COMPILE", "1") == "0":
        print("[verify] local Rust compile checking is off (SOLVER_RUST_COMPILE=0)")
        return None
    _rustc = shutil.which("rustc")
    if _rustc is None:
        print(
            "[verify] no local `rustc`, so Rust answers get the structural check "
            "only. Installing a toolchain lets the miner reject an answer that "
            "will not build instead of submitting it. Once per run."
        )
    return _rustc


def compile_defect(code: str) -> Optional[str]:
    """Why this source will not build, in one line, or None. Never raises.

    None also means "could not tell" -- no compiler, or the attempt itself went
    wrong. Silence has to mean the same thing as success here, because the
    alternative is a miner that stops submitting answers the moment a toolchain
    goes missing.
    """
    rustc = rustc_path()
    if rustc is None or not code.strip():
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="hone-rustc-") as work:
            source = Path(work) / "candidate.rs"
            source.write_text(code, encoding="utf-8")
            built = subprocess.run(
                [
                    rustc,
                    f"--edition={RELEASE_POLICY.rust_edition}",
                    *RELEASE_POLICY.rustc_flags,
                    "-o", str(Path(work) / "candidate"),
                    str(source),
                ],
                capture_output=True, text=True, cwd=work,
                timeout=COMPILE_TIMEOUT_S,
            )
            if built.returncode == 0:
                return None
            return f"it does not compile: {_first_error(built.stderr)}"
    except subprocess.TimeoutExpired:
        return None  # a slow compile is not evidence the answer is wrong
    except Exception:  # noqa: BLE001 - a broken check must not cost the answer
        return None


def _first_error(stderr: str) -> str:
    """The first real error line, which is the one a model can act on.

    Warnings come first often enough that handing back `stderr[:200]` reports a
    style lint as the reason a program was rejected -- and a repair round spent
    on an unused-variable warning is a repair round wasted.
    """
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("error"):
            return stripped[:200]
    head = next((l.strip() for l in stderr.splitlines() if l.strip()), "")
    return (head or "rustc reported no reason")[:200]

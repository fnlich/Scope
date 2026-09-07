"""How two outputs are judged the same, and what a candidate is asked about
itself.

Two modes, because the corpus has two. A Rust answer is a complete program
judged on whitespace tokens; a Python answer is a function judged on the value
it returns, which must not be the arguments it was handed.

Nothing here runs code. `probe_source` builds the program that does, so that
the questions which can only be answered from INSIDE the sandbox -- was an
argument mutated, what type was actually returned -- come back as data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# The validator's own split: bytes, not str, and these six characters only.
# Anything else -- a locale's idea of whitespace, a non-breaking space -- is
# content and must survive into the comparison.
TOKEN_SPLIT = re.compile(rb"[\t\n\x0b\x0c\r ]+")

# What the probe prefixes its answer with, so a program that happens to return
# a dict cannot be mistaken for one.
PROBE_KEY = "__hone_probe__"


def tokens(output: Any) -> list[bytes]:
    """A program-mode output as the tokens it will actually be judged on."""
    if output is None:
        return []
    raw = output if isinstance(output, bytes) else str(output).encode("utf-8", "replace")
    return [piece for piece in TOKEN_SPLIT.split(raw.strip()) if piece]


def same_program(left: Any, right: Any) -> bool:
    """Program mode: the token lists are equal, exactly."""
    return tokens(left) == tokens(right)


def same_function(left: Any, right: Any) -> bool:
    """Function mode: raw equality.

    Deliberately `==` and not a tolerance. The validator compares values; a
    checker that is more forgiving than the validator reports a pass the
    validator will not give, which is the one error this whole package exists
    to avoid.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        # `True == 1` in Python and the validator does not have to agree.
        return False
    return left == right


def same(language: str, left: Any, right: Any) -> bool:
    return same_program(left, right) if language == "rust" else same_function(left, right)


# --------------------------------------------------------------------------- #
# The probe: the questions only the sandbox can answer.
# --------------------------------------------------------------------------- #

_PROBE = '''

# -- appended by the verification ladder; not part of the candidate -------- #
def {name}(*__args, **__kwargs):
    import copy as __copy
    import json as __json

    def __shape(__v):
        if __v is None:
            return "None"
        return type(__v).__name__

    def __widths(__v, __out):
        if isinstance(__v, bool):
            return
        if isinstance(__v, int):
            __out.append(abs(__v))
            return
        if isinstance(__v, (list, tuple)):
            for __x in __v:
                __widths(__x, __out)
        elif isinstance(__v, dict):
            for __k, __x in __v.items():
                __widths(__k, __out)
                __widths(__x, __out)

    __before = __copy.deepcopy((__args, __kwargs))
    __value = {entry}(*__args, **__kwargs)
    try:
        __mutated = (__args, __kwargs) != __before
    except Exception:
        __mutated = False
    __sizes = []
    try:
        __widths(__value, __sizes)
    except Exception:
        __sizes = []
    try:
        __json.dumps(__value)
        __serialisable = True
    except Exception:
        __serialisable = False
    return {{
        "{key}": 1,
        "mutated": bool(__mutated),
        "shape": __shape(__value),
        "max_int": max(__sizes) if __sizes else 0,
        "serialisable": __serialisable,
        "value": __value,
    }}
'''

PROBE_ENTRY = "__hone_probe_entry"


def probe_source(code: str, entrypoint: str) -> str:
    """`code` with a wrapper appended that reports what the sandbox can see.

    Run this instead of the candidate, with `PROBE_ENTRY` as the entrypoint,
    and every case comes back carrying its own mutation and shape verdict. The
    candidate is untouched above the marker, so a failure inside it still
    surfaces as that failure.
    """
    return code + _PROBE.format(name=PROBE_ENTRY, entry=entrypoint, key=PROBE_KEY)


@dataclass
class Probed:
    """One case's answer, as the probe saw it."""

    value: Any = None
    shape: str = ""
    mutated: bool = False
    max_int: int = 0
    serialisable: bool = True
    ok: bool = False
    error: Optional[str] = None

    @classmethod
    def read(cls, run: Any) -> "Probed":
        """Unpack a `_Grader.outputs` row that ran `probe_source`."""
        if not getattr(run, "ok", False):
            return cls(ok=False, error=getattr(run, "error", None) or "did not run")
        payload = getattr(run, "value", None)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:  # noqa: BLE001 - a string answer is just a value
                payload = None
        if not isinstance(payload, dict) or PROBE_KEY not in payload:
            # The probe did not run -- the candidate raised before it returned,
            # or something replaced the entrypoint. The value stands on its own.
            return cls(ok=True, value=getattr(run, "value", None), shape="?")
        return cls(
            ok=True,
            value=payload.get("value"),
            shape=str(payload.get("shape") or "?"),
            mutated=bool(payload.get("mutated")),
            max_int=int(payload.get("max_int") or 0),
            serialisable=bool(payload.get("serialisable", True)),
        )


# --------------------------------------------------------------------------- #
# The register's return shape, checked against what came back.
# --------------------------------------------------------------------------- #

# What the statement can ask for and what Python calls it. A register that says
# "a list of pairs" is checked as a list; the pairs are the comparison's job.
_SHAPES = {
    "list": {"list"},
    "tuple": {"tuple"},
    "dict": {"dict"},
    "set": {"set", "frozenset"},
    "str": {"str"},
    "int": {"int"},
    "float": {"float", "int"},
    "bool": {"bool"},
    "none": {"NoneType", "None"},
}


def shape_matches(declared: str, observed: str) -> Optional[str]:
    """None when the shape is right or unknown, else what is wrong with it.

    Unknown is not a failure. The register is one model's reading of a
    statement, and a shape it did not name is a shape this cannot judge.
    """
    want = _SHAPES.get((declared or "").strip().lower())
    if not want or not observed or observed == "?":
        return None
    seen = observed.strip()
    if seen in want or seen.lower() in want:
        return None
    return f"the register says the answer is a {declared}, and a {observed} came back"


@dataclass
class Mismatch:
    """One differential disagreement, kept whole for the repair prompt."""

    case: dict[str, Any] = field(default_factory=dict)
    candidate: Any = None
    reference: Any = None
    note: str = ""

    def describe(self, language: str, entrypoint: str, limit: int = 300) -> str:
        call = _render_call(language, entrypoint, self.case)
        return (
            f"{call} -> {_clip(self.candidate, limit)}"
            f"; the reference says {_clip(self.reference, limit)}"
            + (f" ({self.note})" if self.note else "")
        )


def _clip(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _render_call(language: str, entrypoint: str, case: dict[str, Any]) -> str:
    if language == "rust":
        stdin = (case.get("args") or [""])[0]
        return f"stdin {_clip(stdin, 200)}"
    args = ", ".join(_clip(a, 80) for a in case.get("args") or [])
    kwargs = case.get("kwargs") or {}
    if kwargs:
        args += ", " + ", ".join(f"{k}={_clip(v, 80)}" for k, v in kwargs.items())
    return f"{entrypoint}({args})"

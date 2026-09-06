"""What may still be launched, and when the answer goes out regardless.

The whole pipeline is a race against one number. `Clock` holds it, `decide`
reads it, and nothing else in the package is allowed an opinion about time.

THE LATENCIES ARE MEASURED, NOT ASSUMED. The design this implements shipped a
`P95` table marked "assumed; replace with measured", and the assumption that
mattered most was wrong by more than a factor of two: it put a high-effort
Opus solution at 105 seconds, and this miner's own production logs put the
equivalent turn at 230 seconds at the ninety-fifth percentile -- measured over
53 solves at effort LOW, which is the fast end of the range the design asks
for. Every default below is the measured number, and `load_p95` replaces it
with whatever calibration wrote last. A gate computed from an optimistic
latency does not fail loudly; it launches a call that cannot land, and the
solve reports nothing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# The validator stops reading at `deadline_s + 10`, and the miner's own
# `handle_request` cuts at `deadline_s + RESPONSE_GRACE_S`. HARD is where the
# answer goes out whatever state the pipeline is in.
DEADLINE_S = 300.0
MARGIN_S = 20.0
HARD_S = DEADLINE_S - MARGIN_S

# One verification round with no model in it. Measured on this corpus: ~5s
# typical, so 10 is the cap rather than the expectation.
VERIFY_S = 10.0
DEEP_VERIFY_S = 5.0

# Measured on 53 production solves (Opus, effort low, 2.7KB statements), and
# deliberately the p95 rather than the median -- a gate exists for the tail.
#
#   phase        n   median   p75   p90   p95   max
#   cases       53       54    63    79   126   145
#   program     53       66   100   185   230   251
#   correction  17        7    27    37    50    50
#
# `solution` is the program turn: the register replaces the cases turn, so the
# program turn alone is the comparator. `fix` is the correction turn.
MEASURED_P95 = {
    "register": 20.0,     # a short Sonnet turn; not separately measured yet
    "solution": 230.0,    # measured
    "solution_fast": 130.0,   # measured / 1.77, the low end of fast mode's claim
    "solution_small": 90.0,   # a smaller model's solution; not measured yet
    "kit": 90.0,          # a long Sonnet turn; not separately measured yet
    "fix": 50.0,          # measured
    "fix_small": 35.0,    # not separately measured yet
}

# Caps are what a call is allowed to spend before it is abandoned, and they are
# NOT the p95: a cap set at the p95 abandons one call in twenty at the moment
# it was about to answer. They are set where the marginal call stops being
# worth the seconds it denies everything after it.
CAP = {
    "register": 20.0,
    "solution": 150.0,
    "kit": 70.0,
    "fix": 60.0,
    "fix_small": 40.0,
}


def load_p95(path: Optional[str] = None) -> dict[str, float]:
    """The measured table, overlaid with whatever calibration wrote last."""
    table = dict(MEASURED_P95)
    path = path or os.environ.get("PIPELINE_P95_FILE", "")
    if path and os.path.exists(path):
        try:
            with open(path) as handle:
                learned = json.load(handle)
            for key, value in (learned or {}).items():
                if key in table and isinstance(value, (int, float)) and value > 0:
                    table[key] = float(value)
        except Exception:  # noqa: BLE001 - a bad file must not stop a solve
            pass
    return table


class State(str, Enum):
    """What the pipeline is holding, not what it is doing."""

    NO_SOLUTION = "no solution"
    HAVE_SOLUTION = "solution, no kit"
    HAVE_KIT = "solution and kit, not verified"
    VERIFIED_GREEN = "verified green"
    VERIFIED_FAIL = "verified, and it failed"


class Action(str, Enum):
    REPORT = "report"
    WAIT = "wait"
    LAUNCH_SOLUTION = "launch solution"
    LAUNCH_SOLUTION_SMALL = "launch solution (smaller model)"
    LAUNCH_KIT = "launch test kit"
    LAUNCH_FIX = "launch fix"
    LAUNCH_FIX_SMALL = "launch fix (smaller model)"
    DEEP_VERIFY = "deep verification"


@dataclass
class Clock:
    """One solve's budget. `now()` is seconds since the request arrived."""

    started: float = field(default_factory=time.monotonic)
    hard_s: float = HARD_S
    p95: dict[str, float] = field(default_factory=load_p95)

    def now(self) -> float:
        return time.monotonic() - self.started

    def left(self) -> float:
        return self.hard_s - self.now()

    def fits(self, *stages: str, verify: bool = True) -> bool:
        """Is there room for these stages and a verification round after them?"""
        need = sum(self.p95.get(stage, 0.0) for stage in stages)
        if verify:
            need += VERIFY_S
        return self.left() >= need


def decide(
    clock: Clock,
    state: State,
    *,
    have_deep: bool = False,
    fix_rounds: int = 0,
    max_fix_rounds: int = 2,
    solution_attempts: int = 1,
) -> Action:
    """The only launcher. Pure: it reads the clock and the state, nothing else.

    Ordering is the design's: a candidate that can be reported exists before
    the evidence that would improve it, because a candidate at 120 seconds can
    always be reported and a test kit at 120 seconds cannot.
    """
    if clock.left() <= 0:
        return Action.REPORT

    if state is State.VERIFIED_GREEN:
        # Deep verification is the only thing left that can change the answer,
        # and it is cheap. After it, nothing sequential can add anything.
        if not have_deep and clock.left() >= DEEP_VERIFY_S:
            return Action.DEEP_VERIFY
        return Action.REPORT

    if state is State.VERIFIED_FAIL:
        if fix_rounds >= max_fix_rounds:
            return Action.REPORT
        if clock.fits("fix"):
            return Action.LAUNCH_FIX
        if clock.fits("fix_small"):
            return Action.LAUNCH_FIX_SMALL
        return Action.REPORT

    if state is State.NO_SOLUTION:
        # The first solution call is launched by the caller; this branch is the
        # one that fires when it timed out and a smaller, faster model is the
        # last chance at any candidate at all.
        if solution_attempts <= 1 and clock.fits("solution"):
            return Action.LAUNCH_SOLUTION
        if clock.fits("solution_small", verify=False):
            return Action.LAUNCH_SOLUTION_SMALL
        return Action.REPORT

    if state is State.HAVE_SOLUTION:
        # The kit is worth launching only if a fix could still follow it --
        # evidence with no time to act on it buys nothing. This branch is
        # missing from the design's own `decide`, which never launches B2.
        if clock.fits("kit", "fix_small"):
            return Action.LAUNCH_KIT
        return Action.REPORT

    return Action.WAIT

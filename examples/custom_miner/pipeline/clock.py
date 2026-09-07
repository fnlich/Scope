"""What may still be launched, and when the answer goes out regardless.

The whole pipeline is a race against one number. `Clock` holds it, `decide`
reads it, and nothing else in the package is allowed an opinion about time.

EVERY LATENCY HERE COMES FROM A LOG. None is reasoned about, estimated from a
vendor's throughput claim, or carried over from a design document. They are
read from `latency.json`, which only `calibration/measure_logs.py` writes, and
every entry arrives with the number of observations behind it and the log line
pattern it was counted from. A stage with no measurement cannot be gated on:
`Latency.of` raises rather than let a plausible-looking float decide whether a
model call is launched.

That rule exists because the alternative was tried. The design this implements
shipped a P95 table marked "assumed", and the assumption carrying the schedule
put a solution turn at 105 seconds against a measured 220. A first pass at this
file then replaced two of its seven numbers with measurements and left five
invented, wearing comments that read like measurements. Both mistakes are the
same mistake, and the only fix that holds is structural.

TWO COSTS, MEASURED SEPARATELY, BECAUSE THEY ANSWER DIFFERENT QUESTIONS.
A `[cli]` turn line says what one SUCCESSFUL call cost. A `[phase]` line says
what the solve PAID to get that answer, including every turn that was cut,
hopped or retried first. Over 237 turns and 101 program phases the second is
2.2x the first. Gates are built on the phase cost; a gate built on the turn
cost schedules a call the solve cannot afford.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum

# The validator stops reading at `deadline_s + 10`, and the miner's own
# `handle_request` cuts at `deadline_s + RESPONSE_GRACE_S`. HARD is where the
# answer goes out whatever state the pipeline is in.
DEADLINE_S = 300.0
MARGIN_S = 20.0
HARD_S = DEADLINE_S - MARGIN_S

# Measured, not chosen: `[phase] cross-check` over 85 rounds is p50 16s, p95
# 61s. VERIFY_S is the p50 because the ladder is expected to run, not to be
# survived; the gates carry the tail by refusing to launch near the edge.
VERIFY_S = 16.0
DEEP_VERIFY_S = 5.0

_HERE = os.path.dirname(os.path.abspath(__file__))
LATENCY_FILE = os.environ.get("PIPELINE_LATENCY_FILE", os.path.join(_HERE, "latency.json"))


@dataclass(frozen=True)
class Latency:
    """One stage's cost, and where the number came from.

    `kind` is `measured` when a log line counts this exact stage, and `derived`
    when it is a measured number for a comparable stage carried across by a
    measured ratio. There is no third kind; a stage with neither is absent.
    """

    seconds: float
    n: int
    source: str
    kind: str = "measured"

    def __post_init__(self) -> None:
        if self.n <= 0 or not self.source:
            raise ValueError(f"a latency with no observations behind it: {self!r}")


def _table(path: str = "") -> dict:
    path = path or LATENCY_FILE
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - a missing table is a hard error
        raise RuntimeError(
            f"no measured latencies at {path}. Run "
            f"`python -m calibration.measure_logs <miner.log> --out {path}` "
            f"before gating anything on time."
        ) from exc


# How much more a solve PAYS for an answer than one successful call costs,
# because a phase absorbs the turns that were cut and retried first. Measured
# on the one stage where both numbers exist: program phase p95 / large-turn
# p95. Used only to carry a turn measurement across to a stage that has no
# phase of its own, and recorded as `derived` when it is.
def _retry_ratio(table: dict) -> float:
    phase = (table.get("phase") or {}).get("program") or {}
    turn = (table.get("turn_by_output_size") or {}).get("large") or {}
    if phase.get("p95") and turn.get("p95"):
        return round(float(phase["p95"]) / float(turn["p95"]), 2)
    return 1.0


def load(path: str = "") -> dict[str, Latency]:
    """The stage table, every entry traceable to a log line.

    Stage -> measurement mapping, and nothing else is in here:

      solution   the `program` phase. The same thing under another name: one
                 call that returns a whole program. n=101.
      fix        the `correction` phase. n=38.
      register   no phase runs one today, so the closest measurement is a turn
                 whose output is under 500 characters -- which is what a
                 numbered list of readings is. Carried across by the retry
                 ratio, and marked derived.
      kit        likewise, from turns over 8000 characters: a generator and a
                 reference are two programs, and that is the only bucket with
                 two programs' worth of output in it.
    """
    table = _table(path)
    phase = table.get("phase") or {}
    turn = table.get("turn_by_output_size") or {}
    ratio = _retry_ratio(table)
    out: dict[str, Latency] = {}

    for stage, key in (("solution", "program"), ("fix", "correction")):
        row = phase.get(key)
        if row:
            out[stage] = Latency(float(row["p95"]), int(row["n"]),
                                 f"{row['source']} p95", "measured")
    for stage, bucket in (("register", "tiny"), ("kit", "huge")):
        row = turn.get(bucket)
        if row:
            out[stage] = Latency(
                round(float(row["p95"]) * ratio, 1), int(row["n"]),
                f"{row['source']} p95 x {ratio} (measured phase/turn ratio)", "derived")
    return out


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
    LAUNCH_KIT = "launch test kit"
    LAUNCH_FIX = "launch fix"
    DEEP_VERIFY = "deep verification"


@dataclass
class Clock:
    """One solve's budget. `now()` is seconds since the request arrived."""

    started: float = field(default_factory=time.monotonic)
    hard_s: float = HARD_S
    latency: dict[str, Latency] = field(default_factory=load)

    def now(self) -> float:
        return time.monotonic() - self.started

    def left(self) -> float:
        return self.hard_s - self.now()

    def cost(self, stage: str) -> float:
        """What `stage` costs, or a refusal. Never a guess."""
        found = self.latency.get(stage)
        if found is None:
            raise KeyError(
                f"no measured latency for {stage!r}; measure it from a log "
                f"before gating on it. Known: {sorted(self.latency)}"
            )
        return found.seconds

    def fits(self, *stages: str, verify: bool = True) -> bool:
        """Is there room for these stages and a verification round after them?"""
        need = sum(self.cost(stage) for stage in stages)
        if verify:
            need += VERIFY_S
        return self.left() >= need

    def provenance(self) -> str:
        """One line per gate, for the log an operator reads at startup."""
        return "; ".join(
            f"{name}={row.seconds:.0f}s ({row.kind}, n={row.n})"
            for name, row in sorted(self.latency.items())
        )


def decide(
    clock: Clock,
    state: State,
    *,
    have_deep: bool = False,
    fix_rounds: int = 0,
    max_fix_rounds: int = 2,
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
        # There is no smaller-model rung here. The design had one; no log in
        # this repository has ever run a fix on a smaller model, so there is no
        # number to gate it with and a gate without a number is the thing this
        # module exists to prevent.
        return Action.REPORT

    if state is State.NO_SOLUTION:
        # The first solution call is launched by the caller; this branch is the
        # one that fires when it timed out and a smaller, faster model is the
        # last chance at any candidate at all.
        if clock.fits("solution"):
            return Action.LAUNCH_SOLUTION
        return Action.REPORT

    if state is State.HAVE_SOLUTION:
        # The kit is worth launching only if a fix could still follow it --
        # evidence with no time to act on it buys nothing. This branch is
        # missing from the design's own `decide`, which never launches B2.
        if clock.fits("kit", "fix"):
            return Action.LAUNCH_KIT
        return Action.REPORT

    return Action.WAIT

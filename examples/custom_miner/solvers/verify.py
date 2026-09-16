"""Grade your own answer with the validator's grader, then repair it.

The subnet pays only for a submission that passes the COMPLETE hidden suite —
a partial answer earns exactly as much as no answer at all. That makes a
one-shot "ask the model, return whatever it said" miner leave a lot on the
table: models routinely produce something that is nearly right.

Two facts turn that into an advantage:

1. ``TaskRequest.public_examples`` carries real ``{args, kwargs, expected}``
   cases, shipped with every task.
2. The comparison the validator will apply to you is IN THIS REPOSITORY —
   ``rlvr.execution.compare.values_equal`` for Python and
   ``rlvr.execution.rust_judge.outputs_match`` for Rust, reached through the
   same ``Executor`` the validator uses.

So a miner can run its own candidate through the validator's executor before
answering, and when a case fails, hand the model the concrete failure and ask
for a fix. Passing every public example is not proof of passing the hidden
suite, but it is a strong precondition and it eliminates the large class of
answers that are simply wrong on the stated contract.

A note on the executor. Grading runs under Docker, the same backend the
validator uses, because the point of grading locally is to learn what the
validator will find and the two backends do not agree about that. The
validator gives a candidate 256 MiB with swap off and runs every hidden test
in ONE container, where an OOM kill fails the entire suite rather than the
case that caused it; ``SubprocessExecutor`` gives it 1 GiB and a fresh process
per case. An answer that allocates 400 MB passes every local case and loses
every hidden one, and under the subprocess backend nothing here could see it.

``SOLVER_VERIFY_EXECUTOR=subprocess`` selects the old backend, and a host with
no reachable daemon falls back to it automatically for Python rather than
grading nothing -- with a line saying so, because a solve graded under the
wrong limits is worth less than it looks. Rust has no subprocess path at all
and always requires the container.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple, Optional, Protocol

from rlvr.types import TestCase

from .analyze import analysis_from_json, heuristic_analyze
from .differential import Differential, DifferentialReport, Router
from .prompts import (
    build_analysis_prompt,
    build_candidate_prompt,
    build_differential_repair_prompt,
    build_inputs_prompt,
    build_oracle_prompt,
    dropped_definitions,
    extract_analysis,
    extract_code,
    extract_generator,
    extract_inputs,
    python_defect,
    rust_defect,
)
from . import solution_cache
from .config import env_on
from .rust_compile import compile_defect, rustc_path

# Per-case wall clock for every local run: grading against the bar, the size
# probe, and the too_slow / OOM verdicts are all calibrated to it. It EQUALS the
# validator's `per_test_timeout_s` (5.0) and that is not a coincidence to tune
# away: set lower, the probe calls programs slow that the validator passes and
# keeps them out of the cache; set higher, a program that times out there
# passes here. Overridable for experiments only.
VERIFY_TIMEOUT_S = float(os.environ.get("SOLVER_VERIFY_TIMEOUT_S", "5"))


# The one boolean grammar every solver setting is read with; see `config`.
_env_on = env_on

# The least a case may be given when the budget cannot afford the full timeout.
#
# `check` would rather shorten every case's clock than refuse cases outright --
# see there for why -- but only down to a point where a timeout still MEANS
# something. A correct answer to these problems runs in milliseconds, so one
# second is already a thousandfold margin, and a program that cannot finish in
# it is being reported honestly rather than harshly. Below this the number
# stops being evidence and starts being an artefact of the deadline, so the run
# gives up cases instead.
MIN_CASE_TIMEOUT_S = 1.0

# How long an executor that could not be built stays unavailable before the
# next solve is allowed to try again.
#
# A hold, not a verdict: a Docker daemon started after the miner is the
# ordinary case, and a permanent answer would mean grading nothing in that
# language until someone restarted the process. Long enough that Rust tasks
# stop paying for the probe, short enough that a box the operator has just
# fixed heals on its own.
EXECUTOR_RETRY_S = float(os.environ.get("SOLVER_EXECUTOR_RETRY_S", "300"))

# What must be left when a read stops, for the answer to reach the validator at
# all. THE ONLY reserve taken out of the deadline, and the only thing standing
# between a full-budget solve and a 504 with nothing in it.
#
# `handle_request` wraps the WHOLE solve -- including `fit_response`,
# `save_solution` and `save_exchange` -- in one `asyncio.wait_for(...,
# timeout=min(deadline_s + RESPONSE_GRACE_S, GLM_REQUEST_TIMEOUT_S))`, and a
# solve that overruns it is cancelled and answered 504 with nothing. So
# everything after the last read has to fit in here: `send`'s post-read phases
# (copy, stream, salvage, post-mortem -- `FULL_TAIL_S`, 11s, itself scaled
# down on short deadlines by `tail_budget`),
# then the archive writes and the response.
#
# There used to be a second reserve as well, `SOLVER_SAFETY_MARGIN_S` at 20s,
# which also held time back for GRADING. Two reserves for one deadline is one
# too many: the working budget was 20s short of the limit while the true limit
# was 12s short, and the 8s difference did nothing but complicate the
# arithmetic. One number, and it is the real one -- a 300s deadline now gives
# the reads 285s rather than 280s.
DELIVERY_RESERVE_S = 15.0

# What must be left after the LAST read for the answer to reach the wire --
# and so the point past which `DELIVERY_RESERVE_S` is not being saved for
# anything, and a read still in progress may have the rest of it.
#
# The reserve is sized for the browser backend, whose post-read tail is real:
# copy, stream, salvage and post-mortem, `FULL_TAIL_S`, 11 seconds. The CLI
# backend has no such tail. What actually happens after its last read is
# `_kill` -- and only when the turn is CUT, since a reply that finishes leaves
# the process exiting on its own -- which waits at most 5s for the child to
# die, and then `fit_response`, both archive writes and signing a maximum-size
# 128 KB payload, measured at 2.4 ms median and 3.2 ms worst over thirty runs.
# Six seconds covers the kill with the measured delivery three orders of
# magnitude inside it.
#
# So on the CLI backend roughly nine seconds of the reserve is held for work
# that costs milliseconds, and it is held hardest in the one case where it is
# worth least: a program turn cut mid-write ships a TRUNCATED program, which
# does not compile and scores zero, and the reserve is then protecting the
# delivery of a zero. `_attempt` hands that time to the read instead, and only
# when there is no finished program in hand to lose. The payment rule is why
# this is free: correctness is a hard gate, speed is a multiplier floored at
# 0.95, and the validator listens until `deadline_s + 10` while this miner
# stops at `deadline_s + 5`.
WIRE_TAIL_S = 6.0

# ---------------------------------------------------------------- the probe --
# The largest input the sandbox can hand back, MEASURED against the real
# executor rather than assumed.
#
# `subprocess_executor` caps captured stdout at `_MAX_OUTPUT_BYTES` = 256 KiB
# and keeps the TAIL of it, so a framed status line longer than that loses its
# opening marker and `_extract_framed` returns None. The caller is then told
# "sandbox produced no verdict (crashed or exited early)" -- which reads as a
# broken generator and is nothing of the kind. Bisected on this machine
# through `_Grader.outputs`: 261,093 bytes comes back, 262,187 does not, and
# 256 KiB is 262,144.
#
# The arm this replaces set its own limit to 1,000,000 and checked it AFTER
# the run, so the check could never fire: the run had already died. Measured
# over 102 production solves, that arm obtained 8 large inputs and reported 26
# crashed generators.
#
# 240 KiB, for the framing and the JSON quoting around the payload.
PROBE_MAX_BYTES = 240 * 1024

# The byte budgets offered to the generator, in order, until one comes back.
# Descending because the reason a budget fails is nearly always that what came
# back was too big for the pipe, and a program that is quadratic at the limit
# is quadratic at a quarter of it: 60 KiB of input is tens of thousands of
# values, where an O(n^2) loop is billions of operations and a five-second
# limit is not close.
PROBE_SCALES = (PROBE_MAX_BYTES, PROBE_MAX_BYTES // 4, PROBE_MAX_BYTES // 16)


# The least slice a read can be handed: `send(prompt, max(MIN_SLICE_S, left))`
# pads every slice up to it, so a round or a pass started with less than this
# left would run PAST the deadline by construction. Not a budget -- the
# granularity of the clock -- and the only thing besides the deadline itself
# that decides whether another round, pass or handoff is started.
MIN_SLICE_S = 1.0

# A repair round that changed nothing AND cost less than this did not involve
# the model at all: the read came back with text that was already on the page.
# Re-asking that costs no time, so the loop would resend the same prompt at
# machine speed until the deadline -- measured at 2,567 rounds for one answer
# that never changed. A round that took a real round trip is the other thing
# entirely: the model answered, and answered the same, and the next ask is a
# real chance the remaining budget is there to pay for. Time is what tells the
# two apart; the count cannot.
STALE_ROUND_S = 2.0






class Conversation(Protocol):
    """One live, isolated model conversation.

    The repair loop deliberately stays inside a single conversation so the
    model can see its own previous attempt alongside the failure report.
    """

    # Two arguments, always. A backend MAY also accept ``extend_to_s`` -- a
    # hard bound past the slice, for a caller that hands out less than its whole
    # remaining budget -- but nothing here does, so nothing here passes one.
    async def send(self, text: str, timeout_s: float) -> str: ...
    async def close(self) -> None: ...


class Backend(Protocol):
    # ``avoid`` names a provider to steer away from, so a second attempt can ask
    # a different model. It is a preference, not a guarantee.
    async def open(self, avoid: Optional[str] = None) -> Conversation: ...
    async def aclose(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


@dataclass
class Answer:
    """What a solver hands back to the miner.

    Structurally compatible with ``custom_miner.SolveResult`` (the miner reads
    ``.code``, ``.raw_response`` and, when present, ``.diagnostics``, which it
    archives beside the exchange) and defined here on purpose: importing
    the miner module from inside a request would make a path problem surface as
    a failed solve at serving time rather than at startup.
    """

    code: str
    raw_response: str = ""
    # Whether this answer reproduced every public example. A chain of providers
    # needs this to know when to stop trying; the miner itself ignores it.
    verified: bool = False
    passed: int = 0
    total: int = 0
    # The same question asked of the model's OWN cases, which on live traffic is
    # the only suite that ever runs. Kept beside `verified` rather than merged
    # into it -- see `Candidate.self_verified`.
    self_verified: bool = False
    self_passed: int = 0
    self_total: int = 0
    # Everything the summary line says, kept as data so the archive can hold
    # it beside the answer. The line is what an operator reads live; this is
    # what makes a solve answerable AFTER the fact -- which bar the program
    # cleared, which cases were disputed and how they were settled, whether
    # it was timed at scale, which models touched it. Without the bar itself
    # on disk, "it passed its own cases" is a claim with no way to check it.
    diagnostics: dict = field(default_factory=dict)


@dataclass
class Candidate:
    code: str
    raw: str
    passed: int = 0
    total: int = 0
    defect: Optional[str] = None
    failures: list[str] = field(default_factory=list)
    # Cases the MODEL wrote for its own program, kept apart from the validator's
    # examples on purpose. They are evidence, not verification: a model cannot
    # confirm its own reading of a statement, so folding these into
    # `passed`/`total` would let `verified` -- which gates the answer cache --
    # go True on nothing more than the model agreeing with itself.
    self_passed: int = 0
    self_total: int = 0
    from_self_tests: bool = False
    # How many of the model's own cases were IN HAND for this candidate,
    # whether or not there was time to run them. `from_self_tests` says they
    # ran; this says they existed. Keeping only the first made "no time to run
    # the cases" indistinguishable from "the model never sent any", and the
    # warning in `solve_task` said the second when the truth was the first.
    self_cases: int = 0
    # How many of `self_total` actually RAN. They differ when the budget cut a
    # run short, and without this the two are indistinguishable in the one line
    # an operator reads: `self=5/18` says five passed, and leaves them to assume
    # the other thirteen failed when in fact nobody looked at them. Kept beside
    # the counts rather than folded into them because `total` is deliberately
    # the full suite -- an unrun case is unknown, which is neither a pass nor a
    # failure, and that is exactly what keeps `self_verified` false.
    self_observed: int = 0
    # This reply is part of a program rather than a program: it uses something
    # only the round above it defined. Kept beside `defect` rather than folded
    # into it because `_supersedes` has to tell this apart from every other way
    # code can be wrong -- see there.
    partial: bool = False
    # The cases from `self_cases`' suite that this program did NOT pass. What
    # the differential found it disagreed with the reference on.
    failed_cases: list[dict[str, Any]] = field(default_factory=list)
    # What the program actually produced for each of those, in the same order.
    # `failures` renders this for a model to read; the value itself is what a
    # repair prompt quotes beside the reference's answer, so a model can see
    # the two side by side rather than be told they differ.
    failed_actuals: list["_Run"] = field(default_factory=list)
    # The cases this program was actually graded against. Kept for the
    # solution cache: an answer handed out again without being re-run is a
    # question about what it was checked against, and without the bar beside
    # it there is no way to ask.
    self_bar: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Every public example reproduced exactly."""
        return self.defect is None and self.total > 0 and self.passed == self.total

    @property
    def self_verified(self) -> bool:
        """Every case the MODEL wrote for itself reproduced, and it is all the
        evidence there was.

        Deliberately not folded into `verified`, and the reason is unchanged: a
        model cannot confirm its own reading of a statement, so a program that
        agrees with itself must not be able to earn the flag that gates the
        answer cache and tells a chain of providers to stop trying.

        But it is not nothing, either -- it is the ONLY evidence a live solve
        ever has. Production ships no `public_examples` at all, so `verified`
        is False on every real answer this miner sends, and a log that reports
        only that cannot tell "ran every case it had and passed" from "was
        never run at all". Those are the two ends of the range, and they read
        identically. This is the one that says which.

        `total == 0` is part of it: with public examples in hand THEY are the
        verdict, and `verified` already reports it.
        """
        return (
            self.defect is None
            and self.total == 0
            and self.self_total > 0
            and self.self_passed == self.self_total
            and not self.failures
        )

    @property
    def score(self) -> tuple[int, int, int, int]:
        """Ranking key for 'best so far' — the validator's examples, then the
        model's own cases, then non-empty, then runnable.

        Used BETWEEN PASSES only, where two models answered the same problem
        independently and neither saw the other -- there, grading is the only
        thing that can separate them. Within one pass the rounds are corrections
        of each other and the latest simply wins; see `_supersedes`, and the
        damage this ranking did when it was applied there too.

        A defect ranks BELOW clean code that merely could not be graded, and
        that is not cosmetic. Without a defect term at all, a first answer with
        no `fn main()` and a corrected second answer score identically -- a tie,
        which `>` loses, so the repair round lands a good program and the broken
        one is submitted anyway. The whole repair loop is dead weight for
        structural defects until this ranks them apart.

        Non-empty comes BEFORE runnable, and the order is the whole point.
        Emptiness is not a defect -- there is nothing there to be wrong -- so
        with the terms the other way round a round that captured nothing at all
        outranks a round that returned a program with a fixable flaw, and
        replaces it as "best". Both score zero on chain, but one of them is
        still an answer and the other is the absence of one, and the answer is
        the one to keep: `python_defect` is a static check, and a static check
        that is too strict must not be able to throw work away.
        """
        # `self_passed` sits second: below the validator's own examples, which
        # are ground truth, and above everything structural, because a program
        # that reproduces cases it was checked against is better evidence than
        # one that merely parses. A candidate with no self-tests scores 0 there
        # and ties with one that failed all of them -- deliberately. Having
        # tests must not rank an answer below not having them.
        return (
            self.passed,
            self.self_passed,
            1 if self.code.strip() else 0,
            0 if self.defect else 1,
        )


class _Phases:
    """Every phase of one pass, timed, on its own line as it finishes.

    The solve already reported one number at the end -- `0.3s/290s` -- and that
    is the only number it reported. It says a solve was fast or slow and
    nothing whatever about WHERE the time went, which is the single question an
    operator has when a 290-second budget comes back empty. Every defect fixed
    in this file was found by reconstructing that breakdown by hand from
    scattered lines, or by adding a stopwatch and running the solve again.

    Phases are numbered the way the prompts are: 1 is the cases turn, 2 is the
    program, 3 and up are corrections. Opening a tab is not a phase and is not
    numbered, but it IS timed -- a fleet with no free tab has spent whole
    budgets waiting, and that shows up here as a long `open` rather than as a
    solve that mysteriously started late.

    Each phase is measured from the end of the one before it, which is exact
    because they run back to back, and it means the caller records nothing and
    only says when a phase ended.

    Except for the ones that do not. A phase marked `beside=True` ran
    CONCURRENTLY with the phase after it -- the independent bar is written
    while the program is -- so it reports its own elapsed time, passed in by
    the caller who started it, and does not move the cursor the next phase is
    measured from. Letting it move the cursor would charge the program turn
    only the sliver between the two returning and quietly show a solve whose
    phases no longer sum to its duration.
    """

    def __init__(
        self, budget: float, started: float, pass_no: int = 1, ident: str = ""
    ) -> None:
        self._budget = budget
        self._solve_started = started
        self._pass = pass_no
        self._at = time.monotonic()
        self._wall = datetime.now()
        # The request this solve is for, at the end of every line. Solves
        # interleave -- validators send every four minutes or so and a solve
        # takes two -- and without it the phases of two solves read as one.
        self._ident = f"  id={ident}" if ident else ""

    def mark(
        self, label: str, *, model_s: Optional[float] = None,
        checked_s: Optional[float] = None, beside: bool = False,
        ended: Optional[float] = None,
    ) -> None:
        now, wall = time.monotonic(), datetime.now()
        if beside:
            spent = model_s if model_s is not None else now - self._at
            end_wall = wall if ended is None else wall - timedelta(seconds=now - ended)
            began = end_wall - timedelta(seconds=spent)
        else:
            # `ended` is the monotonic instant the phase actually finished,
            # for the one caller that marks a phase AFTER a later one has
            # already run: the program turn, when an empty bar is retried
            # between the turn returning and the turn being graded. Measured
            # without it -- program 0.3s, retry 0.4s -- the retry printed
            # "took 0.7s" and the program "took 0.2s": the program's seconds
            # credited to the retry, and the program left with the grading.
            # The cursor advances to where the phase ENDED, not to now, so the
            # phase marked next is measured from the right place.
            end = now if ended is None else ended
            spent = end - self._at
            end_wall = wall - timedelta(seconds=now - end)
            began, self._at, self._wall = self._wall, end, end_wall
        left = self._budget - (now - self._solve_started)
        parts = []
        if beside:
            parts.append("alongside")
        if model_s is not None:
            parts.append(f"model {model_s:.1f}s")
        if checked_s is not None:
            parts.append(f"checked {checked_s:.1f}s")
        detail = f"  ({', '.join(parts)})" if parts else ""
        # The pass number only when there IS more than one, so the ordinary
        # solve is not made to carry a column that always reads the same.
        where = f"pass {self._pass} " if self._pass > 1 else ""
        print(
            f"[phase] {where}{label:<14} "
            f"{began:%H:%M:%S}.{began.microsecond // 100000} "
            f"took {spent:6.1f}s{detail}"
            f"  — {max(0.0, left):.0f}s of {self._budget:.0f}s left{self._ident}"
        )


def _ident(task) -> str:
    """The request id as the log carries it: short, and empty when unknown."""
    return str(getattr(task, "problem_id", "") or "")[:12]






def _ago(when: Any) -> str:
    """`saved_at` as a human interval, for the cache-hit line."""
    try:
        seconds = max(0.0, time.time() - float(when))
    except (TypeError, ValueError):
        return "at an unknown time"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.0f}d ago"


def _model_of(provider: Optional[str]) -> str:
    """The model out of a provider label like `cli:opus@primary`."""
    name = (provider or "").split(":")[-1]
    return name.split("@")[0].strip().lower()


def _transport_limit(error: Optional[str]) -> bool:
    """Whether a failed run failed because the ANSWER would not fit back.

    `rlvr/execution/_batch_runner.py` caps a case's framed status at 256 KiB
    and the Rust runner caps raw stdout at 1 MB; past either, the run comes
    back failed with a message about size. That is the harness declining to
    carry a value, not the program doing anything wrong, and the two must not
    be reported the same way -- one is a defect worth a repair round and the
    other is the probe having asked for too much.
    """
    low = (error or "").lower()
    return "too large" in low or "too big" in low










def _record_differential(
    candidate: Candidate, report: "DifferentialReport",
    inputs: list[dict[str, Any]],
) -> None:
    """Write what the comparison found onto the candidate.

    The rest of the solver -- `_supersedes`, the solution cache, the summary
    line -- already speaks in `self_passed`/`self_total`, so the differential
    reports through those rather than beside them. The mapping is the strict
    one on purpose:

    `self_total` is every input, INCLUDING the ones the reference could not
    answer and the ones the clock never reached. `self_passed` is only the
    inputs where both programs ran and agreed. So `self_verified`, which wants
    `self_passed == self_total`, stays false for a suite that was cut short or
    whose reference fell over -- neither of which is evidence about the
    program, and neither of which may open the answer cache.
    """
    candidate.self_cases = len(inputs)
    candidate.self_total = len(inputs)
    candidate.self_passed = report.agreed
    candidate.self_observed = report.agreed + report.mismatch
    candidate.from_self_tests = report.ran > 0
    candidate.failures = [
        f"{case.name}: {case.detail}" for case in report.cases
    ]
    candidate.self_bar = list(inputs)


def _inherit_evidence(candidate: Candidate, prior: Candidate) -> None:
    """Give `candidate` what grading already established about the same source.

    Evidence belongs to the CODE, not to the round that happened to run it, and
    two rounds carrying byte-identical source are the same program however
    differently they were read. Not a ranking, and not in tension with "the
    latest version wins": the answer that ships is unchanged either way. What
    changes is what is known about it.

    Both losses this repairs were measured.

    A round that arrives with the budget gone is not graded at all -- `_grade`'s
    own gate refuses to start a run there is no time for -- so it reports 0 of
    0. When the source is one an earlier round already ran, that is not "this
    answer was never checked" but "this answer was checked and nobody wrote it
    down": the operator's log showed `self=17/20` on one round and `self=0/0` on
    the answer that shipped, which was the same program.

    And `partial` was worse than lost, it was CLEARED. A reply that corrects
    only the cases is graded against the program already in hand, so the
    "previous" it is compared to is itself -- `dropped_definitions` finds
    nothing missing, because nothing can be missing from a comparison with
    itself. A fragment flagged one round earlier came out of that looking like a
    whole program, which is precisely the flag `_supersedes` relies on to keep a
    fragment from displacing one.
    """
    if not (candidate.total or candidate.self_total):
        # Nothing ran for this one, so there is nothing of its own to overwrite.
        # All of it moves together: a pass count without the failures it came
        # with would be a reading nobody could act on.
        #
        # Only when nothing ran. A round that WAS graded has the current answer
        # -- most sharply when the bar moved under it, which is the whole point
        # of a corrected case array: the same program that failed one round
        # passes the next, and carrying the old failures forward there would
        # re-report a disagreement that no longer exists and cost a round trip
        # doing it.
        candidate.passed, candidate.total = prior.passed, prior.total
        candidate.self_passed, candidate.self_total = prior.self_passed, prior.self_total
        candidate.self_observed = prior.self_observed
        candidate.failures = list(prior.failures)
        # WITH the cases they came from. A repair prompt quotes the concrete
        # inputs the two programs disagreed on, so inherited failure TEXT
        # without the cases behind it describes evidence the round no longer
        # holds. The feature and its own evidence move together.
        candidate.failed_cases = list(prior.failed_cases)
        candidate.failed_actuals = list(prior.failed_actuals)
        if not candidate.self_bar:
            candidate.self_bar = list(prior.self_bar)
        candidate.from_self_tests = prior.from_self_tests
        candidate.defect = candidate.defect or prior.defect
    if not candidate.self_cases:
        candidate.self_cases = prior.self_cases
    # One-way, and independent of the above. A fragment does not stop being a
    # fragment because a later round had no round above it to miss anything
    # from.
    candidate.partial = candidate.partial or prior.partial


def _supersedes(candidate: Candidate, best: Candidate, still_writing: bool) -> bool:
    """Should `candidate` replace `best` as the answer that ships?

    THE LATEST VERSION WINS. No score is compared, and that is the whole rule.

    A round only happens because the one before it was wrong: the loop ends the
    moment there is no defect and no failure, so every candidate after the first
    exists BECAUSE the model was shown what was wrong with its predecessor and
    asked to correct it. The later program is the corrected one. Ranking them
    against each other asks a question that has already been answered.

    Scoring them did real damage. `Candidate.score` cannot tell "failed its
    tests" from "was never tested" -- both put 0 in the same slot -- so a
    correction that arrived too late in the budget to grade scored (0,0,1,1)
    against the wrong program's (0,1,1,1) and LOST to the answer it was
    correcting. Reproduced end to end: phase 3 returned the right program,
    `self=1/3` went out, and the file held phase 2's code. Every refinement of
    the comparison was another way to get that wrong; not comparing cannot.

    Two things are still not versions of the answer, and neither is a judgement
    about how good the code is:

    * NOTHING ARRIVED. An empty capture is the absence of an answer rather than
      a worse one -- a dead tab, a reply that rendered as prose, a read that
      timed out. It must never displace a program already in hand.
    * THE MODEL IS STILL WRITING. What arrived is a fragment of a reply rather
      than a revision of one, and a fragment that happens to parse must not
      displace the finished program above it.
    * ONLY THE PART THAT CHANGED ARRIVED. The same thing said by the model
      instead of by the clock: a round asked for the whole program sent back
      the one function it fixed, and it uses a helper that lives in the round
      above. `dropped_definitions` is what knows. This is not a judgement about
      which program is better -- a fragment is not a program, and every hidden
      test would die on `NameError` for a helper that was right there.

    None of the three ranks anything. Two fragments still go latest-first,
    because between two fragments the later one is still the correction.
    """
    if not candidate.code.strip():
        return False
    if still_writing:
        return not best.code.strip()
    if candidate.partial and best.code.strip() and not best.partial:
        return False
    return True


class _Grader:
    """Lazily-built executors, reused across solves (Docker startup is slow)."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._settings = self._build_settings()
        # Built from worker threads now (see `_graded`), and concurrently: the
        # miner serves several solves at once. Two threads missing the cache
        # together would each construct an executor, and for the Docker backend
        # that is a container's worth of startup thrown away -- on the one code
        # path whose entire reason for caching is that Docker startup is slow.
        self._lock = threading.Lock()
        # An executor that could not be BUILT, by language: (when, why).
        #
        # `self._cache` is written after `get_executor` returns, so a
        # constructor that raises leaves nothing behind and every later solve
        # repeats it. For Rust without Docker that is
        # `DockerExecutor._resolve_docker` shelling out to `docker info` --
        # 60ms when the socket is simply absent, and up to its own 20 second
        # timeout when the daemon is hung or still starting. Per Rust task,
        # inside the solve's budget, for the life of the process.
        self._broken: dict[str, tuple[float, str]] = {}
        self._reported: set[str] = set()
        # A language being graded under the subprocess FALLBACK, by when it
        # fell back. The fallback used to go into `_cache` like a real
        # executor and stay there for the life of the process: a Docker
        # daemon started after the miner -- the ordinary case, and the one
        # `EXECUTOR_RETRY_S` exists for -- was never looked for again, Python
        # was graded at 1 GiB for days, and `state()` said "ready".
        self._fallback_since: dict[str, float] = {}

    @staticmethod
    def _build_settings():
        from rlvr.config import Settings

        # `docker`, matching the validator. See the module docstring: the
        # subprocess backend grades under 1 GiB where the validator gives
        # 256 MiB with swap off, so an answer can pass every case here and
        # lose the whole hidden suite to an OOM that this never saw.
        kind = os.environ.get("SOLVER_VERIFY_EXECUTOR", "docker")
        # _env_file=None so the miner's .env cannot accidentally repoint this.
        return Settings(_env_file=None, executor=kind, per_test_timeout_s=VERIFY_TIMEOUT_S)

    def _current(self, language: str):
        """The cached executor, unless it is a fallback due for another try."""
        cached = self._cache.get(language)
        if cached is None:
            return None
        since = self._fallback_since.get(language)
        if since is not None and time.monotonic() - since >= EXECUTOR_RETRY_S:
            return None
        return cached

    def degraded(self, language: str) -> bool:
        """Whether this language is graded under looser limits than the
        validator's. Read by the solution cache's gate: a size probe that
        passed at 1 GiB has not shown what 256 MiB would do."""
        return language in self._fallback_since

    def executor(self, language: str):
        cached = self._current(language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._current(language)
            if cached is not None:
                return cached
            held = self._broken.get(language)
            if held is not None and time.monotonic() - held[0] < EXECUTOR_RETRY_S:
                # Remembered, not re-probed. The message is the original one
                # verbatim: the caller prints it per solve and an operator
                # counts those lines, so it must not change shape here.
                raise RuntimeError(held[1])
            try:
                executor, degraded = self._build(language)
            except Exception as exc:  # noqa: BLE001 - remembered, then re-raised
                self._broken[language] = (time.monotonic(), f"{exc}")
                self._report_unavailable(language, exc)
                raise
            self._broken.pop(language, None)
            self._cache[language] = executor
            if degraded:
                self._fallback_since[language] = time.monotonic()
            else:
                self._fallback_since.pop(language, None)
            return executor

    def state(self, language: str) -> str:
        """What grading this language would do right now. Never probes.

        Read from `/solver-status`, so it must not shell out: a `docker info`
        against a hung daemon blocks for twenty seconds, and the endpoint an
        operator polls to find out whether the miner is healthy is the last
        place to put that.
        """
        if language in self._fallback_since:
            return (
                f"degraded: subprocess at 1 GiB, Docker unavailable "
                f"(tried again every {EXECUTOR_RETRY_S:.0f}s)"
            )
        if language in self._cache:
            return "ready"
        held = self._broken.get(language)
        if held is not None:
            return f"unavailable: {held[1]}"
        return "not checked yet"

    def _build(self, language: str):
        """Construct the executor for ``language``. Assumes the lock is held.

        Returns `(executor, degraded)`: `degraded` is True when Python fell
        back to the subprocess backend, so the caller can remember to try
        Docker again later and say so in `state()`.

        Python falls back to the subprocess backend when Docker cannot be
        built, and that asymmetry with Rust is the whole point. Rust has no
        subprocess path -- rustc lives in the pinned image -- so a missing
        daemon there means the language cannot be graded and saying so is the
        only honest answer. Python can be graded either way, and the choice is
        between grading it under limits that are looser than the validator's
        and not grading it at all. The first is worth much more than the
        second: a bar that runs catches wrong answers whatever the memory cap,
        while `executor()` remembering the failure would leave every Python
        solve for the next `EXECUTOR_RETRY_S` ungraded and unrepaired.

        The cost is stated rather than hidden. Under this fallback an answer
        that would be OOM-killed at 256 MiB passes here, so the line names the
        limit the validator will actually apply.
        """
        from rlvr.execution.executor import get_executor

        settings = self._settings
        if language == "rust" and settings.executor != "docker":
            # Rust needs rustc in the pinned image; no subprocess path.
            from rlvr.config import Settings

            settings = Settings(
                _env_file=None,
                executor="docker",
                per_test_timeout_s=VERIFY_TIMEOUT_S,
            )
        try:
            return get_executor(settings, language=language), False
        except Exception as exc:  # noqa: BLE001 - Python degrades, Rust cannot
            if language == "rust" or settings.executor != "docker":
                raise
            from rlvr.config import Settings

            fallback = get_executor(
                Settings(
                    _env_file=None,
                    executor="subprocess",
                    per_test_timeout_s=VERIFY_TIMEOUT_S,
                ),
                language=language,
            )
            if "python-fallback" not in self._reported:
                self._reported.add("python-fallback")
                print(
                    f"[verify] grading Python under subprocess: Docker is "
                    f"unavailable ({exc}). The validator grades in a container "
                    f"with 256 MiB and no swap, and one OOM there fails every "
                    f"hidden test — an answer that allocates more than that "
                    f"will pass here and score zero. Docker is looked for "
                    f"again every {EXECUTOR_RETRY_S:.0f}s. Once per run."
                )
            return fallback, True

    def _report_unavailable(self, language: str, exc: BaseException) -> None:
        """Say what a missing executor costs, once per language per run.

        The per-solve line names the exception and nothing else, which reads as
        a hiccup. This is the part an operator has to act on: everything in
        this language is now ungraded, and no repair round can fire on it.
        """
        if language in self._reported:
            return
        self._reported.add(language)
        # The hint, not just the error. A daemon that is UP and unreachable
        # reports exactly like one that is not running -- read off a production
        # run, where every Rust answer went out ungraded behind "permission
        # denied ... /var/run/docker.sock", which is a group membership and not
        # a broken install.
        from .rehearse import _executor_hint

        fix = _executor_hint(exc) or ""
        if not fix and language == "rust":
            fix = (
                " Rust has no subprocess path — grading it at all needs a Docker "
                "daemon; see the README's \"Rust needs Docker\" section."
            )
        print(
            f"[verify] the {language} executor could not be built, so no {language} "
            f"answer can be graded here: no repair rounds, and verified=False "
            f"however good the answer is. Not probing again for "
            f"{EXECUTOR_RETRY_S:.0f}s. Once per run.\n"
            f"           {type(exc).__name__}: {exc}{fix}"
        )

    def check_detailed(
        self, code: str, language: str, entrypoint: str,
        examples: list[dict[str, Any]], names: Optional[list[str]] = None,
        budget_s: Optional[float] = None,
    ) -> tuple[int, int, list[str], list[dict[str, Any]], list["_Run"]]:
        """Run ``code`` against the examples.

        Returns ``(passed, total, failures, failed, actuals)`` -- the counts,
        one line of concrete evidence per failure for the repair prompt, the
        failing cases themselves, and what the program actually produced for
        each of them. The last is what lets a correction be merged into the
        suite rather than replace it: the repair prompt shows the model the
        cases that failed and offers to take them back corrected, and knowing
        WHICH cases those were is the difference between correcting a bar and
        letting a model rewrite it.

        ``budget_s`` bounds what the RUN may cost, and it is the difference
        between a check and a solve-ending one. Every case gets
        `VERIFY_TIMEOUT_S` and nothing used to bound the set, so a suite of
        twenty cases against a program that hangs costs twenty times that:
        measured, 100.2 seconds. `_grade`'s gate demanded only 15 seconds be
        left before starting it -- a 6.7x under-estimate, and the run then took
        the rest of the deadline with it. Two of those and a 290-second solve grades nothing and submits
        unverified, which is the failure this argument exists to end.

        A partial run is the point, and it is what the gate's own comment
        already promised: "a partial run that DOES fit is worth more than no
        evidence at all". `total` stays the FULL count, so an unrun case is
        neither a pass nor a failure but simply unknown -- and `passed < total`
        keeps `verified` and `self_verified` false, because three of twenty
        passing is not twenty passing.

        ``names`` labels each failure with the case it came from. Optional
        because the repair prompt does not want it -- the model is being shown
        concrete inputs and outputs, and an authored title is noise there. A
        person reading a test report wants the opposite: `case 3 'resize in
        place'` says which behaviour broke, where a wall of arguments has to be
        decoded first.
        """
        cases = [
            TestCase(
                args=list(case.get("args", []) or []),
                kwargs=dict(case.get("kwargs", {}) or {}),
                expected=case.get("expected"),
            )
            for case in examples
        ]
        if not cases:
            return 0, 0, [], [], []
        executor = self.executor(language)
        started = time.monotonic()
        # What each case was actually given. Reported in the failure text, so a
        # timeout says the clock it was measured against rather than a constant
        # the run may not have been able to afford.
        per_case = VERIFY_TIMEOUT_S
        results: list[Any] = []
        ran = 0
        last_call_s = 0.0
        while ran < len(cases):
            spent = time.monotonic() - started
            left = None if budget_s is None else float(budget_s) - spent
            chunk = len(cases) - ran
            if left is not None:
                # `last_call_s` is the whole cost of the previous call, fixed
                # overhead included, and that overhead is not small everywhere:
                # the Docker executor adds `5 + 0.025n` seconds and a container
                # start to EVERY call, and the Rust one adds a container and a
                # full rustc build on top. Sizing by `VERIFY_TIMEOUT_S` alone
                # models none of that, so a short budget -- which is exactly
                # when this loop chunks -- issued call after call each costing
                # far more than the arithmetic allowed for. Refusing a call the
                # last one's own cost says will not fit is the bound that
                # actually holds, and on the subprocess executor, where a call
                # costs milliseconds, it never fires.
                if ran and (left <= 0 or left < last_call_s):
                    break
                # Buy TIME PER CASE, not cases at a fixed price.
                #
                # The rule here used to be `chunk = left // VERIFY_TIMEOUT_S`:
                # every case costs the full timeout, so a short budget buys few
                # cases and the rest are dropped. On a batch executor that is
                # exactly backwards, because the cost is dominated by the CALL
                # rather than by what is in it -- the Docker executor starts a
                # container per `run_tests`, and the Rust one a container and a
                # full rustc build. Dropping cases to make the arithmetic fit
                # then pays that fixed cost once per surviving case.
                #
                # Measured against an executor with a 1.2s call cost and 18
                # cases, which is the shape of the log this was written from:
                #
                #   budget    calls   graded   spent
                #   none          1    18/18    1.38s
                #   60s           2    18/18    2.58s
                #   20s           7    18/18    8.58s
                #   8s            6     6/18    7.26s   <- six containers for
                #                                          six cases, and one
                #                                          call would have run
                #                                          all eighteen in 1.4s
                #
                # So the budget is spread across the cases instead: each gets
                # `left / remaining`, and they go in ONE call. A suite that is
                # merely fast then always runs whole, because being fast is
                # precisely what makes a short clock enough for it.
                #
                # `MIN_CASE_TIMEOUT_S` is where this stops. Below a second a
                # timeout says more about the deadline than about the program,
                # so rather than shorten further the run does what it used to do
                # and takes fewer cases -- still in one call. That also keeps
                # the original protection intact: a suite that HANGS is bounded
                # by `left` either way, and now reports more failures for the
                # same seconds, because more of it ran.
                #
                # Always at least one case, so a check never reports nothing at
                # all, which bounds the overrun at one per-case timeout -- now
                # at most `MIN_CASE_TIMEOUT_S` when the clock is short, where it
                # used to be the full five seconds.
                if VERIFY_TIMEOUT_S > 0:
                    per_case = min(
                        VERIFY_TIMEOUT_S,
                        max(MIN_CASE_TIMEOUT_S, left / max(1, chunk)),
                    )
                    chunk = max(1, min(chunk, int(left // per_case)))
            call_started = time.monotonic()
            batch = executor.run_tests(
                code, entrypoint, cases[ran:ran + chunk], per_case
            )
            last_call_s = time.monotonic() - call_started
            results.extend(batch)
            ran += chunk
            if len(batch) != chunk:
                # An invariant both executors keep -- one result per test, on
                # every path including the failure ones -- and chunking is what
                # made this code DEPEND on it. A short batch shifts every result
                # after it against the case it belongs to, so a later failure
                # would be reported with the wrong inputs; and the executor is
                # an operator setting (`SOLVER_VERIFY_EXECUTOR`), so a backend
                # this file has never seen can be in the loop. Stop at the last
                # alignment that is certainly right: the rest count as unrun,
                # which they are.
                break
        # `len(results)`, not `ran`: they differ only when a batch came back
        # short, and there the results are what actually ran.
        running = cases[:len(results)]
        if len(running) < len(cases):
            # Which of the two reasons, because they call for different things
            # from whoever reads the log. A budget that ran out is ordinary and
            # says the deadline was tight; an executor that returned fewer
            # results than tests is a broken backend and says so.
            why = (
                f"fit in the {float(budget_s):.0f}s left"
                if budget_s is not None and len(results) == ran
                else "came back from the executor"
            )
            print(
                f"[verify] {len(running)} of {len(cases)} case(s) {why}; the "
                f"rest are unrun rather than passed, so this answer cannot "
                f"read as verified"
            )
        failures: list[str] = []
        failed: list[dict[str, Any]] = []
        actuals: list[_Run] = []
        passed = 0
        for index, (result, case) in enumerate(zip(results, running)):
            if result.passed:
                passed += 1
                continue
            label = ""
            if names and index < len(names) and names[index]:
                label = f"case {index + 1} {names[index]!r}: "
            failures.append(
                label + _describe(result, case, language, entrypoint, per_case)
            )
            failed.append(examples[index])
            # What the program actually produced, kept beside the case it
            # produced it for. `_describe` renders this for a model to read;
            # the differential needs the value itself, because a repair prompt
            # shows what each program returned for the same input.
            actuals.append(
                _Run(
                    ok=bool(getattr(result, "value_ok", False))
                    and not result.timed_out and not result.error,
                    value=getattr(result, "value", None),
                    error=result.error,
                    timed_out=bool(result.timed_out),
                    runtime_ms=float(getattr(result, "runtime_ms", 0.0) or 0.0),
                )
            )
        return passed, len(cases), failures, failed, actuals

    def check(
        self, code: str, language: str, entrypoint: str,
        examples: list[dict[str, Any]], names: Optional[list[str]] = None,
        budget_s: Optional[float] = None,
    ) -> tuple[int, int, list[str], list[dict[str, Any]]]:
        """`check_detailed` without the actual values. See it for everything.

        Kept as the four-tuple every caller but the differential wants,
        rather than widened in place: unpacking is positional, and a fifth
        element would have been a silent `ValueError` in each of them.
        """
        passed, total, failures, failed, _ = self.check_detailed(
            code, language, entrypoint, examples, names=names,
            budget_s=budget_s,
        )
        return passed, total, failures, failed

    def outputs(
        self, code: str, language: str, entrypoint: str,
        inputs: list[dict[str, Any]], budget_s: Optional[float] = None,
    ) -> list["_Run"]:
        """Run ``code`` on ``inputs`` and return what it produced, judged by
        nobody: one `_Run` per input, in order, with the sandbox's own value.

        Grading without a bar. `check` decides pass or fail against an
        expectation; this has none, because the point is to compare two
        programs' outputs with each other and put the difference to a reader.
        Same executor, same per-case timeout, same budget discipline: the run
        stops at what fits, and an input that did not run comes back as
        `ok=False, error="unrun"` rather than as anything it did not do.
        """
        # The Rust runner compares stdout with `expected` unasked, and its
        # token split reads a None as a crash; an empty string compares like
        # any other and is ignored here just the same.
        cases = [
            TestCase(
                args=list(case.get("args", []) or []),
                kwargs=dict(case.get("kwargs", {}) or {}),
                expected="" if language == "rust" else None,
            )
            for case in inputs
        ]
        if not cases:
            return []
        executor = self.executor(language)
        started = time.monotonic()
        results: list[Any] = []
        ran = 0
        last_call_s = 0.0
        while ran < len(cases):
            spent = time.monotonic() - started
            left = None if budget_s is None else float(budget_s) - spent
            chunk = len(cases) - ran
            per_case = VERIFY_TIMEOUT_S
            if left is not None:
                if ran and (left <= 0 or left < last_call_s):
                    break
                if VERIFY_TIMEOUT_S > 0:
                    per_case = min(
                        VERIFY_TIMEOUT_S,
                        max(MIN_CASE_TIMEOUT_S, left / max(1, chunk)),
                    )
                    chunk = max(1, min(chunk, int(left // per_case)))
            call_started = time.monotonic()
            batch = executor.run_tests(code, entrypoint, cases[ran:ran + chunk], per_case)
            last_call_s = time.monotonic() - call_started
            results.extend(batch)
            ran += chunk
            if len(batch) != chunk:
                break
        runs = [
            _Run(
                ok=bool(getattr(r, "value_ok", False)),
                value=getattr(r, "value", None),
                error=getattr(r, "error", None),
                timed_out=bool(getattr(r, "timed_out", False)),
                runtime_ms=float(getattr(r, "runtime_ms", 0.0) or 0.0),
            )
            for r in results
        ]
        runs += [_Run(ok=False, error="unrun")] * (len(cases) - len(runs))
        return runs


class _NoExecutor(Exception):
    """The size probe has no executor to run on. Private to it."""


class _Probe(NamedTuple):
    """What the size probe found, and how much it is worth.

    `sentence` is the report a repair prompt carries, and is None unless the
    program has to change. `state` says what the None means, because four
    different Nones used to mean four different things to a caller that could
    only see one:

    * `passed`   -- the program ran at the statement's own scale and finished
                    inside the per-test limit. EVIDENCE, and the only state
                    `solution_cache.worth_keeping` accepts.
    * `too_slow` -- it ran and did not finish; `oom` -- it ran and was killed
                    for memory. Both carry a sentence and both are verdicts.
    * `crashed`  -- it ran and died some other way. Far likelier the
                    generator handing it an input the statement does not
                    allow than a fault in the program, so no sentence -- but
                    not a pass either, and never cached as one.
    * `skipped`  -- the clock ran out before a run could be finished. Nothing
                    is known.
    * `none`     -- no generator, no valid large input, or no executor.
                    Nothing is known.

    Returned rather than latched on the solver, because the solver serves
    several solves at once and a flag on `self` was read by whichever of them
    finished last.
    """

    sentence: Optional[str]
    state: str


@dataclass
class _Run:
    """One input's outcome from `_Grader.outputs`."""

    ok: bool
    value: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    runtime_ms: float = 0.0


# What an out-of-memory kill looks like coming back from the container.
#
# The validator's Docker executor cannot always tell an OOM from a timeout --
# both can exit 137 -- so it inspects the stopped container and, failing that,
# says so in words: "container killed (likely OOM / memory limit)"
# (`rlvr/execution/docker_executor.py`). `MemoryError` is the subprocess
# backend's version, where `RLIMIT_AS` makes the allocation fail inside the
# interpreter rather than killing the process.
#
# Matched on TEXT because that is all `_Run` carries, and matched on WHOLE
# PHRASES because the text is a traceback: it carries the program's own
# identifiers and messages, and a bare `oom` is inside `rooms`, `bloom` and
# `zoom`. Every phrase here is one the executors write for a memory kill --
# `container killed (likely OOM / memory limit)` and `exceeded its memory
# limit` from the two Docker executors, `MemoryError` from the interpreter
# under `RLIMIT_AS` -- and for nothing else. A looser pattern turned an
# ordinary crash at scale into a repair round that rewrote a correct program
# to fix a memory problem it did not have.
_OOM_RE = re.compile(
    r"likely oom|memory limit|\bmemoryerror\b|\bout of memory\b|\boom[- ]?killed\b",
    re.IGNORECASE,
)


def _looks_out_of_memory(run: "_Run") -> bool:
    """True when this run died for lack of memory rather than of time."""
    if run.timed_out:
        return False
    return bool(_OOM_RE.search(run.error or ""))


# Text that changes between two runs of the SAME failure, and nothing else.
#
# Both sandboxes name their working directory after a random suffix --
# `tempfile.TemporaryDirectory(prefix="rlvr_sbx_")` in the validator's own
# subprocess executor, `prefix="hone-rustc-"` in the Rust compile check -- and
# that directory is inside every traceback line and every rustc diagnostic. So
# a program that crashed identically on two rounds produced two different
# failure strings, and the loop's repeat detector, which is nothing but a
# comparison of those strings, could never fire on a crashing program at all.
# The default `repr` of an object carries the same problem in the form of a
# heap address.
#
# Narrow on purpose. This runs over failure text that the MODEL reads, and a
# normaliser that also rewrote returned values would hide the very difference
# the model is being asked about -- an expected `"0xdeadbeef"` is an ordinary
# string. The address pattern therefore matches only CPython's `<... at 0x...>`
# form, which no test value wears by accident.
_SANDBOX_PATH_RE = re.compile(r"(?:/[^\s\"']*/)?(?:rlvr_sbx_|hone-rustc-)[A-Za-z0-9_.-]+")
_HEAP_ADDRESS_RE = re.compile(r"(<[^<>]*? at )0x[0-9a-fA-F]+(>)")


def _stable(text: Optional[str]) -> Optional[str]:
    """`text` with the per-run noise taken out, or None unchanged."""
    if not text:
        return text
    return _HEAP_ADDRESS_RE.sub(r"\g<1>0x...\g<2>", _SANDBOX_PATH_RE.sub("<sandbox>", text))


def _describe(
    result, case: TestCase, language: str, entrypoint: str,
    timeout_s: Optional[float] = None,
) -> str:
    """One line of concrete evidence for the repair prompt.

    ``timeout_s`` is the clock this case was actually given, which is not always
    `VERIFY_TIMEOUT_S`: when the budget cannot afford the full timeout for every
    case, `check` shortens it rather than dropping cases. Reporting the constant
    there would tell the model its program failed to finish in five seconds when
    it was never given five seconds -- and a model told that goes looking for a
    performance problem it may not have.
    """
    limit = VERIFY_TIMEOUT_S if timeout_s is None else timeout_s
    if language == "rust":
        call = f"stdin={_clip(case.args[0] if case.args else '')!r}"
    else:
        call = f"{entrypoint}(*{case.args!r}, **{case.kwargs!r})"
    if result.timed_out:
        short = (
            "" if limit >= VERIFY_TIMEOUT_S
            else f" — that was all the time left, not the usual {VERIFY_TIMEOUT_S:.3g}s"
        )
        return (
            f"{call} timed out after {limit:.3g}s (too slow or an infinite "
            f"loop){short}"
        )
    if result.error:
        return f"{call} raised: {_clip(_stable(_bound(result.error)), 300)}"
    actual = result.value if result.value_ok else result.actual_repr
    # Bounded, then stabilised, then clipped. The middle step is the one that
    # needs the first: `_SANDBOX_PATH_RE` costs O(slashes x run length) on a
    # slash-dense token, and `result.value` is the candidate's return value
    # deserialised in full with no cap on it at all -- so a program that
    # returns a long path-like string made each failure line cost seconds,
    # measured at 2.4s for 39KB and 47s for 180KB. All of it spent AFTER
    # `run_tests` returned, which puts it outside `budget_s`, outside
    # `VERIFY_TIMEOUT_S` and outside the round trip `_grade` holds back. The
    # bound is far larger than anything `_clip` would keep, so nothing a reader
    # or a model would have seen is lost.
    #
    # Clipping LAST still, so a heap address the clip cuts through is
    # normalised before it is cut rather than left half-written.
    return (
        f"{call} returned {_clip(_stable(_bound(repr(actual))))}, "
        f"expected {_clip(repr(case.expected))}"
    )


# The most text `_stable` is ever asked to scan. See `_describe`.
_STABLE_SCAN_LIMIT = 8192


def _bound(text: Optional[str]) -> Optional[str]:
    if text is None or len(text) <= _STABLE_SCAN_LIMIT:
        return text
    return text[:_STABLE_SCAN_LIMIT]


def _clip(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# No time floor gates a further PASS either. Whether another model is asked
# is a question about how many have been asked (`MAX_PASSES`,
# `SECOND_OPINION_PASSES`) and about the deadline, and nothing else: a pass
# opened with seconds left is cut by the same clock as everything else, and
# an empty answer pays exactly zero however early it is sent.

# How many models one task may be put to.
#
# `SECOND_OPINION_PASSES` is the policy for an answer that came back WRONG: ask
# the other model once, and stop. There is something worth submitting either
# way, and each further ask spends a real account's quota to improve on it.
#
# `MAX_PASSES` applies only while holding NOTHING, where that reasoning does not
# apply at all -- there is no answer to improve on and an empty one pays zero.
# Even so it is a runaway guard rather than a target: the clock stops the loop
# long before this does.
SECOND_OPINION_PASSES = 2
MAX_PASSES = 4

class _Shipped(NamedTuple):
    """What the pass that produced the shipped answer did.

    Captured as that pass wins and read when the summary line is built, which
    is the whole point: `plan` is overwritten by every later pass, and a pass
    that scores lower does not replace `best`, so reading `plan` at the end
    described a solve using the numbers of a program that was thrown away.

    Named rather than positional because it is no longer four things. It began
    as `(exit, bar_provider, rounds, corrected)` and every field since -- the
    probe verdict, the second bar, the adjudications, the repair chain -- is
    another index nobody reading `shipped[5]` could identify.
    """

    exit: str = ""
    rounds: int = 0
    corrected: int = 0
    probe: str = ""
    tests_provider: Optional[str] = None
    oracle_provider: Optional[str] = None
    ran: int = 0
    agreed: int = 0
    mismatch: int = 0
    oracle_crash: int = 0
    repair: tuple = ()
    patched: tuple = ()
    regressed: int = 0
    flipped: int = 0

    @classmethod
    def of(cls, plan: "_Plan") -> "_Shipped":
        return cls(
            exit=plan.exit, rounds=plan.rounds, corrected=plan.corrected,
            probe=plan.probe, tests_provider=plan.tests_provider,
            oracle_provider=plan.oracle_provider, ran=plan.ran,
            agreed=plan.agreed, mismatch=plan.mismatch,
            oracle_crash=plan.oracle_crash, repair=tuple(plan.repair),
            patched=tuple(plan.patched), regressed=plan.regressed,
            flipped=plan.flipped,
        )


class _Plan:
    """What one solve did, gathered as it happens. One per `solve_task`.

    Per-solve rather than per-solver: solves run concurrently on one instance,
    and a counter on `self` would mix one task's rounds into another's.

    Everything on it is written to be READ -- it is what `summary()` turns
    into the one `[verify]` line an operator greps, and every field on it
    earns its place by being the detector for a named failure. Nothing here
    steers the solve.
    """

    def __init__(self) -> None:
        # What the summary line reports beside the tally, so a hidden-suite
        # outcome can be joined back to how the solve went: rounds asked, and
        # cases the model corrected that stood. Measured need: a log of 76
        # solves graded 83.5% right, and no way to tell a corrected-bar solve
        # from a clean one.
        self.rounds = 0
        self.corrected = 0
        # WHICH MODEL wrote the inputs and which wrote the reference. With
        # five phases that can each name a different model, "which model
        # answered" stopped being one question -- and these two decide whether
        # a disagreement meant anything. A reference written by the same model
        # at the same effort as the candidate is not an independent reading of
        # the statement, and its agreement is not evidence.
        self.tests_provider: Optional[str] = None
        self.oracle_provider: Optional[str] = None
        # What running the two programs on the same inputs established, for
        # the answer that SHIPPED. `ran=0` is the case worth being able to
        # see: the old line printed `corrected=0/18 disagreed=0/18
        # exit=converged` for a solve that had established nothing, which is
        # the same line a solve that checked eighteen cases and passed them
        # all prints. These four tell them apart.
        self.ran = 0
        self.agreed = 0
        self.mismatch = 0
        self.oracle_crash = 0
        # Which condition ended the loop, and what the SIZE PROBE said about
        # the program that shipped. See `note_exit` and `note_probe`.
        self.exit = ""
        self.probe = ""
        # Which models the repair rounds ran on, in order, and WHICH ARTIFACT
        # each round patched -- `cand` or `oracle`. The second is the
        # instrument for the routing heuristic: a mismatch blames the
        # candidate and a reference crash blames the reference, and neither is
        # certain. Without a record of what each round chose there is no way
        # to ask afterwards whether it chose right.
        self.repair: list[str] = []
        self.patched: list[str] = []
        # Repair rounds that did NOT improve the verdict score. Repair is
        # monotone -- a round that scores no better leaves the previous
        # candidate in place -- so this is free to record, and it is the
        # signature of the one failure the differential cannot rule out by
        # itself: blaming the program that was already correct. A healthy
        # solve prints 0; a rising count with `mismatch` flat is the number to
        # watch.
        self.regressed = 0
        # Times the router gave up on the candidate and patched the reference
        # instead, having blamed the same failure twice with nothing else
        # accusing it. Bounds the damage of a wrong blame and makes it
        # countable.
        self.flipped = 0

    def note_probe(self, verdict: str) -> None:
        """The LAST verdict wins, for the same reason `exit` does.

        A too_slow round that is then repaired and passes must report
        `passed`: the line describes the answer that SHIPPED, and the earlier
        verdict belongs to a program that no longer exists.
        """
        self.probe = verdict

    def note_exit(self, reason: str) -> None:
        """The LAST reason wins.

        A solve runs a second pass only when the first ended without a
        deliverable answer -- `empty`, `cutoff`, `stalled` -- and the summary
        line describes the answer that SHIPPED. Keeping the first reason
        reported a solve whose second pass converged as `exit=empty`, which is
        the one thing an operator reading that line must not be told.
        """
        self.exit = reason

# There is no cap on what the CASES turn may spend, and there has never been a
# version of one that paid. It carried a private cap three times: two removed
# before this, and the half-budget ceiling removed here.
#
# The argument that kept bringing it back: `send` returns the moment the model
# finishes, so a slice is a
# ceiling and never a wait -- a cases turn that takes 90 seconds hands the
# program the other 190 whether a cap exists or not. The only case a cap
# changes is the one where the model has NOT finished, "and there it converts
# a slow answer into no answer, which is the one trade the payment policy says
# never to make". Measured against those caps on a live tab: `Thought for
# 1m 17s` before a single character appeared, against a 60 second cap.
#
# Every word of that holds. What it assumes is that being cut off ends the
# SOLVE, and that was true only because of what happened next: `_ask_for_cases`
# returned None for a conversation mid-answer, and `_attempt` abandoned the pass
# rather than send turn 2 into a tab that had not answered turn 1. So the cap
# did convert a slow answer into no answer -- not because it was a cap, but
# because nothing picked the solve up afterwards.
#
# Turn 1's answer is not the answer. It is a local grading bar, and the code
# has always known how to proceed without one ("the cases turn produced none
# usable; asking for the program without them"). What was missing is that a
# conversation still writing turn 1 cannot be reused -- and the answer to that
# is a different conversation, which this file already opens for other reasons.
# With that path in place a cap costs the CASES, never the answer, and the
# trade the payment policy forbids is not the one being made.
#
# Measured over a live run of thirteen solves, which is why this exists at all:
# the cases turn averaged 97.2s against the program turn's 66.7s and took 54%
# of the mean solve -- 1264 seconds of 2338 -- to produce no code at all. Twice
# it left less than the worst observed program turn behind it, and once it took
# 197.9s of a 290s budget, after which the program turn was cut off mid-write
# at 90.8s and a truncated Rust program went out. That is a zero, and it is the
# only zero in the run.
#
# So the cases turn is bounded by the solve deadline and by nothing else.
# The recovery path that made a ceiling safe -- drop the cases, open a fresh
# conversation, ask it for the program with what the ceiling withheld -- goes
# with it: there is no withheld half to recover, and a turn still writing when
# the whole budget is gone has nothing left to be reopened with.

# Nothing here passes `extend_to_s` any more, and that is the end of a long
# argument rather than an oversight. It existed so a read could spend a repair
# reserve on an answer that was still arriving; with every read given the whole
# remaining budget there is no reserve, and a hard bound equal to the slice is
# an extension that cannot extend. `send` still accepts the keyword -- see
# `browser_pool` -- for a caller that does hand out less than everything.
#
# The happy side effect: every read here is the two-argument `send` that has
# always been the `Conversation` contract, so a backend written outside this
# package needs nothing new to work.


class VerifyingSolver:
    """Wrap any conversational backend in a self-check-and-repair loop.

    Budget discipline matters more here than anywhere else in the miner: a
    correct answer delivered after the cutoff scores the same zero as a wrong
    one. Every attempt is bounded, the loop stops while there is still margin,
    and whatever candidate ranked best is returned rather than nothing.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        max_attempts: int = 0,
        reserve_s: float = DELIVERY_RESERVE_S,
        max_budget_s: float = 3600.0,
        cache_size: int = 256,
        second_opinion: bool = True,
        self_tests: bool = True,
    ):
        self._backend = backend
        # 0 means UNLIMITED, and that is the default. Correctness is the whole
        # of the payment here, so the only thing that should stop a repair loop
        # is the validator's deadline -- a count of three was a second, private
        # deadline layered under the real one, and it fired first.
        self._max_attempts = max(0, int(max_attempts))
        self._reserve = max(0.0, float(reserve_s))
        self._max_budget = max(5.0, float(max_budget_s))
        self._second_opinion = bool(second_opinion)
        self._self_tests = bool(self_tests)
        # Stage 3, the analysis turn. The only stage with a switch of its
        # own, because it is the only one whose output is optional: the free
        # heuristic scan runs either way, and turning this off leaves the
        # solve reading the statement with regexes alone. On by default --
        # with it off the later prompts see fewer traps, which is a worse
        # prompt rather than a broken one.
        self._llm_analysis = _env_on("SOLVER_LLM_ANALYSIS")
        # The size probe: one extra fenced block on the inputs turn, and one
        # local run of the finished program on a large valid input. No extra
        # model turn, and no reference -- see `_timed_out_at_scale`. It is
        # also one of the two signals that accuse the candidate without the
        # reference's help, which is what the router's flip rule listens to.
        self._size_probe = (
            _env_on("SOLVER_SIZE_PROBE")
        )
        self._grader = _Grader()
        self._cache: dict[str, tuple[str, str]] = {}
        self._cache_size = max(0, int(cache_size))
        self._counts = {
            "solved": 0, "verified": 0, "verified_on_local": 0, "cache_hits": 0,
            "empty": 0, "fallback": 0,
        }
        self._by_provider: dict[str, dict[str, int]] = {}
        # The no-examples explanation is worth saying, but only once a run.
        self._warned_ungradeable = False
        # ...and so is a deadline being cut short by our own configuration.
        self._warned_short_deadline = False

    def _next_pass_blocked_by(
        self, best: Candidate, attempt_no: int, remaining: float
    ) -> Optional[str]:
        """Why the pass after this one will not happen, or None if it will.

        Asked in TWO places, and it has to answer the same in both: at the top
        of the next iteration, which ACTS on it, and at the bottom of this one,
        which ANNOUNCES it. They were separate expressions and they drifted --
        the announcement asked only `attempt_no < passes` and knew nothing
        about the clock, so a solve that had spent its entire budget printed

            [verify] claude returned nothing; asking another model
            [verify] -0s left; not enough to ask anyone else, submitting empty

        one line apart. Nobody was asked. An operator reading that goes looking
        for a second provider's failure that never happened, while the real one
        -- a program turn that ran the budget out -- sits three lines above it
        wearing no emphasis at all.
        """
        empty_handed = not best.code.strip()
        if not empty_handed and attempt_no >= SECOND_OPINION_PASSES:
            # There is an answer in hand and it has already had its second
            # opinion. `MAX_PASSES` is for the empty case only: spending it
            # here would double or quadruple what every failing task costs a
            # real account's quota, to improve on something already worth
            # submitting.
            return "the answer in hand has already had its second opinion"
        if remaining < MIN_SLICE_S:
            return (
                "the deadline is gone, submitting empty"
                if empty_handed
                else "the deadline is gone"
            )
        return None

    # -- the Solver interface custom_miner.py expects ---------------------- #
    async def solve_task(self, task, timeout_s: float) -> Answer:
        started = time.monotonic()
        # `timeout_s` is already the cutoff the miner's own `handle_request`
        # will 504 at: `deadline_s + RESPONSE_GRACE_S`, capped by
        # `glm_request_timeout_s`. The margin is what keeps this side of it:
        # `send` runs its copy, stream, salvage and post-mortem phases AFTER
        # its slice expires (5 + 3 + 1 + 2 = 11s), and the
        # answer still has to be graded, archived, signed and put on the wire
        # before the validator stops listening at `deadline_s + 10`.
        #
        # `_max_budget` is a ceiling, not a target, and it deliberately does not
        # bind at the deadlines this subnet advertises. It used to: 240 against a
        # 300s deadline capped the first attempt at 191s, and a model that needed
        # longer had its answer thrown away by this miner rather than by the
        # validator -- which would have paid 96% for the same answer arriving at
        # six minutes. Correctness is worth 100%; speed is worth at most 5%.
        # ONE reserve, and it is what delivery needs. Everything else the
        # solve does -- reading, grading, repairing -- happens inside `budget`,
        # and `budget` runs right up to the point the answer stops being
        # deliverable. Waiting is close to free and giving up is a certain zero:
        # the validator reads until `deadline_s + 10`, and its payment rule has
        # no deadline term at all -- correctness is a hard gate and speed is a
        # relative multiplier floored at 0.95, so the same answer a minute later
        # is still worth 95%.
        budget = min(float(timeout_s), self._max_budget) - self._reserve
        if budget <= 5.0:
            # Too short for the whole margin, so keep the SHAPE of the promise
            # instead of its size: half the request, which leaves the other half
            # for the post-read tail (itself scaled, see `tail_budget`) and for
            # putting the answer on the wire.
            #
            # The floor used to be 5 seconds, and a floor is exactly the wrong
            # instrument here: at a 5-second deadline it budgeted the entire
            # request and `handle_request` cancelled the solve mid-flight. A
            # 504 is indistinguishable from a dead miner, and `deadline_s` is
            # only `Field(gt=0.0, ...)` -- nothing in the protocol promises the
            # comfortable numbers this subnet happens to send today.
            budget = max(1.0, float(timeout_s) * 0.5)

        advertised = float(getattr(task, "deadline_s", 0.0) or 0.0)
        if advertised - float(timeout_s) > 1.0 and not self._warned_short_deadline:
            # `timeout_s` is the request's cutoff, capped by
            # `glm_request_timeout_s`. When it comes back SHORTER than what
            # the validator advertised, the miner
            # is giving up early on its own configuration -- and nothing else
            # says so. The reference miner's docs put GLM_REQUEST_TIMEOUT_S at
            # 280 against a 300s deadline, and a .env copied from there costs
            # every solve 20 seconds it was offered. Once per run.
            self._warned_short_deadline = True
            print(
                f"[verify] the validator offered {advertised:.0f}s but this miner "
                f"caps the solve at {float(timeout_s):.0f}s, so every answer gets "
                f"{advertised - float(timeout_s):.0f}s less than it could. Raise "
                f"GLM_REQUEST_TIMEOUT_S to at least {advertised:.0f} to use it all "
                f"— a correct answer arriving late still earns 95%+, an unfinished "
                f"one earns nothing. Once per run."
            )

        key = _cache_key(task)
        if key in self._cache:
            self._counts["cache_hits"] += 1
            code, raw = self._cache[key]
            # Said and archived, like the disk hit below: a submission served
            # without being re-run is the one whose provenance a wrong result
            # asks about, and it used to leave no line and no `solve` record.
            print(
                f"[verify] {task.language} entrypoint={task.entrypoint} "
                f"cache=hit(memory) {key[:12]} (it reproduced the public "
                f"examples earlier in this process)"
                + (f" id={_ident(task)}" if _ident(task) else "")
            )
            return Answer(
                code=code, raw_response=raw, verified=True,
                diagnostics={"cache": "memory-hit", "cache_key": key,
                             "exit": "cache"},
            )
        # ...and the same question asked of the disk, which outlives the
        # process. Before any conversation is opened, because the whole value
        # of a hit is that it costs a second and no quota.
        #
        # Re-checked rather than trusted: the file was written by a previous
        # run of this code, but the file system is not a memory and an
        # operator may have edited, truncated or copied it. `python_defect` is
        # the same structural check every fresh answer passes and it costs
        # microseconds, so a corrupted entry reads as a miss.
        # Not when the request carries public examples. `worth_keeping`
        # requires `self_verified`, which requires that no public example ran
        # -- so every stored entry is an answer graded by the model's own
        # cases alone. Examples are cheap, decisive, and shipped with the
        # request; serving a cached answer past them would skip the better
        # evidence for the worse. Live traffic ships none, so this costs
        # nothing there and keeps a replay honest.
        # Off the loop: a read of up to 2 MiB and a JSON parse, on the loop
        # every other in-flight solve's deadline timer runs on.
        stored = (
            None if task.public_examples
            else await asyncio.to_thread(solution_cache.load, key)
        )
        if stored is not None and (
            stored.get("probe") != "passed" or not stored.get("bar")
        ):
            # Not what this code writes. The line below claims the answer
            # passed its own cases and was timed at scale, so a file that does
            # not carry both -- hand-edited, or from another build -- is a
            # miss rather than a claim.
            stored = None
        if stored is not None:
            code = str(stored.get("code") or "")
            defect = (
                rust_defect(code)
                if task.language == "rust"
                else python_defect(code, task.entrypoint)
            )
            if defect is None:
                self._counts["cache_hits"] += 1
                print(
                    f"[verify] {task.language} entrypoint={task.entrypoint} "
                    f"cache=hit {key[:12]} "
                    f"(kept {_ago(stored.get('saved_at'))}; it passed its own "
                    f"cases and was timed at scale)"
                    + (f" id={_ident(task)}" if _ident(task) else "")
                )
                return Answer(
                    code=code,
                    raw_response=str(stored.get("raw_response") or ""),
                    self_verified=True,
                    # The archive gets what the cache knows, or the one
                    # submission most in need of an explanation would have
                    # none: an answer served without being re-run is exactly
                    # the one whose "what was this checked against" a wrong
                    # hidden-suite result asks about. It is all in `stored`.
                    diagnostics={
                        "cache": "hit",
                        "cache_key": key,
                        # The same shape a solved answer archives, so a reader
                        # joining on `provider` or `exit` finds the hits too.
                        "provider": (stored.get("providers") or [None])[0],
                        "exit": "cache",
                        "saved_at": stored.get("saved_at"),
                        "bar": stored.get("bar") or [],
                        "probe": stored.get("probe") or "",
                        "providers": stored.get("providers") or [],
                    },
                )
            print(f"[verify] a cached answer for {key[:12]} no longer reads as "
                  f"a program ({defect}); solving it again")

        best = Candidate(code="", raw="")
        # One pass per model. The second only happens if the first could not
        # reproduce the public examples even after its repair rounds — at which
        # point the odds it passes the HIDDEN suite are poor, and the whole
        # payment rides on that. Asking the other model is a fresh chance at the
        # full amount, and with a fleet there is usually an idle tab to ask on.
        asked: list[str] = []
        # WHICH model produced the answer that wins, not merely which were
        # asked. Attribution after the fact was otherwise guesswork: of 43
        # archived submissions only three could be traced to a provider at all,
        # and only because the damage itself carried a fingerprint -- two held
        # ChatGPT's nudge, one quoted `/home/claude/sol`. The other forty were
        # unattributable, which made "is one of these tabs doing worse than the
        # others" an unanswerable question.
        won_with: Optional[str] = None
        # (exit, bar_provider, rounds, corrected) of the pass that produced
        # `best`, or None when no pass ever beat the empty candidate it starts
        # as -- every pass failed, and the LAST one's reason is then the only
        # account of the solve there is. See where it is captured and resolved.
        shipped: Optional[_Shipped] = None
        plan = _Plan()
        # A ceiling, not a plan. Every ordinary path breaks out after one or
        # two: the loop only keeps going while it is holding NOTHING, which is
        # the one state where another ask cannot make things worse.
        passes = MAX_PASSES if self._second_opinion else 1
        attempt_no = 0
        while attempt_no < passes:
            remaining = budget - (time.monotonic() - started)
            # The first pass always runs, however little is left: bailing here
            # would return nothing having asked nobody.
            if attempt_no:
                blocked = self._next_pass_blocked_by(best, attempt_no, remaining)
                if blocked is not None:
                    # `max(0, ...)`: the budget can be a hair past spent by the
                    # time this reads it, and "-0s left" reads as a bug in the
                    # arithmetic rather than as a solve that used everything it
                    # had.
                    print(f"[verify] {max(0.0, remaining):.0f}s left; {blocked}")
                    break
            attempt_no += 1
            candidate, provider = await self._attempt(
                task, remaining, avoid=asked[-1] if asked else None, plan=plan,
                pass_no=attempt_no,
            )
            if provider:
                asked.append(provider)
            if candidate is not None and candidate.score > best.score:
                best = candidate
                won_with = provider
                # What the SHIPPING pass did, captured as it wins. A second
                # pass runs with an answer already in hand whenever public
                # examples ran and something failed, and a pass that then
                # scores lower does not replace `best` -- but it did overwrite
                # `plan`, so the line described a pass whose program was
                # thrown away. `provider=` has always come from `won_with` for
                # exactly this reason; these now do too.
                shipped = _Shipped.of(plan)
            if best.verified and not best.failures:
                break
            # Nothing RAN, whether or not anything was shipped to run. The
            # distinction used to be `not task.public_examples`, and that missed
            # the commoner case by far: examples shipped, and the executor could
            # not run them. Measured on a live miner with no Docker daemon, all
            # three Rust challenges asked a SECOND model -- a full extra solve
            # each, 55 to 108 seconds and a second conversation off the account
            # quota -- and then submitted the first model's answer anyway,
            # because two ungradeable candidates tie at `score` and `>` loses a
            # tie. Twice the time and twice the quota for no information at all.
            if best.total == 0:
                if best.from_self_tests or not best.code.strip():
                    # Two ways this warning would be a lie, and the second was
                    # printed against a solve whose cases turn had worked
                    # perfectly.
                    #
                    # `from_self_tests`: the model shipped cases and they ran,
                    # so there was something to grade after all.
                    #
                    # NO CODE: the differential is gated on a non-empty
                    # candidate, so
                    # an empty candidate reports `from_self_tests=False`
                    # whatever turn 1 produced -- and the warning then blames
                    # the cases turn for the PROGRAM turn's failure. Measured:
                    # a Rust solve whose cases turn returned usable cases in
                    # silence, whose program turn then spent the whole 285s
                    # budget still writing, and which reported "the model sent
                    # no usable cases of its own either". Nothing can be graded
                    # because there is no ANSWER, and the lines that say the
                    # answer is missing already say so, about the right turn.
                    pass
                elif best.self_cases:
                    # A THIRD way it would be a lie, and the one the deadline
                    # produces: the cases turn worked, the cases are right
                    # here, and there was no budget left to run them. Saying
                    # "the model sent no usable cases" sends the reader to fix
                    # a prompt that is working. `_grade` has already named this
                    # one on the line above, so there is nothing to add.
                    pass
                elif not self._warned_ungradeable:
                    self._warned_ungradeable = True
                    why = (
                        "no public examples shipped with this task"
                        if not task.public_examples
                        else "the public examples could not be run here"
                    )
                    print(
                        f"[verify] {why}, and the model sent no usable cases of "
                        f"its own either, so nothing can be graded locally: no "
                        f"repair rounds, no second opinion once an answer is in "
                        f"hand, and verified=False however good it is. Once per run."
                    )
                # A second opinion is only ever worth buying when this one came
                # back EMPTY. Then it is worth a lot: an empty answer scores
                # zero, and the other model is the only remaining chance at the
                # whole payment.
                if best.code.strip():
                    break
            if attempt_no < passes and self._next_pass_blocked_by(
                best, attempt_no, budget - (time.monotonic() - started)
            ) is None:
                print(
                    f"[verify] {provider or 'first'} "
                    + (
                        "returned nothing"
                        if not best.code.strip()
                        else "cleared the examples but not its own cases"
                        if best.verified
                        else "did not verify"
                    )
                    + "; asking another model"
                )
        if asked:
            # `won_with`, not `asked[-1]`. They usually coincide -- a verified
            # answer ends the loop, so the winner is normally the last one asked
            # -- but "usually" is not what a tally is for. A pass whose backend
            # never reported a provider is absent from `asked` while still able
            # to produce the winning answer, and the credit then lands on the
            # PREVIOUS model. This is the number an operator reads to decide
            # which account has started failing; it should say who actually won.
            self._note(won_with if best.verified else None, asked)

        if best.verified:
            self._counts["verified"] += 1
        elif best.self_verified:
            # Counted apart from `verified`, never inside it, and named as
            # the log line names it. On live traffic this is the only counter
            # of the two that can ever move, so a `/solver-status` showing
            # verified=0 over a whole run is the ordinary reading rather than
            # the alarming one -- and this is the number that says whether the
            # answers were any good.
            self._counts["verified_on_local"] += 1
        if best.code.strip():
            self._counts["solved"] += 1
            # `not best.failures` as well as `verified`: with both suites run,
            # an answer can clear the validator's examples and still disagree
            # with the model's own cases. Caching that re-serves one wrong
            # answer for every later task with the same statement, which is the
            # exact harm the cache gate exists to prevent.
            if best.verified and not best.failures and self._cache_size:
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (best.code, best.raw)
        else:
            self._counts["empty"] += 1
        # Nothing ever won, so there is no shipping pass to describe: fall
        # back to the last one, which is what `exit=failed` and `exit=cases`
        # exist to report. Resolved here rather than at the capture so a solve
        # that DID ship is never described by a later pass that lost.
        if shipped is None:
            shipped = _Shipped.of(plan)
        # The disk keeps a different set from the in-memory cache above, on a
        # different gate. That one wants `verified` -- agreement with the
        # PUBLIC examples -- which live traffic never ships, so it has never
        # once fired on a real solve. This asks what a live answer can
        # actually establish about itself: it passed every case its readers
        # wrote, nothing was left contested, and it was timed at the
        # statement's own scale. See `solution_cache.worth_keeping`.
        # ...and not while grading is DEGRADED. Under the subprocess fallback
        # a `passed` probe says the program finished in time at 1 GiB, which
        # is not what the validator's 256 MiB will say; kept, it would be
        # served on every duplicate with no re-run ever.
        if best.code.strip() and solution_cache.worth_keeping(
            self_verified=best.self_verified,
            failures=bool(best.failures),
            contested=0,
            probe=shipped.probe,
        ) and not self._grader.degraded(task.language):
            record = solution_cache.record(
                code=best.code, raw=best.raw, task=task, bar=best.self_bar,
                probe=shipped.probe,
                # Every model that touched what a hit is served on, in the
                # order they touched it, once each: the inputs, the reference,
                # the candidate's author, and the models the repair went
                # through.
                providers=list(dict.fromkeys(
                    p for p in (
                        shipped.tests_provider, shipped.oracle_provider,
                        won_with, *shipped.repair,
                    )
                    if p
                )),
            )
            await asyncio.to_thread(solution_cache.save, key, record)
        elapsed = time.monotonic() - started
        print(
            f"[verify] {task.language} entrypoint={task.entrypoint} "
            f"provider={won_with or 'none'} "
            + (f"inputs={shipped.tests_provider} "
               if shipped.tests_provider else "")
            + (f"reference={shipped.oracle_provider} "
               if shipped.oracle_provider else "")
            + f"examples={best.passed}/{best.total} "
            + (
                f"self={best.self_passed}/{best.self_total} "
                + (
                    f"({best.self_observed} ran, "
                    f"{best.self_total - best.self_observed} never did) "
                    if best.self_observed and best.self_observed < best.self_total
                    else ""
                )
                if best.self_total else ""
            )
            + f"verified={best.verified} "
            # `verified=False` is the only thing a live solve could ever print,
            # because no request ships public examples -- so on its own it said
            # the same thing about an answer that agreed with the reference
            # everywhere and one that was never run at all.
            + (
                f"(verified on local: agreed with an independently written "
                f"reference on all {best.self_total} inputs; no public "
                f"examples exist to confirm it) "
                if best.self_verified
                else ""
            )
            # What the comparison actually established. `ran=` is the field
            # that stops this line saying the same thing about a checked
            # answer and an unchecked one -- the failure that motivated the
            # whole design.
            + f"ran={shipped.ran} agreed={shipped.agreed} "
            + f"mismatch={shipped.mismatch} "
            + (f"oracle_crash={shipped.oracle_crash} "
               if shipped.oracle_crash else "")
            + f"rounds={shipped.rounds} corrected={shipped.corrected}/{best.self_total}"
            # WHICH artifact each repair round patched, and whether the router
            # ever had to take a round back off the candidate. `regressed=` is
            # the instrument for a wrong blame; `flipped=` says the flip rule
            # fired.
            + (
                " repair=" + ">".join(
                    f"{who}:{what}" for who, what
                    in zip(shipped.repair, shipped.patched)
                )
                if shipped.repair else ""
            )
            + (f" regressed={shipped.regressed}" if shipped.regressed else "")
            + (f" flipped={shipped.flipped}" if shipped.flipped else "")
            + (f" probe={shipped.probe}" if shipped.probe else "")
            + (f" exit={shipped.exit}" if shipped.exit else "")
            + " "
            + f"{elapsed:.1f}s/{budget:.0f}s"
            + (f" id={_ident(task)}" if _ident(task) else "")
        )
        return Answer(
            code=best.code, raw_response=best.raw,
            verified=best.verified, passed=best.passed, total=best.total,
            self_verified=best.self_verified,
            self_passed=best.self_passed, self_total=best.self_total,
            # The join key between a shipped answer and its hidden-suite
            # outcome. The names that were here before and still mean the same
            # thing keep their spelling, so anything reading an archive from
            # before this change still finds them.
            diagnostics={
                "provider": won_with,
                "tests_provider": shipped.tests_provider,
                "oracle_provider": shipped.oracle_provider,
                # What the answer was checked against. Kept for the solution
                # cache and for the archive: an answer served again without
                # being re-run is a question about its evidence.
                "bar": best.self_bar,
                "self_passed": best.self_passed,
                "self_total": best.self_total,
                "self_observed": best.self_observed,
                "self_verified": best.self_verified,
                # WHICH inputs it disagreed on, not just how many. A wrong
                # answer in the archive is a question about one case, and a
                # count cannot answer it.
                "failed_cases": best.failed_cases,
                "failures": best.failures,
                # What the differential established, and what the repair loop
                # did about it.
                "ran": shipped.ran,
                "agreed": shipped.agreed,
                "mismatch": shipped.mismatch,
                "oracle_crash": shipped.oracle_crash,
                "rounds": shipped.rounds,
                "corrected": shipped.corrected,
                "repair": list(shipped.repair),
                "patched": list(shipped.patched),
                "regressed": shipped.regressed,
                "flipped": shipped.flipped,
                "probe": shipped.probe,
                "exit": shipped.exit,
                "elapsed_s": round(elapsed, 1),
                "budget_s": round(budget, 1),
            },
        )

    # -- stage 3: what the statement hides -------------------------------- #
    async def _analyse(
        self, task, heuristic, budget: float, started: float,
        avoid: Optional[str], phases: "_Phases",
    ):
        """Ask a model to add to the free trap scan. Never costs the solve.

        This is the ONE stage allowed to produce nothing. Every failure --
        an unreadable conversation, a reply that is not JSON, a backend that
        raises -- ends with the heuristic analysis still in hand, which is the
        same place the solve would have been had the stage never run. So it is
        wrapped whole rather than guarded condition by condition.
        """
        conversation = None
        try:
            conversation = await self._open_within(
                budget, started, avoid, phase="analysis"
            )
            reply = await self._send_within(
                conversation,
                build_analysis_prompt(task, heuristic),
                max(1.0, budget - (time.monotonic() - started)),
            )
            merged = analysis_from_json(extract_analysis(reply), heuristic)
            phases.mark("1 analysis")
            added = len(merged.traps) - len(heuristic.traps)
            print(f"[verify] the analysis turn added {added} trap(s) to the "
                  f"{len(heuristic.traps)} the scan already had")
            return merged
        except Exception as exc:  # noqa: BLE001 - the heuristic pass stands
            print(f"[verify] the analysis turn produced nothing usable "
                  f"({type(exc).__name__}: {exc}); going on with the scan "
                  f"alone")
            return heuristic
        finally:
            if conversation is not None:
                with contextlib.suppress(Exception):
                    await conversation.close()

    # -- stage 4: the inputs, with no answers ----------------------------- #
    async def _write_inputs(
        self, task, analysis, budget: float, started: float,
        avoid: Optional[str], probe: Optional[list], plan: Optional["_Plan"],
        phases: "_Phases",
    ) -> list[dict[str, Any]]:
        """Inputs only. Runs beside the oracle and the candidate.

        A failure here costs the CHECK and not the answer: with no inputs
        there is nothing to run either program on, the differential reports
        that it established nothing, and the candidate ships ungraded rather
        than not at all.
        """
        conversation = None
        turn = time.monotonic()
        try:
            conversation = await self._open_within(
                budget, started, avoid, phase="tests"
            )
            if plan is not None:
                plan.tests_provider = getattr(conversation, "provider", None)
            reply = await self._send_within(
                conversation,
                build_inputs_prompt(task, analysis, want_probe=probe is not None),
                max(1.0, budget - (time.monotonic() - started)),
            )
            if probe is not None:
                found = extract_generator(reply)
                if found:
                    probe.append(found)
            cases = extract_inputs(reply, task.language)
            phases.mark("1 inputs", beside=True, ended=turn)
            if not cases:
                print("[verify] the inputs turn produced none usable; the "
                      "candidate will ship unchecked unless a repair finds one")
            return cases
        except Exception as exc:  # noqa: BLE001 - never lose the answer to the bar
            print(f"[verify] the inputs turn failed ({type(exc).__name__}: {exc}); "
                  f"there is nothing to run either program on")
            return []
        finally:
            if conversation is not None:
                with contextlib.suppress(Exception):
                    await conversation.close()

    # -- stage 5: the reference, written to be obviously right ------------ #
    async def _write_oracle(
        self, task, analysis, budget: float, started: float,
        avoid: Optional[str], plan: Optional["_Plan"], phases: "_Phases",
    ) -> str:
        """The slow literal program the candidate is compared against.

        Opened on its own phase and its own conversation, and that separation
        is the design rather than tidiness: the reference must not be able to
        see the candidate, or it stops being an independent reading of the
        statement and becomes a second draft of the same one.
        """
        conversation = None
        turn = time.monotonic()
        try:
            conversation = await self._open_within(
                budget, started, avoid, phase="oracle"
            )
            if plan is not None:
                plan.oracle_provider = getattr(conversation, "provider", None)
            reply = await self._send_within(
                conversation,
                build_oracle_prompt(task, analysis),
                max(1.0, budget - (time.monotonic() - started)),
            )
            code = extract_code(reply, task.entrypoint, task.language)
            phases.mark("1 reference", beside=True, ended=turn)
            if not code.strip():
                print("[verify] the reference turn produced no program; "
                      "nothing can be compared against the candidate")
            return code
        except Exception as exc:  # noqa: BLE001 - never lose the answer to the bar
            print(f"[verify] the reference turn failed ({type(exc).__name__}: {exc}); "
                  f"the candidate cannot be checked against one")
            return ""
        finally:
            if conversation is not None:
                with contextlib.suppress(Exception):
                    await conversation.close()

    async def _attempt(
        self,
        task,
        remaining: float,
        avoid: Optional[str],
        plan: Optional["_Plan"] = None,
        pass_no: int = 1,
    ) -> tuple[Optional[Candidate], Optional[str]]:
        """One pass of the nine stages. Returns the best candidate and who wrote it.

        The shape, and why it is this shape:

          * stage 3 runs alone, because every later prompt is handed its result;
          * stages 4, 5 and 6 run TOGETHER, because none consumes another's
            output -- the critical path is one analysis plus the slowest of
            three, not the sum of four;
          * stage 7 runs the reference and the candidate on the same inputs and
            is the only stage with no model call in it;
          * stage 8 repairs whichever artifact the router names, on a phase
            whose model is deliberately not the one that wrote the candidate.

        One clock governs all of it. A stage runs if the deadline has not been
        reached; nothing here asks whether a stage is "worth starting", because
        every constant that would answer that is a guess about how long a model
        will take.
        """
        started = time.monotonic()
        budget = max(1.0, remaining)
        best: Optional[Candidate] = None
        best_score: tuple = ()
        best_provider: Optional[str] = None
        conversation = None
        repair_conv = None
        provider: Optional[str] = None

        def left() -> float:
            return budget - (time.monotonic() - started)

        def note_exit(reason: str) -> None:
            if plan is not None:
                plan.note_exit(reason)

        try:
            phases = _Phases(budget, started, pass_no, ident=_ident(task))
            if plan is not None:
                # Per pass, like `exit`: the summary line describes the pass
                # that SHIPPED, and a second pass runs only after the first
                # delivered nothing.
                plan.rounds = 0
                plan.corrected = 0
                plan.probe = ""
                plan.repair = []
                plan.patched = []
                plan.regressed = 0
                plan.flipped = 0
                plan.ran = plan.agreed = 0
                plan.mismatch = plan.oracle_crash = 0
                plan.tests_provider = None
                plan.oracle_provider = None

            # -- 2. the free trap scan -------------------------------------
            analysis = heuristic_analyze(task)

            # -- 3. the analysis turn --------------------------------------
            if self._llm_analysis and left() >= MIN_SLICE_S:
                analysis = await self._analyse(
                    task, analysis, budget, started, avoid, phases
                )

            # -- 4, 5, 6 side by side --------------------------------------
            probe: Optional[list] = [] if self._size_probe else None
            probed: dict[str, _Probe] = {}
            # `SOLVER_SELF_TESTS=0` turns the local check off, and with it both
            # turns that exist only to produce one. Live traffic ships no
            # public examples, so with this off nothing is graded, nothing is
            # compared and the repair loop never fires -- the answer is
            # whatever the candidate turn said. It is off the default path and
            # it is a shipped configuration, for an operator who wants one turn
            # per solve and will take the score that comes with it.
            #
            # Both turns go together. Inputs with no reference cannot be
            # answered and a reference with no inputs has nothing to run on, so
            # keeping either one alone would buy a conversation and a model's
            # time for a comparison that cannot happen.
            inputs_task = (
                asyncio.create_task(self._write_inputs(
                    task, analysis, budget, started, avoid, probe, plan, phases))
                if self._self_tests else None
            )
            oracle_task = (
                asyncio.create_task(self._write_oracle(
                    task, analysis, budget, started, avoid, plan, phases))
                if self._self_tests else None
            )

            conversation = await self._open_within(
                budget, started, avoid, phase="candidate"
            )
            provider = best_provider = getattr(conversation, "provider", None)
            phases.mark(f"open {provider or 'tab'}")
            candidate_reply = await self._send_within(
                conversation, build_candidate_prompt(task, analysis),
                max(1.0, left()),
            )
            phases.mark("2 candidate")

            inputs = await inputs_task if inputs_task is not None else []
            oracle = await oracle_task if oracle_task is not None else ""
            if not self._self_tests:
                print("[verify] the local check is off "
                      "(SOLVER_SELF_TESTS=0); the candidate ships unchecked")

            # -- 7 and 8: compare, then repair whichever is wrong ----------
            differential = Differential(self._grader, VERIFY_TIMEOUT_S)
            router = Router()
            # Every (candidate, reference, failing cases) this pass has seen.
            # A round that leaves all three unchanged moved nothing, and the
            # next round would send the same prompt and read the same reply --
            # so it is a stall, not progress, and the budget is better spent
            # shipping what is in hand. Measured under the previous design,
            # which compared only against the PREVIOUS round rather than all
            # of them: 59 sends in one solve, cycling between two answers.
            seen: set[tuple] = set()
            # Which artifact the previous round patched. A stall only ends the
            # pass when the router names that same one again -- a stalled
            # candidate repair is exactly when trying the reference is worth a
            # round, and giving up there would leave a wrong reference
            # unexamined.
            last_patched = ""
            last_code = ""
            # How long the round that produced the reply now in hand took. Only
            # read when that round changed nothing -- see `STALE_ROUND_S`.
            round_s = float("inf")
            # Which conversation produced the reply now being graded. Stage 6
            # writes the first candidate; from round 2 a candidate repair is
            # written by the repair conversation instead, and "the model was
            # still writing" is a fact about whichever tab actually wrote it.
            # Reading it off the stage-6 conversation forever would let one
            # unfinished first draft end every later round as a cutoff.
            writer = conversation
            attempt = 0
            while True:
                attempt += 1
                if attempt > 1 and left() < MIN_SLICE_S:
                    print("[verify] the deadline is gone; submitting the last "
                          "version" + ("" if best is not None
                                       and best.verified else " unverified"))
                    note_exit("budget")
                    break
                if plan is not None:
                    plan.rounds += 1

                still_writing = getattr(writer, "still_writing", False)
                # `cases=None`: the structural check, the Rust compile and the
                # validator's own examples happen here; the candidate is RUN by
                # the differential below, once, rather than twice.
                candidate = await self._graded(
                    candidate_reply, task, left(), previous=last_code,
                )
                # The validator's OWN examples are ground truth, and they
                # take precedence exactly as they did before: when they fail
                # the program is wrong, there is nothing to weigh, and running
                # the reference against it would buy an executor run per input
                # on a question already answered. Only once they are green --
                # which on live traffic is always, since no task ships any --
                # does a disagreement with the reference become the open
                # question. Without this the repair loop consulted only the
                # differential and a program failing ground truth was never
                # repaired at all.
                examples_failed = bool(
                    candidate.total and candidate.passed < candidate.total
                )
                # A structural defect -- a program that will not parse, or Rust
                # that will not compile -- accuses the candidate on its own
                # evidence, with no reference involved. It has to drive a
                # repair by itself: live traffic ships no public examples, so
                # `compile_defect` is the ONLY check a Rust answer gets, and a
                # loop that consulted only the differential shipped a program
                # that does not build without ever asking for a fix.
                defective = bool(candidate.defect)
                if examples_failed or defective:
                    report = DifferentialReport(
                        note=("the validator's own examples failed"
                              if examples_failed else candidate.defect)
                    )
                else:
                    report = await asyncio.to_thread(
                        differential.compare, candidate.code, oracle,
                        task.language, task.entrypoint, inputs, left(),
                    )
                    _record_differential(candidate, report, inputs)
                if plan is not None:
                    plan.ran, plan.agreed = report.ran, report.agreed
                    plan.mismatch = report.mismatch
                    plan.oracle_crash = report.oracle_crash
                print(f"[verify] the differential: {report.summary()}")

                signature = (
                    candidate.code, oracle,
                    tuple(case.name for case in report.cases),
                )
                stalled = signature in seen
                seen.add(signature)

                # The differential first, then the candidate's OWN quality as
                # the tie-break. Ranking on the comparison alone made every
                # report that established nothing score identically -- so a
                # program repaired out of a structural defect could not
                # outrank the defective one it replaced, and the broken
                # version shipped. `Candidate.score`'s last term is "has no
                # defect", which is exactly the missing comparison.
                score = (report.score(), candidate.score)
                if _supersedes(candidate, best, still_writing) and (
                    best is None or score > best_score
                ):
                    best, best_score, best_provider = candidate, score, provider
                elif attempt > 1 and plan is not None:
                    # Repair is MONOTONE: a round that did not raise the score
                    # leaves the previous version in place. Counted, because a
                    # rising count with `mismatch` flat is the one signature of
                    # the routing heuristic blaming the program that was right.
                    plan.regressed += 1

                if report.ok:
                    slow, probe_state = await self._probe_now(
                        candidate, probe or [], probed, task, budget, started
                    )
                    if plan is not None:
                        plan.note_probe(probe_state)
                    if slow is None:
                        note_exit("verified")
                        break
                    # The program agrees with the reference and is too slow at
                    # the statement's own scale. That is the candidate's fault
                    # on evidence the reference had no part in.
                    report.note = slow

                if still_writing:
                    # The model had not finished when the read stopped, so what
                    # is in hand is a FRAGMENT of an answer rather than a wrong
                    # one, and a repair report about a half-written program
                    # describes a bug its author was still in the middle of not
                    # writing. Measured against the old browser backend:
                    #
                    #   captured=''   -> "your reply did not reach me as code",
                    #                    sent to a model still writing it
                    #   captured='def g(n):\n    total = 0\n    while n > 0:'
                    #                 -> "the code is not valid Python", about a
                    #                    program the model had not finished
                    #
                    # `_send_within` already reads past its own slice rather
                    # than stop early, so reaching here means the whole budget
                    # is gone and there is no round to spend anyway. Stop, and
                    # say which turn ran out -- the summary otherwise blames
                    # the inputs turn for the candidate turn's failure.
                    print(
                        f"[verify] {provider or 'this model'} was still writing "
                        f"when the budget ran out; "
                        + ("submitting the part that arrived"
                           if candidate.code.strip()
                           else "nothing arrived to submit")
                        + " rather than interrupting it with a repair prompt"
                    )
                    note_exit("cutoff")
                    break

                blamed_elsewhere = defective or examples_failed or (
                    plan is not None and plan.probe in ("too_slow", "oom")
                )
                if examples_failed or defective:
                    # Ground truth accuses the candidate and nothing else is in
                    # dispute, so the router is not consulted at all.
                    blame = "candidate"
                else:
                    blame = router.choose(
                        report, candidate_blamed_independently=blamed_elsewhere,
                        stalled=stalled,
                    )
                if plan is not None:
                    plan.flipped = router.flipped
                if not blame:
                    note_exit("converged")
                    break
                if stalled and round_s < STALE_ROUND_S:
                    # Nothing moved, and the round that moved nothing was free.
                    # Whatever wrote that reply was not a model. Flipping to
                    # the reference would ask the same tab the same way, so
                    # this stops the pass whoever the router blames.
                    print(
                        f"[verify] {provider or 'this model'} returned the same "
                        f"program in {round_s:.1f}s without being asked again — "
                        f"the tab is replaying an old reply rather than "
                        f"answering; submitting the last version"
                    )
                    note_exit("stalled")
                    break
                if stalled and blame == last_patched:
                    # The round that just ran changed neither program, and the
                    # router still wants the artifact that produced nothing.
                    # It has nowhere else to go, so the next round would send
                    # the same prompt and read the same reply.
                    print("[verify] the last round changed neither program "
                          "and there is nothing else to try; submitting what "
                          "is in hand")
                    note_exit("stalled")
                    break
                if left() < MIN_SLICE_S:
                    note_exit("budget")
                    break

                if repair_conv is None:
                    repair_conv = await self._open_within(
                        budget, started, avoid, phase="repair"
                    )
                repair_provider = getattr(repair_conv, "provider", None)
                if plan is not None:
                    plan.repair.append(_model_of(repair_provider))
                    plan.patched.append(
                        "cand" if blame == "candidate" else "oracle"
                    )
                last_code = candidate.code
                last_patched = blame
                round_started = time.monotonic()
                reply = await self._send_within(
                    repair_conv,
                    build_differential_repair_prompt(
                        task, analysis,
                        candidate.code if blame == "candidate" else oracle,
                        ("\n".join(candidate.failures) if examples_failed
                         else report.prompt_text()),  # the defect rides in `defect=`
                        kind=blame,
                        defect=candidate.defect if blame == "candidate" else None,
                        # What the prompt may claim was RUN. A defect is found
                        # before execution, so a repair that says "I compared
                        # it against a reference" is describing a run that did
                        # not happen -- and a model told its logic disagreed
                        # rewrites logic that was never the problem.
                        found_by=("examples" if examples_failed
                                  else "unrun" if defective
                                  else "differential"),
                    ),
                    max(1.0, left()),
                )
                round_s = time.monotonic() - round_started
                phases.mark(f"{attempt + 2} repair {blame}")
                if not reply.strip():
                    note_exit("empty")
                    break
                if blame == "oracle":
                    # The candidate is untouched and is re-graded next round
                    # against a reference that may now agree with it.
                    patched = extract_code(reply, task.entrypoint, task.language)
                    if not patched.strip():
                        note_exit("empty")
                        break
                    oracle = patched
                else:
                    candidate_reply = reply
                    writer = repair_conv
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never lose an answer in hand
            print(f"[verify] the solve failed: {type(exc).__name__}: {exc}")
            note_exit("failed")
        finally:
            for conv in (conversation, repair_conv):
                if conv is not None:
                    with contextlib.suppress(Exception):
                        await conv.close()
        return best, best_provider



    @staticmethod
    def _takes_extension(conversation) -> bool:
        """Whether this backend's `send` accepts `extend_to_s`.

        Asked of the signature rather than found out by calling: the fallback
        for a backend that does not take the keyword would be to call `send`
        again, and `send` puts a PROMPT on a conversation. A TypeError raised
        from inside a send that already went out would be answered by sending
        it a second time.
        """
        try:
            return "extend_to_s" in inspect.signature(conversation.send).parameters
        except (TypeError, ValueError):  # a C callable, or no signature at all
            return False

    async def _send_within(
        self, conversation, prompt: str, timeout_s: float,
        extend_to_s: Optional[float] = None,
    ) -> str:
        """One turn, with the hard bound past the slice where one is offered.

        `Conversation.send` is two arguments in the protocol and `extend_to_s`
        is optional, so a backend written outside this package need not have
        grown one -- and a keyword it does not take would be a TypeError
        inside the one call the whole solve depends on.
        """
        if extend_to_s is not None and self._takes_extension(conversation):
            return await conversation.send(
                prompt, timeout_s, extend_to_s=float(extend_to_s)
            )
        return await conversation.send(prompt, timeout_s)


    async def _probe_now(
        self, candidate, probe: list, probed: dict, task, budget: float,
        started: float,
    ) -> tuple[Optional[str], str]:
        """The size probe's verdict at a success exit, and what it amounts to.

        Returns `(sentence, state)` -- a `_Probe`, unpacked. The sentence is
        None unless the program must be repaired; the state says what the
        probe actually learned, and `solution_cache.worth_keeping` reads it:
        "finished inside five seconds at the statement's own scale" is
        evidence worth keeping an answer over, and "there was no time to ask"
        or "no large input could be built" are not.

        `probed` is keyed by SOURCE, not a flag, and that is the whole fix for
        a hole this had: `probed = True` used to be set the moment a too_slow
        repair was issued, so the rewritten program -- the one thing the round
        existed to produce -- shipped having never been timed. A flag cannot
        tell "already asked about this program" from "already asked about a
        program that no longer exists", and only the first is a reason not to
        ask again. Only VERDICTS are remembered: a probe that ran out of clock
        or could build no input knows nothing about the program, and a later
        call with a generator the first bar did not have should be free to
        find out.

        Every generator in hand is tried, in order, until one produces a
        verdict. Both bars are asked for one, and the second exists precisely
        so that a first generator that returns the wrong shape at every scale
        is not the end of the probe.

        There is no budget floor. There was one, and it was a sub-budget
        pretending to be prudence: it skipped the probe whenever less than
        45s remained, on the theory that a verdict with no room for the round
        it may cause is wasted. But the run is bounded by what is left anyway,
        an unfinished probe reports nothing and ships the answer as it stands,
        and a too_slow verdict with ten seconds left still buys one rotation
        round that might land. The deadline is the only clock.
        """
        if not self._size_probe or not probe:
            return None, "none"
        code = candidate.code.strip()
        if not code:
            return None, "none"
        known = probed.get(code)
        if known is not None:
            return known.sentence, known.state
        result = _Probe(None, "none")
        for generator in probe:
            left = budget - (time.monotonic() - started)
            if left <= 0:
                # Never run, so nothing is known -- and reported apart from
                # `passed` because the difference decides whether this answer
                # may be kept and handed out again without being run.
                result = _Probe(None, "skipped")
                break
            result = await self._timed_out_at_scale(code, generator, task, left)
            if result.state != "none":
                break
        if result.state in ("passed", "too_slow", "oom", "crashed"):
            probed[code] = result
        return result.sentence, result.state

    async def _timed_out_at_scale(
        self, code: str, generator: str, task, left: float
    ) -> _Probe:
        """One large valid input, run under the validator's own per-test limit.

        Returns a `_Probe`: the sentence a repair prompt reports when the
        program must change, and the state that says what was learned. NEVER
        raises into the solve: every way this can fail ends with the answer
        shipping exactly as it would have without it.

        This is the one check here that needs no oracle. Every other verdict
        in this file compares what the program produced against what something
        else said it should produce, and is therefore only as good as that
        second opinion -- which is written by the same model reading the same
        statement. "Did it finish in five seconds" has no second opinion in it
        at all, and the validator asks exactly that question, of
        `per_test_timeout_s`, at sizes the bar structurally cannot reach: every
        case on the bar carries an `expected` derived by hand, and nobody
        derives one by hand for two hundred thousand elements.

        Bounded by `left` and by nothing else. Each run is offered what is
        left of the solve, and is ALSO waited for no longer than that: the
        Docker executor cannot cut a container short, and for Rust one run
        includes a release build that takes what it takes. A run the clock
        cuts is `skipped`, not a verdict -- a timeout inside a window shorter
        than the per-test limit says nothing about whether the program would
        have finished in the full one.
        """
        if not code.strip() or not generator.strip():
            return _Probe(None, "none")
        began = time.monotonic()

        def remaining() -> float:
            return left - (time.monotonic() - began)

        async def run(*args) -> Optional[list]:
            """`_Grader.outputs` off the loop, held to the clock, never raising.

            None when the clock cut it or the executor could not be had; the
            caller decides which state that is. A thread cannot be cancelled,
            so a cut run finishes on its own in the background -- bounded by
            the executor's own timeout -- with nothing waiting on it.
            """
            allowed = remaining()
            if allowed <= 0:
                return None
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._grader.outputs, *args),
                    timeout=allowed,
                )
            except asyncio.TimeoutError:
                return None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - no executor is no verdict
                print(f"[verify] the size probe could not run "
                      f"({type(exc).__name__}: {exc}); the program is graded "
                      f"on the bar's own cases alone")
                raise _NoExecutor() from exc

        try:
            # The last rung that crashed, if any -- so a ladder that crashes
            # all the way down still reports `crashed` rather than "no input".
            crashed: Optional[tuple] = None
            for scale in PROBE_SCALES:
                if remaining() <= 0:
                    print("[verify] the size probe: out of time before a large "
                          "input could be built")
                    return _Probe(None, "skipped")
                made = await run(
                    generator, "python", "generate",
                    [{"args": [424242, scale]}],
                    max(1.0, min(remaining(), VERIFY_TIMEOUT_S * 2)),
                )
                if made is None:
                    print("[verify] the size probe: out of time while building "
                          "a large input")
                    return _Probe(None, "skipped")
                if not made or not made[0].ok:
                    # Nearly always the transport, not the generator: a return
                    # value over `PROBE_MAX_BYTES` comes back as a crash. Try a
                    # smaller budget rather than conclude anything.
                    continue
                case = made[0].value
                if not isinstance(case, dict) or not isinstance(case.get("args"), list):
                    continue
                size = len(json.dumps(case, default=str))
                if size > PROBE_MAX_BYTES:
                    continue
                # The budget the run is given. Twice the per-case limit when
                # the solve can afford it, which is what makes `outputs` hand
                # the one case the WHOLE limit: it sizes the per-case clock
                # from what is left of the budget, and a case given less than
                # the executor's own per-test timeout is killed by the outer
                # watchdog first -- "no verdict", the shape of a crash. So a
                # run is a VERDICT only when the case had the full clock
                # (`full`); cut shorter, a timeout or a crash-shaped death
                # says nothing about the program and is `skipped`.
                window = max(1.0, min(remaining(), VERIFY_TIMEOUT_S * 2))
                full = window >= VERIFY_TIMEOUT_S + 1.0
                ran = await run(
                    code, task.language, task.entrypoint, [case], window,
                )
                if ran is None:
                    print("[verify] the size probe: out of time while the "
                          "program ran at scale; nothing is known about it")
                    return _Probe(None, "skipped")
                if not ran:
                    return _Probe(None, "none")
                if not full and not ran[0].ok:
                    print(
                        f"[verify] the size probe: the program had not "
                        f"finished a {size:,}-byte input when the solve's "
                        f"clock cut it short of the validator's "
                        f"{VERIFY_TIMEOUT_S:.0f}s; that is not a verdict"
                    )
                    return _Probe(None, "skipped")
                if ran[0].timed_out:
                    print(
                        f"[verify] the size probe: the program did not finish on a "
                        f"valid {size:,}-byte input inside {VERIFY_TIMEOUT_S:.0f}s — "
                        f"the bar's own cases are all small by construction, and the "
                        f"validator runs the hidden tests at the statement's limits"
                    )
                    return _Probe(
                        f"I ran the program on one valid input of {size:,} bytes, "
                        f"generated to the limits this statement states, and it did "
                        f"not finish within {VERIFY_TIMEOUT_S:.0f} seconds.",
                        "too_slow",
                    )
                if not ran[0].ok and _looks_out_of_memory(ran[0]):
                    # The other way a program dies at size, and the one the probe
                    # could not see until grading moved into a container: the
                    # validator caps a candidate at 256 MiB with swap off and runs
                    # every hidden test in ONE of them, so a single OOM kill fails
                    # the whole suite rather than the case that caused it. That
                    # makes it worth a repair round on exactly the same footing as
                    # a timeout.
                    print(
                        f"[verify] the size probe: the program ran out of memory on "
                        f"a valid {size:,}-byte input — the validator grades in a "
                        f"container with 256 MiB and no swap, and one OOM there "
                        f"fails every hidden test"
                    )
                    return _Probe(
                        f"I ran the program on one valid input of {size:,} bytes, "
                        f"generated to the limits this statement states, and it ran "
                        f"out of memory. The validator runs every hidden test in a "
                        f"single container limited to 256 MiB with no swap, where "
                        f"one such kill fails the entire suite.",
                        "oom",
                    )
                if not ran[0].ok:
                    # A CRASH DESCENDS THE LADDER rather than ending it, which
                    # is the only way to tell the two things it can mean apart.
                    #
                    # `PROBE_MAX_BYTES` bounds the INPUT; the runner also caps
                    # what a case may hand BACK -- 256 KiB of framed status in
                    # `_batch_runner`, and the subprocess executor keeps only
                    # the last 256 KiB of stdout, where an over-long reply
                    # loses its opening frame marker. So a correct program
                    # whose answer is about the size of its input fails at the
                    # top rung, and the two failures do not even read alike:
                    # one says "serialized return value is too large" and the
                    # other says "sandbox produced no verdict (crashed or
                    # exited early)", which is also what a real crash says.
                    #
                    # No message can separate them; a smaller input can. The
                    # generator has always retried at a smaller rung for
                    # exactly this reason, and the program abandoning the two
                    # smaller rungs meant the one shape a smaller input surely
                    # fixes was the one shape that never got one. If every
                    # rung crashes it is reported as a crash, as before.
                    head = (ran[0].error or "").strip().splitlines()
                    detail = head[-1] if head else "no detail"
                    crashed = (size, detail)
                    print(
                        f"[verify] the size probe: the program did not survive "
                        f"a {size:,}-byte input ({detail}); "
                        + (
                            "the answer was too large for the runner to carry, "
                            "which is this harness rather than the program"
                            if _transport_limit(ran[0].error)
                            else "trying a smaller input to tell a fault from "
                                 "an input the statement does not allow"
                        )
                    )
                    continue
                # `runtime_ms` is the executor's, and both Docker executors
                # stamp a result with the whole container's wall time -- for
                # Rust that includes the release build. Reported as what it is.
                timing = (
                    f"{ran[0].runtime_ms / 1000.0:.1f}s of container time"
                    if task.language == "rust"
                    else f"{ran[0].runtime_ms / 1000.0:.1f}s of {VERIFY_TIMEOUT_S:.0f}s"
                )
                print(
                    f"[verify] the size probe: finished a valid {size:,}-byte input "
                    f"in {timing}"
                )
                # The one path where the program was actually timed at size and
                # finished. Everything else that returns here knows nothing
                # about the program, and says so in its state.
                return _Probe(None, "passed")
        except _NoExecutor:
            return _Probe(None, "none")
        if crashed is not None:
            size, detail = crashed
            print(
                f"[verify] the size probe: the program crashed at every size "
                f"down to {size:,} bytes ({detail}); more likely an input the "
                f"statement does not allow than a fault, so no repair is asked "
                f"-- and no pass is recorded"
            )
            return _Probe(None, "crashed")
        print(
            "[verify] the size probe: no large input could be had; the program "
            "is graded on the bar's own cases alone"
        )
        return _Probe(None, "none")

    async def _open_within(
        self, budget: float, started: float, avoid: Optional[str],
        phase: Optional[str] = None, profile=None,
    ):
        """`backend.open`, bounded by the solve's own clock.

        A fleet backend waits for a free tab, and the wait it defaults to is an
        operator setting about fleet capacity that knows nothing about this
        request's deadline -- `MINER_TAB_WAIT_S` ships at 120s and a solve
        could spend it three times over against a 280s budget. So the wait is
        bounded by THIS request's clock, and by nothing else: everything that
        is left, which is the same bound every other read in this file gets.

        The bound is offered as a keyword and the two-argument form is bounded
        from out here instead. `Backend.open` has always been `open(avoid=...)`,
        so a backend written outside this package need not have grown a
        `timeout_s` -- and a keyword it does not take would be a TypeError
        inside the one call the whole solve depends on.
        """
        share = max(1.0, budget - (time.monotonic() - started))
        # `open_for` is the CLI backend's; a browser fleet has no models to
        # choose between and never grew one. Asked for by name rather than by
        # duck-typed keyword, because a backend that HAS `open` and not
        # `open_for` is the normal case, not an error to report.
        # A NAMED model, when the caller wants one specific seat rather than
        # the best available. The repair rotation does: its whole purpose is
        # that the next round is answered by a different reading, and `avoid`
        # cannot express "this one" -- it says only "not that one", and the
        # ladder's answer to that is whatever rung happens to be next.
        if profile is not None:
            by_profile = getattr(self._backend, "open_profile", None)
            if by_profile is not None:
                try:
                    return await by_profile(profile.model, profile.effort)
                except TypeError:
                    pass
        opener = getattr(self._backend, "open_for", None)
        if phase and opener is not None:
            try:
                return await opener(phase=phase, avoid=avoid, timeout_s=share)
            except TypeError:
                pass
        try:
            return await self._backend.open(avoid=avoid, timeout_s=share)
        except TypeError:
            return await asyncio.wait_for(
                self._backend.open(avoid=avoid), timeout=max(share, 1.0)
            )

    async def _graded(
        self, reply: str, task, left: float, previous: str = "",
    ) -> Candidate:
        """`_grade`, run OFF the event loop.

        Grading is blocking work wearing an async coat: `compile_defect` shells
        out to rustc and `_Grader.check` runs the validator's own executor,
        each a `subprocess.run` of seconds -- and for the Docker backend, of a
        container start. Called straight from a coroutine it stops the event
        loop dead, and the loop is not this solve's alone.

        Measured, a 3s subprocess called from inside a coroutine, beside a
        second task holding a 1.0 second deadline:

            the other solve's 1.0s deadline fired after  3.05s

        The miner answers several validators at once (`solve_slots` is a
        semaphore, not a mutex), so that other task is another live solve. Worse,
        the deadline that decides whether a solve is PAID is itself an
        `asyncio.wait_for` in `handle_request` -- and a timer cannot fire on a
        loop that is not running. One Rust compile could therefore push every
        other in-flight solve past its cutoff, and each of those answers 504
        with no answer at all, however finished the answer already was.

        `to_thread` costs nothing here: the calling coroutine is going to wait
        for this result either way. What it buys is that everything ELSE keeps
        running while it waits.
        """
        try:
            return await asyncio.to_thread(
                self._grade, reply, task, left, previous
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never lose the answer to the check
            print(f"[verify] grading unavailable: {type(exc).__name__}: {exc}")
            return Candidate(
                code=extract_code(reply, task.entrypoint, task.language), raw=reply
            )

    def _note(self, winner: Optional[str], asked: list[str]) -> None:
        """Per-provider tally, so a model that has started failing is visible."""
        for name in asked:
            row = self._by_provider.setdefault(name, {"asked": 0, "verified": 0})
            row["asked"] += 1
        if winner:
            row = self._by_provider.setdefault(winner, {"asked": 0, "verified": 0})
            row["verified"] += 1


    def _grade(
        self, reply: str, task, left: Optional[float] = None,
        previous: str = "",
    ) -> Candidate:
        code = extract_code(reply, task.entrypoint, task.language)
        candidate = Candidate(code=code, raw=reply)
        defect = (
            rust_defect(code)
            if task.language == "rust"
            else python_defect(code, task.entrypoint)
        )
        if defect is None and task.language != "rust":
            # Only once the structural checks are happy, because they are the
            # ones that say what is wrong most precisely. A reply that will not
            # parse is not "incomplete", it is broken, and saying the wrong one
            # sends the repair round after the wrong thing.
            #
            # `partial` rides along separately: `defect` tells the MODEL what to
            # fix, and this tells `_supersedes` that what arrived is not a
            # version of the answer at all.
            defect = dropped_definitions(code, previous)
            candidate.partial = defect is not None
        # Not `left <= 0`. `_Grader.check` gives every case
        # `VERIFY_TIMEOUT_S` and nothing bounds the run as a whole, so a
        # candidate that times out on each of its cases spends that many
        # multiples of it: measured, 6 cases x 5s = 30s of executor time bought
        # with 0.2s of budget, on a verdict nothing could act on -- there is no
        # time left for a repair round and `verified` never reaches the
        # validator. The guard's own reason ("running anything buys nothing that
        # can still be acted on") is as true at 0.2s as at 0, so it asks what
        # the run could actually cost.
        #
        # The floor is ONE CASE, and it used to be `GRADE_FLOOR_S` = 15.
        #
        # Fifteen was picked to cap a demand computed from the suite size --
        # twenty cases would otherwise refuse to grade below a hundred seconds
        # -- and nothing reconciled it with the 12s round-trip floor the repair
        # loop then had (since removed: the deadline is the only clock). The
        # two constants left a band, 12 to 15 seconds, in which the loop would
        # happily spend a whole model round trip but refused a local check that
        # measures 0.78 seconds for twenty cases. A cases-only correction
        # landing there was graded 0 of 0, `_inherit_evidence` restored the
        # failures the merge had just corrected, and the round went out
        # re-reporting a disagreement that no longer existed.
        #
        # Grading must never be the thing that is too expensive when another
        # prompt is not, so the demand is now what a run actually costs at
        # minimum: one case at the per-case timeout. `check` bounds the rest
        # itself -- it chunks against the budget and stops on time -- which is
        # what makes a size-derived demand unnecessary rather than merely
        # capped.
        out_of_budget = left is not None and left < VERIFY_TIMEOUT_S
        # What the RUN may spend, as opposed to what it must have to start.
        #
        # A round trip is held back, and the first attempt at that was reverted
        # for a good reason which no longer applies. It used to trade one
        # failure for a worse one: `check` sized its run by the WORST case, so a
        # shortened budget meant only the first case or two ran, they happened
        # to pass, and a partial run with no failures in it ended the repair
        # loop -- stopping on ignorance rather than on evidence. `check` now
        # sizes each chunk by what the cases have actually cost, so an ordinary
        # twenty-case suite finishes inside three seconds and holding twelve
        # back costs nothing at all. What it buys is the round the evidence is
        # FOR: without it one grading pass could spend every second that was
        # left, and a list of failures nobody has time to report is not worth
        # the run that produced it.
        #
        # Only when a round trip is actually on the table, AND only when what
        # is left after holding it back still buys a case. Below the round-trip
        # floor the loop will not start another round whatever happens, so
        # reserving for one would throw the seconds away; and between the two,
        # subtracting the reserve leaves a sliver -- a run that grades one case
        # of twenty, passes it, reports no failures, and ends the loop on
        # ignorance rather than on evidence. That is the exact regression this
        # reserve was reverted for once already, and the guard is what keeps it
        # from coming back through the other side.
        # The whole of `left`. A round trip used to be held back from it,
        # and that was a budget inside the deadline: with twenty cases and
        # twenty seconds left it graded eight of them, called the rest unrun,
        # and a right answer went out not `self_verified` for want of the
        # twelve seconds it was saving for a round the loop then did not
        # start. Whether another round fits is the loop's question, asked
        # against the deadline; the grade's job is to grade.
        grading_budget = left
        # When the reserve starts running out from, so a SECOND grading pass
        # spends what the first one left rather than the same allowance over
        # again. Both suites can run in one `_grade` -- the validator's
        # examples, then the model's own cases -- and handing each the full
        # `grading_budget` made the reserve promise something it did not keep:
        # with 100s left, up to 176s of grading, and the round trip it was
        # holding back gone twice over. Production ships no `public_examples`,
        # so only one pass runs there and this is the belt to that brace.
        grading_started = time.monotonic()

        def _budget_left() -> Optional[float]:
            if grading_budget is None:
                return None
            return grading_budget - (time.monotonic() - grading_started)
        if defect is None and task.language == "rust" and not out_of_budget:
            # Python's check PARSED that code; Rust's only grepped it for
            # `fn main`. Ask the compiler the same question the validator will,
            # which is the only check a Rust answer gets at all when no public
            # examples shipped -- and on the run this was written for, none
            # ever did. Returns None when there is no local toolchain.
            #
            # `left` caps it: a compile is allowed to be slow, but not slower
            # than the answer it is checking is worth. See `compile_defect`.
            #
            # And when `left` has gone NEGATIVE the compile is skipped outright,
            # not merely capped. `compile_defect` floors its timeout at one
            # second, so an overrun budget still bought a temp directory and a
            # rustc process -- a whole second, spent past the deadline, on a
            # verdict nothing can act on: there is no time left for a repair
            # round and `defect` never reaches the validator. The read now
            # extends into this reserve whenever the model is still writing, so
            # arriving here with nothing left is the ordinary case rather than
            # the strange one.
            defect = _stable(compile_defect(code, left))
        if defect is not None:
            # Structurally unusable: report it without paying for execution.
            candidate.defect = defect
            candidate.code = "" if not code.strip() else code
            return candidate
        if out_of_budget:
            # The budget is gone, so running anything buys nothing that can
            # still be acted on: there is no time for a repair round, and
            # `verified` never reaches the validator -- it feeds this process's
            # cache and its stats and nothing else. It is not free, either:
            # every case gets VERIFY_TIMEOUT_S, in a subprocess or a container,
            # and the deadline above is an `asyncio.wait_for` that answers 504
            # rather than late. The check would be paid for with the answer it
            # was checking. The structural checks above already ran; they cost
            # microseconds and are what ranks this candidate.
            # Gated on `task.public_examples` until now, which is every
            # live task: production ships none, so the one line explaining why
            # an answer went out ungraded was the one line that never printed.
            # What could not be run is what to name.
            # The comparison against the reference is run by the caller, not
            # here, so the only suite this can report as unrun is the
            # validator's own -- which live traffic never ships.
            if task.public_examples:
                print(
                    f"[verify] out of budget before the "
                    f"{len(task.public_examples)} public example(s) could be "
                    f"run; submitting the answer unverified"
                )
            return candidate

        # BOTH suites, in this order, because turn 2 was asked to pass both:
        # the validator's examples (in `<examples>`) and the model's own cases
        # from turn 1 (in `<must_pass>`). Only one of them used to run. With
        # examples shipped the own cases were quoted in the prompt and never
        # executed, so a program right on the one example and wrong on its own
        # boundary case verified, ended the loop and shipped -- the repair round
        # that exists to catch exactly that never fired. Live traffic ships no
        # examples, which is why it went unnoticed rather than why it was fine.
        #
        # The ORDER is the whole of the precedence. The validator's examples are
        # ground truth: when they fail, the program is wrong, there is nothing
        # to weigh, and the own cases are not run at all -- a second opinion
        # from the same model on a program already known wrong tells us nothing
        # and costs an executor run per case. Only once they are all green does
        # a disagreement with the model's OWN cases become the open question,
        # and `failures` then carries that instead. So `failures` names one
        # suite at a time and `from_self_tests` says which, which is what lets
        # the repair prompt ask for the right thing.
        if task.public_examples:
            try:
                passed, total, failures, _ = self._grader.check(
                    code, task.language, task.entrypoint, task.public_examples,
                    # What is left NOW: the compile above may have spent some
                    # of it, and a Rust compile spends seconds.
                    budget_s=_budget_left(),
                )
            except Exception as exc:  # noqa: BLE001 - a broken grader loses no answer
                print(f"[verify] local grading unavailable: {type(exc).__name__}: {exc}")
                return candidate
            candidate.passed, candidate.total, candidate.failures = (
                passed, total, failures
            )
            if failures:
                return candidate
        # The candidate is not RUN here. It used to be, against cases the
        # model wrote for itself; now the differential runs it once against
        # the reference's answers and `_record_differential` writes the result
        # back onto this candidate. Running it in both places would pay for
        # every case twice -- a container apiece on Rust.
        return candidate

    # -- what a Rust answer is actually checked with ----------------------- #
    def rust_support(self) -> dict[str, str]:
        """The two independent checks a Rust answer gets, and their state.

        Both can be off at once, and on a box with neither the only thing
        standing between a model's reply and the validator is `rust_defect`,
        which greps a fenced block for `fn main`. Measured over 45 archived
        submissions: 6 of 18 Rust answers would not build, and among them were
        a prompt echo, a tool call and a program truncated mid-identifier --
        all three of which carry those characters.

        Cheap on purpose: `rustc_path` is memoised after its first lookup and
        `_Grader.state` never probes, so `/solver-status` can report this.
        """
        compiler = rustc_path()
        return {
            "compile_gate": f"rustc at {compiler}" if compiler else "off: no local rustc",
            "grading": self._grader.state("rust"),
        }

    async def check_rust_support(self) -> dict[str, str]:
        """Probe both now, at startup, and say what is missing.

        Neither is looked at until the first Rust task arrives otherwise, so a
        box with no toolchain and no daemon looks perfectly healthy -- the
        fleet is up, the doctor is clean, `/health` answers -- right until a
        Rust challenge is graded by nobody. That is the failure this miner is
        least able to see and the operator most able to fix, so it is worth one
        `which` and one `docker info` before serving.
        """
        await asyncio.to_thread(self._probe_rust)
        support = self.rust_support()
        blind = support["compile_gate"].startswith("off") and support[
            "grading"
        ].startswith("unavailable")
        if blind:
            print(
                "[verify] WARN: no local rustc and no working Rust executor, so a "
                "Rust answer is checked only by a grep for `fn main` before it is "
                "submitted. Installing a toolchain restores the compile gate "
                "without Docker; grading Rust needs the daemon."
            )
        else:
            print(
                f"[verify] rust: compile gate {support['compile_gate']}, "
                f"grading {support['grading']}"
            )
        return support

    def _probe_rust(self) -> None:
        """Both lookups, in a worker thread. Neither is allowed to raise."""
        rustc_path()
        try:
            self._grader.executor("rust")
        except Exception:  # noqa: BLE001 - `_Grader` has already said why
            pass

    def stats(self) -> dict[str, Any]:
        return {
            "solver": dict(self._counts),
            "providers": {k: dict(v) for k, v in self._by_provider.items()},
            "fleet": self._backend.stats(),
            "rust": self.rust_support(),
        }

    async def aclose(self) -> None:
        await self._backend.aclose()


def _cache_key(task) -> str:
    import hashlib

    material = f"{task.language}\0{task.entrypoint}\0{task.statement}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

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
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple, Optional, Protocol, Sequence

from rlvr.types import TestCase

from .prompts import (
    build_code_prompt,
    build_expected_prompt,
    build_repair_prompt,
    build_resume_prompt,
    build_tests_prompt,
    dropped_definitions,
    extract_code,
    extract_expected,
    extract_generator,
    extract_self_tests,
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

# The most failing cases one round will put to an independent reader.
#
# Not a budget -- it is a statement about what the judge is FOR. One or two
# cases disagreeing is two readings of a statement differing about a clause,
# which is the question a third reading can settle. Half the bar disagreeing
# is not that: it is a wrong program, nobody is confused about the statement,
# and the round is better spent asking for a better program. Measured over 54
# live solves, ten of the fifteen correction rounds were single-case
# disagreements and three were 6, 14 and 19 cases at once -- the split is real
# and it falls here.
MAX_ADJUDICATED = 2

# The least slice a read can be handed: `send(prompt, max(MIN_SLICE_S, left))`
# pads every slice up to it, so a round or a pass started with less than this
# left would run PAST the deadline by construction. Not a budget -- the
# granularity of the clock -- and the only thing besides the deadline itself
# that decides whether another round, pass or handoff is started.
MIN_SLICE_S = 1.0

# There is deliberately no round-trip floor here. There was one (12s, 20s on
# the CLI backend), and it was a second clock under the deadline: below it the
# loop refused another correction round and shipped a program it knew to be
# wrong with time still on the validator's clock. A round started with too
# little left is cut by the deadline like any other read, `_supersedes` keeps
# the fragment from displacing the answer in hand, and the delivery reserve
# ships that answer. The deadline is the only clock. (Operator's rule.)

# Under this, a round did not involve the model. `send` blocks on a chat UI
# until the reply finishes -- tens of seconds, normally -- so a round that came
# back in under a couple of seconds read something that was already on the page.
# It is the difference between a model that answered the same way twice and a
# tab that is handing back the previous answer forever, and only the second is
# a reason to stop correcting.
STALE_ROUND_S = 2.0

# How many rounds running may leave the PROGRAM untouched before the repair
# prompt stops offering to correct a case at all and asks for the program.
#
# The escape hatch is there because the model's own cases can be wrong -- turn 1
# reasons its `expected` values out before any program exists -- and a round
# that blames the code for a wrong case breaks a correct program. Left open
# while nothing converges it is no longer that: two rounds running without the
# program changing is the one thing a correction phase cannot afford, whether
# they were spent correcting the bar or re-sending the same code. Two, not one,
# because the first correction is the ordinary case this whole path was built
# for.
CASES_ONLY_ROUNDS = 2

# ------------------------------------------------------- the round budget --
# How many correction rounds one conversation gets before the repair is
# carried to another model, and how many that one gets before the answer in
# hand is the answer.
#
# Measured over 102 production solves: 76 needed no correction at all, 20
# finished after ONE round of it, 4 after two, 1 after three, and 1 ran to
# eight. So 96 of 102 are done inside two correction rounds and 101 inside
# three -- a third round on the first conversation costs one solve in a
# hundred, and everything past it has never once paid.
#
# What it prevents is on a later log, after the case-escape hatch fell silent:
# three solves of eleven spent SEVEN, NINE and NINE correction rounds
# alternating between a reply the parser dropped and a rewrite that failed the
# same case, 115-200s each, one of them ending `exit=cutoff` with the budget
# gone. Nothing in the loop stopped it: `SOLVER_MAX_ATTEMPTS` defaults to 0,
# which is unlimited, and the deadline was the only terminator.
#
# Both are `CASES_ONLY_ROUNDS + 1`, and that is a relationship rather than two
# numbers that happen to be equal. The case offer is withdrawn after
# CASES_ONLY_ROUNDS rounds that left the program untouched, and the withdrawal
# is worth nothing unless a round remains for it to act in: a budget equal to
# the threshold would fire the withdrawal and end the conversation on the same
# round, so the one prompt that says "this time it is the program that has to
# change" would never be sent. One more, and it is sent exactly once, which is
# what it is for. The handoff gets the same for the same reason.
FIRST_PHASE_ROUNDS = CASES_ONLY_ROUNDS + 1
HANDOFF_ROUNDS = CASES_ONLY_ROUNDS + 1

# There is no "rounds one model gets before the repair moves on" any more, and
# `SOLVER_REPAIR_ROTATE_FROM` is gone with it. A per-model round quota beside
# the rotation list was the only way to say a RATIO while the list could name
# each model once; the schedule says it in the list -- `opus, fable, opus` is
# opus on two rounds in three -- so the quota is one setting expressing what
# another setting already says. See `_scheduled_profile`.
#
# The measured shape it answered is unchanged and the schedule answers it too:
# of ten single-case disagreements in a live run, nine ended with the program's
# own author ruling its own case wrong and keeping its program, so the round
# after the author's first is the one least likely to find anything new -- and
# on the shipped schedule that round is fable's.
# The most of the bar ONE repair reply may rewrite, as a fraction. A reply
# that corrects more than this is re-specifying the problem rather than fixing
# a case, and is refused whole.
#
# Measured on a production day: the share of the bar rewritten by each of the
# thirteen correction replies was 5, 7, 7, 8, 8, 10, 10, 10, 10, 11, 15, 29
# and 68 percent. A third fires on the 68 and on nothing else. That reply came
# from a program already known wrong on two inputs: it rewrote fifteen of its
# twenty-two cases with nine seconds left, and the solve shipped reporting it
# had passed all twenty-two.
BULK_CORRECTION_SHARE = 3
# How many bulk rewrites a pass tolerates before the case offer is withdrawn
# for good. Refusing without this is a spin: the reply comes back the same.
MAX_BULK_REFUSALS = 2


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
    # a correction round is allowed to change: see `_merge_cases`.
    failed_cases: list[dict[str, Any]] = field(default_factory=list)
    # What the program actually produced for each of those, in the same order.
    # `failures` renders this for a model to read; the judge needs the value,
    # because settling which of the program and the case is wrong means
    # comparing an independent reading against both.
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


def _expectation(case: dict) -> str:
    """What a case expects, as one comparable string. A correction is a case
    that came back with the same call and a different one of these; a case
    re-sent under a new name with the same answer is not a correction."""
    return json.dumps(case.get("expected"), sort_keys=True, default=str)


def _case_key(case: dict) -> tuple:
    """What makes two cases the same case: the call, not the answer.

    A correction changes what a case EXPECTS. Keying on the input is what lets
    the corrected version be recognised as the same case rather than an
    additional one -- and it is why `expected` is deliberately absent from the
    key.
    """
    # `key=repr` on the sort, not the default comparison: a kwargs dict with
    # keys of mixed type (`{1: 2, "a": 3}`) makes `sorted` raise TypeError, and
    # `ast.literal_eval` -- the tolerant parser a corrected array may come
    # through -- can produce exactly that where `json.loads` cannot. The whole
    # pass is wrapped in a handler that would report the crash as a dead
    # backend and abandon the solve.
    return (
        repr(case.get("args", [])),
        repr(sorted((case.get("kwargs") or {}).items(), key=repr)),
    )


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


def _scheduled_profile(rotation: Sequence, correction_round: int):
    """Which model answers correction round `correction_round`, 1-indexed.

    A POSITION in the schedule rather than a search through it, and the change
    is what lets the schedule state a RATIO. The search this replaced skipped
    every model that had already had the repair, so a model could appear once
    per lap and no more -- which makes every model equal by construction and
    leaves no way to say "opus twice for every fable once" except by inventing
    a per-model round quota beside the list.

    Indexing says it in the list itself. `opus, fable, opus` cycles as

        1 opus   2 fable   3 opus   4 opus   5 fable   6 opus   7 opus   8 fable

    -- opus on two rounds in three, fable on one, and the pattern is readable
    off the setting without reference to anything here. A schedule naming one
    model is a repair that never moves; naming each once is strict alternation;
    repeating one is a ratio. None of those is a special case.

    The counterpart is that the caller must not hand off on a round the
    schedule does not move on: two consecutive `opus` entries mean the same
    conversation continues, which is what keeps its context.
    """
    if not rotation:
        return None
    return rotation[(correction_round - 1) % len(rotation)]


def _same_expected(left: Any, right: Any, language: str) -> bool:
    """Whether two answers to the same call are the same answer.

    THE VALIDATOR'S OWN COMPARISON, imported rather than reimplemented, and
    that is not fastidiousness. Two earlier attempts to measure whether two
    readings agree compared with `json.dumps` and reported 0% and 48%
    agreement; every point of the difference was whitespace, because a Rust
    answer is stdout judged on whitespace tokens and `"OK 2\\nD 0"` and
    `"OK 2 D 0"` are the same answer. A comparison stricter than the grader's
    invents disagreements, and one looser reports a pass the validator will
    not give.
    """
    if language == "rust":
        from rlvr.execution.rust_judge import outputs_match

        if not isinstance(left, str) or not isinstance(right, str):
            return False
        return outputs_match(left, right)
    from rlvr.execution.compare import values_equal

    try:
        return bool(values_equal(left, right))
    except Exception:  # noqa: BLE001 - an uncomparable pair is not a match
        return False


def _union_bars(
    first: list[dict], second: list[dict], language: str = "python"
) -> tuple[list[dict], dict[tuple, Any]]:
    """Two independently written bars, joined. Returns (cases, split).

    UNION, not intersection, and the measurement says why. Two models asked to
    choose test inputs for the same statement shared five of the 233 they
    wrote (`calibration/two_bar_overlap.py`) -- so an intersection would be
    empty on almost every solve, and agreement about WHICH inputs to test can
    carry no signal at all. What that same 2% overlap means for a union is the
    opposite and is the whole point: the second reader's cases are almost
    entirely inputs the first never thought to probe, written by a reading of
    the statement that is not the one that produced the program.

    Keyed by the CALL, so a case both bars happened to write appears once.
    Where they wrote the same call and disagree about what it returns, the
    first bar's answer stands and the key is returned in `split`. A
    disagreement there is the statement being genuinely ambiguous rather than
    either model being careless, so it is not evidence against the program,
    and `split` is what lets the caller treat it that way.

    How OFTEN that happens is measured for one pair only:
    `calibration/fixed_inputs.py` held the inputs fixed, asked opus and sonnet
    for the expected values, and they split on 6 of 97. The second bar is no
    longer sonnet, so that 6% is not this pair's rate and nothing here should
    be read as claiming it is -- what carries over is the ARGUMENT, which
    never depended on the number: two readings of one statement that differ
    about a value have found an ambiguity in the statement, whoever the two
    readers are.
    """
    seen = {_case_key(case): case for case in first}
    split: dict[tuple, Any] = {}
    merged = list(first)
    for case in second:
        key = _case_key(case)
        mine = seen.get(key)
        if mine is None:
            seen[key] = case
            merged.append(case)
        elif not _same_expected(
            mine.get("expected"), case.get("expected"), language
        ):
            # The GRADER's comparison, not `_expectation`'s string form. For
            # Rust a verdict is stdout compared on whitespace tokens, so
            # `"OK 2\nD 0"` and `"OK 2 D 0"` are the same answer -- and
            # calling them a disagreement would mark the case `split`, which
            # is what keeps it away from the judge. Two bars that agree would
            # then look like an ambiguity, and the case would go back to being
            # settled by the program's own author.
            split[key] = case.get("expected")
    return merged, split


def _merge_cases(
    agreed: list[dict], revised: list[dict], failed: list[dict]
) -> tuple[list[dict], str]:
    """The agreed suite with a correction applied. Returns (cases, what changed).

    A correction round happens because the program disagreed with a case, and
    only one of the two can be wrong. The repair prompt says so and offers both
    ways out: send the program back fixed, or -- if the CASE was the thing that
    was wrong -- send that case back corrected.

    What a correction may touch is exactly the cases the program FAILED. That
    single rule is what makes accepting a short array safe:

      * A case the program PASSES cannot be corrected, dropped or weakened. The
        bar a program has already cleared is not up for negotiation, so the
        obvious way to game this -- delete the case you cannot pass -- is not
        reachable from here.
      * A failing case may be corrected in place (same call, new expectation)
        or swapped for a different one, when the call itself was the thing that
        made no sense. Both are what "the case was wrong" means in practice.
      * The suite keeps its SIZE. Every failing case removed must be replaced,
        so a bar cannot be cleared by deleting what the program could not pass;
        and it cannot GROW here either, because cases written beside a program
        agree with its bugs and the program turn is already refused its own for
        that reason.

    Replacing the whole suite was the alternative, and refusing anything shorter
    was what came before. Both were wrong in the same place. Demanding the full
    array back meant a twenty-case suite was re-sent to correct one of them,
    which is slower, likelier to be truncated mid-array, and -- when it came
    back one case short for any reason at all -- refused outright, so the one
    wrong case broke a correct program on every remaining round of the solve.
    """
    if not revised:
        return list(agreed), ""
    out = {_case_key(case) for case in failed}
    keep = [case for case in agreed if _case_key(case) not in out]
    seen = {_case_key(case) for case in keep}
    merged = list(keep)
    for case in revised:
        key = _case_key(case)
        if key in seen:
            # It re-states a case the program PASSES. Not a correction this
            # round is entitled to make, and not a hostile act either -- a
            # model that re-sends its whole suite lands here on every passing
            # case. Keep the version that was agreed.
            continue
        merged.append(case)
        seen.add(key)
    corrected = len(merged) - len(keep)
    # A failing case the reply did NOT re-state stays as it was. The prompt
    # may report two disagreements and the natural reply corrects one of
    # them; read as "the other was dropped" that reply was refused outright,
    # every round, and the one case the model had fixed never landed.
    # Untouched is not deleted: the case is still on the bar, still failing,
    # and still reported next round.
    for case in failed:
        if len(merged) >= len(agreed):
            break
        if _case_key(case) not in seen:
            merged.append(case)
            seen.add(_case_key(case))
    if corrected == 0:
        # Nothing moved. Either the reply re-sent the suite unchanged -- the
        # ordinary shape, and nothing to say about it -- or every case it
        # carried matched one the program PASSES and was skipped above. The
        # second is worth a word: it is what a correction looks like when the
        # failing set this round was merged against is not the set the prompt
        # actually quoted, and that has been a real bug rather than a
        # hypothetical one.
        return list(agreed), "" if not failed else "NOTHING: the correction changed no case that failed"
    if len(merged) != len(agreed):
        # The suite keeps its SIZE. Only the failing cases may change, and each
        # one has to be replaced rather than simply removed -- "the program can
        # drop the case it disagrees with and attach the corrected one in its
        # place", with the second half enforced.
        #
        # Both directions of this were measured breaking a solve.
        #
        # SHORTER is a bar cleared by deletion. A repair that came back with
        # only the cases it already passed dropped the one it did not, and a
        # program failing 1 of 2 went out reported `self=1/1`, verified on
        # local, on a bar it had rewritten in the same breath.
        #
        # LONGER is back-filling wearing a correction's clothes. Cases written
        # beside a program agree with its bugs -- which is the entire argument
        # for asking for them in a separate turn -- and a repair round that
        # re-sends the program with its own cases attached would otherwise get
        # them onto the bar by the side door the program turn is refused at.
        # Measured: a buggy program adding two cases of its own to a one-case
        # bar and finishing 2/3 instead of 0/1.
        return list(agreed), (
            f"REFUSED: {len(agreed) - len(keep)} case(s) failed and "
            f"{corrected} came back; a failing case may be "
            f"corrected, not dropped, and the suite may not grow here"
        )
    return merged, f"{corrected} failing case(s) corrected"


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
        # WITH the cases they came from. Splitting these was a silent hole: the
        # inherited `failures` built a repair prompt quoting concrete cases and
        # offering to take them back corrected, while the empty `failed_cases`
        # left `_merge_cases` with nothing in play -- so the correction the
        # prompt had just asked for matched an existing key, read as "it
        # re-states a case the program passes", and was dropped without a word.
        # The feature and its own evidence have to move together.
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
            # a judge needs the value itself, because deciding whether the
            # program or the case is wrong means comparing an independent
            # reading against BOTH of them.
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

        Kept as the four-tuple every caller but the judge wants, rather than
        widened in place: unpacking is positional, and a fifth element would
        have been a silent `ValueError` in each of them.
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
    bar_provider: Optional[str] = None
    rounds: int = 0
    corrected: int = 0
    probe: str = ""
    bar2_provider: Optional[str] = None
    union: Optional[tuple] = None
    adjudicated: tuple = ()
    contested: int = 0
    repair: tuple = ()
    split: int = 0

    @classmethod
    def of(cls, plan: "_Plan") -> "_Shipped":
        return cls(
            exit=plan.exit, bar_provider=plan.bar_provider, rounds=plan.rounds,
            corrected=plan.corrected, probe=plan.probe,
            bar2_provider=plan.bar2_provider, union=plan.union,
            adjudicated=tuple(plan.adjudicated.items()),
            contested=plan.contested, repair=tuple(plan.repair),
            split=plan.split,
        )


@dataclass
class _Plan:
    """What the NEXT pass of one solve should do. One per `solve_task`.

    Only `two_phase` so far, and it exists because a cases turn that costs a
    whole pass used to cost EVERY pass. `solve_task` retries `_attempt` up to
    `MAX_PASSES` times while it is holding nothing, and nothing remembered that
    turn 1 had already proved unaffordable here -- so a task whose cases turn
    timed out burned all four passes on four more cases turns and submitted
    nothing, having never once asked for a program.

    Per-solve rather than per-solver: solves run concurrently on one instance,
    and a flag on `self` would let one task's bad luck disable the split for
    every other task in flight.

    It answers "was turn 1 unaffordable HERE", so only a failure that belongs to
    the task and the site clears it. A tab that went blind is retired on the
    spot and the next pass is served by another one, so its failure says nothing
    about the task -- see `_attempt`, where the two are told apart.

    Cleared, the next pass asks for the program ALONE. There is no combined
    turn to fall back to: cases written beside a program are back-filled from
    what it happens to do and agree with its bugs, which is the whole argument
    for splitting the turns, and a prompt that asks for a second block the
    grader will not trust spends output tokens inside the deadline. The cost is
    real and it is the right one: that task goes out ungraded rather than
    graded against evidence worth nothing.
    """

    two_phase: bool = True

    def __init__(self) -> None:
        # What the summary line reports beside the tally, so a hidden-suite
        # outcome can be joined back to how the solve went: rounds asked, and
        # cases the model corrected that stood. Measured need: a log of 76
        # solves graded 83.5% right, and no way to tell a corrected-bar solve
        # from a clean one.
        self.rounds = 0
        self.corrected = 0
        # Which model wrote the BAR, when it was written somewhere other than
        # where the program was. The summary line is what a hidden-suite
        # outcome is joined back to, and with a cases model and a program
        # model that can differ, "which model answered" is two questions.
        # This is also the instrument that makes `SOLVER_CLI_PHASE_PROFILES`
        # measurable rather than guessable: without it there is no way to ask
        # whether the bar's model changed anything.
        self.bar_provider: Optional[str] = None
        # What the FIRST grade found, before any repair: how many of the
        # program's cases it failed the moment it was written. This is phase
        # 3's trigger rate, and it is the number the correction phase turns on.
        # Measured over the 102 archived solves, 76 of them never entered a
        # correction round at all -- so the loop that converges 25 times out of
        # 26 was only ever offered a quarter of the traffic, and the ~20 wrong
        # answers those runs shipped are mostly among the 76 it never saw. A
        # log that reports `rounds=1` cannot tell "the program was right" from
        # "the cases could not tell", and those need opposite work.
        #
        # The first grade of the SOLVE, not of the pass that shipped. A second
        # pass runs only after the first failed to deliver, and this asks
        # whether the bar found anything on this task at all -- so a 3/18 from
        # an abandoned first pass is the answer even when the second pass's
        # fresh program cleared the same bar. Unlike `exit`, which describes
        # how the solve ended and so takes the LAST pass.
        self.disagreed: Optional[tuple[int, int]] = None
        # Which condition ended the correction loop. Every way out of the pass
        # notes a reason before returning -- the seven `break`s, the cases
        # turn that never answered, and the backend failure -- so the line
        # never carries an earlier pass's reason for this one, or nothing.
        self.exit = ""
        # What the SIZE PROBE said about the program that shipped. The probe
        # answers the one question the bar structurally cannot -- every case on
        # the bar carries a hand-derived `expected`, so none of them is ever the
        # size the validator runs -- and until this field existed the log could
        # not tell an answer timed at scale from one that was never timed. The
        # states are worth keeping apart -- see `_Probe`: `passed` is
        # evidence; `too_slow` and `oom` are answers that shipped known-bad at
        # scale because the clock ran out mid-repair; `crashed` died at scale
        # some other way; `skipped` and `none` were never timed. Describes the
        # program that SHIPPED: cleared whenever a different program becomes
        # the one in hand, so a verdict never outlives the program it was
        # about.
        self.probe = ""
        # Calls the two bars both wrote and disagreed about. Neither a failure
        # nor a verdict on its own -- see `_union_bars` and the split routing
        # in `_attempt` -- but the one number that says how ambiguous the
        # statement was, so it is recorded.
        self.split = 0
        # The SECOND bar: which model wrote it, and what the union came to.
        # `union` is (bar1, bar2, union) case counts, so a line can say whether
        # the second reader contributed anything the first had not thought of
        # -- two models share about 2% of their chosen inputs, measured, which
        # is the entire reason a second bar is worth a turn.
        self.bar2_provider: Optional[str] = None
        self.union: Optional[tuple[int, int, int]] = None
        # How the judge settled cases the bar and the program disagreed about,
        # counted by route. `case` means an independent reading agreed with the
        # bar and the program had to change; `program` means it agreed with the
        # program and the case was corrected; `contested` means it agreed with
        # neither, which is the statement being genuinely ambiguous.
        #
        # This is the instrument for the finding that motivated the judge: over
        # 54 live solves, nine of ten single-case disagreements ended with the
        # program's own author editing its own case. Without a route breakdown
        # there is no way to see whether that ratio moved.
        self.adjudicated: dict[str, int] = {}
        self.contested = 0
        # Which models the correction rounds ran on, in order. A repair that
        # stays where the program was written is a model reviewing its own
        # reading; this says whether it left, and where it went.
        self.repair: list[str] = []

    def note_adjudicated(self, route: str) -> None:
        self.adjudicated[route] = self.adjudicated.get(route, 0) + 1

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
        independent_bar: Optional[bool] = None,
        second_bar: Optional[bool] = None,
        judge: Optional[bool] = None,
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
        # Where the cases are written -- see `_attempt`. Read from the
        # environment when the caller says nothing, because it decides how
        # many conversations a solve holds open and a scripted backend that
        # hands every conversation the same replies cannot serve two.
        self._independent_bar = (
            _env_on("SOLVER_INDEPENDENT_BAR")
            if independent_bar is None
            else bool(independent_bar)
        )
        # A SECOND bar, beside the first, on another model. Requires the
        # independent bar: it is a third concurrent conversation, and the
        # sequential shape has no place to put it.
        #
        # Costs one cases turn of output tokens per solve and no wall-clock,
        # and it is the only thing here that reaches a solve where the bar and
        # the program agreed with each other and were both wrong.
        self._second_bar = self._independent_bar and (
            _env_on("SOLVER_SECOND_BAR")
            if second_bar is None
            else bool(second_bar)
        )
        # The judge: a third reader, asked what a disputed call returns.
        # Costs a short turn and only on the rounds that have a disagreement
        # to settle -- about a quarter of solves. See `_adjudicate`.
        self._judge = (
            _env_on("SOLVER_JUDGE")
            if judge is None
            else bool(judge)
        )
        # The models a correction round moves through. Empty for any backend
        # that has no models to choose between -- a browser fleet -- in which
        # case the loop behaves as it always did: one handoff to a fresh tab.
        try:
            from .claude_cli import cli_repair_rotation

            self._rotation = cli_repair_rotation()
        except (Exception, SystemExit):
            # `SystemExit`, explicitly: `cli_repair_rotation` raises it on a
            # malformed value, as every other config reader in that module
            # does, and it is a BaseException -- so `except Exception` was a
            # fallback that could never run. A typo in one env var must cost
            # the rotation, not the miner.
            print("[verify] SOLVER_REPAIR_ROTATION could not be read; "
                  "correction rounds stay where the program was written")
            self._rotation = ()
        # The size probe: one extra fenced block on the cases turn, and one
        # local run of the finished program on a large valid input. No extra
        # model turn, and no oracle -- see `_timed_out_at_scale`.
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
                    # NO CODE: `_run_self_tests` is gated on `code.strip()`, so
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
            contested=shipped.contested,
            probe=shipped.probe,
        ) and not self._grader.degraded(task.language):
            record = solution_cache.record(
                code=best.code, raw=best.raw, task=task, bar=best.self_bar,
                probe=shipped.probe,
                # Every model that touched what a hit is served on, in the
                # order they touched it, once each: the bars it cleared, the
                # program's author, the models the repair went through.
                providers=list(dict.fromkeys(
                    p for p in (
                        shipped.bar_provider, shipped.bar2_provider,
                        won_with, *shipped.repair,
                    )
                    if p and p != "late"
                )),
            )
            await asyncio.to_thread(solution_cache.save, key, record)
        elapsed = time.monotonic() - started
        print(
            f"[verify] {task.language} entrypoint={task.entrypoint} "
            f"provider={won_with or 'none'} "
            + (f"bar={shipped.bar_provider} " if shipped.bar_provider else "")
            + (f"bar2={shipped.bar2_provider} " if shipped.bar2_provider else "")
            + (
                f"union={shipped.union[2]}({shipped.union[0]}+{shipped.union[1]}) "
                if shipped.union else ""
            )
            + (f"split={shipped.split} " if shipped.split else "")
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
            # the same thing about an answer that passed every case it had and
            # about one that was never run. This says which, without ever
            # claiming the word `verified` for a model agreeing with itself.
            + (
                f"(verified on local: passed all {best.self_total} of its "
                f"own cases; no public examples exist to confirm it) "
                if best.self_verified
                else ""
            )
            + f"rounds={shipped.rounds} corrected={shipped.corrected}/{best.self_total}"
            # What the FIRST grade found and what ended the loop. `rounds=1`
            # alone cannot tell a program that was right from one whose cases
            # could not tell, and those need opposite work: 76 of the 102
            # archived solves reported rounds=1, and most of that run's wrong
            # answers are among them.
            + (
                f" disagreed={plan.disagreed[0]}/{plan.disagreed[1]}"
                if plan.disagreed is not None
                else " disagreed=none"
            )
            # How the judge settled what the bar and the program disagreed
            # about, and what the probe said about the program that shipped.
            + (
                " adjudicated="
                + ",".join(f"{route}:{n}" for route, n in shipped.adjudicated)
                if shipped.adjudicated else ""
            )
            + (f" contested={shipped.contested}" if shipped.contested else "")
            + (f" repair={'>'.join(shipped.repair)}" if shipped.repair else "")
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
            diagnostics={
                "provider": won_with,
                "bar_provider": shipped.bar_provider,
                "bar2_provider": shipped.bar2_provider,
                "union": list(shipped.union) if shipped.union else None,
                "split": shipped.split,
                "bar": best.self_bar,
                "self_passed": best.self_passed,
                "self_total": best.self_total,
                "self_observed": best.self_observed,
                "self_verified": best.self_verified,
                # WHICH cases it failed, and what it produced for them, not
                # just how many. A wrong answer in the archive is a question
                # about one case, and a count cannot answer it.
                "failed_cases": best.failed_cases,
                "failures": best.failures,
                "disagreed": list(plan.disagreed) if plan.disagreed else None,
                "adjudicated": dict(shipped.adjudicated),
                "contested": shipped.contested,
                "rounds": shipped.rounds,
                "corrected": shipped.corrected,
                "repair": list(shipped.repair),
                "probe": shipped.probe,
                "exit": shipped.exit,
                "elapsed_s": round(elapsed, 1),
                "budget_s": round(budget, 1),
            },
        )

    async def _ask_for_cases(
        self, conversation, task, left: float, probe=None, independent: bool = False,
    ):
        """Turn 1: the model's cases, before it has written the program.

        Returns the cases, or ``[]`` when the reply carried none usable, or
        ``None`` when the CONVERSATION is the problem -- unreadable, or still
        writing -- in which case turn 2 must not be sent into it at all.

        Read against the SOLVE's clock and nothing else: turn 1 gets whatever
        is left, the same ceiling the program turn gets. There is no partial
        budget here to cut a thinking model off with -- see the note above
        `_Plan` for why every version of that cap was a mistake.
        """
        slice_s = max(1.0, left)
        # `probe` is a list the caller passes to receive the generator, rather
        # than a second return value: this returns cases, [] and None with
        # three different meanings the whole solve turns on, and widening that
        # to a tuple to carry an optional extra would put the probe in the way
        # of the bar. Absent list, no ask, and the turn is byte-identical to
        # what it was.
        want_probe = probe is not None
        prompt = build_tests_prompt(
            task.language, task.statement, task.entrypoint, task.public_examples,
            want_probe=want_probe,
            # Whether this conversation will be asked for the program. The
            # bar's own conversation never is, and the prompt used to promise
            # it would be.
            independent=independent,
        )
        # The slice IS everything left, and everything left runs to the point
        # the answer stops being deliverable. Nothing is held back to extend to.
        reply = await conversation.send(prompt, slice_s)
        if getattr(conversation, "still_writing", False) or getattr(
            conversation, "empty_reason", None
        ) in ("unreadable", "unfinished"):
            print(
                f"[verify] the cases turn came back "
                f"{getattr(conversation, 'empty_reason', None) or 'unfinished'}; "
                f"the program request cannot go into a conversation that has "
                f"not answered the last one"
            )
            return None
        if want_probe:
            found = extract_generator(reply)
            if found:
                probe.append(found)
        cases = extract_self_tests(reply, task.entrypoint, task.language)
        if not cases:
            print("[verify] the cases turn produced none usable; "
                  "asking for the program without them")
        return cases

    async def _attempt(
        self,
        task,
        remaining: float,
        avoid: Optional[str],
        plan: Optional["_Plan"] = None,
        pass_no: int = 1,
    ) -> tuple[Optional[Candidate], Optional[str]]:
        """One model, one conversation: initial answer plus repair rounds.

        The repair rounds deliberately stay in that single conversation so the
        model sees its own previous attempt beside the failure report. Returns
        the best candidate it produced and which provider produced it.
        """
        started = time.monotonic()
        # 1.0, not 5.0. A floor above what the caller can afford does not buy a
        # longer read, it buys a cancelled solve: `solve_task` has already cut
        # the request down to something deliverable, and raising it back here
        # undoes that silently.
        budget = max(1.0, remaining)
        best: Optional[Candidate] = None
        conversation = None
        provider: Optional[str] = None
        # Which provider produced `best` -- not which one this pass is talking
        # to NOW. They part company the moment a repair is carried elsewhere:
        # the pass then ends holding an answer from the first model and a
        # conversation with the second, and returning the latter credits the
        # wrong account. That is the one question the per-provider tally exists
        # to answer, so it follows `best` rather than the conversation.
        #
        # Bound out here beside `provider`, for the same reason: `open()` can
        # raise, the handler below catches it, and the return then reads a name
        # the try block never got to bind.
        best_provider: Optional[str] = None
        # The cases conversation and the turn running in it, when the bar is
        # written somewhere other than where the program is. Bound out here
        # beside `best_provider` and for the same reason: the cleanup below
        # runs for every way this returns, including the ones that never got
        # as far as opening them.
        bar: list = []
        bar_task: Optional[asyncio.Task] = None
        bar_started = 0.0
        # The SECOND bar, written beside the first by a different model. Same
        # shape, same lifetime, and bound out here for the same reason: the
        # cleanup below runs for every way this returns.
        bar2: list = []
        bar2_task: Optional[asyncio.Task] = None
        bar2_started = 0.0
        # Cases the two bars wrote for the same call and disagree about, as
        # {key: what the second bar expected}. Not a failure of either: two
        # readings differing about one call is the statement being ambiguous
        # there, and a program that disagrees with such a case is not thereby
        # shown to be wrong.
        split_keys: dict[tuple, Any] = {}
        # Cases an independent reader has already ruled on, so no case is put
        # to one twice in a pass: the answer would be the same and the turn
        # would not.
        adjudicated: dict[tuple, tuple[str, Any]] = {}
        # One retry of an empty cases turn, per pass. See where it fires.
        bar_retried = False
        # The size probe's generator, filled in by the cases turn, and what it
        # said about each program it was asked about. Keyed by source: asking
        # twice about the SAME program answers the same, and a round that
        # changes the program produces a new key, so the replacement is timed
        # in its turn rather than shipping on the strength of its predecessor's
        # verdict.
        probe: list = []
        probed: dict[str, _Probe] = {}
        # Correction rounds this PASS has sent, never reset -- the index the
        # repair schedule is read at. See `_scheduled_profile`.
        correction_rounds = 0
        # Correction rounds sent into the conversation now in hand. Reset when
        # the repair is carried elsewhere, so each conversation is judged on
        # what it did rather than on what the pass has spent.
        rounds_here = 0
        # Whether the program turn's phase line already went out -- it does,
        # early, when the retry runs between the turn and its grading.
        program_marked = False
        try:
            # BOUNDED by what is left. `BrowserFleet.open` waits for a free tab
            # up to `MINER_TAB_WAIT_S`, which ships at 120s, and nothing here
            # ever passed a smaller number -- so on a busy fleet a solve could
            # spend 120s per pass waiting, three passes, 360s against a 280s
            # deadline, and return empty having sent no prompt at all. Measured:
            # budget 40s, elapsed 50.1s, `open()` called at t=0 and t=25.1,
            # prompts sent 0.
            phases = _Phases(budget, started, pass_no, ident=_ident(task))
            if plan is not None:
                # Per pass, like `exit=`: the summary line describes the pass
                # that shipped, and a second pass runs only after the first
                # delivered nothing. Left accumulating, a solve whose first
                # pass spent three rounds and died reported `rounds=4` for
                # the one round its second pass actually took.
                #
                # Everything here is per pass for that reason, and `probe`
                # most of all. It gates the on-disk cache, so a first pass
                # that timed a program at scale and then failed to deliver
                # would have vouched for the SECOND pass's program -- a
                # different program, never timed -- and put it on disk to be
                # served again unexamined.
                plan.rounds = 0
                plan.corrected = 0
                plan.probe = ""
                plan.adjudicated = {}
                plan.contested = 0
                plan.repair = []
                plan.union = None
                plan.split = 0
                plan.bar_provider = None
                plan.bar2_provider = None
            conversation = await self._open_within(
                budget, started, avoid, phase="program"
            )
            provider = best_provider = getattr(conversation, "provider", None)
            phases.mark(f"open {provider or 'tab'}")
            # Turn 1: the cases, before the program exists. Cases written
            # ALONGSIDE a program can be back-filled from what the program
            # happens to do, and then they agree with its bugs; cases written
            # first cannot. That is the whole argument for spending a round
            # trip here.
            cases: Optional[list] = None
            two_phase = plan is None or plan.two_phase
            if self._self_tests and two_phase and self._independent_bar:
                # A SECOND conversation, on the `cases` phase's model, asked at
                # the same moment the program is. Two things follow, and both
                # are what the archived runs say is missing.
                #
                # Independence. Sequentially in one session the program turn is
                # written with the cases already in its context, by the model
                # that wrote them, from that model's one reading of the
                # statement. A bar the author can see is not a check. Measured
                # across the 97 solves the two archived runs report: 96 shipped
                # a program that passed EVERY one of its own ~18 cases, and 71
                # never had a single disagreement to repair -- against a hidden
                # suite those runs passed 78-83% of the time. About one shipped
                # answer in five clears a bar it wrote for itself and still
                # fails, which is what a bar and a program wrong the same way
                # looks like from outside.
                #
                # Time. Those two turns are 83.8% of all phase time and they
                # run back to back: cases p50 56.9s, program p50 63.5s, so
                # p50 120.4s of a 280s budget spent before the first grade.
                # Side by side that is max() rather than sum: p50 ~63.5s, and
                # at p90 (95.1s and 171.2s) ~171s instead of ~266s. Shipped
                # solves ran p90 274s against a 280s stop; this is where the
                # headroom for the repair rounds comes from.
                bar_started = time.monotonic()
                # The OPEN goes inside the task too, not just the turn. A
                # fleet backend waits for a free tab, and awaiting that here
                # would put the bar's queue on the program's critical path --
                # the one thing the split exists to take it off. `bar` is a
                # one-element list so the cleanup below can close whatever the
                # task got as far as opening.
                bar = []
                bar_task = asyncio.create_task(
                    self._write_the_bar(
                        bar, task, budget, started, avoid, plan,
                        probe=probe if self._size_probe else None,
                    )
                )
                # A SECOND bar, beside the first, on a different model.
                #
                # It is the only mechanism here that reaches the solves where
                # nothing ever disagreed -- 39 of 54 in the last live run, and
                # 71 of 97 before that. A bar and a program written by one
                # model from one reading of the statement share that reading's
                # mistakes, so the bar ratifies the bug instead of catching it,
                # and no amount of repair-loop work helps a round that never
                # fires. A reader that is not the program's author is the only
                # thing that changes which cases get written at all.
                #
                # Free in wall-clock and not free in tokens: it runs beside the
                # program turn like the first bar, so the critical path is
                # still max() rather than sum, and it costs one cases turn's
                # output. `cases2` is pinned to a different model in
                # `cli_phase_profiles` rather than steered with `avoid`,
                # because the first bar's provider is not known yet -- both
                # turns start together.
                if self._second_bar:
                    bar2_started = time.monotonic()
                    bar2 = []
                    bar2_task = asyncio.create_task(
                        self._write_the_bar(
                            bar2, task, budget, started, avoid, plan,
                            phase="cases2", provider_slot="bar2_provider",
                            # The generator too: when the first bar's turn
                            # comes back without one, this is a second chance
                            # at a probe rather than none.
                            probe=probe if self._size_probe else None,
                        )
                    )
                print(
                    f"[verify] the bar is being written in a separate "
                    f"conversation while {provider or 'this one'} writes the "
                    f"program, so neither reading of the statement can see the "
                    f"other"
                    + (
                        ", and a second bar beside it on another model"
                        if bar2_task is not None else ""
                    )
                )
            elif self._self_tests and two_phase:
                asked_at = time.monotonic()
                left_for_cases = budget - (time.monotonic() - started)
                # Everything left. A ceiling here is a second deadline on a
                # solve that has one, and the only thing it can do that the
                # deadline does not is cut a model off mid-answer -- which
                # costs the whole turn, because half a bar is worse than none.
                cases = await self._ask_for_cases(
                    conversation, task, left_for_cases
                )
                # Re-read after EVERY turn, here and below: a backend may move
                # a conversation to another model or seat inside a turn (the
                # CLI ladder does), and `provider` bound at open would then
                # credit the answer to the pair that refused it -- and, passed
                # back as `avoid`, ask "anyone but the refuser" when the ask
                # was "anyone but the one that just answered".
                provider = best_provider = getattr(conversation, "provider", provider)
                phases.mark("1 cases", model_s=time.monotonic() - asked_at)
                if cases is None:
                    # The tab could not be read, or the model was still writing.
                    # Sending turn 2 into it would queue behind an answer that
                    # has not arrived, so this conversation is finished either
                    # way. What differs is whether anything else is worth doing,
                    # and the clock decides it.
                    #
                    # Whether anything ELSE happens is already decided, by the
                    # clock, in `solve_task`: it will not open another pass
                    # once the deadline is gone. That is the whole
                    # guarantee "a turn that ran the deadline out is never
                    # retried on another tab" rests on, and it holds only
                    # because turn 1 now reads against the real budget -- when
                    # it was capped at 60s the budget survived it and four tabs
                    # were spent in a row. Deciding it a second time here would
                    # be a duplicate of that floor, and a duplicate that drifts.
                    #
                    # So the only job left is to say which failure this was,
                    # because "asking another model" was printed even when
                    # nothing else would be asked.
                    left_after = budget - (time.monotonic() - started)
                    # WHOSE failure was it, though. `unreadable` is set by
                    # `_read` only when the tab went blind or the page died,
                    # and it retires that tab at the same moment -- so the next
                    # pass is served by a different one and the reason cannot
                    # follow it there. Every other way this returns None is a
                    # model that had not finished writing, which belongs to the
                    # task and the site and WOULD repeat exactly.
                    tab_side = (
                        getattr(conversation, "empty_reason", None) == "unreadable"
                    )
                    if plan is not None and not tab_side:
                        # Not the same way twice. A long thinking phase, a slow
                        # account, a hard problem: the next pass would repeat
                        # it. It did: four passes, four timed-out cases turns,
                        # nothing submitted. The remaining passes ask for the
                        # program alone, and that task is submitted ungraded.
                        #
                        # Clearing this for a DEAD TAB was the same mistake
                        # pointed the other way. It cost the rest of the solve
                        # the split -- and on live traffic, which ships no
                        # public examples, the model's own cases are the only
                        # grading there is. Turn 2 alone falls back to the
                        # combined prompt, where the cases are written beside
                        # the program and can be back-filled from whatever it
                        # happens to do. One blind tab is not evidence about
                        # the task, and `BLIND_TAB_GRACE_S` bounds what finding
                        # that out costs.
                        plan.two_phase = False
                    if left_after <= 0:
                        print(
                            f"[verify] the cases turn used the whole "
                            f"{budget:.0f}s budget; nothing left to ask anyone "
                            f"else with"
                        )
                    elif tab_side:
                        print(f"[verify] {left_after:.0f}s left; that tab is gone, "
                              f"not the cases turn — the next pass asks another "
                              f"one for cases as usual")
                    else:
                        print(f"[verify] {left_after:.0f}s left; not asking for "
                              f"cases again this task, the remaining attempts "
                              f"go straight to the program")
                    # The cases turn never answered, so no program was asked
                    # for this pass. Unrecorded, `exit=` kept whatever an
                    # earlier pass had left in it -- or nothing.
                    if plan is not None:
                        plan.note_exit("cases")
                    return best, best_provider
            prompt = build_code_prompt(
                task.language, task.statement, task.entrypoint,
                task.public_examples, cases=cases,
            )
            # `cases or []` covers None as well as [], and it matters: the
            # early return above is the only thing that keeps None out of here,
            # so `list(cases)` was one edit away from a TypeError -- which this
            # module CATCHES as a backend failure and reports as "provider=none"
            # with no answer. A crash that looks like a dead tab is the worst
            # kind, so the line does not depend on the guard above surviving.
            agreed = list(cases or [])
            # The program the LAST round produced, so a repair that corrects a
            # case can be told apart from one that rewrites both -- and the
            # reply that carried it, so a correction sent WITHOUT the program
            # can still be graded against something.
            last_code: Optional[str] = None
            last_program_reply: Optional[str] = None
            # The models this pass has already handed the repair to, in
            # order. A rotation rather than a single handoff: see
            # `_resume_elsewhere`. The deadline ends it, not a count.
            rotated: list = []
            # Empty unless the backend can be asked for a NAMED model. A
            # browser fleet cannot -- every tab is the same subscription --
            # so there a handoff buys a fresh conversation and nothing more,
            # and the once-per-pass rule it always had still applies.
            rotation = (
                self._rotation
                if getattr(self._backend, "open_profile", None) is not None
                else ()
            )
            # What each round AMOUNTED to -- the program, the defect and the
            # failing cases -- so a round that changed nothing can be told from
            # one that did. See `duplicate` below.
            #
            # Every round of the pass, not just the one before. Comparing
            # against the previous round only is blind to the shape this loop
            # actually spins in, which alternates: a repair round reports
            # failures F on program P, the next reply comes back as cases the
            # revision guard refuses, that round grades as something else, and
            # the round after is P and F again. No two CONSECUTIVE signatures
            # ever matched, so the guard never fired once -- measured, fifty-
            # nine sends inside a single solve.
            seen_signatures: dict[tuple, int] = {}
            # How many times each repair report has already gone out, so a
            # prompt is never sent byte-identical twice without saying so. See
            # `stalled` where the next prompt is built.
            reports_sent: dict[str, int] = {}
            # What grading established about each program this pass has seen,
            # by its source. See `_inherit_evidence` for what it is for.
            judged: dict[str, Candidate] = {}
            # The cases the LAST round failed -- the ones the repair prompt
            # just quoted, and so the only ones the reply to it is entitled to
            # change. See `_merge_cases`.
            reported_failed: list[dict] = []
            # Consecutive rounds that left the program exactly as it was. See
            # `CASES_ONLY_ROUNDS`.
            program_unchanged = 0
            # Replies refused whole for rewriting too much of the bar at once.
            # See `BULK_CORRECTION_SHARE`.
            bulk_refusals = 0
            # Whether the prompt just sent WITHDREW the offer to correct a
            # case. A withdrawal the reply can ignore is not one.
            program_only = False
            def _schedule_moves() -> bool:
                """Whether the NEXT correction round belongs somewhere else.

                With a schedule this is not a budget question at all. The old
                test asked whether this conversation had had its allotted
                rounds; the schedule already says who answers round N, so the
                only question left is whether that is who is answering now. Two
                consecutive `opus` entries mean no handoff and the conversation
                keeps its context, which is the whole reason the test is on the
                MODEL rather than on a count.

                A backend with no rotation to move along -- a browser fleet,
                where every tab is the same model -- keeps the round budgets it
                always had, because there a handoff buys a fresh conversation
                and nothing else.
                """
                if not rotation:
                    return rounds_here >= (
                        HANDOFF_ROUNDS if rotated else FIRST_PHASE_ROUNDS
                    )
                want = _scheduled_profile(rotation, correction_rounds + 1)
                return bool(
                    want
                    and want.model
                    and want.model.lower() not in (provider or "").lower()
                )

            async def _resume_elsewhere(
                why: str, avoid: Optional[str] = None, slow: Optional[str] = None,
            ):
                """Carry the repair to a FRESH conversation, or None.

                `slow` is the size probe's sentence when the program in hand
                is correct on every case and merely too slow at scale. That is
                a fault worth carrying -- the validator fails every large
                hidden test on it -- and it used to be refused here, because
                the guard below asked for failures or a defect and a slow
                program has neither. So the too_slow handoff was dead code:
                the loop printed "nobody else to ask" with two models on the
                rotation never asked, and shipped a program known to time out.

                Declines say why. Every caller used to word its exit line from
                its own guess -- "every model has had it", "nobody else to
                ask" -- and each guess was wrong on some path.

                The one move available when a conversation will not produce a
                new answer -- because it is unreadable, or because it just
                repeated itself. Both are the same situation from here: nothing
                more is coming from this tab, and the repair is still worth
                making somewhere else.

                `avoid` is what separates the two. An unreadable tab is a TAB
                problem -- the model is fine and any tab will do, so it passes
                none. A conversation that repeated itself is a MODEL problem,
                and the answer to that is the other model: it arrives holding
                the previous program and the cases it failed, which is a far
                better start than the fresh pass `solve_task` would give it.
                On live traffic that pass does not happen at all -- with no
                public examples nothing can be graded, and `solve_task` breaks
                rather than spend a second account on an answer it cannot
                compare. So this is the only place the other model gets asked.

                Once per pass. A second tab that also fails is a fleet problem
                rather than something to keep paying for, and the caller then
                ends the loop holding the best answer it ever had.

                Returns `(conversation, provider, prompt)` for the caller to
                install, so the loop's own bindings stay the single source of
                truth for what it is talking to.
                """
                nonlocal program_only, program_unchanged, reported_failed
                nonlocal rounds_here
                left_now = budget - (time.monotonic() - started)
                if best is None or not best.code.strip():
                    print("[verify] the repair is not carried on: no program "
                          "in hand to carry")
                    return None
                if not (best.failures or best.defect or slow):
                    print("[verify] the repair is not carried on: the program "
                          "in hand has no fault to report")
                    return None
                if left_now < MIN_SLICE_S:
                    # The only clock. There was a 40s floor here on the theory
                    # that a fresh conversation needs that long to be worth
                    # opening; it stopped the rotation with time on the
                    # validator's clock, which is the one thing the rotation
                    # must not do. A handoff opened with seconds left is cut by
                    # the deadline like any other read; one that could not be
                    # handed the minimum slice without padding past the
                    # deadline is not opened.
                    print("[verify] the repair is not carried on: the deadline "
                          "is gone")
                    return None
                # ONCE per model, not once per pass. The old rule stopped after
                # a single handoff on the theory that a second tab failing is a
                # fleet problem rather than something to keep paying for -- true
                # of a DEAD TAB, and not true of the case this is now mostly
                # used for. A repair that stays where the program was written
                # asks the model to find a fault in its own reading of the
                # statement, and measured over 54 live solves it does not:
                # nine of ten single-case disagreements ended with the model
                # editing its own case and keeping its program. The answer to
                # that is another reading, and then another, for as long as the
                # deadline allows -- which is what bounds this now.
                if rotation:
                    # WHERE THE SCHEDULE SAYS, for the round about to be sent.
                    # `correction_rounds` counts the rounds already sent, so
                    # the next is `correction_rounds + 1`. It is a POSITION and
                    # not a search, so it never runs out and there is no lap to
                    # announce: the correction runs until the program passes or
                    # the deadline stops it, which is the operator's rule.
                    nxt = _scheduled_profile(rotation, correction_rounds + 1)
                    # NOT recorded in `rotated`. That list answers "has this
                    # pass already been carried once", which is a question only
                    # the no-rotation branch below asks; a schedule has no such
                    # state, and a list that looks like it tracks the rotation
                    # while nothing consults it is worse than no list.
                else:
                    # No models to rotate through -- a browser fleet, where
                    # every tab is the same model and a handoff buys a fresh
                    # conversation and nothing else. There the original rule
                    # still holds: a second tab that also fails is a fleet
                    # problem rather than something to keep paying for.
                    if rotated:
                        print("[verify] the repair is not carried on: it has "
                              "already been carried once and there is no "
                              "other model to carry it to")
                        return None
                    nxt = None
                    rotated.append(None)
                # A fresh conversation is a fresh start, and three pieces of
                # state are about the OLD one.
                #
                # `program_only` most of all: the resume prompt offers the case
                # correction by name, and leaving the withdrawal set meant the
                # loop asked the second model for a corrected case and then
                # threw the answer away when it arrived. The one path that ever
                # reaches a second model, spending its correction on a rule the
                # prompt it answered never stated.
                #
                # `reported_failed` because the resume prompt quotes `best`'s
                # failures, so `best`'s cases are the ones in play -- not
                # whatever the last round happened to leave behind.
                program_only = False
                program_unchanged = 0
                rounds_here = 0
                reported_failed = list(best.failed_cases)
                print(
                    f"[verify] {why}; carrying the repair to a fresh "
                    f"conversation with {left_now:.0f}s left"
                )
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - it may already be broken
                    pass
                # `phase="repair"` -- which matters on exactly one of the two
                # call sites. A resume that names an `avoid` is fleeing a model
                # that just got this wrong, and `open_for` lets `avoid` beat
                # the preference for that reason; a resume with none is fleeing
                # a DEAD TAB, where the model was never the problem and the
                # operator's choice for repairs is the right seat to land on.
                fresh = await self._open_within(
                    budget, started, avoid, phase="repair", profile=nxt,
                )
                landed = getattr(fresh, "provider", None)
                # Did the MODEL actually change? A browser fleet has none to
                # change, and `open_profile` falls through to the ladder when
                # the one asked for is out everywhere -- so a handoff can land
                # right back on the model that wrote the program. Telling that
                # model it is looking at someone else's work is the same false
                # premise the flag exists to remove, pointed the other way.
                elsewhere = bool(
                    landed and provider
                    and _model_of(landed) != _model_of(provider)
                )
                phases.mark(f"open {landed or 'tab'}")
                if plan is not None and landed:
                    plan.repair.append(landed)
                return (
                    fresh,
                    landed or provider,
                    build_resume_prompt(
                        task.language, task.statement, task.entrypoint,
                        task.public_examples, agreed, best.code,
                        best.failures, defect=best.defect,
                        from_self_tests=best.from_self_tests,
                        bar_is_independent=self._independent_bar,
                        failed_cases=best.failed_cases,
                        # This model did not write the program it is being
                        # shown. Saying otherwise is a false premise about the
                        # one thing the round turns on, and a false premise
                        # gets argued with rather than acted on.
                        foreign=elsewhere,
                        # ALL of them, not any: the prompt's confirmation
                        # withdraws the case offer for every failure it lists,
                        # so it may only be made when every listed case was
                        # confirmed. One confirmed and one merely failing used
                        # to lock the model out of correcting the second.
                        case_confirmed=bool(best.failed_cases) and all(
                            adjudicated.get(_case_key(case), ("", None))[0]
                            == "case"
                            for case in best.failed_cases
                        ),
                        too_slow=slow,
                    ),
                )

            bar2_joined = False

            async def join_bar2() -> bool:
                """The second bar, joined to the first IF it has landed.

                Never waits, and is asked often: when the first bar is
                collected, before every grade, and once more before a program
                is called converged. That is what makes it incremental. The
                second bar is an addition to a bar that already exists, so
                nothing is ever held for it -- but it was once CANCELLED the
                first time it was not ready, at the first join, and that cut
                the stage on the program turn's clock rather than the
                deadline: whenever the second bar's cases turn ran longer
                than the program turn, the one mechanism aimed at solves where bar
                and program agree was thrown away with most of the budget
                left. Now a bar that lands during round three is graded
                against from round three.

                True when cases were added, so a caller about to ship can
                re-grade against them first.
                """
                nonlocal agreed, split_keys, bar2_task, bar2, bar2_joined
                if bar2_task is None or not bar2_task.done():
                    return False
                extra = await self._collect_bar(
                    bar2_task, bar2, phases, budget, started, bar2_started,
                    label="1 bar2", who="the second bar", quiet=True,
                )
                bar2_task, bar2 = None, []
                bar2_joined = True
                if not extra:
                    return False
                before = len(agreed)
                agreed, split_keys = _union_bars(agreed, extra, task.language)
                if plan is not None:
                    plan.union = (before, len(extra), len(agreed))
                    plan.split = len(split_keys)
                print(
                    f"[verify] the two bars union to {len(agreed)} case(s) "
                    f"({before} + {len(extra)}, {len(split_keys)} of them the "
                    f"same call read two ways)"
                )
                return len(agreed) > before

            def _too_slow_report(slow: str) -> str:
                """The too_slow repair prompt, with a repeat named as one.

                Through the same `reports_sent` bookkeeping as every other
                report. With the verdict cached per program, a model that
                re-sends the same slow program used to receive the identical
                prompt word for word every round -- the exact pattern the
                stalled mechanism exists to break.
                """
                report = build_repair_prompt(
                    [], task.language, task.entrypoint, too_slow=slow
                )
                stalled = reports_sent.get(report, 0)
                reports_sent[report] = stalled + 1
                if stalled:
                    print(
                        f"[verify] this same too_slow report has now gone out "
                        f"{stalled + 1} times; naming the repetition in it"
                    )
                    report = build_repair_prompt(
                        [], task.language, task.entrypoint, too_slow=slow,
                        stalled=stalled,
                    )
                return report

            attempt = 0
            while True:
                attempt += 1
                left = budget - (time.monotonic() - started)
                if attempt > 1 and left < MIN_SLICE_S:
                    # The deadline, and only the deadline. This used to be a
                    # round-trip floor (12s; 20s on the CLI backend) on the
                    # theory that a round started under it cannot finish --
                    # and it stopped the loop with a wrong program in hand and
                    # time on the validator's clock, which is the one thing the
                    # correction must not do (operator's rule). A round started
                    # with seconds left is cut by the same deadline as every
                    # other read, and `_supersedes` keeps a fragment from
                    # displacing the answer in hand. `MIN_SLICE_S` is not a
                    # budget: it is the least slice `send` can be handed, and
                    # a round that would have to be padded past the deadline
                    # to be sent at all is one the deadline has ended. The
                    # first attempt always runs, however little there is,
                    # exactly as the first pass does in `solve_task`.
                    #
                    # And this branch itself used to be the silent one. The
                    # deadline is the ordinary way a correction loop ends -- it
                    # runs until the answer passes or the clock stops it -- so
                    # it is the last thing that should happen without a word.
                    print(
                        "[verify] the deadline is gone; submitting the last version"
                        + (
                            " unverified"
                            if best is None or not best.verified
                            else ""
                        )
                    )
                    if plan is not None:
                        plan.note_exit("budget")
                    break
                if self._max_attempts and attempt > self._max_attempts:
                    print(
                        f"[verify] SOLVER_MAX_ATTEMPTS={self._max_attempts} "
                        f"reached; submitting the last version"
                        + (
                            " unverified"
                            if best is None or not best.verified
                            else ""
                        )
                    )
                    if plan is not None:
                        plan.note_exit("maxattempts")
                    break
                # Every round reads against EVERYTHING that is left. Earlier
                # builds handed the first attempt a fraction (60% with public
                # examples, 85% without) so a repair would have something to
                # spend, and that reserve was worth least exactly when it cost
                # most: `send` returns the moment the model finishes, so the
                # slice was never a wait -- only a ceiling on a read that ran
                # long, which is the one case where cutting it short throws away
                # the answer. The loop below stops when a round trip no longer
                # fits; nothing is carved out in advance.
                # No `extend_to_s`, for the same reason the cases turn passes
                # none: the slice already IS everything left, so there is
                # nothing being held back to extend into. Passing one equal to
                # the slice makes `send`'s extension a no-op by construction and
                # only reads as though a reserve existed.
                # No `extend_to_s`: `left` already runs to the point the
                # answer stops being deliverable, so there is nothing past it to
                # extend into.
                # Counted HERE, where a prompt is about to go out -- not at the
                # top of the loop, which counts entries. The budget and
                # max-attempts breaks sit between the two and send nothing, so
                # every solve that ended either way reported one round more
                # than it asked for. Before the send rather than after it: a
                # round whose send raises or is cut off was still asked, and
                # `exit=cutoff` should not also lose its round.
                if plan is not None:
                    plan.rounds += 1
                if attempt > 1:
                    # CORRECTION rounds only. Attempt 1 is the program itself,
                    # and counting it would spend a third of this
                    # conversation's budget before a single failure existed.
                    rounds_here += 1
                    # ...and the PASS's own count, which is what indexes the
                    # schedule. `rounds_here` restarts at every handoff, so it
                    # cannot say which round of the solve this is; the schedule
                    # has to be positioned against a number that only ever goes
                    # up. Counted here, before the send, so the round about to
                    # go out is `correction_rounds` and the next is one more.
                    correction_rounds += 1
                round_started = time.monotonic()
                correction_refused = False
                # Empty-handed: nothing finished is in hand to ship, so a cut
                # here ships a fragment or nothing, and both score zero. The
                # delivery reserve is worth more spent finishing this reply
                # than protecting the delivery of that -- see `WIRE_TAIL_S`.
                # With a program already in hand the budget stands: there the
                # reserve is protecting an answer that can actually be paid
                # for, and `_supersedes` already refuses to let a fragment
                # displace it.
                spare = (
                    budget + self._reserve - WIRE_TAIL_S
                    - (time.monotonic() - started)
                )
                reply = await self._send_within(
                    conversation, prompt, max(1.0, left),
                    extend_to_s=(
                        spare
                        if (best is None or not best.code.strip())
                        and spare > left
                        else None
                    ),
                )
                provider = getattr(conversation, "provider", provider)
                # How long the round trip actually took. Used only by the
                # duplicate branch below, and only to tell a model from a tab.
                # Read BEFORE the bar is collected: a round that came back in
                # no time is how a replayed reply is caught, and charging it
                # for a bar that was still being written elsewhere would hide
                # exactly the case it exists to find.
                round_s = time.monotonic() - round_started
                if bar_task is not None:
                    # The bar, now that the program it grades exists. Awaited
                    # HERE rather than before the send, because awaiting it
                    # before the send is the sequential shape this replaced;
                    # and here rather than at first use, because `revised`
                    # below already reasons about what the bar holds.
                    agreed = await self._collect_bar(
                        bar_task, bar, phases, budget, started, bar_started
                    )
                    bar_task, bar = None, []
                    # The second reader, joined to the first if it has ALREADY
                    # finished -- and asked again before every grade if it has
                    # not. It is never waited for: bar1 has landed by now, so a
                    # hung second model would otherwise hold the whole
                    # remaining deadline and the answer would ship UNGRADED
                    # with a perfectly good bar in hand.
                    await join_bar2()
                    if not agreed and not bar_retried:
                        # A bar that came back with nothing leaves the solve
                        # with NOTHING TO GRADE: `self_total` is 0, there are
                        # no failures, the loop breaks on the next line and the
                        # answer ships ungraded. Measured over the 102 archived
                        # solves, that is 5 of them -- phase 2 and phase 3 both
                        # inert -- while the median solve hands back 134s of
                        # its 290s unspent. One more ask is the cheapest thing
                        # that budget can buy.
                        #
                        # ONCE, and only while the plan still believes a cases
                        # turn is affordable here. Whatever made the first one
                        # come back empty -- a refusal, a reply that carried no
                        # JSON, a turn cut off -- belongs to the task and the
                        # site as often as not, and `_Plan.two_phase` is where
                        # that lesson already lives.
                        bar_retried = True
                        left_now = budget - (time.monotonic() - started)
                        if (
                            self._self_tests
                            and (plan is None or plan.two_phase)
                            and left_now >= MIN_SLICE_S
                        ):
                            # The program's turn is already over, and the
                            # retry is about to run AFTER it. Mark the turn
                            # now, at the instant it ended, so the retry is
                            # measured from there and not from the open --
                            # and so the turn's own seconds are not handed to
                            # the retry. Its grading has not happened yet;
                            # that gets its own line below.
                            phases.mark(
                                "2 program", model_s=round_s,
                                ended=round_started + round_s,
                            )
                            program_marked = True
                            print(
                                f"[verify] the cases turn came back with "
                                f"nothing, so there is no bar to grade against "
                                f"and the repair loop has nothing to work on; "
                                f"asking once more with {left_now:.0f}s left"
                            )
                            bar, bar_started = [], time.monotonic()
                            retry = asyncio.create_task(
                                self._write_the_bar(
                                    bar, task, budget, started, avoid, plan
                                )
                            )
                            agreed = await self._collect_bar(
                                retry, bar, phases, budget, started,
                                bar_started, beside=False,
                                label="1 cases again",
                            )
                            bar = []
                            if not agreed and plan is not None:
                                # Twice is evidence about the task, not the
                                # tab. The remaining passes go straight to the
                                # program rather than paying for a third.
                                plan.two_phase = False
                # A repair reply may carry a CORRECTED case array: the repair
                # prompt offers it outright ("or, if the case was wrong rather
                # than the program, a `json` array holding ALL of the cases").
                # Freezing turn 1's cases would kill that escape hatch and let
                # one wrong case break a correct program on every round.
                #
                # WHEN it takes effect differs by the shape the repair came
                # back in, and both shapes matter.
                #
                # Program UNCHANGED, cases corrected -- exactly what was asked
                # for. Applied to this same reply, because the alternative is to
                # report the identical failure it was sent to fix: the round is
                # spent, the next prompt quotes the same disagreement, and the
                # correction lands only on the round after -- one more round
                # trip spent re-reporting a failure already fixed, against a
                # deadline. Measured on a live
                # solve: turn 1 wrote three cases whose `final_records` order
                # was wrong, the program was right, the model corrected the
                # cases exactly as asked, and the answer still went out
                # reported 17/20. Nothing is conceded by grading it now: the
                # program is the one already judged, so a weakened case cannot
                # launder a rewrite that did not happen.
                #
                # Program CHANGED as well -- the reply rewrote both sides of the
                # disagreement. The prompt no longer spends a sentence
                # forbidding that, because forbidding it was never what stopped
                # it: this is. Such a reply is graded against the bar as it
                # stood BEFORE it arrived, so a model cannot make a rewritten
                # program pass by rewriting the bar in the same breath. Its
                # cases apply from the next round.
                revised = extract_self_tests(reply, task.entrypoint, task.language)
                if revised and attempt == 1:
                    # The PROGRAM turn. Its cases are back-filled from what the
                    # program happens to do -- they agree with its bugs, which
                    # is the entire argument for splitting the turns -- so the
                    # bar stays the one turn 1 wrote before any program existed.
                    #
                    # Not a theoretical objection. Measured: turn 1 wrote a case
                    # that CAUGHT the bug, turn 2 sent the buggy program with
                    # two cases of its own, round 1 reported the real failure
                    # and then adopted them, and round 2 re-graded the same
                    # buggy program against the bar it had brought with it --
                    # `self=2/2`, no failures, loop over, buggy program
                    # submitted as passing everything. The rule that a reply is
                    # judged against the bar as it stood before it arrived
                    # covered repair rounds and left this one round short.
                    print(
                        f"[verify] the program turn sent {len(revised)} case(s) "
                        f"of its own; keeping turn 1's — cases written beside a "
                        f"program are back-filled from it"
                    )
                    revised = []
                if revised and program_only:
                    # The last prompt stopped offering the case and asked for
                    # the program, because two rounds running had corrected the
                    # bar and left the program alone. A reply that sends cases
                    # anyway is that same round again, and accepting it would
                    # make the withdrawal a sentence rather than a rule.
                    print(
                        "[verify] the cases came back again after the prompt "
                        "stopped offering them; keeping the bar as it stands — "
                        "this round was asked for the program"
                    )
                    revised = []
                if revised:
                    # MERGED into the agreed suite, not swapped for it, and the
                    # cases the last round FAILED are the only ones a correction
                    # may touch. That rule is what makes a short array safe to
                    # accept, and accepting one matters: the repair prompt asks
                    # about one disagreement, so the natural reply is that one
                    # case corrected. Demanding the whole array back re-sent
                    # twenty cases to fix one of them -- slower, likelier to be
                    # truncated mid-array, and refused outright whenever it came
                    # back one short, which left the wrong case breaking a
                    # correct program on every remaining round of the solve.
                    #
                    # What the old refusal was protecting is protected here by
                    # construction rather than by suspicion: a case the program
                    # PASSES cannot be corrected, dropped or weakened, so the
                    # way to game this -- delete the case you cannot pass -- is
                    # not reachable. See `_merge_cases`.
                    # A confirmed case is not the model's to correct.
                    # A reply that rewrites a THIRD of the bar is not
                    # correcting a case, it is re-specifying the problem.
                    #
                    # Kept after the judge was removed, and on narrower grounds
                    # than it was built on. The judge upheld the model's rewrite
                    # in 22 of the 25 corrections it ever checked and sided with
                    # the original case in none of them, so "the model launders
                    # its own bar" is not what the logs show and is not why this
                    # is here. What the logs do NOT contain is a single judged
                    # BULK rewrite -- the one that replaced fifteen of twenty-two
                    # cases with nine seconds left was never checked by anything.
                    # The cap costs one correction in thirteen and covers the
                    # case no evidence speaks to.
                    # How many cases actually CHANGED, not how many were sent.
                    # The repair prompt asks for "a json array holding ALL of
                    # the cases, corrected", so a well-behaved reply carries the
                    # whole bar every time; counting the array's length refused
                    # every correct correction and left the bar frozen.
                    held = {_case_key(c): _expectation(c) for c in agreed}
                    moved = sum(1 for c in revised
                                if held.get(_case_key(c)) != _expectation(c))
                    allowed = max(2, len(agreed) // BULK_CORRECTION_SHARE)
                    if moved > allowed:
                        bulk_refusals += 1
                        # A refused correction is not the model repeating
                        # itself about the PROGRAM -- the program was never
                        # asked for -- and the duplicate guard below must not
                        # read it as one. Both setters of this flag went out
                        # with the judge in d6c9fc6; this is the one that is
                        # still needed.
                        correction_refused = True
                        print(
                            f"[verify] {moved} of the {len(agreed)} case(s) "
                            f"on the bar came back rewritten in one reply, which is "
                            f"a re-specification rather than a correction; the bar "
                            f"stands and the program is re-graded against it"
                        )
                        merged, changed = list(agreed), ""
                    else:
                        merged, changed = _merge_cases(agreed, revised, reported_failed)
                    if changed.startswith("REFUSED"):
                        print(f"[verify] {changed[len('REFUSED:'):].strip()}")
                        revised = []
                    elif changed.startswith("NOTHING"):
                        print(f"[verify] {changed[len('NOTHING:'):].strip()}")
                        revised = merged
                    else:
                        if changed:
                            print(
                                f"[verify] the repair corrected the cases rather "
                                f"than the program: {changed}. The program is "
                                f"re-graded against the {len(merged)} that now "
                                f"stand; if it still disagrees the loop keeps "
                                f"going."
                            )
                            # What the summary line's `corrected=` counts. The
                            # increment went out with the judge in d6c9fc6 and
                            # the line has printed 0 ever since; `moved` is the
                            # number of cases whose expectation this reply
                            # actually changed, counted above for the bulk cap.
                            if plan is not None:
                                plan.corrected += moved
                        # Kept even when the merge changed nothing, because
                        # `revised` is not only the new bar -- it is what tells
                        # the branches below that this reply carried CASES. A
                        # reply that sends the suite back untouched and no
                        # program is still asking for the program in hand to be
                        # re-graded, and zeroing it here made that reply read as
                        # "nothing reached me as code" and spend a round trip
                        # saying so.
                        revised = merged
                # Whether this reply carried CASES, recorded before the
                # branches below consume `revised` -- they set it to None once
                # it has been applied, and reading it afterwards to decide
                # whether the round produced anything said "nothing arrived"
                # about the one reply the repair prompt asks for by name.
                carried_cases = bool(revised)
                now_code = extract_code(reply, task.entrypoint, task.language).strip()
                # Which reply the candidate is actually graded FROM. Normally
                # this one; see the cases-only branch below for when it is not.
                graded = reply
                if revised and last_code:
                    if now_code == last_code:
                        agreed, revised = revised, None
                    elif not now_code:
                        # Cases corrected, program deliberately NOT resent --
                        # the one reply the repair prompt asks for by name when
                        # the case was the thing that was wrong, and until this
                        # branch existed the answer to it was "your previous
                        # reply did not reach me as code". Measured: the program
                        # was right, turn 1's case was not, the model corrected
                        # exactly the case it was asked to, and the miner spent
                        # the rest of the budget demanding a program it already
                        # had before submitting one reported 0/1 on a bogus bar.
                        #
                        # Nothing here is taken on trust. The program is the one
                        # already in hand and already judged, so a weakened bar
                        # cannot launder a rewrite -- there was no rewrite. It
                        # is re-graded, not assumed to pass.
                        agreed, revised = revised, None
                        graded = last_program_reply or reply
                # A second bar that landed during this round's turn is graded
                # against from this round on.
                await join_bar2()
                graded_at = time.monotonic()
                candidate = await self._graded(
                    graded, task, budget - (time.monotonic() - started), agreed,
                    # The round ABOVE this one, still un-updated here -- see the
                    # `last_code = now_code` below, which runs after this. That
                    # is what a reply has to be complete with respect to: a
                    # round that sends back only the function it fixed is using
                    # helpers that exist in the reply above it and nowhere in
                    # the file that would be submitted.
                    previous=last_code or "",
                )
                # Numbered as the prompts are: turn 1 asked for the cases, so
                # the program is phase 2 and the first correction is phase 3.
                # The operator's own vocabulary all through this file's history
                # -- "3rd phase output should be full code" -- and a log that
                # numbered them differently would be answering a question
                # nobody asked in words nobody used.
                if program_marked:
                    # The turn went out above, before the retried bar; what is
                    # left of this phase is the grading alone.
                    phases.mark("2 graded", checked_s=time.monotonic() - graded_at)
                    program_marked = False
                else:
                    phases.mark(
                        "2 program" if attempt == 1 else f"{attempt + 1} correction",
                        model_s=round_s,
                        checked_s=time.monotonic() - graded_at,
                    )
                # Before anything reads it: a round that could not be graded,
                # or one graded against itself, must not report less about a
                # program than an earlier round already established.
                # The trigger rate, taken at the FIRST grade that actually
                # ran the program against cases -- before any repair moved
                # either side. `self_total` is zero when the round produced a
                # defect or there were no cases, and neither of those is a
                # disagreement, so both correctly leave this unset.
                if (
                    plan is not None
                    and plan.disagreed is None
                    and candidate.self_total
                ):
                    plan.disagreed = (
                        candidate.self_total - candidate.self_passed,
                        candidate.self_total,
                    )
                # The model's own case correction, applied now that the
                # candidate has been graded against the bar as it stood when
                # the reply arrived. It used to be applied at the END of the
                # round, and that was where the judge's work went to die: the
                # judge writes into `agreed`, and a reply carrying BOTH a
                # corrected case and a changed program left `revised` pending,
                # so `agreed = revised` restored the pre-judge bar a few lines
                # later. The dropped case came back, and it could never be
                # re-judged -- its key is already in `adjudicated` -- so every
                # remaining round re-reported a disagreement an independent
                # reader had already said the statement does not decide.
                if revised:
                    agreed, revised = revised, None
                # A DISPUTED case, put to a reader with no stake in either
                # side, before the repair prompt asks the program's own author
                # to rule on its own reading. Here rather than beside the
                # prompt because a "the case was wrong" verdict is applied and
                # re-graded WITHOUT a model turn, and everything below --
                # `judged`, `best`, `program_unchanged` -- has to see the
                # candidate that results rather than the one that arrived.
                #
                # Bounded to a couple of cases: a program failing half its bar
                # is not a disagreement about a reading, it is a wrong program,
                # and the round is better spent on it.
                if (
                    self._judge
                    and candidate.from_self_tests
                    and not candidate.defect
                    and 1 <= len(candidate.failed_cases) <= MAX_ADJUDICATED
                    and candidate.failed_actuals
                    and any(
                        _case_key(case) not in adjudicated
                        for case in candidate.failed_cases
                    )
                ):
                    # Only the ones nobody has ruled on. Asked again, a case
                    # gets the same answer for a second turn's money, and
                    # `note_adjudicated` would count it twice -- the archived
                    # breakdown of how disputes were settled is the whole
                    # reason the counters exist.
                    # ...and only the ones the program ANSWERED. A case the
                    # program crashed or timed out on is not a disagreement
                    # between two readings, it is a program that produced no
                    # value, and it is wrong under both. Put to the judge it
                    # could be ruled `contested` and dropped, shipping a
                    # program that crashes on the input.
                    fresh = [
                        (case, actual)
                        for case, actual in zip(
                            candidate.failed_cases, candidate.failed_actuals
                        )
                        if _case_key(case) not in adjudicated and actual.ok
                    ]
                    # A call the two BARS split on already has its second
                    # reading, and the judge -- pinned to the second bar's
                    # model -- would not be a third. The program's value
                    # decides: agreeing with the second bar is two readings
                    # against one, so the case was wrong; agreeing with
                    # neither is the statement being ambiguous there, and the
                    # case is dropped. This routing was designed and then
                    # never wired in: `split_keys` only kept the case away
                    # from the judge, and the case fell to the ordinary repair
                    # prompt -- the program's own author, with the case-edit
                    # offer, deciding a call two other readers disagreed on.
                    verdicts: dict[tuple, tuple[str, Any]] = {}
                    for case, actual in fresh:
                        key = _case_key(case)
                        if key in split_keys:
                            other = split_keys[key]
                            verdicts[key] = (
                                ("program", other)
                                if _same_expected(actual.value, other, task.language)
                                else ("contested", other)
                            )
                    unsettled = [
                        (c, a) for c, a in fresh if _case_key(c) not in verdicts
                    ]
                    if unsettled:
                        verdicts.update(await self._adjudicate(
                            task, [c for c, _ in unsettled],
                            [a for _, a in unsettled],
                            provider, budget, started, split_keys,
                        ))
                    adjudicated.update(verdicts)
                    corrections = [
                        dict(case, expected=verdicts[_case_key(case)][1])
                        for case in candidate.failed_cases
                        if verdicts.get(_case_key(case), ("", None))[0]
                        == "program"
                    ]
                    dropped = [
                        case for case in candidate.failed_cases
                        if verdicts.get(_case_key(case), ("", None))[0]
                        == "contested"
                    ]
                    if plan is not None:
                        for route, _ in verdicts.values():
                            plan.note_adjudicated(route)
                    if plan is not None:
                        plan.contested += len(dropped)
                    if corrections or dropped:
                        if corrections:
                            agreed, _ = _merge_cases(
                                agreed, corrections, candidate.failed_cases
                            )
                            if plan is not None:
                                plan.corrected += len(corrections)
                        if dropped:
                            drop = {_case_key(case) for case in dropped}
                            agreed = [
                                case for case in agreed
                                if _case_key(case) not in drop
                            ]
                        print(
                            f"[verify] an independent reading settled "
                            f"{len(corrections) + len(dropped)} disputed "
                            f"case(s): {len(corrections)} where the case was "
                            f"wrong and {len(dropped)} the statement does not "
                            f"decide; re-grading the same program against them"
                        )
                        # Re-graded, not re-asked. The program did not change,
                        # so nothing here spends a turn: it may now pass a bar
                        # that no longer holds a case it was right to fail.
                        #
                        # `graded`, not `reply`, and the difference is a lost
                        # answer. On a cases-only correction round `reply` is
                        # a JSON array -- the model was asked for the case
                        # alone and sent exactly that -- so re-grading from it
                        # extracts no code, and a candidate that was sound a
                        # line ago becomes a defect. `graded` is the reply the
                        # program actually came from, which is what the round
                        # above already used for the same reason.
                        # `_graded`, not `_grade`: off the event loop, and
                        # with the fallback candidate its handler builds.
                        # Grading is a `subprocess.run` of seconds -- a
                        # container start, or a rustc build for Rust -- and
                        # now that Docker is the default backend, running it
                        # straight from this coroutine would stall every other
                        # solve in flight, including the `wait_for` that
                        # decides whether they are paid.
                        candidate = await self._graded(
                            graded, task,
                            budget - (time.monotonic() - started), agreed,
                            # `or ""`: on the first round there is no previous
                            # program, and the comparison wants a string.
                            previous=last_code or "",
                        )
                key = candidate.code.strip()
                if key:
                    prior = judged.get(key)
                    if prior is not None:
                        _inherit_evidence(candidate, prior)
                    judged[key] = candidate
                # What this round AMOUNTED to. Compared against the rounds
                # before rather than the replies themselves, because the same
                # program under a different sentence of prose is the same
                # program: byte equality misses that and this does not. The
                # failures are in it so a corrected CASE reads as progress even
                # when the program is untouched -- which is exactly what the
                # repair prompt asks for.
                signature = (
                    candidate.code.strip(), candidate.defect, tuple(candidate.failures)
                )
                repeats = seen_signatures.get(signature, 0)
                # A round whose correction the judge refused is not the model
                # repeating itself about the PROGRAM: the program was never
                # asked for. It is neither counted nor held against it.
                if not correction_refused:
                    seen_signatures[signature] = repeats + 1
                duplicate = attempt > 1 and repeats > 0 and not correction_refused
                # Only when one ARRIVED. A cases-only reply leaves the program
                # in hand standing, and forgetting it here would make the very
                # next correction unattributable to any program at all.
                # A round that CHANGED the program resets it; a round that
                # left it as it was counts. A round where nothing arrived at
                # all does neither -- an unreadable tab or a dead read is not
                # the model refusing to touch its program, and counting it
                # withdrew the case offer over rounds in which the model was
                # never heard from. The offer exists for a real reason and is
                # taken away on evidence, not on silence.
                if now_code or carried_cases:
                    program_unchanged = (
                        0 if now_code and now_code != last_code
                        else program_unchanged + 1
                    )
                if now_code:
                    last_code = now_code
                    last_program_reply = reply
                if best is None or _supersedes(
                    candidate, best,
                    getattr(conversation, "still_writing", False),
                ):
                    if plan is not None and (
                        best is None
                        or best.code.strip() != candidate.code.strip()
                    ):
                        # A different program is now the one in hand, and the
                        # probe verdict describes the one that shipped: the
                        # predecessor's `too_slow` must not be reported for a
                        # program that was never timed. What is known about
                        # THIS program, if it was timed before, is in `probed`.
                        known = probed.get(candidate.code.strip())
                        plan.note_probe(known.state if known is not None else "")
                    best, best_provider = candidate, provider
                # A second bar that landed since the grade above: joined, and
                # the program re-graded against the union before anything is
                # called converged. `graded` is the reply the program came
                # from; `_graded` is off the loop, as every grade is.
                if (
                    not candidate.defect and not candidate.failures
                    and best is candidate and await join_bar2()
                ):
                    candidate = await self._graded(
                        graded, task, budget - (time.monotonic() - started),
                        agreed, previous=last_code or "",
                    )
                    judged[candidate.code.strip()] = candidate
                    best, best_provider = candidate, provider
                # `best is candidate`: a fragment the model was still writing
                # that happens to pass is not the answer -- `_supersedes` just
                # refused it -- and probing it, then recording `exit=verified`
                # for it, described a program that did not ship.
                if candidate.verified and not candidate.failures and best is candidate:
                    slow, probe_state = await self._probe_now(
                        candidate, probe, probed, task, budget, started
                    )
                    if plan is not None:
                        plan.note_probe(probe_state)
                    if slow is not None:
                        # A too_slow round is a correction round like any
                        # other, so it faces the handoff. It did not: this
                        # `continue` jumps over the `rounds_here` check below,
                        # and with the verdict now cached per program a model
                        # that keeps re-sending the same slow answer would be
                        # asked the same thing until the deadline, never
                        # reaching the rotation that is the one thing left.
                        #
                        # ASKED, not counted. `correction_rounds` advances
                        # where a correction prompt is SENT, at the top of the
                        # loop, and the prompt this branch builds is sent by
                        # the next iteration -- so the schedule is read here
                        # for the round that is about to go out, and advancing
                        # it here as well would skip a round of it.
                        if _schedule_moves():
                            carried = await _resume_elsewhere(
                                f"{rounds_here} round(s) with "
                                f"{provider or 'this model'} did not make it "
                                f"fast enough",
                                avoid=provider, slow=slow,
                            )
                            if carried is not None:
                                conversation, provider, prompt = carried
                                continue
                            # Nowhere left to take it -- `_resume_elsewhere`
                            # said why. The answer in hand is correct and slow,
                            # which still beats nothing.
                            print(
                                f"[verify] the program is still too slow after "
                                f"{rounds_here} round(s) and the repair cannot "
                                f"be carried on; submitting it as it stands"
                            )
                            if plan is not None:
                                plan.note_exit("exhausted")
                            break
                        prompt = _too_slow_report(slow)
                        continue
                    if plan is not None:
                        plan.note_exit("verified")
                    break

                if getattr(conversation, "still_writing", False):
                    # The model had not finished when the read stopped, so
                    # whatever is in hand is a fragment of an answer rather than
                    # a wrong one -- and there is nothing to say to a
                    # conversation that is mid-sentence. The composer is
                    # usually disabled while a reply streams; where it is not,
                    # the prompt queues behind the answer it is asking about.
                    #
                    # Measured, with the site's busy selector dropped at startup
                    # (which `usable_busy_selectors` does whenever a candidate
                    # matches an idle page) and the model still writing:
                    #
                    #   captured=''             -> "your reply did not reach me
                    #                              as code", sent to a model
                    #                              that was still writing it
                    #   captured='def g(n):\n    total = 0\n    while n > 0:'
                    #                           -> "the code is not valid
                    #                              Python", about a program the
                    #                              model had not finished
                    #
                    # `send` already reads past its slice rather than stop here,
                    # so reaching this means the whole budget is gone. Stop.
                    print(
                        f"[verify] {provider or 'this model'} was still writing when "
                        f"the budget ran out; "
                        + (
                            "submitting the part that arrived"
                            if candidate.code.strip()
                            else "nothing arrived to submit"
                        )
                        + " rather than interrupting it with a repair prompt"
                    )
                    if plan is not None:
                        plan.note_exit("cutoff")
                    break
                if not candidate.code.strip() and (
                    getattr(conversation, "empty_reason", None)
                    in ("unreadable", "unfinished")
                ):
                    # Nothing was captured, and the conversation itself is why.
                    # A repair round here sends the fix-this prompt into a tab
                    # that just proved it cannot be read, or queues it behind an
                    # answer the model has not finished writing. Measured on a
                    # live miner, twice in one run: the first read spent 191s
                    # and returned nothing, the repair spent another 29s on the
                    # same dead conversation and returned nothing, and the task
                    # ended with 5s left -- too few to ask any of the five
                    # healthy tabs standing idle.
                    #
                    # A reply that RENDERED and simply had no code block in it
                    # is the opposite case and deliberately not caught here:
                    # that is the model breaking the output contract, telling it
                    # so is what fixes it, and the conversation is fine.
                    #
                    # The distinction cannot be made from the candidate: an
                    # empty one always carries a `defect`, because the
                    # structural checks reject empty source exactly as they
                    # reject a broken program. Only the tab knows.
                    reason = getattr(conversation, "empty_reason", "?")
                    if reason == "unreadable":
                        # The conversation is gone; the REPAIR is not. Carry it
                        # to a fresh tab rather than submitting an answer whose
                        # failures nobody asked the model to fix. Measured over
                        # a production run: fifteen solves ended exactly here,
                        # each holding a candidate that failed its own cases,
                        # with an average of 129 seconds still on the clock.
                        carried = await _resume_elsewhere(
                            f"{provider or 'this model'} returned nothing and "
                            f"that conversation is unreadable"
                        )
                        if carried is not None:
                            conversation, provider, prompt = carried
                            continue
                    # No repair to carry, or nothing left to carry it with. Say
                    # which -- this line used to promise that somebody else
                    # would be asked, and when an answer was already in hand
                    # `solve_task` submitted it instead and asked nobody.
                    print(
                        f"[verify] {provider or 'this model'} returned nothing and "
                        f"the conversation is {reason}; "
                        + (
                            "submitting the answer already in hand"
                            if best is not None and best.code.strip()
                            else "asking elsewhere"
                        )
                    )
                    if plan is not None:
                        plan.note_exit("empty")
                    break
                if duplicate:
                    # Same program, same failures: the round changed nothing.
                    # Change something rather than re-ask -- a FRESH
                    # conversation is a real change where a re-ask is not, and
                    # `_resume_elsewhere` is that move.
                    #
                    # With what is KNOWN about this program: a repeat of a
                    # program the probe already called too slow has a fault to
                    # carry even though its failure list is empty, and without
                    # it the handoff declined ("no fault to report") and the
                    # stale-tab check below ended the pass.
                    known = probed.get(candidate.code.strip())
                    carried = await _resume_elsewhere(
                        f"{provider or 'this model'} sent back the same program "
                        f"and the same failures after being shown them",
                        avoid=provider,
                        slow=known.sentence if known is not None else None,
                    )
                    if carried is not None:
                        conversation, provider, prompt = carried
                        continue
                    # Nowhere left to carry it. Whether that ends the loop turns
                    # on WHY the round changed nothing, and those are two
                    # different things wearing one shape.
                    #
                    # A round that cost no time did not involve the model: the
                    # read returned text that was already on the page. Re-asking
                    # that spins at machine speed for the rest of the budget,
                    # and this branch is the only thing standing between a
                    # broken tab and that spin. Stop.
                    #
                    # A round that took a real round trip is the other thing
                    # entirely -- the model answered, and answered the same.
                    # Stopping there ended solves with the whole budget unspent:
                    # measured, a model repeating itself once ended the loop at
                    # 0.2s of a 60s budget, throwing away every round the clock
                    # would still have paid for. Correcting runs until the
                    # answer passes or the deadline stops it, and a model is
                    # stochastic -- the next ask is a real chance, not a
                    # certainty, and a real chance is what the remaining budget
                    # is for. Fall through and ask again.
                    if round_s < STALE_ROUND_S:
                        print(
                            f"[verify] {provider or 'this model'} returned the same "
                            f"program in {round_s:.1f}s without being asked again — "
                            f"the tab is replaying an old reply rather than "
                            f"answering; submitting the last version"
                        )
                        if plan is not None:
                            plan.note_exit("stalled")
                        break
                if not candidate.defect and not candidate.failures and best is candidate:
                    # Nothing the program's cases can say against it -- and the
                    # cases are all small, by a rule this file cannot relax: an
                    # `expected` is derived by hand, so no case on the bar is
                    # ever the size the validator will run. The probe asks the
                    # one question the bar cannot, and asks it before this
                    # counts as converged.
                    slow, probe_state = await self._probe_now(
                        candidate, probe, probed, task, budget, started
                    )
                    if plan is not None:
                        plan.note_probe(probe_state)
                    if slow is not None:
                        # A too_slow round is a correction round like any
                        # other, so it faces the handoff. It did not: this
                        # `continue` jumps over the `rounds_here` check below,
                        # and with the verdict now cached per program a model
                        # that keeps re-sending the same slow answer would be
                        # asked the same thing until the deadline, never
                        # reaching the rotation that is the one thing left.
                        #
                        # ASKED, not counted. `correction_rounds` advances
                        # where a correction prompt is SENT, at the top of the
                        # loop, and the prompt this branch builds is sent by
                        # the next iteration -- so the schedule is read here
                        # for the round that is about to go out, and advancing
                        # it here as well would skip a round of it.
                        if _schedule_moves():
                            carried = await _resume_elsewhere(
                                f"{rounds_here} round(s) with "
                                f"{provider or 'this model'} did not make it "
                                f"fast enough",
                                avoid=provider, slow=slow,
                            )
                            if carried is not None:
                                conversation, provider, prompt = carried
                                continue
                            # Nowhere left to take it -- `_resume_elsewhere`
                            # said why. The answer in hand is correct and slow,
                            # which still beats nothing.
                            print(
                                f"[verify] the program is still too slow after "
                                f"{rounds_here} round(s) and the repair cannot "
                                f"be carried on; submitting it as it stands"
                            )
                            if plan is not None:
                                plan.note_exit("exhausted")
                            break
                        prompt = _too_slow_report(slow)
                        continue
                    # Reported as `converged`, and the summary line says beside
                    # it how many cases DISAGREED to begin with -- because
                    # converging from nothing and converging from four failures
                    # are the same word here and very different evidence.
                    if plan is not None:
                        plan.note_exit("converged")
                    break
                # Kept apart, not merged into one list of "problems": a defect
                # means the code never ran, and the repair prompt has to say so
                # rather than blame logic that was never executed.
                # The round did not clear it. Whether to ask THIS conversation
                # again is a question about the conversation, not the budget:
                # the deadline will stop the loop eventually, and eventually is
                # measured -- seven, nine and nine correction rounds on three
                # solves of eleven, alternating between a reply the parser
                # dropped and a rewrite that failed the same case, one ending
                # with the budget gone and nothing submitted.
                if _schedule_moves():
                    carried = await _resume_elsewhere(
                        f"{rounds_here} correction round(s) with "
                        f"{provider or 'this model'} did not clear it",
                        avoid=provider,
                    )
                    if carried is not None:
                        # Holding the program and the whole bar, so its first
                        # round starts where this conversation's last one
                        # ended. Where it LANDS is the operator's, through the
                        # `repair` phase profile -- `open_for` honours that pin
                        # now even against an `avoid`, provided the pin is not
                        # the model being fled.
                        conversation, provider, prompt = carried
                        continue
                    # Nowhere to carry it to -- `_resume_elsewhere` said why,
                    # and this line no longer guesses. The answer in hand is
                    # the best this pass is going to hold.
                    print(
                        f"[verify] {rounds_here} correction round(s) here and "
                        f"the repair cannot be carried on; submitting the best "
                        f"version in hand"
                    )
                    if plan is not None:
                        plan.note_exit("exhausted")
                    break
                # An independent reader agreed with a case this program
                # failed. Two readings against one is as much as anything here
                # establishes, so the case stands and the offer to rewrite it
                # is withdrawn for THIS round -- by evidence rather than by the
                # exhaustion the two conditions below describe.
                # ALL of the failing cases, not any of them: the confirmation
                # withdraws the case offer for every failure the prompt lists,
                # and one confirmed case beside one that was merely failing
                # used to lock the model out of correcting the second -- which
                # may be the very call the two bars split on.
                case_confirmed = bool(candidate.failed_cases) and all(
                    adjudicated.get(_case_key(case), ("", None))[0] == "case"
                    for case in candidate.failed_cases
                )
                insist = (
                    program_unchanged >= CASES_ONLY_ROUNDS
                    # A pass that has twice answered a repair by rewriting a
                    # third of its bar has said what it has to say about the
                    # cases. The offer does not come back.
                    or bulk_refusals >= MAX_BULK_REFUSALS
                )
                program_only = (
                    insist or case_confirmed
                ) and candidate.from_self_tests
                if program_only and program_unchanged >= CASES_ONLY_ROUNDS:
                    print(
                        f"[verify] the program has not changed in "
                        f"{program_unchanged} round(s); asking for the program "
                        f"this time and not offering the cases"
                    )

                report = build_repair_prompt(
                    candidate.failures,
                    task.language,
                    task.entrypoint,
                    defect=candidate.defect,
                    from_self_tests=candidate.from_self_tests,
                    insist_on_program=insist,
                    # WHOSE cases these are, said truthfully. The conversation
                    # being repaired never saw the bar when the bar is written
                    # elsewhere, and a repair round turns on exactly that.
                    bar_is_independent=self._independent_bar,
                    failed_cases=candidate.failed_cases,
                    case_confirmed=case_confirmed,
                )
                # Asking the same question a second time is worth doing -- a
                # model is stochastic and the budget is there to spend on the
                # chance. Asking it in the same WORDS is not: the conversation
                # still holds the reply it gave, and the likeliest continuation
                # of an identical prompt is an identical answer. So the report
                # goes out again, with the repetition named in it.
                # The cases this prompt is about. The reply may correct these
                # and nothing else. Kept from the last GRADED round when this
                # one ran nothing: an empty candidate has no failed cases, and
                # forgetting the reported set here made the next correction --
                # the JSON for exactly those cases -- merge against nothing
                # and vanish without a line of log.
                if candidate.total or candidate.self_total:
                    reported_failed = list(candidate.failed_cases)
                stalled = reports_sent.get(report, 0)
                reports_sent[report] = stalled + 1
                if stalled:
                    print(
                        f"[verify] this same report has now gone out "
                        f"{stalled + 1} times; naming the repetition in it "
                        f"rather than re-sending it word for word"
                    )
                    report = build_repair_prompt(
                        candidate.failures,
                        task.language,
                        task.entrypoint,
                        defect=candidate.defect,
                        from_self_tests=candidate.from_self_tests,
                        stalled=stalled,
                        insist_on_program=insist,
                        bar_is_independent=self._independent_bar,
                        failed_cases=candidate.failed_cases,
                        case_confirmed=case_confirmed,
                    )
                prompt = report
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed solve scores zero, never crashes
            print(f"[verify] backend failure: {type(exc).__name__}: {exc}")
            if plan is not None:
                plan.note_exit("failed")
        finally:
            # The bar's turn outlives the pass whenever the pass ended before
            # the program's first reply -- a backend failure, a cancelled
            # solve. Left alone it would hold a CLI process and a seat for as
            # long as the model kept writing, for a bar nothing will read.
            if bar_task is not None and not bar_task.done():
                bar_task.cancel()
            if bar2_task is not None and not bar2_task.done():
                bar2_task.cancel()
            for live in (conversation, *bar, *bar2):
                if live is None:
                    continue
                try:
                    await live.close()
                except Exception:  # noqa: BLE001 - cleanup must not mask a result
                    pass
            # Observe the cancellation, bounded, AFTER the closes so a cancel
            # of this pass mid-wait cannot skip them. On this interpreter
            # (3.11) `asyncio.wait_for` can swallow a cancel that lands in the
            # loop iteration where its inner future has just completed -- at
            # the bar's slot acquire, that means the cases turn spawns a
            # child and runs to the end of its slice on the seat's quota,
            # with nothing left holding a reference to cancel it again.
            # Reproduced through this method with the real backend: cancels
            # landing 1-3 iterations after the send began returned normally.
            # So: wait a bounded moment, and if it is still running, cancel
            # once more. The wait is `asyncio.wait`, which does not cancel
            # what it waits on, and a cancel of THIS task during it
            # propagates, as it should, with the re-cancel in the finally.
            for pending in (bar_task, bar2_task):
                if pending is None or pending.done():
                    continue
                try:
                    await asyncio.wait({pending}, timeout=6.0)
                finally:
                    if not pending.done():
                        pending.cancel()
            # A bar task that finished with an exception before the pass
            # ended -- its open raised, its send raised -- is done, so it was
            # neither cancelled nor awaited above, and asyncio logs "Task
            # exception was never retrieved" at collection, reading as a crash
            # in the miner log. Retrieved here, and that is all.
            for finished in (bar_task, bar2_task):
                if finished is not None and finished.done() and not finished.cancelled():
                    finished.exception()
            # The second bar never landed while this pass ran. Its conversation
            # was closed above like the others; what remains is to say so on
            # the summary line, where `bar2=late` is the instrument for
            # deciding whether the second reader is fast enough to be worth
            # its turn.
            if bar2_task is not None and not bar2_joined and plan is not None:
                plan.bar2_provider = "late"
                print("[verify] the second bar never landed before the pass "
                      "ended; the first bar stood on its own")
        return best, best_provider

    async def _write_the_bar(
        self, holder: list, task, budget: float, started: float,
        avoid: Optional[str], plan, probe=None, phase: str = "cases",
        provider_slot: str = "bar_provider",
    ) -> tuple[Optional[list], float]:
        """Open a conversation of the bar's own and ask it for the cases.

        Returns the cases and how long the TURN itself took.

        Runs as a task beside the program turn, which is why the open is in
        here: a backend that queues would otherwise make the program wait for
        a tab the program does not use.

        `holder` receives the conversation as soon as there is one, so
        `_attempt`'s cleanup can close it however this ends -- including the
        ways that never return, where the task is cancelled mid-open.

        The whole remaining budget is offered, and no share of it withheld.
        The share existed because the program turn had to wait its turn; it no
        longer does, and a ceiling here would cost the bar without buying the
        program anything. What bounds the wait is `_collect_bar`, on the other
        side, where the cost is actually paid.
        """
        conversation = await self._open_within(
            budget, started, avoid, phase=phase
        )
        holder.append(conversation)
        left = budget - (time.monotonic() - started)
        # Timed HERE, and handed back with the cases. Timing it where it is
        # collected instead measures from the bar's start to the moment the
        # program's turn happened to finish and got round to awaiting it --
        # which is `max(bar, program)`, not the bar, and reads as the bar
        # whenever the program is the slower of the two. That is the one
        # comparison the split exists to let an operator make.
        began = time.monotonic()
        cases = await self._ask_for_cases(
            conversation, task, max(1.0, left), probe=probe, independent=True,
        )
        # Read AFTER the turn, not at open: the ladder may hop the
        # conversation to another model inside the turn, and `bar=` on the
        # summary line is the instrument for deciding which model should
        # write the bar -- bound at open it credited the one that refused.
        if plan is not None and getattr(plan, provider_slot, None) != "late":
            setattr(plan, provider_slot, getattr(conversation, "provider", None))
        return cases, time.monotonic() - began

    async def _collect_bar(
        self, bar_task, bar: list, phases, budget: float, started: float,
        bar_started: float, beside: bool = True, label: str = "1 cases",
        who: str = "the bar", quiet: bool = False,
    ) -> list[dict]:
        """The independent cases turn's result, or an empty bar.

        Bounded by what is left of the solve, and by nothing else. Reserving a
        correction round out of it was a second deadline on a phase: it bought
        the repair loop time to act on a disagreement, at the price of
        sometimes discarding the bar that would have found one. A bar that
        arrives too late to repair against is still the difference between an
        answer graded and an answer shipped unchecked -- `disagreed=` and
        `self=` on the summary line are worth more than a round nobody may
        need, and 74% of solves never use one.

        Returns `[]` for every way the BAR can fail -- a cases turn still
        writing, an unreadable tab, a reply carrying nothing usable. None of
        them are the program's problem: the program was written in a
        conversation this one never touched, and it ships either way. That is
        the difference this split buys and the reason no failure here is
        allowed to propagate. A cancelled SOLVE is the one exception and is
        re-raised; see below.
        """
        left = budget - (time.monotonic() - started)
        cases: Optional[list] = None
        # How long the bar's own turn took, as measured beside it. Falls back
        # to the elapsed wall time only when the turn never reported one,
        # which is every path where there are no cases to report anyway.
        spent: Optional[float] = None
        try:
            cases, spent = await asyncio.wait_for(
                bar_task, timeout=max(1.0, left)
            )
        except asyncio.CancelledError:
            # The SOLVE was cancelled, not the bar. `wait_for` raises the same
            # exception either way, and swallowing it here would leave a
            # cancelled solve grading and reporting an answer nobody is
            # waiting for -- `CancelledError` is a BaseException precisely so
            # that a blanket handler cannot do that.
            bar_task.cancel()
            raise
        except asyncio.TimeoutError:
            bar_task.cancel()
            # Measured NOW, when the wait ended -- not before it began, which
            # reported a bar that ran out the budget as having taken no time.
            spent = time.monotonic() - bar_started
            print(
                f"[verify] {who} was still being written when the "
                f"{budget:.0f}s budget ran out"
                + ("" if quiet else "; the program is graded against "
                   "the public examples alone")
            )
        except Exception as exc:  # noqa: BLE001 - the program ships regardless
            spent = time.monotonic() - bar_started
            print(f"[verify] {who} came back unusable ({exc})"
                  + ("" if quiet else "; the program is graded against "
                     "the public examples alone"))
        finally:
            for conversation in bar:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - cleanup must not mask a result
                    pass
            # Emptied HERE, not by the caller. When this is cancelled mid-wait
            # the re-raise skips the caller's `bar = []`, and `_attempt`'s own
            # finally then closes the same conversations a second time --
            # `release()` clamps at zero, so the live-session count drifts
            # low and a real leak later reads as nothing.
            bar.clear()
        # `beside` only when it really ran beside the program. A RETRY does
        # not: the program's turn has already returned by then, so the retry is
        # sequential and must move the phase cursor like any other phase.
        # Marking it alongside would leave the cursor behind it and charge the
        # next correction round with the retry's seconds -- the same
        # misreporting, pointed the other way.
        phases.mark(
            label,
            model_s=spent if spent is not None else time.monotonic() - bar_started,
            beside=beside,
        )
        if cases:
            print(f"[verify] {who} holds {len(cases)} case(s), written "
                  f"without sight of the program")
        return list(cases or [])

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

    async def _adjudicate(
        self, task, cases: list[dict], actuals: list, provider: Optional[str],
        budget: float, started: float, split_keys: dict,
    ) -> dict[tuple, tuple[str, Any]]:
        """Who is right about a disputed case: the bar, the program, or nobody.

        Returns `{case key: (route, what the reader said)}` where route is
        "case", "program" or "contested" -- the value beside it is what makes
        a "the case was wrong" verdict applicable without asking anyone
        again. `{}` for
        every way this can fail. A judge that does not answer leaves the round
        exactly as it was -- the repair prompt still goes out offering both
        ways, which is what happened before this existed.

        THE PROBLEM. A disagreement is one reading of the statement against
        another, and the repair prompt asks the program's own author to say
        which was wrong. Measured over 54 live solves: of ten single-case
        disagreements, nine ended with the model editing its own case and
        keeping its program. That may often be right -- an earlier run had a
        judge uphold case rewrites 22 times in 25 -- but the model deciding it
        is the one whose reading is on trial, and when the bar has caught a
        real bug this is exactly where the catch is erased.

        WHY THIS QUESTION AND NOT ANOTHER. The judge is never asked "is this
        program correct" or "which of these two is right". It is shown the
        statement and the failing CALLS, no program and no expected values,
        and asked what each call must return -- the one question two
        independent readings actually agree about. `two_bar_overlap.py`: two
        models choosing their own inputs shared 5 of 233. `fixed_inputs.py`:
        the same two models, inputs held fixed, agreed on 91 of 97 expected
        values. So the judge is put to the second question and never the first.

        WHAT ITS ANSWER MEANS. Agreeing with the bar makes two independent
        readings against the program's one, which is as much as anything here
        can establish, and the case stands. Agreeing with the program means
        the case was wrong and it is corrected without spending a model turn
        on it. Agreeing with neither is the statement being ambiguous at that
        call, and the case is dropped rather than held against a program that
        may well be right.

        Cases the two BARS already split on are not brought here: the second
        bar is itself an independent reading of that call, and the judge --
        pinned to the same model as the second bar -- would not be a third one.
        """
        wanted = [
            (case, actual)
            for case, actual in zip(cases, actuals)
            if _case_key(case) not in split_keys
        ]
        if not wanted:
            return {}
        left = budget - (time.monotonic() - started)
        if left <= 0:
            return {}
        asked = [case for case, _ in wanted]
        conversation = None
        try:
            conversation = await self._open_within(
                budget, started, provider, phase="judge"
            )
            reply = await self._send_within(
                conversation,
                build_expected_prompt(
                    task.language, task.statement, task.entrypoint, asked,
                    examples=task.public_examples,
                ),
                max(1.0, budget - (time.monotonic() - started)),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - no verdict is not a failure
            print(f"[verify] the independent reading did not arrive ({exc}); "
                  f"the round asks as it always did")
            return {}
        finally:
            if conversation is not None:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - cleanup must not mask this
                    pass
        said = extract_expected(reply, len(asked))
        if not said:
            print("[verify] the independent reading came back unusable; the "
                  "round asks as it always did")
            return {}
        verdicts: dict[tuple, tuple[str, Any]] = {}
        for index, (case, actual) in enumerate(wanted):
            if index not in said:
                # Left out on purpose, by a reader told to leave out a call
                # the statement does not determine. That is a vote for
                # ambiguity, but a weak one, so it decides nothing.
                continue
            judged = said[index]
            if task.language == "rust" and isinstance(judged, (int, float, bool)):
                # A Rust case's `expected` is stdout, a string, and the grader
                # compares strings; a reader that answered a numeric problem
                # with the number -- `6` for `"6"` -- matched neither side
                # and had the case DROPPED as contested. JSON's spelling of
                # the scalar is what the program would have printed.
                judged = json.dumps(judged)
            # The GRADER's comparison, both times. Two earlier passes at this
            # measurement compared with `json.dumps` and reported agreement of
            # 0% and 48%, every point of the difference whitespace: for Rust a
            # verdict is stdout compared on whitespace tokens, and nothing
            # else may decide it.
            like_bar = _same_expected(judged, case.get("expected"), task.language)
            like_program = bool(actual.ok) and _same_expected(
                judged, actual.value, task.language
            )
            key = _case_key(case)
            if like_bar and not like_program:
                verdicts[key] = ("case", judged)
            elif like_program and not like_bar:
                verdicts[key] = ("program", judged)
            elif not like_bar and not like_program:
                verdicts[key] = ("contested", judged)
            # Agreeing with both is impossible unless the program already
            # passed, which is not how a case gets here.
        return verdicts

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
        self, reply: str, task, left: float, cases: Optional[list] = None,
        previous: str = "",
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
                self._grade, reply, task, left, cases, previous
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

    # ---------------------------------------------------------------------- #
    def _run_self_tests(
        self, candidate: Candidate, task, cases: Optional[list] = None,
        left: Optional[float] = None,
    ) -> None:
        """Grade a candidate against the cases turn 1 obtained.

        Never raises and never blocks the answer. A model wrote both halves of
        this -- the cases and the JSON they arrived in -- so every failure mode
        here ends in "no self-tests ran", which is exactly where this code path
        started.

        It does not read the reply. The program turn asks for ONE block, so
        there is nothing to extract from it -- and mining it for cases anyway
        would mean grading a program against whatever it volunteered about
        itself, which is the back-filling the split exists to prevent. A repair
        reply that CORRECTS a case is handled where it belongs, in `_attempt`,
        which feeds the corrected array to the next round.
        """
        cases = list(cases or [])
        if not cases:
            return
        names = [case.get("name", "") for case in cases]
        try:
            passed, total, failures, failed, actuals = (
                self._grader.check_detailed(
                    candidate.code, task.language, task.entrypoint, cases,
                    names, budget_s=left,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a broken executor loses no answer
            # Same four words as the public-examples path below, deliberately.
            # They are what an operator counts to decide whether a missing
            # executor is costing anything -- and on live traffic, which ships
            # no public examples, THIS is the only path that can print it. The
            # other wording made that count read zero while every answer in the
            # language went out ungraded.
            print(f"[verify] local grading unavailable, so the model's own cases "
                  f"could not be run: {type(exc).__name__}: {exc}")
            return
        if not total:
            return
        candidate.self_passed, candidate.self_total = passed, total
        # Exact for THIS call: `check` counts every result as either a pass or
        # a failure line, so the two together are what ran.
        candidate.self_observed = passed + len(failures)
        candidate.from_self_tests = True
        # Only failures drive a repair. A clean run is left silent: it is the
        # ordinary outcome and saying so on every solve would bury the line
        # that matters.
        candidate.failures = failures
        # WHICH cases failed, not just what the failures looked like. A repair
        # round may take these back corrected, and the merge that does it has
        # to know exactly which of the agreed cases are in play.
        candidate.failed_cases = failed
        candidate.failed_actuals = actuals
        candidate.self_bar = list(cases)

    def _grade(
        self, reply: str, task, left: Optional[float] = None,
        cases: Optional[list] = None, previous: str = "",
    ) -> Candidate:
        code = extract_code(reply, task.entrypoint, task.language)
        candidate = Candidate(code=code, raw=reply, self_cases=len(cases or []))
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
            unrun = []
            if task.public_examples:
                unrun.append(f"the {len(task.public_examples)} public example(s)")
            if cases:
                unrun.append(f"the model's {len(cases)} own case(s)")
            if unrun:
                print(
                    f"[verify] out of budget before {' or '.join(unrun)} could be "
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
        # Running these is not verification and is never recorded as any:
        # `passed`/`total` are the validator's examples alone, so `verified`
        # cannot be earned by a model agreeing with itself. What they catch is
        # the commonest failure by far -- the model knowing what the answer
        # should be and coding it wrong -- and that is objectively checkable
        # with the validator's own executor.
        if self._self_tests and code.strip():
            # Bounded by what is left of the grading budget NOW, after the
            # compile and the examples above have spent their share: the two
            # suites share one deadline, and `run_tests` still stops at
            # `VERIFY_TIMEOUT_S` per case inside it.
            self._run_self_tests(candidate, task, cases, left=_budget_left())
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

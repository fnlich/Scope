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

A note on the executor: ``SubprocessExecutor`` is documented as dev-grade
because it does not confine untrusted code on Linux. Here the code being run
is YOUR OWN solver's output rather than an adversary's, which is the one
setting that backend is appropriate for. Set ``SOLVER_VERIFY_EXECUTOR=docker``
to use the container backend instead; Rust verification always requires it.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from rlvr.types import TestCase

from .prompts import (
    build_code_prompt,
    build_repair_prompt,
    build_resume_prompt,
    build_tests_prompt,
    extract_code,
    extract_self_tests,
    python_defect,
    rust_defect,
)
from .rust_compile import compile_defect, rustc_path

# Per-example wall clock when checking our own candidate. Kept small: this is
# a smoke test against tiny public examples, not the real grading run.
VERIFY_TIMEOUT_S = float(os.environ.get("SOLVER_VERIFY_TIMEOUT_S", "5"))

# How long an executor that could not be built stays unavailable before the
# next solve is allowed to try again.
#
# A hold, not a verdict: a Docker daemon started after the miner is the
# ordinary case, and a permanent answer would mean grading nothing in that
# language until someone restarted the process. Long enough that Rust tasks
# stop paying for the probe, short enough that a box the operator has just
# fixed heals on its own.
EXECUTOR_RETRY_S = float(os.environ.get("SOLVER_EXECUTOR_RETRY_S", "300"))

# What a repair needs to be worth carrying to a FRESH conversation: a tab, a
# prompt that restates the whole problem, and a read long enough to answer it.
# More than the 12s an in-conversation round trip needs, because none of that
# is warm.
RESUME_FLOOR_S = 40.0


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

    Structurally compatible with ``custom_miner.SolveResult`` (the miner only
    reads ``.code`` and ``.raw_response``) and defined here on purpose: importing
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

    @property
    def verified(self) -> bool:
        """Every public example reproduced exactly."""
        return self.defect is None and self.total > 0 and self.passed == self.total

    @property
    def score(self) -> tuple[int, int, int]:
        """Ranking key for 'best so far' — passes, then non-empty, then runnable.

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

    @staticmethod
    def _build_settings():
        from rlvr.config import Settings

        kind = os.environ.get("SOLVER_VERIFY_EXECUTOR", "subprocess")
        # _env_file=None so the miner's .env cannot accidentally repoint this.
        return Settings(_env_file=None, executor=kind, per_test_timeout_s=VERIFY_TIMEOUT_S)

    def executor(self, language: str):
        cached = self._cache.get(language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(language)
            if cached is not None:
                return cached
            held = self._broken.get(language)
            if held is not None and time.monotonic() - held[0] < EXECUTOR_RETRY_S:
                # Remembered, not re-probed. The message is the original one
                # verbatim: the caller prints it per solve and an operator
                # counts those lines, so it must not change shape here.
                raise RuntimeError(held[1])
            try:
                executor = self._build(language)
            except Exception as exc:  # noqa: BLE001 - remembered, then re-raised
                self._broken[language] = (time.monotonic(), f"{exc}")
                self._report_unavailable(language, exc)
                raise
            self._broken.pop(language, None)
            self._cache[language] = executor
            return executor

    def state(self, language: str) -> str:
        """What grading this language would do right now. Never probes.

        Read from `/solver-status`, so it must not shell out: a `docker info`
        against a hung daemon blocks for twenty seconds, and the endpoint an
        operator polls to find out whether the miner is healthy is the last
        place to put that.
        """
        if language in self._cache:
            return "ready"
        held = self._broken.get(language)
        if held is not None:
            return f"unavailable: {held[1]}"
        return "not checked yet"

    def _build(self, language: str):
        """Construct the executor for ``language``. Assumes the lock is held."""
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
        return get_executor(settings, language=language)

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

    def check(
        self, code: str, language: str, entrypoint: str,
        examples: list[dict[str, Any]], names: Optional[list[str]] = None,
    ) -> tuple[int, int, list[str]]:
        """Run ``code`` against the public examples. Returns (passed, total, failures).

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
            return 0, 0, []
        results = self.executor(language).run_tests(
            code, entrypoint, cases, VERIFY_TIMEOUT_S
        )
        failures: list[str] = []
        passed = 0
        for index, (result, case) in enumerate(zip(results, cases)):
            if result.passed:
                passed += 1
                continue
            label = ""
            if names and index < len(names) and names[index]:
                label = f"case {index + 1} {names[index]!r}: "
            failures.append(label + _describe(result, case, language, entrypoint))
        return passed, len(cases), failures


def _describe(result, case: TestCase, language: str, entrypoint: str) -> str:
    """One line of concrete evidence for the repair prompt."""
    if language == "rust":
        call = f"stdin={_clip(case.args[0] if case.args else '')!r}"
    else:
        call = f"{entrypoint}(*{case.args!r}, **{case.kwargs!r})"
    if result.timed_out:
        return f"{call} timed out after {VERIFY_TIMEOUT_S:g}s (too slow or an infinite loop)"
    if result.error:
        return f"{call} raised: {_clip(result.error, 300)}"
    actual = result.value if result.value_ok else result.actual_repr
    return f"{call} returned {_clip(repr(actual))}, expected {_clip(repr(case.expected))}"


def _clip(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# How much budget must be left to ask ANOTHER model, by what is already in hand.
#
# The two numbers differ because what they risk differs, and the payment policy
# is what sets the price. `all_passed` is a hard gate: an empty answer pays
# exactly zero, and no amount of hurrying changes that. Above the gate, speed is
# a multiplier floored at 0.95 -- the slowest correct answer still earns 95% of
# what the fastest earns. So:
#
#   * Empty-handed, ANY time worth a round trip is worth spending. A failed
#     extra ask costs nothing that was not already lost; a successful one is the
#     whole payment. So the floor is the mechanical minimum and nothing more:
#     12s is what `_attempt`'s own loop refuses to start a round below, which
#     makes this "as long as an ask can happen at all".
#   * Holding an unverified answer, the ask is speculative rather than free --
#     it spends a real account's quota to improve on something that may already
#     be right. That bar stays where it was.
#
# The ordering is the invariant, not the values: empty-handed must never be the
# HARDER case to justify. It is the one with nothing to lose.
EMPTY_HANDED_FLOOR_S = 12.0
SECOND_OPINION_FLOOR_S = 20.0

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

# Turn 1 has NO timeout of its own. There is one clock on a solve -- the
# deadline the validator advertised -- and turn 1 reads against that, exactly
# like every other read here.
#
# It carried a private cap twice, and both were wrong in the same direction.
# The first passed `extend_to_s` equal to the slice, which makes `send`'s
# extension a no-op by construction; the second kept a soft 60s slice and a hard
# 100s cap. Both cut the model off MID-THINK, and cutting a model off mid-think
# is the one thing that cannot help: the reply does not exist yet, so what the
# cap saves is time that bought nothing and what it costs is the whole turn.
# Measured on a live tab: `Thought for 1m 17s` before a single character
# appeared, against a 60 second cap.
#
# A cap LOOKS like it protects the program's read, and it does not. `send`
# returns the moment the model finishes -- the slice is a ceiling, never a wait
# -- so a cases turn that takes 90 seconds hands the program the other 190
# whether or not a cap exists. The only case a cap changes is the one where the
# model has NOT finished, and there it converts a slow answer into no answer,
# which is the one trade the payment policy says never to make.
#
# What protects the program now is the budget itself and `_Plan`: turn 1 cannot
# outlive the deadline, and a turn 1 that fails the TASK's way does not get a
# second chance in the same solve.


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
        safety_margin_s: float = 20.0,
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
        self._margin = max(0.0, float(safety_margin_s))
        self._max_budget = max(5.0, float(max_budget_s))
        self._second_opinion = bool(second_opinion)
        self._self_tests = bool(self_tests)
        self._grader = _Grader()
        self._cache: dict[str, tuple[str, str]] = {}
        self._cache_size = max(0, int(cache_size))
        self._counts = {"solved": 0, "verified": 0, "cache_hits": 0, "empty": 0}
        self._by_provider: dict[str, dict[str, int]] = {}
        # The no-examples explanation is worth saying, but only once a run.
        self._warned_ungradeable = False
        # ...and so is a deadline being cut short by our own configuration.
        self._warned_short_deadline = False

    # -- the Solver interface custom_miner.py expects ---------------------- #
    async def solve_task(self, task, timeout_s: float) -> Answer:
        started = time.monotonic()
        # `timeout_s` is already `min(deadline_s, glm_request_timeout_s)` -- the
        # deadline the miner's own `handle_request` will 504 at. The margin is
        # what keeps this side of it: `send` runs its copy, stream and
        # post-mortem phases AFTER its slice expires (5 + 4 + 2 = 11s), and the
        # answer still has to be graded, archived, signed and put on the wire
        # before the validator stops listening at `deadline_s + 10`.
        #
        # `_max_budget` is a ceiling, not a target, and it deliberately does not
        # bind at the deadlines this subnet advertises. It used to: 240 against a
        # 300s deadline capped the first attempt at 191s, and a model that needed
        # longer had its answer thrown away by this miner rather than by the
        # validator -- which would have paid 96% for the same answer arriving at
        # six minutes. Correctness is worth 100%; speed is worth at most 5%.
        budget = min(float(timeout_s), self._max_budget) - self._margin
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
            # `timeout_s` is `min(deadline_s, glm_request_timeout_s)`. When it
            # comes back SHORTER than what the validator advertised, the miner
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
            return Answer(code=code, raw_response=raw, verified=True)

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
                empty_handed = not best.code.strip()
                if not empty_handed and attempt_no >= SECOND_OPINION_PASSES:
                    # There is an answer in hand and it has already had its
                    # second opinion. `MAX_PASSES` is for the empty case only:
                    # spending it here would double or quadruple what every
                    # failing task costs a real account's quota, to improve on
                    # something already worth submitting.
                    break
                floor_s = (
                    EMPTY_HANDED_FLOOR_S if empty_handed else SECOND_OPINION_FLOOR_S
                )
                if remaining < floor_s:
                    print(
                        f"[verify] {remaining:.0f}s left; "
                        + (
                            "not enough to ask anyone else, submitting empty"
                            if empty_handed
                            else "no time for a second opinion"
                        )
                    )
                    break
            attempt_no += 1
            candidate, provider = await self._attempt(
                task, remaining, avoid=asked[-1] if asked else None, plan=plan
            )
            if provider:
                asked.append(provider)
            if candidate is not None and candidate.score > best.score:
                best = candidate
                won_with = provider
            if best.verified:
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
                if best.from_self_tests:
                    # Not ungradeable after all: the model shipped cases and
                    # they ran. The warning below is about having NOTHING to
                    # run and would be false here.
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
            if attempt_no < passes:
                print(
                    f"[verify] {provider or 'first'} "
                    + (
                        "returned nothing; asking another model"
                        if not best.code.strip()
                        else "did not verify; asking another model"
                    )
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
        if best.code.strip():
            self._counts["solved"] += 1
            if best.verified and self._cache_size:
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (best.code, best.raw)
        else:
            self._counts["empty"] += 1
        elapsed = time.monotonic() - started
        print(
            f"[verify] {task.language} entrypoint={task.entrypoint} "
            f"provider={won_with or 'none'} "
            f"examples={best.passed}/{best.total} "
            + (f"self={best.self_passed}/{best.self_total} " if best.self_total else "")
            + f"verified={best.verified} "
            f"{elapsed:.1f}s/{budget:.0f}s"
        )
        return Answer(
            code=best.code, raw_response=best.raw,
            verified=best.verified, passed=best.passed, total=best.total,
        )

    async def _ask_for_cases(self, conversation, task, left: float):
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
        prompt = build_tests_prompt(
            task.language, task.statement, task.entrypoint, task.public_examples
        )
        # No `extend_to_s`: the slice already IS everything left, so there is
        # nothing to extend to and nothing being held back to extend into.
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
        try:
            conversation = await self._backend.open(avoid=avoid)
            provider = getattr(conversation, "provider", None)
            # Turn 1: the cases, before the program exists. Cases written
            # ALONGSIDE a program can be back-filled from what the program
            # happens to do, and then they agree with its bugs; cases written
            # first cannot. That is the whole argument for spending a round
            # trip here.
            cases: Optional[list] = None
            two_phase = plan is None or plan.two_phase
            if self._self_tests and two_phase:
                cases = await self._ask_for_cases(
                    conversation, task, budget - (time.monotonic() - started)
                )
                if cases is None:
                    # The tab could not be read, or the model was still writing.
                    # Sending turn 2 into it would queue behind an answer that
                    # has not arrived, so this conversation is finished either
                    # way. What differs is whether anything else is worth doing,
                    # and the clock decides it.
                    #
                    # Whether anything ELSE happens is already decided, by the
                    # clock, in `solve_task`: it will not open another pass with
                    # less than `EMPTY_HANDED_FLOOR_S` left. That is the whole
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
                    if left_after < EMPTY_HANDED_FLOOR_S:
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
                    return best, provider
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
            # case can be told apart from one that rewrites both.
            last_code: Optional[str] = None
            # A repair may be carried to a fresh conversation ONCE per pass.
            resumed = False
            # The last reply, verbatim. With the round count gone, a tab that
            # answers INSTANTLY is the one thing that can spin: `send` normally
            # blocks on the model for tens of seconds, but a read that returns
            # stale text returns it at once, and the loop would resend the same
            # repair at machine speed for the whole budget -- hammering the site
            # from an account the operator is signed in to. A byte-identical
            # reply to a prompt quoting a fresh failure is not a revision.
            last_reply: Optional[str] = None
            attempt = 0
            while True:
                attempt += 1
                if self._max_attempts and attempt > self._max_attempts:
                    break
                left = budget - (time.monotonic() - started)
                if attempt > 1 and left < 12.0:
                    # Not enough left to be worth another ROUND TRIP -- which is
                    # what this has always been about, and it never should have
                    # gated the first one. It did: below a 32-second deadline
                    # the budget lands under twelve seconds and the model was
                    # never asked at all, so the miner returned an empty answer
                    # without a single line of log to say why. The first attempt
                    # always runs, however little there is, exactly as the first
                    # pass does in `solve_task`.
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
                reply = await conversation.send(prompt, left)
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
                if attempt > 1 and reply == last_reply:
                    print(
                        f"[verify] {provider or 'this model'} sent back the "
                        f"identical reply after being shown the failure; "
                        f"stopping rather than asking again"
                    )
                    break
                last_reply = reply
                revised = extract_self_tests(reply, task.entrypoint, task.language)
                if revised and len(revised) < len(agreed):
                    # A revision may CORRECT a case. It may not delete one --
                    # and dropping the case you cannot pass is exactly how a
                    # bar gets cleared without the program improving. The
                    # repair prompt asks for the complete array back, so a
                    # short one is either disobedience or the thing this
                    # guards against; either way the old bar stands.
                    print(
                        f"[verify] the repair sent back {len(revised)} case(s) "
                        f"where {len(agreed)} were agreed; keeping the fuller "
                        f"set — a case may be corrected, not dropped"
                    )
                    revised = []
                now_code = extract_code(reply, task.entrypoint, task.language).strip()
                if revised and last_code is not None and now_code == last_code:
                    agreed, revised = revised, None
                candidate = await self._graded(
                    reply, task, budget - (time.monotonic() - started), agreed
                )
                last_code = now_code
                if revised:
                    agreed = revised
                if best is None or candidate.score > best.score:
                    best = candidate
                elif candidate.score == best.score and not getattr(
                    conversation, "still_writing", False
                ):
                    # A TIE goes to the later candidate, and that is not a
                    # coin toss: this one was written after seeing the failure
                    # report, so it is the model's considered revision of the
                    # one already in hand.
                    #
                    # Strict `>` made every repair round a no-op whenever the
                    # score could not move -- and the score cannot move when a
                    # case is WRONG. Measured: turn 1 wrote a case no correct
                    # program can pass, attempt 1 scored 1/2, the model then
                    # rewrote the program properly, and the rewrite tied at 1/2
                    # and was thrown away. The miner submitted the first draft
                    # and the model's last word never left the tab. With three
                    # bogus cases pinning a solve at 17/20, that is every
                    # remaining round.
                    #
                    # Not when the model was STILL WRITING, though. What
                    # arrived there is a fragment of an answer rather than a
                    # revision of one, and a fragment that happens to parse and
                    # tie must not displace the finished program above it.
                    best = candidate
                if candidate.verified:
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
                    left_now = budget - (time.monotonic() - started)
                    if (
                        reason == "unreadable"
                        and not resumed
                        and best is not None
                        and best.code.strip()
                        and (best.failures or best.defect)
                        and left_now >= RESUME_FLOOR_S
                    ):
                        # The conversation is gone; the REPAIR is not. Carry it
                        # to a fresh tab rather than submitting an answer whose
                        # failures nobody asked the model to fix. Measured over
                        # a production run: fifteen solves ended exactly here,
                        # each holding a candidate that failed its own cases,
                        # with an average of 129 seconds still on the clock.
                        #
                        # Once per pass. A tab that dies on the resume too is a
                        # fleet problem, not something to keep paying for.
                        resumed = True
                        print(
                            f"[verify] {provider or 'this model'} returned nothing "
                            f"and that conversation is unreadable; carrying the "
                            f"repair to a fresh one with {left_now:.0f}s left"
                        )
                        try:
                            await conversation.close()
                        except Exception:  # noqa: BLE001 - it is already broken
                            pass
                        conversation = await self._backend.open(avoid=None)
                        provider = getattr(conversation, "provider", provider)
                        prompt = build_resume_prompt(
                            task.language, task.statement, task.entrypoint,
                            task.public_examples, agreed, best.code,
                            best.failures, defect=best.defect,
                            from_self_tests=best.from_self_tests,
                        )
                        continue
                    # No repair to carry, or nothing left to carry it with. Say
                    # which -- this line used to promise that somebody else
                    # would be asked, and when an answer was already in hand
                    # `solve_task` submitted it instead and asked nobody.
                    print(
                        f"[verify] {provider or 'this model'} returned nothing and "
                        f"the conversation is {reason}; "
                        + (
                            f"submitting the answer already in hand"
                            if best is not None and best.code.strip()
                            else "asking elsewhere"
                        )
                    )
                    break
                if not candidate.defect and not candidate.failures:
                    break  # nothing actionable to report (no examples shipped)
                # Kept apart, not merged into one list of "problems": a defect
                # means the code never ran, and the repair prompt has to say so
                # rather than blame logic that was never executed.
                prompt = build_repair_prompt(
                    candidate.failures,
                    task.language,
                    task.entrypoint,
                    defect=candidate.defect,
                    from_self_tests=candidate.from_self_tests,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed solve scores zero, never crashes
            print(f"[verify] backend failure: {type(exc).__name__}: {exc}")
        finally:
            if conversation is not None:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - cleanup must not mask a result
                    pass
        return best, provider

    async def _graded(
        self, reply: str, task, left: float, cases: Optional[list] = None
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
            return await asyncio.to_thread(self._grade, reply, task, left, cases)
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
            passed, total, failures = self._grader.check(
                candidate.code, task.language, task.entrypoint, cases, names
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
        candidate.from_self_tests = True
        # Only failures drive a repair. A clean run is left silent: it is the
        # ordinary outcome and saying so on every solve would bury the line
        # that matters.
        candidate.failures = failures

    def _grade(
        self, reply: str, task, left: Optional[float] = None,
        cases: Optional[list] = None,
    ) -> Candidate:
        code = extract_code(reply, task.entrypoint, task.language)
        candidate = Candidate(code=code, raw=reply)
        defect = (
            rust_defect(code)
            if task.language == "rust"
            else python_defect(code, task.entrypoint)
        )
        out_of_budget = left is not None and left <= 0
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
            defect = compile_defect(code, left)
        if defect is not None:
            # Structurally unusable: report it without paying for execution.
            candidate.defect = defect
            candidate.code = "" if not code.strip() else code
            return candidate
        if not task.public_examples:
            # Nothing SHIPPED to verify against -- which is every task on live
            # traffic -- so the only cases that can exist are the ones the model
            # wrote for its own program. Running them is not verification and is
            # never recorded as any: `passed`/`total` stay at zero, so `verified`
            # stays False and the answer is never cached. What it does catch is
            # the commonest failure by far, the model knowing what the answer
            # should be and coding it wrong, and that is objectively checkable
            # with the validator's own executor.
            if self._self_tests and not out_of_budget and code.strip():
                self._run_self_tests(candidate, task, cases)
            return candidate
        if out_of_budget:
            # The budget is gone, so running the examples buys nothing that can
            # still be acted on: there is no time for a repair round, and
            # `verified` never reaches the validator -- it feeds this process's
            # cache and its stats and nothing else. It is not free, either:
            # every case gets VERIFY_TIMEOUT_S, in a subprocess or a container,
            # and the deadline above is an `asyncio.wait_for` that answers 504
            # rather than late. The check would be paid for with the answer it
            # was checking. The structural checks above already ran; they cost
            # microseconds and are what ranks this candidate.
            print(
                "[verify] out of budget before the examples could be run; "
                "submitting the answer unverified"
            )
            return candidate
        try:
            passed, total, failures = self._grader.check(
                code, task.language, task.entrypoint, task.public_examples
            )
        except Exception as exc:  # noqa: BLE001 - a broken grader must not lose the answer
            print(f"[verify] local grading unavailable: {type(exc).__name__}: {exc}")
            return candidate
        candidate.passed, candidate.total, candidate.failures = passed, total, failures
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

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
import inspect
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from rlvr.types import TestCase

from .prompts import (
    build_initial_prompt,
    build_repair_prompt,
    extract_code,
    python_defect,
    rust_defect,
)
from .rust_compile import compile_defect

# Per-example wall clock when checking our own candidate. Kept small: this is
# a smoke test against tiny public examples, not the real grading run.
VERIFY_TIMEOUT_S = float(os.environ.get("SOLVER_VERIFY_TIMEOUT_S", "5"))


class Conversation(Protocol):
    """One live, isolated model conversation.

    The repair loop deliberately stays inside a single conversation so the
    model can see its own previous attempt alongside the failure report.
    """

    # A backend MAY also accept ``extend_to_s`` -- the caller's real remaining
    # budget, as opposed to the slice it allocated for this one read. One that
    # does may read past its slice while the model is still writing rather than
    # stop mid-answer. One that does not is simply never offered it; see
    # ``_reads_past_its_slice``.
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
        return (self.passed, 1 if self.code.strip() else 0, 0 if self.defect else 1)


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
            executor = get_executor(settings, language=language)
            self._cache[language] = executor
            return executor

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
# Which backends accept `extend_to_s`, by class, asked once each.
#
# The parameter is optional on purpose. `Conversation` is a Protocol, so a
# backend written outside this package satisfies it with the two-argument
# `send` that has always been the contract -- and calling it with a keyword it
# does not take would be a TypeError inside the one call the whole solve
# depends on. Catching that TypeError is not an option either: one raised from
# INSIDE `send` is indistinguishable, and retrying would send the prompt twice.
_EXTEND_SUPPORT: dict[type, bool] = {}


def _reads_past_its_slice(conversation: Any) -> bool:
    cls = type(conversation)
    known = _EXTEND_SUPPORT.get(cls)
    if known is None:
        try:
            known = "extend_to_s" in inspect.signature(cls.send).parameters
        except (TypeError, ValueError):  # builtins, C callables, no signature
            known = False
        _EXTEND_SUPPORT[cls] = known
    return known


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
        max_attempts: int = 3,
        safety_margin_s: float = 20.0,
        max_budget_s: float = 3600.0,
        cache_size: int = 256,
        second_opinion: bool = True,
    ):
        self._backend = backend
        self._max_attempts = max(1, int(max_attempts))
        self._margin = max(0.0, float(safety_margin_s))
        self._max_budget = max(5.0, float(max_budget_s))
        self._second_opinion = bool(second_opinion)
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
            budget = max(5.0, float(timeout_s) * 0.5)

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
                task, remaining, avoid=asked[-1] if asked else None
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
                if not self._warned_ungradeable:
                    self._warned_ungradeable = True
                    why = (
                        "no public examples shipped with this task"
                        if not task.public_examples
                        else "the public examples could not be run here"
                    )
                    print(
                        f"[verify] {why}, so nothing can be graded locally: no "
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
            f"examples={best.passed}/{best.total} verified={best.verified} "
            f"{elapsed:.1f}s/{budget:.0f}s"
        )
        return Answer(
            code=best.code, raw_response=best.raw,
            verified=best.verified, passed=best.passed, total=best.total,
        )

    async def _attempt(
        self, task, remaining: float, avoid: Optional[str]
    ) -> tuple[Optional[Candidate], Optional[str]]:
        """One model, one conversation: initial answer plus repair rounds.

        The repair rounds deliberately stay in that single conversation so the
        model sees its own previous attempt beside the failure report. Returns
        the best candidate it produced and which provider produced it.
        """
        started = time.monotonic()
        budget = max(5.0, remaining)
        best: Optional[Candidate] = None
        conversation = None
        provider: Optional[str] = None
        try:
            conversation = await self._backend.open(avoid=avoid)
            provider = getattr(conversation, "provider", None)
            # The budget is told to the MODEL, not just enforced against it.
            # Everything above this line rations the clock on the model's
            # behalf; nothing until now let the model ration it for itself, and
            # a model rationing an unknown either hurries a generous budget
            # away or explores a tight one until the reply is cut off. Both pay
            # zero. Measured after the lease, so the number is the truth rather
            # than what it was before opening a conversation cost anything.
            prompt = build_initial_prompt(
                task.language, task.statement, task.entrypoint, task.public_examples,
                budget_s=budget - (time.monotonic() - started),
            )
            for attempt in range(1, self._max_attempts + 1):
                left = budget - (time.monotonic() - started)
                if left < 12.0:
                    break  # not enough left to be worth another round trip
                # Give the first attempt the larger share; repairs are cheaper.
                #
                # How much larger depends on whether a repair can even happen.
                # With public examples a repair is likely, and reserving 40% for
                # it is well spent. With NONE -- every task on the run this was
                # written for -- the only repair possible is defect-driven, and
                # a first answer that is structurally fine ends the loop right
                # there. Measured: a Claude tab spent its whole 135 second slice
                # and the remaining 90 seconds of a 225 second budget went
                # unused, on the one attempt that had to succeed.
                first_share = 0.6 if task.public_examples else 0.85
                slice_s = (
                    left if attempt == self._max_attempts else left * first_share
                )

                # The slice is what this read is ALLOCATED; `left` is what the
                # attempt actually still has. Handing over both lets the read
                # spend the repair reserve on an answer that is still arriving,
                # which is the only thing that can still produce one.
                if _reads_past_its_slice(conversation):
                    reply = await conversation.send(prompt, slice_s, extend_to_s=left)
                else:
                    reply = await conversation.send(prompt, slice_s)
                candidate = await self._graded(
                    reply, task, budget - (time.monotonic() - started)
                )
                if best is None or candidate.score > best.score:
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
                        f"the budget ran out; submitting what arrived rather than "
                        f"interrupting it with a repair prompt"
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
                    print(
                        f"[verify] {provider or 'this model'} returned nothing and "
                        f"the conversation is "
                        f"{getattr(conversation, 'empty_reason', '?')}; asking "
                        f"elsewhere rather than repairing it"
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
                    budget_s=budget - (time.monotonic() - started),
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

    async def _graded(self, reply: str, task, left: float) -> Candidate:
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
            return await asyncio.to_thread(self._grade, reply, task, left)
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
    def _grade(self, reply: str, task, left: Optional[float] = None) -> Candidate:
        code = extract_code(reply, task.entrypoint, task.language)
        candidate = Candidate(code=code, raw=reply)
        defect = (
            rust_defect(code)
            if task.language == "rust"
            else python_defect(code, task.entrypoint)
        )
        if defect is None and task.language == "rust":
            # Python's check PARSED that code; Rust's only grepped it for
            # `fn main`. Ask the compiler the same question the validator will,
            # which is the only check a Rust answer gets at all when no public
            # examples shipped -- and on the run this was written for, none
            # ever did. Returns None when there is no local toolchain.
            #
            # `left` caps it: a compile is allowed to be slow, but not slower
            # than the answer it is checking is worth. See `compile_defect`.
            defect = compile_defect(code, left)
        if defect is not None:
            # Structurally unusable: report it without paying for execution.
            candidate.defect = defect
            candidate.code = "" if not code.strip() else code
            return candidate
        if not task.public_examples:
            return candidate  # nothing to verify against; take it as-is
        if left is not None and left <= 0:
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

    def stats(self) -> dict[str, Any]:
        return {
            "solver": dict(self._counts),
            "providers": {k: dict(v) for k, v in self._by_provider.items()},
            "fleet": self._backend.stats(),
        }

    async def aclose(self) -> None:
        await self._backend.aclose()


def _cache_key(task) -> str:
    import hashlib

    material = f"{task.language}\0{task.entrypoint}\0{task.statement}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

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

    @staticmethod
    def _build_settings():
        from rlvr.config import Settings

        kind = os.environ.get("SOLVER_VERIFY_EXECUTOR", "subprocess")
        # _env_file=None so the miner's .env cannot accidentally repoint this.
        return Settings(_env_file=None, executor=kind, per_test_timeout_s=VERIFY_TIMEOUT_S)

    def executor(self, language: str):
        if language in self._cache:
            return self._cache[language]
        from rlvr.execution.executor import get_executor

        settings = self._settings
        if language == "rust" and settings.executor != "docker":
            # Rust needs rustc in the pinned image; there is no subprocess path.
            from rlvr.config import Settings

            settings = Settings(
                _env_file=None, executor="docker", per_test_timeout_s=VERIFY_TIMEOUT_S
            )
        executor = get_executor(settings, language=language)
        self._cache[language] = executor
        return executor

    def check(
        self, code: str, language: str, entrypoint: str, examples: list[dict[str, Any]]
    ) -> tuple[int, int, list[str]]:
        """Run ``code`` against the public examples. Returns (passed, total, failures)."""
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
        for result, case in zip(results, cases):
            if result.passed:
                passed += 1
                continue
            failures.append(_describe(result, case, language, entrypoint))
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
        safety_margin_s: float = 15.0,
        max_budget_s: float = 240.0,
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

    # -- the Solver interface custom_miner.py expects ---------------------- #
    async def solve_task(self, task, timeout_s: float) -> Answer:
        started = time.monotonic()
        # The validator advertises one deadline but may enforce the problem
        # server's shorter one, so never spend the full advertised budget.
        budget = min(float(timeout_s), self._max_budget) - self._margin
        if budget <= 5.0:
            budget = max(5.0, float(timeout_s) * 0.5)

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
        passes = 2 if self._second_opinion else 1
        for attempt_no in range(passes):
            remaining = budget - (time.monotonic() - started)
            # The first pass always runs, however little is left: bailing here
            # would return nothing having asked nobody.
            if attempt_no and remaining < 20.0:
                print(f"[verify] {remaining:.0f}s left; no time for a second opinion")
                break
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
            if not task.public_examples:
                if not self._warned_ungradeable:
                    self._warned_ungradeable = True
                    print(
                        "[verify] no public examples shipped with this task, so "
                        "nothing can be graded locally: no repair rounds, and "
                        "verified=False however good the answer is. Once per run."
                    )
                # Nothing can rank two ungradeable answers apart -- `score` ties
                # at (0, has_code) -- so a second opinion is only ever worth
                # buying when this one came back EMPTY. Then it is worth a lot:
                # an empty answer scores zero, and the other model is the only
                # remaining chance at the whole payment.
                if best.code.strip():
                    break
            if attempt_no + 1 < passes:
                print(f"[verify] {provider or 'first'} did not verify; asking another model")
        if asked:
            self._note(asked[-1] if best.verified else None, asked)

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
            prompt = build_initial_prompt(
                task.language, task.statement, task.entrypoint, task.public_examples
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

                reply = await conversation.send(prompt, slice_s)
                candidate = self._grade(reply, task)
                if best is None or candidate.score > best.score:
                    best = candidate
                if candidate.verified:
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

    def _note(self, winner: Optional[str], asked: list[str]) -> None:
        """Per-provider tally, so a model that has started failing is visible."""
        for name in asked:
            row = self._by_provider.setdefault(name, {"asked": 0, "verified": 0})
            row["asked"] += 1
        if winner:
            self._by_provider[winner]["verified"] += 1

    # ---------------------------------------------------------------------- #
    def _grade(self, reply: str, task) -> Candidate:
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
            defect = compile_defect(code)
        if defect is not None:
            # Structurally unusable: report it without paying for execution.
            candidate.defect = defect
            candidate.code = "" if not code.strip() else code
            return candidate
        if not task.public_examples:
            return candidate  # nothing to verify against; take it as-is
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

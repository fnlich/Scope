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
    async def open(self) -> Conversation: ...
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
    def score(self) -> tuple[int, int]:
        """Ranking key for 'best so far' — pass count, then having any code."""
        return (self.passed, 1 if self.code.strip() else 0)


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
    ):
        self._backend = backend
        self._max_attempts = max(1, int(max_attempts))
        self._margin = max(0.0, float(safety_margin_s))
        self._max_budget = max(5.0, float(max_budget_s))
        self._grader = _Grader()
        self._cache: dict[str, tuple[str, str]] = {}
        self._cache_size = max(0, int(cache_size))
        self._counts = {"solved": 0, "verified": 0, "cache_hits": 0, "empty": 0}

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
        conversation = None
        try:
            conversation = await self._backend.open()
            prompt = build_initial_prompt(
                task.language, task.statement, task.entrypoint, task.public_examples
            )
            for attempt in range(1, self._max_attempts + 1):
                remaining = budget - (time.monotonic() - started)
                if remaining < 12.0:
                    break  # not enough left to be worth another round trip
                # Give the first attempt the larger share; repairs are cheaper.
                slice_s = remaining if attempt == self._max_attempts else remaining * 0.6

                reply = await conversation.send(prompt, slice_s)
                candidate = self._grade(reply, task)
                if candidate.score > best.score:
                    best = candidate
                if candidate.verified:
                    self._counts["verified"] += 1
                    break
                problems = ([candidate.defect] if candidate.defect else []) + candidate.failures
                if not problems:
                    break  # nothing actionable to report (no examples shipped)
                prompt = build_repair_prompt(problems, task.language, task.entrypoint)
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
            f"examples={best.passed}/{best.total} verified={best.verified} "
            f"{elapsed:.1f}s/{budget:.0f}s"
        )
        return Answer(
            code=best.code, raw_response=best.raw,
            verified=best.verified, passed=best.passed, total=best.total,
        )

    # ---------------------------------------------------------------------- #
    def _grade(self, reply: str, task) -> Candidate:
        code = extract_code(reply)
        candidate = Candidate(code=code, raw=reply)
        defect = (
            rust_defect(code)
            if task.language == "rust"
            else python_defect(code, task.entrypoint)
        )
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
        return {"solver": dict(self._counts), "backend": self._backend.stats()}

    async def aclose(self) -> None:
        await self._backend.aclose()


def _cache_key(task) -> str:
    import hashlib

    material = f"{task.language}\0{task.entrypoint}\0{task.statement}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

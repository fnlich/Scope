"""Use several model providers for one task, and pick backends from the env.

The subnet pays only for a complete hidden-suite pass, so a second opinion is
worth far more than a faster first one: the latency tiebreaker spans about 3.5%,
while being wrong costs 100%. ``FallbackSolver`` therefore runs providers in
order and stops at the first answer that reproduces every public example.

Ordering is a cost decision, not a quality one — put the provider you would
rather pay for first. A verified answer ends the chain immediately, so later
providers cost nothing on tasks the first one solves.

    MINER_BACKENDS=claude,gemini,chatgpt   # in preference order
    MINER_BACKENDS=claude                  # just one

Every backend implements the same protocol, so adding another is one class.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from .verify import Answer, VerifyingSolver

# Backends that need no browser. `chatgpt` is imported lazily below because it
# pulls in Playwright, which most deployments of the API backends will not have.
_API_BACKENDS = {"claude", "gemini"}
KNOWN_BACKENDS = sorted(_API_BACKENDS | {"chatgpt"})


def build_backend(name: str):
    """Construct one backend by name."""
    key = name.strip().lower()
    if key == "claude":
        from .claude_api import ClaudeBackend

        return ClaudeBackend()
    if key == "gemini":
        from .gemini_api import GeminiBackend

        return GeminiBackend()
    if key == "chatgpt":
        from .chatgpt_cdp import ChatGPTPool

        ports = [
            int(p) for p in os.environ.get("CHATGPT_PORTS", "9222").replace(",", " ").split()
        ]
        return ChatGPTPool(
            ports,
            host=os.environ.get("CHATGPT_HOST", "127.0.0.1"),
            tabs_per_browser=int(os.environ.get("CHATGPT_TABS_PER_BROWSER", "2")),
        )
    raise SystemExit(f"unknown backend {name!r}; expected one of {KNOWN_BACKENDS}")


def _tuning() -> dict[str, Any]:
    return dict(
        max_attempts=int(os.environ.get("SOLVER_MAX_ATTEMPTS", "3")),
        safety_margin_s=float(os.environ.get("SOLVER_SAFETY_MARGIN_S", "15")),
        max_budget_s=float(os.environ.get("SOLVER_MAX_BUDGET_S", "240")),
    )


class FallbackSolver:
    """Try each provider in turn; stop at the first verified answer.

    The whole budget is not handed to the first provider: it is divided so a
    later one still has room to work. A provider that returns an unverified
    answer is not wasted — the best-ranked candidate across all of them is
    returned if none verify, which is still better than nothing.
    """

    def __init__(
        self,
        solvers: list[tuple[str, VerifyingSolver]],
        *,
        safety_margin_s: float = 15.0,
    ):
        if not solvers:
            raise SystemExit("FallbackSolver needs at least one backend")
        self._solvers = solvers
        self._margin = max(0.0, float(safety_margin_s))
        self._wins: dict[str, int] = {name: 0 for name, _ in solvers}
        self._attempts: dict[str, int] = {name: 0 for name, _ in solvers}

    async def solve_task(self, task, timeout_s: float) -> Answer:
        started = time.monotonic()
        budget = max(5.0, float(timeout_s) - self._margin)
        best = Answer(code="")

        for index, (name, solver) in enumerate(self._solvers):
            elapsed = time.monotonic() - started
            remaining = budget - elapsed
            if remaining < 12.0:
                break  # not enough left for another provider to be useful
            # Give each remaining provider an equal share of what is left, so a
            # slow first provider cannot starve the rest.
            share = remaining / max(1, len(self._solvers) - index)
            slice_s = remaining if index == len(self._solvers) - 1 else share

            self._attempts[name] += 1
            answer = await solver.solve_task(task, slice_s)
            if answer.verified:
                self._wins[name] += 1
                print(f"[multi] {name} verified in {time.monotonic() - started:.1f}s")
                return answer
            if (answer.passed, bool(answer.code.strip())) > (
                best.passed, bool(best.code.strip())
            ):
                best = answer
            print(
                f"[multi] {name} unverified ({answer.passed}/{answer.total}); "
                f"{'trying next' if index + 1 < len(self._solvers) else 'no provider left'}"
            )
        return best

    def stats(self) -> dict[str, Any]:
        return {
            "chain": [name for name, _ in self._solvers],
            "verified_by": dict(self._wins),
            "attempts": dict(self._attempts),
            "providers": {name: s.stats() for name, s in self._solvers},
        }

    async def aclose(self) -> None:
        for _, solver in self._solvers:
            try:
                await solver.aclose()
            except Exception:  # noqa: BLE001 - one bad shutdown is not the others'
                pass


def build_solver(names: Optional[list[str]] = None):
    """Build the solver named by ``MINER_BACKENDS`` (default: claude).

    One backend gives a plain ``VerifyingSolver``; several give a
    ``FallbackSolver`` over them, in the order listed.
    """
    if names is None:
        raw = os.environ.get("MINER_BACKENDS", "claude")
        names = [n for n in raw.replace(",", " ").split() if n]
    if not names:
        raise SystemExit("MINER_BACKENDS is empty")

    tuning = _tuning()
    solvers = [(n.lower(), VerifyingSolver(build_backend(n), **tuning)) for n in names]
    if len(solvers) == 1:
        return solvers[0][1]
    return FallbackSolver(solvers, safety_margin_s=tuning["safety_margin_s"])

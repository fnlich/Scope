"""Use several model providers for one task, and pick backends from the env.

The subnet pays only for a complete hidden-suite pass, so a second opinion is
worth far more than a faster first one: the latency tiebreaker spans 5 points
(`payment_speed_floor` is 0.95, so the multiplier lives in [0.95, 1.0]), while
being wrong costs 100%. ``FallbackSolver`` therefore runs providers in
order and stops at the first answer that reproduces every public example.

Ordering is a cost decision, not a quality one — put the provider you would
rather pay for first. A verified answer ends the chain immediately, so later
providers cost nothing on tasks the first one solves.

    MINER_BACKENDS=claude,chatgpt   # in preference order
    MINER_BACKENDS=claude           # just one

Both backends drive a Chrome you started and signed in to yourself, attached
over CDP. No API key is read anywhere in this package.

Running both is also the only redundancy a browser miner has. There is no API
path to fall back to, so when one provider's DOM changes or its login expires,
the other is what stops the score going to zero — see the fallback chain below.

Every backend implements the same protocol, so adding another is one class.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from .verify import Answer, VerifyingSolver

# Both backends attach to a logged-in Chrome; neither reads an API key.
# Imported lazily inside build_backend so that importing this module does not
# require Playwright.
KNOWN_BACKENDS = ["chatgpt", "claude"]


def _pool_kwargs(prefix: str) -> dict[str, Any]:
    """Which browsers a backend attaches to, and how many tabs in each.

    ``<PREFIX>_CDP`` is the endpoint list — a bare port, ``host:port``, a full
    URL, or several separated by commas, one per account. Unset, it defaults to
    the port ``scripts/start_debug_browser.sh`` uses, so the common
    one-browser setup needs no configuration at all.
    """
    from .browser_pool import DEFAULT_CDP_PORT

    return dict(
        cdp=os.environ.get(f"{prefix}_CDP", "").strip() or str(DEFAULT_CDP_PORT),
        tabs_per_browser=int(os.environ.get(f"{prefix}_TABS_PER_BROWSER", "2")),
    )


def build_backend(name: str):
    """Construct one backend by name."""
    key = name.strip().lower()
    if key == "claude":
        # The browser, not an API key: your Claude subscription is the quota.
        from .claude_web import ClaudeBrowserPool

        return ClaudeBrowserPool(**_pool_kwargs("CLAUDE"))
    if key == "chatgpt":
        from .chatgpt_web import ChatGPTPool

        return ChatGPTPool(**_pool_kwargs("CHATGPT"))
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
            # Never skip the FIRST provider. With a short deadline the margin can
            # eat the whole budget, and bailing here would return an empty answer
            # having called nobody — no attempt recorded, nothing logged, a
            # guaranteed zero that looks like the model failed. VerifyingSolver
            # has the same floor for the same reason.
            if remaining < 12.0 and index > 0:
                print(f"[multi] {remaining:.0f}s left; stopping before {name}")
                break
            if remaining < 12.0:
                remaining = max(5.0, float(timeout_s) * 0.5)
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


def backends(solver) -> list[Any]:
    """Every leaf backend behind a solver — one, or a whole chain of them."""
    chain = getattr(solver, "_solvers", None)
    if chain is None:
        return [solver._backend]
    return [inner._backend for _, inner in chain]


async def warm_up(solver, min_capacity: int = 1) -> None:
    """Start any browser pool before serving, and warn if capacity is short.

    Lazy start already covers correctness. What it does not cover is
    visibility: an expired login would otherwise first surface as a failed
    solve on a real validator request, minutes or hours later, instead of at
    launch where someone is watching.
    """
    for backend in backends(solver):
        start = getattr(backend, "start", None)
        if start is None:
            continue  # a backend with nothing to warm up
        await start()
        tabs = backend.stats().get("tabs", 0)
        if tabs < min_capacity:
            print(
                f"[multi] NOTE: {backend.site.name} has {tabs} tab(s) but "
                f"MINER_MAX_CONCURRENT_REQUESTS={min_capacity}. Tasks beyond the "
                "tab count queue and burn their deadline — add browsers/tabs, or "
                "lower the concurrency."
            )

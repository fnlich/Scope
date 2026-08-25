"""Turn the browsers you started into one fleet, from the environment.

Your side of the deal: run N browsers, each signed in to one provider, each on
its own debugging port. This side: attach to all of them and treat them as a
single fleet of conversation slots.

    CLAUDE_CDP=9222,9223,9224      # three browsers signed in to claude.ai
    CHATGPT_CDP=9225,9226          # two signed in to chatgpt.com

Set either, or both. Tasks are spread across whatever is listed — a task does
not care which model answers it, so the useful unit is "the next free tab",
which is also what spreads load over your accounts.

There is deliberately no provider-preference setting. Naming one provider
"first" would mean queueing on its browsers while the others idle, which is the
opposite of what a fleet is for. The only place the provider is consulted is the
second opinion below, and there it is used to pick a DIFFERENT one.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from .browser_pool import DEFAULT_CDP_PORT, Browser, BrowserFleet, Site, normalize_cdp
from .verify import VerifyingSolver

PROVIDERS = ("claude", "chatgpt")


def site_for(provider: str) -> Site:
    if provider == "claude":
        from .claude_web import claude_site

        return claude_site()
    if provider == "chatgpt":
        from .chatgpt_web import chatgpt_site

        return chatgpt_site()
    raise SystemExit(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def roster(env: Optional[dict[str, str]] = None) -> list[Browser]:
    """Every browser named in the environment, in provider order.

    Nothing set at all is treated as one Claude browser on the default port —
    the single-browser case then needs no configuration, matching what
    ``scripts/start_debug_browser.sh`` starts by default.
    """
    source = os.environ if env is None else env
    browsers: list[Browser] = []
    for provider in PROVIDERS:
        for endpoint in normalize_cdp(source.get(f"{provider.upper()}_CDP", "")):
            browsers.append(Browser(endpoint, site_for(provider)))
    if not browsers:
        browsers.append(
            Browser(normalize_cdp(str(DEFAULT_CDP_PORT))[0], site_for("claude"))
        )
    seen, unique = set(), []
    for browser in browsers:
        # One browser cannot be signed in to two providers at once in the same
        # profile, and attaching twice would double-count its capacity.
        if browser.endpoint in seen:
            print(
                f"[fleet] WARN: {browser.endpoint} is listed more than once; "
                f"using it as {browser.site.name} only"
            )
            continue
        seen.add(browser.endpoint)
        unique.append(browser)
    return unique


def tabs_per_browser(env: Optional[dict[str, str]] = None) -> int:
    source = os.environ if env is None else env
    return int(source.get("MINER_TABS_PER_BROWSER", "2"))


def build_solver(browsers: Optional[Sequence[Browser]] = None) -> VerifyingSolver:
    """The fleet, wrapped in the self-verify-and-repair loop."""
    fleet = BrowserFleet(
        list(browsers) if browsers is not None else roster(),
        tabs_per_browser=tabs_per_browser(),
    )
    return VerifyingSolver(
        fleet,
        max_attempts=int(os.environ.get("SOLVER_MAX_ATTEMPTS", "3")),
        safety_margin_s=float(os.environ.get("SOLVER_SAFETY_MARGIN_S", "15")),
        max_budget_s=float(os.environ.get("SOLVER_MAX_BUDGET_S", "240")),
        second_opinion=os.environ.get("SOLVER_SECOND_OPINION", "true").strip().lower()
        not in ("0", "false", "no"),
    )


async def warm_up(solver, min_capacity: int = 1) -> None:
    """Attach and fill the fleet before serving, and warn if capacity is short.

    Lazy start already covers correctness. What it does not cover is visibility:
    a browser that is not signed in would otherwise first surface as a failed
    solve on a real validator request, minutes or hours later, instead of at
    launch where someone is watching.
    """
    fleet = getattr(solver, "_backend", None)
    start = getattr(fleet, "start", None)
    if start is None:
        return
    await start()
    tabs = fleet.stats().get("tabs", 0)
    if tabs < min_capacity:
        print(
            f"[fleet] NOTE: {tabs} tab(s) but MINER_MAX_CONCURRENT_REQUESTS="
            f"{min_capacity}. Tasks beyond the tab count queue and burn their "
            f"deadline — add a browser, raise MINER_TABS_PER_BROWSER, or lower "
            f"the concurrency."
        )


def describe(browsers: Sequence[Browser]) -> str:
    counts: dict[str, int] = {}
    for browser in browsers:
        counts[browser.site.name] = counts.get(browser.site.name, 0) + 1
    return ", ".join(f"{n} {c}" for c, n in sorted(counts.items())) or "no browsers"


def stats(solver) -> dict[str, Any]:
    return solver.stats()

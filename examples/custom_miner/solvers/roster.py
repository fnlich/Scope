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


def backend_kind(env: Optional[dict[str, str]] = None) -> str:
    """Which backend answers: ``browser`` (default) or ``cli``.

    The browser fleet stays the default because it is what every existing
    deployment is configured for, and a solver that silently changed where the
    answers came from would be the worst kind of upgrade. `SOLVER_BACKEND=cli`
    opts in.
    """
    source = os.environ if env is None else env
    kind = (source.get("SOLVER_BACKEND", "") or "browser").strip().lower()
    if kind not in ("browser", "cli"):
        raise SystemExit(
            f"SOLVER_BACKEND={kind!r}; expected 'browser' or 'cli'"
        )
    return kind


def roster(env: Optional[dict[str, str]] = None) -> list[Browser]:
    """Every browser named in the environment, in provider order.

    Nothing set at all is treated as one Claude browser on the default port —
    the single-browser case then needs no configuration, matching what
    ``scripts/start_debug_browser.sh`` starts by default.
    """
    source = os.environ if env is None else env
    if backend_kind(source) == "cli":
        # No browsers to attach, and saying so beats defaulting to one on the
        # standard port and reporting that it could not be reached.
        return []
    browsers: list[Browser] = []
    for provider in PROVIDERS:
        for endpoint in normalize_cdp(source.get(f"{provider.upper()}_CDP", "")):
            browsers.append(Browser(endpoint, site_for(provider)))
    if not browsers:
        browsers.append(
            Browser(normalize_cdp(str(DEFAULT_CDP_PORT))[0], site_for("claude"))
        )
    # One endpoint, one browser, one provider. Attaching to the same browser
    # twice is not merely double-counted capacity: `_fill` reclaims a browser's
    # leftover tabs each time it attaches, so the second entry would close the
    # tabs the first had just spawned. Sign a second provider in to a SECOND
    # browser on its own port instead.
    kept: dict[str, Browser] = {}
    unique: list[Browser] = []
    for browser in browsers:
        winner = kept.get(browser.endpoint)
        if winner is not None:
            print(
                f"[fleet] WARN: {browser.endpoint} is listed under both "
                f"{winner.site.name.upper()}_CDP and "
                f"{browser.site.name.upper()}_CDP. Serving it as "
                f"{winner.site.name} only — {browser.site.name} on that port is "
                f"ignored. Give {browser.site.name} its own browser on its own "
                f"port to use both."
            )
            continue
        kept[browser.endpoint] = browser
        unique.append(browser)
    return unique


def tabs_per_browser(env: Optional[dict[str, str]] = None) -> int:
    source = os.environ if env is None else env
    return int(source.get("MINER_TABS_PER_BROWSER", "2"))


def build_solver(browsers: Optional[Sequence[Browser]] = None) -> VerifyingSolver:
    """The backend, wrapped in the self-verify-and-repair loop.

    Everything below the first two lines is the same whichever backend answers.
    That is the point of `Backend` being a protocol: the repair loop, the
    grading, the budget discipline and the archive have no idea whether a turn
    came from a browser tab or a `claude` subprocess, and none of them had to
    change to gain one.
    """
    if backend_kind() == "cli":
        from .claude_cli import CliBackend

        fleet: Any = CliBackend()
    else:
        fleet = BrowserFleet(
            list(browsers) if browsers is not None else roster(),
            tabs_per_browser=tabs_per_browser(),
        )
    return VerifyingSolver(
        fleet,
        # 0 = keep correcting until the answer passes or the request's deadline
        # stops it. A count here is a second, private deadline: the loop is what
        # turns a nearly-right answer into a paid one, and there is no partial
        # credit for stopping early. Set it to cap the rounds anyway.
        max_attempts=int(os.environ.get("SOLVER_MAX_ATTEMPTS", "0")),
        # The largest deadline the protocol allows (`TaskRequest.deadline_s` is
        # `le=3600`), so it cannot bind on a spec-compliant request: the budget
        # is the deadline the validator advertised, minus what delivering the
        # answer costs, and nothing else. A smaller value here is a second,
        # private deadline that throws away answers the validator would still
        # have paid for -- see VerifyingSolver.solve_task.
        max_budget_s=float(os.environ.get("SOLVER_MAX_BUDGET_S", "3600")),
        second_opinion=os.environ.get("SOLVER_SECOND_OPINION", "true").strip().lower()
        not in ("0", "false", "no"),
        # The model's own cases, run with the validator's executor. Live traffic
        # ships no `public_examples` at all, so without these there is nothing to
        # grade and the repair loop never fires. `SOLVER_SELF_TESTS=0` turns the
        # cases turn off: one round trip per solve instead of two, and every
        # answer submitted ungraded. It no longer ASKS for cases either -- the
        # combined prompt requested a second block that this switch then told
        # the grader to ignore, so the model spent output tokens inside the
        # deadline writing something nothing read.
        self_tests=os.environ.get("SOLVER_SELF_TESTS", "true").strip().lower()
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
    # Whatever this backend calls a slot. A browser fleet counts tabs; the CLI
    # counts processes it will run at once. Reading only `tabs` reported the CLI
    # backend as having no capacity at all and advised adding a browser to it.
    stats = fleet.stats()
    slots = stats.get("tabs", stats.get("concurrency", 0))
    if slots < min_capacity:
        unit = "tab(s)" if "tabs" in stats else "slot(s)"
        remedy = (
            "add a browser, raise MINER_TABS_PER_BROWSER, or lower the "
            "concurrency"
            if "tabs" in stats else
            "raise SOLVER_CLI_CONCURRENCY, or lower the concurrency"
        )
        print(
            f"[fleet] NOTE: {slots} {unit} but MINER_MAX_CONCURRENT_REQUESTS="
            f"{min_capacity}. Tasks beyond that queue and burn their "
            f"deadline — {remedy}."
        )


def describe(browsers: Sequence[Browser]) -> str:
    if not browsers and backend_kind() == "cli":
        from .claude_cli import cli_backup_dirs, cli_effort, cli_emergency_profiles, cli_models

        effort = cli_effort()
        ladder = [f"{cli_models()[0]}/{effort}"]
        ladder += [p.label for p in cli_emergency_profiles(effort) if p.label not in ladder]
        seats = 1 + len(cli_backup_dirs())
        return (f"claude CLI ({' > '.join(ladder)}; "
                f"{seats} account{'s' if seats != 1 else ''})")
    counts: dict[str, int] = {}
    for browser in browsers:
        counts[browser.site.name] = counts.get(browser.site.name, 0) + 1
    return ", ".join(f"{n} {c}" for c, n in sorted(counts.items())) or "no browsers"


def stats(solver) -> dict[str, Any]:
    return solver.stats()

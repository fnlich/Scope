"""Check a browser backend's selectors against your own logged-in Chrome.

A browser-backed miner fails silently: a DOM change looks exactly like an idle
miner, and by the time the score drops the zeros are already inside the
200-observation window. This is the five-second check that stops that.

    cd examples/custom_miner
    python -m solvers.doctor claude
    python -m solvers.doctor claude --probe      # also send a real prompt
    python -m solvers.doctor chatgpt --port 9223

It reports, for every role, which candidate selector your page actually has,
and it flags the three mistakes that matter: no composer (not logged in, or the
DOM moved), a "still generating" selector that matches an idle page (every
answer would look unfinished forever), and an assistant selector that already
matches in an empty conversation (it is matching something that is not a reply
— possibly your own message, which would make the miner answer with its own
prompt). Every role is overridable in ``.env`` with ``|`` between candidates.

``--probe`` goes further and drives the real ``_Tab.send`` path with a trivial
prompt, so what you see is exactly what the miner would read.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .cdp_pool import Site, import_playwright, usable_busy_selectors, wait_for_any
from .config import find_env_file, load_env_file

PROBE = (
    "Reply with exactly one fenced Python code block containing only "
    "`def pong():\n    return 'pong'` and nothing else."
)


def _site(name: str) -> Site:
    if name == "claude":
        from .claude_cdp import claude_site

        return claude_site()
    if name == "chatgpt":
        from .chatgpt_cdp import chatgpt_site

        return chatgpt_site()
    raise SystemExit(f"unknown backend {name!r}; expected 'claude' or 'chatgpt'")


async def _report_role(
    page, label: str, candidates, *, expect_zero: bool = False, on_miss: str = ""
) -> bool:
    """Print one role's candidates. Returns True if the role looks healthy."""
    print(f"  {label}:")
    winner = None
    for selector in candidates:
        try:
            count = await page.locator(selector).count()
        except Exception as exc:  # noqa: BLE001 - a malformed selector is a finding
            print(f"    [ERR ] {selector}  ({type(exc).__name__})")
            continue
        mark = "MATCH" if count else "  -  "
        print(f"    [{mark}] {selector}   ({count} node(s))")
        if count and winner is None:
            winner = selector
    if expect_zero:
        if winner is not None:
            print(
                f"    !! {label} already matches in an empty conversation. It is "
                "matching something that is not a reply — verify it before use."
            )
            return False
        print("    ok: nothing matches yet, as expected in an empty conversation")
        return True
    if winner is None:
        print("    !! nothing matched. Set the override and re-run.")
        if on_miss:
            print(f"       {on_miss}")
        return False
    print(f"    -> using {winner}")
    return True


async def run(name: str, port: int, host: str, probe: bool) -> int:
    site = _site(name)
    async_playwright = import_playwright()
    print(f"[doctor] {site.name}: attaching to http://{host}:{port}")
    pw = await async_playwright().start()
    try:
        try:
            browser = await pw.chromium.connect_over_cdp(f"http://{host}:{port}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[doctor] FAIL: cannot attach: {exc}\n"
                f"         Start Chrome with --remote-debugging-port={port} "
                f"--user-data-dir=<dir> and log in to {site.url}."
            )
            return 2
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        healthy = True
        try:
            print(f"[doctor] opening {site.url}")
            await page.goto(site.url, wait_until="domcontentloaded")
            composer = await wait_for_any(page, site.composer, site.ready_timeout_ms)
            if composer is None:
                print("[doctor] FAIL: no composer selector matched.")
                await _report_role(page, "composer", site.composer)
                print(
                    "         Most often this means the profile is not logged in. "
                    f"Log in to {site.url} in that Chrome window and re-run."
                )
                return 1
            print(f"[doctor] page ready (composer: {composer})\n")
            healthy &= await _report_role(page, "composer", site.composer)
            healthy &= await _report_role(
                page,
                "send button",
                site.send,
                on_miss="Not fatal: with no send button the prompt is submitted with "
                "Enter, which these composers accept. Fix it anyway if you can.",
            )
            print("  still-generating (must NOT match an idle page):")
            kept = await usable_busy_selectors(page, site.busy, site.name)
            for selector in site.busy:
                print(f"    [{'keep ' if selector in kept else 'DROP '}] {selector}")
            if not kept:
                print(
                    "    note: no usable busy selector. Answers are still detected by "
                    "text going unchanged across two polls, but expect them to be "
                    "accepted a little early or late."
                )
            print()
            healthy &= await _report_role(
                page, "assistant message", site.assistant, expect_zero=True
            )
            if site.message_id_attr:
                print(f"\n  message id attribute: {site.message_id_attr}")
            else:
                print("\n  message id attribute: none — replies identified by position")

            if probe:
                healthy &= await _probe(page, site, composer, kept)
        finally:
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"\n[doctor] {'OK' if healthy else 'PROBLEMS FOUND (see !! above)'}")
        return 0 if healthy else 1
    finally:
        await pw.stop()


async def _probe(page, site: Site, composer: str, busy) -> bool:
    """Send a real prompt through the real read path."""
    from .cdp_pool import _Tab

    class _Solo:
        """Minimal pool: the probe never releases the tab."""

        def __init__(self, site: Site):
            self.site = site

        async def release(self, tab) -> None:
            pass

    print("\n[doctor] probing with a real prompt (up to 120s)...")
    from dataclasses import replace

    tab = _Tab(_Solo(replace(site, busy=busy)), page, None, "probe", composer=composer)
    reply = await tab.send(PROBE, 120.0)
    if not reply.strip():
        print("[doctor] !! the probe read nothing back. The assistant selector or the")
        print("            still-generating selector is wrong for this page.")
        return False
    from .prompts import extract_code

    print(f"[doctor] read {len(reply)} chars back:")
    print("    " + "\n    ".join(reply.strip().splitlines()[:12]))
    code = extract_code(reply)
    if "pong" not in code:
        print("[doctor] !! that does not look like the answer to the probe prompt.")
        return False
    print("[doctor] probe answered correctly — the read path works end to end.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m solvers.doctor")
    parser.add_argument("backend", choices=("claude", "chatgpt"))
    parser.add_argument("--port", type=int, default=None, help="CDP port")
    parser.add_argument("--host", default=None, help="CDP host")
    parser.add_argument(
        "--probe", action="store_true", help="also send a real prompt and read it back"
    )
    args = parser.parse_args(argv)
    # preflight.py sits beside custom_miner.py, one level above this package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preflight import require_linux

    require_linux("The browser-backend doctor")
    found = find_env_file()
    if load_env_file(found):
        print(f"[doctor] loaded {found}")

    prefix = "CLAUDE" if args.backend == "claude" else "CHATGPT"
    port = args.port
    if port is None:
        raw = os.environ.get(f"{prefix}_PORTS", "9222").replace(",", " ").split()
        port = int(raw[0]) if raw else 9222
    host = args.host or os.environ.get(f"{prefix}_HOST", "127.0.0.1")
    return asyncio.run(run(args.backend, port, host, args.probe))


if __name__ == "__main__":
    sys.exit(main())

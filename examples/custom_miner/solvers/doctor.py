"""Check a browser backend's selectors against your own logged-in Chrome.

A browser-backed miner fails silently: a DOM change looks exactly like an idle
miner, and by the time the score drops the zeros are already inside the
200-observation window. This is the five-second check that stops that.

    cd examples/custom_miner
    python -m solvers.doctor claude
    python -m solvers.doctor claude --probe        # also send a real prompt
    python -m solvers.doctor chatgpt --cdp 9223    # a second browser

It attaches to the same Chrome the miner would, opens its own tab, and reports
for every role which candidate selector your page actually has. It flags the
three mistakes that matter: no composer (not signed in, or the DOM moved), a
"still generating" selector that matches an idle page (every answer would look
unfinished forever), and an assistant selector that already matches in an empty
conversation (it is matching something that is not a reply — possibly your own
message, which would make the miner answer with its own prompt). Every role is
overridable in ``.env`` with ``|`` between candidates.

``--probe`` goes further and drives the real ``_Tab.send`` path with a trivial
prompt, so what you see is exactly what the miner would read.

Safe to run while nothing else is using the browser: it opens a fresh tab,
closes that tab, and disconnects without closing your browser.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from .browser_pool import (
    DEFAULT_CDP_PORT,
    Site,
    import_playwright,
    normalize_cdp,
    usable_busy_selectors,
    wait_for_any,
)
from .config import find_env_file, load_env_file

# One prompt, asked in the shape that used to break. Two blocks on purpose:
# the reader takes every block and the grader picks the one that defines the
# entrypoint, so a usage example appended after the answer must not win. Asking
# for one block would leave that untested, and asking twice would spend a
# second prompt of the operator's quota to learn the same thing.
PROBE = (
    "Reply with exactly TWO fenced code blocks and no other text.\n"
    "Block 1: `def pong():` returning the string 'pong'.\n"
    "Block 2: a single line calling it, like `print(pong())`."
)


def _forensics(label: str, text: str) -> None:
    """Show what the page actually handed over, byte for byte where it matters.

    A rendered page is not a text file. Characters the UI uses for its own
    bookkeeping are invisible on screen and fatal in source, so the only honest
    way to know what a provider's DOM yields is to look at the codepoints.
    """
    odd: dict[str, int] = {}
    for ch in text:
        if ch in "\n\t" or " " <= ch <= "~":
            continue
        name = f"U+{ord(ch):04X}"
        odd[name] = odd.get(name, 0) + 1
    print(f"    {label}: {len(text)} chars")
    if odd:
        listed = ", ".join(f"{k}x{v}" for k, v in sorted(odd.items())[:12])
        print(f"      non-ASCII/invisible: {listed}")
    else:
        print("      non-ASCII/invisible: none")


def _site(name: str) -> Site:
    if name == "claude":
        from .claude_web import claude_site

        return claude_site()
    if name == "chatgpt":
        from .chatgpt_web import chatgpt_site

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


async def _attach(pw, site, endpoint: str):
    """Attach the way the miner does, or print why it could not.

    Returns the browser, or None. The caller disconnects it; nothing here ever
    closes the browser you started.
    """
    print(f"[doctor] {site.name}: attaching over CDP to {endpoint}")
    try:
        return await pw.chromium.connect_over_cdp(endpoint)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[doctor] FAIL: cannot attach: {str(exc).splitlines()[0]}\n"
            f"         Start Chrome with --remote-debugging-port "
            f"(scripts/start_debug_browser.sh does it) and confirm "
            f"{endpoint}/json/version answers."
        )
        return None


async def run(name: str, cdp: str, probe: bool) -> int:
    site = _site(name)
    endpoint = normalize_cdp(cdp)[0]
    async_playwright = import_playwright()
    pw = await async_playwright().start()
    try:
        browser = await _attach(pw, site, endpoint)
        if browser is None:
            return 2
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        # Our own tab, so the doctor never disturbs one you are looking at.
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
                    f"         Most often it is not signed in: open {site.url} in the "
                    f"Chrome on {endpoint} and sign in by hand."
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
            print()
            await _report_role(
                page,
                "new chat control",
                site.new_chat,
                on_miss="Not fatal: with no new-chat control each task starts its "
                "conversation by reloading the page instead, which always works "
                "but costs a few seconds per task. Set "
                f"{site.env_prefix}_NEW_CHAT to get them back.",
            )
            if site.message_id_attr:
                print(f"\n  message id attribute: {site.message_id_attr}")
            else:
                print("\n  message id attribute: none — replies identified by position")

            if probe:
                healthy &= await _probe(page, site, composer, kept)
        finally:
            # Close only our tab, then disconnect. Your browser keeps running.
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"\n[doctor] {'OK' if healthy else 'PROBLEMS FOUND (see !! above)'}")
        return 0 if healthy else 1
    finally:
        await pw.stop()


async def _probe(page, site: Site, composer: str, busy) -> bool:
    """Send a real prompt through the real read path."""
    from .browser_pool import _Tab

    class _Solo:
        """Minimal pool: the probe never releases the tab, so `release` is all
        a tab asks of the fleet it came from."""

        async def release(self, tab) -> None:
            pass

    print("\n[doctor] probing with a real prompt (up to 120s)...")
    from dataclasses import replace

    # The tab carries its own Site: a fleet spans providers and has no single
    # one to fall back on. Hand it the site with only the busy selectors this
    # page proved usable, which is exactly what the fleet hands a real tab.
    tab = _Tab(
        _Solo(), page, None, "probe", replace(site, busy=busy), composer=composer
    )
    reply = await tab.send(PROBE, 120.0)
    if not reply.strip():
        print("[doctor] !! the probe read nothing back. The assistant selector or the")
        print("            still-generating selector is wrong for this page.")
        return False
    from .prompts import extract_code, python_defect

    blocks = reply.count("```") // 2
    print(f"[doctor] read {len(reply)} chars back; code blocks seen: {blocks}")
    _forensics("raw reply", reply)

    # Not `extract_code(reply)`: the grader always knows the entrypoint, so the
    # doctor must ask the same question the miner asks, or it would pass on a
    # page where the miner picks the wrong block.
    code = extract_code(reply, "pong")
    _forensics("what would be submitted", code)
    print("    ----- what the miner would submit -----")
    for line in code.splitlines()[:12]:
        print(f"    | {line}")
    print("    ---------------------------------------")

    defect = python_defect(code, "pong")
    if defect is not None:
        print(f"[doctor] !! this page's replies do not survive extraction: {defect}")
        print("           That is a DOM difference, not a broken selector — the")
        print("           block above is what the reader got.")
        return False
    scope: dict = {}
    try:
        exec(compile(code, "<doctor>", "exec"), scope)
        result = scope["pong"]()
    except Exception as exc:  # noqa: BLE001 - that is the finding
        print(f"[doctor] !! the extracted code parses but will not run: {exc!r}")
        return False
    if result != "pong":
        print(f"[doctor] !! it ran but returned {result!r}, not 'pong'.")
        return False
    if blocks > 1:
        print("[doctor] probe answered correctly, and with TWO blocks offered the")
        print("         ANSWER was chosen, not the usage example. This page's")
        print("         markup is handled end to end.")
    else:
        print("[doctor] probe answered correctly. The model sent one block rather")
        print("         than the two asked for, so multi-block choice is untested")
        print("         here — re-run if you want that covered.")
    await _report_reset(tab)
    return True


async def _report_reset(tab) -> None:
    """Show how this tab will start its NEXT conversation, and what it costs.

    The probe has just left a real transcript on the page, which is exactly the
    state a tab is in between two tasks. Whether the in-app new-chat control
    works here is the one thing a selector list cannot tell you — it either
    routes and clears the transcript or it does not — and if it does not, the
    miner falls back to a reload silently and correctly, which is precisely why
    it is worth measuring rather than assuming.
    """
    print("\n[doctor] starting the next conversation the way the miner will...")
    started = time.monotonic()
    if await tab._new_chat():
        print(
            f"[doctor] in-app new chat: {time.monotonic() - started:.1f}s. The "
            f"transcript is gone and the page was never reloaded — this is what "
            f"the miner will do between tasks."
        )
        return
    print(
        f"[doctor] no usable new-chat control ({time.monotonic() - started:.1f}s "
        f"to find that out). Falling back to a reload, which always works:"
    )
    started = time.monotonic()
    await tab._reload()
    print(
        f"[doctor] reload: {time.monotonic() - started:.1f}s — paid on every "
        f"task after the first. Not a failure, but set "
        f"{tab.site.env_prefix}_NEW_CHAT to a selector for this page's "
        f"'New chat' control and it becomes the line above."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m solvers.doctor")
    parser.add_argument("backend", choices=("claude", "chatgpt"))
    parser.add_argument(
        "--cdp", default=None,
        help=f"CDP endpoint of the Chrome you started: a port, host:port or URL "
        f"(default: <BACKEND>_CDP from .env, else {DEFAULT_CDP_PORT})",
    )
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

    # Check the same browser the miner would: the first endpoint it would use.
    cdp = args.cdp or os.environ.get(f"{args.backend.upper()}_CDP") or str(DEFAULT_CDP_PORT)
    if not normalize_cdp(cdp):
        raise SystemExit(f"--cdp {cdp!r} is not a port, host:port or URL")
    return asyncio.run(run(args.backend, cdp, args.probe))


if __name__ == "__main__":
    sys.exit(main())

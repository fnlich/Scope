"""Log in to a provider once, into a Firefox profile the miner will reuse.

Playwright launches Firefox itself (``connect_over_cdp`` is Chromium-only, and
Firefox dropped CDP for WebDriver BiDi), so there is no "start a browser and
attach to it" step any more. Instead the login lives in a profile directory, and
this opens a real browser window on that directory so you can sign in by hand.

    python -m solvers.login claude
    python -m solvers.login chatgpt --profile ~/.hone-miner/firefox/chatgpt-2

It stays open until you press Enter, then reports whether the session actually
stuck, so you find out now rather than from a run of zeros.

**The profile can be open in one process at a time.** Stop the miner before
running this, and stop this before starting the miner.

## On a machine with no screen

Two ways, both fine:

  * Run it under Xvfb with a VNC server and tunnel in — ``scripts/login.sh``
    does exactly that and prints the ssh command.
  * Log in on a desktop, then copy the directory over:
    ``rsync -a ~/.hone-miner/firefox/claude-1/ server:~/.hone-miner/firefox/claude-1/``
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .browser_pool import default_profile, import_playwright, wait_for_any
from .config import find_env_file, load_env_file

SITES = {"claude": "https://claude.ai/", "chatgpt": "https://chatgpt.com/"}


def _site(name: str):
    if name == "claude":
        from .claude_web import claude_site

        return claude_site()
    from .chatgpt_web import chatgpt_site

    return chatgpt_site()


async def run(name: str, profile: str, headless: bool) -> int:
    site = _site(name)
    async_playwright = import_playwright()
    Path(profile).mkdir(parents=True, exist_ok=True)

    if headless:
        print(
            "[login] WARNING: no display found. A headless window cannot be typed\n"
            "        into. Use scripts/login.sh (Xvfb + VNC), or log in on a\n"
            "        desktop and copy the profile directory over.",
            file=sys.stderr,
        )
        return 2

    pw = await async_playwright().start()
    try:
        try:
            context = await pw.firefox.launch_persistent_context(
                profile, headless=False, viewport={"width": 1280, "height": 900}
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[login] cannot open {profile}: {str(exc).splitlines()[0]}\n"
                "        A profile can be open in one process at a time — is the "
                "miner running?",
                file=sys.stderr,
            )
            return 2
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(SITES[name], wait_until="domcontentloaded")
        print(f"[login] {name}: sign in in the window, then press Enter here.")
        print(f"[login] profile: {profile}")
        await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)

        # Verifying beats trusting: the composer only exists once you are in, and
        # "I thought I logged in" is otherwise indistinguishable from success
        # until the miner has been scoring zero for a while.
        await page.goto(site.url, wait_until="domcontentloaded")
        composer = await wait_for_any(page, site.composer, 20_000)
        await context.close()
        if composer is None:
            print(
                "[login] FAILED: the composer never appeared, so the session did "
                "not stick. Try again, and complete any verification step.",
                file=sys.stderr,
            )
            return 1
        print(f"[login] OK — session saved. Next: python -m solvers.doctor {name} --probe")
        return 0
    finally:
        await pw.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m solvers.login")
    parser.add_argument("backend", choices=("claude", "chatgpt"))
    parser.add_argument("--profile", default=None, help="Firefox profile directory")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preflight import require_linux

    require_linux("The login helper")
    load_env_file(find_env_file())

    profile = args.profile
    if profile is None:
        listed = os.environ.get(f"{args.backend.upper()}_PROFILES", "").replace(",", " ").split()
        profile = listed[0] if listed else str(default_profile(args.backend, 1))
    profile = str(Path(profile).expanduser())
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return asyncio.run(run(args.backend, profile, headless=not has_display))


if __name__ == "__main__":
    sys.exit(main())

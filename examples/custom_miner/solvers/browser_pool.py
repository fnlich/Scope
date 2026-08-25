"""A pool of logged-in Firefox tabs, shared by every browser backend.

Claude and ChatGPT need exactly the same machinery: keep N logged-in tabs,
lease one per task, start every task in a fresh conversation, notice when a tab
dies and replace it rather than recycle it, and bound every wait by the caller's
deadline. Only the DOM differs. So the machinery lives here once and each site
contributes a ``Site``: where to type, what to click, what "still generating"
looks like, and where the reply lands.

That split is not tidiness. Two of the behaviours here were real failure modes —
a tab that dies and gets recycled turns a miner into one that answers
``/health`` cheerfully while scoring zero forever, and an unbounded submit
overruns the solver's budget by ~90s because Playwright auto-waits 30s *per
action*. Duplicating them per site means fixing them twice and testing them
once.

## Why Playwright owns the browser

An earlier version attached over CDP to a Chrome you started yourself. Firefox
cannot do that: Playwright's ``connect_over_cdp`` is Chromium-only, and Mozilla
removed its CDP implementation in favour of WebDriver BiDi. So Playwright
launches Firefox itself, with ``launch_persistent_context`` against a profile
directory that keeps the login.

That is a better shape for a miner anyway — one process to supervise instead of
two, and a crash restarts already logged in — but it comes with one rule worth
knowing before it bites you: **a profile directory can only be open in one
process at a time.** Run ``python -m solvers.login`` while the miner is running
and the launch fails; the message says so.

## Selectors are configuration, not code

Chat UIs change their DOM without notice, and a miner whose selector broke
overnight looks exactly like a miner that is merely idle. So every role takes a
LIST of candidate selectors tried in order, and every list is overridable from
the environment with ``|`` between candidates (``,`` is already CSS's own
"either" operator, and we need to know *which* one matched):

    CLAUDE_ASSISTANT='div[data-is-streaming]|div.font-claude-message'

``python -m solvers.doctor claude`` reports which candidate won for each role,
so a DOM change is a one-line .env fix instead of a patch. Run it before you
point a registered hotkey at any backend.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Site:
    """Everything that differs between one chat UI and another."""

    name: str
    env_prefix: str
    url: str
    composer: tuple[str, ...]
    send: tuple[str, ...]
    # Selectors that are present only while the model is generating.
    busy: tuple[str, ...]
    # Selectors for an ASSISTANT message. Must not be able to match the user's
    # own message — see the echo guard in ``_Tab.send``.
    assistant: tuple[str, ...]
    # Attribute that uniquely identifies a message, if the site has one.
    # ChatGPT does (``data-message-id``); without one the reply is identified
    # by position, which works because every task starts a fresh conversation.
    message_id_attr: Optional[str] = None
    # Appended to every prompt. Used to keep a site from answering in a form
    # the reader cannot see (Claude's artifacts panel, for instance).
    nudge: str = ""
    poll_s: float = 2.0
    ready_timeout_ms: int = 60_000


PROFILE_ROOT = Path(os.environ.get("MINER_PROFILE_ROOT", "~/.hone-miner/firefox")).expanduser()


def default_profile(backend: str, index: int = 1) -> Path:
    """Where a profile lives when nobody said otherwise."""
    return PROFILE_ROOT / f"{backend}-{index}"


def import_playwright():
    """Playwright, or an actionable message instead of an ImportError traceback.

    It is in no extra of this project on purpose: it is a large dependency and
    the browser download is larger still, so the failure has to name both steps.
    """
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "The browser backends need Playwright and its Firefox build:\n"
            "    pip install playwright\n"
            "    python -m playwright install firefox\n"
            "Playwright launches Firefox itself, so both steps are required."
        ) from exc
    return async_playwright


async def wait_for_any(page, candidates: Sequence[str], timeout_ms: int) -> Optional[str]:
    """Return the first candidate selector that appears, or None on timeout.

    Polling rather than ``wait_for_selector`` so a candidate list can mix
    selector engines without one bad entry failing the whole wait.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        for selector in candidates:
            try:
                if await page.locator(selector).count() > 0:
                    return selector
            except Exception:  # noqa: BLE001 - a malformed candidate is just skipped
                continue
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.5)


async def valid_selectors(
    page, candidates: Sequence[str], name: str, role: str
) -> tuple[str, ...]:
    """Drop candidates this page cannot even evaluate.

    Done once at startup rather than at answer time, because at answer time a
    selector that raises is indistinguishable from a page that has died — and
    the tab would be retired on every request. A typo in a ``.env`` override
    should cost that one candidate, not the miner.
    """
    kept: list[str] = []
    for selector in candidates:
        try:
            await page.locator(selector).count()
        except Exception as exc:  # noqa: BLE001 - that is the finding
            print(
                f"[{name}] WARN: {role} selector {selector!r} is not usable "
                f"({type(exc).__name__}); ignoring it"
            )
            continue
        kept.append(selector)
    return tuple(kept)


async def usable_busy_selectors(page, candidates: Sequence[str], name: str) -> tuple[str, ...]:
    """Drop 'still generating' selectors that match an idle page.

    This is the one selector mistake that cannot degrade gracefully: a busy
    selector which is always true means the reader believes the model is
    forever mid-answer and every solve burns its whole budget for nothing. An
    idle, freshly-loaded page is exactly the ground truth needed to catch it,
    and it costs one DOM query per candidate at startup.

    The opposite mistake — a busy selector that never matches — is harmless
    here, because the reader also requires the text to be unchanged across two
    polls before it accepts an answer.
    """
    kept: list[str] = []
    for selector in candidates:
        try:
            if await page.locator(selector).count() > 0:
                print(
                    f"[{name}] WARN: busy selector {selector!r} matches an IDLE page; "
                    "ignoring it (it would make every answer look unfinished)"
                )
                continue
        except Exception:  # noqa: BLE001 - malformed selector, same treatment
            continue
        kept.append(selector)
    return tuple(kept)


class _Tab:
    """One chat tab: a single conversation slot leased from the pool.

    ``alive`` is the pool's health signal. A tab whose browser or page has gone
    away must never be recycled: leasing it again fails the next request too,
    and a miner that keeps handing out dead tabs returns zeros indefinitely
    while still answering ``/health``. Any Playwright failure clears the flag,
    and the pool disposes of the tab and tries to spawn a replacement.
    """

    def __init__(
        self,
        pool: "CDPPool",
        page,
        context,
        label: str,
        *,
        site: Optional[Site] = None,
        composer: Optional[str] = None,
    ):
        self._pool = pool
        self._page = page
        self.context = context
        self.label = label
        # A per-tab Site: the pool prunes the selectors this page cannot use
        # before handing it over, so the answer path never has to guess whether
        # a raising selector means a bad override or a dead page.
        self.site: Site = site or pool.site
        self._composer = composer or (self.site.composer[0] if self.site.composer else "")
        self._sent = ""
        self._warned_echo = False
        self.uses = 0
        self.alive = True

    async def start(self) -> None:
        """Open a fresh conversation so each task begins with empty context.

        Context bleed is not a cosmetic problem for a miner: a model that can
        still see the previous task's code will happily blend the two, and the
        result fails the hidden suite in a way that is very hard to diagnose.
        """
        try:
            await self._page.goto(self.site.url, wait_until="domcontentloaded")
            found = await wait_for_any(self._page, self.site.composer, 30_000)
            if found is None:
                raise RuntimeError("composer did not reappear after navigation")
            self._composer = found
        except Exception:
            self.alive = False
            raise

    async def _submit(self, text: str, ui_ms: int) -> None:
        """Type the prompt and press send, with every step bounded.

        Playwright's default auto-wait is 30s PER action, so an unbounded
        click/insert/click can burn ~90s that the solver's budget never
        accounted for — which would blow the response deadline and score zero
        even though the answer was on its way.
        """
        composer = self._page.locator(self._composer)
        await composer.click(timeout=ui_ms)
        # insert_text handles newlines safely — typing them would submit early.
        await self._page.keyboard.insert_text(text)
        button = await self._first_match(self.site.send)
        if button is None:
            # No send button matched. Every one of these composers also submits
            # on Enter, and that is safe *here* only because the whole prompt,
            # newlines included, is already in the box via insert_text.
            await self._page.keyboard.press("Enter")
            return
        await self._page.locator(button).click(timeout=ui_ms)

    async def send(self, text: str, timeout_s: float) -> str:
        self.uses += 1
        if self.site.nudge:
            text = f"{text}\n\n{self.site.nudge}"
        self._sent = text
        # Reserve a slice of the caller's budget for getting the prompt in;
        # the rest is for waiting on the answer.
        submit_budget_s = max(5.0, min(20.0, timeout_s * 0.3))
        try:
            before = await self._fingerprint()
            await asyncio.wait_for(
                self._submit(text, int(submit_budget_s * 1000)),
                timeout=submit_budget_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - includes the submit timeout
            # The prompt never reached the model. Treat the tab as unusable so
            # the pool replaces it rather than failing the next request too.
            self.alive = False
            print(f"[{self.site.name}] tab {self.label} failed to submit: {type(exc).__name__}")
            return ""

        deadline = time.monotonic() + max(1.0, timeout_s)
        stable: Optional[str] = None
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(self.site.poll_s)
                if await self._busy_now():
                    stable = None
                    continue
                reply = await self._new_reply(before)
                if reply is None:
                    continue  # ours hasn't rendered yet
                text_now = await self._read(reply)
                if text_now is None or self._echoes_prompt(text_now):
                    continue
                if text_now == stable:
                    return text_now  # finished: not busy, and unchanged text
                stable = text_now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the page died mid-read
            self.alive = False
            print(f"[{self.site.name}] tab {self.label} died while reading: {type(exc).__name__}")
        # Timed out mid-stream: a partial answer still beats nothing, because
        # the verifier can grade it and may repair it on the next attempt.
        return stable or ""

    # -- DOM helpers -------------------------------------------------------- #
    async def _first_match(self, candidates: Sequence[str]) -> Optional[str]:
        """First candidate present right now. Deliberately not cached: the
        earlier, more specific candidates only start matching once the reply
        exists, so latching onto a late-matching fallback would be permanent."""
        for selector in candidates:
            if await self._page.locator(selector).count() > 0:
                return selector
        return None

    async def _messages(self):
        selector = await self._first_match(self.site.assistant)
        return None if selector is None else self._page.locator(selector)

    async def _fingerprint(self) -> tuple[int, Optional[str]]:
        """How the conversation looked before we submitted."""
        messages = await self._messages()
        if messages is None:
            return 0, None
        count = await messages.count()
        if count == 0:
            return 0, None
        last = messages.nth(count - 1)
        mid = (
            await last.get_attribute(self.site.message_id_attr)
            if self.site.message_id_attr
            else None
        )
        return count, mid

    async def _new_reply(self, before: tuple[int, Optional[str]]):
        """The last assistant message, if it is not the one that was there before."""
        count_before, id_before = before
        messages = await self._messages()
        if messages is None:
            return None
        count = await messages.count()
        if count == 0:
            return None
        last = messages.nth(count - 1)
        if self.site.message_id_attr:
            mid = await last.get_attribute(self.site.message_id_attr)
            if mid is not None:
                return None if mid == id_before else last
        # No id attribute: identify by position. Sound because every task opens
        # a fresh conversation, so the reply we want is simply an assistant
        # message that was not there when we pressed send.
        return last if count > count_before else None

    async def _busy_now(self) -> bool:
        for selector in self.site.busy:
            if await self._page.locator(selector).count() > 0:
                return True
        return False

    def _echoes_prompt(self, text: str) -> bool:
        """Guard against an assistant selector that matches the USER's message.

        Without this the miner would submit its own prompt back as the answer —
        a total failure that produces no error and no empty reply, just a
        permanent zero. Cheap to check, and it names the fix in the log.
        """
        head = " ".join(self._sent.split())[:80]
        if not head or not " ".join(text.split()).startswith(head):
            return False
        if not self._warned_echo:
            self._warned_echo = True
            print(
                f"[{self.site.name}] WARN: tab {self.label}: an assistant selector is "
                f"matching the user's message. Run `python -m solvers.doctor "
                f"{self.site.name}` and set {self.site.env_prefix}_ASSISTANT."
            )
        return True

    @staticmethod
    async def _read(reply) -> Optional[str]:
        code_blocks = reply.locator("pre code")
        count = await code_blocks.count()
        if count > 0:
            return await code_blocks.nth(count - 1).inner_text()
        text = (await reply.inner_text()).strip()
        return text or None

    async def close(self) -> None:
        """Release the tab back to the pool (the tab itself is reused)."""
        await self._pool.release(self)

    async def dispose(self) -> None:
        try:
            await self._page.close()
        except Exception:  # noqa: BLE001 - the browser may already be gone
            pass


def normalize_cdp(value: str | Sequence[str] | None) -> list[str]:
    """Turn ``"9222"`` / ``"host:9222"`` / a full URL / a list into CDP URLs.

    A bare port is the common case, so ``9222`` means ``http://127.0.0.1:9222``.
    """
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else str(value).replace(",", " ").split()
    endpoints: list[str] = []
    for raw in parts:
        item = str(raw).strip()
        if not item:
            continue
        if "://" in item:
            endpoints.append(item.rstrip("/"))
        elif ":" in item:
            endpoints.append(f"http://{item}")
        else:
            endpoints.append(f"http://127.0.0.1:{item}")
    return endpoints


class BrowserPool:
    """Tabs leased one per task, from one of two browser sources.

    **Launch mode (default).** Playwright launches Firefox against a persistent
    profile directory. Simple — one process, no ports — but a Playwright-driven
    Firefox is a distinctive automation build, and providers that fingerprint
    aggressively (Google sign-in, above all) refuse it.

    **Attach mode.** You start an ordinary Chrome/Chromium yourself with
    ``--remote-debugging-port`` and log in by hand; the pool attaches over CDP.
    Because *you* launched it, it is not in automation mode — ``navigator.webdriver``
    is false and it looks like the normal browser it is, which is what gets past
    those sign-in checks. Pass ``cdp=[...]`` (or set ``<PREFIX>_CDP``) to select
    it. Only Chromium exposes CDP; Firefox cannot be attached to this way.

    Either way, each source (profile *or* browser) is expected to hold a DIFFERENT
    account: accounts are the rate-limit unit, so N sources give N times the
    throughput. Tabs within one source share that account and its limit.
    """

    def __init__(
        self,
        site: Site,
        profiles: list[str],
        *,
        tabs_per_profile: int = 2,
        headless: bool = True,
        cdp: str | Sequence[str] | None = None,
    ):
        self.site = site
        self._cdp = normalize_cdp(cdp)
        self._attach = bool(self._cdp)
        self._profiles = [str(Path(p).expanduser()) for p in profiles] or [
            str(default_profile(site.name, 1))
        ]
        self._tabs_per_profile = max(1, int(tabs_per_profile))
        self._headless = bool(headless)
        self._pw = None
        self._contexts: list[Any] = []
        self._browsers: list[Any] = []  # CDP-attached; disconnected, never killed
        self._free: asyncio.Queue[_Tab] = asyncio.Queue()
        self._size = 0
        self._lost = 0
        self._started = False
        self._start_lock = asyncio.Lock()

    def _login_hint(self, source: str = "") -> str:
        """How to fix 'not logged in', worded for whichever mode is in use."""
        if self._attach:
            where = f" on {source}" if source else ""
            return (
                f"open {self.site.url} in the Chrome{where} you started with "
                f"--remote-debugging-port and sign in by hand"
            )
        return f"python -m solvers.login {self.site.name}" + (
            f" --profile {source}" if source else ""
        )

    async def _spawn(self, context, label: str, source: str = "") -> Optional[_Tab]:
        """Open one logged-in tab, or return None with a reason logged."""
        site = self.site
        page = None
        try:
            page = await context.new_page()
            await page.goto(site.url, wait_until="domcontentloaded")
            composer = await wait_for_any(page, site.composer, site.ready_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{site.name}] WARN: tab {label} never loaded {site.url} "
                f"({type(exc).__name__}) — is it logged in? {self._login_hint(source)}"
            )
            return None
        if composer is None:
            print(
                f"[{site.name}] WARN: tab {label}: no composer selector matched. "
                f"Either it is not logged in ({self._login_hint(source)}), or the DOM "
                f"changed — run `python -m solvers.doctor {site.name}` and set "
                f"{site.env_prefix}_COMPOSER."
            )
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
            return None
        checked = replace(
            site,
            send=await valid_selectors(page, site.send, site.name, "send"),
            assistant=await valid_selectors(page, site.assistant, site.name, "assistant"),
            # Chained so a typo'd busy override is named, not just dropped.
            busy=await usable_busy_selectors(
                page,
                await valid_selectors(page, site.busy, site.name, "busy"),
                site.name,
            ),
        )
        return _Tab(self, page, context, label, site=checked, composer=composer)

    async def start(self) -> None:
        """Attach to the browsers and fill the tab pool. Idempotent.

        Safe to call explicitly at startup, and called lazily by ``open()`` so
        the pool also works under a host that has no startup hook — a local test
        harness, for instance. Without that, a host which never called start()
        would leave ``open()`` blocked on an empty queue and every solve would
        quietly return nothing.
        """
        async with self._start_lock:
            if self._started:
                return
            await self._connect()
            self._started = True

    async def _connect(self) -> None:
        async_playwright = import_playwright()
        site = self.site
        self._pw = await async_playwright().start()
        # (context, label_base, source_description) per browser source.
        sources = (
            [(await self._attach_cdp(e), self._label(e), e) for e in self._cdp]
            if self._attach
            else [(await self._open_profile(p), Path(p).name, p) for p in self._profiles]
        )
        for context, label_base, source in sources:
            if context is None:
                continue
            self._contexts.append(context)
            for i in range(self._tabs_per_profile):
                tab = await self._spawn(context, f"{label_base}#{i + 1}", source)
                if tab is None:
                    continue
                await self._free.put(tab)
                self._size += 1
                print(f"[{site.name}] tab {tab.label} ready")
        if self._size == 0:
            how = (
                f"start Chrome with --remote-debugging-port, {self._login_hint()}, "
                "and check the endpoint is reachable"
                if self._attach
                else f"log in once with:\n    {self._login_hint()}"
            )
            raise RuntimeError(f"No usable {site.name} tabs. To fix: {how}")
        kind = "browser" if self._attach else "profile"
        print(
            f"[{site.name}] pool ready: {self._size} tab(s) "
            f"across {len(self._contexts)} {kind}(s)"
        )

    @staticmethod
    def _label(endpoint: str) -> str:
        """A short, stable tab-label base from a CDP endpoint (its host:port)."""
        return endpoint.split("://", 1)[-1]

    async def _attach_cdp(self, endpoint: str):
        """Attach to a browser you started yourself, or explain why it could not.

        The browser is NOT owned here: ``aclose`` disconnects but never kills it,
        so restarting the miner keeps your hand-made login.
        """
        site = self.site
        try:
            browser = await self._pw.chromium.connect_over_cdp(endpoint)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{site.name}] WARN: cannot attach to {endpoint}: "
                f"{str(exc).splitlines()[0]}\n"
                f"         Start Chrome/Chromium with --remote-debugging-port and a "
                f"--user-data-dir, and confirm {endpoint}/json/version answers."
            )
            return None
        self._browsers.append(browser)
        return browser.contexts[0] if browser.contexts else await browser.new_context()

    async def _open_profile(self, profile: str):
        """Launch Firefox on one profile directory, or explain why it could not."""
        site = self.site
        if not Path(profile).is_dir():
            print(
                f"[{site.name}] WARN: no profile at {profile}. Log in once with:\n"
                f"    python -m solvers.login {site.name} --profile {profile}"
            )
            return None
        try:
            return await self._pw.firefox.launch_persistent_context(
                profile, headless=self._headless, viewport={"width": 1280, "height": 900}
            )
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            # A profile can be open in exactly one process. This is the mistake
            # people make: leaving the login helper running, then starting the
            # miner, and reading the raw Playwright error as a crash.
            hint = (
                "\n         That profile is already open in another process — "
                "close the login helper (or the other miner) first."
                if "profile" in detail.lower() or "lock" in detail.lower()
                else ""
            )
            print(f"[{site.name}] WARN: cannot open {profile}: {detail.splitlines()[0]}{hint}")
            return None

    async def open(self) -> _Tab:
        """Lease a tab and put it in a fresh conversation."""
        if not self._started:
            await self.start()
        tab = await self._free.get()
        try:
            await tab.start()
        except Exception:
            # start() already marked it dead; release() disposes and replaces it.
            await self.release(tab)
            raise
        return tab

    async def release(self, tab: _Tab) -> None:
        """Return a tab to the pool, or retire and replace a dead one.

        Recycling a dead tab is the failure that matters here: it would fail
        every future request leased onto it, so capacity is rebuilt instead.
        """
        if tab.alive:
            await self._free.put(tab)
            return
        self._lost += 1
        await tab.dispose()
        replacement = await self._spawn(tab.context, tab.label)
        if replacement is not None:
            await self._free.put(replacement)
            print(f"[{self.site.name}] tab {tab.label} replaced after failure")
            return
        self._size -= 1
        print(
            f"[{self.site.name}] WARN: tab {tab.label} retired and could not be "
            f"replaced; {self._size} tab(s) left"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.site.name,
            "mode": "attach" if self._attach else "launch",
            "tabs": self._size,
            "idle": self._free.qsize(),
            "sources": len(self._contexts),
            "headless": None if self._attach else self._headless,
            "lost": self._lost,
        }

    async def aclose(self) -> None:
        if not self._started:
            return
        while not self._free.empty():
            await (await self._free.get()).dispose()
        if self._attach:
            # Disconnect, do NOT kill: you own these browsers, and the login in
            # them is what we do not want to throw away on a miner restart.
            for browser in self._browsers:
                try:
                    await browser.close()
                except Exception:  # noqa: BLE001
                    pass
        else:
            for context in self._contexts:
                try:
                    await context.close()  # also stops the Firefox process
                except Exception:  # noqa: BLE001
                    pass
        if self._pw is not None:
            await self._pw.stop()

"""A pool of CDP-attached chat tabs, shared by every browser backend.

ChatGPT and Claude need exactly the same machinery: attach to an
already-running Chrome over CDP, hold N logged-in tabs, lease one per task,
start every task in a fresh conversation, notice when a tab dies and replace it
rather than recycle it, and bound every wait by the caller's deadline. Only the
DOM differs. So the machinery lives here once and each site contributes a
``Site``: where to type, what to click, what "still generating" looks like, and
where the reply lands.

That split is not tidiness. Two of the behaviours in this file were real
failure modes — a tab that dies and gets recycled turns a miner into one that
answers ``/health`` cheerfully while scoring zero forever, and an unbounded
Playwright submit overruns the solver's budget by ~90s because auto-wait is 30s
*per action*. Duplicating them per site means fixing them twice and testing
them once.

## Selectors are configuration, not code

Chat UIs change their DOM without notice, and a miner whose selector broke
overnight looks exactly like a miner that is merely idle. So every role takes a
LIST of candidate selectors tried in order, and every list is overridable from
the environment with ``|`` between candidates (``,`` is already CSS's own
"either" operator, and we need to know *which* one matched):

    CLAUDE_ASSISTANT='div[data-is-streaming]|div.font-claude-message'

``python -m solvers.doctor claude`` attaches to your browser and reports which
candidate won for each role, so a DOM change is a one-line .env fix instead of
a patch. Run it before you point a registered hotkey at any browser backend.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
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


def import_playwright():
    """Playwright, or an actionable message instead of an ImportError traceback.

    It is in no extra of this project on purpose: the API backends do not need
    it, and it is a large dependency. Note that no ``playwright install`` is
    required either — these backends attach over CDP to a Chrome you started
    yourself, so there is no bundled browser to download.
    """
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "The browser backends need Playwright:\n"
            "    pip install playwright\n"
            "No `playwright install` is needed — they attach over CDP to a "
            "Chrome you start yourself."
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


class CDPPool:
    """Tabs across one or more CDP-attached browsers, leased one per task.

    Each port is expected to be a browser logged into a DIFFERENT account:
    accounts are the real rate-limit unit, so N accounts give N times the
    throughput. Tabs within one browser share that browser's account and
    therefore its limit.
    """

    def __init__(
        self,
        site: Site,
        ports: list[int],
        *,
        host: str = "127.0.0.1",
        tabs_per_browser: int = 2,
    ):
        self.site = site
        self._ports = ports or [9222]
        self._host = host
        self._tabs_per_browser = max(1, int(tabs_per_browser))
        self._pw = None
        self._browsers: list[Any] = []
        self._free: asyncio.Queue[_Tab] = asyncio.Queue()
        self._size = 0
        self._lost = 0
        self._started = False
        self._start_lock = asyncio.Lock()

    async def _spawn(self, context, label: str) -> Optional[_Tab]:
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
                f"({type(exc).__name__}) — is this profile logged in?"
            )
            return None
        if composer is None:
            print(
                f"[{site.name}] WARN: tab {label}: no composer selector matched. "
                f"Either the profile is not logged in, or the DOM changed — run "
                f"`python -m solvers.doctor {site.name}` and set {site.env_prefix}_COMPOSER."
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
        for port in self._ports:
            endpoint = f"http://{self._host}:{port}"
            try:
                browser = await self._pw.chromium.connect_over_cdp(endpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"[{site.name}] WARN: cannot attach to {endpoint}: {exc}")
                continue
            self._browsers.append(browser)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            for i in range(self._tabs_per_browser):
                tab = await self._spawn(context, f"{port}#{i + 1}")
                if tab is None:
                    continue
                await self._free.put(tab)
                self._size += 1
                print(f"[{site.name}] tab {tab.label} ready")
        if self._size == 0:
            raise RuntimeError(
                f"No usable {site.name} tabs. Start Chrome with "
                f"--remote-debugging-port=<port> --user-data-dir=<dir>, open "
                f"{site.url} in it, and log in."
            )
        print(
            f"[{site.name}] pool ready: {self._size} tab(s) "
            f"across {len(self._browsers)} browser(s)"
        )

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
            "tabs": self._size,
            "idle": self._free.qsize(),
            "browsers": len(self._browsers),
            "lost": self._lost,
        }

    async def aclose(self) -> None:
        if not self._started:
            return
        while not self._free.empty():
            await (await self._free.get()).dispose()
        for browser in self._browsers:
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw is not None:
            await self._pw.stop()

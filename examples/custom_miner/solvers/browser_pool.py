"""A pool of logged-in Chrome tabs, shared by every browser backend.

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

## You start the browser; this attaches to it

This pool never launches a browser. You start an ordinary Chrome or Chromium
yourself with ``--remote-debugging-port`` (``scripts/start_debug_browser.sh``
does it), sign in **by hand**, and the pool attaches over the Chrome DevTools
Protocol.

That division of labour is the point, not an inconvenience. A browser launched
by an automation driver announces itself as one — ``navigator.webdriver`` is
true and the build is distinctive — and provider sign-in flows reject it. The
most visible of those is Google's OAuth, which answers *"Couldn't sign you in.
This browser or app may not be secure."* A browser **you** started is not in
automation mode, so the same sign-in succeeds; attaching afterwards does not
change that.

Two consequences worth knowing before they surprise you:

* **The browser is yours, not ours.** ``aclose`` closes the tabs this pool
  opened and then *disconnects*; it never closes your browser. Restarting the
  miner therefore keeps the login you made by hand.
* **CDP is a Chromium protocol.** Only Chrome and Chromium expose it. That is
  what fixes the browser choice here — it is a constraint, not a preference.

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
from typing import Any, Optional, Sequence

# The port `scripts/start_debug_browser.sh` uses unless told otherwise, and so
# the port every backend assumes when `<PREFIX>_CDP` is not set.
DEFAULT_CDP_PORT = 9222


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

    It is in no extra of this project on purpose: it is a large dependency and
    most of this example does not need it. Note that no ``playwright install``
    step is required — this attaches to a browser you started, so there is no
    bundled browser to download.
    """
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "The browser backends need Playwright:\n"
            "    pip install playwright\n"
            "No `playwright install` is needed — the backends attach over CDP to "
            "a Chrome you start yourself."
        ) from exc
    return async_playwright


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

    The opposite mistake — a busy selector that never matches — is survivable
    rather than harmless: the reader falls back to requiring the text to be
    unchanged across two polls, so an answer is accepted after ~2x poll_s of
    quiet. A model that pauses longer than that mid-generation can have a
    truncated answer accepted as final, which the repair round then has to
    catch. Keeping one working busy selector is worth it.
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
        pool: "BrowserPool",
        page,
        context,
        label: str,
        *,
        site: Optional[Site] = None,
        composer: Optional[str] = None,
        source: str = "",
    ):
        self._pool = pool
        self._page = page
        self.context = context
        self.label = label
        # Which browser this tab came from. Carried so a replacement spawned
        # after a failure can still name the right endpoint in its log.
        self.source = source
        # A per-tab Site: the pool prunes the selectors this page cannot use
        # before handing it over, so the answer path never has to guess whether
        # a raising selector means a bad override or a dead page.
        self.site: Site = site or pool.site
        self._composer = composer or (self.site.composer[0] if self.site.composer else "")
        self._sent = ""
        # Assistant selector latched for the current send(); see _messages().
        self._assistant: Optional[str] = None
        self._warned_echo = False
        self.uses = 0
        self.alive = True
        # Set while this tab is out on loan. The pool uses it to make a second
        # release() a no-op: without it a double close would queue the same tab
        # twice and two tasks would end up driving one page.
        self.leased = False

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

        ``.first`` on both clicks is not cosmetic: a Locator click is strict and
        RAISES when the selector matches more than one node, while the candidate
        lists deliberately end in broad fallbacks that can. Without it a page
        with two matching nodes fails every submit, and since a failed submit
        retires the tab, the pool would churn tabs forever and never answer.
        """
        composer = self._page.locator(self._composer).first
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
        await self._page.locator(button).first.click(timeout=ui_ms)

    async def send(self, text: str, timeout_s: float) -> str:
        if not self.alive:
            # The pool has not recycled this tab yet. Retrying a known-dead tab
            # only burns the caller's budget one submit-timeout at a time.
            return ""
        # Start the clock BEFORE the submit. Deriving the read deadline after it
        # would hand the read a fresh `timeout_s` on top of however long typing
        # took — an overrun larger than the solver's whole safety margin.
        started = time.monotonic()
        deadline = started + max(1.0, timeout_s)
        self.uses += 1
        if self.site.nudge:
            text = f"{text}\n\n{self.site.nudge}"
        self._sent = text
        # One selector for the whole call: `_first_match` is free to pick a
        # different candidate once the reply renders, and comparing a count taken
        # from one selector against a count taken from another is meaningless.
        self._assistant = None
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

        # `stable` drives the completion test and is reset whenever the model is
        # mid-generation. `best` is the newest text actually read and is never
        # reset, so a deadline that lands mid-answer still returns something.
        best, stable = "", None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self.site.poll_s, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Every DOM call below auto-waits up to 30s on its own, which
                # would sail past the deadline the loop just checked.
                text_now, busy = await asyncio.wait_for(
                    self._poll(before), timeout=remaining
                )
                if text_now is not None:
                    best = text_now  # keep it even mid-generation
                if busy or text_now is None:
                    stable = None  # still typing, or ours has not rendered yet
                    continue
                if text_now == stable:
                    return text_now  # finished: not busy, and unchanged text
                stable = text_now
        except asyncio.TimeoutError:
            pass  # out of budget mid-read; `best` still holds what we had
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the page died mid-read
            self.alive = False
            print(f"[{self.site.name}] tab {self.label} died while reading: {type(exc).__name__}")
        return best

    async def _poll(
        self, before: tuple[int, Optional[str]]
    ) -> tuple[Optional[str], bool]:
        """One read round: ``(text_or_None, still_generating)``.

        The read happens whether or not the model is still typing. Checking busy
        first and returning early — the obvious shape — means a reply that never
        stops streaming before the deadline is never read at all, so a timeout
        mid-answer returns nothing instead of the part that had arrived.
        """
        busy = await self._busy_now()
        reply = await self._new_reply(before)
        if reply is None:
            return None, busy  # ours hasn't rendered yet
        text_now = await self._read(reply)
        if text_now is None or await self._echoes_prompt(reply, text_now):
            return None, busy
        return text_now, busy

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
        """Assistant messages, through ONE selector for the whole send().

        Latched on first match and reset by ``send()``. Re-resolving per call
        would let the fingerprint count nodes matching one candidate and the
        poll count nodes matching another, and ``_new_reply`` compares those two
        numbers directly — so a site with several viable candidates and no
        message id (claude.ai exactly) would mis-detect replies on every repair
        round. Latching per send, not per tab, still lets a later send pick a
        better candidate once the DOM settles.
        """
        if self._assistant is None:
            self._assistant = await self._first_match(self.site.assistant)
            if self._assistant is None:
                return None
        return self._page.locator(self._assistant)

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

    async def _echoes_prompt(self, reply, text: str) -> bool:
        """Guard against an assistant selector that matches the USER's message.

        Without this the miner would submit its own prompt back as the answer —
        a total failure that produces no error and no empty reply, just a
        permanent zero. Cheap to check, and it names the fix in the log.

        Checked against the WHOLE message, not the text ``_read`` returned:
        ``_read`` prefers the last ``pre code`` block, and task statements
        routinely contain fenced code, so comparing against the extracted block
        would never match the prompt's opening words and the guard would never
        fire — precisely when it is needed.
        """
        head = " ".join(self._sent.split())[:80]
        if not head:
            return False
        try:
            whole = await reply.inner_text()
        except Exception:  # noqa: BLE001 - fall back to what we already read
            whole = text
        if not " ".join(whole.split()).startswith(head):
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
        """Hand the tab back. Reused if it is still healthy, retired if not."""
        await self._pool.release(self)

    async def dispose(self) -> None:
        """Close this tab's page. Safe to call twice."""
        try:
            await self._page.close()
        except Exception:  # noqa: BLE001 - the browser may already be gone
            pass


class BrowserPool:
    """Tabs leased one per task, from Chrome browsers you started and attached to.

    Each endpoint is expected to be a browser holding a DIFFERENT account:
    accounts are the rate-limit unit, so N browsers give N times the throughput.
    Tabs within one browser share that account and therefore its limit.

    Nothing here launches or closes a browser. See the module docstring for why
    that is the whole point rather than a limitation.
    """

    def __init__(
        self,
        site: Site,
        cdp: str | Sequence[str] | None = None,
        *,
        tabs_per_browser: int = 2,
    ):
        self.site = site
        self._cdp = normalize_cdp(cdp) or normalize_cdp(str(DEFAULT_CDP_PORT))
        self._tabs_per_browser = max(1, int(tabs_per_browser))
        self._pw = None
        self._contexts: list[Any] = []
        self._browsers: list[Any] = []  # attached; disconnected, never closed
        # Every tab ever handed out, so shutdown can close the pages this pool
        # opened in YOUR browser — including the ones currently leased, which a
        # free-queue-only sweep would leave behind.
        self._tabs: list[_Tab] = []
        self._free: asyncio.Queue[_Tab] = asyncio.Queue()
        self._size = 0
        self._lost = 0
        self._started = False
        self._closing = False
        # How long open() waits for a free tab before saying so. Unbounded would
        # be worse: a caller with its own deadline gets cancelled, and a caller
        # without one hangs with no explanation.
        self._lease_timeout_s = float(os.environ.get("MINER_TAB_WAIT_S", "120"))
        self._start_lock = asyncio.Lock()

    def _login_hint(self, source: str = "") -> str:
        """How to fix 'not logged in', naming the browser it applies to."""
        where = f" on {source}" if source else ""
        return (
            f"open {self.site.url} in the Chrome{where} you started with "
            f"--remote-debugging-port and sign in by hand"
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
            if page is not None:
                try:
                    await page.close()
                except Exception:  # noqa: BLE001 - do not leave a stray tab behind
                    pass
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
        if not checked.assistant:
            # Every assistant candidate was unusable — a one-entry .env override
            # with a typo does this. Such a tab can never find a reply, so it
            # would burn the whole budget on every request and return nothing,
            # while still being logged as ready. Refuse it loudly instead.
            print(
                f"[{site.name}] WARN: tab {label}: no usable assistant selector "
                f"(tried {list(site.assistant)}). Fix {site.env_prefix}_ASSISTANT — "
                f"without one this tab could never read a reply."
            )
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
            return None
        tab = _Tab(self, page, context, label, site=checked, composer=composer, source=source)
        self._tabs.append(tab)
        return tab

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
        self._pw = await async_playwright().start()
        try:
            await self._fill()
        except Exception:
            # Started the driver, then failed. Without this the driver process
            # and every attached connection leak, and the next open() — start()
            # having left _started False — would spawn a second set.
            await self._teardown()
            raise

    async def _fill(self) -> None:
        site = self.site
        for endpoint in self._cdp:
            context = await self._attach(endpoint)
            if context is None:
                continue
            self._contexts.append(context)
            label_base = endpoint.split("://", 1)[-1]
            for i in range(self._tabs_per_browser):
                tab = await self._spawn(context, f"{label_base}#{i + 1}", endpoint)
                if tab is None:
                    continue
                await self._free.put(tab)
                self._size += 1
                print(f"[{site.name}] tab {tab.label} ready")
        if self._size == 0:
            raise RuntimeError(
                f"No usable {site.name} tabs. To fix: start Chrome with "
                f"--remote-debugging-port (scripts/start_debug_browser.sh does it), "
                f"{self._login_hint()}, and confirm the endpoint answers."
            )
        print(
            f"[{site.name}] pool ready: {self._size} tab(s) "
            f"across {len(self._contexts)} browser(s)"
        )

    async def _attach(self, endpoint: str):
        """Attach to a browser you started yourself, or explain why it could not."""
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

    async def open(self, timeout_s: Optional[float] = None) -> _Tab:
        """Lease a tab and put it in a fresh conversation.

        Retries once if the leased tab turns out to be dead, because
        ``release`` will have queued a fresh replacement by then and failing the
        request while a healthy tab sits idle would be a zero for nothing.
        """
        if not self._started:
            await self.start()
        wait_s = self._lease_timeout_s if timeout_s is None else timeout_s
        last: Optional[BaseException] = None
        for _ in range(2):
            try:
                tab = await asyncio.wait_for(self._free.get(), timeout=wait_s)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"no free {self.site.name} tab within {wait_s:g}s — every tab is "
                    f"busy. Raise {self.site.env_prefix}_TABS_PER_BROWSER, add a "
                    f"browser, or lower MINER_MAX_CONCURRENT_REQUESTS."
                ) from None
            tab.leased = True
            try:
                await tab.start()
                return tab
            except asyncio.CancelledError:
                # CancelledError is a BaseException, so the `except Exception`
                # below would miss it and the tab would be lost: never requeued,
                # never disposed, still counted. Enough cancellations and the
                # pool bleeds to zero and open() blocks on an empty queue while
                # /health stays green. Requeue synchronously — awaiting while
                # unwinding a cancellation is not guaranteed to complete.
                tab.leased = False
                if tab.alive:
                    self._free.put_nowait(tab)
                else:
                    self._retire(tab)
                raise
            except Exception as exc:  # noqa: BLE001
                # start() already marked it dead; release() disposes and replaces it.
                last = exc
                await self.release(tab)
        raise last if last is not None else RuntimeError("could not lease a tab")

    def _retire(self, tab: _Tab) -> None:
        """Book-keeping for a tab that is gone. No I/O, so it is cancel-safe."""
        self._lost += 1
        if tab in self._tabs:
            self._tabs.remove(tab)
        self._size = max(0, self._size - 1)

    async def release(self, tab: _Tab) -> None:
        """Return a tab to the pool, or retire and replace a dead one.

        Recycling a dead tab is the failure that matters here: it would fail
        every future request leased onto it, so capacity is rebuilt instead.
        """
        if not tab.leased:
            # Already released. Queuing it again would hand one page to two
            # tasks, which corrupts both answers and is near-impossible to
            # diagnose from the outside.
            return
        tab.leased = False
        if self._closing:
            # A solve cancelled by shutdown releases its tab AFTER aclose() has
            # run. Requeuing then would put a closed page back in the pool and
            # respawning would talk to a severed connection, driving _size
            # negative and blaming a login for what is purely ordering.
            await tab.dispose()
            return
        if tab.alive:
            await self._free.put(tab)
            return
        await tab.dispose()
        self._retire(tab)
        # Same browser, same label, same endpoint — so a replacement that also
        # fails still reports which browser needs attention.
        replacement = await self._spawn(tab.context, tab.label, tab.source)
        if replacement is not None:
            await self._free.put(replacement)
            self._size += 1  # _retire took one off; this puts the capacity back
            print(f"[{self.site.name}] tab {tab.label} replaced after failure")
            return
        print(
            f"[{self.site.name}] WARN: tab {tab.label} retired and could not be "
            f"replaced; {self._size} tab(s) left"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.site.name,
            "tabs": self._size,
            "idle": self._free.qsize(),
            "browsers": len(self._contexts),
            "endpoints": list(self._cdp),
            "lost": self._lost,
        }

    async def aclose(self) -> None:
        """Close the tabs this pool opened, then disconnect. Never closes your browser."""
        await self._teardown()

    async def _teardown(self) -> None:
        """Undo whatever was set up, from any stage. Safe to call twice.

        Keyed off ``_pw`` rather than ``_started`` so a start() that failed
        part-way is still cleaned up — that is the path where the driver
        process would otherwise be left running.
        """
        if self._pw is None:
            return
        # Announce it before any awaiting: a solve cancelled by this shutdown
        # releases its tab part-way through, and release() has to know not to
        # requeue a page that is about to be closed.
        self._closing = True
        # Drain the queue first so nothing is handed out mid-shutdown, then
        # close every tab this pool opened — leased ones included, or they would
        # be left open in your browser.
        while not self._free.empty():
            self._free.get_nowait()
        for tab in self._tabs:
            tab.alive = False
            await tab.dispose()
        self._tabs.clear()
        for browser in self._browsers:
            try:
                # For a CDP-attached browser this severs the connection; the
                # browser process you started keeps running, with your login.
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        self._browsers.clear()
        self._contexts.clear()
        self._size = 0
        try:
            await self._pw.stop()
        finally:
            self._pw = None
            self._started = False
            self._closing = False

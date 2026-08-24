"""ChatGPT-over-CDP backend, adapted from fnlich/Automation for live serving.

The answer-detection logic here is a direct port of
``homework_automation_cdp.py``: identify the new reply by its
``data-message-id`` rather than by message count, treat it as finished only
when the Stop button is gone AND the text is unchanged across two polls, and
start every problem in a fresh conversation. That logic is the hard-won part
and it is preserved as-is.

Three things had to change for a miner:

* **Async.** ``sync_playwright`` refuses to run inside a running asyncio event
  loop, and the miner is an asyncio HTTP server. This uses ``async_api``, so
  concurrent solves are ordinary tasks instead of thread-affine handles.
* **A pool instead of a queue.** The batch script pulls the next problem off a
  filesystem queue; a miner has work pushed at it and must serve several at
  once. The same insight from ``run_parallel.py`` still applies and is what the
  pool is built on: one browser per ChatGPT account gives a true N-fold rate
  limit, so tabs are leased from a pool that can span several CDP ports.
* **Deadlines everywhere.** A batch job can wait five minutes; here a late
  answer is worth exactly zero, so every wait is bounded by the caller.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

CHATGPT_URL = os.environ.get("CHATGPT_URL", "https://chatgpt.com/")

# Selectors, unchanged from the automation script.
COMPOSER = "#prompt-textarea"
SEND_BUTTON = '[data-testid="send-button"]'
STOP_BUTTON = '[data-testid="stop-button"]'
ASSISTANT_MSG = '[data-message-author-role="assistant"]'

POLL_S = float(os.environ.get("CHATGPT_POLL_S", "2"))


class _Tab:
    """One ChatGPT tab: a single conversation slot leased from the pool.

    ``alive`` is the pool's health signal. A tab whose browser or page has gone
    away must never be recycled: leasing it again fails the next request too,
    and a miner that keeps handing out dead tabs returns zeros indefinitely
    while still answering /health. Any Playwright failure clears the flag, and
    the pool disposes of the tab and tries to spawn a replacement.
    """

    def __init__(self, pool: "ChatGPTPool", page, context, label: str):
        self._pool = pool
        self._page = page
        self.context = context
        self.label = label
        self.uses = 0
        self.alive = True

    async def start(self) -> None:
        """Open a fresh conversation so each task begins with empty context.

        Context bleed is not a cosmetic problem for a miner: a model that can
        still see the previous task's code will happily blend the two, and the
        result fails the hidden suite in a way that is very hard to diagnose.
        """
        try:
            await self._page.goto(CHATGPT_URL, wait_until="domcontentloaded")
            await self._page.wait_for_selector(COMPOSER, timeout=30_000)
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
        composer = self._page.locator(COMPOSER)
        await composer.click(timeout=ui_ms)
        # insert_text handles newlines safely — Enter alone would submit early.
        await self._page.keyboard.insert_text(text)
        await self._page.locator(SEND_BUTTON).click(timeout=ui_ms)

    async def send(self, text: str, timeout_s: float) -> str:
        page = self._page
        self.uses += 1
        # Reserve a slice of the caller's budget for getting the prompt in;
        # the rest is for waiting on the answer.
        submit_budget_s = max(5.0, min(20.0, timeout_s * 0.3))
        try:
            id_before = await self._last_message_id()
            await asyncio.wait_for(
                self._submit(text, int(submit_budget_s * 1000)),
                timeout=submit_budget_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - includes the submit timeout
            # The prompt never reached ChatGPT. Treat the tab as unusable so
            # the pool replaces it rather than failing the next request too.
            self.alive = False
            print(f"[chatgpt] tab {self.label} failed to submit: {type(exc).__name__}")
            return ""

        deadline = time.monotonic() + max(1.0, timeout_s)
        stable: Optional[str] = None
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(POLL_S)
                # Still generating or thinking while the Stop button is present.
                if await page.locator(STOP_BUTTON).count() > 0:
                    stable = None
                    continue
                messages = page.locator(ASSISTANT_MSG)
                n = await messages.count()
                if n == 0:
                    continue
                reply = messages.nth(n - 1)
                reply_id = await reply.get_attribute("data-message-id")
                if reply_id is not None and reply_id == id_before:
                    continue  # still the previous answer; ours hasn't rendered yet
                text_now = await self._read(reply)
                if text_now is None:
                    continue
                if text_now == stable:
                    return text_now  # finished: no Stop button and unchanged text
                stable = text_now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the page died mid-read
            self.alive = False
            print(f"[chatgpt] tab {self.label} died while reading: {type(exc).__name__}")
        # Timed out mid-stream: a partial answer still beats nothing, because
        # the verifier can grade it and may repair it on the next attempt.
        return stable or ""

    async def _last_message_id(self) -> Optional[str]:
        messages = self._page.locator(ASSISTANT_MSG)
        n = await messages.count()
        if n == 0:
            return None
        return await messages.nth(n - 1).get_attribute("data-message-id")

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


class ChatGPTPool:
    """A pool of ChatGPT tabs, optionally spread across several browsers.

    Each CDP port is expected to be a browser logged into a DIFFERENT ChatGPT
    account: accounts are the real rate-limit unit, so N accounts give N times
    the throughput, exactly as in ``run_parallel.py``. Tabs within one browser
    share that browser's account and therefore its limit.
    """

    def __init__(
        self,
        ports: list[int],
        *,
        host: str = "127.0.0.1",
        tabs_per_browser: int = 2,
    ):
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
        """Open one logged-in ChatGPT tab, or return None with a reason logged."""
        try:
            page = await context.new_page()
            await page.goto(CHATGPT_URL, wait_until="domcontentloaded")
            await page.wait_for_selector(COMPOSER, timeout=60_000)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[chatgpt] WARN: tab {label} never reached the composer "
                f"({type(exc).__name__}) — is this profile logged in?"
            )
            return None
        return _Tab(self, page, context, label)

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
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        for port in self._ports:
            endpoint = f"http://{self._host}:{port}"
            try:
                browser = await self._pw.chromium.connect_over_cdp(endpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"[chatgpt] WARN: cannot attach to {endpoint}: {exc}")
                continue
            self._browsers.append(browser)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            for i in range(self._tabs_per_browser):
                tab = await self._spawn(context, f"{port}#{i + 1}")
                if tab is None:
                    continue
                await self._free.put(tab)
                self._size += 1
                print(f"[chatgpt] tab {tab.label} ready")
        if self._size == 0:
            raise RuntimeError(
                "No usable ChatGPT tabs. Start Chrome with "
                "--remote-debugging-port=<port> --user-data-dir=<dir>, open "
                "https://chatgpt.com in it, and log in."
            )
        print(f"[chatgpt] pool ready: {self._size} tab(s) across {len(self._browsers)} browser(s)")

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
            print(f"[chatgpt] tab {tab.label} replaced after failure")
            return
        self._size -= 1
        print(
            f"[chatgpt] WARN: tab {tab.label} retired and could not be replaced; "
            f"{self._size} tab(s) left"
        )

    def stats(self) -> dict[str, Any]:
        return {
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

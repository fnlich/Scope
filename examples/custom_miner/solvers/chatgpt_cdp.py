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
    """One ChatGPT tab: a single conversation slot leased from the pool."""

    def __init__(self, pool: "ChatGPTPool", page, label: str):
        self._pool = pool
        self._page = page
        self.label = label
        self.uses = 0

    async def start(self) -> None:
        """Open a fresh conversation so each task begins with empty context.

        Context bleed is not a cosmetic problem for a miner: a model that can
        still see the previous task's code will happily blend the two, and the
        result fails the hidden suite in a way that is very hard to diagnose.
        """
        await self._page.goto(CHATGPT_URL, wait_until="domcontentloaded")
        await self._page.wait_for_selector(COMPOSER, timeout=30_000)

    async def send(self, text: str, timeout_s: float) -> str:
        page = self._page
        self.uses += 1
        id_before = await self._last_message_id()

        composer = page.locator(COMPOSER)
        await composer.click()
        # insert_text handles newlines safely — Enter alone would submit early.
        await page.keyboard.insert_text(text)
        await page.locator(SEND_BUTTON).click()

        deadline = time.monotonic() + max(1.0, timeout_s)
        stable: Optional[str] = None
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

    async def start(self) -> None:
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
                label = f"{port}#{i + 1}"
                try:
                    page = await context.new_page()
                    await page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                    await page.wait_for_selector(COMPOSER, timeout=60_000)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[chatgpt] WARN: tab {label} never reached the composer "
                        f"({type(exc).__name__}) — is this profile logged in?"
                    )
                    continue
                await self._free.put(_Tab(self, page, label))
                self._size += 1
                print(f"[chatgpt] tab {label} ready")
        if self._size == 0:
            raise RuntimeError(
                "No usable ChatGPT tabs. Start Chrome with "
                "--remote-debugging-port=<port> --user-data-dir=<dir>, open "
                "https://chatgpt.com in it, and log in."
            )
        print(f"[chatgpt] pool ready: {self._size} tab(s) across {len(self._browsers)} browser(s)")

    async def open(self) -> _Tab:
        """Lease a tab and put it in a fresh conversation."""
        tab = await self._free.get()
        try:
            await tab.start()
        except Exception:
            # A tab that cannot even open a chat is dead to us; drop it rather
            # than cycling it back into the pool to fail the next request too.
            self._lost += 1
            self._size -= 1
            await tab.dispose()
            raise
        return tab

    async def release(self, tab: _Tab) -> None:
        await self._free.put(tab)

    def stats(self) -> dict[str, Any]:
        return {
            "tabs": self._size,
            "idle": self._free.qsize(),
            "browsers": len(self._browsers),
            "lost": self._lost,
        }

    async def aclose(self) -> None:
        while not self._free.empty():
            await (await self._free.get()).dispose()
        for browser in self._browsers:
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw is not None:
            await self._pw.stop()

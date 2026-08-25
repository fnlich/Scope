"""ChatGPT-in-Firefox backend, adapted from fnlich/Automation for live serving.

The answer-detection logic is a direct port of ``homework_automation_cdp.py``:
identify the new reply by its ``data-message-id`` rather than by message count,
treat it as finished only when the Stop button is gone AND the text is unchanged
across two polls, and start every problem in a fresh conversation. That logic is
the hard-won part; it now lives in ``browser_pool.py``, shared with the Claude
backend, and this module is just ChatGPT's half of it — the URL, the selectors,
and the fact that ChatGPT *does* have a per-message id.

Three things had to change for a miner, and all three are in ``browser_pool.py``:

* **Async.** ``sync_playwright`` refuses to run inside a running asyncio event
  loop, and the miner is an asyncio HTTP server. The pool uses ``async_api``,
  so concurrent solves are ordinary tasks instead of thread-affine handles.
* **A pool instead of a queue.** The batch script pulls the next problem off a
  filesystem queue; a miner has work pushed at it and must serve several at
  once. The same insight from ``run_parallel.py`` still applies and is what the
  pool is built on: one profile per ChatGPT account gives a true N-fold rate
  limit, so tabs are leased from a pool spanning several profiles.
* **Deadlines everywhere.** A batch job can wait five minutes; here a late
  answer is worth exactly zero, so every wait is bounded by the caller.
"""

from __future__ import annotations

import os

from .browser_pool import BrowserPool, Site, _Tab  # noqa: F401  (_Tab re-exported for tests)
from .config import selectors

# Selectors, unchanged from the automation script, but each is now the first
# candidate in an overridable list — see browser_pool's module docstring.
COMPOSER = "#prompt-textarea"
SEND_BUTTON = '[data-testid="send-button"]'
STOP_BUTTON = '[data-testid="stop-button"]'
ASSISTANT_MSG = '[data-message-author-role="assistant"]'


def chatgpt_site() -> Site:
    return Site(
        name="chatgpt",
        env_prefix="CHATGPT",
        url=os.environ.get("CHATGPT_URL", "https://chatgpt.com/"),
        composer=selectors("CHATGPT_COMPOSER", (COMPOSER, 'div[contenteditable="true"]')),
        send=selectors("CHATGPT_SEND_BUTTON", (SEND_BUTTON, 'button[aria-label*="Send"]')),
        busy=selectors("CHATGPT_STOP_BUTTON", (STOP_BUTTON, 'button[aria-label*="Stop"]')),
        assistant=selectors("CHATGPT_ASSISTANT", (ASSISTANT_MSG,)),
        message_id_attr="data-message-id",
        poll_s=float(os.environ.get("CHATGPT_POLL_S", "2")),
    )


class ChatGPTPool(BrowserPool):
    """A pool of ChatGPT tabs, optionally spread across several profiles."""

    def __init__(
        self,
        profiles: list[str],
        *,
        tabs_per_profile: int = 2,
        headless: bool = True,
        cdp=None,
    ):
        super().__init__(
            chatgpt_site(), profiles,
            tabs_per_profile=tabs_per_profile, headless=headless, cdp=cdp,
        )

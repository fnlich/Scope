"""ChatGPT-in-Chrome backend, adapted from fnlich/Automation for live serving.

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
  pool is built on: one browser per ChatGPT account gives a true N-fold rate
  limit, so tabs are leased from a pool spanning several browsers.
* **Deadlines everywhere.** A batch job can wait five minutes; here a late
  answer is worth exactly zero, so every wait is bounded by the caller.
"""

from __future__ import annotations

import os

from .browser_pool import Site, _Tab  # noqa: F401  (_Tab re-exported for tests)
from .config import selectors

# Selectors, unchanged from the automation script, but each is now the first
# candidate in an overridable list — see browser_pool's module docstring.
COMPOSER = "#prompt-textarea"
# Both spellings, newest first: a live page was found using `chat-input-send`
# where this had only ever known `send-button`. The aria-label fallback had been
# quietly carrying the submit ever since, which is exactly the kind of drift
# `python -m solvers.doctor chatgpt` exists to make visible.
SEND_BUTTON = '[data-testid="chat-input-send"]'
SEND_BUTTON_LEGACY = '[data-testid="send-button"]'
STOP_BUTTON = '[data-testid="chat-input-stop"]'
STOP_BUTTON_LEGACY = '[data-testid="stop-button"]'
ASSISTANT_MSG = '[data-message-author-role="assistant"]'

# Same hazard as Claude's artifacts panel, different name: a long program is
# exactly when a model moves the answer out of the message and into a side
# panel, where the reader cannot see it. The reply then looks like prose with
# no code, which costs a whole solve to discover.
NUDGE = (
    "Reply directly in the chat with one ordinary fenced code block, however "
    "long the program is. Do not use canvas."
)


def chatgpt_site() -> Site:
    return Site(
        name="chatgpt",
        env_prefix="CHATGPT",
        url=os.environ.get("CHATGPT_URL", "https://chatgpt.com/"),
        composer=selectors("CHATGPT_COMPOSER", (COMPOSER, 'div[contenteditable="true"]')),
        send=selectors(
            "CHATGPT_SEND_BUTTON",
            (SEND_BUTTON, SEND_BUTTON_LEGACY, 'button[aria-label*="Send"]'),
        ),
        busy=selectors(
            "CHATGPT_STOP_BUTTON",
            (STOP_BUTTON, STOP_BUTTON_LEGACY, 'button[aria-label*="Stop"]'),
        ),
        assistant=selectors("CHATGPT_ASSISTANT", (ASSISTANT_MSG,)),
        # An in-app new chat rather than a reload; see `_Tab.start`. A miss here
        # only costs speed, since the reset falls back to loading `url`.
        new_chat=selectors(
            "CHATGPT_NEW_CHAT",
            ('[data-testid="create-new-chat-button"]', 'button[aria-label*="New chat"]'),
        ),
        # The code block's own control. NOT "Copy response" (the whole message)
        # and emphatically NOT "Run code". See `_Tab._copied_code`.
        copy=selectors("CHATGPT_COPY", ('button[aria-label="Copy"]',)),
        message_id_attr="data-message-id",
        nudge=os.environ.get("CHATGPT_NUDGE", NUDGE),
        poll_s=float(os.environ.get("CHATGPT_POLL_S", "2")),
    )

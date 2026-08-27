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
# Fallbacks behind it, and the reason this role gets them at all: it is the one
# whose failure is TOTAL. A composer or copy control that stops matching
# degrades -- the submit falls back, the copy falls back to scraping the DOM.
# An assistant selector that matches nothing reads no answer at all, for every
# task, until somebody notices. A live run showed what that costs:
#
#     no assistant selector matched anything. Tried
#     ['[data-message-author-role="assistant"]']
#
# One candidate is one deploy away from that, and Claude's list has had three
# since it was written. What these are NOT is a fix for that particular tab:
# five other ChatGPT tabs answered the same minute with the same selector, so
# the page on 9227 was in a bad state rather than a new shape. These are
# insurance against the shape changing under all of them at once.
#
# Both are assistant-only BY CONSTRUCTION -- the attribute value and the class
# name each say so -- which is the hazard `claude_web`'s docstring documents:
# a candidate that also matched the user's turn would have the miner hand its
# own prompt back as the answer. `_Tab.send`'s echo guard is the backstop, and
# `test_no_assistant_candidate_can_match_a_user_turn` is the check.
#
# They are unverified against a live page, unlike the first. Run
# `python -m solvers.doctor chatgpt --probe` to see which one your account's
# DOM actually has, and pin it in CHATGPT_ASSISTANT.
ASSISTANT_TURN = 'article[data-turn="assistant"]'
ASSISTANT_AGENT_TURN = ".agent-turn"

# Same hazard as Claude's artifacts panel, different name: a long program is
# exactly when a model moves the answer out of the message and into a side
# panel, where the reader cannot see it. The reply then looks like prose with
# no code, which costs a whole solve to discover.
NUDGE = (
    "START your reply with the fenced block. No preamble and no explanation — "
    "an answer that arrives after a paragraph of prose may not arrive at all. "
    "Reply directly in the chat with exactly the ordinary fenced block or "
    "blocks the output contract above asks for and nothing else, however long "
    "they are. Do not use canvas. Do not run code and do not try to "
    "compile or test anything — there is no toolchain here and every tool "
    "call is time the answer does not get."
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
        assistant=selectors(
            "CHATGPT_ASSISTANT",
            (ASSISTANT_MSG, ASSISTANT_TURN, ASSISTANT_AGENT_TURN),
        ),
        # An in-app new chat rather than a reload; see `_Tab.start`. A miss here
        # only costs speed, since the reset falls back to loading `url`.
        new_chat=selectors(
            "CHATGPT_NEW_CHAT",
            ('[data-testid="create-new-chat-button"]', 'button[aria-label*="New chat"]'),
        ),
        # The code block's own control. NOT "Copy response" (the whole message)
        # and emphatically NOT "Run code". See `_Tab._copied_code`.
        copy=selectors("CHATGPT_COPY", ('button[aria-label="Copy"]',)),
        copy_name=os.environ.get("CHATGPT_COPY_NAME", "copy").casefold(),
        # The answer as it came off the wire, ahead of every render.
        # `CHATGPT_STREAM=0` turns the capture off entirely;
        # `CHATGPT_STREAM_FIRST=1` promotes it over what the page shows.
        stream=os.environ.get("CHATGPT_STREAM", "1") != "0",
        stream_first=os.environ.get("CHATGPT_STREAM_FIRST", "0") == "1",
        message_id_attr="data-message-id",
        nudge=os.environ.get("CHATGPT_NUDGE", NUDGE),
        poll_s=float(os.environ.get("CHATGPT_POLL_S", "2")),
    )

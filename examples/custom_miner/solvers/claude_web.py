"""Claude-in-the-browser backend: drive claude.ai in Chrome, no API key.

Same shape as the ChatGPT backend and the same machinery underneath
(``browser_pool.py``): the pool attaches to a Chrome you started and signed in
to, holds one tab per concurrent solve, starts every task in a fresh
conversation, and feeds the reply into the self-verify-and-repair loop in
``verify.py``. Your existing Claude subscription is the quota; no API key is
read anywhere.

    ./scripts/start_debug_browser.sh --port 9222   # then sign in to claude.ai
    CLAUDE_CDP=9222 python examples/custom_miner/run_miner.py

There is deliberately no API-key path in this package. The consequence is that
a browser backend has no supported fallback to switch to, which is exactly why
the doctor tool below is not optional and why running ``claude,chatgpt`` as a
chain is worth the second account.

## Read this before you trust the selectors below

These selectors are candidate lists, not verified facts. claude.ai's markup is
not a published interface and it changes; the defaults are ordered
most-specific-first with a broader fallback behind each. **Verify them once
against your own logged-in browser before pointing a registered hotkey at
this:**

    cd examples/custom_miner
    python -m solvers.doctor claude --probe

That prints which candidate matched for each role and shows the text it read
back, and every role is overridable in ``.env`` (``|`` between candidates):

    CLAUDE_COMPOSER='div[contenteditable="true"].ProseMirror'
    CLAUDE_ASSISTANT='div[data-is-streaming]'

Three specific hazards, and what is done about each:

* **The assistant selector must not match your own message.** If it did, the
  miner would hand its own prompt back as the answer — no error, no empty
  reply, just a permanent zero. So no deliberately-generic fallback is offered
  for this role, and ``_Tab.send`` refuses any reply that starts with the
  prompt it just sent, naming the doctor in the log.
* **Artifacts.** Long code can land in the artifacts side panel instead of the
  message, where the reader cannot see it. The prompt therefore asks explicitly
  for an inline code block.
* **A 'still generating' selector that is always true** would make every answer
  look unfinished and burn the whole budget. The pool checks each busy
  candidate against a freshly-loaded idle page at startup and drops any that
  matches.
"""

from __future__ import annotations

import os

from .browser_pool import Site
from .config import selectors

# Claude answers in the chat by default, but long code can be moved to the
# artifacts panel, which lives outside the message the reader scrapes. One
# sentence is cheaper than trying to scrape the panel.
NUDGE = (
    "Reply directly in the chat with one ordinary fenced code block, however "
    "long the program is. Do not create an artifact. Do not use the analysis "
    "tool and do not try to compile, run or test anything — there is no "
    "toolchain here and every tool call is time the answer does not get."
)


def claude_site() -> Site:
    return Site(
        name="claude",
        env_prefix="CLAUDE",
        url=os.environ.get("CLAUDE_URL", "https://claude.ai/new"),
        composer=selectors(
            "CLAUDE_COMPOSER",
            (
                'div[contenteditable="true"].ProseMirror',
                '[data-testid="chat-input"] div[contenteditable="true"]',
                'fieldset div[contenteditable="true"]',
                'div[contenteditable="true"]',
            ),
        ),
        send=selectors(
            "CLAUDE_SEND_BUTTON",
            (
                'button[aria-label="Send message"]',
                'button[data-testid="send-button"]',
                'button[aria-label*="Send"]',
            ),
        ),
        busy=selectors(
            "CLAUDE_STOP_BUTTON",
            (
                'button[aria-label="Stop response"]',
                'button[aria-label*="Stop"]',
                'div[data-is-streaming="true"]',
            ),
        ),
        # No broad fallback on purpose: a selector that also matched the user's
        # turn would make the miner answer with its own prompt.
        assistant=selectors(
            "CLAUDE_ASSISTANT",
            (
                "div[data-is-streaming]",
                "div.font-claude-message",
                '[data-testid="assistant-message"]',
            ),
        ),
        # Starting a new chat in the app beats reloading the page: no bundle
        # refetch, no app boot. `$=` on the href so a full URL matches as well
        # as a bare path. Nothing here matching is not a failure -- the reset
        # falls back to loading `url`, which is what it always did.
        new_chat=selectors(
            "CLAUDE_NEW_CHAT",
            (
                'a[href$="/new"]',
                'button[aria-label="New chat"]',
                'a[aria-label="New chat"]',
            ),
        ),
        # claude.ai has no per-message id attribute we can rely on, so the reply
        # is identified by position. Sound here because every task starts a
        # fresh conversation.
        # The code block's own control. NOT the message-level "Copy", which
        # takes the whole reply. See `_Tab._copied_code`.
        copy=selectors("CLAUDE_COPY", ('button[aria-label="Copy to clipboard"]',)),
        copy_name=os.environ.get("CLAUDE_COPY_NAME", "copy").casefold(),
        # The answer as it came off the wire, ahead of every render.
        # `CLAUDE_STREAM=0` turns the capture off entirely;
        # `CLAUDE_STREAM_FIRST=1` promotes it over what the page shows.
        stream=os.environ.get("CLAUDE_STREAM", "1") != "0",
        stream_first=os.environ.get("CLAUDE_STREAM_FIRST", "0") == "1",
        message_id_attr=None,
        nudge=os.environ.get("CLAUDE_NUDGE", NUDGE),
        poll_s=float(os.environ.get("CLAUDE_POLL_S", "2")),
    )

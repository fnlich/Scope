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
import re
import time
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Any, NamedTuple, Optional, Sequence

# The port `scripts/start_debug_browser.sh` uses unless told otherwise, and so
# the port every backend assumes when `<PREFIX>_CDP` is not set.
DEFAULT_CDP_PORT = 9222

# How long the in-app "new chat" click gets before the reload takes over. Short
# on purpose: the click exists only because it is faster than a reload, so
# waiting the site's full ready timeout for one that did not route would make
# the fast path slower than the thing it replaces.
NEW_CHAT_TIMEOUT_MS = 5_000


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
    # The site's own "new chat" control. Optional: when nothing here matches,
    # a fresh conversation is had by reloading ``url`` instead, which always
    # works. So a wrong or missing candidate costs speed, never correctness.
    new_chat: tuple[str, ...] = ()
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
        pool: "BrowserFleet",
        page,
        context,
        label: str,
        site: Site,
        *,
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
        # A per-tab Site — required, because a fleet spans providers and has no
        # single one of its own. The fleet prunes the selectors this page cannot
        # use before handing it over, so the answer path never has to guess
        # whether a raising selector means a bad override or a dead page.
        self.site: Site = site
        self._composer = composer or (self.site.composer[0] if self.site.composer else "")
        self._sent = ""
        # Assistant selector latched for the current send(); see _messages().
        self._assistant: Optional[str] = None
        # ...and the index of the reply element, so two streaming branches
        # cannot be read alternately. See _new_reply.
        self._reply_index: Optional[int] = None
        self._reply_key: Optional[str] = None
        self._warned_echo = False
        self._warned_branches = False
        self.uses = 0
        self.alive = True
        # True while this tab is known to be sitting in an EMPTY conversation.
        # It starts true because `_spawn` builds a tab only after loading
        # `site.url` — the site's own new-conversation URL — and seeing the
        # composer. That is what makes the first task on a new tab need no
        # reset at all, rather than reloading a page that just loaded.
        self._fresh = True
        # Set while this tab is out on loan. The pool uses it to make a second
        # release() a no-op: without it a double close would queue the same tab
        # twice and two tasks would end up driving one page.
        self.leased = False

    async def start(self) -> None:
        """Put this tab in an empty conversation, as cheaply as that can be done.

        Context bleed is not a cosmetic problem for a miner: a model that can
        still see the previous task's code will happily blend the two, and the
        result fails the hidden suite in a way that is very hard to diagnose.
        So every task must begin with an empty transcript. What *is* negotiable
        is the price, and it used to be the highest one available — a full page
        load per task, on the task's own clock:

        1. **Already empty** — a tab straight from ``_spawn``, or one whose last
           task never submitted. Nothing to do. This is the whole of the "open
           the tab once" saving: without it every tab reloads the page it just
           loaded before it may answer anything.
        2. **The site's own new-chat control** — an in-app route change. No
           bundle refetch, no app boot, no re-auth; typically under a second
           against several for a reload. Taken only when the transcript is
           demonstrably gone afterwards (see ``_new_chat``).
        3. **Reload ``site.url``** — slow but certain, and therefore the
           fallback that lets tiers 1 and 2 be attempted at no risk.

        Note what is NOT here: closing the tab. A tab is opened once and reused
        for the life of the miner. Closing and reopening per task would throw
        away the page's warm state for nothing, and — since a tab is only
        replaced when it *dies* — would make ordinary work indistinguishable
        from failure in the logs.
        """
        if self._fresh and await self._composer_present():
            return
        if await self._new_chat():
            return
        await self._reload()

    async def _composer_present(self) -> bool:
        """Is the composer still there? One count — no navigation, no wait.

        Guards the tier-1 shortcut. A tab can sit idle for hours between tasks
        and the page under it may have been reloaded or redirected by the site
        in the meantime, which would leave ``_fresh`` describing a page that no
        longer exists. Confirming the composer costs a millisecond and turns
        that from a failed task into a reset.
        """
        try:
            return await self._page.locator(self._composer).count() > 0
        except Exception:  # noqa: BLE001 - a dead page cannot be shortcut past
            return False

    async def _new_chat(self) -> bool:
        """Start a new conversation from inside the app. True only if it worked.

        Never trusts the click: a control that did not route — a changed DOM, a
        modal in the way, a click landing on a disabled button — leaves the
        previous task's transcript in place, and the next prompt would then be
        answered with that context in view. That is precisely the failure this
        method exists to avoid, and it is invisible from the outside: the answer
        comes back promptly and is simply wrong. So the transcript is checked
        afterwards, and anything short of proof falls through to the reload.

        Raises nothing, ever — including from resolving the selectors, which is
        why that happens inside the try. ``open()`` retires a tab whose
        ``start()`` raised, on the promise that ``start()`` marked it dead
        first; only ``_reload`` does that. A raise from here would reach
        ``open()`` with ``alive`` still true, and the tab would be requeued —
        the recycled-dead-tab failure this pool exists to prevent.
        """
        try:
            button = await self._first_match(self.site.new_chat)
            if button is None:
                return False
            await self._page.locator(button).first.click(timeout=NEW_CHAT_TIMEOUT_MS)
            found = await wait_for_any(
                self._page, self.site.composer, NEW_CHAT_TIMEOUT_MS
            )
            if found is None or not await self._transcript_cleared():
                return False
            self._composer = found
        except Exception:  # noqa: BLE001 - the reload below is the fallback
            return False
        self._fresh = True
        return True

    async def _transcript_cleared(self) -> bool:
        """No assistant message anywhere on the page.

        Deliberately reads every candidate rather than the latched one: this
        runs between sends, when the latch is meaningless, and a transcript that
        survives under *any* candidate is a transcript that survived. A selector
        that raises proves nothing, so it counts as not cleared.
        """
        for selector in self.site.assistant:
            try:
                if await self._page.locator(selector).count() > 0:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    async def _reload(self) -> None:
        """Reload the site's new-conversation URL. Slow, but always works."""
        try:
            await self._page.goto(self.site.url, wait_until="domcontentloaded")
            found = await wait_for_any(self._page, self.site.composer, 30_000)
            if found is None:
                raise RuntimeError("composer did not reappear after navigation")
            self._composer = found
        except Exception:
            self.alive = False
            raise
        self._fresh = True

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
        # Dirty from here on, whatever happens next: a submit that raises
        # part-way may still have left the prompt in the composer or even sent
        # it. Claiming freshness we cannot prove would skip the next reset and
        # bleed this task into the following one.
        self._fresh = False
        if self.site.nudge:
            text = f"{text}\n\n{self.site.nudge}"
        self._sent = text
        # One selector for the whole call: `_first_match` is free to pick a
        # different candidate once the reply renders, and comparing a count taken
        # from one selector against a count taken from another is meaningless.
        self._assistant = None
        self._reply_index = None
        self._reply_key = None
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
        #
        # What `stable` holds is the WHOLE rendered message, not just the code
        # `_read` pulls out of it, and that distinction decides whether this
        # returns an answer or a model's rough work. `_read` discards the prose
        # whenever the message has code blocks, so a message growing by prose
        # ALONE reads identically twice running -- which is precisely what a
        # reply looks like in the gap between a reasoning section that quoted
        # some code and the final answer, not yet written. Comparing the code
        # alone accepts that gap as the finished reply and hands the grader a
        # fragment lifted out of the model's thinking; comparing the message
        # closes it. The busy selector is supposed to cover this and usually
        # does, but it is per-site, overridable, and dropped at startup when it
        # matches an idle page -- so it is a guard, not the guarantee.
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
                text_now, busy, whole = await asyncio.wait_for(
                    self._poll(before), timeout=remaining
                )
                if text_now is not None:
                    best = text_now  # keep it even mid-generation
                if busy or text_now is None:
                    stable = None  # still typing, or ours has not rendered yet
                    continue
                mark = (text_now, whole)
                if mark == stable:
                    return text_now  # finished: not busy, and nothing moved
                stable = mark
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
    ) -> tuple[Optional[str], bool, str]:
        """One read round: ``(code_or_None, still_generating, whole_message)``.

        The read happens whether or not the model is still typing. Checking busy
        first and returning early — the obvious shape — means a reply that never
        stops streaming before the deadline is never read at all, so a timeout
        mid-answer returns nothing instead of the part that had arrived.

        The message as rendered comes back beside the code because both of the
        callers that matter need the prose ``_read`` throws away: the echo guard
        to recognise the miner's own prompt, and ``send``'s completion test to
        tell a finished reply from one still being written around code it has
        already shown. One DOM read serves both.
        """
        busy = await self._busy_now()
        reply = await self._new_reply(before)
        if reply is None:
            return None, busy, ""  # ours hasn't rendered yet
        text_now = await self._read(reply)
        whole = await self._whole(reply)
        if text_now is None or self._echoes_prompt(whole or text_now):
            return None, busy, whole
        return text_now, busy, whole

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
        """The reply to THIS send, latched for the rest of it.

        One send can produce more than one assistant message: ChatGPT sometimes
        runs a side-by-side comparison and streams two candidate answers for a
        single prompt. Reading "the last message" then means reading whichever
        branch happens to be last at that instant — and while both are
        streaming, that flips. The text never settles, so the poll loop never
        sees two identical reads, and the whole budget is spent before the
        deadline forces a partial answer out.

        So the FIRST message that was not there before is latched on sight and
        read for the rest of the send. Both branches are real answers; what
        matters is committing to one. With a single reply this is exactly the
        old behaviour — index ``count_before`` is the only new message.
        """
        count_before, id_before = before
        messages = await self._messages()
        if messages is None:
            return None
        count = await messages.count()
        if count == 0:
            return None
        if self._reply_key is not None:
            # Latched by id: survives the branches being reordered, which an
            # index does not. Cheap — there are never many messages on screen.
            for i in range(count):
                node = messages.nth(i)
                if await node.get_attribute(self.site.message_id_attr) == self._reply_key:
                    return node
            # The id we latched is gone, and that is NOT proof the answer is.
            # A chat UI re-renders a streaming message and can swap the id it
            # first painted for the one the server assigns. Returning None here
            # returned it for the REST OF THE SEND: the read never recovered,
            # `best` stayed empty, and a perfectly good answer was reported as
            # "the reply contained no code". So fall back to the position the
            # reply was latched at and adopt whatever id it wears now.
            if self._reply_index is not None and count > self._reply_index:
                fresh = messages.nth(self._reply_index)
                self._reply_key = await fresh.get_attribute(self.site.message_id_attr)
                return fresh
            return None
        if self._reply_index is not None:
            return messages.nth(self._reply_index) if count > self._reply_index else None

        if count > count_before:
            if count - count_before > 1 and not self._warned_branches:
                self._warned_branches = True
                print(
                    f"[{self.site.name}] note: tab {self.label} got "
                    f"{count - count_before} answers to one prompt (a "
                    f"side-by-side comparison). Reading the first and ignoring "
                    f"the rest."
                )
            fresh = messages.nth(count_before)
            # Position is recorded even when an id is available -- it is the id
            # latch's only way back if the id changes under it.
            self._reply_index = count_before
            # Prefer an id: two branches keep their ids whatever order they are
            # painted in, so the read cannot drift from one answer to the other.
            if self.site.message_id_attr:
                self._reply_key = await fresh.get_attribute(self.site.message_id_attr)
            return fresh
        # Some sites replace the last message rather than appending one, so a
        # changed id still means a new reply even when the count did not move.
        if self.site.message_id_attr:
            last = messages.nth(count - 1)
            mid = await last.get_attribute(self.site.message_id_attr)
            if mid is not None and mid != id_before:
                self._reply_key = mid
                self._reply_index = count - 1
                return last
        return None

    async def _busy_now(self) -> bool:
        for selector in self.site.busy:
            if await self._page.locator(selector).count() > 0:
                return True
        return False

    def _echoes_prompt(self, whole: str) -> bool:
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
    async def _whole(reply) -> str:
        """The message exactly as rendered, or ``""`` if it cannot be read.

        Swallowing the failure is deliberate: this is only ever used to compare
        one poll against the next, and a page caught mid-navigation should cost
        a poll, not the tab. ``send`` degrades to the old code-only test when
        this returns ``""`` twice, which is what it did before this existed.
        """
        try:
            return await reply.inner_text()
        except Exception:  # noqa: BLE001 - mid-navigation, or the node went away
            return ""

    @staticmethod
    async def _read(reply) -> Optional[str]:
        """The reply's code blocks, re-fenced, or the message text.

        Every block, not a guess at which one matters. Taking only the last
        used to submit the "example usage" snippet whenever a model appended
        one -- the solution was right there in the block before it, and the
        whole solve was spent to report that `solve` was never defined.
        Choosing between them needs the entrypoint, which belongs to the task
        rather than the tab, so hand them all over and let the caller pick.
        """
        code_blocks = reply.locator("pre code")
        count = await code_blocks.count()
        if count > 0:
            fenced = []
            for i in range(count):
                # textContent, not innerText, and measured rather than assumed:
                # claude.ai splits a <code> into `data-code-line-group` blocks,
                # and innerText adds a line break at every block boundary -- a
                # 15-line program came back as 17. Rust does not care about a
                # stray blank line; a Python multi-line string literal does, and
                # the corruption is invisible until a hidden test disagrees
                # about the text. Each line already carries its own newline, so
                # the raw text IS the source. Only safe on the code block: at
                # message level textContent returns the markup's own
                # indentation as whitespace, which is why the fallback below
                # and the echo guard both still read innerText.
                block = await code_blocks.nth(i).text_content() or ""
                if not block.strip():
                    continue
                # Markdown's own rule: the fence must outrun any backtick run
                # inside, or a block containing ``` would close itself early.
                longest = max((len(r) for r in re.findall(r"`+", block)), default=0)
                fence = "`" * max(3, longest + 1)
                fenced.append(f"{fence}\n{block}\n{fence}")
            if fenced:
                return "\n".join(fenced)
        text = (await reply.inner_text()).strip()
        return text or None

    @property
    def provider(self) -> str:
        """Which model answers in this tab — 'claude' or 'chatgpt'."""
        return self.site.name

    async def close(self) -> None:
        """Hand the tab back. Reused if it is still healthy, retired if not."""
        await self._pool.release(self)

    async def dispose(self) -> None:
        """Close this tab's page. Safe to call twice."""
        try:
            await self._page.close()
        except Exception:  # noqa: BLE001 - the browser may already be gone
            pass


class Browser(NamedTuple):
    """One browser you started, and which provider is signed in to it."""

    endpoint: str
    site: Site


# Written into `window.name` on every tab this fleet opens. It is per-tab, it
# survives navigation, and it reads back as "" on tabs you opened yourself — so
# a restart can find its own leftovers without ever touching yours.
TAB_MARK = "hone-miner"


class BrowserFleet:
    """Every tab across every browser you started, leased one per task.

    One fleet, not one pool per provider. A task does not care whether it is
    answered by Claude or ChatGPT, so the useful unit is "the next free tab in
    the fleet" — which spreads load over your accounts, the thing that actually
    limits throughput. Tabs are handed out first-in-first-out and returned to the
    back, so leases rotate; they are also enqueued browser-interleaved, so two
    tasks arriving together land on two different accounts rather than doubling
    up on one.

    Nothing here launches or closes a browser. See the module docstring for why
    that is the whole point rather than a limitation.
    """

    def __init__(
        self,
        browsers: Sequence[Browser],
        *,
        tabs_per_browser: int = 2,
    ):
        self._browsers_wanted = list(browsers)
        self._tabs_per_browser = max(1, int(tabs_per_browser))
        self._pw = None
        self._contexts: list[Any] = []
        self._connections: list[Any] = []  # attached; disconnected, never closed
        # Every tab ever handed out, so shutdown can close the pages this fleet
        # opened in YOUR browsers — including the ones currently leased, which a
        # free-queue-only sweep would leave behind.
        self._tabs: list[_Tab] = []
        self._free: asyncio.Queue[_Tab] = asyncio.Queue()
        # The rotation: the endpoints that actually produced tabs, in the order
        # you configured them, and whose turn is next. Requests go 1st browser,
        # 2nd, ... nth, 1st again. Seeding the queue interleaved gives that too
        # while tabs come back in the order they went out, but they do not: a
        # tab is released when its task ENDS, and concurrent tasks end out of
        # order, so a queue alone drifts out of rotation within a few requests.
        # This makes the rotation the rule rather than a side effect of timing.
        self._order: list[str] = []
        self._cursor = 0
        self._size = 0
        self._lost = 0
        self._reclaimed = 0
        self._started = False
        self._closing = False
        # How long open() waits for a free tab before saying so. Unbounded would
        # be worse: a caller with its own deadline gets cancelled, and a caller
        # without one hangs with no explanation.
        self._lease_timeout_s = float(os.environ.get("MINER_TAB_WAIT_S", "120"))
        self._start_lock = asyncio.Lock()

    # -- setup -------------------------------------------------------------- #
    async def start(self) -> None:
        """Attach to every browser and fill the fleet. Idempotent.

        Safe to call explicitly at startup, and called lazily by ``open()`` so
        the fleet also works under a host that has no startup hook — a local test
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
        # Spawned per browser, then enqueued interleaved: with two tabs each
        # across browsers A and B the queue is A1, B1, A2, B2, so the first two
        # concurrent tasks use two different accounts instead of hammering one.
        columns: list[list[_Tab]] = []
        for browser in self._browsers_wanted:
            context = await self._attach(browser)
            if context is None:
                continue
            self._contexts.append(context)
            await self._reclaim(context, browser)
            label = browser.endpoint.split("://", 1)[-1]
            spawned: list[_Tab] = []
            for i in range(self._tabs_per_browser):
                tab = await self._spawn(context, browser, f"{label}#{i + 1}")
                if tab is not None:
                    spawned.append(tab)
                    print(f"[{browser.site.name}] tab {tab.label} ready")
            if spawned:
                columns.append(spawned)

        # Only browsers that produced a tab are in the rotation: one that could
        # not be attached to is not an active browser, and leaving it in would
        # spend a turn on nothing.
        self._order = [column[0].source for column in columns if column]
        self._cursor = 0
        for row in zip_longest(*columns):
            for tab in row:
                if tab is not None:
                    self._free.put_nowait(tab)
                    self._size += 1

        if self._size == 0:
            wanted = ", ".join(
                f"{b.site.name}@{b.endpoint}" for b in self._browsers_wanted
            ) or "(none configured)"
            raise RuntimeError(
                f"No usable tabs. Wanted: {wanted}. To fix: start each browser "
                f"with --remote-debugging-port (scripts/start_debug_browser.sh "
                f"does it), sign in to the provider by hand, and confirm the "
                f"endpoint answers."
            )
        print(f"[fleet] ready: {self._size} tab(s) across {self._describe()}")

    def _describe(self) -> str:
        counts: dict[str, int] = {}
        for tab in self._tabs:
            counts[tab.site.name] = counts.get(tab.site.name, 0) + 1
        return ", ".join(f"{n}×{c}" for c, n in sorted(counts.items())) or "nothing"

    async def _attach(self, browser: Browser):
        """Attach to a browser you started yourself, or explain why it could not."""
        site = browser.site
        try:
            connection = await self._pw.chromium.connect_over_cdp(browser.endpoint)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{site.name}] WARN: cannot attach to {browser.endpoint}: "
                f"{str(exc).splitlines()[0]}\n"
                f"         Start Chrome/Chromium with --remote-debugging-port and a "
                f"--user-data-dir, and confirm {browser.endpoint}/json/version answers."
            )
            return None
        self._connections.append(connection)
        return (
            connection.contexts[0]
            if connection.contexts
            else await connection.new_context()
        )

    async def _reclaim(self, context, browser: Browser) -> None:
        """Close tabs a previous run left behind in this browser.

        A miner under a process supervisor gets SIGKILLed sooner or later, and an
        unclean exit never runs shutdown — so every restart would add another set
        of dead tabs to a browser that stays up for weeks. Ours are identifiable
        because ``_spawn`` stamps ``window.name``; yours never are, so signing-in
        tabs are never touched.
        """
        found = 0
        for page in list(getattr(context, "pages", []) or []):
            try:
                if not str(await page.evaluate("window.name") or "").startswith(TAB_MARK):
                    continue
                await page.close()
            except Exception:  # noqa: BLE001 - a page we cannot ask about is not ours
                continue
            found += 1
        self._reclaimed += found
        if found:
            print(
                f"[{browser.site.name}] reclaimed {found} tab(s) at {browser.endpoint} "
                f"left by an earlier run of this miner"
            )

    async def _spawn(self, context, browser: Browser, label: str) -> Optional[_Tab]:
        """Open one signed-in tab, or return None with a reason logged."""
        site, page = browser.site, None
        hint = (
            f"open {site.url} in the Chrome on {browser.endpoint} and sign in by hand"
        )
        try:
            page = await context.new_page()
            await page.goto(site.url, wait_until="domcontentloaded")
            # Stamp it before anything can go wrong later, so even a tab that
            # fails its checks below is reclaimable after an unclean exit.
            await page.evaluate(f"window.name = {TAB_MARK + '/' + site.name!r}")
            composer = await wait_for_any(page, site.composer, site.ready_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{site.name}] WARN: tab {label} never loaded {site.url} "
                f"({type(exc).__name__}) — is it signed in? {hint}"
            )
            await _close_quietly(page)
            return None
        if composer is None:
            print(
                f"[{site.name}] WARN: tab {label}: no composer selector matched. "
                f"Either it is not signed in ({hint}), or the DOM changed — run "
                f"`python -m solvers.doctor {site.name}` and set "
                f"{site.env_prefix}_COMPOSER."
            )
            await _close_quietly(page)
            return None
        checked = replace(
            site,
            send=await valid_selectors(page, site.send, site.name, "send"),
            # Unlike the roles below, an empty result here is survivable: with
            # no usable new-chat control every reset falls back to a reload.
            new_chat=await valid_selectors(
                page, site.new_chat, site.name, "new chat"
            ),
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
            await _close_quietly(page)
            return None
        tab = _Tab(
            self, page, context, label, checked,
            composer=composer, source=browser.endpoint,
        )
        self._tabs.append(tab)
        return tab

    # -- leasing ------------------------------------------------------------ #
    async def open(
        self, avoid: Optional[str] = None, timeout_s: Optional[float] = None
    ) -> _Tab:
        """Lease the next free tab and put it in a fresh conversation.

        ``avoid`` asks for a different provider than the one named — used to get
        a second opinion from the other model. It is a preference, not a
        guarantee: if every free tab is the avoided provider, one of those is
        still better than failing the task.
        """
        if not self._started:
            await self.start()
        wait_s = self._lease_timeout_s if timeout_s is None else timeout_s
        last: Optional[BaseException] = None
        for _ in range(2):
            tab = await self._lease(avoid, wait_s)
            tab.leased = True
            try:
                await tab.start()
                return tab
            except asyncio.CancelledError:
                # CancelledError is a BaseException, so the `except Exception`
                # below would miss it and the tab would be lost: never requeued,
                # never disposed, still counted. Enough cancellations and the
                # fleet bleeds to zero and open() blocks on an empty queue while
                # /health stays green. Requeue synchronously — awaiting while
                # unwinding a cancellation is not guaranteed to complete.
                tab.leased = False
                if tab.alive:
                    self._free.put_nowait(tab)
                else:
                    self._retire(tab)
                raise
            except Exception as exc:  # noqa: BLE001
                # start() already marked it dead; release() disposes and replaces
                # it, so the second pass gets the healthy replacement rather than
                # failing the task while a good tab sits idle.
                last = exc
                await self.release(tab)
        raise last if last is not None else RuntimeError("could not lease a tab")

    async def _lease(self, avoid: Optional[str], wait_s: float) -> _Tab:
        """The next free tab, taken in browser rotation order.

        Request 1 goes to the 1st browser, request 2 to the 2nd, ... request n
        to the nth, request n+1 back to the 1st. That is the point of running
        several browsers at all: the account is the rate limit, so spreading
        consecutive tasks across accounts is what buys the throughput.

        The one deviation, and it is deliberate: if the browser whose turn it is
        has no free tab, the turn passes to the next browser that does rather
        than waiting for it. A miner is paid for answers that arrive before the
        deadline, so idling behind one busy account while another sits free
        would trade real money for a tidier sequence. The cursor follows the
        browser actually used, so the rotation resumes from there instead of
        replaying the gap.
        """
        try:
            first = await asyncio.wait_for(self._free.get(), timeout=wait_s)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"no free tab within {wait_s:g}s — every tab is busy. Add a "
                f"browser, raise MINER_TABS_PER_BROWSER, or lower "
                f"MINER_MAX_CONCURRENT_REQUESTS."
            ) from None
        # Everything from here down is synchronous, and has to be: draining the
        # queue and putting back what we did not take is one step, and an await
        # in the middle of it is a cancellation point that would drop every
        # drained tab on the floor.
        pool = [first]
        while not self._free.empty():
            pool.append(self._free.get_nowait())
        chosen = self._pick(pool, avoid)
        for tab in pool:
            if tab is not chosen:
                self._free.put_nowait(tab)
        return chosen

    def _pick(self, pool: list[_Tab], avoid: Optional[str]) -> _Tab:
        """Choose from the free tabs and advance the rotation. No I/O.

        ``avoid`` outranks the rotation. It is asked for only by the second
        opinion, after the first model has already failed to produce a
        verifiable answer, and getting the OTHER model is the entire value of
        that attempt — spending it on the same model to keep the browsers in
        order would be the wrong trade. Among the tabs that satisfy it, the one
        whose browser is nearest in rotation order wins, so the ordinary case is
        unaffected.
        """
        n = len(self._order)
        position = {endpoint: i for i, endpoint in enumerate(self._order)}

        def rank(tab: _Tab) -> tuple[int, int]:
            wrong_model = 1 if (avoid is not None and tab.site.name == avoid) else 0
            if n == 0:  # a fleet assembled without _fill, i.e. in a test
                return (wrong_model, 0)
            index = position.get(tab.source)
            # Distance forward from whoever's turn it is, so the browser due
            # next is 0, the one after it 1, and so on around the ring.
            turn = n if index is None else (index - self._cursor) % n
            return (wrong_model, turn)

        # min() keeps the first of any tie, so tabs of equal standing are still
        # handed out in the order they were freed.
        chosen = min(pool, key=rank)
        index = position.get(chosen.source)
        if index is not None:
            self._cursor = (index + 1) % n
        return chosen

    def _retire(self, tab: _Tab) -> None:
        """Book-keeping for a tab that is gone. No I/O, so it is cancel-safe."""
        self._lost += 1
        if tab in self._tabs:
            self._tabs.remove(tab)
        self._size = max(0, self._size - 1)

    async def release(self, tab: _Tab) -> None:
        """Return a tab to the fleet, or retire and replace a dead one.

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
            # run. Requeuing then would put a closed page back in the fleet and
            # respawning would talk to a severed connection, driving _size
            # negative and blaming a login for what is purely ordering.
            await tab.dispose()
            return
        if tab.alive:
            await self._free.put(tab)
            return
        await tab.dispose()
        self._retire(tab)
        browser = Browser(tab.source, tab.site)
        replacement = await self._spawn(tab.context, browser, tab.label)
        if replacement is not None:
            await self._free.put(replacement)
            self._size += 1  # _retire took one off; this puts the capacity back
            print(f"[{tab.site.name}] tab {tab.label} replaced after failure")
            return
        print(
            f"[{tab.site.name}] WARN: tab {tab.label} retired and could not be "
            f"replaced; {self._size} tab(s) left"
        )

    # -- reporting and shutdown --------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        per_provider: dict[str, int] = {}
        for tab in self._tabs:
            per_provider[tab.site.name] = per_provider.get(tab.site.name, 0) + 1
        return {
            "tabs": self._size,
            "idle": self._free.qsize(),
            "by_provider": per_provider,
            "browsers": len(self._contexts),
            "endpoints": [b.endpoint for b in self._browsers_wanted],
            "lost": self._lost,
            "reclaimed": self._reclaimed,
        }

    async def aclose(self) -> None:
        """Close the tabs this fleet opened, then disconnect. Never closes a browser."""
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
        # close every tab this fleet opened — leased ones included, or they would
        # be left open in your browser.
        while not self._free.empty():
            self._free.get_nowait()
        for tab in self._tabs:
            tab.alive = False
            await tab.dispose()
        self._tabs.clear()
        for connection in self._connections:
            try:
                # For a CDP-attached browser this severs the connection; the
                # browser process you started keeps running, with your login.
                await connection.close()
            except Exception:  # noqa: BLE001
                pass
        self._connections.clear()
        self._contexts.clear()
        self._size = 0
        try:
            await self._pw.stop()
        finally:
            self._pw = None
            self._started = False
            self._closing = False


async def _close_quietly(page) -> None:
    if page is None:
        return
    try:
        await page.close()
    except Exception:  # noqa: BLE001 - do not leave a stray tab behind
        pass

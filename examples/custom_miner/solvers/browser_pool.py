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

# One scanner for fenced markdown, not two. This module used to carry its own
# copy while the extractor in `prompts` matched fences with a regular
# expression instead -- and the two disagreed about where a block ends, in four
# measured ways that each cost a whole answer. Two readers of the same markdown
# that disagree is a bug waiting for the reply that tells them apart.
from .prompts import fenced_blocks as _fenced_blocks

# The port `scripts/start_debug_browser.sh` uses unless told otherwise, and so
# the port every backend assumes when `<PREFIX>_CDP` is not set.
DEFAULT_CDP_PORT = 9222

# How long the in-app "new chat" click gets before the reload takes over. Short
# on purpose: the click exists only because it is faster than a reload, so
# waiting the site's full ready timeout for one that did not route would make
# the fast path slower than the thing it replaces.
NEW_CHAT_TIMEOUT_MS = 5_000

# How long ONE read round gets before it is abandoned and retried.
#
# Not a tidiness limit. A poll that straddles a DOM change finds a node, and
# then waits on it after the site has already replaced it -- and Playwright
# auto-waits 30 SECONDS on a locator by default. Bounded only by the send's
# remaining budget, that single poll eats every second the send had left, on a
# page where the answer is on screen the whole time. Measured on the DOM
# transition claude.ai actually makes: an 8s send spent 7.85s inside one
# `inner_text()` and returned "". The next poll re-resolves and costs
# milliseconds, so the only thing this buys back is the entire budget.
POLL_READ_TIMEOUT_S = 5.0

# How long ONE click on a copy control gets. Tight on purpose: it runs after the
# read already came back empty, so it is spending time the solve has notionally
# run out of, to turn a guaranteed zero into a possible answer. Per click, not
# per send -- a reply with several code blocks is several clicks, and the whole
# phase is bounded separately by COPY_PHASE_TIMEOUT_S.
COPY_TIMEOUT_MS = 1_500

# How long each phase AFTER the read loop gets.
#
# Everything past the loop runs on time the solve has notionally already spent:
# the loop exits at the deadline, and then the copy control, the network stream
# and the post-mortem each go back to the page. Every one of those is a
# Playwright call, and a Playwright call auto-waits 30 SECONDS unless told
# otherwise -- measured here against a node that was resolved and then removed,
# which is the ordinary shape of a page still settling after an answer:
#
#     button.inner_text()   raised after 30.0s
#     node.inner_text()     raised after 30.0s
#     code.text_content()   raised after 30.0s
#
# Three phases, several calls apiece, and `send` can return two minutes after
# the `timeout_s` it was given -- long after the validator stopped listening,
# and with nothing gained, because the answer it is holding was complete before
# any of it started. `_submit` has been bounded against exactly this since it
# was written; the way out was not.
#
# Each bound degrades into the reading already in hand rather than into
# nothing: copy falls back to what scraping saw, the stream leaves `best`
# untouched, and the post-mortem is a log line whose absence costs no answer.
#
# The three together must stay inside VerifyingSolver's safety margin
# (`SOLVER_SAFETY_MARGIN_S`, 20 seconds), and the reason is not tidiness either:
# `handle_request` wraps the whole solve in an `asyncio.wait_for` and answers
# 504 -- NOTHING -- when it is exceeded. Overrunning does not deliver the answer
# late; it throws away an answer already in hand. 5 + 4 + 2 = 11 leaves the last
# grade, the tab close, and the archive-and-sign on the way out the rest of it.
COPY_PHASE_TIMEOUT_S = 5.0
STREAM_PHASE_TIMEOUT_S = 4.0
POSTMORTEM_TIMEOUT_S = 2.0
FULL_TAIL_S = COPY_PHASE_TIMEOUT_S + STREAM_PHASE_TIMEOUT_S + POSTMORTEM_TIMEOUT_S


def tail_budget(timeout_s: float) -> float:
    """How long the post-read phases may take, for a read of ``timeout_s``.

    The eleven seconds above are sized against the solver's safety margin, and
    that margin is sized against the deadlines this subnet usually advertises.
    `TaskRequest.deadline_s` is only `Field(gt=0.0, le=3600.0)`, and at the
    small end the arithmetic inverts: `solve_task` falls back to half the
    timeout when the margin would leave nothing, and then a fixed eleven-second
    tail is larger than the whole request. Measured, before this existed:

        deadline  budget   read + tail   504 at
            25     12.5        23.5        25    ok
            15      7.5        18.5        15    504, NO ANSWER
            10      5.0        16.0        10    504, NO ANSWER
             5      5.0        16.0         5    504, NO ANSWER

    Every one of those is a total loss with no log line to explain it: the
    solve is cancelled mid-flight by `handle_request` and the validator gets an
    error. A miner that cannot answer a short request at all is worse than one
    that answers it badly.

    Half the read, because a rescue that costs more than half again what the
    attempt cost has stopped being a rescue. At any read of 22 seconds or more
    this returns the full eleven and nothing changes, which is every read on
    the deadlines actually seen in production.
    """
    return min(FULL_TAIL_S, max(0.0, float(timeout_s)) / 2.0)

# How long a tab gets to put SOMETHING on screen before it is declared unreadable.
#
# A healthy chat UI paints the assistant bubble within a second or two of the
# submit, empty, and fills it in afterwards. So "the reply element has still not
# appeared" and "the model is thinking" are different states, and the read loop
# used to treat them identically -- it polled a tab that could not be read at
# all for the whole slice, then reported the same empty answer a slow model
# would have produced.
#
# Measured on a live miner, both shapes of the failure:
#
#     chatgpt tab 9227: no assistant selector matched anything
#     claude  tab 9222: matched 1 message(s), the same as before the prompt
#                       was sent -- the answer never rendered
#
# Each burned 191 seconds of a 225 second budget and then a 29 second repair
# round on the same dead conversation, leaving 5 seconds -- too few to ask
# anyone else. Five healthy tabs sat idle through both. Detecting it here turns
# a whole-task loss into a 30 second one, and hands the rest of the budget to a
# tab that works.
#
# Generous on purpose: this must never fire on a model that is merely slow. A
# reply that has rendered -- even as an empty bubble with a cursor in it --
# resets nothing, because the check is "did it EVER appear", not "has it
# finished".
BLIND_TAB_GRACE_S = 30.0

# Installed before the copy button is clicked. It makes the page hand the code
# to US instead of to the operating system.
#
# Reading the system clipboard back would be the obvious way to use a copy
# button, and it is the one thing that must not happen here: the clipboard is
# ONE object shared by every tab, every browser on the display, and every miner
# process the operator runs. Measured, two tabs in one browser: A wrote
# 'TAB-A-CODE', B wrote 'TAB-B-CODE', A read back 'TAB-B-CODE'. A pool that
# read the clipboard would submit another task's program, silently, whenever
# two solves overlapped. Intercepting `writeText` keeps the value in the page,
# so nothing is shared and nothing can collide -- and it also leaves the
# operator's own clipboard alone while the miner works.
_COPY_HOOK = """() => {
  if (window.__honeHooked) return;
  window.__honeHooked = true;
  window.__honeCopied = null;
  const c = navigator.clipboard;
  if (!c) return;
  c.writeText = (t) => { window.__honeCopied = String(t); return Promise.resolve(); };
  c.write = (items) => {
    try {
      const item = items && items[0];
      if (item && item.getType) {
        return item.getType('text/plain')
          .then((b) => b.text())
          .then((t) => { window.__honeCopied = t; });
      }
    } catch (e) { /* fall through: the read below reports nothing copied */ }
    return Promise.resolve();
  };
}"""


# How long the streamed answer gets to be reconstructed, in total. It is a
# `page.evaluate` over text already in the tab, so this is a guard against a
# wedged renderer rather than a real budget.
STREAM_TIMEOUT_MS = 2_000

# Installed with `add_init_script` on every tab, before the site's own code
# runs, so it survives navigation, reloads and the in-app "new chat".
#
# This is the answer BEFORE the page exists. Everything else this class reads
# is downstream of a render: `pre code` is the source after a syntax
# highlighter rebuilt it as DOM, and even the copy control is the framework's
# own copy of a block it has already parsed. The wire is the markdown the model
# emitted, and it is the only source that still has something to say when the
# DOM read comes back empty -- which is exactly the failure that surfaces,
# much later and much less usefully, as "the reply contained no code".
#
# The patch is written to be unable to break the site, because it runs on a
# signed-in account the operator cares about:
#   * `res.clone()`, never `res.body.tee()` plus a constructed `Response`. A
#     Response built by hand loses `url` and `redirected`, and a chat UI that
#     reads either would break in a way that looks like the site's bug.
#   * the original response object is returned untouched, on every path,
#     including every path where this code throws.
#   * only streaming content types are touched, so ordinary requests -- images,
#     avatars, telemetry -- keep their bodies unread and uncloned.
#   * the buffer is bounded in both records and characters, because a tab lives
#     for weeks.
_STREAM_HOOK = r"""() => {
  if (window.__honeStreamHooked) return;
  window.__honeStreamHooked = true;
  window.__honeStreams = [];
  window.__honeStreamSeq = 0;
  var MAX_RECORDS = 6;
  var MAX_CHARS = 2000000;
  var original = window.fetch;
  if (typeof original !== 'function') return;
  var keep = function (res) {
    var type = '';
    try { type = (res.headers && res.headers.get('content-type')) || ''; } catch (e) { return; }
    if (!/event-stream|ndjson/i.test(type)) return;
    var copy = res.clone();
    if (!copy.body || !copy.body.getReader) return;
    var rec = { seq: ++window.__honeStreamSeq, text: '' };
    window.__honeStreams.push(rec);
    while (window.__honeStreams.length > MAX_RECORDS) window.__honeStreams.shift();
    var reader = copy.body.getReader();
    var dec = new TextDecoder();
    var pump = function () {
      return reader.read().then(function (r) {
        if (r.done) return;
        if (rec.text.length < MAX_CHARS) rec.text += dec.decode(r.value, { stream: true });
        return pump();
      });
    };
    pump().catch(function () {});
  };
  window.fetch = function () {
    var out;
    try {
      out = original.apply(this, arguments);
    } catch (e) {
      throw e;                       // the site's own failure, unchanged
    }
    if (!out || typeof out.then !== 'function') return out;
    return out.then(function (res) {
      try { keep(res); } catch (e) { /* the site gets its response regardless */ }
      return res;
    });
  };
}"""

# `add_init_script` takes SCRIPT SOURCE, not a function the way `evaluate`
# does: handed the arrow function above it would build a function, discard it,
# and install nothing at all -- measured, `window.__honeStreamHooked` came back
# False and every stream went uncaptured, in total silence. Kept as a separate
# constant so the hook itself stays callable from `evaluate` in tests.
_STREAM_INSTALL = f"({_STREAM_HOOK})()"

# Reconstructs the assistant's markdown out of whatever the site streamed,
# without knowing the site's schema -- because there is no published schema to
# know. Both formats in use here are undocumented, private, and changed without
# notice, so anything hard-coded is a thing that breaks silently one Tuesday.
#
# What holds across both is structural: an SSE stream carries many small JSON
# events, and the answer is the ONE field appended to over and over. So group
# every string leaf and take the group that accumulated the most text. On
# Claude that group is `/delta/text`; on ChatGPT's operation encoding it is
# `/v`. Neither name appears below.
#
# A group is a path PLUS the short strings that came with it in the same event,
# and that second half is not decoration. ChatGPT sends the answer and its own
# bookkeeping down the same `/v` field, told apart only by the sibling
# operation: `{p:/message/content/parts/0, o:append, v:"..."}` is the message,
# `{p:/message/status, o:replace, v:"finished_successfully"}` is not. Measured,
# grouping on the path alone appended `finished_successfully` to the answer.
#
# Those qualifiers carry forward, because the encoding omits them: after one
# append, ChatGPT sends bare `{v:"..."}` meaning "same operation as before".
# Reading each of those as its own group splits the answer and drops its
# opening chunk -- which is the whole first line when a reply opens with code.
#
# Two kinds of string are dropped outright or they win on volume alone:
#   * reasoning, by name, on the path or on its qualifiers. Claude streams
#     thinking as `/delta/thinking` and ChatGPT under `/message/content/
#     thoughts/...`, and a long reasoning block dwarfs the answer. Submitting
#     the model's rough work is a failure this miner has already had once.
#   * tool arguments. A model that reaches for its tools streams what it is
#     asking them to do as `/delta/partial_json`, and that is not an answer --
#     it is a shell command with a program quoted inside it. Measured: a 5,442
#     byte tool call beat the 54 byte answer beside it on volume alone, and the
#     wire handed back `{"command": "cat > main.rs << 'EOF' ..."}`.
#   * enums. `/delta/type` repeats "text_delta" on every event and can outweigh
#     a short program. A handful of distinct values across many events is a
#     tag, not prose.
#
# Not handled, and deliberately: an SSE payload split across several `data:`
# lines. Neither site does it, and guessing at reassembly would corrupt more
# than it recovered.
_STREAM_READ = r"""(since) => {
  var recs = (window.__honeStreams || []).filter(function (r) { return r.seq > since; });
  if (!recs.length) return null;
  var NOISE = /think|thought|reason|scratch|signature|citation|websearch|tool|input_json|partial_json/i;
  // Bookkeeping keys, with an optional prefix: `content_type` and
  // `conversation_id` are as much tags as `type` and `id`, and matching only
  // the bare word let `content_type` become the answer -- measured, a reply
  // with no text in it reconstructed as the single word "text".
  var TAG = /(^|\/)([a-z]+_)*(id|role|type|model|status|name|kind|reason|created|uuid|parent|slug|index|version|mime|lang|language)$/i;
  // Text this stream attributes to somebody other than the model. A chat
  // response carries the CONVERSATION, not just the reply: ChatGPT opens with
  // a snapshot holding the user's own turn under `author.role = "user"`, and
  // "the field appended to most" is then the PROMPT whenever the answer is
  // shorter than it -- which it usually is. Measured on ChatGPT's real payload
  // shape: a 1,384 character user turn beat the 41 character answer beside it,
  // and two validators were sent this miner's own instructions as Rust.
  var NOT_THE_MODEL = /\/role=(user|system|tool)\b/i;
  var leaves = function (node, path, out) {
    if (node === null || typeof node === 'undefined') return out;
    if (typeof node === 'string') { out.push([path, node]); return out; }
    if (typeof node !== 'object') return out;
    if (Array.isArray(node)) {
      for (var i = 0; i < node.length; i++) leaves(node[i], path + '/*', out);
      return out;
    }
    for (var k in node) {
      if (Object.prototype.hasOwnProperty.call(node, k)) leaves(node[k], path + '/' + k, out);
    }
    return out;
  };
  var best = null;
  for (var r = 0; r < recs.length; r++) {
    var buckets = Object.create(null);
    var sticky = Object.create(null);
    var lines = recs[r].text.split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (line.indexOf('data:') === 0) line = line.slice(5).trim();
      if (!line || line === '[DONE]') continue;
      if (line.charAt(0) !== '{' && line.charAt(0) !== '[') continue;
      var event;
      try { event = JSON.parse(line); } catch (e) { continue; }
      var found = leaves(event, '', []);
      for (var j = 0; j < found.length; j++) {
        var path = found[j][0], value = found[j][1];
        if (TAG.test(path)) continue;
        var marks = [];
        for (var m = 0; m < found.length; m++) {
          if (found[m][0] === path) continue;
          if (found[m][1].length <= 64) marks.push(found[m][0] + '=' + found[m][1]);
        }
        var sig = marks.sort().join('|');
        if (sig) { sticky[path] = sig; } else { sig = sticky[path] || ''; }
        if (NOISE.test(path) || NOISE.test(sig)) continue;
        if (NOT_THE_MODEL.test(sig)) continue;
        var key = path + '\u0000' + sig;
        var e = buckets[key];
        if (!e) { e = buckets[key] = { text: '', count: 0, distinct: Object.create(null), seen: 0 }; }
        e.text += value;
        e.count++;
        if (e.seen < 8 && !(value in e.distinct)) { e.distinct[value] = 1; e.seen++; }
      }
    }
    for (var b in buckets) {
      var q = buckets[b];
      if (q.seen <= 3 && q.count >= 8) continue;
      if (best === null || q.text.length > best.length) best = q.text;
    }
  }
  return best;
}"""


# What the composer holds, read back before anything is submitted. Both sites
# put a contenteditable div behind the composer selector rather than a real
# input, so `input_value()` does not apply; `value ?? innerText` covers a plain
# textarea too without asking the page which it is.
_COMPOSER_TEXT_JS = (
    "el => (el.value === undefined || el.value === null ? el.innerText : el.value)"
)


# How long to let the editor finish painting what was inserted, and how often
# to look. A composer is React over ProseMirror: `insert_text` returns when the
# input event is delivered, not when the DOM shows the result, and on a box
# running four Chrome instances in 5 GB that gap is visible.
COMPOSER_SETTLE_S = 3.0
COMPOSER_POLL_S = 0.1

_WORDS_RE = re.compile(r"[0-9A-Za-z_]+")


def _words(text: str) -> list[str]:
    return _WORDS_RE.findall(text)


def _same_message(typed: str, seen: str) -> bool:
    """Is the box holding our message, and nothing that is not ours?

    Not equality, and the difference is the whole of it. The question is "did
    somebody else's text get in" -- and no editor invents a word, while every
    rich-text editor rewrites punctuation. A composer applies input rules as
    text arrives: `- ` at the start of a line becomes a bullet and `1. `
    becomes an ordered list, at which point the marker is list STRUCTURE and
    `innerText` hands the line back without it -- the digit of an ordered
    marker included. This prompt carries four of the first and five of the
    second, and demanding the text back verbatim reported those sends as
    contaminated, twice each (retyping reproduces it), and retired the tab.
    Measured on a live miner: "the composer does not hold the prompt as typed,
    twice over", on a tab that was holding the prompt exactly as intended.

    So: every word the box shows must be one of ours, in our order, and nearly
    all of ours must be there. An editor may DROP a marker; it may not add a
    word. A leftover human draft is made of words that are not ours and fails
    on the first of them, which is the job this check exists to do. A read that
    caught the editor half-painted fails the second test rather than the first,
    and `_settled_composer` is what waits that out instead of retyping into it.
    """
    ours, shown = _words(typed), _words(seen)
    if len(shown) < 0.95 * len(ours):
        return False
    i = 0
    for word in shown:
        while i < len(ours) and ours[i] != word:
            i += 1
        if i == len(ours):
            return False          # a word the box shows that we never typed
        i += 1
    return True


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
    # The code block's own "copy" control, used ONLY as a last resort — see
    # `_Tab._copied_code`. Must name the button that copies ONE code block, not
    # the one that copies the whole message and not, on ChatGPT, "Run code".
    copy: tuple[str, ...] = ()
    # What the copy control must call itself before it is clicked. See
    # `_Tab._copied_code`: the selector says where to look, this says what we
    # are willing to press. Override for a non-English UI.
    copy_name: str = "copy"
    # Read the answer off the network stream as well as off the page. See
    # `_STREAM_HOOK`: it is the markdown the model emitted, before any of this
    # became DOM, and it is the last source left when the page read comes back
    # empty. Off makes the tab behave exactly as it did before that existed.
    stream: bool = True
    # Take the streamed answer even when the page produced one of its own.
    # Off by default and deliberately so: the wire format is private and
    # undocumented, so until an operator has watched the two agree on THEIR
    # accounts, the stream only rescues an empty read and reports differences.
    # Turn it on once `python -m solvers.doctor` shows the sources agreeing.
    stream_first: bool = False
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


def _describe_char(ch: str) -> str:
    """A character a human can act on: the codepoint, not a blank space."""
    if not ch:
        return "nothing (it ends here)"
    return f"{ch!r} (U+{ord(ch):04X})"


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
        self._warned_copy = False
        # Said once when the editor reformats what we typed (see `_same_message`).
        self._warned_rewrite = False
        # A COUNT, not a flag. The other notes on this class are configuration
        # advice and say themselves once; this one is damage to the answer being
        # submitted right now, and how OFTEN it happens is the number an
        # operator needs. Silencing it after the first would have reported two
        # incidents in a run of 56 solves on two tabs -- exactly one per tab,
        # which is what a once-per-tab flag reports whether it happened twice
        # or forty times.
        self._cut_short_count = 0
        self._warned_diff = False
        self._warned_relatch = False
        self._warned_stream = False
        self._warned_stream_diff = False
        # Highest network record seen before this send's prompt went out, so a
        # reply is never reconstructed out of the PREVIOUS answer's stream.
        self._stream_before = 0
        self.uses = 0
        self.alive = True
        # Why the last `send` came back empty, or None if it did not. Three
        # causes look identical from "" and only one of them is worth another
        # prompt in the same conversation:
        #
        #   "unreadable" -- no reply ever rendered, or the tab died. Prompting
        #                   again sends the repair into a conversation that has
        #                   just proved it cannot be read.
        #   "unfinished" -- the model was still writing when the budget ran out.
        #                   A second prompt queues behind an answer that has not
        #                   been produced yet.
        #   "no-code"    -- a FINISHED reply that simply had no code block in
        #                   it. That is the model breaking the output contract,
        #                   and telling it so is exactly what fixes it.
        #
        # Callers that cannot see this attribute (any non-browser backend) get
        # the historic behaviour, because only the first two suppress a repair.
        self.empty_reason: Optional[str] = None
        # Was the model STILL WRITING when the last send stopped reading?
        #
        # Separate from `empty_reason` because it is true whether or not
        # anything was captured, and both cases matter: nothing captured, and a
        # code block captured half-written. Prompting into a conversation that
        # is mid-answer cannot help in either -- the composer is usually
        # disabled, and if it is not the prompt queues behind the answer that
        # has not arrived yet.
        self.still_writing = False
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

    async def _settled_composer(self, text: str, ui_ms: int) -> str:
        """What the composer holds once it has finished painting.

        `insert_text` returns when the input event is delivered; a React editor
        applies it a frame or more later, and reading immediately catches an
        empty box or half a prompt. That is not contamination and must not be
        treated as any: it read as "the composer does not hold the prompt as
        typed", retyped, raced again, and retired a working tab.

        Returns as soon as it matches, so the ordinary send pays one poll.

        Bounded by the caller's own slice as well as by COMPOSER_SETTLE_S: the
        submit phase has a budget, and waiting past it does not buy a send, it
        buys a timeout that retires the tab. On a one-second slice this waits
        fractions of a second and then retypes, which is the right order.
        """
        deadline = time.monotonic() + min(
            COMPOSER_SETTLE_S, max(0.1, (ui_ms / 1000.0) * 0.4)
        )
        seen = await self._composer_holds()
        while not _same_message(text, seen) and time.monotonic() < deadline:
            await asyncio.sleep(COMPOSER_POLL_S)
            seen = await self._composer_holds()
        return seen

    async def _composer_holds(self) -> str:
        """What the composer contains right now.

        Both sites put a contenteditable ProseMirror div behind the composer
        selector rather than a real input, so `input_value()` does not apply to
        either. Reading `value ?? innerText` in the page covers a plain textarea
        too, without having to ask which one this is.

        Deliberately not guarded: a composer we cannot read is a composer we
        cannot vouch for, and `_submit`'s caller already turns a raise into a
        retired tab and a solve that goes elsewhere.
        """
        composer = self._page.locator(self._composer).first
        return await composer.evaluate(_COMPOSER_TEXT_JS) or ""

    async def _clear_composer(self, ui_ms: int) -> None:
        """Empty the composer, and prove it emptied. Raises when it will not.

        This is not housekeeping. These accounts are shared with people, so the
        box can already hold somebody's half-typed message -- and `insert_text`
        inserts at the CARET, which the click above it has just placed at the
        element's centre. The prompt is spliced INTO that draft and the whole
        thing is sent as one message: their text, our prompt, the rest of their
        text. Nothing downstream would notice it. `_is_our_own_prompt` inspects
        the REPLY for our prompt's head, and in a contaminated send that head is
        still there, intact. The only symptom is a model answering a mangled
        question, which looks exactly like a hard task.
        """
        composer = self._page.locator(self._composer).first
        await composer.click(timeout=ui_ms)
        # Control, not Meta: every entrypoint here refuses to start anywhere but
        # Linux (`preflight.require_linux`), and `fill` below covers a composer
        # that swallows the shortcut anyway.
        await self._page.keyboard.press("Control+A")
        await self._page.keyboard.press("Delete")
        if not (await self._composer_holds()).strip():
            return
        # The shortcut did not reach it -- an editor with its own key handling,
        # or a selection that never landed. `fill` goes through the element
        # instead of the keyboard, and the click restores the caret it moves.
        await composer.fill("", timeout=ui_ms)
        await composer.click(timeout=ui_ms)
        left = (await self._composer_holds()).strip()
        if left:
            raise RuntimeError(
                f"the composer still holds {len(left)} character(s) we did not "
                f"type and would not clear"
            )

    async def _submit(self, text: str, ui_ms: int) -> None:
        """Put the prompt in the composer ALONE, then press send.

        Playwright's default auto-wait is 30s PER action, so an unbounded
        click/insert/click can burn ~90s that the solver's budget never
        accounted for — which would blow the response deadline and score zero
        even though the answer was on its way.

        ``.first`` on both clicks is not cosmetic: a Locator click is strict and
        RAISES when the selector matches more than one node, while the candidate
        lists deliberately end in broad fallbacks that can. Without it a page
        with two matching nodes fails every submit, and since a failed submit
        retires the tab, the pool would churn tabs forever and never answer.

        Nothing is ever sent unread. The box is emptied, the prompt is typed,
        and the box is read BACK and compared before the send button is touched
        -- because the one thing this method must never do is hand a validator's
        problem to a model wrapped in somebody else's sentence.
        """
        for attempt in (1, 2):
            await self._clear_composer(ui_ms)
            # insert_text handles newlines safely — typing them would submit early.
            await self._page.keyboard.insert_text(text)
            seen = await self._settled_composer(text, ui_ms)
            if _same_message(text, seen):
                if " ".join(text.split()) != " ".join(seen.split()) and not self._warned_rewrite:
                    self._warned_rewrite = True
                    print(
                        f"[{self.site.name}] note: tab {self.label}: the composer "
                        f"REWROTE the prompt's punctuation as it went in — every "
                        f"word is still there and in order, so this is the editor "
                        f"formatting markdown, not somebody else's text. Once per run."
                    )
                break
            if attempt == 2:
                raise RuntimeError(
                    "the composer does not hold the prompt as typed, twice over"
                )
            print(
                f"[{self.site.name}] tab {self.label}: the composer did not hold the "
                f"prompt as typed; clearing it and typing it again, once"
            )
        button = await self._first_match(self.site.send)
        if button is None:
            # No send button matched. Every one of these composers also submits
            # on Enter, and that is safe *here* only because the whole prompt,
            # newlines included, is already in the box via insert_text.
            await self._page.keyboard.press("Enter")
            return
        await self._page.locator(button).first.click(timeout=ui_ms)

    async def _open_turn(self, text: str, ui_ms: int) -> tuple[int, Optional[str]]:
        """Snapshot the conversation, then put the prompt in.

        One coroutine so `send` can bound the whole "getting the prompt in"
        phase with a single `wait_for`, rather than bounding the click and
        leaving the two reads either side of it unbounded.
        """
        before = await self._fingerprint()
        # Before the prompt, not after: the response starts streaming while
        # `_submit` is still returning, so a floor taken afterwards can already
        # have this answer's own record under it.
        self._stream_before = await self._stream_seq()
        await self._submit(text, ui_ms)
        return before

    async def _copy_phase(
        self, before: tuple[int, Optional[str]]
    ) -> tuple[Optional[list[str]], Optional[list[str]]]:
        """What the copy control gives, and what the DOM shows, for comparison.

        Returns ``(copied, rendered)``; ``rendered`` is only read when
        ``copied`` is non-empty, so it is not fetched otherwise. Separated from
        `send` for the same reason as `_open_turn`: it is three Playwright
        calls that need ONE bound around them.
        """
        reply = await self._new_reply(before)
        if reply is None:
            return None, None
        copied = await self._copied_blocks(reply)
        rendered = await self._dom_blocks(reply) if copied else None
        return copied, rendered

    async def send(
        self, text: str, timeout_s: float, extend_to_s: Optional[float] = None
    ) -> str:
        """Ask, then read. ``timeout_s`` is the slice; ``extend_to_s`` the cap.

        They differ because a slice is an internal ALLOCATION rather than a
        deadline. When the slice runs out with the model still writing, the read
        extends ONCE to ``extend_to_s`` -- the caller's real remaining budget --
        rather than stopping to do something that cannot help: the composer is
        usually disabled mid-stream, and where it is not the prompt simply
        queues behind the answer it is asking about. Waiting is the only move
        that can still produce the answer, and the payment policy agrees: a
        correct answer arriving late earns at least 95% of what the fastest
        earns, and an unfinished one earns nothing.

        ``VerifyingSolver`` no longer slices. Every read it makes is given the
        whole remaining budget, so it passes no ``extend_to_s`` and there is
        nothing here to extend into -- ``timeout_s`` IS the request's deadline,
        less what delivering the answer costs. The parameter stays for a caller
        that does hand out less than everything.
        """
        if not self.alive:
            # The pool has not recycled this tab yet. Retrying a known-dead tab
            # only burns the caller's budget one submit-timeout at a time.
            self.empty_reason = "unreadable"
            self.still_writing = False
            return ""
        # Start the clock BEFORE the submit. Deriving the read deadline after it
        # would hand the read a fresh `timeout_s` on top of however long typing
        # took — an overrun larger than the solver's whole safety margin.
        started = time.monotonic()
        deadline = started + max(1.0, timeout_s)
        # Never before the slice, and never past the caller's own budget.
        hard = started + max(max(1.0, timeout_s), float(extend_to_s or 0.0))
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
        #
        # Bounded by the read's own deadline, and that bound is the third place
        # on this branch where a floor sat above what the caller could afford.
        # `max(5.0, ...)` alone gave a one-second read a five-second submit --
        # it does not buy a better submit, it buys an overrun, and on a short
        # request `handle_request` answers 504 with nothing at all. A submit
        # that fills the whole read is a read that finds nothing, which is bad;
        # a submit that outlives it is a solve that is cancelled, which is
        # worse.
        submit_budget_s = min(
            max(5.0, min(20.0, timeout_s * 0.3)), max(1.0, timeout_s)
        )
        try:
            # The snapshot is inside the bound, not outside it. `_fingerprint`
            # ends in a `get_attribute` on the last message, and that is the
            # 30s auto-wait again: a conversation that re-renders as the
            # prompt goes out hands back a node the site has already replaced,
            # and the whole read budget is gone before a single poll runs --
            # on a page that would have answered. The budget is described as
            # the slice for getting the prompt in; this makes that true.
            before = await asyncio.wait_for(
                self._open_turn(text, int(submit_budget_s * 1000)),
                timeout=submit_budget_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - includes the submit timeout
            # The prompt never reached the model. Treat the tab as unusable so
            # the pool replaces it rather than failing the next request too.
            self.alive = False
            self.empty_reason = "unreadable"
            self.still_writing = False
            print(
                f"[{self.site.name}] tab {self.label} failed to submit: "
                f"{type(exc).__name__}: {exc}"
            )
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
        # What the PAGE gave us, block by block, for the stream to be checked
        # against. None means "not looked at yet" and is not the same as [].
        # Sized once per send: the three phases below share it in proportion,
        # so the whole tail is what `tail_budget` says and no phase can quietly
        # spend the deadline of a short request on its own.
        tail = tail_budget(timeout_s) / FULL_TAIL_S
        page_blocks: Optional[list[str]] = None
        best, stable = "", None
        # Has the reply to THIS prompt ever appeared on screen? Not "has it
        # finished" -- an empty bubble counts. A tab that never manages even
        # that inside BLIND_TAB_GRACE_S cannot be read, and every further poll
        # spends budget the healthy tabs could have had.
        submitted_at = time.monotonic()
        saw_reply = blind = False
        # Two independent readings of "is it still writing", because either
        # alone is wrong often enough to matter.
        #
        # `last_busy` is the site's own stop control. It is authoritative when
        # present and absent altogether when it is not: `usable_busy_selectors`
        # DROPS any candidate that matches an idle page at startup, so a site
        # can legitimately run with none, and then this reads False through the
        # whole of an answer that is still arriving. Measured, with the busy
        # selector dropped and the model mid-sentence at the deadline:
        #
        #     empty_reason='no-code'     <- and so a repair round fired, telling
        #                                   a model still writing its first
        #                                   answer that it had sent no code
        #
        # `grew` is the fallback and needs no selector at all: a message that
        # is longer than it was one poll ago is being written, whatever the DOM
        # calls its stop button.
        #
        # What it does NOT rescue, and neither did anything before it: with no
        # busy selector, a reply whose CODE stops changing for two polls is
        # accepted as finished by the settle test and the read exits early --
        # so a model that pauses longer than 2x poll_s mid-block can still have
        # a truncated answer taken as final. That is the known cost of losing
        # the busy selector, documented on `usable_busy_selectors`, and the
        # reason it says keeping one working is worth it. `grew` covers the
        # commoner shape by far: a model thinking in prose, where `_read`
        # returns None throughout and the read runs to its deadline.
        last_busy = False
        grew = False
        last_whole: Optional[str] = None
        def out_of_time() -> bool:
            """Stop reading? Extends once, while the answer may still arrive."""
            nonlocal deadline
            if time.monotonic() < deadline:
                return False
            # `hard <= deadline` is also what makes this happen at most once:
            # the extension below sets them equal.
            #
            # `not saw_reply` extends too, and it is the case a slice serves
            # worst: a model that thinks before it writes renders nothing while
            # it thinks (77 seconds, measured on a live tab), so stopping at the
            # slice with an empty page trades an answer that was still coming
            # for time there is nothing left to spend it on.
            #
            # With no slice -- which is every read `VerifyingSolver` makes now
            # -- `hard` equals `deadline` and none of this runs: the first test
            # returns True and the read has already had the whole budget.
            if hard <= deadline or not (last_busy or grew or not saw_reply):
                return True
            deadline = hard
            print(
                f"[{self.site.name}] tab {self.label} "
                + ("is still writing at" if saw_reply else "has rendered nothing by")
                + f" its {timeout_s:.0f}s slice; reading on to "
                f"{hard - started:.0f}s rather than "
                + (
                    "interrupting an answer that is still arriving."
                    if saw_reply
                    else "abandoning an answer that may still arrive."
                )
            )
            return time.monotonic() >= deadline

        try:
            while True:
                if out_of_time():
                    break
                remaining = max(0.0, deadline - time.monotonic())
                await asyncio.sleep(min(self.site.poll_s, remaining))
                if out_of_time():
                    break
                remaining = max(0.0, deadline - time.monotonic())
                # Every DOM call below auto-waits up to 30s on its own, which
                # would sail past the deadline the loop just checked.
                try:
                    text_now, busy, whole, on_screen = await asyncio.wait_for(
                        self._poll(before),
                        timeout=min(remaining, POLL_READ_TIMEOUT_S),
                    )
                except asyncio.TimeoutError:
                    # This read straddled a change in the page: it resolved a
                    # node and then waited on one the site had already
                    # replaced. Abandoning it costs one round; letting it run
                    # to the deadline costs the answer. See POLL_READ_TIMEOUT_S.
                    stable = None
                    continue
                last_busy = busy
                grew = last_whole is not None and whole != last_whole
                last_whole = whole
                if on_screen:
                    saw_reply = True
                elif (
                    not blind
                    and time.monotonic() - submitted_at >= BLIND_TAB_GRACE_S
                ):
                    # Nothing has rendered in the time a working tab needs to
                    # paint an empty bubble. That is worth SAYING and it is not
                    # worth acting on, and the difference cost a run.
                    #
                    # It used to stop the read here and retire the tab, on the
                    # reasoning that the budget was better spent elsewhere.
                    # Measured over a live run: every one of eighteen tabs this
                    # fired on had its answer recovered off the wire moments
                    # later -- the model was not silent, the DOM was late. So
                    # the answer was never what the bail saved; what it threw
                    # away was the CONVERSATION, and with it every repair round
                    # that solve could still have run. Fifteen answers went out
                    # with failing cases and an average of 129 unused seconds.
                    #
                    # A model that thinks before it writes renders nothing for
                    # as long as it thinks -- 77 seconds, measured on a live
                    # tab. There is no per-turn deadline to protect here, only
                    # the request's own, so the read waits for the answer.
                    blind = True
                    print(
                        f"[{self.site.name}] tab {self.label} has shown no reply yet "
                        f"after {BLIND_TAB_GRACE_S:.0f}s — still waiting. A model "
                        f"that thinks before it writes renders nothing while it "
                        f"thinks, and the answer may also arrive off the wire."
                    )
                if text_now is not None:
                    best = text_now  # keep it even mid-generation
                if busy or text_now is None:
                    stable = None  # still typing, or ours has not rendered yet
                    continue
                mark = (text_now, whole)
                if mark == stable:
                    # Finished: not busy, and nothing moved. Break rather than
                    # return, so the one exit below is the only exit — a
                    # `return` here left the fenced-block check unreachable on
                    # every successful read, which is every read that matters.
                    best = text_now
                    break
                stable = mark
        except asyncio.TimeoutError:
            pass  # out of budget mid-read; `best` still holds what we had
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the page died mid-read
            self.alive = False
            print(f"[{self.site.name}] tab {self.label} died while reading: {type(exc).__name__}")
        if self.alive and self.site.copy:
            # Once per send, never per poll: the answer is finished by now, and
            # clicking a control on every poll would be dozens of clicks a
            # solve. Scraping decided WHEN to read; the copy control decides
            # WHAT was read, because it predates the syntax highlighter.
            try:
                copied, rendered = await asyncio.wait_for(
                    self._copy_phase(before), timeout=COPY_PHASE_TIMEOUT_S * tail
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - fall back to what scraping saw
                copied = rendered = None
            page_blocks = copied
            if copied:
                # The copy control wins on FIDELITY -- it is the source before
                # the highlighter rebuilt it as DOM -- but it has no authority
                # at all on COMPLETENESS, and the two are different questions.
                #
                # Measured on a real page: the DOM held a complete 182-character
                # Rust program, the copy control gave the first 60, and those 60
                # were submitted. `rust_defect` returned None on them, because a
                # truncated program keeps its `fn main` -- so nothing downstream
                # caught it and it went to the validator as a confident answer.
                # In the live log this arrived twice, on the only two tasks that
                # spent the entire budget:
                #
                #   what the page RENDERS and what it COPIES are not the same --
                #   they differ at character 1630: rendered '\n', copied nothing
                #   (it ends here). Using the copy.
                #
                # When the copied source is the rendered source CUT SHORT, the
                # two readings were simply taken at different moments and the
                # shorter one is older. Take the fuller one. Anything else --
                # a difference in the middle, which is what a highlighter
                # artefact looks like -- leaves the copy in charge as before.
                lost = self._copy_was_cut_short(rendered or [], copied)
                if lost:
                    self._cut_short_count += 1
                    if self._cut_short_count == 1:
                        print(
                            f"[{self.site.name}] note: tab {self.label}: the copy "
                            f"control gave the answer CUT SHORT — {lost} character(s) "
                            f"fewer than the page shows, and the page's version "
                            f"starts with all of it. Submitting what the page shows. "
                            f"This is what a copy taken while the reply was still "
                            f"streaming looks like, and every one of these is an "
                            f"answer that would have gone out truncated."
                        )
                    else:
                        print(
                            f"[{self.site.name}] tab {self.label}: copy CUT SHORT "
                            f"again, {lost} character(s) missing "
                            f"(#{self._cut_short_count} on this tab)"
                        )
                    copied = list(rendered or [])
                    page_blocks = copied
                elif rendered and not self._warned_diff:
                    difference = self._disagreement(rendered, copied)
                    if difference:
                        self._warned_diff = True
                        print(
                            f"[{self.site.name}] note: tab {self.label}: what the page "
                            f"RENDERS and what it COPIES are not the same — {difference}. "
                            f"Using the copy, which is the source before syntax "
                            f"highlighting. If answers start failing for no visible "
                            f"reason, this line is where to look."
                        )
                best = "\n".join(self._fence(b) for b in copied)
            elif not self._warned_copy:
                self._warned_copy = True
                print(
                    f"[{self.site.name}] note: tab {self.label} could not use the "
                    f"code block's copy control, falling back to reading the DOM. "
                    f"Run `python -m solvers.doctor {self.site.name}` if answers "
                    f"start arriving mangled — the control may have been renamed."
                )
        if self.alive and self.site.stream:
            try:
                best = await asyncio.wait_for(
                    self._reconcile_stream(before, best, page_blocks),
                    timeout=STREAM_PHASE_TIMEOUT_S * tail,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the page's reading
                print(
                    f"[{self.site.name}] note: tab {self.label} could not finish "
                    f"checking the answer against the network stream in "
                    f"{STREAM_PHASE_TIMEOUT_S:.0f}s. Submitting what the page gave."
                )
        # A tab that showed no reply is NOT retired here, and that is the
        # point. Retiring it threw away the conversation the repair loop needs
        # -- and the tab was usually fine: the page rendered late, or not at
        # all, while the answer came off the wire. A tab is dead when the PAGE
        # dies (the read above clears `alive`) or when the prompt cannot be put
        # into it (`send` clears it on a failed submit). Being slow to paint is
        # neither.
        if best and self._is_our_own_prompt(best):
            # Last line of defence, at the ONE exit, because everything above
            # it can produce a submission and only one of them was guarded.
            self._warned_echo = True
            print(
                f"[{self.site.name}] WARN: tab {self.label} was about to submit the "
                f"miner's OWN PROMPT as the answer, and did not. Run `python -m "
                f"solvers.doctor {self.site.name}` — either "
                f"{self.site.env_prefix}_ASSISTANT is matching your own message, or "
                f"this tab is reconstructing the conversation rather than the reply."
            )
            best = ""
        # One exit, one verdict. `best` non-empty clears it: the attribute
        # describes the LAST send, and a stale reason would suppress a repair
        # round on a later one.
        # `self.alive`, not `not blind`: the read also exits with the tab dead
        # when the PAGE died mid-answer, and there `last_busy` can still be True
        # from the last poll that worked. Reporting that as "still writing"
        # sends the caller looking for a slow model instead of a dead tab --
        # the same class of misdirection every other diagnostic here exists to
        # remove. Blind already clears `alive` above, so this covers both.
        self.still_writing = bool(last_busy or grew) and self.alive
        if best:
            self.empty_reason = None
        elif blind or not self.alive:
            self.empty_reason = "unreadable"
        elif self.still_writing:
            self.empty_reason = "unfinished"
        else:
            self.empty_reason = "no-code"
        if not best and (self.alive or blind):
            try:
                await asyncio.wait_for(
                    self._explain_empty(before), timeout=POSTMORTEM_TIMEOUT_S * tail
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a diagnostic, never a failure
                print(
                    f"[{self.site.name}] tab {self.label} captured NOTHING from this "
                    f"reply, and the page did not answer in time to say why."
                )
        return best

    async def _reconcile_stream(
        self, before: tuple[int, Optional[str]], best: str, page_blocks: Optional[list[str]]
    ) -> str:
        """Bring the network stream in as a third reading of the same answer.

        The stream sits above everything else this class reads: it is the
        markdown the model emitted, captured before the page turned it into
        DOM, so none of the damage the other paths have to survive has happened
        to it yet. It is also the only source that still holds the answer when
        the page read comes back empty — a selector that stopped matching, a
        message whose id was swapped mid-stream, a render this tab cannot see.
        Every one of those has cost a whole solve here, and every one of them
        arrives as the same five words: "the reply contained no code".

        It does not simply win, and that is a statement about what can be
        checked rather than about what is likely. Both wire formats are
        private, undocumented and free to change on any deploy, and the
        reconstruction in `_STREAM_READ` is a heuristic over their JSON. A
        heuristic that silently replaced a good answer with a bad one would be
        strictly worse than the bug it was written to fix. So it is used where
        being wrong costs nothing that is not already lost — when the page
        produced nothing at all — and everywhere else it reports, loudly and
        once, whether it agrees. An operator who has watched it agree on their
        own accounts turns `stream_first` on and gets it as the primary.
        """
        streamed = await self._streamed_markdown()
        if streamed is None:
            if not best and not self._warned_stream:
                self._warned_stream = True
                print(
                    f"[{self.site.name}] note: tab {self.label} captured nothing from "
                    f"the network either — this site may not stream its answers over "
                    f"`fetch`. Run `python -m solvers.doctor {self.site.name}` to see "
                    f"what each source returned."
                )
            return best
        blocks = _fenced_blocks(streamed)
        if page_blocks is None:
            try:
                reply = await self._new_reply(before)
                page_blocks = await self._dom_blocks(reply) if reply is not None else []
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - nothing to compare against, then
                page_blocks = []
        if not best:
            # The read loop finished holding nothing. Everything read since is
            # strictly better than the empty string, and the ONLY question left
            # is which of the two later readings to take.
            #
            # The page first, when it has one. `page_blocks` here was fetched
            # after the deadline -- so a non-empty one means the answer landed
            # in the moments between the last poll and now, which is common
            # enough to be the ordinary shape of a near-miss. Requiring the page
            # to be empty before looking anywhere threw that away: measured,
            # `best=""` beside `page_blocks=["def g(n): ..."]` returned "", with
            # the answer sitting in a list in this function's own arguments.
            if page_blocks:
                print(
                    f"[{self.site.name}] tab {self.label} read nothing before its "
                    f"deadline, and found {len(page_blocks)} code block(s) on the "
                    f"page just after it. Submitting those rather than nothing."
                )
                return "\n".join(self._fence(b) for b in page_blocks)
            # Then the wire. This is the case the whole path exists for -- but
            # only when the wire actually holds an answer.
            if blocks:
                print(
                    f"[{self.site.name}] tab {self.label} read NOTHING from the page "
                    f"and recovered {len(blocks)} code block(s) from the network "
                    f"stream instead. The answer below came off the wire, not the DOM."
                )
                return "\n".join(self._fence(b) for b in blocks)
            # Neither. Returning the raw streamed text anyway was worse than
            # returning nothing in three separate ways, all of them measured: it
            # announced a rescue that had not happened, it made `best` non-empty
            # and so SILENCED the post-mortem that would have said what the page
            # contained, and the value was thrown away at extraction regardless.
            # Worst of all it could be the miner's own prompt -- a chat stream
            # carries the conversation, not just the reply -- and two prompts
            # reached a validator as Rust programs that way.
            print(
                f"[{self.site.name}] tab {self.label} read nothing from the page, "
                f"and the network stream had no code block in it either "
                f"({len(streamed)} chars captured). Nothing to submit."
            )
            return ""
        if blocks and page_blocks and not self._warned_stream_diff:
            gap = self._first_difference(page_blocks, blocks, "the page", "the wire")
            if gap:
                self._warned_stream_diff = True
                print(
                    f"[{self.site.name}] note: tab {self.label}: what the page shows and "
                    f"what came off the wire are not the same — {gap}. Using the page. "
                    f"If the wire is the one that is right, set "
                    f"{self.site.env_prefix}_STREAM_FIRST=1."
                )
        if self.site.stream_first and blocks:
            return "\n".join(self._fence(b) for b in blocks)
        return best

    async def _poll(
        self, before: tuple[int, Optional[str]]
    ) -> tuple[Optional[str], bool, str, bool]:
        """One round: ``(code_or_None, still_generating, whole_message, on_screen)``.

        ``on_screen`` is the answer to "has this reply rendered at all", which
        is not the same question as either of the first two and is the only one
        that separates a tab that cannot be read from a model that is thinking.
        ``code_or_None`` is None for both; ``still_generating`` is False for
        both once the page settles. See BLIND_TAB_GRACE_S.

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
            return None, busy, "", False  # ours hasn't rendered yet
        text_now = await self._read(reply)
        whole = await self._whole(reply)
        if text_now is None or self._echoes_prompt(whole or text_now):
            return None, busy, whole, True
        return text_now, busy, whole, True

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

        Latched on first match and reset by ``send()``. Re-resolving *per call*
        would let the fingerprint count nodes matching one candidate and the
        poll count nodes matching another, and ``_new_reply`` compares those two
        numbers directly — so a site with several viable candidates and no
        message id (claude.ai exactly) would mis-detect replies on every repair
        round. Latching per send, not per tab, still lets a later send pick a
        better candidate once the DOM settles.

        The latch is dropped in exactly one case: the candidate it holds matches
        nothing at all. There is no count to corrupt at zero, and the
        alternative is reading nothing for the rest of the send.
        """
        if self._assistant is not None:
            found = self._page.locator(self._assistant)
            if await found.count() > 0:
                return found
            # The winning candidate has stopped matching. Sites stream a
            # message under one attribute and drop it when the message is
            # finished, so the selector that found the answer can be the one
            # that cannot see it any more. Holding the latch here reads NOTHING
            # for the rest of the send while the answer sits on screen under a
            # sibling candidate -- captured nothing, reported as "the reply
            # contained no code". Re-resolving costs one DOM query.
            if not self._warned_relatch:
                self._warned_relatch = True
                print(
                    f"[{self.site.name}] note: tab {self.label}: assistant selector "
                    f"{self._assistant!r} stopped matching mid-answer; re-resolving. "
                    f"Run `python -m solvers.doctor {self.site.name}` — it will show "
                    f"which candidate to pin in {self.site.env_prefix}_ASSISTANT."
                )
            self._assistant = None
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

    def _is_our_own_prompt(self, text: str) -> bool:
        """Would this submission be the miner's own prompt handed back?

        `_echoes_prompt` already asks a version of this, and it is applied in
        exactly one place: the scrape path, inside `_poll`. Every other route
        to a submission -- the copy control, the network stream -- went around
        it, and the stream took that route twice in production. Two validators
        were sent this file's own instruction text, ending in the words "Do not
        use canvas", as a Rust program.

        The test here is containment, not the prefix test `_echoes_prompt`
        uses, and the difference is the point. By this stage the text has been
        through fencing, a copy control or a wire reconstruction, so the prompt
        need not be at the front any more. Containment is safe because what is
        searched for is the head of a prompt this miner generated -- "Solve
        this programming problem in ..." -- which appears in no answer to it.
        """
        head = " ".join(self._sent.split())[:80]
        return bool(head) and head in " ".join(text.split())

    async def _explain_empty(self, before: tuple[int, Optional[str]]) -> None:
        """Say why nothing was captured, while the page is still there to ask.

        "The reply contained no code" is what the grader says afterwards, and it
        describes the symptom of every one of these causes identically: a
        selector that matches nothing, a reply that never rendered, an answer
        still streaming when the budget ran out. The page can distinguish them
        in four DOM queries, and only right now.
        """
        try:
            resolved = await self._first_match(self.site.assistant)
            count = 0
            if resolved is not None:
                count = await self._page.locator(resolved).count()
            reply = await self._new_reply(before)
            busy = await self._busy_now()
        except Exception as exc:  # noqa: BLE001 - the page is gone; say that
            print(
                f"[{self.site.name}] tab {self.label} captured nothing and the page "
                f"could not be inspected ({type(exc).__name__})."
            )
            return
        if resolved is None:
            why = (
                f"no assistant selector matched anything. Tried "
                f"{list(self.site.assistant)}; set {self.site.env_prefix}_ASSISTANT"
            )
        elif count <= before[0]:
            why = (
                f"{resolved!r} matched {count} message(s), the same as before the "
                f"prompt was sent — the answer never rendered"
            )
        elif reply is None:
            why = (
                f"{resolved!r} matched {count} message(s) but none of them could be "
                f"identified as the answer to this prompt"
            )
        elif busy:
            why = "the answer was still being written when the budget ran out"
        else:
            # Distinguish "there is a message and it simply has no code in it"
            # from "the selector matched a container with no text". They look
            # identical from `best == ""` and they need opposite fixes -- the
            # first is the model's doing, the second is ours.
            spoken = " ".join((await self._whole(reply)).split())
            if spoken:
                why = (
                    f"the message has no code block in it. It says: "
                    f"{spoken[:160]!r}{'...' if len(spoken) > 160 else ''}"
                )
            else:
                why = (
                    f"the answer rendered but read as empty — "
                    f"{self.site.env_prefix}_ASSISTANT may be matching a wrapper "
                    f"rather than the message"
                )
        print(
            f"[{self.site.name}] tab {self.label} captured NOTHING from this reply: "
            f"{why}. This is what surfaces later as \"the reply contained no code\"."
        )

    async def _copied_blocks(self, reply) -> Optional[list[str]]:
        """The code as the PAGE would copy it, without touching the clipboard.

        This is the preferred extractor, and the reason is not tidiness. What
        the copy control hands over is the source the model wrote; what
        `pre code` hands over is the source AFTER a syntax highlighter has
        rebuilt it as DOM. Those differ, and they have differed here in
        production: a highlighter put U+E027 -- a Private Use Area character
        that exists in no source file -- inside a Python answer, and the solve
        died on a character no human could see. Re-fencing, chip stripping and
        invisible-character scrubbing are all repairs for damage that this path
        never takes, because it reads from before the render.

        All-or-nothing on purpose. A reply with two blocks has two controls; if
        only one answers, taking that one would silently drop a block and could
        drop the answer. Returning None instead hands the whole read back to
        scraping, which at least sees every block.
        """
        if not self.site.copy:
            return None
        buttons = None
        for selector in self.site.copy:
            found = reply.locator(selector)
            if await found.count() > 0:
                buttons = found
                break
        if buttons is None:
            return None
        await self._page.evaluate(_COPY_HOOK)
        fenced: list[str] = []
        expected = await buttons.count()
        for i in range(expected):
            button = buttons.nth(i)
            # Check the control's own name before pressing it. A selector is a
            # guess about structure and can drift onto a neighbour; ChatGPT puts
            # "Run code" in the very same header as "Copy". Reading the code is
            # worth a click, executing it is not, so nothing gets pressed unless
            # it says what it is. A UI in another language degrades to scraping
            # until `<PREFIX>_COPY_NAME` is set, which is the safe direction.
            try:
                name = await button.get_attribute("aria-label") or await button.inner_text()
            except Exception:  # noqa: BLE001 - unreadable name is not a name
                name = None
            if self.site.copy_name not in (name or "").casefold():
                return None
            await self._page.evaluate("window.__honeCopied = null")
            try:
                # force: the control is often transparent until hover, and
                # actionability would wait for a hover that never comes.
                await button.click(timeout=COPY_TIMEOUT_MS, force=True)
                await self._page.wait_for_function(
                    "window.__honeCopied !== null", timeout=COPY_TIMEOUT_MS
                )
                block = await self._page.evaluate("window.__honeCopied")
            except Exception:  # noqa: BLE001 - one uncooperative button, not a dead tab
                continue
            if not block or not block.strip():
                continue
            fenced.append(block)
        if len(fenced) != expected or not fenced:
            return None  # see the docstring: all of them, or none of them
        return fenced

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

    async def _stream_seq(self) -> int:
        """How many streamed responses this tab has seen so far.

        Taken before the prompt goes out and used as a floor afterwards, so the
        answer is never reconstructed out of the previous turn's stream -- the
        buffer holds several records, and a repair round would otherwise be at
        risk of re-submitting the reply it was sent to repair.
        """
        try:
            return int(await self._page.evaluate("window.__honeStreamSeq || 0") or 0)
        except Exception:  # noqa: BLE001 - no hook, or a page mid-navigation
            return 0

    async def _streamed_markdown(self) -> Optional[str]:
        """The answer as it came off the wire, or None if nothing was captured."""
        if not self.site.stream:
            return None
        try:
            text = await asyncio.wait_for(
                self._page.evaluate(_STREAM_READ, self._stream_before),
                timeout=STREAM_TIMEOUT_MS / 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a reading, not the reading; never fatal
            return None
        text = (text or "").strip()
        return text or None

    async def _copied_code(self, reply) -> Optional[str]:
        """`_copied_blocks`, fenced. Convenience for callers that want text."""
        blocks = await self._copied_blocks(reply)
        return "\n".join(self._fence(b) for b in blocks) if blocks else None

    # How much shorter the copy may be before it is treated as cut short
    # rather than as an artefact. A truncated program loses a line at the very
    # least; nothing that loses four non-whitespace characters or fewer is a
    # truncation, and down there the copy's fidelity advantage is worth more
    # than the render's extra character -- a highlighter that appends one is
    # precisely the thing the copy control exists to beat.
    _CUT_SHORT_SLACK = 4

    @staticmethod
    def _copy_was_cut_short(rendered: list[str], copied: list[str]) -> int:
        """Characters the copy is missing, when it is the same answer cut short.

        Zero unless the copied source is a strict PREFIX of the rendered source
        with whitespace ignored, which is what distinguishes the two ways these
        readings disagree. A copy taken while the reply was still streaming is
        the beginning of the answer and nothing else. A highlighter artefact --
        the measured one was a Private Use Area character inside a Python
        program -- is a difference in the MIDDLE, so the prefix test fails and
        the copy keeps its authority, which is the whole reason it is preferred.

        Whitespace is ignored on both sides so that indentation rebuilt by the
        renderer, or the newline a copy control trims, is never mistaken for
        lost code.
        """
        if not rendered or not copied:
            return 0
        whole = "".join("".join(b.split()) for b in rendered)
        part = "".join("".join(b.split()) for b in copied)
        if len(part) >= len(whole) or not whole.startswith(part):
            return 0
        missing = len(whole) - len(part)
        return missing if missing > _Tab._CUT_SHORT_SLACK else 0

    def _disagreement(self, dom: list[str], copied: list[str]) -> Optional[str]:
        """How the rendered code differs from the copied code, in one line.

        The point is not to choose — the copy already wins, because it is the
        source before the highlighter touched it. The point is to SAY SO. Every
        extraction bug this miner has had was silent: a Private Use Area
        character, a leaked language chip, a blank line inserted at a render
        boundary. Each one looked exactly like the model writing bad code, and
        each cost days to find. Two independent readings of the same answer are
        already in hand here, so disagreement costs nothing to detect and turns
        a mystery into a log line.
        """
        if len(dom) != len(copied):
            return (
                f"the page renders {len(dom)} code block(s) but its copy "
                f"controls give {len(copied)}"
            )
        return self._first_difference(dom, copied, "rendered", "copied")

    @staticmethod
    def _first_difference(
        first: list[str], second: list[str], left: str, right: str
    ) -> Optional[str]:
        """Where two readings of the same answer part company, in one line.

        Blank lines at the very start and end are not a difference. They are an
        artefact of where each reading was taken -- `textContent` on a `<code>`
        keeps the newline before the closing tag, a copy control usually trims
        it, a fenced block always ends in one -- and reporting them would fire
        the warning on every clean answer, which is how a warning stops being
        read before the one that matters arrives.
        """
        if len(first) != len(second):
            return f"{left} has {len(first)} code block(s), {right} has {len(second)}"
        for raw_a, raw_b in zip(first, second):
            a, b = raw_a.strip("\n"), raw_b.strip("\n")
            if a == b:
                continue
            at = next(
                (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)),
            )
            return (
                f"they differ at character {at}: {left} {_describe_char(a[at:at + 1])}, "
                f"{right} {_describe_char(b[at:at + 1])}"
            )
        return None

    @staticmethod
    def _fence(block: str) -> str:
        """Wrap one block. Markdown's own rule: the fence must outrun any
        backtick run inside, or a block containing ``` closes itself early."""
        longest = max((len(r) for r in re.findall(r"`+", block)), default=0)
        return "`" * max(3, longest + 1) + f"\n{block}\n" + "`" * max(3, longest + 1)

    @staticmethod
    async def _read(reply) -> Optional[str]:
        """The reply's code blocks, re-fenced, or None. Never the prose.

        Every block, not a guess at which one matters. Taking only the last
        used to submit the "example usage" snippet whenever a model appended
        one -- the solution was right there in the block before it, and the
        whole solve was spent to report that `solve` was never defined.
        Choosing between them needs the entrypoint, which belongs to the task
        rather than the tab, so hand them all over and let the caller pick.

        Returning None rather than the message text is the important half, and
        it is a statement about what this miner is for: it only ever wants a
        code block, so prose is never an answer and handing it back has no
        upside at all. It has a large downside. claude.ai renders extended
        thinking INSIDE the element the assistant selector matches, and that
        thinking arrives long before any code does -- so for the whole first
        stretch of an answer, the message on screen is reasoning and nothing
        else. Falling back to `inner_text` turned that into "here is your
        program": measured on a real solve, 13,200 characters of the model
        working through the problem were submitted as Rust, the grader replied
        "the program does not define `fn main()`", and the repair round told
        the model to fix a program it had never sent. Twice.

        Reading nothing is the honest answer to "has it written the code yet",
        and it is what the ChatGPT tabs have always done -- ChatGPT keeps its
        reasoning outside the matched element, so the same situation there read
        as empty and was reported as empty. This makes the two sites agree.
        What the message actually says is not lost: `_explain_empty` quotes it
        when a send comes back with nothing.
        """
        blocks = await _Tab._dom_blocks(reply)
        return "\n".join(_Tab._fence(b) for b in blocks) if blocks else None

    @staticmethod
    async def _dom_blocks(reply) -> list[str]:
        """Every rendered code block, raw. The second opinion on what was said."""
        blocks: list[str] = []
        code_blocks = reply.locator("pre code")
        for i in range(await code_blocks.count()):
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
            blocks.append(block)
        return blocks

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
        # Replacement spawns in flight. Tracked, not fired and forgotten: each
        # one opens a page in YOUR browser, and one that lands after shutdown
        # has swept `_tabs` is a signed-in tab nothing will ever close.
        self._pending: set[Any] = set()
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
            if site.stream:
                # Before the goto, and as an init script rather than an
                # evaluate: it has to be in place before the site's own bundle
                # installs its fetch wrappers, and it has to survive every
                # navigation this tab makes for the weeks it stays open.
                await page.add_init_script(_STREAM_INSTALL)
            await page.goto(site.url, wait_until="domcontentloaded")
            # Stamp it before anything can go wrong later, so even a tab that
            # fails its checks below is reclaimable after an unclean exit.
            await page.evaluate(f"window.name = {TAB_MARK + '/' + site.name!r}")
            composer = await wait_for_any(page, site.composer, site.ready_timeout_ms)
        except asyncio.CancelledError:
            # Shutdown, almost always. Close the page on the way out: this
            # coroutine is the only thing that knows it exists yet -- it is not
            # in `_tabs` until the very end -- so unwinding without it leaves a
            # tab open in the operator's browser for good.
            await _close_quietly(page)
            raise
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
                # start() already marked it dead; release() disposes it and
                # rebuilds the capacity in the background, so the second pass
                # takes whichever tab reaches `_free` first -- another idle one
                # straight away, or the replacement when it finishes loading --
                # rather than failing the task while a good tab sits idle.
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
        # Rebuilt off to the side, NOT awaited here. Building a tab means a new
        # page, a navigation, and a wait for the composer -- `ready_timeout_ms`
        # alone is 60 SECONDS. This runs from the solver's `finally`, after the
        # answer is already in hand and after the budget is spent, and the
        # deadline above it is an `asyncio.wait_for` in `handle_request` that
        # answers 504 rather than late. Awaited, the replacement would throw
        # away the very answer whose failure asked for it -- and it is precisely
        # the failing solves, the ones with the least budget left, that get
        # here. Nothing waits on the new tab except the next lease, which waits
        # on `_free` anyway and has its own budget to do it in.
        self._replace_later(tab)

    def _replace_later(self, tab: _Tab) -> None:
        """Rebuild one tab's worth of capacity, out of everyone's way."""
        if self._closing:
            return
        task = asyncio.ensure_future(self._replace(tab))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _replace(self, tab: _Tab) -> None:
        browser = Browser(tab.source, tab.site)
        replacement = await self._spawn(tab.context, browser, tab.label)
        if replacement is None:
            print(
                f"[{tab.site.name}] WARN: tab {tab.label} retired and could not be "
                f"replaced; {self._size} tab(s) left"
            )
            return
        if self._closing:
            # Shutdown started while this was loading. `_teardown` has already
            # swept `_tabs`, so nothing else will ever close this page.
            self._retire(replacement)
            await replacement.dispose()
            return
        await self._free.put(replacement)
        self._size += 1  # _retire took one off; this puts the capacity back
        print(f"[{tab.site.name}] tab {tab.label} replaced after failure")

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
        # Stop the background replacements BEFORE the sweep below. One may be
        # mid-navigation right now, and a tab that lands after `_tabs.clear()`
        # is a page this fleet opened, still signed in, that nothing will close.
        for task in list(self._pending):
            task.cancel()
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
            self._pending.clear()
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

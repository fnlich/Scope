"""Subnet-specific prompt construction and code extraction.

The homework automation's prompt ("solve this homework problem in Python")
is not sufficient here: the validator runs the returned source against a
HIDDEN suite with an exact contract, so the prompt has to state that contract
precisely or a perfectly reasonable answer scores zero.

What the validator actually does with the code:

* Python — imports the source and calls ``entrypoint(*args, **kwargs)`` for
  each hidden case, comparing the RETURN VALUE structurally
  (``rlvr.execution.compare.values_equal``). Anything printed is ignored;
  anything read from stdin hangs. So the answer must be a pure function.
* Rust — compiles the source as one file with ``rustc --edition=2021`` and
  runs it once per case, feeding the case to STDIN and comparing STDOUT
  token-by-token on ASCII whitespace (``rlvr.execution.rust_judge``). So the
  answer must be a complete program with ``fn main()``.

The public examples are rendered into the prompt because the statement alone
is frequently ambiguous about ordering, tie-breaking and output shape, and
those examples are the only disambiguation a miner is given. All 97 recorded
production tasks ship none, so on live traffic that paragraph never renders
and the statement is the whole of the specification.

FIVE PROMPTS, one per stage of a solve, and every one of them is built the
same way -- statement header, worked examples where there are any, the trap
block, the one task this stage is for, then the output contract LAST:

    build_analysis_prompt   stage 3: add to the free trap scan
    build_inputs_prompt     stage 4: inputs only, never their answers
    build_oracle_prompt     stage 5: a reference, correctness-first
    build_candidate_prompt  stage 6: the program that ships, complexity-first
    build_differential_repair_prompt
                            stage 8: whichever of the two is wrong

Stages 5 and 6 are deliberately opposed -- that difference is the whole of the
evidence the comparison produces; see ``solvers/differential.py``. Stage 4 is
forbidden to supply an expected value, and the extractor drops any that arrive
anyway: the reference program computes every expectation by being RUN, which
is what stops the bar and the program sharing one misreading of the statement.

The extraction half of this file is older than the stage design and outlives
it: ``fenced_blocks``, ``sanitize_code``, ``extract_code`` and the JSON
salvage below were each written against a way a real reply was thrown away.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from typing import Any, Optional

# ChatGPT wraps code in ``` fences; the DOM reader already returns the inner
# text of a <pre><code> block, but a reply that arrived as plain text (or a
# non-DOM backend) can still carry fences, so strip them defensively.
# Markdown fences are 3 OR MORE backticks, and the closer must be at least as
# long as the opener. Hard-coding three cut a block short the moment its own
# source contained ``` -- a docstring showing markdown was enough -- and missed
# a longer fence entirely. The backreference makes the closer match the opener.
def fenced_blocks(markdown: str) -> list[str]:
    """Every fenced block in a markdown string, in order, without its fences.

    Scanned line by line rather than matched with one regular expression, and
    that is not a style preference -- the regex this replaced lost a whole
    answer in four separate ways, each measured against the shape that produces
    it:

    * A closing fence has to be a whole LINE. The regex matched its backticks
      anywhere, so `fence = "```"` inside a program ENDED the block, and the
      answer was truncated at that line.
    * A block written with four backticks because it contains three is the
      markdown rule for exactly that case. The regex's trailing `` `* `` ate
      the wrong run and left a stray fence inside the code.
    * A reply cut off mid-block has no closing fence at all. The regex matched
      nothing, so the extractor fell through to its "this reply is all prose"
      path and returned NOTHING -- discarding a program that was fully written,
      on the one failure a deadline causes most often.
    * `~~~` is a fence too, and CommonMark says so.

    An unclosed final fence is therefore kept: a reply cut off by a deadline
    still has its program in it, and dropping it turns a partial answer into no
    answer at all.
    """
    blocks: list[str] = []
    body: Optional[list[str]] = None
    fence = ""
    # How far the OPENING fence was indented. Markdown requires that
    # indentation of every line inside a block nested under a list item, and it
    # is not part of the source:
    #
    #     1. Sort, then sum:
    #
    #        ```python
    #        import math
    #
    #        def solve(nums):
    #            return sum(sorted(nums)[-2:])
    #        ```
    #
    # Keeping it handed `extract_code` a block whose every line began with three
    # spaces; `.strip()` then removed them from the FIRST line only, and the
    # result was `unexpected indent, line 3` on a program the model had written
    # correctly. CommonMark strips exactly this, and never more than the fence
    # itself had -- so a line the author genuinely indented further keeps the
    # difference.
    indent = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if body is None:
            # The fence may END a line of prose rather than start one. Markdown
            # says a fence opens a line, and a model that writes
            # `Here you go: ```python` has broken that rule -- but it has still
            # answered, and the reader that came before this one accepted it.
            # Requiring the line to START with the fence dropped that answer
            # entirely: no block found, so the extractor fell through to its
            # all-prose path and returned "". Caught by the suite.
            #
            # A fence that STARTS its line is markdown's own rule and takes
            # the info string markdown allows. The mid-line tolerance is
            # deliberately narrower: the fence has to be the last thing on the
            # line with its language word attached, so `Use ```code``` inline`
            # is not an opener and cannot swallow the paragraph beneath it.
            # Allowing a SPACE before that word was enough to break exactly
            # that -- ``` inline.` read as a fence with the info string
            # "inline.".
            opener = re.match(r"(`{3,}|~{3,})", stripped) or re.search(
                r"(`{3,}|~{3,})[A-Za-z0-9_+#.-]*$", line.rstrip()
            )
            if opener:
                fence = opener.group(1)
                indent = len(line) - len(line.lstrip(" "))
                body = []
            continue
        if re.fullmatch(re.escape(fence[0]) + "{%d,}" % len(fence), stripped):
            if "\n".join(body).strip():
                blocks.append("\n".join(body) + "\n")
            body, fence, indent = None, "", 0
            continue
        body.append(_unindent(line, indent))
    if body is not None and "\n".join(body).strip():
        blocks.append("\n".join(body) + "\n")
    return blocks


def _unindent(line: str, indent: int) -> str:
    """Drop up to ``indent`` leading SPACES -- never more, never a tab.

    Never more, because a line the author indented past the fence keeps the
    difference. Never a tab, because a tab cannot be partially removed and
    guessing its width would corrupt source that a chat UI does render with
    them; a block opened at column zero, which is every unnested block, is
    returned untouched either way.
    """
    if indent <= 0:
        return line
    kept = 0
    while kept < indent and kept < len(line) and line[kept] == " ":
        kept += 1
    return line[kept:]


# Characters that only ever arrive from a RENDERED page, never from source a
# grader would accept: zero-width marks, line/paragraph separators, the BOM,
# and the Private Use Area, which chat UIs use for syntax-highlight and cursor
# bookkeeping. One of these is enough to make the whole file a SyntaxError —
# `invalid non-printable character U+E027` — after the model wrote a perfectly
# good answer, so they are stripped rather than reported.
_INVISIBLE_RE = re.compile(
    "[\u200b-\u200f\u2028\u2029\u2060\ufeff\ue000-\uf8ff]"
    "|[\U000f0000-\U000ffffd]|[\U00100000-\U0010fffd]"
)
# Exotic spaces render like a space and break indentation. Fold them.
_ODD_SPACE_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

# Some models narrate in a <think> block before answering, and the narration
# quotes code -- half a struct, a function it then discards. When that block
# arrives as TEXT rather than as its own collapsed UI element, every fragment in
# it looks exactly like a candidate answer to the fence scanner, and one of them
# is the last block whenever the real answer has not arrived. So the reasoning
# is removed before anything is matched. An unclosed opener takes the rest of
# the reply with it: there is no answer after a `<think>` that never ended, and
# "nothing arrived" is a far better thing to report than a fragment of the
# model's rough work.
_THINK_RE = re.compile(
    r"<(think|thinking|reasoning|scratchpad)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(
    r"<(?:think|thinking|reasoning|scratchpad)\b[^>]*>.*",
    re.DOTALL | re.IGNORECASE,
)

# A rendered code block puts its language chip inside the element the reader
# scrapes, so the inner text can begin with a bare "python" line. That is worse
# than a syntax error: it PARSES, defines the entrypoint, passes every check —
# and then raises NameError the moment the grader imports it, failing every
# hidden test with nothing anywhere saying why.
_LANG_LABEL_RE = re.compile(
    r"^[ \t]*(?:python|python3|py|rust|rs|javascript|js|typescript|ts|json"
    r"|bash|sh|shell|text|plaintext|plain|code|output)[ \t]*\r?\n",
    re.IGNORECASE,
)


def sanitize_code(text: str) -> str:
    """Undo what rendering did to the source, without touching the source."""
    if not text:
        return ""
    text = _INVISIBLE_RE.sub("", text)
    text = _ODD_SPACE_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Only ever one chip, and only at the very front.
    return _LANG_LABEL_RE.sub("", text, count=1)


PYTHON_RULES = """\
- There is no partial credit: a program wrong on ONE hidden case scores
  exactly what no answer scores. Correctness is the whole of it.
- Define exactly one top-level function named `{entrypoint}`. It is called
  directly as `{entrypoint}(*args, **kwargs)`.
- RETURN the answer. Do not print it, do not read stdin, do not call input().
  Printed output is ignored by the grader.
- Standard library only. No pip packages, no network, no file access.
- Put no tests, example calls or `if __name__ == "__main__"` INSIDE the
  program block. Nothing but the function and whatever it needs to run."""

RUST_RULES = """\
- There is no partial credit: a program wrong on ONE hidden case scores
  exactly what no answer scores. Correctness is the whole of it.
- Write ONE complete program with `fn main()`, compiled as a single file with
  `rustc --edition=2021 -C opt-level=2`. No Cargo, no crates, std only.
- READ the input from stdin and WRITE only the requested answer to stdout.
- Output is compared token-by-token after splitting on ASCII whitespace, so
  extra prose, labels or prompts make the answer wrong."""

# The two output contracts, and why they share an opening.
#
# `_Tab._is_our_own_prompt` (browser_pool.py) recognises a stale scrape by
# testing whether the message begins with the first 80 normalised characters of
# whatever was last sent. Give the two turns DIFFERENT openings and a scrape
# that returns turn 1's text is no longer recognised after turn 2 -- it is
# reported as a defect instead, and a repair round is spent on it. So the first
# sentence is identical in both, and only the line naming the block differs.
_ONE_BLOCK = """\
Reply with ONE fenced block written directly in the chat — not into an artifact
or canvas — and nothing else. No preamble, no explanation before it or after it.
Only what is inside the fence is ever read.

"""


# ...and the same contract when a SIZE PROBE is wanted as well. Two blocks, in
# a fixed order, because the second one is not a test case and must not be
# parsed as one.
#
# Why the probe cannot be a case. Every case on the bar carries an `expected`
# the model derives BY HAND from the statement, which is what makes the bar
# evidence rather than an echo of the program. That rule puts a hard ceiling on
# case SIZE: nobody hand-computes the answer for n = 200000. So the bar is,
# structurally and permanently, a suite of small inputs -- and a program that
# is right on every small input can still be quadratic, and the validator runs
# it at the statement's real limits with `per_test_timeout_s` of five seconds.
#
# A timeout needs no expected value. "Did it finish" is answerable with no
# oracle at all, which is what makes this affordable: one more fenced block on
# a turn that is already paid for, and no judgement about what the answer
# should be.
TESTS_OUTPUT_CONTRACT_WITH_PROBE = """\
Send TWO fenced blocks and nothing else — no preamble, nothing between or
after them. Only what is inside the fences is ever read.

The FIRST block is `json` and holds the test cases.
The SECOND block is `python` and holds one function, `generate`.

Do NOT write the program yet — you will be asked for it next."""

CODE_OUTPUT_CONTRACT = _ONE_BLOCK + """\
That block is the {language} program. Nothing else is graded."""

# What "the corrected program" has to spell out, in every repair round.
#
# A model shown one failing case answers about that case: it sends back the one
# function it changed, correct in itself and unrunnable on its own, because the
# imports and the helpers are in the reply above it. Each round is graded as a
# WHOLE FILE and submitted as one -- there is no conversation on the other side
# to reassemble it -- so a reply that is a diff in spirit is a zero in fact, and
# `compile()` will not say so: a lone corrected function parses perfectly.
#
# `_carry_imports` rescues the import half of this, and nothing rescues the rest.
WHOLE_PROGRAM = (
    "the corrected program, COMPLETE — every import, helper and definition it "
    "needs to run on its own, not a diff and not only the part you changed"
)


# What the size probe asks for, appended to the cases task when one is wanted.
#
# `scale` is a BYTE BUDGET, not a percentage, and that is the whole lesson of
# the arm this replaces. That one asked for "1 to 100, where 100 means the
# largest input every limit in the statement allows" and then shipped the
# result back through the sandbox's framed status line -- which caps captured
# stdout at 256 KiB and keeps only the TAIL, so a reply larger than that lost
# its opening frame marker and came back as "sandbox produced no verdict
# (crashed or exited early)". Measured over 102 production solves: 8 large
# inputs obtained, 26 reported as a crashed generator. The generators were
# almost certainly fine; the transport ate them, and the size check that would
# have caught it ran AFTER the run and was set at 1000000 bytes -- 3.8 times
# the ceiling it was meant to respect.
#
# Naming a byte budget puts the one number that matters where the model can
# respect it, and `PROBE_MAX_BYTES` in `verify.py` keeps the ladder honest
# regardless.
GENERATOR_TASK = """\
Then, in a second fenced `python` block, write `generate(seed, scale)`.

It RETURNS one valid test input for this problem — the input only, never the
answer. Call `random.seed(seed)` first. `scale` is a BYTE BUDGET: return the
LARGEST input this problem allows whose serialised size stays under `scale`
bytes, subject to every limit the statement states. If the statement's own
limits are smaller than `scale`, return the largest input those limits allow.

{shape}

This exists to time the program, not to check its answer, so nothing here
needs an expected value. Make it as large as the budget permits: a program
that is too slow is only visibly too slow at size. Use only the standard
library, define everything in the same block, and print nothing."""

_PROBE_SHAPE_PYTHON = (
    'Return a dict `{{"args": [...], "kwargs": {{}}}}` giving the arguments for '
    "`{entrypoint}(*args, **kwargs)`. Every value must be JSON-serialisable: no "
    "tuples, no sets, no `inf`, no `NaN`."
)
_PROBE_SHAPE_RUST = (
    'Return a dict `{{"args": ["<the complete stdin>"]}}` — one string, the '
    "entire standard input the program is to be run on, exactly as it would "
    "arrive, newlines included."
)


def extract_generator(reply: str) -> str:
    """The fenced block defining `generate`, or ''.

    Returns '' for everything unexpected. The probe is an optional extra on a
    turn whose real product is the bar: a reply that sends no generator, or one
    this cannot find, must cost the solve nothing at all.
    """
    if not reply:
        return ""
    for block in fenced_blocks(sanitize_code(reply)):
        if re.search(r"^\s*def\s+generate\s*\(", block, re.M):
            return block
    return ""


# Two of these are facts about THIS grader, not general advice, and both cost a
# solve when guessed at: the comparison is structural and strict about bools,
# and each test is on a five-second clock.
PYTHON_ENVIRONMENT = """\
- Python integers never overflow — but when the statement says an operation
  exceeding a named MAX must be rejected, that is a rule for you to implement,
  not an error Python will raise for you. The default recursion limit is 1000, so a
  recursive answer dies at n = 10^4 with RecursionError. Write it iteratively,
  or raise the limit yourself at the top of the file.
- Each test gets about 5 seconds. O(n^2) over n = 10^5 does not fit.
- Iterating a `set` or `dict` of STRINGS gives a different order in every
  process — `PYTHONHASHSEED` is random by default, and a solution tested with
  small ints looks stable and is not. Sort before returning anything
  order-sensitive.
- Return the exact shape the examples show. The comparison is structural:
  `True` is not `1`, so a boolean answer must be a real bool; a dict must have
  exactly the expected keys; two integers must match exactly. (A list and a
  tuple with equal contents do compare equal, so that one is safe.)"""

# The overflow line is the single most valuable sentence in this file, and it is
# measured rather than assumed -- see the test that compiles it. `rustc`
# switches overflow checks off whenever opt-level > 0, and the grader compiles
# at opt-level=2, so the arithmetic wraps and the program exits 0 with a
# plausible wrong number. There is no panic, no message, and nothing in the
# failure that points at the cause.
RUST_ENVIRONMENT = """\
- INTEGER OVERFLOW IS SILENT HERE. The grader compiles with `-C opt-level=2`,
  which turns overflow checks OFF: `i32` arithmetic wraps around and the
  program exits normally with a wrong answer instead of panicking. Two `i32`
  values of 2_000_000_000 add up to -294967296. Use `i64` everywhere by
  default, `i128` for products, and reach for `i32` only where you have proved
  the range cannot be exceeded. A running total overflows before any single
  term does — accumulate sums of products in `i128`, and reduce modular
  quantities at every step, never only at the end.
- Read ALL of stdin and parse it as one stream of whitespace-separated tokens,
  never line by line: counts may be zero, records may cross line boundaries,
  and trailing newlines, blank lines and repeated spaces are all legal.
- Deep recursion overflows the stack. Prefer iteration for n up to 10^5.
- `HashMap` and `HashSet` iteration order is unspecified and differs run to run.
  Use `BTreeMap`/`BTreeSet`, or sort, before emitting anything order-sensitive.
- Each test gets about 5 seconds, so lock stdout once and wrap it in a
  BufWriter rather than printing in a loop."""


# What the worked examples ARE, said where they are read rather than in a
# checklist further down. Two claims, and they pull in opposite directions on
# purpose: they are ground truth, so they settle the ordering and tie-breaking
# a statement leaves open -- and they are a FLOOR, because a program written to
# satisfy the two examples shown and nothing else is the commonest way to pass
# every check this miner has and score zero on the hidden suite.
#
# Live traffic ships none of these (all 97 archived requests carry an empty
# list), so this text is read by `rehearse --examples N` and by any task that
# starts shipping them -- not by production today.
EXAMPLES_LABEL = (
    "WORKED EXAMPLES — already known to be right, and a floor, not the "
    "specification.\nWhere the statement is ambiguous they decide; where it is "
    "not, the hidden tests\ngo far beyond them."
)


def _render_examples(language: str, examples: list[dict[str, Any]]) -> str:
    """Render public examples in the shape the grader will actually use."""
    if not examples:
        return ""
    if language == "rust":
        blocks = []
        for i, case in enumerate(examples, 1):
            args = case.get("args") or [""]
            stdin = args[0] if args else ""
            blocks.append(
                f"Example {i}:\nSTDIN:\n{stdin}\nEXPECTED STDOUT:\n{case.get('expected', '')}"
            )
        return "\n\n".join(blocks)
    lines = []
    for i, case in enumerate(examples, 1):
        args = json.dumps(case.get("args", []), ensure_ascii=False)
        kwargs = json.dumps(case.get("kwargs", {}), ensure_ascii=False)
        expected = json.dumps(case.get("expected"), ensure_ascii=False)
        lines.append(f"Example {i}: args={args} kwargs={kwargs} -> returns {expected}")
    return "\n".join(lines)


# The most inputs one reply may contribute. Each one is now TWO executor runs
# against the solve's own budget -- the reference produces the expectation and
# the candidate is graded on it -- a subprocess apiece for Python and a
# container apiece for Rust. A model that emits forty would spend the deadline
# measuring instead of getting an answer submitted, and the prompt asks for 8
# to 20, so this is the ceiling it was already told about.
MAX_INPUTS = 20


def _thin(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """At most ``limit`` cases, keeping the SHAPE of the coverage.

    A head-slice was what this used to do, and the turn-1 prompt inverts the
    assumption that made it safe. Cases now arrive easiest-first by explicit
    instruction, so `cases[:limit]` keeps the cheap ones and throws away the
    boundaries -- discarding exactly the cases the whole mechanism exists to
    run, and doing it silently.

    The first three are kept because that is where the prompt's first three
    classes sit -- the ordinary case, the empty value, and one -- and the rest
    are sampled at an even stride so no class is dropped wholesale whatever
    order the model actually used. Three stays right now that only ONE of them
    is the ordinary case: the head is the common path plus the two degenerate
    inputs every implementation has to survive, which is the smallest set worth
    protecting from the stride.
    """
    if len(cases) <= limit:
        return list(cases)
    head = cases[: min(3, limit)]
    rest, room = cases[len(head):], limit - len(head)
    if room <= 0:
        return head
    stride = len(rest) / float(room)
    return head + [rest[int(i * stride)] for i in range(room)]


def _scrub_json(text: str) -> str:
    r"""Strip line comments and trailing commas, leaving strings untouched.

    Both are things a model writes and JSON forbids, and both are cheap to undo
    -- but only with a scanner that knows where the strings are. `re.sub` on
    `//` corrupts an expected value of `"http://x"`, and on `,\s*]` corrupts
    `"a,]"`. Those are exactly the values a test case is made of, so a
    string-blind scrubber trades one silent drop for a silent corruption.
    """
    out: list[str] = []
    quote = ""          # the character that opened the string we are inside
    escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        # BOTH quote characters, because by the time this runs the text may be
        # Python rather than JSON -- `ast.literal_eval` is one of the parsers
        # downstream, and a model reasoning in Python single-quotes its strings.
        # Tracking only `"` ate the `#` out of `'a#b'` and the `//` out of
        # `'http://x'`, turning a silent drop into a silent corruption, which is
        # worse: the cases still run, against values nobody wrote.
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        # A line comment, in either of the two spellings a model reaches for.
        if text.startswith("//", i) or ch == "#":
            end = text.find("\n", i)
            i = n if end == -1 else end
            continue
        if ch == ",":
            # Trailing comma: the next non-space character closes a container.
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "]}":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _loads_cases(text: str) -> Any:
    """`json.loads`, then the two dialects a model actually writes. Never raises.

    Strict JSON first, because it is the common case and the cheapest. What
    follows is not permissiveness for its own sake -- each fallback was a
    measured way a real correction was thrown away:

    * comments and trailing commas -- JSON forbids both, models write both;
    * `True` / `False` / `None` and single-quoted strings -- a model reasoning
      in Python writes Python, and `ast.literal_eval` reads exactly that
      dialect. It evaluates literals only: no names, no calls, no attribute
      access, so a hostile array cannot execute anything.
    """
    for attempt in (text, _scrub_json(text)):
        try:
            return json.loads(attempt)
        except Exception:  # noqa: BLE001 - a model wrote it; anything is possible
            pass
    for attempt in (text, _scrub_json(text)):
        try:
            return ast.literal_eval(attempt)
        except Exception:  # noqa: BLE001 - not a literal either, then
            pass
    return None


# How many bracketed spans to try before giving up. A reply is prose, not a
# haystack: a handful covers "the model quoted a list or two while explaining
# itself", and the cap is what stops a pathological block turning the search
# into the solve's own budget.
_MAX_SPAN_TRIES = 8


def _array_spans(text: str) -> list[str]:
    """Every bracketed span in `text`, outermost-first, up to `_MAX_SPAN_TRIES`.

    Taking only the FIRST `[` was wrong, and wrong on the commonest shape there
    is: the most natural way a model corrects one of its own cases is to quote
    that case's INPUT, and for this miner an input is a list literal.

        You are right, for the input [3, 1, 2] the sum is 6, not 5.
        Here are ALL of the cases, corrected:
        [{"name": "ordinary", ...}, ...]

    The first span is `[3, 1, 2]`, which holds no case dicts, and the search
    stopped there -- so the corrected suite below it scored zero, which is
    exactly the failure this was written to fix. `Case [2]`, `- [x] fixed` and
    `nums[0]` all do the same.
    """
    spans: list[str] = []
    at = 0
    while len(spans) < _MAX_SPAN_TRIES:
        span = _array_span(text, at)
        if span is None:
            break
        spans.append(span)
        at = text.find(span, at) + 1
    return spans


def _array_span(text: str, start_at: int = 0) -> Optional[str]:
    """The first bracketed array in `text`, brackets included, or None.

    A model answers "send back ALL of the cases" with prose around the array
    about as often as with the array alone, and a reply read off the DOM rather
    than off the copy control arrives with its fence already gone. Both put the
    array somewhere other than at character zero.

    Bracket-matched rather than regex-matched, and string-aware for the same
    reason `_scrub_json` is: an expected value of `"]"` ends the array under any
    cheaper rule.
    """
    start = text.find("[", start_at)
    if start == -1:
        return None
    depth = 0
    closed = -1
    quote = ""
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "\"'":       # both, for the same reason `_scrub_json` does
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
            if depth == 1:
                closed = i          # a complete element ended here
    return _truncated_span(text, start, closed)


def _truncated_span(text: str, start: int, closed: int) -> Optional[str]:
    """An unterminated array, cut back to its last complete element.

    A twenty-case array is long, and the ways a reply gets cut short are
    ordinary rather than exotic: a deadline stops the model mid-sentence, or the
    copy control fails and the DOM read returns what had rendered so far. Either
    way `json.loads` sees an array with no `]` and returns nothing at all --
    so nineteen perfectly good corrected cases are thrown away because the
    twentieth did not arrive.

    Nineteen cases are worth more than none. The `len(revised) < len(agreed)`
    guard in `verify.py` still refuses a set that came back SHORTER than the one
    it replaces, so a salvage cannot quietly shrink the bar -- it only rescues
    the case where the array was complete enough to keep.
    """
    if closed < 0:
        return None
    return text[start: closed + 1] + "]"


# What a model names the array when it wraps it in an object instead of sending
# it bare. `{"cases": [...]}` is a reply to "send back ALL of the cases" that
# reads perfectly to a human and parsed to nothing here.
_CASE_KEYS = ("cases", "tests", "test_cases", "testcases", "examples")


def _parse_cases(
    block: str, language: str, *, scan_prose: bool = False
) -> list[dict[str, Any]]:
    """One block as a case list, or []. Never raises.

    Strict-first and tolerant-after. The strictness that used to be the whole
    function is still the fast path -- a program can be a hundred kilobytes and
    parsing one as JSON comes out of the solve's own budget -- but it was also
    the only path, and on a REPAIR round the array it dropped was the corrected
    cases. Losing those means the same wrong case breaks the program again on
    every remaining round, which is a solve that spends its whole deadline
    reporting a failure the model already fixed.

    Measured against the shapes a model actually sends: an array wrapped in
    `{"cases": ...}`, a trailing comma, `//` comments, single quotes,
    `True`/`False`/`None`, prose around the array, and no fence at all. Every
    one of them was silently worth nothing.

    What did NOT loosen is the structural gate below: an item is a case only if
    it is a dict carrying `expected`. That is what keeps a program, a prompt
    echo or a stray list of numbers from being read as a suite, and no amount
    of dialect tolerance touches it.
    """
    text = block.strip()
    if not text.startswith("["):
        # One leaked language chip, and no more. A copy control that hands back
        # `json\n[{...}]` would otherwise fail the fast path and drop the
        # array -- and on a repair round that array is the corrected cases,
        # so losing it means the same wrong case breaks the program again on
        # every remaining round.
        head, _, rest = text.partition("\n")
        if head.strip().casefold() in ("json", "jsonc", "json5"):
            text = rest.strip()
    raw = _loads_cases(text)
    if isinstance(raw, dict):
        # Wrapped in an object. Take a named list, or the only list it holds.
        for key in _CASE_KEYS:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            lists = [v for v in raw.values() if isinstance(v, list)]
            raw = lists[0] if len(lists) == 1 else None
    if not isinstance(raw, list) and (text.lstrip().startswith("[") or scan_prose):
        # Two different jobs share this call, and only the second is gated.
        #
        # When the text ALREADY starts with `[`, this is not digging through
        # prose -- the model plainly sent an array and something is wrong with
        # it, most often that it stops mid-element because the reply was cut
        # off. `_array_span` bracket-matches it and `_truncated_span` salvages
        # the complete prefix. Gating that behind `scan_prose` cost the whole
        # rescue: a twenty-case correction truncated at the twentieth parsed to
        # nothing, which is the failure this was written to fix.
        #
        # Digging an array out of surrounding prose is the other job, and OFF
        # by default.
        #
        # Inside a fenced block prose-digging stays off because the contract asked for one
        # block and a model that obeyed it sent the array alone: an array buried
        # in a fence full of prose is more likely something the model was
        # talking ABOUT than a suite it meant to send, and grading a program
        # against cases nobody wrote is worse than grading it against none.
        #
        # It is on for the reply as a whole, where the alternative is not a
        # stricter reading but no reading at all -- a model that ignored the
        # fence entirely still answered, and the structural gate below is what
        # decides whether what it wrote is a suite.
        for span in _array_spans(text):
            candidate = _loads_cases(span)
            if isinstance(candidate, list) and _case_items(candidate, language):
                raw = candidate
                break
    if not isinstance(raw, list):
        return []
    return _case_items(raw, language)


def salvage_case_array(text: str) -> Optional[str]:
    """An UNFENCED test-case array inside `text`, re-fenced, or None.

    The browser layer only ever hands back fenced code blocks -- deliberately,
    and for a reason that has cost whole solves: claude.ai renders extended
    thinking inside the element the assistant selector matches, so falling back
    to the message text once submitted 13,200 characters of reasoning as a Rust
    program. That rule has one blind spot, and it is exactly the reply the
    INPUTS turn asks for -- the only turn that still wants a JSON array rather
    than code. `inputs_from_payload` can read an array out of anything, but it
    never gets the chance: a model that writes the array as ordinary text
    renders no `pre code`, so the page read returns None and the reply reaches
    `prompts.py` as the empty string. The fallback was unreachable from the one
    path that needed it.

    This is the narrow way through, and it stays narrow on purpose:

      * Only when NOTHING was fenced. A reply that obeyed the contract is read
        the way it always was.
      * Only an array whose items pass the same structural gate everything else
        here uses -- a dict carrying `expected`. Prose, a program, a stray list
        of numbers and a quoted prompt all fail it.
      * The result is CASES, never a program. `extract_code` skips a block that
        parses as cases, so nothing salvaged here can be submitted as an answer;
        the worst it can do is move the bar, which every other case array can
        do too and which the caller's own guards already cover.

    The gate is applied as `python` because a tab does not know the task's
    language, and it is the permissive of the two. Nothing is conceded: the
    Rust-specific filter still runs downstream in `_parse_cases`, where the
    language is known.
    """
    if not text or "[" not in text:
        return None
    cleaned = _OPEN_THINK_RE.sub("", _THINK_RE.sub("", sanitize_code(text)))
    if fenced_blocks(cleaned):
        return None
    for span in _array_spans(cleaned):
        raw = _loads_cases(span)
        if isinstance(raw, list) and _case_items(raw, "python"):
            return f"```json\n{span.strip()}\n```"
    return None


def _case_items(raw: list, language: str) -> list[dict[str, Any]]:
    """The structural gate, and the only thing that decides what a case is.

    Kept apart from the parsing above because the span search has to ask this
    question too: "did this bracketed span actually hold cases, or should I look
    at the next one?" is the same question as "is this a suite".

    An item counts only if it is a dict carrying `args` or `expected`. No
    amount of dialect tolerance upstream touches that -- it is what keeps a
    program, a prompt echo or a stray list of numbers from being read as a
    suite.

    `args` alone is enough because the inputs turn is now forbidden to supply
    an expected value: the reference program computes every expectation by
    being run. Requiring `expected` made this gate reject exactly the array
    the live design asks for, which silently disabled the prose salvage on the
    one turn that still sends a JSON array. An item with neither key is not a
    case in any dialect.
    """
    cases: list[dict[str, Any]] = []
    # One case per CALL. Two cases with the same arguments are either the same
    # case twice -- an executor run bought for nothing -- or a contradiction no
    # program can satisfy, and neither is worth carrying. It also keeps the
    # suite key-unique, which the correction merge in `verify.py` depends on:
    # there a case is identified by its call, so a duplicate call meant one
    # failing case took a PASSING one off the bar with it and left room for a
    # model-authored case to replace both. Measured on the real function.
    seen: set[tuple] = set()
    for item in raw:
        if not isinstance(item, dict) or not ({"args", "expected"} & set(item)):
            continue
        args = item.get("args", [])
        if not isinstance(args, list):
            args = [args]
        kwargs = item.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        if language == "rust":
            # The Rust judge feeds `args[0]` to stdin and compares stdout, so a
            # case shaped for a function call cannot run at all. Discarding it
            # is the honest outcome: a case that cannot run is not evidence.
            if len(args) != 1 or not isinstance(args[0], str):
                continue
            # An expected value, IF the model supplied one, must be the stdout
            # string the judge compares. A number there matches neither the
            # reference's output nor the program's. Absent is fine: that is
            # what the inputs turn is asked for.
            if "expected" in item and not isinstance(item["expected"], str):
                continue
        key = (repr(args), repr(sorted(kwargs.items(), key=repr)))
        if key in seen:
            continue
        seen.add(key)
        name = item.get("name")
        cases.append({
            "args": args,
            "kwargs": kwargs,
            "expected": item.get("expected"),
            "name": str(name)[:80] if isinstance(name, str) else "",
        })
    return cases


def extract_code(
    reply: str, entrypoint: Optional[str] = None, language: str = "python"
) -> str:
    """Pull the source out of a model reply.

    The DOM reader already returns a code block's inner text for ChatGPT, but
    a reply can still arrive fenced (or as prose). Prefer the LAST fenced
    block — models often show a wrong first draft before the final answer.
    """
    if not reply:
        return ""
    # Clean BEFORE matching: a stray invisible character inside the opening
    # fence would stop the block being recognised at all.
    reply = sanitize_code(reply)
    # ...and drop the model's own reasoning before matching too, so the fences
    # it quoted while thinking never compete with the answer.
    reply = _OPEN_THINK_RE.sub("", _THINK_RE.sub("", reply))
    matches = fenced_blocks(reply)
    if not matches:
        # No fence anywhere. That is usually a reply that is ALL prose -- a
        # refusal, a clarifying question, or a model's reasoning scraped before
        # it wrote anything -- and handing prose back as `code` is not a
        # harmless guess. It parses as "a program with a defect", so the caller
        # reports "the program does not define `fn main()`" about a program
        # that was never sent, and the repair round asks the model to fix
        # logic instead of telling it that nothing arrived. That has cost whole
        # solves: two rounds spent rewriting an algorithm over a reply that had
        # no code in it at all.
        #
        # But an unfenced reply is not always prose: a model that ignores the
        # formatting rule and types the program bare has still answered, and
        # dropping that would trade one silent failure for another. The
        # existing notion of "gradeable" settles it -- the same defect check
        # used to choose between fenced blocks below. Code passes and is kept;
        # prose fails and is reported as nothing arrived, which is true.
        bare = reply.strip()
        if not bare or entrypoint is None:
            return bare
        defect = (
            rust_defect(bare) if language == "rust"
            else python_defect(bare, entrypoint)
        )
        return bare if defect is None else ""
    blocks = [b for b in (sanitize_code(m).strip() for m in matches) if b]
    # A block that IS the cases is not a candidate program, and saying so here
    # rather than downstream is what keeps a cases-only correction readable as
    # one. `_converse` decides between "the model corrected its cases" and "the
    # model rewrote both" by asking whether any code arrived; a `json` array
    # answered to that question makes a reply that changed no code look like a
    # rewrite, and the corrected cases are then deferred a round they do not
    # have. The gate is `_parse_cases`'s own -- dicts carrying `expected` -- so
    # nothing that could be a program is excluded by it.
    blocks = [b for b in blocks if not _parse_cases(b, language)]
    if not blocks:
        return ""
    # With a target in hand, prefer the LAST block that would actually grade.
    # Models append usage examples and print() demos after the answer, and the
    # last block is then the demo. Reusing the defect check as the test means
    # "gradeable" here is exactly what it means everywhere else.
    if entrypoint:
        for i in range(len(blocks) - 1, -1, -1):
            defect = (
                rust_defect(blocks[i]) if language == "rust"
                else python_defect(blocks[i], entrypoint)
            )
            if defect is None:
                return _carry_imports(blocks[i], blocks[:i], language)
    # Nothing clean. Fall back only as far as something that is plausibly
    # source: a broken program is still an attempt, and the defect it reports
    # is what the repair round needs to hear, but a tool call is not an attempt
    # at all. Handing one over submits a guaranteed zero AND archives it as
    # "the solution", which is how a tool call came to be saved as a Rust
    # program on a solve where the model had answered correctly.
    usable = [b for b in reversed(blocks) if plausible_source(b, language)]
    if not usable:
        return ""
    # Among those, prefer one that at least DEFINES what the grader is going to
    # call. Taking the last plausible block instead was measured doing real
    # damage: a model answers, then appends `print(g([1, 2, 3]))` as a usage
    # example, and that demo is perfectly plausible Python. The moment the
    # answer above it picks up any defect at all -- a genuine truncation, or a
    # false positive from the fall-off-the-end check -- the demo becomes the
    # last plausible block and wins. A whole correct program was replaced by its
    # own one-line example, submitted, and archived as the solution. The demo
    # also teaches the repair round nothing: it is told the code does not define
    # `g`, about a block that was never trying to.
    if entrypoint:
        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i] in usable and _defines(blocks[i], entrypoint, language):
                return _carry_imports(blocks[i], blocks[:i], language)
    return usable[0]


_BUILTIN_NAMES = frozenset(dir(builtins))


def _import_bindings(code: str) -> dict[str, str]:
    """``{name: the top-level import statement that binds it}``.

    Only import statements are ever read out, and that is the safety property.
    Whatever this returns gets PREPENDED to the answer, so it must not be able
    to do work: an assignment, a call, a definition beside the imports stays
    where it is and is never copied. An `import` is the one statement that
    cannot surprise -- and where it can, because the module is unavailable, the
    answer that needed it was already lost.

    An earlier attempt required the whole block to be nothing but imports. That
    read well and bought nothing: a block mixing `import sys` with a
    `sys.setrecursionlimit(...)` call is still a header the model split off,
    and refusing it lost the import as well as the call. Mutation testing is
    what surfaced it -- removing the restriction changed no behaviour any test
    could see, because only the import lines were ever taken either way.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    bound: dict[str, str] = {}
    lines = code.splitlines()
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name == "*":
                return {}  # no way to know what it binds
            bound[alias.asname or alias.name.split(".")[0]] = "\n".join(
                lines[node.lineno - 1: node.end_lineno]
            )
    return bound


def _bound(code: str) -> set[str]:
    """Every name this block binds, in any scope.

    Collected from every scope at once rather than per-scope, and that is
    deliberately over-permissive: a local named `math` in some other function
    hides a genuinely missing module-level `math`. Both callers want the error
    in that direction -- `_unbound` would otherwise prepend an import the answer
    never asked for, and `dropped_definitions` would otherwise report a program
    incomplete when it is not.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                # `import *` is NOT treated as binding everything. Doing that
                # was the cautious-looking choice and it cost a carry: a block
                # holding `from collections import *` beside a use of `math`
                # reported nothing missing, so the `math` split into an earlier
                # block was left behind and every test failed on NameError.
                # A name a star-import really does supply is simply one that no
                # earlier block binds either, so nothing is carried for it.
                if alias.name != "*":
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _reads(code: str) -> set[str]:
    """Every name this block reads."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _unbound(code: str) -> set[str]:
    """Names this block READS and never binds anywhere, builtins aside."""
    return _reads(code) - _bound(code) - _BUILTIN_NAMES


def dropped_definitions(code: str, previous: str) -> Optional[str]:
    """Names this reply uses that only the reply BEFORE it defined, or None.

    A repair round is asked for the whole program and sometimes sends back only
    the part it changed. That reply is not a worse answer, it is a fragment of
    one: it parses, `compile()` is happy with it, `python_defect` reports it
    clean, and every hidden test dies on `NameError` -- for a helper that was
    right there in the round above. Submitted at the deadline, ungraded, it is
    a certain zero wearing a clean bill of health.

    So the check is not "does this look like a diff". Marker-hunting was
    measured against the 97 archived answers and found two hits, BOTH FALSE --
    Rust holding `b"UNCHANGED "` and `let mut unchanged` -- and no true ones.
    This asks the only question with an answer: does this program use something
    that only its predecessor supplied?

    Python only, and the narrowness is the point:

    * `_carry_imports` runs inside `extract_code` first, so a dropped IMPORT is
      already spliced back before this sees it. What reaches here is helpers,
      classes and constants -- which is precisely "only the updated code".
    * Rust name resolution is not something to guess at (the same reason
      `_carry_imports` gives for its own Rust path), and a Rust answer goes
      through `compile_defect`'s real `rustc`, which reports a dropped `fn` far
      better than any text search could.
    * Both halves lean over-permissive -- `_bound` collects from every scope --
      so a name has to be genuinely unresolvable HERE and genuinely bound THERE
      before anything is said. Measured: fires on none of the 26 archived
      Python answers.
    """
    if not code.strip() or not previous.strip():
        return None
    lost = sorted(_unbound(code) & _bound(previous))
    if not lost:
        return None
    names = ", ".join(f"`{n}`" for n in lost[:4])
    if len(lost) > 4:
        names += f" and {len(lost) - 4} more"
    return (
        f"this is only part of the program: it uses {names}, which your "
        f"previous reply defined and this one does not"
    )


_RUST_USE_RE = re.compile(r"^[ \t]*(?:pub\s+)?use\s", re.MULTILINE)
# A preamble block: `use` lines, `extern crate`, inner/outer attributes and
# comments, and nothing else. Anything with a body is a program, not a preamble.
_RUST_PREAMBLE_LINE_RE = re.compile(
    r"^[ \t]*(?:(?:pub\s+)?use\s.*;|extern\s+crate\s.*;|#!?\[.*\]|//.*)?[ \t]*$"
)


def _is_rust_preamble(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    return bool(lines) and all(_RUST_PREAMBLE_LINE_RE.fullmatch(ln) for ln in lines)


def _carry_imports(block: str, earlier: list[str], language: str) -> str:
    """Bring forward an imports-only block the chosen one turns out to need.

    A model told to send ONE code block sometimes sends its imports in a block
    of their own. Taking the block that defines the entrypoint then leaves the
    imports behind, and the result is worse than a visibly broken answer: it
    parses, it defines the right function, `python_defect` passes it, and every
    hidden test fails with `NameError: name 'math' is not defined`. Nothing
    anywhere says so. Measured on exactly that pair of blocks.

    Narrow on purpose. Only top-level `import` statements are eligible, only
    the ones binding a name this block reads and never binds are taken, and a
    block that needs nothing is returned untouched.

    Rust used to be left alone on the grounds that `use` has the same shape but
    a Rust answer goes through the compiler, which says so. The compiler is
    allowed not to be there -- no local `rustc`, or `SOLVER_RUST_COMPILE=0` --
    and then nothing says so at all, exactly as with `_rust_unclosed`. Narrower
    still than the Python path, because Rust name resolution is not something to
    guess at: an earlier block that is NOTHING but `use` lines and attributes is
    a preamble the model split off, and it is carried only when the chosen block
    has no `use` of its own.
    """
    if language == "rust":
        if _RUST_USE_RE.search(block):
            return block
        carried = [b.strip() for b in earlier if _is_rust_preamble(b)]
        if not carried:
            return block
        return "\n".join(carried) + "\n\n" + block
    missing = _unbound(block)
    if not missing:
        return block
    carried: list[str] = []
    for other in earlier:
        bindings = _import_bindings(other)
        for name in sorted(missing & set(bindings)):
            if bindings[name] not in carried:
                carried.append(bindings[name])
        missing -= set(bindings)
    if not carried:
        return block
    return _splice_imports(block, carried)


def _splice_imports(block: str, carried: list[str]) -> str:
    """Put ``carried`` at the top of ``block`` -- but below what must be first.

    `from __future__` has to be the first statement in the file, after at most a
    docstring, and `import math` above it is a file `ast.parse` accepts and the
    grader's import rejects:

        SyntaxError: from __future__ imports must occur at the beginning of the
        file

    Nothing downstream noticed, so it went out as a confident answer and scored
    zero. Anything the module may legally open with -- its docstring, a
    `__future__` import, a comment, a shebang, an encoding line -- stays where
    it is and the carried imports follow it.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        # No tree to ask, so no claim to make about what must come first.
        return "\n".join(carried) + "\n\n" + block
    after = 0
    for node in tree.body:
        docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if not (docstring or future):
            break
        after = node.end_lineno or after
    if not after:
        return "\n".join(carried) + "\n\n" + block
    lines = block.splitlines()
    head, tail = lines[:after], lines[after:]
    return "\n".join(head + [""] + carried + [""] + [t for t in tail if True]).rstrip("\n") + "\n"


def _defines(code: str, entrypoint: str, language: str = "python") -> bool:
    """Does this block define the thing the grader will call?

    Not "is it correct" and not "is it complete". A truncated answer still
    defines its function, and that block is precisely the one a repair round
    needs to be shown -- which is why this is a separate question from
    `*_defect` rather than a re-use of it.
    """
    if language == "rust":
        return bool(_RUST_MAIN_RE.search(code))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Cut off mid-statement, so there is no tree to ask. The definition line
        # itself is the best evidence left, and it is the part that survives a
        # truncation: the cut is at the END of the answer, not the start.
        return re.search(
            rf"^[ \t]*(?:async\s+)?def\s+{re.escape(entrypoint)}\s*\(", code, re.MULTILINE
        ) is not None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == entrypoint for t in node.targets
        ):
            return True
    return False


# Said by both defect checks, and carried into the repair prompt as a `defect`
# with `found_by="unrun"`, because "nothing arrived" needs a different
# conversation from "what arrived is wrong".
NO_CODE = "the reply contained no code"


# A code block a chat UI paints is not necessarily an ANSWER. When a model
# reaches for its tools, every tool call is painted as a `pre code` block too,
# and the reader cannot tell one from the other -- so the question "is this
# plausibly source in the target language" has to be asked before a block is
# allowed to become a submission.
#
# The two languages need different tests, and the asymmetry is real rather than
# laziness. Rust's top level is a CLOSED grammar: a file can only begin with an
# item, an attribute or a comment, so an allowlist of openers is exact and a
# shell command or a JSON object fails at its first character. Python's top
# level is arbitrary statements -- a perfectly good answer may open with
# `MOD = 10**9 + 7` -- so no allowlist can be written that does not reject real
# code. What CAN be named there is the short list of things a tool call opens
# with.
_SHELL_COMMANDS = (
    r"cat|cd|mkdir|echo|ls|rm|cp|mv|touch|chmod|export|sudo"
    r"|apt|apt-get|yum|brew|pip3?|python3?|rustc|cargo|npm|yarn|git|curl|wget"
    r"|bash|sh|zsh|make|which|pytest|node"
)
# `$ ` and `> ` open no Python statement, so a bare prompt character is enough.
# `#` is different and the difference cost real answers: it is a ROOT shell
# prompt and it is also how a great many Python programs begin. `# Sliding
# window over the log lines.` matched `[$#>]\s` and the whole block was declared
# "not source at all" -- so `extract_code` fell past it and submitted the
# model's own one-line usage example instead. Deleting only that comment made
# the same reply return the program.
#
# So `#` counts only when a command follows it, which is what a root prompt
# actually looks like and what no comment does. The bare command alternatives
# below still catch an unprompted command line.
_SHELL_OPENER_RE = re.compile(
    r"^[ \t]*(?:[$>]\s|\#\s*(?:" + _SHELL_COMMANDS + r")\b|(?:"
    + _SHELL_COMMANDS + r")\b)"
)


def plausible_source(code: str, language: str = "python") -> bool:
    """Could this block be source at all, before asking whether it is correct?

    Deliberately not the same question as `*_defect`. A program with a fixable
    flaw -- no entrypoint, a syntax error, a truncated line -- IS an attempt at
    an answer, and it is worth submitting and worth showing the repair round.
    A tool call is not an attempt at anything: submitting it guarantees a zero
    and tells the model nothing it can act on.
    """
    first = next((line for line in code.splitlines() if line.strip()), "")
    if not first:
        return False
    if language == "rust":
        return bool(_RUST_OPENER_RE.match(first))
    if first.lstrip()[:1] in "{[":
        return False  # a JSON payload, not a program
    return not _SHELL_OPENER_RE.match(first)


def _is_generator(fn) -> bool:
    """Does THIS function yield? Not one nested inside it."""
    stack: list = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a yield in there makes IT a generator, not us
        stack.extend(ast.iter_child_nodes(node))
    return False


def _always_returns(body: list) -> bool:
    """Does this statement list guarantee a `return` or a `raise`?

    The question a compiler asks about a Rust function and nothing asks about a
    Python one. Only the LAST statement matters: anything before it can be
    skipped, so only the tail decides whether control can fall off the end.

    Conservative in the direction that costs least. A bare `for` loop is never
    treated as guaranteeing a return even when it obviously does, because the
    price of being wrong here is one repair round, while the price of missing a
    truncated answer is the whole solve.

    A loop with an `else`, though, is not a guess: the clause runs on every exit
    that is not a `break`, so an `else` that always returns leaves no way to
    fall through -- unless a `break` bound to THIS loop skips it. That is the
    same rule `while True:` already gets, and the same helper decides it.
    Without this, the ordinary "search, else report not found" shape was
    reported as `can reach the end of its body without returning ... which is
    what a reply cut off mid-answer looks like`, about a correct program. The
    cost is not only the wasted round: a block carrying a defect loses
    `extract_code`'s gradeable preference, and a trailing usage example can then
    outrank the answer.
    """
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return (
            bool(last.orelse)
            and _always_returns(last.body)
            and _always_returns(last.orelse)
        )
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _always_returns(last.body)
    if isinstance(last, ast.Try):
        if last.finalbody and _always_returns(last.finalbody):
            return True
        head = _always_returns(last.orelse) if last.orelse else _always_returns(last.body)
        return head and all(_always_returns(h.body) for h in last.handlers)
    if isinstance(last, ast.While):
        # `while True:` with no way out never falls through to the end -- and
        # `while 1:` is the same loop. Testing `is True` recognised only the
        # keyword, so the numeric spelling (which competitive-programming
        # answers use constantly) was reported as "can reach the end of its body
        # without returning ... which is what a reply cut off mid-answer looks
        # like", about a correct program. Any truthy constant reads the same
        # way to the interpreter, so it reads the same way here.
        if isinstance(last.test, ast.Constant) and bool(last.test.value):
            return not _breaks_out_of(last)
        return _loop_else_returns(last)
    if isinstance(last, (ast.For, ast.AsyncFor)):
        return _loop_else_returns(last)
    if isinstance(last, ast.Match):
        return bool(last.cases) and all(_always_returns(c.body) for c in last.cases)
    return False


def _loop_else_returns(loop) -> bool:
    """A loop whose `else` always returns, and which cannot `break` past it."""
    return (
        bool(loop.orelse)
        and _always_returns(loop.orelse)
        and not _breaks_out_of(loop)
    )


def _breaks_out_of(loop) -> bool:
    """Is there a `break` bound to THIS loop, rather than to one inside it?

    `ast.walk` sees every `break` in the subtree, and an inner loop's break
    exits the inner loop -- it says nothing about whether the outer `while True`
    can ever end. Counting those flagged a correct program as "can reach the end
    without returning", and the consequence was not merely a wasted repair
    round: the block was then outranked by the model's own usage example, and
    the example was what got submitted.
    """
    stack: list = list(loop.body) + list(loop.orelse)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(
            node,
            (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef,
             ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue  # a `break` in there belongs to it, not to us
        stack.extend(ast.iter_child_nodes(node))
    return False


def python_defect(code: str, entrypoint: str) -> Optional[str]:
    """Return a reason string if the source can't possibly be graded, else None.

    Catches the two failure modes that make a reply worthless before it is even
    executed: it isn't Python at all (prose, a refusal, a truncated stream), or
    it never defines the function the validator is going to call.
    """
    if not code.strip():
        return NO_CODE
    try:
        # `compile`, not `ast.parse`. The validator IMPORTS this source, and
        # import compiles it -- so `ast.parse` is the wrong question by exactly
        # the set of programs that parse and will not compile. `from __future__`
        # in the wrong place is the one that reached a validator: `ast.parse`
        # said fine, the import raised `SyntaxError: from __future__ imports
        # must occur at the beginning of the file`, and this function had
        # reported the answer clean. Same exception, same message shape, one
        # more class of certain zero caught before it ships.
        tree = ast.parse(code)
        compile(code, "<solution>", "exec")
    except SyntaxError as exc:
        return f"the code is not valid Python ({exc.msg}, line {exc.lineno})"
    except ValueError as exc:  # noqa: BLE001 - null bytes and the like
        return f"the code is not valid Python ({exc})"
    # A top-level statement that is just a bare name runs at import time and
    # raises NameError, so every hidden test fails. It is never meaningful code,
    # and it is exactly what a leaked language chip looks like once it parses.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
            return (
                f"line {node.lineno} is a bare name `{node.value.id}` at top "
                f"level; it raises NameError on import and fails every test"
            )
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            if _is_generator(node):
                # Also a defect, and a real one -- the grader compares RETURN
                # VALUES structurally, so what it receives here is a generator
                # object rather than the answer. But it is not a truncation, and
                # the sentence below would tell the model its reply was cut off.
                return (
                    f"`{entrypoint}` is a generator: it yields instead of "
                    f"returning, so the grader is handed a generator object "
                    f"rather than the answer"
                )
            if not _always_returns(node.body):
                # `ast.parse` is Python's version of grepping for `fn main`: it
                # is happy with source that was CUT OFF, because a reply
                # truncated at a statement boundary is still a valid module.
                # Measured on 25 real archived answers, two ended deep inside a
                # loop with no return after it -- both parsed, both were
                # submitted, both returned None on every hidden test, and
                # nothing anywhere noticed. This flagged exactly those two and
                # none of the other twenty-three.
                #
                # It is also a real defect when the model meant it: a grader
                # compares RETURN VALUES, so a function that falls off its own
                # end answers None. Rust gets this from the compiler for free.
                ending = type(node.body[-1]).__name__.lower()
                return (
                    f"`{entrypoint}` can reach the end of its body without "
                    f"returning, so it answers None — the body ends on a "
                    f"`{ending}` rather than a return, which is what a reply "
                    f"cut off mid-answer looks like"
                )
            return None
    # An assignment such as `f = lambda x: ...` is also callable, so accept it.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == entrypoint for t in node.targets
        ):
            return None
    return f"the code does not define a top-level function named `{entrypoint}`"


# `fn main` at the START of a line. Inside an escaped string -- a tool call's
# JSON arguments, say -- it only ever appears mid-line, after a literal `\n`.
_RUST_MAIN_RE = re.compile(
    r"^[ \t]*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?"
    r"fn\s+main\s*\(",
    re.MULTILINE,
)
# What a single-file Rust program can begin with. Everything a chat UI renders
# as a code block that ISN'T a program -- a shell command, a JSON payload, a
# diff -- begins with something else, and this is the cheapest way to tell them
# apart that does not need a compiler.
_RUST_OPENER_RE = re.compile(
    r"^[ \t]*(?:#!|#\[|//|/\*|use\b|fn\b|pub\b|mod\b|struct\b|enum\b|impl\b"
    r"|trait\b|const\b|static\b|type\b|unsafe\b|extern\b|async\b|macro_rules!)"
)


# `r"..."`, `r#"..."#`, `b"..."`, `br#"..."#`. The quote is required, which is
# what keeps a raw IDENTIFIER (`r#type`, `r#match`) from being read as one.
_RUST_RAW_RE = re.compile(r'b?r(#*)"')
_IDENT_CH = re.compile(r"[A-Za-z0-9_]")


def _rust_unclosed(code: str) -> Optional[str]:
    """The delimiter a truncated Rust program leaves open, or None.

    The check Python gets from `ast.parse` and `_always_returns`, and Rust had
    only from a compiler that is allowed not to be there. `rust_defect`'s two
    other tests both pass a truncation -- the first line still opens like Rust
    and `fn main` still begins a line -- so an answer the deadline cut in half
    went out as a confident one. Measured on a real submission: 10,608 bytes,
    75 `{` against 71 `}`, ending mid-identifier four blocks deep. `rustc` says
    `error: this file contains an unclosed delimiter`; nothing here did.

    Conservative in the one direction that matters. A false positive does not
    merely cost a repair round: a block carrying a defect loses `extract_code`'s
    "last gradeable" preference, and a trailing usage example can then outrank
    the real answer -- which is the damage `_breaks_out_of` was written for. So
    this reports ONLY a delimiter still open at the end of the input, and gives
    up (returns None) the moment the scan meets anything it cannot account for:
    a mismatched closer, a closer with nothing open, a string or comment that
    never ends. Each of those is at least as likely to be this function
    misreading Rust as it is to be a broken program.
    """
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    i, n, line = 0, len(code), 1
    while i < n:
        c = code[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n:
            if code[i + 1] == "/":
                while i < n and code[i] != "\n":
                    i += 1
                continue
            if code[i + 1] == "*":
                # Rust nests block comments, unlike C.
                depth, i = 1, i + 2
                while i < n and depth:
                    if code.startswith("/*", i):
                        depth, i = depth + 1, i + 2
                    elif code.startswith("*/", i):
                        depth, i = depth - 1, i + 2
                    else:
                        line += code[i] == "\n"
                        i += 1
                if depth:
                    return None  # ran off the end inside a comment
                continue
        raw = _RUST_RAW_RE.match(code, i)
        if raw and not (i and _IDENT_CH.match(code[i - 1])):
            close = '"' + raw.group(1)
            end = code.find(close, raw.end())
            if end < 0:
                return None
            line += code.count("\n", i, end)
            i = end + len(close)
            continue
        if c == '"':
            i += 1
            while i < n and code[i] != '"':
                line += code[i] == "\n"
                i += 2 if code[i] == "\\" else 1
            if i >= n:
                return None
            i += 1
            continue
        if c == "'":
            # A char literal, or a lifetime/loop label. `'a'` is a literal;
            # `'a` in `&'a str` or `'outer: loop` is not, and reading it as one
            # would swallow the rest of the line looking for a closing quote.
            nxt = code[i + 1] if i + 1 < n else ""
            after = code[i + 2] if i + 2 < n else ""
            if nxt and nxt != "\\" and _IDENT_CH.match(nxt) and after != "'":
                i += 1  # a lifetime: skip the tick, scan the name as code
                continue
            j = i + 1
            if j < n and code[j] == "\\":
                # PAST what the backslash escapes, not onto it. Landing on it
                # made `'\\''` close on its own escaped quote, so the real
                # closing tick opened a second literal and the scan
                # resynchronised on whatever tick came next -- a lifetime, a
                # later char literal. Differential-fuzzed against rustc over 726
                # generated programs the verdict never actually changed: the
                # swallowed span kept its own delimiters balanced, or the scan
                # ran off the end and declined to judge. So this is a
                # correctness fix, not a measured loss.
                j += 2
            while j < n and code[j] != "'":
                line += code[j] == "\n"
                j += 1
            if j >= n:
                return None
            i = j + 1
            continue
        if c in "([{":
            stack.append((c, line))
        elif c in pairs:
            if not stack or stack[-1][0] != pairs[c]:
                return None  # more likely this scanner than a broken program
            stack.pop()
        i += 1
    if not stack:
        return None
    opener, where = stack[0]
    return (
        f"the program ends with {len(stack)} unclosed `{opener}` — the outermost "
        f"opens on line {where} and is never closed, which is what a reply cut "
        f"off mid-answer looks like"
    )


def rust_defect(code: str) -> Optional[str]:
    """Cheap structural check before paying for a compile.

    Both tests here are stricter than they look, and they are stricter because
    of what a chat UI renders as a code block. When a model reaches for its
    tools, every tool call is painted as a `pre code` block too -- the reader
    cannot tell one from an answer, and it should not have to. A block holding
    `{"command": "cat > main.rs << 'EOF'\nfn main() ..."}` used to pass, because
    the old test was `"fn main" in code`: it merely MENTIONS `fn main`, inside
    a quoted shell string, inside JSON. Submitted, it is a guaranteed zero, and
    the grader's only complaint would have been a compile error nobody could
    trace back to a tool call.

    So: `fn main` must begin a line, which it never does inside an escaped
    string, and the file must begin the way a Rust file begins. A shell command
    or a JSON object fails the second test at its first character.

    The order of the two matters. Telling a model "your program does not define
    `fn main()`" about something that was never a program is the contradiction
    `_FOUND_BY` exists to avoid, so a block that is not Rust at all says exactly
    that instead.
    """
    if not code.strip():
        return NO_CODE
    first = next((line for line in code.splitlines() if line.strip()), "")
    if not _RUST_OPENER_RE.match(first):
        return (
            f"this does not look like a Rust program — it begins with "
            f"{first.strip()[:48]!r}"
        )
    if not _RUST_MAIN_RE.search(code):
        return "the program does not define `fn main()`"
    # Last, because it is the only one of the three that a program which IS
    # Rust can fail. See `_rust_unclosed` for why it only ever reports a
    # delimiter left open at the end.
    return _rust_unclosed(code)


# =========================================================================== #
# The nine-stage solve: analysis, inputs, oracle, candidate, repair
# =========================================================================== #
#
# Five prompts, and what separates them is the whole design rather than a
# tuning preference.
#
# The ORACLE and the CANDIDATE are asked for the same program from opposed
# instructions. The oracle is told the inputs are tiny and that nested loops
# are fine; the candidate is told the hidden tests are at the stated maximums.
# If they were one prompt the two programs would copy the same misreading and
# comparing them would establish nothing -- the difference between the
# instructions IS the evidence.
#
# The INPUTS turn is forbidden to supply expected values. It used to be asked
# for them, and that made the bar an echo of the same model's reading of the
# statement. Here the reference program computes them by being run, so the
# turn only has to invent inputs -- a strictly easier thing to be right about.


def _statement_header(task) -> str:
    """The statement, and the few facts about it every stage needs."""
    language = str(getattr(task, "language", "") or "python")
    return (
        f"language: {language}\n"
        f"entrypoint: {getattr(task, 'entrypoint', '') or ''}\n"
        "\nPROBLEM STATEMENT:\n"
        f"{getattr(task, 'statement', '') or ''}\n"
    )


def _is_python(task) -> bool:
    return str(getattr(task, "language", "") or "").strip().lower() == "python"


ANALYSIS_TASK = """\
You are reading a competitive-programming statement to find what would make a
straightforward implementation fail the hidden tests. You are NOT writing the
solution, and you will not be asked to.

Return JSON with exactly this shape:

{
  "summary": "one paragraph",
  "signature": "def name(...) or fn main()",
  "traps": [{"name": "", "evidence": "", "mitigation": "", "severity": "high|medium"}],
  "invariants": ["..."],
  "edge_cases": ["..."],
  "algorithm_sketch": "a concrete data structure and algorithm, not buzzwords",
  "complexity_time": "",
  "complexity_memory": "",
  "naive_failure": "why writing the obvious program straight out fails here",
  "io_notes": ""
}

Look hardest at: bounds too large to iterate, structures too large to build,
behaviour that continues after an explicit list is exhausted, index bases,
bytes against characters against graphemes, state that is versioned or shared,
counting rules that fire on one event and not another, and the exact tokens the
output must contain."""

ANALYSIS_OUTPUT_CONTRACT = _ONE_BLOCK + """\
That block is `json`, and it holds the analysis described above."""

# The inputs turn. Two rules do the work: inputs only, and keep them small
# enough that a deliberately slow reference can answer them.
_INPUTS_TASK_PYTHON = """\
Invent test INPUTS for this problem. Inputs only — you are not being asked what
they should return, and any expected value you write will be discarded.

A separate, deliberately slow reference implementation will be run on these
inputs to compute the answers, so every input must be one such a program can
finish: keep n small, usually 20 or less.

Each case is an argument list for `{entrypoint}`, called as
`{entrypoint}(*args, **kwargs)`.

{{
  "cases": [
    {{"name": "short name", "args": [...], "kwargs": {{}},
     "notes": "which trap this one is aimed at"}}
  ]
}}

Write 8 to 20 cases. `args` must be JSON — lists, not tuples. Cover the
ordinary path first, then: empty and single-element inputs, the boundaries of
every range the statement names, records the statement calls invalid, and one
case per trap listed above. Every case must be legal input under the statement;
a case the statement forbids proves nothing about the program."""

_INPUTS_TASK_RUST = """\
Invent test INPUTS for this problem. Inputs only — you are not being asked what
they should print, and any expected output you write will be discarded.

A separate, deliberately slow reference implementation will be run on these
inputs to compute the answers, so every input must be one such a program can
finish: keep the sizes small.

Each case is the exact bytes written to the program's stdin.

{
  "cases": [
    {"name": "short name", "stdin": "raw stdin, including its newlines",
     "notes": "which trap this one is aimed at"}
  ]
}

Write 6 to 15 cases. Cover the ordinary path first, then: the smallest legal
input, the boundaries of every range the statement names, and one case per trap
listed above. Every stdin must be well-formed under the statement — if it says
all input is valid, do not send input that is not."""

INPUTS_OUTPUT_CONTRACT = _ONE_BLOCK + """\
That block is `json`, and it holds the cases. Do NOT write any program."""

_ORACLE_TASK = """\
Write a REFERENCE implementation of this problem.

It will only ever be run on tiny inputs, and it exists to be obviously correct
rather than fast. Optimise for nothing except following the statement exactly:

- Prefer the direct, literal reading of every sentence. Simulate what the
  statement describes step by step.
- Nested loops are fine. Recomputing from scratch is fine. Do not reach for a
  closed form, and do not skip a branch because it looks rare — a rare branch
  is exactly what this is for.
- Handle every case the statement admits, including the empty and degenerate
  ones, rather than assuming they will not come up.
- Do not optimise. If you find yourself choosing a clever structure, choose the
  obvious one instead."""

_CANDIDATE_TASK = """\
Write the SOLUTION to this problem — the program that will be submitted.

The hidden tests include the largest inputs the statement allows, and there is
no partial credit, so it must be both correct and fast enough at those sizes:

- Correct on every case the statement admits, the degenerate ones included.
- Fast at the stated maximums. If the statement names a bound you would have to
  iterate to reach, you need a closed form, a compressed representation or an
  implicit one — not a faster loop.
- Every trap listed above is a trap you are expected to have handled."""

# The repair turn, and the one thing about it that is structural: the model
# answering it did NOT write the program it is being shown. The repair phase
# names a different model from the candidate phase deliberately -- a model
# asked to repair its own program defends its own reading of the statement --
# and the cost of that is exactly this prompt, which cannot rely on a
# conversation and has to carry the statement, the traps and the code itself.
_REPAIR_TASK = """\
The program below fails the check reported under it. Repair it.

You did not write this program. Read the failure as evidence about the program,
not as a claim you have to defend — and read the statement yourself rather than
trusting that the program's author read it correctly."""

# HOW the failure was found, and it has to be true. A repair prompt that
# describes a run that did not happen asks the wrong question: told its logic
# disagreed with a reference, a model rewrites logic -- and when the real fault
# is that the program never compiled, or never arrived, the rewrite goes
# straight back to the same place. Measured under the previous design on a Rust
# task: two complete, plausible programs, both reported as "I ran the program
# and got: the reply contained no code", both repaired against evidence that
# did not exist.
_FOUND_BY = {
    "differential": (
        "The failure was found by running this program and a separate reference\n"
        "implementation on the same inputs and comparing what they produced. Where\n"
        "they disagree, at least one of them is wrong about the statement."
    ),
    "examples": (
        "The failure was found by running this program against the worked examples\n"
        "that shipped with the statement. Those are ground truth: where the program\n"
        "disagrees with one, the program is wrong."
    ),
    "unrun": (
        "NOTHING WAS RUN. A local check refused this program before it could\n"
        "execute, so there is no failing input and no wrong answer — only the\n"
        "reason below. Fix that reason; the logic has not been judged."
    ),
}


def build_analysis_prompt(task, heuristic) -> str:
    """Stage 3. The free heuristic pass is shown so the model adds to it."""
    return (
        _statement_header(task)
        + "\nAlready found by a mechanical scan of the statement:\n"
        + heuristic.trap_block()
        + "\n\n"
        + ANALYSIS_TASK
        + "\n\n"
        + ANALYSIS_OUTPUT_CONTRACT
    )


def build_inputs_prompt(task, analysis, want_probe: bool = False) -> str:
    """Stage 4. Inputs only; the reference computes the answers.

    `want_probe` asks for the size-probe generator as a second block. It rides
    on this turn rather than costing one of its own for the same reason it
    always has: every case here is small enough for a deliberately slow
    reference to answer, so none of them is ever the size the validator runs.
    "Did it finish at scale" needs no expected value and so needs no oracle.
    """
    python = _is_python(task)
    entrypoint = getattr(task, "entrypoint", "") or "solve"
    body = (
        _INPUTS_TASK_PYTHON.format(entrypoint=entrypoint)
        if python
        else _INPUTS_TASK_RUST
    )
    if want_probe:
        # `{shape}` is what the generator must RETURN, and it is the only part
        # of this that `_probe_now` reads by name. Concatenating the task
        # unformatted shipped the literal five characters to the model, which
        # then invented a shape: the generator ran, returned something the
        # probe could not call the program with, and the size check was
        # skipped in silence on every solve that asked for it.
        tail = GENERATOR_TASK.format(
            shape=(_PROBE_SHAPE_PYTHON if python else _PROBE_SHAPE_RUST).format(
                entrypoint=entrypoint
            )
        ) + "\n\n" + TESTS_OUTPUT_CONTRACT_WITH_PROBE
    else:
        tail = INPUTS_OUTPUT_CONTRACT
    return (
        _statement_header(task)
        + "\nWHAT THE STATEMENT HIDES:\n"
        + analysis.as_prompt_block()
        + "\n"
        + body
        + "\n\n"
        + tail
    )


def build_oracle_prompt(task, analysis) -> str:
    """Stage 5. Correctness-first, small-n, deliberately not clever."""
    python = _is_python(task)
    entrypoint = getattr(task, "entrypoint", "") or "solve"
    shape = (
        f"Define exactly one top-level function named `{entrypoint}` and RETURN "
        "the answer. Standard library only, no printing, no stdin."
        if python
        else "One complete program with `fn main()`, reading stdin and writing "
        "stdout. Standard library only, no crates, no unsafe."
    )
    return (
        _statement_header(task)
        + "\nWHAT THE STATEMENT HIDES:\n"
        + analysis.as_prompt_block()
        + "\n"
        + _ORACLE_TASK
        + "\n\n"
        + shape
        + "\n"
        + (
            "Python integers are unbounded; use them rather than worrying "
            "about overflow.\n"
            if python
            else "Use i128 wherever a value might exceed i64. This program is "
            "never run on large input, so a slow but safe choice costs "
            "nothing.\n"
        )
        + "\n"
        + CODE_OUTPUT_CONTRACT.format(
            language="python" if python else "rust"
        )
    )


def build_candidate_prompt(task, analysis) -> str:
    """Stage 6. Complexity-first. This is the program that ships."""
    python = _is_python(task)
    entrypoint = getattr(task, "entrypoint", "") or "solve"
    rules = (
        PYTHON_RULES.format(entrypoint=entrypoint) if python else RUST_RULES
    )
    environment = PYTHON_ENVIRONMENT if python else RUST_ENVIRONMENT
    examples = _render_examples(
        "python" if python else "rust",
        list(getattr(task, "public_examples", None) or []),
    )
    return (
        _statement_header(task)
        + (f"\n{EXAMPLES_LABEL}\n{examples}\n" if examples else "")
        + "\nWHAT THE STATEMENT HIDES:\n"
        + analysis.as_prompt_block()
        + "\n"
        + _CANDIDATE_TASK
        + "\n\nRULES:\n"
        + rules
        + "\n\nTHE ENVIRONMENT IT RUNS IN:\n"
        + environment
        + "\n\n"
        + CODE_OUTPUT_CONTRACT.format(
            language="python" if python else "rust"
        )
    )


def build_differential_repair_prompt(
    task, analysis, code: str, report: str, kind: str = "candidate",
    defect: Optional[str] = None, found_by: str = "differential",
) -> str:
    """Stage 8. Self-contained: the repair model has no conversation to read.

    `kind` says WHICH artifact is being repaired, and it changes what the model
    is told it is looking at. Repairing the reference is not the same job as
    repairing the program that ships: the reference may be as slow as it likes
    and only has to stop falling over, while the candidate has to stay fast.

    `found_by` says how the failure was found, and it is not decoration -- see
    `_FOUND_BY`. An unknown value falls back to the differential wording, which
    is the one every repair the orchestrator actually sends uses unless it says
    otherwise.
    """
    python = _is_python(task)
    language = "python" if python else "rust"
    entrypoint = getattr(task, "entrypoint", "") or "solve"
    oracle = kind == "oracle"
    keep = (
        f"Keep the function name `{entrypoint}`."
        if python
        else "Keep a single `fn main()`."
    )
    aim = (
        "This is the REFERENCE implementation, not the submitted one. It only "
        "runs on tiny inputs, so do not make it faster — make it stop failing "
        "and keep it obviously correct."
        if oracle
        else "This is the program that will be SUBMITTED. It must stay correct "
        "at the largest inputs the statement allows; a repair that fixes this "
        "case and makes the program quadratic has not helped."
    )
    return (
        _statement_header(task)
        + "\nWHAT THE STATEMENT HIDES:\n"
        + analysis.as_prompt_block()
        + "\n"
        + _REPAIR_TASK
        + "\n\n"
        + _FOUND_BY.get(found_by, _FOUND_BY["differential"])
        + "\n\n"
        + aim
        + "\n\nTHE PROGRAM:\n"
        + f"```{language}\n{code.strip()}\n```\n"
        + (f"\nA local check also reports: {defect}\n" if defect else "")
        + "\nWHAT THE CHECK FOUND:\n"
        + report.strip()
        + "\n\n"
        + keep
        + "\n\nReply with "
        + WHOLE_PROGRAM
        + ".\n\n"
        + CODE_OUTPUT_CONTRACT.format(language=language)
    )


def extract_analysis(reply: str) -> Any:
    """The analysis JSON out of a reply, or None. Never raises.

    Stage 3 is the one stage allowed to produce nothing, so every failure here
    is a `None` the caller keeps the heuristic analysis for.
    """
    for block in fenced_blocks(reply or ""):
        value = _loads_cases(block)
        if isinstance(value, dict):
            return value
    value = _loads_cases(reply or "")
    if isinstance(value, dict):
        return value
    salvaged = _object_span(reply or "")
    return _loads_cases(salvaged) if salvaged else None


def _object_span(text: str) -> Optional[str]:
    """The outermost `{...}` in `text`, for a model that skipped the fence."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start:end + 1]


def extract_inputs(reply: str, language: str) -> list[dict[str, Any]]:
    """Stage 4's cases, as inputs. Any expected value present is DROPPED.

    Dropping it is not defensive tidying. A model that supplies an expected
    value has answered a question it was told not to answer, and taking it
    would put that model's reading of the statement back into the bar -- which
    is the exact correlation the reference program exists to break.
    """
    payload: Any = None
    for block in fenced_blocks(reply or ""):
        payload = _loads_cases(block)
        if payload is not None:
            break
    if payload is None:
        payload = _loads_cases(reply or "")
    if payload is None:
        # `salvage_case_array` hands back a RE-FENCED block, because its other
        # caller wants something that reads like an ordinary reply. Passing
        # that straight to the JSON reader parses the backticks as part of the
        # array, so the salvage succeeded and its result was dropped one line
        # later -- the prose rescue was unreachable from here.
        for block in fenced_blocks(salvage_case_array(reply or "") or ""):
            payload = _loads_cases(block)
            if payload is not None:
                break
    return inputs_from_payload(payload, language)


def inputs_from_payload(payload: Any, language: str) -> list[dict[str, Any]]:
    """The parsed JSON as input-only cases. Never raises."""
    raw: Any = None
    if isinstance(payload, dict):
        for key in _CASE_KEYS:
            if isinstance(payload.get(key), list):
                raw = payload[key]
                break
    elif isinstance(payload, list):
        raw = payload
    if not isinstance(raw, list):
        return []

    rust = str(language or "").strip().lower() == "rust"
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip() or f"case {index + 1}"
        if rust:
            stdin = item.get("stdin")
            if stdin is None:
                stdin = item.get("input")
            if not isinstance(stdin, str):
                continue
            args: list[Any] = [stdin]
            kwargs: dict[str, Any] = {}
        else:
            args = item.get("args")
            if args is None:
                args = item.get("input")
            if args is None:
                args = []
            if not isinstance(args, list):
                args = [args]
            kwargs = item.get("kwargs")
            kwargs = kwargs if isinstance(kwargs, dict) else {}
        try:
            key = json.dumps([args, kwargs], sort_keys=True, default=str)
        except Exception:  # noqa: BLE001 - an unserialisable case is not one
            continue
        if key in seen:
            continue
        seen.add(key)
        cases.append({
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "notes": str(item.get("notes") or item.get("note") or "").strip(),
        })
    # Capped, and thinned rather than truncated -- see `_thin`. The cap was
    # lost when the cases turn became the inputs turn, and it costs more here
    # than it did there: every input is now run TWICE, once by the reference
    # and once by the candidate, so an over-long reply buys double the
    # executor time it used to.
    return _thin(cases, MAX_INPUTS)

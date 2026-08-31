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
those examples are the only disambiguation a miner is given.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from typing import Any, Optional, Sequence

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

TESTS_OUTPUT_CONTRACT = _ONE_BLOCK + """\
That block is `json`, and it holds test cases. Do NOT write the program yet —
you will be asked for it next."""

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


# Turn 1. The cases, before the program exists.
#
# The ordering is the user's and it is right: the ordinary case first, so the
# common path is checked at all, then the boundaries where implementations
# actually break. Counts are stated per class rather than as a total, because a
# total invites a model to spend it all on the easy classes.
#
# ONE ordinary case, not three. Three of them are three runs of the same code
# path: the common path is either right or it is not, and a second and third
# typical input almost never disagree with the first. They are not free either
# -- every case is an executor run inside the solve's own deadline, a
# subprocess for Python and a container for Rust, and the repair loop re-runs
# the whole suite on every round. Two of those runs were buying a re-answer to
# a question already answered. The classes below are where implementations
# actually break, and the budget belongs to them.
#
# "You have not written the program yet, and that is deliberate" is the whole
# argument for splitting the turns. Cases written ALONGSIDE a program can be
# back-filled from what the program happens to do, and then they agree with its
# bugs. Cases written first cannot.
TESTS_TASK_PYTHON = """\
Write the test cases for this problem — the cases, not the program. You will be
asked for the program in my next message, and these cases are what it will be
RUN against before it is submitted, so a case you leave out is a case nobody
runs.

[{{"name": "ordinary",   "args": [[3, 1, 2]], "expected": 6}},
 {{"name": "empty",      "args": [[]],        "expected": 0}},
 {{"name": "one item",   "args": [[5]],       "expected": 5}}]

- `args` is the argument list for `{entrypoint}(*args)`; `kwargs` is optional.
- `expected` is the exact value the program must RETURN, written as JSON.
- Every value must be JSON: no tuples, no sets, no `inf`, no `NaN`, no code.
- `name` is a short label so a failure report can say which case broke.

Write, in this order:

1. ONE ordinary case. A typical input, nothing special about it. It is the
   common path, and a suite that tests only boundaries never checks it.
2. THE EMPTY VALUE, or zero: an empty list, an empty string, `0`, `{{}}` —
   whichever of them this statement allows.
3. ONE: a single element, `n = 1`, the smallest legal input.
4. THE BOUNDARY: every limit, threshold and modulus the statement names, tested
   AT that exact value, and the largest value it allows.
5. THE CASES THIS PROBLEM IS LIKELY TO BE GOT WRONG ON: inputs where a
   plausible implementation returns something the statement does not — ties and
   duplicates, every element equal, already sorted, exactly reversed, a rule the
   statement states given one input that makes it fire and one that NEARLY does,
   and whatever else this particular problem makes easy to get wrong.

Skip a class only when the statement makes it impossible. At most {limit} cases
in total. Derive every `expected` from the STATEMENT by reasoning it out.
You have not written the program yet, and that is deliberate: a case computed
from code agrees with the code's bugs, which is exactly what a test is supposed
to catch."""


TESTS_TASK_RUST = """\
Write the test cases for this problem — the cases, not the program. You will be
asked for the program in my next message, and these cases are what it will be
RUN against before it is submitted, so a case you leave out is a case nobody
runs.

[{{"name": "ordinary", "args": ["3\\n1 2 3\\n"], "expected": "6"}},
 {{"name": "one item", "args": ["1\\n5\\n"],     "expected": "5"}}]

- `args` holds exactly ONE string: the complete stdin the program reads.
- `expected` is the complete stdout it must write, as a string.
- Output is compared after splitting on whitespace, so spacing is forgiving but
  an extra or missing token is not.
- `name` is a short label so a failure report can say which case broke.

Write, in this order:

1. ONE ordinary case. A typical input, nothing special about it. It is the
   common path, and a suite that tests only boundaries never checks it.
2. THE EMPTY VALUE, or zero: the smallest legal input, a count of zero, and an
   empty payload after the count if the format allows one.
3. ONE: a single element, `n = 1`.
4. THE BOUNDARY: every limit, threshold and modulus the statement names, tested
   AT that exact value, and the largest value it allows.
5. THE CASES THIS PROBLEM IS LIKELY TO BE GOT WRONG ON: inputs where a
   plausible implementation writes something the statement does not — ties and
   duplicates, every element equal, already sorted, exactly reversed, a rule the
   statement states given one input that makes it fire and one that NEARLY does,
   and whatever else this particular problem makes easy to get wrong.

Skip a class only when the statement makes it impossible. At most {limit} cases
in total. Derive every `expected` from the STATEMENT by reasoning it out.
You have not written the program yet, and that is deliberate: a case computed
from code agrees with the code's bugs."""


def build_tests_prompt(
    language: str, statement: str, entrypoint: str, examples: list[dict[str, Any]]
) -> str:
    """Turn 1: ask for the cases and nothing else."""
    is_rust = language == "rust"
    parts = [
        "<output>", TESTS_OUTPUT_CONTRACT, "</output>", "",
        f'<problem language="{"rust" if is_rust else "python"}" '
        f'entrypoint="{entrypoint}">',
        statement.strip(),
        "</problem>", "",
    ]
    rendered = _render_examples(language, examples)
    if rendered:
        parts += [
            '<examples note="PUBLIC EXAMPLES — already known to be right. Your '
            'cases must AGREE with these and go far beyond them.">',
            rendered, "</examples>", "",
        ]
    parts += [
        "<task>",
        (TESTS_TASK_RUST if is_rust else TESTS_TASK_PYTHON).format(
            entrypoint=entrypoint, limit=MAX_SELF_TESTS
        ),
        "</task>",
    ]
    return "\n".join(parts)


def _render_cases(cases: list[dict[str, Any]], language: str, entrypoint: str) -> str:
    """The agreed cases, as the call the grader will actually make."""
    lines = []
    for index, case in enumerate(cases, 1):
        name = case.get("name") or ""
        label = f"{index}. {name}: " if name else f"{index}. "
        if language == "rust":
            lines.append(
                f"{label}stdin {json.dumps(case.get('args', [''])[0])} "
                f"-> stdout {json.dumps(case.get('expected'))}"
            )
        else:
            args = ", ".join(json.dumps(a) for a in case.get("args", []))
            kwargs = "".join(
                f", {k}={json.dumps(v)}" for k, v in (case.get("kwargs") or {}).items()
            )
            lines.append(
                f"{label}{entrypoint}({args}{kwargs}) -> {json.dumps(case.get('expected'))}"
            )
    return "\n".join(lines)


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


def build_code_prompt(
    language: str,
    statement: str,
    entrypoint: str,
    examples: list[dict[str, Any]],
    cases: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    """Turn 2: ask for the program, and only the program.

    ``cases`` is what turn 1 obtained, and it decides one thing:

    * a list      -- the cases exist, so they are restated as the bar this
                     program has to clear.
    * ``None``/[] -- turn 1 produced nothing usable, or was never asked. No bar
                     to point at, and the answer still goes out.

    Either way the contract is ONE block. A reply carrying a second one is a
    reply that spent output tokens inside the deadline on something nothing
    reads: the cases were settled a turn ago, and `extract_code` would have to
    step over whatever else arrived.

    Laid out in delimited sections, and the order is the argument. The output
    contract goes FIRST because it is the only instruction whose failure costs
    the entire answer rather than degrading it; the site's nudge repeats it last,
    so it holds both the primacy and the recency slot. The problem and its
    examples come next, because instructions about how to solve something are
    unreadable before you know what it is. Everything that shapes HOW to answer
    comes last, closest to where generation begins.

    The examples are labelled a floor rather than the specification. They are the
    friendliest thing in the message and the easiest to over-fit to, and the
    label is what stops them being read as the whole job.

    """
    is_rust = language == "rust"
    given = list(cases or [])
    rules = (RUST_RULES if is_rust else PYTHON_RULES).format(entrypoint=entrypoint)
    environment = RUST_ENVIRONMENT if is_rust else PYTHON_ENVIRONMENT
    contract = CODE_OUTPUT_CONTRACT.format(language="Rust" if is_rust else "Python")

    parts = [
        "<output>", contract, "</output>", "",
        f'<problem language="{"rust" if is_rust else "python"}" '
        f'entrypoint="{entrypoint}">',
        statement.strip(),
        "</problem>", "",
    ]
    rendered = _render_examples(language, examples)
    if rendered:
        parts += [
            "<examples note=\"PUBLIC EXAMPLES — a floor, not the specification, "
            "and already known to be right. Your program must reproduce these "
            "exactly, and where the statement is ambiguous they decide.\">",
            rendered,
            "</examples>", "",
        ]
    parts += ["<contract>", rules, "", environment, "</contract>"]
    if given:
        # The bar, stated as the calls the grader will actually make. These are
        # the model's OWN cases from the previous turn, echoed rather than
        # referred to: a model asked to honour "the cases you sent" has to
        # scroll back past its own JSON to find them, and what it half-
        # remembers is what the program gets checked against.
        #
        # Last in the message, which is where it belongs: it is what the
        # program has to clear, read immediately before the program is written.
        parts += [
            "",
            '<must_pass note="YOUR OWN cases from the previous message. Every '
            'one of these is RUN against your program before it is submitted.">',
            _render_cases(given, language, entrypoint),
            "</must_pass>",
        ]
    return "\n".join(parts)


def build_resume_prompt(
    language: str,
    statement: str,
    entrypoint: str,
    examples: list[dict[str, Any]],
    cases: Optional[Sequence[dict[str, Any]]],
    code: str,
    failures: list[str],
    defect: Optional[str] = None,
    from_self_tests: bool = False,
) -> str:
    """A repair round for a conversation that no longer exists.

    Repairs normally stay inside one conversation, because the model can see
    its own previous attempt there and the prompt need only carry what went
    wrong. When the tab that produced the answer cannot be used again -- the
    prompt will not go into it, or the page died -- that context is gone with
    it, and the round used to be abandoned along with it. Measured over a
    production run: fifteen answers went out carrying failures nobody had asked
    the model to fix, with an average of 129 seconds of budget unspent.

    So the whole conversation is reconstituted in one message: the problem as
    turn 2 states it, the program that was produced, and what happened when it
    ran. A fresh tab has no history, so nothing here may assume any.
    """
    base = build_code_prompt(language, statement, entrypoint, examples, cases=cases)
    report = build_repair_prompt(
        failures, language, entrypoint, defect=defect, from_self_tests=from_self_tests
    )
    return "\n".join([
        base,
        "",
        '<previous_attempt note="YOUR program, from a conversation that ended '
        'before it could be corrected. This is what happened when I ran it.">',
        f"```{'rust' if language == 'rust' else 'python'}",
        code.strip(),
        "```",
        "",
        report,
        "</previous_attempt>",
    ])


def build_repair_prompt(
    failures: list[str],
    language: str,
    entrypoint: str,
    defect: Optional[str] = None,
    from_self_tests: bool = False,
) -> str:
    """Ask for a fix, quoting the concrete failures the local grader found.

    The failures come from running the candidate through the validator's own
    executor, so this is real evidence rather than a vague 'try again' — which
    is the difference between a repair loop that converges and one that drifts.

    A ``defect`` is the other kind of problem entirely, and it must not be
    dressed up as the first. Defects are found BEFORE anything is executed —
    nothing arrived, it will not parse, there is no ``fn main`` — so telling the
    model "I ran the program against the examples and got: the program does not
    define `fn main()`" is not evidence but a contradiction. Faced with one, a
    model rewrites the logic, which was never the problem, and the repair round
    is spent for nothing. Ask about delivery when delivery failed, and about
    shape when the shape is wrong.

    What is NOT here is method. Earlier versions spent a paragraph on how to
    think about the failure -- trace the call, do not guess from the shape of
    it, re-check the fix against every other case silently, do not change both
    to make them agree. That is work which never reaches the reply, competing
    with the failure itself for attention, and it is the same class of
    instruction the two-phase rewrite already took out of turns 1 and 2. The
    error, and the one line naming what may come back. Nothing else.

    "Do not change both" is not lost by leaving it unsaid: it is enforced in
    ``verify.py``, where a reply that rewrites the program AND the cases is
    graded against the bar as it stood before it arrived, and a revision that
    drops cases is refused outright. The grader keeps the promise, so the
    prompt stops asking for it.
    """
    if defect == NO_CODE:
        body = (
            "Your previous reply did not reach me as code. I can only read the "
            "chat message itself.\n\n"
            f"Send the program again as one ordinary fenced code block written "
            f"directly in the chat, with nothing outside it: {WHOLE_PROGRAM}."
        )
    elif defect:
        body = (
            f"I could not run your previous reply: {defect}.\n\n"
            f"Send back ONE fenced code block, with nothing outside it: "
            f"{WHOLE_PROGRAM}."
        )
    elif from_self_tests:
        # Deliberately not "your solution is WRONG". These cases came from the
        # model itself, so a disagreement proves only that two things it wrote
        # contradict each other -- and telling it the CODE is at fault when the
        # CASE was wrong is how a repair round breaks a correct program. The
        # output rule names both ways out and lets the model pick.
        detail = "\n".join(f"  - {line}" for line in failures)
        target = "the program" if language == "rust" else f"`{entrypoint}`"
        body = (
            f"I ran {target} against the test cases you sent and got:\n"
            f"{detail}\n\n"
            f"Send back ONE fenced block: {WHOLE_PROGRAM} — or, if the case "
            f"was wrong rather than the program, a `json` array holding ALL of "
            f"the cases, corrected."
        )
    else:
        detail = "\n".join(f"  - {line}" for line in failures)
        target = "the program" if language == "rust" else f"`{entrypoint}`"
        body = (
            f"I ran {target} against the examples and got:\n"
            f"{detail}\n\n"
            f"Send back ONE fenced block, with nothing outside it: "
            f"{WHOLE_PROGRAM}."
        )
    return body


# The most cases one reply may contribute. Each one is an executor run against
# the solve's own budget -- a subprocess for Python, a container for Rust -- so
# a model that emits forty of them would spend the deadline proving its own
# program right instead of getting it submitted.
MAX_SELF_TESTS = 20


def extract_self_tests(
    reply: str, entrypoint: str, language: str = "python"
) -> list[dict[str, Any]]:
    """The cases the model wrote for its OWN program, or [].

    Production ships no `public_examples`: measured over a live run, 56 solves
    in a row reported `examples=0/0`. So the repair loop -- the one mechanism
    here that turns a nearly-right answer into a right one -- never had anything
    to run, and `verified` was False on every answer because nothing could be
    checked rather than because anything was wrong.

    A model cannot verify its own understanding of a statement, and nothing here
    pretends otherwise: cases that encode the same misreading as the code agree
    with it, and that class goes uncaught. What they DO catch is the commoner
    one by far -- the model knows what the answer should be and coded it wrong.
    That is exactly the "nearly right" class this miner exists to close, and it
    is objectively checkable: the program either produces the model's own stated
    value or it does not.

    Returns [] for anything unexpected. A malformed block must degrade to the
    behaviour that existed before this function did, never to an exception --
    it is parsed on the path that decides what gets submitted.
    """
    if not reply or not entrypoint:
        return []
    # `sanitize_code` first, as `extract_code` has always done. A reply read off
    # the RENDERED PAGE -- which is what happens whenever the copy control fails
    # -- carries the page's own characters: a non-breaking space where the model
    # typed a space, a zero-width joiner from a syntax highlighter. Neither
    # `json.loads` nor `ast.literal_eval` accepts one as whitespace, so a single
    # invisible character dropped an entire corrected suite. That fallback is in
    # the log this was found from.
    reply = sanitize_code(reply)
    fenced = fenced_blocks(reply)
    for block in fenced:
        # The program is skipped by `_parse_cases` rather than by an
        # is-this-the-program check, and that is deliberate. Such a check has no
        # reachable upside -- no Python or Rust program starts with `[` -- and a
        # real downside: `_defines` for Rust is a text search for `fn main`, so a
        # task about generating Rust would have its cases thrown away for
        # quoting the phrase in an expected value. The structural test is both
        # sufficient and the one that cannot misfire.
        cases = _parse_cases(block, language)
        if cases:
            return _thin(cases, MAX_SELF_TESTS)
    if fenced:
        # It used fences and none of them was a suite, so it did not send one.
        # Digging through the prose AROUND a block a model deliberately fenced
        # would read an array it was discussing as cases it meant to run.
        return []
    # NO fence anywhere. The contract asks for one, and a reply that ignores it
    # entirely is still a reply -- the array is right there in the text, and
    # `_parse_cases`'s structural gate is what decides, not the fence. Measured:
    # a corrected suite sent as bare text scored zero cases, so the same wrong
    # case broke the program on every remaining round.
    return _thin(_parse_cases(reply, language, scan_prose=True), MAX_SELF_TESTS)


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
    """Strip line comments and trailing commas, leaving strings untouched.

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


def _array_span(text: str) -> Optional[str]:
    """The first bracketed array in `text`, brackets included, or None.

    A model answers "send back ALL of the cases" with prose around the array
    about as often as with the array alone, and a reply read off the DOM rather
    than off the copy control arrives with its fence already gone. Both put the
    array somewhere other than at character zero.

    Bracket-matched rather than regex-matched, and string-aware for the same
    reason `_scrub_json` is: an expected value of `"]"` ends the array under any
    cheaper rule.
    """
    start = text.find("[")
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
        span = _array_span(text)
        raw = _loads_cases(span) if span else None
    if not isinstance(raw, list):
        return []
    cases: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "expected" not in item:
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
            if not isinstance(item.get("expected"), str):
                continue
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


# Said by both defect checks, and recognised by `build_repair_prompt`, because
# "nothing arrived" needs a different conversation from "what arrived is wrong".
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
    `build_repair_prompt` exists to avoid, so a block that is not Rust at all
    says exactly that instead.
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

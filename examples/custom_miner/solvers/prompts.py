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
                body = []
            continue
        if re.fullmatch(re.escape(fence[0]) + "{%d,}" % len(fence), stripped):
            if "\n".join(body).strip():
                blocks.append("\n".join(body) + "\n")
            body, fence = None, ""
            continue
        body.append(line)
    if body is not None and "\n".join(body).strip():
        blocks.append("\n".join(body) + "\n")
    return blocks


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


# Turn 1. The cases, before the program exists.
#
# The ordering is the user's and it is right: three ordinary cases first, so the
# common path is checked at all, then the boundaries where implementations
# actually break. Counts are stated per class rather than as a total, because a
# total invites a model to spend it all on the easy classes.
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

1. THREE ordinary cases. Typical inputs, nothing special about them. These are
   the common path, and a suite that tests only boundaries never checks it.
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

1. THREE ordinary cases. Typical inputs, nothing special about them. These are
   the common path, and a suite that tests only boundaries never checks it.
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
    """
    if defect == NO_CODE:
        body = (
            "Your previous reply did not reach me as code. I can only read the "
            "chat message itself, so an artifact, a canvas, a preview pane or a "
            "collapsed block is invisible to me.\n\n"
            "Send the COMPLETE program again as one ordinary fenced code block "
            "written directly in the chat. Do not create an artifact or canvas. "
            "Do not abbreviate it or replace any part with a comment. Same rules "
            "as before, and nothing outside the block."
        )
    elif defect:
        body = (
            f"I could not run your previous reply: {defect}.\n\n"
            "Nothing was executed, so none of this is about your logic yet — it "
            "is about what arrived. Put that right and send the COMPLETE program "
            "again as one ordinary fenced code block written directly in the "
            "chat, with nothing outside it. Same rules as before."
        )
    elif from_self_tests:
        # Deliberately not "your solution is WRONG". These cases came from the
        # model itself, so a disagreement proves only that two things it wrote
        # contradict each other -- and telling it the CODE is at fault when the
        # CASE was wrong is how a repair round breaks a correct program. Naming
        # the real question is also the more useful prompt: one of the two is
        # wrong, and deciding which is exactly the work.
        detail = "\n".join(f"  - {line}" for line in failures)
        target = "the program" if language == "rust" else f"`{entrypoint}`"
        body = (
            f"Your program and your own test cases DISAGREE. I ran {target} "
            f"against the cases you sent and got:\n"
            f"{detail}\n\n"
            "Exactly one of the two is wrong, and which one is the question. In "
            "your reasoning — not in the reply — go back to the STATEMENT and "
            "work out what it says the answer for that input is. If the case is "
            "right, fix the program; if the case was wrong, fix the case and "
            "leave the program alone. Do not change both to make them agree.\n\n"
            "Then re-check the fix against every OTHER case you were sent, "
            "silently: a repair that fixes one case and breaks another is still "
            "wrong. Reply with the corrected program block. "
            "If it was the CASE that was wrong, add a second `json` block "
            "holding the COMPLETE corrected list of cases — all of them, not "
            "only the ones you changed; otherwise send the program alone."
        )
    else:
        detail = "\n".join(f"  - {line}" for line in failures)
        target = "the program" if language == "rust" else f"`{entrypoint}`"
        body = (
            f"Your solution is WRONG. I ran {target} against the examples and got:\n"
            f"{detail}\n\n"
            "In your reasoning — not in the reply — trace the failing call through "
            "your code until you find the actual line where the computed value and "
            "the expected one part company; do not guess at the fix from the shape "
            "of the failure. Re-check the fix against every OTHER case you were "
            "sent, silently: a repair that fixes this one and breaks another is "
            "still wrong. Then reply with "
            "ONLY ONE corrected code block and nothing else."
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
    for block in fenced_blocks(reply):
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
    return []


def _thin(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """At most ``limit`` cases, keeping the SHAPE of the coverage.

    A head-slice was what this used to do, and the turn-1 prompt inverts the
    assumption that made it safe. Cases now arrive ordinary-first by explicit
    instruction, so `cases[:limit]` keeps the three easy ones and throws away
    the boundaries -- discarding exactly the cases the whole mechanism exists to
    run, and doing it silently.

    The first three are kept because the prompt puts the common path there, and
    the rest are sampled at an even stride so no class is dropped wholesale
    whatever order the model actually used.
    """
    if len(cases) <= limit:
        return list(cases)
    head = cases[: min(3, limit)]
    rest, room = cases[len(head):], limit - len(head)
    if room <= 0:
        return head
    stride = len(rest) / float(room)
    return head + [rest[int(i * stride)] for i in range(room)]


def _parse_cases(block: str, language: str) -> list[dict[str, Any]]:
    """One block as a case list, or []. Never raises.

    The `[` test is a FAST PATH, not a guard: `json.loads` on a program raises
    and this returns [] either way. It is here because it runs against every
    fenced block of every reply, and a program can be a hundred kilobytes -- the
    cost of parsing one as JSON comes out of the solve's own budget. Removing it
    changes no outcome, only how much work is done to reach the same one.
    """
    text = block.strip()
    if not text.startswith("["):
        # One leaked language chip, and no more. A copy control that hands back
        # `json\n[{...}]` would otherwise fail the fast path and drop the
        # array -- and on a repair round that array is the corrected cases,
        # so losing it means the same wrong case breaks the program again on
        # every remaining round.
        head, _, rest = text.partition("\n")
        if head.strip().casefold() not in ("json", "jsonc", "json5"):
            return []
        text = rest.strip()
        if not text.startswith("["):
            return []
    try:
        raw = json.loads(text)
    except Exception:  # noqa: BLE001 - a model wrote it; anything is possible
        return []
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


def _unbound(code: str) -> set[str]:
    """Names this block READS and never binds anywhere, builtins aside.

    Bindings are collected from every scope at once rather than per-scope. That
    is deliberately over-permissive: a local named `math` in some other function
    hides a genuinely missing module-level `math`, and the cost of that is one
    import not carried forward -- the failure this already had. The opposite
    error would prepend an import the answer never asked for.
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
    used = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return used - bound - _BUILTIN_NAMES


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
    block that needs nothing is returned untouched. Rust is left alone: `use` has the
    same shape, but a Rust answer is put through the compiler, which says so.
    """
    if language == "rust":
        return block
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
    return "\n".join(carried) + "\n\n" + block


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
_SHELL_OPENER_RE = re.compile(
    r"^[ \t]*(?:[$#>]\s|cat|cd|mkdir|echo|ls|rm|cp|mv|touch|chmod|export|sudo"
    r"|apt|apt-get|yum|brew|pip3?|python3?|rustc|cargo|npm|yarn|git|curl|wget"
    r"|bash|sh|zsh|make|which|pytest|node)\b"
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


def _always_returns(body: list) -> bool:
    """Does this statement list guarantee a `return` or a `raise`?

    The question a compiler asks about a Rust function and nothing asks about a
    Python one. Only the LAST statement matters: anything before it can be
    skipped, so only the tail decides whether control can fall off the end.

    Conservative in the direction that costs least. A `for` loop is never
    treated as guaranteeing a return even when it obviously does, because the
    price of being wrong here is one repair round, while the price of missing a
    truncated answer is the whole solve.
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
        # `while True:` with no way out never falls through to the end.
        if isinstance(last.test, ast.Constant) and last.test.value is True:
            return not _breaks_out_of(last)
        return False
    if isinstance(last, ast.Match):
        return bool(last.cases) and all(_always_returns(c.body) for c in last.cases)
    return False


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
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"the code is not valid Python ({exc.msg}, line {exc.lineno})"
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
    return None

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


# Stated first, and again last by the site's nudge. It is the only instruction
# whose failure costs the whole answer rather than degrading it, so it gets both
# the primacy and the recency slot and nothing else competes for either.
OUTPUT_CONTRACT = """\
Reply with ONE fenced {language} code block and nothing else. No preamble, no
explanation before it or after it. Only the code inside the fence is ever read."""

PYTHON_RULES = """\
- There is no partial credit. A program wrong on ONE hidden case scores
  exactly what no answer at all scores, so prefer the safe implementation
  over the clever one every time.
- Define exactly one top-level function named `{entrypoint}`. It is called
  directly as `{entrypoint}(*args, **kwargs)`.
- RETURN the answer. Do not print it, do not read stdin, do not call input().
  Printed output is ignored by the grader.
- Standard library only. No pip packages, no network, no file access.
- Write no comments and no docstrings. Nothing reads them, and the answer is
  scored partly on how fast it arrives.
- Do not include tests, example calls, or `if __name__ == "__main__"`."""

RUST_RULES = """\
- There is no partial credit. A program wrong on ONE hidden case scores
  exactly what no answer at all scores, so prefer the safe implementation
  over the clever one every time.
- Write ONE complete program with `fn main()`, compiled as a single file with
  `rustc --edition=2021 -C opt-level=2`. No Cargo, no crates, std only.
- READ the input from stdin and WRITE only the requested answer to stdout.
- Output is compared token-by-token after splitting on ASCII whitespace, so
  extra prose, labels or prompts make the answer wrong.
- Write no comments and no docstrings. Nothing reads them, and the answer is
  scored partly on how fast it arrives."""

# The public examples are the friendly ones. The hidden suite is where the
# score comes from, and it is written to break a solution that only handles the
# shape it was shown -- so the cases below are named one at a time rather than
# summarised as "handle edge cases", which every model agrees to and none acts
# on. Each line is a case that has a right answer the statement implies and a
# wrong answer a plausible implementation produces.
#
# The ORDER of the first sentence is the expensive part. It used to read "walk
# your solution through every one of these before you answer", and a model does
# what it is told: it narrated the walkthrough, at length, and only then began
# the program. Reported from a live tab, and the reason it costs a solve is not
# style -- the first attempt has about 135 seconds of a 225 second budget, and
# prose spent before the code is time the code does not get. Written the other
# way round the artifact exists first, so a reply cut short loses the checking
# pass rather than the whole answer.
# An ordered procedure, not advice. Ordering is the thing this prompt has had to
# fight hardest: told to check "before you answer", models narrated the check and
# ran out of time before the program existed. A numbered list leaves no room to
# read the steps in a different order, and step 6 is the one that must be last.
METHOD = """\
Work in this order:
1. Read the problem, then the examples. Where the statement is ambiguous, the
   examples decide — they are the only disambiguation you are given.
2. Write the program FIRST, complete and runnable.
3. Trace every example by hand through the code you wrote. If one disagrees,
   the code is wrong, not the example.
4. Put the code through the edge cases below and fix what breaks.
5. Read the program back against itself, using the self-check below.
6. Send the code and nothing else.

Steps 1 to 5 happen silently, in your reasoning. None of that work appears in
the reply."""

EDGE_CASES = """\
The hidden tests are adversarial; the examples are not. Each line below is a
case with a right answer the statement implies and a wrong answer a plausible
implementation produces:
- NOTHING: an empty list, an empty string, n = 0. Work out what the statement
  says the answer is, then make sure your code reaches it instead of crashing
  or dividing by a length of zero.
- ONE: n = 1, a single element, a one-character string. Almost every
  off-by-one bug is visible here and nowhere else.
- TWO: the smallest case where "first" and "last" are different elements, and
  where adjacent-pair logic stops agreeing with the general case.
- BOTH ENDS: the first element and the last, an empty range, and an inclusive
  bound against an exclusive one. Check the far end explicitly — a loop that is
  right at the start and short by one at the finish passes every example.
- EXTREME VALUES: 0, 1, -1, negative numbers, and the largest magnitude the
  statement allows. If no bound is given, assume values up to 10^18 and n up to
  10^5, and pick an algorithm and a type that survive both.
- DEGENERATE SHAPE: every element equal, every element a duplicate, already
  sorted, sorted backwards, all zeros."""


# Two of these are facts about THIS grader, not general advice, and both cost a
# solve when guessed at: the comparison is structural and strict about bools,
# and each test is on a five-second clock.
PYTHON_ENVIRONMENT = """\
- Python integers never overflow, but the default recursion limit is 1000, so a
  recursive answer dies at n = 10^4 with RecursionError. Write it iteratively,
  or raise the limit yourself at the top of the file.
- Each test gets about 5 seconds. O(n^2) over n = 10^5 does not fit.
- Iterating a `set` or `dict` of STRINGS gives a different order in every
  process — `PYTHONHASHSEED` is random by default. Measured: four runs of
  `list({'alpha','beta','gamma'})` gave four different orders, while a set of
  small ints gave the same order every time. So a solution tested with integers
  looks stable and is not. Sort before returning anything order-sensitive.
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
  the range cannot be exceeded.
- Read to EOF. Tolerate trailing newlines, blank lines and repeated spaces, and
  do not assume a fixed number of lines unless the statement fixes it.
- Deep recursion overflows the stack. Prefer iteration for n up to 10^5.
- `HashMap` and `HashSet` iteration order is unspecified and differs run to run.
  Use `BTreeMap`/`BTreeSet`, or sort, before emitting anything order-sensitive.
- Each test gets about 5 seconds, so lock stdout once and wrap it in a
  BufWriter rather than printing in a loop."""


# Read off 43 answers a live miner submitted. Ten were the model's own bugs, and
# EIGHT of those ten were visible on a careful re-read of the program itself --
# no test, no execution, no cleverness required. A helper called but never
# written. A `[-1]` on a list the program's own parser can empty. A sentinel of
# 255 pushed into a list and later used to index five buckets. An id used where
# a position was meant. A condition already guaranteed by the match arm it sat
# in, so the arm was dead. An index of 16 into a length-16 string.
#
# None of that is a hard-problem failure. It is a re-reading failure, and it is
# the one kind of mistake a prompt can actually reach: the model has everything
# it needs to catch these and simply does not look again. So it is asked to,
# once, against a list of exactly the things that have gone wrong.
SELF_CHECK = """\
Read the program back against itself and answer each of these about it. Every
line is a bug that has reached this grader inside a program that looked
finished:
- Every function you CALL, you also wrote. A helper you meant to add and did not
  is a compile error sitting in a file that otherwise looks complete.
- Every index is in range. Watch for two numberings of the same objects — an id
  used where a position is meant — and for a sentinel value that later reaches a
  subscript.
- Every branch is reachable and every branch produces its answer. A condition
  already guaranteed by the arm it sits in is dead code; an arm that computes a
  value and then discards it emits nothing at all.
- Every `[0]`, `[-1]`, `.pop()`, `.first()`, `.last()` has an answer for the
  empty case — including the empty case your OWN code can produce.
- When you rebuild derived state, you rebuild ALL of it. A refresh that updates
  four fields and forgets the fifth leaves the fifth silently stale.
- State you commit before a branch, that branch can undo. A counter advanced and
  not rewound on the path that should not have advanced it is a wrong answer
  that shows up under one ordering and not another."""


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


def build_initial_prompt(
    language: str, statement: str, entrypoint: str, examples: list[dict[str, Any]]
) -> str:
    """The one prompt that has to carry the whole contract.

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
    rules = (RUST_RULES if is_rust else PYTHON_RULES).format(entrypoint=entrypoint)
    environment = RUST_ENVIRONMENT if is_rust else PYTHON_ENVIRONMENT

    parts = [
        "<output>",
        OUTPUT_CONTRACT.format(language="Rust" if is_rust else "Python"),
        "</output>", "",
        f'<problem language="{"rust" if is_rust else "python"}" '
        f'entrypoint="{entrypoint}">',
        statement.strip(),
        "</problem>", "",
    ]
    rendered = _render_examples(language, examples)
    if rendered:
        parts += [
            "<examples note=\"PUBLIC EXAMPLES — a floor, not the specification. "
            "Your code must reproduce these exactly AND survive the cases below.\">",
            rendered,
            "</examples>", "",
        ]
    parts += [
        "<contract>", rules, "", environment, "</contract>", "",
        "<method>", METHOD, "",
        "<edge_cases>", EDGE_CASES, "</edge_cases>", "",
        "<self_check>", SELF_CHECK, "</self_check>",
        "</method>",
    ]
    return "\n".join(parts)


def build_repair_prompt(
    failures: list[str],
    language: str,
    entrypoint: str,
    defect: Optional[str] = None,
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
        return (
            "Your previous reply did not reach me as code. I can only read the "
            "chat message itself, so an artifact, a canvas, a preview pane or a "
            "collapsed block is invisible to me.\n\n"
            "Send the COMPLETE program again as one ordinary fenced code block "
            "written directly in the chat. Do not create an artifact or canvas. "
            "Do not abbreviate it or replace any part with a comment. Same rules "
            "as before, and nothing outside the code block."
        )
    if defect:
        return (
            f"I could not run your previous reply: {defect}.\n\n"
            "Nothing was executed, so none of this is about your logic yet — it "
            "is about what arrived. Put that right and send the COMPLETE program "
            "again as ONE ordinary fenced code block written directly in the "
            "chat, with nothing outside it. Same rules as before."
        )
    detail = "\n".join(f"  - {line}" for line in failures)
    target = "the program" if language == "rust" else f"`{entrypoint}`"
    return (
        f"Your solution is WRONG. I ran {target} against the examples and got:\n"
        f"{detail}\n\n"
        "Work out why, then reply with ONLY ONE corrected code block. "
        "Before you send it, run the fix back through the edge-case checklist "
        "from my first message — the empty case, n = 1, both ends, and the "
        "largest values allowed — because a repair that fixes the example and "
        "breaks a boundary scores the same zero. "
        "Same rules as before. Do not explain outside the code block."
    )


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

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
_FENCE_RE = re.compile(r"(`{3,})[^\n`]*\n(.*?)\n?\1`*", re.DOTALL)

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
Rules — the grader is automated and unforgiving:
- Reply with ONLY ONE Python code block and nothing else outside it.
- Define exactly one top-level function named `{entrypoint}`. It is called
  directly as `{entrypoint}(*args, **kwargs)`.
- RETURN the answer. Do not print it, do not read stdin, do not call input().
  Printed output is ignored by the grader.
- Standard library only. No pip packages, no network, no file access.
- Write no comments and no docstrings. Nothing reads them, and the answer is
  scored partly on how fast it arrives.
- Do not include tests, example calls, or `if __name__ == "__main__"`."""

RUST_RULES = """\
Rules — the grader is automated and unforgiving:
- Reply with ONLY ONE Rust code block and nothing else outside it.
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
EDGE_CASES = """\
Edge cases are where this is won or lost. Walk your solution through every one
of these before you answer, and fix what breaks:
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
PYTHON_EDGE_CASES = """\
- Python integers never overflow, but the default recursion limit is 1000, so a
  recursive answer dies at n = 10^4 with RecursionError. Write it iteratively,
  or raise the limit yourself at the top of the file.
- Each test gets about 5 seconds. O(n^2) over n = 10^5 does not fit.
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
RUST_EDGE_CASES = """\
- INTEGER OVERFLOW IS SILENT HERE. The grader compiles with `-C opt-level=2`,
  which turns overflow checks OFF: `i32` arithmetic wraps around and the
  program exits normally with a wrong answer instead of panicking. Two `i32`
  values of 2_000_000_000 add up to -294967296. Use `i64` everywhere by
  default, `i128` for products, and reach for `i32` only where you have proved
  the range cannot be exceeded.
- Read to EOF. Tolerate trailing newlines, blank lines and repeated spaces, and
  do not assume a fixed number of lines unless the statement fixes it.
- Deep recursion overflows the stack. Prefer iteration for n up to 10^5.
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


def build_initial_prompt(
    language: str, statement: str, entrypoint: str, examples: list[dict[str, Any]]
) -> str:
    """The one prompt that has to carry the whole contract.

    The examples come LAST on purpose. They are the friendliest thing in the
    message and the easiest to over-fit to, so the edge-case checklist is read
    first and the examples arrive already framed as a lower bound rather than
    as the specification.
    """
    is_rust = language == "rust"
    rules = (RUST_RULES if is_rust else PYTHON_RULES).format(entrypoint=entrypoint)
    edges = EDGE_CASES + "\n" + (RUST_EDGE_CASES if is_rust else PYTHON_EDGE_CASES)
    parts = [
        f"Solve this programming problem in {'Rust' if is_rust else 'Python'}.",
        "", rules,
        "", edges,
        "", "PROBLEM:", statement.strip(),
    ]
    rendered = _render_examples(language, examples)
    if rendered:
        parts += [
            "",
            "PUBLIC EXAMPLES — these are a floor, not the specification. Your "
            "code must reproduce them exactly AND survive the cases above:",
            rendered,
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
    matches = [m.group(2) for m in _FENCE_RE.finditer(reply)]
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
        for block in reversed(blocks):
            defect = (
                rust_defect(block) if language == "rust"
                else python_defect(block, entrypoint)
            )
            if defect is None:
                return block
    # Nothing clean: the last block is still the best guess, and the defect it
    # reports is what the repair round needs to hear.
    return blocks[-1]


# Said by both defect checks, and recognised by `build_repair_prompt`, because
# "nothing arrived" needs a different conversation from "what arrived is wrong".
NO_CODE = "the reply contained no code"


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
            return None
    # An assignment such as `f = lambda x: ...` is also callable, so accept it.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == entrypoint for t in node.targets
        ):
            return None
    return f"the code does not define a top-level function named `{entrypoint}`"


def rust_defect(code: str) -> Optional[str]:
    """Cheap structural check before paying for a compile."""
    if not code.strip():
        return NO_CODE
    if "fn main" not in code:
        return "the program does not define `fn main()`"
    return None

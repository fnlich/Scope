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
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


PYTHON_RULES = """\
Rules — the grader is automated and unforgiving:
- Reply with ONLY ONE Python code block and nothing else outside it.
- Define exactly one top-level function named `{entrypoint}`. It is called
  directly as `{entrypoint}(*args, **kwargs)`.
- RETURN the answer. Do not print it, do not read stdin, do not call input().
  Printed output is ignored by the grader.
- Standard library only. No pip packages, no network, no file access.
- Handle edge cases and large inputs; hidden tests go beyond the examples.
- Do not include tests, example calls, or `if __name__ == "__main__"`."""

RUST_RULES = """\
Rules — the grader is automated and unforgiving:
- Reply with ONLY ONE Rust code block and nothing else outside it.
- Write ONE complete program with `fn main()`, compiled as a single file with
  `rustc --edition=2021 -C opt-level=2`. No Cargo, no crates, std only.
- READ the input from stdin and WRITE only the requested answer to stdout.
- Output is compared token-by-token after splitting on ASCII whitespace, so
  extra prose, labels or prompts make the answer wrong.
- Handle edge cases and large inputs; hidden tests go beyond the examples.
- Prefer fast I/O (lock stdout, use a BufWriter) — inputs can be large."""


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
    rules = (RUST_RULES if language == "rust" else PYTHON_RULES).format(
        entrypoint=entrypoint
    )
    parts = [f"Solve this programming problem in {'Rust' if language == 'rust' else 'Python'}.", "", rules, "", "PROBLEM:", statement.strip()]
    rendered = _render_examples(language, examples)
    if rendered:
        parts += ["", "PUBLIC EXAMPLES (your code must reproduce these exactly):", rendered]
    return "\n".join(parts)


def build_repair_prompt(failures: list[str], language: str, entrypoint: str) -> str:
    """Ask for a fix, quoting the concrete failures the local grader found.

    The failures come from running the candidate through the validator's own
    executor, so this is real evidence rather than a vague 'try again' — which
    is the difference between a repair loop that converges and one that drifts.
    """
    detail = "\n".join(f"  - {line}" for line in failures)
    target = "the program" if language == "rust" else f"`{entrypoint}`"
    return (
        f"Your solution is WRONG. I ran {target} against the examples and got:\n"
        f"{detail}\n\n"
        "Work out why, then reply with ONLY ONE corrected code block. "
        "Same rules as before. Do not explain outside the code block."
    )


def extract_code(reply: str) -> str:
    """Pull the source out of a model reply.

    The DOM reader already returns a code block's inner text for ChatGPT, but
    a reply can still arrive fenced (or as prose). Prefer the LAST fenced
    block — models often show a wrong first draft before the final answer.
    """
    if not reply:
        return ""
    matches = _FENCE_RE.findall(reply)
    if matches:
        return matches[-1].strip()
    return reply.strip()


def python_defect(code: str, entrypoint: str) -> Optional[str]:
    """Return a reason string if the source can't possibly be graded, else None.

    Catches the two failure modes that make a reply worthless before it is even
    executed: it isn't Python at all (prose, a refusal, a truncated stream), or
    it never defines the function the validator is going to call.
    """
    if not code.strip():
        return "the reply contained no code"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"the code is not valid Python ({exc.msg}, line {exc.lineno})"
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
        return "the reply contained no code"
    if "fn main" not in code:
        return "the program does not define `fn main()`"
    return None

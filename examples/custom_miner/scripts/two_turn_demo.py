"""See the two-turn split without a browser, a wallet or a chain.

`try_solver.py` next door needs a signed-in Chrome and spends a real solve to
tell you anything. This needs neither: the backend is a scripted fake that
returns whatever replies each scenario lists, so every branch of the split --
including the ones a live run almost never reaches -- is reachable in under a
second. What it CANNOT tell you is whether a real model obeys the prompts; for
that, `try_solver.py` and `solvers.rehearse --examples 0`.

    python3 scripts/two_turn_demo.py                    # every branch
    python3 scripts/two_turn_demo.py --show cases       # turn 1, verbatim
    python3 scripts/two_turn_demo.py --show code        # turn 2, with <must_pass>
    python3 scripts/two_turn_demo.py --show code-bare   # turn 2 with no cases to clear
    python3 scripts/two_turn_demo.py --lang rust --show cases

Reading the two prompts by eye is the half no test can do for you. Turn 1 must
ask for ONE json block and no program; turn 2 for ONE code block, with the
model's own cases echoed under <must_pass> and no mention of a second block
anywhere in it. Every turn this miner sends asks for exactly one block --
there is no combined prompt to fall back to.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent          # examples/custom_miner
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parent.parent))            # the repo root, for `rlvr`

from custom_miner import SolveTask  # noqa: E402
from solvers.prompts import build_code_prompt, build_tests_prompt  # noqa: E402
from solvers.verify import VerifyingSolver  # noqa: E402

STATEMENT = "Return the sum of the decimal digits of a non-negative integer n."

# The classic off-by-one: right for 12345, wrong for 0 and every single digit.
RIGHT = ("```python\ndef g(n):\n    s = 0\n    while n > 0:\n"
         "        s += n % 10\n        n //= 10\n    return s\n```")
WRONG = RIGHT.replace("while n > 0", "while n > 9")

CASES = ('```json\n[{"name": "zero", "args": [0], "expected": 0},\n'
         ' {"name": "one digit", "args": [7], "expected": 7},\n'
         ' {"name": "carry", "args": [12345], "expected": 15}]\n```')
# A case that is ITSELF wrong. No program can pass it, so it is what the repair
# round's "fix the case and leave the program alone" escape hatch exists for.
BAD_CASES = CASES.replace('"expected": 0}', '"expected": 99}')


def drive(deadline: float, replies: list[str], **kw) -> None:
    """Run one solve against a backend that replies from a script."""
    seen: list[tuple] = []

    class Chat:
        provider = "claude"

        def __init__(self) -> None:
            self.n = -1

        async def send(self, text, timeout_s, extend_to_s=None):
            self.n += 1
            seen.append((text, timeout_s, extend_to_s))
            return replies[min(self.n, len(replies) - 1)]

        async def close(self) -> None:
            pass

    class Fleet:
        async def open(self, avoid=None):
            return Chat()

        async def aclose(self) -> None:
            pass

        def stats(self) -> dict:
            return {}

    task = SolveTask(
        problem_id="demo", language="python", statement=STATEMENT,
        entrypoint="g", public_examples=[], deadline_s=deadline,
    )
    answer = asyncio.run(
        VerifyingSolver(Fleet(), **kw).solve_task(task, timeout_s=deadline)
    )

    def kind(text: str) -> str:
        # `<task>` is turn 1's; `<contract>` is turn 2's. A repair has neither:
        # it carries the failure and nothing else, because it is sent into the
        # conversation that already holds both.
        if "<task>" in text:
            return "CASES"
        return "CODE" if "<contract>" in text else "REPAIR"

    print(f"  turns   {[kind(t) for t, _, _ in seen]}")
    print(f"  slices  {[round(s, 1) for _, s, _ in seen]}s"
          f"   what each read is allocated")
    print(f"  caps    {[None if c is None else round(c, 1) for _, _, c in seen]}s"
          f"   None on turn 1: nothing is held back, so nothing to extend into")
    # What each turn ASKED for, which is the half no test can check for you: a
    # repair may legitimately ask for a corrected `json` block beside the
    # program, and nothing else here ever may.
    print(f"  asks    "
          f"{['ONE block' if 'second `json` block' not in t else 'program + fixed cases' for t, _, _ in seen]}")
    print(f"  answer  "
          f"{'CORRECT' if 'while n > 0' in (answer.code or '') else 'WRONG or EMPTY'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two_turn_demo.py")
    parser.add_argument(
        "--show", choices=["cases", "code", "code-bare"],
        help="print one prompt verbatim instead of running the scenarios",
    )
    parser.add_argument("--lang", default="python", choices=["python", "rust"])
    args = parser.parse_args(argv)
    entry = "main" if args.lang == "rust" else "g"

    if args.show == "cases":
        print(build_tests_prompt(args.lang, STATEMENT, entry, []))
        return 0
    if args.show == "code":
        print(build_code_prompt(
            args.lang, STATEMENT, entry, [],
            cases=[{"name": "zero", "args": [0], "expected": 0},
                   {"name": "carry", "args": [12345], "expected": 15}],
        ))
        return 0
    if args.show == "code-bare":
        print(build_code_prompt(args.lang, STATEMENT, entry, []))
        return 0

    for title, deadline, replies, kw in (
        ("300s — two turns, the model gets its cases right",
         300.0, [CASES, RIGHT], {}),
        ("300s — the model's OWN case catches the model's bug",
         300.0, [CASES, WRONG, RIGHT], {}),
        ("300s — turn 1 wrote a BAD case; the repair round corrects it",
         300.0, [BAD_CASES, RIGHT, RIGHT + "\n\n" + CASES, RIGHT], {}),
        ("300s — turn 1 came back useless; turn 2 still goes out",
         300.0, ["Happy to help!", RIGHT], {}),
        ("60s — a short deadline splits too; turn 1 costs what it took",
         60.0, [CASES, RIGHT], {}),
        ("300s — SOLVER_SELF_TESTS=0: one turn, and the answer goes out ungraded",
         300.0, [RIGHT], {"self_tests": False}),
    ):
        print(f"\n{title}")
        drive(deadline, replies, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

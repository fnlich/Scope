#!/usr/bin/env python3
"""Solve one task with the configured backends. No chain, no wallet, no signing.

    cd examples/custom_miner
    python scripts/try_solver.py
    python scripts/try_solver.py --statement "Return n factorial." --entrypoint fact \
                                --example '{"args": [5], "expected": 120}'

This is the layer worth testing first, because it is where nearly all the risk
lives: attaching to your browser, it still being signed in, the selectors
matching, the model answering, and the self-verification loop grading that
answer with the validator's own executor.

Deliberately free of the chain dependencies -- it never imports bittensor or
``custom_miner`` -- so it runs before you have a wallet, and a failure here can
only be the solver.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent.parent          # examples/custom_miner
sys.path.insert(0, str(_HERE))
# ...and the repo root, so `rlvr` imports whether or not the package was pip
# installed and whatever directory this is run from.
sys.path.insert(1, str(_HERE.parent.parent))

from preflight import require_linux  # noqa: E402

require_linux("The solver test")

from solvers.config import find_env_file, load_env_file  # noqa: E402
from solvers.roster import build_solver, warm_up  # noqa: E402

# A task the model should get right first try, with examples that catch the
# classic off-by-one. If this one does not verify, the problem is the setup.
DIGITS = dict(
    statement="Return the sum of the decimal digits of a non-negative integer n.",
    entrypoint="sum_of_digits",
    examples=[
        {"args": [12345], "kwargs": {}, "expected": 15},
        {"args": [0], "kwargs": {}, "expected": 0},
        {"args": [999], "kwargs": {}, "expected": 27},
    ],
)


@dataclass
class Task:
    """The fields a solver reads. Structurally what the miner passes it."""

    problem_id: str
    language: str
    statement: str
    entrypoint: str
    public_examples: list[dict[str, Any]] = field(default_factory=list)
    deadline_s: float = 300.0


async def main(args: argparse.Namespace) -> int:
    found = find_env_file()
    if load_env_file(found):
        print(f"[try] loaded {found}")

    examples = [json.loads(e) for e in args.example] if args.example else DIGITS["examples"]
    task = Task(
        problem_id="try-1",
        language=args.language,
        statement=args.statement or DIGITS["statement"],
        entrypoint=args.entrypoint or DIGITS["entrypoint"],
        public_examples=examples,
        deadline_s=float(args.timeout),
    )

    solver = build_solver()
    started = time.monotonic()
    if args.repeat > 1:
        # A verified answer is cached by statement, so repeating the same task
        # would serve the cache and never touch the browser -- the opposite of
        # what the flag is for. Zero is the documented "off": the store is
        # guarded by `if best.verified and self._cache_size`.
        solver._cache_size = 0
        solver._cache.clear()
    try:
        await warm_up(solver, 1)
        # `--repeat` exists to make the tab lifecycle visible. A one-shot run
        # cannot show it: any short-lived process opens a tab and closes it on
        # the way out, which looks identical to "a tab per task". Only the
        # SECOND task proves the tab was kept and merely given a new
        # conversation -- which is what the miner does for its whole life.
        for round_no in range(1, max(1, args.repeat) + 1):
            if args.repeat > 1:
                print(f"\n[try] --- task {round_no} of {args.repeat} ---")
                print("[try] watch the browser: same tab, new conversation")
            print(f"[try] solving: {task.statement!r} -> {task.entrypoint}()")
            answer = await solver.solve_task(task, task.deadline_s)
            if round_no < args.repeat:
                print(f"[try] task {round_no}: verified={answer.verified} "
                      f"({answer.passed}/{answer.total})")
        # Before aclose(), which drains the pool and would report idle=0.
        stats = solver.stats()
    finally:
        # Only now, at process exit, are the tabs closed -- because the whole
        # fleet is going away. A running miner never reaches this point.
        await solver.aclose()
    elapsed = time.monotonic() - started

    print("\n" + "=" * 68)
    print(answer.code or "(no code came back)")
    print("=" * 68)
    print(f"verified   {answer.verified}   ({answer.passed}/{answer.total} public examples)")
    print(f"elapsed    {elapsed:.1f}s")
    print(f"stats      {json.dumps(stats, indent=2)}")

    if answer.verified:
        print("\n[try] PASS — the solver works end to end.")
        return 0
    if answer.code.strip():
        print(
            "\n[try] PARTIAL — code came back but did not reproduce every example.\n"
            "      The plumbing works; the model got it wrong. Re-run before"
            " blaming the setup."
        )
        return 1
    print(
        "\n[try] FAIL — nothing came back. In order of likelihood:\n"
        "      1. the browser is not signed in   -> sign in to the provider in it\n"
        "      2. a selector no longer matches   -> python -m solvers.doctor <backend> --probe\n"
        "      3. the deadline was too short     -> --timeout 300"
    )
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="try_solver.py")
    parser.add_argument("--language", default="python", choices=("python", "rust"))
    parser.add_argument("--statement", default="")
    parser.add_argument("--entrypoint", default="")
    parser.add_argument(
        "--example", action="append",
        help='JSON case, repeatable: \'{"args": [5], "expected": 120}\'',
    )
    parser.add_argument("--timeout", default=180, type=float)
    parser.add_argument(
        "--repeat", default=1, type=int,
        help="solve the same task N times in one process. Use it to watch the "
        "tab being REUSED: one tab, N conversations, closed only at exit.",
    )
    try:
        sys.exit(asyncio.run(main(parser.parse_args())))
    except KeyboardInterrupt:
        sys.exit(130)

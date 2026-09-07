"""Read the paired runs in `bar-ab/` and print what the split cost and bought.

The measurement behind `SOLVER_INDEPENDENT_BAR`. Each sample challenge is
solved twice -- once with the cases turn one turn ahead of the program in the
same conversation, once with it written beside the program in its own -- shown
NO public examples and graded on all of them, which is the shape live traffic
has. Reproduce with:

    for MODE in 0 1; do
      for C in asset-rebuild-planner extent-journal reactive-stat-board \\
               revocable-verification-gate sparse-circular-array; do
        SOLVER_BACKEND=cli SOLVER_INDEPENDENT_BAR=$MODE \\
          python -m solvers.rehearse --challenge $C --examples 0 \\
          > calibration/bar-ab/$([ $MODE = 0 ] && echo sequential || echo split)-$C.log 2>&1
      done
    done

Read what it prints for what it is. The TIMES are five paired measurements.
The GRADED column is two problems of three cases each -- the other three are
Rust, which needs Docker to grade, and the box this ran on had none -- so it
is a smoke test that the split did not break correctness, not evidence that it
improved it. The evidence that the shared bar was catching nothing is in
`logs-2026-09-05*.log`: 96 of 97 solves shipped a program that passed every one
of its own cases, against a hidden-suite pass rate of 78-83%.
"""

from __future__ import annotations

import collections
import glob
import os
import re
import statistics as st

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bar-ab")

PHASE = re.compile(r"\[phase\] (.+?)\s+\d\d:\d\d:\d\d\.\d took\s+([\d.]+)s")
DONE = re.compile(
    r"\[verify\] (?:python|rust) entrypoint=\S+.*?examples=\d+/\d+\s+"
    r"(?:self=(\d+)/(\d+)\s+)?.*?rounds=(\d+) corrected=\d+/\d+.*?([\d.]+)s/"
)
SCORE = re.compile(r"\[rehearse\] SCORES: (?:passed all (\d+)|(\d+) of (\d+))")
SEAT = re.compile(r"\[rehearse\] seat: (\d+) turn\(s\), ([\d,]+) output tokens")


def _score(text):
    m = SCORE.search(text)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1)), int(m.group(1))
    return int(m.group(2)), int(m.group(3))


def read(path: str) -> dict:
    text = open(path).read()
    done, seat = DONE.search(text), SEAT.search(text)
    phases: dict[str, list[float]] = {}
    for label, spent in PHASE.findall(text):
        phases.setdefault(re.sub(r"^open .*", "open", label.strip()), []).append(
            float(spent)
        )
    return {
        "score": _score(text),
        "own": (int(done.group(1)), int(done.group(2)))
        if done and done.group(1)
        else None,
        "rounds": int(done.group(3)) if done else 0,
        "elapsed": float(done.group(4)) if done else 0.0,
        "turns": int(seat.group(1)) if seat else 0,
        "tokens": int(seat.group(2).replace(",", "")) if seat else 0,
        "cases": sum(phases.get("1 cases", [0.0])),
        "program": sum(phases.get("2 program", [0.0])),
    }


def main() -> None:
    runs: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(HERE, "*.log"))):
        mode, _, name = os.path.basename(path)[:-4].partition("-")
        runs[name][mode] = read(path)

    head = (
        f"{'challenge':30s} {'mode':10s} {'GRADED':>7s} {'own':>7s} {'rnds':>4s} "
        f"{'cases':>7s} {'prog':>7s} {'solve':>7s} {'turns':>5s} {'out tok':>8s}"
    )
    print(head)
    print("-" * len(head))
    totals: dict[str, list] = {"sequential": [[], [], 0, 0], "split": [[], [], 0, 0]}
    for name in sorted(runs):
        for mode in ("sequential", "split"):
            r = runs[name].get(mode)
            if r is None:
                continue
            g = f"{r['score'][0]}/{r['score'][1]}" if r["score"] else "-"
            o = f"{r['own'][0]}/{r['own'][1]}" if r["own"] else "-"
            print(
                f"{name:30s} {mode:10s} {g:>7s} {o:>7s} {r['rounds']:4d} "
                f"{r['cases']:7.1f} {r['program']:7.1f} {r['elapsed']:7.1f} "
                f"{r['turns']:5d} {r['tokens']:8,d}"
            )
            totals[mode][0].append(r["elapsed"])
            totals[mode][1].append(r["tokens"])
            if r["score"]:
                totals[mode][2] += 1
                totals[mode][3] += r["score"][0] == r["score"][1]
        print()
    print("-" * len(head))
    for mode, (times, tokens, graded, right) in totals.items():
        if not times:
            continue
        print(
            f"{mode:10s} median solve {st.median(times):5.0f}s   "
            f"max {max(times):5.0f}s   output tokens {sum(tokens):7,d}   "
            f"fully correct {right}/{graded} (the two that can be graded here)"
        )


if __name__ == "__main__":
    main()

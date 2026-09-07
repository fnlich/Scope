"""Run the ladder over an archived corpus and report what it learns.

Offline, before going live, and again after every change to a rung. It answers
the questions the design says to answer without hidden verdicts:

  * how often a real answer clears each rung, which is the false-positive rate
    of that rung -- these are answers the miner actually shipped, so a rung
    that rejects many of them is too strict, not vigilant;
  * where the seconds go, so `ladder.CAPS` is measured rather than guessed;
  * the shape of the corpus itself, so a schedule can be built on it.

Usage:
    python -m calibration.corpus_eval --corpus /path/to/hone-examples
    python -m calibration.corpus_eval --corpus ... --limit 20 --verbose
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ladder as _ladder  # noqa: E402
from pipeline.ladder import Candidate, Ladder  # noqa: E402


def records(corpus: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Every archived exchange under `corpus`, oldest path first."""
    paths: list[str] = []
    for sub in ("solutions", "solutions-2", "."):
        paths.extend(sorted(glob.glob(os.path.join(corpus, sub, "*.json"))))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path) as handle:
                blob = json.load(handle)
        except Exception:  # noqa: BLE001 - a bad file is skipped, not fatal
            continue
        request = blob.get("request") or {}
        response = blob.get("response") or {}
        if not request.get("statement") or not response.get("code"):
            continue
        out.append({
            "id": blob.get("problem_id") or os.path.basename(path)[:12],
            "language": request.get("language") or "python",
            "entrypoint": request.get("entrypoint") or "",
            "statement": request.get("statement") or "",
            "code": response.get("code") or "",
        })
        if limit and len(out) >= limit:
            break
    return out


def _grader():
    """The miner's own grader, or None when its executor cannot be built."""
    try:
        from solvers.verify import _Grader
        return _Grader()
    except Exception as exc:  # noqa: BLE001
        print(f"[calib] no grader ({type(exc).__name__}: {exc}); "
              f"rungs that run code will be skipped")
        return None


def _rust_compile():
    try:
        from solvers.rust_compile import compile_defect, rustc_path
        if rustc_path() is None:
            print("[calib] no rustc; the build rung will be skipped for Rust")
            return None
        return compile_defect
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        help="a checkout of the archive (expects solutions/ and solutions-2/)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="", help="write the learned numbers here as JSON")
    args = parser.parse_args(argv)

    rows = records(args.corpus, args.limit or None)
    if not rows:
        print(f"[calib] no usable records under {args.corpus}")
        return 1

    run = Ladder(_grader(), _rust_compile())
    langs = collections.Counter(r["language"] for r in rows)
    print(f"[calib] {len(rows)} record(s): " +
          ", ".join(f"{v} {k}" for k, v in sorted(langs.items())))

    step_seconds: dict[str, list[float]] = collections.defaultdict(list)
    failures: collections.Counter = collections.Counter()
    rules: collections.Counter = collections.Counter()
    green = 0
    started = time.monotonic()

    for row in rows:
        candidate = Candidate(row["code"], row["language"], row["entrypoint"])
        verdict = run.verify(candidate, None, left=_ladder.ROUND_S)
        for step in verdict.steps:
            step_seconds[step.name].append(step.seconds)
        if verdict.green:
            green += 1
        else:
            failures[verdict.failure.value] += 1
            rules[verdict.report[:70]] += 1
            if args.verbose:
                print(f"  {row['id'][:12]} {row['language']:6} "
                      f"{verdict.failure.value:9} {verdict.report[:90]}")

    elapsed = time.monotonic() - started
    print(f"\n[calib] cleared every rung that could run: {green}/{len(rows)} "
          f"({green / len(rows) * 100:.0f}%) in {elapsed:.1f}s")

    if failures:
        print("[calib] failure classes:")
        for name, count in failures.most_common():
            print(f"    {name:10} {count:>4}")
        print("[calib] what the rung actually said:")
        for text, count in rules.most_common(12):
            print(f"    {count:>4}x {text}")

    learned: dict[str, float] = {}
    print("\n[calib] seconds per rung (this is what CAPS should be built from):")
    print(f"    {'rung':<16}{'n':>5}{'median':>9}{'p90':>8}{'max':>8}")
    for name, values in sorted(step_seconds.items()):
        values.sort()
        p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
        learned[f"ladder_{name.replace(' ', '_').replace('-', '_')}"] = round(p90, 3)
        print(f"    {name:<16}{len(values):>5}{st.median(values):>9.3f}"
              f"{p90:>8.3f}{values[-1]:>8.3f}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(learned, handle, indent=2, sort_keys=True)
        print(f"\n[calib] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Derive every latency constant from production logs. Nothing else may set one.

The pipeline's gates are only as good as the numbers under them, and a number
that came from reasoning rather than a log is indistinguishable from a number
that came from a log once it is written down as a float. So this is the only
thing allowed to produce them: it reads miner logs, counts, and writes
`latency.json` with the observation count and the source line pattern beside
every value. `pipeline.latency` refuses to gate on an entry that has neither.

Two different costs are measured, because they answer different questions:

  TURN   what one successful model call costs, from the `[cli] ... ok:` lines.
         Indexed by output size, because that is what the logs show it depends
         on -- a turn under 500 characters is a different animal from one over
         8000.

  PHASE  what the solve actually PAYS to get that answer, from the `[phase]`
         lines. Higher than the turn cost, because a phase absorbs the turns
         that were cut, hopped and retried before one answered. Gates are
         built on this one; a gate built on the turn cost schedules a call the
         solve cannot afford.

Usage:
    python -m calibration.measure_logs LOG [LOG ...] --out pipeline/latency.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from typing import Any, Optional

TURN = re.compile(
    r"\[cli\] cli:(\w+)\S* (?:ok|partial): \d+ event\(s\), "
    r"first text after (\d+)s, (\d+) character\(s\) in (\d+)s"
)
PHASE = re.compile(
    r"\[phase\] (?:pass \d+ )?(\S+(?: \S+)?)\s+\d\d:\d\d:\d\d\.\d took\s+[\d.]+s\s+\(model ([\d.]+)s"
)
# Output-size buckets, in characters. Chosen where the logs already separate:
# a register is a short list, a solution is a program, a kit is two programs.
BUCKETS = [(0, 500, "tiny"), (500, 1500, "short"), (1500, 4000, "medium"),
           (4000, 8000, "large"), (8000, 10 ** 9, "huge")]


def _q(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(p * len(ordered)))])


def _stat(values: list[float], source: str) -> dict[str, Any]:
    return {
        "p50": round(st.median(values), 1),
        "p90": round(_q(values, 0.90), 1),
        "p95": round(_q(values, 0.95), 1),
        "max": round(max(values), 1),
        "n": len(values),
        "source": source,
    }


def measure(paths: list[str]) -> dict[str, Any]:
    turns: list[tuple[str, int, int, int]] = []
    phases: dict[str, list[float]] = {}
    for path in paths:
        with open(path, errors="replace") as handle:
            for line in handle:
                found = TURN.search(line)
                if found:
                    turns.append((found.group(1), int(found.group(2)),
                                  int(found.group(3)), int(found.group(4))))
                    continue
                found = PHASE.search(line)
                if found:
                    label = found.group(1)
                    key = ("cases" if label.startswith("1 cases")
                           else "program" if label.startswith("2 program")
                           else "correction" if "correction" in label
                           else "cross-check" if "cross-check" in label else None)
                    if key:
                        phases.setdefault(key, []).append(float(found.group(2)))

    out: dict[str, Any] = {"logs": paths, "turns": len(turns)}

    by_bucket: dict[str, dict[str, Any]] = {}
    for low, high, name in BUCKETS:
        seconds = [t[3] for t in turns if low <= t[2] < high]
        if seconds:
            by_bucket[name] = _stat(seconds, f"[cli] turns with {low}-{high} output chars")
    out["turn_by_output_size"] = by_bucket

    first_text = [t[1] for t in turns]
    if first_text:
        out["first_text"] = _stat(first_text, "[cli] first text after Ns, all turns")
    rates = [t[2] / max(1, t[3] - t[1]) for t in turns if t[3] - t[1] >= 1]
    if rates:
        ordered = sorted(rates)
        out["write_chars_per_s"] = {
            "p50": round(st.median(rates), 1),
            "p10": round(ordered[int(0.10 * len(ordered))], 1),
            "p05": round(ordered[int(0.05 * len(ordered))], 1),
            "n": len(rates),
            "source": "[cli] chars / (total - first text)",
        }

    out["phase"] = {name: _stat(values, f"[phase] {name}, model seconds")
                    for name, values in sorted(phases.items()) if values}

    sizes: dict[str, dict[str, Any]] = {}
    for name in ("cases", "program", "correction"):
        chars = [t[2] for t in turns]
        if name == "program" and chars:
            sizes[name] = _stat([float(c) for c in chars], "[cli] output chars, all turns")
    out["output_chars_all_turns"] = _stat([float(t[2]) for t in turns],
                                          "[cli] output chars") if turns else {}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    table = measure(args.logs)
    if not table.get("turns"):
        print("[measure] no turn lines found; is this a miner log?")
        return 1

    print(f"[measure] {table['turns']} turn(s) from {len(args.logs)} log(s)")
    print("\n  what ONE SUCCESSFUL CALL costs, by output size:")
    print(f"    {'bucket':<10}{'n':>5}{'p50':>7}{'p90':>7}{'p95':>7}{'max':>7}")
    for name, row in table["turn_by_output_size"].items():
        print(f"    {name:<10}{row['n']:>5}{row['p50']:>7.0f}{row['p90']:>7.0f}"
              f"{row['p95']:>7.0f}{row['max']:>7.0f}")
    print("\n  what the SOLVE PAYS per phase (includes cut, hopped and retried turns):")
    print(f"    {'phase':<14}{'n':>5}{'p50':>7}{'p90':>7}{'p95':>7}{'max':>7}")
    for name, row in table["phase"].items():
        print(f"    {name:<14}{row['n']:>5}{row['p50']:>7.0f}{row['p90']:>7.0f}"
              f"{row['p95']:>7.0f}{row['max']:>7.0f}")
    ft = table.get("first_text", {})
    if ft:
        print(f"\n  first text: p50 {ft['p50']:.0f}s  p90 {ft['p90']:.0f}s  "
              f"p95 {ft['p95']:.0f}s  max {ft['max']:.0f}s  (n={ft['n']})")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
        print(f"\n[measure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

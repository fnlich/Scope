#!/usr/bin/env python
"""Measure Docker grading throughput with synthetic Python submissions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import statistics
import time

from rlvr.config import get_settings
from rlvr.execution.docker_executor import DockerExecutor
from rlvr.types import TestCase


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _grade_one(
    executor: DockerExecutor,
    code: str,
    tests: list[TestCase],
    timeout_s: float,
) -> tuple[float, int]:
    started = time.monotonic()
    results = executor.run_tests(code, "solve", tests, timeout_s)
    elapsed = time.monotonic() - started
    failures = sum(not result.passed for result in results)
    return elapsed, failures


def _run_challenge(
    executor: DockerExecutor,
    submissions: int,
    tests: list[TestCase],
    concurrency: int,
    timeout_s: float,
) -> dict[str, float | int]:
    code = "def solve(value):\n    return value * value\n"
    started = time.monotonic()
    submission_times: list[float] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_grade_one, executor, code, tests, timeout_s)
            for _ in range(submissions)
        ]
        for future in as_completed(futures):
            elapsed, failed = future.result()
            submission_times.append(elapsed)
            failures += failed
    wall_s = time.monotonic() - started
    return {
        "wall_s": wall_s,
        "submission_p50_s": statistics.median(submission_times),
        "submission_p95_s": _percentile(submission_times, 0.95),
        "submission_max_s": max(submission_times),
        "submissions_per_s": submissions / wall_s,
        "test_processes_per_s": submissions * len(tests) / wall_s,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=int, default=123)
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if min(args.submissions, args.tests, args.concurrency, args.repeats) < 1:
        parser.error("count arguments must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        parser.error("--timeout-s must be finite and positive")

    settings = get_settings()
    executor = DockerExecutor(settings)
    tests = [TestCase(args=[index], expected=index * index) for index in range(args.tests)]

    _grade_one(executor, "def solve(value): return value * value", tests, args.timeout_s)
    runs = [
        _run_challenge(
            executor,
            args.submissions,
            tests,
            args.concurrency,
            args.timeout_s,
        )
        for _ in range(args.repeats)
    ]
    walls = [float(run["wall_s"]) for run in runs]
    output = {
        "configuration": {
            "submissions": args.submissions,
            "tests_per_submission": args.tests,
            "concurrency": args.concurrency,
            "repeats": args.repeats,
            "timeout_s": args.timeout_s,
            "docker_image": executor.image,
            "docker_memory": executor.memory,
            "docker_cpus": executor.cpus,
            "docker_pids_limit": executor.pids_limit,
        },
        "challenge_wall_s": {
            "p50": statistics.median(walls),
            "p95": _percentile(walls, 0.95),
            "max": max(walls),
        },
        "runs": runs,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if any(int(run["failures"]) for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

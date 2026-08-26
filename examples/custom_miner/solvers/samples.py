"""Problems to rehearse against when there is no validator to ask.

Deliberately not toys. `solvers.doctor --probe` already asks for sixty comment
lines and a `return 'pong'`, and every model alive answers that -- which is the
point of it, because it is testing the SELECTORS. A rehearsal is testing
whether this browser and this account solve the kind of thing the subnet
actually sends, so a sample that any model answers on the first try would say
nothing about a miner that is about to score zero.

So each of these has the properties the real ones have, and each is here
because it catches something different:

* An edge case the statement states and the examples do not cover, so a model
  that skims the examples and ignores the prose fails on the hidden suite while
  passing everything it was shown. That is the single most common shape of a
  wrong answer, and it is invisible to a local check against public examples.
* Enough arithmetic to need thinking. On claude.ai the thinking is rendered
  INSIDE the element the assistant selector matches, so a problem that provokes
  none never exercises the reader's hardest case.
* A hidden suite that is genuinely hidden: the public examples are a strict,
  small subset, so "passed the examples" and "would have scored" are different
  answers and the rehearsal can tell them apart.

The Rust one is not the Python one translated. Rust answers reach the grader
through stdin and stdout rather than through a function call, and the miner has
lost whole answers to that difference alone -- so the rehearsal has to put a
real one through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlvr.protocol import TaskRequest
from rlvr.types import TestCase


@dataclass(frozen=True)
class Sample:
    problem_id: str
    language: str
    statement: str
    entrypoint: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    hidden: list[dict[str, Any]] = field(default_factory=list)
    deadline_s: float = 300.0

    def request(self) -> TaskRequest:
        return TaskRequest(
            problem_id=self.problem_id,
            language=self.language,
            statement=self.statement,
            entrypoint=self.entrypoint,
            public_examples=[TestCase(**case) for case in self.examples],
            deadline_s=self.deadline_s,
        )

    def hidden_tests(self) -> list[TestCase]:
        """The examples AND the cases the model never saw.

        Both, because a validator grades the complete suite and the interesting
        failure is an answer that passes what it was shown and fails what it
        was not.
        """
        return [TestCase(**case) for case in (self.examples + self.hidden)]


PYTHON = Sample(
    problem_id="rehearsal-python-1",
    language="python",
    entrypoint="longest_run",
    statement="""\
Given a list of integers `values`, return the length of the longest run of
consecutive EQUAL elements.

Rules:
  * An empty list has a longest run of 0.
  * A list of one element has a longest run of 1.
  * Elements are compared by equality, not by ordering, so the run does not
    have to be increasing or decreasing -- only equal.
  * `True` and `1` are equal in Python and count as the same element; so do
    `False` and `0`.

Return the length as an int.""",
    # The examples show only the ordinary case. Everything the statement
    # promises about the boundaries is left to the hidden suite, which is
    # exactly the trap a real problem sets.
    examples=[
        {"args": [[1, 1, 2, 2, 2, 3]], "kwargs": {}, "expected": 3},
        {"args": [[5, 4, 3, 2, 1]], "kwargs": {}, "expected": 1},
    ],
    hidden=[
        {"args": [[]], "kwargs": {}, "expected": 0},
        {"args": [[7]], "kwargs": {}, "expected": 1},
        {"args": [[2, 2, 2, 2]], "kwargs": {}, "expected": 4},
        {"args": [[1, True, 1, 0, False]], "kwargs": {}, "expected": 3},
        {"args": [[0, 0, 1, 1, 1, 0, 0, 0, 0]], "kwargs": {}, "expected": 4},
        {"args": [[-1, -1, -1, 2]], "kwargs": {}, "expected": 3},
    ],
)

RUST = Sample(
    problem_id="rehearsal-rust-1",
    language="rust",
    entrypoint="main",
    statement="""\
Read from standard input and write to standard output.

The first line holds an integer N. The second line holds N integers separated
by single spaces.

Print the length of the longest run of consecutive EQUAL integers, followed by
a newline.

Rules:
  * If N is 0, the second line is empty or absent, and the answer is 0.
  * The integers fit in i64. They may be negative.
  * Print nothing but the number and a trailing newline.""",
    examples=[
        {"args": ["6\n1 1 2 2 2 3\n"], "kwargs": {}, "expected": "3\n"},
        {"args": ["5\n5 4 3 2 1\n"], "kwargs": {}, "expected": "1\n"},
    ],
    hidden=[
        {"args": ["0\n\n"], "kwargs": {}, "expected": "0\n"},
        {"args": ["1\n7\n"], "kwargs": {}, "expected": "1\n"},
        {"args": ["4\n2 2 2 2\n"], "kwargs": {}, "expected": "4\n"},
        {"args": ["9\n0 0 1 1 1 0 0 0 0\n"], "kwargs": {}, "expected": "4\n"},
        # i64, not i32: an answer that reaches for `i32` overflows silently in
        # release mode, which is the exact failure `rust_compile` was written
        # for and which no example here would otherwise provoke.
        {"args": ["3\n3000000000 3000000000 1\n"], "kwargs": {}, "expected": "2\n"},
    ],
)

SAMPLES: dict[str, Sample] = {"python": PYTHON, "rust": RUST}

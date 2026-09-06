"""Tests for the sequential pipeline's model-free half."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import compare  # noqa: E402
from pipeline.clock import (  # noqa: E402
    Action, Clock, State, decide,
)
from pipeline.ladder import (  # noqa: E402
    Candidate, Fail, Kit, Ladder, static_defect,
)


# --------------------------------------------------------------------------- #
# The static rung, tuned against 177 archived answers.
# --------------------------------------------------------------------------- #
def test_a_function_answer_may_import_sys_because_real_ones_do():
    """Calibration over the corpus rejected ten shipped Python solutions for
    `import sys`, which they use for `setrecursionlimit` -- a 185-line median
    solution recurses. Stdlib is allowed; what performs I/O is not."""
    assert static_defect(Candidate("import sys\ndef g(n):\n    return n", "python", "g")) is None
    assert static_defect(Candidate(
        "import sys\ndef g(n):\n    sys.setrecursionlimit(10000)\n    return n", "python", "g")) is None
    # ...but the attributes that actually do I/O are still refused.
    bad = static_defect(Candidate("import sys\ndef g(n):\n    return sys.stdin.read()", "python", "g"))
    assert bad and "sys.stdin" in bad


def test_the_static_rung_refuses_what_will_not_run_where_it_is_graded():
    for code, want in (
        ("", "no code"),
        ("def h():\n    pass", "entrypoint"),
        ("import numpy\ndef g(n):\n    return n", "standard library"),
        ("def g(n):\n    return input()", "may not call"),
        ("def g(n):\n    return (", "not valid Python"),
    ):
        defect = static_defect(Candidate(code, "python", "g"))
        assert defect and want in defect, (code, defect)
    assert static_defect(Candidate("fn helper() {}", "rust", "main")) is not None
    assert static_defect(Candidate("fn main() {}", "rust", "main")) is None


def test_a_truncated_answer_is_caught_for_free():
    """Two of 177 archived answers were shipped cut off mid-file -- certain
    zeros. Both are caught by a rung that makes no model call."""
    python = "def g(n):\n    total = 0\n    for c in"
    assert static_defect(Candidate(python, "python", "g")) is not None


# --------------------------------------------------------------------------- #
# Comparison.
# --------------------------------------------------------------------------- #
def test_program_mode_compares_whitespace_tokens_and_nothing_else():
    assert compare.same_program("1 2 3", "1\n2\t3\r\n")
    assert compare.same_program(" 1  2 ", "1 2")
    assert not compare.same_program("1 2", "1 3")
    assert not compare.same_program("12", "1 2")
    # A non-breaking space is content, not whitespace: the validator splits on
    # six bytes and so does this.
    assert not compare.same_program("1 2", "1 2")


def test_function_mode_does_not_confuse_a_bool_with_an_int():
    assert compare.same_function(1, 1)
    assert not compare.same_function(True, 1)
    assert not compare.same_function([1], (1,))


def test_the_probe_reports_what_only_the_sandbox_can_see():
    """Mutation and the real return type are invisible from outside the
    executor -- a value round-trips through JSON and a tuple arrives as a list."""
    source = compare.probe_source(
        "def f(a):\n    a.append(9)\n    return (1, 2)\n", "f")
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - the point of the test
    out = namespace[compare.PROBE_ENTRY]([1, 2])
    assert out[compare.PROBE_KEY] == 1
    assert out["mutated"] is True
    assert out["shape"] == "tuple"

    clean = compare.probe_source("def f(a):\n    return sum(a)\n", "f")
    namespace = {}
    exec(clean, namespace)  # noqa: S102
    out = namespace[compare.PROBE_ENTRY]([1, 2])
    assert out["mutated"] is False and out["shape"] == "int"


def test_an_integer_too_wide_for_the_declared_type_is_reported():
    source = compare.probe_source("def f():\n    return 1 << 70\n", "f")
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102
    assert namespace[compare.PROBE_ENTRY]()["max_int"] >= 1 << 63


def test_the_declared_return_shape_is_checked_but_an_unknown_one_is_not():
    assert compare.shape_matches("list", "list") is None
    assert compare.shape_matches("tuple", "list") is not None
    # The register is one model's reading; a shape it did not name is not a
    # failure, it is an absence of evidence.
    assert compare.shape_matches("", "list") is None
    assert compare.shape_matches("whatever", "list") is None


# --------------------------------------------------------------------------- #
# The clock.
# --------------------------------------------------------------------------- #
def _at(seconds: float) -> Clock:
    clock = Clock()
    clock.started -= seconds
    return clock


def test_the_answer_goes_out_at_the_hard_deadline_whatever_the_state():
    for state in State:
        assert decide(_at(281), state) is Action.REPORT


def test_a_green_verdict_buys_deep_verification_once_and_then_reports():
    assert decide(_at(200), State.VERIFIED_GREEN) is Action.DEEP_VERIFY
    assert decide(_at(200), State.VERIFIED_GREEN, have_deep=True) is Action.REPORT
    # ...and not even that when there is no room for it.
    assert decide(_at(279), State.VERIFIED_GREEN) is Action.REPORT


def test_a_fix_is_launched_only_while_one_could_still_land():
    assert decide(_at(100), State.VERIFIED_FAIL) is Action.LAUNCH_FIX
    # Past the Opus gate but inside the smaller model's.
    assert decide(_at(232), State.VERIFIED_FAIL) is Action.LAUNCH_FIX_SMALL
    assert decide(_at(250), State.VERIFIED_FAIL) is Action.REPORT
    # And never more than the cap allows.
    assert decide(_at(100), State.VERIFIED_FAIL, fix_rounds=2) is Action.REPORT


def test_the_kit_is_launched_only_when_a_fix_could_still_follow_it():
    """Evidence with no time to act on it buys nothing. The design's own
    `decide` has no branch for this at all and never launches the kit."""
    assert decide(_at(20), State.HAVE_SOLUTION) is Action.LAUNCH_KIT
    assert decide(_at(200), State.HAVE_SOLUTION) is Action.REPORT


def test_a_timed_out_solution_falls_back_to_a_smaller_model_while_one_fits():
    assert decide(_at(160), State.NO_SOLUTION, solution_attempts=2) is Action.LAUNCH_SOLUTION_SMALL
    assert decide(_at(250), State.NO_SOLUTION, solution_attempts=2) is Action.REPORT


# --------------------------------------------------------------------------- #
# The ladder degrades rather than refusing.
# --------------------------------------------------------------------------- #
class _NoGrader:
    """A grader whose executor cannot be built, which is the live case on a
    box with no Docker daemon."""

    def outputs(self, code, language, entrypoint, inputs, budget_s=None):
        raise RuntimeError("no executor here")


def test_with_no_kit_the_ladder_still_reports_what_it_established():
    verdict = Ladder(_NoGrader()).verify(
        Candidate("def g(n):\n    return n", "python", "g"), None, left=5.0)
    assert verdict.green, verdict.report
    # The evidence names the rungs that ran, so a short ladder cannot be
    # mistaken for a full one.
    assert "static" in verdict.evidence
    assert "differential" not in verdict.evidence


def test_a_static_failure_stops_the_round_before_anything_runs_code():
    verdict = Ladder(_NoGrader()).verify(
        Candidate("import numpy\ndef g(n):\n    return n", "python", "g"), None, left=5.0)
    assert not verdict.green and verdict.failure is Fail.STATIC
    assert verdict.failure.blames_solution
    assert len(verdict.steps) == 1, "nothing after static should have run"


def test_a_broken_kit_does_not_condemn_the_candidate():
    """A kit that fails its own self-test is the kit's problem. The ladder
    drops it and grades on what is left."""
    kit = Kit(generate=lambda seed, size: {"args": [seed]},
              validate=lambda case: False, reference="def g(n):\n    return n")
    verdict = Ladder(_NoGrader()).verify(
        Candidate("def g(n):\n    return n", "python", "g"), kit, left=5.0)
    names = [s.name for s in verdict.steps]
    assert "kit self-test" in names
    assert not [s for s in verdict.steps if s.name == "kit self-test"][0].passed


def test_a_mismatch_does_not_blame_the_solution_by_itself():
    """Two programs disagree and the ladder does not know which is wrong, so
    the class routes to arbitration, not to a patch of the candidate."""
    assert not Fail.MISMATCH.blames_solution
    for cls in (Fail.COMPILE, Fail.CRASH, Fail.MUTATION, Fail.SHAPE,
                Fail.OVERFLOW, Fail.TIMEOUT, Fail.STATIC):
        assert cls.blames_solution


def test_a_verdict_scores_so_a_patch_that_made_things_worse_is_refused():
    good = Ladder(_NoGrader()).verify(
        Candidate("def g(n):\n    return n", "python", "g"), None, left=5.0)
    bad = Ladder(_NoGrader()).verify(
        Candidate("def g(n):\n    return (", "python", "g"), None, left=5.0)
    assert good.score() > bad.score()


@pytest.mark.parametrize("language,code", [
    ("python", "def g(n):\n    return n"),
    ("rust", "fn main() { println!(\"1\"); }"),
])
def test_both_modes_reach_a_verdict_without_an_executor(language, code):
    verdict = Ladder(_NoGrader()).verify(Candidate(code, language, "g" if language == "python" else "main"),
                                         None, left=5.0)
    assert verdict.steps, "a round must always produce evidence"

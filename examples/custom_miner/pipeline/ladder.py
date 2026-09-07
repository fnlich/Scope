"""The verification ladder: everything that can be established without a model.

Ordered cheapest-first, and every rung can end the round. A candidate that
does not compile is not run against sixty inputs to find that out again, and
the class of the first failure is what routes the repair -- a crash goes to
the solution, a mismatch goes to arbitration, a kit that fails its own
self-test goes to the generator.

The ladder DEGRADES rather than refuses. With no test kit it still builds,
still reads the source statically, and still runs the stress input; a
candidate verified on that much is reported on that much, and `Verdict.evidence`
says which rungs actually ran so nothing downstream can mistake a short ladder
for a clean one.
"""

from __future__ import annotations

import ast
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from . import compare

# Per-rung caps. Measured on this corpus at roughly a fifth of each: the cap is
# what stops a rung from eating the round, not what it is expected to spend.
CAPS = {
    "static": 2.0,
    "build": 8.0,
    "kit_self_test": 4.0,
    "edge": 4.0,
    "differential": 6.0,
    "invariants": 3.0,
    "stress": 10.0,
}
ROUND_S = 10.0
DEEP_S = 5.0

# Python source that is disqualifying in function mode. The contract is a pure
# function: no I/O, nothing imported from outside the standard library.
_BANNED_CALLS = {"input", "open", "eval", "exec", "compile", "__import__"}
# I/O reached through a module rather than a bare call. `sys` itself is NOT
# banned: calibration over 177 archived answers rejected ten real, shipped
# Python solutions for `import sys`, which they use for `setrecursionlimit` on
# deep recursion -- a 185-line median solution needs it. What is banned is the
# handful of attributes that actually perform I/O.
_BANNED_ATTRS = {
    ("sys", "stdin"), ("sys", "stdout"), ("sys", "stderr"), ("sys", "argv"),
    ("os", "system"), ("os", "popen"), ("os", "environ"), ("os", "remove"),
    ("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
}
# Modules that are never in the standard library, so importing one means the
# answer will not run where it is graded. Detected against the interpreter's own
# list rather than a hand-written denylist, which goes stale in both directions.
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


class Fail(str, Enum):
    """The class of a failure, which is the only thing that routes a repair."""

    NONE = "none"
    COMPILE = "compile"
    STATIC = "static"
    CRASH = "crash"
    MUTATION = "mutation"
    SHAPE = "shape"
    OVERFLOW = "overflow"
    TIMEOUT = "timeout"
    MISMATCH = "mismatch"
    KIT = "kit"

    @property
    def blames_solution(self) -> bool:
        """Whether the candidate is what has to change.

        A MISMATCH does not appear here: two programs disagree and the ladder
        does not know which is wrong, so it goes to arbitration instead.
        """
        return self in {
            Fail.COMPILE, Fail.STATIC, Fail.CRASH, Fail.MUTATION,
            Fail.SHAPE, Fail.OVERFLOW, Fail.TIMEOUT,
        }


@dataclass
class Step:
    """One rung's outcome."""

    name: str
    passed: bool
    seconds: float = 0.0
    detail: str = ""
    failure: Fail = Fail.NONE
    ran: int = 0

    def __str__(self) -> str:
        mark = "ok " if self.passed else "FAIL"
        count = f" ({self.ran})" if self.ran else ""
        return f"{mark} {self.name}{count} {self.seconds:.2f}s{': ' + self.detail if self.detail else ''}"


@dataclass
class Verdict:
    """What the ladder established about one candidate."""

    green: bool = False
    failure: Fail = Fail.NONE
    steps: list[Step] = field(default_factory=list)
    mismatches: list[compare.Mismatch] = field(default_factory=list)
    report: str = ""

    @property
    def evidence(self) -> str:
        """Which rungs actually ran. A short ladder must not read as a clean one."""
        return ", ".join(s.name for s in self.steps if s.passed) or "nothing ran"

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def score(self) -> tuple:
        """Higher is better. Used to refuse a patch that made things worse."""
        return (1 if self.green else 0, sum(1 for s in self.steps if s.passed),
                -len(self.mismatches))


@dataclass
class Candidate:
    code: str
    language: str
    entrypoint: str


@dataclass
class Kit:
    """The generated test kit. Every field is optional: the ladder degrades."""

    generate: Optional[Callable[[int, int], Any]] = None
    validate: Optional[Callable[[Any], bool]] = None
    edge_cases: Optional[Callable[[], list]] = None
    check: Optional[Callable[[Any, Any], bool]] = None
    reference: str = ""
    source: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.reference) or self.generate is not None


# --------------------------------------------------------------------------- #


def static_defect(candidate: Candidate) -> Optional[str]:
    """What is wrong with the source before anything runs it, or None."""
    code = candidate.code
    if not code.strip():
        return "no code"
    if candidate.language == "rust":
        if "fn main" not in code:
            return "a Rust answer is a complete program and there is no `fn main`"
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"the code is not valid Python ({exc.msg}, line {exc.lineno})"
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if candidate.entrypoint and candidate.entrypoint not in names:
        return f"the entrypoint `{candidate.entrypoint}` is not defined"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                return f"a function answer may not call `{node.func.id}`"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in _BANNED_ATTRS:
                return f"a function answer may not touch `{node.value.id}.{node.attr}`"
        module = ""
        if isinstance(node, ast.Import):
            module = (node.names[0].name or "").split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = (node.module or "").split(".")[0]
        if module and _STDLIB and module not in _STDLIB:
            return f"a function answer may not import `{module}`; it is not in the standard library"
    return None


def _timed(name: str, cap: float, body: Callable[[], tuple[bool, str, Fail, int]]) -> Step:
    started = time.monotonic()
    try:
        passed, detail, failure, ran = body()
    except Exception as exc:  # noqa: BLE001 - a broken rung fails, never raises
        return Step(name, False, time.monotonic() - started,
                    f"{type(exc).__name__}: {exc}", Fail.KIT)
    return Step(name, passed, time.monotonic() - started, detail, failure, ran)


class Ladder:
    """Runs the rungs. `grader` is `solvers.verify._Grader` or anything with
    the same `outputs(code, language, entrypoint, inputs, budget_s)`."""

    def __init__(self, grader: Any, rust_compile: Any = None) -> None:
        self._grader = grader
        self._rust_compile = rust_compile

    # -- the rungs ---------------------------------------------------------- #

    def _static(self, candidate: Candidate) -> Step:
        def body():
            defect = static_defect(candidate)
            return (defect is None, defect or "", Fail.STATIC if defect else Fail.NONE, 0)
        return _timed("static", CAPS["static"], body)

    def _build(self, candidate: Candidate, left: float) -> Step:
        def body():
            if candidate.language != "rust" or self._rust_compile is None:
                return True, "not a Rust answer" if candidate.language != "rust" else "no compiler", Fail.NONE, 0
            defect = self._rust_compile(candidate.code, min(left, CAPS["build"]))
            return (defect is None, defect or "", Fail.COMPILE if defect else Fail.NONE, 0)
        return _timed("build", CAPS["build"], body)

    def _run(self, candidate: Candidate, cases: list[dict], left: float,
             probe: bool = False) -> list:
        code = candidate.code
        entry = candidate.entrypoint
        if probe and candidate.language != "rust":
            code = compare.probe_source(candidate.code, candidate.entrypoint)
            entry = compare.PROBE_ENTRY
        return self._grader.outputs(code, candidate.language, entry, cases, left)

    def _kit_self_test(self, kit: Kit, left: float, seeds: int = 40) -> Step:
        def body():
            if kit.generate is None or kit.validate is None:
                return True, "no generator to test", Fail.NONE, 0
            bad = 0
            ran = 0
            for seed in range(seeds):
                for size in (0, 1, 2, 3):
                    try:
                        case = kit.generate(seed, size)
                        ran += 1
                        if not kit.validate(case):
                            bad += 1
                    except Exception:  # noqa: BLE001
                        bad += 1
            if bad:
                return False, f"{bad} of {ran} generated input(s) failed the kit's own validate()", Fail.KIT, ran
            return True, "", Fail.NONE, ran
        return _timed("kit self-test", CAPS["kit_self_test"], body)

    def _cases_step(self, name: str, candidate: Candidate, kit: Kit,
                    cases: list[dict], left: float, cap: float) -> Step:
        """Run `cases` against candidate and reference and compare."""
        def body():
            if not cases:
                return True, "no cases", Fail.NONE, 0
            if not kit.reference:
                return True, "no reference to compare with", Fail.NONE, 0
            budget = min(left, cap)
            mine = self._run(candidate, cases, budget * 0.5, probe=True)
            theirs = self._grader.outputs(
                kit.reference, candidate.language, candidate.entrypoint, cases, budget * 0.5)
            bad = []
            for case, a, b in zip(cases, mine, theirs):
                got = compare.Probed.read(a)
                if not got.ok:
                    return False, f"the candidate did not run: {got.error}", (
                        Fail.TIMEOUT if getattr(a, "timed_out", False) else Fail.CRASH), len(cases)
                if not getattr(b, "ok", False):
                    continue  # the reference is what failed; not the candidate's problem
                value = got.value if candidate.language != "rust" else getattr(a, "value", None)
                if not compare.same(candidate.language, value, getattr(b, "value", None)):
                    bad.append(compare.Mismatch(case=case, candidate=value,
                                                reference=getattr(b, "value", None)))
            if bad:
                return False, f"{len(bad)} of {len(cases)} disagree with the reference", Fail.MISMATCH, len(cases)
            return True, "", Fail.NONE, len(cases)
        step = _timed(name, cap, body)
        return step

    def _invariants(self, candidate: Candidate, cases: list[dict], left: float,
                    return_shape: str = "") -> Step:
        """Shape, mutation, exactness -- the questions only the probe answers."""
        def body():
            if candidate.language == "rust" or not cases:
                return True, "function mode only" if candidate.language == "rust" else "no cases", Fail.NONE, 0
            runs = self._run(candidate, cases[:8], min(left, CAPS["invariants"]), probe=True)
            for case, run in zip(cases, runs):
                got = compare.Probed.read(run)
                if not got.ok:
                    continue
                if got.mutated:
                    return False, "the answer mutated an argument it was handed", Fail.MUTATION, len(runs)
                if not got.serialisable:
                    return False, f"the answer returned a {got.shape} that will not serialise", Fail.SHAPE, len(runs)
                wrong = compare.shape_matches(return_shape, got.shape)
                if wrong:
                    return False, wrong, Fail.SHAPE, len(runs)
                if got.max_int >= 1 << 63:
                    return False, f"an integer of {got.max_int.bit_length()} bits came back; check the width", Fail.OVERFLOW, len(runs)
            return True, "", Fail.NONE, len(runs)
        return _timed("invariants", CAPS["invariants"], body)

    def _stress(self, candidate: Candidate, kit: Kit, left: float) -> Step:
        def body():
            if kit.generate is None:
                return True, "no generator", Fail.NONE, 0
            case = None
            for size in (100, 40, 15):
                try:
                    made = kit.generate(424242, size)
                except Exception:  # noqa: BLE001
                    continue
                if made is not None:
                    case = made if isinstance(made, dict) else {"args": [made]}
                    break
            if case is None:
                return True, "the generator produced no large input", Fail.NONE, 0
            runs = self._run(candidate, [case], min(left, CAPS["stress"]))
            run = runs[0] if runs else None
            if run is None or not getattr(run, "ok", False):
                if getattr(run, "timed_out", False):
                    return False, "the largest input timed out", Fail.TIMEOUT, 1
                return True, "the largest input did not run; not held against it", Fail.NONE, 1
            return True, f"largest input in {getattr(run, 'runtime_ms', 0)/1000:.1f}s", Fail.NONE, 1
        return _timed("stress", CAPS["stress"], body)

    # -- the round ---------------------------------------------------------- #

    def verify(self, candidate: Candidate, kit: Optional[Kit] = None,
               left: float = ROUND_S, return_shape: str = "") -> Verdict:
        """One round. Stops at the first failure; every rung is timed."""
        kit = kit or Kit()
        verdict = Verdict()
        started = time.monotonic()

        def remaining() -> float:
            return max(0.5, left - (time.monotonic() - started))

        for step in (self._static(candidate), self._build(candidate, remaining())):
            verdict.steps.append(step)
            if not step.passed:
                verdict.failure = step.failure
                verdict.report = step.detail
                return verdict

        if kit.usable:
            step = self._kit_self_test(kit, remaining())
            verdict.steps.append(step)
            if not step.passed:
                # A broken kit does not condemn the candidate. The ladder
                # degrades to what does not depend on it and says so.
                verdict.failure = Fail.KIT
                verdict.report = step.detail
                kit = Kit(reference=kit.reference)

        edge = []
        if kit.edge_cases is not None:
            try:
                edge = [c if isinstance(c, dict) else {"args": [c]} for c in kit.edge_cases()]
            except Exception:  # noqa: BLE001
                edge = []
        if edge:
            step = self._cases_step("edge", candidate, kit, edge, remaining(), CAPS["edge"])
            verdict.steps.append(step)
            if not step.passed:
                verdict.failure = step.failure
                verdict.report = step.detail
                return verdict

        generated = []
        if kit.generate is not None:
            for seed in range(60):
                try:
                    made = kit.generate(seed, seed % 2)
                except Exception:  # noqa: BLE001
                    continue
                if made is not None:
                    generated.append(made if isinstance(made, dict) else {"args": [made]})
        if generated:
            step = self._cases_step("differential", candidate, kit, generated,
                                    remaining(), CAPS["differential"])
            verdict.steps.append(step)
            if not step.passed:
                verdict.failure = step.failure
                verdict.report = step.detail
                return verdict

        step = self._invariants(candidate, generated or edge, remaining(), return_shape)
        verdict.steps.append(step)
        if not step.passed:
            verdict.failure = step.failure
            verdict.report = step.detail
            return verdict

        step = self._stress(candidate, kit, remaining())
        verdict.steps.append(step)
        if not step.passed:
            verdict.failure = step.failure
            verdict.report = step.detail
            return verdict

        verdict.green = verdict.failure is Fail.NONE
        return verdict

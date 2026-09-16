"""Two programs, opposed instructions, the same inputs.

The question this answers is the one the old two-bar design could not. That
design asked two readers what a call SHOULD return and compared their answers;
when they agreed it learned nothing, because two readings of one statement by
one family of model agree for reasons that have nothing to do with being right.
The measured shape of it: 39 of 54 live solves never produced a disagreement at
all, and the judge brought in to settle the rest upheld the rewrite 22 times,
sided against it 0 times, and was unusable 7 -- a rubber stamp.

Here nothing is ASKED for an expected value. A naive oracle is written under a
correctness-first instruction (slow, literal, small inputs assumed) and the
candidate under a complexity-first one (fast at the stated maximums). Both are
executed on the same synthesized inputs and the oracle's outputs ARE the
expectations. Two programs written to optimise for different things that still
agree is evidence; two that disagree is a fault with a concrete input attached.

Three consequences worth stating, because each one removed code:

  * The comparison is the VALIDATOR'S. Feeding the oracle's output back in as
    `expected` and grading the candidate with `_Grader.check_detailed` means
    `values_equal` for Python and `outputs_match` for Rust -- the same functions
    the hidden suite will use, not a reimplementation of them. Whitespace-token
    equality on Rust comes free, so a `println!` oracle and a `print!("{}\\n")`
    candidate are correctly the same answer.

  * Which artifact a failure blames is a FIELD, not a substring. The upstream
    version routed repairs with `all("oracle" in detail ...)`, and the detail
    for an honest disagreement reads `candidate != oracle` -- which contains
    "oracle". So a report where every case was a real disagreement patched the
    reference instead of the program that ships. `CaseResult.blames` is set
    where the failure is classified and there is a test on it.

  * Mutation of caller-owned arguments is NOT checked here, and cannot be. The
    upstream harness deep-copies arguments and fails a case that changed them.
    This grader forks a child per case, exactly as the validator's does, and
    never compares the arguments before and after -- so mutation is invisible
    to the hidden suite too. Enforcing it locally would fail programs that
    would have scored, and there is no signal to report it with even if we
    wanted to. A test pins that a mutating program still passes.

The oracle is memoised per solve on the hash of its source. It rarely changes
while the candidate is being repaired, and for Rust every run of it otherwise
costs a fresh container plus a full `opt-level=2` build.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Failures carried into a repair prompt. Past a handful the model stops reading
# them as evidence and starts pattern-matching the list, and the prompt grows
# without the repair improving.
MAX_REPORTED = 3

# How much of one side's output a repair prompt may carry. A disagreement on a
# 200 KB return value is still a disagreement; the model does not need all of it
# to see which one is wrong.
OUTPUT_CLIP = 400


@dataclass
class CaseResult:
    """One input the two programs did NOT agree on.

    Only failures are recorded -- a passing case is counted and discarded --
    so there is no `ok` field to read. What a case needs to carry is which
    artifact it accuses and what each side produced.
    """

    name: str
    detail: str = ""
    # The artifact to repair: `oracle` when the reference could not produce a
    # value at all, `candidate` when both ran and disagreed. Set where the
    # failure is classified and never inferred from `detail`.
    blames: str = ""
    oracle_out: str = ""
    candidate_out: str = ""


@dataclass
class DifferentialReport:
    """What running both programs on the same inputs established."""

    ran: int = 0
    agreed: int = 0
    mismatch: int = 0
    oracle_crash: int = 0
    # Cases that were sent to be graded and never produced a verdict, because
    # the budget ran out mid-suite or the executor came back short. NOT
    # agreement: `check_detailed` counts `total` as every case it was handed,
    # so without this an unrun case is silently indistinguishable from a
    # passing one -- which is the whole failure this design exists to stop.
    unrun: int = 0
    # Failures only. A passing case is counted, not stored.
    cases: list[CaseResult] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        """Every input ran, and the two programs agreed on all of them.

        Three clauses, and each rules out a different way of looking finished
        without having been checked. `ran > 0`: a report with no cases has
        established nothing. `oracle_crash == 0`: a reference that fell over
        cannot vouch for anything. `unrun == 0`: a suite the clock cut short
        proved only as much as it got through, and reading the rest as
        agreement is exactly the `corrected=0/18 exit=converged` line this
        design was built to stop printing.
        """
        return (
            self.ran > 0
            and self.mismatch == 0
            and self.oracle_crash == 0
            and self.unrun == 0
        )

    @property
    def blame(self) -> str:
        """Which artifact the next repair should patch, by default.

        A mismatch outranks an oracle crash: if the two programs disagreed
        anywhere, there is a concrete input the candidate got wrong, and that
        is better evidence than a reference that fell over on some other one.
        The orchestrator can override this -- see the flip rule -- but it
        overrides a named decision rather than re-deriving one from text.
        """
        if self.mismatch:
            return "candidate"
        if self.oracle_crash:
            return "oracle"
        return ""

    def score(self) -> tuple:
        """Higher is better. Repair is monotone against this.

        A repair round that does not raise this leaves the previous version in
        place, so the cost of blaming the wrong program is a wasted round and
        never a worse answer shipped.
        """
        return (1 if self.ok else 0, self.agreed, -self.mismatch,
                -self.oracle_crash, -self.unrun)

    def summary(self) -> str:
        parts = [
            f"ran={self.ran}", f"agreed={self.agreed}",
            f"mismatch={self.mismatch}", f"oracle_crash={self.oracle_crash}",
        ]
        if self.unrun:
            parts.append(f"unrun={self.unrun}")
        if self.note:
            parts.append(self.note)
        return " ".join(parts)

    def prompt_text(self, limit: int = 3500) -> str:
        """The failure report a repair prompt carries."""
        lines: list[str] = [self.summary()]
        for case in self.cases[:MAX_REPORTED]:
            lines.append(f"FAIL {case.name}: {case.detail}")
            if case.oracle_out:
                lines.append(f"  reference produced: {case.oracle_out}")
            if case.candidate_out:
                lines.append(f"  this program produced: {case.candidate_out}")
        text = "\n".join(lines)
        return text if len(text) <= limit else text[:limit] + " ..."


# How many times the same failure may be blamed on the candidate before the
# router stops believing itself and patches the reference instead.
#
# Two, not one. A first disagreement genuinely is more likely the candidate's
# fault: it was written under the harder instruction, against the larger
# inputs, with the compression and closed forms that is where bugs live. One
# failed repair is ordinary -- models miss on the first try. Two failed repairs
# on the SAME case, with nothing else accusing the candidate, is the signature
# of a reference that is itself wrong about the statement, and continuing to
# patch a correct program is how a solve spends its whole deadline going
# backwards.
FLIP_AFTER = 2


class Router:
    """Which artifact the next repair should patch. One per solve.

    The default is `report.blame`: a disagreement accuses the candidate, a
    reference that fell over accuses the reference. That default is a
    heuristic and it can be wrong -- on the closest available measurement, two
    independent encodings of one statement, the shipped program was the wrong
    party 7 times in 27 and the second encoding 11. So the router does two
    things the bare default cannot.

    It listens to signals that do not come from the reference. `compile_defect`
    and the size probe accuse the candidate on their own evidence, and when
    either has spoken the router never second-guesses them -- there is no
    ambiguity about who is at fault.

    Otherwise it counts. The same set of failing cases blamed on the candidate
    `FLIP_AFTER` times over hands the next round to the reference instead, and
    `flipped` records that it happened. A DIFFERENT set of failures is a
    different argument and starts its own count, so a solve making real
    progress never flips.

    One caveat, stated because it is asymmetric and the code cannot fix it:
    `compile_defect` is Rust-only. On Python the independent signal is the size
    probe alone, so a wrongly-blamed Python candidate gets both rounds before
    the router reconsiders, on weaker evidence than a Rust one would.
    """

    def __init__(self, flip_after: int = FLIP_AFTER) -> None:
        self._blamed: dict[tuple, int] = {}
        self.flipped = 0

    def choose(
        self, report: "DifferentialReport", *,
        candidate_blamed_independently: bool = False,
    ) -> str:
        """`"candidate"`, `"oracle"`, or `""` when there is nothing to repair."""
        default = report.blame
        if default != "candidate":
            return default
        if candidate_blamed_independently:
            # Something that is not the reference says the candidate is wrong.
            # There is no dispute to arbitrate.
            return "candidate"

        key = tuple(sorted(case.name for case in report.cases if case.blames == "candidate"))
        seen = self._blamed.get(key, 0)
        if seen >= FLIP_AFTER:
            # Reset rather than latch: if the reference was not the problem
            # either, the next round goes back to the candidate instead of
            # flipping for the rest of the solve.
            self._blamed[key] = 0
            self.flipped += 1
            return "oracle"
        self._blamed[key] = seen + 1
        return "candidate"


def _clip(value: Any, limit: int = OUTPUT_CLIP) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:  # noqa: BLE001 - an unserialisable value is still evidence
        text = repr(value)
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" ...[{len(text)} chars]"


def inputs_key(inputs: list[dict[str, Any]]) -> str:
    """A stable key for a case list, for the oracle memo."""
    try:
        blob = json.dumps(
            [[case.get("args", []), case.get("kwargs", {})] for case in inputs],
            sort_keys=True, separators=(",", ":"), default=str,
        )
    except Exception:  # noqa: BLE001
        blob = repr(inputs)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def source_key(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8", "replace")).hexdigest()[:20]


class Differential:
    """Runs the pair for one solve. Holds the oracle memo for that solve only.

    One per solve, not one per miner: the memo is keyed on source text, and a
    long-lived instance would hand one solve's expectations to another problem
    that happened to produce a byte-identical oracle.
    """

    def __init__(self, grader: Any) -> None:
        self._grader = grader
        self._memo: dict[tuple[str, str], list[Any]] = {}
        self.oracle_runs = 0
        self.oracle_reused = 0

    def expectations(
        self, oracle: str, language: str, entrypoint: str,
        inputs: list[dict[str, Any]], budget_s: Optional[float] = None,
    ) -> list[Any]:
        """Run the oracle and return one `_Run` per input, in order.

        Memoised on (source, inputs). The oracle usually survives a candidate
        repair untouched, and for Rust re-running it costs a container and a
        full build.
        """
        key = (source_key(oracle), inputs_key(inputs))
        cached = self._memo.get(key)
        if cached is not None:
            self.oracle_reused += 1
            return cached
        try:
            runs = self._grader.outputs(
                oracle, language, entrypoint, inputs, budget_s=budget_s
            )
        except Exception as exc:  # noqa: BLE001 - a broken grader loses no answer
            # The standing rule everywhere else in this solver: an executor
            # that cannot be built costs the CHECK, never the answer. An empty
            # run list makes every input an unanswered one, which is what it
            # is -- and the report then reads as having established nothing
            # rather than as having established agreement.
            print(f"[verify] the reference could not be run, so nothing was "
                  f"checked against it: {type(exc).__name__}: {exc}")
            runs = []
        self._memo[key] = runs
        self.oracle_runs += 1
        return runs

    def compare(
        self, candidate: str, oracle: str, language: str, entrypoint: str,
        inputs: list[dict[str, Any]], budget_s: Optional[float] = None,
    ) -> DifferentialReport:
        """Run both programs on `inputs` and say what that established."""
        report = DifferentialReport()
        if not inputs:
            report.note = "no synthesized inputs"
            return report
        if not (candidate or "").strip():
            report.note = "no candidate to run"
            return report
        if not (oracle or "").strip():
            report.note = "no oracle to compare against"
            return report

        runs = self.expectations(
            oracle, language, entrypoint, inputs, budget_s=budget_s
        )

        # An input the oracle could not answer has no expectation, so the
        # candidate is not graded on it -- being unable to check a case is not
        # the same as the case failing, and grading it against a missing value
        # would blame the candidate for the reference's crash.
        gradable: list[dict[str, Any]] = []
        gradable_names: list[str] = []
        for index, case in enumerate(inputs):
            name = str(case.get("name") or f"case {index + 1}")
            run = runs[index] if index < len(runs) else None
            if run is None or not getattr(run, "ok", False):
                report.oracle_crash += 1
                report.cases.append(CaseResult(
                    name=name, blames="oracle",
                    detail=_oracle_detail(run),
                ))
                continue
            gradable.append({
                "args": list(case.get("args", []) or []),
                "kwargs": dict(case.get("kwargs", {}) or {}),
                "expected": getattr(run, "value", None),
            })
            gradable_names.append(name)

        if not gradable:
            report.note = "the reference answered nothing"
            return report

        try:
            passed, total, failures, failed, actuals = (
                self._grader.check_detailed(
                    candidate, language, entrypoint, gradable,
                    names=gradable_names, budget_s=budget_s,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a broken grader loses no answer
            print(f"[verify] local grading unavailable, so the candidate could "
                  f"not be compared: {type(exc).__name__}: {exc}")
            report.unrun = len(gradable)
            report.note = "the candidate could not be run"
            return report
        # `total` is every case that was HANDED OVER, not every case that ran:
        # a budget that expires mid-suite leaves the rest unrun, and
        # `check_detailed` reports them in neither `passed` nor `failures`.
        # Counting them as agreement is how a suite that proved a third of
        # itself reads as a clean one.
        observed = passed + len(failures)
        report.agreed = passed
        report.unrun = max(0, total - observed)
        report.ran = report.oracle_crash + observed

        # `failed` and `actuals` are built together, one append each per
        # failing case, so they are parallel TO EACH OTHER and not to the case
        # list. Indexing `actuals` by a position in `gradable` reads the wrong
        # element whenever the failures are not a prefix -- and silently, since
        # a short read just falls off the end and becomes "produced nothing",
        # which is the one thing a repair prompt must not be told about a
        # program that did produce something.
        by_index = {id(case): i for i, case in enumerate(gradable)}
        for case, produced in zip(failed, actuals):
            index = by_index.get(id(case))
            if index is None:
                index = _match_case(gradable, case)
            name = gradable_names[index] if index is not None else "case"
            expected = case.get("expected") if isinstance(case, dict) else None
            report.mismatch += 1
            report.cases.append(CaseResult(
                name=name,
                blames="candidate",
                detail=_candidate_detail(produced),
                oracle_out=_clip(expected),
                candidate_out=_clip(getattr(produced, "value", None)),
            ))
        return report


def _oracle_detail(run: Any) -> str:
    """Why the reference produced nothing for this input."""
    if run is None:
        return "the reference did not run"
    if getattr(run, "timed_out", False):
        return "the reference timed out"
    error = getattr(run, "error", None)
    return f"the reference failed: {error}" if error else "the reference failed"


def _candidate_detail(run: Any) -> str:
    """Why this case counts against the candidate."""
    if run is None:
        return "this program produced nothing"
    if getattr(run, "timed_out", False):
        return "this program timed out where the reference did not"
    error = getattr(run, "error", None)
    if error:
        return f"this program failed where the reference did not: {error}"
    return "this program and the reference disagree"


def _match_case(
    gradable: list[dict[str, Any]], case: dict[str, Any]
) -> Optional[int]:
    """Where a returned failed case sits in the list that was sent.

    `check_detailed` normally hands back the very dicts it was given, so
    identity finds them. This is the fallback for an executor that rebuilt
    them: a failure reported against the wrong case is worse than one reported
    against no case, so it compares the inputs rather than guessing by order.
    """
    for index, sent in enumerate(gradable):
        if sent.get("args") == case.get("args") and \
                sent.get("kwargs") == case.get("kwargs"):
            return index
    return None

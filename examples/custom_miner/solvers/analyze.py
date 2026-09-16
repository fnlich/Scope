"""What the statement says that a naive reading would miss.

This runs BEFORE any model call and costs nothing -- regexes over the statement,
milliseconds. That is the whole reason it exists. Every later prompt in the
solve is handed the same trap block, so the oracle, the candidate and every
repair round start from one reading of the problem instead of three.

The catalog is not a general-purpose linter. Each entry was put here because the
wording it matches is wording this subnet's statements actually use, and because
a solver that misses it produces a program that passes small cases and fails the
hidden suite -- the exact failure that scores zero while looking fine.

Measured over the 97 recorded tasks in `examples/problems` (48 python, 49 rust),
median 5 traps per task, never fewer than 3:

    97/97  sandbox_constraints          32/97  index_base
    97/97  no_public_examples            6/97  unicode_indexing
    88/97  large_n_hidden_tests          6/97  negative_index_wrap
    49/97  rust_contract                 5/97  persistent_or_branching_state
    48/97  python_contract               3/97  implicit_default_behavior
    47/97  token_output_compare          3/97  no_mutation_of_inputs
    40/97  huge_numeric_bounds           1/97  retry_accounting

Three entries -- `no_full_materialization`, `exact_arithmetic` and `dag_not_tree`
-- fire on NONE of those 97. They are kept rather than deleted because they do
fire on the upstream ChallengeBox statements (2/10, 1/10 and 1/10 of that set),
so the wording is real and this corpus simply has not drawn it yet. A regex that
has never matched costs microseconds; the trap it would have caught costs the
solve. If a later corpus still shows them at zero, delete them then -- but say so
with a number, not an impression.

Two rules govern the merge with the model's own analysis in `merge_analysis`:

  - Traps are unioned BY NAME and the heuristic wins a collision. A regex that
    matched is evidence the statement contains those words; a model that names
    the same trap is agreeing, not correcting.
  - Prose fields (sketch, complexities, summary) let the model win, because a
    regex cannot write an algorithm sketch and the heuristic's is a placeholder.

So the model can only ADD traps here, never remove one. That asymmetry is
deliberate: a missed trap costs the solve, an extra one costs a few tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# A number this big is never something to loop over. The forms are the ones the
# statements actually use -- `10**18`, `10^18`, `10^{18}` and `1e18` all appear.
_LARGE_BOUND = re.compile(
    r"10\s*\*\*\s*(9|12|18|30)|10\s*\^\s*(9|12|18|30)|10\^\{?(9|12|18|30)\}?|"
    r"1e(?:9|12|18|30)",
    re.I,
)
_N_BOUND = re.compile(
    r"(?:at most|<=|≤|less than or equal to)\s*([0-9][0-9_,]{4,})",
    re.I,
)
_PLAIN_INT = re.compile(r"\b([1-9][0-9]{5,})\b")
_COMMA_INT = re.compile(r"\b([1-9]\d{0,2}(?:,\d{3})+)\b")

# The size at which an O(n^2) answer stops being acceptable. Below it a quadratic
# program finishes inside the per-case timeout; above it, it does not.
_LARGE_N = 100_000

# (pattern, name, evidence, mitigation). Severity is `high` unless the trap is
# a standing contract rather than something the statement singled out.
_KEYWORD_TRAPS: list[tuple[re.Pattern[str], str, str, str]] = [
    (
        re.compile(
            r"do not enumerate|cannot be fully built|must not enumerate|"
            r"too large to materialize|enormous",
            re.I,
        ),
        "no_full_materialization",
        "The statement forbids building the whole structure in memory.",
        "Use implicit counts, lazy views, compressed trees or closed-form "
        "ranges. Never expand it.",
    ),
    (
        re.compile(r"implicit|after all runs are exhausted|later attempt", re.I),
        "implicit_default_behavior",
        "Behaviour continues after the explicit stream is exhausted.",
        "Model the remaining default tail; do not stop when the table ends.",
    ),
    (
        re.compile(r"must not modify|not mutate|do not mutate|caller-owned", re.I),
        "no_mutation_of_inputs",
        "Caller-owned inputs must come back unchanged.",
        "Copy, or treat them as read-only. Never sort or reverse in place.",
    ),
    (
        re.compile(r"exact arithmetic|all arithmetic is exact", re.I),
        "exact_arithmetic",
        "Rounding or wrapping would change the answer.",
        "Python ints are unbounded -- use them. Rust: i128, or checked ops; "
        "never silent wrap unless the statement asks for it.",
    ),
    (
        re.compile(r"retry is counted when|counts one retry|scheduled", re.I),
        "retry_accounting",
        "Retries are counted at one specific moment, not at every attempt.",
        "Follow the wording exactly: scheduled, attempted and dropped are "
        "three different events and may each count differently.",
    ),
    (
        re.compile(r"persistent|versions may branch|ancestry", re.I),
        "persistent_or_branching_state",
        "State is versioned or reused across operations.",
        "Persistent structures, copy-on-write or parent pointers. A single "
        "mutable snapshot gives the wrong answer on a branch.",
    ),
    (
        re.compile(r"unfolding|acyclic directed graph|multiple incoming", re.I),
        "dag_not_tree",
        "Shared nodes must be read by path, not by identity.",
        "Walk paths, or memoise on (node, environment). Never assume one "
        "visit per node.",
    ),
    (
        re.compile(r"whitespace bytes|splitting on ASCII whitespace", re.I),
        "token_output_compare",
        "The judge splits output on ASCII whitespace 0x09-0x0D and 0x20.",
        "Separate tokens with spaces or newlines. Any other separator "
        "becomes part of the token.",
    ),
    (
        re.compile(r"UTF-8|UTF-16|grapheme|surrogate|Unicode", re.I),
        "unicode_indexing",
        "Byte, code-unit, scalar and grapheme indexes are four things.",
        "Build the boundary table once. Never index a Python str as UTF-16, "
        "nor a Rust String as scalars, without converting first.",
    ),
    (
        re.compile(
            r"standard library|no I/O|perform no input/output|no filesystem|"
            r"no unsafe",
            re.I,
        ),
        "sandbox_constraints",
        "The solution is restricted to the language standard library.",
        "No packages, no files, no network, and no module-level state that "
        "would carry from one hidden test to the next.",
    ),
    (
        re.compile(r"one-based|1-based|zero-based|0-based", re.I),
        "index_base",
        "The index origin is stated, which means it is not the obvious one.",
        "Convert to one internal convention at the boundary and say so in a "
        "comment.",
    ),
    (
        re.compile(r"clamp|negative.*index|G\+x", re.I),
        "negative_index_wrap",
        "Negative indexes wrap and then clamp.",
        "Resolve with n+x first, then clamp. Python slice semantics are not "
        "the same thing.",
    ),
]


@dataclass
class Trap:
    """One thing about the statement that breaks a naive reading."""

    name: str
    evidence: str
    mitigation: str
    severity: str = "high"


@dataclass
class Analysis:
    """The reading of the statement every later prompt is given."""

    summary: str = ""
    signature: str = ""
    traps: list[Trap] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    algorithm_sketch: str = ""
    complexity_time: str = ""
    complexity_memory: str = ""
    naive_failure: str = ""
    io_notes: str = ""
    source: str = "heuristic"

    def trap_block(self) -> str:
        if not self.traps:
            return "(no traps recorded)"
        return "\n".join(
            f"- [{trap.severity}] {trap.name}: {trap.evidence} "
            f"| mitigation: {trap.mitigation}"
            for trap in self.traps
        )

    def as_prompt_block(self) -> str:
        invariants = "\n".join(f"- {i}" for i in self.invariants) or "- (none)"
        edges = "\n".join(f"- {e}" for e in self.edge_cases) or "- (none)"
        return (
            f"Summary: {self.summary or '(heuristic only)'}\n"
            f"Signature: {self.signature}\n"
            f"Time complexity target: {self.complexity_time}\n"
            f"Memory complexity target: {self.complexity_memory}\n"
            f"Naive solutions fail because: {self.naive_failure}\n"
            f"I/O notes: {self.io_notes}\n"
            f"Traps:\n{self.trap_block()}\n"
            f"Invariants:\n{invariants}\n"
            f"Edge cases:\n{edges}\n"
            f"Algorithm sketch:\n{self.algorithm_sketch}\n"
        )


_EDGE_CASES = (
    "Empty or zero-length inputs, if the statement allows them",
    "Single-element inputs",
    "Maximum n at tiny numeric values (stresses speed, not overflow)",
    "Tiny n at maximum numeric values (stresses overflow and exactness)",
    "Invalid or out-of-range records, if the statement defines them",
    "Off-by-one where a range is inclusive at one end and exclusive at the other",
)

_INVARIANTS = (
    "Honour every named status string or token exactly, including its case.",
    "Shared state persists only where the statement says it does.",
    "A dropped or invalid path must not apply its side effects later.",
)


def _parse_int(raw: str) -> int:
    return int(raw.replace("_", "").replace(",", ""))


def _is_python(language: str) -> bool:
    return (language or "").strip().lower() == "python"


def heuristic_analyze(task: Any) -> Analysis:
    """Read the statement with regexes. No model call, no network, no clock.

    `task` is duck-typed the way the rest of the solver takes it: `.statement`,
    `.language`, `.entrypoint` and `.public_examples`. Anything missing is
    treated as absent rather than raising -- this runs before everything else
    and must never be the reason a solve produces nothing.
    """
    text = str(getattr(task, "statement", "") or "")
    language = str(getattr(task, "language", "") or "")
    entrypoint = str(getattr(task, "entrypoint", "") or "")
    examples = getattr(task, "public_examples", None) or []

    traps: list[Trap] = []
    seen: set[str] = set()

    def add(trap: Trap) -> None:
        # First writer wins. The keyword catalog runs before the language
        # contract, so a statement that SAYS "standard library only" keeps the
        # high-severity version of `sandbox_constraints` rather than the
        # standing medium one.
        if trap.name in seen:
            return
        seen.add(trap.name)
        traps.append(trap)

    if _LARGE_BOUND.search(text):
        add(
            Trap(
                name="huge_numeric_bounds",
                evidence="The statement carries 1e9/1e18/1e30-scale bounds.",
                mitigation="Never loop over those counts. Closed form, two "
                "pointers, difference arrays or lazy simulation.",
            )
        )

    hits = [_parse_int(m.group(1)) for m in _N_BOUND.finditer(text)]
    for pattern in (_PLAIN_INT, _COMMA_INT):
        hits.extend(
            value
            for value in (_parse_int(m.group(1)) for m in pattern.finditer(text))
            if value >= _LARGE_N
        )
    if any(value >= _LARGE_N for value in hits):
        add(
            Trap(
                name="large_n_hidden_tests",
                evidence=f"Input sizes named in the statement include "
                f"{sorted(set(hits))[:8]}.",
                mitigation="Aim for near-linear time and O(n) memory. A "
                "quadratic answer that looks simple still times out.",
            )
        )

    for pattern, name, evidence, mitigation in _KEYWORD_TRAPS:
        if pattern.search(text):
            add(Trap(name=name, evidence=evidence, mitigation=mitigation))

    if _is_python(language):
        add(
            Trap(
                name="python_contract",
                evidence=f"Must define `{entrypoint}`, standard library only, "
                "no I/O.",
                mitigation="Do not read stdin, do not print, do not import "
                "anything outside the standard library.",
                severity="medium",
            )
        )
        add(
            Trap(
                name="sandbox_constraints",
                evidence="Python standard library only, and no I/O.",
                mitigation="No packages, no files, no network, no state that "
                "survives from one hidden test to the next.",
                severity="medium",
            )
        )
        signature = f"def {entrypoint}(...)"
        io_notes = "Called as a function. Return the value; never print it."
    else:
        add(
            Trap(
                name="rust_contract",
                evidence="One program with fn main(), stdin to stdout, std "
                "only, no unsafe.",
                mitigation="Read the input up front or with a fast scanner. "
                "Print exactly the tokens asked for.",
                severity="medium",
            )
        )
        add(
            Trap(
                name="sandbox_constraints",
                evidence="Rust standard library only, stdin/stdout, no unsafe.",
                mitigation="No crates, no files, no network, no randomness.",
                severity="medium",
            )
        )
        signature = "fn main()"
        io_notes = (
            "Stdin to stdout. The judge tokenises on ASCII whitespace "
            "0x09-0x0D and 0x20, so spacing between tokens is free but the "
            "tokens themselves are not."
        )

    if not examples:
        add(
            Trap(
                name="no_public_examples",
                evidence="No public examples came with this task, so the "
                "hidden suite is the only thing that scores it.",
                mitigation="Synthesize small inputs from the wording and "
                "check them against an independently written oracle.",
                severity="medium",
            )
        )

    naive = (
        "A direct simulation, or a fully expanded structure, runs out of time "
        "or memory on the hidden maximum-size tests."
    )
    if any(trap.name == "huge_numeric_bounds" for trap in traps):
        naive = (
            "Looping to a 1e18-scale counter, or allocating a 1e9-scale grid, "
            "cannot finish. This needs a closed form or a compressed structure."
        )

    return Analysis(
        summary="Heuristic pass over the statement; the model may add to it.",
        signature=signature,
        traps=traps,
        invariants=list(_INVARIANTS),
        edge_cases=list(_EDGE_CASES),
        algorithm_sketch="Choose the asymptotically safe shape first, then "
        "fill in the exact branching from the wording.",
        complexity_time="Near-linear in the stated n, and sublinear in any "
        "huge numeric bound.",
        complexity_memory="Never proportional to a 1e9-scale coordinate space "
        "or to an unfolded layout.",
        naive_failure=naive,
        io_notes=io_notes,
        source="heuristic",
    )


def merge_analysis(base: Analysis, extra: Analysis) -> Analysis:
    """Fold the model's analysis into the heuristic one.

    Traps union by name with the HEURISTIC winning -- see the module docstring.
    Prose fields let the model win when it said anything at all.
    """
    known = {trap.name: trap for trap in base.traps}
    for trap in extra.traps:
        known.setdefault(trap.name, trap)
    return Analysis(
        summary=extra.summary or base.summary,
        signature=extra.signature or base.signature,
        traps=list(known.values()),
        invariants=_unique(list(base.invariants) + list(extra.invariants)),
        edge_cases=_unique(list(base.edge_cases) + list(extra.edge_cases)),
        algorithm_sketch=extra.algorithm_sketch or base.algorithm_sketch,
        complexity_time=extra.complexity_time or base.complexity_time,
        complexity_memory=extra.complexity_memory or base.complexity_memory,
        naive_failure=extra.naive_failure or base.naive_failure,
        io_notes=extra.io_notes or base.io_notes,
        source="heuristic+llm" if extra.source.startswith("llm") else base.source,
    )


def analysis_from_json(payload: Any, heuristic: Analysis) -> Analysis:
    """Build an `Analysis` out of whatever the model returned.

    Every field is coerced and every field is optional. A model that returns
    half the shape still contributes that half; a model that returns something
    unusable contributes nothing and the heuristic stands. This never raises --
    the caller's alternative to a partial analysis is no analysis.
    """
    if not isinstance(payload, dict):
        return heuristic

    traps: list[Trap] = []
    for item in payload.get("traps") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "llm_trap").strip() or "llm_trap"
        # `evidence` is the documented key; `why` is what models write when
        # they paraphrase the schema, and dropping it loses the whole trap.
        evidence = str(item.get("evidence") or item.get("why") or "").strip()
        severity = str(item.get("severity") or "high").strip().lower()
        traps.append(
            Trap(
                name=name,
                evidence=evidence,
                mitigation=str(item.get("mitigation") or "").strip(),
                severity=severity if severity in ("high", "medium") else "high",
            )
        )

    extra = Analysis(
        summary=str(payload.get("summary") or "").strip(),
        signature=str(payload.get("signature") or "").strip()
        or heuristic.signature,
        traps=traps,
        invariants=_strings(payload.get("invariants")),
        edge_cases=_strings(payload.get("edge_cases")),
        algorithm_sketch=str(payload.get("algorithm_sketch") or "").strip(),
        complexity_time=str(payload.get("complexity_time") or "").strip(),
        complexity_memory=str(payload.get("complexity_memory") or "").strip(),
        naive_failure=str(payload.get("naive_failure") or "").strip(),
        io_notes=str(payload.get("io_notes") or "").strip(),
        source="llm",
    )
    return merge_analysis(heuristic, extra)


def _strings(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def trap_names(analysis: Optional[Analysis]) -> tuple[str, ...]:
    """The trap names, for a log line or a test."""
    if analysis is None:
        return ()
    return tuple(trap.name for trap in analysis.traps)

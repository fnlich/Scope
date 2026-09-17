"""What the statement says that a naive reading would miss.

This runs BEFORE any model call and costs nothing -- regexes over the statement,
milliseconds. That is the whole reason it exists. Every later prompt in the
solve is handed the same trap block, so the oracle, the candidate and every
repair round start from one reading of the problem instead of three.

The catalog is not a general-purpose linter. Each entry was put here because the
wording it matches is wording this subnet's statements actually use, and because
a solver that misses it produces a program that passes small cases and fails the
hidden suite -- the exact failure that scores zero while looking fine.

Measured over 275 recorded tasks -- the 97 in `examples/problems` plus the 178
in `fnlich/hone-examples`, which also ships the ACCEPTED solution beside each
statement. Median 7 traps per task:

    275/275  sandbox_constraints        30/275  preserve_untouched
    275/275  no_public_examples         25/275  case_sensitivity_stated
    248/275  large_n_hidden_tests       16/275  fixpoint_closure
    147/275  rust_contract              14/275  modular_arithmetic
    141/275  token_output_compare       13/275  unicode_indexing
    128/275  python_contract            13/275  persistent_or_branching_state
    109/275  huge_numeric_bounds        10/275  error_priority_order
     81/275  index_base                 10/275  no_mutation_of_inputs
     71/275  rust_wide_arithmetic       10/275  negative_index_wrap
     60/275  deterministic_tiebreak      8/275  all_branches_no_shortcircuit
     52/275  duplicates_defined          8/275  exact_output_shape
     51/275  inclusive_bounds            7/275  retry_accounting
     47/275  cycle_self_reference        6/275  float_exactness
     36/275  recursive_descent_depth     5/275  bool_is_not_int
                                         4/275  implicit_default_behavior

SIX of those entries are structural -- they say what language this is and that
the suite is hidden, and they fire on everything. What a solve actually gains is
the rest, and that is the number worth watching. Before the 15 entries mined
from `hone-examples`, a THIRD of statements drew none of them at all:

                          problem-specific traps      tasks with none
    hone-examples (178)   median 1 -> 2               61 (34%) -> 14 (8%)
    examples/problems(97) median 1 -> 3               30 (31%) ->  3 (3%)

The 97 were not a blind holdout -- they were in the set the hit rates were
measured over -- but the wording was mined from the OTHER 178 and from what
their accepted solutions had to do, so carrying across is some evidence the
patterns are not overfitted to one draw.

How an entry earned its place, and the bar for the next one: the wording must be
greppable and copied verbatim from real statements; the hit rate must
discriminate rather than fire on everything (nothing below is above 26% outside
the structural six); and every match was read to confirm it is not a false
positive. Two patterns were narrowed during that read -- `in this order` was
matching "in this ordering" and a bare `reserved` was matching reserved TOKENS
-- which is the whole reason the read is not optional.

Three entries -- `no_full_materialization`, `exact_arithmetic` and `dag_not_tree`
-- fire on NONE of the 275. They are kept rather than deleted because they do
fire on the upstream ChallengeBox statements (2/10, 1/10 and 1/10 of that set),
so the wording is real and this corpus has not drawn it. A regex that has never
matched costs microseconds; the trap it would have caught costs the solve. But
`exact_arithmetic` is now the weakest of the three for a reason worth recording:
the overflow it was reaching for turns out to be REAL and common -- 27 of 98
accepted Rust solutions reach for i128 -- and it missed every one of them,
because it waits on the words `exact arithmetic` and the statements instead say
`10^18`. `rust_wide_arithmetic` is what catches that, off the bound rather than
off a phrase. The lesson generalises: match what statements SAY, not what the
trap is called.

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
    # --------------------------------------------------------------------- #
    # Mined from 275 recorded statements (97 in `examples/problems`, 178 in
    # fnlich/hone-examples) and, where the corpus carries one, from the
    # ACCEPTED solution beside each. A trap earned a place here only when the
    # wording is greppable, the hit rate is discriminating rather than
    # universal, and every match was read to confirm it is not a false
    # positive. Rates below are over all 275.
    # --------------------------------------------------------------------- #
    (
        re.compile(
            r"no (?:[a-z ]{0,30})?depth limit|arbitrarily deep|deeply nested|"
            r"nesting depth|nested (?:expression|structure|object|list|value|"
            r"union|group)|recursiv|\bsubtree\b|\bdescendant",
            re.I,
        ),
        "recursive_descent_depth",
        "The input itself is a nested structure, so its depth is an INPUT and "
        "not a constant.",
        "Python recurses 1000 deep by default and a 200k-node chain is legal "
        "input -- write it with an explicit stack, or raise the limit at the "
        "top of the file. Rust overflows its 8 MiB stack the same way. "
        "Measured on this corpus: 30 of 80 accepted Python solutions either "
        "used an explicit stack or called `sys.setrecursionlimit`.",
    ),
    (
        re.compile(
            r"\bties?\b (?:are|use|is|go)|tie-break|ties use|resolve ties|"
            r"lexicographically smallest|lexicographically largest|"
            r"smallest .{0,20}(?:key|name|index)|earliest .{0,20}(?:branch|index)",
            re.I,
        ),
        "deterministic_tiebreak",
        "The statement names the rule for breaking a tie, which means ties "
        "happen and the hidden tests contain them.",
        "Sort by the FULL key the statement gives, tie-break included, in one "
        "comparison. Never leave the order to the sort's stability or to a "
        "dict's iteration order.",
    ),
    (
        re.compile(
            r"\bcyclic\b|\bcycles?\b|self-referen|leads back to|"
            r"nonempty chain|circular",
            re.I,
        ),
        "cycle_self_reference",
        "The input may contain a cycle, and the statement defines what that "
        "means rather than forbidding it.",
        "A self-loop is a cycle: read `a nonempty chain back to itself` "
        "literally. Detect with colours (white/grey/black), not a visited "
        "set -- and note that `is on a cycle` and `can reach a cycle` are "
        "different questions.",
    ),
    (
        re.compile(
            r"duplicate|appears more than once|repeated (?:entries|keys|names)|"
            r"after their first occurrence|first occurrence",
            re.I,
        ),
        "duplicates_defined",
        "Duplicates are possible and the statement says what to do with them.",
        "Follow the rule exactly -- keep the first, keep the last, reject, or "
        "merge are four different answers. A `set` or a `dict` silently picks "
        "one of them for you.",
    ),
    (
        re.compile(r"\binclusive\b|\bexclusive\b|half-open", re.I),
        "inclusive_bounds",
        "A range endpoint is called out as inclusive or exclusive, which is "
        "said only when it is not the obvious one.",
        "Write the interval one way internally and convert at the boundary. "
        "Python slices and `range` are half-open; the statement usually is "
        "not.",
    ),
    (
        re.compile(
            r"reserved (?:bit|field|key|name|word|value)|must be preserved|"
            r"must not be (?:modified|changed|masked|touched)|"
            r"leave .{0,25} unchanged|remain(?:s)? unchanged|"
            r"never (?:changed|modified)|bits? .{0,25}(?:must not|never) ",
            re.I,
        ),
        "preserve_untouched",
        "Part of the structure must come back exactly as it went in.",
        "Touch only what the statement names. Build the result from the "
        "original rather than from a normalised copy, and never rewrite a "
        "field just because you parsed it.",
    ),
    (
        re.compile(
            r"case-sensitive|case sensitive|casefold|case-insensitive", re.I
        ),
        "case_sensitivity_stated",
        "Case handling is stated, so it is load-bearing somewhere.",
        "Casefold once at the boundary if the match is insensitive, and keep "
        "the ORIGINAL spelling for output. `lower()` and `casefold()` differ, "
        "and Rust `to_lowercase` is not ASCII-only.",
    ),
    (
        re.compile(
            r"check,? in order\b|in the following order\b|"
            r"report the (?:first|earliest)|error (?:precedence|priority)|"
            r"whichever (?:comes|occurs) first|takes precedence|"
            r"check .{0,40}before (?:descend|child|its )|"
            r"(?:first|earliest) (?:failure|error|violation) (?:in|is|encountered)",
            re.I,
        ),
        "error_priority_order",
        "WHICH failure is reported is specified, not just that one is.",
        "Check in the stated order and return the first hit. A program that "
        "validates in its own order reports a real error at the wrong "
        "priority and scores zero on a case it almost got right.",
    ),
    (
        re.compile(
            r"smallest set|least fixed point|fixpoint|transitive(?:ly)? closure|"
            r"until no (?:more|further|additional)|repeat until|"
            r"such that every|is also included|propagat|cascad",
            re.I,
        ),
        "fixpoint_closure",
        "The answer is the least set closed under a rule, not a single pass.",
        "Iterate to a fixed point, or walk the REVERSED dependency edges from "
        "the seeds with a worklist. One forward sweep terminates early and "
        "under-reports.",
    ),
    (
        re.compile(
            r"modulo|modulus|1000000007|998244353|10\^9 \+ 7", re.I
        ),
        "modular_arithmetic",
        "Results are reduced modulo something.",
        "Reduce at every step, not at the end. Negative intermediates must "
        "come back non-negative -- Python `%` already does, Rust `%` does "
        "not. Division means a modular inverse, never `/`.",
    ),
    (
        re.compile(
            r"exactly one branch|more than one .{0,25}succeed|\bambiguous\b|"
            r"all branches|every branch",
            re.I,
        ),
        "all_branches_no_shortcircuit",
        "Ambiguity is an outcome, so every alternative has to be tried even "
        "after one succeeds.",
        "Evaluate them all and count the successes. Returning on the first "
        "match cannot tell one from two, which is the case the statement "
        "singled out.",
    ),
    (
        re.compile(
            r"exactly these keys|exactly those keys|with exactly the keys|"
            r"exactly these fields",
            re.I,
        ),
        "exact_output_shape",
        "The returned container's key set is pinned exactly.",
        "Emit every named key on every path, including the empty and error "
        "ones, and emit nothing else. The comparison is structural: a missing "
        "key and an extra key both fail.",
    ),
    (
        re.compile(
            r"booleans? are not (?:integers|ints|numbers)|are distinct kinds|"
            r"bool(?:ean)?s? (?:are|is) not (?:an? )?(?:integer|number)",
            re.I,
        ),
        "bool_is_not_int",
        "Booleans, integers and floats are separate kinds here.",
        "Python disagrees: `True == 1`, `isinstance(True, int)` is true, and "
        "`1 == 1.0`, so a dict key, a `set` or a bare `==` conflates them. "
        "Compare `type(x) is bool` FIRST, and key on `(type(x).__name__, x)`.",
    ),
    (
        re.compile(
            r"floating-point|floating point|IEEE ?754|binary64|"
            r"relative error|absolute error|round-half",
            re.I,
        ),
        "float_exactness",
        "Floating-point behaviour is named, so the answer turns on it.",
        "Do not accumulate error: compare with the stated tolerance, or stay "
        "in integers or `fractions.Fraction` and convert once at the end. "
        "`round()` is banker's rounding in Python and is not round-half-up.",
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
        # Rust has no big integers, and this corpus says the difference bites.
        # `i64` tops out just past 9.2e18, so a statement carrying a 1e18-scale
        # bound leaves under one decimal order of headroom: a single sum of two
        # such values, or any product, is already over. Overflow there is not a
        # panic -- the validator builds with `-C opt-level=2`, which turns the
        # checks OFF -- so the program exits 0 with a plausible wrong number and
        # nothing in the failure points at the cause.
        #
        # Measured: 27 of 98 accepted Rust solutions in this corpus reach for
        # i128/u128, and 63% of those statements carry 1e18-scale wording
        # against 20% of the rest -- a 3.2x lift. This is the trap the
        # never-firing `exact_arithmetic` entry was reaching for; that one
        # waits on the words `exact arithmetic`, which no statement here says.
        if _LARGE_BOUND.search(text):
            add(
                Trap(
                    name="rust_wide_arithmetic",
                    evidence="A 1e9/1e18-scale bound in Rust, where i64 holds "
                    "only to 9.2e18 and overflow at opt-level=2 is SILENT.",
                    mitigation="i64 for stated values, i128 for any sum or "
                    "product of them, and cast before multiplying rather than "
                    "after. Where wrapping is wanted, say so with "
                    "`wrapping_mul`; where it is not, `checked_`/`saturating_` "
                    "turns a silent wrong answer into a visible one.",
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

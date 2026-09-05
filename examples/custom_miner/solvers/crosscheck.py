"""A second, independent reading of the statement, and what it disagrees on.

The repair loop verifies a program against cases the SAME model wrote from the
SAME reading of the statement, in the same conversation. When that reading is
wrong the cases encode the mistake, the program satisfies them, and the loop
ends with every case passing and the answer wrong. Measured on this miner's
own archived production answers, replayed off-chain: of fifteen that had
passed their own cases, two were wrong -- confirmed wrong, by an independent
program disagreeing with them on generated inputs and a third reader, reasoning
from the statement, siding with the independent program both times. Both were
Rust, both the same shape: one rule misread, every case written to the
misreading. Thirteen percent, which is the size of the gap between this miner
and the ones it is measured against.

So, beside the primary solve and in parallel with it, a SECOND READING: a fresh
conversation, a different model where there is one, the statement and nothing
else. It writes its own program and a generator of random valid inputs. Once
the primary program passes its own cases, both programs run on the generated
inputs and every disagreement is put to a THIRD reader, who reasons the exact
output out from the statement alone. Two readings against one decides. When
the primary loses, the input becomes a case on its bar -- one it cannot talk
its way out of, because the repair prompt says the case was confirmed and the
merge rule locks it -- and the ordinary repair round does the rest.

The generator also produces one maximum-size input, on which the primary
program is timed. Hand-written cases are small; hidden tests are not, and an
O(n^2) answer passes every case it wrote for itself and times out on the
validator's. Nothing else in the loop ever ran the program at scale.

None of this replaces the primary solve or ranks against it. It finds the
inputs the primary was never going to test itself, and hands them to the loop
that already knows what to do with a failing case.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .prompts import build_code_prompt, extract_code

# Generated inputs per solve. Small ones -- sizes a wrong answer can be read
# off -- because a disagreement is only useful if a reader can adjudicate it.
DEFAULT_INPUTS = 60

# The least of the budget worth starting a cross-check in: inputs, two runs,
# a judge or two and a repair round have to fit behind it.
XCHECK_FLOOR_S = 45.0

# The least worth asking a judge in, and the most one judge turn may take.
# Measured: the same judge answered two Rust disputes in ten seconds each and
# two Python disputes on a larger statement in ninety each. The second is a
# turn the solve cannot afford, so it is cut, and asked again next round.
JUDGE_FLOOR_S = 25.0
JUDGE_TURN_MAX_S = 60.0

# What the check leaves untouched for the repair round it may cause. A check
# that spent everything it was given -- measured: 188s of 214s -- found
# nothing it could still act on.
REPAIR_RESERVE_S = 40.0

# Disagreements put to the judge per round. Two is enough to tell a wrong
# program from an unlucky generator, and each costs a turn.
MAX_ADJUDICATIONS = 2

# A maximum-size input larger than this is not sent to the sandbox at all: the
# runners bound their status payloads, and a rejected run says nothing.
STRESS_MAX_BYTES = 1_000_000

# The scales tried for the large input, in order, until one comes back.
STRESS_SCALES = (100, 40, 15)

# How much of what is left the primary will wait for the second reading to
# arrive, when it has not yet. The reading started at the same time as the
# primary and is usually done first; this is for the small problem the primary
# finishes in fifteen seconds.
READ_WAIT_SHARE = 0.5


def _flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def enabled() -> bool:
    raw = _flag("SOLVER_CROSSCHECK", "1").lower()
    return raw not in ("0", "false", "no", "off")


def _profile(name: str, default: str) -> tuple[str, str]:
    raw = _flag(name, default) or default
    model, _, effort = raw.partition(":")
    return model.strip() or default.partition(":")[0], (effort.strip() or "low")


def reader_profile() -> tuple[str, str]:
    """Who writes the second reading. `fable:low` by default: measured on a
    real Rust problem, 38s for a program against 86s for opus at low effort
    and 161s for sonnet -- and a different model reads a statement
    differently, which is the point."""
    return _profile("SOLVER_CROSSCHECK_PROFILE", "fable:low")


def judge_profile() -> tuple[str, str]:
    """Who adjudicates a disagreement. `opus:low` by default, measured: at
    medium effort the same judge answered two Rust disputes in ten seconds
    each and was cut off at sixty on a larger Python statement without a
    verdict. The inputs are small and the question exact; what the judge
    needs is to finish. A fresh conversation, so still a reading of its own."""
    return _profile("SOLVER_JUDGE_PROFILE", "opus:low")


def inputs_wanted() -> int:
    try:
        return max(5, int(_flag("SOLVER_CROSSCHECK_INPUTS", str(DEFAULT_INPUTS))))
    except ValueError:
        return DEFAULT_INPUTS


# -- prompts ---------------------------------------------------------------- #

_GENERATOR_CONTRACT = (
    "Reply with ONE fenced `python` block written directly in the chat and "
    "nothing else. No preamble, no explanation before it or after it. Only "
    "what is inside the fence is ever read."
)

_GENERATOR_TASK = """\
Write a Python 3 function `generate(seed, scale)` that RETURNS one random test
input for this problem. Call `random.seed(seed)` first. `scale` is an integer
from 1 to 100: 1 means the smallest interesting inputs -- sizes of one to six,
so that a wrong answer can be read off by hand -- and 100 means the largest
input every limit in the statement allows.

{shape}

Every input returned must be VALID under every constraint the statement states.
Vary the structure so that, across seeds, every rule in the statement gets
exercised: the shapes each rule fires on and the shapes that nearly do, ties,
duplicates, boundaries, empty parts, and the ordinary case. Use only the
standard library, define everything the function needs in the same block, and
do not print anything."""

_SHAPE_PYTHON = (
    'Return a dict `{{"args": [...], "kwargs": {{}}}}` giving the arguments '
    "for `{entrypoint}(*args, **kwargs)`. Every value must be JSON-serialisable: "
    "no tuples, no sets, no `inf`, no `NaN`."
)
_SHAPE_RUST = (
    "Return one string: the complete stdin the program reads, laid out exactly "
    "as the statement's input format describes, ending in a newline."
)

_JUDGE_CONTRACT = (
    "Reply with ONE fenced `json` block written directly in the chat and "
    "nothing else. No preamble, no explanation before it or after it."
)

_JUDGE_TASK_PYTHON = """\
Two independently written programs disagree on the calls below, so one of them
has misread the statement. Decide each from the STATEMENT ALONE: work out, step
by step in your head, the exact value `{entrypoint}(*args, **kwargs)` must
return for every call, in order.

{calls}

Reply with `{{"expected": [<value for call 1>, <value for call 2>, ...]}}` -- one
JSON value per call, in the same order, and nothing else."""

_JUDGE_TASK_RUST = """\
Two independently written programs disagree on the inputs below, so one of them
has misread the statement. Decide each from the STATEMENT ALONE: work out, step
by step in your head, the exact stdout the program must write for every stdin,
in order.

{inputs}

Reply with `{{"expected": ["<stdout for input 1>", "<stdout for input 2>", ...]}}`
-- one JSON string per input, in the same order, and nothing else."""


def build_generator_prompt(language: str, statement: str, entrypoint: str) -> str:
    shape = (_SHAPE_RUST if language == "rust" else _SHAPE_PYTHON).format(
        entrypoint=entrypoint
    )
    return "\n".join([
        "<output>", _GENERATOR_CONTRACT, "</output>", "",
        f'<problem language="{language}" entrypoint="{entrypoint}">',
        statement.strip(), "</problem>", "",
        "<task>", _GENERATOR_TASK.format(shape=shape), "</task>",
    ])


def build_judge_prompt(language: str, statement: str, entrypoint: str, cases: list[dict]) -> str:
    """One turn for every disputed input: a judge reads the statement once."""
    if language == "rust":
        inputs = "\n\n".join(
            f"Input {i}:\n```\n{str((case.get('args') or [''])[0])}\n```"
            for i, case in enumerate(cases, 1)
        )
        task = _JUDGE_TASK_RUST.format(inputs=inputs)
    else:
        calls = "\n\n".join(
            f"Call {i}:\n```json\n"
            + json.dumps({"args": case.get("args", []), "kwargs": case.get("kwargs") or {}})
            + "\n```"
            for i, case in enumerate(cases, 1)
        )
        task = _JUDGE_TASK_PYTHON.format(entrypoint=entrypoint, calls=calls)
    return "\n".join([
        "<output>", _JUDGE_CONTRACT, "</output>", "",
        f'<problem language="{language}" entrypoint="{entrypoint}">',
        statement.strip(), "</problem>", "",
        "<task>", task, "</task>",
    ])


_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def extract_generator(reply: str) -> str:
    """The fenced block that defines `generate`, or ''."""
    for block in _FENCE.findall(reply or ""):
        if re.search(r"^\s*def\s+generate\s*\(", block, re.M):
            return block.strip()
    return ""


def extract_expected(reply: str, count: int = 1) -> tuple[bool, list[Any]]:
    """The judge's `{"expected": [...]}` as (found, values), `count` of them.

    A single value where a list of one was asked for is read as that list;
    a list of the wrong length is no verdict at all, because nothing says
    which input the missing one was."""
    for block in _FENCE.findall(reply or ""):
        try:
            data = json.loads(block)
        except Exception:  # noqa: BLE001 - not every fence is JSON
            continue
        if isinstance(data, dict) and "expected" in data:
            values = data["expected"]
            if count == 1 and not isinstance(values, list):
                return True, [values]
            if isinstance(values, list) and len(values) == count:
                return True, values
            return False, []
    return False, []


# -- outcomes --------------------------------------------------------------- #

@dataclass
class Outcome:
    """One program run on one input, as the sandbox reported it."""

    ok: bool
    value: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    runtime_ms: float = 0.0

    def describe(self, language: str) -> str:
        if self.timed_out:
            return "timed out"
        if not self.ok:
            return f"crashed ({(self.error or 'error')[:160]})"
        if language == "rust":
            return f"printed {json.dumps(str(self.value))[:400]}"
        return f"returned {json.dumps(self.value, default=str)[:400]}"


def same(language: str, a: Outcome, b: Outcome) -> bool:
    """The validator's own equality: tokens for Rust, structure for Python."""
    if a.timed_out or b.timed_out or not a.ok or not b.ok:
        return (a.timed_out, a.ok) == (b.timed_out, b.ok)
    if language == "rust":
        from rlvr.execution.rust_judge import outputs_match

        return outputs_match(str(a.value), str(b.value))
    from rlvr.execution.compare import values_equal

    return values_equal(a.value, b.value)


def _case_from(language: str, value: Any) -> Optional[dict]:
    """A generator's return value as a case dict, or None if it is not one."""
    if language == "rust":
        return {"args": [value]} if isinstance(value, str) and value else None
    if not isinstance(value, dict) or not isinstance(value.get("args"), list):
        return None
    kwargs = value.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        return None
    return {"args": list(value["args"]), "kwargs": dict(kwargs)}


def _render_call(language: str, entrypoint: str, case: dict) -> str:
    if language == "rust":
        return f"stdin {json.dumps(str((case.get('args') or [''])[0]))[:500]}"
    args = ", ".join(json.dumps(a, default=str) for a in case.get("args", []))
    kwargs = "".join(f", {k}={json.dumps(v, default=str)}" for k, v in (case.get("kwargs") or {}).items())
    return f"{entrypoint}({args}{kwargs})"[:500]


Opener = Callable[[], Awaitable[Any]]


# -- the second reading ----------------------------------------------------- #

class SecondReading:
    """An independent program and input generator, fetched in the background.

    Started when the solve starts, so it costs no wall-clock the primary would
    have used: the primary's cases turn alone takes longer than this whole
    reading on most problems. Its conversation is its own; nothing it says is
    ever shown to the primary, and nothing the primary says reaches it.
    """

    def __init__(self, opener: Opener, task, budget_s: float) -> None:
        self._opener = opener
        self._task_in = task
        self._budget = max(1.0, float(budget_s))
        self.code = ""
        self.generator = ""
        self.provider: Optional[str] = None
        self.error: Optional[str] = None
        self.spent = 0.0
        self._job: Optional[asyncio.Task] = None
        self._started = time.monotonic()

    def start(self) -> "SecondReading":
        self._job = asyncio.ensure_future(self._run())
        return self

    @property
    def done(self) -> bool:
        return self._job is not None and self._job.done()

    async def _run(self) -> None:
        task = self._task_in
        conversation = None
        try:
            left = lambda: max(1.0, self._budget - (time.monotonic() - self._started))  # noqa: E731
            conversation = await self._opener(min(left() * 0.5, 120.0))
            self.provider = getattr(conversation, "provider", None)
            reply = await conversation.send(
                build_code_prompt(task.language, task.statement, task.entrypoint,
                                  list(task.public_examples or []), None),
                left(),
            )
            self.code = extract_code(reply, task.entrypoint, task.language) or ""
            if not self.code.strip():
                self.error = getattr(conversation, "empty_reason", None) or "no program"
                return
            reply = await conversation.send(
                build_generator_prompt(task.language, task.statement, task.entrypoint),
                left(),
            )
            self.generator = extract_generator(reply)
            if not self.generator:
                self.error = "no generator"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a lost reading loses no answer
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.spent = time.monotonic() - self._started
            if conversation is not None:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001
                    pass

    async def wait(self, timeout_s: float) -> bool:
        """True when a program is in hand, waiting up to `timeout_s` for it."""
        if self._job is None:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(self._job), timeout=max(0.0, timeout_s))
        except asyncio.TimeoutError:
            return False
        except Exception:  # noqa: BLE001 - reported through `error`
            pass
        return bool(self.code.strip())

    async def close(self) -> None:
        if self._job is not None and not self._job.done():
            self._job.cancel()
            try:
                await self._job
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# -- the check -------------------------------------------------------------- #

@dataclass
class CheckResult:
    """What one cross-check round established about one program."""

    failures: list[str] = field(default_factory=list)
    new_cases: list[dict] = field(default_factory=list)
    inputs: int = 0
    disagreements: int = 0
    judged: int = 0
    stress: str = ""
    note: str = ""

    @property
    def clean(self) -> bool:
        return not self.failures and self.inputs > 0


class CrossCheck:
    """Runs the primary program against the second reading and judges the gaps.

    One per solve: the generated inputs and the second program's outputs are
    computed once and reused across repair rounds and passes, so each round
    pays only for running the new program and for any new disagreement.
    """

    def __init__(
        self, grader, task, reading: SecondReading, judge: Opener,
        inputs: Optional[int] = None,
    ) -> None:
        self._grader = grader
        self._task = task
        self._reading = reading
        self.reading = reading
        self._judge = judge
        self._wanted = inputs or inputs_wanted()
        self._cases: Optional[list[dict]] = None
        self._second: Optional[list[Outcome]] = None
        self._stress_case: Optional[dict] = None
        self._stress_second: Optional[Outcome] = None
        self._stress_note = "no large input"
        self._stress_scale = 0
        self._verdicts: dict[str, str] = {}
        # Cases confirmed against the primary, and their keys: on the bar for
        # every later round and pass, and never up for correction.
        self.confirmed: list[dict] = []
        self.locked: set[str] = set()
        self.rounds = 0
        # The second reading's program graded against the primary's own
        # cases, in the background, so that a fallback verdict is in hand the
        # moment the primary runs out. See `pregrade` and `fallback`.
        self._pregrade: Optional[asyncio.Task] = None
        self.second_passed = 0
        self.second_total = 0

    @staticmethod
    def key(case: dict) -> str:
        return json.dumps(
            [case.get("args", []), case.get("kwargs") or {}], sort_keys=True, default=str
        )

    def confirm(self, case: dict) -> None:
        """Lock a case the primary must satisfy: on the bar for every later
        round and pass, and never the primary's to correct."""
        key = self.key(case)
        if key not in self.locked:
            self.locked.add(key)
            confirmed = dict(case)
            # `run` reports a confirmed case by name, and a case the model
            # wrote need not have one.
            confirmed.setdefault("name", f"confirmed {len(self.confirmed) + 1}")
            self.confirmed.append(confirmed)

    # -- the second reading as the answer of last resort ------------------- #

    def pregrade(self, cases: list[dict], left_s: float) -> None:
        """Grade the second reading's program against the primary's own cases,
        in the background, once per solve.

        Measured on a production log: the primary's program turn ran past
        200 seconds in six of seventy-six solves, and four of those ended
        with nothing, a fragment, or a program failing its own cases -- while
        a second program had been sitting finished for two minutes. The
        verdict on that program is computed while there is still time, so
        that using it costs nothing at the deadline.
        """
        if self._pregrade is not None or not cases:
            return
        self._pregrade = asyncio.ensure_future(self._pregrade_run(list(cases), left_s))

    async def _pregrade_run(self, cases: list[dict], left_s: float) -> None:
        started = time.monotonic()
        try:
            if not await self._reading.wait(max(1.0, left_s * 0.9)):
                return
            left = left_s - (time.monotonic() - started)
            if left < 3.0:
                return
            runs = await self._outputs(self._reading.code, cases, left)
            language = self._task.language
            passed = sum(
                1 for case, run in zip(cases, runs)
                if same(language, Outcome(True, case.get("expected")), run)
            )
            self.second_passed, self.second_total = passed, len(cases)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a lost pregrade loses no answer
            return

    async def fallback(self, wait_s: float = 0.0) -> Optional[tuple[str, int, int]]:
        """The second reading's program, when it passed every one of the
        primary's own cases: (code, passed, total). None otherwise.

        Waits up to `wait_s` for a grading still running. Ordinarily it
        finished minutes ago -- the primary's program turn is the long part
        of a solve -- and the wait is for the other shape, a primary that
        died early, where a few seconds buys the only answer there is."""
        if self._pregrade is None:
            return None
        if not self._pregrade.done() and wait_s > 0:
            try:
                await asyncio.wait_for(asyncio.shield(self._pregrade), wait_s)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        if not self._pregrade.done():
            return None
        if self.second_total and self.second_passed == self.second_total:
            code = self._reading.code
            if code.strip():
                return code, self.second_passed, self.second_total
        return None

    async def close(self) -> None:
        if self._pregrade is not None and not self._pregrade.done():
            self._pregrade.cancel()
            try:
                await self._pregrade
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self._reading.close()

    # -- a correction, judged before it is accepted --------------------------- #

    async def judge_corrections(
        self, disputes: list[tuple[dict, dict]], left_s: float
    ) -> list[tuple[str, Any]]:
        """Each dispute is (the case as it stood, the case as the primary
        wants it), same call, different expectation. One judge turn decides
        them all: per dispute, `original`, `corrected`, `neither` or `no
        verdict`, with the judge's own expected value.

        Measured on a production log: eighteen times in seventy-six solves
        the primary answered a failing case by rewriting the case, and every
        one of those rewrites was accepted on its own say-so. The author of
        a program is the one party with a reason to want its cases changed;
        a reader with no program decides instead.
        """
        cases = [original for original, _ in disputes]
        values = await self._ask_judge(cases, left_s)
        if values is None:
            return [("no verdict", None)] * len(disputes)
        language = self._task.language
        verdicts: list[tuple[str, Any]] = []
        for (original, corrected), expected in zip(disputes, values):
            if language == "rust" and not isinstance(expected, str):
                verdicts.append(("no verdict", None))
                continue
            judge = Outcome(True, expected)
            with_original = same(language, judge, Outcome(True, original.get("expected")))
            with_corrected = same(language, judge, Outcome(True, corrected.get("expected")))
            verdicts.append((
                "corrected" if with_corrected and not with_original
                else "original" if with_original and not with_corrected
                else "neither" if not with_original else "corrected",
                expected,
            ))
        return verdicts

    async def run(self, code: str, left_s: float) -> CheckResult:
        """Cross-check `code` inside `left_s`. Never raises."""
        task = self._task
        language = task.language
        result = CheckResult()
        deadline = time.monotonic() + max(0.0, left_s - REPAIR_RESERVE_S)
        left = lambda: deadline - time.monotonic()  # noqa: E731
        self.rounds += 1
        try:
            if not await self._reading.wait(min(left() * READ_WAIT_SHARE, 90.0)):
                result.note = (
                    f"no second reading ({self._reading.error or 'still writing'}"
                    f" after {self._reading.spent:.0f}s)" if self._reading.done
                    else f"second reading not in yet after {time.monotonic() - self._reading._started:.0f}s"
                )
                return result
            if self._cases is None:
                await self._generate(left)
            if not self._cases:
                result.note = "the generator produced no valid input"
                return result
            if self._second is None:
                self._second = await self._outputs(self._reading.code, self._cases, left())
            first = await self._outputs(code, self._cases, left())
            result.inputs = len(self._cases)
            gaps = [
                (case, a, b) for case, a, b in zip(self._cases, first, self._second)
                if not same(language, a, b)
            ]
            result.disagreements = len(gaps)
            # Judge the gaps not judged before, a few per round, in ONE turn,
            # while it fits.
            disputed = [
                (case, a, b) for case, a, b in gaps if self.key(case) not in self._verdicts
            ][:MAX_ADJUDICATIONS]
            verdicts = (
                await self._adjudicate(disputed, left())
                if disputed and left() >= JUDGE_FLOOR_S else []
            )
            for (case, a, b), (verdict, expected) in zip(disputed, verdicts):
                key = self.key(case)
                result.judged += 1
                if verdict not in ("primary wrong", "second wrong", "tie", "undecided"):
                    # A judge that could not be reached, or said nothing
                    # usable, is asked again next round; a verdict is not.
                    continue
                self._verdicts[key] = verdict
                if verdict == "primary wrong":
                    confirmed = dict(case)
                    confirmed["name"] = f"cross-check {len(self.confirmed) + 1}"
                    confirmed["expected"] = expected
                    self.confirmed.append(confirmed)
                    self.locked.add(key)
                    result.new_cases.append(confirmed)
                    result.failures.append(
                        f"case {confirmed['name']!r} (an independent program and a "
                        f"third reading of the statement agree on this; the case is "
                        f"right, fix the program): "
                        f"{_render_call(language, task.entrypoint, case)} should "
                        f"{'print' if language == 'rust' else 'return'} "
                        f"{json.dumps(expected, default=str)[:400]}; yours "
                        f"{a.describe(language)}"
                    )
            # Cases confirmed in earlier rounds that this program still fails.
            for confirmed in self.confirmed:
                if any(self.key(c) == self.key(confirmed) for c in result.new_cases):
                    continue
                if left() < 10.0:
                    break
                run = await self._outputs(code, [confirmed], left())
                if run and not same(language, run[0], Outcome(True, confirmed["expected"])):
                    result.failures.append(
                        f"case {confirmed['name']!r} (confirmed by two independent "
                        f"readings; the case is right, fix the program): "
                        f"{_render_call(language, task.entrypoint, confirmed)} should "
                        f"{'print' if language == 'rust' else 'return'} "
                        f"{json.dumps(confirmed['expected'], default=str)[:400]}; yours "
                        f"{run[0].describe(language)}"
                    )
            if left() > 10.0:
                await self._stress(code, result, left)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken check loses no answer
            result.note = f"cross-check failed: {type(exc).__name__}: {exc}"
        return result

    async def _generate(self, left) -> None:
        """Run the generator for the small inputs and the one large one."""
        language = self._task.language
        seeds = [[seed, 1 + seed % 3] for seed in range(self._wanted)]
        runs = await self._outputs(
            self._reading.generator, [{"args": s} for s in seeds], left(), language="python"
        )
        cases = []
        for run in runs:
            case = _case_from(language, run.value) if run.ok else None
            if case is not None:
                cases.append(case)
        self._cases = cases
        # The largest input that can be had. Full scale first; when that is
        # too big for the sandbox to hand back, or too slow to generate in a
        # case's time, a smaller scale -- a program that is quadratic at the
        # limit is quadratic at a third of it, and a third is what fits.
        for scale in STRESS_SCALES:
            if left() < 10.0:
                self._stress_note = "no large input (no time to generate one)"
                break
            big = await self._outputs(
                self._reading.generator, [{"args": [424242, scale]}],
                min(left(), 30.0), language="python",
            )
            if not big or not big[0].ok:
                self._stress_note = "no large input (the generator " + (
                    "timed out" if big and big[0].timed_out
                    else f"failed: {(big[0].error or '?')[:80]}" if big else "did not run"
                ) + f" at scale {scale})"
                continue
            case = _case_from(language, big[0].value)
            if case is None:
                self._stress_note = f"no large input (the wrong shape at scale {scale})"
                continue
            if len(json.dumps(case, default=str)) > STRESS_MAX_BYTES:
                self._stress_note = f"no large input (over {STRESS_MAX_BYTES} bytes at scale {scale})"
                continue
            self._stress_case = case
            self._stress_scale = scale
            break

    async def _outputs(
        self, code: str, cases: list[dict], left_s: float, language: Optional[str] = None
    ) -> list[Outcome]:
        language = language or self._task.language
        entrypoint = "generate" if language == "python" and code is self._reading.generator else self._task.entrypoint
        raw = await asyncio.to_thread(
            self._grader.outputs, code, language, entrypoint, cases, max(1.0, left_s)
        )
        return [Outcome(ok=r.ok, value=r.value, error=r.error, timed_out=r.timed_out,
                        runtime_ms=r.runtime_ms) for r in raw]

    async def _ask_judge(self, cases: list[dict], left_s: float) -> Optional[list[Any]]:
        """One judge turn over `cases`: the expected value for each, in order,
        or None when no usable verdict came back. A turn that cannot finish
        inside `JUDGE_TURN_MAX_S` is cut and counted as no verdict."""
        task = self._task
        conversation = None
        started = time.monotonic()
        try:
            conversation = await self._judge(min(left_s * 0.5, 30.0))
            reply = await conversation.send(
                build_judge_prompt(task.language, task.statement, task.entrypoint, cases),
                max(1.0, min(left_s, JUDGE_TURN_MAX_S)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] cross-check: no judge ({type(exc).__name__}: {exc})")
            return None
        finally:
            if conversation is not None:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001
                    pass
        spent = time.monotonic() - started
        found, values = extract_expected(reply, len(cases))
        if not found:
            print(f"[verify] cross-check: the judge gave no usable verdict in {spent:.0f}s"
                  + (" (cut off)" if getattr(conversation, "still_writing", False) else ""))
            return None
        print(f"[verify] cross-check: the judge decided {len(cases)} input(s) in {spent:.0f}s")
        return values

    async def _adjudicate(
        self, disputed: list[tuple[dict, Outcome, Outcome]], left_s: float
    ) -> list[tuple[str, Any]]:
        """Put the disputed inputs to the judge in one turn: one (verdict,
        expected) per input, in order."""
        task = self._task
        values = await self._ask_judge([case for case, _, _ in disputed], left_s)
        if values is None:
            return [("no verdict", None)] * len(disputed)
        verdicts: list[tuple[str, Any]] = []
        for (case, a, b), expected in zip(disputed, values):
            if task.language == "rust" and not isinstance(expected, str):
                # A Rust verdict is stdout, and the grader's token split
                # reads anything else as a crash -- on every later round.
                verdicts.append(("no verdict", None))
                continue
            judge = Outcome(True, expected)
            with_a = same(task.language, judge, a)
            with_b = same(task.language, judge, b)
            verdict = ("primary wrong" if with_b and not with_a
                       else "second wrong" if with_a and not with_b
                       else "undecided" if not with_a else "tie")
            verdicts.append((verdict, expected))
        print("[verify] cross-check: " + ", ".join(v for v, _ in verdicts))
        return verdicts

    async def _stress(self, code: str, result: CheckResult, left) -> None:
        """Time the program on the largest generated input. A timeout is a
        failure only when the second program finishes it: that proves the
        input can be done in time, so the slowness is this program's."""
        case = self._stress_case
        if case is None:
            result.stress = self._stress_note
            return
        size = len(json.dumps(case, default=str))
        scale = f"scale {self._stress_scale}, " if self._stress_scale < 100 else ""
        if self._stress_second is None:
            second = await self._outputs(self._reading.code, [case], min(left(), 20.0))
            self._stress_second = second[0] if second else Outcome(False, error="unrun")
        if left() < 10.0:
            result.stress = "no time to run the largest input"
            return
        run = await self._outputs(code, [case], min(left(), 20.0))
        first = run[0] if run else Outcome(False, error="unrun")
        if first.timed_out and self._stress_second.ok and not self._stress_second.timed_out:
            result.stress = f"{scale}{size} bytes: TIMED OUT (the independent program took {self._stress_second.runtime_ms / 1000:.1f}s)"
            result.failures.append(
                f"a maximum-size input ({size} bytes) timed out: each test gets about "
                f"5 seconds, and an independent program finished this one in "
                f"{self._stress_second.runtime_ms / 1000:.1f}s. Reduce the complexity; "
                f"do not ask for the case, there is no case to correct"
            )
        elif first.ok:
            result.stress = f"{scale}{size} bytes in {first.runtime_ms / 1000:.1f}s"
        elif first.timed_out:
            result.stress = f"{size} bytes: timed out, and so did the independent program; not held against it"
        else:
            result.stress = f"{size} bytes: {first.describe(self._task.language)}; not held against it"


def describe(result: CheckResult) -> str:
    """One log line for a round's result."""
    if result.note and not result.inputs:
        return result.note
    parts = [f"{result.inputs} generated input(s), {result.disagreements} disagreement(s)"]
    if result.judged:
        parts.append(f"{result.judged} put to the judge")
    if result.new_cases:
        parts.append(f"{len(result.new_cases)} CONFIRMED against this program")
    elif result.failures:
        parts.append(f"{len(result.failures)} failure(s) stand")
    if result.stress:
        parts.append(
            result.stress if result.stress.startswith("no large input")
            else f"largest input {result.stress}"
        )
    if result.note:
        parts.append(result.note)
    return "; ".join(parts) + ("" if result.failures else " -- clean")

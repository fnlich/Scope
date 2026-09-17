"""Regression tests for the custom-miner example.

Run with:  pytest examples/custom_miner

These are kept out of ``tests/`` (the validator's own suite, which
``pyproject.toml`` pins via ``testpaths``) because they exercise the example
and need the ``miner`` extra. What they lock in is the part that is expensive
to get wrong in production:

* the reply a validator receives passes all four of its acceptance checks;
* the self-verify loop actually repairs a wrong answer;
* a solve never outruns its budget and never returns nothing when it has
  something;
* a browser tab that dies is never recycled back into the pool.
"""

from __future__ import annotations

import json
import contextlib
import io
import os
import re
import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_miner import TRUNCATED, CustomMiner, SolveTask, fit_response  # noqa: E402
from solvers import browser_pool as _browser_pool  # noqa: E402
from solvers.browser_pool import (  # noqa: E402
    BLIND_TAB_GRACE_S,
    Browser,
    BrowserFleet,
    Site,
    _STREAM_INSTALL,
    _STREAM_READ,
    _Tab,
    _fenced_blocks,
    usable_busy_selectors,
)
from solvers.chatgpt_web import chatgpt_site  # noqa: E402
from solvers.claude_web import claude_site  # noqa: E402
from solvers.prompts import (  # noqa: E402
    NO_CODE,
    extract_code,
    python_defect,
)
from solvers.verify import (  # noqa: E402
    MAX_PASSES,
    SECOND_OPINION_PASSES,
    Answer,
    VerifyingSolver,
    _Phases,
)

from rlvr.config import Settings  # noqa: E402
from rlvr.neurons.demo_miner import DemoMinerSettings, build_demo_miner_app  # noqa: E402
from rlvr.problemserver.api import derive_request_id  # noqa: E402
from rlvr.protocol import (  # noqa: E402
    SolutionPayload,
    TaskRequest,
    sign_message,
    verify_signature,
)
from rlvr.types import TestCase  # noqa: E402

keypair = pytest.importorskip("bittensor_wallet").Keypair
TestClient = pytest.importorskip("starlette.testclient").TestClient


DIGITS = SolveTask(
    problem_id="req-1",
    language="python",
    statement="Return the sum of the decimal digits of n.",
    entrypoint="g",
    public_examples=[
        {"args": [12345], "kwargs": {}, "expected": 15},
        {"args": [0], "kwargs": {}, "expected": 0},
    ],
    deadline_s=300.0,
)

WRONG = "```python\ndef g(n):\n    s = 0\n    while n > 9:\n        s += n % 10\n        n //= 10\n    return s\n```"
RIGHT = "```python\ndef g(n):\n    s = 0\n    while n > 0:\n        s += n % 10\n        n //= 10\n    return s\n```"
# What turn 1 sends back when a test needs the cases turn to produce cases
# rather than be wasted on a stray code block. `WRONG` FAILS this one -- it
# stops at `n > 9`, so 12345 sums to 14 -- which is what makes a repair round
# in these tests real rather than decorative.
CASES = '```json\n[{"name": "all five digits", "args": [12345], "expected": 15}]\n```'


# Which stage a prompt belongs to, by a marker unique to that prompt. Checked
# against the real builders in `test_the_phase_markers_the_fakes_key_on_are_real`
# -- a marker that stops matching would silently hand one stage's scripted
# reply to another, which is exactly how a defect test once passed its
# candidate reply to the analysis turn and then asserted nothing.
_PHASE_MARKERS = (
    ("repair", "Repair it"),
    ("analysis", "algorithm_sketch"),
    ("inputs", "Invent test INPUTS"),
    ("oracle", "REFERENCE implementation"),
)


def _phase_of(text: str) -> str:
    for phase, marker in _PHASE_MARKERS:
        if marker in text:
            return phase
    return "candidate"


class _Script:
    """The reply list, and a cursor that belongs to the SOLVE.

    It used to live on the conversation, which was the same thing while a
    solve held one. It no longer does: the candidate and the repair are
    different phases and so different conversations, and a per-conversation
    cursor handed the repair reply[0] again -- the wrong answer, for ever.
    """

    def __init__(self, replies):
        self._replies, self._n = list(replies), -1

    def next(self) -> str:
        if not self._replies:
            return ""
        self._n += 1
        return self._replies[min(self._n, len(self._replies) - 1)]


class _Chat:
    """A scripted conversation, keyed on WHAT is being asked rather than when.

    The reply list means "the candidate, then each repair in turn", which is
    what a test writing `_solver([WRONG, RIGHT])` intends. Serving it by call
    order stopped meaning that once a solve opened five conversations: the
    analysis turn would eat reply[0] and the candidate would be handed the
    repair. So the three turns that are not the candidate answer for
    themselves, and the ordered list is spent only on the turns the tests are
    actually about.
    """

    def __init__(self, script, provider="claude"):
        # Several tests build a `_Chat` directly with a plain list. Accept
        # both: a shared `_Script` when the solve owns the cursor, a list when
        # the test only has one conversation to script.
        self._script = script if isinstance(script, _Script) else _Script(script)
        self.provider = provider

    async def send(self, text, timeout_s, extend_to_s=None):
        if _phase_of(text) in ("analysis", "inputs", "oracle"):
            # Nothing to add, no synthesized inputs, no reference: grading
            # falls to the validator's own examples, which is the shape every
            # one of these tests was written against.
            return ""
        return self._script.next()

    async def close(self): pass


class _Backend:
    def __init__(self, replies, provider="claude"):
        self._script, self._provider = _Script(replies), provider
    async def open(self, avoid=None): return _Chat(self._script, self._provider)
    async def aclose(self): pass
    def stats(self): return {}


def _solver(replies, **kw):
    kw.setdefault("reserve_s", 0)
    kw.setdefault("max_budget_s", 120)
    return VerifyingSolver(_Backend(replies), **kw)




class _TwoSeats:
    """A backend that hands out a different conversation per PHASE.

    Records which phase asked for each seat and what each seat was sent, which
    is how a test asserts the property the phases exist for: that the
    candidate was written by a conversation the reference never entered.
    """

    def __init__(self, per_phase, provider="claude"):
        self.per_phase, self.opened, self.sent = per_phase, [], {}

    async def open_for(self, phase=None, avoid=None, timeout_s=None):
        self.opened.append(phase)
        return self._seat(phase)

    async def open(self, avoid=None, timeout_s=None):
        self.opened.append(None)
        return self._seat(None)

    def _seat(self, phase):
        seat = _Chat(self.per_phase.get(phase, self.per_phase[None]),
                     provider=f"claude:{phase or 'ladder'}")
        sent = self.sent.setdefault(phase, [])
        send = seat.send

        async def _record(text, timeout_s, extend_to_s=None):
            sent.append(text)
            return await send(text, timeout_s, extend_to_s)

        seat.send = _record
        return seat

    async def aclose(self): pass
    def stats(self): return {}

def _two_seat_task():
    return SolveTask(
        problem_id="two-seat", language="python",
        statement="Return the sum of the decimal digits of n.", entrypoint="g",
        public_examples=[], deadline_s=120.0,
    )


def _solver_seeing(replies, **kw):
    """A solver whose every conversation shares one script, and the prompts it
    saw. The script is shared across conversations because a solve now opens
    one per phase; a per-conversation cursor would hand each phase reply[0].
    """
    sent: list[str] = []
    script = _Script(replies)

    class _Seen(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            sent.append(text)
            return await super().send(text, timeout_s, extend_to_s)

    class _Fleet:
        async def open(self, avoid=None): return _Seen(script, "claude")
        async def aclose(self): pass
        def stats(self): return {}

    return VerifyingSolver(_Fleet(), **kw), sent

@pytest.fixture(autouse=True)
def _never_archive_into_the_operators_corpus(tmp_path, monkeypatch):
    """Point the solution archive at a scratch directory for EVERY test.

    Autouse and unconditional, because the alternative was measured rather than
    imagined. Two tests here drive the real `/solve` path through a TestClient,
    and `CustomMiner.solve` archives every answer it produces — so running the
    suite wrote its own canned fixtures into `solutions/` beside answers a live
    miner had produced for real validators. Deleting the two files and running
    just those two tests put them straight back: 43 files, then 45.

    That corpus is evidence. It is what an operator reads to find out what their
    miner actually submitted, and a test that quietly adds rows to it makes that
    evidence untrustworthy in a way nobody would think to check. Per-test opt-in
    would have left the same hole open for the next test somebody writes.
    """
    monkeypatch.setenv("SOLVER_SOLUTION_DIR", str(tmp_path / "solutions"))
    # And the solution cache, for the same reason twice over: a test must not
    # write into the operator's corpus, and a test must not READ one either.
    # A cache hit returns an answer without opening a conversation, so an
    # entry left by a previous run -- or by a live miner -- would silently
    # replace whatever a test's scripted backend was about to say.
    monkeypatch.setenv("SOLVER_SOLUTION_CACHE_DIR", str(tmp_path / "cache"))
    # A solve opens one conversation per stage, and the scripted backends here
    # hand every conversation the same `_Script`. That is deliberate and it is
    # why `_Chat` answers the analysis, inputs and reference turns for itself:
    # the ordered reply list then means "the candidate, then each repair",
    # which is what a test writing `_solver([WRONG, RIGHT])` intends. Tests
    # about the other stages build a backend that tells its conversations
    # apart -- see `_TwoSeats` and the content-aware fakes.
    #
    # Grading defaults to Docker, matching the validator's limits. A test host
    # need not have a daemon, and the tests that build a `_Grader` directly are
    # about what the grader DOES with a verdict rather than about which backend
    # produced it -- so they get the backend that always builds. The fallback
    # in `_Grader._build` is covered by a test that asks for it by name.
    monkeypatch.setenv("SOLVER_VERIFY_EXECUTOR", "subprocess")


# --------------------------------------------------------------------------- #
# The wire contract: what the validator actually checks before paying.
# --------------------------------------------------------------------------- #
def test_reply_passes_every_validator_acceptance_check():
    miner_kp, validator_kp = keypair.create_from_uri("//Bob"), keypair.create_from_uri("//Alice")
    prompts: list[str] = []

    class _Recording(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            prompts.append(text)
            return await super().send(text, timeout_s, extend_to_s)

    class _Recorded(_Backend):
        async def open(self, avoid=None):
            return _Recording(self._script, self._provider)

    # WRONG then RIGHT, and the second is the point. `_Chat` answers the
    # analysis, inputs and reference turns for itself, so the ordered list is
    # spent only on the candidate and the repairs -- the turns this test is
    # about. Without that, an earlier version had the analysis turn eat
    # reply[0] and the candidate answer with the repair, and the one test
    # covering the wire path proved nothing about the loop that decides what
    # goes ON that wire.
    solver = VerifyingSolver(
        _Recorded([WRONG, RIGHT]), reserve_s=0, max_budget_s=120
    )
    metagraph = SimpleNamespace(
        hotkeys=[validator_kp.ss58_address], validator_permit=[True], S=[0.0]
    )
    miner = CustomMiner(
        DemoMinerSettings(_env_file=None), solver,
        wallet=SimpleNamespace(hotkey=miner_kp), subtensor=None, metagraph=metagraph,
    )
    request_id = derive_request_id("chal-1", 7, miner_kp.ss58_address)
    request = TaskRequest(
        problem_id=request_id, language="python", statement=DIGITS.statement,
        entrypoint="g",
        public_examples=[TestCase(args=[12345], kwargs={}, expected=15)],
    )
    body = request.model_dump_json().encode()
    headers = sign_message(validator_kp, body, signed_for=miner_kp.ss58_address)
    headers["Content-Type"] = "application/json"

    with TestClient(build_demo_miner_app(miner)) as client:
        response = client.post("/solve", content=body, headers=headers)

    assert response.status_code == 200
    reply_headers = {
        name: response.headers.get(name, "")
        for name in (
            "Epistula-Version", "Epistula-Timestamp", "Epistula-Uuid",
            "Epistula-Signed-By", "Epistula-Signed-For", "Epistula-Request-Signature",
        )
    }
    # 1. signed by the miner hotkey the validator expects
    assert reply_headers["Epistula-Signed-By"] == miner_kp.ss58_address
    # 2. signature verifies and is bound to this validator
    assert verify_signature(
        reply_headers, response.content, expected_signed_for=validator_kp.ss58_address
    )
    payload = SolutionPayload.model_validate_json(response.content)
    # 3. the per-dispatch request id is echoed back unchanged
    assert payload.problem_id == request.problem_id
    # 4. within the per-RESPONSE byte cap. Not the request cap, which is eight
    #    times larger and belongs to the other direction: the validator reads a
    #    bounded number of bytes back and discards the whole response if it runs
    #    over, so checking the wrong one here would pass a reply that is thrown
    #    away on arrival.
    assert len(response.content) <= Settings().miner_max_response_bytes
    # 5. ...and the answer is the REPAIRED one. Both halves are asserted: the
    #    code that shipped, and that a repair round is what produced it.
    assert "while n > 0" in payload.code, payload.code
    # Every stage writes in its own conversation, so this fixture's single
    # prompt list interleaves them and position is not turn order. Ask by what
    # a prompt IS instead: each stage was asked exactly once, and a repair
    # round -- the thing that turned WRONG into RIGHT -- actually ran.
    asked = [_phase_of(p) for p in prompts]
    for stage in ("analysis", "inputs", "oracle", "candidate"):
        assert asked.count(stage) == 1, f"{stage} was asked {asked.count(stage)}x: {asked}"
    assert asked.count("repair") >= 1, f"no repair round was sent: {asked}"


def test_the_correction_ships_on_chain_and_lands_in_the_archive(tmp_path):
    """The rule -- THE LATEST VERSION WINS -- carried through `/solve` to the
    file an operator reads, in the shape live traffic actually has.

    Three things make this the case that matters, and all three are how the
    original bug hid:

    * `public_examples=[]`. All 97 archived requests carry none, so the
      synthesized inputs run against an independently written reference are
      the only bar. That is the path the repair loop runs on in production.
    * THE CORRECTION IS TOO LATE TO GRADE. Grading declines with less than one
      case's worth of budget left (`VERIFY_TIMEOUT_S`), so the repaired
      program arrives carrying no evidence at all, and `Candidate.score` puts
      the same 0 in the self-tests slot for "failed them" and for "was never
      run". The draft is measurably wrong; the correction is merely unmeasured,
      and the rule that tells them apart is THE LATEST VERSION WINS. A
      correction that can be graded is picked by either rule -- only this one
      tells them apart.
    * IT ASSERTS THE ARCHIVED FILE. The report was "the solution file holds
      the draft's code", and that file is written by `save_solution` after
      `fit_response`, two hops past anything a `solve_task`-level test sees.

    Betting on the ungraded correction is right rather than merely safe:
    payment is all-or-nothing, so a program known to fail one of its own cases
    is a certain zero, while an ungraded correction is at worst the same zero
    and was written by a model that had just been shown what was wrong.
    """
    miner_kp = keypair.create_from_uri("//Bob")
    validator_kp = keypair.create_from_uri("//Alice")
    # Inputs only. The reference below is what turns them into expectations.
    inputs = ('```json\n[{"name": "zero", "args": [0]},\n'
              ' {"name": "single", "args": [7]},\n'
              ' {"name": "carry", "args": [12345]}]\n```')
    reference = "```python\ndef g(n):\n    return sum(int(d) for d in str(n))\n```"
    draft = "```python\ndef g(n):\n    return 0\n```"            # agrees on 1 of 3
    fixed = ("```python\ndef g(n):\n    t = 0\n    while n > 0:\n"
             "        t += n % 10\n        n //= 10\n    return t\n```")
    prompts: list[str] = []

    class _Slow(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            prompts.append(text)
            phase = _phase_of(text)
            if phase == "inputs":
                return inputs
            if phase == "oracle":
                return reference
            if phase == "analysis":
                return ""
            if phase == "repair":
                # The correction takes most of the budget to arrive, so by the
                # time it is back there is less than one case's worth of clock
                # left and nothing can be run against it.
                await asyncio.sleep(7.0)
                return fixed
            return draft

    class _SlowBackend(_Backend):
        async def open(self, avoid=None):
            return _Slow(self._script, self._provider)

    miner = CustomMiner(
        DemoMinerSettings(_env_file=None),
        VerifyingSolver(_SlowBackend([]), reserve_s=0,
                        max_budget_s=10, second_opinion=False),
        wallet=SimpleNamespace(hotkey=miner_kp), subtensor=None, metagraph=None,
    )
    request = TaskRequest(
        problem_id="live-shape-1", language="python",
        statement="Return the sum of the decimal digits of n.",
        entrypoint="g", public_examples=[],
    )
    body = request.model_dump_json().encode()
    headers = sign_message(validator_kp, body, signed_for=miner_kp.ss58_address)
    headers["Content-Type"] = "application/json"

    with TestClient(build_demo_miner_app(miner)) as client:
        response = client.post("/solve", content=body, headers=headers)

    assert response.status_code == 200
    payload = SolutionPayload.model_validate_json(response.content)
    # The repair round happened at all. Stages write in their own
    # conversations and finish out of order, so ask what each prompt IS.
    asked = [_phase_of(p) for p in prompts]
    assert asked.count("repair") >= 1, f"no repair round was sent: {asked}"
    for stage in ("analysis", "inputs", "oracle", "candidate"):
        assert asked.count(stage) == 1, f"{stage} was asked {asked.count(stage)}x: {asked}"
    # What went out is phase 3, not the phase-2 draft it corrects.
    assert "while n > 0" in payload.code, (
        f"submitted the program the repair round corrected: {payload.code!r}"
    )

    archived = tmp_path / "solutions" / "live-shape-1.py"
    assert archived.is_file(), (
        f"nothing archived: {sorted((tmp_path / 'solutions').glob('*'))}"
    )
    assert archived.read_text() == payload.code, (
        f"the file is not the submission: {archived.read_text()!r}"
    )


def test_a_long_transcript_is_trimmed_to_what_the_validator_will_read():
    """The validator reads a bounded number of bytes and discards the WHOLE
    response if it runs over. A correct, correctly-signed answer then scores
    zero and neither log says why — the miner sees 200, the validator sees a
    reply it never read. `code` is what gets graded and `raw_response` is the
    transcript kept for the dataset, so the transcript is what gives way."""
    cap = Settings().miner_max_response_bytes
    code = "def g(n):\n    return n"
    payload = fit_response(
        SolutionPayload(problem_id="p-1", code=code, raw_response="x" * (cap * 2))
    )
    assert len(payload.model_dump_json().encode()) <= cap
    assert payload.code == code, "the graded field must never be trimmed to fit"
    assert payload.raw_response.endswith(TRUNCATED)


def test_a_chatty_model_still_produces_a_reply_the_validator_accepts():
    """The same thing end to end, because the cap applies to the serialized
    payload rather than to any field the solver can see."""
    miner_kp = keypair.create_from_uri("//Bob")
    validator_kp = keypair.create_from_uri("//Alice")
    cap = Settings().miner_max_response_bytes
    rambling = "I will explain at length. " * (cap // 10) + RIGHT
    miner = CustomMiner(
        DemoMinerSettings(_env_file=None), _solver([rambling]),
        wallet=SimpleNamespace(hotkey=miner_kp), subtensor=None, metagraph=None,
    )
    request = TaskRequest(
        problem_id="chatty-1", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[TestCase(args=[12345], kwargs={}, expected=15)],
    )
    body = request.model_dump_json().encode()
    headers = sign_message(validator_kp, body, signed_for=miner_kp.ss58_address)
    headers["Content-Type"] = "application/json"

    with TestClient(build_demo_miner_app(miner)) as client:
        response = client.post("/solve", content=body, headers=headers)

    assert response.status_code == 200
    assert len(response.content) <= cap, "the validator would discard this unread"
    reply_headers = {
        name: response.headers.get(name, "")
        for name in (
            "Epistula-Version", "Epistula-Timestamp", "Epistula-Uuid",
            "Epistula-Signed-By", "Epistula-Signed-For", "Epistula-Request-Signature",
        )
    }
    # Trimming happens before signing, so the signature covers what is sent.
    assert verify_signature(
        reply_headers, response.content, expected_signed_for=validator_kp.ss58_address
    )
    payload = SolutionPayload.model_validate_json(response.content)
    assert "while n > 0" in payload.code, "the answer survived the trim intact"


# --------------------------------------------------------------------------- #
# The value-add: self-grading against the public examples repairs a wrong answer.
# --------------------------------------------------------------------------- #
def test_verify_loop_repairs_a_wrong_answer():
    result = asyncio.run(_solver([WRONG, RIGHT]).solve_task(DIGITS, timeout_s=120))
    assert "while n > 0" in result.code


def test_a_reply_with_no_code_is_rejected_and_retried():
    result = asyncio.run(_solver(["Sure! Here's the approach...", RIGHT]).solve_task(DIGITS, 120))
    assert "while n > 0" in result.code


def test_a_verified_answer_is_cached_by_statement():
    solver = _solver([RIGHT])
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert solver.stats()["solver"]["cache_hits"] == 1


def test_a_zero_cache_size_disables_caching_without_crashing():
    """`try_solver.py --repeat` sets `_cache_size = 0` so each round really
    drives the browser; a cached answer would make the flag a no-op and show
    nothing. Zero must therefore mean OFF rather than "evict every time", which
    on an empty dict would raise."""
    solver = _solver([RIGHT])
    solver._cache_size = 0
    for _ in range(3):
        assert asyncio.run(solver.solve_task(DIGITS, 60.0)).verified
    assert solver.stats()["solver"].get("cache_hits", 0) == 0
    assert solver._cache == {}


def test_an_unverified_answer_is_not_cached():
    solver = _solver([WRONG])
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert solver.stats()["solver"]["cache_hits"] == 0


# --------------------------------------------------------------------------- #
# Budget: a correct answer after the cutoff pays exactly the same as a wrong one.
# --------------------------------------------------------------------------- #
class _SlowBackend:
    async def open(self, avoid=None):
        class Slow:
            async def send(self, text, timeout_s):
                # A slow model that honours the slice, as every real backend
                # does: the slice is a ceiling on the read.
                await asyncio.sleep(min(4.0, timeout_s))
                return WRONG
            async def close(self): pass
        return Slow()
    async def aclose(self): pass
    def stats(self): return {}


def test_a_solve_never_outruns_its_advertised_deadline():
    # One second of reserve: with none at all the loop is allowed to run to the
    # very last instant of the budget, which is the point of having no floor.
    solver = VerifyingSolver(_SlowBackend(), max_attempts=9, reserve_s=1, max_budget_s=30)
    started = time.monotonic()
    result = asyncio.run(solver.solve_task(DIGITS, timeout_s=30))
    assert time.monotonic() - started < 30
    assert result.code.strip()  # best-so-far, never nothing when we have something


def test_the_safety_margin_is_held_back_from_the_advertised_deadline():
    solver = VerifyingSolver(_SlowBackend(), max_attempts=9, reserve_s=15, max_budget_s=240)
    started = time.monotonic()
    asyncio.run(solver.solve_task(DIGITS, timeout_s=40))
    assert time.monotonic() - started < 30  # 40s advertised minus a 15s margin


def test_a_dead_backend_yields_an_empty_answer_rather_than_an_exception():
    class Broken:
        async def open(self): raise RuntimeError("browser died")
        async def aclose(self): pass
        def stats(self): return {}
    result = asyncio.run(
        VerifyingSolver(Broken(), reserve_s=0, max_budget_s=60).solve_task(DIGITS, 60)
    )
    assert isinstance(result, Answer) and result.code == ""


# --------------------------------------------------------------------------- #
# Pool health: recycling a dead tab would fail every request leased onto it.
# --------------------------------------------------------------------------- #
class _DeadPage:
    def __getattr__(self, name):
        def boom(*args, **kwargs):
            raise RuntimeError("Target page, context or browser has been closed")
        return boom


async def _done(value):
    """An already-resolved awaitable, for stubbing out async methods."""
    return value


def _fleet(*sites, tabs_per_browser: int = 2) -> BrowserFleet:
    """A real fleet, no browser and no network — __init__ does no I/O."""
    sites = sites or (chatgpt_site(),)
    browsers = [
        Browser(f"http://127.0.0.1:{9222 + i}", s) for i, s in enumerate(sites)
    ]
    return BrowserFleet(browsers, tabs_per_browser=tabs_per_browser)


def _pool(replaceable: bool, site=None) -> BrowserFleet:
    """A real fleet with a stubbed _spawn.

    Built through the real constructor on purpose. An earlier version assembled
    the object with __new__ and set four attributes by hand, which silently went
    stale the moment it grew a fifth; the resulting AttributeError was then
    swallowed by the solver's catch-all and surfaced as a confusing count.
    """
    pool = _fleet(site or chatgpt_site())
    pool._size = 1                       # pretend one tab was spawned at startup

    async def spawn(context, browser, label):
        if not replaceable:
            return None
        tab = _Tab(pool, _DeadPage(), context, f"{label}-new", site=browser.site)
        return tab

    pool._spawn = spawn
    return pool


# Both browser backends share one pool implementation, so the dead-tab fix has
# to be proven for both — that sharing is the reason it exists only once.
SITES = pytest.mark.parametrize(
    "site", [chatgpt_site(), claude_site()], ids=["chatgpt", "claude"]
)


async def _use_dead_tab(pool: BrowserFleet, leased: list = None) -> None:
    await pool._free.put(_Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site()))
    handed = leased if leased is not None else []

    class LeaseOnly:
        # Leases exactly as BrowserPool.open() does, minus tab.start() (which
        # would fail first on a dead page). Setting `leased` matters: release()
        # ignores a tab that was never leased, so skipping it here would make
        # the test assert against a no-op.
        async def open(self, avoid=None):
            # `get_nowait`, not `get`. A solve opens one conversation per
            # phase, and a fleet with nothing left to lease raises rather than
            # blocking for ever -- blocking here made the pass wait out its
            # whole budget per phase instead of degrading.
            try:
                tab = pool._free.get_nowait()
            except asyncio.QueueEmpty:
                raise RuntimeError("no tab available") from None
            tab.leased = True
            handed.append(id(tab))
            return tab

        async def aclose(self): pass
        def stats(self): return pool.stats()

    await VerifyingSolver(
        LeaseOnly(), max_attempts=1, reserve_s=0, max_budget_s=30,
        second_opinion=False,   # this is about tab replacement, not two models
    ).solve_task(DIGITS, timeout_s=30)
    await _settle(pool)


async def _settle(pool: BrowserFleet) -> None:
    """Wait for the fleet's background replacements to land.

    `release()` deliberately does not wait for one (it runs in the solver's
    `finally`, past the deadline). Production has an event loop that keeps
    running afterwards; a test that ends at `asyncio.run` does not, so the wait
    has to be explicit here or these assertions would be measuring scheduling
    luck rather than the fleet.
    """
    while pool._pending:
        await asyncio.gather(*list(pool._pending), return_exceptions=True)


@SITES
def test_a_tab_that_dies_is_replaced_not_recycled(site):
    """The invariant is RECYCLING, asserted by identity rather than by a count.

    It used to read `_lost == 1`, which held only while a solve leased exactly
    one tab. A solve now opens one conversation per phase, and the stubbed
    `_spawn` hands back another dead page, so more than one tab legitimately
    dies. What must never happen -- and what the fix was for -- is the same
    dead tab coming back out of the free queue to fail a second request.
    """
    pool = _pool(replaceable=True, site=site)
    handed: list = []
    asyncio.run(_use_dead_tab(pool, handed))

    assert len(handed) == len(set(handed)), (
        f"a dead tab was leased twice: {handed}"
    )
    assert pool._lost >= 1, "the dead tab was never noticed"
    assert pool._lost == len(handed), (
        f"{len(handed)} tab(s) leased but {pool._lost} recorded lost"
    )
    # The fleet neither shrank nor leaked: one tab in, one replacement waiting.
    assert pool._size == 1 and pool._free.qsize() == 1


@SITES
def test_an_unreplaceable_dead_tab_retires_instead_of_poisoning_the_pool(site):
    """A tab that cannot be replaced is RETIRED, not put back. The fleet ends
    smaller rather than holding a tab that fails every request leased onto
    it."""
    pool = _pool(replaceable=False, site=site)
    asyncio.run(_use_dead_tab(pool))
    assert pool._lost == 1 and pool._size == 0 and pool._free.qsize() == 0


def test_the_submit_phase_is_bounded_so_playwright_cannot_overrun_the_budget():
    import inspect

    send = inspect.getsource(_Tab.send)
    submit = inspect.getsource(_Tab._submit)
    # Playwright auto-waits 30s per action by default; unbounded that is ~90s
    # of overrun on a budget the solver carefully computed.
    assert "asyncio.wait_for" in send
    # Every DOM action on the way to a send carries the bound: two clicks and a
    # fill while clearing the box, one click to send. Clearing moved into its
    # own method, so the whole path is checked rather than `_submit` alone.
    clear = inspect.getsource(_Tab._clear_composer)
    assert submit.count("timeout=ui_ms") == 1, submit
    assert clear.count("timeout=ui_ms") == 3, clear
    for line in (submit + clear).splitlines():
        stripped = line.strip()
        if stripped.startswith("await") and (".click(" in stripped or ".fill(" in stripped):
            assert "timeout=ui_ms" in stripped, f"unbounded DOM action: {stripped}"


def test_the_pool_starts_lazily_so_any_host_can_serve_it():
    """Regression: start() was once only called by a launcher that needs a live
    chain. Hosted anywhere else the fleet stayed empty and open() blocked on the
    queue forever, so every solve returned nothing."""
    import inspect

    assert "await self.start()" in inspect.getsource(BrowserFleet.open)
    start = inspect.getsource(BrowserFleet.start)
    assert "_start_lock" in start and "if self._started" in start, "start must be idempotent"


def test_starting_twice_connects_only_once():
    import asyncio

    pool = _fleet()
    calls = []

    async def fake_connect():
        calls.append(1)

    pool._connect = fake_connect

    async def go():
        await pool.start()   # an explicit start, as run_miner.py does
        await pool.start()   # a second call, as a lazy start from open() would be

    asyncio.run(go())
    assert calls == [1], "start() must connect once no matter how often it is called"


# --------------------------------------------------------------------------- #
# Multi-provider chain
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Linux only.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "platform,name", [("win32", "Windows"), ("cygwin", "Windows"), ("darwin", "macOS")]
)
def test_a_non_linux_host_is_refused_with_the_reason(monkeypatch, platform, name):
    """A Windows install dies compiling bittensor-wallet through Rust, three
    layers below anything this repo wrote. One clear line beats that."""
    import preflight

    monkeypatch.setattr(preflight.sys, "platform", platform)
    with pytest.raises(SystemExit) as raised:
        preflight.require_linux("The custom miner")
    message = str(raised.value)
    assert name in message and "bittensor-wallet" in message
    assert "WSL2" in message, "Windows users need to be told where to go"


def test_wsl2_and_any_linux_pass(monkeypatch):
    """WSL2 is real Linux to Python and to Chrome; it must not be refused.

    monkeypatch, not assignment: `preflight.sys` IS the stdlib sys module, so a
    bare assignment would change sys.platform for the whole test session.
    """
    import preflight

    for platform in ("linux", "linux2"):
        monkeypatch.setattr(preflight.sys, "platform", platform)
        preflight.require_linux()  # must not raise


def test_every_entrypoint_checks_the_platform_before_importing_anything_heavy():
    """A guard only one launcher calls is a guard nobody has — and one that runs
    after `import rlvr` never speaks at all, because on a Windows box that
    import is what already failed."""
    from pathlib import Path

    root = Path(__file__).resolve().parent
    for name in ("custom_miner.py", "run_miner.py"):
        body = (root / name).read_text()
        assert "require_linux(" in body, name
        guard = body.index('require_linux("')
        for heavy in ("import httpx", "from rlvr", "from custom_miner"):
            if heavy in body:
                assert guard < body.index(heavy), f"{name}: guard runs after {heavy}"
    assert "require_linux(" in (root / "solvers" / "doctor.py").read_text()


def test_the_debug_browser_script_is_linux_only_and_keeps_cdp_on_loopback():
    """That debug port is full control of a browser holding your logged-in
    sessions, on a box already exposing a public axon port."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent / "scripts" / "start_debug_browser.sh"
    ).read_text()
    assert "uname -s" in script and "Linux" in script
    # The default binds to loopback; naming the address flag would undo that.
    # It appears once, in the comment warning never to use it.
    assert script.count("--remote-debugging-address") == 1
    assert "NEVER add --remote-debugging-address" in script
    assert "--no-sandbox" in script   # needed when running as root
    assert "xvfb-run" in script       # a headless host has no screen


# --------------------------------------------------------------------------- #
# Fake page objects. There is no browser in CI, so these stand in for Playwright
# where the logic under test is ours rather than the DOM's. `_Loc.first` exists
# because real locators have it and the click path relies on it.
# --------------------------------------------------------------------------- #
class _Node:
    def __init__(self, text="", code=(), attrs=None):
        self._text, self._code, self._attrs = text, list(code), dict(attrs or {})

    def locator(self, selector):
        inner = [_Node(text=c) for c in self._code] if selector == "pre code" else []
        return _Loc(None, selector, inner)

    async def inner_text(self):
        return self._text

    async def text_content(self):
        """`_read` asks code blocks for this; see the note there on why."""
        return self._text

    async def get_attribute(self, name):
        return self._attrs.get(name)


class _Loc:
    def __init__(self, page, selector, nodes):
        self._page, self._selector, self._nodes = page, selector, list(nodes)

    @property
    def first(self):
        """Playwright locators have this; clicks use it to avoid strict mode."""
        return _Loc(self._page, self._selector, self._nodes[:1])

    async def count(self):
        return len(self._nodes)

    def nth(self, index):
        return self._nodes[index]

    async def click(self, timeout=None):
        if not self._nodes:
            raise RuntimeError(f"nothing to click for {self._selector}")
        self._page.clicked.append(self._selector)
        if self._page.on_click:
            self._page.on_click(self._selector)

    async def evaluate(self, expression):
        """Only ever asked one thing: what does the composer hold."""
        if self._page.composer_unreadable:
            raise RuntimeError("the page will not say what the composer holds")
        return self._page.composer

    async def fill(self, value, timeout=None):
        self._page.filled.append(value)
        if not self._page.composer_unclearable:
            self._page.composer = value


class _FakePage:
    """A page whose DOM is `{selector: [nodes]}` and can change on submit."""

    def __init__(self, dom, on_click=None, composer=""):
        self.dom, self.on_click = dom, on_click
        self.typed, self.pressed, self.clicked = [], [], []
        # What the composer BOX holds, as opposed to what we tried to type into
        # it. A shared account can arrive with somebody's draft already there.
        self.composer = composer
        self.filled = []
        self.composer_unclearable = False
        self.composer_unreadable = False
        # Set to mangle what lands in the box, to stand in for an editor that
        # does not take the text verbatim.
        self.on_insert = None
        # Every navigation and close, so "did this reload?" and "did this throw
        # the tab away?" are assertable rather than inferred.
        self.navigated, self.closed = [], False
        self.on_goto = None

        page = self

        class _Keyboard:
            async def insert_text(self, text):
                page.typed.append(text)
                # At the caret, which is what makes a leftover draft dangerous:
                # the fake appends, the real thing can splice into the middle.
                page.composer += text
                if page.on_insert:
                    page.composer = page.on_insert(page.composer)

            async def press(self, key):
                page.pressed.append(key)
                if key == "Delete" and "Control+A" in page.pressed:
                    if not page.composer_unclearable:
                        page.composer = ""
                # Only a key that SUBMITS moves the conversation on. Control+A
                # and Delete are the composer being cleared, and firing the
                # page's on_click for them would have the reply arrive before
                # the prompt was sent.
                if key == "Enter" and page.on_click:
                    page.on_click(key)

        self.keyboard = _Keyboard()

    def locator(self, selector):
        return _Loc(self, selector, self.dom.get(selector, []))

    async def goto(self, url, wait_until=None):
        self.navigated.append(url)
        if self.on_goto:
            self.on_goto(url)

    async def close(self):
        self.closed = True


class _SoloPool:
    def __init__(self, site):
        self.site = site

    async def release(self, tab):
        pass


def _site(**kw) -> Site:
    base = dict(
        name="t", env_prefix="T", url="about:blank", composer=("#composer",),
        send=("#send",), busy=(), assistant=("#assistant",), poll_s=0.01,
    )
    base.update(kw)
    return Site(**base)


def _tab(page, site) -> _Tab:
    return _Tab(_SoloPool(site), page, None, "probe", site, composer="#composer")


# --- one tab, opened once, reused for every task ------------------------- #
# The tab is the expensive object here: it is signed in by hand and warm. So a
# tab is opened once and kept, and each task is separated from the last by a
# fresh CONVERSATION, not a fresh tab. `_Tab.start` does that in three tiers --
# already-empty, the site's new-chat control, a reload -- and these pin all
# three, plus the guard that keeps tier 2 from bleeding context.


def _chat_page(new_chat: bool = True) -> _FakePage:
    dom = {"#composer": [_Node()], "#send": [_Node()], "#assistant": []}
    if new_chat:
        dom["#newchat"] = [_Node()]
    return _FakePage(dom)


def _answers(page, *, clears: bool = True):
    """Click handler: sending produces a reply, new-chat clears the transcript."""

    def handler(selector):
        if selector == "#send":
            page.dom["#assistant"] = [_Node(code=["def f():\n    return 1"])]
        elif selector == "#newchat" and clears:
            page.dom["#assistant"] = []

    return handler


def test_the_first_task_on_a_new_tab_does_not_reload_the_page():
    """`_spawn` builds a tab only after loading the site's new-conversation URL
    and seeing the composer, so the tab arrives empty. Reloading it before the
    first task is a full app boot spent to reach the state the page is already
    in — paid once per tab at startup, on the first task's clock."""
    page = _chat_page()
    asyncio.run(_tab(page, _site(new_chat=("#newchat",))).start())
    assert page.navigated == []
    assert page.clicked == []


def test_a_later_task_gets_a_new_chat_without_reloading_the_page():
    """Tier 2. Once a task has run the transcript must go, but reloading throws
    away a booted SPA to reach a state the app can route to itself."""
    page = _chat_page()
    tab = _tab(page, _site(new_chat=("#newchat",)))
    page.on_click = _answers(page)
    assert asyncio.run(tab.send("first task", 2.0))       # tab is now dirty

    asyncio.run(tab.start())
    assert "#newchat" in page.clicked
    assert page.navigated == []


def test_a_new_chat_that_leaves_the_transcript_behind_falls_back_to_a_reload():
    """The failure tier 2 must never cause. A new-chat control that did not
    route — changed DOM, a modal in the way, a disabled button — leaves the last
    task's transcript in place, and the next answer comes back promptly and
    quietly wrong. So the click is never trusted: the transcript is checked, and
    anything short of proof pays for the reload."""
    page = _chat_page()
    tab = _tab(page, _site(new_chat=("#newchat",)))
    page.on_click = _answers(page)
    assert asyncio.run(tab.send("first task", 2.0))
    page.on_click = _answers(page, clears=False)          # the click does nothing

    asyncio.run(tab.start())
    assert page.navigated == ["about:blank"]


def test_a_site_with_no_new_chat_control_still_gets_a_fresh_conversation():
    """Tier 3 alone. These selectors are candidate lists against markup nobody
    publishes, so 'none of them matched' is a state to design for, not an
    accident: it costs the reload it always cost, never correctness."""
    page = _chat_page(new_chat=False)
    tab = _tab(page, _site())
    page.on_click = _answers(page)
    assert asyncio.run(tab.send("first task", 2.0))

    asyncio.run(tab.start())
    assert page.navigated == ["about:blank"]


def test_a_tab_that_looked_fresh_but_went_stale_is_reset_not_trusted():
    """A tab can idle for hours between tasks, and the site may reload or
    redirect the page underneath it. The freshness flag would then describe a
    page that no longer exists, and submitting into it wastes a whole task to
    learn that. One count of the composer is cheap enough to spend every time."""
    page = _chat_page(new_chat=False)
    page.dom["#composer"] = []                            # the app moved on
    page.on_goto = lambda _: page.dom.__setitem__("#composer", [_Node()])
    asyncio.run(_tab(page, _site()).start())
    assert page.navigated == ["about:blank"]


def test_a_new_chat_selector_that_raises_falls_back_instead_of_escaping():
    """`open()` retires a tab whose `start()` raised, on the promise that
    `start()` marked it dead first — and only the reload does that. So every
    other tier has to swallow its own failures, selector resolution included: a
    raise from the fast path would reach `open()` with the tab still flagged
    alive, and it would be requeued. That is the recycled-dead-tab failure the
    pool exists to prevent, and it would come back through the door added to
    make the pool faster."""

    class _Raising(_FakePage):
        def locator(self, selector):
            if selector == "#boom":
                raise RuntimeError("Execution context was destroyed")
            return super().locator(selector)

    page = _Raising({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    tab = _tab(page, _site(new_chat=("#boom",)))
    page.on_click = _answers(page)
    assert asyncio.run(tab.send("first task", 2.0))

    asyncio.run(tab.start())                              # must not raise
    assert page.navigated == ["about:blank"]
    assert tab.alive is True


def test_a_tab_is_opened_once_and_never_closed_between_tasks():
    """The whole lifecycle, end to end: two tasks, one page, no reload, no
    close. It matters that tab churn is NOT what ordinary work looks like —
    the fleet replaces a tab only when it dies, so a tab closing is the signal
    that something is wrong, and a design that closed one per task would hide
    every real failure in the noise."""
    site = _site(new_chat=("#newchat",))
    fleet = _fleet(site)
    pages = []

    async def spawn(context, browser, label):
        page = _chat_page()
        page.on_click = _answers(page)
        pages.append(page)
        return _Tab(fleet, page, context, label, browser.site, composer="#composer")

    fleet._spawn = spawn

    async def two_tasks():
        fleet._started = True                             # no browser to attach to
        fleet._free.put_nowait(await spawn(None, fleet._browsers_wanted[0], "t#1"))
        fleet._size = 1
        for _ in range(2):
            tab = await fleet.open()
            assert await tab.send("solve it", 2.0)
            await tab.close()

    asyncio.run(two_tasks())
    assert len(pages) == 1, "a second tab was opened; the first should be reused"
    assert pages[0].closed is False, "the tab was closed between tasks"
    assert pages[0].navigated == [], "the page was reloaded between tasks"


def test_tabs_are_enqueued_interleaved_across_browsers():
    """Two tasks arriving together must land on two different accounts. Enqueuing
    browser-by-browser would give both to the first account, which is the one
    thing that actually rate-limits."""
    async def go():
        fleet = _fleet(claude_site(), chatgpt_site(), tabs_per_browser=2)
        made = []

        async def spawn(context, browser, label):
            tab = _Tab(fleet, _DeadPage(), None, label, browser.site)
            made.append(tab)
            return tab

        fleet._spawn = spawn
        fleet._attach = lambda browser: _done(object())
        fleet._reclaim = lambda context, browser: _done(None)
        fleet._pw = object()
        await fleet._fill()
        return [t.label for t in list(fleet._free._queue)]

    order = asyncio.run(go())
    # A#1, B#1, A#2, B#2 — not A#1, A#2, B#1, B#2.
    assert [l.split("#")[1] for l in order] == ["1", "1", "2", "2"], order
    assert order[0].split(":")[1].startswith("9222")
    assert order[1].split(":")[1].startswith("9223")


# --- the browser rotation ------------------------------------------------ #
# Request 1 to the 1st browser, 2 to the 2nd, ... n to the nth, n+1 back to the
# 1st. The account is what rate-limits, so this is the whole reason for running
# several browsers rather than several tabs in one.


async def _filled(n_browsers: int, tabs: int = 2, attach_fails=()) -> BrowserFleet:
    """A fleet through the REAL `_fill`, so `_order` is what production sets.

    Building the queue by hand instead would test the rotation against a fleet
    state no miner ever has, which is how the pre-fleet doctor call rotted.
    """
    sites = [claude_site(), chatgpt_site()]
    fleet = _fleet(*[sites[i % 2] for i in range(n_browsers)], tabs_per_browser=tabs)

    async def spawn(context, browser, label):
        return _Tab(
            fleet, _DeadPage(), None, label, browser.site, source=browser.endpoint
        )

    fleet._spawn = spawn
    fleet._attach = lambda b: _done(None if b.endpoint in attach_fails else object())
    fleet._reclaim = lambda context, browser: _done(None)
    fleet._pw = object()
    await fleet._fill()
    return fleet


def _port(tab) -> str:
    return tab.source.rsplit(":", 1)[-1]


async def _take(fleet, avoid=None):
    tab = await fleet._lease(avoid, 5)
    tab.leased = True                    # release() ignores a tab never leased
    return tab


def test_requests_go_round_the_browsers_in_order():
    """The plain case: one task at a time, three browsers, nine requests."""

    async def go():
        fleet = await _filled(3)
        seen = []
        for _ in range(9):
            tab = await _take(fleet)
            seen.append(_port(tab))
            await fleet.release(tab)
        return seen

    assert asyncio.run(go()) == ["9222", "9223", "9224"] * 3


def test_the_rotation_survives_tasks_finishing_out_of_order():
    """The case a queue alone gets wrong, and the reason the cursor exists.

    A tab is freed when its task ENDS, and concurrent tasks end in whatever
    order the models happen to answer. Take the next free tab and the sequence
    stops being a rotation within a few requests — two hard problems on one
    account and an easy one on another, and the fast account starts taking more
    than its share of the tasks. Here the second of each pair always finishes
    first, which is enough to scramble a queue and must not scramble this.
    """

    async def go():
        fleet = await _filled(3)                    # 3 browsers, 2 tabs each
        seen = []
        for _ in range(6):
            first = await _take(fleet)
            second = await _take(fleet)
            seen += [_port(first), _port(second)]
            await fleet.release(second)             # out of order, deliberately
            await fleet.release(first)
        return seen

    assert asyncio.run(go()) == ["9222", "9223", "9224"] * 4


def test_a_busy_browser_passes_its_turn_instead_of_stalling_the_fleet():
    """The one deliberate deviation. A miner is paid for answers that beat the
    deadline, so waiting for the browser whose turn it is while another sits
    free would trade money for a tidier sequence."""

    async def go():
        fleet = await _filled(3, tabs=1)
        held = [await _take(fleet) for _ in range(3)]
        assert [_port(t) for t in held] == ["9222", "9223", "9224"]
        await fleet.release(held[1])                # only 9223 is free again
        # It is 9222's turn, but 9222 is still working. Take 9223 rather than
        # wait, and resume the rotation from there.
        return _port(await _take(fleet))

    assert asyncio.run(go()) == "9223"


def test_a_browser_that_never_attached_is_not_in_the_rotation():
    """`n` is the number of browsers actually serving. One that could not be
    attached to — not started, wrong port — is not one of them, and leaving it
    in the ring would spend every nth turn discovering that again."""

    async def go():
        fleet = await _filled(3, attach_fails={"http://127.0.0.1:9223"})
        seen = []
        for _ in range(4):
            tab = await _take(fleet)
            seen.append(_port(tab))
            await fleet.release(tab)
        return seen, fleet._order

    seen, order = asyncio.run(go())
    assert seen == ["9222", "9224", "9222", "9224"]
    assert order == ["http://127.0.0.1:9222", "http://127.0.0.1:9224"]


def test_the_second_opinion_outranks_the_rotation():
    """`avoid` is asked for only after the first model failed to produce a
    verifiable answer, and reaching the OTHER model is the entire value of that
    attempt. Spending it on the same model to keep the browsers in order would
    be the wrong trade."""

    async def go():
        fleet = await _filled(2, tabs=1)            # 9222 claude, 9223 chatgpt
        tab = await _take(fleet, avoid="claude")    # 9222's turn, but avoid it
        return _port(tab), tab.site.name

    assert asyncio.run(go()) == ("9223", "chatgpt")


def test_a_lease_can_ask_for_a_different_provider():
    """The second opinion is only worth asking if it reaches the OTHER model."""
    async def go():
        fleet = _fleet()
        claude = _Tab(fleet, _DeadPage(), None, "c#1", claude_site())
        gpt = _Tab(fleet, _DeadPage(), None, "g#1", chatgpt_site())
        fleet._free.put_nowait(claude)
        fleet._free.put_nowait(gpt)
        first = await fleet._lease(None, 1.0)
        second = await fleet._lease(avoid=first.provider, wait_s=1.0)
        return first.provider, second.provider

    first, second = asyncio.run(go())
    assert first == "claude" and second == "chatgpt"


def test_avoiding_a_provider_still_returns_a_tab_when_it_is_the_only_one():
    """A preference, not a guarantee: one of the avoided provider's tabs still
    beats failing the task."""
    async def go():
        fleet = _fleet()
        fleet._free.put_nowait(_Tab(fleet, _DeadPage(), None, "c#1", claude_site()))
        fleet._free.put_nowait(_Tab(fleet, _DeadPage(), None, "c#2", claude_site()))
        tab = await fleet._lease(avoid="claude", wait_s=1.0)
        return tab.provider, fleet._free.qsize()

    provider, left = asyncio.run(go())
    assert provider == "claude" and left == 1, "it must not drop or duplicate tabs"


def test_a_second_opinion_asks_the_other_model_only_when_the_first_fails():
    """A verified answer must end it — the second model costs a real account's
    quota and a tab another task could be using."""
    seen = []

    class _Fleet:
    # One script per PASS, not per open: a pass opens one conversation
    # per phase, so indexing by the number of opens ran off the end of a
    # two-element list. `avoid` is what distinguishes the second pass.
        def __init__(self, replies):
            self._scripts = [_Script(r) for r in replies]
        async def open(self, avoid=None):
            provider = "chatgpt" if avoid == "claude" else "claude"
            seen.append(provider)
            which = min(1 if avoid else 0, len(self._scripts) - 1)
            return _Chat(self._scripts[which], provider)
        async def aclose(self): pass
        def stats(self): return {}

    # First model gets it right: one provider asked.
    seen.clear()
    solver = VerifyingSolver(_Fleet([[RIGHT], [RIGHT]]), reserve_s=0, max_budget_s=120)
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified and set(seen) == {"claude"}, seen
    assert solver.stats()["providers"]["claude"]["verified"] == 1

    # First model keeps failing: the other one is asked and wins.
    seen.clear()
    solver = VerifyingSolver(
        _Fleet([[WRONG, WRONG, WRONG], [RIGHT]]), reserve_s=0, max_budget_s=120
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified and list(dict.fromkeys(seen)) == ["claude", "chatgpt"], seen
    assert solver.stats()["providers"]["chatgpt"]["verified"] == 1


def test_an_ungradeable_task_does_not_pay_for_a_second_opinion():
    """Live traffic ships tasks with no public examples, and then the second
    model's answer can never win: `verified` needs total > 0, and `score` ties
    at (0, has_code). Asking anyway spends a second account's quota and doubles
    the latency to produce an answer that is discarded on return."""
    calls: list[str] = []

    class _Counting(_Backend):
        async def open(self, avoid=None):
            calls.append(avoid or "first")
            return _Chat([RIGHT], "chatgpt" if avoid == "claude" else "claude")

    task = SolveTask(
        problem_id="none", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[], deadline_s=120.0,
    )
    solver = VerifyingSolver(
        _Counting([RIGHT]), reserve_s=0, max_budget_s=120, second_opinion=True
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))
    assert answer.code, "the answer still comes back"
    assert answer.verified is False, "nothing can verify it, and it must not claim to"
    assert set(calls) == {"first"}, f"asked a second model for nothing: {calls}"


def test_an_ungradeable_task_still_buys_a_second_opinion_when_the_first_is_empty():
    """The exception that makes the rule safe, and the case a live log caught.

    Two ungradeable answers cannot be told apart -- unless one of them is
    EMPTY. `score` is (passed, has_code), so (0,1) beats (0,0): the other model
    is the only remaining chance at the whole payment. Skipping it because the
    task happens to ship no examples turns a recoverable submit failure into a
    guaranteed zero.
    """
    calls: list[str] = []

    class _Silent(_Backend):
        async def open(self, avoid=None):
            calls.append(avoid or "first")
            # The first model's tab failed to submit, so send() returns "".
            return _Chat([""] if avoid is None else [RIGHT],
                         "chatgpt" if avoid else "claude")

    task = SolveTask(
        problem_id="none-empty", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[], deadline_s=120.0,
    )
    solver = VerifyingSolver(
        _Silent([""]), reserve_s=0, max_budget_s=120, second_opinion=True
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))
    assert set(calls) == {"first", "claude"}, f"never fell back: {calls}"
    assert "while n > 0" in answer.code, "the fallback answer was not used"


def test_the_second_opinion_can_be_turned_off():
    """Pure throughput: never spend a second account on one task."""
    seen = []

    class _Fleet:
        async def open(self, avoid=None):
            seen.append(avoid)
            return _Chat([WRONG], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=1, reserve_s=0, max_budget_s=120, second_opinion=False
    )
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert set(seen) == {None}, seen


def test_shutdown_is_armed_before_the_slow_attach_not_after():
    """A supervisor restarting while the miner is attaching to eight browsers
    would otherwise raise KeyboardInterrupt straight through the cleanup and
    leave the tabs already opened behind."""
    from pathlib import Path

    serve = Path(__file__).resolve().parent.joinpath("run_miner.py").read_text()
    body = serve[serve.index("async def serve()"):serve.index("asyncio.run(serve())")]
    handler = body.index("add_signal_handler")
    attach = body.index("await warm_up(")
    assert handler < attach, "signal handlers must be installed before attaching"
    # Both signals a supervisor uses, and cleanup that a second one cannot cut short.
    assert "signal.SIGINT" in body and "signal.SIGTERM" in body
    assert "asyncio.shield" in body
    assert "solver.aclose()" in body


def test_a_restart_reclaims_the_tabs_its_predecessor_left_behind():
    """A supervised miner gets SIGKILLed sooner or later, and an unclean exit
    never runs shutdown. Without this, every restart adds dead tabs to a browser
    that stays up for weeks."""
    from solvers.browser_pool import TAB_MARK

    closed = []

    class _Page:
        def __init__(self, name): self._name = name
        async def evaluate(self, script): return self._name
        async def close(self): closed.append(self._name)

    class _Unreadable(_Page):
        async def evaluate(self, script): raise RuntimeError("page is gone")

    class _Ctx:
        pages = [
            _Page(""),                       # your own tab: never touched
            _Page(f"{TAB_MARK}/claude"),     # ours, from a previous run
            _Page(f"{TAB_MARK}/chatgpt"),    # ours
            _Unreadable("weird"),            # cannot be asked -> not ours
        ]

    async def go():
        fleet = _fleet()
        await fleet._reclaim(_Ctx(), Browser("http://127.0.0.1:9222", claude_site()))
        return fleet._reclaimed

    count = asyncio.run(go())
    assert closed == [f"{TAB_MARK}/claude", f"{TAB_MARK}/chatgpt"], closed
    assert count == 2


def test_the_fleet_attaches_over_cdp_and_never_launches_a_browser():
    """Launching it here would put the browser back in automation mode, which is
    exactly what provider sign-in checks reject — the reason this design exists."""
    import inspect

    from solvers import browser_pool

    source = inspect.getsource(browser_pool)
    assert "chromium.connect_over_cdp" in source
    for launcher in ("launch_persistent_context", ".launch(", "launch_server"):
        assert launcher not in source, f"the fleet must not call {launcher}"


def test_the_roster_reads_both_provider_lists_into_one_fleet():
    """Six to ten browsers, mixed providers, is the shape this is built for."""
    from solvers.roster import roster

    browsers = roster({"CLAUDE_CDP": "9222,9223,9224", "CHATGPT_CDP": "9225,9226"})
    assert [b.endpoint for b in browsers] == [
        "http://127.0.0.1:9222", "http://127.0.0.1:9223", "http://127.0.0.1:9224",
        "http://127.0.0.1:9225", "http://127.0.0.1:9226",
    ]
    assert [b.site.name for b in browsers] == ["claude"] * 3 + ["chatgpt"] * 2


def test_an_empty_roster_falls_back_to_one_browser_on_the_default_port():
    """The single-browser case must need no configuration at all."""
    from solvers.roster import roster

    browsers = roster({})
    assert len(browsers) == 1
    assert browsers[0].endpoint == "http://127.0.0.1:9222"
    assert browsers[0].site.name == "claude"


def test_the_same_endpoint_listed_under_two_providers_is_not_double_counted(capsys):
    """Attaching to one browser twice does not just invent capacity: `_fill`
    reclaims a browser's leftover tabs on every attach, so the second entry
    would close the tabs the first had just spawned."""
    from solvers.roster import roster

    browsers = roster({"CLAUDE_CDP": "9222", "CHATGPT_CDP": "9222"})
    assert len(browsers) == 1 and browsers[0].site.name == "claude"
    warning = capsys.readouterr().out
    # The warning has to name the provider that is actually SERVED. Naming the
    # dropped one instead — which it used to — tells the operator the opposite
    # of what happened, and they go looking for the fault in the wrong browser.
    assert "Serving it as claude only" in warning, warning
    assert "chatgpt on that port is ignored" in warning, warning


def test_shutdown_disconnects_but_never_closes_your_browser():
    """The operator owns the browser. Closing it would throw away the login they
    made by hand, which a miner restart must never do."""
    import inspect


    teardown = inspect.getsource(BrowserFleet._teardown)
    # Tabs this pool opened are closed; the browser is only disconnected.
    assert "tab.dispose()" in teardown
    assert "connection.close()" in teardown, "the attachment is severed, not the browser"
    assert "context.close()" not in teardown, "closing the context would close your window"


def test_shutdown_closes_leased_tabs_too_not_just_idle_ones():
    """A free-queue-only sweep leaves every in-flight tab open in your browser.

    Behavioural, not a source grep: build a pool holding one idle tab and one
    leased tab (leased = tracked but not in the queue, which is what a shutdown
    mid-solve looks like) and assert both pages get closed.
    """
    closed = []

    class _Page:
        def __init__(self, name): self.name = name
        async def close(self): closed.append(self.name)

    class _Browser:
        def __init__(self): self.disconnected = False
        async def close(self): self.disconnected = True

    class _Driver:
        def __init__(self): self.stopped = False
        async def stop(self): self.stopped = True

    async def go():
        pool = _fleet()
        idle = _Tab(pool, _Page("idle"), None, "idle", chatgpt_site())
        leased = _Tab(pool, _Page("leased"), None, "leased", chatgpt_site())
        pool._tabs = [idle, leased]
        await pool._free.put(idle)          # only the idle one is in the queue
        browser, driver = _Browser(), _Driver()
        pool._connections = [browser]
        pool._pw = driver                   # teardown keys off the driver
        pool._started = True
        await pool.aclose()
        return browser, driver

    browser, driver = asyncio.run(go())
    assert sorted(closed) == ["idle", "leased"], f"only closed {closed}"
    assert browser.disconnected, "the pool must disconnect from the browser"
    assert driver.stopped, "the Playwright driver must be stopped too"


def test_releasing_a_tab_twice_does_not_queue_it_twice():
    """Two tasks driving one page corrupt both answers, and the symptom — two
    solves interleaving in one conversation — looks like a model failure."""
    async def go():
        pool = _fleet()
        tab = _Tab(pool, _DeadPage(), None, "t#1", chatgpt_site())
        tab.leased = True
        await pool.release(tab)
        await pool.release(tab)     # a second close must be a no-op
        return pool._free.qsize()

    assert asyncio.run(go()) == 1, "the tab was queued twice"


def test_a_start_that_fails_stops_the_playwright_driver():
    """Otherwise the driver process leaks, and because start() left _started
    False the next open() would spawn a second one."""
    async def go():
        pool = BrowserFleet([Browser("http://127.0.0.1:59999", chatgpt_site())])

        class _Driver:
            def __init__(self): self.stopped = False
            async def stop(self): self.stopped = True

        driver = _Driver()

        async def fake_pw_start():
            pool._pw = driver

        # Stand in for `async_playwright().start()`, then let _fill() fail for
        # real by finding no reachable endpoint.
        async def connect():
            await fake_pw_start()
            try:
                await pool._fill()
            except Exception:
                await pool._teardown()
                raise

        pool._connect = connect
        with pytest.raises(RuntimeError):
            await pool.start()
        return driver, pool

    driver, pool = asyncio.run(go())
    assert driver.stopped, "a failed start must stop the driver it started"
    assert pool._pw is None and not pool._started


def test_an_unreachable_endpoint_names_the_debug_browser_script():
    """A wrong CDP endpoint is a browser-setup problem, and the message must say
    so rather than leaving the operator to guess."""
    import asyncio

    pool = BrowserFleet([Browser("http://127.0.0.1:59999", claude_site())])
    with pytest.raises(RuntimeError, match="remote-debugging-port"):
        asyncio.run(pool.start())


def test_tabs_per_browser_comes_from_the_environment(monkeypatch):
    from solvers.roster import tabs_per_browser

    monkeypatch.delenv("MINER_TABS_PER_BROWSER", raising=False)
    assert tabs_per_browser() == 2
    monkeypatch.setenv("MINER_TABS_PER_BROWSER", "4")
    assert tabs_per_browser() == 4


def test_no_backend_anywhere_reads_an_api_key(monkeypatch):
    """Every backend reaches a seat somebody already pays for. Nothing in the
    package authenticates with a key, and nothing imports a provider SDK — a
    regression would be invisible otherwise, because a key-reading backend
    works fine right up until it bills someone.

    This used to ban the substring `API_KEY` anywhere in the package, which was
    the right intent and became the wrong instrument: `claude_cli` has to name
    `ANTHROPIC_API_KEY` in order to REMOVE it from the environment its child
    gets. A ban that cannot tell "reads a key" from "deletes a key" would have
    forced the one module that actively prevents metered billing to stop saying
    so. So the ban is on the act — the idioms that read a key's VALUE — and the
    behaviour is then asserted directly, which is stronger than either.
    """
    import inspect
    from pathlib import Path

    from solvers import claude_web
    from solvers.claude_cli import child_env
    from solvers.roster import PROVIDERS, site_for

    assert set(PROVIDERS) == {"claude", "chatgpt"}
    assert "claude.ai" in site_for("claude").url
    assert "chatgpt.com" in site_for("chatgpt").url

    # Reading a key's value, in every form that would work, and importing an
    # SDK that would then use it.
    banned = (
        'environ["ANTHROPIC_API_KEY"]', "environ['ANTHROPIC_API_KEY']",
        'environ.get("ANTHROPIC_API_KEY"', "environ.get('ANTHROPIC_API_KEY'",
        'getenv("ANTHROPIC_API_KEY"', "getenv('ANTHROPIC_API_KEY'",
        "import anthropic", "from anthropic", "google.genai",
    )
    package = Path(inspect.getfile(claude_web)).parent
    for module in sorted(package.glob("*.py")):
        body = module.read_text()
        for needle in banned:
            assert needle not in body, f"{module.name} references {needle!r}"

    # And the one backend that spawns something capable of using a key takes it
    # away first. Asserted, not merely unmentioned: this is the difference
    # between a subscription seat and an invoice.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-be-used")
    monkeypatch.delenv("SOLVER_CLI_ALLOW_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" not in child_env()


def test_the_doctor_probe_drives_a_real_tab():
    """The doctor builds a `_Tab` by hand, so it is the one caller that a change
    to that constructor can break without any other test noticing — and did:
    it kept passing the pre-fleet argument list and died with a TypeError at the
    probe, after all the selector checks had already printed OK. Call the
    doctor's own function, not a hand-built tab, so the signature stays bound.
    """
    from solvers import doctor

    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    # Two blocks, as the probe now asks for: the answer and a usage example.
    # `print(pong())` alone used to satisfy the doctor, which is exactly the
    # weakness that let a page shipping only the usage example look healthy.
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant",
        [_Node(code=["def pong():\n    return 'pong'", "print(pong())"])],
    )
    assert asyncio.run(doctor._probe(page, _site(), "#composer", ())) is True
    # ...and it reports how the next conversation starts. With no new-chat
    # control that is the reload, which is the honest answer, not a skip.
    assert page.navigated == ["about:blank"]


def test_the_doctor_reports_the_in_app_new_chat_when_the_page_has_one():
    """The one thing a selector list cannot tell you is whether clicking the
    control actually clears the transcript. The probe has just left a real one
    on the page, so this is the only place that can be answered against your
    own browser rather than assumed."""
    from solvers import doctor

    page = _chat_page()

    def handler(selector):
        if selector == "#send":
            page.dom["#assistant"] = [
                _Node(code=["def pong():\n    return 'pong'", "print(pong())"])
            ]
        elif selector == "#newchat":
            page.dom["#assistant"] = []

    page.on_click = handler
    assert asyncio.run(
        doctor._probe(page, _site(new_chat=("#newchat",)), "#composer", ())
    ) is True
    assert "#newchat" in page.clicked
    assert page.navigated == [], "the in-app path should not reload the page"


def test_a_long_answer_survives_the_reader_intact():
    """Real solutions are not three lines. Re-fencing every block and choosing
    between them must not quietly lose the middle of a big one, and the tail is
    where truncation shows: a cut answer still parses surprisingly often, and
    then fails the hidden tests for reasons nothing logs."""
    body = "\n".join(f"    # line {i}" for i in range(1, 2001))
    answer = f"def pong():\n{body}\n    return 'pong'"
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant", [_Node(code=[answer, "print(pong())"])]
    )
    reply = asyncio.run(_tab(page, _site()).send("solve it", 2.0))
    code = extract_code(reply, "pong")
    assert "# line 1\n" in code and "# line 2000" in code, "the block was truncated"
    scope: dict = {}
    exec(compile(code, "<submitted>", "exec"), scope)
    assert scope["pong"]() == "pong"


def test_two_answers_to_one_prompt_do_not_flip_between_polls():
    """ChatGPT sometimes streams TWO candidate answers for a single prompt and
    asks which you prefer. Reading "the last message" then means reading
    whichever branch is last at that instant, and while both stream that
    changes: the text never repeats across two polls, the completion test never
    fires, and the whole budget is spent before the deadline forces a partial
    answer out. Latch one branch on sight and read only that.

    Latched by message id, not by index -- an index still drifts if the two are
    repainted in the other order, which is the failure this reproduces.
    """
    A = _Node(code=["def pong():\n    return 'A'"], attrs={"data-message-id": "id-A"})
    B = _Node(code=["def pong():\n    return 'B'"], attrs={"data-message-id": "id-B"})
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    site = _site(message_id_attr="data-message-id")
    tab = _tab(page, site)

    async def go():
        before = await tab._fingerprint()          # empty conversation
        page.dom["#assistant"] = [A]               # first branch renders
        seen = set()
        for poll in range(6):
            if poll:                               # then both, order flipping
                page.dom["#assistant"] = [A, B] if poll % 2 else [B, A]
            reply = await tab._new_reply(before)
            if reply is not None:
                seen.add(await _Tab._read(reply))
        return seen

    seen = asyncio.run(go())
    assert len(seen) == 1, f"read drifted between branches: {seen}"
    assert "return 'A'" in seen.pop(), "did not commit to the branch it saw first"


def test_a_reply_whose_id_changes_mid_stream_is_still_read():
    """Latching the reply by id has to have a way back, and did not.

    A chat UI paints a streaming message with a provisional id and can swap it
    for the server's once the message is confirmed. The id latch searched for a
    key that no longer existed and returned None -- and kept returning None for
    the rest of the send, because nothing ever re-latched. If the swap happened
    before the first readable frame (it does: the message is painted empty and
    filled afterwards) then `best` was never set either, so `send` returned ""
    and a complete, correct answer was reported as "the reply contained no
    code". Seen live on ChatGPT three attempts running.
    """
    ANSWER = "def pong():\n    return 'pong'"

    class _Swapping(_Node):
        """One assistant message: painted empty, re-identified, then filled."""

        def __init__(self):
            super().__init__(attrs={"data-message-id": "provisional"})
            self._polls = 0

        def locator(self, selector):
            self._code = [ANSWER] if self._polls else []   # fills after the swap
            return super().locator(selector)

        async def inner_text(self):
            self._polls += 1
            self._attrs["data-message-id"] = "server-assigned"
            return self._text

    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_Swapping()])
    site = _site(message_id_attr="data-message-id")
    reply = asyncio.run(_tab(page, site).send("solve it", 2.0))

    assert extract_code(reply, "pong") == ANSWER, f"lost the answer to an id swap: {reply!r}"


def test_a_reply_is_found_by_position_when_the_site_has_no_message_id():
    """claude.ai has no per-message id, so the reply is 'an assistant message
    that was not there before we pressed send'. Sound only because every task
    starts a fresh conversation."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant", [_Node(code=["def g(n):\n    return n"])]
    )
    reply = asyncio.run(_tab(page, _site()).send("solve it", 2.0))
    # The reader re-fences what it scraped so every block reaches the caller;
    # picking between them needs the entrypoint, which the tab does not have.
    assert extract_code(reply, "g") == "def g(n):\n    return n"
    assert page.typed == ["solve it"]


def _wire_page(stream, renders=True):
    """A page whose network stream returns `stream` (a value or a callable).

    `renders=False` is the BLIND tab: the DOM never shows the reply at all, and
    the wire is the only place the answer exists.
    """
    class _Page(_FakePage):
        async def evaluate(self, expression, arg=None):
            if "since" in str(expression):
                return stream() if callable(stream) else stream
            return 0

    page = _Page({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    if renders:
        page.on_click = lambda _: page.dom.__setitem__(
            "#assistant", [_Node(code=["def g(n):\n    return n"])])
    return page


def test_a_blind_tab_stops_when_the_wire_has_settled(monkeypatch):
    """The whole budget, spent polling a DOM that was never going to render,
    while the finished answer sat on the wire the entire time.

    This is the failure the operator reported, and the class already half knew
    about it: the blind-tab notice says "the answer may also arrive off the
    wire", and the comment above it records eighteen tabs whose answers were
    recovered off the wire moments later. What was missing was reading the wire
    BEFORE the deadline rather than after it. Measured before this existed:
    10.00s of a 10s slice and 30.00s of a 30s slice — 100% both times, with the
    answer complete on the wire from the first poll.

    On a live 290-second solve that is the whole budget, and the phase-3
    correction then arrives with nothing left to grade it against.
    """
    from solvers import browser_pool

    monkeypatch.setattr(browser_pool, "BLIND_TAB_GRACE_S", 1.0)
    started = time.monotonic()
    answer = "```python\ndef g(n):\n    return n\n```"
    got = asyncio.run(
        _tab(_wire_page(answer, renders=False), _site(stream=True)).send("solve it", 30.0)
    )
    spent = time.monotonic() - started

    assert extract_code(got, "g") == "def g(n):\n    return n", f"lost it: {got!r}"
    assert spent < 10.0, (
        f"a blind tab spent {spent:.1f}s of a 30s slice on an answer that was "
        f"complete on the wire before the first poll"
    )


def test_the_wire_is_only_taken_once_it_has_stopped_changing(monkeypatch):
    """The guard on the fix above, and the reason stillness is the signal.

    `fenced_blocks` keeps an unclosed final fence on purpose — a reply cut off
    by a deadline still has its program in it — so "there is a fenced block" is
    true of a half-written answer too and proves nothing about being finished.
    Breaking on structure would submit whatever had arrived by the blind grace.

    Two things must not end the read: a stream still growing, and a stream that
    has settled with no answer in it (a preamble, or the model's reasoning).
    """
    from solvers import browser_pool

    monkeypatch.setattr(browser_pool, "BLIND_TAB_GRACE_S", 1.0)

    # Still arriving: every read is longer than the last.
    lines = {"n": 0}

    def growing():
        lines["n"] += 1
        return "```python\ndef g(n):\n" + "    x = 1\n" * lines["n"]

    started = time.monotonic()
    got = asyncio.run(
        _tab(_wire_page(growing, renders=False), _site(stream=True)).send("solve it", 6.0)
    )
    assert time.monotonic() - started >= 5.0, "cut a stream off mid-answer"
    assert got.count("x = 1") > 3, f"took a truncated prefix: {got!r}"

    # Settled, but it is a preamble rather than an answer.
    started = time.monotonic()
    got = asyncio.run(
        _tab(_wire_page("Let me think about this problem.", renders=False),
             _site(stream=True)).send("solve it", 6.0)
    )
    assert time.monotonic() - started >= 5.0, (
        "stopped on a stream holding no answer, which submits nothing"
    )
    assert not (got or "").strip(), f"returned prose as an answer: {got!r}"


def test_a_reply_that_is_all_prose_reaches_the_solver_as_prose():
    """The page read must hand back what the model actually wrote, and the
    extractor must refuse to call it a program.

    The rule the second half pins is right and stays: claude.ai renders
    extended thinking inside the element the assistant selector matches, and
    falling back to the message text once submitted 13,200 characters of
    reasoning as a Rust program. Both halves are needed together -- a read that
    returned nothing would report "your reply did not reach me" about a reply
    that plainly did, and an extractor that took prose would submit it."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_Node(
        text="You are right — the case was wrong, not the program. Corrected:\n\n"
             '[{"name": "zero", "args": [0]}, {"name": "carry", "args": [12345]}]'
             "\n\nThe program itself is fine as sent."
    )])
    reply = asyncio.run(_tab(page, _site()).send("fix it", 1.0))

    from solvers.prompts import extract_inputs

    # The array is dug out of the prose, because that fallback is the only
    # thing standing between a model that ignored the fence and a solve with
    # no bar at all -- and the inputs turn is the one turn that still asks for
    # a JSON array. The gate it passes is structural: `args` is enough, since
    # the inputs turn is forbidden to supply an expected value.
    assert extract_inputs(reply, "python") == [
        {"name": "zero", "args": [0], "kwargs": {}, "notes": ""},
        {"name": "carry", "args": [12345], "kwargs": {}, "notes": ""},
    ], f"the salvaged inputs were lost: {reply!r}"
    assert extract_code(reply, "g", "python") == "", (
        f"prose came back as a program, which is what the None rule prevents: "
        f"{reply!r}"
    )


def test_prose_without_an_array_is_still_read_as_nothing():
    """The guard on the salvage above, and the reason it is safe.

    Reasoning, a refusal and a clarifying question are all prose, and every one
    of them must still read as "no answer". Only a bracketed array whose items
    carry `expected` gets through — the same structural gate everything else
    here uses.
    """
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_Node(
        text="Let me think. The digits of 12345 are [1, 2, 3, 4, 5], so the sum "
             "is 15. I will use a while loop and accumulate the remainder."
    )])
    assert asyncio.run(_tab(page, _site()).send("solve it", 1.0)) == "", (
        "reasoning was returned as an answer"
    )


def test_a_partial_answer_survives_a_deadline_that_lands_mid_stream():
    """The commonest timeout there is: the model is still typing when the budget
    runs out. Returning "" there throws away a gradeable answer and hands the
    repair round nothing to work with.

    Half a CODE BLOCK, not half a message. What survives a deadline is whatever
    the model had written into the block, because that is the only thing this
    miner can submit -- a half-finished program still defines the entrypoint
    often enough to be worth grading, and the repair round has something
    concrete to work from. Prose caught mid-stream is not a partial answer; see
    `_read`.
    """
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})

    def stream(_):
        page.dom["#assistant"] = [_Node(code=["def solve(xs):\n    total = 0"])]
        page.dom["#stop"] = [_Node()]        # still generating, and stays that way

    page.on_click = stream
    site = _site(busy=("#stop",))
    got = asyncio.run(_tab(page, site).send("solve it", 1.5))
    assert "total = 0" in got, f"threw away the half-written program: {got!r}"


def test_send_honours_its_deadline_including_the_time_spent_submitting():
    """Deriving the read deadline after the submit hands the read a fresh full
    budget on top of it — an overrun bigger than the solver's safety margin."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})

    async def slow_insert(text):
        await asyncio.sleep(1.0)
        page.typed.append(text)

    page.keyboard.insert_text = slow_insert
    started = time.monotonic()
    asyncio.run(_tab(page, _site()).send("solve it", 2.0))
    assert time.monotonic() - started < 3.0, "the submit time was added on top"


def test_a_dead_tab_is_not_driven_again():
    """Retrying a known-dead tab burns the budget one submit-timeout at a time."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    tab = _tab(page, _site())
    tab.alive = False
    assert asyncio.run(tab.send("solve it", 30.0)) == ""
    assert page.typed == [], "it typed into a tab it knew was dead"


def test_a_shorter_message_list_is_a_re_render_not_a_new_reply():
    """`_new_reply` treated "the last message's id changed" as proof of a new
    reply. That holds when the list grew or held steady; when it SHRANK, the
    last message is an OLDER one wearing a different id — so this prompt was
    answered with a previous turn's program, silently, `empty_reason=None`.

    Reproduced: before=(2, 'id-B'), the DOM re-rendered down to one message, and
    `send` returned branch A of the turn before."""
    site = _site(message_id_attr="data-message-id")
    old_a = _Node(text="OLD A", code=["def pong():\n    return 'OLD-BRANCH-A'"],
                  attrs={"data-message-id": "id-A"})
    old_b = _Node(text="OLD B", code=["def pong():\n    return 'OLD-BRANCH-B'"],
                  attrs={"data-message-id": "id-B"})
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()],
                      "#assistant": [old_a, old_b]})
    # The click re-renders the list DOWN to one message and never grows it: the
    # answer to this prompt has not been painted yet.
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [old_a])

    got = asyncio.run(_tab(page, site).send("solve it", 0.4))
    assert "OLD-BRANCH-A" not in got, (
        f"answered this prompt with a previous turn's program: {got!r}"
    )

    # ...and the case the branch exists for still works: a site that REPLACES
    # the last message rather than appending one.
    replaced = _Node(text="new", code=["def pong():\n    return 'THE ANSWER'"],
                     attrs={"data-message-id": "id-NEW"})
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()],
                      "#assistant": [old_a, old_b]})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [old_a, replaced])
    got = asyncio.run(_tab(page, site).send("solve it", 0.4))
    assert "THE ANSWER" in got, f"stopped seeing a replaced last message: {got!r}"


def test_a_transient_selector_miss_does_not_strand_the_send_on_a_coarser_one():
    """`_messages` dropped its latch when the candidate matched nothing,
    reasoning "there is no count to corrupt at zero". There is: `before[0]` was
    counted with the OLD candidate, and `_new_reply` compares the new one's
    count against it directly. chatgpt.com ships two candidates that count on
    different scales — an A/B pair is two messages inside one article — so after
    a re-resolve the comparison is meaningless and no reply is ever found.

    Measured on the code this replaces, step by step:

        before = (2, 'm2') latched: #msg
        note: assistant selector '#msg' stopped matching mid-answer; re-resolving
        after the blink, latched: #article
        once #msg matches again, latched: #article      <- never re-examined
        reply found: False
    """
    MSG, ART = "#msg", "#article"
    site = _site(assistant=(MSG, ART), message_id_attr="data-message-id")
    m1 = _Node(text="old one", attrs={"data-message-id": "m1"})
    m2 = _Node(text="old two", attrs={"data-message-id": "m2"})
    answer = _Node(text="here", code=["def g(n):\n    return n * 7"],
                   attrs={"data-message-id": "m3"})
    art = _Node(text="one article holds every message")
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()],
                      MSG: [m1, m2], ART: [art]})
    tab = _tab(page, site)

    async def go():
        before = await tab._fingerprint()
        assert before == (2, "m2") and tab._assistant == MSG, (before, tab._assistant)

        page.dom[MSG] = []                       # the candidate blinks out
        await tab._messages()

        page.dom[MSG] = [m1, m2, answer]         # ...and the answer lands
        await tab._messages()
        assert tab._assistant == MSG, (
            f"stayed on the coarser candidate: {tab._assistant} — the baseline "
            f"was counted with {MSG} and the two count on different scales"
        )
        reply = await tab._new_reply(before)
        assert reply is not None, "never found the reply after the blink"
        assert await tab._read(reply) is not None

    asyncio.run(go())


def test_a_count_taken_with_one_selector_is_never_compared_against_another():
    """The other half. While the latch is NOT the candidate the baseline was
    counted with, the count comparison is meaningless and is not made at all —
    `_new_reply` waits for the original to come back rather than guessing from
    numbers on two different scales."""
    MSG, ART = "#msg", "#article"
    site = _site(assistant=(MSG, ART))          # no message id: counts only
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()],
                      MSG: [_Node(text="a"), _Node(text="b")],
                      ART: [_Node(text="t1"), _Node(text="t2"), _Node(text="t3")]})
    tab = _tab(page, site)

    async def go():
        before = await tab._fingerprint()
        assert before[0] == 2 and tab._counted_with == MSG

        page.dom[MSG] = []                       # forced onto the coarser one
        await tab._messages()
        assert tab._assistant == ART
        # ART's count is 3 against a baseline of 2 taken under MSG. Reading that
        # as "a new message arrived" is how a whole send was lost.
        assert await tab._new_reply(before) is None, (
            "compared a count taken with one selector against another"
        )

    asyncio.run(go())


def test_the_post_mortem_names_the_selector_the_read_actually_used(capsys):
    """`_explain_empty` re-resolved from scratch, so it could count one selector
    and quote `before[0]`, which was counted with another — reporting "matched 2
    message(s), the same as before the prompt was sent" of two different things,
    about a page holding the finished answer."""
    MSG, ART = "#msg", "#article"
    site = _site(assistant=(MSG, ART))
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()],
                      MSG: [], ART: [_Node(text="a turn")]})
    tab = _tab(page, site)
    asyncio.run(tab._messages())          # latches ART, the only one matching
    assert tab._assistant == ART

    page.dom[MSG] = [_Node(text="x"), _Node(text="y")]
    asyncio.run(tab._explain_empty((1, None)))
    out = capsys.readouterr().out
    assert ART in out, f"the post-mortem named a selector the read never used: {out}"
    assert MSG not in out, out


def test_the_echo_guard_reads_the_whole_message_not_the_code_block():
    """`_read` prefers the last `pre code`, and task statements routinely contain
    fenced code — so comparing the extracted block against the prompt would never
    match and the guard would never fire, exactly when it is needed."""
    prompt = "Solve this problem.\n```\nexample\n```\nReturn the digit sum."
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    # The assistant selector wrongly matches the user's turn: whole text echoes
    # the prompt, while the code block inside it does not.
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant", [_Node(text=prompt, code=["print('lifted from my own prompt')"])]
    )
    assert asyncio.run(_tab(page, _site()).send(prompt, 1.5)) == ""


def test_a_reply_that_echoes_the_prompt_is_refused():
    """If an assistant selector also matches the USER's turn, the miner would
    submit its own prompt back as the answer: no error, no empty reply, just a
    permanent zero. It must be refused instead."""
    prompt = "Solve this programming problem in Python.\nReturn the digit sum."
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_Node(text=prompt)])
    assert asyncio.run(_tab(page, _site()).send(prompt, 1.0)) == ""


def test_a_draft_left_in_the_composer_never_reaches_a_validator():
    """The failure this exists for, and it is silent from end to end.

    These chat accounts are shared with people, so the box can already hold
    somebody's half-typed message. `insert_text` inserts at the CARET, which the
    click before it just put at the element's centre — so the prompt is spliced
    INTO that draft and the whole thing goes as one message. Nothing downstream
    catches it: `_is_our_own_prompt` inspects the REPLY for our prompt's head,
    and in a contaminated send that head is still there, intact. The only
    symptom is a model answering a mangled question, which looks exactly like a
    hard task."""
    page = _FakePage(
        {"#composer": [_Node()], "#send": [], "#assistant": []},
        composer="hey, quick question about my mortgage",
    )
    asyncio.run(_tab(page, _site()).send("SOLVE THIS", 1.0))

    assert page.composer == "SOLVE THIS", (
        f"what was sent was not the prompt alone: {page.composer!r}"
    )
    assert "mortgage" not in page.composer
    assert page.pressed[:2] == ["Control+A", "Delete"], page.pressed


def test_an_editor_that_reformats_the_prompt_is_not_contamination():
    """Read off a live miner, where this retired a working tab:

        the composer did not hold the prompt as typed; clearing it and typing
        it again, once
        failed to submit: RuntimeError: the composer does not hold the prompt
        as typed, twice over

    The box was holding the prompt exactly as intended. claude.ai's composer is
    a rich-text editor and applies input rules as text arrives: `- ` at the
    start of a line becomes a bullet, `1. ` becomes an ordered list, and the
    marker is then list STRUCTURE rather than text — so `innerText` gives the
    line back without it, the ordered marker's digit included. Turn 1 carries
    nine such lines. Demanding the text back verbatim called every one of those
    sends contaminated, and retyping reproduces it exactly, so the second look
    failed too and the tab was thrown away.

    What may not happen is a word appearing that we never typed."""
    prompt = _all_stage_prompts(_stage_task("python"))["inputs"]
    reformatted = "\n".join(
        re.sub(r"^(- |[0-9]+\. )", "", line) for line in prompt.splitlines()
    )
    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    page.on_insert = lambda _: reformatted          # the editor rewrites it
    tab = _tab(page, _site())
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(tab.send(prompt, 1.0))

    assert page.pressed.count("Enter") == 1, "a reformatted prompt was not sent"
    assert tab.alive is True, "the tab was retired over the editor's own markup"
    # One clear, one insert: it must not have retyped either.
    assert page.typed == [prompt], f"retyped a prompt that was already right: {len(page.typed)}"


def test_a_composer_read_before_it_has_painted_is_waited_for_not_retyped():
    """`insert_text` returns when the input event is delivered, not when the
    editor has rendered it. On a box running four Chrome instances in 5 GB that
    gap is visible, and reading straight after catches an empty box or half a
    prompt — which is not contamination and must not be answered by retyping
    into an editor that is still catching up."""
    prompt = "solve this problem please"
    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    state = {"inserted": None, "reads": 0}

    def paints_late(text):
        state["inserted"] = text
        return ""                      # the box shows nothing yet

    page.on_insert = paints_late

    class _Slow(_Loc):
        @property
        def first(self):               # a real locator's `.first` keeps its type
            return self

        async def evaluate(self, expression):
            if state["inserted"] is None:
                return ""              # before we type: an empty box, as clearing wants
            state["reads"] += 1
            return state["inserted"] if state["reads"] >= 3 else ""

    plain = page.locator
    page.locator = lambda sel: (
        _Slow(page, sel, page.dom.get(sel, [])) if sel == "#composer" else plain(sel)
    )
    tab = _tab(page, _site())
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(tab.send(prompt, 3.0))

    assert state["reads"] >= 3, "did not wait for the editor to paint"
    assert page.pressed.count("Enter") == 1, "never sent a prompt that did arrive"
    assert page.typed == [prompt], (
        f"retyped into an editor that was merely slow to paint: {page.typed}"
    )
    assert tab.alive is True

def test_a_composer_that_will_not_empty_is_never_sent_to():
    """A box we cannot empty is a box whose contents we cannot vouch for, so the
    tab is thrown away rather than the prompt sent into it. `send` already turns
    a raise here into a retired tab reporting `unreadable`, and the solver takes
    that to another tab — so this needs no plumbing of its own, only the raise."""
    page = _FakePage(
        {"#composer": [_Node()], "#send": [], "#assistant": []},
        composer="somebody else's draft",
    )
    page.composer_unclearable = True
    tab = _tab(page, _site())
    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        reply = asyncio.run(tab.send("SOLVE THIS", 1.0))

    assert reply == ""
    assert tab.alive is False, "a tab that cannot be cleared was kept"
    assert tab.empty_reason == "unreadable"
    assert page.pressed.count("Enter") == 0, "sent anyway"
    assert "failed to submit" in chatter.getvalue()
    assert "would not clear" in chatter.getvalue(), chatter.getvalue()


def test_the_prompt_is_read_back_before_it_is_sent():
    """Clearing is not proof. An editor that mangles, truncates or autocompletes
    what was inserted would otherwise send whatever it happened to keep, so the
    box is read BACK and compared before the send control is touched. One
    retype, then the tab goes."""
    mangled = []

    def eat_the_end(text):
        mangled.append(text)
        return text[:4]                      # the editor kept only a fragment

    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    page.on_insert = eat_the_end
    tab = _tab(page, _site())
    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        reply = asyncio.run(tab.send("SOLVE THIS", 1.0))

    assert reply == ""
    assert tab.alive is False
    assert len(mangled) == 2, f"did not retype once before giving up: {mangled}"
    assert page.pressed.count("Enter") == 0, "sent a prompt it could not read back"
    assert "did not hold the prompt as typed" in chatter.getvalue()


def test_a_composer_that_matches_after_a_retype_is_sent():
    """The retype is worth having: a transient mangle costs one round trip, not
    the tab."""
    state = {"n": 0}

    def once(text):
        state["n"] += 1
        return text if state["n"] > 1 else "junk" + text

    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    page.on_insert = once
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(_tab(page, _site()).send("SOLVE THIS", 1.0))

    assert page.composer == "SOLVE THIS"
    assert page.pressed.count("Enter") == 1, "the retyped prompt was not sent"


def test_the_read_back_tolerates_the_editors_own_whitespace():
    """A contenteditable turns our newlines into block elements and hands them
    back as its own arrangement of them. Comparing exactly would fail on every
    send; comparing on collapsed whitespace still catches the only thing that
    matters, which is text we did not type."""
    from solvers.browser_pool import _same_message

    assert _same_message("a\n\nb", "a\nb")
    assert _same_message("a b", " a  b ")
    assert not _same_message("a b", "hi a b")
    assert not _same_message("a b", "a b bye")


def test_pressing_enter_is_the_fallback_when_no_send_button_matches():
    """Safe only because insert_text already put the whole multi-line prompt in."""
    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    asyncio.run(_tab(page, _site()).send("line one\nline two", 1.0))
    assert page.typed == ["line one\nline two"]
    # The clear comes first and the submit last: a send that did not empty the
    # box first is a send that may carry somebody else's draft.
    assert page.pressed == ["Control+A", "Delete", "Enter"], page.pressed


def test_a_selector_a_page_cannot_evaluate_is_dropped_at_startup():
    """A typo in a `.env` override should cost that one candidate. At answer
    time a raising selector is indistinguishable from a dead page, so the tab
    would be retired on every request instead."""
    from solvers.browser_pool import valid_selectors

    class _Strict(_FakePage):
        def locator(self, selector):
            if selector == "!!bad":
                raise ValueError("unexpected token")
            return super().locator(selector)

    page = _Strict({"#ok": [_Node()]})
    kept = asyncio.run(valid_selectors(page, ("!!bad", "#ok"), "t", "assistant"))
    assert kept == ("#ok",)


def test_a_busy_selector_that_matches_an_idle_page_is_dropped():
    """An always-true 'still generating' selector is the one selector mistake
    that cannot degrade gracefully: every answer would look unfinished and burn
    the whole budget. An idle page is the ground truth that catches it."""
    page = _FakePage({"#always": [_Node()], "#real-stop": []})
    kept = asyncio.run(usable_busy_selectors(page, ("#always", "#real-stop"), "t"))
    assert kept == ("#real-stop",)


def test_a_rendered_code_block_survives_being_scraped():
    """Both halves of a real solve that scored zero.

    The reader scrapes a RENDERED page, so what comes back is not the source
    the model wrote. Two things ride along, and both were seen live:

    1. A Private Use Area character the UI uses for its own bookkeeping. It is
       invisible, and it makes the whole file `invalid non-printable character
       U+E027` — after a perfectly good answer.
    2. The code block's language chip, which sits inside the element being
       scraped and becomes a bare `python` first line. That is the worse one:
       it parses, it defines the entrypoint, it passes every check, and then
       raises NameError the instant the grader imports it.
    """
    from solvers.prompts import extract_code, python_defect

    scraped = "python\ndef g(n):" + chr(0xE027) + "\n    return n"
    code = extract_code(scraped)
    assert python_defect(code, "g") is None, "the artefacts were not cleaned"
    exec(compile(code, "<submitted>", "exec"), {})          # must not raise

    # ...and the same with only the chip, which used to pass silently.
    code = extract_code("python\ndef g(n):\n    return n")
    assert python_defect(code, "g") is None
    ns: dict = {}
    exec(compile(code, "<submitted>", "exec"), ns)
    assert ns["g"](3) == 3


def test_the_reader_hands_over_every_code_block_not_its_favourite():
    """The tab must not choose. It has no entrypoint and no language, so any
    choice it makes is a guess -- and the guess it used to make (the last one)
    threw the answer away whenever a usage example followed it. Re-fence them
    all and let the grader, which knows the task, decide."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant",
        [_Node(code=["def g(n):\n    return n * 2", "print(g(21))"])],
    )
    reply = asyncio.run(_tab(page, _site()).send("solve it", 2.0))
    assert "def g(n)" in reply and "print(g(21))" in reply, "a block was dropped"
    assert extract_code(reply, "g") == "def g(n):\n    return n * 2"


def test_the_answer_wins_over_a_usage_example_that_follows_it():
    """Models append `print(solve(21))` demos however firmly the prompt says
    not to. Taking the last block submitted the demo, and the whole solve was
    spent reporting that the entrypoint was never defined -- with the real
    answer sitting in the block just before it."""
    reply = "```\ndef solve(n):\n    return n * 2\n```\n```\nprint(solve(21))\n```"
    assert extract_code(reply) == "print(solve(21))", "no target: last block"
    assert extract_code(reply, "solve") == "def solve(n):\n    return n * 2"


def test_a_corrected_answer_still_beats_the_draft_before_it():
    """The other half of the rule: when both blocks are gradeable, the LAST one
    wins, because a model that shows a draft then fixes it means the fix."""
    reply = "```\ndef solve(n):\n    return n\n```\n```\ndef solve(n):\n    return n * 2\n```"
    assert extract_code(reply, "solve") == "def solve(n):\n    return n * 2"


def test_nothing_gradeable_still_returns_something_to_complain_about():
    """Returning "" would report `no code` when the real defect is more
    specific, and the repair round is only as good as the evidence it gets."""
    assert extract_code("```\nnot code\n```\n```\nalso not\n```", "solve") == "also not"


def test_a_fence_inside_the_source_does_not_cut_the_block_short():
    """A docstring showing markdown was enough to truncate the answer: the
    hard-coded three-backtick closer matched the docstring's own fence."""
    inner = 'def solve(n):\n    """```md"""\n    return n'
    assert extract_code("````\n" + inner + "\n````", "solve") == inner


def test_a_rust_program_wins_over_a_sample_output_block():
    """Same rule, other language: `rust_defect` is the gradeability test."""
    reply = '```\nfn main() { println!("1"); }\n```\n```\n1 2 3\n```'
    assert extract_code(reply, "main", "rust").startswith("fn main()")


def test_a_language_chip_inside_a_fence_is_dropped_too():
    """Belt and braces: a fenced reply can carry the chip as its first line."""
    from solvers.prompts import extract_code

    code = extract_code("```python\npython\ndef g(n):\n    return n\n```")
    assert code.startswith("def g("), code


def test_exotic_spaces_do_not_break_indentation():
    """A non-breaking space renders like a space and is not one."""
    from solvers.prompts import extract_code, python_defect

    code = extract_code("def g(n):\n" + "\u00a0" * 4 + "return n")
    assert python_defect(code, "g") is None, "NBSP indentation was not folded"


def test_a_bare_name_at_top_level_is_reported_not_submitted():
    """The general form of the chip bug. A top-level bare name is never
    meaningful code and always raises NameError on import, so every hidden test
    fails. Reporting it turns a silent zero into a repair round."""
    from solvers.prompts import python_defect

    defect = python_defect("import os\nfoo\ndef g(n):\n    return n", "g")
    assert defect is not None and "bare name" in defect, defect


def test_clean_code_is_left_exactly_alone():
    """The sanitiser must not be creative with source that was already fine."""
    from solvers.prompts import extract_code

    source = "def g(n):\n    # keep  spacing\n    return {'ok': True}"
    assert extract_code(source) == source


def test_a_delivery_failure_is_not_reported_as_a_wrong_answer():
    """The repair round is the second and last chance at a task, and it used to
    be spent on a contradiction. When nothing arrived, the miner said "I ran the
    program against the examples and got: the reply contained no code" — nothing
    was run, there was nothing to run. A model told its program failed the
    examples rewrites the program, which was never the problem, and the rewrite
    goes to the same place the first one did.

    Seen live on a Rust task: two complete, plausible programs, both reported as
    no code, both repaired against evidence that did not exist.
    """
    prompt = _repair_prompt("rust", defect=NO_CODE, found_by="unrun")

    assert NO_CODE in prompt
    # Where the reply has to be WRITTEN is the whole of the fix, and it is said
    # positively rather than as a list of the places it must not go. The ban on
    # artifacts and canvases is the nudge's job, and the nudge is appended to
    # every send including this one -- see `_submit`.
    assert "directly in the chat" in prompt
    assert "NOTHING WAS RUN" in prompt, "still claims to have run something"
    assert "comparing what they produced" not in prompt, (
        "described a comparison against a reference that never happened"
    )


def test_a_real_failure_still_quotes_the_evidence():
    """The other branch must keep working: when code DID arrive and failed, the
    concrete counter-example is what makes the repair loop converge."""
    prompt = _repair_prompt("python", report="g(*[12345]) returned 14, expected 15")

    assert "returned 14, expected 15" in prompt
    assert "running this program and a separate reference" in prompt


# --- a model that reasons before it answers ------------------------------- #
# Reasoning quotes code, and quoted code is not an answer. Three separate
# things had to hold for a fragment out of the model's rough work to reach the
# grader, and each of these pins one of them: the read must not stop in the gap
# between the reasoning and the answer, the extractor must not treat reasoning
# as a candidate, and a defect must be reported as one rather than as a failed
# run. Live symptom on a Rust task, three attempts running: "no code", then
# "does not define fn main()", then "no code" again.


def test_a_pause_between_the_reasoning_and_the_answer_is_not_the_answer():
    """`_read` keeps a message's code blocks and drops its prose. So while the
    model writes the sentence that introduces its real answer, the read does not
    move at all -- two identical polls, which is exactly what "finished" used to
    mean. The reasoning's fragment is returned, `fn main` is missing from it,
    and the repair round is spent on a program that was never the answer.

    The busy selector normally covers this, but it is per-site, overridable and
    dropped at startup when it matches an idle page, so the completion test has
    to hold without one. There is none here on purpose.
    """
    from solvers.prompts import rust_defect

    FRAGMENT = "struct SegTree { n: usize }"          # quoted mid-reasoning
    ANSWER = 'fn main() {\n    println!("42");\n}'    # written much later
    intro = "Let me think. A segment tree works."
    frames = [
        ([FRAGMENT], "Let me think."),
        ([FRAGMENT], intro),                          # code identical, prose grew
        ([FRAGMENT], intro + " Here it is:"),         # and again
        ([FRAGMENT, ANSWER], intro + " Here it is:"), # the answer finally lands
        ([FRAGMENT, ANSWER], intro + " Here it is:"), # settled
    ]

    class _StreamNode(_Node):
        """One assistant message that advances a frame each time it is polled."""

        def __init__(self):
            super().__init__()
            self._i = 0

        def _frame(self):
            return frames[min(self._i, len(frames) - 1)]

        def locator(self, selector):
            self._code = list(self._frame()[0])
            return super().locator(selector)

        async def inner_text(self):
            text = self._frame()[1]
            self._i += 1     # `_whole` reads this last, so one poll is one frame
            return text

    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_StreamNode()])
    reply = asyncio.run(_tab(page, _site(busy=())).send("solve it", 5.0))

    code = extract_code(reply, "main", "rust")
    assert rust_defect(code) is None, f"stopped reading mid-reasoning: {reply!r}"
    assert "println!" in code, "handed over the fragment instead of the answer"


def test_code_quoted_while_thinking_is_not_a_candidate_answer():
    """Some replies carry the reasoning as literal `<think>` text, and the
    reasoning quotes code. Every fragment in it looks like a candidate to a
    fence scanner -- and one of them is the LAST block whenever the answer did
    not arrive, so rough work goes to the grader looking like a solution.

    An opener with no closer is the same problem with no bottom: there is no
    answer after it, and "nothing arrived" is the only honest thing to report.
    It is also the more useful one, because it is the branch of the repair
    prompt that asks for the code again instead of blaming the logic.
    """
    from solvers.prompts import NO_CODE, rust_defect

    reply = (
        "<think>\nA segment tree, maybe:\n"
        "```rust\nstruct SegTree { n: usize }\n```\n"
        "no, too slow.\n</think>\n"
        'Here it is:\n```rust\nfn main() { println!("42"); }\n```'
    )
    code = extract_code(reply, "main", "rust")
    assert rust_defect(code) is None and "println!" in code
    assert "SegTree" not in code, "mined the model's own reasoning for an answer"

    cut = reply.split("</think>")[0]        # the read landed mid-thought
    assert rust_defect(extract_code(cut, "main", "rust")) == NO_CODE


def test_the_reader_returns_a_code_block_byte_for_byte():
    """Only a real browser can answer this, so this one uses one.

    claude.ai splits a <code> into `data-code-line-group` blocks. `innerText`
    puts a line break at every block boundary, so a 15-line program comes back
    as 17 -- measured, on the DOM of a real answer. Rust shrugs at a stray blank
    line. A Python multi-line string literal does not, and nothing anywhere
    reports the corruption: the code parses, defines the entrypoint, passes
    every check, and disagrees with a hidden test about the contents of a
    string. textContent is the raw text, and each line already carries its own
    newline, so it round-trips exactly.
    """
    chrome = Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    if not chrome.exists():
        pytest.skip("no browser on this host")
    playwright = pytest.importorskip("playwright.async_api")

    source = "\n".join(
        ['fn main() {', '    let s = "line one', 'line two";', '    println!("{}", s);', '}']
    )
    # The real shape: per-line spans, chunked into display:block line groups.
    lines = source.split("\n")
    groups = "".join(
        '<span class="block" data-code-line-group="">'
        + "".join(f"<span>{ln}\n</span>" for ln in lines[i : i + 2])
        + "</span>"
        for i in range(0, len(lines), 2)
    )
    html = (
        '<!doctype html><meta charset="utf-8"><style>.block{display:block}'
        "code{white-space:pre}</style>"
        '<div data-is-streaming="false"><div class="p-3.5">rust</div>'
        f'<pre><code class="language-rust">{groups}</code></pre></div>'
    )
    page_file = Path(tempfile.mkdtemp()) / "reply.html"
    page_file.write_text(html, encoding="utf-8")

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=str(chrome), args=["--no-sandbox"]
            )
            page = await (await browser.new_context()).new_page()
            await page.goto(page_file.as_uri())
            read = await _Tab._read(page.locator("div[data-is-streaming]").first)
            await browser.close()
            return read

    code = extract_code(asyncio.run(go()), "main", "rust")
    assert code == source, f"the reader did not return the block verbatim:\n{code!r}"
    assert not code.lstrip().startswith("rust"), "the language chip leaked in"


# The code block's own copy control, used as a LAST RESORT when the code
# selectors match nothing. The copied value is intercepted inside the page:
# reading the system clipboard back would be catastrophic here, because there
# is one clipboard shared by every tab, every browser on the display, and every
# miner process the operator runs. Measured, two tabs in one browser: A wrote
# 'TAB-A-CODE', B wrote 'TAB-B-CODE', A read back 'TAB-B-CODE'. A pool reading
# the clipboard would submit another task's program whenever two solves
# overlapped -- silently, and with no way to tell afterwards.

COPY_PROGRAM = 'def pong():\n    note = """one\ntwo"""\n    return note'


def _chromium_or_skip():
    chrome = Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    if not chrome.exists():
        pytest.skip("no browser on this host")
    return pytest.importorskip("playwright.async_api"), str(chrome)


def _served(body: str) -> str:
    page = Path(tempfile.mkdtemp()) / "reply.html"
    page.write_text(body, encoding="utf-8")
    return page.as_uri()


def test_the_copy_control_recovers_code_the_selectors_cannot_see():
    """The recurring failure this covers: the DOM moves, `pre code` stops
    matching, and a complete answer is reported as no answer at all."""
    playwright, chrome = _chromium_or_skip()
    url = _served('<!doctype html><meta charset="utf-8">\n<div data-message-author-role="assistant">\n  <div id="src">__PROGRAM__</div>\n  <button aria-label="Copy" id="c">copy</button>\n  <button aria-label="Run code">run</button>\n</div>\n<script>\ndocument.getElementById(\'c\').onclick = () =>\n  navigator.clipboard.writeText(document.getElementById(\'src\').textContent);\n</script>'.replace("__PROGRAM__", COPY_PROGRAM))
    site = _site(copy=('button[aria-label="Copy"]',))

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            ctx = await browser.new_context()
            await ctx.grant_permissions(["clipboard-read", "clipboard-write"])
            page = await ctx.new_page()
            await page.goto(url)
            await page.evaluate("navigator.clipboard.writeText('SENTINEL')")
            reply = page.locator('[data-message-author-role="assistant"]').first
            recovered = await _tab(page, site)._copied_code(reply)
            clipboard = await page.evaluate("navigator.clipboard.readText()")
            await browser.close()
            return recovered, clipboard

    recovered, clipboard = asyncio.run(go())
    assert recovered is not None, "the copy control was not used"
    assert extract_code(recovered, "pong") == COPY_PROGRAM, f"garbled: {recovered!r}"
    assert clipboard == "SENTINEL", (
        f"the system clipboard was written to ({clipboard!r}); it is shared by "
        f"every tab and miner on this machine, so two overlapping solves could "
        f"swap answers"
    )


def _answering_site():
    return _site(
        composer=("#composer",), send=("#send",),
        assistant=('[data-message-author-role="assistant"]',),
        copy=('button[aria-label="Copy"]',),
    )


def _send_in_browser(body, then=None):
    playwright, chrome = _chromium_or_skip()
    url = _served(body)

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            reply = await _tab(page, _answering_site()).send("solve it", 5.0)
            extra = await page.evaluate(then) if then else None
            await browser.close()
            return reply, extra

    return asyncio.run(go())


def test_the_copy_control_is_preferred_over_the_rendered_dom(capsys):
    """Why the copy control leads rather than backs up.

    `pre code` hands over the source AFTER a syntax highlighter has rebuilt it
    as DOM; the copy control hands over what the model actually wrote. They
    differ, and they differed in production: a highlighter put U+E027 -- a
    Private Use Area character present in no source file -- inside a Python
    answer, and the solve died on a character nobody could see. Sanitising that
    after the fact is chasing damage the copy path never takes.

    Also pins the cost: one click per send, not one per poll.
    """
    from solvers.prompts import python_defect

    reply, clicks = _send_in_browser('<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\n// The source the model wrote. The copy control hands this over verbatim.\nconst SOURCE = "def pong():\\n    total = 1 + 2\\n    return total";\nwindow.__clicks = 0;\ndocument.getElementById(\'send\').onclick = () => {\n  const wrap = document.createElement(\'div\');\n  wrap.setAttribute(\'data-message-author-role\', \'assistant\');\n  const pre = document.createElement(\'pre\');\n  const code = document.createElement(\'code\');\n  // A syntax highlighter rebuilding the source as DOM, and slipping in a\n  // Private Use Area character that exists in no source file. This is the\n  // real production bug: U+E027 inside a Python answer.\n  code.textContent = SOURCE.replace("1 + 2", "1 \\uE027+ 2");\n  pre.appendChild(code);\n  wrap.appendChild(pre);\n  const btn = document.createElement(\'button\');\n  btn.setAttribute(\'aria-label\', \'Copy\');\n  btn.textContent = \'copy\';\n  btn.onclick = () => { window.__clicks++; navigator.clipboard.writeText(SOURCE); };\n  wrap.appendChild(btn);\n  document.getElementById(\'host\').appendChild(wrap);\n};\n</script>', then="window.__clicks")
    code = extract_code(reply, "pong")

    assert "\ue027" not in code, f"took the highlighter's DOM over the source: {code!r}"
    assert python_defect(code, "pong") is None, python_defect(code, "pong")
    scope: dict = {}
    exec(compile(code, "<submitted>", "exec"), scope)
    assert scope["pong"]() == 3, "the recovered source does not run"
    assert clicks == 1, f"clicked the copy control {clicks} times, expected once per send"

    # Preferring the copy silently would leave the next render bug as invisible
    # as the last three were. Two readings are already in hand, so say when they
    # disagree, and name the character a human can act on.
    logged = capsys.readouterr().out
    assert "RENDERS and what it COPIES are not the same" in logged, (
        f"took the copy but never said the page rendered something else: {logged!r}"
    )
    assert "U+E027" in logged, f"did not name the offending codepoint: {logged!r}"


def test_agreeing_readings_are_not_reported_as_a_problem(capsys):
    """The warning has to mean something. If it fires on every answer, nobody
    reads it, and the one time it matters it is lost in the noise."""
    body = '<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\nconst SOURCE = "def pong():\\n    return 5";\ndocument.getElementById(\'send\').onclick = () => {\n  const wrap = document.createElement(\'div\');\n  wrap.setAttribute(\'data-message-author-role\', \'assistant\');\n  const pre = document.createElement(\'pre\');\n  const code = document.createElement(\'code\');\n  code.textContent = SOURCE;                       // render and source agree\n  pre.appendChild(code);\n  wrap.appendChild(pre);\n  const btn = document.createElement(\'button\');\n  btn.setAttribute(\'aria-label\', \'Copy\');\n  btn.onclick = () => navigator.clipboard.writeText(SOURCE);\n  wrap.appendChild(btn);\n  document.getElementById(\'host\').appendChild(wrap);\n};\n</script>'
    reply, _ = _send_in_browser(body)
    assert "return 5" in extract_code(reply, "pong"), f"lost the answer: {reply!r}"
    assert "not the same" not in capsys.readouterr().out, "cried wolf on a clean read"


def test_a_missing_copy_control_falls_back_to_reading_the_dom():
    """The copy control is preferred, not required: a site that never had one,
    or renamed it, must still be read rather than reported as silent."""
    reply, _ = _send_in_browser('<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\ndocument.getElementById(\'send\').onclick = () => {\n  const wrap = document.createElement(\'div\');\n  wrap.setAttribute(\'data-message-author-role\', \'assistant\');\n  const pre = document.createElement(\'pre\');\n  const code = document.createElement(\'code\');\n  code.textContent = "def pong():\\n    return 7";\n  pre.appendChild(code);\n  wrap.appendChild(pre);\n  document.getElementById(\'host\').appendChild(wrap);   // no copy control at all\n};\n</script>')
    assert "return 7" in extract_code(reply, "pong"), f"lost the answer: {reply!r}"


def test_a_control_that_does_not_call_itself_copy_is_never_pressed():
    """A selector is a guess about structure and can drift onto a neighbour.
    ChatGPT keeps "Run code" in the same header as "Copy": reading the answer is
    worth a click, executing it is not. So the control's own name is checked
    before anything is pressed, and a mismatch falls back to scraping."""
    playwright, chrome = _chromium_or_skip()
    url = _served('<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\nwindow.__ran = false;\ndocument.getElementById(\'send\').onclick = () => {\n  const wrap = document.createElement(\'div\');\n  wrap.setAttribute(\'data-message-author-role\', \'assistant\');\n  const pre = document.createElement(\'pre\');\n  const code = document.createElement(\'code\');\n  code.textContent = "def pong():\\n    return 9";\n  pre.appendChild(code);\n  wrap.appendChild(pre);\n  // The selector has drifted onto the neighbour ChatGPT keeps in the same\n  // header. Pressing this would execute the answer instead of reading it.\n  const btn = document.createElement(\'button\');\n  btn.setAttribute(\'data-role\', \'copyish\');\n  btn.setAttribute(\'aria-label\', \'Run code\');\n  btn.onclick = () => { window.__ran = true; };\n  wrap.appendChild(btn);\n  document.getElementById(\'host\').appendChild(wrap);\n};\n</script>')
    site = _site(
        composer=("#composer",), send=("#send",),
        assistant=('[data-message-author-role="assistant"]',),
        copy=('button[data-role="copyish"]',),   # matches, but it is not Copy
    )

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            reply = await _tab(page, site).send("solve it", 5.0)
            ran = await page.evaluate("window.__ran")
            await browser.close()
            return reply, ran

    reply, ran = asyncio.run(go())
    assert not ran, "pressed a control labelled 'Run code'"
    assert "return 9" in extract_code(reply, "pong"), f"lost the answer: {reply!r}"


def test_an_assistant_selector_that_dies_mid_answer_is_re_resolved(capsys):
    """Captured nothing, and the answer was on screen the whole time.

    Sites stream a message under one attribute and drop it when the message is
    done. The candidate that found the message is then the one that cannot see
    it, and a latch held for the whole send reads nothing for the rest of it --
    which surfaces, much later and much less usefully, as "the reply contained
    no code".

    The transition is driven from here rather than a page timer: Chromium
    throttles timers in background pages, and every tab this miner owns is a
    background page.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served('<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\ndocument.getElementById(\'send\').onclick = () => {\n  const d = document.createElement(\'div\');\n  d.setAttribute(\'data-is-streaming\', \'true\');   // painted, still empty\n  document.getElementById(\'host\').appendChild(d);\n};\n</script>')
    site = _site(
        composer=("#composer",), send=("#send",),
        assistant=("div[data-is-streaming]", "div.done-msg"),
    )
    finish = """() => {
        const d = document.querySelector('[data-is-streaming]');
        d.removeAttribute('data-is-streaming');
        d.className = 'done-msg';
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.textContent = 'def pong():\\n    return 4';
        pre.appendChild(code);
        d.appendChild(pre);
    }"""

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            sending = asyncio.create_task(tab.send("solve it", 8.0))
            # Wait for the latch itself, not for a length of time: a fixed
            # sleep encodes a guess about how fast this machine is, and this
            # suite runs a browser per test. The latch going non-None IS the
            # state under test — the streaming candidate has won.
            deadline = time.monotonic() + 5
            while tab._assistant is None and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert tab._assistant == "div[data-is-streaming]", tab._assistant
            await page.evaluate(finish)   # ...which now matches nothing
            reply = await sending
            await browser.close()
            return reply

    reply = asyncio.run(go())
    assert "return 4" in extract_code(reply, "pong"), f"captured nothing: {reply!r}"
    assert "stopped matching mid-answer" in capsys.readouterr().out, "re-resolved silently"


def test_capturing_nothing_says_why_while_the_page_can_still_be_asked(capsys):
    """"The reply contained no code" describes a selector that matches nothing,
    a reply that never rendered and an answer still streaming, identically. The
    page can tell them apart in four queries, and only at the time."""
    playwright, chrome = _chromium_or_skip()
    url = _served('<!doctype html><meta charset="utf-8">\n<div id="composer" contenteditable="true"></div><button id="send">go</button>\n<div id="host"></div>\n<script>\ndocument.getElementById(\'send\').onclick = () => {\n  const d = document.createElement(\'div\');\n  d.className = \'renamed-by-the-site\';     // nothing the miner knows about\n  d.textContent = \'def pong(): return 1\';\n  document.getElementById(\'host\').appendChild(d);\n};\n</script>')
    site = _site(
        composer=("#composer",), send=("#send",),
        assistant=('[data-message-author-role="assistant"]',),
    )

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            reply = await _tab(page, site).send("solve it", 3.0)
            await browser.close()
            return reply

    assert asyncio.run(go()) == "", "expected an empty capture for this page"
    logged = capsys.readouterr().out
    assert "captured NOTHING" in logged, f"stayed silent about an empty read: {logged!r}"
    assert "no assistant selector matched" in logged, f"did not name the cause: {logged!r}"
    assert "_ASSISTANT" in logged, f"did not name the fix: {logged!r}"


def test_a_defect_is_not_reported_as_a_failed_run():
    """Defects are found BEFORE anything executes. "I ran the program against
    the examples and got: the program does not define `fn main()`" is not
    evidence, it is a contradiction -- and a model told its logic failed will
    rewrite the logic, which was never the problem."""
    prompt = _repair_prompt(
        "rust", defect="the program does not define `fn main()`", found_by="unrun",
    )
    assert "does not define `fn main()`" in prompt
    assert "NOTHING WAS RUN" in prompt, "still claims to have executed it"
    assert "the logic has not been judged" in prompt, (
        "still blames logic that never ran"
    )


def test_the_repair_round_hears_about_the_defect_not_about_the_examples():
    """The wiring, not the wording: `_grade` returns a defect OR failures, never
    both, and merging them into one list of "problems" was what put a defect
    under the "I ran it" heading in the first place."""
    prompts: list[str] = []

    class _Recording(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            prompts.append(text)
            return await super().send(text, timeout_s, extend_to_s)

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Recording(self._script, self._provider)

    task = SolveTask(
        problem_id="defect", language="rust", statement="Print 42.",
        entrypoint="main", public_examples=[], deadline_s=120.0,
    )
    solver = VerifyingSolver(
        # A helper with no `fn main` -- the defect this test is about -- then a
        # real program. Live traffic ships no public examples, so the compile
        # gate is the ONLY thing that can catch the first one, and it has to
        # drive the repair on its own.
        _Backend2(["```rust\nfn helper() {}\n```",
                   '```rust\nfn main() { println!("42"); }\n```']),
        reserve_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))

    assert "println!" in answer.code, "never got past the defect"
    repairs = [p for p in prompts if "Repair it" in p]
    assert len(repairs) == 1, f"expected exactly one repair, got {len(repairs)}"
    # The defect is what the repair hears about, under its own heading rather
    # than under "I ran it against the examples".
    assert "fn main" in repairs[0], repairs[0]
    assert "A local check also reports" in repairs[0], repairs[0]


def test_both_providers_are_told_to_keep_long_code_in_the_chat():
    """A long program is exactly when a model moves the answer into a side panel
    the reader cannot see, so both nudges have to say so — and say it about
    length, which is the trigger."""
    assert "however long" in claude_site().nudge
    assert "artifact" in claude_site().nudge
    assert "however long" in chatgpt_site().nudge
    assert "canvas" in chatgpt_site().nudge


def test_the_claude_prompt_asks_for_an_inline_code_block():
    """Long code can land in the artifacts panel, outside the message the
    reader scrapes. One sentence is cheaper than scraping the panel."""
    site = claude_site()
    assert "artifact" in site.nudge.lower()
    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    asyncio.run(_tab(page, _site(nudge=site.nudge)).send("solve it", 1.0))
    assert page.typed[0].endswith(site.nudge)


def test_selector_lists_are_overridable_from_the_environment(monkeypatch):
    """A DOM change must be a one-line .env fix, not a patch. `|` separates
    candidates because `,` is already CSS's own 'either' operator."""
    from solvers.config import selectors

    assert selectors("T_ASSISTANT", ("a", "b")) == ("a", "b")
    monkeypatch.setenv("T_ASSISTANT", 'div[x="1"] | .y')
    assert selectors("T_ASSISTANT", ("a",)) == ('div[x="1"]', ".y")


def test_dotenv_values_fill_in_without_overriding_the_real_environment(monkeypatch, tmp_path):
    """The miner's settings come from .env via pydantic-settings, which never
    touches os.environ — so backend knobs written there were being ignored."""
    from solvers.config import load_env_file

    env = tmp_path / ".env"
    env.write_text('# comment\nCLAUDE_CDP=9222,9223\nexport CLAUDE_URL="https://claude.ai/new"\nSHELL_WINS=from-file\n')
    monkeypatch.setenv("SHELL_WINS", "from-shell")
    monkeypatch.delenv("CLAUDE_CDP", raising=False)
    monkeypatch.delenv("CLAUDE_URL", raising=False)
    assert load_env_file(env) == 2
    import os

    assert os.environ["CLAUDE_CDP"] == "9222,9223"
    assert os.environ["CLAUDE_URL"] == "https://claude.ai/new"
    assert os.environ["SHELL_WINS"] == "from-shell"


def test_dotenv_parses_the_way_pydantic_settings_will(tmp_path, monkeypatch):
    """These values get PROMOTED into os.environ, where they outrank the file
    pydantic-settings reads — so anything parsed differently here silently
    changes the miner's own settings. A trailing comment is the common case."""
    from solvers.config import load_env_file

    env = tmp_path / ".env"
    env.write_text('AXON_PORT=8091  # the port to open\nWALLET_NAME="my wallet"  # note\n')
    monkeypatch.delenv("AXON_PORT", raising=False)
    monkeypatch.delenv("WALLET_NAME", raising=False)
    load_env_file(env)
    import os

    assert os.environ["AXON_PORT"] == "8091"
    assert os.environ["WALLET_NAME"] == "my wallet"


def test_the_env_file_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    """One .env at the repo root has to configure both the miner (run from the
    root) and the doctor (run from examples/custom_miner)."""
    from solvers.config import find_env_file

    (tmp_path / ".env").write_text("CLAUDE_CDP=9222\n")
    nested = tmp_path / "examples" / "custom_miner"
    nested.mkdir(parents=True)
    assert find_env_file(nested) == tmp_path / ".env"


def test_a_missing_playwright_names_the_fix_instead_of_raising_importerror():
    """The API backends do not need Playwright, so it is in no extra — which
    makes a bare ImportError the most likely first experience."""
    import inspect

    from solvers.browser_pool import import_playwright

    source = inspect.getsource(import_playwright)
    assert "pip install playwright" in source and "SystemExit" in source
    # No browser download is needed — the pool attaches to one you started —
    # so the message must not send anyone chasing a `playwright install`.
    assert "No `playwright install` is needed" in source


def test_a_browser_backend_is_started_before_serving_not_on_first_request():
    """An expired login must surface at launch, where someone is watching, not
    hours later as a failed solve on a real validator request."""
    from solvers.roster import warm_up

    started = []

    class _Pool:
        site = _site()

        async def start(self):
            started.append(True)

        def stats(self):
            return {"tabs": 1}

    solver = VerifyingSolver(_Pool())
    asyncio.run(warm_up(solver, 1))
    assert started == [True]
    # An API backend has nothing to warm up and must not be a problem.
    asyncio.run(warm_up(_solver([RIGHT]), 1))


# --- the answer as it came off the wire ---------------------------------- #
# The source above every other one this reader has: the markdown the model
# emitted, captured before the page turned any of it into DOM. `pre code` is
# that source after a syntax highlighter rebuilt it; even the copy control is
# the framework's own copy of a block it has already parsed. The wire is also
# the only source with anything left to say when the page read comes back
# empty, which is the failure that keeps arriving as "the reply contained no
# code" — a selector that stopped matching, an id swapped mid-stream, a render
# this tab cannot see.


def test_a_block_that_contains_a_fence_is_not_cut_at_the_inner_one():
    """Markdown's own rule: the closing fence must be at least as long as the
    opening one. A parser that stops at the first ``` truncates every answer
    written with four backticks — which is exactly how a model writes a block
    that has markdown inside it."""
    blocks = _fenced_blocks("````md\n```py\nx = 1\n```\n````\n")
    assert blocks == ["```py\nx = 1\n```\n"], blocks


def test_a_fence_the_deadline_cut_off_is_still_an_answer():
    """A reply the budget interrupted still has its program in it. Requiring a
    closing fence would turn a recoverable partial answer into no answer."""
    assert _fenced_blocks("here:\n```python\ndef f():\n    return 1\n") == [
        "def f():\n    return 1\n"
    ]


CLAUDE_ANSWER = "```python\ndef solve(xs):\n    return sorted(xs)\n```\n\nDone."
LONG_THOUGHTS = "Let me reason this through carefully. " * 120


def _claude_wire(answer: str = CLAUDE_ANSWER) -> str:
    """A Claude-shaped SSE body: reasoning, a signature, then the answer."""
    events = [
        "event: message_start",
        'data: {"type":"message_start","message":'
        '{"id":"msg_1","role":"assistant","model":"claude"}}',
    ]
    for i in range(0, len(LONG_THOUGHTS), 25):
        events.append("data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": LONG_THOUGHTS[i:i + 25]},
        }))
    events.append("data: " + json.dumps({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "signature_delta", "signature": "x" * 900},
    }))
    for i in range(0, len(answer), 6):
        events.append("data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": answer[i:i + 6]},
        }))
    events.append('data: {"type":"message_stop"}')
    return "\n\n".join(events) + "\n\n"


def _chatgpt_wire(answer: str = CLAUDE_ANSWER) -> str:
    """A ChatGPT-shaped SSE body, in its operation encoding.

    Three things here are not padding. The answer and the site's own
    bookkeeping come down the SAME `v` field, told apart only by the sibling
    operation. After the first append the operation is OMITTED and bare
    `{"v": "..."}` means "as before" — so the qualifier has to carry forward or
    the answer's opening chunk lands in its own group and is lost, and this
    answer opens with the fence. And a metadata `replace` lands in the middle.
    """
    events = []

    def send(obj):
        events.append("data: " + json.dumps(obj))

    send({"v": {"message": {"id": "abc", "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": [""]},
                            "status": "in_progress"}, "c": 0}})
    first = True
    for i in range(0, len(LONG_THOUGHTS), 30):
        if first:
            send({"p": "/message/content/thoughts/0/content", "o": "append",
                  "v": LONG_THOUGHTS[i:i + 30]})
            first = False
        else:
            send({"v": LONG_THOUGHTS[i:i + 30]})
    send({"p": "/message/metadata/finished_text", "o": "replace",
          "v": "Thought for 12 seconds"})
    first = True
    for i in range(0, len(answer), 5):
        if first:
            send({"p": "/message/content/parts/0", "o": "append",
                  "v": answer[i:i + 5]})
            first = False
        else:
            send({"v": answer[i:i + 5]})
        if i == 20:
            send({"p": "/message/metadata/model_slug", "o": "replace", "v": "gpt-x"})
            first = True   # the site re-states the operation after interrupting
    send({"p": "/message/status", "o": "replace", "v": "finished_successfully"})
    events.append("data: [DONE]")
    return "\n\n".join(events) + "\n\n"


def _streaming_page(playwright, chrome, bodies, page_html="<!doctype html>hi"):
    """A browser whose `/sse` returns each body in turn, with the hook armed.

    Routed rather than served from a file because the capture only fires on a
    streaming content type, and that is the property being tested.
    """

    async def opened(p):
        browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        page = await (await browser.new_context()).new_page()
        await page.add_init_script(_STREAM_INSTALL)
        turn = {"n": 0}

        def stream(route):
            body = bodies[min(turn["n"], len(bodies) - 1)]
            turn["n"] += 1
            return asyncio.ensure_future(route.fulfill(
                status=200, content_type="text/event-stream", body=body))

        await page.route("**/sse", stream)
        await page.route("**/blank", lambda r: asyncio.ensure_future(
            r.fulfill(status=200, content_type="text/html", body=page_html)))
        await page.goto("https://example.test/blank")
        return browser, page

    return opened


@pytest.mark.parametrize("shape", ["claude", "chatgpt"])
def test_the_wire_answer_is_reconstructed_without_knowing_the_schema(shape):
    """Neither site publishes its stream format, and both change theirs without
    telling anyone, so nothing about either is hard-coded. What is relied on is
    structural: an SSE stream is many small JSON events and the answer is the
    one field appended to over and over. Both real shapes are pinned here
    because a heuristic that fits one and not the other is not a heuristic."""
    playwright, chrome = _chromium_or_skip()
    body = _claude_wire() if shape == "claude" else _chatgpt_wire()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(playwright, chrome, [body])(p)
            await page.evaluate("async () => { await (await fetch('/sse')).text(); }")
            await page.wait_for_function("(window.__honeStreams||[])[0]")
            await asyncio.sleep(0.2)
            out = await page.evaluate(_STREAM_READ, 0)
            await browser.close()
            return out

    got = asyncio.run(go())
    assert got == CLAUDE_ANSWER, f"reconstructed something else: {got!r}"
    assert _fenced_blocks(got) == ["def solve(xs):\n    return sorted(xs)\n"]


@pytest.mark.parametrize("shape", ["claude", "chatgpt"])
def test_the_models_rough_work_is_never_mistaken_for_its_answer(shape):
    """Both bodies above carry far more reasoning than answer — Claude under
    `delta.thinking`, ChatGPT under `/message/content/thoughts/...`. Picking
    the largest group without excluding those submits the model's scratchpad,
    which is a failure this miner has already had once."""
    playwright, chrome = _chromium_or_skip()
    body = _claude_wire() if shape == "claude" else _chatgpt_wire()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(playwright, chrome, [body])(p)
            await page.evaluate("async () => { await (await fetch('/sse')).text(); }")
            await asyncio.sleep(0.2)
            out = await page.evaluate(_STREAM_READ, 0)
            await browser.close()
            return out

    got = asyncio.run(go()) or ""
    assert "reason this through" not in got, f"submitted the thinking: {got[:120]!r}"
    assert len(got) < len(LONG_THOUGHTS), "took the larger group on size alone"


def test_the_network_capture_cannot_break_the_page():
    """This patches `fetch` on a signed-in account the operator cares about, so
    what it must never do matters more than what it does.

    `res.clone()` and not `res.body.tee()` with a hand-built `Response`: a
    constructed one loses `url` and `redirected`, and a chat UI reading either
    breaks in a way that looks like the site's own bug. The body must be left
    unread for the app, non-streaming requests must not be touched at all, and
    a fetch that fails must still fail.
    """
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(playwright, chrome, ['data: {"a":"x"}\n\n'])(p)
            await page.route("**/thing.json", lambda r: asyncio.ensure_future(
                r.fulfill(status=200, content_type="application/json", body='{"k":1}')))
            out = await page.evaluate("""async () => {
                const s = await fetch('/sse');
                const seen = {url: s.url, status: s.status, ok: s.ok,
                              redirected: s.redirected, type: s.type,
                              bodyUsed: s.bodyUsed};
                seen.body = await s.text();
                const j = await fetch('/thing.json');
                seen.json = await j.json();
                seen.captured = (window.__honeStreams || []).length;
                try { await fetch('http://127.0.0.1:1/nope'); seen.failed = false; }
                catch (e) { seen.failed = true; }
                return seen;
            }""")
            await browser.close()
            return out

    seen = asyncio.run(go())
    assert seen["url"].endswith("/sse"), f"lost the response url: {seen['url']!r}"
    assert (seen["status"], seen["ok"], seen["type"]) == (200, True, "basic"), seen
    assert seen["redirected"] is False, seen
    assert seen["bodyUsed"] is False, "the app's own body had already been consumed"
    assert seen["body"] == 'data: {"a":"x"}\n\n', "the app got a different body"
    assert seen["json"] == {"k": 1}, seen
    assert seen["captured"] == 1, (
        f"cloned {seen['captured']} responses; ordinary requests must be untouched"
    )
    assert seen["failed"], "a fetch that should have failed resolved instead"


def _wire_site(**kw):
    return _site(
        composer=("#composer",), send=("#send",),
        assistant=('[data-message-author-role="assistant"]',),
        **kw,
    )


# A page that streams its answer but never renders it. That is not contrived:
# it is what every one of these failures looked like from Python — the reply
# was there, and the reader could not see it.
SILENT_PAGE = (
    '<!doctype html><meta charset="utf-8">'
    '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    '<div id="host"></div>'
    "<script>document.getElementById('send').onclick = () => "
    "{ fetch('/sse').then(r => r.text()); };</script>"
)


def test_the_wire_answers_when_the_page_reads_back_nothing(capsys):
    """The whole reason this path exists. Every other reading is downstream of
    a render, so when the render is unreadable they all return the same empty
    string and the solve is a guaranteed zero. The wire still has the answer."""
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [_claude_wire()], page_html=SILENT_PAGE)(p)
            reply = await _tab(page, _wire_site()).send("solve it", 6.0)
            await browser.close()
            return reply

    reply = asyncio.run(go())
    assert "def solve" in extract_code(reply, "solve"), f"still empty: {reply!r}"
    assert "read NOTHING from the page" in capsys.readouterr().out, (
        "took the wire without saying the page had failed"
    )


def test_a_previous_turns_stream_is_not_read_as_this_turns_answer():
    """The buffer holds several responses, and a repair round asks again in the
    same tab. Reading the whole buffer would re-submit the very answer the
    repair was sent to replace, which reads as the model ignoring the fix."""
    playwright, chrome = _chromium_or_skip()
    # Built from a different ANSWER, not by editing the finished body: the
    # answer is cut into 5- and 6-character JSON chunks before it is encoded,
    # so no phrase of it survives contiguously in the bytes and a string
    # replacement on the body silently changes nothing at all.
    #
    # And deliberately SHORTER than the first. Reading the whole buffer picks
    # the largest group in it, so a longer second answer would come out right
    # by accident and this test would pass with the floor removed — checked,
    # it did.
    second = _claude_wire("```python\ndef solve(xs):\n    return xs[::-1]\n```")

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [_claude_wire(), second], page_html=SILENT_PAGE)(p)
            tab = _tab(page, _wire_site())
            first = await tab.send("solve it", 6.0)
            again = await tab.send("no, reverse it", 6.0)
            await browser.close()
            return first, again

    first, again = asyncio.run(go())
    assert "sorted(xs)" in first, f"lost the first answer: {first!r}"
    assert "xs[::-1]" in again, f"re-submitted the first answer: {again!r}"
    assert "sorted(xs)" not in again, f"the old turn leaked in: {again!r}"


RENDERING_PAGE = (
    '<!doctype html><meta charset="utf-8">'
    '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    '<div id="host"></div><script>'
    "document.getElementById('send').onclick = () => {"
    # Rendered FROM the response, as a real chat UI does. A page that paints
    # independently races the stream, and the comparison under test would
    # then depend on which of the two happened to win.
    "  fetch('/sse').then(r => r.text()).then(() => {"
    "  const w = document.createElement('div');"
    "  w.setAttribute('data-message-author-role', 'assistant');"
    "  const pre = document.createElement('pre'), code = document.createElement('code');"
    # No trailing newline, where the wire's block has one. That is the single
    # most common difference between two honest readings of the same answer --
    # `textContent` keeps the newline before a closing tag, a copy control
    # trims it, a fenced block always has one -- and reporting it would fire
    # the warning on every clean reply until nobody read it any more.
    "  code.textContent = 'def solve(xs):\\n    return sorted(xs)';"
    "  pre.appendChild(code); w.appendChild(pre);"
    "  document.getElementById('host').appendChild(w); });"
    "};</script>"
)


def test_the_page_is_believed_over_the_wire_until_an_operator_says_otherwise(capsys):
    """The wire is the better source in principle and unverifiable in practice:
    both formats are private, so the reconstruction is a heuristic that could be
    silently wrong after any deploy. A heuristic that quietly replaced a good
    answer with a bad one would be worse than the bug it was written to fix. So
    it rescues an empty read, it reports every disagreement, and it takes over
    only when an operator who has watched the two agree turns it on."""
    playwright, chrome = _chromium_or_skip()
    wire = _claude_wire(CLAUDE_ANSWER.replace("sorted(xs)", "WIRE_ONLY(xs)"))

    async def go(stream_first):
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [wire], page_html=RENDERING_PAGE)(p)
            site = _wire_site(stream_first=stream_first)
            reply = await _tab(page, site).send("solve it", 6.0)
            await browser.close()
            return reply

    reply = asyncio.run(go(False))
    assert "sorted(xs)" in reply, f"took the wire by default: {reply!r}"
    assert "WIRE_ONLY" not in reply, f"took the wire by default: {reply!r}"
    logged = capsys.readouterr().out
    assert "not the same" in logged, f"said nothing about the disagreement: {logged!r}"
    assert "T_STREAM_FIRST=1" in logged, f"did not name the override: {logged!r}"

    # ...and with it on, the wire is what gets submitted.
    assert "WIRE_ONLY" in asyncio.run(go(True)), "the override does nothing"


def test_agreeing_sources_are_not_reported_as_a_disagreement(capsys):
    """A warning that fires on every clean answer is a warning nobody reads.
    Trailing newlines especially: `textContent` on a `<code>` keeps the one
    before the closing tag and a fenced block always ends in one, so comparing
    them raw would cry wolf on literally every reply."""
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [_claude_wire()], page_html=RENDERING_PAGE)(p)
            reply = await _tab(page, _wire_site()).send("solve it", 6.0)
            await browser.close()
            return reply

    reply = asyncio.run(go())
    assert "sorted(xs)" in reply, f"lost the answer: {reply!r}"
    assert "not the same" not in capsys.readouterr().out, "cried wolf on a clean read"


def test_the_capture_can_be_switched_off_entirely():
    """A tab told not to touch the network must behave exactly as it did before
    any of this existed — including reading nothing when the page shows
    nothing, which is the honest old answer."""
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [_claude_wire()], page_html=SILENT_PAGE)(p)
            reply = await _tab(page, _wire_site(stream=False)).send("solve it", 4.0)
            await browser.close()
            return reply

    assert asyncio.run(go()) == "", "read the wire with the capture switched off"


def test_one_read_that_hangs_does_not_eat_the_whole_budget():
    """Found by accident, and the most expensive bug in this file.

    A poll resolves a node and then reads it. If the site swaps the message
    between those two steps, the read waits on an element that no longer
    matches — and Playwright auto-waits THIRTY SECONDS. Bounded only by the
    send's remaining budget, that one poll spends every second the solve had
    left and returns nothing, while the finished answer sits on screen the
    whole time. Measured on the real DOM transition, before the fix: an 8s
    send spent 7.85s inside a single `inner_text()` and returned "". On a solve
    with a five-minute budget that is a five-minute stall and a certain zero,
    and it arrives as "the reply contained no code" like everything else.

    The hang is injected rather than raced for. Aiming a DOM change at the
    inside of an in-flight read does reproduce it — that is how it was found —
    but only sometimes, and a test that catches a budget-eating bug two runs in
    three is worse than no test at all: it goes green on the broken code and
    gets believed. What must hold is simply that ONE unreturning read cannot
    spend the whole send, whatever made it hang.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div');"
        "  d.className = 'done-msg';"
        "  const pre = document.createElement('pre');"
        "  const code = document.createElement('code');"
        "  code.textContent = 'def pong():\\n    return 4';"
        "  pre.appendChild(code); d.appendChild(pre);"
        "  document.getElementById('host').appendChild(d);"
        "};</script>"
    )
    site = _site(composer=("#composer",), send=("#send",), assistant=("div.done-msg",))

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            reads, real = {"n": 0}, tab._poll

            async def hangs_once(before):
                reads["n"] += 1
                if reads["n"] == 1:
                    await asyncio.sleep(3600)   # the auto-wait that never lands
                return await real(before)

            tab._poll = hangs_once
            started = time.monotonic()
            reply = await tab.send("solve it", 20.0)
            elapsed = time.monotonic() - started
            await browser.close()
            return reply, elapsed, reads["n"]

    reply, elapsed, reads = asyncio.run(go())
    assert reads > 1, "the loop gave up after the first read instead of retrying"
    assert "return 4" in extract_code(reply, "pong"), f"captured nothing: {reply!r}"
    assert elapsed < 10, (
        f"the answer took {elapsed:.1f}s of a 20s budget: one read is still "
        f"allowed to run to the deadline"
    )


def test_a_tab_the_fleet_opens_has_the_capture_armed():
    """`_spawn` is the ONLY place a real tab gets the hook, and every other
    test in this file installs it by hand — so removing the line from `_spawn`
    left the whole suite green. Checked: it did.

    It also pins how the hook is installed. `add_init_script` takes script
    source, not a function the way `evaluate` does; handed the arrow function
    it builds one, discards it, and arms nothing, in total silence. It has to
    go in before the goto, too, or the site's own bundle wraps `fetch` first.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    )
    async def go(stream):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            ctx = await browser.new_context()
            fleet = BrowserFleet([Browser("http://127.0.0.1:0", _site(
                url=url, composer=("#composer",), send=("#send",),
                assistant=("div.msg",), stream=stream))])
            tab = await fleet._spawn(ctx, fleet._browsers_wanted[0], "t")
            armed = None
            if tab is not None:
                armed = await tab._page.evaluate("!!window.__honeStreamHooked")
            await browser.close()
            return tab, armed

    tab, armed = asyncio.run(go(True))
    assert tab is not None, "the fleet could not open a tab on this page at all"
    assert armed is True, "a tab the fleet opened has no network capture on it"
    # ...and a site told not to touch the network gets a tab that never does.
    assert asyncio.run(go(False))[1] is False, "armed a tab with the capture off"


def test_a_round_that_captured_nothing_never_replaces_a_flawed_program():
    """`best so far` is what gets submitted, so what it ranks matters.

    Emptiness is not a defect — there is nothing there to be wrong — so a
    ranking that asks "is it runnable?" before "is there anything there?" lets
    a round that read NOTHING outrank a round that returned a program with a
    fixable flaw, and take its place as best. Both score zero on chain, but one
    is an answer and the other is the absence of one. `python_defect` is a
    static check, and a static check that is too strict must not be able to
    throw away work by being wrong.
    """
    from solvers.verify import Candidate

    def candidate(code, defect=None, passed=0):
        return Candidate(code=code, raw="", defect=defect, passed=passed, total=2)

    clean = candidate("def pong():\n    return 'pong'")
    flawed = candidate("pong = 1", defect="does not define pong()")
    nothing = candidate("")

    assert clean.score > flawed.score, "a repaired answer does not beat the broken one"
    assert flawed.score > nothing.score, "an empty round displaced a real program"
    assert candidate("x", passed=1).score > clean.score, "passing examples must win"


# --- what the prompt promises about the grader must stay true ------------- #
# The edge-case section makes specific factual claims: overflow is silent,
# `True` is not `1`, each test gets five seconds. Claims like those rot without
# anyone noticing — the prompt keeps saying them long after the policy that
# made them true has moved — and a prompt that confidently states something
# false is worse than one that says nothing, because the model acts on it.


def test_each_language_is_warned_about_its_own_way_of_losing_a_large_number():
    """The large-number failure is not the same failure in both languages, and
    telling either one the other's story wastes the only prompt there is:
    Python cannot overflow at all, and Rust cannot grow an integer."""
    # The ENVIRONMENT block, not the whole prompt. The trap block above it is
    # written from the statement and is language-agnostic -- "stresses overflow
    # and exactness" is a note about which INPUTS are worth trying, true in
    # both languages. What must not cross over is the environment's own story
    # about how each language loses a number, which is where the decision is.
    def environment(prompt):
        return prompt.split("THE ENVIRONMENT IT RUNS IN:", 1)[1].lower()

    rust = environment(_candidate_prompt("rust"))
    python = environment(_candidate_prompt("python"))

    assert "overflow is silent" in rust and "i64" in rust
    assert "wraps" in rust, "did not say what silent overflow actually does"
    for rust_only in ("i64", "i128", "wraps", "opt-level"):
        assert rust_only not in python, (
            f"told Python about {rust_only!r}, which is Rust's failure"
        )
    assert "never overflow" in python, (
        "left Python to guess whether its integers can overflow"
    )
    assert "recursion limit is 1000" in python
    assert "recursion limit" not in rust, "told Rust about Python's limit"


def test_the_prompts_claim_about_silent_overflow_is_true_of_this_grader():
    """The most valuable sentence in the prompt, and the one most able to go
    quietly wrong.

    `rustc` disables overflow checks whenever opt-level > 0, and the validator
    compiles at opt-level=2 — so `i32` arithmetic WRAPS and the program exits 0
    with a plausible wrong number instead of panicking. There is no message and
    nothing in the failure that points at the cause, which is exactly why the
    model has to be told up front.

    Compiled here with the validator's own flags rather than a copy of them, so
    that changing `RELEASE_POLICY.rustc_flags` — adding `-C
    debug-assertions=on`, say — fails this test instead of leaving the prompt
    asserting something that stopped being true.
    """
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("no rustc on this host")
    from rlvr.policy import RELEASE_POLICY
    from solvers.prompts import RUST_ENVIRONMENT

    work = Path(tempfile.mkdtemp())
    src = work / "ov.rs"
    src.write_text(
        "fn main() {\n"
        "    let vals: Vec<i32> = vec![2_000_000_000, 2_000_000_000];\n"
        "    let mut t: i32 = 0;\n"
        "    for v in &vals { t += *v; }\n"
        "    println!(\"{}\", t);\n"
        "}\n",
        encoding="utf-8",
    )
    built = subprocess.run(
        [rustc, f"--edition={RELEASE_POLICY.rust_edition}",
         *RELEASE_POLICY.rustc_flags, "-o", str(work / "ov"), str(src)],
        capture_output=True, text=True, cwd=work,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(work / "ov")], capture_output=True, text=True)

    assert ran.returncode == 0, (
        f"the overflow panicked (exit {ran.returncode}); the prompt says it is "
        f"silent, so either the flags changed or the prompt is now wrong"
    )
    assert ran.stdout.strip() == "-294967296", (
        f"i32 overflow produced {ran.stdout.strip()!r}; the prompt tells the "
        f"model it produces -294967296"
    )
    assert "-294967296" in RUST_ENVIRONMENT, "the prompt stopped quoting the value"


def test_the_prompts_claims_about_answer_comparison_are_true():
    """The Python prompt tells the model `True` is not `1` and that a list and
    a tuple are interchangeable. Both are load-bearing — one makes it wrap a
    boolean answer, the other stops it wasting a repair round converting a
    perfectly acceptable tuple — and both are somebody else's code."""
    from rlvr.execution.compare import values_equal
    from solvers.prompts import PYTHON_ENVIRONMENT

    assert "`True` is not `1`" in PYTHON_ENVIRONMENT
    assert not values_equal(True, 1), "the prompt's bool claim is now false"

    assert "tuple with equal contents do compare equal" in PYTHON_ENVIRONMENT
    assert values_equal([1, 2], (1, 2)), "the prompt's list/tuple claim is false"

    assert "two integers must match exactly" in PYTHON_ENVIRONMENT
    assert not values_equal(1_000_000, 1_000_001), (
        "a float tolerance is accepting wrong integers; the prompt says it cannot"
    )


def test_the_prompt_quotes_the_real_per_test_timeout():
    """A budget the model is told about has to be the budget it gets. Quoting a
    generous one invites an algorithm that does not fit."""
    from rlvr.config import Settings
    from solvers.prompts import PYTHON_ENVIRONMENT, RUST_ENVIRONMENT

    seconds = Settings.model_fields["per_test_timeout_s"].default
    assert seconds == 5.0, (
        f"the per-test timeout is now {seconds}s; both prompts still say 5"
    )
    assert "5 seconds" in PYTHON_ENVIRONMENT and "5 seconds" in RUST_ENVIRONMENT


def test_the_examples_are_framed_as_a_floor_not_the_specification():
    """The examples now sit WITH the problem rather than after the checklists,
    and the reversal is deliberate.

    They used to come last so the checklist would be read first. That bought
    one thing and cost another: instructions about how to solve something are
    unreadable before you know what it is, and the task was buried under two
    kilobytes of advice. The label does the anti-over-fitting work on its own —
    it says in the same breath that these are a floor and that the cases below
    still apply — so the task can be where a task belongs.
    """
    prompt = _candidate_prompt(
        "python", [{"args": [[1]], "kwargs": {}, "expected": 1}]
    )
    flat = " ".join(prompt.split())
    assert "a floor, not the specification" in flat
    assert "already known to be right" in flat, "the label drops their standing"
    assert prompt.index("PROBLEM STATEMENT") < prompt.index("WORKED EXAMPLES"), (
        "the examples are separated from the problem they belong to"
    )
    # ...and a task with none says nothing about examples at all, rather than
    # labelling an empty list. Every one of the 97 archived requests is this.
    assert "WORKED EXAMPLES" not in _candidate_prompt("python")


def test_the_output_contract_holds_the_first_word_and_the_nudge_the_last():
    """The only instruction whose failure costs the ENTIRE answer rather than
    degrading it, so it gets both ends and nothing competes for either."""
    for site, language in ((claude_site(), "rust"), (chatgpt_site(), "python")):
        for name, prompt in _all_stage_prompts(_stage_task(language)).items():
            # The contract is the LAST thing every stage says, which is the
            # other end from where it used to sit. A model reads an
            # instruction about how to answer at the moment it starts
            # answering, not two kilobytes before it knows the question.
            tail = prompt.rsplit("Reply with", 1)[-1] if "Reply with" in prompt \
                else prompt[-600:]
            # ONE block, in every stage but the one that asks for a probe
            # generator beside the inputs. The count is the load-bearing part:
            # `extract_code` picks the block that DEFINES the entrypoint, so a
            # second one is what could confuse it, and only the probe turn
            # wants two -- where it says so, by number.
            if name == "inputs+probe":
                assert "TWO fenced blocks" in tail, (name, language, tail[:200])
            else:
                assert "ONE fenced block" in tail, (name, language, tail[:200])
                assert "TWO fenced blocks" not in tail, (name, language)
        # ...and the site's nudge, appended after everything, repeats it.
        assert site.nudge.startswith(
            "START your reply with the fenced block"
        ), site.nudge[:70]


def test_a_repair_carries_the_error_and_no_method_for_thinking():
    """A repair round is the evidence plus one sentence naming what may come
    back. Nothing else.

    It used to carry a paragraph of method as well -- trace the failing call
    through your code, do not guess at the fix from the shape of the failure,
    re-check the fix against every OTHER case you were sent silently, do not
    change both to make them agree. That is work which never reaches the reply,
    competing with the failure itself for attention, and it is the same class of
    instruction the two-phase rewrite already took out of turns 1 and 2."""
    prompt = _repair_prompt("python", report="solve([]) raised IndexError")
    assert "solve([]) raised IndexError" in prompt, prompt
    for method in ("in your reasoning", "silently", "trace the failing",
                   "do not guess", "re-check", "same rules as before",
                   "do not change both", "every other case"):
        assert method not in prompt.lower(), (
            f"method is back in the repair prompt: {method!r}"
        )
    # ...and a DEFECT is answered with delivery, never with a run report.
    for defect in ("the program does not define fn main()", NO_CODE):
        repair = _repair_prompt("rust", defect=defect, found_by="unrun")
        assert "comparing what they produced" not in repair, (
            f"answered a delivery failure with evidence that does not exist: {defect!r}"
        )
        assert "NOTHING WAS RUN" in repair, defect


# --- a message that is all reasoning is not an answer --------------------- #
# claude.ai renders extended thinking INSIDE the element the assistant selector
# matches, and the thinking arrives long before any code does. ChatGPT keeps its
# reasoning outside the matched element, so the same situation there read as
# empty and was honestly reported as empty. That single DOM difference is why
# one site failed with "the reply contained no code" and the other with "the
# program does not define `fn main()`" — the same moment, two symptoms.


def test_a_message_that_is_all_reasoning_is_not_read_as_the_answer():
    """Measured on a real solve: 13,200 characters of the model working through
    the problem were submitted as Rust. The grader answered "the program does
    not define `fn main()`", and the repair round told the model to fix a
    program it had never sent. Twice, before the budget ran out."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    thinking = "Let me carefully work through this problem. " * 300

    def answers(_):
        # A finished turn, as far as any selector can tell: nothing is
        # streaming, nothing is busy. There is simply no code in it yet.
        page.dom["#assistant"] = [_Node(text=thinking)]

    page.on_click = answers
    got = asyncio.run(_tab(page, _site()).send("solve it", 1.5))
    assert got == "", f"submitted {len(got)} characters of reasoning as code: {got[:80]!r}"


def test_reasoning_is_reported_as_nothing_arrived_not_as_a_broken_program():
    """The two are different conversations, and giving the wrong one costs the
    round. "Your program has no fn main()" is a contradiction when no program
    was sent: the model rewrites logic that was never the problem. "Nothing
    reached me as code" is the one that gets a code block back."""
    from solvers.prompts import rust_defect

    prose = (
        "Let me carefully work through this problem.\n"
        "We have a stream of bytes described by a DAG of nodes.\n"
        "State during processing: the current pending record length."
    )
    defect = rust_defect(extract_code(prose, "main", "rust"))
    assert defect == NO_CODE, f"reasoning was diagnosed as {defect!r}"

    repair = _repair_prompt("rust", code="", defect=defect, found_by="unrun")
    assert NO_CODE in repair, repair[:200]
    assert "does not define" not in repair, (
        "still telling the model to fix a program it never sent"
    )


def test_a_program_sent_without_a_fence_is_still_an_answer():
    """The other half of the same rule. A model that ignores the formatting and
    types the program bare has still answered, and dropping that would trade one
    silent failure for another. Gradeability decides, not punctuation."""
    bare_rust = 'use std::io;\nfn main() { println!("1"); }'
    assert extract_code(bare_rust, "main", "rust") == bare_rust

    bare_py = "def pong():\n    return 'pong'"
    assert extract_code(bare_py, "pong") == bare_py

    # ...and something that merely looks like code but cannot be graded is not
    # rescued by this: it is still nothing arriving.
    assert extract_code("I would start by sorting xs, then return xs[0].", "pong") == ""


def test_a_reply_with_no_code_block_says_what_it_said_instead(capsys):
    """What is given up by never submitting prose is the chance to SEE it, and
    that is the thing that makes a silent failure take days. A message with no
    code and a selector matching an empty wrapper both arrive as `best == ""`
    and need opposite fixes, so the post-mortem quotes the message."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})

    def answers(_):
        page.dom["#assistant"] = [
            _Node(text="I need more detail about the framing rules before I can answer.")
        ]

    page.on_click = answers
    asyncio.run(_tab(page, _site()).send("solve it", 1.5))
    logged = capsys.readouterr().out
    assert "no code block in it" in logged, logged
    assert "I need more detail" in logged, f"did not quote the message: {logged!r}"


def test_claude_and_chatgpt_now_fail_the_same_way_on_a_thinking_message():
    """The bug was never in either model. It was that Claude's reasoning lands
    inside the matched element and ChatGPT's does not, so the identical moment
    produced two different diagnoses. Reading only code blocks makes the site's
    markup stop mattering."""
    thinking = "Working through the constraints. " * 50
    results = {}
    for name, dom in (
        # Claude: the thinking is inside the message.
        ("claude", [_Node(text=thinking)]),
        # ChatGPT: it is somewhere the selector cannot see.
        ("chatgpt", [_Node(text="")]),
    ):
        page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
        page.on_click = lambda _, d=dom, p=None: None
        page.dom["#assistant"] = dom
        results[name] = asyncio.run(_tab(page, _site()).send("solve it", 1.0))
    assert results["claude"] == results["chatgpt"] == "", results


# --- every solve leaves a file ------------------------------------------- #
# A browser-backed miner is hard to look at afterwards: the reply that produced
# a zero is gone the moment the tab starts its next conversation, and the
# validator keeps the only other copy of what was sent.


def _request(problem_id, language="rust"):
    from rlvr.protocol import TaskRequest

    return TaskRequest(
        problem_id=problem_id, language=language, statement="do a thing",
        entrypoint="main" if language == "rust" else "solve",
    )


def _solved_by(solver, request, directory):
    """Drive the REAL solve path, including its own failure handling.

    Called unbound against a stub rather than through a constructed miner:
    `solve` touches nothing but `self._solver`, and building a DemoMiner would
    drag in settings, a client and an axon to test six lines of file writing.
    """
    from custom_miner import CustomMiner

    previous = os.environ.get("SOLVER_SOLUTION_DIR")
    os.environ["SOLVER_SOLUTION_DIR"] = str(directory)
    try:
        return asyncio.run(CustomMiner.solve(SimpleNamespace(_solver=solver), request, 5.0))
    finally:
        if previous is None:
            os.environ.pop("SOLVER_SOLUTION_DIR", None)
        else:
            os.environ["SOLVER_SOLUTION_DIR"] = previous


def _solver_returning(code, transcript="transcript"):
    from custom_miner import SolveResult

    class _S:
        async def solve_task(self, task, timeout_s):
            return SolveResult(code=code, raw_response=transcript)

    return _S()


def test_the_answer_that_was_sent_is_the_answer_on_disk(tmp_path):
    """Written from the PAYLOAD, not from the variable that fed it. The file is
    only worth having if it is the submission rather than something that
    resembles it — anything that rewrites `code` on the way out would otherwise
    leave a copy that quietly disagrees with what was graded."""
    program = 'use std::io;\nfn main() { println!("1"); }\n'
    payload = _solved_by(_solver_returning(program), _request("prob-1"), tmp_path)
    written = tmp_path / "prob-1.rs"
    assert written.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert written.read_text() == payload.code == program


def test_the_language_picks_the_extension(tmp_path):
    _solved_by(_solver_returning("def solve():\n    return 1\n"),
               _request("py-task", "python"), tmp_path)
    _solved_by(_solver_returning("fn main() {}"), _request("rs-task", "rust"), tmp_path)
    assert (tmp_path / "py-task.py").exists()
    assert (tmp_path / "rs-task.rs").exists()


@pytest.mark.parametrize(
    "kind, code",
    [("empty", ""), ("blank", "   \n\n  ")],
)
def test_a_solve_that_produced_nothing_still_leaves_an_empty_file(kind, code, tmp_path):
    """The deliberate part. Absence would be ambiguous — never dispatched,
    crashed before the solver ran, or answered with silence — and those need
    different fixes. A zero-byte file says which one it was.

    Whitespace-only counts as nothing: a few blank lines on disk read as an
    answer at a glance and to anything measuring size."""
    _solved_by(_solver_returning(code), _request(f"{kind}-task"), tmp_path)
    written = tmp_path / f"{kind}-task.rs"
    assert written.exists(), f"a silent solve left no record at all ({kind})"
    assert written.stat().st_size == 0, (
        f"wrote {written.stat().st_size} bytes for an answer that was empty"
    )


def test_a_solver_that_raises_still_leaves_a_file(tmp_path):
    """The path most likely to be the one you need afterwards, and the one that
    never reaches the solver's own return statement."""

    class _Dies:
        async def solve_task(self, task, timeout_s):
            raise RuntimeError("the tab died")

    payload = _solved_by(_Dies(), _request("boom"), tmp_path)
    assert payload.code == "", "a crashed solve must submit nothing"
    written = tmp_path / "boom.rs"
    assert written.exists() and written.stat().st_size == 0


def test_a_problem_id_cannot_write_outside_the_archive(tmp_path):
    """`problem_id` arrives over the network and is used to build a path, so it
    is sanitised as hostile input rather than trusted as an identifier."""
    from solution_archive import save_solution

    for hostile in ("../../etc/passwd", "..\\..\\windows\\system32", "/abs/olute",
                    "..", ".", "", "  ", "._-"):
        written = save_solution(hostile, "python", "x = 1", tmp_path)
        assert written is not None, hostile
        assert tmp_path in written.parents, f"{hostile!r} escaped to {written}"
        assert written.parent == tmp_path, f"{hostile!r} nested to {written}"
    assert not (tmp_path / "etc").exists(), "created a directory from a path segment"


def test_a_very_long_problem_id_still_produces_a_usable_name(tmp_path):
    """Filesystems cap a single name at 255 bytes; an id is capped at 256."""
    from solution_archive import save_solution

    written = save_solution("z" * 256, "rust", "fn main(){}", tmp_path)
    assert written is not None and written.exists()
    assert len(written.name) < 255, len(written.name)


def test_archiving_can_be_switched_off(tmp_path):
    """It writes to disk on every solve, so there has to be a way to stop it."""
    from solution_archive import archive_dir, save_solution

    previous = os.environ.get("SOLVER_SOLUTION_DIR")
    os.environ["SOLVER_SOLUTION_DIR"] = ""
    try:
        assert archive_dir() is None
        assert save_solution("p", "python", "x = 1") is None
    finally:
        if previous is None:
            os.environ.pop("SOLVER_SOLUTION_DIR", None)
        else:
            os.environ["SOLVER_SOLUTION_DIR"] = previous


def test_a_disk_that_cannot_be_written_does_not_cost_the_solve(tmp_path, capsys):
    """A miner that dies because a disk filled up has turned a lost point into a
    lost session. The answer still goes out; the failure is explained once."""
    import solution_archive

    blocked = tmp_path / "wall"
    blocked.write_text("I am a file, not a directory")

    solution_archive._warned = False
    payload = _solved_by(_solver_returning("fn main() {}"), _request("p1"), blocked)
    assert payload.code == "fn main() {}", "a failed archive swallowed the answer"
    assert "could not write solutions" in capsys.readouterr().out

    # ...and it does not say so again on every subsequent solve.
    _solved_by(_solver_returning("fn main() {}"), _request("p2"), blocked)
    assert "could not write solutions" not in capsys.readouterr().out


# --- a tool call is not an answer ----------------------------------------- #
# When a model reaches for its tools, a chat UI paints every tool call as a
# `pre code` block — the same markup an answer gets. So "read only code blocks"
# is not enough on its own: the blocks have to be asked whether they are
# plausibly source in the target language before one becomes a submission.

TOOL_JSON = (
    '{"command": "mkdir -p /home/claude/sol && cat > /home/claude/sol/main.rs '
    '<< \'RUST_EOF\'\\nuse std::io;\\nfn main() {\\n    println!(\\"draft\\");\\n}'
    '\\nRUST_EOF\\necho written"}'
)
TOOL_SHELL = (
    "cat > /home/claude/sol/main.rs << 'RUST_EOF'\n"
    "use std::io;\n"
    "fn main() {\n"
    '    println!("draft");\n'
    "}\n"
    "RUST_EOF\n"
    "echo written"
)
REAL_RUST = 'use std::io;\nfn main() {\n    println!("42");\n}'


def _blocks(*bodies):
    return "\n".join(f"```\n{b}\n```" for b in bodies)


def test_a_tool_call_is_not_a_rust_program():
    """`"fn main" in code` was the whole test, and a tool call passes it: the
    program is quoted INSIDE a shell heredoc, inside JSON. Submitted, that is a
    compile error nobody could trace back to a tool call."""
    from solvers.prompts import rust_defect

    for name, block in (("JSON", TOOL_JSON), ("shell", TOOL_SHELL)):
        defect = rust_defect(block)
        assert defect is not None, f"a {name} tool call passed as a program"
        assert "does not look like a Rust program" in defect, defect
        assert "does not define" not in defect, (
            "called it a program with a missing main; it was never a program"
        )
    assert rust_defect(REAL_RUST) is None
    # ...and a real file that opens with an attribute rather than `use` or `fn`.
    assert rust_defect("#![allow(unused)]\nfn main() {}") is None


def test_a_program_that_only_mentions_fn_main_in_a_string_has_no_main():
    """The opener check catches a tool call; this catches the subtler one it
    cannot. A genuine Rust file that merely QUOTES `fn main` — in a string, a
    macro, a `write!` template — opens like Rust and passes every structural
    test except the one that asks where `fn main` actually is. Submitted, it is
    a link error, which costs a compile to discover instead of a search."""
    from solvers.prompts import rust_defect

    quoted = 'use std::io;\nfn helper() { let s = "fn main() {}"; }'
    assert rust_defect(quoted) == "the program does not define `fn main()`", rust_defect(quoted)

    for real in (
        "fn main() {}",
        "    fn main() {}",
        "pub fn main() {}",
        "use std::io;\nfn main () {\n}",
        "async fn main() {}",
    ):
        assert rust_defect(real) is None, f"rejected a real program: {real!r}"


def test_the_answer_wins_even_when_tool_calls_come_after_it():
    """The case that produced this. The model wrote the program, then went on
    running things — so the LAST code block in the message is a tool call, and
    "the last gradeable block" picked the one that merely mentioned `fn main`."""
    assert extract_code(_blocks(TOOL_JSON, REAL_RUST, TOOL_SHELL), "main", "rust") == REAL_RUST
    assert extract_code(_blocks(REAL_RUST, TOOL_JSON, TOOL_SHELL), "main", "rust") == REAL_RUST


def test_a_reply_of_nothing_but_tool_calls_submits_nothing():
    """Submitting one is a guaranteed zero AND archives a shell command as "the
    solution". Nothing arrived is both true and actionable."""
    from solvers.prompts import rust_defect

    got = extract_code(_blocks(TOOL_JSON, TOOL_SHELL), "main", "rust")
    assert got == "", f"submitted a tool call: {got[:60]!r}"
    assert rust_defect(got) == NO_CODE


def test_a_broken_program_is_still_an_attempt_and_is_kept():
    """The line this draws. A program with a fixable flaw — no entrypoint, a
    syntax error, a line the deadline cut in half — IS an attempt at an answer,
    and both the grader and the repair round need to see it. Only things that
    were never attempts get dropped."""
    missing_main = "use std::io;\nfn helper() -> i64 { 1 }"
    assert extract_code(_blocks(missing_main), "main", "rust") == missing_main

    truncated_py = "import sys\ndef solve(xs):\n    return sorted(xs"
    assert extract_code(_blocks(truncated_py), "solve") == truncated_py

    # Python opens with arbitrary statements, so a constant first line is fine.
    constant_first = "MOD = 10**9 + 7\ndef solve(xs):\n    return len(xs) % MOD"
    assert extract_code(_blocks(constant_first), "solve") == constant_first


def test_python_tool_calls_are_dropped_too():
    """Python has no closed top-level grammar, so it cannot use Rust's
    allowlist of openers — but the handful of things a tool call starts with
    can still be named."""
    from solvers.prompts import plausible_source

    assert not plausible_source("python3 /home/claude/sol/sim.py")
    assert not plausible_source("cat > sim.py << 'EOF'\ndef solve(): pass\nEOF")
    assert not plausible_source(TOOL_JSON)
    assert plausible_source("def solve(xs):\n    return xs")
    assert plausible_source("MOD = 10**9 + 7")


def test_both_backends_ask_the_model_not_to_reach_for_its_tools():
    """The root cause, and the only fix that costs nothing: the model used its
    tools because nothing said not to. There is no toolchain behind a chat UI —
    the session that produced this tried `apt-get install rustc` — so every tool
    call is time the answer does not get."""
    for site in (claude_site(), chatgpt_site()):
        nudge = site.nudge.lower()
        assert "compile" in nudge and "test anything" in nudge, site.nudge


def test_the_wire_does_not_mistake_tool_arguments_for_the_answer():
    """The stream had the same bug, from the other direction. A model asking a
    tool to do something streams the request as `partial_json`, and "the field
    appended to most" is then the tool call: a shell command with a draft
    program quoted inside it. Measured before the fix — a 5,442 byte tool call
    beat the 54 byte answer beside it purely on volume."""
    playwright, chrome = _chromium_or_skip()
    answer = 'Here it is:\n\n```rust\nfn main() { println!("42"); }\n```'
    tool = json.dumps({"command": "cat > main.rs << 'EOF'\n"
                                  + ("fn main() { /* draft */ }\n" * 200) + "EOF"})
    events = [
        'data: {"type":"message_start","message":{"id":"m","role":"assistant"}}',
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"t1","name":"bash","input":{}}}',
    ]
    for i in range(0, len(tool), 30):
        events.append("data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": tool[i:i + 30]}}))
    events.append('data: {"type":"content_block_stop","index":0}')
    for i in range(0, len(answer), 8):
        events.append("data: " + json.dumps({
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "text_delta", "text": answer[i:i + 8]}}))
    body = "\n\n".join(events) + "\n\n"

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(playwright, chrome, [body])(p)
            await page.evaluate("async () => { await (await fetch('/sse')).text(); }")
            await asyncio.sleep(0.2)
            out = await page.evaluate(_STREAM_READ, 0)
            await browser.close()
            return out

    got = asyncio.run(go())
    assert got == answer, f"the wire took {len(tool)} bytes of tool call: {(got or '')[:80]!r}"


# --- the miner must never submit its own prompt --------------------------- #
# It did. Twice, to real validators, archived as Rust programs ending in the
# words "Do not use canvas". `_echoes_prompt` was written to stop exactly this
# and was applied in exactly one place — inside `_poll`, guarding the scrape.
# The copy control and the network stream both went around it.

OUR_PROMPT = (
    "Solve this programming problem in Rust.\n\nRules — the grader is automated "
    "and unforgiving:\n- Write ONE complete program with `fn main()`.\n\n"
    "PROBLEM:\nDo a thing.\n\nReply directly in the chat with one ordinary "
    "fenced code block. Do not use canvas."
)


def _text_stream(text):
    """A Claude-shaped SSE body carrying `text` as the assistant's message."""
    events = ['data: {"type":"message_start","message":{"id":"m","role":"assistant"}}']
    for i in range(0, len(text), 20):
        events.append("data: " + json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[i:i + 20]}}))
    return "\n\n".join(events) + "\n\n"


# A page that renders nothing readable but does stream. The wire is then the
# only source with anything in it, which is the situation the rescue exists for.
SILENT_STREAMING_PAGE = (
    '<!doctype html><meta charset="utf-8">'
    '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    '<div id="host"></div><script>'
    "document.getElementById('send').onclick = () => {"
    "  fetch('/sse').then(r => r.text()).then(() => {"
    "    const d = document.createElement('div');"
    "    d.setAttribute('data-message-author-role', 'assistant');"
    "    document.getElementById('host').appendChild(d); }); };</script>"
)


def _send_against_stream(body):
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(
                playwright, chrome, [body], page_html=SILENT_STREAMING_PAGE)(p)
            reply = await _tab(page, _wire_site()).send(OUR_PROMPT, 6.0)
            await browser.close()
            return reply

    return asyncio.run(go())


def test_the_wire_never_hands_back_the_miners_own_prompt(capsys):
    """The production failure, end to end.

    A chat stream carries the CONVERSATION, not just the reply, so the prompt
    that was just sent can be the largest block of text in it. With the page
    unreadable and the wire holding no fenced code, the rescue branch returned
    that raw text — and a validator received this file's own instructions as a
    Rust program. Twice.
    """
    reply = _send_against_stream(_text_stream(OUR_PROMPT))
    assert reply == "", f"submitted the miner's own prompt: {reply[:80]!r}"
    logged = capsys.readouterr().out
    assert "no code block in it either" in logged, logged


def test_the_wire_does_not_claim_a_rescue_it_did_not_make(capsys):
    """The old message said "recovered no code block(s) ... The answer below
    came off the wire" — announcing a rescue in the same breath as admitting
    there was nothing to rescue. Worse, returning that text made `best`
    non-empty, which SILENCED the post-mortem that would have said what the
    page actually contained."""
    _send_against_stream(_text_stream("I need more detail before I can answer."))
    logged = capsys.readouterr().out
    assert "The answer below came off the wire" not in logged, (
        f"still claiming a rescue with nothing recovered: {logged!r}"
    )
    assert "captured NOTHING from this reply" in logged, (
        f"the post-mortem was suppressed: {logged!r}"
    )


def test_a_real_answer_on_the_wire_is_still_rescued(capsys):
    """The guard must not cost the thing the wire is there for."""
    answer = 'Here it is:\n\n```rust\nfn main() { println!("42"); }\n```'
    reply = _send_against_stream(_text_stream(answer))
    assert "fn main" in reply, f"lost a genuine wire answer: {reply!r}"
    assert "came off the wire" in capsys.readouterr().out


def test_the_copy_control_cannot_smuggle_the_prompt_past_the_guard(capsys):
    """The other unguarded route. `_copied_blocks` presses a control and takes
    what it is handed, with no echo check anywhere on that path — so a selector
    that has drifted onto the user's own turn submits the prompt from a source
    the scrape guard never sees."""
    playwright, chrome = _chromium_or_skip()

    def page_for(src):
        return (
            '<!doctype html><meta charset="utf-8">'
            '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
            '<div id="host"></div><script>const SRC = ' + json.dumps(src) + ';'
            "document.getElementById('send').onclick = () => {"
            "  const w=document.createElement('div');"
            "  w.setAttribute('data-message-author-role','assistant');"
            "  const pre=document.createElement('pre'), c=document.createElement('code');"
            "  c.textContent=SRC; pre.appendChild(c); w.appendChild(pre);"
            "  const b=document.createElement('button'); b.setAttribute('aria-label','Copy');"
            "  b.onclick=()=>navigator.clipboard.writeText(SRC); w.appendChild(b);"
            "  document.getElementById('host').appendChild(w); };</script>"
        )

    async def go(src):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.route("**/x", lambda r: asyncio.ensure_future(r.fulfill(
                status=200, content_type="text/html", body=page_for(src))))
            await page.goto("https://example.test/x")
            site = _wire_site(copy=('button[aria-label="Copy"]',), stream=False)
            reply = await _tab(page, site).send(OUR_PROMPT, 6.0)
            await browser.close()
            return reply

    assert asyncio.run(go(OUR_PROMPT)) == "", "the copy control smuggled the prompt through"
    assert "OWN PROMPT" in capsys.readouterr().out

    # ...and a real program handed over by the same control still gets through.
    real = 'use std::io;\nfn main() { println!("42"); }'
    assert "fn main" in asyncio.run(go(real)), "the guard ate a genuine answer"


def test_the_suite_never_archives_into_the_operators_corpus(tmp_path):
    """Measured, not imagined: two tests here drive the real `/solve` path
    through a TestClient, and `CustomMiner.solve` archives everything it
    produces — so running the suite wrote its own fixtures into `solutions/`
    beside answers a live miner had produced for real validators. Deleting the
    two files and re-running just those tests put them straight back."""
    from solution_archive import DEFAULT_DIR, archive_dir

    where = archive_dir()
    assert where is not None
    assert where != Path(DEFAULT_DIR), (
        "a test is pointed at the real solutions directory; the autouse fixture "
        "is not in force"
    )
    assert "pytest" in str(where) or str(tmp_path.parent) in str(where), where


# --- Rust gets a compiler, not a grep ------------------------------------- #
# Python's structural check PARSES the source. Rust's greps it for `fn main`.
# That asymmetry is why every answer this miner has destroyed in transit was a
# Rust one: the model's reasoning, a tool call, and the miner's own prompt all
# contain the characters `fn main`, and all three were submitted as programs.


def _rustc_or_skip():
    from solvers.rust_compile import rustc_path

    if rustc_path() is None:
        pytest.skip("no rustc on this host")


BUILDS = 'use std::io::{self, Read};\nfn main() {\n    let mut s = String::new();\n    io::stdin().read_to_string(&mut s).unwrap();\n    println!("{}", s.trim().len());\n}\n'


@pytest.mark.parametrize(
    "name, code",
    [
        ("prompt echo", OUR_PROMPT),
        ("tool call", '{"command": "cat > main.rs << \'EOF\'\\nfn main() {}\\nEOF"}'),
        ("truncated mid-token", "use std::io;\nfn main() {\n    let x = 1;\n    le"),
        ("undefined function", "fn main() {\n    let _ = missing_helper(1);\n}"),
        ("borrow error", "fn main() {\n    let mut v = vec![1];\n    let r = &v[0];\n    v.push(2);\n    println!(\"{}\", r);\n}"),
    ],
)
def test_the_rust_gate_rejects_what_will_not_build(name, code):
    """Replayed against the real archive, this rejected exactly the six Rust
    submissions that did not build and passed all twelve that did. Three of the
    six would otherwise have reached a validator as programs; the other three
    become a repair round instead of a certain zero."""
    _rustc_or_skip()
    from solvers.rust_compile import compile_defect

    defect = compile_defect(code)
    assert defect is not None, f"a {name} was accepted as a Rust program"
    assert defect.startswith("it does not compile:"), defect


def test_the_rust_gate_passes_a_program_that_builds():
    """The expensive half of being wrong. A gate that rejects real answers is
    worse than no gate: this one has to be silent on anything that compiles."""
    _rustc_or_skip()
    from solvers.rust_compile import compile_defect

    assert compile_defect(BUILDS) is None


def test_the_first_error_is_reported_not_the_first_warning():
    """The exact shape that makes this necessary, found by looking rather than
    guessing: rustc reports most errors before warnings, but a MISSING `main`
    comes from a very late pass, so an unused import is printed first.

        warning: unused import: `std::collections::HashMap`
        ...
        error[E0601]: `main` function not found in crate `candidate`

    That is the worst possible case to get wrong. "No main function" is the one
    defect this miner most needs to report, and handing back the head of stderr
    instead sends the repair round off to delete an import while the program
    still has no entry point.
    """
    _rustc_or_skip()
    from solvers.rust_compile import compile_defect

    warning_first = "use std::collections::HashMap;\nfn helper() {}\n"
    defect = compile_defect(warning_first)
    assert defect is not None, "a program with no `main` was accepted"
    assert "main" in defect and "E0601" in defect, (
        f"reported the warning instead of the error: {defect}"
    )
    assert "unused import" not in defect, defect


def test_no_local_toolchain_means_no_opinion(monkeypatch):
    """Silence must mean the same thing as success. A miner that stops
    submitting answers because a toolchain went missing has turned a missing
    convenience into an outage."""
    import solvers.rust_compile as rc

    monkeypatch.setattr(rc, "_looked", False)
    monkeypatch.setattr(rc, "_rustc", None)
    monkeypatch.setattr(rc.shutil, "which", lambda _: None)
    assert rc.rustc_path() is None
    assert rc.compile_defect("this is not rust at all") is None

    # ...and the operator can switch it off even where a compiler exists.
    monkeypatch.setattr(rc, "_looked", False)
    monkeypatch.setenv("SOLVER_RUST_COMPILE", "0")
    assert rc.rustc_path() is None


def test_the_gate_asks_the_same_question_the_validator_will():
    """Flags read from RELEASE_POLICY rather than copied, so a change to the
    validator's toolchain cannot leave this quietly asking something else."""
    import inspect

    from rlvr.policy import RELEASE_POLICY
    from solvers import rust_compile

    source = inspect.getsource(rust_compile.compile_defect)
    assert "RELEASE_POLICY.rustc_flags" in source
    assert "RELEASE_POLICY.rust_edition" in source
    assert RELEASE_POLICY.rustc_flags == ("-C", "opt-level=2"), RELEASE_POLICY.rustc_flags


def test_a_compile_failure_becomes_a_defect_the_repair_round_can_use():
    """End to end: it has to reach `_grade` and come out as a DEFECT, because
    that is what turns a dead solve into another round. With no public examples
    — every task on the run this was written for — a defect is the only thing
    that can make the loop ask again at all."""
    _rustc_or_skip()

    solver = _solver([])
    task = SimpleNamespace(
        language="rust", entrypoint="main", statement="do a thing", public_examples=[],
    )
    candidate = solver._grade("```rust\nfn main() { nope(); }\n```", task)

    assert candidate.defect is not None, "a program that cannot build was taken as-is"
    assert "does not compile" in candidate.defect
    assert candidate.code.strip(), "threw the answer away instead of repairing it"

    repair = _repair_prompt(
        "rust", code=candidate.code, defect=candidate.defect, found_by="unrun",
    )
    assert "NOTHING WAS RUN" in repair
    assert "cannot find function" in repair, repair[:200]


# --- a Python answer can be cut off and still parse ----------------------- #
# `ast.parse` is Python's version of grepping for `fn main`: perfectly happy
# with source that was truncated, because a reply cut at a statement boundary
# is still a valid module. Two archived answers ended deep inside a loop with
# no return after them. Both parsed. Both were submitted. Both answered None on
# every hidden test, and nothing anywhere noticed.


def test_a_truncated_python_answer_is_caught_even_though_it_parses():
    """The shape that got through, taken from the archive: the function ends
    on a `while` sixteen columns deep, with no return after it."""
    cut_off = (
        "def plan_workflow(jobs):\n"
        "    ready = []\n"
        "    cand = set(jobs)\n"
        "    while cand:\n"
        "        allowed = cand & set(ready)\n"
        "        if not allowed:\n"
        "            break"
    )
    import ast

    ast.parse(cut_off)  # it really does parse — that is the whole problem
    defect = python_defect(cut_off, "plan_workflow")
    assert defect is not None, "a truncated answer was taken as finished"
    assert "without returning" in defect and "while" in defect, defect


def test_a_function_that_ends_on_a_return_is_left_alone():
    """Replayed against the archive this flagged 2 of 25 and passed the other
    23. A check that fires on real answers is worse than no check."""
    for good in (
        "def solve(xs):\n    return sorted(xs)",
        "def solve(xs):\n    if not xs:\n        return []\n    return sorted(xs)",
        "def solve(xs):\n    try:\n        return xs[0]\n    except IndexError:\n        return None",
        "def solve(xs):\n    with open('/dev/null') as f:\n        return len(xs)",
        "def solve(xs):\n    raise ValueError('no')",
        "def solve(xs):\n    while True:\n        return xs",
    ):
        assert python_defect(good, "solve") is None, f"rejected a finished function:\n{good}"


def test_falling_off_the_end_is_a_defect_even_when_the_model_meant_it():
    """Not only a truncation detector. The grader compares RETURN VALUES, so a
    function that runs off its own end answers None — which is wrong for almost
    every task. Rust gets this from its compiler for free; Python did not get
    it at all."""
    meant_it = (
        "def solve(xs):\n"
        "    for x in xs:\n"
        "        if x > 0:\n"
        "            return x\n"
    )
    defect = python_defect(meant_it, "solve")
    assert defect is not None, "an empty list would answer None and nothing said so"
    assert "answers None" in defect, defect

    # An `if` with no `else` is the same hole and the commonest shape of it —
    # it is precisely the n = 0 case the prompt spends six lines asking about.
    no_else = "def solve(xs):\n    if xs:\n        return max(xs)\n"
    assert python_defect(no_else, "solve") is not None, (
        "an unguarded `if` at the end answers None for the empty input"
    )
    # ...but an if/else where BOTH branches return is finished.
    both = "def solve(xs):\n    if xs:\n        return max(xs)\n    else:\n        return 0\n"
    assert python_defect(both, "solve") is None


def test_the_conservative_direction_is_the_cheap_one():
    """A `for` loop that obviously always returns is still flagged, and that is
    deliberate: being wrong here costs one repair round, while missing a
    truncated answer costs the whole solve. The answer is never thrown away —
    a defective candidate still outranks an empty one."""
    from solvers.verify import Candidate

    obvious = "def solve(xs):\n    for x in [1]:\n        return x\n"
    assert python_defect(obvious, "solve") is not None

    flawed = Candidate(code=obvious, raw="", defect="falls off the end")
    assert flawed.score > Candidate(code="", raw="").score, (
        "a wrongly-flagged answer must still beat submitting nothing"
    )


# --- the stream carries the conversation, not just the reply -------------- #
# Attributed by provider from the archive itself: the two prompt echoes end in
# CHATGPT_NUDGE ("Do not use canvas"), and the tool call quotes `/home/claude/sol`
# — Claude's analysis sandbox. Two different sites, two different mechanisms,
# and the miner had been treating them as one.

CHATGPT_USER_TURN = (
    "Solve this programming problem in Rust.\n\nRules — the grader is automated\n"
    + "- some rule about edge cases\n" * 40
    + "\nDo not use canvas."
)


def _chatgpt_conversation(answer, thoughts=""):
    """ChatGPT's real shape: a snapshot of the CONVERSATION, then deltas.

    The snapshot holds the user's own turn under `author.role = "user"`, which
    is the whole problem — it is usually far longer than the answer beside it.
    """
    events = []

    def send(obj):
        events.append("data: " + json.dumps(obj))

    send({"v": {"message": {
        "id": "u1", "author": {"role": "user"},
        "content": {"content_type": "text", "parts": [CHATGPT_USER_TURN]},
        "status": "finished"}, "conversation_id": "abc-123", "c": 0}})
    send({"v": {"message": {
        "id": "a1", "author": {"role": "assistant"},
        "content": {"content_type": "text", "parts": [""]},
        "status": "in_progress"}, "c": 1}})
    for body, path in ((thoughts, "/message/content/thoughts/0/content"),
                       (answer, "/message/content/parts/0")):
        first = True
        for i in range(0, len(body), 6):
            if first:
                send({"p": path, "o": "append", "v": body[i:i + 6]})
                first = False
            else:
                send({"v": body[i:i + 6]})
    events.append("data: [DONE]")
    return "\n\n".join(events) + "\n\n"


def _reconstruct(body):
    playwright, chrome = _chromium_or_skip()

    async def go():
        async with playwright.async_playwright() as p:
            browser, page = await _streaming_page(playwright, chrome, [body])(p)
            await page.evaluate("async () => { await (await fetch('/sse')).text(); }")
            await asyncio.sleep(0.2)
            out = await page.evaluate(_STREAM_READ, 0)
            await browser.close()
            return out or ""

    return asyncio.run(go())


def test_the_wire_never_takes_text_the_stream_says_the_user_wrote():
    """The mechanism behind two real submissions, reproduced on ChatGPT's own
    payload shape: a 1,384-character user turn beat the 41-character answer
    beside it purely on volume, and the miner's instructions reached a validator
    as a Rust program. Who said it decides, not how much of it there is."""
    answer = '```rust\nfn main() { println!("42"); }\n```'
    assert _reconstruct(_chatgpt_conversation(answer)) == answer


def test_a_reply_with_no_answer_in_it_reconstructs_as_nothing():
    """The production case: the model had not written anything yet, so the only
    long text in the stream was the prompt. Nothing is the honest answer."""
    assert _reconstruct(_chatgpt_conversation("")) == ""


def test_bookkeeping_keys_are_tags_whatever_they_are_prefixed_with():
    """`content_type` is as much a tag as `type`, and `conversation_id` as much
    as `id`. Matching only the bare word left a reply with no text in it
    reconstructing as the single word "text" — the value of `content_type`."""
    assert "text" != _reconstruct(_chatgpt_conversation("")), (
        "a bookkeeping value was taken as the answer"
    )


def test_reasoning_and_the_user_turn_lose_to_a_short_answer_together():
    """All three kinds of text a chat stream carries, in one reply, with the
    real answer the smallest of them."""
    answer = '```rust\nfn main() { println!("42"); }\n```'
    got = _reconstruct(_chatgpt_conversation(answer, thoughts="Let me reason. " * 200))
    assert got == answer, f"took reasoning or the prompt over the answer: {got[:70]!r}"


def test_the_log_names_which_model_produced_the_answer(capsys):
    """Attribution after the fact was guesswork. Of 43 archived submissions
    only three could be traced to a provider at all, and only because the
    DAMAGE carried a fingerprint — two held ChatGPT's nudge, one quoted
    `/home/claude/sol`. The other forty were unattributable, which made "is one
    of these tabs doing worse than the others" unanswerable.

    It has to be the model that WON, not merely the ones asked: a second
    opinion is bought precisely when the first answer was poor, so "who was
    asked" and "whose answer went out" are different questions.
    """
    # Two providers, and the FIRST one wins: its answer passes the examples, so
    # no second opinion is bought. Crediting "whoever was asked last" would
    # coincide with the truth here only by accident, which is why the second
    # case below asks two and still expects the first to be named.
    class _TwoModels:
        # One entry per PASS. A pass opens one conversation per phase, so
        # indexing by the number of opens walked the script four times too
        # fast; `avoid` changes exactly once per pass.
        _unset = object()

        def __init__(self, script):
            self._script, self.seen = script, []
            self._avoid, self._i, self._chat = self._unset, 0, None

        async def open(self, avoid=None):
            if avoid != self._avoid:
                self._avoid = avoid
                name, replies = self._script[min(self._i, len(self._script) - 1)]
                self._i += 1
                self.seen.append(name)
                self._chat = (name, _Script(replies))
            name, script = self._chat
            return _Chat(script, name)

        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement=DIGITS.statement,
        entrypoint="g", deadline_s=60.0,
        public_examples=[{"args": [12345], "kwargs": {}, "expected": 15}],
    )

    # chatgpt answers correctly and is never followed up.
    backend = _TwoModels([("chatgpt", [RIGHT])])
    asyncio.run(VerifyingSolver(backend, reserve_s=0, max_budget_s=120)
                .solve_task(task, 60.0))
    logged = capsys.readouterr().out
    assert "provider=chatgpt" in logged, f"the log cannot say who answered: {logged!r}"

    # ...and when the first model fails and the second is asked but does WORSE,
    # the credit must stay with the answer that actually went out.
    # Three entries, not two: a conversation that repeats itself now carries
    # its repair to the OTHER model inside the same pass, so `claude` is reached
    # once there and once again on the second-opinion pass.
    backend = _TwoModels([("chatgpt", [WRONG, WRONG, WRONG]),
                          ("claude", ["no code here"]),
                          ("claude", ["no code here"])])
    asyncio.run(VerifyingSolver(backend, reserve_s=0, max_budget_s=120)
                .solve_task(task, 60.0))
    logged = capsys.readouterr().out
    assert backend.seen[0] == "chatgpt" and "claude" in backend.seen, backend.seen
    assert "provider=chatgpt" in logged, (
        f"credited the last model ASKED rather than the one whose answer was "
        f"submitted: {logged!r}"
    )


# --- prose before the code costs time, not correctness -------------------- #
# Reported from a live Claude tab: long explanations arriving before the
# program. The extractor was never the problem — it handles a preamble fine.
# The clock is: the first attempt had 135 seconds of a 225 second budget, and
# prose spent before the code is time the code does not get.


def test_a_preamble_before_the_code_is_extracted_correctly():
    """Worth pinning so the fix is aimed at the right thing. A model that
    explains itself first has still answered, and nothing downstream should
    care — including when it appends an example block afterwards, which is the
    shape that WOULD break a reader that took the last block blindly."""
    reply = (
        "I'll solve this step by step. The values reach 10^18 so i64 is needed\n"
        "throughout, and n = 0 must answer 0 rather than divide by a length.\n\n"
        "Here is the complete program:\n\n"
        "```rust\nuse std::io::{self, Read};\nfn main() {\n"
        '    let mut s = String::new();\n'
        "    io::stdin().read_to_string(&mut s).unwrap();\n"
        '    println!("{}", s.trim().len());\n}\n```\n\n'
        "Example run:\n\n```\n3\n1 2 3\n```\n"
    )
    code = extract_code(reply, "main", "rust")
    assert code.startswith("use std::io"), f"a preamble broke extraction: {code[:60]!r}"
    from solvers.prompts import rust_defect

    assert rust_defect(code) is None


def test_both_nudges_use_the_last_word_to_demand_code_first():
    """The nudge is appended after everything else, so it is the last thing the
    model reads before it starts generating. That slot is worth the strongest
    version of the one instruction that decides whether the answer arrives."""
    for site in (claude_site(), chatgpt_site()):
        assert site.nudge.startswith(
            "START your reply with the fenced block"
        ), site.nudge[:70]
        # The nudge holds the recency slot AND is appended to EVERY send, so
        # any count named here overrides the contract of whichever turn it
        # happens to ride on. Both solve turns ask for one block; a repair
        # round may ask for a corrected `json` block beside the program. So it
        # names no count at all and defers to the message above it -- pinning
        # "ONE" here told a repair round to send the program alone, and a model
        # obeying that can never correct a case that was wrong.
        assert "the message above asks for" in site.nudge, site.nudge
        for pinned in ("ONE ordinary fenced block", "two ordinary fenced blocks",
                       "program first", "JSON cases second"):
            assert pinned not in site.nudge, f"{site.name}: nudge pins {pinned!r}"
        # And it does not hurry the model. Correctness is the whole payment;
        # "an answer that arrives after a paragraph of prose may not arrive at
        # all" traded the thing being paid for against a thing that is not.
        for rush in ("may not arrive at all", "time the answer does not get",
                     "deadline", "quickly"):
            assert rush not in site.nudge, f"{site.name}: nudge still rushes: {rush!r}"


def test_the_examples_decide_when_the_statement_is_ambiguous():
    """The examples are the only disambiguation a solver is given — the README
    says so and nothing in the prompt used to. Without the rule the model has
    to guess which of its readings the author meant."""
    for language in ("rust", "python"):
        # Normalised, because the prompt is hard-wrapped: the phrase under test
        # spans a line break and an indent, and asserting on the raw text would
        # fail on formatting rather than on meaning.
        prompt = " ".join(
            _candidate_prompt(
                language, [{"args": [1], "kwargs": {}, "expected": 1}]
            ).split()
        ).lower()
        # It lives on the worked-examples label, which is the one place it can
        # be read at the moment it applies -- and the only place it survives
        # deleting the procedure that used to carry it.
        assert "where the statement is ambiguous they decide" in prompt, (
            "no disambiguation rule"
        )
        assert "already known to be right" in prompt


def test_both_contracts_say_there_is_no_partial_credit():
    """It changes the risk calculus. A model that thinks a near-miss scores
    something will reach for the clever implementation; one that knows a single
    wrong hidden case scores zero will not."""
    for language in ("rust", "python"):
        prompt = _candidate_prompt(language)
        assert "no partial credit" in prompt
        assert "Correctness is the whole of it" in prompt
        # The stake, not the tariff. What follows "no partial credit" used to be
        # the payment curve -- 95% for the slowest correct answer -- which tells
        # a model that speed is worth something. It is worth at most 5%, and
        # there is no per-turn deadline to spend it against.
        for tariff in ("95%", "fastest", "slowest correct"):
            assert tariff not in prompt, f"the payment curve is back: {tariff!r}"


def test_each_language_is_warned_that_hash_order_is_not_stable():
    """Measured rather than assumed, and it is the kind that hides: four runs of
    `list({'alpha','beta','gamma'})` gave four different orders because
    PYTHONHASHSEED is random per process, while a set of small ints gave the
    same order every time. A solution tested with integers looks stable and is
    not. Rust randomises HashMap/HashSet iteration for the same reason."""
    from solvers.prompts import PYTHON_ENVIRONMENT, RUST_ENVIRONMENT

    assert "PYTHONHASHSEED" in PYTHON_ENVIRONMENT
    assert "Sort before returning" in PYTHON_ENVIRONMENT
    assert "HashMap" in RUST_ENVIRONMENT and "BTreeMap" in RUST_ENVIRONMENT

    # ...and the claim itself is true of this interpreter.
    import subprocess
    import sys

    orders = {
        subprocess.run([sys.executable, "-c",
                        "print(list({'alpha','beta','gamma','delta','epsilon'}))"],
                       capture_output=True, text=True).stdout
        for _ in range(8)
    }
    assert len(orders) > 1, (
        "string set order was stable across 8 processes; the prompt's claim "
        "about PYTHONHASHSEED no longer holds on this interpreter"
    )


def _archived(pid, solver, tmp_path, language="rust"):
    from rlvr.protocol import TaskRequest
    from rlvr.types import TestCase

    request = TaskRequest(
        problem_id=pid, language=language,
        statement="Read N then N integers and print their sum.",
        entrypoint="main" if language == "rust" else "solve",
        public_examples=[TestCase(args=["3\n1 2 3"], kwargs={}, expected="6")],
        deadline_s=240.0,
    )
    payload = _solved_by(solver, request, tmp_path)
    stem = tmp_path / pid
    return payload, stem


def test_the_request_and_the_reply_are_archived_beside_the_code(tmp_path):
    """Same stem, different extension: the pair is obvious in a listing and
    trivial to join, and the code file stays a program that something can
    compile, diff or grade without stripping a header off it first."""
    program = 'fn main(){ println!("6"); }'
    transcript = f"Here is the program:\n\n```rust\n{program}\n```"

    class _S:
        async def solve_task(self, task, timeout_s):
            from custom_miner import SolveResult

            return SolveResult(code=program, raw_response=transcript)

    payload, stem = _archived("pair", _S(), tmp_path)

    assert stem.with_suffix(".rs").read_text() == program, "the code file is not pure code"
    record = json.loads(stem.with_suffix(".json").read_text())
    assert record["problem_id"] == "pair"
    # the question...
    assert record["request"]["statement"].startswith("Read N then N integers")
    assert record["request"]["entrypoint"] == "main"
    assert record["request"]["public_examples"][0]["expected"] == "6"
    assert record["request"]["deadline_s"] == 240.0
    # ...and the answer, including what the model actually said.
    assert record["response"]["code"] == payload.code == program
    assert record["response"]["raw_response"] == transcript


def test_a_solve_that_crashed_still_records_what_it_was_asked(tmp_path):
    """The path most worth having afterwards, and the one that never reaches
    the solver's own return. An empty `.rs` says a problem was seen and
    answered with silence; only the record says WHICH problem, and why."""

    class _Dies:
        async def solve_task(self, task, timeout_s):
            raise RuntimeError("the tab died")

    payload, stem = _archived("boom", _Dies(), tmp_path)

    assert payload.code == ""
    assert stem.with_suffix(".rs").stat().st_size == 0
    record = json.loads(stem.with_suffix(".json").read_text())
    assert record["request"]["statement"], "the question was lost with the answer"
    assert record["response"]["code"] == ""
    assert record["response"]["raw_response"] == "<solver failed>"


def test_the_two_expensive_bugs_would_have_been_one_glance(tmp_path):
    """Why `raw_response` is in the record rather than summarised out of it.

    A tool call submitted as Rust and a prompt submitted as Rust were both
    invisible in the code file — each looked like a finished program. Both are
    unmistakable in the transcript beside it.
    """
    tool_call = '{"command": "cat > /home/claude/sol/main.rs << \'EOF\'\\nfn main(){}\\nEOF"}'

    class _ToolCall:
        async def solve_task(self, task, timeout_s):
            from custom_miner import SolveResult

            return SolveResult(code=tool_call, raw_response=f"```\n{tool_call}\n```")

    _, stem = _archived("tool", _ToolCall(), tmp_path)
    record = json.loads(stem.with_suffix(".json").read_text())
    assert "/home/claude/sol" in record["response"]["raw_response"], (
        "the transcript that identifies the provider and the failure was dropped"
    )


def test_archiving_off_writes_neither_file(tmp_path):
    from solution_archive import save_exchange, save_solution

    previous = os.environ.get("SOLVER_SOLUTION_DIR")
    os.environ["SOLVER_SOLUTION_DIR"] = ""
    try:
        assert save_solution("p", "rust", "fn main(){}") is None
        assert save_exchange("p", {"a": 1}, {"b": 2}) is None
    finally:
        if previous is None:
            os.environ.pop("SOLVER_SOLUTION_DIR", None)
        else:
            os.environ["SOLVER_SOLUTION_DIR"] = previous


def test_a_record_that_will_not_serialise_does_not_cost_the_solve(tmp_path):
    """Same rule as everywhere else in this file: the archive is a convenience
    and the answer is the product. Neither an unserialisable field nor an
    unwritable disk may take a solve with it."""
    from solution_archive import save_exchange

    class _Opaque:
        def __repr__(self):
            return "<opaque>"

    # `default=str` catches most of it; a repr that itself raises is the case
    # that must still not propagate.
    class _Hostile:
        def __repr__(self):
            raise ValueError("no")

    assert save_exchange("p", {"weird": _Opaque()}, {}, tmp_path) is not None
    assert save_exchange("q", {"weird": _Hostile()}, {}, tmp_path) is None

    blocked = tmp_path / "wall"
    blocked.write_text("I am a file, not a directory")
    assert save_exchange("r", {"a": 1}, {"b": 2}, blocked) is None


def test_a_hostile_problem_id_cannot_place_the_record_outside_the_archive(tmp_path):
    """Same sanitisation as the code file — `problem_id` still arrives over the
    network and is still being used to build a path."""
    from solution_archive import save_exchange

    for hostile in ("../../etc/passwd", "..\\..\\windows", "/abs/olute", "..", ""):
        written = save_exchange(hostile, {"a": 1}, {"b": 2}, tmp_path)
        assert written is not None and written.parent == tmp_path, (
            f"{hostile!r} escaped to {written}"
        )


# --- nothing may outlive the deadline it was given ----------------------- #
# Playwright auto-waits 30 SECONDS on a locator unless told otherwise, and
# `set_default_timeout` is never called anywhere in this miner. Measured here
# against a node that was resolved and then removed -- the ordinary shape of a
# chat page still settling after an answer:
#
#     button.inner_text()   raised after 30.0s
#     node.inner_text()     raised after 30.0s
#     code.text_content()   raised after 30.0s
#
# `_submit` has been bounded against that since it was written. The rest of
# `send` was not, and the tail is the dangerous half: it runs AFTER the read
# loop has hit the deadline, so every second it spends is a second the solve
# has already promised away. `handle_request` wraps the whole solve in an
# `asyncio.wait_for` and answers 504 -- nothing at all -- rather than late, so
# an unbounded tail does not deliver the answer slowly. It destroys it.


def _answered_page(code="def pong():\n    return 'pong'"):
    """A page that renders one finished answer as soon as send is clicked."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__("#assistant", [_Node(code=[code])])
    return page


def _forever(*_a, **_kw):
    async def hang():
        await asyncio.sleep(3600)

    return hang()


def test_a_wedged_copy_control_cannot_spend_the_answer_it_was_checking(monkeypatch):
    """The copy phase runs past the deadline to improve an answer already in
    hand. Unbounded it can wait 30s per button on a page that is mid-rerender,
    and the answer it was polishing is thrown away by the deadline above it."""
    from solvers import browser_pool

    monkeypatch.setattr(_Tab, "_copy_phase", _forever)
    page = _answered_page()
    tab = _tab(page, _site(copy=("#copy",)))
    started = time.monotonic()
    reply = asyncio.run(tab.send("solve it", 1.0))
    spent = time.monotonic() - started
    assert spent < browser_pool.COPY_PHASE_TIMEOUT_S + 4.0, f"tail ran {spent:.1f}s"
    assert "return 'pong'" in reply, f"scraped answer lost to the copy phase: {reply!r}"


def test_a_wedged_stream_check_cannot_spend_the_answer_it_was_checking(monkeypatch):
    """Same hazard, second phase. The stream is a cross-check on an answer the
    page already gave; failing to finish it must cost the check, not the
    answer."""
    from solvers import browser_pool

    monkeypatch.setattr(_Tab, "_reconcile_stream", _forever)
    page = _answered_page()
    tab = _tab(page, _site(stream=True))
    started = time.monotonic()
    reply = asyncio.run(tab.send("solve it", 1.0))
    spent = time.monotonic() - started
    assert spent < browser_pool.STREAM_PHASE_TIMEOUT_S + 4.0, f"tail ran {spent:.1f}s"
    assert "return 'pong'" in reply, f"answer lost to the stream check: {reply!r}"


def test_a_wedged_post_mortem_cannot_be_the_slowest_thing_in_the_solve(capsys, monkeypatch):
    """`_explain_empty` exists to explain a zero. It is a LOG LINE. Unbounded it
    can outlast everything that produced the zero -- and it was not even inside
    a try, so a raise from its final `inner_text` propagated out of `send`."""
    from solvers import browser_pool

    monkeypatch.setattr(_Tab, "_explain_empty", _forever)
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    started = time.monotonic()
    reply = asyncio.run(_tab(page, _site()).send("solve it", 1.0))
    spent = time.monotonic() - started
    assert reply == ""
    assert spent < browser_pool.POSTMORTEM_TIMEOUT_S + 4.0, f"post-mortem ran {spent:.1f}s"
    assert "did not answer in time" in capsys.readouterr().out


def test_the_whole_tail_fits_inside_the_solvers_safety_margin():
    """Arithmetic, not behaviour, and it is the assumption the three bounds
    above are chosen against: they run after the budget is gone, so their sum
    has to fit in what VerifyingSolver held back -- with room left for the last
    grade and the tab close."""
    from solvers import browser_pool

    tail = (
        browser_pool.COPY_PHASE_TIMEOUT_S
        + browser_pool.STREAM_PHASE_TIMEOUT_S
        + browser_pool.POSTMORTEM_TIMEOUT_S
    )
    assert tail < 15.0, f"the tail ({tail}s) can outlast the default safety margin"


def test_a_wedged_snapshot_cannot_eat_the_budget_before_a_prompt_is_even_sent():
    """`_fingerprint` ends in a `get_attribute` on the last message, which is
    the 30s auto-wait again -- and it ran OUTSIDE the submit bound. A page that
    re-renders as the prompt goes out could burn the whole read budget before a
    single poll, on a conversation that would have answered."""

    class _Wedged(_FakePage):
        def locator(self, selector):
            if selector == "#assistant":
                return _Loc(self, selector, [_Slow()])
            return super().locator(selector)

    class _Slow(_Node):
        def __init__(self):
            super().__init__()

        async def get_attribute(self, name):
            await asyncio.sleep(3600)

    page = _Wedged({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    site = _site(message_id_attr="data-id")
    tab = _tab(page, site)
    started = time.monotonic()
    reply = asyncio.run(tab.send("solve it", 1.0))
    spent = time.monotonic() - started
    # The submit budget floors at 5s; anything beyond that is the unbounded read.
    assert spent < 12.0, f"the snapshot ran {spent:.1f}s"
    assert reply == "" and tab.alive is False, "a wedged page must retire the tab"


def test_open_turn_still_snapshots_before_it_submits():
    """Bounding the snapshot moved it into a helper. The ORDER is the thing that
    must survive: a floor taken after the prompt can already have this answer's
    own stream record under it, and the reply would be rebuilt from its own
    prompt."""
    page = _answered_page()
    tab = _tab(page, _site())
    order: list[str] = []
    tab._fingerprint = lambda: _record(order, "fingerprint", (0, None))
    tab._stream_seq = lambda: _record(order, "stream_seq", 7)
    tab._submit = lambda text, ui_ms: _record(order, "submit", None)
    before = asyncio.run(tab._open_turn("hello", 1000))
    assert order == ["fingerprint", "stream_seq", "submit"], order
    assert before == (0, None) and tab._stream_before == 7


async def _record(log: list, name: str, value):
    log.append(name)
    return value


# --- rebuilding capacity must not be billed to the solve that lost it ----- #
# `release()` runs from the solver's `finally`, after the answer is in hand and
# after the budget is spent. Building a tab means a new page, a navigation and
# a wait for the composer -- `ready_timeout_ms` alone is 60 SECONDS -- and the
# deadline above it is an `asyncio.wait_for` in `handle_request` that answers
# 504 rather than late. Awaited there, the replacement would destroy the very
# answer whose failure asked for it, and it is exactly the failing solves, the
# ones with the least budget left, that reach this path.


def _slow_pool(delay: float = 3600.0, site=None) -> BrowserFleet:
    """A fleet whose `_spawn` takes as long as a real signed-in page can."""
    site = site or chatgpt_site()
    pool = _fleet(site)
    pool._size = 1

    async def spawn(context, browser, label):
        await asyncio.sleep(delay)
        tab = _Tab(pool, _DeadPage(), context, f"{label}-new", site=browser.site)
        pool._tabs.append(tab)
        return tab

    pool._spawn = spawn
    return pool


def test_replacing_a_dead_tab_does_not_hold_up_the_answer():
    pool = _slow_pool()

    async def go():
        dead = _Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site())
        dead.alive, dead.leased = False, True
        pool._tabs.append(dead)
        started = time.monotonic()
        await pool.release(dead)
        spent = time.monotonic() - started
        pending = list(pool._pending)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return spent, pending

    spent, pending = asyncio.run(go())
    assert spent < 1.0, f"release() waited {spent:.1f}s for the new tab to load"
    assert pending, "the replacement was dropped instead of being handed off"
    assert pool._lost == 1 and pool._size == 0, "book-keeping is not deferred"


def test_a_replacement_still_lands_in_the_fleet_once_it_finishes_loading():
    """Deferring it must not mean losing it: capacity has to come back, or the
    fleet bleeds a tab on every failure until nothing is left."""
    pool = _slow_pool(delay=0.01)

    async def go():
        dead = _Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site())
        dead.alive, dead.leased = False, True
        pool._tabs.append(dead)
        await pool.release(dead)
        await _settle(pool)

    asyncio.run(go())
    assert pool._free.qsize() == 1, "the replacement never reached the free queue"
    assert pool._size == 1, f"capacity not restored: {pool._size}"


def test_shutdown_does_not_leave_a_half_built_tab_open_in_your_browser():
    """A replacement in flight during shutdown is a page this fleet opened,
    signed in, that nothing else knows about — `_teardown` sweeps `_tabs`, and
    the new tab is not in it until `_spawn` returns."""
    pool = _slow_pool(delay=3600.0)
    pool._pw = SimpleNamespace(stop=lambda: _done(None))

    async def go():
        dead = _Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site())
        dead.alive, dead.leased = False, True
        pool._tabs.append(dead)
        await pool.release(dead)
        spawn = next(iter(pool._pending), None)
        assert spawn is not None, "nothing was spawned to cancel"
        started = time.monotonic()
        await asyncio.wait_for(pool.aclose(), timeout=10)
        # Read the state HERE, inside the loop. Asserting after `asyncio.run`
        # returns proves nothing: its own shutdown cancels whatever is left and
        # the done-callback empties `_pending`, so a teardown that ignored the
        # replacement entirely would look identical from outside.
        return time.monotonic() - started, spawn.done(), list(pool._pending)

    spent, finished, leftover = asyncio.run(go())
    assert spent < 5.0, f"shutdown waited {spent:.1f}s on a replacement"
    assert finished, "the replacement was still loading when the fleet went away"
    assert leftover == [], f"a replacement outlived the fleet: {leftover}"
    assert pool._tabs == [] and pool._size == 0


def test_a_replacement_that_finishes_after_shutdown_is_closed_not_leaked():
    """The other half of the race: the spawn completes before the cancellation
    reaches it. `_teardown` has already swept `_tabs`, so this tab would stay
    open in the operator's browser forever."""
    pool = _fleet(chatgpt_site())
    pool._size = 1
    built: list = []

    async def spawn(context, browser, label):
        pool._closing = True          # shutdown ran while this was loading
        tab = _Tab(pool, _FakePage({}), context, f"{label}-new", site=browser.site)
        pool._tabs.append(tab)
        built.append(tab)
        return tab

    pool._spawn = spawn

    async def go():
        dead = _Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site())
        dead.alive, dead.leased = False, True
        pool._tabs.append(dead)
        await pool.release(dead)
        await _settle(pool)

    asyncio.run(go())
    assert built and built[0]._page.closed, "the late replacement was left open"
    assert pool._free.qsize() == 0, "a page that was just closed was queued as free"


# --- grading is blocking work wearing an async coat ---------------------- #
# `compile_defect` shells out to rustc and `_Grader.check` runs the validator's
# own executor -- each a `subprocess.run` of seconds, and for the Docker backend
# of a container start. Called straight from a coroutine they stop the event
# loop dead, and the loop is not one solve's alone: the miner answers several
# validators at once (`solve_slots` is a semaphore), and the deadline that
# decides whether a solve is PAID is itself an `asyncio.wait_for` -- which
# cannot fire on a loop that is not running.


def test_grading_does_not_stop_the_world_for_every_other_solve():
    """Measured before the fix, a 3s subprocess beside a 1.0s deadline:

        the other solve's 1.0s deadline fired after  3.05s

    Every concurrent solve is pushed past its cutoff by one Rust compile, and
    each of those answers 504 with no answer at all."""
    from solvers.verify import Candidate

    solver = _solver([RIGHT])
    solver._grade = (
        lambda reply, task, left=None, cases=None, previous="":
        time.sleep(1.0) or Candidate(code="x = 1", raw=reply)
    )
    late: list[float] = []

    async def other_solve():
        started = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.sleep(30), timeout=0.2)
        except asyncio.TimeoutError:
            late.append(time.monotonic() - started)

    async def go():
        return await asyncio.gather(
            other_solve(), solver._graded("reply", DIGITS, 30.0)
        )

    _, candidate = asyncio.run(go())
    assert candidate.code == "x = 1", "the grade itself was lost"
    assert late and late[0] < 0.6, (
        f"a concurrent solve's 0.2s deadline fired after {late[0]:.2f}s — "
        f"the event loop was blocked by grading"
    )


def test_a_grade_that_explodes_still_yields_the_answer_it_was_checking():
    """Off-loop or not, the check is subordinate to the answer: a candidate that
    cannot be graded is still a candidate, and an ungraded answer can pass the
    hidden suite where nothing at all cannot."""
    solver = _solver([RIGHT])

    def boom(reply, task, left=None):
        raise RuntimeError("executor gone")

    solver._grade = boom
    candidate = asyncio.run(solver._graded(RIGHT, DIGITS, 30.0))
    assert "def g(n)" in candidate.code, f"answer lost with the grade: {candidate.code!r}"


def test_the_executor_cache_is_built_once_even_under_concurrent_grades():
    """Now reached from worker threads, and concurrently. Two threads missing
    the cache together would each construct an executor — for the Docker backend
    that is a container's worth of startup thrown away, on the one code path
    whose entire reason for caching is that Docker startup is slow."""
    import threading
    from solvers.verify import _Grader

    grader = _Grader()
    built: list[int] = []
    start = threading.Barrier(4)

    def make(settings, language):
        built.append(1)
        time.sleep(0.05)          # widen the window a real construction would have
        return object()

    import rlvr.execution.executor as ex_mod
    original = ex_mod.get_executor
    ex_mod.get_executor = make
    try:
        seen: list = []

        def worker():
            start.wait()
            seen.append(grader.executor("python"))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        ex_mod.get_executor = original

    assert len(built) == 1, f"built the executor {len(built)} times"
    assert len(set(map(id, seen))) == 1, "different threads got different executors"


def test_an_executor_that_cannot_be_built_is_not_rebuilt_on_every_solve(capsys):
    """The cache is written after `get_executor` RETURNS, so a constructor that
    raises left nothing behind and every later solve repeated it.

    For Rust without Docker that constructor is
    `DockerExecutor._resolve_docker`, which shells out to `docker info`: 60ms
    against a missing socket, and up to its own 20 second timeout against a
    daemon that is hung or still starting — inside the solve's budget, per Rust
    task, for the life of the process."""
    from solvers.verify import _Grader

    grader = _Grader()
    tried: list[int] = []

    def unavailable(settings, language):
        tried.append(1)
        raise RuntimeError("DockerExecutor could not contact the Docker daemon")

    import rlvr.execution.executor as ex_mod
    original = ex_mod.get_executor
    ex_mod.get_executor = unavailable
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="Docker daemon"):
                grader.executor("rust")
    finally:
        ex_mod.get_executor = original

    assert len(tried) == 1, f"probed the daemon {len(tried)} times, once per solve"
    out = capsys.readouterr().out
    assert out.count("could not be built") == 1, (
        f"the once-per-run explanation was printed {out.count('could not be built')} "
        f"times: {out}"
    )
    # And it says what it COSTS, not merely what failed: an operator reading
    # one exception per solve has no way to tell that everything in the
    # language is now going out ungraded.
    assert "verified=False" in out and "repair rounds" in out, out


def test_a_docker_daemon_started_after_the_miner_is_picked_up():
    """The hold is a hold, not a verdict. Starting Docker after the miner is the
    ordinary case, and a permanent answer would mean grading no Rust at all
    until someone restarted the process."""
    from solvers import verify as verify_mod
    from solvers.verify import _Grader

    grader = _Grader()
    built = object()
    calls: list[int] = []

    def flaky(settings, language):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no daemon yet")
        return built

    import rlvr.execution.executor as ex_mod
    original, hold = ex_mod.get_executor, verify_mod.EXECUTOR_RETRY_S
    ex_mod.get_executor = flaky
    verify_mod.EXECUTOR_RETRY_S = 0.0     # the hold has elapsed
    try:
        with pytest.raises(RuntimeError):
            grader.executor("rust")
        assert grader.executor("rust") is built, "never tried again"
        assert grader.executor("rust") is built, "the working executor is cached"
    finally:
        ex_mod.get_executor = original
        verify_mod.EXECUTOR_RETRY_S = hold

    assert len(calls) == 2, f"built {len(calls)} times; the second must be cached"


def test_a_missing_executor_says_the_same_thing_on_live_traffic(capsys):
    """Both grading paths print the same four words on purpose.

    An operator counts those lines to decide whether a missing executor is
    costing anything. Live traffic ships no public examples, so the only path
    that can print is the differential one — and while it said something else,
    that count read zero while every answer in the language went out
    ungraded."""
    from solvers.differential import Differential

    class _Unused:
        async def open(self, avoid=None): raise AssertionError("not needed")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Unused(), max_attempts=1, reserve_s=0)

    def unavailable(*a, **kw):
        raise RuntimeError("DockerExecutor could not contact the Docker daemon")

    # 1. The path a task WITH public examples takes.
    solver._grader.check = unavailable
    with_examples = SolveTask(
        problem_id="p", language="python", statement="s", entrypoint="g",
        public_examples=[TestCase(args=[12345], kwargs={}, expected=15)],
        deadline_s=60.0,
    )
    candidate = solver._grade(RIGHT, with_examples, 30.0)
    assert candidate.code.strip(), "the answer was lost with the grading"
    assert "local grading unavailable" in capsys.readouterr().out

    # 2. The path live traffic takes, every time, because it ships none.
    solver._grader.check_detailed = unavailable
    differential = Differential(solver._grader)
    report = differential.compare(
        extract_code(RIGHT, "g", "python"), extract_code(RIGHT, "g", "python"),
        "python", "g",
        [{"name": "carry", "args": [12345], "kwargs": {}}], 30.0,
    )
    out = capsys.readouterr().out
    assert "local grading unavailable" in out, out
    # And the report says it established nothing, rather than reading as a
    # clean agreement that no executor was ever asked about.
    assert not report.ok and report.unrun == 1, report.summary()


def _stub_solver():
    class _Unused:
        async def open(self, avoid=None): raise AssertionError("not needed")
        async def aclose(self): pass
        def stats(self): return {"tabs": 1}

    return VerifyingSolver(_Unused(), max_attempts=1, reserve_s=0)


def test_a_box_with_neither_rust_check_says_so_before_it_costs_anything(capsys):
    """Both Rust checks can be off at once, and nothing said so until a Rust
    challenge had already been answered.

    Without a local toolchain `compile_defect` returns None — which means
    "could not tell", not "fine" — and without a daemon nothing grades. What is
    left is `rust_defect`, a grep of a fenced block for `fn main`: a prompt
    echo, a tool call and a program truncated mid-identifier all carry those
    characters, and all three have been submitted."""
    from solvers import verify as verify_mod

    solver = _stub_solver()

    def no_daemon(settings, language):
        raise RuntimeError("could not contact the Docker daemon")

    import rlvr.execution.executor as ex_mod
    original, had_rustc = ex_mod.get_executor, verify_mod.rustc_path
    ex_mod.get_executor = no_daemon
    verify_mod.rustc_path = lambda: None
    try:
        support = asyncio.run(solver.check_rust_support())
    finally:
        ex_mod.get_executor = original
        verify_mod.rustc_path = had_rustc

    assert support["compile_gate"].startswith("off"), support
    assert support["grading"].startswith("unavailable"), support
    out = capsys.readouterr().out
    assert "WARN" in out and "fn main" in out, out


def test_a_working_toolchain_is_reported_rather_than_warned_about(capsys):
    from solvers import verify as verify_mod

    solver = _stub_solver()

    import rlvr.execution.executor as ex_mod
    original, had_rustc = ex_mod.get_executor, verify_mod.rustc_path
    ex_mod.get_executor = lambda settings, language: object()
    verify_mod.rustc_path = lambda: "/usr/bin/rustc"
    try:
        support = asyncio.run(solver.check_rust_support())
    finally:
        ex_mod.get_executor = original
        verify_mod.rustc_path = had_rustc

    assert support == {"compile_gate": "rustc at /usr/bin/rustc", "grading": "ready"}
    out = capsys.readouterr().out
    assert "WARN" not in out, out


def test_solver_status_reports_the_rust_checks_without_probing_for_them():
    """`/solver-status` is what an operator polls to find out whether the miner
    is healthy. A `docker info` against a hung daemon blocks for twenty seconds,
    so the endpoint must report what is already known and probe nothing."""
    from solvers import verify as verify_mod

    solver = _stub_solver()

    def never(settings, language):
        raise AssertionError("/solver-status probed the daemon")

    import rlvr.execution.executor as ex_mod
    original, had_rustc = ex_mod.get_executor, verify_mod.rustc_path
    ex_mod.get_executor = never
    verify_mod.rustc_path = lambda: None
    try:
        rust = solver.stats()["rust"]
    finally:
        ex_mod.get_executor = original
        verify_mod.rustc_path = had_rustc

    assert rust["grading"] == "not checked yet", rust
    assert rust["compile_gate"].startswith("off"), rust


def test_the_rust_checks_are_probed_before_serving_not_on_the_first_rust_task():
    """At launch, beside the fleet's own warm-up, where someone is watching."""
    from pathlib import Path

    serve = Path(__file__).resolve().parent.joinpath("run_miner.py").read_text()
    body = serve[serve.index("async def serve()"):serve.index("asyncio.run(serve())")]
    assert "check_rust_support" in body, "the Rust checks are never probed"
    assert body.index("await warm_up(") < body.index("check_rust_support"), (
        "probe after the fleet: a browser that will not attach is the more "
        "urgent failure and should print first"
    )


def test_a_compile_check_cannot_outlive_the_answer_it_is_checking():
    """`COMPILE_TIMEOUT_S` defaults to 25s — a hang guard sized for a compiler,
    not for a deadline. The solver's whole safety margin is 15s, and overrunning
    it does not deliver the answer late, it discards it."""
    from solvers import rust_compile

    seen: dict = {}

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        seen.update(kw)
        return _Done()

    original_run, original_path = rust_compile.subprocess.run, rust_compile.rustc_path
    rust_compile.subprocess.run = fake_run
    rust_compile.rustc_path = lambda: "/usr/bin/rustc"
    try:
        rust_compile.compile_defect("fn main() {}", 3.0)
        assert seen["timeout"] == 3.0, f"budget ignored: {seen['timeout']}"
        rust_compile.compile_defect("fn main() {}", -5.0)
        assert seen["timeout"] == 1.0, "a spent budget must still allow one try"
        rust_compile.compile_defect("fn main() {}", 9_999.0)
        assert seen["timeout"] == rust_compile.COMPILE_TIMEOUT_S, "the hang guard was raised"
        rust_compile.compile_defect("fn main() {}")
        assert seen["timeout"] == rust_compile.COMPILE_TIMEOUT_S, "no budget, no change"
    finally:
        rust_compile.subprocess.run = original_run
        rust_compile.rustc_path = original_path


def test_the_compiler_lookup_is_safe_to_race():
    """`rustc_path` caches in module globals and is now called from worker
    threads. `_looked` is read OUTSIDE the lock, so setting it before the lookup
    finishes would let a second thread see True and take `_rustc` — still None —
    as the answer, silently skipping the compile check on that solve."""
    from solvers import rust_compile

    saved = (rust_compile._rustc, rust_compile._looked)
    rust_compile._rustc, rust_compile._looked = None, False
    observed: list[bool] = []

    def slow_which(_name):
        observed.append(rust_compile._looked)
        return "/usr/bin/rustc"

    original = rust_compile.shutil.which
    rust_compile.shutil.which = slow_which
    try:
        assert rust_compile.rustc_path() == "/usr/bin/rustc"
        assert observed == [False], (
            "`_looked` was already True while the lookup was still running — "
            "a racing thread would have read `_rustc` as None"
        )
        assert rust_compile._looked is True
        rust_compile.rustc_path()
        assert observed == [False], "the second call did not use the cache"
    finally:
        rust_compile.shutil.which = original
        rust_compile._rustc, rust_compile._looked = saved


# --- a usage example is not an answer ------------------------------------ #
# Models append `print(g(5))` after the program. That demo is perfectly
# plausible Python, so the moment the block above it picks up ANY defect --
# a genuine truncation, or a false positive -- the demo becomes the last
# plausible block and wins. What gets submitted is a one-liner that calls a
# function nobody defined, and it is archived as "the solution".


def test_a_usage_example_never_outranks_the_program_it_demonstrates():
    """Measured: a correct program whose entrypoint ends in `while True:` was
    replaced by its own `print(g(...))` example, submitted, and archived."""
    answer = (
        "def g(grid):\n"
        "    seen = set()\n"
        "    while True:\n"
        "        for row in grid:\n"
        "            if row in seen:\n"
        "                break\n"
        "            seen.add(row)\n"
        "        if len(seen) == len(grid):\n"
        "            return sorted(seen)\n"
    )
    reply = f"Here:\n\n```python\n{answer}```\n\nUsage:\n\n```python\nprint(g([1, 2, 3]))\n```\n"
    got = extract_code(reply, "g")
    assert got.strip().startswith("def g(grid)"), f"submitted the demo: {got!r}"


def test_a_truncated_attempt_still_beats_the_demo_beneath_it():
    """The fallback exists to hand the repair round a real attempt. A demo
    teaches it nothing — it would be told the code does not define `g`, about a
    block that was never trying to."""
    cut = "def g(n):\n    total = 0\n    for d in str(n):\n        total += int(d)\n"
    reply = f"```python\n{cut}```\n\n```python\nprint(g(5))\n```\n"
    got = extract_code(reply, "g")
    assert got.strip().startswith("def g(n)"), f"kept the demo instead: {got!r}"
    assert "without returning" in (python_defect(got, "g") or ""), (
        "the repair round would hear about the wrong block"
    )


def test_a_break_inside_a_nested_loop_does_not_end_the_outer_while():
    """`ast.walk` sees every break in the subtree, and an inner loop's break
    exits the INNER loop. Counting it marked a correct program as truncated."""
    nested = (
        "def g(n):\n"
        "    while True:\n"
        "        for d in range(n):\n"
        "            if d > 2:\n"
        "                break\n"
        "        return n\n"
    )
    assert python_defect(nested, "g") is None, python_defect(nested, "g")
    # ...and a break that really is bound to the `while` still counts.
    escapes = "def g(n):\n    while True:\n        n -= 1\n        if n < 0:\n            break\n"
    assert "without returning" in (python_defect(escapes, "g") or "")


def test_the_fallback_still_refuses_a_block_that_is_not_source_at_all():
    """Preferring the block that defines the entrypoint must not become a way
    for a tool call to get in: `plausible_source` is still the gate."""
    tool = '{"command": "cat > main.rs << \'EOF\'\\nfn main() {}"}'
    assert extract_code(f"```\n{tool}\n```", "g") == ""
    assert extract_code(f"```\n{tool}\n```", "main", "rust") == ""


def test_defines_reads_a_definition_out_of_source_too_cut_to_parse():
    """A truncation lands at the END of an answer, so the `def` line survives it.
    Without that path a half-written program loses to any demo beside it."""
    from solvers.prompts import _defines

    assert _defines("def g(n):\n    return {", "g") is True   # unparseable
    assert _defines("async def g(n):\n    x = [", "g") is True
    assert _defines("def other(n):\n    return {", "g") is False
    assert _defines("g = lambda n: n", "g") is True
    assert _defines("fn main() {\n    let x =", "main", "rust") is True
    assert _defines("let x = 1;", "main", "rust") is False


def test_the_examples_are_not_run_once_the_budget_is_already_gone(capsys):
    """Each case gets VERIFY_TIMEOUT_S, in a subprocess or a container, and
    after the last send there is nothing left to spend it from. `verified` never
    reaches the validator — it feeds this process's cache and stats — so the
    only thing the run could still buy is a repair round there is no time for.
    The deadline above answers 504 rather than late, so the check would be paid
    for with the answer it was checking."""
    solver = _solver([RIGHT])
    ran: list = []
    solver._grader.check_detailed = (
        lambda *a, **kw: ran.append(a) or (2, 2, [], [], [])
    )

    spent = solver._grade(RIGHT, DIGITS, -0.5)
    assert ran == [], "the grader ran on a budget that was already spent"
    assert spent.code.strip() and spent.defect is None, "the answer was lost with the check"
    assert "unverified" in capsys.readouterr().out

    # With budget left, and with none stated at all, it still runs.
    assert solver._grade(RIGHT, DIGITS, 30.0).passed == 2
    assert solver._grade(RIGHT, DIGITS).passed == 2
    assert len(ran) == 2


def test_a_win_is_not_credited_to_a_model_that_did_not_produce_it():
    """`asked[-1]` was a proxy for the winner. A pass whose backend reports no
    provider is absent from `asked` while still able to produce the winning
    answer, and the credit then landed on the PREVIOUS model — in the one number
    an operator reads to decide which account has started failing."""

    class _Anonymous:
        """First a named model that gets it wrong, then one that will not say
        who it is and gets it right."""

        # Per PASS, not per open: a pass opens one conversation per phase, so
        # counting opens put the anonymous model inside the first pass.
        _unset = object()

        def __init__(self):
            self._avoid, self._pass = self._unset, 0
            self._scripts = {}

        async def open(self, avoid=None):
            if avoid != self._avoid:
                self._avoid, self._pass = avoid, self._pass + 1
                self._scripts[self._pass] = _Script(
                    [WRONG] if self._pass == 1 else [RIGHT]
                )
            script = self._scripts[self._pass]
            if self._pass == 1:
                return _Chat(script, provider="claude")
            chat = _Chat(script)
            del chat.provider          # reports nothing about itself
            return chat

        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Anonymous(), max_attempts=1, reserve_s=0, max_budget_s=120
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified, "the second pass was supposed to win"
    rows = solver.stats()["providers"]
    assert rows["claude"]["asked"] == 1, rows
    assert rows["claude"]["verified"] == 0, (
        f"claude was credited with an answer it did not produce: {rows}"
    )


def test_a_winner_outside_the_asked_list_is_still_counted():
    """`_note` used to index `_by_provider[winner]` directly, on the assumption
    that a winner is always someone who was asked. It is one KeyError away from
    losing a whole solve to the stats line at the end of it."""
    solver = _solver([RIGHT])
    solver._note("chatgpt", ["claude"])
    rows = solver.stats()["providers"]
    assert rows["claude"] == {"asked": 1, "verified": 0}, rows
    assert rows["chatgpt"] == {"asked": 0, "verified": 1}, rows


def test_an_answer_that_lands_just_after_the_deadline_is_still_submitted(capsys):
    """`page_blocks` is fetched AFTER the read loop gave up, so a non-empty one
    means the reply rendered in the moments between the last poll and now. The
    rescue used to require the page to be empty before it looked anywhere, so
    that reading was discarded: measured, `best=""` beside
    `page_blocks=["def g(n): ..."]` returned "" — with the answer sitting in a
    list in the function's own arguments."""
    site = _site(stream=True)
    tab = _tab(None, site)
    tab._sent = "solve it"

    async def nothing_on_the_wire():
        return "prose the model emitted, with no fenced block in it"

    tab._streamed_markdown = nothing_on_the_wire
    got = asyncio.run(tab._reconcile_stream((0, None), "", ["def g(n):\n    return n"]))
    assert "def g(n)" in got, f"threw away the page's late answer: {got!r}"
    assert "just after it" in capsys.readouterr().out


def test_the_wire_is_used_when_the_page_has_nothing_at_all():
    """The other order: no page blocks, an answer on the wire."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"

    async def wire():
        return "here:\n\n```python\ndef g(n):\n    return n\n```\n"

    tab._streamed_markdown = wire
    got = asyncio.run(tab._reconcile_stream((0, None), "", []))
    assert "def g(n)" in got, f"the wire rescue stopped working: {got!r}"


def test_a_page_read_taken_before_the_answer_finished_never_beats_the_wire(capsys):
    """The rule the copy control has always had, on the pair that never got it.

    A reading wins on FIDELITY and has no authority at all on COMPLETENESS, and
    the two are different questions. `_cut_short_by` decided that for copy vs
    page; page vs wire got the comparison, a printed note, and no decision.
    Measured on a live solve, in the operator's own log:

        what the page shows and what came off the wire are not the same — they
        differ at character 5605: the page nothing (it ends here), the wire ' '
        (U+0020). Using the page.

    The page ENDED at 5605 and the wire carried on. The page was the answer cut
    short, the wire held the whole of it, and the truncation was submitted."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"
    whole = "def g(n):\n    t = 0\n    while n > 0:\n        t += n % 10\n        n //= 10\n    return t"

    async def wire():
        return f"```python\n{whole}\n```\n"

    tab._streamed_markdown = wire
    # What the page had when the deadline landed: the same answer, cut off.
    cut = "def g(n):\n    t = 0\n    while n > 0:\n        t"
    got = asyncio.run(tab._reconcile_stream((0, None), _Tab._fence(cut), [cut]))

    assert "return t" in got, f"submitted the page's truncation: {got!r}"
    log = capsys.readouterr().out
    assert "CUT SHORT" in log and "came off the wire" in log, log
    # And the count is a count, not a once-per-tab flag: every one of these is
    # an answer that would have gone out truncated, so the NUMBER is the thing.
    assert tab._cut_short_stream == 1


def test_the_wire_wins_only_when_the_page_is_its_PREFIX(capsys):
    """The other half, and the reason the wire does not simply win.

    Both stream formats are private, undocumented and free to change on any
    deploy, and the reconstruction is a heuristic over their JSON. The one thing
    to fear is a reading that picked up the CONVERSATION rather than the reply —
    and such a reading cannot have the page as its prefix. So a difference in
    the MIDDLE leaves the page in charge exactly as before, and only says so."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"
    page = "def g(n):\n    return n + 1"

    async def different_in_the_middle():
        return "```python\ndef g(n):\n    return n + 2\n```\n"

    tab._streamed_markdown = different_in_the_middle
    got = asyncio.run(tab._reconcile_stream((0, None), _Tab._fence(page), [page]))

    assert "n + 1" in got, f"a mid-answer disagreement handed the wire the answer: {got!r}"
    log = capsys.readouterr().out
    assert "Using the page" in log, log
    assert "CUT SHORT" not in log, log
    assert tab._cut_short_stream == 0


def test_a_wire_shorter_than_the_page_never_replaces_it(capsys):
    """The direction that must never fire. A stream capture that started late
    holds the END of the answer, not the whole of it, and the page is then the
    fuller reading — the prefix test is what tells the two apart."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"
    page = "def g(n):\n    t = 0\n    for d in str(n):\n        t += int(d)\n    return t"

    async def truncated_wire():
        return "```python\ndef g(n):\n    t = 0\n```\n"

    tab._streamed_markdown = truncated_wire
    got = asyncio.run(tab._reconcile_stream((0, None), _Tab._fence(page), [page]))

    assert "return t" in got, f"took a wire reading shorter than the page: {got!r}"
    assert tab._cut_short_stream == 0


def test_the_deadline_read_is_tested_for_completeness_like_every_other(capsys):
    """`best` comes off the read loop, which stops at the deadline. The page and
    the wire are both read AFTER it, so `best` is the oldest of the three and
    the likeliest to be short — and until this ran, nothing ever asked.

    Measured: `best` a truncation, the page and the wire each holding the whole
    program and agreeing with each other exactly. `_cut_short_by` saw no gap
    between THEM and `_first_difference` found nothing to report, so control
    fell through to `return best` and the truncation was submitted without a
    single line of log. Every comparison in the function was made and the one
    that mattered was not among them."""
    whole = ("def g(n):\n    total = 0\n    for i in range(n):\n"
             "        total += i\n    return total")
    cut = "def g(n):\n    total = 0\n    for i in ra"
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"

    async def wire():
        return f"```python\n{whole}\n```\n"

    tab._streamed_markdown = wire
    got = asyncio.run(tab._reconcile_stream((0, None), _Tab._fence(cut), [whole]))

    assert "return total" in got, f"submitted the deadline read's truncation: {got!r}"
    log = capsys.readouterr().out
    assert "CUT SHORT" in log, log
    assert tab._cut_short_stream == 1


def test_the_page_is_re_read_whatever_the_network_did(capsys):
    """These are independent questions — did the network hold the answer, has
    the page since rendered it — and the second used to be asked only when the
    first said yes.

    The refetch and the late-page rescue sat BELOW the early return taken when
    the capture came back empty. So a wire that captured prose recovered the
    program off the page, and a wire that captured nothing threw the identical
    page away and submitted "". Whether the answer was looked at depended on
    something with no bearing on it."""
    whole = "def g(n):\n    return n"

    for label, wire in (("nothing", None), ("prose only", "just prose, no block")):
        tab = _tab(None, _site(stream=True))
        tab._sent = "solve it"
        looked = []

        async def streamed(w=wire):
            return w

        async def new_reply(before):
            looked.append("page")
            return object()

        tab._streamed_markdown = streamed
        tab._new_reply = new_reply
        tab._dom_blocks = lambda reply: _done([whole])
        tab._whole = lambda reply: _done("here is the answer")

        got = asyncio.run(tab._reconcile_stream((0, None), "", None))
        assert looked == ["page"], f"wire={label}: never re-read the page"
        assert "return n" in got, f"wire={label}: threw the page's answer away: {got!r}"


def test_the_page_refetch_cannot_hand_back_our_own_prompt():
    """The refetch is a second route to a submission, so it asks the same
    question the scrape path asks in `_poll`.

    An assistant selector that also matches the USER's turn hands back a message
    whose whole text is our prompt and whose code block is whatever the
    statement quoted — so the block alone looks like a fine answer, and the
    guard at the single exit is testing the block, not the message. That is
    exactly how two of this miner's own prompts reached a validator as Rust
    programs."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "Solve this programming problem in Python.\nReturn the digit sum."

    async def no_wire():
        return None

    tab._streamed_markdown = no_wire
    tab._new_reply = lambda before: _done(object())
    tab._dom_blocks = lambda reply: _done(["example_from_the_statement()"])
    tab._whole = lambda reply: _done(tab._sent)

    got = asyncio.run(tab._reconcile_stream((0, None), "", None))
    assert got == "", f"handed back a block from our own echoed prompt: {got!r}"


def test_opening_a_tab_is_bounded_by_the_solve_budget():
    """`BrowserFleet.open` waits for a free tab up to `MINER_TAB_WAIT_S`, which
    ships at 120s, and nothing here ever passed a smaller number. On a busy
    fleet that is 120s per pass against a deadline that knows nothing about it:
    measured, budget 40s, elapsed 50.1s, `open()` called at t=0 and t=25.1,
    prompts sent 0, answer empty."""
    asked = []

    class _Slow:
        async def open(self, avoid=None, timeout_s=None):
            asked.append(timeout_s)
            raise RuntimeError("no free tab")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Slow(), reserve_s=0, max_budget_s=40,
                             second_opinion=False)
    asyncio.run(solver.solve_task(
        SolveTask(problem_id="p", language="python", statement="s", entrypoint="g",
                  public_examples=[], deadline_s=40.0),
        timeout_s=40.0,
    ))
    assert asked, "open() was never called"
    assert all(t is not None for t in asked), f"unbounded lease wait: {asked}"
    assert all(t <= 40.0 for t in asked), f"waited longer than the whole solve: {asked}"


def test_a_backend_without_a_lease_timeout_is_still_bounded():
    """`Backend.open` has always been `open(avoid=...)`. A keyword a custom
    backend does not take would be a TypeError inside the one call the whole
    solve depends on, so the bound is offered and then applied from outside."""
    class _TwoArg:
        async def open(self, avoid=None):
            await asyncio.sleep(30)
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_TwoArg(), reserve_s=0, max_budget_s=12,
                             second_opinion=False)
    started = time.monotonic()
    answer = asyncio.run(solver.solve_task(
        SolveTask(problem_id="p", language="python", statement="s", entrypoint="g",
                  public_examples=[], deadline_s=12.0),
        timeout_s=12.0,
    ))
    elapsed = time.monotonic() - started
    assert elapsed < 20.0, f"a two-argument backend hung for {elapsed:.0f}s"
    assert answer.code == ""


def test_grading_is_not_started_when_it_cannot_finish():
    """`_Grader.check` gives every case `VERIFY_TIMEOUT_S` and nothing bounds
    the run as a whole, so a candidate that times out on each of its cases
    spends that many multiples of it. Measured: 6 cases x 5s = 30s of executor
    time bought with 0.2s of budget, on a verdict nothing could act on — there
    is no time for a repair round and `verified` never reaches the validator."""
    from types import SimpleNamespace

    from solvers.verify import VERIFY_TIMEOUT_S

    solver, _ = _solver_seeing([])
    ran = []
    solver._grader = SimpleNamespace(
        check_detailed=lambda *a, **k: (ran.append(1), (0, 1, ["boom"], [], []))[1]
    )
    task = SimpleNamespace(language="python", entrypoint="g", statement="s",
                           public_examples=[])
    reply = "```python\ndef g(n):\n    return n\n```"
    cases = [{"name": f"c{i}", "args": [i], "expected": i} for i in range(6)]

    from solvers.differential import Differential

    solver._grader.outputs = lambda *a, **k: [
        SimpleNamespace(ok=True, value=i, error=None, timed_out=False)
        for i in range(6)
    ]

    def compare(budget):
        return Differential(solver._grader).compare(
            reply, "def g(n):\n    return n\n", "python", "g", cases,
            budget_s=budget,
        )

    # Six cases at VERIFY_TIMEOUT_S each cannot fit in a fifth of a second.
    compare(0.2)
    assert not ran, "started a grading run that could not finish inside the budget"

    # ...and with the time to do it, it still runs.
    compare(300.0)
    assert ran, "stopped grading when there was plenty of budget"

    # The demand is ONE case, not the whole suite: a task with twenty cases must
    # not refuse to grade anything under a hundred seconds, because a partial
    # run that DOES fit is worth more than no evidence at all, and `check`
    # bounds the rest itself.
    ran.clear()
    compare(VERIFY_TIMEOUT_S + 0.1)
    assert ran, "refused a six-case suite the budget could start"

    # And grading must never be the thing that is too expensive when another
    # model round trip is not. A band where the loop would spend twelve seconds
    # on a prompt while refusing a check that measures under a second is not a
    # trade anything would choose; it existed for two years because the two
    # floors were never reasoned about together. (The loop's own floor is gone
    # now; the grade's demand is still one case.)
    ran.clear()
    compare(12.1)
    assert ran, "refused to grade with 12.1s left"


def test_nothing_anywhere_still_submits_nothing(capsys):
    """The streamed text is never handed back raw — a chat stream carries the
    conversation, and two of the miner's own prompts reached a validator as Rust
    programs that way."""
    tab = _tab(None, _site(stream=True))
    tab._sent = "solve it"

    async def prose():
        return "I need more information about the input format."

    tab._streamed_markdown = prose
    assert asyncio.run(tab._reconcile_stream((0, None), "", [])) == ""
    assert "Nothing to submit" in capsys.readouterr().out


# --- one reader for fenced markdown, not two ----------------------------- #
# `browser_pool` scanned fences line by line while `extract_code` matched them
# with a regular expression, and the two disagreed about where a block ends.
# Each disagreement below was measured losing a whole answer.


def _grades(reply, entry="g", lang="python"):
    from solvers.prompts import rust_defect

    code = extract_code(reply, entry, lang)
    defect = rust_defect(code) if lang == "rust" else python_defect(code, entry)
    return code, defect


def test_a_closing_fence_has_to_be_a_whole_line():
    """`fence = "```"` inside a program is not the end of the block. The regex
    matched its backticks anywhere and truncated the answer at that line."""
    prog = 'def g(t):\n    fence = "```"\n    return t.count(fence)'
    code, defect = _grades(f"Here:\n\n```python\n{prog}\n```\n")
    assert defect is None, defect
    assert code.strip() == prog, f"truncated at the inner fence: {code!r}"


def test_a_reply_cut_off_mid_block_keeps_the_program_it_did_write():
    """The commonest thing a deadline does. With no closing fence the regex
    matched nothing, the extractor fell through to its all-prose path, and a
    fully written program was returned as ''."""
    code, defect = _grades(
        "Here you go:\n\n```python\ndef g(n):\n    return sum(int(c) for c in str(n))"
    )
    assert defect is None, defect
    assert code.strip().startswith("def g(n)"), f"lost the answer: {code!r}"
    assert "Here you go" not in code, "the prose came with it"


def test_tilde_fences_are_fences():
    """CommonMark says so, and a model that uses them is not wrong."""
    code, defect = _grades("~~~python\ndef g(n):\n    return len(str(n))\n~~~")
    assert defect is None and code.strip().startswith("def g(n)"), code


def test_a_four_backtick_fence_keeps_the_three_backticks_inside_it():
    """Markdown's rule for a block that contains a fence. The regex's trailing
    backtick run ate the wrong one and left a stray fence in the code."""
    prog = 'FENCE = """\n```\n"""\n\n\ndef g(t):\n    return t.count(FENCE.strip())'
    code, defect = _grades(f"````python\n{prog}\n````")
    assert defect is None, defect
    assert code.strip() == prog, f"the fence rule was not applied: {code!r}"
    scope: dict = {}
    exec(compile(code, "<test>", "exec"), scope)
    assert scope["g"]("``` and ``` again") == 2


def test_both_readers_are_now_the_same_function():
    """The point of the change: two readers of one markdown format that
    disagree is a bug waiting for the reply that tells them apart."""
    from solvers import browser_pool
    from solvers.prompts import fenced_blocks

    assert browser_pool._fenced_blocks is fenced_blocks


# --- an answer split across two blocks ----------------------------------- #
# A model told to send ONE code block sometimes sends its imports in a block of
# their own. Taking the block that defines the entrypoint then leaves the
# imports behind, and the result is worse than a visibly broken answer: it
# parses, it defines the right function, `python_defect` passes it, and every
# hidden test fails with `NameError: name 'math' is not defined` while nothing
# anywhere says so.


def _runs(reply, entry="g", args=(16,)):
    """Extract, then actually run it the way the grader will."""
    code = extract_code(reply, entry)
    assert python_defect(code, entry) is None, python_defect(code, entry)
    scope: dict = {}
    exec(compile(code, "<test>", "exec"), scope)
    return code, scope[entry](*args)


def test_imports_left_in_their_own_block_are_carried_to_the_answer():
    code, value = _runs("```python\nimport math\n```\n\n"
                        "```python\ndef g(n):\n    return math.isqrt(n)\n```")
    assert value == 4, value
    assert code.startswith("import math"), code


def test_only_the_imports_the_answer_actually_needs_are_carried():
    code, value = _runs("```python\nimport math\nimport json\n```\n\n"
                        "```python\ndef g(n):\n    return math.isqrt(n)\n```")
    assert value == 4 and "import json" not in code, code


def test_a_from_import_is_carried_the_same_way():
    code, value = _runs(
        "```python\nfrom collections import Counter\n```\n\n"
        "```python\ndef g(s):\n    return len(Counter(s))\n```",
        args=("aab",),
    )
    assert value == 2 and "from collections import Counter" in code, code


def test_an_import_the_answer_does_not_use_is_never_dragged_in():
    """The dangerous direction. `import numpy` prepended to a self-contained
    answer turns a working program into an ImportError on every hidden test."""
    code, value = _runs("```python\nimport numpy\n```\n\n"
                        "```python\ndef g(n):\n    return n * 2\n```", args=(3,))
    assert value == 6 and "numpy" not in code, code


def test_only_the_import_lines_are_taken_never_the_code_beside_them():
    """The safety property. A header block that also RUNS something — a
    `sys.setrecursionlimit` call, a `print` — still yields its imports, and the
    statement that would execute at import time is left exactly where it is."""
    code, value = _runs("```python\nprint('setting up')\nimport math\n```\n\n"
                        "```python\ndef g(n):\n    return math.isqrt(n)\n```")
    assert value == 4, value
    assert "import math" in code, "the import was lost with the block around it"
    assert "print(" not in code, f"a running statement was prepended: {code!r}"


def test_a_block_with_no_imports_at_all_contributes_nothing():
    code, value = _runs("```python\nprint('setting up')\n```\n\n"
                        "```python\ndef g(n):\n    return n + 1\n```", args=(1,))
    assert value == 2 and "print(" not in code, code


def test_a_local_name_that_shadows_a_module_is_not_a_missing_import():
    code, value = _runs("```python\nimport math\n```\n\n"
                        "```python\ndef g(n):\n    math = 2\n    return n * math\n```",
                        args=(3,))
    assert value == 6 and "import math" not in code, code


def test_carrying_imports_does_not_resurrect_the_usage_demo():
    """The two rules have to hold together: take the block that defines the
    entrypoint, and give it the imports it needs — not the demo below it."""
    code, value = _runs("```python\nimport math\n```\n\n"
                        "```python\ndef g(n):\n    return math.isqrt(n)\n```\n\n"
                        "```python\nprint(g(16))\n```")
    assert value == 4 and "print(g(" not in code, code


# `examples/problems` holds 97 archived exchanges -- the exact request the
# validator sent and the exact answer that went back. That makes the answers a
# regression fixture rather than a hypothetical: they are what this miner
# actually produced, including the ways it went wrong. 65 of the 97 carry an
# answer at all; the other 32 submitted nothing, which is the failure the
# reading-path work exists to reduce.
#
# Read from the archives rather than from `solutions/`, which is where a local
# rehearsal WRITES and is therefore not a record of anything.
def _archived_answers(language: str) -> list:
    import json
    import pathlib

    where = pathlib.Path(__file__).resolve().parents[2] / "examples" / "problems"
    out = []
    for f in sorted(where.glob("*.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        if record.get("request", {}).get("language") != language:
            continue
        code = (record.get("response") or {}).get("code", "")
        if code.strip():
            out.append((f.stem, code))
    return out


def test_a_truncated_rust_program_is_caught_without_a_compiler():
    """The check Python gets from `ast.parse` and `_always_returns`, and Rust
    had only from a compiler that is allowed not to be there.

    `rust_defect`'s other two tests both PASS a truncation: the first line still
    opens like Rust and `fn main` still begins a line. Measured on a real
    archived submission — 10,608 bytes, 75 `{` against 71 `}`, ending
    mid-identifier four blocks deep. `rustc` calls it `error: this file contains
    an unclosed delimiter`; nothing here did, and it went out as a confident
    answer that cannot compile."""
    from solvers.prompts import rust_defect

    cut = ("fn main() {\n    for i in 0..10 {\n        if i > 3 {\n"
           "            let x = i * 2;\n            i")
    defect = rust_defect(cut)
    assert defect and "unclosed" in defect, defect
    assert "cut off mid-answer" in defect, defect

    assert rust_defect('fn main() {\n    println!("ok");\n}\n') is None


def test_the_rust_truncation_check_does_not_fire_on_valid_rust():
    """A false positive here does not merely cost a repair round: a block
    carrying a defect loses `extract_code`'s "last gradeable" preference, so a
    trailing usage example can outrank the real answer. Every construct below is
    one a naive delimiter counter gets wrong."""
    from solvers.prompts import _rust_unclosed, rust_defect

    valid = {
        "brace in a string": 'fn main() { println!("}{ not real"); }',
        "brace in a char": "fn main() { let c = '}'; let d = '{'; }",
        "escaped quote char": "fn main() { let q = '\\''; let b = '\\\\'; }",
        "lifetime": 'struct S<\'a> { s: &\'a str }\nfn main() { let _ = S { s: "h" }; }',
        "loop label": "fn main() { 'outer: loop { break 'outer; } }",
        "raw string": 'fn main() { let s = r#"he said "}" loudly"#; }',
        "nested block comment": "fn main() { /* a /* b { */ c */ let x = 1; }",
        "line comment": "fn main() { // } not real\n    let x = 1;\n}",
        "byte string and byte char": 'fn main() { let b = b"}"; let c = b\'{\'; }',
        "raw identifier": "fn main() { let r#type = 1; let _ = r#type; }",
        "unicode escape char": "fn main() { let e = '\\u{1F600}'; }",
        "closure": 'fn main() { let f = |x: i32| { x + 1 }; println!("{}", f(1)); }',
    }
    for label, src in valid.items():
        assert rust_defect(src) is None, f"{label}: {rust_defect(src)}"

    # And it gives UP rather than guessing when the scan meets something it
    # cannot account for — an unterminated string or comment is as likely to be
    # this scanner misreading Rust as it is to be a broken program.
    assert _rust_unclosed('fn main() {\n    println!("unterminated') is None
    assert _rust_unclosed("fn main() {\n    /* thinking about it") is None


def test_a_loop_with_an_else_that_returns_is_not_a_truncation():
    """`_always_returns` handled `If`, `With`, `Try`, `while True` and `Match`,
    and fell through to False for every `For`. So the ordinary "search, else
    report not found" shape was reported as *can reach the end of its body
    without returning ... which is what a reply cut off mid-answer looks like*,
    about a correct program.

    A loop's `else` runs on every exit that is not a `break`, so an `else` that
    always returns leaves no way to fall through — the same rule `while True:`
    already had, decided by the same helper."""
    from solvers.prompts import python_defect

    found = ("def g(n):\n    for i in range(n):\n        if i == 3:\n"
             "            return i\n    else:\n        return -1\n")
    assert python_defect(found, "g") is None, python_defect(found, "g")

    while_else = "def g(n):\n    while n:\n        n -= 1\n    else:\n        return n\n"
    assert python_defect(while_else, "g") is None, python_defect(while_else, "g")

    # A `break` bound to the loop SKIPS the else, so control can still fall
    # through — and a bare loop with no else says nothing either way.
    breaks = ("def g(n):\n    for i in range(n):\n        if i == 3:\n"
              "            break\n    else:\n        return -1\n")
    assert python_defect(breaks, "g"), "a break past the else was not noticed"
    bare = "def g(n):\n    for i in range(n):\n        pass\n"
    assert python_defect(bare, "g"), "a bare loop stopped being a truncation signal"


def test_a_generator_is_reported_as_a_generator_not_as_a_truncation():
    """Still a defect — the grader compares RETURN VALUES structurally, so what
    it receives is a generator object rather than the answer. But it is not a
    reply that was cut off, and saying so sent the model looking for a
    truncation that was not there."""
    from solvers.prompts import python_defect

    gen = "def g(n):\n    for i in range(n):\n        yield i\n"
    defect = python_defect(gen, "g")
    assert defect and "generator" in defect, defect
    assert "cut off" not in defect, defect

    # A nested helper that yields makes IT a generator, not `g`.
    nested = "def g(n):\n    def h():\n        yield 1\n    return list(h())\n"
    assert python_defect(nested, "g") is None, python_defect(nested, "g")


def test_every_archived_rust_answer_is_judged_the_same_way_as_rustc():
    """The 43 committed submissions in `solutions/` are the exact bytes the
    validator received, which makes them a regression fixture rather than a
    hypothetical. Exactly one is truncated; the structural check has to find
    that one and leave the other seventeen alone."""
    import pathlib

    from solvers.prompts import rust_defect

    answers = _archived_answers("rust")
    if not answers:
        pytest.skip("the archived exchanges are not checked out")
    assert len(answers) > 30, f"only {len(answers)} rust answers to check"

    # Exactly the one rustc calls truncated, and no others. Verified against
    # `rustc --edition=2021` over all of them: 39 agree, 0 disagree.
    truncated = [name for name, code in answers
                 if "unclosed" in (rust_defect(code) or "")]
    assert truncated == ["252c5febd7c1eacb670775cc5bbc99e4e2b180c15b64c81648fe3cb89afcb3ca"], (
        truncated
    )


def test_a_commented_first_line_is_python_not_a_root_shell_prompt():
    r"""`_SHELL_OPENER_RE` opened with `[$#>]\s`, where `#` meant a root shell
    prompt. In Python `# ` is a comment, and a commented first line is one of
    the commonest ways a program starts — so the whole block was declared "not
    source at all", `extract_code` fell past it, and the model's own one-line
    usage example was submitted instead. Deleting only the comment made the same
    reply return the program."""
    from solvers.prompts import plausible_source

    answer = ("# Sliding window over the log lines.\n"
              "def g(lines):\n    best = 0\n    for ln in lines:\n"
              "        if ln.startswith('E'):\n            best += 1\n    return best\n")
    assert plausible_source(answer, "python"), "a commented answer is not source"

    reply = f"```python\n{answer}```\n\nExample:\n\n```python\nprint(g(['E1']))\n```\n"
    got = extract_code(reply, "g", "python")
    assert "def g(lines)" in got, f"submitted the demo instead of the answer: {got!r}"
    assert "print(g(" not in got, got

    # ...and the thing the `#` alternative was there for still fails: a root
    # prompt is a prompt CHARACTER followed by a command, which no comment is.
    for tool_call in ("# cat > main.rs << 'EOF'\nfn main() {}\n",
                      "# pip install numpy\n",
                      "$ python3 solve.py\n",
                      "> npm run build\n",
                      "cd /home/claude && python sol.py\n",
                      '{"command": "mkdir -p /home/claude"}\n'):
        assert not plausible_source(tool_call, "python"), tool_call


def test_a_code_block_nested_under_a_list_item_still_parses():
    """Markdown REQUIRES the indentation of every line inside a block nested
    under a list item, and it is not part of the source. Keeping it handed
    `extract_code` a block whose every line began with three spaces; `.strip()`
    then removed them from the first line only, and a program the model wrote
    correctly came back as `unexpected indent, line 3`."""
    from solvers.prompts import fenced_blocks

    reply = ("Plan:\n\n"
             "1. Sort the distinct values.\n"
             "2. Sum the top two:\n\n"
             "   ```python\n"
             "   import math\n"
             "\n"
             "   def solve(nums):\n"
             "       vals = sorted(set(nums), reverse=True)\n"
             "       return sum(vals[:2]) if len(vals) >= 2 else 0\n"
             "   ```\n")
    got = extract_code(reply, "solve", "python")
    assert python_defect(got, "solve") is None, python_defect(got, "solve")
    assert got.startswith("import math"), got

    # Never MORE than the fence had: a line the author indented further keeps
    # the difference, which is the whole of the program's own structure.
    assert fenced_blocks("  ```py\n  a\n      b\n  ```\n") == ["a\n    b\n"]
    # An unnested block — every ordinary reply — is untouched.
    assert fenced_blocks("```py\ndef g():\n    return 1\n```\n") == [
        "def g():\n    return 1\n"
    ]
    # A tab is never partially removed; guessing its width would corrupt source.
    assert fenced_blocks(" ```py\n\tdef g():\n\t\treturn 1\n ```\n") == [
        "\tdef g():\n\t\treturn 1\n"
    ]


def test_a_carried_import_never_displaces_a_future_import():
    """`from __future__` must be the first statement in the file, after at most
    a docstring. `import math` above it is source `ast.parse` accepts and the
    grader's import rejects — so it was reported CLEAN and scored zero."""
    reply = ("```python\nimport math\n```\n\n"
             "```python\nfrom __future__ import annotations\n\n"
             "def g(n):\n    return math.isqrt(n)\n```\n")
    got = extract_code(reply, "g", "python")
    assert got.lstrip().startswith("from __future__"), got
    assert "import math" in got, got
    compile(got, "<solution>", "exec")          # the question the grader asks
    assert python_defect(got, "g") is None

    # A docstring may precede it, and has the same must-be-first rule.
    with_doc = ('```python\nimport math\n```\n\n```python\n"""Solve it."""\n'
                "from __future__ import annotations\n\ndef g(n):\n    return math.isqrt(n)\n```\n")
    got = extract_code(with_doc, "g", "python")
    compile(got, "<solution>", "exec")
    assert got.lstrip().startswith('"""Solve it."""'), got

    # With nothing that must come first, the import still goes to the top.
    plain = "```python\nimport math\n```\n\n```python\ndef g(n):\n    return math.isqrt(n)\n```\n"
    assert extract_code(plain, "g", "python").startswith("import math"), plain


def test_the_parse_gate_asks_what_the_grader_will_ask():
    """The validator IMPORTS this source, and import COMPILES it — so
    `ast.parse` is the wrong question by exactly the set of programs that parse
    and will not compile."""
    bad = "import math\nfrom __future__ import annotations\ndef g(n):\n    return n\n"
    import ast as _ast
    _ast.parse(bad)                              # parses...
    defect = python_defect(bad, "g")             # ...and is caught anyway
    assert defect and "not valid Python" in defect, defect

    # Nothing that compiles today may start failing: the archived answers are
    # the exact bytes the validator received.
    checked = 0
    for _, src in _archived_answers("python"):
        try:
            _ast.parse(src)
        except SyntaxError:
            continue
        compile(src, "<archived>", "exec")
        checked += 1
    assert checked > 20, f"only {checked} archived python answers were checked"


def test_a_rust_preamble_split_into_its_own_block_is_carried_too():
    """Rust used to be left to its compiler here: `use` has the same shape as
    `import`, and a Rust answer is put through rustc, which says so.

    The compiler is allowed not to be there. With no local `rustc`, or with
    `SOLVER_RUST_COMPILE=0`, nothing says so at all — and on the operator's own
    host every Rust solve went out ungraded for want of a Docker daemon. So the
    `use` lines are carried, on the same argument as the Python path.

    Narrower than the Python path, though, because Rust name resolution is not
    something to guess at: the earlier block has to be NOTHING but `use` lines,
    attributes and comments, and the chosen block has to have no `use` of its
    own."""
    got = extract_code("```rust\nuse std::io;\n```\n\n"
                       "```rust\nfn main() {\n    println!(\"x\");\n}\n```", "main", "rust")
    assert "use std::io;" in got, f"lost the preamble the program needs: {got!r}"
    assert got.rstrip().endswith("}"), got

    # A block with its own `use` is complete; nothing is prepended to it.
    own = extract_code("```rust\nuse std::io;\n```\n\n"
                       "```rust\nuse std::fmt;\nfn main() {}\n```", "main", "rust")
    assert "std::io" not in own, f"stacked a preamble onto a complete program: {own!r}"

    # An earlier block that is a PROGRAM is not a preamble, and is left alone.
    prog = extract_code("```rust\nfn helper() {}\n```\n\n"
                        "```rust\nfn main() {}\n```", "main", "rust")
    assert prog.strip() == "fn main() {}", prog


def test_a_fence_that_ends_a_line_of_prose_still_opens_a_block():
    """Markdown says a fence opens a line, and a model that writes
    `Here you go: ```python` has broken that rule — but it has still answered.
    Requiring the line to START with the fence dropped that answer entirely:
    no block found, so the extractor fell through to its all-prose path and
    returned "". Caught by this suite when the reader was unified."""
    ans = "def g(n):\n    return sum(int(c) for c in str(n))"
    for reply in (
        f"I will explain at length. ```python\n{ans}\n```",
        f"Here you go: ```\n{ans}\n```",
    ):
        code, defect = _grades(reply)
        assert defect is None, f"{reply[:30]!r}: {defect}"
        assert code.strip() == ans, code


def test_inline_backticks_in_a_sentence_do_not_open_a_block():
    """The other side of that tolerance. A fence run mid-sentence, with prose
    after it, would swallow the paragraph beneath — and the answer with it."""
    ans = "def g(n):\n    return sum(int(c) for c in str(n))"
    code, defect = _grades(f"Use ```code``` inline.\n\nThen:\n\n```python\n{ans}\n```")
    assert defect is None and code.strip() == ans, code
    assert extract_code("You can wrap it in ```fences``` if you like.", "g") == ""


def test_an_indented_fence_and_a_spaced_info_string_are_still_fences():
    ans = "def g(n):\n    return len(str(n))"
    assert _grades(f"  ```python\n{ans}\n  ```")[1] is None
    assert _grades(f"``` python\n{ans}\n```")[1] is None


def test_a_star_import_does_not_hide_a_genuinely_missing_module():
    """Treating `import *` as binding everything looked like the cautious
    choice and cost a carry: a block holding `from collections import *` beside
    a use of `math` reported nothing missing, the `math` split into an earlier
    block was left behind, and every hidden test failed on NameError."""
    code, value = _runs(
        "```python\nimport math\n```\n\n"
        "```python\nfrom collections import *\ndef g(n):\n    return math.isqrt(n)\n```"
    )
    assert value == 4, value
    assert "import math" in code, code


def test_a_star_import_is_never_itself_carried():
    """The other half: `from x import *` binds names this cannot enumerate, so
    it can never be the statement that answers a missing name."""
    from solvers.prompts import _import_bindings

    assert _import_bindings("from collections import *") == {}
    assert _import_bindings("import math\nfrom os import *") == {}


# --------------------------------------------------------------------------- #
# The local rehearsal: one real problem, solved through the miner's own code.
#
# What is under test here is mostly that it does NOT reimplement the miner. A
# rehearsal that solved the problem its own way would agree with the miner
# right up until the day they diverged, and would then report success about
# code nobody runs. So these pin the path: the request is signed, it goes
# through `handle_request`, the answer comes back through `fit_response`, and
# the archive is written by `save_solution` — the same objects a validator's
# request meets.
# --------------------------------------------------------------------------- #
def _rehearsal_args(**kw):
    import argparse

    base = dict(sample="python", source_file=None, lease=False, challenge=None,
                examples=2, timeout=300.0, insecure=False, statement=False, show=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _rehearsal_solver(reply, provider="claude"):
    """A backend that answers with `reply`, so the rehearsal needs no browser."""
    class _Chat:
        def __init__(self):
            self.provider = provider
            self.asked: list[str] = []

        async def send(self, text, timeout_s):
            self.asked.append(text)
            return reply

        async def close(self): pass

    class _Backend:
        def __init__(self): self.chats: list[_Chat] = []
        async def open(self, avoid=None):
            chat = _Chat()
            self.chats.append(chat)
            return chat
        async def aclose(self): pass
        def stats(self): return {}

    backend = _Backend()

    def factory():
        return VerifyingSolver(backend, max_attempts=1, reserve_s=0,
                               max_budget_s=60, second_opinion=False)

    return factory, backend


RIGHT_RUN = """Here you go.

```python
def longest_run(values):
    best = 0
    run = 0
    previous = object()
    for value in values:
        run = run + 1 if run and value == previous else 1
        previous = value
        best = max(best, run)
    return best
```
"""


def test_the_rehearsal_solves_a_real_problem_and_says_it_would_score(tmp_path, capsys):
    from solvers import rehearse

    factory, backend = _rehearsal_solver(RIGHT_RUN)
    code = asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "SCORES: passed all" in out, out
    # The MINER'S prompt reached the model, not one the rehearsal invented.
    # Every stage opens its own conversation, so the candidate turn is found by
    # what it asks for rather than by being first.
    asked = [text for chat in backend.chats for text in chat.asked]
    written = [p for p in asked if _phase_of(p) == "candidate"]
    assert written, f"no candidate turn: {[_phase_of(p) for p in asked]}"
    assert "longest run" in written[0]
    assert "Reply with ONE fenced block" in written[0], "not the miner's own prompt"


def test_the_rehearsal_writes_the_solution_to_a_file(tmp_path, monkeypatch, capsys):
    """The archive is written by the miner's own `save_solution`, so a rehearsal
    leaves the same evidence a live solve does — including the empty file that
    records an answer of silence."""
    from solvers import rehearse

    monkeypatch.setenv("SOLVER_SOLUTION_DIR", str(tmp_path))
    factory, _ = _rehearsal_solver(RIGHT_RUN)
    asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["rehearsal-python-1.json", "rehearsal-python-1.py"], written
    assert "def longest_run" in (tmp_path / "rehearsal-python-1.py").read_text()
    record = json.loads((tmp_path / "rehearsal-python-1.json").read_text())
    assert record["request"]["entrypoint"] == "longest_run"
    assert "def longest_run" in record["response"]["code"]


def test_a_rehearsal_that_answers_with_prose_leaves_an_empty_file(tmp_path, monkeypatch, capsys):
    from solvers import rehearse

    monkeypatch.setenv("SOLVER_SOLUTION_DIR", str(tmp_path))
    factory, _ = _rehearsal_solver("Could you clarify whether the list can nest?")
    code = asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    out = capsys.readouterr().out
    assert code == 1 and "DOES NOT SCORE: nothing was submitted" in out, out
    assert (tmp_path / "rehearsal-python-1.py").read_text() == ""


def test_the_hidden_cases_catch_an_answer_that_passed_every_example(tmp_path, capsys):
    """The reason the samples carry a hidden suite at all. This answer passes
    both public examples, so the miner's own local check reports `verified=True`
    — and it is still a zero, because the statement promises something about the
    empty list that no example shows. That gap is invisible to every check the
    miner has, and it is the commonest shape of a wrong answer."""
    from solvers import rehearse

    skimmed = (
        "```python\n"
        "def longest_run(values):\n"
        "    best = 1\n"
        "    run = 1\n"
        "    for i in range(1, len(values)):\n"
        "        run = run + 1 if values[i] == values[i - 1] else 1\n"
        "        best = max(best, run)\n"
        "    return best\n"
        "```"
    )
    factory, _ = _rehearsal_solver(skimmed)
    code = asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    out = capsys.readouterr().out
    assert "verified=True" in out, "the miner's own check should have been happy"
    assert code == 1, out
    assert "DOES NOT SCORE: passed 7/8" in out, out
    assert "longest_run(*[[]]" in out, "it should name the case that failed"


def test_the_rehearsal_replays_an_archived_request(tmp_path, monkeypatch, capsys):
    """`--from` takes what `save_exchange` writes, so the natural thing to hand
    it is the record of a solve that went wrong."""
    from solvers import rehearse
    from solution_archive import save_exchange

    monkeypatch.setenv("SOLVER_SOLUTION_DIR", str(tmp_path))
    request = TaskRequest(
        problem_id="replayed-1", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[TestCase(args=[12345], kwargs={}, expected=15)],
    )
    record = save_exchange("replayed-1", request.model_dump(mode="json"),
                           {"problem_id": "replayed-1", "code": "", "raw_response": ""},
                           tmp_path)
    factory, backend = _rehearsal_solver(RIGHT)
    code = asyncio.run(
        rehearse.run(_rehearsal_args(source_file=str(record)), solver_factory=factory)
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "problem replayed-1" in out and "SCORES" in out, out
    # An archive has no hidden suite in it, and saying so is the point.
    assert "only the public examples" in out, out


def test_replaying_a_bare_task_request_works_too():
    """A validator's own logs hold the request without the answer beside it."""
    from solvers import rehearse

    path = Path(tempfile.mkdtemp()) / "bare.json"
    path.write_text(TaskRequest(
        problem_id="bare-1", language="python", statement="Return n.",
        entrypoint="g",
    ).model_dump_json())
    problem = rehearse._from_file(str(path))
    assert problem.request.problem_id == "bare-1"
    assert "only the public examples" in problem.tests_are


def test_an_archive_with_extra_fields_is_not_a_validation_error():
    """`TaskRequest` forbids extras, and an archived record has more in it than
    a request. A confusing pydantic error is a poor way to say so."""
    from solvers import rehearse

    path = Path(tempfile.mkdtemp()) / "fat.json"
    path.write_text(json.dumps({
        "problem_id": "fat-1",
        "request": {
            "problem_id": "fat-1", "language": "python", "statement": "Return n.",
            "entrypoint": "g", "public_examples": [], "deadline_s": 90.0,
            "challenge_id": "not part of a TaskRequest", "leased_at": 1.0,
        },
        "response": {"code": "", "raw_response": ""},
    }))
    problem = rehearse._from_file(str(path))
    assert problem.request.problem_id == "fat-1" and problem.request.deadline_s == 90.0


def test_a_rust_answer_that_will_not_build_is_a_failure_not_an_unknown(capsys):
    """A missing Docker daemon means the tests cannot run, which is an unknown.
    A program that will not COMPILE is not an unknown — it is a zero, and
    reporting it as unknown hides the most definite failure there is behind a
    note about the operator's docker socket."""
    _rustc_or_skip()
    from solvers import rehearse

    broken = '```rust\nfn main() {\n    let x: i32 = "not a number";\n    println!("{}", x)\n}\n```'
    factory, _ = _rehearsal_solver(broken)
    code = asyncio.run(rehearse.run(_rehearsal_args(sample="rust"), solver_factory=factory))
    out = capsys.readouterr().out
    assert code == 1, out
    assert "DOES NOT SCORE" in out and "does not compile" in out, out


def test_an_unknown_sample_names_the_ones_that_exist():
    from solvers import rehearse

    with pytest.raises(SystemExit) as raised:
        rehearse._from_sample("cobol")
    assert "python" in str(raised.value) and "rust" in str(raised.value)


def test_every_sample_is_a_valid_request_whose_examples_are_a_strict_subset():
    """The samples only mean something if the hidden suite really is hidden:
    "passed the examples" and "would have scored" have to be able to differ."""
    from solvers.samples import SAMPLES

    for name, sample in SAMPLES.items():
        request = sample.request()
        assert request.entrypoint and request.statement.strip(), name
        assert request.public_examples, f"{name} shows the model nothing"
        assert len(sample.hidden_tests()) > len(request.public_examples), (
            f"{name} has no hidden cases, so it cannot tell a skimmed answer apart"
        )


def test_the_rehearsal_goes_through_the_signed_handler_not_a_shortcut(monkeypatch, capsys):
    """The load-bearing claim of the whole tool. Calling `solve()` directly
    would be simpler and would look identical on a good day — and would stop
    testing the signature check, the replay cache, the concurrency slot and the
    deadline that answers 504 rather than late. Proven by breaking the
    signature: only a path that actually verifies it can reject this."""
    from solvers import rehearse

    monkeypatch.setattr(
        rehearse, "sign_message",
        lambda *a, **kw: {"Epistula-Version": "2", "Epistula-Signed-By": "nobody"},
    )
    factory, _ = _rehearsal_solver(RIGHT_RUN)
    code = asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    out = capsys.readouterr().out
    assert code == 1, out
    assert "answered 401" in out, f"the signature was never checked: {out}"


def test_the_rehearsal_closes_the_fleet_even_when_the_solve_explodes():
    """It opens real browser tabs. Leaving them behind on a failure would be a
    tab per run, in the operator's own signed-in Chrome."""
    from solvers import rehearse

    closed: list[bool] = []

    class _Exploding:
        async def solve_task(self, task, timeout_s):
            raise RuntimeError("the fleet fell over")
        async def aclose(self):
            closed.append(True)
        def stats(self):
            return {"tabs": 1}

    code = asyncio.run(
        rehearse.run(_rehearsal_args(), solver_factory=lambda: _Exploding())
    )
    assert closed == [True], "the fleet was left open"
    # A solve that raises is caught by the miner, which answers with silence.
    assert code == 1


def test_no_browser_is_reported_as_unchecked_not_as_a_wrong_answer(capsys):
    """An operator who has not started Chrome yet is the likeliest person ever
    to run this. The fleet already says what is wrong and how to fix it; a
    traceback on top of that buries the one line worth reading, and calling it
    a failed answer would blame the miner for a browser that is not running."""
    from solvers import rehearse

    class _NoFleet:
        async def solve_task(self, task, timeout_s):
            raise AssertionError("should never get as far as solving")
        async def aclose(self): pass
        def stats(self): return {"tabs": 0}
        async def start(self):
            raise RuntimeError("No usable tabs. Wanted: claude@http://127.0.0.1:9222")

    class _Solver(_NoFleet):
        _backend = None

    solver = _Solver()
    solver._backend = solver          # `warm_up` reaches for `_backend.start`
    code = asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=lambda: solver))
    out = capsys.readouterr().out
    assert code == 2, f"a missing browser is not a wrong answer: {out}"
    assert "COULD NOT BE CHECKED: no backend" in out, out
    assert "No usable tabs" in out, "the fleet's own advice was swallowed"


def test_a_quiet_rehearsal_does_not_call_a_real_answer_empty(capsys):
    """`--show 0` prints no code, which is not the same as there being none.
    Folding the two into one branch put "the answer was EMPTY" directly beneath
    "submitted 197 chars of python"."""
    from solvers import rehearse

    factory, _ = _rehearsal_solver(RIGHT_RUN)
    asyncio.run(rehearse.run(_rehearsal_args(show=0), solver_factory=factory))
    out = capsys.readouterr().out
    assert "submitted" in out and "chars of python" in out, out
    assert "EMPTY" not in out, f"a real answer was announced as empty:\n{out}"

    factory, _ = _rehearsal_solver("I need a clarification before I can answer.")
    asyncio.run(rehearse.run(_rehearsal_args(show=0), solver_factory=factory))
    assert "the answer was EMPTY" in capsys.readouterr().out


def test_the_doctor_explains_a_site_it_cannot_reach(capsys, monkeypatch):
    """Measured by running the doctor behind a network that blocks the site:
    twenty-five lines of Playwright internals ending in
    `net::ERR_CONNECTION_RESET`, with the one useful word buried in the middle.
    Attaching had already succeeded — that part is printed — so what failed is
    reaching the site, and that has causes an operator can act on."""
    import solvers.doctor as doctor

    page = _FakePage({"#composer": [_Node()]})
    page.add_init_script = lambda script: _done(None)

    async def refuse(url, wait_until=None):
        raise RuntimeError(f"net::ERR_CONNECTION_RESET at {url}")

    page.goto = refuse
    site = _site(url="https://example.invalid/new", stream=True)

    class _Browser:
        contexts = [SimpleNamespace(new_page=lambda: _done(page))]
        async def close(self): pass

    monkeypatch.setattr(doctor, "_site", lambda name: site)
    monkeypatch.setattr(doctor, "_attach", lambda pw, s, endpoint: _done(_Browser()))
    monkeypatch.setattr(
        doctor, "import_playwright",
        lambda: lambda: SimpleNamespace(
            start=lambda: _done(SimpleNamespace(stop=lambda: _done(None)))
        ),
    )
    code = asyncio.run(doctor.run("claude", "9222", False))
    out = capsys.readouterr().out
    assert code == 2, out
    assert "could not open" in out and "ERR_CONNECTION_RESET" in out, out
    assert "Traceback" not in out
    assert "proxy or firewall" in out, "the operator was left without a next step"
    assert page.closed, "the doctor's own tab was left open in your browser"


# --------------------------------------------------------------------------- #
# The sample challenges: the closest thing in this repository to what a
# validator actually sends. Five real problems, two Python and three Rust, each
# a page of prose with its edge cases stated rather than shown.
# --------------------------------------------------------------------------- #
def test_every_sample_challenge_loads_as_a_validator_request():
    from solvers.challenges import load_all, names

    found = names()
    assert len(found) == 5, found
    for challenge in load_all():
        assert challenge.language in ("python", "rust"), challenge.name
        assert challenge.entrypoint, challenge.name
        assert len(challenge.statement) > 500, f"{challenge.name} statement looks empty"
        assert challenge.cases, challenge.name
        for case in challenge.cases:
            assert set(case) >= {"args", "kwargs", "expected"}, case


def test_the_model_is_shown_fewer_cases_than_it_is_graded_on():
    """The decision that makes a challenge run mean anything. Show the model all
    three and grade it on all three and the result is circular:
    `VerifyingSolver` repairs until the public examples pass, so the grade can
    only agree with the check already made — it would report a success it was
    incapable of failing."""
    from solvers import rehearse

    problems = rehearse._from_challenges(None, 2, 300.0)
    assert len(problems) == 5
    for problem in problems:
        shown = len(problem.request.public_examples)
        graded = len(problem.tests)
        assert shown < graded, (
            f"{problem.request.problem_id}: shown {shown} of {graded} — "
            f"the grade cannot fail"
        )
        assert "not shown" in problem.tests_are


def test_no_examples_at_all_reproduces_the_run_this_miner_was_built_for():
    """On that run nothing shipped public examples, so the whole repair loop was
    dead code. It is the condition worth measuring against, not a handicap."""
    from solvers import rehearse

    for problem in rehearse._from_challenges(None, 0, 300.0):
        assert problem.request.public_examples == []
        assert len(problem.tests) == 3


def test_showing_every_case_says_the_grade_cannot_fail():
    """If someone asks for it anyway, the report has to admit what it is."""
    from solvers import rehearse

    problem = rehearse._from_challenges(["extent-journal"], 99, 300.0)[0]
    assert len(problem.request.public_examples) == len(problem.tests)
    assert "cannot fail" in problem.tests_are


def test_a_directory_of_archived_requests_replays_every_one(tmp_path):
    """`save_exchange` writes one record per solve, so a directory of them is a
    corpus of exactly what this miner was asked in production — the statements,
    the entrypoints, the deadlines, and the fact that no public examples shipped
    with any of them. Replaying it is the off-chain regression run.

    Sorted by name, so two runs are comparable line for line."""
    import json

    from solvers.rehearse import _from_archive

    for i, language in enumerate(["python", "rust", "python"]):
        (tmp_path / f"{i}-problem.json").write_text(json.dumps({
            "problem_id": f"{i}-problem",
            "request": {
                "problem_id": f"{i}-problem", "language": language,
                "statement": "do a thing", "entrypoint": "g" if language == "python" else "main",
                "public_examples": [], "deadline_s": 300.0,
            },
            # The answer the miner gave LAST time is in the file too, and it is
            # not part of the request. Replaying it would grade the old answer.
            "response": {"problem_id": f"{i}-problem",
                         "code": "def g():\n    return 'THE OLD ANSWER'",
                         "raw_response": "..."},
        }))

    problems = _from_archive(str(tmp_path))
    assert [p.request.problem_id for p in problems] == [
        "0-problem", "1-problem", "2-problem"
    ], "not sorted, so two runs are not comparable"
    assert [p.request.language for p in problems] == ["python", "rust", "python"]
    assert all(not p.request.public_examples for p in problems)

    # ONLY the request. The stored answer never reaches the solve.
    assert not any("THE OLD ANSWER" in p.request.statement for p in problems)

    # A single file still works, and so does a bare TaskRequest.
    one = _from_archive(str(tmp_path / "1-problem.json"))
    assert len(one) == 1 and one[0].request.language == "rust"

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit) as caught:
        _from_archive(str(empty))
    assert "no .json requests" in str(caught.value)


def test_a_corpus_with_no_tests_is_not_reported_as_a_catastrophe(capsys):
    """Every archived request carries zero public examples — that is the live
    condition this miner was built for, and it means nothing in the corpus CAN
    score. Saying "0/97 would have scored" of it reads as a disaster rather than
    as a missing yardstick.

    What a replay does measure without any tests: whether an answer came back at
    all, and whether a Rust answer compiles. An empty answer is the failure this
    miner has most of — of 97 archived solves, 32 submitted nothing."""
    from solvers import rehearse

    def _p(language, tests=()):
        return rehearse.Problem(
            SolveTask(problem_id="p", language=language, statement="s",
                      entrypoint="g", public_examples=[], deadline_s=300.0),
            list(tests), "the archive", "nothing to check it against",
        )

    rehearse._summarise([
        (_p("python"), rehearse.UNKNOWN, "no tests came with this problem"),
        (_p("python"), rehearse.FAILED, "nothing was submitted"),
        (_p("rust"), rehearse.UNKNOWN, "no tests came with this problem"),
        (_p("rust"), rehearse.FAILED, "it does not compile: unclosed delimiter"),
    ])
    out = capsys.readouterr().out
    assert "would have scored" not in out, out
    assert "could be graded against public examples" in out, out
    assert "3/4 produced an answer" in out, out
    assert "(1 submitted nothing)" in out, out
    assert "1/2 rust answer(s) compile" in out, out
    # No solver stats passed, so nothing is claimed about local verification.
    assert "wrote for itself" not in out, out

    # ...and when tests DID come with the problem, the old line is still the
    # headline. A SCORED verdict is only reachable when they did.
    graded = _p("python", tests=[TestCase(args=[1], kwargs={}, expected=1)])
    rehearse._summarise([(graded, rehearse.SCORED, "passed all 3 test(s)")])
    out = capsys.readouterr().out
    assert "1/1 would have scored" in out, out
    assert "1/1 produced an answer" in out, out


def test_a_replay_with_no_tests_still_reports_what_the_solver_could_check(capsys):
    """The yardstick a corpus with no public examples DOES have.

    "none of the 97 could be graded against public examples" is true and, on its
    own, was the whole verdict a replay reached -- which cannot tell an answer
    that ran every case the model wrote and passed from one that was never run.
    That distinction is the entire question a replay exists to answer, and the
    solver already knows it: `VerifyingSolver` runs the model's own cases with
    the validator's executor and counts the clean ones.

    Reported apart from "would have scored" and never merged into it. A model
    agreeing with itself is weaker evidence than a public example, and the line
    says so rather than letting the number be read as a score."""
    from solvers import rehearse

    problems = [
        rehearse.Problem(
            SolveTask(problem_id=f"p{i}", language="python", statement="s",
                      entrypoint="g", public_examples=[], deadline_s=300.0),
            [], "the archive", "nothing to check it against",
        )
        for i in range(3)
    ]
    results = [(p, rehearse.UNKNOWN, "no tests came with this problem")
               for p in problems]

    rehearse._summarise(results, {"solver": {"verified_on_local": 2, "empty": 0}})
    out = capsys.readouterr().out
    assert "2/3 verified on local" in out, out
    assert "passed every case the model wrote for itself" in out, out
    assert "weaker than a public example" in out, out
    assert "would have scored" not in out, "a self-check was reported as a score"

    # A solver that checked nothing claims nothing -- no line at all, rather
    # than a zero that reads as a failure.
    rehearse._summarise(results, {"solver": {"verified_on_local": 0}})
    assert "wrote for itself" not in capsys.readouterr().out

    # ...and a stats() that is missing, broken or shaped differently is a
    # missing line, never a crashed replay.
    class _Broken:
        def stats(self): raise RuntimeError("no")

    assert rehearse._solver_stats(_Broken()) == {}
    assert rehearse._solver_stats(SimpleNamespace(stats=lambda: None)) == {}
    rehearse._summarise(results, rehearse._solver_stats(_Broken()))
    assert "wrote for itself" not in capsys.readouterr().out


def test_your_own_problems_directory_wins_over_the_shipped_samples(tmp_path):
    """`examples/problems` is where an operator drops their own problems, and it
    is found with no flag and no environment variable. The shipped samples are
    the fallback so a fresh checkout still has something to run."""
    from solvers.challenges import challenge_dir, names

    root = tmp_path / "repo"
    mine = root / "examples" / "problems" / "my-problem"
    shipped = root / "examples" / "sample_challenges" / "extent-journal"
    for d in (mine, shipped):
        d.mkdir(parents=True)
        (d / "PROBLEM.md").write_text("statement")
        (d / "cases.json").write_text(
            '{"language": "python", "entrypoint": "g", "cases": []}'
        )
    start = root / "examples" / "custom_miner" / "solvers"
    start.mkdir(parents=True)

    found = challenge_dir(start)
    assert found == root / "examples" / "problems", found
    assert names(found) == ["my-problem"]


def test_an_empty_problems_directory_does_not_shadow_the_samples(tmp_path):
    """A directory created and not yet filled must not silently take over and
    report "(none found)" — which is what a plain `is_dir()` test would do, and
    the directory ships with only a README in it."""
    from solvers.challenges import challenge_dir

    root = tmp_path / "repo"
    (root / "examples" / "problems").mkdir(parents=True)
    (root / "examples" / "problems" / "README.md").write_text("drop them here")
    shipped = root / "examples" / "sample_challenges" / "extent-journal"
    shipped.mkdir(parents=True)
    (shipped / "PROBLEM.md").write_text("statement")
    (shipped / "cases.json").write_text(
        '{"language": "python", "entrypoint": "g", "cases": []}'
    )
    start = root / "examples" / "custom_miner" / "solvers"
    start.mkdir(parents=True)

    assert challenge_dir(start) == root / "examples" / "sample_challenges"


def test_a_local_run_can_be_told_where_to_archive_and_logged_verbatim(
    tmp_path, capsys, monkeypatch
):
    """The two things a local run has to leave behind: the answers where the
    operator asked for them, and the output — the SAME lines the on-chain miner
    prints, because it is the same code printing them.

    `SOLVER_SOLUTION_DIR` is relative to the working directory and this package
    runs from `examples/custom_miner`, so the default lands beside the miner
    rather than at the repository root. An operator went looking in the wrong
    one; `--solutions` settles it."""
    import os

    from solvers import rehearse

    log = tmp_path / "runs" / "local.log"          # a directory that must be made
    with rehearse._tee(str(log)):
        print("[verify] python entrypoint=g provider=claude examples=0/0")
        print("[rehearse] DOES NOT SCORE: nothing was submitted")

    written = log.read_text()
    assert "[verify] python entrypoint=g" in written, written
    assert "DOES NOT SCORE" in written, written
    assert str(log.resolve()) in written, "the log does not say where it is"
    # ...and the terminal still had it live. A log that only exists afterwards
    # is no use while a run is going wrong.
    assert "DOES NOT SCORE" in capsys.readouterr().out

    # No --log, no file, and nothing swallowed.
    with rehearse._tee(None):
        print("[rehearse] still on the terminal")
    assert "still on the terminal" in capsys.readouterr().out

    # `--solutions` is the archive directory, set before the solve reaches it.
    where = tmp_path / "answers"
    # Scoped: `archive_to` sets a process-wide environment variable, and every
    # other test that archives reads it.
    monkeypatch.setenv("SOLVER_SOLUTION_DIR", "unset")
    rehearse.archive_to(str(where))
    assert os.environ["SOLVER_SOLUTION_DIR"] == str(where)
    from solution_archive import archive_dir
    assert archive_dir() == where

    # ...and no --solutions leaves it exactly where it was.
    rehearse.archive_to(None)
    assert os.environ["SOLVER_SOLUTION_DIR"] == str(where)


def test_a_challenge_name_cannot_read_outside_the_challenge_directory(tmp_path):
    """`name` arrives from the command line and becomes a path.

    The escape target has to be a REAL challenge, or the "is there a cases.json
    there" check catches it on its own and the guard under test never runs.
    So: one valid challenge inside the directory, an identical one just outside
    it, and `../elsewhere` must not reach the second."""
    from solvers.challenges import load

    root = tmp_path / "challenges"
    (root / "inside").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    manifest = json.dumps({
        "language": "python", "entrypoint": "f",
        "cases": [{"name": "c", "args": [], "kwargs": {}, "expected": 1}],
    })
    for directory in (root / "inside", outside):
        (directory / "cases.json").write_text(manifest)
        (directory / "PROBLEM.md").write_text("# t\n\nDo the thing.")

    assert load("inside", root).entrypoint == "f", "the ordinary case broke"
    for hostile in ("../elsewhere", "inside/../../elsewhere", ".."):
        with pytest.raises(SystemExit):
            load(hostile, root)


def test_an_unknown_challenge_names_the_ones_that_exist():
    from solvers.challenges import load

    with pytest.raises(SystemExit) as raised:
        load("no-such-challenge")
    assert "sparse-circular-array" in str(raised.value)


def test_challenges_run_as_a_batch_and_end_in_one_table(capsys):
    """A run over five challenges prints several hundred lines and takes long
    enough that nobody watches it. Which of them scored must not require
    scrolling back through four other answers."""
    from solvers import rehearse

    factory, backend = _rehearsal_solver("```python\ndef f():\n    return 1\n```")
    code = asyncio.run(
        rehearse.run(
            _rehearsal_args(challenge=["all"], examples=2, timeout=30.0),
            solver_factory=factory,
        )
    )
    out = capsys.readouterr().out
    assert "[rehearse] summary" in out, out
    for name in ("asset-rebuild-planner", "extent-journal", "sparse-circular-array"):
        assert name in out, f"{name} missing from the run"
    assert "0/5 would have scored" in out, out
    assert code != 0
    # One conversation per PHASE now, not one per challenge, so the count is
    # a multiple of the five challenges rather than five.
    assert len(backend.chats) >= 5, "not every challenge was attempted"
    assert len(backend.chats) % 5 == 0, (
        f"{len(backend.chats)} conversations over 5 challenges is not a whole "
        f"number of phases each"
    )
    # Every summary row on one line. A failure detail runs to several hundred
    # characters, and five of those wrapped is the scrollback the table was
    # added to replace.
    table = out[out.index("[rehearse] summary"):]
    rows = [line for line in table.splitlines() if line.startswith(("  PASS", "  FAIL", "  ????"))]
    assert len(rows) == 5, table
    over = [line for line in rows if len(line) > 130]
    assert not over, f"summary row too long to be a table: {over[0][:160]}..."


def test_the_summary_stays_a_table():
    """A failure detail runs to several hundred characters — it names the
    arguments, what came back and what was wanted. Five of those wrapped is the
    scrollback the table was added to replace."""
    from solvers import rehearse

    long_why = "passed 0/3 — " + "x" * 400
    assert len(rehearse._fit(long_why, 96)) == 96
    assert rehearse._fit("short", 96) == "short"
    assert rehearse._fit("a\n  b   c", 96) == "a b c"


def test_the_local_check_can_be_turned_off_and_then_nothing_is_asked_for_one():
    """`SOLVER_SELF_TESTS=0` is a shipped configuration and it has to mean
    something.

    It was a constructor argument the solver stored and never read, so an
    operator who set it paid for the inputs and reference turns anyway and got
    a differential they had asked not to have. Off means off: neither turn is
    opened, nothing is compared, no repair round fires, and the line says so
    rather than leaving `ran=0` to be read as a failure."""
    # 12345: the reference sums every digit and gets 15; `WRONG` stops at the
    # leading digit and gets 14. They have to disagree or there is no repair
    # round for the ON case to find.
    inputs = '```json\n[{"name": "carry", "args": [12345]}]\n```'
    reference = "```python\ndef g(n):\n    return sum(int(d) for d in str(n))\n```"
    asked: list[str] = []

    class _Counting(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            phase = _phase_of(text)
            asked.append(phase)
            if phase == "inputs":
                return inputs
            if phase == "oracle":
                return reference
            if phase == "analysis":
                return ""
            return WRONG          # disagrees with the reference on every input

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Counting(self._script, self._provider)

    task = SolveTask(problem_id="off", language="python",
                     statement="Return the sum of the decimal digits of n.",
                     entrypoint="g", public_examples=[], deadline_s=60.0)

    def solve(self_tests):
        backend = _Backend2([])
        solver = VerifyingSolver(backend, reserve_s=0, max_budget_s=60,
                                 second_opinion=False, self_tests=self_tests)
        asked.clear()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            answer = asyncio.run(solver.solve_task(task, timeout_s=60.0))
        return answer, log.getvalue(), list(asked)

    # ON: the two turns run, the disagreement is found, a repair is sent.
    answer, out, phases = solve(True)
    assert "inputs" in phases and "oracle" in phases, phases
    assert "repair" in phases, phases
    assert "mismatch=1" in out, out

    # OFF: neither turn is opened at all, and nothing is repaired.
    answer, out, phases = solve(False)
    assert "inputs" not in phases, f"asked for inputs with the check off: {phases}"
    assert "oracle" not in phases, f"asked for a reference with the check off: {phases}"
    assert "repair" not in phases, f"repaired against nothing: {phases}"
    assert "the local check is off" in out, out
    # ...and the candidate still ships. Off is a cheaper solve, not a lost one.
    assert answer.code.strip(), out
    assert answer.verified is False and answer.self_verified is False


def test_the_summary_line_still_parses_with_the_calibration_regex():
    """`calibration/bar_ab.py` reads the `[verify]` and `[phase]` lines, and it
    is the ONLY consumer of their format.

    Its failure mode is silence: a regex that no longer matches returns zero
    rows rather than raising, so a summary line that grew a field would take
    the comparison tool with it and nobody would find out. `rounds=` and
    `corrected=` were kept through the rebuild for exactly this, and this test
    is what says so out loud — run against a line a real solve just printed,
    not against one written here by hand.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bar_ab_under_test", "calibration/bar_ab.py"
    )
    bar_ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bar_ab)

    inputs = ('```json\n[{"name": "zero", "args": [0]},\n'
              ' {"name": "carry", "args": [12345]}]\n```')
    reference = "```python\ndef g(n):\n    return sum(int(d) for d in str(n))\n```"

    class _Differential(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            phase = _phase_of(text)
            if phase == "inputs":
                return inputs
            if phase == "oracle":
                return reference
            if phase == "analysis":
                return ""
            return RIGHT

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Differential(self._script, self._provider)

    task = SolveTask(problem_id="ab", language="python",
                     statement="Return the sum of the decimal digits of n.",
                     entrypoint="g", public_examples=[], deadline_s=60.0)
    solver = VerifyingSolver(_Backend2([]), reserve_s=0, max_budget_s=60,
                             second_opinion=False)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        asyncio.run(solver.solve_task(task, timeout_s=60.0))
    out = log.getvalue()

    summary = [l for l in out.splitlines() if l.startswith("[verify] python entrypoint=")]
    assert summary, out
    match = bar_ab.DONE.search(summary[0])
    assert match, f"bar_ab would read zero rows off:\n{summary[0]}"
    self_passed, self_total, rounds, elapsed = match.groups()
    assert (self_passed, self_total) == ("2", "2"), match.groups()
    assert int(rounds) >= 1 and float(elapsed) >= 0.0, match.groups()

    phases = [l for l in out.splitlines() if l.startswith("[phase] ")]
    assert phases and all(bar_ab.PHASE.search(l) for l in phases), phases


def test_a_mixed_batch_reports_the_worst_outcome_in_it(capsys, monkeypatch):
    """A run with one wrong answer in it is not a passing run."""
    from solvers import rehearse

    verdicts = iter([
        (rehearse.SCORED, "passed all 3 test(s)"),
        (rehearse.UNKNOWN, "the tests could not be run here"),
        (rehearse.FAILED, "passed 1/3"),
    ])
    monkeypatch.setattr(rehearse, "_verdict", lambda *a: next(verdicts))
    factory, _ = _rehearsal_solver("```python\ndef f():\n    return 1\n```")
    code = asyncio.run(
        rehearse.run(
            _rehearsal_args(
                challenge=["asset-rebuild-planner", "extent-journal",
                           "sparse-circular-array"],
                examples=2, timeout=30.0,
            ),
            solver_factory=factory,
        )
    )
    assert code == 1, capsys.readouterr().out

    # ...and with nothing failed, an unknown still outranks a pass.
    verdicts = iter([(rehearse.SCORED, "ok"), (rehearse.UNKNOWN, "no docker")])
    monkeypatch.setattr(rehearse, "_verdict", lambda *a: next(verdicts))
    factory, _ = _rehearsal_solver("```python\ndef f():\n    return 1\n```")
    code = asyncio.run(
        rehearse.run(
            _rehearsal_args(
                challenge=["asset-rebuild-planner", "extent-journal"],
                examples=2, timeout=30.0,
            ),
            solver_factory=factory,
        )
    )
    assert code == 2, capsys.readouterr().out


def test_the_fleet_is_opened_once_for_a_whole_batch():
    """Opening browsers per challenge would spend a minute of page loads five
    times over, and the tabs are designed to be reused — that is what a miner
    does for its whole life."""
    from solvers import rehearse

    closes: list[int] = []
    factory, backend = _rehearsal_solver("```python\ndef f():\n    return 1\n```")

    def counting_factory():
        solver = factory()
        original = solver.aclose

        async def aclose():
            closes.append(1)
            await original()

        solver.aclose = aclose
        return solver

    asyncio.run(
        rehearse.run(
            _rehearsal_args(challenge=["all"], examples=2, timeout=30.0),
            solver_factory=counting_factory,
        )
    )
    assert closes == [1], f"the fleet was closed {len(closes)} times, not once"


def test_an_ungradeable_answer_does_not_buy_a_second_opinion(capsys):
    """Measured on a live miner with no Docker daemon: all three Rust
    challenges asked a SECOND model — a full extra solve each, 55 to 108
    seconds and a second conversation off the account quota — and then
    submitted the FIRST model's answer anyway, because two ungradeable
    candidates tie at `score` and `>` loses a tie.

    The old condition was `not task.public_examples`, which missed this
    entirely: the examples were shipped, they just could not be run."""
    asked: list[str] = []

    class _Backend:
        async def open(self, avoid=None):
            asked.append(avoid or "first")
            return _Chat([RIGHT], provider="m2" if avoid else "m1")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, reserve_s=0,
                             max_budget_s=60, second_opinion=True)

    def unavailable(*a, **kw):
        raise RuntimeError("DockerExecutor requires the 'docker' CLI on PATH")

    solver._grader.check = unavailable
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    out = capsys.readouterr().out
    assert answer.code.strip(), "the answer itself was lost"
    # Counted as DISTINCT `avoid` values rather than as `open()` calls: a
    # pass now opens one conversation per phase, and what this is about is how
    # many PASSES ran -- a second pass is the one that passes `avoid`.
    assert len(set(asked)) == 1, f"asked {set(asked)} for an ungradeable task"
    assert "could not be run here" in out, out


def test_an_empty_ungradeable_answer_still_buys_a_second_opinion():
    """The other half. An empty answer scores zero, so the other model is the
    only remaining chance at the whole payment — that trade is still worth it
    when nothing can be graded."""
    asked: list[str] = []

    class _Backend:
        async def open(self, avoid=None):
            asked.append(avoid or "first")
            # First model says nothing; the second answers.
            second = avoid is not None
            return _Chat([RIGHT] if second else ["I cannot help."],
                         provider="m2" if second else "m1")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, reserve_s=0,
                             max_budget_s=60, second_opinion=True)
    solver._grader.check = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("no docker")
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    # Counted as DISTINCT `avoid` values rather than as `open()` calls: a
    # pass now opens one conversation per phase, and what this is about is how
    # many PASSES ran -- a second pass is the one that passes `avoid`.
    assert len(set(asked)) == 2, "an empty answer must still get a second chance"
    assert "def g(n)" in answer.code, answer.code


def test_a_gradeable_failure_still_buys_a_second_opinion():
    """Only 'nothing ran' skips it. An answer that ran and failed some examples
    IS rankable, so the other model is worth asking."""
    asked: list[str] = []

    class _Backend:
        async def open(self, avoid=None):
            asked.append(avoid or "first")
            return _Chat([WRONG], provider="m2" if avoid else "m1")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, reserve_s=0,
                             max_budget_s=60, second_opinion=True)
    asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    # Counted as DISTINCT `avoid` values rather than as `open()` calls: a
    # pass now opens one conversation per phase, and what this is about is how
    # many PASSES ran -- a second pass is the one that passes `avoid`.
    assert len(set(asked)) == 2, "a rankable failure should still ask the other model"


def test_a_solve_that_used_its_whole_budget_does_not_promise_another_model(capsys):
    """Both halves of a real log, from a Rust solve that submitted nothing.

        [verify] claude was still writing when the budget ran out; nothing
                 arrived to submit rather than interrupting it with a repair
                 prompt
        [verify] no public examples shipped with this task, and the model sent
                 no usable cases of its own either, so nothing can be graded
                 locally...
        [verify] claude returned nothing; asking another model
        [verify] -0s left; the deadline is gone, submitting empty

    Two of those four lines are false, and both point the reader away from what
    actually went wrong -- a program turn that spent the entire 285s budget with
    the model still writing.

    "asking another model" was decided by `attempt_no < passes` alone, which
    knows nothing about the clock; the loop head then re-decided it WITH the
    clock and asked nobody. `_next_pass_blocked_by` is now the single answer
    both lines read.

    "the model sent no usable cases of its own either" was decided by
    `from_self_tests`, which `_run_self_tests` only ever sets when there was
    code to run the cases against. The cases turn here worked perfectly -- it
    is the reply BELOW it that never arrived -- so an empty answer always
    libelled turn 1 for turn 2's failure.
    """
    burned: list[str] = []

    class _StillWriting(_Chat):
        still_writing = False

        async def send(self, text, timeout_s):
            burned.append(text)
            reply = await super().send(text, timeout_s)
            if len(burned) >= 2:
                # The program turn: the model is thinking, and it is still
                # thinking when the budget is gone.
                self.still_writing = True
                self.empty_reason = "unfinished"
                await asyncio.sleep(4.0)
                return ""
            return reply

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _StillWriting(self._script, self._provider)

    task = SolveTask(
        problem_id="burned", language="python",
        statement="Return the sum of the decimal digits of n.", entrypoint="g",
        public_examples=[], deadline_s=3.0,
    )
    # `CASES` first, so turn 1 genuinely produces usable cases -- the whole
    # point of the second assertion below.
    solver = VerifyingSolver(_Backend2([CASES, ""]), reserve_s=0, max_budget_s=3,
                             second_opinion=True)
    answer = asyncio.run(solver.solve_task(task, timeout_s=3.0))
    out = capsys.readouterr().out

    assert not answer.code.strip(), "this reproduction is meant to submit nothing"
    # One turn per phase now, and the point stands: the budget was spent on
    # turns that produced nothing, and no second model was promised for it.
    assert len(burned) >= 2, f"expected the pass to have spent turns: {len(burned)}"
    assert "still writing when the budget ran out" in out, out
    # 1. Nobody was asked, so nobody may be promised.
    assert "asking another model" not in out, (
        "promised a second model with the budget already spent:\n" + out
    )
    assert "the deadline is gone, submitting empty" in out, out
    # 2. Turn 1 delivered cases. The failure was turn 2's.
    assert "no usable cases of its own" not in out, (
        "blamed the cases turn for the program turn's failure:\n" + out
    )
    # 3. A budget that is a hair past spent reads as spent, not as negative.
    assert "-0s left" not in out, out


def _grace_miner(sleep_s, *, settings=None):
    """A miner whose solve takes exactly `sleep_s`, through the real handler."""
    from custom_miner import SolveResult

    class _Slow:
        async def solve_task(self, task, timeout_s):
            await asyncio.sleep(sleep_s)
            return SolveResult(code="def g(n): return n", raw_response="r")

        async def aclose(self): pass

    return CustomMiner(settings or DemoMinerSettings(_env_file=None), _Slow(),
                       wallet=None, subtensor=None, metagraph=None)


def _signed(deadline_s, problem_id="grace"):
    from rlvr.protocol import sign_message as _sign

    request = TaskRequest(problem_id=problem_id, language="python", statement="s",
                          entrypoint="g", public_examples=[], deadline_s=deadline_s)
    body = request.model_dump_json().encode()
    return _sign(keypair.create_from_uri("//Alice"), body), body


def test_the_solve_runs_past_the_deadline_into_the_validators_grace(monkeypatch):
    """The validator does not stop at `deadline_s`.

    `_dispatch_committed` bounds the whole exchange at
    `deadline_s + _MINER_RESPONSE_GRACE_S` — 10.0 in
    `rlvr/neurons/decentralized.py` — and the reference `handle_request`
    cancelled our own solve at `deadline_s` flat, answering 504 with nothing
    while the validator was still listening. Seconds given back for free, and
    there is nothing on the other side to fear for using them:
    `rlvr/scoring/payment.py` gates on `all_passed` and then applies a speed
    multiplier floored at 0.95, so late-but-correct is worth ~96% and
    unfinished is worth 0.

    Not all ten seconds — see `RESPONSE_GRACE_S`. This asserts we take the
    part we decided was safe: a solve that runs PAST the deadline still
    returns 200 with its answer."""
    import custom_miner

    monkeypatch.setattr(custom_miner, "RESPONSE_GRACE_S", 3.0)
    miner = _grace_miner(1.5)
    headers, body = _signed(deadline_s=1.0)     # solve overruns it by 0.5s
    status, payload = asyncio.run(miner.handle_request(headers, body))

    assert status == 200, f"504'd inside the validator's own grace: {payload}"
    assert payload.code == "def g(n): return n"


def test_the_grace_is_bounded_and_an_overrun_still_ends_in_504(monkeypatch):
    """The grace widens the cutoff; it does not remove it. Past
    `deadline_s + RESPONSE_GRACE_S` the validator is about to stop reading, and
    a solve still running then is worth nothing either way — so it ends the
    same way the reference miner ends it."""
    import custom_miner

    monkeypatch.setattr(custom_miner, "RESPONSE_GRACE_S", 1.0)
    miner = _grace_miner(5.0)
    headers, body = _signed(deadline_s=1.0)     # cutoff at 2.0s, solve wants 5.0
    status, payload = asyncio.run(miner.handle_request(headers, body))

    assert status == 504, f"ran past the validator's ceiling: {status}"


def test_an_operator_asking_for_a_shorter_solve_is_not_overruled(monkeypatch):
    """`min` with `glm_request_timeout_s` is kept. Someone who turns that knob
    DOWN is asking for a shorter solve, and a grace that ignored it would make
    the knob a lie."""
    import custom_miner

    monkeypatch.setattr(custom_miner, "RESPONSE_GRACE_S", 30.0)
    settings = DemoMinerSettings(_env_file=None, glm_request_timeout_s=1.0)
    miner = _grace_miner(3.0, settings=settings)
    headers, body = _signed(deadline_s=60.0)
    status, _ = asyncio.run(miner.handle_request(headers, body))

    assert status == 504, "the grace overruled an explicit, lower cap"


def test_zero_grace_is_exactly_the_reference_behaviour(monkeypatch):
    """`MINER_RESPONSE_GRACE_S=0` puts the cutoff back on `deadline_s`, so an
    operator who wants the upstream behaviour can have it without editing
    code."""
    import custom_miner

    monkeypatch.setattr(custom_miner, "RESPONSE_GRACE_S", 0.0)
    miner = _grace_miner(1.5)
    headers, body = _signed(deadline_s=1.0)
    status, _ = asyncio.run(miner.handle_request(headers, body))

    assert status == 504


def test_the_overridden_handler_rejects_exactly_what_the_reference_rejects():
    """The override copies four checks, and a copy drifts.

    `handle_request` is where the wire contract is enforced — signature, replay
    nonce, signer authorization, request shape — and the only reason to
    override it is one number in the last line. So every rejection path is
    driven through BOTH implementations and required to answer identically. If
    upstream tightens a check and this copy does not, this test says so rather
    than a validator silently accepting something it should not."""
    from rlvr.neurons.demo_miner import DemoMiner
    from rlvr.protocol import sign_message as _sign

    def _pair():
        # Two miners over the same solver, differing only in which
        # handle_request runs. Fresh each time: `nonces` is stateful.
        from custom_miner import SolveResult

        class _Instant:
            async def solve_task(self, task, timeout_s):
                return SolveResult(code="def g(n): return n", raw_response="r")

            async def aclose(self): pass

        settings = DemoMinerSettings(_env_file=None)
        mine = CustomMiner(settings, _Instant(), wallet=None, subtensor=None,
                           metagraph=None)
        theirs = DemoMiner(settings, client=_Instant(), wallet=None,
                           subtensor=None, metagraph=None)
        theirs.solve = mine.solve          # same answer, different gate
        return mine, theirs

    good_headers, good_body = _signed(deadline_s=5.0, problem_id="diff")

    cases = {
        "unsigned": ({}, good_body),
        "tampered body": (good_headers, good_body + b" "),
        "bad signature": ({**good_headers, "Epistula-Request-Signature": "0xdead"},
                          good_body),
        "stale timestamp": ({**good_headers, "Epistula-Timestamp": "1"}, good_body),
        "unparseable body": _signed_over(b'{"not": "a task"}'),
        "accepted": (good_headers, good_body),
    }
    for name, (headers, body) in cases.items():
        mine, theirs = _pair()
        ours = asyncio.run(mine.handle_request(dict(headers), body))[0]
        ref = asyncio.run(theirs.handle_request(dict(headers), body))[0]
        assert ours == ref, f"{name}: override said {ours}, reference said {ref}"

    # ...and the replay check, which needs the SAME miner asked twice.
    mine, theirs = _pair()
    assert asyncio.run(mine.handle_request(dict(good_headers), good_body))[0] == 200
    assert asyncio.run(mine.handle_request(dict(good_headers), good_body))[0] == 409
    assert asyncio.run(theirs.handle_request(dict(good_headers), good_body))[0] == 200
    assert asyncio.run(theirs.handle_request(dict(good_headers), good_body))[0] == 409


def _signed_over(body: bytes):
    from rlvr.protocol import sign_message as _sign

    return _sign(keypair.create_from_uri("//Alice"), body), body


def test_every_entry_point_gives_a_solve_the_same_budget(monkeypatch, capsys):
    """A replay is only evidence if it runs the solve production runs.

    `handle_request` bounds the whole solve at
    `min(deadline_s, glm_request_timeout_s)`, and `DemoMinerSettings` defaults
    that to 280 -- below the 300s this subnet advertises.
    `apply_solve_timeout_default` fills in 3600 instead, which
    `TaskRequest.deadline_s`'s own `le=3600` makes structurally incapable of
    binding, so the request's deadline is the only deadline.

    It was called by `custom_miner.run_custom_miner` and by `rehearse`, and NOT
    by `run_miner.py` -- the on-chain entry point. With nothing in the
    environment the replay therefore ran every solve at 300s of a 300s deadline
    while the miner it was rehearsing ran at 280: twenty seconds of divergence,
    in the direction that flatters the replay. `config.py`'s own docstring said
    the opposite ("The miner and the rehearsal both call it").

    `load_miner_env` is the fix and this is the test that it stays fixed: the
    steps are in one function now, so a fourth entry point cannot lose one."""
    from solvers import config

    monkeypatch.delenv("GLM_REQUEST_TIMEOUT_S", raising=False)
    config.load_miner_env("miner")
    assert os.environ["GLM_REQUEST_TIMEOUT_S"] == config.DEFAULT_SOLVE_TIMEOUT_S

    settings = DemoMinerSettings(_env_file=None)
    assert settings.glm_request_timeout_s == float(config.DEFAULT_SOLVE_TIMEOUT_S)
    # The number that actually matters: what a 300s request gets to spend.
    assert min(300.0, settings.glm_request_timeout_s) == 300.0, (
        "a spec-compliant deadline was cut short by the miner's own default"
    )

    # Every entry point calls it. Read from the source rather than trusted,
    # because the whole failure was one caller quietly not doing so.
    here = Path(__file__).resolve().parent
    for name in ("run_miner.py", "custom_miner.py", "solvers/rehearse.py"):
        text = (here / name).read_text(encoding="utf-8")
        assert "load_miner_env(" in text, f"{name} builds its settings by hand"
        assert "apply_solve_timeout_default()" not in text, (
            f"{name} still open-codes a step of the sequence"
        )


def test_an_operators_own_solve_timeout_is_never_overwritten(monkeypatch):
    """`setdefault`, not assignment. The default exists to stop a bare
    environment costing 20s a solve, not to overrule someone who chose a value
    -- and a knob that ignores what you set it to is worse than no knob."""
    from solvers import config

    monkeypatch.setenv("GLM_REQUEST_TIMEOUT_S", "120")
    config.load_miner_env("miner")
    assert os.environ["GLM_REQUEST_TIMEOUT_S"] == "120"
    assert DemoMinerSettings(_env_file=None).glm_request_timeout_s == 120.0


def test_the_archive_line_names_a_directory_you_can_find(tmp_path, monkeypatch, capsys):
    """`SOLVER_SOLUTION_DIR` defaults to the relative "solutions", so the line
    read "archived under solutions/" and left the reader to work out which
    directory that was relative to — and the repository has a `solutions/` at
    its root as well as the one this creates beside the miner. An operator went
    looking in the wrong one."""
    from solvers import rehearse

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOLVER_SOLUTION_DIR", "solutions")
    factory, _ = _rehearsal_solver(RIGHT_RUN)
    asyncio.run(rehearse.run(_rehearsal_args(), solver_factory=factory))
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "archived under" in l)
    assert str(tmp_path / "solutions") in line, line
    assert (tmp_path / "solutions" / "rehearsal-python-1.py").is_file()


def test_a_failing_case_is_named_and_says_whether_it_was_held_back(capsys):
    """A wall of arguments has to be decoded before the reader knows which of
    three behaviours broke. And "(held back)" is the fact worth having in front
    of them: a model that fails a case it was SHOWN has ignored its own worked
    example, one that fails a case it never saw has simply never exercised that
    path. Opposite diagnoses, and the arguments alone say neither."""
    from solvers import rehearse

    # Rejects everything: fails all three, two of them shown.
    rejects = ("```python\ndef extent_journal(chunk_size, address_limit, "
               "large_threshold, events):\n    return [('rejected',) for _ in events]\n```")
    factory, _ = _rehearsal_solver(rejects)
    asyncio.run(
        rehearse.run(
            _rehearsal_args(challenge=["extent-journal"], examples=2, timeout=30.0),
            solver_factory=factory,
        )
    )
    out = capsys.readouterr().out
    assert "case 1 'claim lookup and release'" in out, out
    assert "(held back)" not in out.split("case 3")[0], "a shown case was labelled held back"


def test_only_the_cases_beyond_the_shown_ones_are_marked_held_back():
    from solvers import rehearse
    from solvers.verify import _Grader

    seen: dict = {}
    original = _Grader.check

    def spy(self, code, language, entrypoint, examples, names=None):
        seen["names"] = names
        return original(self, code, language, entrypoint, examples, names)

    _Grader.check = spy
    try:
        problem = rehearse._from_challenges(["extent-journal"], 2, 60.0)[0]
        payload = SimpleNamespace(code="def extent_journal(*a):\n    return []")
        rehearse._verdict(payload, problem.request, problem.tests,
                          problem.case_names, len(problem.request.public_examples))
    finally:
        _Grader.check = original
    names = seen["names"]
    assert len(names) == 3, names
    assert "(held back)" not in names[0] and "(held back)" not in names[1], names
    assert names[2].endswith("(held back)"), names[2]


def test_grading_without_names_is_unchanged():
    """The repair prompt does not want them: the model is being shown concrete
    inputs and outputs, and an authored title is noise there."""
    from solvers.verify import _Grader

    passed, total, failures, _ = _Grader().check(
        "def g(n):\n    return 0", "python", "g",
        [{"args": [12345], "kwargs": {}, "expected": 15}],
    )
    assert (passed, total) == (0, 1)
    assert failures and not failures[0].startswith("case "), failures[0]


def test_a_docker_daemon_you_cannot_reach_is_told_how_to_reach_it():
    """"permission denied ... /var/run/docker.sock" is not a broken install and
    not a stopped daemon — it is a running daemon the miner's user is not in
    the group for. The raw error names the socket and never names the group, so
    an operator reads it as "Docker is broken" and reinstalls Docker."""
    from solvers import rehearse

    denied = rehearse._executor_hint(RuntimeError(
        "DockerExecutor could not contact the Docker daemon ('docker info' rc=1: "
        "permission denied while trying to connect to the docker API at "
        "unix:///var/run/docker.sock)"
    ))
    assert "usermod -aG docker" in denied and "newgrp docker" in denied, denied

    missing = rehearse._executor_hint(RuntimeError(
        "DockerExecutor requires the 'docker' CLI on PATH, but it was not found."
    ))
    assert "not installed" in missing, missing

    assert rehearse._executor_hint(RuntimeError("the grader exploded")) == ""


def test_an_unpulled_rust_image_is_not_reported_as_a_wrong_answer(monkeypatch):
    """`docker run` pulls a missing image, and that pull happens INSIDE the
    executor's own ~80s budget (compile 60s + cases + slack). A first pull of a
    Rust image is several hundred megabytes; over budget the run is killed,
    every case comes back failed, and an unfinished download is reported as
    `passed 0/3` — a wrong answer. That is the one thing a tool built to say
    "would this have scored" must never get backwards."""
    from solvers import rehearse
    from rlvr.policy import RELEASE_POLICY

    monkeypatch.setattr(rehearse, "_compile_defect_if_possible", lambda code: None)
    monkeypatch.setattr(
        rehearse, "_rust_sandbox_missing", lambda: RELEASE_POLICY.rust_image
    )
    request = TaskRequest(
        problem_id="r", language="rust", statement="Do it.", entrypoint="main",
    )
    verdict, why = rehearse._verdict(
        SimpleNamespace(code="fn main() {}"), request,
        [TestCase(args=["1\n"], kwargs={}, expected="1\n")],
    )
    assert verdict == rehearse.UNKNOWN, (verdict, why)
    assert "docker pull" in why and RELEASE_POLICY.rust_image in why, why


def test_the_image_check_only_speaks_when_docker_actually_said_no_such_image(monkeypatch):
    """An ambiguous Docker error must not be relabelled as a missing download:
    the operator would go and pull an image they already have, twice, while the
    real failure went unreported."""
    from solvers import rehearse
    import subprocess

    monkeypatch.setattr(rehearse.shutil if hasattr(rehearse, "shutil") else __import__("shutil"),
                        "which", lambda name: "/usr/bin/docker", raising=False)

    def answering(returncode, stderr):
        def run(cmd, **kw):
            return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")
        return run

    monkeypatch.setattr(subprocess, "run", answering(1, "Error: No such image: ghcr.io/x@sha256:y"))
    assert rehearse._rust_sandbox_missing() is not None

    monkeypatch.setattr(subprocess, "run", answering(0, ""))
    assert rehearse._rust_sandbox_missing() is None

    monkeypatch.setattr(subprocess, "run", answering(1, "permission denied on /var/run/docker.sock"))
    assert rehearse._rust_sandbox_missing() is None, "an ambiguous error was called a missing image"

    def explode(cmd, **kw):
        raise OSError("docker vanished")

    monkeypatch.setattr(subprocess, "run", explode)
    assert rehearse._rust_sandbox_missing() is None


# --------------------------------------------------------------------------- #
# Quality over speed: the prompt must tell the model the truth about what the
# payment rule actually rewards.
# --------------------------------------------------------------------------- #
def test_a_repair_ends_on_the_rule_for_what_may_come_back():
    """The last thing a repair says is what it will accept, because that is the
    sentence the reply has to obey.

    There is now ONE sentence, and that is the change. A repair used to offer
    the model a second way out -- rewrite the failing CASE instead of the
    program -- because the bar's expected values were themselves reasoned to
    by a model and could be wrong. Nothing reasons to an expected value any
    more: the reference program is run and its outputs are the expectations,
    so a disagreement is between two programs and the only thing that can come
    back is a program."""
    from solvers.prompts import WHOLE_PROGRAM

    prompt = _repair_prompt("python", report="g(*[0], **{}) returned 1, expected 0")

    assert WHOLE_PROGRAM in prompt, prompt
    assert "json" not in prompt.lower(), (
        "offered to rewrite a case, which no longer exists as a way out"
    )
    assert "before you send" not in prompt.lower(), (
        "the repair prompt reintroduced the phrase that caused narration"
    )
    # ...and the same holds when the REFERENCE is the thing being patched: it
    # is still a program that has to come back whole.
    reference = _repair_prompt("python", kind="oracle")
    assert WHOLE_PROGRAM in reference, reference
    assert "REFERENCE implementation" in reference, reference


def test_a_round_that_sends_only_what_it_changed_is_told_to_send_it_all():
    """The repair round asks for the whole program; a model shown one failing
    case answers about that case.

    What comes back is the one function it fixed -- correct in itself, and
    unrunnable, because the helper it calls is in the reply ABOVE it and the
    file that gets submitted is this reply alone. `compile()` accepts it,
    `python_defect` reports it clean, and every hidden test dies on `NameError`
    for a function that was right there a round ago.

    So the round is reported as incomplete, by name, and the next prompt asks
    for the whole program rather than for different logic -- the same
    distinction the defect branch already makes between "your logic is wrong"
    and "I could not run this at all"."""
    from solvers.prompts import dropped_definitions

    previous = ("import math\n"
                "def digits(n):\n    return [int(c) for c in str(n)]\n"
                "def g(n):\n    return sum(digits(n))")
    only_the_fix = "def g(n):\n    return sum(digits(abs(n)))"

    defect = dropped_definitions(only_the_fix, previous)
    assert defect and "`digits`" in defect, defect
    assert "only part of the program" in defect, defect

    prompt = _repair_prompt("python", code=only_the_fix, defect=defect,
                            found_by="unrun")
    assert "digits" in prompt, "the repair never named what was missing"
    assert "not only the part you changed" in prompt, prompt
    assert "comparing what they produced" not in prompt, (
        "blamed logic that was never run"
    )


def test_a_whole_program_that_drops_a_helper_it_no_longer_calls_is_fine():
    """The guard that keeps the one above honest.

    A correction is allowed to be a rewrite. Inlining a helper, or replacing two
    functions with one, drops a definition the previous version had -- and that
    is a complete program, not a fragment. Only a name still USED and no longer
    defined says the reply is part of something.

    Measured against the corpus rather than argued: `_unbound` finds nothing
    unresolvable in any of the 26 archived Python answers, so this check has no
    false-positive surface on real submissions at all."""
    from solvers.prompts import _unbound, dropped_definitions

    previous = ("def digits(n):\n    return [int(c) for c in str(n)]\n"
                "def g(n):\n    return sum(digits(n))")
    inlined = "def g(n):\n    return sum(int(c) for c in str(n))"
    assert dropped_definitions(inlined, previous) is None, "refused a rewrite"
    # ...and the first answer of a solve, which has no predecessor at all.
    assert dropped_definitions(previous, "") is None

    archived = _archived_answers("python")
    assert len(archived) >= 20, f"expected the archived answers, got {len(archived)}"
    flagged = [name for name, code in archived if _unbound(code)]
    assert not flagged, f"would have called a real submission incomplete: {flagged}"


def test_a_fragment_does_not_displace_the_whole_program_above_it():
    """The other half, and the one the prompt cannot cover.

    THE LATEST VERSION WINS is the rule, and a fragment is not a version. If the
    last round of a solve sends back only the function it fixed and the budget
    ends before anything can be run, `_supersedes` would ship it -- a certain
    zero, over a complete program that at least runs.

    Nothing is ranked here: no score is compared, and between two fragments the
    later one still wins, because between two fragments the later one is still
    the correction."""
    from solvers.verify import Candidate, _supersedes

    whole = Candidate(code="def digits(n): return []\ndef g(n): return digits(n)",
                      raw="", self_passed=1, self_total=3, failures=["case 2 ..."],
                      from_self_tests=True)
    fragment = Candidate(code="def g(n): return sum(digits(n))", raw="",
                         defect="this is only part of the program: it uses `digits`",
                         partial=True)
    later_fragment = Candidate(code="def g(n): return sum(digits(abs(n)))", raw="",
                               defect="this is only part of the program", partial=True)

    assert not _supersedes(fragment, whole, False), "shipped a fragment"
    # A whole program still supersedes, failures and all -- the rule is unchanged
    # for everything that is actually a program.
    assert _supersedes(whole, fragment, False)
    # Two fragments: latest still wins. There is no complete answer to protect.
    assert _supersedes(later_fragment, fragment, False)
    # And a fragment is still better than nothing at all.
    assert _supersedes(fragment, Candidate(code="", raw=""), False)


def test_a_tab_replaying_an_old_reply_stops_instead_of_spinning(monkeypatch):
    """The other side of it, and the reason the guard exists at all.

    `send` blocks on the chat UI until the reply finishes, so a round that comes
    back instantly read something already on the page. Re-asking that does not
    cost a round trip — it costs nothing — and the loop would resend the same
    repair at machine speed until the budget was gone, thousands of times, for
    one answer that never changes.

    Time is what tells the two apart, not the count: the model answering the
    same way twice is worth another ask, and a tab that never asked anything is
    not."""
    from solvers import verify

    cases = ('```json\n[{"name": "zero", "args": [0]},\n'
             ' {"name": "carry", "args": [12345]}]\n```')
    right = ("```python\ndef g(n):\n    return sum(int(d) for d in str(n))\n```")
    stale = "```python\ndef g(n):\n    return 0\n```"
    sent: list[str] = []

    class _Stale(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            sent.append((id(self), text))
            phase = _phase_of(text)
            if phase == "inputs":
                return cases
            if phase == "oracle":
                return right     # the reference is right, so `stale` mismatches
            if phase == "analysis":
                return ""
            return stale         # the candidate, and every repair: instantly

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Stale(self._script, self._provider)

    task = SolveTask(problem_id="stale", language="python",
                     statement="Return the sum of the decimal digits of n.",
                     entrypoint="g", public_examples=[], deadline_s=30.0)
    solver = verify.VerifyingSolver(_Backend2([]), reserve_s=0, max_budget_s=30,
                                    second_opinion=False)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        answer = asyncio.run(solver.solve_task(task, timeout_s=30.0))
    out = log.getvalue()

    from collections import Counter

    worst = max(Counter(who for who, _ in sent).values(), default=0)
    assert worst <= 4, f"spun on a tab that was not answering: {worst} prompts"
    assert "replaying an old reply" in out, out
    # The last version still goes out. A stale tab is a reason to stop asking,
    # never a reason to submit nothing.
    assert answer.code.strip() == "def g(n):\n    return 0", answer.code


def test_an_answer_that_passed_every_case_it_had_is_reported_as_such():
    """`verified=False` is the only thing a live solve can print, and on its own
    it says nothing.

    `verified` means the VALIDATOR's public examples all reproduced. Live
    traffic ships none — all 97 archived requests carry zero — so `verified` is
    False on every real answer this miner sends, whether it passed every case
    the model wrote for it or was never run at all. Those are the two ends of
    the range and the log gave them the same word.

    So the state is reported, without ever letting a model agreeing with itself
    claim `verified`: that flag gates the answer cache and tells a chain of
    providers to stop trying, and self-agreement must not be able to earn it."""
    # INPUTS only -- no `expected` anywhere. The reference program supplies
    # every expectation by being run, which is the whole point of the design.
    inputs = ('```json\n[{"name": "zero", "args": [0]},\n'
              ' {"name": "single", "args": [7]},\n'
              ' {"name": "carry", "args": [12345]}]\n```')
    right = ("```python\ndef g(n):\n    t = 0\n    while n > 0:\n"
             "        t += n % 10\n        n //= 10\n    return t\n```")
    # The reference is written differently on purpose: same answers, different
    # program, so agreeing with it is evidence rather than a tautology.
    reference = "```python\ndef g(n):\n    return sum(int(d) for d in str(n))\n```"

    class _Differential(_Chat):
        async def send(self, text, timeout_s, extend_to_s=None):
            phase = _phase_of(text)
            if phase == "inputs":
                return inputs
            if phase == "oracle":
                return reference
            if phase == "analysis":
                return ""
            return right

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Differential(self._script, self._provider)

    task = SolveTask(problem_id="sv", language="python",
                     statement="Return the sum of the decimal digits of n.",
                     entrypoint="g", public_examples=[], deadline_s=60.0)
    solver = VerifyingSolver(_Backend2([]), reserve_s=0, max_budget_s=60,
                             second_opinion=False)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        answer = asyncio.run(solver.solve_task(task, timeout_s=60.0))
    out = log.getvalue()

    assert "verified on local" in out, out
    assert "reference on all 3 inputs" in out, out
    assert "no public examples exist to confirm it" in out, (
        "claimed more than the evidence supports:\n" + out
    )
    # `verified` itself is untouched: no public example ran, so nothing is
    # verified in the sense that word is used everywhere else.
    assert "verified=False" in out, out
    assert answer.verified is False
    assert answer.self_verified is True
    assert (answer.self_passed, answer.self_total) == (3, 3)
    # Counted apart, so `/solver-status` showing verified=0 over a live run
    # reads as the ordinary case rather than as a catastrophe.
    counts = solver.stats()["solver"]
    assert counts["verified_on_local"] == 1 and counts["verified"] == 0, counts


def test_a_self_check_that_failed_a_case_is_not_reported_as_verified():
    """The other half. `self_verified` is every case, not most of them — the
    payment rule is all-or-nothing, so "2 of 3" is worth exactly what 0 of 3 is
    and must not read as a pass."""
    from solvers.verify import Candidate

    passed = Candidate(code="def g(n): return n", raw="", self_passed=3,
                       self_total=3, from_self_tests=True)
    partial = Candidate(code="def g(n): return n", raw="", self_passed=2,
                        self_total=3, failures=["case 3 ..."], from_self_tests=True)
    never_run = Candidate(code="def g(n): return n", raw="")
    broken = Candidate(code="def g(", raw="", defect="not valid Python",
                       self_passed=3, self_total=3, from_self_tests=True)
    with_examples = Candidate(code="def g(n): return n", raw="", passed=2, total=2,
                              self_passed=3, self_total=3, from_self_tests=True)

    assert passed.self_verified
    assert not partial.self_verified, "a partial pass read as verified"
    assert not never_run.self_verified, "never running read as passing"
    assert not broken.self_verified, "a defect read as verified"
    # With public examples in hand THEY are the verdict, and `verified` reports
    # it. Saying both would be two answers to one question.
    assert with_examples.verified and not with_examples.self_verified


# --------------------------------------------------------------------------- #
# The archive is only evidence if it is EXACTLY what the validator received.
# --------------------------------------------------------------------------- #
def test_the_file_holds_the_exact_bytes_the_validator_was_sent(tmp_path, monkeypatch):
    """Byte-for-byte, against the code decoded from the real signed HTTP
    response — not against the variable that fed it, and not through
    `read_text`, which applies universal-newline translation on READ and would
    hide a CRLF answer being rewritten on the way to disk.

    Every shape below is one the reader can actually produce: CRLF from a
    Windows-flavoured paste, non-ASCII from a string literal, characters JSON
    has to escape, and a transcript big enough to force `fit_response` to trim
    (which must trim `raw_response` and never touch `code`)."""
    from custom_miner import response_limit

    shapes = {
        "ordinary": "def solve(n):\n    return n + 1\n",
        "no trailing newline": "def solve(n):\n    return n",
        "trailing whitespace": "def solve(n):\n    return n   \n   \n",
        "non-ascii": "def solve(n):\n    return '→ ✓ 日本語 🎯'\n",
        "json metachars": 'def solve(n):\n    return "q\\" b\\\\s \\ttab"\n',
        "CRLF": "def solve(n):\r\n    return n\r\n",
        "control chars": "def solve(n):\n    return '\\x01\\x02'\n",
        "huge transcript": "def solve(n):\n    return n\n",
    }
    for name, program in shapes.items():
        transcript = "x" * (response_limit() * 2) if name == "huge transcript" else "t"
        payload = _solved_by(
            _solver_returning(program, transcript), _request("shape", "python"), tmp_path
        )
        assert payload.code == program, f"{name}: the graded field was rewritten"
        written = tmp_path / "shape.py"
        assert written.read_bytes() == program.encode("utf-8"), (
            f"{name}: the file is not the bytes that were sent"
        )
        record = json.loads((tmp_path / "shape.json").read_text(encoding="utf-8"))
        assert record["response"]["code"] == program, f"{name}: the exchange disagrees"


def test_two_problem_ids_can_never_share_one_file(tmp_path):
    """Sanitising is lossy in both directions — `abc/def` and `abc:def` both
    became `abc_def`, and two over-long ids sharing a prefix truncated to the
    same name. Either way the second solve overwrote the first and the file
    then held an answer to a DIFFERENT problem than its name claimed, which is
    the one promise this module makes."""
    from solution_archive import save_solution

    collide = [("abc/def", "abc:def"), ("z" * 120 + "X", "z" * 120 + "Y"),
               ("p 1", "p_1"), ("../x", "x"), ("a.b", "a:b")]
    for first, second in collide:
        a = save_solution(first, "python", "x = 1", tmp_path)
        b = save_solution(second, "python", "x = 2", tmp_path)
        assert a != b, f"{first!r} and {second!r} share the file {a}"
        assert a.read_text() == "x = 1", f"{second!r} overwrote {first!r}"
        assert a.parent == b.parent == tmp_path


def test_an_id_that_needs_no_sanitising_keeps_its_readable_name(tmp_path):
    """The disambiguating digest is only for ids that were ALTERED. Real ones —
    sha256 hex, or a slug like `extent-journal` — must stay exactly themselves,
    or every filename an operator has learned changes for nothing."""
    from solution_archive import save_solution

    for readable in ("extent-journal", "rehearsal-python-1", "a" * 64, "req-1"):
        written = save_solution(readable, "python", "x = 1", tmp_path)
        assert written.name == f"{readable}.py", written.name


def test_the_file_follows_the_payload_even_if_fit_response_rewrites_the_code(tmp_path, monkeypatch):
    """`save_solution` is handed `payload.code`, not the variable that fed it.
    Today those are identical — `fit_response` trims only `raw_response` — so
    nothing observable distinguishes the two, and a refactor could quietly swap
    them back. This forces the difference: a `fit_response` that DOES rewrite
    the graded field must take the file with it, because the file is only worth
    having if it is the submission rather than something that resembles it."""
    import custom_miner

    original = custom_miner.fit_response

    def rewriting(payload, limit=None):
        trimmed = original(payload, limit)
        return trimmed.model_copy(update={"code": trimmed.code + "\n# rewritten\n"})

    monkeypatch.setattr(custom_miner, "fit_response", rewriting)
    payload = _solved_by(_solver_returning("x = 1\n"), _request("rw", "python"), tmp_path)
    assert payload.code.endswith("# rewritten\n"), "the stand-in did not fire"
    assert (tmp_path / "rw.py").read_bytes() == payload.code.encode("utf-8"), (
        "the file kept the pre-fit_response code, not what was sent"
    )


# --------------------------------------------------------------------------- #
# Never give up on a task while an answer is still obtainable.
#
# `all_passed` is a hard gate and speed is a multiplier floored at 0.95: the
# slowest CORRECT answer earns 95% of what the fastest earns, and an empty one
# earns nothing at all. Every test below pins one of the places the miner used
# to stop early and hand the validator an empty response it would have paid for.
# --------------------------------------------------------------------------- #
def test_a_tab_that_renders_nothing_is_waited_for_and_kept():
    """The reverse of what this file used to assert, and a live run is why.

    Giving up on a tab that had painted nothing after a grace period looked
    free: the recovery phases still ran, so the ANSWER was not lost. What was
    lost was the CONVERSATION. Over one production run the bail fired eighteen
    times, the wire produced the answer in nearly every one — the model was not
    silent, the DOM was late — and each of those solves then had its repair
    round sent into a tab that had just been retired. Fifteen answers went out
    with failing cases and an average of 129 unused seconds behind them.

    A model that thinks before it writes renders nothing for as long as it
    thinks: 77 seconds, measured on a live tab. There is no per-turn deadline
    to protect, only the request's own. So the read waits, and the tab stays.

    A tab is dead when the PAGE dies or the prompt cannot be submitted into it.
    Being slow to paint is neither.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    )
    site = _site(assistant=("#assistant",))  # matches nothing, now and forever

    async def go(grace, budget):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            chatter = io.StringIO()
            try:
                started = time.monotonic()
                with contextlib.redirect_stdout(chatter):
                    reply = await tab.send("solve it", budget)
                elapsed = time.monotonic() - started
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return reply, elapsed, tab.alive, chatter.getvalue()

    reply, elapsed, alive, log = asyncio.run(go(1.0, 6.0))
    assert reply == "", f"there was nothing on the page to capture: {reply!r}"
    assert elapsed >= 5.0, (
        f"stopped after {elapsed:.1f}s of a 6s budget instead of waiting for an "
        f"answer that may still have been on its way"
    )
    assert alive is True, (
        "a tab that is merely slow to paint was retired; the conversation the "
        "repair loop needs went with it"
    )
    assert "still waiting" in log, log

def test_the_tab_says_WHY_a_send_came_back_empty():
    """`""` is the same string for three different failures, and only one of
    them is worth another prompt in the same conversation. The tab is the only
    place that can tell them apart, so it records which it was.

    Getting this wrong in either direction costs a task: treat a prose reply as
    unreadable and the repair round that would have fixed it never happens;
    treat an unreadable tab as prose and the repair goes into a conversation
    that has already proved it cannot be read.
    """
    playwright, chrome = _chromium_or_skip()

    BUBBLE = (
        "  const d = document.createElement('div');"
        "  d.className = 'msg';"
        "  d.textContent = 'Sure! Here is the approach...';"
        "  document.getElementById('host').appendChild(d);"
    )
    shell = (
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div>__STOP__<script>'
        "document.getElementById('send').onclick = () => {__BODY__};</script>"
    )
    cases = [
        # A finished reply that simply has no code block in it.
        ("no-code", _served(shell.replace("__STOP__", "").replace("__BODY__", BUBBLE)),
         _site(assistant=("div.msg",))),
        # The same reply, with the site still showing its stop control.
        ("unfinished",
         _served(shell.replace("__STOP__", '<button id="stop">stop</button>')
                      .replace("__BODY__", BUBBLE)),
         _site(assistant=("div.msg",), busy=("#stop",))),
        # Nothing renders at all.
        ("unreadable", _served(shell.replace("__STOP__", "").replace("__BODY__", "")),
         _site(assistant=("div.msg",))),
    ]

    async def go(url, site, grace):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            try:
                await tab.send("solve it", 3.0)
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return tab.empty_reason

    for expected, url, site in cases:
        got = asyncio.run(go(url, site, 1.0))
        assert got == expected, f"expected {expected!r}, the tab said {got!r}"


def test_the_grace_is_long_enough_that_a_thinking_model_is_never_dropped():
    """The bound this fix must not break. A reply that has rendered -- even as
    an empty bubble with a cursor in it -- keeps its whole slice, because the
    check is 'did it EVER appear', not 'has it finished'. Getting this wrong
    would trade two zeros a run for a zero on every slow answer."""
    assert BLIND_TAB_GRACE_S >= 20.0, (
        "a site under load can take seconds to paint the assistant bubble; "
        "anything tighter starts killing tabs that were about to answer"
    )


@pytest.mark.parametrize("reason", ["unreadable", "unfinished"])
def test_a_conversation_that_cannot_answer_is_not_asked_to_repair_itself(reason):
    """Nothing captured, and the CONVERSATION is why.

    "unreadable" is a tab that never rendered a reply or died; "unfinished" is a
    model still writing when the budget ran out. A repair round goes straight
    back into the first, and queues behind an answer that does not exist yet in
    the second. Measured on a live miner, twice in one run: 191s for nothing,
    then 29s more for nothing, ending the task with 5s left while five healthy
    tabs sat idle.

    The distinction cannot be made from the candidate -- an empty one always
    carries a `defect`, because the structural checks reject empty source
    exactly as they reject a broken program. Only the tab knows, and this is it
    saying so.
    """
    sends: list[str] = []

    class _Silent(_Chat):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.empty_reason = reason
        async def send(self, text, timeout_s, extend_to_s=None):
            sends.append((id(self), text))
            return ""

    class _Fleet:
        async def open(self, avoid=None):
            return _Silent([""], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=3, reserve_s=0, max_budget_s=120,
        second_opinion=False,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code == "", "nothing was captured; nothing is the honest answer"
    # Per CONVERSATION. A pass opens one per phase, so several prompts go out
    # in total; what must never happen is asking the SAME silent conversation
    # twice, since each ask is a full slice of the budget spent to be told the
    # same thing.
    from collections import Counter

    worst = max(Counter(who for who, _ in sends).values(), default=0)
    assert worst == 1, (
        f"asked one conversation {worst} times after it returned nothing"
    )


def test_an_empty_answer_keeps_asking_while_the_clock_allows_it():
    """Two passes was a policy for 'the first model was WRONG'. Empty is a
    different state and has a different price: a wrong answer pays zero and so
    does no answer, but an extra ask can only turn the second into a payment.
    So while nothing is in hand, keep asking until the clock says stop.
    """
    seen: list[str] = []

    class _Fleet:
        # One script per PASS, not per open. A pass opens one conversation per
        # phase and they all carry the same `avoid`, so the script advances
        # when `avoid` CHANGES -- which is exactly once per pass, and unlike a
        # truthiness test it still tells four alternating passes apart.
        _unset = object()

        def __init__(self, replies):
            self._replies, self._i = replies, 0
            self._avoid, self._script = self._unset, None

        async def open(self, avoid=None):
            provider = "chatgpt" if avoid == "claude" else "claude"
            seen.append(provider)
            if avoid != self._avoid:
                self._avoid = avoid
                which = min(self._i, len(self._replies) - 1)
                self._script = _Script(self._replies[which])
                self._i += 1
            return _Chat(self._script, provider)
        async def aclose(self): pass
        def stats(self): return {}

    # Three tabs in a row capture nothing; the fourth answers.
    solver = VerifyingSolver(
        _Fleet([["nope"], ["nope"], ["nope"], [RIGHT]]),
        max_attempts=1, reserve_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified, (
        f"gave up after {len(seen)} model(s) with time still on the clock and "
        f"submitted nothing. The answer was one ask away."
    )
    assert list(dict.fromkeys(seen)) == ["claude", "chatgpt"], seen


def test_the_run_of_asks_is_capped_so_a_dead_fleet_cannot_spin():
    """The other side of the same loop. Every tab failing must cost a bounded
    number of asks, not one per poll until the deadline."""
    seen: list[str] = []

    class _Fleet:
        async def open(self, avoid=None):
            seen.append(avoid or "first")
            return _Chat(["nothing here"], "chatgpt" if avoid == "claude" else "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=1, reserve_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code == ""
    assert len(set(seen)) <= MAX_PASSES, (
        f"asked {len(set(seen))} distinct providers, cap is {MAX_PASSES}"
    )


def test_no_time_floor_gates_a_further_pass():
    """The operator's rule: the deadline is the only clock. Whether another
    model is asked is a question about how many have been asked and about the
    deadline -- an ask with any time left is allowed to happen, and one with
    none is not, whatever is in hand."""
    from solvers import verify

    assert not hasattr(verify, "EMPTY_HANDED_FLOOR_S")
    assert not hasattr(verify, "SECOND_OPINION_FLOOR_S")
    assert not hasattr(verify, "RESUME_FLOOR_S")
    assert not hasattr(verify, "ROUND_TRIP_FLOOR_S")
    solver = verify.VerifyingSolver.__new__(verify.VerifyingSolver)
    empty = verify.Candidate(code="", raw="")
    held = verify.Candidate(code="def g(n): return 0", raw="")
    assert solver._next_pass_blocked_by(empty, 1, 1.5) is None
    assert solver._next_pass_blocked_by(held, 1, 1.5) is None
    # Under the send's own minimum slice a pass could only be padded past the
    # deadline; that is the one thing besides the deadline that decides.
    assert "deadline" in solver._next_pass_blocked_by(empty, 1, 0.5)
    assert "deadline" in solver._next_pass_blocked_by(held, 1, -1.0)


def test_the_first_read_gets_the_deadline_the_validator_actually_advertised():
    """The self-imposed cap that produced both live zeros.

    `SOLVER_MAX_BUDGET_S=240` against the 300s deadline this subnet advertises
    made the budget 225s and the first read 191s. A model still writing at 191s
    had its answer discarded HERE -- not by the validator, which pays ~96% for
    the same answer arriving at six minutes, because the speed multiplier is
    floored at 0.95 while correctness is a hard gate.

    The cap stays as a runaway guard. It must not bind at the advertised
    deadline.
    """
    slices: list[float] = []

    class _Timed(_Chat):
        async def send(self, text, timeout_s):
            slices.append(timeout_s)
            return RIGHT

    class _Fleet:
        async def open(self, avoid=None): return _Timed([RIGHT], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    # No public examples -- which is every task on live traffic.
    task = SolveTask(
        problem_id="live", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[], deadline_s=300.0,
    )
    # Defaults on purpose: this is what an operator who has tuned nothing gets.
    asyncio.run(VerifyingSolver(_Fleet()).solve_task(task, timeout_s=300.0))
    # slices[0] is the cases turn, and it reads against the WHOLE of what is
    # left. The ceiling that used to sit here was a third cap on this turn --
    # two removed before it -- and every one of them was a second deadline on
    # a solve that already has one. What the cases turn costs is decided by
    # when the model finishes, which is what the program read below shows.
    assert slices[0] > 230.0, (
        f"the cases turn was given {slices[0]:.0f}s of a 300s deadline; a read "
        f"that thinks before it writes gets the budget, not a share of it"
    )
    slices = slices[1:]
    assert slices[0] > 230.0, (
        f"the first read got {slices[0]:.0f}s of a 300s deadline. The old cap "
        f"gave it 191.2s and threw away answers the validator would have paid "
        f"~96% for."
    )


def test_a_shorter_advertised_deadline_still_wins():
    """The cap is a ceiling, and raising it must not let the miner overrun a
    validator that advertises less. `min()` is the whole guarantee: a 60s
    deadline stays a 60s deadline."""
    slices: list[float] = []

    class _Timed(_Chat):
        async def send(self, text, timeout_s):
            slices.append(timeout_s)
            return RIGHT

    class _Fleet:
        async def open(self, avoid=None): return _Timed([RIGHT], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    asyncio.run(VerifyingSolver(_Fleet()).solve_task(DIGITS, timeout_s=60.0))
    assert slices[0] < 60.0, (
        f"the first read got {slices[0]:.0f}s of a 60s deadline; the whole solve "
        f"has to be signed and on the wire before it expires"
    )


def test_a_reply_that_has_rendered_keeps_its_whole_slice_however_slow_it_is():
    """The bound the fail-fast must not cross, and the one that would hurt most
    if it did.

    `BLIND_TAB_GRACE_S` retires a tab that renders NOTHING. Almost every real
    answer takes longer than the grace to finish — so if the check ever asks
    "has it finished" instead of "did it appear", it stops being a fix for two
    zeros a run and becomes a zero on every answer slower than 30 seconds,
    which is most of them.

    The page here does exactly what a chat UI does: paints an empty assistant
    bubble on submit, and fills the code in long after the grace has passed.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div');"
        "  d.className = 'msg';"                       # the bubble, empty
        "  document.getElementById('host').appendChild(d);"
        "  setTimeout(() => {"                          # ...the code, much later
        "    const pre = document.createElement('pre');"
        "    const code = document.createElement('code');"
        "    code.textContent = 'def pong():\\n    return 4';"
        "    pre.appendChild(code); d.appendChild(pre);"
        "  }, 3000);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",))

    async def go(grace):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            try:
                reply = await tab.send("solve it", 30.0)
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return reply, tab.alive

    # The grace expires at 1s; the code does not arrive until 3s.
    reply, alive = asyncio.run(go(1.0))
    assert "return 4" in extract_code(reply, "pong"), (
        f"a reply that had already rendered was dropped for being slow: {reply!r}. "
        f"The check is 'did it EVER appear', not 'has it finished'."
    )
    assert alive is True, "a tab that answered must not be retired"


def test_a_wrong_answer_still_gets_exactly_one_second_opinion():
    """`MAX_PASSES` is for the empty case only.

    Letting it apply to a wrong-but-running answer would quietly double or
    quadruple what every failing task costs a real account's quota — to improve
    on something that was already worth submitting. The empty case is different
    precisely because there is nothing there to improve on.
    """
    seen: list[str] = []

    class _Fleet:
        async def open(self, avoid=None):
            seen.append(avoid or "first")
            return _Chat([WRONG], "chatgpt" if avoid == "claude" else "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=1, reserve_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code, "the wrong answer is still submitted; it may pass the hidden suite"
    assert answer.verified is False
    assert len(set(seen)) == SECOND_OPINION_PASSES, (
        f"asked {len(set(seen))} models about an answer that was already in hand; "
        f"the policy for a wrong answer is one second opinion"
    )


def test_a_deadline_the_miner_shortens_itself_is_reported_once(capsys):
    """`GLM_REQUEST_TIMEOUT_S` is named for the reference miner's model client,
    but `handle_request` applies it to whatever solver is plugged in. Left at
    the 280 that `docs/DEMO_MINER.md` documents, it silently cuts 20 seconds off
    every solve against this subnet's 300s deadline — and nothing else in the
    logs says so. That gap is worth more than it looks: a correct answer
    arriving late still earns 95%+, an unfinished one earns nothing.
    """
    class _Fleet:
        async def open(self, avoid=None): return _Chat([RIGHT], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Fleet())
    # deadline_s=300 (what the validator advertised) vs timeout_s=280 (ours).
    asyncio.run(solver.solve_task(DIGITS, timeout_s=280.0))
    said = capsys.readouterr().out
    assert "300s" in said and "280s" in said, said
    assert "GLM_REQUEST_TIMEOUT_S" in said, f"the fix has to be nameable: {said}"

    # Once per run, not once per task: this is a configuration fact, and a
    # miner answering hundreds of tasks would otherwise say it hundreds of times.
    asyncio.run(solver.solve_task(DIGITS, timeout_s=280.0))
    assert "GLM_REQUEST_TIMEOUT_S" not in capsys.readouterr().out

    # And silent when nothing is being given up.
    quiet = VerifyingSolver(_Fleet())
    asyncio.run(quiet.solve_task(DIGITS, timeout_s=300.0))
    assert "GLM_REQUEST_TIMEOUT_S" not in capsys.readouterr().out


def test_a_reply_that_vanishes_mid_answer_is_waited_for_not_written_off():
    """Why the fail-fast asks "did it EVER appear" and not "is it there now".

    Sites stream a message under one attribute and drop it when the message is
    finished, so the selector that found the answer can be the one that cannot
    see it any more — `_messages` has a whole re-resolve path for exactly this,
    and says so in the log. During that window the reply is off screen while
    being written.

    A tab in that window looks identical to one that never rendered at all:
    both report nothing on screen. Only the memory that it was there ONCE
    separates them — and getting it wrong means retiring tabs in the middle of
    producing the answer, which is worse than the bug the fail-fast fixes.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div');"
        "  d.className = 'msg';"                       # the bubble appears...
        "  document.getElementById('host').appendChild(d);"
        "  setTimeout(() => { d.className = 'gone'; }, 1200);"   # ...stops matching...
        "  setTimeout(() => {"                                    # ...and comes back with the code
        "    const pre = document.createElement('pre');"
        "    const code = document.createElement('code');"
        "    code.textContent = 'def pong():\\n    return 4';"
        "    pre.appendChild(code); d.appendChild(pre);"
        "    d.className = 'msg';"
        "  }, 3000);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",))

    async def go(grace):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            try:
                reply = await tab.send("solve it", 30.0)
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return reply, tab.alive

    # The grace expires at 1s. The reply goes off screen at 1.2s -- after the
    # grace, so every poll from then until 3s reports nothing on screen.
    reply, alive = asyncio.run(go(1.0))
    assert "return 4" in extract_code(reply, "pong"), (
        f"the tab was written off while its answer was between selectors: "
        f"{reply!r}. It had already rendered once; that is the whole test."
    )
    assert alive is True, "a tab that answered must not be retired"


def test_the_solve_timeout_default_tracks_the_deadline_the_subnet_advertises(monkeypatch):
    """`GLM_REQUEST_TIMEOUT_S` is the miner's own 504 deadline, and leaving it
    below what validators advertise is a self-inflicted cut on every solve.

    The assertion that matters is the last one: it ties the default to
    `Settings.solve_deadline_s` rather than to a number someone typed, so if the
    subnet moves its deadline this fails instead of quietly going back to
    throwing answers away.
    """
    from solvers.config import DEFAULT_SOLVE_TIMEOUT_S, apply_solve_timeout_default

    monkeypatch.delenv("GLM_REQUEST_TIMEOUT_S", raising=False)
    apply_solve_timeout_default()
    assert os.environ["GLM_REQUEST_TIMEOUT_S"] == DEFAULT_SOLVE_TIMEOUT_S

    # An operator who set it keeps it, whether from the shell or from .env --
    # `load_env_file` has already copied .env into the environment by now.
    monkeypatch.setenv("GLM_REQUEST_TIMEOUT_S", "120")
    apply_solve_timeout_default()
    assert os.environ["GLM_REQUEST_TIMEOUT_S"] == "120", "the operator's value must win"

    advertised = Settings().solve_deadline_s
    assert float(DEFAULT_SOLVE_TIMEOUT_S) > advertised, (
        f"the miner caps solves at {DEFAULT_SOLVE_TIMEOUT_S}s while validators "
        f"advertise {advertised:g}s. Every solve gets less time than it was "
        f"offered, and an unfinished answer earns nothing while a late correct "
        f"one still earns 95%+."
    )
    # Strictly greater, not equal. Sitting exactly ON the advertised deadline
    # binds the moment the subnet raises it, which is the same bug one level up
    # and just as quiet. The runaway guard for an absurd deadline is
    # SOLVER_MAX_BUDGET_S, in the solver; a second one here only adds a way to
    # be wrong.
    assert float(DEFAULT_SOLVE_TIMEOUT_S) >= 2 * advertised or (
        float(DEFAULT_SOLVE_TIMEOUT_S) - advertised >= 60.0
    ), "leave real headroom above the deadline, not a rounding error"


def test_a_send_that_never_starts_still_reports_why():
    """Tabs are recycled across tasks, so `empty_reason` outlives the
    conversation that set it. The two paths that return before the read loop
    runs at all — a tab already known dead, and a prompt that never reached the
    composer — would otherwise leave the PREVIOUS task's reason standing, and a
    stale `no-code` buys a repair round in a conversation that was never opened.
    """
    site = _site()

    async def go(prepare):
        tab = _Tab(_SoloPool(site), None, None, "probe", site, composer="#composer")
        tab.empty_reason = "no-code"          # left over from an earlier task
        prepare(tab)
        return await tab.send("solve it", 5.0), tab.empty_reason

    def kill(tab): tab.alive = False

    reply, reason = asyncio.run(go(kill))
    assert reply == ""
    assert reason == "unreadable", (
        f"a dead tab reported {reason!r} — the previous task's reason, which "
        f"would earn this one a repair round in a conversation that is gone"
    )


def test_a_blind_tab_still_gets_its_answer_off_the_wire():
    """Retiring an unreadable tab must not stop it being READ one last time.

    The network stream is captured by CDP off the wire and has never touched the
    DOM, so a selector matching nothing says nothing at all about it —
    `_reconcile_stream`'s own docstring names this case: "a selector that
    stopped matching, a render this tab cannot see". Clearing `alive` inside the
    read loop skipped both the copy control and the stream, which is the
    difference between a zero and the whole payment on the one tab that needed
    them.

    So the loop stops polling, the recovery phases run, and only then is the tab
    retired.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    )
    site = _site(assistant=("#nothing-matches-this",), stream=True)

    async def go(grace):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)

            async def wire():
                return "here it is:\n\n```python\ndef pong():\n    return 4\n```\n"

            tab._streamed_markdown = wire
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            try:
                reply = await tab.send("solve it", 60.0)
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return reply, tab.alive, tab.empty_reason

    reply, alive, reason = asyncio.run(go(1.0))
    assert "return 4" in extract_code(reply, "pong"), (
        f"the DOM was unreadable and the answer was on the wire, and nothing "
        f"looked: {reply!r}"
    )
    assert alive is True, (
        "the tab was retired after the wire had just proved the model was "
        "answering it — and the repair round for this solve dies with it"
    )
    assert reason is None, "an answer came back, so there is no empty to explain"


def test_no_assistant_candidate_can_match_a_user_turn():
    """The one selector mistake that costs money silently.

    An assistant candidate that also matched the user's turn would have the
    miner read its own prompt back and submit it: no error, no empty reply, a
    permanent zero. `_Tab.send`'s echo guard is the backstop and has caught it
    in production — this is the check that stops it reaching the guard.

    Every candidate on both sites is asserted against a page carrying only user
    turns, in the shapes each site actually uses. A new fallback that is merely
    *broad* fails here rather than in a live solve.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        # ChatGPT's user turn: the same attributes as an assistant turn, the
        # other value.
        '<article data-turn="user" class="group/conversation-turn">'
        '  <div data-message-author-role="user" data-message-id="aaa">'
        '    <div class="whitespace-pre-wrap">solve it</div>'
        '  </div>'
        '</article>'
        # claude.ai's user turn.
        '<div data-testid="user-message"><p>solve it</p></div>'
        '<div class="font-user-message"><p>solve it</p></div>'
    )

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            hits = {}
            for site in (chatgpt_site(), claude_site()):
                for candidate in site.assistant:
                    n = await page.locator(candidate).count()
                    if n:
                        hits[f"{site.name}: {candidate}"] = n
            await browser.close()
            return hits

    hits = asyncio.run(go())
    assert not hits, (
        f"these assistant candidates match a USER turn: {hits}. The miner would "
        f"read its own prompt back and submit it as the answer."
    )


def test_the_assistant_role_has_more_than_one_candidate_on_every_site():
    """The role whose failure is total is the role that must not be a single
    point of failure. Every other role degrades when its selector stops
    matching — the submit falls back, the copy falls back to scraping. An
    assistant selector matching nothing reads no answer at all, for every task,
    until somebody notices. A live run cost a whole task to exactly that, with
    one candidate on the list."""
    for site in (chatgpt_site(), claude_site()):
        assert len(site.assistant) >= 2, (
            f"{site.name} has {len(site.assistant)} assistant candidate(s); one "
            f"deploy away from reading nothing at all"
        )


# --------------------------------------------------------------------------- #
# A model still writing is waited for, never interrupted.
#
# The slice a read is given is an internal ALLOCATION -- part of the budget is
# held back for a repair round. That reserve is well spent on an answer that
# arrived WRONG and worth nothing at all on one that has not finished arriving.
# --------------------------------------------------------------------------- #
_STREAMING_PAGE = (
    '<!doctype html><meta charset="utf-8">'
    '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    '<button id="stop" style="display:none">stop</button>'
    '<div id="host"></div><script>'
    "const LINES = ['def g(n):','    t = 0','    while n > 0:',"
    "               '        t += n % 10','        n //= 10','    return t'];"
    "document.getElementById('send').onclick = () => {"
    "  const stop = document.getElementById('stop'); stop.style.display = '';"
    "  const d = document.createElement('div'); d.className='msg';"
    "  const pre = document.createElement('pre'); const code = document.createElement('code');"
    "  pre.appendChild(code); d.appendChild(pre);"
    "  document.getElementById('host').appendChild(d);"
    "  let i = 0;"
    "  const t = setInterval(() => {"
    "    if (i < LINES.length) { code.textContent += LINES[i++] + '\\n'; }"
    "    else { clearInterval(t); stop.style.display='none'; }"
    "  }, 800);"                                     # finishes at ~4.8s
    "};</script>"
)


def _send_on_streaming_page(slice_s, extend_to, *, busy=("#stop:visible",)):
    playwright, chrome = _chromium_or_skip()
    url = _served(_STREAMING_PAGE)
    site = _site(assistant=("div.msg",), busy=busy, poll_s=0.2)

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            started = time.monotonic()
            reply = await tab.send("solve it", slice_s, extend_to_s=extend_to)
            elapsed = time.monotonic() - started
            await browser.close()
            return reply, elapsed, tab.still_writing, tab.empty_reason

    return asyncio.run(go())


def test_a_read_waits_out_an_answer_that_is_still_arriving():
    """The slice is not a deadline, and stopping at it threw away answers.

    Measured on this page, whose model finishes at ~4.8s against a 3s slice:

        slice only:      read 3.0s, code = 'def g(n):\\n    t = 0\\n    while n > 0:'
        slice + budget:  read 6.0s, the whole program

    The truncated version is not merely worse, it is a certain zero AND a wasted
    repair round: `python_defect` reports "expected an indented block after
    'while' statement", so the miner interrupts a model that is still writing to
    tell it about a syntax error in a program it has not finished.
    """
    reply, elapsed, writing, _ = _send_on_streaming_page(3.0, 8.0)
    code = extract_code(reply, "g", "python")
    assert "return t" in code, f"stopped mid-answer and kept a fragment: {code!r}"
    assert python_defect(code, "g") is None, "the program that came back must be whole"
    assert elapsed > 3.0, "it cannot have waited without spending longer than the slice"
    assert writing is False, "the model finished; nothing is still being written"


def test_waiting_stops_at_the_budget_and_not_a_second_later():
    """The extension is bounded by what the CALLER still has, not by the model.

    `handle_request` answers 504 -- nothing at all -- past its own deadline, so
    a read that waits for ever does not deliver a late answer, it throws away
    the whole solve. `extend_to_s` is that bound.
    """
    _, elapsed, writing, _ = _send_on_streaming_page(1.0, 3.0)
    assert elapsed < 8.0, (
        f"read for {elapsed:.1f}s against a 3s cap; the model is still writing "
        f"and would be waited for indefinitely"
    )
    assert writing is True, (
        "the caller has to be told the answer was cut off rather than absent — "
        "it is what stops a repair prompt going into a live conversation"
    )


def test_a_finished_reply_does_not_spend_the_repair_reserve():
    """The other half of the bound. A reply that has SETTLED must stop at its
    slice, because the reserve it would eat is exactly what pays for the repair
    round that fixes it. Extending on everything would trade one bug for a
    worse one: a no-code reply with no budget left to ask again."""
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div'); d.className='msg';"
        "  d.textContent = 'Sure! Here is the approach...';"   # settles at once
        "  document.getElementById('host').appendChild(d);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",), poll_s=0.2)

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            started = time.monotonic()
            await tab.send("solve it", 2.0, extend_to_s=30.0)
            elapsed = time.monotonic() - started
            await browser.close()
            return elapsed, tab.still_writing, tab.empty_reason

    elapsed, writing, reason = asyncio.run(go())
    assert elapsed < 6.0, (
        f"a settled reply held the read for {elapsed:.1f}s of a 30s budget. That "
        f"time belongs to the repair round that turns it into an answer."
    )
    assert writing is False
    assert reason == "no-code", "a finished reply with no code is the model's doing"


def test_still_writing_is_detected_without_a_busy_selector():
    """`usable_busy_selectors` DROPS any busy candidate that matches an idle
    page at startup, so a site legitimately runs with none — and then the stop
    control says False through the whole of an answer that is still arriving.
    Measured before the fallback existed, with the model mid-sentence:

        busy selector present:  empty_reason='unfinished'   (no repair)
        busy selector dropped:  empty_reason='no-code'      (REPAIRED)

    A message longer than it was one poll ago is being written, whatever the
    DOM calls its stop button. That needs no selector at all.

    The page here is the shape the live failure had: the model thinking out
    loud, with no code block yet. `_read` returns None throughout, so the read
    runs to its deadline rather than settling, and the growth of the message is
    the only thing left that knows an answer is on its way.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div'); d.className='msg';"
        "  document.getElementById('host').appendChild(d);"
        "  let n = 0;"
        "  setInterval(() => { d.textContent += 'considering case ' + (++n) + '. '; }, 150);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",), busy=(), poll_s=0.2)   # no stop control

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            await tab.send("solve it", 2.0)
            await browser.close()
            return tab.still_writing, tab.empty_reason

    writing, reason = asyncio.run(go())
    assert writing is True, "the message was growing on every poll"
    assert reason == "unfinished", (
        f"reported {reason!r} with no busy selector, which buys a repair prompt "
        f"in a conversation the model is still writing into"
    )


@pytest.mark.parametrize(
    "captured, what",
    [("", "nothing at all"), ("```python\ndef g(n):\n    t = 0\n    while n > 0:\n```", "a half-written program")],
)
def test_a_model_still_writing_is_never_sent_a_repair_prompt(captured, what):
    """The two shapes of the same mistake, both measured on a real page.

    Nothing captured, the model mid-sentence:
        -> "Your previous reply did not reach me as code..."
    A code block half-rendered, the model mid-sentence:
        -> "the code is not valid Python (expected an indented block after
            'while' statement)"

    Neither can help. The composer is usually disabled while a reply streams,
    and where it is not the prompt queues behind the answer it is asking about.
    The second is the worse of the two because nothing else catches it: the
    candidate is non-empty, so the empty-capture guard does not apply, and a
    truncated program looks exactly like a broken one to `python_defect`.
    """
    sends: list[str] = []

    class _StillWriting(_Chat):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.still_writing = True
            self.empty_reason = "unfinished" if not captured else None
        async def send(self, text, timeout_s, extend_to_s=None):
            sends.append((id(self), text))
            return captured

    class _Fleet:
        async def open(self, avoid=None): return _StillWriting([""], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=3, reserve_s=0, max_budget_s=120,
        second_opinion=False,
    )
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    # Per CONVERSATION. A pass opens one per phase, so several prompts go out
    # in total; what must never happen is a SECOND prompt into the one that is
    # still writing, where it queues behind the answer it asks about.
    from collections import Counter

    worst = max(Counter(who for who, _ in sends).values(), default=0)
    assert worst == 1, (
        f"captured {what} from a model that had not finished, then sent "
        f"{worst - 1} more prompt(s) into the conversation it was still "
        f"writing in"
    )


def test_a_backend_that_cannot_wait_is_never_asked_to():
    """`extend_to_s` is optional on the `Conversation` protocol.

    A backend written outside this package satisfies the protocol with the
    two-argument `send` that has always been the contract, and calling it with
    a keyword it does not take would raise TypeError inside the single call the
    whole solve depends on. Catching that TypeError is not an option either: one
    raised from INSIDE `send` is indistinguishable, and retrying would send the
    prompt twice.

    It used to be a per-class probe of `send`'s signature. It is structural now:
    every read here is given the whole remaining budget, so there is no reserve
    to extend into and no reason to pass the keyword to anyone.
    """
    seen: list[tuple] = []

    class _OldStyle:
        provider = "claude"
        async def send(self, text, timeout_s):      # the historic two-arg form
            seen.append((timeout_s,))
            return RIGHT
        async def close(self): pass

    class _NewStyle(_OldStyle):
        async def send(self, text, timeout_s, extend_to_s=None):
            seen.append((timeout_s, extend_to_s))
            return RIGHT

    # Both are called the same way -- with ONE budget argument -- so the
    # new-style backend simply sees `extend_to_s` at its default. (`arity` is
    # what each fake RECORDS, which is fixed; what is being checked is that the
    # old-style two-argument `send` is reached at all, and that the new-style
    # one is never handed a bound it would have to honour.)
    for backend, arity in ((_OldStyle, 1), (_NewStyle, 2)):
        seen.clear()

        class _Fleet:
            async def open(self, avoid=None): return backend()
            async def aclose(self): pass
            def stats(self): return {}

        answer = asyncio.run(
            VerifyingSolver(_Fleet(), reserve_s=0, max_budget_s=120)
            .solve_task(DIGITS, timeout_s=120)
        )
        assert answer.verified, f"{backend.__name__} stopped producing answers"
        assert len(seen[0]) == arity, (
            f"{backend.__name__}.send was called with {len(seen[0])} budget "
            f"argument(s); it takes {arity}"
        )
        if arity == 2:
            assert seen[0][1] is None, (
                f"a hard bound of {seen[0][1]} was passed to a read that was "
                f"already given the whole budget"
            )


# --------------------------------------------------------------------------- #
# The copy control wins on FIDELITY, never on COMPLETENESS.
# --------------------------------------------------------------------------- #
def test_a_copy_taken_mid_stream_never_replaces_the_fuller_page(capsys):
    """Measured on a live miner, twice, on the only two tasks that spent the
    whole budget:

        what the page RENDERS and what it COPIES are not the same — they differ
        at character 1630: rendered '\\n', copied nothing (it ends here).
        Using the copy.

    Reproduced: the DOM held a complete 182-character Rust program, the copy
    control gave the first 60, and those 60 were submitted. `rust_defect`
    returned None on them — a truncated program keeps its `fn main` — so
    nothing downstream caught it and it reached the validator as a confident
    answer that cannot compile.

    A copy clicked while the reply is still streaming is the beginning of the
    answer and nothing else. The two readings were taken at different moments;
    the shorter one is simply older.
    """
    from solvers.prompts import rust_defect

    FULL = (
        "fn main() {\\n"
        "    let mut total = 0i64;\\n"
        "    for line in std::io::stdin().lines() {\\n"
        "        total += line.unwrap().trim().parse::<i64>().unwrap_or(0);\\n"
        "    }\\n"
        "    println!(\\\"{}\\\", total);\\n"
        "}"
    )
    reply, _ = _send_in_browser(
        '<!doctype html><meta charset="utf-8">\n'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>\n'
        '<div id="host"></div>\n<script>\n'
        f'const FULL = "{FULL}";\n'
        "document.getElementById('send').onclick = () => {\n"
        "  const wrap = document.createElement('div');\n"
        "  wrap.setAttribute('data-message-author-role', 'assistant');\n"
        "  const pre = document.createElement('pre');\n"
        "  const code = document.createElement('code');\n"
        "  code.textContent = FULL;\n"                     # the page has it all
        "  pre.appendChild(code); wrap.appendChild(pre);\n"
        "  const btn = document.createElement('button');\n"
        "  btn.setAttribute('aria-label', 'Copy');\n"
        "  btn.onclick = () => navigator.clipboard.writeText(FULL.slice(0, 60));\n"
        "  wrap.appendChild(btn);\n"
        "  document.getElementById('host').appendChild(wrap);\n"
        "};\n</script>"
    )
    code = extract_code(reply, "main", "rust")
    assert "println!" in code and code.rstrip().endswith("}"), (
        f"submitted the copy control's truncated version: {code!r}"
    )
    assert rust_defect(code) is None

    said = capsys.readouterr().out
    assert "CUT SHORT" in said, f"took the fuller reading but never said why: {said!r}"
    assert "98 character(s) fewer" in said, (
        f"the warning has to carry the AMOUNT — without it a reader cannot tell "
        f"a truncated program from a trailing newline: {said!r}"
    )


def test_a_copy_that_differs_in_the_middle_still_wins():
    """The other half, and the reason the guard is a PREFIX test rather than a
    length test. A highlighter artefact is a difference in the MIDDLE — the
    measured one was U+E027 inside a Python program — and there the copy is
    complete and authoritative. Only a copy that is the same answer cut short
    loses. A plain "is it shorter" test would hand every highlighter bug back
    to the DOM, which is the damage the copy control exists to avoid."""
    cut = _Tab._cut_short_by

    whole = ["def g(n):\n    total = 1 + 2\n    return total"]
    # Truncated: a strict prefix, materially shorter -> the render wins.
    assert cut(whole, ["def g(n):\n    total = 1"]) > 0
    # Damaged in the middle: same length, different content -> the copy wins.
    assert cut(whole, ["def g(n):\n    total = 1 * 2\n    return total"]) == 0
    # A highlighter INSERTION. The render is LONGER than the copy, so a plain
    # "is the copy shorter" test would hand the answer back to the DOM -- and
    # the DOM is the one carrying the damage. Only the prefix test tells them
    # apart: an insertion in the middle breaks the prefix, a truncation does not.
    assert cut(["def g(n):\n    total = 1 + 2\n    return total"], whole) == 0
    # Trailing whitespace the renderer kept and the copy control trimmed. RAW
    # this is a strict prefix and five characters short -- exactly the false
    # alarm that would fire on clean answers; normalised it is no loss at all.
    assert cut(["def g(n):\n    return 1\n\n   "], ["def g(n):\n    return 1"]) == 0
    # Below the slack -> the copy keeps its fidelity advantage.
    assert cut(["def g(n):\n    return 12"], ["def g(n):\n    return 1"]) == 0
    # Nothing to compare against.
    assert cut([], ["x"]) == 0 and cut(["x"], []) == 0


# --------------------------------------------------------------------------- #
# The request's deadline is the only deadline, and the model is told what it is.
# --------------------------------------------------------------------------- #
def test_no_miner_cap_can_bind_on_a_spec_compliant_request():
    """`TaskRequest.deadline_s` is `Field(gt=0.0, le=3600.0)`. Every cap the
    miner keeps must sit at or above that ceiling, or it is a second, PRIVATE
    deadline — and a private deadline silently costs the difference the moment
    a validator advertises more than it.

    This has already happened twice on this branch: `SOLVER_MAX_BUDGET_S=240`
    against a 300s deadline cut the first read from 238s to 191s, and
    `GLM_REQUEST_TIMEOUT_S=280` cut the whole solve by 20s. Both were invisible
    except as answers that arrived unfinished.
    """
    from solvers.config import DEFAULT_SOLVE_TIMEOUT_S

    ceiling = next(
        m.le for m in TaskRequest.model_fields["deadline_s"].metadata
        if getattr(m, "le", None) is not None
    )
    caps = {
        "GLM_REQUEST_TIMEOUT_S": float(DEFAULT_SOLVE_TIMEOUT_S),
        "VerifyingSolver(max_budget_s=)": VerifyingSolver(object())._max_budget,
        "SOLVER_MAX_BUDGET_S": float(
            _roster_default("SOLVER_MAX_BUDGET_S")
        ),
    }
    for name, value in caps.items():
        assert value >= ceiling, (
            f"{name} is {value:g} but a validator may legally advertise "
            f"deadline_s={ceiling:g}. That gap is a private deadline, and every "
            f"second of it is time the model is not given."
        )


def _roster_default(var: str) -> str:
    """The default `roster.build_solver` uses for one env var, read from source
    so the test cannot drift from the code it is checking."""
    text = (Path(__file__).parent / "solvers" / "roster.py").read_text("utf-8")
    m = re.search(rf'os\.environ\.get\("{var}",\s*"([\d.]+)"\)', text)
    assert m, f"{var} is no longer read with a literal default"
    return m.group(1)


# --------------------------------------------------------------------------- #
# A short deadline must still get an answer.
#
# `TaskRequest.deadline_s` is only `Field(gt=0.0, le=3600.0)`. Nothing in the
# protocol promises the comfortable numbers this subnet happens to send today,
# and at the small end three separate floors — each sensible alone — combined
# into a guaranteed total loss.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deadline", [3600.0, 300.0, 60.0, 40.0, 32.0, 25.0, 20.0, 15.0, 10.0, 5.0, 2.0])
def test_every_deadline_the_protocol_allows_leaves_room_to_answer(deadline):
    """The arithmetic that used to invert, checked across the legal range.

    Measured before the fix, and every row is a silent total loss:

        deadline  budget   asks?   read+tail   504 at
             32    12.0     NO        --         32    empty, model never asked
             20    10.0     NO        --         20    empty, model never asked
             15     7.5     yes      18.5        15    504, NO ANSWER
              5     5.0     yes      16.0         5    504, NO ANSWER

    `handle_request` wraps the solve in an `asyncio.wait_for` and answers 504
    with nothing at all past the deadline, so an overrun does not deliver a
    late answer — it throws away the whole solve. And a 504 is indistinguishable
    from a dead miner.
    """
    from solvers.browser_pool import tail_budget

    solver = VerifyingSolver(object())
    budget = min(deadline, solver._max_budget) - solver._reserve
    if budget <= 5.0:
        budget = max(1.0, deadline * 0.5)
    attempt_budget = max(1.0, budget)

    # The first read always happens, however little there is — pinned as
    # behaviour by `test_the_first_attempt_runs_however_little_time_there_is`.
    first_slice = max(1.0, attempt_budget * 0.85)

    spent = attempt_budget + tail_budget(first_slice)
    assert spent < deadline, (
        f"a {deadline:g}s deadline budgets {budget:.1f}s and then spends "
        f"{spent:.1f}s before the answer is even signed — `handle_request` "
        f"cancels the solve and the validator gets nothing"
    )


def test_the_post_read_tail_never_outlives_the_read_it_rescues():
    """Eleven seconds is sized against the safety margin, which is sized against
    a 300-second deadline. A rescue that costs more than half again what the
    attempt cost has stopped being a rescue — and below ~16s the fixed tail
    outlived the whole request."""
    from solvers.browser_pool import (
        FULL_TAIL_S, COPY_PHASE_TIMEOUT_S, STREAM_PHASE_TIMEOUT_S,
        SALVAGE_PHASE_TIMEOUT_S, POSTMORTEM_TIMEOUT_S, tail_budget,
    )
    from solvers.verify import DELIVERY_RESERVE_S

    # EVERY phase, and the point of naming them is that a phase without a slice
    # runs outside the promise. The prose salvage arrived that way and spent
    # `STREAM_TIMEOUT_MS` beside the tail rather than inside it — measured, a
    # constant +2.0s at every slice, which at a 5-second slice made the tail
    # 180% of its own budget. It has a slice now, taken out of the stream phase
    # rather than added to the total.
    assert FULL_TAIL_S == (
        COPY_PHASE_TIMEOUT_S + STREAM_PHASE_TIMEOUT_S
        + SALVAGE_PHASE_TIMEOUT_S + POSTMORTEM_TIMEOUT_S
    )
    # And the total is what `DELIVERY_RESERVE_S` was sized to absorb. Growing it
    # spends a reserve that also has to cover the last grade, the tab close and
    # the archive-and-sign — and overrunning does not deliver the answer late,
    # it answers 504 and throws away an answer already in hand.
    assert FULL_TAIL_S < DELIVERY_RESERVE_S, (
        f"the post-read tail ({FULL_TAIL_S}s) no longer fits inside the "
        f"delivery reserve ({DELIVERY_RESERVE_S}s)"
    )
    # Unchanged wherever there is room — which is every read in production.
    for generous in (22.0, 34.0, 238.0, 3600.0):
        assert tail_budget(generous) == FULL_TAIL_S
    # Never more than half the read below that, and never negative.
    for tight in (21.0, 10.0, 4.25, 1.0, 0.0, -5.0):
        assert 0.0 <= tail_budget(tight) <= max(0.0, tight) / 2.0 + 1e-9, tight
    assert tail_budget(10.0) == 5.0


def test_the_first_attempt_runs_however_little_time_there_is():
    """"Not enough left to be worth another ROUND TRIP" is what that guard has
    always been about, and it never should have gated the first one. It did:
    below a 32-second deadline the budget lands under twelve seconds and the
    model was never asked at all — an empty answer with no line of log to say
    why, which reads exactly like a broken browser."""
    asked: list[float] = []

    class _Chat:
        provider = "claude"
        async def send(self, text, timeout_s, extend_to_s=None):
            asked.append(timeout_s)
            return RIGHT
        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="tight", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[], deadline_s=20.0,
    )
    answer = asyncio.run(VerifyingSolver(_Fleet()).solve_task(task, timeout_s=20.0))
    assert asked, "the model was never asked at all on a 20s deadline"
    assert answer.code, f"returned nothing having asked nobody: {answer!r}"


def test_a_send_that_never_reads_reports_no_stale_writing_state():
    """`still_writing` decides whether `_attempt` may send a repair prompt, and
    the tab outlives the send that set it. The two paths that return before the
    read loop runs left the PREVIOUS send's verdict standing — so a tab that had
    been mid-answer last time reported "still writing" for a send it never even
    submitted."""
    site = _site()

    async def go():
        tab = _Tab(_SoloPool(site), None, None, "probe", site, composer="#composer")
        tab.alive = False
        tab.still_writing = True            # left over from an earlier send
        reply = await tab.send("solve it", 5.0)
        return reply, tab.still_writing, tab.empty_reason

    reply, writing, reason = asyncio.run(go())
    assert reply == ""
    assert reason == "unreadable"
    assert writing is False, (
        "a send that never submitted anything cannot have left a model writing"
    )


def test_a_page_that_died_mid_answer_is_not_reported_as_still_writing():
    """The read also exits with the tab dead when the PAGE died, and there the
    last successful poll can leave `last_busy` True. Reporting that as "still
    writing" sends the reader looking for a slow model instead of a dead tab —
    the same misdirection every other diagnostic here exists to remove."""
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<button id="stop">stop</button><div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div'); d.className='msg';"
        "  d.textContent = 'thinking';"
        "  document.getElementById('host').appendChild(d);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",), busy=("#stop",), poll_s=0.2)

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real = tab._poll
            calls = {"n": 0}

            async def dies_after_one(before):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("Target page, context or browser has been closed")
                return await real(before)

            tab._poll = dies_after_one
            await tab.send("solve it", 4.0)
            await browser.close()
            return tab.alive, tab.still_writing, tab.empty_reason

    alive, writing, reason = asyncio.run(go())
    assert alive is False, "the page died; the tab must be retired"
    assert writing is False, f"a dead page is not a model at work (reason={reason!r})"
    assert reason == "unreadable"


def test_a_rust_compile_is_not_paid_for_past_the_deadline(monkeypatch):
    """`compile_defect` floors its own timeout at one second, so an overrun
    budget still bought a temp directory and a rustc process — a whole second,
    spent past the deadline, on a verdict nothing can act on: there is no time
    for a repair round and `defect` never reaches the validator.

    The read now extends into that reserve whenever the model is still writing,
    so arriving here with nothing left is the ordinary case rather than a
    strange one."""
    import solvers.verify as verify

    calls: list[Any] = []

    def _counting(code, budget_s=None):
        calls.append(budget_s)
        return None

    monkeypatch.setattr(verify, "compile_defect", _counting)
    solver = VerifyingSolver(object())
    task = SimpleNamespace(
        language="rust", entrypoint="main", public_examples=[],
        statement="s", problem_id="p", deadline_s=300.0,
    )
    reply = "```rust\nfn main() { println!(\"hi\"); }\n```"

    solver._grade(reply, task, left=30.0)
    assert calls == [30.0], "a compile with budget left must still happen"

    calls.clear()
    solver._grade(reply, task, left=-4.0)
    solver._grade(reply, task, left=0.0)
    assert calls == [], f"paid for a compile past the deadline: {calls}"


@pytest.mark.parametrize("deadline", [40.0, 32.0, 25.0, 20.0, 15.0, 10.0, 5.0, 2.0])
def test_a_short_deadline_leaves_room_to_answer_from_the_real_numbers(deadline):
    """The same guarantee as the arithmetic test above, but read OUT of the code
    instead of recomputed beside it.

    That distinction is the whole point of this test existing separately: a test
    that re-derives the formula it is checking passes whatever the code does,
    and both short-deadline floors survived exactly that mistake here. The slice
    and the cap below are the numbers `solve_task` and `_attempt` actually
    produced, taken off the conversation they were handed to.
    """
    from solvers.browser_pool import tail_budget

    seen: list[tuple] = []

    class _Chat:
        provider = "claude"
        async def send(self, text, timeout_s, extend_to_s=None):
            seen.append((timeout_s, extend_to_s))
            return RIGHT
        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="short", language="python", statement=DIGITS.statement,
        entrypoint="g", public_examples=[], deadline_s=deadline,
    )
    asyncio.run(VerifyingSolver(_Fleet()).solve_task(task, timeout_s=deadline))
    assert seen, f"the model was never asked at all on a {deadline:g}s deadline"

    slice_s, cap = seen[0]
    # The read may run to `cap` (it extends there while the model is writing),
    # and the tail is sized from the slice.
    worst = (cap if cap else slice_s) + tail_budget(slice_s)
    assert worst < deadline, (
        f"a {deadline:g}s deadline hands out a {slice_s:.1f}s read capped at "
        f"{cap:.1f}s, then up to {tail_budget(slice_s):.1f}s of post-read "
        f"phases — {worst:.1f}s before the answer is even signed. "
        f"`handle_request` cancels the solve and the validator gets nothing."
    )


def test_the_post_read_phases_are_actually_bounded_by_the_short_read():
    """`tail_budget` being right is worth nothing if `send` does not apply it.

    The hang is injected rather than raced for, exactly as
    `test_one_unreturning_read_cannot_spend_the_whole_send` does: what must hold
    is that a two-second read cannot be followed by eleven seconds of rescue,
    whatever made the rescue slow.
    """
    playwright, chrome = _chromium_or_skip()
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
        '<div id="host"></div><script>'
        "document.getElementById('send').onclick = () => {"
        "  const d = document.createElement('div'); d.className='msg';"
        "  const pre = document.createElement('pre'); const code = document.createElement('code');"
        "  code.textContent = 'def pong():\\n    return 4';"
        "  pre.appendChild(code); d.appendChild(pre);"
        "  document.getElementById('host').appendChild(d);"
        "};</script>"
    )
    site = _site(assistant=("div.msg",), copy=('button[aria-label="Copy"]',), poll_s=0.2)

    async def go():
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)

            async def never_returns(before):
                await asyncio.sleep(3600)

            tab._copy_phase = never_returns
            started = time.monotonic()
            reply = await tab.send("solve it", 2.0)
            elapsed = time.monotonic() - started
            await browser.close()
            return reply, elapsed

    reply, elapsed = asyncio.run(go())
    assert "return 4" in extract_code(reply, "pong"), (
        f"the DOM reading has to survive a copy control that hangs: {reply!r}"
    )
    assert elapsed < 5.0, (
        f"a 2s read was followed by {elapsed - 2:.1f}s of post-read phases. On a "
        f"short deadline that alone outlives the whole request, and "
        f"`handle_request` answers 504 with nothing."
    )


def test_getting_the_prompt_in_never_outlives_the_read_itself():
    """The third place on this branch where a floor sat above what the caller
    could afford. `max(5.0, ...)` alone gave a one-second read a five-second
    submit — which does not buy a better submit, it buys an overrun, and on a
    short request `handle_request` answers 504 with nothing at all.

    A submit that fills the whole read is a read that finds nothing, which is
    bad. A submit that outlives it is a solve that is cancelled, which is worse.
    """
    playwright, chrome = _chromium_or_skip()
    # A composer that is never there, so the submit spends its whole allowance
    # looking for it.
    url = _served('<!doctype html><meta charset="utf-8"><div id="host"></div>')
    site = _site(composer=("#nothing",), send=("#nothing",), assistant=("div.msg",))

    async def go(slice_s):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            started = time.monotonic()
            await tab.send("solve it", slice_s)
            elapsed = time.monotonic() - started
            await browser.close()
            return elapsed

    elapsed = asyncio.run(go(1.5))
    assert elapsed < 4.0, (
        f"a 1.5s read spent {elapsed:.1f}s failing to submit. The whole request "
        f"may be shorter than that, and the solve is cancelled rather than late."
    )


def test_the_readme_table_matches_the_defaults_it_documents():
    """Documented defaults drift, and this branch drifted twice in one session:
    the table said 600 for `SOLVER_MAX_BUDGET_S` and `GLM_REQUEST_TIMEOUT_S`
    while the code had moved to 3600.

    That is worse than an out-of-date sentence. These are the knobs an operator
    reaches for when a miner is misbehaving, and a table that lies about the
    default sends them to change a value that was never the one in effect.
    """
    here = Path(__file__).parent
    readme = (here / "README.md").read_text("utf-8")
    roster = (here / "solvers" / "roster.py").read_text("utf-8")
    from solvers.config import DEFAULT_SOLVE_TIMEOUT_S

    documented = {
        name: value
        for name, value in re.findall(r"^\| `([A-Z_]+)` \| `([\d.]+)` \|", readme, re.M)
    }
    assert documented, "the environment table is gone or no longer parses"

    actual = {
        name: value
        for name, value in re.findall(
            r'os\.environ\.get\("([A-Z_]+)",\s*"([\d.]+)"\)', roster
        )
    }
    actual["GLM_REQUEST_TIMEOUT_S"] = DEFAULT_SOLVE_TIMEOUT_S

    wrong = {
        name: (said, actual[name])
        for name, said in documented.items()
        if name in actual and float(said) != float(actual[name])
    }
    assert not wrong, (
        "the README documents defaults the code does not use "
        + ", ".join(f"{n}: says {s}, is {a}" for n, (s, a) in sorted(wrong.items()))
    )


# --------------------------------------------------------------------------- #
# The CLI backend: the same subscription, reached without a browser.
#
# Driven against a FAKE `claude` binary rather than the real one. These tests
# have to run hundreds of times; the real CLI would spend the operator's
# subscription to prove things about argv and stdin that a stub proves exactly
# as well. The one thing a stub cannot prove -- that the real CLI answers
# correctly -- is proved by the rehearsal instead, which scored 2/2 on the
# gradeable challenges.
# --------------------------------------------------------------------------- #
_FAKE_CLI = r'''#!/usr/bin/env python3
"""A `claude` stand-in: emits the stream-json events the real one emits."""
import json, os, sys, time

argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

# Which account: the CLI keeps one sign-in per CLAUDE_CONFIG_DIR.
account = os.path.basename(os.environ.get("CLAUDE_CONFIG_DIR", "").rstrip("/")) or "default"
model = opt("--model")

# The mode, per account and model. FAKE_CLI_MODE is the default; a JSON file
# beside the log overrides it mid-test -- the backend fixes the child's
# environment when it is built, so a file is the only way to reach a later
# turn -- keyed "<account>|<model>", "<account>", "model:<model>" or "*".
mode = os.environ.get("FAKE_CLI_MODE", "ok")
if os.path.exists(os.environ["FAKE_CLI_LOG"] + ".modes"):
    table = json.load(open(os.environ["FAKE_CLI_LOG"] + ".modes"))
    for key in (account + "|" + str(model), account, "model:" + str(model), "*"):
        if key in table:
            mode = table[key]
            break

if argv and argv[0] == "auth":
    print(json.dumps({"loggedIn": mode != "unauth", "authMethod": "oauth_token"}))
    sys.exit(0)

prompt = sys.stdin.read()
session = opt("--session-id") or opt("--resume") or "?"

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

# What the fake was asked, echoed so the test can assert on it. Written
# FIRST, so a turn that then fails is still on the record.
record = {"prompt": prompt, "argv": argv, "session": session,
          "model": model, "effort": opt("--effort"), "account": account,
          "resumed": "--resume" in argv,
          "api_key": os.environ.get("ANTHROPIC_API_KEY"),
          "claudecode": os.environ.get("CLAUDECODE"),
          "cwd": os.getcwd(), "cwd_entries": sorted(os.listdir("."))}
with open(os.environ["FAKE_CLI_LOG"], "a") as fh:
    fh.write(json.dumps(record) + "\n")

if mode == "exit1":
    sys.stderr.write("No conversation found with session ID: " + session + "\n")
    sys.exit(1)
if mode == "unauth":
    sys.stderr.write("Not logged in \u00b7 Please run /login\n")
    sys.exit(1)
if mode == "server-error":
    emit({"type": "result", "is_error": True, "subtype": "error_during_execution",
          "result": "API Error: 500 Internal server error", "session_id": session})
    sys.exit(1)
if mode == "error-after-text":
    # The connection broke mid-answer: text arrived, then the CLI gave up.
    for piece in ("```python\n", "def g(n):\n"):
        emit({"type": "stream_event",
              "event": {"type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": piece}}})
    emit({"type": "result", "is_error": True, "subtype": "error_during_execution",
          "result": "API Error: fetch failed", "session_id": session})
    sys.exit(1)
if mode in ("overloaded", "overloaded0"):
    # The CLI retrying an overloaded request itself, one event per retry,
    # then waiting out a delay that only grows. `overloaded0` numbers its
    # attempts from zero, as nothing promises the real one will not.
    first = 0 if mode == "overloaded0" else 1
    for attempt, delay in ((first, 1000), (first + 1, 4000), (first + 2, 16000)):
        emit({"type": "system", "subtype": "api_retry", "attempt": attempt,
              "max_retries": 10, "retry_delay_ms": delay, "error_status": 529,
              "error": "API Error: 529 {\"type\":\"overloaded_error\"}"})
        time.sleep(delay / 1000.0)
    sys.exit(1)
if mode == "authrace":
    # Two miners racing to refresh one login's token. The CLI says so itself
    # -- "usually transient" -- and it is over in about a second. Read off a
    # production replay, where hopping on it spent the backup account's quota
    # to avoid a one-second wait.
    emit({"type": "system", "subtype": "api_retry", "attempt": 1,
          "max_retries": 10, "retry_delay_ms": 1000, "error_status": 500,
          "error": "authentication_failed: another Claude Code process "
                   "exited mid-refresh; this is usually transient"})
    time.sleep(0.2)
    for piece in ("```python\n", "def g(n):\n    return 1\n", "```"):
        emit({"type": "stream_event",
              "event": {"type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": piece}}})
    emit({"type": "result", "is_error": False, "session_id": session})
    sys.exit(0)
if mode == "silent":
    emit({"type": "result", "is_error": False, "session_id": session})
    sys.exit(0)
if mode == "latewrite":
    # A reply that is still being written when an ordinary slice would end.
    # The delay is however long SOLVER_FAKE_WRITE_S says; the answer that
    # follows is whole, which is the point -- cut at the slice it is a
    # fragment that cannot compile, read to the end it is a program.
    time.sleep(float(os.environ.get("SOLVER_FAKE_WRITE_S", "1")))
    emit({"type": "stream_event",
          "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta",
                              "text": "```python\ndef g(n):\n"}}})
    emit({"type": "stream_event",
          "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "    return n\n```"}}})
    emit({"type": "result", "is_error": False, "session_id": session})
    sys.exit(0)
if mode == "heartbeat":
    # The shape measured in production: events arriving forever, none of them
    # carrying a character of the answer. A keepalive, not a model writing.
    while True:
        emit({"type": "stream_event", "event": {"type": "ping"}})
        time.sleep(0.005)
if mode == "result-only":
    # No deltas at all: the whole reply arrives in the `result` event. The
    # backend used to open this event only to read `is_error`.
    emit({"type": "result", "is_error": False, "session_id": session,
          "result": "```python\ndef g(n):\n    return n\n```",
          "usage": {"input_tokens": 3, "output_tokens": 40}})
    sys.exit(0)
if mode == "assistant-only":
    # The complete message, with no partial-message deltas behind it.
    emit({"type": "assistant", "session_id": session, "message": {"content": [
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": "```python\ndef g(n):\n    return n\n```"}]}})
    emit({"type": "result", "is_error": False, "session_id": session})
    sys.exit(0)
if mode == "stall":
    # The real CLI at the subscription's usage limit: no event, ever.
    time.sleep(600)
if mode in ("ratelimit", "limited", "opus-limit", "overage"):
    # The event's shape as the CLI emits it: no utilisation at the top level,
    # one entry per rolling window under `unifiedWindows`.
    full = mode == "limited"
    info = {"status": "rejected" if full else "allowed_warning",
            "rateLimitType": "five_hour", "resetsAt": time.time() + 600,
            "overageStatus": "rejected", "isUsingOverage": False,
            "unifiedWindows": {
                "five_hour": {"utilization": 1.0 if full else 0.93,
                              "resetsAt": time.time() + 600},
                "seven_day": {"utilization": 0.42,
                              "resetsAt": time.time() + 86400}}}
    if mode == "opus-limit":
        # A weekly cap on ONE model: the seat's windows are fine.
        info.update({"status": "rejected", "rateLimitType": "seven_day_opus",
                     "resetsAt": time.time() + 3600})
        info["unifiedWindows"]["five_hour"]["utilization"] = 0.5
    if mode == "overage":
        # The window is spent and the plan's paid extra usage is covering it.
        info.update({"status": "allowed", "overageStatus": "allowed",
                     "isUsingOverage": True})
        info["unifiedWindows"]["five_hour"]["utilization"] = 1.0
    emit({"type": "rate_limit_event", "rate_limit_info": info})
    if full or (mode == "opus-limit" and opt("--model") == "opus"):
        emit({"type": "result", "is_error": True, "subtype": "error_rate_limit",
              "session_id": session})
        sys.exit(1)

for piece in ("```python\n", "def g(n):\n", "    return n\n", "```"):
    emit({"type": "stream_event",
          "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": piece}}})
    if mode == "slow":
        time.sleep(2.0)
emit({"type": "result", "is_error": False, "session_id": session,
      "total_cost_usd": 0.01,
      "usage": {"input_tokens": 3, "output_tokens": 40,
                "cache_creation_input_tokens": 500, "cache_read_input_tokens": 120,
                "output_tokens_details": {"thinking_tokens": 7}}})
'''


def _fake_cli(tmp_path, monkeypatch, mode="ok", backups=0):
    """Install the stub as SOLVER_CLI_BIN and return its call log path.

    `backups` names that many backup accounts, each a directory under
    `tmp_path` (`claude-2`, `claude-3`, ...), the way an operator would."""
    binary = tmp_path / "fake-claude"
    binary.write_text(_FAKE_CLI, encoding="utf-8")
    binary.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("SOLVER_CLI_BIN", str(binary))
    monkeypatch.setenv("SOLVER_CLI_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    monkeypatch.setenv("FAKE_CLI_MODE", mode)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    if backups:
        dirs = [tmp_path / f"claude-{n + 2}" for n in range(backups)]
        monkeypatch.setenv("SOLVER_CLI_BACKUP_ACCOUNTS", ",".join(map(str, dirs)))
    else:
        monkeypatch.delenv("SOLVER_CLI_BACKUP_ACCOUNTS", raising=False)
    # The SHIPPED ladder, pinned explicitly so a test that hops runs onto the
    # rung production runs onto. Two models, opus and fable, and no sonnet
    # anywhere -- the operator's decision. A test that needs a LONGER ladder
    # than production ships sets its own; this one may only ever be what the
    # miner really runs, or a hop test proves nothing about the miner.
    monkeypatch.setenv("SOLVER_CLI_EMERGENCY_PROFILES", "fable:low")
    monkeypatch.delenv("SOLVER_CLI_MODELS", raising=False)
    monkeypatch.delenv("SOLVER_CLI_RECOVERY_S", raising=False)
    return log


def _cli_modes(log, table):
    """Change what the fake does from here on, per account and model."""
    (log.parent / (log.name + ".modes")).write_text(json.dumps(table))


def _cli_calls(log):
    # A fake that fails or is refused exits before it records itself, so a
    # log that was never written means no child got as far as running.
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def test_the_cli_backend_carries_one_conversation_across_turns(tmp_path, monkeypatch):
    """The repair loop's whole premise: the model sees its own previous attempt
    beside the failure report. With a browser that is one tab; here it is one
    session id — created on the first turn, reopened with `--resume` on every
    one after."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        first = await conversation.send("turn one", 60.0)
        second = await conversation.send("turn two", 60.0)
        await conversation.close()
        return first, second

    first, second = asyncio.run(go())
    assert extract_code(first, "g") == "def g(n):\n    return n", first
    assert first == second

    calls = _cli_calls(log)
    assert len(calls) == 2, calls
    # The prompt goes on STDIN, not in argv: measured, the CLI otherwise waits
    # three seconds for input it is never given (5.2s against 2.4s for the
    # identical turn), and an argv-borne 63KB statement is bounded by ARG_MAX.
    assert calls[0]["prompt"] == "turn one" and calls[1]["prompt"] == "turn two"
    assert not any(a in ("turn one", "turn two") for a in calls[0]["argv"])
    # One session, created then resumed.
    assert calls[0]["session"] == calls[1]["session"]
    assert calls[0]["resumed"] is False and calls[1]["resumed"] is True


def test_the_cli_backend_never_hands_a_child_the_api_key(tmp_path, monkeypatch):
    """The whole reason this backend exists.

    The CLI resolves credentials in a fixed order and an API key OUTRANKS the
    subscription, so a stray `ANTHROPIC_API_KEY` would move every solve onto
    metered billing without a word — while answering exactly as well, which is
    what makes it invisible. The miner's own environment is left alone; only
    the child's is scrubbed.

    `CLAUDE_CODE_*` goes for a different reason, measured: launched from inside
    a Claude Code session the child inherited the PARENT's session id, so
    `--resume` would have appended every solve to the operator's own
    conversation.
    """
    from solvers.claude_cli import CliBackend, child_env

    log = _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "the-parent-session")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/srv/miner/.claude")

    env = child_env()
    assert "ANTHROPIC_API_KEY" not in env, "the child would bill the API key"
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env
    # Operator configuration is not this module's to edit -- and the config
    # dir is where the sign-in lives, so dropping it would sign the child out.
    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example"
    assert env["CLAUDE_CONFIG_DIR"] == "/srv/miner/.claude"
    assert "PATH" in env

    asyncio.run(_one_cli_turn(CliBackend()))
    seen = _cli_calls(log)[0]
    assert seen["api_key"] is None and seen["claudecode"] is None

    # ...and an operator who genuinely wants the key can have it back.
    monkeypatch.setenv("SOLVER_CLI_ALLOW_API_KEY", "1")
    assert child_env()["ANTHROPIC_API_KEY"] == "sk-ant-should-not-be-used"


async def _one_cli_turn(backend, prompt="solve it", timeout_s=60.0):
    conversation = await backend.open()
    try:
        return await conversation.send(prompt, timeout_s)
    finally:
        await conversation.close()


def test_a_long_cli_answer_is_not_lost_to_a_line_limit(tmp_path, monkeypatch):
    """The failure this had, found by asking what a 200KB answer does.

    `readline()` is bounded by the stream reader's limit -- 64KB by default --
    and the CLI's final `result` event embeds the whole answer a second time, so
    one long program puts a single line over it. Measured before the fix: the
    read did not merely lose the answer, it STALLED and then reported the turn
    unfinished after the entire slice. Silent, slow and total, which is the
    worst shape a failure can have.

    Reading in chunks and splitting lines here removes the limit rather than
    raising it, because a raised limit only moves the cliff.
    """
    from solvers.claude_cli import CliBackend

    binary = tmp_path / "long-claude"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "big = 'x' * 200000\n"
        "d = {'type':'stream_event','event':{'type':'content_block_delta',"
        "'delta':{'type':'text_delta','text':'```python\\n'+big+'\\n```'}}}\n"
        "sys.stdout.write(json.dumps(d)+'\\n')\n"
        "sys.stdout.write(json.dumps({'type':'result','is_error':False,"
        "'result':'```python\\n'+big+'\\n```'})+'\\n')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("SOLVER_CLI_BIN", str(binary))

    async def go():
        backend = CliBackend()
        conversation = await backend.open()
        try:
            started = time.monotonic()
            body = await conversation.send("solve it", 60.0)
            return body, time.monotonic() - started, conversation.empty_reason
        finally:
            await conversation.close()

    body, spent, reason = asyncio.run(go())
    assert len(body) > 199_000, f"a long answer came back as {len(body)} chars"
    assert spent < 20.0, (
        f"took {spent:.0f}s of a 60s slice — the read stalled rather than "
        f"failing, which spends the deadline and submits nothing"
    )
    assert reason is None


def test_a_cli_turn_cut_off_keeps_what_arrived(tmp_path, monkeypatch):
    """The same rule every other read here keeps: submit the part that arrived.

    This is why the backend reads `stream-json` rather than `json`. The single
    JSON result emits nothing until the turn ends, so a deadline landing
    mid-answer would be a total loss; the event stream hands over the text
    already written. `still_writing` is what then stops the repair loop asking a
    model that never finished to fix what it did not say.
    """
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="slow")
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        body = await conversation.send("solve it", 3.0)
        spent = time.monotonic() - started
        return body, spent, conversation.still_writing, conversation.empty_reason

    body, spent, writing, reason = asyncio.run(go())
    assert spent < 10.0, f"a 3s slice took {spent:.1f}s"
    assert writing is True, "a turn killed mid-answer must read as unfinished"
    assert body, "threw away the text that had already arrived"
    assert body.startswith("```python"), body
    assert reason is None, "text arrived, so nothing is missing to explain"


def test_a_cli_session_that_fails_is_told_apart_from_one_that_says_nothing(
    tmp_path, monkeypatch
):
    """Two empty turns that need opposite handling, and `_attempt` reads the
    difference off `empty_reason`.

    A non-zero exit is the SESSION failing — a lost conversation, a refused
    resume, a broken install — and the loop answers that by carrying the repair
    to a fresh one. A clean exit with no text is the model declining to say
    anything, which is the conversation working and the turn being wasted.
    """
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="exit1")
    conversation_reason = _cli_reason(CliBackend())
    assert conversation_reason == "unreadable", conversation_reason

    _fake_cli(tmp_path, monkeypatch, mode="silent")
    assert _cli_reason(CliBackend()) == "no-code"

    # A binary that is not there at all must be a failed turn, never a crash:
    # this runs inside a solve, and an exception here would surface as a dead
    # backend for the whole pass.
    monkeypatch.setenv("SOLVER_CLI_BIN", str(tmp_path / "does-not-exist"))
    assert _cli_reason(CliBackend()) == "unreadable"


def _cli_reason(backend):
    async def go():
        conversation = await backend.open()
        body = await conversation.send("solve it", 30.0)
        assert not body, body
        return conversation.empty_reason
    return asyncio.run(go())


def test_the_cli_backend_asks_a_different_model_for_a_second_opinion(
    tmp_path, monkeypatch
):
    """`avoid` is how a second pass reaches something other than what just
    failed. A browser fleet answers it with another account; here the better
    answer is another model, and it costs nothing to offer."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_MODELS", "opus,sonnet")
    backend = CliBackend()

    async def go():
        first = await backend.open()
        second = await backend.open(avoid=first.provider)
        return first.provider, second.provider, first._session, second._session

    a, b, sid_a, sid_b = asyncio.run(go())
    assert a == "cli:opus" and b == "cli:sonnet", (a, b)
    assert sid_a != sid_b, "two conversations shared one session id"


def test_the_cli_child_is_given_no_tools_and_no_customisations(
    tmp_path, monkeypatch
):
    """What the child is allowed to be, checked as argv because that is the only
    place it is decided.

    Three of these are load-bearing. `--tools ""` because the answer is text and
    a tool call is a way for the turn to end without one. `--safe-mode` because
    the operator's CLAUDE.md, skills and hooks are not part of this task and
    could only steer it. And NOT `--bare`, whose own help says Anthropic auth is
    then strictly an API key and OAuth is never read — precisely backwards for a
    subscription.
    """
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_EFFORT", "low")
    asyncio.run(_one_cli_turn(CliBackend()))
    argv = _cli_calls(log)[0]["argv"]

    assert "--bare" not in argv, (
        "--bare makes the CLI read an API key and never OAuth, which defeats "
        "the entire purpose of this backend"
    )
    assert argv[argv.index("--tools") + 1] == ""
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
                 "--include-partial-messages"):
        assert flag in argv, f"{flag} missing from {argv}"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    # A system prompt of our own: the default one is a coding AGENT's, all
    # tools and files and git, and none of it applies to "write one program".
    assert "--system-prompt" in argv
    # An empty working directory, so there is nothing in scope to read, and
    # the SAME one for every child: the CLI keys its per-project state on the
    # cwd, and a temp dir per conversation left a new `~/.claude/projects/`
    # entry behind for every solve.
    asyncio.run(_one_cli_turn(CliBackend()))
    calls = _cli_calls(log)
    assert calls[0]["cwd_entries"] == [], calls[0]["cwd_entries"]
    assert calls[0]["cwd"] == calls[1]["cwd"] == str(tmp_path / "work"), calls


def test_the_cli_backend_says_when_the_subscription_is_running_out(
    tmp_path, monkeypatch, capsys
):
    """The one failure mode a subscription has that an API key does not.

    Past the limit every solve fails identically, for a reason no other line of
    the log would name. The CLI reports it on the event stream and this is the
    only place it is ever seen."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="ratelimit")
    backend = CliBackend()
    asyncio.run(_one_cli_turn(backend))
    asyncio.run(_one_cli_turn(backend))
    out = capsys.readouterr().out

    assert "93% of the five_hour subscription limit" in out, out
    assert "resets in about" in out, out
    # Once per tenth of the budget, not once per turn: two turns at the same
    # utilisation say it once. And only the window that is running out: the
    # seven-day one at 42% has nothing to say.
    assert out.count("subscription limit used") == 1, out
    assert "seven_day" not in out, out
    # A warning is not the limit. Both turns went through.
    assert backend.limited_for() == 0
    assert backend.stats()["turns"] == 2


def test_at_the_subscription_limit_every_solve_is_turned_away_at_once(
    tmp_path, monkeypatch, capsys
):
    """One solve discovers the limit; every solve after it, until the reset, is
    told in a millisecond rather than each spending its slice finding out.

    Backend-wide, because the limit is: a fresh conversation, a different
    model, a second opinion — all the same seat, all equally spent."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, mode="limited")
    backend = CliBackend()

    async def go():
        first = await backend.open()
        body = await first.send("solve it", 60.0)
        reason = first.empty_reason
        # The second opinion, on the other model, and a third solve entirely.
        second = await backend.open(avoid=first.provider)
        started = time.monotonic()
        again = await second.send("solve it", 60.0)
        spent = time.monotonic() - started
        return body, reason, again, second.empty_reason, spent

    body, reason, again, reason2, spent = asyncio.run(go())
    assert body == "" and reason == "unreadable", (body, reason)
    assert again == "" and reason2 == "unreadable", (again, reason2)
    assert spent < 0.5, f"a known limit still cost {spent:.1f}s to rediscover"
    assert len(_cli_calls(log)) <= 1, "the second solve started a child"
    # Until the CLI's own reset time, which it reported as ten minutes out.
    assert 500 < backend.limited_for() <= 600, backend.limited_for()
    assert 500 < backend.limited_for("sonnet") <= 600
    out = capsys.readouterr().out
    assert "OUT for" in out and "account primary, every model -- limit" in out, out
    assert "turned away" in out, out


def test_a_limit_on_one_model_leaves_the_other_answering(
    tmp_path, monkeypatch, capsys
):
    """The CLI's schema names weekly windows for ONE model -- `seven_day_opus`,
    `seven_day_sonnet` -- and a limit on Opus is not a limit on Sonnet. Read
    as a limit on the seat it would turn away the model that still works."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, mode="opus-limit")
    monkeypatch.setenv("SOLVER_CLI_MODELS", "opus,sonnet")
    # Sonnet is the SUBJECT here, not a preference: `seven_day_sonnet` is a
    # window name in the CLI's own schema, and reading it as a limit on the
    # seat would turn away a model that still works. The miner does not run
    # sonnet, and it still has to understand what the CLI says about it -- so
    # this test names the ladder it needs rather than inheriting the shipped
    # one, which is opus then fable.
    monkeypatch.setenv("SOLVER_CLI_EMERGENCY_PROFILES", "sonnet:low")
    backend = CliBackend()

    async def go():
        first = await backend.open()
        body = await first.send("solve it", 60.0)
        # A second solve: the ladder already knows, and goes straight there.
        second = await backend.open()
        again = await second.send("solve it", 60.0)
        return body, first, again, second

    body, first, again, second = asyncio.run(go())
    # The turn itself moved on: opus refused, sonnet answered, same slice.
    assert extract_code(body, "g") == "def g(n):\n    return n", body
    assert first.provider == "cli:sonnet" and first.hops == 1
    assert extract_code(again, "g") == "def g(n):\n    return n", again
    assert second.provider == "cli:sonnet" and second.hops == 0
    # The CLI said an hour; the backend takes its word for half of that at
    # most before one solve is let through to check (`LIMIT_RECHECK_S`).
    assert 1700 < backend.limited_for("opus") <= 1800, backend.limited_for("opus")
    assert backend.limited_for("sonnet") == 0
    assert backend.limited_for("claude-opus-5") > 1700, "a full model id"
    calls = _cli_calls(log)
    assert [c["model"] for c in calls] == ["opus", "sonnet", "sonnet"], calls
    # The same report on sonnet's turn was the same limit, not news.
    out = capsys.readouterr().out
    assert out.count("OUT for") == 1, out
    assert "EMERGENCY MODE: cli:sonnet" in out, out


def test_paid_extra_usage_is_off_unless_asked_for(tmp_path, monkeypatch, capsys):
    """Extra usage is metered billing, and this backend's whole premise is a
    seat that is already paid for. So it is treated as the limit unless the
    operator says otherwise, exactly as the API key is."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="overage")
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        body = await conversation.send("solve it", 60.0)
        return body, conversation.empty_reason

    body, reason = asyncio.run(go())
    # The fake reports overage on every model, so every rung refuses.
    assert body == "" and reason == "unreadable", (body, reason)
    assert backend.limited_for("opus") > 0
    out = capsys.readouterr().out
    assert "extra usage is not enabled" in out, out

    monkeypatch.setenv("SOLVER_CLI_ALLOW_OVERAGE", "1")
    backend = CliBackend()
    body, reason = asyncio.run(go())
    assert extract_code(body, "g") == "def g(n):\n    return n", body
    assert backend.limited_for("opus") == 0
    assert "paid extra usage" in capsys.readouterr().out


def test_a_cli_that_hangs_without_a_word_is_given_up_on_quickly(
    tmp_path, monkeypatch, capsys
):
    """Measured at the usage limit: the CLI blocks silently, no event, no exit.

    Left to the slice, that is a whole deadline burnt per solve with nothing to
    show and no line of log to say why. A working turn emits its first event in
    under a second, so a turn that has said nothing at all inside the first
    window is not thinking, it is wedged."""
    from solvers import claude_cli

    _fake_cli(tmp_path, monkeypatch, mode="stall")
    monkeypatch.setattr(claude_cli, "FIRST_EVENT_S", 1.0)
    backend = claude_cli.CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        body = await conversation.send("solve it", 60.0)
        return body, conversation.empty_reason, time.monotonic() - started

    body, reason, spent = asyncio.run(go())
    assert body == "" and reason == "unreadable", (body, reason)
    # Every rung of the ladder was tried -- a stall is one pair's until
    # proven otherwise -- and each cost the first-event window, no more.
    assert spent < 12.0, f"a silent CLI held the slice for {spent:.1f}s"
    # Two rungs now, not three: the shipped ladder is opus then fable.
    assert backend.stats()["stalls"] == 2
    out = capsys.readouterr().out
    assert "produced no event at all" in out
    assert "hop: cli:opus -> cli:fable" in out, out


def test_a_failed_first_cli_turn_does_not_reuse_its_session_id(
    tmp_path, monkeypatch
):
    """`--session-id` on an id that already exists is a hard error (measured:
    "already in use", exit 1), and a first turn that failed may or may not have
    created it. So a failed first turn takes a fresh id: the conversation was
    empty either way, and a fresh one cannot collide."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, mode="exit1")
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        failed = conversation._session
        await conversation.send("turn one", 60.0)
        _cli_modes(log, {"*": "ok"})
        await conversation.send("turn two", 60.0)
        await conversation.send("turn three", 60.0)
        return failed

    failed = asyncio.run(go())
    calls = _cli_calls(log)
    assert [c["resumed"] for c in calls] == [False, False, True], calls
    assert calls[0]["session"] == failed
    assert calls[1]["session"] != failed, "reused the id of a failed first turn"
    assert calls[1]["session"] == calls[2]["session"], "the session was lost"


def test_waiting_for_a_cli_slot_counts_against_the_slice(tmp_path, monkeypatch):
    """The slot is acquired INSIDE the slice. Acquired outside it, a solve queued
    behind the others waited with no bound at all, and the wait was invisible
    to every clock in `verify.py`."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="slow")
    backend = CliBackend(concurrency=1)

    async def go():
        holder = await backend.open()
        waiter = await backend.open()
        holding = asyncio.ensure_future(holder.send("first", 60.0))
        await asyncio.sleep(0.5)
        started = time.monotonic()
        body = await waiter.send("second", 2.0)
        spent = time.monotonic() - started
        await holding
        return body, waiter.empty_reason, spent

    body, reason, spent = asyncio.run(go())
    assert body == "" and reason == "unreadable", (body, reason)
    assert spent < 4.0, f"a 2s slice waited {spent:.1f}s for a slot"


def test_a_usage_limit_moves_the_solve_to_the_backup_account(
    tmp_path, monkeypatch, capsys
):
    """The first rung. A usage limit is the ACCOUNT's, so the answer is the
    other account on the same model -- inside the same turn, since a session
    that holds nothing yet loses nothing by moving."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    _cli_modes(log, {"default": "limited", "*": "ok"})
    backend = CliBackend()
    assert [a.name for a in backend.accounts] == ["primary", "claude-2"]

    async def go():
        first = await backend.open()
        body = await first.send("solve it", 60.0)
        more = await first.send("fix it", 60.0)
        # The next solve does not rediscover the limit.
        second = await backend.open()
        again = await second.send("solve it", 60.0)
        return first, body, more, second, again

    first, body, more, second, again = asyncio.run(go())
    assert extract_code(body, "g") == "def g(n):\n    return n", body
    assert first.provider == "cli:opus@claude-2" and first.hops == 1
    assert extract_code(more, "g"), more
    assert second.provider == "cli:opus@claude-2" and second.hops == 0
    assert extract_code(again, "g"), again
    calls = _cli_calls(log)
    assert [(c["account"], c["model"], c["resumed"]) for c in calls] == [
        ("default", "opus", False),   # refused
        ("claude-2", "opus", False),  # the hop: fresh session on the other seat
        ("claude-2", "opus", True),   # the repair turn resumes IT
        ("claude-2", "opus", False),  # the next solve goes straight there
    ], calls
    assert calls[1]["session"] == calls[2]["session"]
    assert calls[0]["session"] != calls[1]["session"], "a session cannot change seats"
    out = capsys.readouterr().out
    assert "hop: cli:opus@primary -> cli:opus@claude-2" in out, out
    assert "EMERGENCY MODE: cli:opus@claude-2" in out and "limit" in out, out
    assert "primary/*" in backend.stats()["out"], backend.stats()


def test_a_limit_mid_conversation_hands_the_repair_to_a_fresh_one(
    tmp_path, monkeypatch
):
    """A session lives in its account's config directory and cannot follow a
    hop, so a limit that lands on a LATER turn ends the conversation as
    unreadable -- and `VerifyingSolver`'s fresh-conversation path, asking
    `open(avoid=<that provider>)`, lands on the other seat with the SAME model:
    the failed pair is out, so any healthy pair is the answer, not a second
    opinion."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        first = await conversation.send("solve it", 60.0)
        _cli_modes(log, {"default": "limited", "*": "ok"})
        second = await conversation.send("fix it", 60.0)
        carried = await backend.open(avoid=conversation.provider)
        return first, second, conversation, carried

    first, second, conversation, carried = asyncio.run(go())
    assert extract_code(first, "g"), first
    assert second == "" and conversation.empty_reason == "unreadable"
    assert conversation.hops == 0, "a started session must not change seats"
    assert conversation.provider == "cli:opus@primary"
    assert carried.provider == "cli:opus@claude-2", carried.provider


def test_a_first_round_fragment_is_kept_rather_than_crashing_the_solve():
    """REGRESSION, off a replay of the 97 recorded tasks.

    `_supersedes` answers "should this displace the answer in hand", and two of
    its rules read `best` to decide. On the FIRST round there is no `best` --
    the caller guards `best is None`, but in the second conjunct of an `and`,
    so this runs first. A candidate turn that was still writing at the budget
    and left a fragment behind dereferenced None and raised straight out of
    the round:

        [verify] the solve failed: AttributeError: 'NoneType' object has no
                 attribute 'code'
        [rehearse] submitted 0 chars of python

    5,489 characters were in hand and went in the bin. A crash is the one
    outcome this function exists to prevent, and it turned a partial answer --
    which might have scored -- into a certain zero.
    """
    from solvers.verify import Candidate, _supersedes

    fragment = Candidate(code="def g(n):\n    while n > 0:", raw="...")

    # Both rules that read `best`, with nothing to read.
    assert _supersedes(fragment, None, True) is True
    fragment.partial = True
    assert _supersedes(fragment, None, False) is True
    # ...and the one rule that does not: an empty capture is still not an
    # answer, with or without something to beat.
    assert _supersedes(Candidate(code="", raw=""), None, True) is False


def test_a_still_writing_first_round_submits_the_part_that_arrived(capsys):
    """The same bug through a whole solve, which is where it was found.

    The candidate turn runs out of budget with a partial program in hand and
    no inputs to check it against. That must end the pass as a cutoff and
    SUBMIT the fragment -- a partial answer can score, and a crash cannot."""
    from solvers import verify

    partial = "```python\ndef g(n):\n    total = 0\n    while n > 0:\n```"

    class _Unfinished(_Chat):
        still_writing = False

        async def send(self, text, timeout_s, extend_to_s=None):
            phase = _phase_of(text)
            if phase in ("analysis", "inputs", "oracle"):
                return ""
            # The candidate turn: the model is still writing when the budget
            # is gone, and what arrived is a fragment.
            self.still_writing = True
            self.empty_reason = "unfinished"
            return partial

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Unfinished(self._script, self._provider)

    task = SolveTask(problem_id="frag", language="python",
                     statement="Return the sum of the decimal digits of n.",
                     entrypoint="g", public_examples=[], deadline_s=30.0)
    solver = verify.VerifyingSolver(_Backend2([]), reserve_s=0, max_budget_s=30,
                                    second_opinion=False)
    answer = asyncio.run(solver.solve_task(task, timeout_s=30.0))
    out = capsys.readouterr().out

    assert "the solve failed" not in out, out
    assert "AttributeError" not in out, out
    assert "while n > 0" in answer.code, (
        f"threw away the fragment that was in hand: {answer.code!r}\n{out}"
    )
    assert "still writing when the budget ran out" in out, out


def test_a_seat_with_nothing_left_says_so_instead_of_reading_as_recovery(
    tmp_path, monkeypatch, capsys
):
    """REGRESSION, and the worst line this backend could print.

    `pick` hands out the DEFAULT pair when nothing on the ladder is healthy --
    deliberately, so `send` turns it away with the reason in a millisecond
    rather than spawning a process that cannot work. `_announce` compared only
    the (account, model) pair, so it could not tell that fall-through from a
    genuine recovery.

    The sequence that produced it, measured on one account: a 529 storm parks
    opus, `fable` takes over, EMERGENCY MODE is printed and the mode is now
    fable. The seat then spends its five-hour window, EVERY model goes out,
    `pick` falls back to opus -- and because opus is the default pair, the line
    that reached the operator at the moment nothing could answer was

        [cli] back to normal: cli:opus (effort low) answers again

    An operator reading that goes back to sleep while every solve scores zero.
    The announced state is now (pair, can it serve), so going fully out is a
    change and says what it costs.
    """
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()

    # 1. A refused model: the emergency rung answers, on this one account.
    backend.note_degraded("opus", "529 overloaded")
    backend._announce(*backend.pick())
    assert "EMERGENCY MODE" in capsys.readouterr().out

    # 2. The whole seat is spent. Nothing on the ladder can answer.
    backend.note_limit(backend.accounts[0], "*", time.time() + 3600, "five_hour")
    assert backend._healthy() == [], "this reproduction needs an empty ladder"
    backend._announce(*backend.pick())
    out = capsys.readouterr().out
    assert "back to normal" not in out, (
        "announced a recovery at the moment the seat went fully out:\n" + out
    )
    assert "NOTHING CAN ANSWER" in out, out
    # It says what it COSTS, and -- on a single seat -- what fixes it.
    assert "score zero" in out and "SOLVER_CLI_BACKUP_ACCOUNTS" in out, out

    # 3. Saying it once is the rule everywhere else here; it holds.
    backend._announce(*backend.pick())
    assert capsys.readouterr().out == "", "repeated the same state"

    # 4. And a seat that comes BACK from fully out still announces recovery --
    #    the guard must not latch.
    backend._out.clear()
    backend._announce(*backend.pick())
    assert "back to normal" in capsys.readouterr().out


def test_solver_status_says_which_rung_is_answering(tmp_path, monkeypatch, capsys):
    """`out` says what is BROKEN; it cannot say who took over.

    Most of its entries are neither a limit nor an emergency -- a wedged pair,
    a refused model, a signed-out seat -- and a seat can be steered away from
    at 95% of its window with `out` completely empty. So the one question an
    operator asks of /solver-status, "am I on the emergency rung right now?",
    had no answer there: the current pair was printed by `_announce` and kept
    nowhere a machine could read it."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()

    assert backend.stats()["answering"] == {
        "account": "primary", "model": "opus",
        "serving": True, "is_default": True,
    }

    # The emergency rung, and `is_default` is the flag to alert on.
    backend.note_degraded("opus", "529 overloaded")
    backend._announce(*backend.pick())
    answering = backend.stats()["answering"]
    assert answering["model"] == "fable" and answering["is_default"] is False
    assert answering["serving"] is True, "fable can answer; it is not an outage"

    # Nothing left: the pair is handed out to be turned away, and says so.
    backend.note_limit(backend.accounts[0], "*", time.time() + 3600, "five_hour")
    backend._announce(*backend.pick())
    assert backend.stats()["answering"]["serving"] is False

    capsys.readouterr()


def test_an_overloaded_model_hops_to_the_emergency_profile_and_keeps_the_session(
    tmp_path, monkeypatch, capsys
):
    """The second rung. The CLI retries an overloaded request itself and says
    so on the stream, once per retry with the status; after the second the
    next wait is already seconds long and a different model will answer
    sooner. A model can change under a session, so the repair turn keeps its
    history -- `--resume` on the same id, now with `--model sonnet --effort
    high`."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        first = await conversation.send("solve it", 60.0)
        _cli_modes(log, {"model:opus": "overloaded", "*": "ok"})
        started = time.monotonic()
        second = await conversation.send("fix it", 60.0)
        spent = time.monotonic() - started
        after = await backend.open()
        return first, second, spent, conversation, after

    first, second, spent, conversation, after = asyncio.run(go())
    assert extract_code(first, "g") and extract_code(second, "g"), (first, second)
    assert conversation.provider == "cli:fable" and conversation.hops == 1
    # The EFFORT comes with the rung, not from the conversation it hopped out
    # of: taken from the ladder rather than written here, so this keeps saying
    # what it means when the ladder's efforts change.
    from solvers.claude_cli import cli_emergency_profiles

    assert conversation.effort == cli_emergency_profiles("low")[0].effort
    # ONE account, so there is nowhere for the turn to go and the wait is the
    # right answer: the first retry is sat through, the second is the signal,
    # and the 4s and 16s behind it are never paid. Moving on is only better
    # than waiting when somewhere else can actually take the turn -- another
    # MODEL on this same sign-in shares its processes and its quota, so it is
    # not somewhere else. See
    # `test_a_retry_is_not_waited_out_when_another_seat_is_free`.
    assert 0.9 < spent < 5.0, spent
    calls = _cli_calls(log)
    assert [(c["model"], c["effort"], c["resumed"]) for c in calls] == [
        ("opus", "low", False), ("opus", "low", True), ("fable", "low", True),
    ], calls
    assert len({c["session"] for c in calls}) == 1, "the session was lost in the hop"
    # Every account, ten minutes: the service's problem, not a seat's -- and
    # reached on the model's OWN evidence, two retries in, rather than on the
    # first one. `_Busy` moves a turn without recording anything; this is the
    # other path, where the retries ran out.
    out_table = backend.stats()["out"]
    assert "*/opus" in out_table and 500 < out_table["*/opus"]["seconds"] <= 600, out_table
    assert "refused" in out_table["*/opus"]["why"] and "529" in out_table["*/opus"]["why"]
    # The next fresh solve goes straight to the emergency profile.
    assert after.provider == "cli:fable"
    assert after.effort == cli_emergency_profiles("low")[0].effort
    out = capsys.readouterr().out
    assert "hop: cli:opus -> cli:fable" in out, out
    assert (
        f"EMERGENCY MODE: cli:fable "
        f"(effort {cli_emergency_profiles('low')[0].effort})"
    ) in out, out


def test_a_server_error_result_is_read_the_same_way(tmp_path, monkeypatch):
    """The CLI giving up outright -- a `result` with `is_error` and a 5xx in
    its text -- is the same refusal as a retry storm, and moves the same way."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    _cli_modes(log, {"model:opus": "server-error", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        return await conversation.send("solve it", 60.0), conversation

    body, conversation = asyncio.run(go())
    assert extract_code(body, "g"), body
    assert conversation.provider == "cli:fable"
    assert "*/opus" in backend.stats()["out"]


def test_the_default_model_is_tried_again_after_the_recovery_window(
    tmp_path, monkeypatch, capsys
):
    """Recovery is not a background probe, it is the next real solve. When a
    refused model's window expires the ladder puts it back on top, the next
    solve asks it, and a failure re-parks it for another window."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_RECOVERY_S", "1")
    _cli_modes(log, {"model:opus": "server-error", "*": "ok"})
    backend = CliBackend()

    async def go():
        first = await backend.open()
        await first.send("solve it", 60.0)
        parked = await backend.open()           # still out: emergency profile
        await asyncio.sleep(1.2)                # the window passes
        _cli_modes(log, {"*": "ok"})            # and the service recovered
        probe = await backend.open()            # ...so the next solve tries it
        answer = await probe.send("solve it", 60.0)
        _cli_modes(log, {"model:opus": "server-error", "*": "ok"})
        await asyncio.sleep(1.2)
        again = await backend.open()            # tried again, fails again
        await again.send("solve it", 60.0)
        parked_again = await backend.open()
        return parked, probe, answer, again, parked_again

    parked, probe, answer, again, parked_again = asyncio.run(go())
    assert parked.provider == "cli:fable"
    assert probe.provider == "cli:opus" and extract_code(answer, "g"), answer
    # `again` was handed opus -- the window had passed -- and hopped inside
    # its own turn; the solve after it is parked from the start.
    assert again.provider == "cli:fable" and again.hops == 1
    assert parked_again.provider == "cli:fable" and parked_again.hops == 0
    out = capsys.readouterr().out
    assert out.count("EMERGENCY MODE") == 2, out
    assert out.count("back to normal") == 1, out
    assert out.count("OUT for") == 2, out
    assert not out.startswith("[cli] back to normal"), "nothing to come back from"


def test_a_signed_out_backup_is_reported_at_launch_and_skipped(
    tmp_path, monkeypatch, capsys
):
    """A backup that is not signed in is not fatal -- the primary answers --
    but it is not a backup either, and the operator hears that at launch with
    the command that fixes it, not at the limit."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    _cli_modes(log, {"claude-2": "unauth", "*": "ok"})
    backend = CliBackend()
    asyncio.run(backend.start())
    out = capsys.readouterr().out
    assert "WARNING: backup account 'claude-2'" in out, out
    assert f"python -m solvers.claude_cli login {tmp_path / 'claude-2'}" in out, out
    assert "claude-2/*" in backend.stats()["out"]

    # And at the limit, with the backup out too, the ladder still moves:
    # the primary's next PROFILE rather than the seat that is not there.
    _cli_modes(log, {"claude-2": "unauth", "default|opus": "limited", "*": "ok"})

    async def go():
        conversation = await backend.open()
        return await conversation.send("solve it", 60.0), conversation

    body, conversation = asyncio.run(go())
    assert body == "" and conversation.empty_reason == "unreadable", body
    assert [c["account"] for c in _cli_calls(log)] == ["default"], _cli_calls(log)


def test_a_signed_out_account_found_mid_run_is_set_aside(tmp_path, monkeypatch):
    """The CLI's own words -- "Not logged in" on stderr -- sort the failure:
    an account's, not a model's, so the ladder moves seats and not models."""
    from solvers.claude_cli import CliBackend, classify

    assert classify("Not logged in · Please run /login") == "auth"
    assert classify('API Error: 529 {"type":"overloaded_error"}') == "server"
    assert classify("API Error: 500 Internal server error") == "server"
    assert classify("fetch failed") == "server"
    # A bad model alias, as the CLI words it: another model is the answer.
    assert classify("There's an issue with the selected model (nonsense). It may "
                    "not exist or you may not have access to it.") == "server"
    assert classify("[claude-code:unrecognized_model] {}") == "server"
    assert classify("Rate limit reached, resets at 5pm") == "limit"
    assert classify("No conversation found with session ID: 3f0e") is None
    assert classify("") is None
    # A three-digit number is a status only beside a word that makes it one.
    # Each of these once parked a model or a seat.
    assert classify("    at Object.<anonymous> (/opt/cli.js:512:98765)") is None
    assert classify("request took 503 ms and was aborted") is None
    assert classify("127.0.0.1:401 refused the tunnel") is None
    assert classify("HTTP 500") == "server" and classify("status 429") == "limit"

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    _cli_modes(log, {"default": "unauth", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        return await conversation.send("solve it", 60.0), conversation

    body, conversation = asyncio.run(go())
    assert extract_code(body, "g"), body
    assert conversation.provider == "cli:opus@claude-2"
    assert "primary/*" in backend.stats()["out"]
    assert "signed out" in backend.stats()["out"]["primary/*"]["why"]


def test_text_that_arrived_before_the_turn_failed_is_an_unfinished_reply(
    tmp_path, monkeypatch
):
    """The connection breaks mid-answer and the CLI gives up: what arrived is
    a fragment, kept and marked unfinished, so the repair loop does not ask
    the model to fix what it never finished saying."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="error-after-text")
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        return await conversation.send("solve it", 60.0), conversation

    body, conversation = asyncio.run(go())
    assert body.startswith("```python"), body
    assert conversation.still_writing is True
    assert conversation.empty_reason is None
    assert conversation.hops == 0, "a fragment in hand is not a reason to hop"


def test_the_cli_retry_events_are_counted_here_not_read_off_the_event(
    tmp_path, monkeypatch
):
    """Retries are counted per TURN, whatever base the CLI numbers them from:
    the second retry event this turn is the signal, one wait paid, not two.

    `overloaded0` numbers its attempts from zero, as nothing promises the real
    CLI will not.
    """
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    _cli_modes(log, {"model:opus": "overloaded0", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        body = await conversation.send("solve it", 60.0)
        return body, conversation, time.monotonic() - started

    body, conversation, spent = asyncio.run(go())
    assert extract_code(body, "g"), body
    assert conversation.provider == "cli:fable" and conversation.hops == 1
    # One account, so the wait is paid: one retry sat through, the second the
    # signal. What this is really about is that the count belongs to this
    # TURN rather than to the event's own `attempt` field.
    assert 0.9 < spent < 5.0, spent


def test_extra_usage_is_the_seats_whatever_window_the_event_named(
    tmp_path, monkeypatch
):
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()
    info = {"status": "rejected", "rateLimitType": "seven_day_opus",
            "isUsingOverage": True, "resetsAt": time.time() + 600}
    assert backend.note_rate_limit(info, "opus") is True
    # Not just opus: a hop to sonnet on this seat would start a metered request.
    assert backend.limited_for("sonnet") > 0


def test_the_status_command_names_each_account(tmp_path, monkeypatch, capsys):
    from solvers import claude_cli

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    _cli_modes(log, {"claude-2": "unauth", "*": "ok"})
    assert claude_cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert "primary" in out and "signed in (subscription)" in out, out
    assert "claude-2" in out and "NOT signed in" in out, out
    assert "ladder: opus/low, fable/low" in out, out

    _cli_modes(log, {"*": "ok"})
    assert claude_cli.main(["status"]) == 0


def test_the_operators_opinion_model_answers_at_the_default_effort(
    tmp_path, monkeypatch
):
    """With `SOLVER_CLI_MODELS=opus,sonnet` the ladder holds sonnet twice --
    the emergency rung at high effort and the operator's opinion rung at the
    default -- and a second opinion is the operator's rung, as documented."""
    from solvers.claude_cli import CliBackend, Profile

    _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_MODELS", "opus,sonnet")
    backend = CliBackend()
    assert backend.pick(avoid="cli:opus")[1] == Profile("sonnet", "low")
    # An unexplained failure elsewhere does not turn an opinion request into
    # a retry on the pair it named.
    backend.note_failure(backend.accounts[0], "opus")
    assert backend.pick(avoid="cli:opus")[1].model != "opus"


def test_a_spent_seat_window_widens_a_limit_reported_on_one_model(
    tmp_path, monkeypatch
):
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()
    info = {"status": "rejected", "rateLimitType": "seven_day_opus",
            "resetsAt": time.time() + 3600,
            "unifiedWindows": {"five_hour": {"utilization": 1.0,
                                             "resetsAt": time.time() + 600}}}
    assert backend.note_rate_limit(info, "sonnet") is True
    assert backend.limited_for("sonnet") > 0, "the whole seat is spent"


def test_an_empty_model_name_is_refused_at_launch(tmp_path, monkeypatch):
    """`:high` would have made `Profile("", "high")`, and an empty name
    matches every model in the outage table -- one refusal parked them all."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_EMERGENCY_PROFILES", "sonnet:high,:low")
    with pytest.raises(SystemExit):
        CliBackend()
    monkeypatch.setenv("SOLVER_CLI_EMERGENCY_PROFILES", "sonnet:high")
    monkeypatch.setenv("SOLVER_CLI_MODELS", "opus,son net")
    with pytest.raises(SystemExit):
        CliBackend()


def test_a_drill_lets_the_operator_watch_the_ladder_move(
    tmp_path, monkeypatch, capsys
):
    """A real usage limit cannot be ordered up, so `SOLVER_CLI_DRILL` pretends
    one at launch: the first solve goes to the backup seat, on real traffic,
    and the operator sees the lines a real limit would produce."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    monkeypatch.setenv("SOLVER_CLI_DRILL", "limit:primary")
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        return await conversation.send("solve it", 60.0), conversation

    body, conversation = asyncio.run(go())
    assert extract_code(body, "g"), body
    assert conversation.provider == "cli:opus@claude-2" and conversation.hops == 0
    assert [c["account"] for c in _cli_calls(log)] == ["claude-2"]
    out = capsys.readouterr().out
    assert "DRILL: pretending account primary is at its usage limit" in out, out
    assert "EMERGENCY MODE: cli:opus@claude-2" in out, out
    # It expires like a real limit would, so it cannot become the config.
    assert 0 < backend.limited_for("opus") <= 300

    monkeypatch.setenv("SOLVER_CLI_DRILL", "refuse:opus")
    backend = CliBackend()
    assert backend.pick()[1].label == "fable/low"

    monkeypatch.setenv("SOLVER_CLI_DRILL", "limit:nobody")
    with pytest.raises(SystemExit):
        CliBackend()


def test_the_ladder_tracks_every_model_a_phase_can_ask_for(tmp_path, monkeypatch):
    """The ladder is the OUTAGE TABLE's key set, not a preference order.

    It used to be the default model plus the emergency rung, because those
    were the only two anything could be asked on. Now each phase names a model
    and effort of its own, and a phase's model going out is the same event as
    the default's going out -- answered the same way, by `open_for` falling
    through. So every profile that can be asked for has to be a rung, or an
    outage on it would be invisible to the table that is supposed to notice.

    `fable/low` is the emergency rung and `fable/medium` is the repair phase.
    They are the same model at two efforts and they are two DIFFERENT rungs on
    purpose: an outage is reported per model and effort, and collapsing them
    would bench a working repair phase because an emergency turn was refused.
    """
    from solvers.claude_cli import CliBackend, cli_phase_profiles

    _fake_cli(tmp_path, monkeypatch)
    monkeypatch.delenv("SOLVER_CLI_EMERGENCY_PROFILES", raising=False)
    monkeypatch.delenv("SOLVER_CLI_PHASE_PROFILES", raising=False)
    ladder = [p.label for p in CliBackend().profiles]

    assert ladder[0] == "opus/low", f"the default model must lead: {ladder}"
    assert "fable/low" in ladder, f"the emergency rung is missing: {ladder}"
    # Every phase's profile is a rung.
    for phase, profile in cli_phase_profiles("low").items():
        assert profile.label in ladder, f"{phase} ({profile.label}) not in {ladder}"
    # ...and no rung appears twice, or an outage would be recorded against one
    # copy and read off the other.
    assert len(ladder) == len(set(ladder)), ladder


def test_the_cli_backend_counts_what_a_solve_costs_the_seat(tmp_path, monkeypatch):
    """Tokens, from the CLI's own usage report on every result event, in the
    four kinds it reports -- what the subscription's windows are spent in."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch)
    backend = CliBackend()
    asyncio.run(_one_cli_turn(backend))
    asyncio.run(_one_cli_turn(backend))
    stats = backend.stats()
    assert stats["turns"] == 2
    assert stats["tokens"] == {"input": 6, "output": 80, "cache_read": 240, "cache_write": 1000}
    assert stats["list_price_usd"] == 0.02


def test_the_cli_summary_names_the_ladder(tmp_path, monkeypatch):
    from solvers import roster as roster_module

    _fake_cli(tmp_path, monkeypatch, backups=1)
    monkeypatch.setenv("SOLVER_BACKEND", "cli")
    assert roster_module.describe([]) == (
        "claude CLI (opus/low > fable/low; 2 accounts)"
    )


def test_rust_outputs_can_be_taken_without_an_expectation(monkeypatch):
    """The Rust runner compares stdout with `expected` whether or not anyone
    asked, and its token split reads a None as a crash. Every Rust run taken
    without an expectation died there, before a single input ran, until the
    grader handed it an empty string instead."""
    from rlvr.execution.rust_judge import outputs_match
    from rlvr.types import ExecutionResult
    from solvers.verify import _Grader

    class _RustLike:
        def run_tests(self, code, entrypoint, tests, timeout_s):
            out = []
            for i, test in enumerate(tests):
                stdout = "42\n"
                # What rust_docker_executor does with every result.
                passed = outputs_match(stdout, test.expected)
                out.append(ExecutionResult(test_index=i, passed=passed, value=stdout,
                                           value_ok=True, runtime_ms=1.0))
            return out

    grader = _Grader()
    monkeypatch.setattr(grader, "executor", lambda language: _RustLike())
    runs = grader.outputs("fn main(){}", "rust", "main", [{"args": ["1\n"]}], budget_s=10)
    assert len(runs) == 1 and runs[0].ok and runs[0].value == "42\n", runs


def test_selecting_the_cli_backend_needs_no_browser(monkeypatch):
    """`SOLVER_BACKEND=cli` and nothing else. The default stays `browser`,
    because a solver that silently changed where the answers came from would be
    the worst kind of upgrade."""
    from solvers import roster as roster_module

    monkeypatch.setenv("SOLVER_BACKEND", "cli")
    assert roster_module.backend_kind() == "cli"
    # No browser is attached, and nothing defaults to one on the standard port
    # only to report that it could not be reached.
    assert roster_module.roster() == []
    assert "claude CLI" in roster_module.describe([])
    solver = roster_module.build_solver()
    assert type(solver._backend).__name__ == "CliBackend"

    monkeypatch.delenv("SOLVER_BACKEND", raising=False)
    assert roster_module.backend_kind() == "browser"
    monkeypatch.setenv("SOLVER_BACKEND", "nonsense")
    with pytest.raises(SystemExit):
        roster_module.backend_kind()


# --------------------------------------------------------------------------- #
# A turn that streams a heartbeat and no answer.
# --------------------------------------------------------------------------- #
def test_a_turn_that_streams_no_answer_text_is_cut_and_asked_elsewhere(
    tmp_path, monkeypatch, capsys
):
    """Measured over a production day: six turns emitted 34 to 388 events, one
    every 750ms, with zero characters of text, and two of those solves
    submitted nothing at all. `_Stalled` cannot see it -- events are arriving.
    The turn is cut once the stream has proved itself alive but silent, and
    the next rung answers."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_FIRST_TEXT_S", "1")
    _cli_modes(log, {"model:opus": "heartbeat", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        body = await conversation.send("solve it", 40.0)
        return body, conversation, time.monotonic() - started

    body, conversation, spent = asyncio.run(go())
    out = capsys.readouterr().out
    assert extract_code(body, "g"), out
    # Cut early, not at the 40s slice, and carried to the next rung.
    assert spent < 15.0, (spent, out)
    # The next rung down the ladder, whichever it is -- not the one that
    # went quiet.
    assert conversation.provider != "cli:opus" and conversation.hops == 1, out
    assert "has sent no answer text at all" in out, out
    assert "no text at all" in out, out
    # The seat is NOT parked: the turn was lost, the account is fine.
    assert backend.stats()["out"] == {}, backend.stats()["out"]


def test_a_silent_turn_is_worth_one_hop_and_no_more(tmp_path, monkeypatch, capsys):
    """Every rung going quiet must not cost a `FIRST_TEXT_S` each. One hop,
    then the answer goes back empty for the caller's own recovery to handle."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("SOLVER_CLI_FIRST_TEXT_S", "1")
    _cli_modes(log, {"*": "heartbeat"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        return await conversation.send("solve it", 60.0), time.monotonic() - started

    body, spent = asyncio.run(go())
    out = capsys.readouterr().out
    assert body == "", out
    # Two silent turns, not five: one hop, then it gives up on the ladder.
    assert out.count("has sent no answer text at all") == 2, out
    assert spent < 20.0, (spent, out)


def test_a_reply_that_never_streamed_is_still_read_from_the_result_event(
    tmp_path, monkeypatch
):
    """Two event shapes carry the whole reply, and the backend used to read
    neither: a `result` event's own text, and a complete `assistant` message.
    A turn whose deltas never came still has its answer right there."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch)
    for shape in ("result-only", "assistant-only"):
        _cli_modes(log, {"*": shape})
        backend = CliBackend()

        async def go():
            conversation = await backend.open()
            return await conversation.send("solve it", 30.0)

        assert extract_code(asyncio.run(go()), "g") == "def g(n):\n    return n", shape


# --------------------------------------------------------------------------- #
# An empty answer is worth zero, so any program beats it.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Which model answers which phase
# --------------------------------------------------------------------------- #


def test_every_phase_names_a_model_and_the_repair_is_a_different_one(monkeypatch):
    """All five phases ship a default, which is a change. Under the two-bar
    design the two writing turns were deliberately unset, because every one of
    the 102 solves in the archived runs opened on the same model and the logs
    therefore said nothing about how another model answers them. That does not
    carry over: these five phases are not five ways of asking the same thing.

    The one that matters is `repair`, pinned to a DIFFERENT MODEL from
    `candidate` on purpose. A model asked to repair its own program defends its
    own reading of the statement -- the failure the differential exists to
    catch, one layer up. Measured over 54 live solves under the old design: of
    ten single-case disagreements, nine ended with the model editing its own
    test case and keeping its program.

    `oracle` sharing the cheap profile is not an economy either. The reference
    is written under a correctness-first instruction so that it is NOT the
    program the candidate would write; two programs from one model at one
    effort copy one misreading, and comparing them then establishes nothing.
    """
    from solvers.claude_cli import PHASES, Profile, cli_phase_profiles

    monkeypatch.delenv("SOLVER_CLI_PHASE_PROFILES", raising=False)
    shipped = cli_phase_profiles("low")
    assert shipped == {
        "analysis": Profile("opus", "low"),
        "tests": Profile("opus", "low"),
        "oracle": Profile("opus", "low"),
        "candidate": Profile("opus", "medium"),
        "repair": Profile("fable", "medium"),
    }, shipped
    assert set(shipped) == set(PHASES), "a phase ships without a model"
    assert shipped["repair"].model != shipped["candidate"].model, (
        "the repair reads a failure in a program written by the same model"
    )

    monkeypatch.setenv(
        "SOLVER_CLI_PHASE_PROFILES", "oracle=fable:low,candidate=opus:high")
    chosen = cli_phase_profiles("low")
    assert chosen["oracle"] == Profile("fable", "low")
    assert chosen["candidate"] == Profile("opus", "high")
    # Naming one phase must not move any other.
    assert chosen["repair"] == Profile("fable", "medium"), chosen
    assert chosen["tests"] == Profile("opus", "low"), chosen


def test_the_orchestrator_only_ever_names_a_phase_that_exists(monkeypatch):
    """`open_for` falls through to the ladder SILENTLY on a phase it does not
    recognise, so a typo is not an error -- it is a phase quietly losing its
    model for the life of the release."""
    import re
    from pathlib import Path as _Path
    from solvers.claude_cli import PHASES

    source = (_Path(__file__).resolve().parent / "solvers" / "verify.py").read_text()
    named = set(re.findall(r'phase="([a-z0-9_]+)"', source))
    assert named, "no phase strings found; did the call shape change?"
    assert named <= set(PHASES), f"unknown phase(s): {sorted(named - set(PHASES))}"

    # And a phase name that is NOT one of them is refused loudly rather than
    # ignored, which is the other half of the same guard.
    from solvers.claude_cli import cli_phase_profiles

    for bad in ("cases", "nosuchphase=opus", "oracle=opus:turbo", "oracle="):
        monkeypatch.setenv("SOLVER_CLI_PHASE_PROFILES", bad)
        with pytest.raises(SystemExit):
            cli_phase_profiles("low")


def test_a_phase_model_is_a_preference_and_never_a_pin(tmp_path, monkeypatch):
    """Three behaviours, and the last two are the safety argument.

    A phase whose model is up gets it. A phase whose model is OUT on every
    account falls through to the ladder, because a preference that can stop a
    solve answering is not a preference. And `avoid` beats the preference: it
    is how a pass says "not the one that just got this wrong", and honouring a
    pin ahead of it would send the retry straight back to the model being
    retried.
    """
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, mode="ok")
    monkeypatch.setenv("SOLVER_CLI_PHASE_PROFILES", "oracle=sonnet,candidate=opus")
    backend = CliBackend()

    async def opened(**kw):
        conversation = await backend.open_for(**kw)
        model = conversation.profile.model
        await conversation.close()
        backend.release()
        return model

    assert asyncio.run(opened(phase="oracle")) == "sonnet"
    assert asyncio.run(opened(phase="candidate")) == "opus"
    # A phase that does not exist is just "open a conversation" -- the
    # ladder's own choice, not an error.
    assert asyncio.run(opened(phase="nosuchphase")) == backend.default.model
    assert asyncio.run(opened(phase=None)) == backend.default.model

    # sonnet out on every account: the reference turn still gets answered.
    for account in backend.accounts:
        backend._set((account.name, "sonnet"), time.time() + 600, "rate limited")
    assert asyncio.run(opened(phase="oracle")) != "sonnet"

    # ...and with sonnet healthy again, a retry that must avoid it is still
    # not sent back to it -- the pin loses to `avoid`, not to the outage.
    backend._out.clear()
    assert asyncio.run(opened(phase="oracle")) == "sonnet", (
        "the outage was not cleared, so the next assertion proves nothing"
    )
    assert asyncio.run(opened(phase="oracle", avoid="cli:sonnet")) != "sonnet"


def test_a_repair_round_says_truthfully_whose_cases_it_ran():
    """The one sentence a repair round turns on, and it has to be true.

    Stage 8 runs on a different model, in its own conversation, and it wrote
    neither the program nor the inputs. "The test cases you sent" would be
    false about all three. The framings ask for different reasoning -- "one of
    my two answers is wrong" against "two programs disagree and at least one
    is wrong about the statement" -- so a false one does not merely misdescribe
    the round, it asks the wrong question.

    Three ways a failure can be found, three sentences, and each is checked
    against what actually ran.
    """
    differential = _repair_prompt("python", report="g(7) -> 0, expected 7")
    examples = _repair_prompt("python", report="g(7) -> 0, expected 7",
                              found_by="examples")
    unrun = _repair_prompt("python", defect="it does not parse", found_by="unrun")

    assert "You did not write this program" in differential
    assert "a separate reference" in differential and "you sent" not in differential
    assert "shipped with the statement" in examples, examples
    assert "ground truth" in examples, examples
    assert "NOTHING WAS RUN" in unrun, unrun
    for prompt in (differential, examples, unrun):
        assert "the test cases you sent" not in prompt, prompt


# --------------------------------------------------------------------------- #
# The size probe: a verdict per program, and no floor under it
# --------------------------------------------------------------------------- #
def test_the_probe_runs_again_on_a_program_it_asked_to_be_rewritten():
    """The hole this closes: `probed` was set the instant a too_slow repair
    went out, so the rewritten program -- the thing the round existed to
    produce -- shipped having never been timed."""
    from solvers.verify import VerifyingSolver

    from solvers.verify import _Probe

    solver = VerifyingSolver.__new__(VerifyingSolver)
    solver._size_probe = True
    timed: list[str] = []

    async def fake(code, generator, task, left):
        timed.append(code)
        return (_Probe("too slow", "too_slow") if "quadratic" in code
                else _Probe(None, "passed"))

    solver._timed_out_at_scale = fake
    probed: dict = {}
    slow = SimpleNamespace(code="quadratic")
    fast = SimpleNamespace(code="linear")
    task = SimpleNamespace(language="python", entrypoint="g")

    async def go():
        started = time.monotonic()
        first, _ = await solver._probe_now(slow, ["gen"], probed, task, 300.0, started)
        # The SAME program again: answered from the cache, not re-run.
        again, _ = await solver._probe_now(slow, ["gen"], probed, task, 300.0, started)
        # A DIFFERENT program: timed in its turn.
        third, _ = await solver._probe_now(fast, ["gen"], probed, task, 300.0, started)
        return first, again, third

    first, again, third = asyncio.run(go())
    assert first == "too slow" and again == "too slow"
    assert third is None, "the rewritten program was never timed"
    assert timed == ["quadratic", "linear"], f"re-ran a verdict it held: {timed}"


def test_the_probe_has_no_budget_floor_under_it():
    """`PROBE_FLOOR_S` is gone. It skipped the probe below 45s on the theory
    that a verdict with no room to act is wasted -- but the run is bounded by
    what is left anyway, an unfinished probe reports nothing and ships the
    answer, and a too_slow verdict with ten seconds left still buys a round."""
    import solvers.verify as verify

    assert not hasattr(verify, "PROBE_FLOOR_S")

    solver = verify.VerifyingSolver.__new__(verify.VerifyingSolver)
    solver._size_probe = True
    offered: list[float] = []

    async def fake(code, generator, task, left):
        offered.append(left)
        return verify._Probe(None, "passed")

    solver._timed_out_at_scale = fake
    task = SimpleNamespace(language="python", entrypoint="g")

    # Ten seconds left, far under the old floor: it runs, with ten seconds.
    asyncio.run(solver._probe_now(
        SimpleNamespace(code="x"), ["gen"], {}, task,
        10.0, time.monotonic(),
    ))
    assert offered and 5.0 < offered[0] <= 10.0, offered

    # Past the deadline is the one case it declines: there is no run to make.
    offered.clear()
    asyncio.run(solver._probe_now(
        SimpleNamespace(code="y"), ["gen"], {}, task,
        0.0, time.monotonic() - 5.0,
    ))
    assert offered == [], "ran a probe with the budget already gone"


# --------------------------------------------------------------------------- #
# Grading under the validator's own limits
# --------------------------------------------------------------------------- #
def test_python_grading_falls_back_when_no_daemon_answers(monkeypatch, capsys):
    """Rust cannot degrade -- rustc lives in the pinned image -- but Python
    can, and the choice there is between grading under looser limits and not
    grading at all. The first is worth much more, and the line says what the
    validator will actually apply."""
    monkeypatch.setenv("SOLVER_VERIFY_EXECUTOR", "docker")
    from solvers.verify import _Grader

    grader = _Grader()
    real = None
    import rlvr.execution.executor as executor_module
    real = executor_module.get_executor

    def only_subprocess(settings, language):
        if getattr(settings, "executor", "") == "docker":
            raise RuntimeError("Cannot connect to the Docker daemon")
        return real(settings, language=language)

    monkeypatch.setattr(executor_module, "get_executor", only_subprocess)

    passed, total, failures, _ = grader.check(
        "def g(n):\n    return n\n", "python", "g",
        [{"args": [3], "kwargs": {}, "expected": 3}],
    )
    assert (passed, total) == (1, 1), failures
    out = capsys.readouterr().out
    assert "256 MiB" in out and "subprocess" in out, out

    # Rust does NOT degrade: there is no subprocess path for it.
    with pytest.raises(Exception):
        grader.executor("rust")


def test_an_out_of_memory_at_scale_is_reported_like_a_timeout():
    """The validator runs every hidden test in ONE container at 256 MiB with
    swap off, so an OOM there fails the whole suite rather than the case that
    caused it. That is worth a repair round on the same footing as a timeout
    -- and every OTHER crash at scale is not, because it is far likelier to be
    a generated input the statement never allowed."""
    from solvers.verify import _looks_out_of_memory, _Run

    assert _looks_out_of_memory(
        _Run(ok=False, error="container killed (likely OOM / memory limit)")
    )
    assert _looks_out_of_memory(_Run(ok=False, error="MemoryError"))
    # A timeout is a timeout; it has its own branch and its own sentence.
    assert not _looks_out_of_memory(
        _Run(ok=False, error="timed out after 5.000s", timed_out=True)
    )
    # An ordinary crash is left alone.
    assert not _looks_out_of_memory(_Run(ok=False, error="IndexError: list index"))
    # ...including one whose traceback merely CONTAINS the letters. `oom` was a
    # substring match, and a program with a function called `count_rooms` had
    # every crash at scale reported as a memory kill.
    assert not _looks_out_of_memory(_Run(
        ok=False,
        error='RecursionError: maximum recursion depth\n  File "<c>", line 12, in count_rooms',
    ))
    assert not _looks_out_of_memory(_Run(ok=False, error="KeyError: 'bedroom'"))
    assert _looks_out_of_memory(_Run(ok=False, error="Rust container exceeded its memory limit"))


# --------------------------------------------------------------------------- #
# Never waiting on a seat that cannot serve
# --------------------------------------------------------------------------- #
def test_a_busy_account_is_tried_after_a_free_one(tmp_path, monkeypatch):
    """The concurrency limit is about one sign-in's processes. Shared across
    accounts it also made a busy account block a free one, so a solve queued
    behind four others waited while another subscription sat idle."""
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, backups=1)
    backend = CliBackend()
    assert len(backend.accounts) == 2, backend.accounts

    # Separate gates, not one.
    primary, backup = backend.accounts
    assert backend.slot_for(primary) is not backend.slot_for(backup)

    # With the primary's processes all in flight, the ladder offers the backup
    # first -- demoted, not skipped, because "busy" is a millisecond-old fact.
    for _ in range(backend.concurrency):
        asyncio.run(backend.slot_for(primary).acquire())
    ordered = backend._with_room_first(list(backend.pairs()))
    assert ordered[0][0] == backup, [a.name for a, _ in ordered]


def test_a_retry_is_not_waited_out_when_another_seat_is_free(tmp_path, monkeypatch):
    """A wait is only worth paying when there is nothing better to do with the
    time, and an idle ACCOUNT is something better.

    Another account, specifically -- not another rung. The ladder is mostly
    other models on this same sign-in, which share its processes and its
    quota, so moving there is not escaping a busy seat: it is answering on a
    weaker model to avoid a one-second wait, which is the measured regression
    `test_a_retrying_auth_error_waits_for_the_cli_instead_of_hopping` exists
    to hold shut.
    """
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    # Only the PRIMARY is retrying; the backup answers. So the turn has
    # somewhere real to go, and nothing about the ladder's other models is
    # involved in the decision.
    # The fake names an account after its `CLAUDE_CONFIG_DIR`, and the
    # primary has none -- so it is `default` here, not `primary`.
    _cli_modes(log, {"default": "overloaded0", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        started = time.monotonic()
        body = await conversation.send("solve it", 60.0)
        return body, conversation, time.monotonic() - started

    with contextlib.redirect_stdout(io.StringIO()) as out:
        body, conversation, spent = asyncio.run(go())
    log_text = out.getvalue()

    assert extract_code(body, "g"), body
    assert conversation.hops == 1, log_text
    # It moved to the other ACCOUNT keeping the model, rather than down the
    # ladder to a weaker one on the same sign-in.
    assert conversation.account.name != "primary", conversation.provider
    assert conversation.model == "opus", conversation.provider
    # And the 1s, 4s and 16s the CLI would have waited went unspent.
    assert spent < 1.0, spent
    # Nothing was benched: one retry is not evidence a model is out.
    assert backend.stats()["out"] == {}, backend.stats()["out"]


def test_a_short_auth_race_is_still_waited_out(tmp_path, monkeypatch):
    """The one measured regression this must not undo. `retry 1/10:
    authentication_failed, next wait 1s` is two miners racing to refresh one
    login's token -- transient by the CLI's own account of it, over in about a
    second, and hopping on it spent the backup account's quota to avoid a
    one-second wait."""
    from solvers.claude_cli import CliBackend

    log = _fake_cli(tmp_path, monkeypatch, backups=1)
    _cli_modes(log, {"model:opus": "authrace", "*": "ok"})
    backend = CliBackend()

    async def go():
        conversation = await backend.open()
        body = await conversation.send("solve it", 60.0)
        return body, conversation

    body, conversation = asyncio.run(go())
    assert extract_code(body, "g"), body
    assert conversation.hops == 0, "abandoned a healthy seat over a token refresh"


# --------------------------------------------------------------------------- #
# The solution cache
# --------------------------------------------------------------------------- #
def test_only_an_answer_with_evidence_behind_it_is_kept(monkeypatch, tmp_path):
    """A cached wrong answer is not one zero, it is a zero every time that
    problem comes round again -- so the gate is about evidence, not
    confidence, and every condition rules out a way an answer can look
    finished without having been checked."""
    from solvers import solution_cache

    monkeypatch.setenv("SOLVER_SOLUTION_CACHE_DIR", str(tmp_path))
    good = dict(self_verified=True, failures=False, contested=0, probe="passed")
    assert solution_cache.worth_keeping(**good)

    # Never graded at all -- the whole hazard.
    assert not solution_cache.worth_keeping(**{**good, "self_verified": False})
    # Best in hand when the clock ran out, not clean.
    assert not solution_cache.worth_keeping(**{**good, "failures": True})
    # Cleared a suite missing a question the readers could not agree on.
    assert not solution_cache.worth_keeping(**{**good, "contested": 1})
    # Never timed at scale, or timed and found slow.
    for probe in ("", "none", "skipped", "too_slow"):
        assert not solution_cache.worth_keeping(**{**good, "probe": probe})


def test_a_cached_answer_survives_a_restart_and_a_corrupt_one_is_a_miss(
    monkeypatch, tmp_path
):
    """The point is a hit that costs a second and no quota. The risk is a file
    an operator edited, truncated or copied, so it is re-checked rather than
    trusted -- and every way it can be wrong reads as a miss."""
    from solvers import solution_cache

    monkeypatch.setenv("SOLVER_SOLUTION_CACHE_DIR", str(tmp_path))
    task = SimpleNamespace(problem_id="p", language="python", entrypoint="g")
    solution_cache.save("k", solution_cache.record(
        code="def g(n):\n    return n\n", raw="raw", task=task,
        bar=[{"args": [1], "expected": 1}], probe="passed", providers=["cli:opus"],
    ))

    stored = solution_cache.load("k")
    assert stored is not None and "def g" in stored["code"]
    # The bar is kept beside it: a cached answer that turns out wrong is a
    # question about what it was checked against.
    assert stored["bar"] == [{"args": [1], "expected": 1}]

    assert solution_cache.load("never-written") is None
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert solution_cache.load("broken") is None
    (tmp_path / "empty.json").write_text('{"code": "  "}', encoding="utf-8")
    assert solution_cache.load("empty") is None

    # Turned off outright, a hit is impossible.
    monkeypatch.setenv("SOLVER_SOLUTION_CACHE", "0")
    assert solution_cache.load("k") is None


def test_a_cache_hit_answers_without_opening_a_conversation(monkeypatch, tmp_path):
    """A solve that does not run is a solve the accounts can spend on a problem
    they have not seen -- and it answers in about a second, which is what the
    fastest-responder multiplier pays for."""
    from solvers import solution_cache
    from solvers.verify import _cache_key

    monkeypatch.setenv("SOLVER_SOLUTION_CACHE_DIR", str(tmp_path))
    task = _two_seat_task()
    solution_cache.save(_cache_key(task), solution_cache.record(
        code="def g(n):\n    return sum(int(c) for c in str(n))\n",
        raw="raw", task=task, bar=[{"args": [12345], "expected": 15}],
        probe="passed", providers=["cli:opus"],
    ))

    class _NeverAsked:
        async def open(self, avoid=None, timeout_s=None):
            raise AssertionError("opened a conversation on a cache hit")
        async def open_for(self, phase=None, avoid=None, timeout_s=None):
            raise AssertionError("opened a conversation on a cache hit")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_NeverAsked(), reserve_s=0, max_budget_s=120)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        answer = asyncio.run(solver.solve_task(task, 120.0))

    assert "sum(int(c)" in answer.code
    assert "cache=hit" in out.getvalue(), out.getvalue()


def test_a_cached_answer_that_no_longer_parses_is_solved_again(
    monkeypatch, tmp_path
):
    """Re-checked rather than trusted: the file system is not a memory."""
    from solvers import solution_cache
    from solvers.verify import _cache_key

    monkeypatch.setenv("SOLVER_SOLUTION_CACHE_DIR", str(tmp_path))
    task = _two_seat_task()
    solution_cache.save(_cache_key(task), solution_cache.record(
        code="def g(n:\n    return  # truncated mid-signature",
        raw="raw", task=task, bar=[{"args": [1], "expected": 1}],
        probe="passed", providers=[],
    ))

    backend = _TwoSeats({
        "cases": [CASES],
        "program": ["```python\ndef g(n):\n    return sum(int(c) for c in str(n))\n```"],
        None: ["```python\ndef g(n):\n    return sum(int(c) for c in str(n))\n```"],
    })
    solver = VerifyingSolver(backend, reserve_s=0, max_budget_s=120)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        answer = asyncio.run(solver.solve_task(task, 120.0))

    assert "sum(int(c)" in answer.code, "served a cached answer that will not run"
    assert "no longer reads as a program" in out.getvalue()


# --------------------------------------------------------------------------- #
# The archive holds how the answer was arrived at
# --------------------------------------------------------------------------- #
def test_the_archive_keeps_the_bar_and_how_the_answer_was_reached(tmp_path):
    """The request and the response say WHAT was submitted. Without this there
    is no way to ask why it was thought to be right -- which is the question a
    wrong answer in the archive actually raises, and the one the 97 archived
    exchanges cannot answer."""
    from solution_archive import save_exchange

    path = save_exchange(
        "p", {"statement": "s"}, {"code": "x"},
        directory=tmp_path,
        solve={"bar": [{"args": [1], "expected": 1}], "probe": "passed",
               "adjudicated": {"case": 1}, "repair": ["cli:sonnet"]},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["solve"]["probe"] == "passed"
    assert record["solve"]["bar"] == [{"args": [1], "expected": 1}]
    assert record["solve"]["repair"] == ["cli:sonnet"]

    # A solver that reports none leaves the record exactly as it was.
    plain = json.loads(
        save_exchange("q", {"a": 1}, {"b": 2}, directory=tmp_path)
        .read_text(encoding="utf-8")
    )
    assert set(plain) == {"problem_id", "request", "response"}, plain


def test_a_slot_hop_keeps_the_model_it_was_pinned_to(tmp_path, monkeypatch):
    """A move about PROCESSES must not quietly change the model.

    The conversations that reach this most often are the ones pinned to a
    model for a reason -- the `oracle` is deliberately the weaker instruction
    and `repair` is deliberately a different model from the candidate's.
    Landing either on the next account's default would collapse both back onto
    the candidate's own model, removing the difference without removing the
    line that claims it.
    """
    from solvers.claude_cli import CliBackend

    _fake_cli(tmp_path, monkeypatch, backups=1)
    backend = CliBackend()
    primary, backup = backend.accounts

    async def go():
        conversation = await backend.open_profile("sonnet", "low")
        assert conversation.model == "sonnet", conversation.provider
        # The primary's processes are all in flight.
        for _ in range(backend.concurrency):
            await backend.slot_for(primary).acquire()
        moved = conversation._hop_account("no free process slot")
        return conversation, moved

    with contextlib.redirect_stdout(io.StringIO()):
        conversation, moved = asyncio.run(go())

    assert moved, "did not move off a full account"
    assert conversation.account == backup, conversation.provider
    assert conversation.model == "sonnet", (
        f"the slot hop changed the pinned model: {conversation.provider}"
    )


def test_a_probe_that_never_ran_is_not_reported_as_a_pass():
    """`passed` is evidence and gates the on-disk cache; the other ways of
    returning no sentence know nothing about the program.

    Recorded as `passed`, an answer the clock ran out on would be written to
    a permanent cache and served again unexamined -- the one thing the cache
    gate exists to prevent. The state comes back FROM the run rather than
    being latched on the solver, because the solver serves several solves at
    once and a flag on `self` was read by whichever finished last."""
    import solvers.verify as verify
    from solvers.verify import _Probe

    solver = verify.VerifyingSolver.__new__(verify.VerifyingSolver)
    solver._size_probe = True
    assert not hasattr(solver, "_probe_ran")
    task = SimpleNamespace(language="python", entrypoint="g")
    candidate = SimpleNamespace(code="def g(n): return n")

    async def never_runs(code, generator, task, left):
        return _Probe(None, "none")          # no valid input could be built

    async def finishes(code, generator, task, left):
        return _Probe(None, "passed")

    async def too_slow(code, generator, task, left):
        return _Probe("did not finish", "too_slow")

    async def crashes(code, generator, task, left):
        return _Probe(None, "crashed")

    now = time.monotonic()

    solver._timed_out_at_scale = finishes
    assert asyncio.run(solver._probe_now(
        candidate, ["gen"], {}, task, 300.0, now)) == (None, "passed")

    solver._timed_out_at_scale = never_runs
    assert asyncio.run(solver._probe_now(
        candidate, ["gen"], {}, task, 300.0, now)) == (None, "none")

    solver._timed_out_at_scale = too_slow
    verdict, state = asyncio.run(
        solver._probe_now(candidate, ["gen"], {}, task, 300.0, now))
    assert verdict == "did not finish" and state == "too_slow"

    # A crash at scale is not a pass either, whatever caused it.
    solver._timed_out_at_scale = crashes
    assert asyncio.run(solver._probe_now(
        candidate, ["gen"], {}, task, 300.0, now)) == (None, "crashed")

    # Out of time: never asked, so nothing is known.
    solver._timed_out_at_scale = finishes
    assert asyncio.run(solver._probe_now(
        candidate, ["gen"], {}, task, 0.0, now - 5.0)) == (None, "skipped")

    # And no generator at all is not a pass either.
    assert asyncio.run(solver._probe_now(
        candidate, [], {}, task, 300.0, now)) == (None, "none")

    # Only VERDICTS are remembered. A `none` is not: a later call with a
    # generator the first bar did not have must be free to find out.
    probed: dict = {}
    solver._timed_out_at_scale = never_runs
    asyncio.run(solver._probe_now(candidate, ["gen"], probed, task, 300.0, now))
    assert probed == {}
    solver._timed_out_at_scale = finishes
    asyncio.run(solver._probe_now(candidate, ["gen"], probed, task, 300.0, now))
    assert probed[candidate.code.strip()].state == "passed"


def test_the_probe_tries_every_generator_it_was_given():
    """Both bars are asked for one, and the second exists precisely so that a
    first generator returning the wrong shape at every scale is not the end
    of the probe."""
    import solvers.verify as verify
    from solvers.verify import _Probe

    solver = verify.VerifyingSolver.__new__(verify.VerifyingSolver)
    solver._size_probe = True
    task = SimpleNamespace(language="python", entrypoint="g")
    used: list[str] = []

    async def fake(code, generator, task, left):
        used.append(generator)
        return _Probe(None, "none") if generator == "broken" else _Probe(None, "passed")

    solver._timed_out_at_scale = fake
    verdict = asyncio.run(solver._probe_now(
        SimpleNamespace(code="x"), ["broken", "works"], {}, task, 300.0,
        time.monotonic(),
    ))
    assert verdict == (None, "passed") and used == ["broken", "works"], used


def test_a_probe_verdict_belongs_to_one_solve():
    """Two solves probing at once on the one solver must not read each other's
    result. Reproduced with the flag this replaced: solve B, whose generator
    built nothing, reported `passed` because solve A's run finished while B
    was waiting -- and B's never-timed program went into the permanent
    cache."""
    import solvers.verify as verify
    from solvers.verify import _Probe

    solver = verify.VerifyingSolver.__new__(verify.VerifyingSolver)
    solver._size_probe = True
    task = SimpleNamespace(language="python", entrypoint="g")
    gate = asyncio.Event()

    async def fake(code, generator, task, left):
        if code == "A":
            await gate.wait()             # A finishes at size, later
            return _Probe(None, "passed")
        gate.set()                        # B built nothing, and says so first
        await asyncio.sleep(0.01)
        return _Probe(None, "none")

    solver._timed_out_at_scale = fake

    async def go():
        now = time.monotonic()
        a = asyncio.create_task(solver._probe_now(
            SimpleNamespace(code="A"), ["gen"], {}, task, 300.0, now))
        b = asyncio.create_task(solver._probe_now(
            SimpleNamespace(code="B"), ["gen"], {}, task, 300.0, now))
        return await asyncio.gather(a, b)

    a, b = asyncio.run(go())
    assert a == (None, "passed"), a
    assert b == (None, "none"), "solve B was credited with solve A's run"


def test_a_handoff_that_lands_on_the_same_model_is_not_called_foreign():
    """`foreign` says the model did not write this program. A browser fleet
    has no models to change between, and `open_profile` falls through to the
    ladder when the one asked for is out everywhere -- so a handoff can land
    right back where it started. Telling that model it is looking at someone
    else's work is the same false premise the flag exists to remove."""
    from solvers.verify import _model_of

    assert _model_of("cli:opus@primary") == "opus"
    assert _model_of("cli:sonnet") == "sonnet"
    assert _model_of(None) == ""
    # The comparison the handoff actually makes.
    assert _model_of("cli:opus@primary") == _model_of("cli:opus@backup"), (
        "the same model on another account is not a different reading"
    )
    assert _model_of("cli:opus@primary") != _model_of("cli:sonnet@primary")


# --------------------------------------------------------------------------- #
# Reading the statement before any model is asked
# --------------------------------------------------------------------------- #
def _recorded_requests():
    """The archived production requests, as solver tasks."""
    directory = Path(__file__).resolve().parent.parent / "problems"
    for path in sorted(directory.glob("*.json")):
        request = json.loads(path.read_text()).get("request") or {}
        if not request.get("statement"):
            continue
        yield SolveTask(
            problem_id=request.get("problem_id", path.stem),
            language=request.get("language", "python"),
            statement=request["statement"],
            entrypoint=request.get("entrypoint", "solve"),
            public_examples=request.get("public_examples") or [],
            deadline_s=float(request.get("deadline_s") or 300.0),
        )


def test_the_trap_scan_has_something_to_say_about_every_recorded_task():
    """The free pass runs on every solve, so a statement it reads nothing out
    of is a statement the oracle and the candidate start on separately. Over
    the recorded corpus it is never silent, and the two traps that hold for
    every task on this subnet -- stdlib only, and no public examples to
    anchor anything -- are on all of them."""
    from solvers.analyze import heuristic_analyze, trap_names

    tasks = list(_recorded_requests())
    assert len(tasks) >= 90, f"corpus looks truncated: {len(tasks)} tasks"

    for task in tasks:
        names = trap_names(heuristic_analyze(task))
        assert len(names) >= 3, f"{task.problem_id} read only {names}"
        assert "sandbox_constraints" in names, task.problem_id
        assert "no_public_examples" in names, task.problem_id
        assert len(set(names)) == len(names), f"duplicate trap: {names}"


# --- the fifteen traps mined from fnlich/hone-examples ---------------------- #
# Each is pinned by REAL statement wording, copied out of the corpus it was
# mined from, plus a control that must NOT fire. A trap whose regex is written
# to match a sentence invented here proves only that the regex matches itself.

_MINED_TRAPS = (
    ("recursive_descent_depth",
     "There is no expression or metadata depth limit. Across the input, the "
     "total number of fields, links and bindings is at most 200000."),
    ("deterministic_tiebreak",
     "dispatch the noncancelled expiration with the smallest timestamp not "
     "exceeding `T`; ties are resolved by smaller insertion rank."),
    ("cycle_self_reference",
     "A bound name is cyclic if a nonempty chain of dependencies leads back "
     "to it."),
    ("duplicates_defined",
     "flatten directly nested unions from left to right, discard structurally "
     "equal items after their first occurrence."),
    ("inclusive_bounds",
     "Its inclusive interval `[lo, hi]` contains `sequence`."),
    ("preserve_untouched",
     "Missing and cyclic references remain unchanged. Preserve each `meta` "
     "wrapper and its extras exactly."),
    ("case_sensitivity_stated",
     "Names are matched case-insensitive after casefolding."),
    ("error_priority_order",
     "Then scan joints structurally. For each index, check in order: "
     "duplicate joint name, unresolved parent, unresolved child."),
    ("fixpoint_closure",
     "The complete preparation set is the smallest set containing all "
     "directly changed fields such that every reachable field whose child is "
     "in the set is also included."),
    ("modular_arithmetic",
     "Report the total modulo 1000000007."),
    ("all_branches_no_shortcircuit",
     "A union matches only when exactly one branch successfully rebuilds the "
     "entire value. If a second branch succeeds, fail with UNION_AMBIGUOUS."),
    ("exact_output_shape",
     "Return a dictionary with exactly these keys: `x`, `y`, `z`."),
    ("bool_is_not_int",
     "Values must exactly match the logical field type: booleans are not "
     "integers."),
    ("float_exactness",
     "Compare with relative error below 1e-9; values are IEEE 754 binary64."),
)


@pytest.mark.parametrize("name,wording", _MINED_TRAPS)
def test_a_mined_trap_fires_on_the_wording_it_was_mined_from(name, wording):
    """Every entry added from the 178-statement corpus, against a sentence
    lifted out of a real statement rather than one written to match."""
    from solvers.analyze import heuristic_analyze, trap_names

    task = SolveTask(problem_id="m", language="python", statement=wording,
                     entrypoint="solve", public_examples=[], deadline_s=300.0)
    assert name in trap_names(heuristic_analyze(task)), (
        f"{name} missed its own wording: {wording!r}"
    )


def test_a_plain_statement_draws_none_of_the_mined_traps():
    """The control, and the one that decides whether any of this is worth
    anything. A catalog that fires on everything has told the solver nothing:
    the block is read by four later prompts, and a trap that is always there
    is noise competing with the traps that are not."""
    from solvers.analyze import heuristic_analyze, trap_names

    plain = SolveTask(
        problem_id="p", language="python",
        statement="Return the sum of the decimal digits of n. n is at most 99.",
        entrypoint="g", public_examples=[], deadline_s=300.0,
    )
    names = set(trap_names(heuristic_analyze(plain)))
    mined = {n for n, _ in _MINED_TRAPS} | {"rust_wide_arithmetic"}
    assert not (names & mined), f"fired on a statement with no traps: {names & mined}"


def test_rust_gets_the_overflow_trap_and_python_never_does():
    """The one mined trap that is conditional on the language, and the reason
    it is: Python integers do not overflow, so the same 1e18 bound means
    `use a closed form` there and `i64 is not wide enough` in Rust. Telling
    Python about a Rust overflow wastes the only prompt there is."""
    from solvers.analyze import heuristic_analyze, trap_names

    wording = ("Each weight is at most 10^18 and the total is the sum of the "
               "selected weights.")
    for language, entry in (("rust", "main"), ("python", "solve")):
        task = SolveTask(problem_id="o", language=language, statement=wording,
                         entrypoint=entry, public_examples=[], deadline_s=300.0)
        names = trap_names(heuristic_analyze(task))
        assert "huge_numeric_bounds" in names, language
        if language == "rust":
            assert "rust_wide_arithmetic" in names, names
        else:
            assert "rust_wide_arithmetic" not in names, (
                "told Python about an overflow it cannot have"
            )


def test_the_mined_traps_moved_the_number_they_were_mined_to_move():
    """The catalog's own claim, checked against the corpus rather than
    asserted in a docstring.

    SIX entries are structural -- they name the language and say the suite is
    hidden -- and fire on everything, so they cannot distinguish one statement
    from another. What a solve gains is the rest. Before this batch a third of
    recorded statements drew NONE of them; the block those solves carried said
    only `this is Python, stdlib only, no examples`, which is true of every
    task on the subnet."""
    from solvers.analyze import heuristic_analyze, trap_names

    structural = {"sandbox_constraints", "no_public_examples", "python_contract",
                  "rust_contract", "token_output_compare", "large_n_hidden_tests"}
    tasks = list(_recorded_requests())
    specific = [len(set(trap_names(heuristic_analyze(t))) - structural) for t in tasks]
    silent = sum(1 for n in specific if n == 0)

    assert silent / len(tasks) <= 0.10, (
        f"{silent}/{len(tasks)} statements draw no problem-specific trap; "
        f"it was 31% before the mined entries and must not regress there"
    )
    assert sorted(specific)[len(specific) // 2] >= 2, (
        f"median problem-specific traps fell to {sorted(specific)[len(specific)//2]}"
    )


def test_the_language_contract_matches_the_language():
    """Python is graded by calling a function and Rust by running a program,
    and a solver told the wrong one writes the wrong shape of answer."""
    from solvers.analyze import heuristic_analyze, trap_names

    for task in _recorded_requests():
        names = trap_names(heuristic_analyze(task))
        if task.language == "python":
            assert "python_contract" in names, task.problem_id
            assert "rust_contract" not in names, task.problem_id
        else:
            assert "rust_contract" in names, task.problem_id
            assert "python_contract" not in names, task.problem_id


def test_the_statements_own_wording_outranks_the_standing_contract():
    """`sandbox_constraints` is added twice: once at high severity when the
    statement itself says so, once at medium as the standing rule. First
    writer wins, so a statement that spells the constraint out keeps the
    version that says the statement spelled it out."""
    from solvers.analyze import heuristic_analyze

    spelled_out = SolveTask(
        problem_id="x", language="python",
        statement="Use the standard library only. Perform no input/output.",
        entrypoint="solve", public_examples=[], deadline_s=300.0,
    )
    silent = SolveTask(
        problem_id="y", language="python",
        statement="Return the sum of the list.",
        entrypoint="solve", public_examples=[], deadline_s=300.0,
    )
    loud = [t for t in heuristic_analyze(spelled_out).traps
            if t.name == "sandbox_constraints"][0]
    quiet = [t for t in heuristic_analyze(silent).traps
             if t.name == "sandbox_constraints"][0]
    assert loud.severity == "high", loud
    assert quiet.severity == "medium", quiet


def test_the_model_may_add_a_trap_and_may_never_remove_one():
    """A regex that matched is evidence the words are there. A model naming
    the same trap is agreeing, not correcting -- so the heuristic's wording
    survives the merge and the model's is dropped. The asymmetry is the
    point: a missed trap costs the solve, a redundant one costs tokens."""
    from solvers.analyze import Analysis, Trap, merge_analysis

    base = Analysis(
        traps=[Trap("index_base", "the heuristic saw it", "convert once")],
        source="heuristic",
    )
    extra = Analysis(
        traps=[
            Trap("index_base", "the model rewrote this", "something else"),
            Trap("brand_new", "only the model saw this", "handle it"),
        ],
        algorithm_sketch="sort, then sweep",
        source="llm",
    )
    merged = merge_analysis(base, extra)
    names = [t.name for t in merged.traps]

    assert names == ["index_base", "brand_new"], names
    kept = [t for t in merged.traps if t.name == "index_base"][0]
    assert kept.evidence == "the heuristic saw it", kept
    # Prose is the other way round: a regex cannot write a sketch.
    assert merged.algorithm_sketch == "sort, then sweep"
    assert merged.source == "heuristic+llm"


def test_an_unusable_analysis_reply_leaves_the_heuristic_standing():
    """Stage 3 is the one stage allowed to produce nothing. Every shape the
    model can fail in has to end with the free pass still in hand, because
    the alternative to a partial analysis is no analysis."""
    from solvers.analyze import analysis_from_json, heuristic_analyze

    task = SolveTask(
        problem_id="x", language="python", statement="Do a thing.",
        entrypoint="solve", public_examples=[], deadline_s=300.0,
    )
    base = heuristic_analyze(task)
    for junk in (None, [], "", 0, {"traps": "not a list"}, {"traps": [1, 2]}):
        out = analysis_from_json(junk, base)
        assert [t.name for t in out.traps] == [t.name for t in base.traps], junk


def test_a_trap_whose_evidence_the_model_called_why_is_still_a_trap():
    """`evidence` is the documented key and `why` is what a model writes when
    it paraphrases the schema. Reading only the documented one silently threw
    the whole trap away."""
    from solvers.analyze import analysis_from_json, heuristic_analyze

    task = SolveTask(
        problem_id="x", language="python", statement="Do a thing.",
        entrypoint="solve", public_examples=[], deadline_s=300.0,
    )
    merged = analysis_from_json(
        {"traps": [{"name": "tail", "why": "the stream runs out",
                    "mitigation": "model the tail"}]},
        heuristic_analyze(task),
    )
    tail = [t for t in merged.traps if t.name == "tail"]
    assert tail and tail[0].evidence == "the stream runs out", merged.traps


def test_the_trap_block_is_what_the_later_prompts_will_read():
    """Every stage after this one is handed the same block. If it renders to
    nothing, three prompts go out having each read the statement alone."""
    from solvers.analyze import heuristic_analyze

    for task in list(_recorded_requests())[:20]:
        block = heuristic_analyze(task).as_prompt_block()
        assert "(no traps recorded)" not in block, task.problem_id
        assert "Traps:" in block and "Algorithm sketch:" in block
        assert len(block) > 400, (task.problem_id, len(block))


# --------------------------------------------------------------------------- #
# Two programs, opposed instructions, the same inputs
# --------------------------------------------------------------------------- #
_PY_INPUTS = [
    {"name": "empty", "args": [[]], "kwargs": {}},
    {"name": "three", "args": [[1, 2, 3]], "kwargs": {}},
]


def _differential(grader):
    from solvers.differential import Differential

    return Differential(grader)


class _FakeGrader:
    """Stands in for `_Grader`, answering the two calls the harness makes."""

    def __init__(self, oracle_runs=None, check=None):
        self._oracle_runs = oracle_runs or {}
        self._check = check
        self.outputs_calls = []
        self.check_calls = []

    def outputs(self, code, language, entrypoint, inputs, budget_s=None):
        self.outputs_calls.append(code)
        return list(self._oracle_runs.get(code, []))

    def check_detailed(self, code, language, entrypoint, examples, names=None,
                       budget_s=None):
        self.check_calls.append((code, list(examples)))
        return self._check(code, examples, names or [])


def _run(value=None, ok=True, error=None, timed_out=False):
    from solvers.verify import _Run

    return _Run(ok=ok, value=value, error=error, timed_out=timed_out)


def test_the_oracles_output_is_the_expectation_and_nothing_is_asked_for_one():
    """The whole design turns on this. The old shape asked a MODEL what a call
    should return; here the reference program is executed and what it produced
    is fed back in as `expected`. That makes the comparison the validator's
    own -- values_equal for Python, outputs_match for Rust -- instead of a
    reimplementation of it."""
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(0), _run(6)]},
        check=lambda code, examples, names: (len(examples), len(examples), [], [], []),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", _PY_INPUTS,
    )
    assert report.ok, report.summary()
    assert report.agreed == 2 and report.mismatch == 0

    # The expectations handed to the grader are the oracle's actual outputs.
    _code, graded = grader.check_calls[0]
    assert [case["expected"] for case in graded] == [0, 6], graded


def test_a_disagreement_blames_the_candidate_and_says_so_in_a_field():
    """THE REGRESSION. The upstream router decided with
    `all("oracle" in case.detail ...)`, and the detail for an honest
    disagreement reads `candidate != oracle` -- which contains the substring
    "oracle". So a report where every case was a real disagreement patched the
    reference instead of the program that ships. `blames` is set where the
    failure is classified, so no wording can move it."""
    failing = {"args": [[1, 2, 3]], "kwargs": {}, "expected": 6}
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(0), _run(6)]},
        check=lambda code, examples, names: (1, 2, ["bad"], [examples[1]], [_run(7)]),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", _PY_INPUTS,
    )
    assert report.mismatch == 1 and report.oracle_crash == 0
    assert report.blame == "candidate", report.summary()
    assert [c.blames for c in report.cases] == ["candidate"], report.cases

    # And the detail is still allowed to mention the reference in prose.
    assert "oracle" not in report.cases[0].blames.replace("candidate", "")


def test_a_reference_that_falls_over_blames_the_reference():
    """A case the oracle could not answer has no expectation, so the candidate
    is never graded on it. Being unable to check a case is not the case
    failing, and grading against a missing value would blame the candidate for
    the reference's crash."""
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(ok=False, error="ZeroDivisionError"), _run(6)]},
        check=lambda code, examples, names: (len(examples), len(examples), [], [], []),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", _PY_INPUTS,
    )
    assert report.oracle_crash == 1 and report.mismatch == 0
    assert report.blame == "oracle", report.summary()
    # Only the one the oracle answered was sent to be graded.
    _code, graded = grader.check_calls[0]
    assert len(graded) == 1 and graded[0]["expected"] == 6, graded


def test_a_disagreement_outranks_a_crash_when_both_happened():
    """A mismatch is a concrete input the candidate got wrong. A reference
    that fell over on some other input is weaker evidence, so the repair goes
    to the candidate."""
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(ok=False, error="boom"), _run(6)]},
        check=lambda code, examples, names: (0, 1, ["bad"], [examples[0]], [_run(7)]),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", _PY_INPUTS,
    )
    assert report.mismatch == 1 and report.oracle_crash == 1
    assert report.blame == "candidate", report.summary()


def test_a_report_that_established_nothing_is_never_a_clean_one():
    """`ran > 0` is the load-bearing clause. An empty suite that reads as ok
    is exactly the 57s-of-290 line this rebuild exists to stop printing."""
    grader = _FakeGrader(check=lambda *a: (0, 0, [], [], []))
    diff = _differential(grader)

    for report in (
        diff.compare("C", "O", "python", "solve", []),
        diff.compare("", "O", "python", "solve", _PY_INPUTS),
        diff.compare("C", "", "python", "solve", _PY_INPUTS),
    ):
        assert not report.ok, report.summary()
        assert report.blame == "", report.summary()


def test_repair_is_monotone_against_the_score():
    """A round that does not improve the score leaves the previous version in
    place, so blaming the wrong program costs a round and never a worse answer
    shipped. Clean outranks everything; among unclean reports fewer
    disagreements win, and agreement is only the tie-break between them."""
    from solvers.differential import DifferentialReport

    clean = DifferentialReport(ran=2, agreed=2)
    partial = DifferentialReport(ran=4, agreed=3, mismatch=1)
    worse = DifferentialReport(ran=4, agreed=1, mismatch=3)

    assert clean.score() > partial.score() > worse.score()
    # More agreement at the same mismatch count is still progress.
    assert DifferentialReport(ran=6, agreed=5, mismatch=1).score() > partial.score()


def test_a_known_disagreement_ranks_below_an_answer_nobody_could_check():
    """The order of `mismatch` and `agreed` inside the score, which decides
    what SHIPS when a repair arrives too late to be graded.

    Ranking `agreed` first read as "more agreement is better" and was measured
    saying something else: a draft that agreed on 1 of 3 inputs and disagreed
    on the other 2 outranked the correction written after it was shown those
    two disagreements, because that correction came back with less than one
    case's worth of clock left and its report was a truthful row of zeroes.
    The KNOWN-WRONG program went out.

    Payment is all or nothing, so that trade is never worth taking: a program
    that disagreed with the reference anywhere is a certain zero, while one
    that was never run is merely unmeasured — and unmeasured still has a
    chance. A verdict outranks an absence, so an absence outranks a bad
    verdict."""
    from solvers.differential import DifferentialReport

    known_wrong = DifferentialReport(ran=3, agreed=1, mismatch=2)
    unmeasured = DifferentialReport(unrun=3)

    assert unmeasured.score() > known_wrong.score()
    # Neither is `ok`: an unrun case is not an agreement, and nothing here may
    # gate the answer cache.
    assert not unmeasured.ok and not known_wrong.ok
    # And among two reports that established nothing, the one that at least
    # tried fewer cases is not somehow better -- fewer unrun cases wins.
    assert DifferentialReport(unrun=1).score() > unmeasured.score()


def test_the_reference_is_not_rerun_while_only_the_candidate_changes():
    """For Rust every run of the oracle otherwise costs a fresh container and
    a full opt-level=2 build, and the oracle normally survives a candidate
    repair untouched. Memoised on the source text, so a real edit to the
    oracle does re-run it."""
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(0), _run(6)], "ORACLE v2": [_run(1), _run(7)]},
        check=lambda code, examples, names: (len(examples), len(examples), [], [], []),
    )
    diff = _differential(grader)

    for candidate in ("C1", "C2", "C3"):
        diff.compare(candidate, "ORACLE", "rust", "main", _PY_INPUTS)
    assert grader.outputs_calls == ["ORACLE"], grader.outputs_calls
    assert diff.oracle_runs == 1 and diff.oracle_reused == 2

    diff.compare("C4", "ORACLE v2", "rust", "main", _PY_INPUTS)
    assert grader.outputs_calls == ["ORACLE", "ORACLE v2"], grader.outputs_calls


def test_a_different_case_list_is_a_different_memo_entry():
    """The bar grows as the model adds cases. Keying only on the oracle source
    would hand the old expectations to the new inputs."""
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(0), _run(6)]},
        check=lambda code, examples, names: (len(examples), len(examples), [], [], []),
    )
    diff = _differential(grader)
    diff.compare("C", "ORACLE", "python", "solve", _PY_INPUTS)
    diff.compare("C", "ORACLE", "python", "solve", _PY_INPUTS + [
        {"name": "more", "args": [[9]], "kwargs": {}},
    ])
    assert grader.outputs_calls == ["ORACLE", "ORACLE"], grader.outputs_calls


def test_the_failure_report_carries_both_sides_and_stays_bounded():
    """The repair prompt is shown what each program produced for the same
    input -- that is the evidence. Past a handful the model stops reading them
    as evidence, and a disagreement on a 200 KB value does not need all of
    it."""
    from solvers.differential import MAX_REPORTED

    big = "x" * 5000
    inputs = [{"name": f"c{i}", "args": [[i]], "kwargs": {}} for i in range(6)]
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(big) for _ in inputs]},
        check=lambda code, examples, names: (
            0, len(examples), ["bad"] * len(examples), list(examples),
            [_run("y" * 5000) for _ in examples],
        ),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    text = report.prompt_text()
    assert report.mismatch == 6
    assert text.count("FAIL ") == MAX_REPORTED, text[:400]
    assert len(text) <= 3500 + 4
    assert "reference produced" in text and "this program produced" in text


def test_two_real_programs_are_compared_by_actually_running_them(monkeypatch):
    """End to end on the real grader, no doubles: a correct candidate agrees
    with the oracle, a subtly wrong one is caught with the concrete input that
    caught it, and the wrong one is blamed rather than the reference.

    The oracle here is written the way the oracle prompt asks for one -- the
    slow literal reading -- and the candidate the way the candidate prompt
    does. They disagree on the empty list, which is exactly the kind of case a
    hidden suite contains and a public example never shows."""
    from solvers.differential import Differential
    from solvers.verify import _Grader

    monkeypatch.setenv("SOLVER_VERIFY_EXECUTOR", "subprocess")

    oracle = (
        "def solve(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += x\n"
        "    return total\n"
    )
    right = "def solve(xs):\n    return sum(xs)\n"
    # max() of an empty sequence raises; sum() returns 0. A naive reading of
    # "the largest running total" that never considers the empty case.
    wrong = "def solve(xs):\n    return max(xs) if xs else None\n"

    inputs = [
        {"name": "empty", "args": [[]], "kwargs": {}},
        {"name": "ones", "args": [[1, 1, 1]], "kwargs": {}},
        {"name": "mixed", "args": [[5, -2, 4]], "kwargs": {}},
    ]

    diff = Differential(_Grader())
    good = diff.compare(right, oracle, "python", "solve", inputs, budget_s=60.0)
    assert good.ok, good.summary()
    assert good.ran == 3 and good.agreed == 3, good.summary()
    assert good.blame == "", good.summary()

    bad = diff.compare(wrong, oracle, "python", "solve", inputs, budget_s=60.0)
    assert not bad.ok, bad.summary()
    assert bad.blame == "candidate", bad.summary()
    assert bad.oracle_crash == 0, bad.summary()
    assert bad.mismatch >= 2, bad.summary()
    assert {c.blames for c in bad.cases} == {"candidate"}, bad.cases

    # The evidence names the case and carries both sides.
    report = bad.prompt_text()
    assert "FAIL " in report and "reference produced" in report, report

    # And the reference was run once for the whole exercise, not once per
    # candidate -- the memo is what makes a repair loop affordable.
    assert diff.oracle_runs == 1 and diff.oracle_reused == 1


def test_a_reference_that_crashes_on_real_code_is_the_one_blamed(monkeypatch):
    """A naive oracle written for small inputs really does fall over on some
    of the cases the model invents. That must never be charged to the
    candidate, which may be perfectly correct on the very same input."""
    from solvers.differential import Differential
    from solvers.verify import _Grader

    monkeypatch.setenv("SOLVER_VERIFY_EXECUTOR", "subprocess")

    oracle = "def solve(xs):\n    return sum(xs) / len(xs)\n"
    candidate = "def solve(xs):\n    return sum(xs) / len(xs) if xs else 0\n"
    inputs = [
        {"name": "empty", "args": [[]], "kwargs": {}},
        {"name": "two", "args": [[2, 4]], "kwargs": {}},
    ]

    report = Differential(_Grader()).compare(
        candidate, oracle, "python", "solve", inputs, budget_s=60.0,
    )
    assert report.oracle_crash == 1, report.summary()
    assert report.mismatch == 0, report.summary()
    assert report.blame == "oracle", report.summary()
    assert report.cases[0].blames == "oracle", report.cases


def test_a_program_that_mutates_its_arguments_is_noted_and_not_failed(monkeypatch):
    """The validator's runner forks a child per case and never compares the
    arguments before and after, so mutation is invisible to the hidden suite.
    Failing it locally would throw away programs that would have scored. The
    upstream harness fails it; this one must not."""
    from solvers.differential import Differential
    from solvers.verify import _Grader

    monkeypatch.setenv("SOLVER_VERIFY_EXECUTOR", "subprocess")

    oracle = "def solve(xs):\n    return sorted(xs)\n"
    # Sorts the caller's list in place, then returns it. Same answer.
    mutating = "def solve(xs):\n    xs.sort()\n    return xs\n"
    inputs = [{"name": "unsorted", "args": [[3, 1, 2]], "kwargs": {}}]

    report = Differential(_Grader()).compare(
        mutating, oracle, "python", "solve", inputs, budget_s=60.0,
    )
    assert report.ok, report.summary()
    assert report.mismatch == 0, report.summary()


def test_the_rust_comparison_is_the_judges_own(monkeypatch):
    """Rust answers are compared by splitting on ASCII whitespace, so a
    println! reference and a print!("{}\\n") candidate are the same answer.
    That comes free from feeding the oracle's stdout back as `expected` and
    letting the grader compare -- there is no second implementation of it
    here to drift."""
    from rlvr.execution.rust_judge import outputs_match

    assert outputs_match("1 2 3\n", "1 2 3")
    assert outputs_match("1\n2\n3\n", "1 2 3")
    assert not outputs_match("1 2 3", "1 2 4")

    # And the harness passes the reference's stdout through untouched, which
    # is the only thing that has to be true for the above to apply.
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run("1 2 3\n")]},
        check=lambda code, examples, names: (1, 1, [], [], []),
    )
    _differential(grader).compare(
        "CANDIDATE", "ORACLE", "rust", "main",
        [{"name": "one", "args": ["3\n1 2 3\n"], "kwargs": {}}],
    )
    _code, graded = grader.check_calls[0]
    assert graded[0]["expected"] == "1 2 3\n", graded


# --------------------------------------------------------------------------- #
# The five stage prompts
# --------------------------------------------------------------------------- #
def _stage_task(language="python"):
    return SolveTask(
        problem_id="x", language=language,
        statement="Implement `solve(xs)`. n is at most 200000.",
        entrypoint="solve" if language == "python" else "main",
        public_examples=[], deadline_s=300.0,
    )


def _repair_prompt(language="python", *, code="def solve(xs):\n    return 0",
                   report="ran=3 agreed=2 mismatch=1", defect=None,
                   found_by="differential", kind="candidate"):
    """One stage-8 prompt, built the way the orchestrator builds it."""
    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    task = _stage_task(language)
    return prompts.build_differential_repair_prompt(
        task, heuristic_analyze(task), code, report,
        kind=kind, defect=defect, found_by=found_by,
    )


def _candidate_prompt(language="python", examples=()):
    """One stage-6 prompt, optionally with worked examples attached."""
    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    task = _stage_task(language)
    task.public_examples = list(examples)
    return prompts.build_candidate_prompt(task, heuristic_analyze(task))


def _all_stage_prompts(task):
    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    analysis = heuristic_analyze(task)
    return {
        "analysis": prompts.build_analysis_prompt(task, analysis),
        "inputs": prompts.build_inputs_prompt(task, analysis),
        # The probe variant is a SEPARATE prompt, not a flag on the same one:
        # it appends the generator task, and appending is where a template
        # goes out unformatted. Listing only the plain variant here is how
        # `{shape}` shipped to the model on every solve that asked for a
        # size probe.
        "inputs+probe": prompts.build_inputs_prompt(
            task, analysis, want_probe=True),
        "oracle": prompts.build_oracle_prompt(task, analysis),
        "candidate": prompts.build_candidate_prompt(task, analysis),
        "repair": prompts.build_differential_repair_prompt(
            task, analysis, "CODE", "ran=3 agreed=2 mismatch=1",
        ),
    }


@pytest.mark.parametrize("language", ["python", "rust"])
def test_no_stage_prompt_ships_an_unrendered_placeholder(language):
    """A `{entrypoint}` that reached a model is a prompt that told it to
    define a function literally called that.

    Named rather than listed, because the list is what failed: this asserted
    `{entrypoint}` and `{language}` by name and passed while `{shape}` — the
    one that tells the probe generator what to RETURN — went out verbatim on
    every solve that asked for a size probe. Any `{lower_case_identifier}`
    surviving into a prompt is the same bug whatever it is called."""
    import re

    # Not every brace is a placeholder: these prompts show JSON shapes on
    # purpose. A format field is a bare lower-case identifier and nothing
    # else -- `{"cases": ...}` and `{}` are not one.
    placeholder = re.compile(r"\{[a-z][a-z0-9_]*\}")
    for name, text in _all_stage_prompts(_stage_task(language)).items():
        assert text.strip(), name
        left = placeholder.findall(text)
        assert not left, f"{name} ({language}) shipped {left}"


def test_the_inputs_turn_is_forbidden_to_answer_its_own_cases():
    """This is what makes the bar evidence instead of an echo. A model that
    supplies expected values has answered a question it was told not to
    answer, and taking them would put that model's reading of the statement
    straight back into the thing meant to check it."""
    from solvers import prompts

    for language in ("python", "rust"):
        text = _all_stage_prompts(_stage_task(language))["inputs"]
        assert "discarded" in text, language
        assert "reference implementation" in text, language

    # And the parser enforces it even when the model ignores the instruction.
    cases = prompts.extract_inputs(
        '```json\n{"cases": ['
        '{"name":"a","args":[[]],"expected":0},'
        '{"name":"b","args":[[1,2]],"expected":3}]}\n```',
        "python",
    )
    assert len(cases) == 2, cases
    assert all("expected" not in case for case in cases), cases


def test_the_two_programs_are_asked_for_opposite_things():
    """The oracle and the candidate are the same problem under opposed
    instructions, and that difference is the entire evidence the design
    produces. If both prompts asked for a fast correct program the two
    answers would share their misreadings and comparing them would establish
    nothing."""
    prompts_by_stage = _all_stage_prompts(_stage_task("python"))
    oracle, candidate = prompts_by_stage["oracle"], prompts_by_stage["candidate"]

    assert "REFERENCE" in oracle and "Do not optimise" in oracle
    assert "tiny inputs" in oracle
    assert "SOLUTION" in candidate and "largest inputs" in candidate
    # The oracle must not be told to be fast, nor the candidate to be slow.
    assert "Nested loops are fine" in oracle
    assert "Nested loops are fine" not in candidate


@pytest.mark.parametrize("language", ["python", "rust"])
def test_the_candidate_is_told_about_the_environment_it_runs_in(language):
    """The measured sentences live in PYTHON_ENVIRONMENT and RUST_ENVIRONMENT
    -- silent integer overflow at opt-level=2 above all -- and the program
    that ships is the one that needs them."""
    text = _all_stage_prompts(_stage_task(language))["candidate"]
    if language == "rust":
        assert "INTEGER OVERFLOW IS SILENT HERE" in text
    else:
        assert "recursion limit" in text


def test_the_repair_prompt_stands_on_its_own():
    """The repair phase names a different model from the candidate phase, so
    the first repair round CANNOT inherit the candidate's conversation. Its
    prompt has to carry the statement, the traps and the program itself or
    the model is reading a failure report about code it cannot see."""
    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    task = _stage_task("python")
    text = prompts.build_differential_repair_prompt(
        task, heuristic_analyze(task), "def solve(xs):\n    return 0\n",
        "ran=3 agreed=2 mismatch=1\nFAIL empty: they disagree",
    )
    assert task.statement in text
    assert "def solve(xs):" in text
    assert "mismatch=1" in text
    assert "Traps:" in text
    assert "You did not write this program" in text


def test_repairing_the_reference_is_not_the_same_job_as_repairing_the_answer():
    """The reference may be as slow as it likes and only has to stop falling
    over; the candidate has to stay fast at the stated maximums. Telling the
    model the wrong one of those is how a repair round makes things worse."""
    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    task = _stage_task("python")
    analysis = heuristic_analyze(task)
    oracle = prompts.build_differential_repair_prompt(
        task, analysis, "CODE", "report", kind="oracle")
    candidate = prompts.build_differential_repair_prompt(
        task, analysis, "CODE", "report", kind="candidate")

    assert "do not make it faster" in oracle
    assert "SUBMITTED" in candidate and "quadratic" in candidate
    assert "do not make it faster" not in candidate


def test_a_reply_that_is_not_json_costs_the_analysis_and_nothing_else():
    """Stage 3 is the one stage allowed to produce nothing."""
    from solvers import prompts

    for junk in ("", "I could not do that", "```python\nx = 1\n```"):
        assert prompts.extract_analysis(junk) is None, junk
    assert prompts.extract_inputs(junk, "python") == []


def test_inputs_are_read_out_of_prose_and_deduplicated():
    """Models write the array unfenced, and they repeat cases. Every case is
    an executor run inside the solve's own deadline, so a duplicate is paid
    for twice and proves nothing the first one did not."""
    from solvers import prompts

    cases = prompts.extract_inputs(
        '```json\n{"cases":[{"args":[[1]]},{"args":[[1]]},{"args":[[2]]}]}\n```',
        "python",
    )
    assert len(cases) == 2, cases
    # A bare list, no wrapper object.
    assert len(prompts.extract_inputs(
        '```json\n[{"args":[[1]]},{"args":[[2]]}]\n```', "python")) == 2


def test_a_rust_case_is_the_bytes_on_stdin():
    """Rust is graded by running a program, so a case is its stdin -- and the
    grader reads that from args[0]."""
    from solvers import prompts

    cases = prompts.extract_inputs(
        '```json\n{"cases":[{"name":"one","stdin":"3\\n1 2 3\\n"}]}\n```', "rust",
    )
    assert cases == [{"name": "one", "args": ["3\n1 2 3\n"],
                      "kwargs": {}, "notes": ""}], cases


def test_a_suite_the_clock_cut_short_never_reads_as_agreement():
    """REGRESSION. `check_detailed` reports `total` as every case it was
    HANDED, not every case that ran -- a budget that expires mid-suite leaves
    the rest in neither `passed` nor `failures`. Counting the difference as
    agreement made a report that proved two cases out of five read exactly as
    clean as one that proved all five, which is the `corrected=0/18
    exit=converged` line this whole design exists to stop printing."""
    inputs = [{"name": f"c{i}", "args": [[i]], "kwargs": {}} for i in range(5)]
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(i) for i in range(5)]},
        # Two ran and passed; three never ran. No failures reported.
        check=lambda code, examples, names: (2, 5, [], [], []),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    assert report.unrun == 3, report.summary()
    assert report.agreed == 2, report.summary()
    assert not report.ok, f"a third of the suite proved nothing: {report.summary()}"
    assert "unrun=3" in report.summary(), report.summary()

    # And a suite that really did run every case is still clean.
    whole = _FakeGrader(
        oracle_runs={"ORACLE": [_run(i) for i in range(5)]},
        check=lambda code, examples, names: (5, 5, [], [], []),
    )
    good = _differential(whole).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    assert good.ok and good.unrun == 0, good.summary()


def test_the_value_a_failing_program_produced_reaches_the_repair_prompt():
    """REGRESSION. `failed` and `actuals` are appended together, once per
    failing case, so they are parallel TO EACH OTHER and not to the case list.
    Indexing `actuals` by a position in the case list read the wrong element
    whenever the failures were not a prefix -- and silently, because a short
    read fell off the end and became 'this program produced nothing'. Telling
    a repair model that a program produced nothing when it produced a wrong
    value points the whole round at the wrong fault."""
    inputs = [{"name": f"c{i}", "args": [[i]], "kwargs": {}} for i in range(5)]
    grader = _FakeGrader(
        oracle_runs={"ORACLE": [_run(i) for i in range(5)]},
        # Only the LAST case fails, so `actuals` has exactly one entry.
        check=lambda code, examples, names: (
            4, 5, ["bad"], [examples[4]], [_run(99)],
        ),
    )
    report = _differential(grader).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    assert report.mismatch == 1, report.summary()
    case = report.cases[0]
    assert case.name == "c4", case
    assert case.candidate_out == "99", case
    assert case.oracle_out == "4", case
    assert "produced nothing" not in case.detail, case
    assert "99" in report.prompt_text(), report.prompt_text()


@pytest.mark.parametrize("language", ["python", "rust"])
def test_no_stage_prompt_ships_a_doubled_brace(language):
    """REGRESSION. The JSON shapes in these prompts are written with `{{` so
    that `str.format` renders them as `{`. The Rust inputs template carries no
    placeholders and so was never formatted, and shipped its braces doubled to
    the model on every Rust solve -- half of live traffic. A model shown
    `{{"cases": ...}}` is being told the wrong shape to reply in.

    `}}` on its own is NOT the tell: `{"args": [...], "kwargs": {}}` is a
    correctly rendered JSON shape with a nested empty object, and forbidding
    it would forbid the shape the probe generator is asked for. An UNRENDERED
    template always carries `{{`, so that is what is banned -- plus the
    balance, which an unrendered one keeps but a half-formatted one does
    not."""
    for name, text in _all_stage_prompts(_stage_task(language)).items():
        assert "{{" not in text, (name, language)
        depth = lowest = 0
        for character in text:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                lowest = min(lowest, depth)
        assert (depth, lowest) == (0, 0), (name, language, depth, lowest)

    # The shape it does ship is the one the parser reads back.
    from solvers import prompts

    text = _all_stage_prompts(_stage_task(language))["inputs"]
    assert '"cases"' in text and '{\n  "cases"' in text, text[:200]
    key = "stdin" if language == "rust" else "args"
    assert f'"{key}"' in text, text


def test_a_grader_that_cannot_run_costs_the_check_and_never_the_answer():
    """The standing rule everywhere else in this solver: an executor that
    cannot be built costs the CHECK, not the answer. `_run_self_tests` has
    caught this since it was written; the differential did not, and would
    have propagated a missing Docker daemon straight up into the solve.

    Both halves have to degrade, and both have to degrade to 'nothing was
    established' rather than to 'everything agreed'."""
    inputs = [{"name": "a", "args": [[1]], "kwargs": {}}]

    class _Broken:
        def outputs(self, *a, **k):
            raise RuntimeError("no docker daemon")

        def check_detailed(self, *a, **k):
            raise RuntimeError("no docker daemon")

    report = _differential(_Broken()).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    assert not report.ok, report.summary()
    assert report.agreed == 0, report.summary()

    # And when only the candidate side is broken, the reference still ran --
    # so the failure is recorded as unrun, not as agreement.
    class _HalfBroken(_Broken):
        def outputs(self, *a, **k):
            return [_run(1)]

    half = _differential(_HalfBroken()).compare(
        "CANDIDATE", "ORACLE", "python", "solve", inputs,
    )
    assert not half.ok, half.summary()
    assert half.unrun == 1, half.summary()
    assert half.agreed == 0, half.summary()


# --------------------------------------------------------------------------- #
# Which artifact a repair round patches
# --------------------------------------------------------------------------- #
def _mismatch_report(*names):
    from solvers.differential import CaseResult, DifferentialReport

    return DifferentialReport(
        ran=len(names) + 1, agreed=1, mismatch=len(names),
        cases=[CaseResult(name=n, blames="candidate") for n in names],
    )


def test_the_same_failure_blamed_twice_hands_the_next_round_to_the_reference():
    """THE FLIP RULE. A disagreement accuses the candidate by default, and
    that default can be wrong -- on the closest measurement available, two
    independent encodings of one statement, the shipped program was the wrong
    party 7 times in 27 and the second encoding 11.

    Two rounds, not one: a first disagreement really is likelier the
    candidate's fault, since it was written under the harder instruction, and
    one failed repair is ordinary. Two failed repairs on the SAME case with
    nothing else accusing it is the signature of a reference that is itself
    wrong."""
    from solvers.differential import Router

    router = Router()
    report = _mismatch_report("boundary")

    assert router.choose(report) == "candidate"
    assert router.choose(report) == "candidate"
    assert router.choose(report) == "oracle", "the router never reconsidered"
    assert router.flipped == 1

    # And it resets rather than latching: if the reference was not the problem
    # either, the next round goes back to the candidate.
    assert router.choose(report) == "candidate"
    assert router.flipped == 1


def test_a_solve_making_progress_never_flips():
    """A DIFFERENT set of failing cases is a different argument, so it starts
    its own count. Without that, three rounds each fixing one case and
    uncovering another would flip on the third and start patching a reference
    that was never implicated."""
    from solvers.differential import Router

    router = Router()
    for name in ("first", "second", "third", "fourth"):
        assert router.choose(_mismatch_report(name)) == "candidate", name
    assert router.flipped == 0


def test_a_signal_that_is_not_the_reference_is_never_second_guessed():
    """`compile_defect` and the size probe accuse the candidate on their own
    evidence. When either has spoken there is no dispute to arbitrate, so the
    count is not even consulted."""
    from solvers.differential import Router

    router = Router()
    report = _mismatch_report("boundary")
    for _ in range(5):
        assert router.choose(
            report, candidate_blamed_independently=True) == "candidate"
    assert router.flipped == 0


def test_a_reference_that_fell_over_is_repaired_without_waiting_for_two_rounds():
    """The flip rule arbitrates DISAGREEMENTS. A reference that could not
    produce a value at all is not a disagreement -- nothing is in dispute, the
    reference is simply broken -- so it is repaired at once."""
    from solvers.differential import CaseResult, DifferentialReport, Router

    router = Router()
    crashed = DifferentialReport(
        ran=2, agreed=1, oracle_crash=1,
        cases=[CaseResult(name="empty", blames="oracle")],
    )
    assert router.choose(crashed) == "oracle"
    assert router.flipped == 0, "a crash is not a flip"

    # And a clean report asks for no repair at all.
    assert Router().choose(DifferentialReport(ran=3, agreed=3)) == ""


def test_the_phase_markers_the_fakes_key_on_are_real():
    """The scripted fakes tell one stage's prompt from another by a marker
    string. A marker that stops matching does not fail loudly -- it hands one
    stage's scripted reply to a different stage, and the test then asserts
    about a solve that never happened. That is not hypothetical: the analysis
    turn silently ate the candidate's reply for a whole afternoon because its
    marker was wording the prompt had never used.

    So the table is checked against the real builders, and each marker has to
    be unique to its own prompt."""
    from types import SimpleNamespace

    from solvers import prompts
    from solvers.analyze import heuristic_analyze

    for language, entrypoint in (("python", "solve"), ("rust", "main")):
        task = SimpleNamespace(
            language=language, statement="S", entrypoint=entrypoint,
            public_examples=[], deadline_s=300.0,
        )
        analysis = heuristic_analyze(task)
        built = {
            "analysis": prompts.build_analysis_prompt(task, analysis),
            "inputs": prompts.build_inputs_prompt(task, analysis),
            "oracle": prompts.build_oracle_prompt(task, analysis),
            "candidate": prompts.build_candidate_prompt(task, analysis),
            "repair": prompts.build_differential_repair_prompt(
                task, analysis, "CODE", "report"),
        }
        for phase, text in built.items():
            assert _phase_of(text) == phase, (
                f"{language}: the {phase} prompt reads as {_phase_of(text)!r}"
            )


def test_a_program_repaired_out_of_a_defect_outranks_the_broken_one():
    """REGRESSION. Repair is monotone against a score, and that score was the
    differential's alone. Every report that established nothing scored the
    same -- so when there were no synthesized inputs, a program repaired out
    of a structural defect could not outrank the defective one it replaced,
    and the broken version shipped with `regressed=1` in the log.

    On live traffic this is the common case, not a corner: no task ships
    public examples, and a Rust answer that will not compile is caught by the
    compile gate alone."""
    task = SolveTask(
        problem_id="defect-rank", language="rust", statement="Print 42.",
        entrypoint="main", public_examples=[], deadline_s=120.0,
    )
    solver = VerifyingSolver(
        _Backend(["```rust\nfn helper() {}\n```",
                  '```rust\nfn main() { println!("42"); }\n```']),
        reserve_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))

    assert "println!" in answer.code, (
        f"the defective program outranked its own repair: {answer.code!r}"
    )
    assert answer.diagnostics["regressed"] == 0, answer.diagnostics
    assert answer.diagnostics["patched"] == ["cand"], answer.diagnostics

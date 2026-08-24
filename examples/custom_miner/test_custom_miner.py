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

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_miner import CustomMiner, SolveTask  # noqa: E402
from solvers.chatgpt_cdp import ChatGPTPool, _Tab  # noqa: E402
from solvers.verify import Answer, VerifyingSolver  # noqa: E402

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


class _Chat:
    def __init__(self, replies): self._replies, self._n = replies, -1
    async def send(self, text, timeout_s):
        self._n += 1
        return self._replies[min(self._n, len(self._replies) - 1)]
    async def close(self): pass


class _Backend:
    def __init__(self, replies): self._replies = replies
    async def open(self): return _Chat(self._replies)
    async def aclose(self): pass
    def stats(self): return {}


def _solver(replies, **kw):
    kw.setdefault("safety_margin_s", 0)
    kw.setdefault("max_budget_s", 120)
    return VerifyingSolver(_Backend(replies), **kw)


# --------------------------------------------------------------------------- #
# The wire contract: what the validator actually checks before paying.
# --------------------------------------------------------------------------- #
def test_reply_passes_every_validator_acceptance_check():
    miner_kp, validator_kp = keypair.create_from_uri("//Bob"), keypair.create_from_uri("//Alice")
    solver = _solver([WRONG, RIGHT])
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
    # 4. within the per-response byte cap
    assert len(response.content) <= DemoMinerSettings(_env_file=None).miner_max_request_bytes
    assert "while n > 0" in payload.code  # the repaired answer, not the first draft


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


def test_an_unverified_answer_is_not_cached():
    solver = _solver([WRONG])
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert solver.stats()["solver"]["cache_hits"] == 0


# --------------------------------------------------------------------------- #
# Budget: a correct answer after the cutoff pays exactly the same as a wrong one.
# --------------------------------------------------------------------------- #
class _SlowBackend:
    async def open(self):
        class Slow:
            async def send(self, text, timeout_s):
                await asyncio.sleep(4)
                return WRONG
            async def close(self): pass
        return Slow()
    async def aclose(self): pass
    def stats(self): return {}


def test_a_solve_never_outruns_its_advertised_deadline():
    solver = VerifyingSolver(_SlowBackend(), max_attempts=9, safety_margin_s=0, max_budget_s=30)
    started = time.monotonic()
    result = asyncio.run(solver.solve_task(DIGITS, timeout_s=30))
    assert time.monotonic() - started < 30
    assert result.code.strip()  # best-so-far, never nothing when we have something


def test_the_safety_margin_is_held_back_from_the_advertised_deadline():
    solver = VerifyingSolver(_SlowBackend(), max_attempts=9, safety_margin_s=15, max_budget_s=240)
    started = time.monotonic()
    asyncio.run(solver.solve_task(DIGITS, timeout_s=40))
    assert time.monotonic() - started < 30  # 40s advertised minus a 15s margin


def test_a_dead_backend_yields_an_empty_answer_rather_than_an_exception():
    class Broken:
        async def open(self): raise RuntimeError("browser died")
        async def aclose(self): pass
        def stats(self): return {}
    result = asyncio.run(
        VerifyingSolver(Broken(), safety_margin_s=0, max_budget_s=60).solve_task(DIGITS, 60)
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


def _pool(replaceable: bool) -> ChatGPTPool:
    pool = ChatGPTPool.__new__(ChatGPTPool)
    pool._free, pool._size, pool._lost, pool._browsers = asyncio.Queue(), 1, 0, []

    async def spawn(context, label):
        return _Tab(pool, _DeadPage(), context, f"{label}-new") if replaceable else None

    pool._spawn = spawn
    return pool


async def _use_dead_tab(pool: ChatGPTPool) -> None:
    await pool._free.put(_Tab(pool, _DeadPage(), object(), "dead#1"))

    class LeaseOnly:
        async def open(self): return await pool._free.get()
        async def aclose(self): pass
        def stats(self): return pool.stats()

    await VerifyingSolver(
        LeaseOnly(), max_attempts=1, safety_margin_s=0, max_budget_s=30
    ).solve_task(DIGITS, timeout_s=30)


def test_a_tab_that_dies_is_replaced_not_recycled():
    pool = _pool(replaceable=True)
    asyncio.run(_use_dead_tab(pool))
    assert pool._lost == 1 and pool._size == 1 and pool._free.qsize() == 1


def test_an_unreplaceable_dead_tab_retires_instead_of_poisoning_the_pool():
    pool = _pool(replaceable=False)
    asyncio.run(_use_dead_tab(pool))
    assert pool._lost == 1 and pool._size == 0 and pool._free.qsize() == 0


def test_the_submit_phase_is_bounded_so_playwright_cannot_overrun_the_budget():
    import inspect

    send = inspect.getsource(_Tab.send)
    submit = inspect.getsource(_Tab._submit)
    # Playwright auto-waits 30s per action by default; unbounded that is ~90s
    # of overrun on a budget the solver carefully computed.
    assert "asyncio.wait_for" in send
    assert submit.count("timeout=ui_ms") == 2


def test_the_pool_starts_lazily_so_any_host_can_serve_it():
    """Regression: start() was only called by run_chatgpt_miner.py, which needs
    a live chain. Hosted anywhere else the pool stayed empty and open() blocked
    on the queue forever, so every solve returned nothing."""
    import inspect

    from solvers.chatgpt_cdp import ChatGPTPool

    assert "await self.start()" in inspect.getsource(ChatGPTPool.open)
    start = inspect.getsource(ChatGPTPool.start)
    assert "_start_lock" in start and "if self._started" in start, "start must be idempotent"


def test_starting_twice_connects_only_once():
    import asyncio

    from solvers.chatgpt_cdp import ChatGPTPool

    pool = ChatGPTPool([9222])
    calls = []

    async def fake_connect():
        calls.append(1)

    pool._connect = fake_connect

    async def go():
        await pool.start()   # an explicit start, as run_chatgpt_miner.py does
        await pool.start()   # a second call, as a lazy start from open() would be

    asyncio.run(go())
    assert calls == [1], "start() must connect once no matter how often it is called"


# --------------------------------------------------------------------------- #
# Multi-provider chain
# --------------------------------------------------------------------------- #
def test_answer_reports_whether_it_verified():
    verified = asyncio.run(_solver([RIGHT]).solve_task(DIGITS, timeout_s=120))
    assert verified.verified and verified.passed == verified.total > 0
    unverified = asyncio.run(_solver([WRONG]).solve_task(DIGITS, timeout_s=120))
    assert not unverified.verified


def test_the_chain_stops_at_the_first_verified_provider():
    """A verified answer must end the chain — later providers cost real money."""
    from solvers.multi import FallbackSolver

    second = _solver([RIGHT])
    chain = FallbackSolver(
        [("first", _solver([RIGHT])), ("second", second)], safety_margin_s=0
    )
    answer = asyncio.run(chain.solve_task(DIGITS, timeout_s=120))
    assert answer.verified
    stats = chain.stats()
    assert stats["verified_by"]["first"] == 1
    assert stats["attempts"]["second"] == 0, "second provider must not be called"


def test_the_chain_falls_through_to_a_provider_that_can_solve_it():
    from solvers.multi import FallbackSolver

    chain = FallbackSolver(
        [("weak", _solver([WRONG, WRONG, WRONG])), ("strong", _solver([RIGHT]))],
        safety_margin_s=0,
    )
    answer = asyncio.run(chain.solve_task(DIGITS, timeout_s=120))
    assert answer.verified and "while n > 0" in answer.code
    assert chain.stats()["verified_by"]["strong"] == 1


def test_when_nothing_verifies_the_best_partial_answer_is_returned():
    """Never return nothing when some provider produced runnable code."""
    from solvers.multi import FallbackSolver

    chain = FallbackSolver(
        [("a", _solver(["no code here"])), ("b", _solver([WRONG]))], safety_margin_s=0
    )
    answer = asyncio.run(chain.solve_task(DIGITS, timeout_s=120))
    assert not answer.verified
    assert answer.code.strip(), "the runnable-but-wrong answer beats an empty one"


def test_an_unknown_backend_name_is_rejected_by_name():
    from solvers.multi import KNOWN_BACKENDS, build_backend

    with pytest.raises(SystemExit, match="unknown backend"):
        build_backend("llama")
    assert {"claude", "gemini", "chatgpt"} == set(KNOWN_BACKENDS)


def test_a_single_backend_builds_a_plain_verifying_solver(monkeypatch):
    """One name must not pay the chain's budget-splitting overhead."""
    import solvers.multi as multi

    monkeypatch.setattr(multi, "build_backend", lambda name: _Backend([RIGHT]))
    assert isinstance(multi.build_solver(["claude"]), VerifyingSolver)
    assert isinstance(multi.build_solver(["claude", "gemini"]), multi.FallbackSolver)

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
    build_code_prompt,
    build_repair_prompt,
    build_tests_prompt,
    extract_code,
    python_defect,
)
from solvers.verify import (  # noqa: E402
    EMPTY_HANDED_FLOOR_S,
    MAX_PASSES,
    SECOND_OPINION_FLOOR_S,
    SECOND_OPINION_PASSES,
    Answer,
    VerifyingSolver,
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


class _Chat:
    def __init__(self, replies, provider="claude"):
        self._replies, self._n, self.provider = replies, -1, provider
    async def send(self, text, timeout_s):
        self._n += 1
        return self._replies[min(self._n, len(self._replies) - 1)]
    async def close(self): pass


class _Backend:
    def __init__(self, replies, provider="claude"):
        self._replies, self._provider = replies, provider
    async def open(self, avoid=None): return _Chat(self._replies, self._provider)
    async def aclose(self): pass
    def stats(self): return {}


def _solver(replies, **kw):
    kw.setdefault("safety_margin_s", 0)
    kw.setdefault("max_budget_s", 120)
    return VerifyingSolver(_Backend(replies), **kw)


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
    # 4. within the per-RESPONSE byte cap. Not the request cap, which is eight
    #    times larger and belongs to the other direction: the validator reads a
    #    bounded number of bytes back and discards the whole response if it runs
    #    over, so checking the wrong one here would pass a reply that is thrown
    #    away on arrival.
    assert len(response.content) <= Settings().miner_max_response_bytes
    assert "while n > 0" in payload.code  # the repaired answer, not the first draft


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


async def _use_dead_tab(pool: BrowserFleet) -> None:
    await pool._free.put(_Tab(pool, _DeadPage(), object(), "dead#1", chatgpt_site()))

    class LeaseOnly:
        # Leases exactly as BrowserPool.open() does, minus tab.start() (which
        # would fail first on a dead page). Setting `leased` matters: release()
        # ignores a tab that was never leased, so skipping it here would make
        # the test assert against a no-op.
        async def open(self, avoid=None):
            tab = await pool._free.get()
            tab.leased = True
            return tab

        async def aclose(self): pass
        def stats(self): return pool.stats()

    await VerifyingSolver(
        LeaseOnly(), max_attempts=1, safety_margin_s=0, max_budget_s=30,
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
    pool = _pool(replaceable=True, site=site)
    asyncio.run(_use_dead_tab(pool))
    assert pool._lost == 1 and pool._size == 1 and pool._free.qsize() == 1


@SITES
def test_an_unreplaceable_dead_tab_retires_instead_of_poisoning_the_pool(site):
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
    assert submit.count("timeout=ui_ms") == 2


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


class _FakePage:
    """A page whose DOM is `{selector: [nodes]}` and can change on submit."""

    def __init__(self, dom, on_click=None):
        self.dom, self.on_click = dom, on_click
        self.typed, self.pressed, self.clicked = [], [], []
        # Every navigation and close, so "did this reload?" and "did this throw
        # the tab away?" are assertable rather than inferred.
        self.navigated, self.closed = [], False
        self.on_goto = None

        page = self

        class _Keyboard:
            async def insert_text(self, text):
                page.typed.append(text)

            async def press(self, key):
                page.pressed.append(key)
                if page.on_click:
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
        def __init__(self, replies): self._replies = replies
        async def open(self, avoid=None):
            provider = "chatgpt" if avoid == "claude" else "claude"
            seen.append(provider)
            return _Chat(self._replies[len(seen) - 1], provider)
        async def aclose(self): pass
        def stats(self): return {}

    # First model gets it right: one provider asked.
    seen.clear()
    solver = VerifyingSolver(_Fleet([[RIGHT], [RIGHT]]), safety_margin_s=0, max_budget_s=120)
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified and seen == ["claude"], seen
    assert solver.stats()["providers"]["claude"]["verified"] == 1

    # First model keeps failing: the other one is asked and wins.
    seen.clear()
    solver = VerifyingSolver(
        _Fleet([[WRONG, WRONG, WRONG], [RIGHT]]), safety_margin_s=0, max_budget_s=120
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified and seen == ["claude", "chatgpt"], seen
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
        _Counting([RIGHT]), safety_margin_s=0, max_budget_s=120, second_opinion=True
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))
    assert answer.code, "the answer still comes back"
    assert answer.verified is False, "nothing can verify it, and it must not claim to"
    assert calls == ["first"], f"asked a second model for nothing: {calls}"


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
        _Silent([""]), safety_margin_s=0, max_budget_s=120, second_opinion=True
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))
    assert calls == ["first", "claude"], f"never fell back: {calls}"
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
        _Fleet(), max_attempts=1, safety_margin_s=0, max_budget_s=120, second_opinion=False
    )
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert seen == [None], seen


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


def test_no_backend_anywhere_reads_an_api_key():
    """Every backend drives a browser. Nothing in the package reads a key, and
    nothing imports a provider SDK — a regression would be invisible otherwise,
    because a key-reading backend works fine right up until it bills someone."""
    import inspect
    from pathlib import Path

    from solvers import claude_web
    from solvers.roster import PROVIDERS, site_for

    assert set(PROVIDERS) == {"claude", "chatgpt"}
    assert "claude.ai" in site_for("claude").url
    assert "chatgpt.com" in site_for("chatgpt").url

    banned = ("API_KEY", "import anthropic", "from anthropic", "google.genai")
    package = Path(inspect.getfile(claude_web)).parent
    for module in sorted(package.glob("*.py")):
        body = module.read_text()
        for needle in banned:
            assert needle not in body, f"{module.name} references {needle!r}"


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


def test_pressing_enter_is_the_fallback_when_no_send_button_matches():
    """Safe only because insert_text already put the whole multi-line prompt in."""
    page = _FakePage({"#composer": [_Node()], "#send": [], "#assistant": []})
    asyncio.run(_tab(page, _site()).send("line one\nline two", 1.0))
    assert page.typed == ["line one\nline two"] and page.pressed == ["Enter"]


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
    from solvers.prompts import NO_CODE, build_repair_prompt

    prompt = build_repair_prompt([], "rust", "main", defect=NO_CODE)
    assert "did not reach me as code" in prompt
    assert "artifact" in prompt and "canvas" in prompt
    assert "I ran" not in prompt, "still claims to have run something"
    assert "WRONG" not in prompt, "still blames the answer for a delivery fault"


def test_a_real_failure_still_quotes_the_evidence():
    """The other branch must keep working: when code DID arrive and failed, the
    concrete counter-example is what makes the repair loop converge."""
    from solvers.prompts import build_repair_prompt

    prompt = build_repair_prompt(["g(*[12345]) returned 14, expected 15"], "python", "g")
    assert "returned 14, expected 15" in prompt
    assert "I ran `g` against the examples" in prompt


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
    from solvers.prompts import build_repair_prompt

    prompt = build_repair_prompt(
        [], "rust", "main", defect="the program does not define `fn main()`"
    )
    assert "does not define `fn main()`" in prompt
    assert "could not run" in prompt
    assert "I ran" not in prompt, "still claims to have executed it"
    assert "WRONG" not in prompt, "still blames logic that never ran"


def test_the_repair_round_hears_about_the_defect_not_about_the_examples():
    """The wiring, not the wording: `_grade` returns a defect OR failures, never
    both, and merging them into one list of "problems" was what put a defect
    under the "I ran it" heading in the first place."""
    prompts: list[str] = []

    class _Recording(_Chat):
        async def send(self, text, timeout_s):
            prompts.append(text)
            return await super().send(text, timeout_s)

    class _Backend2(_Backend):
        async def open(self, avoid=None):
            return _Recording(self._replies, self._provider)

    task = SolveTask(
        problem_id="defect", language="rust", statement="Print 42.",
        entrypoint="main", public_examples=[], deadline_s=120.0,
    )
    solver = VerifyingSolver(
        # Turn 1 answers the cases request; then a helper with no `fn main`,
        # which is the defect this test is about; then a real program.
        _Backend2(['```json\n[{"name": "n", "args": ["\\n"], "expected": "42"}]\n```',
                   "```rust\nfn helper() {}\n```",
                   '```rust\nfn main() { println!("42"); }\n```']),
        safety_margin_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(task, 120.0))

    assert "println!" in answer.code, "never got past the defect"
    # Three prompts now: the cases turn, the program turn, then the repair.
    assert len(prompts) == 3, f"expected cases, program, repair; got {len(prompts)}"
    assert "<task>" in prompts[0], "the first turn must be the cases request"
    assert "could not run" in prompts[2] and "fn main" in prompts[2]
    assert "against the examples" not in prompts[1]


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


def test_the_edge_case_checklist_names_the_cases_rather_than_gesturing_at_them():
    """"Handle edge cases" is a line every model agrees to and none acts on.
    What earns its place in the prompt is a case with a right answer the
    statement implies and a wrong answer a plausible implementation gives."""
    from solvers.prompts import build_code_prompt

    for language, entry in (("python", "solve"), ("rust", "main")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        for case in ("n = 0", "n = 1", "empty", "BOTH ENDS", "EXTREME VALUES",
                     "DEGENERATE SHAPE", "-1"):
            assert case in prompt, f"{language} prompt never mentions {case!r}"
        assert "off-by-one" in prompt, f"{language} prompt omits the boundary case"


def test_each_language_is_warned_about_its_own_way_of_losing_a_large_number():
    """The large-number failure is not the same failure in both languages, and
    telling either one the other's story wastes the only prompt there is:
    Python cannot overflow at all, and Rust cannot grow an integer."""
    from solvers.prompts import build_code_prompt

    rust = build_code_prompt("rust", "Do a thing.", "main", [])
    python = build_code_prompt("python", "Do a thing.", "solve", [])

    assert "OVERFLOW IS SILENT" in rust and "i64" in rust
    assert "overflow" not in python.lower().replace("never overflow", ""), (
        "told Python about an overflow it cannot have"
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
    from solvers.prompts import build_code_prompt

    prompt = build_code_prompt(
        "python", "Do a thing.", "solve",
        [{"args": [[1]], "kwargs": {}, "expected": 1}],
    )
    assert "a floor, not the specification" in prompt
    assert "survive the cases below" in prompt, "the label does not point forward"
    assert prompt.index("<problem") < prompt.index("<examples"), (
        "the examples are separated from the problem they belong to"
    )


def test_how_to_answer_comes_last_where_it_is_most_likely_to_be_obeyed():
    """Everything that shapes HOW to answer sits closest to where generation
    begins. The problem and its examples come first, because that is the thing
    being reasoned about; the method comes last, because that is the thing being
    obeyed."""
    from solvers.prompts import build_code_prompt

    for language, entry in (("rust", "main"), ("python", "solve")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        assert prompt.index("<problem") < prompt.index("<method>")
        assert prompt.index("<contract>") < prompt.index("<method>")
        assert prompt.rstrip().endswith("</method>"), prompt[-80:]


def test_the_output_contract_holds_the_first_word_and_the_nudge_the_last():
    """The only instruction whose failure costs the ENTIRE answer rather than
    degrading it, so it gets both ends and nothing competes for either."""
    from solvers.prompts import build_code_prompt

    for site, language, entry in ((claude_site(), "rust", "main"),
                                  (chatgpt_site(), "python", "solve")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        assert prompt.startswith("<output>"), prompt[:40]
        head = prompt.split("</output>")[0]
        # TWO blocks now, and the count is the load-bearing part: `extract_code`
        # picks the block that DEFINES the entrypoint, so a third one is the
        # thing that could still confuse it.
        assert "EXACTLY TWO fenced blocks" in head, head[:200]
        assert "Never a third block" in head, head[:200]
        # ...and the site's nudge, appended after everything, repeats it.
        assert site.nudge.startswith(
            "START your reply with the fenced block"
        ), site.nudge[:70]


def test_the_section_that_asks_for_a_check_does_not_say_before_you_answer():
    """The phrase itself is what caused the narration. A model told to do
    something "before you answer" writes it down, at length, and only then
    starts the program — so the section is not named that either."""
    from solvers.prompts import build_code_prompt

    prompt = build_code_prompt("rust", "Do a thing.", "main", [])
    assert "before_you_answer" not in prompt, "the tag name reintroduces the phrase"
    assert "Write the program FIRST" in prompt
    assert "silently" in prompt


def test_the_self_check_names_the_bugs_that_actually_reached_the_grader():
    """Read off 43 real submissions. Ten were the model's own bugs and EIGHT of
    those were visible on a careful re-read of the program itself — a helper
    called but never written, a `[-1]` on a list the program's own parser can
    empty, a sentinel of 255 used to index five buckets, an id used where a
    position was meant, a condition already guaranteed by the arm it sat in.

    Generic advice does not reach any of that. Each line below is one of them.
    """
    from solvers.prompts import SELF_CHECK, build_code_prompt

    for phrase in (
        "you also wrote",          # helper called but never defined
        "index is in range",       # sentinel-as-index, id-as-position
        "branch is reachable",     # dead arm, arm that discards its result
        "empty case",              # [-1] on a list its own code can empty
        "rebuild ALL of it",       # refresh that forgets one field
        "that branch can undo",    # counter committed before the wrong path
    ):
        assert phrase in SELF_CHECK, f"the self-check dropped {phrase!r}"
    for language, entry in (("rust", "main"), ("python", "solve")):
        for cases in ("ask", [{"name": "n", "args": [[]], "expected": 0}], []):
            prompt = build_code_prompt(language, "Do a thing.", entry, [], cases=cases)
            reply = "the two blocks" if cases == "ask" else "the program"
            assert SELF_CHECK.format(reply=reply) in prompt


def test_a_repair_round_is_sent_back_through_the_checklist():
    """Repairs go into the SAME conversation, so the checklist is still above
    them — and a repair that fixes the failing example while breaking a
    boundary scores the same zero as the answer it replaced."""
    from solvers.prompts import build_repair_prompt

    prompt = build_repair_prompt(["solve([]) raised IndexError"], "python", "solve")
    assert "edge-case checklist" in prompt
    assert "n = 1" in prompt and "largest values" in prompt
    # ...but a DEFECT is not a logic problem, and must not be answered with it.
    for defect in ("the program does not define fn main()", NO_CODE):
        repair = build_repair_prompt([], "rust", "main", defect=defect)
        assert "edge-case checklist" not in repair, (
            f"answered a delivery failure with logic advice: {defect!r}"
        )


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
    from solvers.prompts import build_repair_prompt, rust_defect

    prose = (
        "Let me carefully work through this problem.\n"
        "We have a stream of bytes described by a DAG of nodes.\n"
        "State during processing: the current pending record length."
    )
    defect = rust_defect(extract_code(prose, "main", "rust"))
    assert defect == NO_CODE, f"reasoning was diagnosed as {defect!r}"

    repair = build_repair_prompt([], "rust", "main", defect=defect)
    assert "did not reach me as code" in repair, repair[:120]
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
    from solvers.prompts import build_repair_prompt

    solver = _solver([])
    task = SimpleNamespace(
        language="rust", entrypoint="main", statement="do a thing", public_examples=[],
    )
    candidate = solver._grade("```rust\nfn main() { nope(); }\n```", task)

    assert candidate.defect is not None, "a program that cannot build was taken as-is"
    assert "does not compile" in candidate.defect
    assert candidate.code.strip(), "threw the answer away instead of repairing it"

    repair = build_repair_prompt([], "rust", "main", defect=candidate.defect)
    assert "could not run your previous reply" in repair
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
        def __init__(self, script):
            self._script, self.seen = script, []

        async def open(self, avoid=None):
            name, replies = self._script[len(self.seen)]
            self.seen.append(name)
            return _Chat(replies, name)

        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement=DIGITS.statement,
        entrypoint="g", deadline_s=60.0,
        public_examples=[{"args": [12345], "kwargs": {}, "expected": 15}],
    )

    # chatgpt answers correctly and is never followed up.
    backend = _TwoModels([("chatgpt", [RIGHT])])
    asyncio.run(VerifyingSolver(backend, safety_margin_s=0, max_budget_s=120)
                .solve_task(task, 60.0))
    logged = capsys.readouterr().out
    assert "provider=chatgpt" in logged, f"the log cannot say who answered: {logged!r}"

    # ...and when the first model fails and the second is asked but does WORSE,
    # the credit must stay with the answer that actually went out.
    backend = _TwoModels([("chatgpt", [WRONG, WRONG, WRONG]), ("claude", ["no code here"])])
    asyncio.run(VerifyingSolver(backend, safety_margin_s=0, max_budget_s=120)
                .solve_task(task, 60.0))
    logged = capsys.readouterr().out
    assert backend.seen == ["chatgpt", "claude"], backend.seen
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


def test_the_checklist_asks_for_the_program_first_not_a_walkthrough():
    """The order of one sentence, and it cost solves. It used to read "walk
    your solution through every one of these BEFORE YOU ANSWER" — and a model
    does what it is told, so it narrated the walkthrough at length and only
    then started the program. Written the other way round the artifact exists
    first, so a reply cut short loses the checking pass rather than the answer.
    """
    from solvers.prompts import EDGE_CASES, METHOD, build_code_prompt

    # The instruction lives in METHOD now — `<edge_cases>` is a list of input
    # shapes and says nothing about when to write anything.
    assert "Write the program FIRST" in METHOD, METHOD[:120]
    assert "silently" in METHOD
    for text in (METHOD, EDGE_CASES):
        assert "before you answer" not in text.lower(), (
            "the prompt still asks the model to narrate before answering"
        )
    for language, entry in (("rust", "main"), ("python", "solve")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        # ...inside <method>, which comes last on purpose — see
        # test_how_to_answer_comes_last_where_it_is_most_likely_to_be_obeyed.
        assert "Write the program FIRST" in prompt.split("<method>")[1]


def test_both_nudges_use_the_last_word_to_demand_code_first():
    """The nudge is appended after everything else, so it is the last thing the
    model reads before it starts generating. That slot is worth the strongest
    version of the one instruction that decides whether the answer arrives."""
    for site in (claude_site(), chatgpt_site()):
        assert site.nudge.startswith(
            "START your reply with the fenced block"
        ), site.nudge[:70]
        # The nudge holds the recency slot AND is appended to every send, so it
        # must not name a layout: the cases turn asks for one block, the
        # combined prompt asks for two, and a nudge that picks one contradicts
        # the other from the slot that wins.
        for layout in ("two ordinary fenced blocks", "program first",
                       "JSON cases second"):
            assert layout not in site.nudge, f"{site.name}: nudge pins {layout!r}"
        assert "may not arrive at all" in site.nudge


def test_the_first_attempt_gets_the_budget_when_no_repair_can_happen():
    """Reserving 40% of the budget for repair rounds is well spent when public
    examples exist and a repair is likely. With none shipped — every task on the
    run this was written for — a structurally fine first answer ends the loop,
    and the reserve is simply discarded. Measured: a tab spent its whole 135s
    slice while 90s of a 225s budget went unused, on the one attempt that had
    to succeed."""
    import inspect

    from solvers.verify import VerifyingSolver

    source = inspect.getsource(VerifyingSolver._attempt)
    assert "first_share = 0.6 if task.public_examples else 0.85" in source, source[:200]

    budget = 225.0
    with_examples = budget * 0.6
    without = budget * 0.85
    assert without > with_examples
    assert without > 135.1, (
        "the first attempt still gets less than the slice that was running out"
    )


def test_the_method_is_a_numbered_procedure_ending_in_send_the_code():
    """Ordering is what this prompt has had to fight hardest. Told to check
    "before you answer", models narrated the check and ran out of time before
    the program existed. A numbered list leaves no room to read the steps in a
    different order, and the last step is the one that must be last."""
    from solvers.prompts import METHOD, build_code_prompt

    steps = [line for line in METHOD.splitlines() if line[:2] in
             ("1.", "2.", "3.", "4.", "5.", "6.")]
    assert len(steps) == 6, steps
    assert "Write the program FIRST" in steps[1], steps[1]
    assert "silently" in METHOD
    # The last step is still the send, and it still names everything the
    # contract of that turn asked for -- both blocks on the single-turn
    # prompt, the program alone once the cases already exist.
    single = build_code_prompt("rust", "Do a thing.", "main", [])
    assert "Send the program, then the cases, and nothing else." in single
    two_turn = build_code_prompt(
        "rust", "Do a thing.", "main", [],
        cases=[{"name": "n", "args": ["1\n"], "expected": "1"}],
    )
    assert "Send the program, and nothing else." in two_turn
    assert "then the cases" not in two_turn


def test_the_examples_decide_when_the_statement_is_ambiguous():
    """The examples are the only disambiguation a solver is given — the README
    says so and nothing in the prompt used to. Without the rule the model has
    to guess which of its readings the author meant."""
    from solvers.prompts import build_code_prompt

    for language, entry in (("rust", "main"), ("python", "solve")):
        # Normalised, because the prompt is hard-wrapped: the phrase under test
        # spans a line break and an indent, and asserting on the raw text would
        # fail on formatting rather than on meaning.
        prompt = " ".join(
            build_code_prompt(language, "Do a thing.", entry, []).split()
        )
        assert "the examples decide" in prompt, "no disambiguation rule"
        assert "the code is wrong, not the example" in prompt


def test_both_contracts_say_there_is_no_partial_credit():
    """It changes the risk calculus. A model that thinks a near-miss scores
    something will reach for the clever implementation; one that knows a single
    wrong hidden case scores zero will not."""
    from solvers.prompts import build_code_prompt

    for language, entry in (("rust", "main"), ("python", "solve")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        assert "no partial credit" in prompt
        assert "prefer the safe implementation" in prompt


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


def test_the_environment_facts_are_not_filed_as_edge_cases():
    """They were, and it was a category error: `<edge_cases>` is about the
    INPUT, `<contract>` is about the machine. Silent overflow and a five-second
    limit are facts about how the grader runs the code, not shapes of data, and
    reading them in a list of "try n = 0" made both lists harder to act on."""
    from solvers.prompts import EDGE_CASES, build_code_prompt

    for banned in ("OVERFLOW", "5 seconds", "recursion", "PYTHONHASHSEED"):
        assert banned not in EDGE_CASES, f"{banned!r} is not an edge case"

    prompt = build_code_prompt("rust", "Do a thing.", "main", [])
    contract = prompt.split("<contract>")[1].split("</contract>")[0]
    assert "OVERFLOW IS SILENT" in contract
    assert "5 seconds" in contract


# --- the question, beside the answer -------------------------------------- #
# The code alone answers "what did we submit". It cannot answer "was that a
# reasonable thing to submit" — the statement, the entrypoint, the examples and
# the model's own reply all lived somewhere the archive could not see.


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
    solver._grade = lambda reply, task, left=None, cases=None: time.sleep(1.0) or Candidate(
        code="x = 1", raw=reply
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
    solver._grader.check = lambda *a, **kw: ran.append(a) or (2, 2, [])

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

        def __init__(self):
            self.opened = 0

        async def open(self, avoid=None):
            self.opened += 1
            if self.opened == 1:
                return _Chat([WRONG], provider="claude")
            chat = _Chat([RIGHT])
            del chat.provider          # reports nothing about itself
            return chat

        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Anonymous(), max_attempts=1, safety_margin_s=0, max_budget_s=120
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


def test_rust_is_left_to_its_compiler():
    """`use` has the same shape, but a Rust answer is put through rustc, which
    says so in a message a repair round can act on."""
    got = extract_code("```rust\nuse std::io;\n```\n\n"
                       "```rust\nfn main() {\n    println!(\"x\");\n}\n```", "main", "rust")
    assert got.strip().startswith("fn main()"), got


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
        return VerifyingSolver(backend, max_attempts=1, safety_margin_s=0,
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
    assert backend.chats and "longest run" in backend.chats[0].asked[0]
    assert "<output>" in backend.chats[0].asked[0], "not the miner's own prompt"


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
    assert "COULD NOT BE CHECKED: no browser" in out, out
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
    assert len(backend.chats) == 5, "one conversation per challenge"
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
            return _Chat([RIGHT], provider=f"m{len(asked)}")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, safety_margin_s=0,
                             max_budget_s=60, second_opinion=True)

    def unavailable(*a, **kw):
        raise RuntimeError("DockerExecutor requires the 'docker' CLI on PATH")

    solver._grader.check = unavailable
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    out = capsys.readouterr().out
    assert answer.code.strip(), "the answer itself was lost"
    assert len(asked) == 1, f"asked {len(asked)} models for an ungradeable task"
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
            return _Chat([RIGHT] if len(asked) > 1 else ["I cannot help."],
                         provider=f"m{len(asked)}")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, safety_margin_s=0,
                             max_budget_s=60, second_opinion=True)
    solver._grader.check = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("no docker")
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    assert len(asked) == 2, "an empty answer must still get a second chance"
    assert "def g(n)" in answer.code, answer.code


def test_a_gradeable_failure_still_buys_a_second_opinion():
    """Only 'nothing ran' skips it. An answer that ran and failed some examples
    IS rankable, so the other model is worth asking."""
    asked: list[str] = []

    class _Backend:
        async def open(self, avoid=None):
            asked.append(avoid or "first")
            return _Chat([WRONG], provider=f"m{len(asked)}")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(_Backend(), max_attempts=1, safety_margin_s=0,
                             max_budget_s=60, second_opinion=True)
    asyncio.run(solver.solve_task(DIGITS, timeout_s=60))
    assert len(asked) == 2, "a rankable failure should still ask the other model"


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

    passed, total, failures = _Grader().check(
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
def test_the_prompt_tells_the_truth_about_how_little_speed_is_worth():
    """The prompt used to say the answer is "scored partly on how fast it
    arrives" — technically true, catastrophically miscalibrated. The payment
    rule gates everything on all-pass correctness and caps the speed spread at
    the floor: fastest correct earns at most (1 - floor) more than slowest
    correct, 5% under this release, with a 180s half-life. A model told speed
    matters trades checks for pace, which is exactly the trade the rule
    punishes. The claim in the prompt is DERIVED here from the payment module's
    own constants, so a policy change fails this test rather than leaving the
    prompt asserting something that stopped being true."""
    from rlvr.policy import RELEASE_POLICY
    from rlvr.scoring.payment import DEFAULT_SPEED_FLOOR
    from solvers.prompts import PYTHON_RULES, RUST_RULES, build_code_prompt

    # Production reads ValidatorPolicy.payment_speed_floor, a constant
    # DUPLICATED from payment.py's default. The prompt's claim is derived from
    # the production one; this assertion is what keeps the duplicate honest,
    # because editing policy.py alone would otherwise leave the prompt
    # test-green and wrong on chain.
    assert RELEASE_POLICY.payment_speed_floor == DEFAULT_SPEED_FLOOR
    floor = f"{round(RELEASE_POLICY.payment_speed_floor * 100)}%"
    for rules in (PYTHON_RULES, RUST_RULES):
        assert f"at least {floor} of what the fastest" in rules, (
            f"the prompt's speed floor does not match the payment policy "
            f"({floor} from RELEASE_POLICY.payment_speed_floor)"
        )
        assert "how fast it arrives" not in rules, (
            "the prompt still tells the model that speed is scored"
        )
    for language, entry in (("python", "solve"), ("rust", "main")):
        prompt = build_code_prompt(language, "Do a thing.", entry, [])
        assert "no partial credit" in prompt
        assert "prefer the safe implementation" in prompt


def test_the_method_licenses_depth_and_forbids_trading_checks_for_speed():
    """The other half of the same correction. Telling the model speed is
    nearly worthless only helps if the method then says what to spend the
    time ON: real traces with computed values, not a skim that ends "looks
    right", and a closing line that makes skipping a check for speed a named
    mistake rather than a judgement call."""
    from solvers.prompts import METHOD, METHOD_CODA, build_code_prompt

    # The license is an ALLOCATION rule, not a claim that reasoning is free —
    # on a UI with little or no hidden reasoning channel, "invisible and free"
    # invites the model to reason visibly in the reply.
    assert "spend the deadline where it buys correctness" in " ".join(METHOD.split())
    assert "invisible and free" not in METHOD
    # Silence is declared ABOVE the numbered steps, so every step is read
    # under it rather than discovering it forty lines later.
    steps_start = METHOD.index("1.")
    assert "silently" in METHOD[:steps_start]
    assert "computing the real" in METHOD and "intermediate values" in METHOD
    assert "in your head" in METHOD, "the trace is not anchored to reasoning"
    assert '"looks right" catches nothing' in " ".join(METHOD.split())
    # The verbs that could be obeyed as artifacts are gone: no building tests
    # (test code in the block), no listing rules (a visible bulleted list).
    assert "Build tests" not in METHOD and "List for yourself" not in METHOD
    assert "Invent inputs of your own" in METHOD
    # The strongest sentence fires once, from the last slot in <method>.
    assert "The check you skip is the hidden test you fail" in " ".join(METHOD_CODA.split())
    assert "the deadline cuts off scores the same zero" in " ".join(METHOD_CODA.split())
    prompt = build_code_prompt("python", "Do a thing.", "solve", [])
    tail = prompt.split("</self_check>")[1]
    assert "The check you skip" in tail, "the coda is not in the final slot"
    # The protective invariants that earlier cost solves must survive the
    # rewrite: program before checking, silence, code-only close.
    steps = [line for line in METHOD.splitlines()
             if line[:2] in ("1.", "2.", "3.", "4.", "5.", "6.")]
    assert len(steps) == 6
    assert "Write the program FIRST" in steps[1]
    assert steps[5].startswith("6. Send ") and steps[5].endswith("nothing else.")
    assert "Send the program, then the cases, and nothing else." in prompt
    assert "before you answer" not in METHOD.lower()


def test_the_model_is_told_to_build_its_own_tests_from_the_statement():
    """The generic checklist catches generic bugs. Both graded failures on the
    real challenges were problem-specific boundaries: extent-journal's resize
    rule (a statement sentence with no code behind it) and the thresholds the
    statement itself named. The method now makes the model derive its battery
    from the statement — every named bound at, below and above; counts at 0,
    1, 2; every rule triggered and NEARLY triggered — and re-trace after
    every fix."""
    from solvers.prompts import EDGE_CASES, METHOD, SELF_CHECK

    normalised_method = " ".join(METHOD.split())
    assert "one input that triggers it and one that NEARLY does" in normalised_method
    assert "re-trace the cases that already passed" in normalised_method
    assert "THE STATEMENT'S OWN CONSTANTS" in EDGE_CASES
    assert "AT that exact value, one below it, and one above" in EDGE_CASES
    for case in ("TIES", "A REJECTED OPERATION", "TWO RULES AT ONCE",
                 "WRAPAROUND", "A BATCH APPLIED AT ONCE"):
        assert case in EDGE_CASES, f"the checklist dropped {case}"
    assert "nothing half-committed" in " ".join(EDGE_CASES.split())
    # The seventh self-check line: a statement sentence with no code behind it.
    assert "code you can point at" in SELF_CHECK
    assert "verify each of these, silently" in " ".join(SELF_CHECK.split()), (
        "the self-check header invites a visible Q&A again"
    )


def test_a_repair_asks_for_a_diagnosis_not_a_guess():
    """A repair that edits from the shape of the failure fixes the symptom the
    failure happened to show. The prompt now demands the model find the actual
    line where computed and expected part company before touching anything."""
    from solvers.prompts import build_repair_prompt

    prompt = build_repair_prompt(["g(*[0], **{}) returned 1, expected 0"],
                                 "python", "g")
    assert "In your reasoning — not in the reply" in prompt
    assert "do not guess at the fix" in prompt
    assert "before you send" not in prompt.lower(), (
        "the repair prompt reintroduced the phrase that caused narration"
    )
    assert prompt.rstrip().endswith("ONLY ONE corrected code block and nothing else."), (
        "the repair prompt no longer ends on the output rule"
    )


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
def test_a_tab_that_never_renders_a_reply_is_dropped_in_seconds_not_at_the_deadline():
    """Measured twice in one live run, on two different sites:

        chatgpt tab 9227: no assistant selector matched anything
        claude  tab 9222: matched 1 message(s), the same as before the prompt
                          was sent -- the answer never rendered

    Both polled a tab that could not be read for the whole 191s first slice,
    then spent a 29s repair round on the same dead conversation, and ended the
    task with 5s left -- fewer than the 20s a second opinion needs. Five healthy
    tabs sat idle through both. The task scored zero.

    The distinction that makes this fixable is `on_screen`: a model that is
    merely slow has ALREADY painted its bubble, so it is never mistaken for
    this. The tab here paints nothing, ever.
    """
    playwright, chrome = _chromium_or_skip()
    # A composer and a send button, and a send that renders nothing at all.
    url = _served(
        '<!doctype html><meta charset="utf-8">'
        '<div id="composer" contenteditable="true"></div><button id="send">go</button>'
    )
    site = _site(assistant=("#assistant",))  # matches nothing, now and forever

    async def go(grace):
        async with playwright.async_playwright() as p:
            browser = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = await (await browser.new_context()).new_page()
            await page.goto(url)
            tab = _tab(page, site)
            real, _browser_pool.BLIND_TAB_GRACE_S = _browser_pool.BLIND_TAB_GRACE_S, grace
            try:
                started = time.monotonic()
                reply = await tab.send("solve it", 120.0)
                elapsed = time.monotonic() - started
            finally:
                _browser_pool.BLIND_TAB_GRACE_S = real
                await browser.close()
            return reply, elapsed, tab.alive

    reply, elapsed, alive = asyncio.run(go(2.0))
    assert reply == "", f"there was nothing on the page to capture: {reply!r}"
    assert elapsed < 30.0, (
        f"the send spent {elapsed:.1f}s of a 120s budget on a tab that showed "
        f"nothing. That time is the whole point: it belongs to another tab."
    )
    assert alive is False, "an unreadable tab must be retired, not handed back"


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
        async def send(self, text, timeout_s):
            sends.append(text)
            return ""

    class _Fleet:
        async def open(self, avoid=None):
            return _Silent([""], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=3, safety_margin_s=0, max_budget_s=120,
        second_opinion=False,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code == "", "nothing was captured; nothing is the honest answer"
    assert len(sends) == 1, (
        f"sent {len(sends)} prompts into a conversation that returned nothing. "
        f"Each one is a full slice of the budget spent to be told the same thing."
    )


def test_a_reply_that_arrived_without_code_is_still_repaired_in_place():
    """The other side of the same guard, and the reason it is not simply
    "empty means give up".

    A FINISHED reply with no code block in it is the model breaking the output
    contract, not a conversation that cannot be used -- and telling it so is
    exactly what fixes it. `test_a_reply_with_no_code_is_rejected_and_retried`
    pins the recovery; this pins that the fail-fast did not swallow it.
    """
    sends: list[str] = []

    class _Prose(_Chat):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.empty_reason = "no-code"   # rendered, settled, simply no code
        async def send(self, text, timeout_s):
            sends.append(text)
            return "Sure! Here is the approach..." if len(sends) == 1 else RIGHT

    class _Fleet:
        async def open(self, avoid=None): return _Prose([""], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=3, safety_margin_s=0, max_budget_s=120,
        second_opinion=False,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified, "the repair round recovers a reply that forgot the fence"
    assert len(sends) == 2, f"expected one repair in place, sent {len(sends)}"


def test_an_empty_answer_keeps_asking_while_the_clock_allows_it():
    """Two passes was a policy for 'the first model was WRONG'. Empty is a
    different state and has a different price: a wrong answer pays zero and so
    does no answer, but an extra ask can only turn the second into a payment.
    So while nothing is in hand, keep asking until the clock says stop.
    """
    seen: list[str] = []

    class _Fleet:
        def __init__(self, replies): self._replies = replies
        async def open(self, avoid=None):
            provider = "chatgpt" if avoid == "claude" else "claude"
            seen.append(provider)
            return _Chat(self._replies[min(len(seen) - 1, len(self._replies) - 1)], provider)
        async def aclose(self): pass
        def stats(self): return {}

    # Three tabs in a row capture nothing; the fourth answers.
    solver = VerifyingSolver(
        _Fleet([["nope"], ["nope"], ["nope"], [RIGHT]]),
        max_attempts=1, safety_margin_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.verified, (
        f"gave up after {len(seen)} model(s) with time still on the clock and "
        f"submitted nothing. The answer was one ask away."
    )
    assert seen == ["claude", "chatgpt", "claude", "chatgpt"], seen


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
        _Fleet(), max_attempts=1, safety_margin_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code == ""
    assert len(seen) == MAX_PASSES, f"asked {len(seen)} times, cap is {MAX_PASSES}"


def test_holding_nothing_lowers_the_bar_for_one_more_ask():
    """The two floors encode what each state has to lose. Holding an unverified
    answer, spending the tail of the budget on another model risks the time.
    Holding nothing, there is nothing to risk -- what is in hand pays zero."""
    assert EMPTY_HANDED_FLOOR_S <= SECOND_OPINION_FLOOR_S, (
        f"empty-handed needs {EMPTY_HANDED_FLOOR_S}s to justify one more ask "
        f"but holding an answer needs only {SECOND_OPINION_FLOOR_S}s. That is "
        f"backwards: the state with nothing to lose is being made the harder "
        f"one to act on."
    )
    assert EMPTY_HANDED_FLOOR_S > 0, "an ask still has to be able to happen"


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
    # slices[0] is the cases turn, and it reads against the SOLVE's clock like
    # everything else -- no private cap. What it actually spends is decided by
    # when the model finishes, not by an allocation.
    assert slices[0] > 230.0, (
        f"the cases turn was given {slices[0]:.0f}s of a 300s deadline; it is "
        f"supposed to read against the whole budget"
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
        _Fleet(), max_attempts=1, safety_margin_s=0, max_budget_s=120,
    )
    answer = asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert answer.code, "the wrong answer is still submitted; it may pass the hidden suite"
    assert answer.verified is False
    assert len(seen) == SECOND_OPINION_PASSES, (
        f"asked {len(seen)} models about an answer that was already in hand; "
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
        f"the DOM was unreadable and the answer was on the wire, and the tab was "
        f"retired before anything looked: {reply!r}"
    )
    assert alive is False, "the DOM is still unreadable; the tab is finished"
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
            sends.append(text)
            return captured

    class _Fleet:
        async def open(self, avoid=None): return _StillWriting([""], "claude")
        async def aclose(self): pass
        def stats(self): return {}

    solver = VerifyingSolver(
        _Fleet(), max_attempts=3, safety_margin_s=0, max_budget_s=120,
        second_opinion=False,
    )
    asyncio.run(solver.solve_task(DIGITS, timeout_s=120))
    assert len(sends) == 1, (
        f"captured {what} from a model that had not finished, and sent "
        f"{len(sends) - 1} more prompt(s) into the conversation it was still "
        f"writing in:\n{sends[1:2]}"
    )


def test_a_backend_that_cannot_wait_is_never_asked_to():
    """`extend_to_s` is optional on the `Conversation` protocol.

    A backend written outside this package satisfies the protocol with the
    two-argument `send` that has always been the contract, and calling it with
    a keyword it does not take would raise TypeError inside the single call the
    whole solve depends on. Catching that TypeError is not an option either: one
    raised from INSIDE `send` is indistinguishable, and retrying would send the
    prompt twice.
    """
    from solvers.verify import _reads_past_its_slice

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

    assert _reads_past_its_slice(_OldStyle()) is False
    assert _reads_past_its_slice(_NewStyle()) is True

    for backend, arity in ((_OldStyle, 1), (_NewStyle, 2)):
        seen.clear()

        class _Fleet:
            async def open(self, avoid=None): return backend()
            async def aclose(self): pass
            def stats(self): return {}

        answer = asyncio.run(
            VerifyingSolver(_Fleet(), safety_margin_s=0, max_budget_s=120)
            .solve_task(DIGITS, timeout_s=120)
        )
        assert answer.verified, f"{backend.__name__} stopped producing answers"
        assert len(seen[0]) == arity, (
            f"{backend.__name__}.send was called with {len(seen[0])} budget "
            f"argument(s); it takes {arity}"
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
    cut = _Tab._copy_was_cut_short

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
    budget = min(deadline, solver._max_budget) - solver._margin
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
        POSTMORTEM_TIMEOUT_S, tail_budget,
    )

    assert FULL_TAIL_S == COPY_PHASE_TIMEOUT_S + STREAM_PHASE_TIMEOUT_S + POSTMORTEM_TIMEOUT_S
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
# The model's own cases, run with the validator's executor.
#
# Live traffic ships no `public_examples`: 56 solves in a row reported
# `examples=0/0`. The repair loop -- the one mechanism here that turns a nearly
# right answer into a right one -- therefore never had anything to run.
# --------------------------------------------------------------------------- #
_WRONG_WITH_CASES = '''```python
def g(n):
    s = 0
    while n > 9:
        s += n % 10
        n //= 10
    return s
```

```json
[{"name": "zero", "args": [0], "expected": 0},
 {"name": "single digit", "args": [7], "expected": 7},
 {"name": "carries", "args": [12345], "expected": 15}]
```'''

_RIGHT_WITH_CASES = _WRONG_WITH_CASES.replace("while n > 9", "while n > 0")

# The two-turn shape: cases alone, then a program alone.
_CASES_ONLY = _WRONG_WITH_CASES.split("```json", 1)[1].join(("```json", ""))
_WRONG_PROGRAM = _WRONG_WITH_CASES.split("\n\n```json", 1)[0]
_RIGHT_PROGRAM = _WRONG_PROGRAM.replace("while n > 9", "while n > 0")

_NO_EXAMPLES = SolveTask(
    problem_id="live", language="python", statement="Sum the digits of n.",
    entrypoint="g", public_examples=[], deadline_s=300.0,
)


def _solver_seeing(replies, **kw):
    sent: list[str] = []

    class _Chat:
        provider = "claude"
        def __init__(self): self.n = -1
        async def send(self, text, timeout_s, extend_to_s=None):
            sent.append(text)
            self.n += 1
            return replies[min(self.n, len(replies) - 1)]
        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    return VerifyingSolver(_Fleet(), **kw), sent


def test_the_models_own_boundary_case_catches_a_wrong_program(capsys):
    """The whole point, end to end, on a task shaped exactly like live traffic:
    no public examples at all.

    The program has the classic `while n > 9` bug — right for 12345, wrong for
    0 and for any single digit. The model's own boundary case catches it, the
    repair round fires, and the corrected program is what gets submitted.
    Before this existed there was nothing to run and the bug shipped.
    """
    solver, sent = _solver_seeing([_CASES_ONLY, _WRONG_PROGRAM, _RIGHT_PROGRAM])
    answer = asyncio.run(solver.solve_task(_NO_EXAMPLES, timeout_s=300.0))

    assert "while n > 0" in answer.code, f"submitted the buggy program: {answer.code!r}"
    assert len(sent) == 3, f"expected cases, program, repair; sent {len(sent)}"
    assert "<task>" in sent[0], "the first turn must ask for the cases alone"
    assert "<must_pass" in sent[1], "the program turn must restate the cases"
    assert "DISAGREE" in sent[2], sent[2][:200]
    assert "'single digit'" in sent[2], (
        f"the repair has to name the case that broke: {sent[2][:300]}"
    )
    assert "self=3/3" in capsys.readouterr().out


def test_self_tests_never_reach_the_validator():
    """The output contract was ONE block for good reasons, and the JSON block
    relaxes it. `extract_code` picks the block that DEFINES the entrypoint, so
    the cases cannot be submitted as a program — checked here against the
    layouts a model actually produces, including the one where it forgets the
    program entirely."""
    layouts = {
        "program then tests": _WRONG_WITH_CASES,
        "tests then program": "\n\n".join(reversed(_WRONG_WITH_CASES.split("\n\n```json"))),
        "untagged json": _WRONG_WITH_CASES.replace("```json", "```"),
    }
    for label, reply in layouts.items():
        code = extract_code(reply, "g", "python")
        assert '"expected"' not in code, f"{label}: the cases leaked into the answer"
        assert "def g" in code, f"{label}: lost the program"

    # Only tests, no program: the honest outcome is "no code", not "here is JSON".
    only = extract_code('```json\n[{"args": [0], "expected": 0}]\n```', "g", "python")
    assert only == "", f"submitted the cases as a program: {only!r}"
    assert python_defect(only, "g") == NO_CODE


def test_the_model_grading_itself_is_never_recorded_as_verification():
    """`verified` gates the answer cache. A model agreeing with itself is not
    proof of anything, so self-tests must never set it — otherwise one wrong
    answer that matched its own wrong cases is cached and re-served for every
    later task with the same statement."""
    solver, _ = _solver_seeing([_RIGHT_WITH_CASES])
    answer = asyncio.run(solver.solve_task(_NO_EXAMPLES, timeout_s=300.0))
    assert answer.code, "the answer still comes back"
    assert answer.verified is False, "self-grading claimed to be verification"
    assert answer.total == 0, "self-tests must not masquerade as public examples"
    assert solver._cache == {}, "an unverified answer was cached"


def test_a_disagreement_is_not_reported_as_the_program_being_wrong():
    """These cases came from the model, so a disagreement proves only that two
    things it wrote contradict each other. Telling it the CODE is at fault when
    the CASE was wrong is how a repair round breaks a correct program."""
    failures = ["g(*[0], **{}) returned 1, expected 0"]
    mine = build_repair_prompt(failures, "python", "g", from_self_tests=True)
    theirs = build_repair_prompt(failures, "python", "g")

    assert "DISAGREE" in mine and "WRONG" not in mine.split("\n")[0]
    assert "Exactly one of the two is wrong" in mine
    assert "leave the program alone" in mine
    # The validator's own examples are ground truth and keep the blunt wording.
    assert "Your solution is WRONG" in theirs
    assert "DISAGREE" not in theirs


def test_self_tests_can_be_turned_off_completely():
    """`SOLVER_SELF_TESTS=0` has to restore exactly the behaviour that preceded
    them — a new mechanism on the path that decides what gets submitted needs a
    way back."""
    solver, sent = _solver_seeing([_WRONG_WITH_CASES], self_tests=False)
    answer = asyncio.run(solver.solve_task(_NO_EXAMPLES, timeout_s=300.0))
    assert len(sent) == 1, "graded anyway with self-tests switched off"
    assert "while n > 9" in answer.code, "the buggy program is submitted, as before"


def test_the_case_parser_survives_whatever_a_model_writes():
    """A model wrote both halves of this — the cases and the JSON they arrived
    in — so every failure mode must end at "no self-tests ran", which is exactly
    where this code path started. It is parsed on the path that decides what
    gets submitted; an exception here loses the answer."""
    from solvers.prompts import MAX_SELF_TESTS, extract_self_tests

    def cases(block, ep="g", lang="python"):
        return extract_self_tests(f"```python\ndef g(n):\n    return n\n```\n\n{block}", ep, lang)

    assert cases('```json\n[{"args": [1], "expected": 2}]\n```') == [
        {"args": [1], "kwargs": {}, "expected": 2, "name": ""}
    ]
    # Junk of every shape degrades to nothing, never to an exception.
    for junk in (
        "```json\nnot json at all\n```",
        "```json\n{}\n```",                       # an object, not a list
        "```json\n[1, 2, 3]\n```",                # not case objects
        '```json\n[{"args": [1]}]\n```',          # no `expected`
        "```json\n[]\n```",
        "```\n\n```",
    ):
        assert cases(junk) == [], junk
    assert extract_self_tests("", "g") == []
    assert extract_self_tests("```json\n[]\n```", "") == []

    # A bare value for `args` is wrapped rather than dropped.
    assert cases('```json\n[{"args": 5, "expected": 5}]\n```')[0]["args"] == [5]
    # Bad `kwargs` is replaced, not trusted.
    assert cases('```json\n[{"args": [], "kwargs": 3, "expected": 0}]\n```')[0]["kwargs"] == {}

    # Capped: each case is an executor run against the solve's own budget.
    many = ", ".join('{"args": [1], "expected": 1}' for _ in range(40))
    assert len(cases(f"```json\n[{many}]\n```")) == MAX_SELF_TESTS


def test_rust_cases_that_cannot_run_are_discarded_rather_than_run():
    """The Rust judge feeds `args[0]` to stdin and compares stdout, so a case
    shaped for a function call cannot run at all. Keeping it would produce a
    failure report about the miner's own parsing rather than the program, and
    send a repair round chasing it."""
    from solvers.prompts import extract_self_tests

    def cases(block):
        return extract_self_tests(
            f'```rust\nfn main() {{}}\n```\n\n{block}', "main", "rust"
        )

    ok = cases('```json\n[{"name": "n", "args": ["3\\n"], "expected": "6"}]\n```')
    assert ok == [{"args": ["3\n"], "kwargs": {}, "expected": "6", "name": "n"}]

    for unusable in (
        '```json\n[{"args": [3], "expected": "6"}]\n```',        # int stdin
        '```json\n[{"args": ["3"], "expected": 6}]\n```',        # int stdout
        '```json\n[{"args": ["a", "b"], "expected": "6"}]\n```', # two streams
        '```json\n[{"args": [], "expected": "6"}]\n```',         # no stdin
    ):
        assert cases(unusable) == [], unusable


def test_a_candidate_that_passes_its_own_cases_outranks_one_that_does_not():
    """`score` decides which round's answer is submitted. Self-test passes rank
    below the validator's own examples, which are ground truth, and above
    anything structural — but a candidate with NO self-tests ties with one that
    failed them all, deliberately: having tests must not rank an answer below
    not having them."""
    from solvers.verify import Candidate

    def c(**kw):
        return Candidate(code="def g(n): return n", raw="", **kw)

    passing = c(self_passed=3, self_total=3)
    failing = c(self_passed=0, self_total=3)
    untested = c()
    assert passing.score > failing.score
    assert failing.score == untested.score, "penalised a candidate for having tests"
    # Ground truth still wins outright.
    assert c(passed=1, total=1).score > passing.score
    # And none of it is verification.
    assert passing.verified is False


def test_the_program_is_never_mistaken_for_the_cases():
    """`_parse_cases` is the only thing separating the two blocks, and it has to
    be enough: no Python or Rust program starts with `[`.

    An is-this-the-program check was tried and removed. It had no reachable
    upside and one real downside — `_defines` for Rust is a text search for
    `fn main`, so a task about GENERATING Rust would have its cases discarded
    for quoting the phrase in an expected value, which is this case.
    """
    from solvers.prompts import extract_self_tests

    reply = (
        '```rust\nfn main() { println!("x"); }\n```\n\n'
        '```json\n[{"name": "emits a program", "args": ["1\\n"], '
        '"expected": "fn main() {}"}]\n```'
    )
    cases = extract_self_tests(reply, "main", "rust")
    assert len(cases) == 1, (
        f"discarded a valid case for quoting `fn main` in its expected output: "
        f"{cases}"
    )
    assert cases[0]["expected"] == "fn main() {}"
    assert extract_code(reply, "main", "rust") == 'fn main() { println!("x"); }'


def test_the_roster_wires_the_self_test_switch_through(monkeypatch):
    """Testing the constructor is not testing the wiring: `SOLVER_SELF_TESTS=0`
    has to reach the solver an operator actually gets."""
    from solvers.roster import build_solver

    monkeypatch.delenv("SOLVER_SELF_TESTS", raising=False)
    assert build_solver([])._self_tests is True, "self-tests are on by default"

    for off in ("0", "false", "NO", "False"):
        monkeypatch.setenv("SOLVER_SELF_TESTS", off)
        assert build_solver([])._self_tests is False, off

    monkeypatch.setenv("SOLVER_SELF_TESTS", "true")
    assert build_solver([])._self_tests is True


# --------------------------------------------------------------------------- #
# Cases first, then the program.
# --------------------------------------------------------------------------- #
def _two_turn(deadline, replies, **kw):
    """Drive a solve and return (prompts, slices, caps, answer)."""
    seen: list[tuple] = []

    class _Chat:
        provider = "claude"
        def __init__(self): self.n = -1
        async def send(self, text, timeout_s, extend_to_s=None):
            self.n += 1
            seen.append((text, timeout_s, extend_to_s))
            return replies[min(self.n, len(replies) - 1)]
        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement="Sum the digits of n.",
        entrypoint="g", public_examples=[], deadline_s=deadline,
    )
    answer = asyncio.run(
        VerifyingSolver(_Fleet(), **kw).solve_task(task, timeout_s=deadline)
    )
    return [t for t, _, _ in seen], [s for _, s, _ in seen], [c for _, _, c in seen], answer


def test_the_cases_are_asked_for_before_the_program_exists():
    """The whole argument for spending a round trip. Cases written ALONGSIDE a
    program can be back-filled from what the program happens to do, and then
    they agree with its bugs. Cases written first cannot — and the prompt says
    so outright, because a model that does not know why will drift back."""
    prompts, _, _, answer = _two_turn(
        300.0, [_CASES_ONLY, _WRONG_PROGRAM, _RIGHT_PROGRAM]
    )
    assert len(prompts) == 3, [p[:60] for p in prompts]
    assert "<task>" in prompts[0] and "<must_pass" not in prompts[0]
    assert "you have not written the program yet" in prompts[0].lower()
    # Said twice, in the contract and in the task, because it is the one
    # instruction that makes turn 1 different from turn 2 -- and a cases turn
    # that also writes the program is just the single-turn prompt with an
    # extra round trip billed to the deadline.
    contract = prompts[0].split("</output>")[0]
    assert "Do NOT write the program yet" in contract, contract
    assert "you will be asked for it next" in contract
    assert "COMPLETE BEFORE CORRECT" not in prompts[0]
    for phrase in ("Write the program FIRST", "Send the program"):
        assert phrase not in prompts[0], f"the cases turn asks for a program: {phrase!r}"
    assert "<must_pass" in prompts[1], "turn 2 must restate the cases as the bar"
    assert "while n > 0" in answer.code


def test_the_cases_turn_does_not_shrink_the_read_the_program_gets():
    """What actually protects the program, now that turn 1 has no private cap.

    A cap looks like the protection and is not: `send` returns the moment the
    model finishes, so the slice is a ceiling and never a wait. A cases turn
    that takes 90 seconds hands the program the other 190 either way. The only
    case a cap changes is the one where the model has NOT finished, and there it
    converts a slow answer into no answer.

    So the guarantee is arithmetic, not allocation: turn 2 opens on what the
    clock says is left, and the elapsed cases turn is the only thing that took
    any of it."""
    _, slices, caps, _ = _two_turn(300.0, [_CASES_ONLY, _RIGHT_PROGRAM])
    assert slices[0] > 230.0, (
        f"the cases turn was allocated {slices[0]:.0f}s of a 300s deadline — it "
        f"is supposed to read against the whole budget, not a share of it"
    )
    assert caps[0] is None or caps[0] <= slices[0] + 0.01, (
        "nothing is held back from the cases turn, so there is nothing to "
        "extend into"
    )
    # A fast cases turn costs the program almost nothing: these fakes reply
    # instantly, so turn 2 still opens on essentially the whole budget.
    assert slices[1] > 200.0, f"the program only got {slices[1]:.0f}s"
    assert caps[1] > slices[1], "the program's read must still be extendable"


def test_every_deadline_asks_for_the_cases_first():
    """There is no budget below which the split is skipped, and there was.

    The floor existed because a cases turn was assumed to COST a fixed 20-30s of
    submit-and-settle before the model writes anything, which a 40s budget
    cannot spare. But turn 1 is a ceiling, not a spend: `send` returns the
    moment the model finishes, so a fast cases turn on a short deadline costs
    what it took and the program gets the rest. A floor priced the worst case
    into every solve."""
    for deadline in (30.0, 60.0, 300.0):
        prompts, slices, _, answer = _two_turn(
            deadline, [_CASES_ONLY, _RIGHT_PROGRAM]
        )
        assert "<task>" in prompts[0], (
            f"a {deadline:.0f}s deadline skipped the cases turn"
        )
        assert "<must_pass" in prompts[1], "turn 2 must restate the cases"
        assert answer.code, f"no program came back at a {deadline:.0f}s deadline"
        # And turn 1 is allocated the whole budget at every size — the ceiling
        # never scales down, only what the model actually spends does.
        assert slices[0] > slices[1], (
            f"turn 1 got {slices[0]:.0f}s and turn 2 {slices[1]:.0f}s at a "
            f"{deadline:.0f}s deadline; turn 1 is allocated everything left"
        )


def _burning_backend(deadline, cases_burns_s):
    """A backend on a REAL clock: turn 1 sleeps `cases_burns_s` and fails.

    The clock has to be real. A fake that fails instantly leaves the budget
    intact, so it cannot tell "the turn failed" from "the turn failed having
    spent everything" — which is the whole distinction under test.
    """
    log: list[str] = []

    class _Chat:
        provider = "claude"
        empty_reason = None
        still_writing = False

        async def send(self, text, timeout_s, extend_to_s=None):
            turn = "CASES" if "<task>" in text else "CODE"
            log.append(turn)
            if turn == "CASES":
                await asyncio.sleep(min(timeout_s, cases_burns_s))
                self.empty_reason, self.still_writing = "unfinished", True
                return ""
            self.empty_reason, self.still_writing = None, False
            return "```python\ndef g(n):\n    return n\n```"

        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement="hard",
        entrypoint="g", public_examples=[], deadline_s=deadline,
    )
    answer = asyncio.run(
        VerifyingSolver(_Fleet()).solve_task(task, timeout_s=deadline)
    )
    return log, answer


def test_a_phase_that_runs_the_deadline_out_stops_instead_of_handing_on():
    """The two ways turn 1 can fail are not the same failure, and the clock is
    what tells them apart.

    A tab that DIES leaves the budget intact, and another tab can still spend
    it — that is worth doing. A turn that ran the DEADLINE out leaves nothing:
    handing on would lease a second tab, ask a second account for a whole
    program with seconds on the clock, and arrive at the same empty answer
    having spent someone's quota to get there.

    Real sleeps, because a fake that fails instantly cannot tell the two
    apart — the budget would be intact in both.

    Nothing new enforces this: `solve_task` has always refused to open a pass
    with less than `EMPTY_HANDED_FLOOR_S` left. What changed is that the rule
    now BINDS, because turn 1 reads against the real budget — while it was
    capped at 60s the budget survived it and four tabs were spent in a row.
    This test is what stops that floor being loosened back out from under it."""
    # The floor has to be a real reserve. At zero the guarantee below still
    # happens to hold, but only because `remaining` lands a hair NEGATIVE —
    # which is luck, not a rule, and a lease that arrives a millisecond earlier
    # would spend a tab on a solve with no time to use it.
    assert EMPTY_HANDED_FLOOR_S >= 5.0, (
        f"EMPTY_HANDED_FLOOR_S is {EMPTY_HANDED_FLOOR_S}; it is the only thing "
        f"stopping a spent deadline being handed to another tab"
    )

    # 40s deadline -> 20s budget. Turn 1 burns all of it.
    log, answer = _burning_backend(40.0, 999.0)
    assert log == ["CASES"], (
        f"the deadline was already gone and it still asked another tab: {log}"
    )
    assert not answer.code

    # Same failure, 3s in, with the budget almost untouched: hand on, and the
    # program still gets asked for.
    log, answer = _burning_backend(40.0, 3.0)
    assert log.count("CASES") == 1, f"asked for cases twice: {log}"
    assert "CODE" in log, f"gave up while the budget was still there: {log}"
    assert answer.code, "submitted nothing despite having the time to answer"


def test_a_cases_turn_that_cannot_be_read_hands_the_task_on():
    """Sending the program request into a conversation that is unreadable, or
    still writing, queues it behind an answer that has not arrived. The tab has
    already proved it cannot answer; another model is the better use of what is
    left."""
    for reason in ("unreadable", "unfinished"):
        seen: list[str] = []

        class _Chat:
            provider = "claude"
            still_writing = False
            def __init__(self): self.empty_reason = reason
            async def send(self, text, timeout_s, extend_to_s=None):
                seen.append(text)
                return ""
            async def close(self): pass

        class _Fleet:
            async def open(self, avoid=None): return _Chat()
            async def aclose(self): pass
            def stats(self): return {}

        task = SolveTask(
            problem_id="p", language="python", statement="s", entrypoint="g",
            public_examples=[], deadline_s=300.0,
        )
        chatter = io.StringIO()
        with contextlib.redirect_stdout(chatter):
            asyncio.run(VerifyingSolver(
                _Fleet(), second_opinion=False
            ).solve_task(task, timeout_s=300.0))
        log = chatter.getvalue()
        assert len(seen) == 1, (
            f"{reason}: sent the program request into a tab that had already "
            f"failed the cases turn"
        )
        # And it stopped DELIBERATELY. Removing the guard once left this
        # assertion green anyway, because `list(None)` raised a TypeError that
        # this module catches as a backend failure -- one send, for entirely
        # the wrong reason. A hand-off is a decision and says so; a crash is
        # not.
        assert f"the cases turn came back {reason}" in log, log
        assert "backend failure" not in log, log


def test_a_repair_may_correct_a_case_the_first_turn_got_wrong():
    """The escape hatch, and without it the split makes the miner WORSE.

    Turn 1 derives its `expected` values from the statement by reasoning, so
    one of them can simply be wrong. Freeze those cases and a CORRECT program
    fails the same bogus case on every repair round, burns all three attempts,
    and is submitted with `verified=False` — the exact failure the two-turn
    split was supposed to remove. So the repair prompt says outright to fix the
    case and leave the program alone, and a repair reply carrying a corrected
    array replaces the frozen one for the next round.

    The correction lands on the NEXT round, not the one that carried it: a
    reply is graded against the cases that were agreed BEFORE it arrived, so a
    model cannot make its program pass by rewriting the bar in the same breath.
    """
    right = "```python\ndef g(n):\n    s = 0\n    while n > 0:\n        s += n % 10\n        n //= 10\n    return s\n```"
    # "zero" is wrong -- the digits of 0 sum to 0, not 99. A correct program
    # cannot pass it, and no rewrite of the program can make it pass.
    bad = ('```json\n[{"name": "zero", "args": [0], "expected": 99},\n'
           ' {"name": "carries", "args": [12345], "expected": 15}]\n```')
    fixed = ('```json\n[{"name": "zero", "args": [0], "expected": 0},\n'
             ' {"name": "carries", "args": [12345], "expected": 15}]\n```')

    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        prompts, _, _, answer = _two_turn(
            300.0,
            [bad,                     # turn 1: cases, one of them bogus
             right,                   # attempt 1: correct program, fails "zero"
             right + "\n\n" + fixed,  # attempt 2: same program, corrected case
             right],                  # attempt 3: graded against the CORRECTION
        )
    log = chatter.getvalue()

    assert len(prompts) == 4, [p[:40] for p in prompts]
    assert "while n > 0" in answer.code
    # The last round ran the CORRECTED array and cleared it. Freeze turn 1's
    # cases and this reads `self=1/2` forever, however right the program is.
    assert "self=2/2" in log, log

    # Attempt 2 was still judged against the bogus case: the repair prompt it
    # was sent quotes the failure, so the correction it carries cannot grade
    # the reply that carried it.
    assert "99" in prompts[2], prompts[2][:400]
    # And the repair prompt is what tells the model correcting a case is
    # allowed at all -- without that sentence the model rewrites the program.
    assert "fix the case" in prompts[2].lower(), prompts[2][:800]


def test_the_cases_turn_asks_for_the_common_path_before_the_boundaries():
    """Three ordinary cases FIRST, then the boundary classes. A suite that is
    all boundaries never checks the common path, and a program that is wrong
    down the middle passes every one of them.

    The counts are stated PER CLASS rather than as a total, because a total
    invites a model to spend it all on the easy classes -- and a total is still
    stated on top, because `MAX_SELF_TESTS` is what the miner will actually
    run."""
    from solvers.prompts import MAX_SELF_TESTS, build_tests_prompt

    for language, entry in (("python", "g"), ("rust", "main")):
        p = build_tests_prompt(language, "Sum the digits of n.", entry, [])
        assert "THREE ordinary cases" in p, p
        assert "between TWO\n   and TEN cases" in p or "between TWO" in p
        # Order is the instruction: ordinary, then the classes.
        ordinary = p.index("THREE ordinary cases")
        for later in ("ZERO", "EVERY LIMIT AND BOUND", "NEGATIVE VALUES",
                      "TIES AND DUPLICATES", "DEGENERATE SHAPE",
                      "MOST LIKELY TO GET WRONG"):
            assert p.index(later) > ordinary, f"{later} comes before the common path"
        # The cap the model is given is the cap the miner enforces, so a model
        # that obeys it never has cases silently thinned away.
        assert f"At most {MAX_SELF_TESTS} cases in total" in p


def _thinking_backend(think_s, deadline=300.0, statement="Do a hard thing."):
    """A tab whose model produces nothing at all for `think_s`, then answers.

    Read off a live tab: `Thought for 1m 17s` before a single character
    appeared. A read whose hard cap is below that comes back `unfinished`.
    """
    log: list[tuple] = []
    program = "```python\ndef g(n):\n    return n\n```"

    class _Chat:
        provider = "claude"
        empty_reason = None
        still_writing = False

        async def send(self, text, timeout_s, extend_to_s=None):
            turn = "CASES" if "<task>" in text else "CODE"
            hard = timeout_s if extend_to_s is None else extend_to_s
            log.append((turn, timeout_s, hard))
            if hard < think_s:
                self.empty_reason, self.still_writing = "unfinished", True
                return ""
            self.empty_reason, self.still_writing = None, False
            return program

        async def close(self): pass

    class _Fleet:
        async def open(self, avoid=None): return _Chat()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement=statement,
        entrypoint="g", public_examples=[], deadline_s=deadline,
    )
    answer = asyncio.run(
        VerifyingSolver(_Fleet()).solve_task(task, timeout_s=deadline)
    )
    return log, answer


def test_the_cases_turn_can_wait_out_a_model_that_is_still_thinking():
    """Turn 1 shipped with `extend_to_s` EQUAL to its slice, which makes send's
    extension a no-op by construction — deliberately, so a rambling model could
    not eat the program's read. It also made turn 1 the one read here that
    cannot wait for a model that is still THINKING, and the one most likely to
    need to: it goes first, on a cold hard problem.

    Measured on a live tab: 77 seconds of thinking before a character appeared,
    against a 60 second cap. Turn 1 timed out, the conversation was unusable,
    and the pass was handed on."""
    log, answer = _thinking_backend(77.0)

    turns = [t for t, _, _ in log]
    assert turns[:2] == ["CASES", "CODE"], turns
    hard_s = log[0][2]
    assert hard_s >= 77.0, (
        f"the cases turn stops at {hard_s:.0f}s, under a measured think of 77s: "
        f"a thinking model is cut off and the turn is lost"
    )
    # Not "a bigger cap" — no cap. Turn 1 reads against the solve's clock.
    assert hard_s > 230.0, (
        f"the cases turn is still capped at {hard_s:.0f}s of a 300s deadline"
    )
    assert answer.code, "no program was submitted"


def test_a_cases_turn_that_costs_a_pass_does_not_cost_every_pass():
    """The regression this pair exists for, and the expensive half.

    `solve_task` retries `_attempt` up to MAX_PASSES while holding nothing, and
    nothing remembered that turn 1 had already proved unaffordable — so whatever
    made it time out, being a property of the TASK and the site rather than of
    one tab, made it time out again on every pass. Live: four passes, four
    timed-out cases turns, 189 seconds, `provider=none`, and the program never
    asked for once. The same task before the split got a single 238s read.

    One pass may be spent finding out. The rest go to the program."""
    log, answer = _thinking_backend(150.0)   # beyond any cases cap

    turns = [t for t, _, _ in log]
    assert turns[0] == "CASES"
    assert turns.count("CASES") == 1, (
        f"spent {turns.count('CASES')} passes on a cases turn that had already "
        f"failed once: {turns}"
    )
    assert "CODE" in turns, f"the program was never asked for: {turns}"
    assert answer.code, "submitted nothing at all"
    # The program's read is the full first-attempt slice, not a leftover.
    code_slice = next(s for t, s, _ in log if t == "CODE")
    assert code_slice > 200.0, f"the program only got {code_slice:.0f}s"


def test_a_blind_tab_does_not_cost_the_task_its_cases_turn():
    """The other half of `_Plan`, and the expensive half on live traffic.

    A cases turn that TIMES OUT is evidence about the task: the model is slow
    on this problem and would be slow again, so the split is dropped for the
    rest of the solve. A tab that goes BLIND is evidence about the tab, which
    `_read` retires at that same moment -- the next pass is served by another
    one. Dropping the split there cost every later pass the model's own cases,
    and with no public examples shipped (which is live traffic) those cases are
    the only grading there is: the combined prompt writes them beside the
    program, where they can be back-filled from whatever it does."""
    log: list[str] = []
    program = "```python\ndef g(n):\n    return n\n```"
    cases = '```json\n[{"args": [1], "kwargs": {}, "expected": 1}]\n```'

    class _Blind:
        provider = "claude"
        still_writing = False
        empty_reason = "unreadable"

        async def send(self, text, timeout_s, extend_to_s=None):
            log.append("CASES-blind" if "<task>" in text else "CODE-blind")
            return ""

        async def close(self): pass

    class _Healthy:
        provider = "chatgpt"
        still_writing = False
        empty_reason = None

        async def send(self, text, timeout_s, extend_to_s=None):
            turn = "CASES" if "<task>" in text else "CODE"
            log.append(turn)
            return cases if turn == "CASES" else program

        async def close(self): pass

    class _Fleet:
        def __init__(self): self.opened = 0
        async def open(self, avoid=None):
            self.opened += 1
            return _Blind() if self.opened == 1 else _Healthy()
        async def aclose(self): pass
        def stats(self): return {}

    task = SolveTask(
        problem_id="p", language="python", statement="s", entrypoint="g",
        public_examples=[], deadline_s=300.0,
    )
    chatter = io.StringIO()
    with contextlib.redirect_stdout(chatter):
        answer = asyncio.run(
            VerifyingSolver(_Fleet()).solve_task(task, timeout_s=300.0)
        )

    assert log[0] == "CASES-blind", log
    assert "CODE-blind" not in log, f"sent the program into the dead tab: {log}"
    assert log[1] == "CASES", (
        f"the second pass skipped the cases turn because the FIRST pass's tab "
        f"died: {log}"
    )
    assert "CODE" in log, f"the program was never asked for: {log}"
    assert answer.code.strip(), "submitted nothing"
    assert "that tab is gone" in chatter.getvalue(), chatter.getvalue()


def test_a_cases_turn_that_yields_nothing_still_gets_a_program():
    """Turn 1 is an improvement, not a gate. A reply with no usable cases must
    cost the time it took and nothing else."""
    prompts, _, _, answer = _two_turn(
        300.0, ["I'd be happy to help with that!", _RIGHT_PROGRAM]
    )
    assert len(prompts) == 2
    assert "<must_pass" not in prompts[1], "invented cases nobody wrote"
    assert "while n > 0" in answer.code, "lost the answer over a failed first turn"


def test_the_time_budget_is_gone_from_every_prompt():
    """A model has no clock. It cannot measure "260 seconds", and the only
    concrete behaviour the number drove was a depth ladder saying how many cases
    to trace — which the cases turn now states outright, by class, which is both
    more precise and checkable."""
    import re

    prompts = [
        build_tests_prompt("python", "s", "g", []),
        build_tests_prompt("rust", "s", "main", []),
        build_code_prompt("python", "s", "g", []),
        build_code_prompt("rust", "s", "main", []),
        build_code_prompt("python", "s", "g", [], cases=[]),
        build_code_prompt("rust", "s", "main", [],
                          cases=[{"name": "n", "args": ["1\n"], "expected": "1"}]),
        build_repair_prompt(["g(1) returned 2, expected 3"], "python", "g"),
        build_repair_prompt([], "rust", "main", defect="there is no fn main"),
    ]
    for p in prompts:
        for ghost in ("<budget>", "seconds for this reply", "seconds left",
                      "generous budget", "working budget", "tight budget"):
            assert ghost not in p, f"{ghost!r} survived in {p[:60]!r}"
        # And the general form, so a number reintroduced in some new wording
        # fails here rather than shipping. The ONE quantity of time a model can
        # act on is the per-test limit, which is a measured fact about the
        # grader and not a budget the model has to ration.
        for hit in re.findall(
            r"\b\d[\d_,]*\s*(?:s|sec|secs|second|seconds|min|mins|minute|minutes)\b",
            p,
        ):
            assert hit == "5 seconds", f"a time budget is back: {hit!r}"
        assert "budget" not in p.lower(), "a budget is back in the prompt"
    # The two rules filed under it were never about time and must survive.
    code = build_code_prompt("python", "s", "g", [])
    assert "COMPLETE BEFORE CORRECT" in code and "SENDABLE AT EVERY MOMENT" in code
    assert "COMPLETE BEFORE CORRECT" not in prompts[0], (
        "the cases turn produces no program to keep sendable"
    )


def test_no_prompt_contradicts_its_own_output_contract():
    """The count of fenced blocks is stated in four places, and three of them
    are shared by turns that ask for different replies.

    `METHOD`, `SELF_CHECK` and `PYTHON_RULES` were written when there was only
    ever one turn, so all three say "the two blocks" and one of them says the
    cases go in the second. Reuse them verbatim under the two-turn contract --
    which says ONE block -- and the turn-2 prompt tells the model both things
    at once. A model resolves that by obeying the more specific half: it emits
    a JSON block `extract_code` then has to step over, and pays for it in
    output tokens spent inside the deadline.

    So: whatever a turn's contract asks for is what every other section of
    THAT turn asks for, and no section names a block the contract did not.
    """
    from solvers.prompts import build_code_prompt, build_tests_prompt

    two_blocks = ("two blocks", "second block", "then the cases",
                  "TWO fenced blocks")
    for language, entry in (("python", "solve"), ("rust", "main")):
        # Turn 1 asks for cases only, and must not ask for a program.
        tests = build_tests_prompt(language, "Do a thing.", entry, [])
        assert "ONE fenced block" in tests
        for phrase in two_blocks:
            assert phrase not in tests, f"the cases turn mentions {phrase!r}"

        # Turn 2, both flavours: cases agreed, and cases never obtained.
        for cases in ([{"name": "n", "args": [[]], "expected": 0}], [], None):
            prompt = build_code_prompt(
                language, "Do a thing.", entry, [], cases=cases
            )
            assert "ONE fenced block" in prompt
            for phrase in two_blocks:
                assert phrase not in prompt, (
                    f"the code turn asks for ONE block and mentions {phrase!r}"
                )
            assert "<self_tests>" not in prompt

        # The single-turn prompt is the other way round and stays that way.
        single = build_code_prompt(language, "Do a thing.", entry, [])
        assert "EXACTLY TWO fenced blocks" in single
        assert "ONE fenced block" not in single
        assert "Send the program, then the cases, and nothing else." in single
        assert "<self_tests>" in single

    # And the one bullet that names where the cases live is only in the turn
    # that has somewhere to put them.
    assert "second block, as JSON" in build_code_prompt(
        "python", "Do a thing.", "solve", []
    )
    assert "second block, as JSON" not in build_code_prompt(
        "python", "Do a thing.", "solve", [], cases=[]
    )


def test_both_turns_open_with_the_same_words():
    """`_is_our_own_prompt` recognises a stale scrape by testing the head of
    whatever was last sent. Give the turns different openings and a scrape
    returning turn 1's text is unrecognised after turn 2 — reported as a defect
    instead, and a repair round spent on it."""
    a = " ".join(build_tests_prompt("python", "s", "g", []).split())
    b = " ".join(build_code_prompt("python", "s", "g", [], cases=[]).split())
    assert a[:88] == b[:88], f"\n  {a[:88]}\n  {b[:88]}"


def test_trimming_the_cases_keeps_the_boundaries():
    """The cases arrive ordinary-first by explicit instruction, so a head-slice
    keeps the three easy ones and throws away every boundary — discarding
    exactly what the mechanism exists to run, and silently."""
    from solvers.prompts import MAX_SELF_TESTS, _thin

    many = [{"name": f"c{i}"} for i in range(40)]
    kept = [c["name"] for c in _thin(many, MAX_SELF_TESTS)]
    assert len(kept) == MAX_SELF_TESTS
    assert kept[:3] == ["c0", "c1", "c2"], "the common path is kept"
    assert int(kept[-1][1:]) > 30, (
        f"trimming stopped at {kept[-1]}; a head-slice would keep c0..c19 and "
        f"drop every boundary after it"
    )
    assert _thin(many[:5], MAX_SELF_TESTS) == many[:5], "no trim when it fits"

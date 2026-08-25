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

from custom_miner import TRUNCATED, CustomMiner, SolveTask, fit_response  # noqa: E402
from solvers.browser_pool import (  # noqa: E402
    Browser,
    BrowserFleet,
    Site,
    _Tab,
    usable_busy_selectors,
)
from solvers.chatgpt_web import chatgpt_site  # noqa: E402
from solvers.claude_web import claude_site  # noqa: E402
from solvers.verify import Answer, VerifyingSolver  # noqa: E402

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
    import inspect
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
    """One profile cannot be signed in as both, and attaching twice would
    invent capacity that does not exist."""
    from solvers.roster import roster

    browsers = roster({"CLAUDE_CDP": "9222", "CHATGPT_CDP": "9222"})
    assert len(browsers) == 1 and browsers[0].site.name == "claude"
    assert "listed more than once" in capsys.readouterr().out


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
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant", [_Node(code=["print('pong')"])]
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
            page.dom["#assistant"] = [_Node(code=["print('pong')"])]
        elif selector == "#newchat":
            page.dom["#assistant"] = []

    page.on_click = handler
    assert asyncio.run(
        doctor._probe(page, _site(new_chat=("#newchat",)), "#composer", ())
    ) is True
    assert "#newchat" in page.clicked
    assert page.navigated == [], "the in-app path should not reload the page"


def test_a_reply_is_found_by_position_when_the_site_has_no_message_id():
    """claude.ai has no per-message id, so the reply is 'an assistant message
    that was not there before we pressed send'. Sound only because every task
    starts a fresh conversation."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})
    page.on_click = lambda _: page.dom.__setitem__(
        "#assistant", [_Node(code=["def g(n):\n    return n"])]
    )
    reply = asyncio.run(_tab(page, _site()).send("solve it", 2.0))
    assert reply == "def g(n):\n    return n"
    assert page.typed == ["solve it"]


def test_a_partial_answer_survives_a_deadline_that_lands_mid_stream():
    """The commonest timeout there is: the model is still typing when the budget
    runs out. Returning "" there throws away a gradeable answer and hands the
    repair round nothing to work with."""
    page = _FakePage({"#composer": [_Node()], "#send": [_Node()], "#assistant": []})

    def stream(_):
        page.dom["#assistant"] = [_Node(text="half an answer")]
        page.dom["#stop"] = [_Node()]        # still generating, and stays that way

    page.on_click = stream
    site = _site(busy=("#stop",))
    assert asyncio.run(_tab(page, site).send("solve it", 1.5)) == "half an answer"


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

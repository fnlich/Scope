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
from solvers.browser_pool import Site, _Tab, usable_busy_selectors  # noqa: E402
from solvers.chatgpt_web import ChatGPTPool, chatgpt_site  # noqa: E402
from solvers.claude_web import claude_site  # noqa: E402
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


def _pool(replaceable: bool, site=None) -> ChatGPTPool:
    pool = ChatGPTPool.__new__(ChatGPTPool)
    pool.site = site or chatgpt_site()
    pool._free, pool._size, pool._lost, pool._browsers = asyncio.Queue(), 1, 0, []

    async def spawn(context, label):
        return _Tab(pool, _DeadPage(), context, f"{label}-new") if replaceable else None

    pool._spawn = spawn
    return pool


# Both browser backends share one pool implementation, so the dead-tab fix has
# to be proven for both — that sharing is the reason it exists only once.
SITES = pytest.mark.parametrize(
    "site", [chatgpt_site(), claude_site()], ids=["chatgpt", "claude"]
)


async def _use_dead_tab(pool: ChatGPTPool) -> None:
    await pool._free.put(_Tab(pool, _DeadPage(), object(), "dead#1"))

    class LeaseOnly:
        async def open(self): return await pool._free.get()
        async def aclose(self): pass
        def stats(self): return pool.stats()

    await VerifyingSolver(
        LeaseOnly(), max_attempts=1, safety_margin_s=0, max_budget_s=30
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
    """Regression: start() was only called by run_chatgpt_miner.py, which needs
    a live chain. Hosted anywhere else the pool stayed empty and open() blocked
    on the queue forever, so every solve returned nothing."""
    import inspect

    from solvers.chatgpt_web import ChatGPTPool

    assert "await self.start()" in inspect.getsource(ChatGPTPool.open)
    start = inspect.getsource(ChatGPTPool.start)
    assert "_start_lock" in start and "if self._started" in start, "start must be idempotent"


def test_starting_twice_connects_only_once():
    import asyncio

    from solvers.chatgpt_web import ChatGPTPool

    pool = ChatGPTPool(["/tmp/does-not-need-to-exist"])
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
    assert {"claude", "chatgpt"} == set(KNOWN_BACKENDS)


def test_a_single_backend_builds_a_plain_verifying_solver(monkeypatch):
    """One name must not pay the chain's budget-splitting overhead."""
    import solvers.multi as multi

    monkeypatch.setattr(multi, "build_backend", lambda name: _Backend([RIGHT]))
    assert isinstance(multi.build_solver(["claude"]), VerifyingSolver)
    assert isinstance(multi.build_solver(["claude", "chatgpt"]), multi.FallbackSolver)


# --------------------------------------------------------------------------- #
# The Claude browser backend.
#
# There is no browser in CI, so these use a fake page. What they pin down is
# everything that does NOT need a real DOM: that `claude` means the browser and
# never an API key, that a reply is found by position (claude.ai has no
# per-message id), that a selector matching the user's own turn can never be
# returned as an answer, and that a "still generating" selector which is always
# true is thrown out before it can freeze every solve.
#
# The selectors themselves cannot be tested here — that is what
# `python -m solvers.doctor claude --probe` is for, against a real login.
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
        pass


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
    return _Tab(_SoloPool(site), page, None, "probe", composer="#composer")


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
    """WSL2 is real Linux to Python and to Firefox; it must not be refused.

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
    for name in ("custom_miner.py", "run_miner.py", "run_chatgpt_miner.py"):
        body = (root / name).read_text()
        assert "require_linux(" in body, name
        guard = body.index('require_linux("')
        for heavy in ("import httpx", "from rlvr", "from custom_miner"):
            if heavy in body:
                assert guard < body.index(heavy), f"{name}: guard runs after {heavy}"
    assert "require_linux(" in (root / "solvers" / "doctor.py").read_text()


def test_the_login_helper_is_linux_only_and_keeps_vnc_on_loopback():
    """That VNC screen is an unauthenticated view of a browser you are about to
    type a password into, on a box already exposing a public axon port."""
    from pathlib import Path

    script = (Path(__file__).resolve().parent / "scripts" / "login.sh").read_text()
    assert "uname -s" in script and "Linux" in script
    assert "-localhost" in script, "the VNC port must never leave loopback"
    assert "-nopw" in script and "-localhost" in script
    assert "Xvfb" in script  # a headless host has no screen to log in on


def test_the_backends_use_firefox_with_a_persistent_profile():
    """Firefox cannot be attached to over CDP — Playwright's connect_over_cdp is
    Chromium-only and Mozilla dropped CDP for WebDriver BiDi — so Playwright has
    to launch it, and the login has to live in a profile directory."""
    import inspect

    from solvers import browser_pool
    from solvers.multi import build_backend

    source = inspect.getsource(browser_pool)
    assert "connect_over_cdp" not in source.replace(
        "cannot do that: Playwright's ``connect_over_cdp`` is Chromium-only", ""
    ), "CDP cannot drive Firefox"
    assert "launch_persistent_context" in source

    pool = build_backend("claude")
    assert pool._profiles and pool._tabs_per_profile >= 1


def test_a_profile_that_is_not_there_names_the_login_command(capsys):
    """The first thing a new operator gets wrong. It must not be a stack trace."""
    from solvers.claude_web import ClaudeBrowserPool

    pool = ClaudeBrowserPool(["/nonexistent/profile"])
    with pytest.raises(RuntimeError, match="solvers.login"):
        asyncio.run(pool.start())
    assert "python -m solvers.login claude" in capsys.readouterr().out


def test_profiles_and_tab_counts_come_from_the_environment(monkeypatch):
    from solvers.multi import _pool_kwargs

    monkeypatch.delenv("CLAUDE_PROFILES", raising=False)
    monkeypatch.delenv("CLAUDE_HEADLESS", raising=False)
    default = _pool_kwargs("CLAUDE")
    assert default["profiles"][0].endswith("claude-1") and default["headless"] is True

    monkeypatch.setenv("CLAUDE_PROFILES", "/a, /b")
    monkeypatch.setenv("CLAUDE_TABS_PER_PROFILE", "4")
    monkeypatch.setenv("CLAUDE_HEADLESS", "false")
    kwargs = _pool_kwargs("CLAUDE")
    assert kwargs["profiles"] == ["/a", "/b"]
    assert kwargs["tabs_per_profile"] == 4 and kwargs["headless"] is False


def test_no_backend_anywhere_reads_an_api_key():
    """Every backend drives a browser. Nothing in the package reads a key, and
    nothing imports a provider SDK — a regression would be invisible otherwise,
    because a key-reading backend works fine right up until it bills someone."""
    import inspect
    from pathlib import Path

    from solvers import claude_web
    from solvers.multi import KNOWN_BACKENDS, build_backend

    backend = build_backend("claude")
    assert isinstance(backend, claude_web.ClaudeBrowserPool)
    assert backend.site.name == "claude" and "claude.ai" in backend.site.url
    assert set(KNOWN_BACKENDS) == {"claude", "chatgpt"}

    banned = ("API_KEY", "import anthropic", "from anthropic", "google.genai")
    package = Path(inspect.getfile(claude_web)).parent
    for module in sorted(package.glob("*.py")):
        body = module.read_text()
        for needle in banned:
            assert needle not in body, f"{module.name} references {needle!r}"


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
    env.write_text('# comment\nCLAUDE_PROFILES=~/a,~/b\nexport CLAUDE_URL="https://claude.ai/new"\nSHELL_WINS=from-file\n')
    monkeypatch.setenv("SHELL_WINS", "from-shell")
    monkeypatch.delenv("CLAUDE_PROFILES", raising=False)
    monkeypatch.delenv("CLAUDE_URL", raising=False)
    assert load_env_file(env) == 2
    import os

    assert os.environ["CLAUDE_PROFILES"] == "~/a,~/b"
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

    (tmp_path / ".env").write_text("CLAUDE_PROFILES=~/a\n")
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
    # The Firefox build is a second, separate download; naming only the first
    # leaves you with a working import and no browser.
    assert "playwright install firefox" in source


def test_a_browser_backend_is_started_before_serving_not_on_first_request():
    """An expired login must surface at launch, where someone is watching, not
    hours later as a failed solve on a real validator request."""
    from solvers.multi import warm_up

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

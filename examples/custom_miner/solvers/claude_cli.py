"""Solve through the Claude Code CLI, on a subscription rather than an API key.

The browser fleet exists because this miner's whole premise is answering with a
seat somebody already pays for rather than with metered API tokens. Driving
claude.ai through CDP is one way to reach that seat. The `claude` CLI is
another, and where it is available it is a far better one: it reaches the same
subscription, it has no DOM to scrape, no copy control to click, no network
stream to reconstruct, and no page that can render nothing for forty seconds
while the answer sits on the wire.

Measured on this machine against the same account, trivial prompt to answer in
hand: 2.4 seconds. The browser path's cases turn averaged 97.2 seconds over a
live run of thirteen solves. That is not a tuning difference, it is a different
order of magnitude, and it is the whole argument for this module.

AUTHENTICATION IS THE POINT, so it is worth being exact. The CLI resolves
credentials in a fixed order and an API key WINS over the subscription, so a
stray `ANTHROPIC_API_KEY` in the miner's environment would silently move every
solve onto metered billing -- the exact thing this miner exists to avoid. The
child environment therefore drops it by default (`SOLVER_CLI_ALLOW_API_KEY=1`
puts it back for an operator who wants that). `claude auth status` reporting
`"authMethod": "oauth_token"` is what a subscription looks like, and `start()`
checks for it at launch rather than letting it surface as a failed solve on a
real validator request hours later.

`--bare` is NOT used and must not be: its own help says Anthropic auth is then
"strictly ANTHROPIC_API_KEY or apiKeyHelper (OAuth and keychain are never
read)", which is precisely backwards for a subscription. `--safe-mode` is the
flag that turns off the operator's CLAUDE.md, skills, plugins, hooks and MCP
servers while leaving auth alone.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Optional

# What the CLI is told it is doing. The default Claude Code system prompt is a
# coding AGENT's -- tools, files, git, workflow -- and none of it applies to
# "write one program into the chat". Replacing it removes thousands of tokens
# of irrelevant instruction from every turn and, more importantly, stops the
# model reaching for behaviour the miner's prompts then have to argue it out of.
#
# Deliberately short. `prompts.py` carries the whole contract -- the output
# rule, the language, the entrypoint, the examples -- and has been tuned against
# live traffic. Anything said twice is a chance for the two to disagree.
SYSTEM_PROMPT = (
    "You answer programming problems. Reply with exactly what the message asks "
    "for and nothing else: no preamble, no explanation, no commentary after it. "
    "You have no tools and no filesystem; write your answer directly into the "
    "reply."
)


# How long a turn may produce NO event at all before it is treated as wedged.
#
# The CLI emits `system/init` before it has so much as called the model, so on
# a working turn the first event arrives in well under a second. Measured at the
# subscription's usage limit: nothing arrives, ever -- the process blocks
# silently rather than failing, and a `json`-format call with a 45-second cap
# produced zero bytes. Left to the slice, that is a whole deadline burnt per
# solve with nothing to show and no line of log to say why. Thirty seconds is
# far past any working start and far short of any budget worth protecting.
FIRST_EVENT_S = 30.0

# The rate-limit statuses under which a turn may proceed. The CLI's own schema
# for the event names three: `allowed`, `allowed_warning`, `rejected`.
_ALLOWED_STATUSES = ("allowed", "allowed_warning")

# The windows the CLI can name in `rateLimitType`. Two of them are for ONE
# model -- a weekly Opus cap is not a weekly Sonnet cap -- and a limit on one
# leaves the other answering. Read as: a limit reported under this type is
# scoped to the model whose name it carries; every other type is the seat.
_MODEL_WINDOWS = {"seven_day_opus": "opus", "seven_day_sonnet": "sonnet"}

# Once the limit has been reported, every solve until the reset is turned away
# at the door rather than each spending a slice rediscovering it. The reset
# time is the CLI's own; this is the longest the backend will take its word
# for it before one solve is allowed through to check. A wrong clock, a limit
# lifted early, a plan upgraded -- each is a reason the report can go stale,
# and one turn every half hour is a cheap way to notice.
LIMIT_RECHECK_S = 1800.0


class _Stalled(Exception):
    """No event arrived inside `FIRST_EVENT_S`."""


class _Limited(Exception):
    """The subscription's usage limit was reported before any answer."""


def _flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _on(name: str, default: bool) -> bool:
    raw = _flag(name)
    if not raw:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def cli_models() -> tuple[str, ...]:
    """The models to solve with, best first.

    More than one is not redundancy, it is the SECOND OPINION. `VerifyingSolver`
    asks `open(avoid=<provider>)` when a pass came back empty or wrong, and with
    a browser fleet that means another account; here it means another model,
    which is the better version of the same idea and costs nothing to offer.
    """
    raw = _flag("SOLVER_CLI_MODELS", "opus,sonnet")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    return models or ("opus",)


def cli_effort() -> str:
    """Reasoning effort, per the CLI's own `--effort` flag.

    `low` by default, and that is a considered choice rather than thrift. These
    are single-function programming problems with a hard correctness gate and a
    300-second deadline; measured here, `low` and `xhigh` both answered a
    trap-shaped arithmetic question correctly, at 25 and 44 thinking tokens.
    Raise it with `SOLVER_CLI_EFFORT` when the problems justify it -- the
    repair loop is the other place to spend that time, and it spends it on
    evidence rather than on guessing.
    """
    return _flag("SOLVER_CLI_EFFORT", "low") or "low"


def child_env() -> dict[str, str]:
    """The environment a `claude` child gets. Three removals, each measured.

    `ANTHROPIC_API_KEY` -- see the module docstring. It outranks the
    subscription, so leaving it in place would move every solve onto metered
    billing without a word.

    `CLAUDE_*` and `CLAUDECODE` -- set when the miner is itself launched from
    inside a Claude Code session, which is exactly how it gets developed.
    Measured with them left in place: the child returned the PARENT session's
    id, so `--resume` would have appended every solve to the operator's own
    conversation and the three turns of a solve would have collided with each
    other. `CLAUDE_CONFIG_DIR` is the one exception, kept: it is where the
    operator's sign-in lives, and dropping it would sign the child out.

    Everything else is kept, deliberately. `ANTHROPIC_BASE_URL`, proxy
    variables, `PATH`, `HOME` -- those are the operator's configuration and this
    module has no business editing them.
    """
    keep_key = _on("SOLVER_CLI_ALLOW_API_KEY", False)
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name == "CLAUDECODE" or (
            name.startswith("CLAUDE_") and name != "CLAUDE_CONFIG_DIR"
        ):
            continue
        if name == "ANTHROPIC_API_KEY" and not keep_key:
            continue
        env[name] = value
    return env


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resets_in(resets_at: Optional[float]) -> str:
    if resets_at is None:
        return ""
    minutes = max(0, int((resets_at - time.time()) / 60))
    return f", resets in about {minutes} minute(s)"


class CliConversation:
    """One `claude` session, driven one turn per subprocess.

    A conversation here is a SESSION ID rather than a live process: the first
    turn creates it with `--session-id`, every later turn reopens it with
    `--resume`. That is what gives the repair loop what it needs -- the model
    sees its own previous attempt beside the failure report -- without holding a
    process open between turns, which would be one more thing to leak.
    """

    def __init__(self, backend: "CliBackend", model: str) -> None:
        self._backend = backend
        self.model = model
        # What `avoid` is matched against, so a second pass asks a different
        # model. See `cli_models`.
        self.provider = f"cli:{model}"
        self._session = str(uuid.uuid4())
        self._started = False
        # Read by `VerifyingSolver` exactly as the browser tabs' are.
        self.still_writing = False
        self.empty_reason: Optional[str] = None

    def _argv(self) -> list[str]:
        argv = [
            self._backend.binary, "-p",
            # `stream-json` rather than `json`, for one reason that pays for the
            # parsing: a turn cut off by the deadline still yields the text that
            # arrived. `json` emits nothing until the turn ends, so a timeout
            # there is a total loss -- and "submit the part that arrived" is a
            # rule this miner already keeps everywhere else.
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",                 # required for stream-json
            "--model", self.model,
            "--effort", self._backend.effort,
            # No tools at all. The answer is text; a tool call is a way for the
            # turn to end without one.
            "--tools", "",
            # The operator's CLAUDE.md, skills, plugins, hooks and settings are
            # not part of this task and could only steer it. Auth is untouched.
            "--safe-mode",
            "--strict-mcp-config",
            "--disable-slash-commands",
            # Nothing may block waiting for a human who is not there.
            "--permission-prompts", "none",
            "--system-prompt", SYSTEM_PROMPT,
        ]
        argv += (["--resume", self._session] if self._started
                 else ["--session-id", self._session])
        return argv

    async def send(self, text: str, timeout_s: float) -> str:
        budget = max(1.0, float(timeout_s))
        deadline = time.monotonic() + budget
        self.still_writing = False
        self.empty_reason = None
        # Known to be at the limit already: fail in a millisecond rather than
        # spend a slice finding out again. One solve discovers it; every solve
        # after that until the reset is told at once.
        wait = self._backend.limited_for(self.model)
        if wait > 0:
            print(f"[cli] {self.provider}: the subscription limit was reached; "
                  f"not asking again for {wait / 60:.0f} minute(s)")
            self.empty_reason = "unreadable"
            return ""
        # The slot is acquired INSIDE the slice. Acquired outside it, a solve
        # queued behind four others waited with no bound at all, and the wait
        # was invisible to every clock in `verify.py`.
        try:
            await asyncio.wait_for(self._backend.slot.acquire(), timeout=budget)
        except asyncio.TimeoutError:
            print(f"[cli] {self.provider}: no free slot inside {budget:.0f}s "
                  f"({self._backend.concurrency} allowed at once)")
            self.empty_reason = "unreadable"
            return ""
        try:
            return await self._send(text, max(1.0, deadline - time.monotonic()))
        finally:
            self._backend.slot.release()

    async def _send(self, text: str, timeout_s: float) -> str:
        started = time.monotonic()
        # stderr goes to a FILE, not a pipe. A pipe is read only after stdout
        # closes, so a child that fills it first blocks on the write while this
        # side blocks on the read -- the classic two-pipe deadlock, and nothing
        # about the CLI promises to keep stderr small.
        errfile = tempfile.TemporaryFile()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errfile,
                env=self._backend.env,
                cwd=self._backend.workdir,
            )
        except Exception as exc:  # noqa: BLE001 - a dead binary is not a crash
            errfile.close()
            print(f"[cli] could not start {self._backend.binary!r}: "
                  f"{type(exc).__name__}: {exc}")
            self.empty_reason = "unreadable"
            return self._settle("")

        # The prompt goes on STDIN, not in argv, and both halves matter.
        # Measured: with stdin left inherited the CLI waits three seconds for
        # input it is never given -- 5.2s against 2.4s for the identical turn --
        # and a statement passed as an argument is bounded by ARG_MAX, which a
        # 63KB problem statement has no business being near.
        chunks: list[str] = []
        ok = False
        try:
            await asyncio.wait_for(
                self._pump(proc, text, chunks), timeout=max(1.0, timeout_s)
            )
            ok = proc.returncode == 0
        except asyncio.TimeoutError:
            # Killed mid-answer. Whatever arrived is kept and reported as
            # unfinished, which is what stops the repair loop asking a model
            # that never finished to fix what it did not say.
            self.still_writing = True
            await self._kill(proc)
            body = "".join(chunks).strip()
            print(f"[cli] {self.provider} did not finish inside "
                  f"{timeout_s:.0f}s; "
                  + (f"keeping the {len(body)} character(s) that arrived"
                     if body else "nothing had arrived"))
            if not body:
                self.empty_reason = "unfinished"
            errfile.close()
            return self._settle(body)
        except _Stalled:
            await self._kill(proc)
            self._backend.note_stall()
            print(f"[cli] {self.provider} produced no event at all in "
                  f"{FIRST_EVENT_S:.0f}s. At the subscription's usage limit the "
                  f"CLI blocks rather than fails; treating this turn as lost "
                  f"rather than spending the slice on it.")
            self.empty_reason = "unreadable"
            errfile.close()
            return self._settle("")
        except _Limited:
            await self._kill(proc)
            print(f"[cli] {self.provider} turn refused: "
                  f"{self._backend.last_error or 'limit reached'}")
            self.empty_reason = "unreadable"
            errfile.close()
            return self._settle("")
        except asyncio.CancelledError:
            await self._kill(proc)
            errfile.close()
            raise
        except Exception as exc:  # noqa: BLE001 - a failed turn is not a crash
            await self._kill(proc)
            print(f"[cli] {self.provider} turn failed: {type(exc).__name__}: {exc}")
            self.empty_reason = "unreadable"
            errfile.close()
            return self._settle("")

        body = "".join(chunks).strip()
        stderr = self._read_stderr(errfile)
        if proc.returncode:
            self._backend.last_error = stderr or f"exit {proc.returncode}"
        if not body:
            # Told apart on purpose. A non-zero exit is the SESSION failing --
            # a lost conversation, a refused resume, a broken install -- and
            # `_attempt` answers that by carrying the repair to a fresh one. A
            # clean exit with no text is the MODEL declining to say anything,
            # which is the conversation working and the turn being wasted.
            self.empty_reason = "unreadable" if proc.returncode else "no-code"
            where = self._backend.last_error if proc.returncode else "exit 0"
            print(f"[cli] {self.provider} returned nothing after "
                  f"{time.monotonic() - started:.1f}s ({where})")
        return self._settle(body if ok or body else "")

    def _settle(self, body: str) -> str:
        """Bookkeeping every exit of `_send` shares: which session the NEXT
        turn opens.

        A first turn that succeeded created the session, so later turns
        `--resume` it. A first turn that did NOT succeed may or may not have
        created it -- a turn killed by the deadline can have written the session
        file a second before -- and `--session-id` on an id that exists is a
        hard error (measured: "Session ID ... is already in use", exit 1). So a
        failed first turn takes a FRESH id: the conversation was empty either
        way, and a fresh one cannot collide.
        """
        if body and not self._started:
            self._started = True
        elif not self._started:
            self._session = str(uuid.uuid4())
        return body

    @staticmethod
    def _read_stderr(errfile) -> str:
        try:
            errfile.seek(0)
            text = errfile.read().decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001
            text = ""
        finally:
            try:
                errfile.close()
            except Exception:  # noqa: BLE001
                pass
        return text.splitlines()[-1] if text else ""

    async def _pump(self, proc, text: str, chunks: list[str]) -> None:
        """Feed the prompt in, read the event stream out, accumulate the answer."""
        assert proc.stdin is not None and proc.stdout is not None

        async def feed() -> None:
            try:
                proc.stdin.write(text.encode("utf-8"))
                await proc.stdin.drain()
            except Exception:  # noqa: BLE001 - the child may have exited early
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass

        writer = asyncio.ensure_future(feed())
        try:
            # Read in CHUNKS and split lines here, rather than `readline()`.
            #
            # `readline` is bounded by the stream reader's limit -- 64KB by
            # default -- and the CLI's final `result` event embeds the whole
            # answer a second time, so one long program puts a single line over
            # it. Measured against a 200KB answer: the read did not merely lose
            # it, it STALLED and then reported the turn unfinished after the
            # entire slice, which is the worst shape a failure can have. Sizing
            # the limit up would only move the cliff; not having one removes it.
            buffered = b""
            first = True
            while True:
                if first:
                    # See `FIRST_EVENT_S`. Only the first read: once the CLI
                    # has spoken at all, a pause is the model thinking, and the
                    # slice is the right clock for that.
                    try:
                        block = await asyncio.wait_for(
                            proc.stdout.read(65536), timeout=FIRST_EVENT_S
                        )
                    except asyncio.TimeoutError:
                        raise _Stalled() from None
                    first = False
                else:
                    block = await proc.stdout.read(65536)
                if not block:
                    break
                buffered += block
                while True:
                    newline = buffered.find(b"\n")
                    if newline < 0:
                        break
                    line, buffered = buffered[:newline], buffered[newline + 1:]
                    self._consume(line, chunks)
            if buffered.strip():
                # A last event with no trailing newline, which is what a killed
                # child leaves behind.
                self._consume(buffered, chunks)
            # Reached only on end of stream: the child has closed stdout, so
            # it is exiting. Waited for here and NOT in the `finally`, where
            # it would wait on a child that is wedged or told to stop -- which
            # the caller kills, and cannot until this returns.
            await proc.wait()
        finally:
            writer.cancel()

    def _consume(self, line: bytes, chunks: list[str]) -> None:
        """One event. A stream this cannot parse is not an error and is skipped;
        the one thing raised on purpose is `_Limited`, for a limit reported
        before any of the answer arrived."""
        try:
            event = json.loads(line.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - not every line is JSON
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "stream_event":
            inner = event.get("event") or {}
            delta = inner.get("delta") or {}
            if delta.get("type") == "text_delta":
                chunks.append(str(delta.get("text") or ""))
            return
        if kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            if self._backend.note_rate_limit(info, self.model) and not chunks:
                raise _Limited()
            return
        if kind == "result":
            self._backend.note_result(event)

    async def _kill(self, proc) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - already gone
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:  # noqa: BLE001 - reaped by the loop, or not ours
            pass

    async def close(self) -> None:
        self._backend.release()


class CliBackend:
    """A `Backend` whose conversations are `claude` sessions.

    Satisfies the same protocol the browser fleet does, so `VerifyingSolver`
    cannot tell the difference: `open`/`aclose`/`stats`, plus the `start` that
    `warm_up` calls to fail loudly at launch instead of quietly on a validator's
    request.
    """

    def __init__(
        self,
        models: Optional[tuple[str, ...]] = None,
        effort: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> None:
        self.binary = _flag("SOLVER_CLI_BIN", "claude") or "claude"
        self.models = models or cli_models()
        self.effort = effort or cli_effort()
        self.env = child_env()
        # One process per solve in flight, and no more. Measured, four at once:
        # 3.1 seconds wall clock for four answers, no contention. The bound is
        # here so a fleet of queued solves cannot become a fleet of processes.
        self._limit = concurrency or max(
            1, int(_flag("SOLVER_CLI_CONCURRENCY", "4") or "4")
        )
        self.concurrency = self._limit
        self.slot = asyncio.Semaphore(self._limit)
        # One fixed, empty directory for every child, and not a temporary one
        # per conversation. The CLI keys its per-project state on the working
        # directory -- a fresh temp dir per conversation left a new entry under
        # `~/.claude/projects/` for every solve (measured: four solves, four
        # directories, 112MB) and would have gone on doing so. Empty, so there
        # is no CLAUDE.md or `.claude/` there to be read even without
        # `--safe-mode`; fixed, so state accrues in one place the operator can
        # find and clear.
        self.workdir = os.path.expanduser(
            _flag("SOLVER_CLI_WORKDIR", "~/.hone-miner/cli") or "~/.hone-miner/cli"
        )
        os.makedirs(self.workdir, exist_ok=True)
        self._opened = 0
        self._live = 0
        self._turns = 0
        self._stalls = 0
        self._cost = 0.0
        self.last_error: Optional[str] = None
        # Wall-clock time until which the subscription is known to be spent,
        # keyed by scope: "*" for the seat, or a model name for a limit that
        # is that model's alone (see `_MODEL_WINDOWS`).
        self._limited: dict[str, float] = {}
        # Whether a turn may run on the plan's paid EXTRA usage once the
        # subscription's own window is spent. Off by default, for the reason
        # the API key is: extra usage is metered billing, and this backend's
        # whole premise is a seat that is already paid for.
        self.allow_overage = _on("SOLVER_CLI_ALLOW_OVERAGE", False)
        # Per window, the utilisation last warned about.
        self._warned: dict[str, float] = {}

    async def start(self) -> None:
        """Prove the CLI is there and signed in, at launch, where it is visible."""
        binary = shutil.which(self.binary)
        if binary is None:
            raise RuntimeError(
                f"no {self.binary!r} on PATH. The CLI backend needs Claude Code "
                f"installed and signed in: `npm i -g @anthropic-ai/claude-code` "
                f"then `claude auth login`. Set SOLVER_CLI_BIN to point at it, "
                f"or SOLVER_BACKEND=browser to use the browser fleet instead."
            )
        proc = await asyncio.create_subprocess_exec(
            binary, "auth", "status",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        try:
            status = json.loads(out.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - an older CLI may not print JSON
            status = {}
        if not status.get("loggedIn"):
            raise RuntimeError(
                f"{binary} is installed but not signed in. Run `claude auth "
                f"login` as the user this miner runs as."
            )
        method = str(status.get("authMethod") or "?")
        # Said out loud because it is the whole point of this backend, and
        # because the failure it warns about is invisible: an API key answers
        # every solve just as well and bills for every one of them.
        billing = (
            "subscription (OAuth)" if method == "oauth_token"
            else f"authMethod={method} — NOT a subscription; this bills per token"
        )
        print(f"[cli] {binary} ready: {billing}, models "
              f"{'/'.join(self.models)}, effort {self.effort}, "
              f"{self._limit} at a time")

    async def open(
        self, avoid: Optional[str] = None, timeout_s: Optional[float] = None
    ) -> CliConversation:
        """A fresh session. `avoid` picks a different model where there is one."""
        model = self.models[0]
        if avoid:
            for candidate in self.models:
                if f"cli:{candidate}" != avoid:
                    model = candidate
                    break
        self._opened += 1
        self._live += 1
        return CliConversation(self, model)

    def release(self) -> None:
        self._live = max(0, self._live - 1)

    def note_result(self, event: dict) -> None:
        self._turns += 1
        try:
            self._cost += float(event.get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        if event.get("is_error"):
            self.last_error = str(event.get("subtype") or "error")

    def limited_for(self, model: Optional[str] = None) -> float:
        """Seconds for which `model` is known to be turned away; 0 if not.

        The seat's limit applies to every model. A model's own weekly limit
        applies to it alone, and the name is matched loosely -- `opus` is what
        the CLI calls the window and `claude-opus-5` is what an operator may
        have configured -- so that the two are read as the same thing.
        """
        now = time.time()
        until = self._limited.get("*", 0.0)
        if model:
            for scope, when in self._limited.items():
                if scope != "*" and scope in model:
                    until = max(until, when)
        return max(0.0, until - now)

    def note_stall(self) -> None:
        self._stalls += 1

    def note_rate_limit(self, info: dict, model: Optional[str] = None) -> bool:
        """Read one `rate_limit_event`. True when this turn is at a limit.

        The one failure mode a subscription has that an API key does not, and
        the one this miner cannot otherwise see: past the limit every solve
        fails identically and for a reason no line of the log would name.

        The event's shape, as the CLI's own schema gives it: a `status` for the
        turn -- `allowed`, `allowed_warning` or `rejected` -- a `rateLimitType`
        naming the window that status is about, and `unifiedWindows` with an
        entry per rolling window (`five_hour`, `seven_day`) each carrying its
        own `utilization` and `resetsAt`. A top-level `utilization`, when there
        is one, describes the `rateLimitType` window and is read as one more.
        """
        if not isinstance(info, dict):
            return False
        status = str(info.get("status") or "allowed")
        kind = str(info.get("rateLimitType") or "")
        resets_at = _number(info.get("resetsAt"))
        windows: dict[str, dict] = {}
        unified = info.get("unifiedWindows")
        if isinstance(unified, dict):
            windows.update({str(k): v for k, v in unified.items()
                            if isinstance(v, dict)})
        if kind and kind not in windows:
            if _number(info.get("utilization")) is not None:
                windows[kind] = info
        overage = bool(info.get("isUsingOverage"))
        limited = status not in _ALLOWED_STATUSES
        scope = _MODEL_WINDOWS.get(kind, "*") if limited else "*"
        for name, window in windows.items():
            used = _number(window.get("utilization"))
            if used is None:
                continue
            reset = _number(window.get("resetsAt"))
            # A window fully used with nothing covering the overflow. The
            # status is the CLI's word on the turn and is believed first; this
            # is for the turn it reports `allowed` on a spent window, which
            # measured here is one that then blocks without another event.
            if (used >= 1.0 and not overage
                    and name not in ("seven_day_overage_included", "overage")):
                if not limited:
                    limited, scope = True, _MODEL_WINDOWS.get(name, "*")
                if reset is not None and (resets_at is None or reset < resets_at):
                    resets_at = reset
            # Said once per tenth of the budget from 80% up, per window, not
            # once per turn: the point is a line the operator sees coming,
            # not one that drowns the log as it does.
            if used >= 0.8 and used >= self._warned.get(name, 0.0) + 0.1:
                self._warned[name] = used
                print(f"[cli] NOTE: {used:.0%} of the {name} subscription limit "
                      f"used{_resets_in(reset)}. Past it every solve fails "
                      f"until it resets.")
        # A turn the subscription's window no longer covers, running on the
        # plan's paid extra usage. Allowed through only on request; otherwise
        # it is the limit, and treated exactly as one.
        if overage and not self.allow_overage:
            if not limited:
                limited, scope = True, "*"
            reason = ("the subscription window is spent and extra usage is "
                      "not enabled")
        else:
            reason = f"status {status}" + (f" on {kind}" if kind else "")
        if overage and self.allow_overage and not self._warned.get("overage"):
            self._warned["overage"] = 1.0
            print("[cli] NOTE: the subscription window is spent; turns are now "
                  "running on the plan's paid extra usage (SOLVER_CLI_ALLOW_OVERAGE)")
        if not limited:
            return False
        now = time.time()
        until = resets_at if resets_at is not None and resets_at > now else now + 300.0
        until = min(until, now + LIMIT_RECHECK_S)
        # Said once per limit, not once per turn: the same report arriving on
        # the next turn lands a few seconds later and must not read as news.
        if until > self._limited.get(scope, 0.0) + 60.0:
            self._limited[scope] = until
            who = "every model" if scope == "*" else f"the {scope} model"
            self.last_error = f"subscription limit reached ({reason})"
            print(f"[cli] the subscription limit has been reached ({reason})"
                  f"{_resets_in(resets_at)}; {who} is turned away for the next "
                  f"{(until - now) / 60:.0f} minute(s)")
        return scope == "*" or (model is not None and scope in model)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "claude-cli",
            "models": list(self.models),
            "effort": self.effort,
            "concurrency": self._limit,
            "sessions_opened": self._opened,
            "sessions_live": self._live,
            "turns": self._turns,
            "stalls": self._stalls,
            "limited_for_s": {scope: round(max(0.0, until - time.time()))
                              for scope, until in self._limited.items()
                              if until > time.time()},
            # List price for the tokens spent. On a subscription nothing is
            # billed this way -- it is here as a measure of how much work went
            # through the seat, not as an invoice.
            "list_price_usd": round(self._cost, 4),
        }

    async def aclose(self) -> None:
        return None

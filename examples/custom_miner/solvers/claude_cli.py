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

    `CLAUDE_CODE_*` and `CLAUDECODE` -- set when the miner is itself launched
    from inside a Claude Code session, which is exactly how it gets developed.
    Measured with them left in place: the child returned the PARENT session's
    id, so `--resume` would have appended every solve to the operator's own
    conversation and the three turns of a solve would have collided with each
    other.

    Everything else is kept, deliberately. `ANTHROPIC_BASE_URL`, proxy
    variables, `PATH`, `HOME` -- those are the operator's configuration and this
    module has no business editing them.
    """
    keep_key = _on("SOLVER_CLI_ALLOW_API_KEY", False)
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name == "CLAUDECODE" or name.startswith("CLAUDE_"):
            continue
        if name == "ANTHROPIC_API_KEY" and not keep_key:
            continue
        env[name] = value
    return env


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
        # A directory of its own, empty, so the CLI's working directory holds
        # nothing it could read and no CLAUDE.md it could pick up. `--safe-mode`
        # already turns discovery off; this makes it true rather than merely
        # configured.
        self._cwd = tempfile.mkdtemp(prefix="hone-cli-")
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
        async with self._backend.slot:
            return await self._send(text, timeout_s)

    async def _send(self, text: str, timeout_s: float) -> str:
        self.still_writing = False
        self.empty_reason = None
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._backend.env,
                cwd=self._cwd,
            )
        except Exception as exc:  # noqa: BLE001 - a dead binary is not a crash
            print(f"[cli] could not start {self._backend.binary!r}: "
                  f"{type(exc).__name__}: {exc}")
            self.empty_reason = "unreadable"
            return ""

        # The prompt goes on STDIN, not in argv, and both halves matter.
        # Measured: with stdin left inherited the CLI waits three seconds for
        # input it is never given -- 5.2s against 2.4s for the identical turn --
        # and a statement passed as an argument is bounded by ARG_MAX, which a
        # 63KB problem statement has no business being near.
        chunks: list[str] = []
        try:
            await asyncio.wait_for(
                self._pump(proc, text, chunks), timeout=max(1.0, timeout_s)
            )
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
            return body
        except asyncio.CancelledError:
            await self._kill(proc)
            raise
        except Exception as exc:  # noqa: BLE001 - a failed turn is not a crash
            await self._kill(proc)
            print(f"[cli] {self.provider} turn failed: {type(exc).__name__}: {exc}")
            self.empty_reason = "unreadable"
            return ""

        self._started = True
        body = "".join(chunks).strip()
        if not body:
            # Told apart on purpose. A non-zero exit is the SESSION failing --
            # a lost conversation, a refused resume, a broken install -- and
            # `_attempt` answers that by carrying the repair to a fresh one. A
            # clean exit with no text is the MODEL declining to say anything,
            # which is the conversation working and the turn being wasted.
            self.empty_reason = (
                "unreadable" if proc.returncode else "no-code"
            )
            where = self._backend.last_error or f"exit {proc.returncode}"
            print(f"[cli] {self.provider} returned nothing after "
                  f"{time.monotonic() - started:.1f}s ({where})")
        return body

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
            while True:
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
        finally:
            writer.cancel()
            stderr = b""
            try:
                stderr = await proc.stderr.read() if proc.stderr else b""
            except Exception:  # noqa: BLE001
                pass
            await proc.wait()
            if proc.returncode:
                self._backend.last_error = (
                    stderr.decode("utf-8", "replace").strip().splitlines()[-1]
                    if stderr.strip() else f"exit {proc.returncode}"
                )

    def _consume(self, line: bytes, chunks: list[str]) -> None:
        """One event. Never raises: a stream this cannot parse is not an error."""
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
            self._backend.note_rate_limit(event.get("rate_limit_info") or {})
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
        shutil.rmtree(self._cwd, ignore_errors=True)


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
        self.slot = asyncio.Semaphore(self._limit)
        self._opened = 0
        self._live = 0
        self._turns = 0
        self._cost = 0.0
        self.last_error: Optional[str] = None
        self._warned_limit = 0.0

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

    def note_rate_limit(self, info: dict) -> None:
        """Say when the subscription is running out, once per tenth of it.

        The one failure mode a subscription has that an API key does not, and
        the one this miner cannot otherwise see: past the limit every solve
        fails identically and for a reason no line of the log would name.
        """
        try:
            used = float(info.get("utilization") or 0.0)
        except (TypeError, ValueError):
            return
        if used < 0.8 or used < self._warned_limit + 0.1:
            return
        self._warned_limit = used
        window = str(info.get("rateLimitType") or "usage")
        resets = info.get("resetsAt")
        when = ""
        if isinstance(resets, (int, float)):
            minutes = max(0, int((float(resets) - time.time()) / 60))
            when = f", resets in about {minutes} minute(s)"
        print(f"[cli] NOTE: {used:.0%} of the {window} subscription limit used"
              f"{when}. Past it every solve fails until it resets.")

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "claude-cli",
            "models": list(self.models),
            "effort": self.effort,
            "concurrency": self._limit,
            "sessions_opened": self._opened,
            "sessions_live": self._live,
            "turns": self._turns,
            # List price for the tokens spent. On a subscription nothing is
            # billed this way -- it is here as a measure of how much work went
            # through the seat, not as an invoice.
            "list_price_usd": round(self._cost, 4),
        }

    async def aclose(self) -> None:
        return None

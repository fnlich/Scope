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

WHAT HAPPENS WHEN IT FAILS is the other half of this module, because a miner
is scored on every request and a seat has three ways of not answering that an
API key does not:

  the subscription's usage limit   -> another ACCOUNT still answers
  the service refusing a model     -> another MODEL still answers
  something neither explains       -> the next rung of both

The backend keeps an OUTAGE TABLE: what is known not to work, for whom, until
when, and why. A usage limit is an account's, for every model on it, until the
CLI's own reset time. A server error is a model's, on every account, for ten
minutes. A turn that never said anything at all is one (account, model) pair's,
briefly. Picking who answers a fresh solve is then a walk down the ladder of
(profile, account) pairs -- the default model on each signed-in account, then
each emergency profile on each -- skipping whatever the table says is out. A
turn that fails part-way hops the same way inside its own slice where the
session can follow it (a model can change under a session; an account cannot),
and otherwise hands the repair back to `VerifyingSolver`, whose fresh-
conversation path was built for exactly this.

Recovery is the same walk: an outage expires, the ladder puts the default pair
back at the top, and the next real solve is the probe. If it fails again the
outage is re-armed for another ten minutes. Nothing runs in the background and
nothing is guessed; the miner's own traffic is the health check.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
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

EFFORTS = ("low", "medium", "high", "xhigh", "max")

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

# Once a limit has been reported, every solve until the reset is turned away
# from that account rather than each spending a slice rediscovering it. The
# reset time is the CLI's own; this is the longest the backend will take its
# word for it before one solve is allowed through to check. A wrong clock, a
# limit lifted early, a plan upgraded -- each is a reason the report can go
# stale, and one turn every half hour is a cheap way to notice.
LIMIT_RECHECK_S = 1800.0

# How long a MODEL is left alone after the service refused it -- an overload,
# a 5xx, a connection that never completed. The operator's number, and the
# recovery cadence: when it expires the default model is back at the top of
# the ladder and the next real solve tries it. `SOLVER_CLI_RECOVERY_S`.
DEFAULT_RECOVERY_S = 600.0

# How long one (account, model) PAIR is left alone after a turn that said
# nothing at all, or failed twice running for no reason the CLI would name.
# Short, because the cause is unknown: the ladder moves on, and the pair gets
# another chance soon.
PAIR_HOLD_S = 300.0

# How long an account is left alone after the CLI reported it signed out or
# refused. A sign-in is an operator's action, and half an hour between checks
# is enough to notice one without spending turns on a seat that is not there.
AUTH_HOLD_S = 1800.0

# The retry event, counted from one, at which the model is declared out. The
# CLI retries an overloaded or failing request itself, with a growing delay
# between attempts, and each retry is reported on the stream as a
# `system/api_retry` event carrying the HTTP status. The first such event is
# ordinary weather and its wait is paid. The second means three requests have
# failed in a row and the next wait is already seconds long: a different model
# will answer sooner than this one will. Counted HERE, per turn, rather than
# read off the event's own `attempt` field, whose base is the CLI's business.
API_RETRIES_TOLERATED = 2

# A single announced retry wait this long is a wait this turn cannot afford
# whatever the attempt number.
LONG_RETRY_MS = 15000

# The least of a slice worth starting another attempt in.
HOP_FLOOR_S = 5.0

# Consecutive unexplained failures on one pair before it is set aside.
FAILURES_BEFORE_HOLD = 2

# What an error says, sorted into what it means for the ladder. Checked
# against the CLI's error text lower-cased; the status codes as whole words
# so a session id cannot match one.
_AUTH_MARKS = ("not logged in", "please run /login", "invalid authentication",
               "authentication_error", "oauth token", "invalid api key",
               "permission_error", "unauthorized")
_LIMIT_MARKS = ("rate limit", "rate_limit", "usage limit", "limit reached",
                "out of extra usage", "out_of_credits")
_SERVER_MARKS = ("overloaded", "internal server error", "api_error",
                 "fetch failed", "econnreset", "econnrefused", "etimedout",
                 "socket hang up", "upstream", "timed out", "network error",
                 "service unavailable", "bad gateway", "gateway time",
                 # A model the service will not serve -- unknown, retired, or
                 # not on this plan. Measured, a bad alias: exit 1 and "There's
                 # an issue with the selected model ... It may not exist or you
                 # may not have access to it". Another model is the answer.
                 "issue with the selected model", "not_found_error",
                 "unrecognized_model", "may not exist")
# A three-digit code counts only next to a word that makes it a status:
# "API Error: 529", "HTTP 500", "status 429", "error code 503". Bare, it
# matched the column of a stack frame (`cli.js:512:98765`), a duration
# ("took 503 ms") and a port -- and each parked a model or a seat.
_STATUS_WORD = re.compile(r"(?:status|error|http|code)\W{0,4}(\d{3})(?![\d:.])")


class _Stalled(Exception):
    """No event arrived inside `FIRST_EVENT_S`."""


class _Limited(Exception):
    """The subscription's usage limit was reported before any answer."""


class _Degraded(Exception):
    """The service is failing this model: overloaded, erroring, unreachable."""


class _Unauthorised(Exception):
    """This account is not signed in, or was refused."""


def _flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _on(name: str, default: bool) -> bool:
    raw = _flag(name)
    if not raw:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def cli_models() -> tuple[str, ...]:
    """The models to solve with, best first.

    The first is the DEFAULT: what every solve is answered with while nothing
    is wrong. Any after it are second-opinion models -- `VerifyingSolver` asks
    `open(avoid=<provider>)` when a pass came back wrong, and with a browser
    fleet that means another account; here it means another model. There is
    no second entry by default, because the emergency ladder already holds a
    different model at a higher effort, and that is the better second opinion.
    """
    raw = _flag("SOLVER_CLI_MODELS", "opus")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    if any(not m or any(ch.isspace() for ch in m) for m in models):
        raise SystemExit(f"SOLVER_CLI_MODELS={raw!r}: expected comma-separated "
                         f"model aliases")
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
    effort = _flag("SOLVER_CLI_EFFORT", "low") or "low"
    if effort not in EFFORTS:
        raise SystemExit(f"SOLVER_CLI_EFFORT={effort!r}; expected one of "
                         f"{', '.join(EFFORTS)}")
    return effort


@dataclass(frozen=True)
class Profile:
    """A model at an effort: what one turn is asked of."""

    model: str
    effort: str

    @property
    def label(self) -> str:
        return f"{self.model}/{self.effort}"


def cli_emergency_profiles(default_effort: Optional[str] = None) -> tuple[Profile, ...]:
    """What answers when the default model will not, in order.

    `SOLVER_CLI_EMERGENCY_PROFILES=fable:low,sonnet:low` -- each entry a model
    alias the CLI accepts, optionally with an effort after a colon. Those are
    the defaults, and the order is measured rather than ranked: on a real
    production problem the program turn took fable 38s at low effort, opus
    86s, sonnet 161s -- and sonnet at high effort, or either model at high,
    did not finish inside 200s. A rung that cannot answer inside the deadline
    is not a rung. Effort above low on the ladder is for problems that
    justify it, set by the operator who measured it.
    """
    default_effort = default_effort or cli_effort()
    raw = _flag("SOLVER_CLI_EMERGENCY_PROFILES", "fable:low,sonnet:low")
    profiles: list[Profile] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        model, _, effort = entry.partition(":")
        model, effort = model.strip(), effort.strip() or default_effort
        if not model or any(ch.isspace() for ch in model):
            raise SystemExit(
                f"SOLVER_CLI_EMERGENCY_PROFILES entry {entry!r}: expected "
                f"model or model:effort"
            )
        if effort not in EFFORTS:
            raise SystemExit(
                f"SOLVER_CLI_EMERGENCY_PROFILES entry {entry!r}: effort must be "
                f"one of {', '.join(EFFORTS)}"
            )
        profiles.append(Profile(model, effort))
    return tuple(profiles)


def cli_backup_dirs() -> tuple[str, ...]:
    """Config directories of the BACKUP accounts, in order.

    A second subscription is the answer to the first's usage limit, and the CLI
    keeps one sign-in per `CLAUDE_CONFIG_DIR`. So a backup account is a
    directory the operator signed in once:

        python -m solvers.claude_cli login ~/.hone-miner/claude-2

    and then named here: `SOLVER_CLI_BACKUP_ACCOUNTS=~/.hone-miner/claude-2`.
    The primary account is whatever `claude` is signed in as with no override.
    """
    raw = _flag("SOLVER_CLI_BACKUP_ACCOUNTS", "")
    return tuple(os.path.expanduser(d.strip()) for d in raw.split(",") if d.strip())


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


@dataclass(frozen=True)
class Account:
    """One sign-in. `config_dir` None is the CLI's own default login."""

    name: str
    config_dir: Optional[str]
    env: dict[str, str] = field(compare=False, hash=False, repr=False)

    @property
    def login_command(self) -> str:
        if self.config_dir is None:
            return "claude auth login"
        return f"python -m solvers.claude_cli login {self.config_dir}"


def _accounts(base_env: dict[str, str]) -> tuple[Account, ...]:
    accounts = [Account("primary", None, dict(base_env))]
    seen = {os.path.realpath(os.path.expanduser(base_env.get("CLAUDE_CONFIG_DIR", "~/.claude")))}
    for directory in cli_backup_dirs():
        real = os.path.realpath(directory)
        if real in seen:
            # The primary named as its own backup would be one account counted
            # twice, and a limit on it "switching" to itself.
            continue
        seen.add(real)
        env = dict(base_env)
        env["CLAUDE_CONFIG_DIR"] = directory
        name = os.path.basename(directory.rstrip("/")) or f"backup-{len(accounts)}"
        if any(a.name == name for a in accounts):
            name = f"{name}-{len(accounts)}"
        accounts.append(Account(name, directory, env))
    return tuple(accounts)


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


def _minutes(seconds: float) -> str:
    return f"{max(1, round(seconds / 60))} minute(s)"


def classify(text: str) -> Optional[str]:
    """What an error message means: `auth`, `limit`, `server`, or None.

    Applied to the CLI's own words -- a `result` event's text, an `api_retry`
    event's error, the last line of stderr -- so that the ladder moves in the
    direction the failure points. An account that is signed out is not fixed
    by another model, and an overloaded model is not fixed by another account.
    """
    low = (text or "").lower()
    if not low:
        return None
    codes = {int(c) for c in _STATUS_WORD.findall(low)}
    if any(mark in low for mark in _AUTH_MARKS) or codes & {401, 403}:
        return "auth"
    if any(mark in low for mark in _LIMIT_MARKS) or 429 in codes:
        return "limit"
    if any(mark in low for mark in _SERVER_MARKS) or any(c >= 500 for c in codes) or 404 in codes:
        return "server"
    return None


@dataclass
class _Outage:
    until: float
    reason: str


class CliConversation:
    """One `claude` session, driven one turn per subprocess.

    A conversation here is a SESSION ID rather than a live process: the first
    turn creates it with `--session-id`, every later turn reopens it with
    `--resume`. That is what gives the repair loop what it needs -- the model
    sees its own previous attempt beside the failure report -- without holding a
    process open between turns, which would be one more thing to leak.

    The session belongs to an account and may be continued by any model: the
    CLI keys it on the config directory, and `--model` is a per-turn choice.
    So a turn that the service refuses can hop models and keep its history,
    while a turn the account cannot serve hops accounts only while the session
    is still empty.
    """

    def __init__(self, backend: "CliBackend", account: Account, profile: Profile) -> None:
        self._backend = backend
        self.account = account
        self.profile = profile
        self._session = str(uuid.uuid4())
        self._started = False
        # Read by `VerifyingSolver` exactly as the browser tabs' are.
        self.still_writing = False
        self.empty_reason: Optional[str] = None
        self.hops = 0
        # What the CLI's `result` said went wrong, this turn, and how many
        # retries it has reported this turn.
        self._turn_error: Optional[str] = None
        self._retries = 0

    @property
    def model(self) -> str:
        return self.profile.model

    @property
    def effort(self) -> str:
        return self.profile.effort

    @property
    def provider(self) -> str:
        """What `avoid` is matched against. Reflects the pair NOW, after any hop."""
        return self._backend.provider_of(self.account, self.profile)

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
            "--effort", self.effort,
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
        """One turn, hopping down the ladder inside its own slice when it must.

        Every attempt ends in a VERDICT, and the verdict decides what happens
        next: an answer, a partial one, or a model that chose to say nothing
        is returned as it is; a limit, a refusal, a stall or a repeated failure
        is a reason to ask someone else, and the slice pays for one more try
        for as long as there is one to make.
        """
        budget = max(1.0, float(timeout_s))
        deadline = time.monotonic() + budget
        self.still_writing = False
        self.empty_reason = None
        while True:
            # Known to be out already: hop in a millisecond rather than spend
            # a slice finding out again. One solve discovers it; every solve
            # after that until the reset is told at once.
            wait, why = self._backend.outage_for(self.account, self.model)
            if wait > 0:
                if self._hop(why):
                    continue
                print(f"[cli] {self.provider}: turned away ({why}; "
                      f"{_minutes(wait)} to go) and nobody else on the ladder "
                      f"can answer")
                self.empty_reason = "unreadable"
                return ""
            left = deadline - time.monotonic()
            # The slot is acquired INSIDE the slice. Acquired outside it, a
            # solve queued behind four others waited with no bound at all, and
            # the wait was invisible to every clock in `verify.py`.
            try:
                await asyncio.wait_for(self._backend.slot.acquire(), timeout=max(0.0, left))
            except asyncio.TimeoutError:
                print(f"[cli] {self.provider}: no free slot inside {budget:.0f}s "
                      f"({self._backend.concurrency} allowed at once)")
                self.empty_reason = "unreadable"
                return ""
            try:
                # Checked again with the slot in hand: a conversation that
                # queued for it may have been passed by another that marked
                # this pair out meanwhile, and "told at once" should hold for
                # it too.
                if self._backend.outage_for(self.account, self.model)[0] > 0:
                    continue
                body, verdict = await self._send(
                    text, max(1.0, deadline - time.monotonic())
                )
            finally:
                self._backend.slot.release()
            if verdict not in ("limited", "stalled", "degraded", "auth"):
                # An answer, a partial one, a model that chose to say nothing
                # -- or a failure with no stated cause. That last one is NOT
                # hopped on: nothing says another rung would do better, and
                # a turn that hopped on every unexplained exit went round the
                # ladder in a circle (measured, in a test: opus, sonnet, opus,
                # sonnet, fable, all on one lost session id). It is counted
                # instead, and the pair is set aside after two in a row.
                return body
            # A failure the ladder has an answer to. `_send` has already
            # recorded it; what remains is whether anyone else can take the
            # turn, and whether there is slice enough left to ask.
            if deadline - time.monotonic() < HOP_FLOOR_S:
                return ""
            if not self._hop(self._backend.last_error or verdict):
                return ""

    def _hop(self, why: str) -> bool:
        """Move this conversation to the next pair that can answer, if any.

        A model can change under a session; an account cannot -- the session
        lives in the account's config directory -- so once the session holds
        anything, only same-account pairs are open, and the caller hands an
        account-shaped failure back to `VerifyingSolver` as an unreadable
        conversation instead. Its fresh-conversation path re-sends the whole
        context, and `open()` then walks the ladder for it.
        """
        pair = self._backend.next_pair(
            self.account, self.model, same_account=self._started
        )
        if pair is None:
            return False
        account, profile = pair
        was = self.provider
        if account != self.account:
            self._session = str(uuid.uuid4())
        self.account, self.profile = account, profile
        self.hops += 1
        self._backend.note_hop()
        print(f"[cli] hop: {was} -> {self.provider} ({why})")
        return True

    async def _send(self, text: str, timeout_s: float) -> tuple[str, str]:
        """One subprocess: the body that arrived and the verdict on the turn.

        Verdicts: `ok`, `partial` (cut off with text in hand), `unfinished`
        (cut off with none), `no-code` (a clean exit that said nothing),
        `limited`, `stalled`, `degraded`, `auth`, `failed`.
        """
        started = time.monotonic()
        self._turn_error = None
        self._retries = 0
        errfile = None
        try:
            # stderr goes to a FILE, not a pipe. A pipe is read only after
            # stdout closes, so a child that fills it first blocks on the write
            # while this side blocks on the read -- the classic two-pipe
            # deadlock, and nothing about the CLI promises to keep stderr small.
            errfile = tempfile.TemporaryFile()
            proc = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errfile,
                env=self.account.env,
                cwd=self._backend.workdir,
            )
        except Exception as exc:  # noqa: BLE001 - a dead binary is not a crash
            if errfile is not None:
                errfile.close()
            print(f"[cli] could not start {self._backend.binary!r}: "
                  f"{type(exc).__name__}: {exc}")
            self._backend.last_error = f"{type(exc).__name__}: {exc}"
            return self._verdict("", "failed")

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
            errfile.close()
            body = "".join(chunks).strip()
            print(f"[cli] {self.provider} did not finish inside "
                  f"{timeout_s:.0f}s; "
                  + (f"keeping the {len(body)} character(s) that arrived"
                     if body else "nothing had arrived"))
            return self._verdict(body, "partial" if body else "unfinished")
        except _Stalled:
            await self._kill(proc)
            errfile.close()
            self._backend.note_stall(self.account, self.model)
            print(f"[cli] {self.provider} produced no event at all in "
                  f"{FIRST_EVENT_S:.0f}s. At the subscription's usage limit the "
                  f"CLI blocks rather than fails; treating this turn as lost "
                  f"rather than spending the slice on it.")
            return self._verdict("", "stalled")
        except _Limited:
            await self._kill(proc)
            errfile.close()
            print(f"[cli] {self.provider} turn refused: "
                  f"{self._backend.last_error or 'limit reached'}")
            return self._verdict("", "limited")
        except _Degraded as exc:
            await self._kill(proc)
            errfile.close()
            self._backend.note_degraded(self.model, str(exc))
            return self._verdict("", "degraded")
        except _Unauthorised as exc:
            await self._kill(proc)
            errfile.close()
            self._backend.note_unauthorised(self.account, str(exc))
            return self._verdict("", "auth")
        except asyncio.CancelledError:
            await self._kill(proc)
            errfile.close()
            raise
        except Exception as exc:  # noqa: BLE001 - a failed turn is not a crash
            await self._kill(proc)
            errfile.close()
            print(f"[cli] {self.provider} turn failed: {type(exc).__name__}: {exc}")
            self._backend.last_error = f"{type(exc).__name__}: {exc}"
            return self._verdict("", "failed")

        body = "".join(chunks).strip()
        stderr = self._read_stderr(errfile) or self._turn_error or ""
        if proc.returncode:
            self._backend.last_error = stderr or f"exit {proc.returncode}"
        if body and (self._turn_error or proc.returncode):
            # Text arrived and THEN the turn failed -- the connection broke
            # mid-answer and the CLI gave up. What arrived is a fragment of a
            # reply, not the reply, and is reported the way a turn cut off by
            # the clock is: kept, and marked unfinished, so the repair loop
            # does not ask the model to fix what it never finished saying.
            self.still_writing = True
            print(f"[cli] {self.provider} failed after {len(body)} character(s) "
                  f"had arrived ({self._backend.last_error or 'error'}); "
                  f"keeping them as an unfinished reply")
            return self._verdict(body, "partial")
        if body:
            return self._verdict(body, "ok")
        if not proc.returncode:
            # A clean exit with no text is the MODEL declining to say
            # anything: the conversation is working and the turn was wasted.
            # Telling it so is what fixes that, and `_attempt` does.
            print(f"[cli] {self.provider} returned nothing after "
                  f"{time.monotonic() - started:.1f}s (exit 0)")
            return self._verdict("", "no-code")
        # A non-zero exit is the SESSION failing -- a lost conversation, a
        # refused resume, a signed-out account, a broken install -- and what
        # the CLI said about it decides which way the ladder moves.
        print(f"[cli] {self.provider} returned nothing after "
              f"{time.monotonic() - started:.1f}s ({self._backend.last_error})")
        kind = classify(stderr)
        if kind == "auth":
            self._backend.note_unauthorised(self.account, stderr)
            return self._verdict("", "auth")
        if kind == "server":
            self._backend.note_degraded(self.model, stderr)
            return self._verdict("", "degraded")
        if kind == "limit":
            self._backend.note_limit(self.account, "*", None, stderr)
            return self._verdict("", "limited")
        return self._verdict("", "failed")

    def _verdict(self, body: str, verdict: str) -> tuple[str, str]:
        """Bookkeeping every exit of `_send` shares.

        The session the NEXT turn opens: a first turn that succeeded created
        it, so later turns `--resume` it. A first turn that did NOT succeed may
        or may not have created it -- a turn killed by the deadline can have
        written the session file a second before -- and `--session-id` on an
        id that exists is a hard error (measured: "Session ID ... is already in
        use", exit 1). So a failed first turn takes a FRESH id: the
        conversation was empty either way, and a fresh one cannot collide.

        And `empty_reason`, which `VerifyingSolver` reads: `unreadable` when
        the CONVERSATION is why nothing came back, `unfinished` when the clock
        was, `no-code` when the model was.
        """
        if body and not self._started:
            self._started = True
        elif not self._started:
            self._session = str(uuid.uuid4())
        if verdict in ("ok", "partial"):
            self.empty_reason = None
            self._backend.note_success(self.account, self.model)
        elif verdict == "unfinished":
            self.empty_reason = "unfinished"
        elif verdict == "no-code":
            self.empty_reason = "no-code"
        else:
            self.empty_reason = "unreadable"
            if verdict == "failed":
                self._backend.note_failure(self.account, self.model)
        return body, verdict

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
        what IS raised on purpose is a failure the ladder answers -- a limit,
        a refusal, a service that is failing -- reported before any of the
        answer arrived, when moving on costs nothing."""
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
            if self._backend.note_rate_limit(info, self.model, self.account) and not chunks:
                raise _Limited()
            return
        if kind == "system" and event.get("subtype") == "api_retry":
            self._retry(event, chunks)
            return
        if kind == "result":
            self._backend.note_result(event)
            if event.get("is_error"):
                self._turn_error = str(event.get("result") or event.get("subtype") or "")
                if not chunks:
                    self._failed_result(event)

    def _retry(self, event: dict, chunks: list[str]) -> None:
        """The CLI is retrying the request itself, and says why.

        `{"type":"system","subtype":"api_retry","attempt":n,"max_retries":m,
        "retry_delay_ms":d,"error_status":s,"error":"..."}`, once per retry.
        A retry restarts the message, so text that arrived before it is not
        the answer's beginning any more and is dropped.
        """
        if chunks:
            del chunks[:]
        self._retries += 1
        status = _number(event.get("error_status"))
        attempt = self._retries
        delay_ms = _number(event.get("retry_delay_ms")) or 0.0
        error = str(event.get("error") or "")
        self._backend.last_error = f"retry {attempt}: {error or status or 'error'}"
        code = int(status) if status is not None else None
        if code in (401, 403) or classify(error) == "auth":
            raise _Unauthorised(error or f"HTTP {code}")
        if code == 429 or classify(error) == "limit":
            self._backend.note_limit(self.account, "*", None, error or "HTTP 429")
            raise _Limited()
        # Anything else the CLI would retry is the service not answering:
        # a 5xx, a 529 overload, a connection with no status at all.
        if attempt >= API_RETRIES_TOLERATED or delay_ms >= LONG_RETRY_MS:
            what = error or (f"HTTP {code}" if code else "connection error")
            raise _Degraded(f"{what}, {attempt} retr{'y' if attempt == 1 else 'ies'} "
                            f"in, next wait {delay_ms / 1000:.0f}s")

    def _failed_result(self, event: dict) -> None:
        """A `result` with `is_error`: the CLI gave up, and says why."""
        text = str(event.get("result") or "")
        errors = event.get("errors")
        if isinstance(errors, list) and errors:
            text = " ".join(str(e) for e in errors) or text
        kind = classify(f"{event.get('subtype') or ''} {text}")
        if kind == "auth":
            raise _Unauthorised(text)
        if kind == "limit":
            self._backend.note_limit(self.account, "*", None, text)
            raise _Limited()
        if kind == "server":
            raise _Degraded(text)

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

    And the ladder. `accounts` are the sign-ins, primary first; `profiles` are
    the (model, effort) pairs, the default first and the emergency profiles
    after it. The pairs of the two, profile-major, are the order in which a
    solve is offered around; the outage table is what it skips.
    """

    def __init__(
        self,
        models: Optional[tuple[str, ...]] = None,
        effort: Optional[str] = None,
        concurrency: Optional[int] = None,
        emergency: Optional[tuple[Profile, ...]] = None,
    ) -> None:
        self.binary = _flag("SOLVER_CLI_BIN", "claude") or "claude"
        self.models = models or cli_models()
        self.effort = effort or cli_effort()
        self.env = child_env()
        self.accounts = _accounts(self.env)
        self.default = Profile(self.models[0], self.effort)
        # Second-opinion models the operator named, at the default effort.
        self.opinions = tuple(Profile(m, self.effort) for m in self.models[1:])
        self.emergency = emergency if emergency is not None else cli_emergency_profiles(self.effort)
        profiles: list[Profile] = [self.default]
        for profile in (*self.emergency, *self.opinions):
            if profile not in profiles:
                profiles.append(profile)
        self.profiles = tuple(profiles)
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
        self.recovery_s = float(_flag("SOLVER_CLI_RECOVERY_S", "") or DEFAULT_RECOVERY_S)
        # Whether a turn may run on the plan's paid EXTRA usage once the
        # subscription's own window is spent. Off by default, for the reason
        # the API key is: extra usage is metered billing, and this backend's
        # whole premise is a seat that is already paid for.
        self.allow_overage = _on("SOLVER_CLI_ALLOW_OVERAGE", False)
        self._opened = 0
        self._live = 0
        self._turns = 0
        self._stalls = 0
        self._hops = 0
        self._cost = 0.0
        self._tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.last_error: Optional[str] = None
        # The outage table. Keyed (account name or "*", model or "*"): what is
        # known not to answer, until when, and why.
        self._out: dict[tuple[str, str], _Outage] = {}
        # Consecutive unexplained failures per (account, model).
        self._failures: dict[tuple[str, str], int] = {}
        # The pair a fresh solve was last handed, so a change is said once.
        self._mode: tuple[str, str] = (self.accounts[0].name, self.default.model)
        # Per (account, window), the utilisation last warned about.
        self._warned: dict[tuple[str, str], float] = {}
        self._drill()

    def _drill(self) -> None:
        """`SOLVER_CLI_DRILL`: start with something pretended out, to watch the
        ladder move on real traffic without waiting for a real outage.

        `limit:<account>` pretends that account is at its usage limit;
        `refuse:<model>` pretends the service is refusing that model. Comma-
        separated, read once at launch, and every entry expires on its own
        (five minutes for a limit, the recovery window for a model) so a drill
        left in `.env` cannot quietly become the configuration. Said loudly at
        launch for the same reason.
        """
        raw = _flag("SOLVER_CLI_DRILL", "")
        if not raw:
            return
        by_name = {a.name: a for a in self.accounts}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            kind, _, what = entry.partition(":")
            kind, what = kind.strip().lower(), what.strip()
            if kind == "limit" and what in by_name:
                print(f"[cli] DRILL: pretending account {what} is at its usage limit")
                self.note_limit(by_name[what], "*", time.time() + PAIR_HOLD_S, "drill")
            elif kind == "refuse" and what:
                print(f"[cli] DRILL: pretending the service refuses {what}")
                self.note_degraded(what, "drill")
            else:
                raise SystemExit(
                    f"SOLVER_CLI_DRILL entry {entry!r}: expected limit:<account> "
                    f"(one of {', '.join(by_name)}) or refuse:<model>"
                )

    # -- the ladder --------------------------------------------------------- #

    def provider_of(self, account: Account, profile: Profile) -> str:
        """`cli:<model>` with one account, `cli:<model>@<account>` with more.

        The account is named only when there is a choice of them, so a
        single-seat log reads as it always did and a two-seat log says which
        seat answered -- the question an operator with two seats has.
        """
        if len(self.accounts) == 1:
            return f"cli:{profile.model}"
        return f"cli:{profile.model}@{account.name}"

    def _parse_provider(self, provider: str) -> tuple[Optional[str], Optional[str]]:
        """(model, account name) named by a provider string, either may be None."""
        if not provider.startswith("cli:"):
            return None, None
        rest = provider[4:]
        model, _, account = rest.partition("@")
        return (model or None), (account or None)

    def pairs(self) -> list[tuple[Account, Profile]]:
        """Every (account, profile), in the order a solve is offered around.

        Profile-major: the default model on every account before any
        emergency profile on any. A usage limit is an account's and a refusal
        is a model's, so for those two the order makes no difference; it
        decides the third case, a turn that failed for no stated reason,
        and there the cheaper move -- same model, other seat -- comes first.
        """
        return [(a, p) for p in self.profiles for a in self.accounts]

    def outage_for(self, account: Account, model: str) -> tuple[float, str]:
        """(seconds this pair is known to be out, why). (0, "") if it is not."""
        now = time.time()
        worst: tuple[float, str] = (0.0, "")
        for key, outage in list(self._out.items()):
            if outage.until <= now:
                del self._out[key]
                continue
            scope_account, scope_model = key
            if scope_account not in ("*", account.name):
                continue
            # Loosely on the model: the CLI's windows say `opus`, the operator
            # may have written `claude-opus-5`, and they are the same thing.
            if scope_model != "*":
                if not scope_model or not model:
                    continue
                if scope_model not in model and model not in scope_model:
                    continue
            if outage.until - now > worst[0]:
                worst = (outage.until - now, outage.reason)
        return worst

    def limited_for(self, model: Optional[str] = None, account: Optional[Account] = None) -> float:
        """Seconds for which `model` on `account` is known to be turned away."""
        account = account or self.accounts[0]
        return self.outage_for(account, model or self.default.model)[0]

    def _healthy(self) -> list[tuple[Account, Profile]]:
        return [(a, p) for a, p in self.pairs() if self.outage_for(a, p.model)[0] <= 0]

    def next_pair(
        self, account: Account, model: str, same_account: bool
    ) -> Optional[tuple[Account, Profile]]:
        """The next healthy pair after (account, model), for a turn to hop to."""
        for candidate, profile in self._healthy():
            if candidate == account and profile.model == model:
                continue
            if same_account and candidate != account:
                continue
            return candidate, profile
        return None

    def pick(self, avoid: Optional[str] = None) -> tuple[Account, Profile]:
        """Who answers a fresh conversation.

        Without `avoid`: the first healthy pair on the ladder, which is the
        default pair whenever nothing is wrong. With it, what `avoid` names
        decides what "different" means. `VerifyingSolver` asks for something
        other than `avoid` in two situations it cannot tell apart from here
        but the table can: the avoided pair FAILED, in which case any healthy
        pair is the answer and the same model on another seat is the best of
        them; or it answered WRONGLY and a second opinion is wanted, in which
        case only a different model is one.
        """
        healthy = self._healthy()
        if avoid:
            model, name = self._parse_provider(avoid)
            avoided = None
            for candidate, profile in self.pairs():
                if profile.model == model and (name is None or candidate.name == name):
                    avoided = (candidate, profile)
                    break
            # Failed: the pair is out. The ask is then for someone who WORKS,
            # and the best of those is the first healthy rung. (An unexplained
            # failure that has not yet parked the pair is NOT read this way:
            # `_failures` is shared by every conversation in flight, and one
            # conversation's lost session must not turn another's request
            # for a second opinion into a retry on the very pair it named.)
            failed = avoided is None or self.outage_for(avoided[0], avoided[1].model)[0] > 0
            if not failed:
                others = [(a, p) for a, p in healthy if p.model != model]
                # The operator's own second-opinion profiles -- at the default
                # effort -- before the emergency ladder, when both are healthy.
                others.sort(key=lambda ap: ap[1] not in self.opinions)
                if others:
                    return others[0]
        if healthy:
            return healthy[0]
        # Nothing on the ladder can answer. The default pair is handed out
        # anyway; `send` turns it away with the reason, in a millisecond.
        return self.pairs()[0]

    def _announce(self, account: Account, profile: Profile) -> None:
        """Say when a fresh solve is not going to the default pair, once."""
        pair = (account.name, profile.model)
        top = (self.accounts[0].name, self.default.model)
        if pair == self._mode:
            return
        self._mode = pair
        label = self.provider_of(account, profile) + f" (effort {profile.effort})"
        if pair == top:
            print(f"[cli] back to normal: {label} answers again")
            return
        wait, why = self.outage_for(self.accounts[0], self.default.model)
        print(f"[cli] EMERGENCY MODE: {label} is answering because "
              f"{self.provider_of(self.accounts[0], self.default)} is out "
              f"({why or 'no healthy pair'}; {_minutes(wait)} until it is tried again)")

    # -- the Backend protocol ---------------------------------------------- #

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
        for account in self.accounts:
            status = await self.auth_status(account)
            if not status.get("loggedIn"):
                if account is self.accounts[0]:
                    raise RuntimeError(
                        f"{binary} is installed but not signed in. Run `claude "
                        f"auth login` as the user this miner runs as."
                    )
                # A backup that is not there is not fatal -- the primary
                # answers -- but it is not a backup either, and the operator
                # should hear that now rather than at the limit.
                self.note_unauthorised(account, "not signed in at launch")
                print(f"[cli] WARNING: backup account {account.name!r} "
                      f"({account.config_dir}) is not signed in; run "
                      f"`{account.login_command}`. It is skipped until it is.")
                continue
            method = str(status.get("authMethod") or "?")
            # Said out loud because it is the whole point of this backend, and
            # because the failure it warns about is invisible: an API key
            # answers every solve just as well and bills for every one of them.
            billing = (
                "subscription (OAuth)" if method == "oauth_token"
                else f"authMethod={method} — NOT a subscription; this bills per token"
            )
            print(f"[cli] account {account.name}: {billing}")
        ladder = ", ".join(p.label for p in self.profiles)
        print(f"[cli] {binary} ready: {len(self.accounts)} account(s), ladder "
              f"{ladder}, {self._limit} at a time; a refused model is tried "
              f"again after {_minutes(self.recovery_s)}")

    async def auth_status(self, account: Account) -> dict:
        binary = shutil.which(self.binary) or self.binary
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, "auth", "status",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=account.env,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            status = json.loads(out.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - an older CLI may not print JSON
            if proc is not None and proc.returncode is None:
                # A hung check must not outlive the question.
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
            return {}
        return status if isinstance(status, dict) else {}

    async def open(
        self, avoid: Optional[str] = None, timeout_s: Optional[float] = None
    ) -> CliConversation:
        """A fresh session on the best pair the ladder has. See `pick`."""
        account, profile = self.pick(avoid)
        if avoid is None:
            self._announce(account, profile)
        self._opened += 1
        self._live += 1
        return CliConversation(self, account, profile)

    async def open_profile(self, model: str, effort: str) -> CliConversation:
        """A fresh session on a NAMED model and effort, where the ladder allows.

        For the second reading and the judge, which want a particular model
        rather than the best available one. The first signed-in account on
        which that model is not out gets it; if the model is out everywhere,
        the ladder's own choice answers instead, and says so through
        `provider`, so the caller can tell.
        """
        wanted = Profile(model, effort if effort in EFFORTS else self.effort)
        for account in self.accounts:
            if self.outage_for(account, wanted.model)[0] <= 0:
                self._opened += 1
                self._live += 1
                return CliConversation(self, account, wanted)
        return await self.open(avoid=f"cli:{model}")

    def release(self) -> None:
        self._live = max(0, self._live - 1)

    # -- what turns report --------------------------------------------------- #

    def note_result(self, event: dict) -> None:
        self._turns += 1
        try:
            self._cost += float(event.get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        # Tokens, as the CLI counts them: what a solve costs the seat, in the
        # unit the seat's windows are measured in.
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            for ours, theirs in (("input", "input_tokens"), ("output", "output_tokens"),
                                 ("cache_read", "cache_read_input_tokens"),
                                 ("cache_write", "cache_creation_input_tokens")):
                value = _number(usage.get(theirs))
                if value:
                    self._tokens[ours] += int(value)
        if event.get("is_error"):
            self.last_error = str(event.get("result") or event.get("subtype") or "error")

    def note_hop(self) -> None:
        self._hops += 1

    def note_success(self, account: Account, model: str) -> None:
        self._failures.pop((account.name, model), None)

    def note_failure(self, account: Account, model: str) -> None:
        """An unexplained failure. Twice running, and the pair is set aside."""
        key = (account.name, model)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= FAILURES_BEFORE_HOLD:
            self._failures.pop(key, None)
            self._set(key, time.time() + PAIR_HOLD_S,
                      f"{FAILURES_BEFORE_HOLD} failures running: {self.last_error or '?'}")

    def note_stall(self, account: Account, model: str) -> None:
        self._stalls += 1
        self._set((account.name, model), time.time() + PAIR_HOLD_S,
                  "no event at all; wedged")

    def note_degraded(self, model: str, reason: str) -> None:
        """The service is failing this model. Every account, ten minutes."""
        self.last_error = f"{model} refused: {reason}"
        self._set(("*", model), time.time() + self.recovery_s, f"refused: {reason}")

    def note_unauthorised(self, account: Account, reason: str) -> None:
        self.last_error = f"{account.name} refused: {reason}"
        self._set((account.name, "*"), time.time() + AUTH_HOLD_S, f"signed out: {reason}")

    def note_limit(
        self, account: Account, scope: str, resets_at: Optional[float], reason: str
    ) -> None:
        """The subscription limit on `account`, for `scope` ("*" or a model)."""
        now = time.time()
        until = resets_at if resets_at is not None and resets_at > now else now + 300.0
        until = min(until, now + LIMIT_RECHECK_S)
        self.last_error = f"subscription limit reached ({reason})"
        self._set((account.name, scope), until, f"limit: {reason}",
                  resets=_resets_in(resets_at))

    def _set(self, key: tuple[str, str], until: float, reason: str, resets: str = "") -> None:
        """Record an outage, and say so once per outage rather than per turn:
        the same report arriving on the next turn lands a few seconds later
        and must not read as news."""
        current = self._out.get(key)
        if current is not None and current.until > time.time() and until <= current.until + 60.0:
            return
        self._out[key] = _Outage(until, reason)
        account, model = key
        who = ("every account" if account == "*" else f"account {account}") + \
              (", every model" if model == "*" else f", model {model}")
        print(f"[cli] OUT for {_minutes(until - time.time())}: {who} -- "
              f"{reason}{resets}")

    def note_rate_limit(
        self, info: dict, model: Optional[str] = None, account: Optional[Account] = None
    ) -> bool:
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
        account = account or self.accounts[0]
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
                window_scope = _MODEL_WINDOWS.get(name, "*")
                # A spent window for the whole seat widens a limit that was
                # reported against one model: the seat is out either way.
                if not limited or window_scope == "*":
                    limited, scope = True, window_scope
                if reset is not None and (resets_at is None or reset < resets_at):
                    resets_at = reset
            # Said once per tenth of the budget from 80% up, per window and
            # account, not once per turn: the point is a line the operator
            # sees coming, not one that drowns the log as it does.
            warned = (account.name, name)
            if used >= 0.8 and used >= self._warned.get(warned, 0.0) + 0.1:
                self._warned[warned] = used
                print(f"[cli] NOTE: {used:.0%} of the {name} subscription limit "
                      f"used on account {account.name}{_resets_in(reset)}. Past "
                      f"it that account's solves move to the next seat, or fail.")
        # A turn the subscription's window no longer covers, running on the
        # plan's paid extra usage. Allowed through only on request; otherwise
        # it is the limit, and treated exactly as one -- for the whole seat,
        # whatever window the event named, because extra usage is the seat's:
        # a hop to another model on it would start a metered request.
        if overage and not self.allow_overage:
            limited, scope = True, "*"
            reason = ("the subscription window is spent and extra usage is "
                      "not enabled")
        else:
            reason = f"status {status}" + (f" on {kind}" if kind else "")
        if overage and self.allow_overage and not self._warned.get((account.name, "overage")):
            self._warned[(account.name, "overage")] = 1.0
            print(f"[cli] NOTE: account {account.name}'s window is spent; its "
                  f"turns now run on the plan's paid extra usage "
                  f"(SOLVER_CLI_ALLOW_OVERAGE)")
        if not limited:
            return False
        self.note_limit(account, scope, resets_at, reason)
        return scope == "*" or (model is not None and scope in model)

    def stats(self) -> dict[str, Any]:
        now = time.time()
        return {
            "backend": "claude-cli",
            "accounts": [a.name for a in self.accounts],
            "default": self.default.label,
            "ladder": [p.label for p in self.profiles],
            "concurrency": self._limit,
            "sessions_opened": self._opened,
            "sessions_live": self._live,
            "turns": self._turns,
            "hops": self._hops,
            "stalls": self._stalls,
            "out": {
                f"{account}/{model}": {"seconds": round(o.until - now), "why": o.reason}
                for (account, model), o in self._out.items() if o.until > now
            },
            # List price for the tokens spent. On a subscription nothing is
            # billed this way -- it is here as a measure of how much work went
            # through the seat, not as an invoice.
            "list_price_usd": round(self._cost, 4),
            "tokens": dict(self._tokens),
        }

    async def aclose(self) -> None:
        return None


# -- operator commands ------------------------------------------------------ #

def main(argv: Optional[list[str]] = None) -> int:
    """`python -m solvers.claude_cli status|login DIR`.

    `status` says which accounts the ladder has and whether each is signed
    in. `login DIR` is how a backup account is made: it runs the CLI's own
    interactive sign-in with `CLAUDE_CONFIG_DIR` pointed at DIR, so the second
    subscription's credentials land in a directory of their own and the first
    account's are untouched. Sign in with the OTHER account when the browser
    opens.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "status"
    if command == "login":
        if len(args) != 2:
            print("usage: python -m solvers.claude_cli login <config-dir>", file=sys.stderr)
            return 2
        directory = os.path.expanduser(args[1])
        os.makedirs(directory, exist_ok=True)
        env = child_env()
        env["CLAUDE_CONFIG_DIR"] = directory
        binary = shutil.which(_flag("SOLVER_CLI_BIN", "claude") or "claude")
        if binary is None:
            print("no `claude` on PATH; install Claude Code first", file=sys.stderr)
            return 1
        print(f"signing in the account for {directory}; use the OTHER "
              f"subscription when the browser asks. Then set\n"
              f"  SOLVER_CLI_BACKUP_ACCOUNTS={directory}\nin .env.")
        os.execvpe(binary, [binary, "auth", "login"], env)
    if command != "status":
        print("usage: python -m solvers.claude_cli [status|login <config-dir>]",
              file=sys.stderr)
        return 2

    async def report() -> int:
        backend = CliBackend()
        worst = 0
        for account in backend.accounts:
            status = await backend.auth_status(account)
            where = account.config_dir or "default login"
            if status.get("loggedIn"):
                method = status.get("authMethod")
                seat = "subscription" if method == "oauth_token" else f"authMethod={method}"
                # Who, when the CLI says: two directories signed in as the
                # SAME account share one limit, and this is where to see it.
                who = " ".join(str(status[k]) for k in ("email", "subscriptionType")
                               if status.get(k))
                print(f"  {account.name:<12} {where}: signed in ({seat})"
                      + (f" as {who}" if who else ""))
            else:
                worst = 1
                print(f"  {account.name:<12} {where}: NOT signed in -> "
                      f"{account.login_command}")
        print(f"  ladder: {', '.join(p.label for p in backend.profiles)}")
        return worst

    return asyncio.run(report())


if __name__ == "__main__":
    sys.exit(main())

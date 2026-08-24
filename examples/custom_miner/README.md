# Run your own application as a miner

The stock demo miner (`rlvr/neurons/demo_miner.py`) answers tasks with GLM-5.2.
`custom_miner.py` keeps every piece of that miner that talks to the subnet —
signature verification, the replay-nonce cache, validator-permit authorization,
response signing, and the byte/concurrency limits — and replaces **only** the
part that produces an answer, so your own solver plugs in without you
re-implementing the wire protocol.

> **Linux only.** Every entrypoint here refuses to start on anything else, and
> says why. See [Linux only, and why](#linux-only-and-why) — on Windows, WSL2 is
> the intended path and counts as Linux.

## What the validator actually requires

A validator sends a signed `POST /solve` and accepts the reply only if it is
HTTP 200, signed by your miner hotkey and bound to the calling validator within
the freshness window, and a `SolutionPayload` whose `problem_id` echoes the
request's and whose `code` field holds runnable source, under the response byte
cap. `custom_miner.py` guarantees all of that; you only supply the `code`.

## Plug in your app

### Option 1 — HTTP (any language)

Run your application as its own service and point the miner at it:

```dotenv
# .env  (in addition to the miner settings below)
MY_APP_URL=http://127.0.0.1:9000/solve
```

Your service receives:

```json
POST /solve
{"problem_id": "...", "language": "python", "statement": "...",
 "entrypoint": "f", "public_examples": [{"args": [...], "kwargs": {}, "expected": ...}],
 "deadline_s": 300.0}
```

and must return:

```json
200
{"code": "def f(x):\n    return x * 2\n", "raw_response": "optional transcript"}
```

### Option 2 — in-process (Python)

```python
from custom_miner import run_custom_miner, SolveTask, SolveResult

class MySolver:
    async def solve_task(self, task: SolveTask, timeout_s: float) -> SolveResult:
        code = await my_agent(task.statement, task.entrypoint, task.language)
        return SolveResult(code=code)
    async def aclose(self): ...

run_custom_miner(MySolver())
```

## Run

```bash
# Debian/Ubuntu — Python 3.12, the build tools the chain wheels expect, and
# (for the browser backends) a virtual screen to log in on.
sudo apt-get update && sudo apt-get install -y \
    python3.12 python3.12-venv build-essential pkg-config libssl-dev \
    xvfb x11vnc

python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[chain,miner]'
cp .env.example .env        # set NETUID, SUBTENSOR_NETWORK, WALLET_NAME,
                            # WALLET_HOTKEY, AXON_PORT, AXON_EXTERNAL_IP, MY_APP_URL
python examples/custom_miner/custom_miner.py
```

Register the hotkey first (see `scripts/register_testnet.sh`), open the axon
port to the internet, and confirm `curl http://127.0.0.1:$AXON_PORT/health`.

### Linux only, and why

`custom_miner.py`, `run_miner.py`, `run_chatgpt_miner.py` and the doctor all
call `require_linux()` before doing anything, so a wrong platform costs one
clear line instead of a build failure three layers down.

**Windows cannot run this at all.** `bittensor-wallet` and `bittensor-drand`
publish manylinux and macOS wheels only — no Windows wheel of any version — so
`pip install '.[chain]'` falls back to compiling them through a Rust toolchain,
and that is where a Windows install dies. Nothing here can route around it: the
miner has to sign with the hotkey, and the hotkey lives in `bittensor-wallet`.
**Use WSL2**; `sys.platform` there is `linux` and everything below applies
unchanged. Install into the WSL filesystem, not `/mnt/c`, or file I/O will
crawl.

**macOS** can install the chain dependencies but is refused here anyway. A
miner is a long-lived server that has to answer inside a deadline around the
clock, and the shape of that — systemd, Firefox under Xvfb, a firewall in front
of the axon port — is Linux shaped. Half-working on a laptop is worse than a
clear no.

Tested against x86-64 glibc 2.28+ (Ubuntu 22.04/24.04, Debian 12, Rocky 9).

## Rules your solver must honor

The subnet grades on the **complete hidden test suite** with only a small
latency tiebreaker, so a partially-correct, late, or empty answer earns zero.

- **Python:** `code` must define the function named `task.entrypoint`, standard
  library only, no I/O or example handling.
- **Rust** (`task.language == "rust"`): `code` must be a complete program with
  `fn main()` that reads stdin and writes only the answer to stdout;
  `entrypoint` is always `"main"`. Output is compared token-by-token on ASCII
  whitespace. Support this only if you intend to solve Rust challenges;
  otherwise return empty `code` for them.
- **Never raise.** On any failure return empty `code` — a zero is survivable, a
  crash loop is not. `custom_miner.py` already wraps your solver this way.

## Backends: Claude and ChatGPT, both from the browser

Two backends ship, behind the same `open()` → `send()`/`close()` protocol, both
feeding the same self-verify-and-repair loop. Pick with `MINER_BACKENDS`:

```bash
MINER_BACKENDS=claude           python examples/custom_miner/run_miner.py
MINER_BACKENDS=claude,chatgpt   python examples/custom_miner/run_miner.py
```

| Backend | Credentials | Quota you spend |
|---|---|---|
| `claude` | a logged-in Firefox profile in `CLAUDE_PROFILES` | your Claude subscription |
| `chatgpt` | a logged-in Firefox profile in `CHATGPT_PROFILES` | your ChatGPT subscription |

Both drive **Firefox**, launched by Playwright against a profile directory you
signed in to once. **No API key is read anywhere in this package** — there is no
API path at all, which is a deliberate choice with a real cost attached: see the
risks below, and run two providers if you can.

`run_chatgpt_miner.py` still exists and is equivalent to `MINER_BACKENDS=chatgpt`.

### More than one backend is a fallback chain — and your only redundancy

With both names, providers run **in order and stop at the first answer that
reproduces every public example**. Ordering is a cost decision, not a quality
one: a verified answer ends the chain, so the second provider costs nothing on
tasks the first one solves. Put the account you would rather spend first.

It is also the only redundancy a browser miner has. With no API backend to fall
back to, a second logged-in provider is what stands between one expired login
and a run of zeros — and the two do not share a DOM, a login, or a rate limit.

This is worth doing because of how the subnet pays. The latency tiebreaker spans
about 3.5%; being wrong costs 100%. A second opinion is therefore worth far more
than a faster first one. If no provider verifies, the best partial answer across
all of them is still returned — never nothing when some provider produced
runnable code.

Budget is split across the remaining providers rather than handed to the first,
so a slow leader cannot starve the rest, and the chain stops early when too
little time remains for another provider to be useful.

| Variable | Default | Meaning |
|---|---|---|
| `MINER_BACKENDS` | `claude` | Comma-separated backends, in preference order |
| `CLAUDE_PROFILES` | `~/.hone-miner/firefox/claude-1` | Profile dirs, one per account |
| `CLAUDE_TABS_PER_PROFILE` | `2` | Conversation slots per profile |
| `CLAUDE_HEADLESS` | `true` | `false` shows the windows (needs a display) |
| `SOLVER_MAX_ATTEMPTS` | `3` | Repair rounds per provider |

These are read from the process environment **and** from `.env` — `run_miner.py`
loads that file into the environment itself, because the miner's own settings go
through pydantic-settings, which reads `.env` directly and leaves `os.environ`
untouched. A shell variable still wins over the file.

`GET /solver-status` reports the chain, which provider verified each task, and
per-provider turn and error counts — watch it, because a provider that starts
failing looks exactly like success until the score drops.

## Running Claude from the browser

`claude` drives Firefox against a profile directory you signed in to once. No
API key is involved; the quota is whatever your Claude plan gives you.

```bash
pip install playwright
python -m playwright install firefox     # a second, separate download

cd examples/custom_miner
python -m solvers.login claude           # sign in; see below if there is no screen
python -m solvers.doctor claude --probe
MINER_BACKENDS=claude python run_miner.py
```

### Why Playwright owns the browser

An earlier version attached over CDP to a Chrome you started yourself. Firefox
cannot do that — Playwright's `connect_over_cdp` is Chromium-only, and Mozilla
removed its CDP implementation in favour of WebDriver BiDi. So Playwright
launches Firefox itself against a persistent profile.

That is a better shape for a miner anyway: **one process to supervise instead of
two**, no debugging port to leave exposed, and a crash restarts already logged
in. It comes with exactly one rule:

> **A profile directory can be open in one process at a time.** Stop the miner
> before running the login helper or the doctor, and stop those before starting
> the miner. All three say so by name when it happens rather than crashing.

### Logging in when the box has no screen

`scripts/login.sh` starts a virtual screen (Xvfb), shares it over VNC bound to
loopback, opens the login window on it, and prints the tunnel command:

```bash
sudo apt-get install -y xvfb x11vnc
./scripts/login.sh claude
# then from your own machine:
ssh -N -L 5900:127.0.0.1:5900 you@your-miner
# point a VNC client at 127.0.0.1:5900, sign in, press Enter in the terminal
```

That VNC screen is an unauthenticated view of a browser you are about to type a
password into, on a box already exposing a public axon port — hence `-localhost`
and the tunnel. Never publish the port.

**Or skip it entirely**: log in on any Linux desktop and copy the directory.

```bash
rsync -a ~/.hone-miner/firefox/claude-1/ you@your-miner:~/.hone-miner/firefox/claude-1/
```

The login helper does not take your word for it — after you press Enter it
reloads the site and checks the composer actually appears, so a half-finished
sign-in fails now rather than as a run of zeros later.

### Accounts, profiles and tabs

The account is the rate-limit unit: one profile per Claude account,
`CLAUDE_TABS_PER_PROFILE` conversation slots inside each, and at least
`MINER_MAX_CONCURRENT_REQUESTS` tabs in total or extra tasks queue and burn
their deadline. The launcher warns when it is short.

```dotenv
CLAUDE_PROFILES=~/.hone-miner/firefox/claude-1,~/.hone-miner/firefox/claude-2
CLAUDE_TABS_PER_PROFILE=2
```

**Headless by default.** Set `CLAUDE_HEADLESS=false` and run the miner under
`xvfb-run` if a provider starts challenging the headless browser; that trades
some memory for looking more like an ordinary session.

**Keeping it alive.** One systemd service with `Restart=always` is enough now —
the browser is a child of the miner, and the profile brings the login back.

### Run the doctor before you point a hotkey at it

claude.ai's markup is not a published interface, so the selectors shipped here
are candidate lists, not verified facts — and a browser miner fails *silently*:
a DOM change looks exactly like an idle miner, and by the time the score drops
the zeros are already inside the 200-observation window (~2.1 days).

```
$ python -m solvers.doctor claude --probe
```

reports which candidate your page actually has for each role, flags the three
mistakes that matter, and then drives the real read path with a trivial prompt
so you see exactly what the miner would see. Every role is overridable in
`.env`, with `|` between candidates (`,` is already CSS's own "either"):

```dotenv
CLAUDE_COMPOSER='div[contenteditable="true"].ProseMirror'
CLAUDE_ASSISTANT='div[data-is-streaming]'
CLAUDE_SEND_BUTTON='button[aria-label="Send message"]'
CLAUDE_STOP_BUTTON='button[aria-label="Stop response"]'
```

Three hazards are handled in code rather than left to the selectors:

- **An assistant selector that also matches your own message** would make the
  miner hand its own prompt back as the answer — no error, no empty reply, just
  a permanent zero. Any reply that starts with the prompt just sent is refused,
  and the log names the doctor.
- **Artifacts.** Long code can land in the side panel, outside the message the
  reader scrapes, so every prompt asks for an inline code block
  (`CLAUDE_NUDGE` overrides the wording).
- **A "still generating" selector that is always true** would make every answer
  look unfinished and burn the whole budget. Each candidate is checked against
  a freshly-loaded idle page at startup, and any that matches is dropped.

The same doctor works for ChatGPT: `python -m solvers.doctor chatgpt`.

## Included backend: ChatGPT in Firefox + self-verification

`run_chatgpt_miner.py` wires up a ready-made solver built from
[fnlich/Automation](https://github.com/fnlich/Automation)'s browser driver:

```
POST /solve -> fresh ChatGPT conversation -> self-grade against the public
examples with the validator's own executor -> repair on failure -> signed reply
```

### Why the self-verification matters

Scoring is accuracy-or-nothing, and models routinely produce *nearly* right
answers. But every task ships real `public_examples`, and the comparators the
validator will judge you with live in this repository
(`rlvr/execution/compare.py`, `rlvr/execution/rust_judge.py`). So the miner
grades its own candidate with the validator's executor before answering, and on
failure hands the model the concrete evidence:

```
Your solution is WRONG. I ran `sum_of_digits` against the examples and got:
  - sum_of_digits(*[12345], **{}) returned 14, expected 15
  - sum_of_digits(*[999], **{}) returned 18, expected 27
```

That turns a one-shot paste into a repair loop that converges. Passing the
public examples is not proof of passing the hidden suite, but it eliminates the
large class of answers that are simply wrong on the stated contract.

### Setup — one browser per ChatGPT account

Accounts are the rate-limit unit, so N accounts give N× throughput (the same
insight as `run_parallel.py`):

```bash
# Same setup as Claude above; one profile per account.
python -m solvers.login chatgpt --profile ~/.hone-miner/firefox/chatgpt-1
python -m solvers.login chatgpt --profile ~/.hone-miner/firefox/chatgpt-2
python -m solvers.doctor chatgpt --probe
CHATGPT_PROFILES=~/.hone-miner/firefox/chatgpt-1,~/.hone-miner/firefox/chatgpt-2 \
    python run_chatgpt_miner.py
```

Keep the tab count at or above `MINER_MAX_CONCURRENT_REQUESTS`, or extra tasks
queue and burn their deadline. The launcher warns when it is short.

| Variable | Default | Meaning |
|---|---|---|
| `CHATGPT_PROFILES` | `~/.hone-miner/firefox/chatgpt-1` | Profile dirs, one per account |
| `CHATGPT_TABS_PER_PROFILE` | `2` | Conversation slots per profile |
| `CHATGPT_HEADLESS` | `true` | `false` shows the windows |
| `SOLVER_MAX_ATTEMPTS` | `3` | Initial answer + repair rounds |
| `SOLVER_SAFETY_MARGIN_S` | `15` | Headroom kept before the cutoff |
| `SOLVER_MAX_BUDGET_S` | `240` | Hard cap on one solve |
| `SOLVER_VERIFY_EXECUTOR` | `subprocess` | `docker` also verifies Rust |

`GET /solver-status` reports pool health and solve counts. Watch it — a
browser-backed miner fails quietly, and silence looks identical to success.

### Know the risks before running either browser backend for money

- **Terms of service.** Driving a consumer chat UI programmatically to power a
  paid service is very likely against the provider's terms — OpenAI's prohibit
  automated extraction of Output, and Anthropic's usage policies and Claude.ai
  terms similarly do not contemplate scripted access to the web app in place of
  the API. The realistic downside is account termination, and it applies to
  `claude` and `chatgpt` equally. The supported way to do this is each
  provider's API; this package does not offer that path.
- **Fragility, with nothing to fall back to.** Firefox updates, DOM changes,
  expired logins, rate limits and CAPTCHAs all break browser automation. A miner
  that does not answer scores zero into a 200-observation window (~2.1 days), so
  one bad night costs most of your score — and there is no `MINER_BACKENDS=...-api`
  to switch to when it happens. Three things stand in for that: run both
  providers so they are not down together, run the doctor before you serve, and
  watch `/solver-status`, because a provider that has started failing looks
  exactly like one that is merely quiet.
- **The `Backend` protocol is three methods.** If you do want an API path,
  `open()` returning something with `send(text, timeout_s)` and `close()` is the
  whole interface; `verify.py` and the fallback chain take it unchanged.
- **Rust.** `SOLVER_VERIFY_EXECUTOR=docker` is required to verify Rust answers;
  without it Rust candidates are returned unverified.

## Testing your setup

Four layers, cheapest first, each isolating a different failure:

```bash
pytest examples/custom_miner        # 1. code only — no browser, no chain
python -m solvers.doctor claude --probe   # 2. profile, login and selectors
python scripts/try_solver.py        # 3. a real solve, end to end, no wallet
                                    # 4. testnet, then finney
```

`scripts/try_solver.py` is the one to reach for when something is wrong. It
builds the solver exactly as the miner does, hands it one task with public
examples, and reports whether the answer reproduced them — while importing
neither bittensor nor `custom_miner`, so it runs before you have a wallet and a
failure there can only be the solver. Its three outcomes are distinct on
purpose: verified (setup is good), code-but-wrong (plumbing works, model
missed), and nothing-came-back (login, selector, or deadline — it names all
three in likelihood order).

```bash
python scripts/try_solver.py --statement "Return n factorial." \
    --entrypoint fact --example '{"args": [5], "expected": 120}'
```

## Tests

```bash
pytest examples/custom_miner
```

Kept out of `tests/` (the validator's own suite) so the default `pytest -q` is
unaffected. They lock in the four validator acceptance checks on a real signed
reply, that the verify loop repairs a wrong answer, that a solve never outruns
its budget and never returns nothing when it has something, that a browser tab
which dies is retired rather than recycled into the pool (for both browser
backends — they share one pool), and that a reply echoing the prompt is refused.

They also pin the platform guard: every entrypoint refuses a non-Linux host
*before* the project imports that would otherwise fail first with a
`ModuleNotFoundError` explaining nothing.

What they cannot test is the selectors themselves, because there is no browser
in CI. That is what `python -m solvers.doctor <backend> --probe` is for, and it
is not optional before a browser backend serves a registered hotkey.

## Two caveats worth knowing

- **Trust the request's `deadline_s` cautiously.** The validator advertises one
  deadline in the request but enforces the *problem server's* deadline as the
  real cutoff, and the two are not guaranteed to match. Budget your solve
  conservatively rather than spending right up to the advertised value.
- **Keep the permit check on.** With `MINER_REQUIRE_VALIDATOR_PERMIT=true` only
  stake-gated validators can spend your compute. Relaxing it lets any registered
  hotkey call your `/solve` and harvest solutions.

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

| Backend | Signs in as | Quota you spend |
|---|---|---|
| `claude` | you, in a real browser | your Claude subscription |
| `chatgpt` | you, in a real browser | your ChatGPT subscription |

**No API key is read anywhere in this package** — there is no API path at all,
which is a deliberate choice with a real cost attached: see the risks below, and
run two providers if you can.

Each backend gets its browser one of two ways, and **which one you pick is the
single most important setup decision**, so it has its own section:
[Two ways to run the browser](#two-ways-to-run-the-browser-attach-vs-launch).
In one line: if a provider's sign-in refuses an automated browser — Google's
"Couldn't sign you in" is the usual one — use **attach mode**, because a browser
you started yourself is not flagged as automation.

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
| `CLAUDE_CDP` | *(unset)* | **Attach mode**: CDP port(s)/URL(s) of Chrome you started |
| `CLAUDE_PROFILES` | `~/.hone-miner/firefox/claude-1` | **Launch mode**: profile dirs, one per account |
| `CLAUDE_TABS_PER_PROFILE` | `2` | Conversation slots per browser source |
| `CLAUDE_HEADLESS` | `true` | Launch mode only; `false` shows the window (needs a display) |
| `SOLVER_MAX_ATTEMPTS` | `3` | Repair rounds per provider |

Setting `CLAUDE_CDP` selects attach mode; leaving it unset uses launch mode with
`CLAUDE_PROFILES`. `CHATGPT_*` are the exact analogues.

These are read from the process environment **and** from `.env` — `run_miner.py`
loads that file into the environment itself, because the miner's own settings go
through pydantic-settings, which reads `.env` directly and leaves `os.environ`
untouched. A shell variable still wins over the file.

`GET /solver-status` reports the chain, which provider verified each task, and
per-provider turn and error counts — watch it, because a provider that starts
failing looks exactly like success until the score drops.

## Two ways to run the browser: attach vs launch

Each backend needs a logged-in browser. There are two ways to give it one, and
the difference is not cosmetic — it decides whether you can log in at all.

| | **attach mode** (recommended) | **launch mode** (default) |
|---|---|---|
| Browser | Chrome/Chromium **you** start | Firefox Playwright starts |
| Sign-in | you, by hand, in a normal browser | you, by hand, in Playwright's Firefox |
| Looks automated? | **no** — `navigator.webdriver` is false | yes — a Playwright build |
| Google / hard sign-in checks | **pass** | often refused |
| Processes to run | two (browser + miner) | one (miner owns the browser) |
| Config | `CLAUDE_CDP=9222` | `CLAUDE_PROFILES=…` (a default exists) |

**Why attach mode exists.** Providers fingerprint the browser. A
Playwright-launched Firefox is a recognisable automation build, and some
sign-in flows — Google's OAuth most visibly, with *"Couldn't sign you in. This
browser or app may not be secure."* — refuse it outright. A browser **you**
started with a debugging port is not in automation mode: `navigator.webdriver`
is `false` and it is the ordinary browser it appears to be, so the same sign-in
succeeds. The miner then attaches over the Chrome DevTools Protocol (CDP), which
is Chromium-only — Firefox exposes WebDriver BiDi instead and cannot be attached
to, which is the whole reason launch mode uses Firefox and attach mode uses
Chrome.

Both were verified end to end on a real browser before this was written: attach
mode solves a task through a hand-started Chrome, and disconnecting the miner
leaves that Chrome (and its login) running.

### Attach mode, step by step (use this if sign-in is refusing you)

```bash
pip install playwright                    # the Python package
# No `playwright install` is needed for attach mode — you bring your own Chrome.
sudo apt-get install -y chromium xvfb     # a browser, and a virtual screen for headless hosts

cd examples/custom_miner

# 1. Start a real Chrome in debug mode. It stays running; you log in in it.
./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/claude-1

# 2. Log in to https://claude.ai in that browser, by hand. On a headless box,
#    reach its window over VNC — see "Logging in with no screen" below.

# 3. Point .env at it and verify, then run:
echo 'CLAUDE_CDP=9222' >> ../../.env
python -m solvers.doctor claude --probe
MINER_BACKENDS=claude python run_miner.py
```

`start_debug_browser.sh` finds `google-chrome-stable` / `chromium` / `$CHROME_BIN`,
keeps the CDP port on loopback, adds `--no-sandbox` only when you are root,
refuses to double-launch a port already in use, and waits for the port before
telling you it is up. On a headless host it runs Chrome under `xvfb-run`.

On a server, Chrome prints a wall of `Failed to connect to the bus` / D-Bus
errors — those are normal with no desktop session and do **not** mean it failed.
The line that matters is `CDP is up on http://127.0.0.1:9222`.

`CLAUDE_CDP` accepts a bare port (`9222` → `http://127.0.0.1:9222`), a `host:port`,
a full URL, or a comma-separated list for several accounts. Set `CHATGPT_CDP` the
same way for ChatGPT.

**Attach mode's one rule and its one strength:**

- The miner does **not** own the browser. On shutdown it disconnects but never
  closes it, so **restarting the miner keeps your hand-made login** — you log in
  once and rarely again.
- Keep the debug port on loopback. Anyone who reaches it has full control of a
  browser holding your logged-in sessions, on a box already exposing a public
  axon port. `start_debug_browser.sh` never binds it off `127.0.0.1`; reach it
  over an SSH tunnel.

### Launch mode, step by step (simpler, when sign-in is not fussy)

`claude` launches Firefox against a profile directory you signed in to once — one
process, no ports. Use it when the provider does not refuse the automated
browser.

```bash
pip install playwright
python -m playwright install firefox      # a second, separate download

cd examples/custom_miner
python -m solvers.login claude            # opens Firefox; sign in, press Enter
python -m solvers.doctor claude --probe
MINER_BACKENDS=claude python run_miner.py
```

Firefox is used here because Playwright cannot attach to an externally-started
Firefox (Chromium-only CDP), so in launch mode Playwright owns the browser. That
brings one rule: **a profile directory can be open in one process at a time.**
Stop the miner before running the login helper or the doctor, and vice versa —
all three say so by name rather than crashing. The login helper verifies the
session stuck (it reloads and checks the composer appears) so a half-finished
sign-in fails now, not as a run of zeros later.

If Google's sign-in refuses this Firefox, that is exactly what attach mode is
for — switch to it.

### Logging in when the box has no screen

Either mode needs you to type a password into a browser once, and a server has
no display. Two ways, both fine:

**Attach mode** — the debug Chrome runs under Xvfb; view it over VNC.

```bash
sudo apt-get install -y x11vnc
# in another terminal, after start_debug_browser.sh is running under Xvfb:
DISPLAY=:99 x11vnc -display :99 -rfbport 5900 -localhost -nopw -forever &
ssh -N -L 5900:127.0.0.1:5900 you@your-miner    # from your own machine
# point a VNC client at 127.0.0.1:5900, sign in, leave the browser running
```

**Launch mode** — `scripts/login.sh` does the Xvfb+VNC dance for you:

```bash
sudo apt-get install -y xvfb x11vnc
./scripts/login.sh claude
ssh -N -L 5900:127.0.0.1:5900 you@your-miner
# VNC to 127.0.0.1:5900, sign in, press Enter in the terminal
```

That VNC screen is an unauthenticated view of a browser you are about to type a
password into — hence `-localhost` and the tunnel. Never publish the port.

**Or skip screens entirely**: log in on any Linux desktop and copy the profile
over. Launch mode: `rsync -a ~/.hone-miner/firefox/claude-1/
you@host:~/.hone-miner/firefox/claude-1/`. Attach mode: same, with the
`~/.hone-miner/chrome/…` directory `start_debug_browser.sh` created.

### Accounts, profiles and tabs

The account is the rate-limit unit: one browser source per account,
`*_TABS_PER_PROFILE` conversation slots inside each, and at least
`MINER_MAX_CONCURRENT_REQUESTS` tabs in total or extra tasks queue and burn
their deadline. The launcher warns when it is short.

```dotenv
# attach mode: one debug Chrome per account, each on its own port
CLAUDE_CDP=9222,9223
# launch mode: one Firefox profile per account
CLAUDE_PROFILES=~/.hone-miner/firefox/claude-1,~/.hone-miner/firefox/claude-2
CLAUDE_TABS_PER_PROFILE=2
```

**Keeping it alive.** In launch mode, one systemd service with `Restart=always`
is enough — the browser is a child of the miner. In attach mode, run two
services: the debug browser (with the same `--profile`, so it restarts logged
in) and the miner; the miner reconnects on its next start.

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

Accounts are the rate-limit unit, so N accounts give N× throughput. ChatGPT's
own login often routes through Google, so **attach mode is usually the one that
works** here — the same two modes as Claude, chosen with `CHATGPT_CDP` vs
`CHATGPT_PROFILES`:

```bash
# Attach mode (recommended): a debug Chrome per account, logged in by hand.
./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/gpt-1
./scripts/start_debug_browser.sh --port 9223 --profile ~/.hone-miner/chrome/gpt-2
python -m solvers.doctor chatgpt --cdp 9222 --probe
CHATGPT_CDP=9222,9223 python run_chatgpt_miner.py

# Launch mode: one Firefox profile per account (if sign-in is not fussy).
python -m solvers.login chatgpt --profile ~/.hone-miner/firefox/chatgpt-1
CHATGPT_PROFILES=~/.hone-miner/firefox/chatgpt-1 python run_chatgpt_miner.py
```

Keep the tab count at or above `MINER_MAX_CONCURRENT_REQUESTS`, or extra tasks
queue and burn their deadline. The launcher warns when it is short.

| Variable | Default | Meaning |
|---|---|---|
| `CHATGPT_CDP` | *(unset)* | Attach mode: CDP port(s)/URL(s) of Chrome you started |
| `CHATGPT_PROFILES` | `~/.hone-miner/firefox/chatgpt-1` | Launch mode: profile dirs, one per account |
| `CHATGPT_TABS_PER_PROFILE` | `2` | Conversation slots per browser source |
| `CHATGPT_HEADLESS` | `true` | Launch mode only; `false` shows the window |
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
- **Detection is the failure you will hit first.** Providers fingerprint the
  browser, and a launched Firefox is the most detectable option — Google's
  sign-in refuses it outright ("Couldn't sign you in"). Attach mode (a real
  Chrome you started) is the answer to that specific wall, because it is not in
  automation mode. This is not an arms race worth entering: do not spoof user
  agents or patch out automation flags — pick the mode that a given provider's
  login actually accepts, and prefer email/one-time-code sign-in over "Continue
  with Google".
- **Fragility, with nothing to fall back to.** Browser/DOM updates, expired
  logins, rate limits and CAPTCHAs all break browser automation. A miner that
  does not answer scores zero into a 200-observation window (~2.1 days), so one
  bad night costs most of your score — and there is no `MINER_BACKENDS=...-api`
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
python -m solvers.doctor claude --probe   # 2. login + selectors (add --cdp 9222 in attach mode)
python scripts/try_solver.py        # 3. a real solve, end to end, no wallet
                                    # 4. testnet, then finney
```

`try_solver.py` honours the same `.env` as the miner, so it exercises whichever
mode you configured — attach or launch — with no extra flags.

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

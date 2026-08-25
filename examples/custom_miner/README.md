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
# Debian/Ubuntu — Python, the build tools the chain wheels expect, curl, and
# (for the browser backends) a virtual screen to sign in on.
sudo apt-get update && sudo apt-get install -y \
    python3 python3-venv build-essential pkg-config libssl-dev \
    curl xvfb x11vnc

python3 -m venv .venv && . .venv/bin/activate   # needs Python 3.10-3.12
pip install -e '.[chain,miner,dev]'             # dev brings pytest
cp .env.example .env        # set NETUID, SUBTENSOR_NETWORK, WALLET_NAME,
                            # WALLET_HOTKEY, AXON_PORT, AXON_EXTERNAL_IP
echo 'MY_APP_URL=http://127.0.0.1:9000/solve' >> .env   # Option 1 only
python examples/custom_miner/custom_miner.py
```

`requires-python` is `>=3.10,<3.13`. Ubuntu 24.04 ships 3.12; Ubuntu 22.04 and
Debian 12 ship 3.10/3.11, which are fine. Include `dev` in the extras or
`pytest` will not be installed and the bare `pytest` on your PATH will be the
system one, running against a different interpreter with none of these
dependencies.

Register the hotkey first (see `scripts/register_testnet.sh`), open the axon
port to the internet, and confirm the health endpoint. `AXON_PORT` lives in
`.env`, which the shell does not read, so name the port explicitly:

```bash
curl http://127.0.0.1:8091/health          # or whichever AXON_PORT you set
```

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
clock, and the shape of that — systemd, a browser under Xvfb, a firewall in
front of the axon port — is Linux shaped. Half-working on a laptop is worse than a
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

You start the browser and sign in by hand; the miner attaches to it. That is
what gets past sign-in checks which reject automated browsers — Google's
"Couldn't sign you in" being the usual one. See
[Running a backend](#running-a-backend-you-start-the-browser-the-miner-attaches).

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
5 points — `payment_speed_floor` is `0.95`, so the speed multiplier lives in
`[0.95, 1.0]` — while being wrong costs 100%. A second opinion is therefore
worth far more than a faster first one. If no provider verifies, the best partial answer across
all of them is still returned — never nothing when some provider produced
runnable code.

Budget is split across the remaining providers rather than handed to the first,
so a slow leader cannot starve the rest, and the chain stops early when too
little time remains for another provider to be useful.

| Variable | Default | Meaning |
|---|---|---|
| `MINER_BACKENDS` | `claude` | Comma-separated backends, in preference order |
| `CLAUDE_CDP` | `9222` | CDP endpoint(s) of the Chrome you started, one per account |
| `CLAUDE_TABS_PER_BROWSER` | `2` | Conversation slots per browser |
| `SOLVER_MAX_ATTEMPTS` | `3` | Model round-trips per provider: 1 initial answer + 2 repairs |

`CLAUDE_CDP` accepts a bare port (`9222` → `http://127.0.0.1:9222`), a
`host:port`, a full URL, or a comma-separated list for several accounts.
`CHATGPT_*` are the exact analogues.

These are read from the process environment **and** from `.env` — `run_miner.py`
loads that file into the environment itself, because the miner's own settings go
through pydantic-settings, which reads `.env` directly and leaves `os.environ`
untouched. A shell variable still wins over the file.

`GET /solver-status` reports per-provider counters — `solved`, `verified`,
`cache_hits`, `empty`, plus pool health (`tabs`, `idle`, `browsers`, `lost`).
With more than one backend it also reports the `chain`, and `verified_by` /
`attempts` totals per provider; with a single backend those two keys are absent,
because there is no chain to describe. Watch it either way: a provider that has
started failing looks exactly like success until the score drops.

## Running a backend: you start the browser, the miner attaches

Each backend needs a browser that is signed in to the provider. You start it and
sign in **by hand**; the miner attaches to it over the Chrome DevTools Protocol
(CDP) and opens its own tabs. Nothing here launches or closes a browser.

**That division of labour is the whole design, not a limitation.** A browser
launched by an automation driver announces itself as one, and provider sign-in
flows reject it — the visible case is Google's OAuth answering *"Couldn't sign
you in. This browser or app may not be secure."* A browser **you** started is
not in automation mode: `navigator.webdriver` is `false` and it is the ordinary
browser it appears to be, so the same sign-in succeeds. Attaching afterwards
does not change that.

It also means **the browser is yours**. On shutdown the miner closes the tabs it
opened and disconnects; it never closes your browser. Restarting the miner
therefore keeps the login you made by hand — you sign in once and rarely again.

CDP is a Chromium protocol, so the browser is Chrome or Chromium. That is a
constraint, not a preference.

### Step by step

```bash
pip install playwright                  # the Python package only
# No `playwright install` is needed: you bring your own browser.
sudo apt-get install -y chromium xvfb   # a browser, and a virtual screen for headless hosts

cd examples/custom_miner

# 1. Start a real Chrome in debug mode. This BLOCKS for the life of the browser,
#    so run it in its own terminal and use a second one for steps 2-3.
./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/claude-1

# 2. Sign in to https://claude.ai in that browser, by hand. On a headless box,
#    reach its window over VNC — see "Signing in when the box has no screen".

# 3. Verify, then run. CLAUDE_CDP defaults to 9222, so with one browser on the
#    default port there is nothing to configure.
python -m solvers.doctor claude --probe
MINER_BACKENDS=claude python run_miner.py
```

`start_debug_browser.sh` uses `$CHROME_BIN` if you set it, else the first of
`google-chrome-stable`, `google-chrome`, `chromium`, `chromium-browser` on your
PATH, else a Playwright-bundled Chromium if one is present. It keeps the CDP
port on loopback, adds
`--no-sandbox` only when you are root, refuses to double-launch a port already in
use, and waits for the port before telling you it is up. On a headless host it
runs Chrome under `xvfb-run`.

The line that means success is `CDP is up on http://127.0.0.1:9222`. Chrome may
print GPU, D-Bus or push-registration errors after it on a server — those are
normal with no desktop session and no Google account, and none of them touch CDP
or the login. Check the browser yourself any time with:

```bash
curl -s http://127.0.0.1:9222/json/version     # JSON back = healthy
```

**Keep the debug port on loopback.** Anyone who reaches it has full control of a
browser holding your logged-in sessions, on a box already exposing a public axon
port. The script never binds it off `127.0.0.1`; reach it over an SSH tunnel.

### Signing in when the box has no screen

The debug browser runs under Xvfb, so its window exists but nothing is showing
it. `start_debug_browser.sh` prints the exact `x11vnc` command to share that
screen — **copy it from the script's own output** rather than from here, because
the display number and the X auth cookie differ per browser:

```
display: :99 (xauth /root/.hone-miner/chrome/claude-1/.Xauthority)
...
  x11vnc -display :99 -auth /root/.hone-miner/chrome/claude-1/.Xauthority \
         -rfbport 5900 -localhost -nopw -forever
```

Run that in a second terminal, then from your own machine:

```bash
ssh -N -L 5900:127.0.0.1:5900 you@your-miner
# point a VNC client at 127.0.0.1:5900, sign in, leave the browser running
```

The display is derived from the port, so a second browser gets its own screen
and its own printed command. The `-auth` argument is not optional: `xvfb-run`
writes a private cookie, and without it `x11vnc` exits with *"No protocol
specified"*.

That VNC screen is an unauthenticated view of a browser you are about to type a
password into — hence `-localhost` and the tunnel. Never publish the port.

**If a sign-in is refused**, prefer email plus a one-time code over "Continue
with Google". Do not try to defeat the check by spoofing a user agent or
patching out automation flags — starting the browser yourself is the supported
way past it, and that is what this design already does.

### Accounts, browsers and tabs

The account is the rate-limit unit: one browser per account on its own port,
`*_TABS_PER_BROWSER` conversation slots inside each, and at least
`MINER_MAX_CONCURRENT_REQUESTS` tabs in total or extra tasks queue and burn
their deadline. The launcher warns when it is short.

```dotenv
# one debug Chrome per account, each on its own port
CLAUDE_CDP=9222,9223
CLAUDE_TABS_PER_BROWSER=2
```

⚠ `MINER_MAX_CONCURRENT_REQUESTS` defaults to **4** while one browser gives
**2** tabs, so the single-browser setup logs a capacity warning on every launch.
Either lower the concurrency to match your tabs, or add tabs/browsers:

```dotenv
MINER_MAX_CONCURRENT_REQUESTS=2     # matches one browser at 2 tabs
```

One browser can serve both backends — sign the same Chrome in to claude.ai and
chatgpt.com, and leave `CLAUDE_CDP` and `CHATGPT_CDP` both on `9222`. Separate
ports are for separate *accounts*, not separate providers.

**Keeping it alive.** Two systemd services with `Restart=always`: the debug
browser (with the same `--profile`, so it restarts already signed in) and the
miner, which reconnects on its next start.

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
- **Artifacts.** Long code can land in Claude's side panel, outside the message
  the reader scrapes, so every *Claude* prompt asks for an inline code block
  (`CLAUDE_NUDGE` overrides the wording). ChatGPT gets no such suffix; its
  shared instructions already ask for a single code block.
- **A "still generating" selector that is always true** would make every answer
  look unfinished and burn the whole budget. Each candidate is checked against
  a freshly-loaded idle page at startup, and any that matches is dropped.

The same doctor works for ChatGPT: `python -m solvers.doctor chatgpt`.

## Included backend: ChatGPT in the browser + self-verification

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
own sign-in often routes through Google, which is exactly the check that rejects
a driver-launched browser — so signing in by hand, as below, is not optional
here. Point `CHATGPT_CDP` at the browsers you started:

Each launcher **stays in the foreground for the life of its browser**, so give
each one its own terminal — they are not a sequence to paste in one go.

```bash
# terminal 1 — stays busy
./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/gpt-1

# terminal 2 — stays busy
./scripts/start_debug_browser.sh --port 9223 --profile ~/.hone-miner/chrome/gpt-2

# terminal 3, after signing in to chatgpt.com in both
python -m solvers.doctor chatgpt --cdp 9222 --probe
CHATGPT_CDP=9222,9223 python run_chatgpt_miner.py
```

Keep the tab count at or above `MINER_MAX_CONCURRENT_REQUESTS`, or extra tasks
queue and burn their deadline. The launcher warns when it is short.

| Variable | Default | Meaning |
|---|---|---|
| `CHATGPT_CDP` | `9222` | CDP endpoint(s) of the Chrome you started, one per account |
| `CHATGPT_TABS_PER_BROWSER` | `2` | Conversation slots per browser |
| `SOLVER_MAX_ATTEMPTS` | `3` | Model round-trips: 1 initial answer + 2 repairs |
| `SOLVER_SAFETY_MARGIN_S` | `15` | Headroom kept before the cutoff |
| `SOLVER_MAX_BUDGET_S` | `240` | Hard cap on one solve |
| `SOLVER_VERIFY_EXECUTOR` | `subprocess` | Python grading backend; Rust always uses Docker |

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
  browser, and a driver-launched one is rejected outright by some sign-in flows
  — Google's answers "Couldn't sign you in". Starting the browser yourself is
  the answer to that specific wall, because it is not in automation mode, and it
  is what this design already does. Do not go further: spoofing user agents or
  patching out automation flags is an arms race on someone else's schedule.
  Prefer email plus a one-time code over "Continue with Google".
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
- **Rust.** Rust verification always uses the Docker executor — the solver
  forces it regardless of `SOLVER_VERIFY_EXECUTOR`, because there is no
  subprocess path to `rustc`. What it actually needs is a working Docker daemon
  and the pinned image; without those, grading is skipped and Rust candidates
  come back unverified.

## Testing your setup

Four layers, cheapest first, each isolating a different failure:

```bash
# from the repo root
python -m pytest examples/custom_miner    # 1. code only — no browser, no chain

# from examples/custom_miner
cd examples/custom_miner
python -m solvers.doctor claude --probe   # 2. the browser's sign-in + selectors
python scripts/try_solver.py              # 3. a real solve, end to end, no wallet
                                          # 4. testnet, then finney
```

Layer 1 runs from the repo root; layers 2 and 3 run from `examples/custom_miner`
— they are a package and a sibling script. Use `python -m pytest`, not bare
`pytest`: the bare binary silently falls through to a system install if the
`dev` extra is missing.

`try_solver.py` honours the same `.env` as the miner, so it drives the same
browsers the miner would, with no extra flags.

`scripts/try_solver.py` is the one to reach for when something is wrong. It
builds the solver exactly as the miner does, hands it one task with public
examples, and reports whether the answer reproduced them — while importing
neither bittensor nor `custom_miner`, so it runs before you have a wallet and a
failure there can only be the solver. Its three outcomes are distinct on
purpose: verified (setup is good), code-but-wrong (plumbing works, model
missed), and nothing-came-back (login, selector, or deadline — it names all
three in likelihood order).

```bash
cd examples/custom_miner
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

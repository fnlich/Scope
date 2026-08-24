# Run your own application as a miner

The stock demo miner (`rlvr/neurons/demo_miner.py`) answers tasks with GLM-5.2.
`custom_miner.py` keeps every piece of that miner that talks to the subnet —
signature verification, the replay-nonce cache, validator-permit authorization,
response signing, and the byte/concurrency limits — and replaces **only** the
part that produces an answer, so your own solver plugs in without you
re-implementing the wire protocol.

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
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[chain,miner]'
cp .env.example .env        # set NETUID, SUBTENSOR_NETWORK, WALLET_NAME,
                            # WALLET_HOTKEY, AXON_PORT, AXON_EXTERNAL_IP, MY_APP_URL
python examples/custom_miner/custom_miner.py
```

Register the hotkey first (see `scripts/register_testnet.sh`), open the axon
port to the internet, and confirm `curl http://127.0.0.1:$AXON_PORT/health`.

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

## Backends: Claude, ChatGPT, Gemini

Four backends ship, all behind the same `open()` → `send()`/`close()` protocol,
all feeding the same self-verify-and-repair loop. Pick with `MINER_BACKENDS`:

```bash
MINER_BACKENDS=claude                  python examples/custom_miner/run_miner.py
MINER_BACKENDS=claude,chatgpt          python examples/custom_miner/run_miner.py
MINER_BACKENDS=claude,chatgpt,gemini   python examples/custom_miner/run_miner.py
```

| Backend | Credentials | Quota you spend |
|---|---|---|
| `claude` | a logged-in Chrome on `CLAUDE_PORTS` | your Claude subscription |
| `chatgpt` | a logged-in Chrome on `CHATGPT_PORTS` | your ChatGPT subscription |
| `claude-api` | `ANTHROPIC_API_KEY`, or an `ant auth login` profile | tokens, per task |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY`; set `GEMINI_MODEL` | tokens, per task |

`claude` and `chatgpt` drive a browser you are already logged in to and read no
API key at all. `claude-api` is the same model over the API for anyone who would
rather pay per token than keep a browser alive; see the risk list below for why
you might.

`run_chatgpt_miner.py` still exists and is equivalent to `MINER_BACKENDS=chatgpt`.

### More than one backend is a fallback chain

With several names, providers run **in order and stop at the first answer that
reproduces every public example**. That ordering is a cost decision, not a
quality one: a verified answer ends the chain, so later providers cost nothing
on tasks the first one solves. Put the provider you would rather pay for first.

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
| `CLAUDE_PORTS` | `9222` | CDP ports for the `claude` browser backend |
| `CLAUDE_MODEL` | `claude-opus-5` | `claude-api` only |
| `CLAUDE_EFFORT` | `high` | `claude-api` only; `low` … `max` |
| `GEMINI_MODEL` | — | Set one your key can actually reach |
| `SOLVER_MAX_ATTEMPTS` | `3` | Repair rounds per provider |

These are read from the process environment **and** from `.env` — `run_miner.py`
loads that file into the environment itself, because the miner's own settings go
through pydantic-settings, which reads `.env` directly and leaves `os.environ`
untouched. A shell variable still wins over the file.

`GET /solver-status` reports the chain, which provider verified each task, and
per-provider turn and error counts — watch it, because a provider that starts
failing looks exactly like success until the score drops.

## Running Claude from the browser

`claude` attaches to a Chrome you have already logged in to, over CDP. No API
key is involved; the quota is whatever your Claude plan gives you.

```bash
pip install playwright     # not in any extra — the API backends do not need it
                           # and no `playwright install` is needed either: these
                           # attach to a Chrome you start yourself

chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-claude-1
# log in to https://claude.ai in that window, then verify the selectors:
cd examples/custom_miner && python -m solvers.doctor claude --probe
# then:
CLAUDE_PORTS=9222 MINER_BACKENDS=claude python examples/custom_miner/run_miner.py
```

As with ChatGPT, the account is the rate-limit unit: one browser per Claude
account, `CLAUDE_TABS_PER_BROWSER` conversation slots inside each, and at least
`MINER_MAX_CONCURRENT_REQUESTS` tabs in total or extra tasks queue and burn
their deadline. The launcher warns when it is short.

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

## Included backend: ChatGPT over CDP + self-verification

`run_chatgpt_miner.py` wires up a ready-made solver built from
[fnlich/Automation](https://github.com/fnlich/Automation)'s CDP driver:

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
pip install playwright   # see the note under "Running Claude from the browser"
chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-1
chrome --remote-debugging-port=9223 --user-data-dir=/tmp/chrome-2
# log in to https://chatgpt.com in each, then check it:
python -m solvers.doctor chatgpt --probe
CHATGPT_PORTS=9222,9223 python examples/custom_miner/run_chatgpt_miner.py
```

Keep the tab count at or above `MINER_MAX_CONCURRENT_REQUESTS`, or extra tasks
queue and burn their deadline. The launcher warns when it is short.

| Variable | Default | Meaning |
|---|---|---|
| `CHATGPT_PORTS` | `9222` | Comma-separated CDP ports (one per account) |
| `CHATGPT_TABS_PER_BROWSER` | `2` | Conversation slots per browser |
| `CHATGPT_HOST` | `127.0.0.1` | CDP host (`CLAUDE_HOST` for Claude) |
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
  the API. The realistic downside is account termination. This applies to
  `claude` and `chatgpt` equally; `claude-api` and `gemini` are the supported
  way to do the same thing.
- **Fragility.** Chrome updates, DOM changes, expired logins, rate limits and
  CAPTCHAs all break browser automation. A miner that does not answer scores
  zero into a 200-observation window (~2.1 days), so one bad night costs most
  of your score. This is what the doctor and the `.env` selector overrides are
  for, and why the API backends exist alongside: switching is one variable,
  `MINER_BACKENDS=claude-api`, with nothing else to change.
- **Rust.** `SOLVER_VERIFY_EXECUTOR=docker` is required to verify Rust answers;
  without it Rust candidates are returned unverified.

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

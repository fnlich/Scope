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

## Backends: Claude, Gemini, ChatGPT

Three backends ship, all behind the same `open()` → `send()`/`close()` protocol,
all feeding the same self-verify-and-repair loop. Pick with `MINER_BACKENDS`:

```bash
MINER_BACKENDS=claude                  python examples/custom_miner/run_miner.py
MINER_BACKENDS=claude,gemini           python examples/custom_miner/run_miner.py
MINER_BACKENDS=claude,gemini,chatgpt   python examples/custom_miner/run_miner.py
```

| Backend | Credentials | Needs a browser |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY`, or an `ant auth login` profile | no |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY`; set `GEMINI_MODEL` | no |
| `chatgpt` | a logged-in Chrome on `CHATGPT_PORTS` | yes |

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
| `CLAUDE_MODEL` | `claude-opus-5` | |
| `CLAUDE_EFFORT` | `high` | `low` … `max` |
| `GEMINI_MODEL` | `gemini-3-pro` | Set one your key can actually reach |
| `SOLVER_MAX_ATTEMPTS` | `3` | Repair rounds per provider |

`GET /solver-status` reports the chain, which provider verified each task, and
per-provider turn and error counts — watch it, because a provider that starts
failing looks exactly like success until the score drops.

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
chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-1
chrome --remote-debugging-port=9223 --user-data-dir=/tmp/chrome-2
# log in to https://chatgpt.com in each, then:
CHATGPT_PORTS=9222,9223 python examples/custom_miner/run_chatgpt_miner.py
```

Keep the tab count at or above `MINER_MAX_CONCURRENT_REQUESTS`, or extra tasks
queue and burn their deadline. The launcher warns when it is short.

| Variable | Default | Meaning |
|---|---|---|
| `CHATGPT_PORTS` | `9222` | Comma-separated CDP ports (one per account) |
| `CHATGPT_TABS_PER_BROWSER` | `2` | Conversation slots per browser |
| `SOLVER_MAX_ATTEMPTS` | `3` | Initial answer + repair rounds |
| `SOLVER_SAFETY_MARGIN_S` | `15` | Headroom kept before the cutoff |
| `SOLVER_MAX_BUDGET_S` | `240` | Hard cap on one solve |
| `SOLVER_VERIFY_EXECUTOR` | `subprocess` | `docker` also verifies Rust |

`GET /solver-status` reports pool health and solve counts. Watch it — a
browser-backed miner fails quietly, and silence looks identical to success.

### Know the risks before running this for money

- **Terms of service.** Driving ChatGPT's web UI programmatically to power a
  paid service is very likely against OpenAI's terms, which prohibit automated
  extraction of Output. The realistic downside is account termination.
- **Fragility.** Chrome updates, DOM changes, expired logins, rate limits and
  CAPTCHAs all break browser automation. A miner that does not answer scores
  zero into a 200-observation window (~2.1 days), so one bad night costs most
  of your score. The backend is deliberately swappable for exactly this reason:
  moving to an API backend means implementing `open()` returning something with
  `send()`/`close()`, and touching nothing else.
- **Rust.** `SOLVER_VERIFY_EXECUTOR=docker` is required to verify Rust answers;
  without it Rust candidates are returned unverified.

## Tests

```bash
pytest examples/custom_miner
```

Kept out of `tests/` (the validator's own suite) so the default `pytest -q` is
unaffected. They lock in the four validator acceptance checks on a real signed
reply, that the verify loop repairs a wrong answer, that a solve never outruns
its budget and never returns nothing when it has something, and that a browser
tab which dies is retired rather than recycled into the pool.

## Two caveats worth knowing

- **Trust the request's `deadline_s` cautiously.** The validator advertises one
  deadline in the request but enforces the *problem server's* deadline as the
  real cutoff, and the two are not guaranteed to match. Budget your solve
  conservatively rather than spending right up to the advertised value.
- **Keep the permit check on.** With `MINER_REQUIRE_VALIDATOR_PERMIT=true` only
  stake-gated validators can spend your compute. Relaxing it lets any registered
  hotkey call your `/solve` and harvest solutions.

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

`custom_miner.py`, `run_miner.py`, the doctor and the login helper all
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

### The hidden suite is where the score is, so the prompt is written for it

The public examples are the friendly ones. Grading is on the **complete hidden
suite**, which is written to break a solution that only handles the shape it was
shown — so the prompt names the cases one at a time instead of saying "handle
edge cases", which every model agrees to and none acts on:

    NOTHING          empty list, empty string, n = 0
    ONE              n = 1, one element, one character
    TWO              where "first" and "last" stop being the same element
    BOTH ENDS        first and last, empty range, inclusive vs exclusive
    EXTREME VALUES   0, 1, -1, negatives, the largest magnitude allowed
    DEGENERATE       all equal, all duplicates, sorted, reverse sorted

The examples are rendered *after* that list and labelled a floor rather than the
specification, because read first they become the spec and the checklist reads
as an afterthought. The model is also asked for one comment at the top of its
code stating what the program does for the empty case and for `n = 1` — writing
it down is what turns the list from something to agree with into something to
do. Repair rounds are sent back through the same checklist, since they share the
conversation and a repair that fixes the failing example while breaking a
boundary scores the same zero.

Each language is then warned about its own way of losing a large number, because
they are not the same failure:

- **Rust — overflow is silent.** The validator compiles with `-C opt-level=2`,
  and `rustc` turns overflow checks off at any opt-level above zero. `i32`
  arithmetic wraps and the program **exits 0 with a plausible wrong number**
  rather than panicking. Measured with the validator's own flags: two `i32`
  values of `2_000_000_000` sum to `-294967296`. So the prompt asks for `i64` by
  default and `i128` for products. There is no message and nothing in the
  failure that points at the cause, which is exactly why it has to be said up
  front.
- **Python — integers never overflow, but recursion dies at 1000.** A recursive
  answer fails at `n = 10^4` with `RecursionError`, so the prompt asks for
  iteration. It also states the comparison rules that cost a solve when guessed
  at: `True` is not `1`, two integers must match exactly, a dict must have
  exactly the expected keys — and a list and a tuple *are* interchangeable, so
  no repair round need be spent converting one.

Both are told the real per-test budget (5 seconds), because a budget quoted
generously invites an algorithm that does not fit.

Every one of those claims is about somebody else's code, and claims like that
rot without anyone noticing — the prompt keeps saying them long after the policy
that made them true has moved. So each is pinned by a test: the overflow test
compiles with `RELEASE_POLICY.rustc_flags` rather than a copy of them, the
comparison claims are asserted against `rlvr.execution.compare.values_equal`,
and the timeout is read from the validator's own config. Change the policy and
the tests fail, instead of the prompt quietly starting to lie.

## How it works: you run the browsers, the miner uses them

You start N browsers — six to ten is a normal fleet — each signed in **by hand**
to one provider, each on its own debugging port. The miner attaches to all of
them and treats their tabs as **one fleet**, and takes the browsers **in turn**:
request 1 to the 1st browser, request 2 to the 2nd, ... request n to the nth,
request n+1 back to the 1st.

```dotenv
CLAUDE_CDP=9222,9223,9224      # three browsers signed in to claude.ai
CHATGPT_CDP=9225,9226,9227     # three signed in to chatgpt.com
MINER_TABS_PER_BROWSER=2       # conversation slots inside each -> 12 tabs
```

Set either list or both. **No API key is read anywhere in this package.**

### Why one fleet and not one pool per provider

Accounts are what actually rate-limits you, so accounts are the axis worth
scaling — and a task does not care which model answers it. So the useful unit is
"the next free tab", not "which provider do we prefer". Two consequences:

- **Throughput scales with browsers.** Six browsers at two tabs is twelve
  concurrent conversations. There is deliberately no provider-preference
  setting: naming one provider "first" would queue tasks on its browsers while
  the others sat idle.
- **Leases rotate, by rule.** A cursor walks the browsers in the order you
  configured them and each request takes the next one's turn, so consecutive
  tasks are spread across accounts by construction. Handing out the next free
  tab instead gets the same answer only while tasks finish in the order they
  started — and they do not: a tab is freed when its task *ends*, so one
  account drawing easy problems finishes sooner, comes back to the front, and
  quietly starts taking more than its share.

  One deliberate exception: if the browser whose turn it is has no free tab, the
  turn passes to the next browser that does. A miner is paid for answers that
  beat the deadline, so idling behind one busy account while another sits free
  would trade money for a tidier sequence. The cursor follows the browser
  actually used and carries on from there.

  Browsers that failed to attach are not in the rotation — `n` is the number of
  browsers actually serving, not the number of ports in your `.env`.

### The one time the provider matters

If an answer still cannot reproduce the task's public examples after its repair
rounds, the odds it passes the **hidden** suite are poor — and the whole payment
rides on that. So the solver asks the *other* model once, on a tab from a
different provider. With a fleet there is usually an idle one, and a second
chance at the full payment is worth far more than the time it costs.

```dotenv
SOLVER_SECOND_OPINION=false     # turn it off for pure throughput
```

Turn it off if you would rather never spend two accounts on one task.

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
python run_miner.py
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
MINER_TABS_PER_BROWSER=2
```

⚠ `MINER_MAX_CONCURRENT_REQUESTS` defaults to **4** while one browser gives
**2** tabs, so the single-browser setup logs a capacity warning on every launch.
Either lower the concurrency to match your tabs, or add tabs/browsers:

```dotenv
MINER_MAX_CONCURRENT_REQUESTS=2     # matches one browser at 2 tabs
```

**One port, one browser, one provider.** Listing the same port under both
`CLAUDE_CDP` and `CHATGPT_CDP` does not give you two backends on one Chrome —
the second is dropped with a warning, and you get the first provider only. It
has to work that way: the fleet reclaims a browser's leftover tabs each time it
attaches, so a second attach to the same browser would close the tabs the first
had just opened. To run both providers, start a second browser on its own port
and sign it in to the other one:

```dotenv
CLAUDE_CDP=9222,9223,9224      # three browsers on claude.ai
CHATGPT_CDP=9225,9226,9227     # three more on chatgpt.com — different ports
```

**A tab is opened once and kept.** What separates one task from the next is a
fresh *conversation*, not a fresh tab — a task must never see the previous
task's code, or the model blends the two and fails the hidden suite in a way
that is very hard to diagnose. The tab itself is the expensive part: signed in
by hand, warm, and only ever replaced when it dies. So the reset is done as
cheaply as it can be:

1. **Nothing to do** — a tab that has just been opened is already in an empty
   conversation. Its first task reloads nothing.
2. **The site's own "new chat"** — an in-app route change: no bundle refetch,
   no app boot, no re-auth. Taken only when the transcript is demonstrably gone
   afterwards, because a control that quietly did not route would leave the last
   task in view and the next answer would come back promptly and wrong.
3. **Reload the page** — what every task used to do, now only when 1 and 2
   cannot be proven.

Because tabs are replaced *only* on death, tab churn in your browser is a real
signal that something is wrong. It is not what ordinary work looks like.

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
CLAUDE_NEW_CHAT='a[href$="/new"]'
CLAUDE_COPY='button[aria-label="Copy to clipboard"]'
```

Two roles are optional. `*_NEW_CHAT` is how a tab starts its next conversation
without a page load; if nothing matches, the tab reloads instead — a few seconds
per task, never a wrong answer. `*_COPY` is the code block's own copy control,
and it is how the code is normally taken (see below); if nothing matches, the
miner reads the DOM instead and says so once. The doctor reports both like any
other role.

Two more are not selectors at all, and both are off by default in the direction
that cannot hurt you:

```dotenv
CLAUDE_STREAM=0          # stop reading the answer off the network entirely
CLAUDE_STREAM_FIRST=1    # ...or trust it over what the page shows
```

## Where the answer is read from

There are four places the same answer can be taken from, and they are not
equally good. In order:

| Source | What it is | Used |
| --- | --- | --- |
| the network stream | the markdown the model emitted, before any of it was a page | when the page reads back empty; as primary with `*_STREAM_FIRST=1` |
| framework state | the source string the site holds per block | not reachable from outside |
| the copy control | that state, handed back on request | **primary** |
| the rendered DOM | `pre code`, after a syntax highlighter rebuilt it | fallback |

Everything below the first line is downstream of a render. The stream is not,
which is why it is the one source with anything left to say when the page read
comes back empty — and an empty page read is where *every* recent `the reply
contained no code` has come from.

## The code comes from the copy control, not the DOM

`pre code` gives you the source *after* a syntax highlighter has rebuilt it as
DOM. The copy control gives you what the model actually wrote. Those differ, and
they have differed here: a highlighter once put U+E027 — a Private Use Area
character that appears in no source file — inside a Python answer, and the solve
died on a character nobody could see. So the reader scrapes the DOM to decide
*when* an answer is finished, then clicks the block's own copy control once to
decide *what* it says.

The value never reaches your clipboard. `navigator.clipboard.writeText` is
patched inside the miner's own tabs so the string comes back to the miner and
goes no further. That is not fastidiousness — there is one clipboard shared by
every tab, every browser on the display, and every miner you run, and reading it
back was measured to cross tabs:

    tab A wrote 'TAB-A-CODE', tab B wrote 'TAB-B-CODE'
    tab A read back: 'TAB-B-CODE'

A pool that read the clipboard would submit another task's program whenever two
solves overlapped, silently. The one visible consequence of the patch: while the
miner owns a tab, that tab's copy buttons no longer write to your real clipboard.

Nothing is *clicked* unless it says what it is. A selector is a guess about
structure and can drift onto a neighbour — ChatGPT keeps "Run code" in the same
header as "Copy", and reading an answer is worth a click where executing it is
not. So the control's accessible name is checked first and must contain `copy`.
On a UI in another language, set `CLAUDE_COPY_NAME` / `CHATGPT_COPY_NAME`; until
you do, the miner reads the DOM instead, which is the safe direction to fail.

It is preferred, not required. If the control is missing or renamed, scraping
answers instead, and a reply with two blocks whose controls only half-respond is
handed back to scraping whole rather than losing a block.

### Both readings are taken, and disagreement is reported

Choosing the better source is only half of it. Every extraction bug this miner
has had was *silent* — a Private Use Area character, a leaked language chip, a
blank line inserted at a render boundary. Each one looked exactly like the model
writing bad code, and each cost days to find.

Two independent readings of the same answer are already in hand by the time a
send finishes, so they are compared. The copy still wins; the difference gets
logged:

```
[claude] note: tab claude#1: what the page RENDERS and what it COPIES are not
the same — they differ at character 14: rendered '\ue027' (U+E027), copied
'+' (U+002B). Using the copy, which is the source before syntax highlighting.
```

It names the codepoint because that is the part you can act on. It fires once
per tab, and only on a real difference — a warning that appears on every answer
is one nobody reads.

### The answer as it came off the wire

Every tab the miner opens patches `window.fetch` before the site's own code
runs, and keeps a copy of any streaming response body. That is the answer
*before* the page exists.

The patch is written so it cannot break a site you are signed in to: it clones
the response rather than replacing it (a hand-built `Response` loses `url` and
`redirected`, and a chat UI reading either breaks in a way that looks like the
site's own bug), it returns the original object untouched on every path
including the ones where it throws, it ignores anything that is not a streaming
content type, and it bounds what it keeps. Verified in a browser: `url`,
`status`, `redirected` and `bodyUsed` all unchanged, ordinary JSON requests not
cloned at all, a fetch that should fail still failing.

Reconstructing the markdown is the part nobody can do for you. Neither site
publishes its stream format and both change theirs without notice, so nothing
about either is hard-coded. What is relied on is structural: an SSE stream is
many small JSON events, and the answer is the one field appended to over and
over. Group every string leaf by its path *plus* the short strings that came
with it, concatenate, take the biggest group — dropping reasoning by name
(`thinking`, `thoughts`), and dropping tags, which repeat a handful of values
across hundreds of events. Both real shapes are pinned in the tests, including
reasoning that outweighs the answer four to one and metadata interleaved into
the middle of it.

Because that is a heuristic over a private format, it does **not** simply win:

- the page read back empty → the wire is used, and says so;
- both produced code and they agree → nothing changes;
- both produced code and they differ → the page is used and the difference is
  logged, with the codepoint;
- `CLAUDE_STREAM_FIRST=1` / `CHATGPT_STREAM_FIRST=1` → the wire is used as the
  primary. Set this once `--probe` has shown you the two agreeing on your own
  accounts, not before.

`CLAUDE_STREAM=0` / `CHATGPT_STREAM=0` turns the capture off entirely.
`python -m solvers.doctor claude --probe` reports all of it side by side:

```
  where the answer can be read from, on this page:
    [ok   ] network — 1 streamed response(s) seen this turn, 812 chars
            reconstructed, 2 code block(s)
    the page and the wire AGREE on all 2 block(s). Setting CLAUDE_STREAM_FIRST=1
    would read this page from the wire, which is the source before any rendering
    happened to it.
```

### When nothing is captured at all, the tab says why

`the reply contained no code` is what the grader reports afterwards, and it
describes several very different causes identically: a selector that matches
nothing, a reply that never rendered, an answer still being written when the
budget ran out. Only the page can tell them apart, and only at the time — so it
is asked before the tab moves on:

```
[claude] tab claude#1 captured NOTHING from this reply: 'div[data-is-streaming]'
matched 1 message(s) but none of them could be identified as the answer to this
prompt. This is what surfaces later as "the reply contained no code".
```

One read is never allowed to spend the whole budget. A poll resolves a node and
then reads it, and if the site swaps the message between those two steps the
read waits on an element that no longer matches — which Playwright does for
thirty seconds. Bounded only by the send's remaining time, that single poll
spends every second the solve had left and returns nothing, while the finished
answer sits on screen. Measured on the transition claude.ai actually makes: an
8s send spent 7.85s inside one `inner_text()` and returned `""`. Each read now
gets five seconds and a timed-out read is retried, not fatal; the retry
re-resolves and costs milliseconds.

One related failure is now repaired rather than reported. A site streams a
message under one attribute and drops it when the message is finished, so the
candidate that *found* the answer can be the one that cannot see it. The
assistant selector is latched for the whole send — deliberately, so message
counts stay comparable — but the latch is dropped the moment it matches
nothing, because there is no count to corrupt at zero and the alternative is
reading nothing while the answer sits on screen.

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

## Self-verification: the part that earns the money

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
large class of answers that are simply wrong on the stated contract. When even
that is not enough, the second opinion asks the other model — see
[The one time the provider matters](#the-one-time-the-provider-matters).

The ChatGPT reader is a direct port of
[fnlich/Automation](https://github.com/fnlich/Automation)'s driver: identify the
reply by its `data-message-id`, treat it as finished only when the Stop button
is gone *and* the text is unchanged across two polls, and start every task in a
fresh conversation.

| Variable | Default | Meaning |
|---|---|---|
| `SOLVER_SAFETY_MARGIN_S` | `15` | Headroom kept before the cutoff |
| `SOLVER_MAX_BUDGET_S` | `240` | Hard cap on one solve |
| `SOLVER_VERIFY_EXECUTOR` | `subprocess` | Python grading backend; Rust always uses Docker |

`GET /solver-status` reports per-provider counters and fleet health. Watch it —
a browser miner fails quietly, and silence looks identical to success.

## Running under pm2

pm2 supervises **only the miner**. The browsers are yours: you start them, you
sign in, and they stay up across miner restarts — which is exactly why sign-in
works at all.

```bash
cd examples/custom_miner
pm2 start ecosystem.config.js
pm2 logs hone-miner
pm2 save && pm2 startup        # survive a reboot
```

Restarts are safe by design, and both halves of that were tested:

- **Clean stop** (`pm2 stop`/`restart` sends SIGINT, then SIGTERM). The miner
  handles both, closes the tabs it opened, and disconnects without touching your
  browsers. The handlers are installed *before* it attaches, so a restart during
  startup — when attaching to eight browsers takes a while — is still clean.
- **Unclean kill** (OOM, `kill -9`, a crash). Nothing runs, so the tabs are
  orphaned. The next start finds them and closes them: every tab this miner
  opens is stamped in `window.name`, which your own tabs never carry. Verified
  over three kill/restart cycles — the tab count stays flat instead of growing
  by `MINER_TABS_PER_BROWSER` each time.

If a browser is down when the miner starts, it logs which endpoint failed and
serves with the rest of the fleet. Bring the browser back and restart the miner
to pick it up again.

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
  bad night costs most of your score — and there is no API backend to switch to
  when it happens. Three things stand in for that: run both providers so they
  are not down together, run the doctor before you serve, and
  watch `/solver-status`, because a provider that has started failing looks
  exactly like one that is merely quiet.
- **The `Backend` protocol is three methods.** If you do want an API path,
  `open()` returning something with `send(text, timeout_s)` and `close()` is the
  whole interface; `verify.py` and the fleet take it unchanged.
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

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

### The prompt is one delimited document, and the order is the argument

Every solve takes **two turns** — the cases first, then the program — and both
are laid out the same way:

```
TURN 1 (cases)                      TURN 2 (program)
<output>   one json block           <output>   one code block          ← first
<problem>  the statement            <problem>  the statement
<examples> a floor to agree with    <examples> a floor, not the spec
<task>     which cases, in order    <contract> what is TRUE: how it is
                                               run, compared, and what
                                               the environment does
                                               silently
                                    <must_pass> the model's OWN cases
                                                from turn 1          ← last
```
Both turns ask for **one** block, and there is no third shape: the single-turn
prompt that asked for a program and its cases together is gone — see
*[The cases are written before the program exists](#the-cases-are-written-before-the-program-exists)*.
…then the site's nudge, appended after everything, repeats the output rule.

The output contract holds **both ends**. It is the only instruction whose
failure costs the entire answer rather than degrading it, so it gets primacy and
recency and nothing else competes for either. The problem comes next, because
instructions about how to solve something are unreadable before you know what it
is. What the program has to clear comes last, immediately before it is written.

**Everything else has been deleted, and the deletion is the design.** Turn 2
used to carry a `<method>` — a six-step numbered procedure — wrapping an
`<edge_cases>` checklist of twelve input shapes, a `<self_check>` of seven
re-reads, and a closing coda. Inside that one message *"a wrong answer pays
zero"* was stated four times, *"work silently, not in the reply"* three times,
and *"send only the program"* three times over plus the nudge; turn 1's case
classes were then restated almost verbatim as turn 2's edge cases, so the model
was told the same thing twice across two turns and neither telling was the one
it was graded against.

None of that was free. Every sentence is an instruction a model can obey
*instead of* answering, and a procedure that says how to think competes with the
task for the same attention. The prompts now state **what is true** — how the
code is invoked, compared and timed — and **what it must clear**, and stop.

The hurrying went with it. `"every character you emit spends wall-clock inside
the deadline"`, `"keep every trace terse"`, `"the payment rule pays the slowest
correct answer at least 95% of what the fastest one earns"`, and a nudge warning
that `"an answer that arrives after a paragraph of prose may not arrive at all"`
all traded thoroughness for speed — on a subnet where a correct answer pays
everything, a wrong one pays nothing, and speed is worth at most 5%. There is no
per-turn deadline to spend either: only the request's own. A test now fails on
the vocabulary — *deadline, wall-clock, terse, fastest, 95%* — in any prompt,
and on any instruction to do work that never reaches the reply.

Three facts were missing that change decisions rather than decorate them.
**There is no partial credit** — a program wrong on one hidden case scores what
no answer scores, which is the difference between reaching for the clever
implementation and the safe one. **The examples decide** when the statement is
ambiguous; they are the only disambiguation a solver is given, and nothing said
so. And **hash order is not stable across processes**: measured, four runs of
`list({'alpha','beta','gamma'})` gave four different orders because
`PYTHONHASHSEED` is random per process, while a set of small ints gave the same
order every time — so a solution tested with integers looks stable and is not.
Rust randomises `HashMap`/`HashSet` iteration for the same reason.

### The hidden suite is where the score is, so turn 1 is written for it

The public examples are the friendly ones, and on live traffic there are usually
none at all. Grading is on the **complete hidden suite**, written to break a
solution that only handles the shape it was shown — so the cases the program
will be checked against are asked for *first, in their own turn*, before the
program exists to back-fill them from. Turn 1 names five classes and their
order is the instruction:

    THREE ORDINARY   typical inputs — the common path an all-boundary suite
                     never checks, and where a program wrong down the middle
                     passes every other case
    EMPTY OR ZERO    empty list, empty string, 0, an empty payload
    ONE              a single element, n = 1, the smallest legal input
    BOUNDARY         every limit, threshold and modulus the statement names,
                     AT that value, and the largest it allows
    LIKELY WRONG     what this particular problem makes easy to get wrong —
                     ties, duplicates, all-equal, sorted, reversed, a rule
                     that fires beside an input that NEARLY makes it fire

That list used to be nine classes with a per-class count, and it was restated
almost verbatim in turn 2 as an `<edge_cases>` checklist the model was asked to
walk before answering. Both are gone: the classes belong to the turn that
produces cases, and turn 2 is handed the cases themselves under `<must_pass>`,
which is the same information as something the grader will actually run.

The examples are rendered *with* the problem and labelled a floor rather than
the specification — and, since the statement is the spec, the label carries the
one disambiguation rule a solver gets: where the statement is ambiguous, the
examples decide.

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

### What was removed, and what it cost

Turn 2 used to end with a `<self_check>`: seven things to re-read the program
for, drawn from real submissions. Read off 43 answers a live miner sent, ten
were the model's own bugs and **eight of those ten were visible on a careful
re-read** — no test, no execution, no cleverness required:

| what shipped | what a re-read would have caught |
| --- | --- |
| `failed_any(&output)` | called, never written |
| `constrained[-1]` | on a list its own parser returns empty |
| `reason.push(255)` then indexed | a sentinel used as a position, into 5 buckets |
| `id = cmd_idx` | an id used where a push-order position was meant |
| `if s.len() != 11` inside `10 =>` | already guaranteed false, so the arm is dead |
| rebuild of `fcnt/first/last/rcnt` | forgot `size`, so every rank descent ran on 1 |
| `pc[jid] = pos + 1` | committed before a branch that must not commit it |

The evidence is kept here because it is worth knowing what these models get
wrong. The *section* is gone, deliberately: it was the third place in one
message telling the model to work silently before replying, it competed with the
task for attention, and the cases from turn 1 catch the same class of bug by
running the program rather than by asking it to look again. That is the trade —
a mechanism that sometimes caught a bug, exchanged for a prompt with one job in
it. If a future run shows those bugs coming back, the answer is a better case
turn, not another checklist.

The code is no longer asked for **unexplained**, either. That bullet — no
comments, no docstrings — existed because output costs wall-clock, and
wall-clock is not what this subnet pays for.

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

### Only a code block counts as an answer

The reader returns the message's code blocks or nothing. It never returns the
prose, and that is a statement about what this miner is for: it only ever wants
a code block, so a message without one is not an answer it can use.

The upside is not tidiness. **claude.ai renders extended thinking inside the
element the assistant selector matches**, and the thinking arrives long before
any code does — so for the whole first stretch of an answer, the message on
screen is reasoning and nothing else. Reading the message text turned that into
"here is your program": measured on a real solve, 13,200 characters of the model
working through the problem were submitted as Rust, the grader replied `the
program does not define fn main()`, and the repair round told the model to fix a
program it had never sent. Twice, until the budget ran out.

ChatGPT keeps its reasoning *outside* the matched element, so the identical
moment there read as empty and was reported as empty. One DOM difference, two
completely different diagnoses for the same situation — which is why the ChatGPT
failure looked like a capture bug and the Claude one looked like a code bug.
Reading only code blocks makes the site's markup stop mattering.

The same rule applies once more at extraction. A reply with no fence anywhere is
kept only if it is *gradeable* — a model that ignores the formatting and types
the program bare has still answered — and is otherwise reported as nothing
arriving, which is true. Punctuation does not decide; the same defect check that
picks between fenced blocks does.

What is given up is the chance to SEE what a code-free reply said, and that is
exactly the thing that makes a silent failure take days. So the post-mortem
quotes it:

```
[claude] tab claude#1 captured NOTHING from this reply: the message has no code
block in it. It says: 'I need more detail about the framing rules before I can
answer.' This is what surfaces later as "the reply contained no code".
```

A message with no code and a selector matching an empty wrapper both arrive as
an empty read and need opposite fixes — the first is the model's doing, the
second is yours — so they are named apart.

### A tool call is not an answer either

This one is Claude's — the archived file quotes `/home/claude/sol`, its analysis
sandbox. Reading only code blocks is not enough on its own, because **when a
model reaches for its tools, the chat UI paints every tool call as a code block
too**
— the same `pre code` markup an answer gets. There is no toolchain behind a chat
window (one session tried `apt-get install rustc`), so those calls achieve
nothing, and the model can end a turn having written its program only *inside*
one:

```
{"command": "cat > main.rs << 'RUST_EOF'\nuse std::io;\nfn main() { … }\nRUST_EOF"}
```

That used to pass every test the miner had. `rust_defect` was `"fn main" in
code`, and the block does mention `fn main` — quoted inside a shell heredoc,
inside JSON. It was picked as the answer, submitted, and archived as the
solution, on a solve where the model had answered correctly further up the
message.

Three things changed:

- **`fn main` must begin a line.** Inside an escaped string it only ever appears
  mid-line, after a literal `\n`. This also catches a genuine Rust file that
  merely *quotes* `fn main` in a string or a macro, which used to pass and then
  fail to link.
- **A block must be plausibly source before it can be submitted.** Rust's top
  level is a closed grammar — a file can only open with an item, an attribute or
  a comment — so an allowlist of openers is exact and a shell command fails at
  its first character. Python's top level is arbitrary statements, so no
  allowlist can be written that does not reject real code (`MOD = 10**9 + 7` is
  a fine first line); there the short list of things a *tool call* opens with is
  named instead.
- **Both prompts ask the model not to use its tools at all** — the root cause,
  and the only fix that costs nothing.

The line this draws matters: a program with a fixable flaw — no entrypoint, a
syntax error, a line the deadline cut in half — **is** an attempt at an answer,
and both the grader and the repair round still get to see it. Only things that
were never attempts are dropped, and a reply of nothing but tool calls reports
that nothing arrived.

The network stream had the same bug from the other direction. A model asking a
tool to do something streams the request as `partial_json`, and "the field
appended to most" was then the tool call rather than the reply — measured, a
5,442-byte tool call beat the 54-byte answer beside it on volume alone. Tool
arguments are excluded by name now, like reasoning.

### The miner must never submit its own prompt

It did. Twice, to real validators, archived as Rust programs ending in the
words "Do not use canvas".

`_echoes_prompt` was written to stop exactly that, and it was applied in
exactly one place: inside `_poll`, guarding the scrape. Every other route to a
submission — the copy control, the network stream — went around it.

**Both came off ChatGPT tabs**, and the mechanism is ChatGPT's specifically. Its
response opens with a snapshot of the *conversation*, which holds the user's own
turn under `author.role = "user"` — and "the field appended to most" is then the
PROMPT whenever the answer is shorter, which it usually is. Reproduced on that
payload shape: a 1,384-character user turn beat the 41-character answer beside
it. So the reconstruction now discards anything the stream attributes to
someone other than the model. Who said it decides, not how much of it there is.

Two things changed. The rescue only claims a rescue when it actually has code
blocks, and returns nothing otherwise — the old message announced a recovery
in the same breath as admitting there was nothing to recover, and worse, made
the result non-empty and so silenced the post-mortem that would have explained
the page. And a final guard sits at the one exit `send()` has, so every route
is covered rather than one. It tests *containment* rather than the prefix test
the scrape guard uses, because by that stage the text has been through fencing,
a copy control or a wire reconstruction and the prompt need not be at the front
any more.

### And nothing is sent that we did not type

The mirror image, and it went the other way through the same box.

These accounts are shared — a person signs the browser in and may use it — so
the composer can already hold somebody's half-typed message. The submit path
clicked the composer and called `insert_text`, which inserts **at the caret**,
and the click had just put the caret at the element's centre. The prompt was
therefore spliced *into* that draft and the whole thing sent as one message:
their text, our prompt, the rest of their text.

Nothing downstream could catch it. `_is_our_own_prompt` inspects the **reply**
for our prompt's head, and in a contaminated send that head is still there,
intact; the stream capture is the one place the submitted text is visible at
all, and it deliberately discards the user's turn. The only symptom is a model
answering a mangled question, which is indistinguishable from a hard task.

So `_submit` now guarantees the box holds our prompt and nothing else:

1. clear it — `Control+A`, `Delete`, then a `fill("")` if the shortcut did not
   reach it — and **read it back** to confirm it emptied;
2. type the prompt, and read the box back **again**, comparing on collapsed
   whitespace (a contenteditable renders our newlines as its own block
   elements, so an exact comparison would fail on every send; what the
   comparison still catches is text we did not type);
3. only then touch the send control.

A box that will not empty, or that does not hold the prompt after one retype, is
not sent to at all: the tab raises, and `send` already treats a raise there as a
dead tab reporting `unreadable`, which sends the solve to another tab. Two extra
DOM round-trips, inside the submit slice the budget already allowed for.

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

## Every solve leaves a file

A browser-backed miner is hard to look at after the fact. The reply that
produced a zero is gone the moment the tab starts its next conversation, the log
says how the solve ended but not what was submitted, and the validator keeps the
only other copy. So each answer is written to disk, named for the problem:

```
solutions/
  bd41f0e2-….rs        the Rust source that was sent
  bd41f0e2-….json      what was asked, and what was actually replied
  7c9a1b04-….py        the Python source that was sent
  7c9a1b04-….json
  3f88d5aa-….rs        0 bytes — this problem was seen and answered with silence
  3f88d5aa-….json      …and this says WHICH problem, and why
```

The empty ones are the point. Absence would be ambiguous — never dispatched,
crashed before the solver ran, or answered with nothing — and those need
different fixes; a zero-byte file says which it was. A solver that *raises*
leaves one too, since that path never reaches the solver's own return.

The content is taken from the payload rather than from the variable that fed it,
so the file is the submission and not something that resembles it. The extension
follows the task's language.

The `.json` beside it holds the validator's **request** and the miner's
**reply**, verbatim:

```json
{
  "problem_id": "…",
  "request":  { "statement": "…", "entrypoint": "main", "public_examples": [ … ],
                "language": "rust", "deadline_s": 240.0 },
  "response": { "code": "fn main(){ … }", "raw_response": "Here is the program:\n\n```rust\n…" }
}
```

The code alone answers *what did we submit*. It cannot answer *was that a
reasonable thing to submit* — the statement, the entrypoint and the examples all
lived only in the request, and the model's actual words only in the tab. Reading
43 archived answers made the gap concrete: half could not be judged without the
problem they were answering, and the two most expensive bugs this miner has had
— a tool call submitted as Rust, a prompt submitted as Rust — are unmistakable
in `raw_response` and invisible in the code file, where each looked like a
finished program.

It sits **beside** the code rather than inside it because the code file is a
program: something has to be able to compile it, diff it or feed it to a grader
without stripping a header off first. Same stem, different extension, so the
pair is obvious in a listing and trivial to join. A solve that crashed writes
one too — the empty `.rs` says a problem was seen and answered with silence, and
only the record says which problem, and that the solver raised.

`SOLVER_SOLUTION_DIR` moves the directory; setting it to nothing turns archiving
off. Nothing here can cost you a solve — a disk that cannot be written explains
itself once and the answer still goes out, because a miner that dies on a full
disk has turned a lost point into a lost session. `problem_id` arrives over the
network and is used to build a path, so it is sanitised as hostile input: it can
only ever name a file directly inside the archive directory.

### A Python answer can be cut off and still parse

`ast.parse` is Python's version of grepping for `fn main`. It is perfectly
happy with source that was **truncated**, because a reply cut at a statement
boundary is still a valid module. Two archived answers ended deep inside a loop
with no return after them — one sixteen columns in, on a bare `break`. Both
parsed. Both were submitted. Both answered `None` on every hidden test, and
nothing anywhere noticed.

So the entrypoint is now asked the question a compiler asks a Rust function:
**can control reach the end of the body without returning?** Replayed against
25 real archived answers this flagged exactly those two and passed the other
twenty-three.

It is not only a truncation detector. The grader compares *return values*, so a
function that runs off its own end answers `None` — which is wrong for almost
every task, and is exactly the `n = 0` case the edge-case checklist spends six
lines asking about:

```python
def solve(xs):
    if xs:
        return max(xs)      # ← answers None for the empty list
```

The analysis is conservative in the cheap direction. A `for` loop is never
treated as guaranteeing a return even when it obviously does, because being
wrong here costs one repair round while missing a truncated answer costs the
whole solve — and a flagged answer is still submitted if the repair produces
nothing better.

### Rust answers are compiled before they are sent

Python's structural check PARSES the source — `ast.parse` rejects prose, a
shell command, a truncated line, anything that is not a program. Rust's greps
it for `fn main`. That asymmetry is why **every answer this miner has destroyed
in transit was a Rust one**: a model's reasoning, a tool call, and the miner's
own prompt all contain the characters `fn main`, and all three have been
submitted as programs.

So when a local `rustc` is available, the candidate is built with the
validator's own flags (read from `RELEASE_POLICY`, not copied) before it is
sent. Replayed against a real archive of 45 submissions — 18 of them Rust —
this rejected exactly the six that would not build and passed all twelve that
would:

```
REJECT  it does not compile: error: unknown start of token: \u{2014}      (prompt echo)
REJECT  it does not compile: error: unknown start of token: \u{2014}      (prompt echo)
REJECT  it does not compile: error: expected item, found `{`             (tool call)
REJECT  it does not compile: error: this file contains an unclosed delimiter   (truncated)
REJECT  it does not compile: error[E0425]: cannot find function `failed_any`   (model bug)
REJECT  it does not compile: error[E0503]: cannot use `head` ...               (model bug)
```

Three of those would otherwise have reached a validator as programs. The other
three become a repair round instead of a certain zero.

It matters most when there is nothing else. With no public examples shipped —
which was every task on the run this was written for — the grader never runs at
all, so this is the *only* check a Rust answer gets before submission.

A local toolchain can differ from the pinned one, and the failure mode of that
is deliberately cheap: a wrong defect costs a repair round, never the answer,
because a defective non-empty candidate outranks an empty one and is still what
gets submitted if the repair produces nothing better. No compiler means no
opinion — silence has to mean the same thing as success, or a missing toolchain
becomes an outage. `SOLVER_RUST_COMPILE=0` turns it off. The candidate is
compiled, never run.

### Correction runs until it passes, on the only deadline there is

Two private deadlines used to sit under the validator's. The loop stopped at
`SOLVER_MAX_ATTEMPTS = 3`, and each round was handed a *fraction* of what was
left — 60% to the first attempt with public examples, 85% without — so a later
round would have something to spend.

Both are gone. `SOLVER_MAX_ATTEMPTS` defaults to `0`, unlimited: a solve keeps
correcting until the answer verifies, until there is nothing left to act on, or
until a round trip no longer fits in the budget. And every round now reads
against **everything that is left**.

The reserve was worth least exactly where it cost most. `send` returns the
moment the model finishes, so holding budget back was never a wait — only a
ceiling on a read that ran long, which is the one case where cutting it short
throws the answer away. Measured: a tab spent its whole 135-second slice while
the remaining 90 seconds of a 225-second budget went unused, on the one attempt
that had to succeed.

`SOLVER_SAFETY_MARGIN_S` stays, because it is not a deadline on the model: it is
the time the answer needs to be graded, archived, signed and put on the wire
before the validator stops listening.

### A repair carries the error and nothing else

The correction prompt is the evidence plus one sentence naming what may come
back:

```
I ran `solve` against the test cases you sent and got:
  - case 2 'single digit': solve(*[7], **{}) returned 0, expected 7

Send back ONE fenced block: the corrected program — or, if the case was wrong
rather than the program, a `json` array holding ALL of the cases, corrected.
```

What it no longer carries is method — *"trace the failing call through your
code"*, *"do not guess at the fix from the shape of the failure"*, *"re-check
the fix against every OTHER case you were sent, silently"*, *"Do not change both
to make them agree"*. That is work which never reaches the reply, competing with
the failure itself for attention, and it is the same class of instruction the
two-phase rewrite already took out of turns 1 and 2.

Nothing is conceded by dropping *"do not change both"*: the grader keeps that
promise rather than the prompt asking for it. A reply that rewrites the program
**and** the cases is graded against the bar as it stood before it arrived, and a
revision that drops cases is refused outright.

The validator's own examples are the one place a case cannot be corrected — they
shipped with the task and are ground truth — so that branch asks for the program
alone.

### Never give up on a task while an answer is still obtainable

An empty answer and a wrong answer pay exactly the same: nothing. `all_passed`
is a hard gate in `rlvr/scoring/payment.py`, checked *before* the speed
multiplier is ever computed. Above that gate speed is worth almost nothing —
the multiplier is `floor + (1 - floor) · 2^(-delay / half_life)` with
`speed_floor = 0.95`, so the slowest correct answer still earns 95% of what the
fastest earns, and one arriving at six minutes earns 96%. Correctness is worth
100%. Speed is worth at most 5%.

The miner used to give up long before the validator did, three ways, and a live
run cost two whole tasks to the combination:

**The budget cap bound below the advertised deadline.** `SOLVER_MAX_BUDGET_S`
was 240 against the 300-second deadline this subnet advertises, so the budget
was 225s and the first read got `225 × 0.85 = 191s`. A model still writing at
191s had its answer discarded *here*, by this miner, not by the validator that
would have paid 96% for it. The cap is now a runaway guard that does not bind:
the budget comes from the deadline the validator actually advertised, minus the
margin. `min()` is what makes raising it safe — a validator advertising 60
seconds still gets 60 seconds.

**A tab that could not be read was polled to the deadline.** Two shapes of the
same failure, both from one run:

```
chatgpt tab 9227: no assistant selector matched anything
claude  tab 9222: matched 1 message(s), the same as before the prompt was
                  sent — the answer never rendered
```

Each burned the full first slice and then a repair round on the same dead
conversation. A working tab paints its assistant bubble within a second or two
of the submit, empty, and fills it in after — so "has the reply appeared at
all" is a different question from "has it finished", and only the first one
separates an unreadable tab from a slow model. A tab that renders nothing for
`BLIND_TAB_GRACE_S` (30s) stops being polled and the rest of the budget goes to
another tab. A model that is merely thinking is never touched: its bubble is
already on screen.

It stops being *polled*, not read. The copy control and the network stream run
first, and only then is the tab retired — because both read the answer by other
means than the selector that just failed. The stream especially: it is captured
off the wire by CDP and has never touched the DOM, so a selector matching
nothing says nothing at all about it. `_reconcile_stream` was written for
exactly this case and names it — *"a selector that stopped matching, a render
this tab cannot see"* — and nine bounded seconds there is the difference
between a zero and the whole payment.

`CHATGPT_ASSISTANT` also had exactly **one** candidate, which is one deploy
away from the same failure across every tab at once. It now has three, and
Claude has had three since it was written. This is the one role whose failure
is total — a composer or copy control that stops matching degrades, an
assistant selector that matches nothing reads no answer at all, for every task,
until somebody notices. Every candidate on both sites is assistant-only *by
construction*, and a test asserts none of them matches a user turn: a candidate
that did would have the miner read its own prompt back and submit it, which is
a silent permanent zero.

What the fallbacks are *not* is a fix for tab 9227 specifically — five other
ChatGPT tabs answered that same minute with the same selector, so that page was
in a bad state rather than a new shape. The wire recovery covers both; the
fallbacks are insurance against the shape changing under all of them at once.

### The request's deadline is the only deadline

`TaskRequest.deadline_s` is `Field(gt=0.0, le=3600.0)`, so every cap the miner
keeps sits at **3600** — at the protocol ceiling, where it is structurally
incapable of binding on a spec-compliant request. `min(deadline_s, 3600)` is
`deadline_s`, always.

Anything smaller is a second, *private* deadline, and a private deadline
silently costs the difference the moment a validator advertises more than it.
That has already happened twice here: `SOLVER_MAX_BUDGET_S=240` against a 300s
deadline cut the first read from 238s to 191s, and `GLM_REQUEST_TIMEOUT_S=280`
cut the whole solve by 20 seconds. Neither was visible as anything except
answers that arrived unfinished.

What remains is not a deadline but the **cost of delivering**:
`SOLVER_SAFETY_MARGIN_S` covers `send`'s post-read phases (5 + 4 + 2 = 11s) plus
grading, archiving, signing and transmission. The budget is the advertised
deadline minus that, and nothing else.

### The model writes the tests, because nobody else does

Live traffic ships **no** `public_examples`. Measured over one run, 56 solves in
a row reported `examples=0/0` — so the repair loop, the one mechanism here that
turns a nearly-right answer into a right one, never had anything to run, and
`verified` was False on every answer because nothing *could* be checked rather
than because anything was wrong.

The only source of cases is the model. So the output contract now asks for two
blocks: the program, then a `json` array of the cases it traced.

```json
[{"name": "empty input",  "args": [[]],        "expected": 0},
 {"name": "single item",  "args": [[5]],       "expected": 5},
 {"name": "stated bound", "args": [[1000000]], "expected": 1000000}]
```

Those run through **the validator's own executor**, and a disagreement fires the
existing repair round. End to end, on a program with the classic `while n > 9`
bug — right for 12345, wrong for 0 and every single digit:

```
[verify] python entrypoint=g examples=0/0 self=3/3 verified=False
  repair: case 2 'single digit': g(*[7], **{}) returned 0, expected 7
  submitted: while n > 0   ← the corrected program
```

**What this does not do is verify anything.** A model cannot confirm its own
reading of a statement, and cases that encode the same misreading agree with the
code. That class stays uncaught. What it catches is the commoner one by far —
the model knowing what the answer should be and coding it wrong — and that is
objectively checkable.

### Turn 2 is asked to pass both suites, so both are run

The validator's examples and the model's own cases are both in the turn-2
prompt — `<examples>` and `<must_pass>`. Only one of them used to be executed:
the self-test path sat under `if not task.public_examples`, so a task that
**shipped** examples had its own cases quoted and never run. A program right on
the one example and wrong on its own boundary cases verified, ended the loop and
shipped, and the repair round that exists to catch precisely that never fired.
Live traffic ships no examples, which is why it went unnoticed rather than why
it was harmless.

The **order** is the whole of the precedence:

1. The validator's examples first. They shipped with the task and are ground
   truth — when they fail, the program is wrong, there is nothing to weigh, and
   the own cases are not run at all. A second opinion from the same model on a
   program already known wrong tells you nothing and costs an executor run per
   case.
2. Only once those are all green does a disagreement with the model's **own**
   cases become the open question, and `failures` carries that instead.

So `failures` names one suite at a time and `from_self_tests` says which — which
is exactly what lets the repair prompt offer a corrected `json` array where a
case may be wrong, and refuse to where the examples are ground truth.

`verified` still means what it always meant: every public example reproduced. It
can now be True on an answer that still disagrees with the model's own cases, so
the answer **cache** is gated on `not failures` as well — caching one of those
re-serves a wrong answer for every later task with the same statement.

So `self_passed` is kept in its own field. It never touches `passed`/`total`,
`verified` is earned by the validator's examples alone — with none shipped it
stays False however many of its own cases a program passes. The answer cache is
gated on `verified` **and** on having no outstanding failures, so an answer that
cleared the examples and still disagrees with its own cases is not cached
either: one wrong answer must not be re-served for every later task with the
same statement.

Two rules keep the bar honest, and they are the same rule at two moments:

- **The program turn cannot bring its own bar.** Cases written beside a program
  are back-filled from what it happens to do, so they agree with its bugs. A
  `json` block in turn 2 is refused and turn 1's cases stand. Measured before
  this existed: turn 1 wrote a case that *caught* the bug, turn 2 sent the buggy
  program with two cases of its own, round 1 reported the real failure and then
  adopted them, and round 2 re-graded the same buggy program against the bar it
  had brought with it — `self=2/2`, no failures, loop over, buggy program
  submitted as passing everything.
- **A repair cannot pass a bar it rewrote in the same breath.** A reply that
  changes the program *and* the cases is graded against the bar as it stood
  before it arrived; its cases apply from the next round.

A repair reply that corrects a case and leaves the program alone is the one the
prompt asks for, and it may arrive **without the program** — a `json` array and
nothing else. The program already in hand is then re-graded against the
corrected bar. Re-graded, not assumed to pass: there is no rewrite to launder
because there is no new program.

The repair wording changes with it. Not *"your solution is WRONG"* — these cases
came from the model, so a disagreement proves only that two things it wrote
contradict each other, and blaming the code when the **case** was wrong is how a
repair round breaks a correct program:

> I ran `solve` against the test cases you sent and got: … Send back ONE fenced
> block: the corrected program — or, if the case was wrong rather than the
> program, a `json` array holding ALL of the cases, corrected.

A repair reply may carry a corrected `json` block beside the program, and that
is safe because `extract_code` picks the block that **defines the entrypoint**,
not the first or last one. Verified against the layouts models actually produce
— program first, cases first, an untagged JSON block, and a reply carrying only
cases, which still reports *"the reply contained no code"* exactly as before.

`SOLVER_SELF_TESTS=0` turns the whole mechanism off: one turn, no cases asked
for, and every answer submitted ungraded.

### The cases are written before the program exists

Asking for both in one reply has a flaw that no amount of prompt wording fixes:
a model writing cases **alongside** a program back-fills the `expected` values
from what the program happens to do. Those cases then agree with the program's
bugs, which is the one thing a test must not do. Cases written *first* cannot.

That is why the combined prompt is not kept as a fallback. It looked free — one
round trip instead of two, and cases where there would otherwise be none — but
what it produced was evidence the grader could not trust, paid for in output
tokens spent inside the deadline. When turn 1 fails in a way that belongs to the
task, the remaining passes ask for the program **alone** and the answer goes out
ungraded, which is the honest version of the same outcome.

So the solve spends a round trip on them, at every deadline:

```
turn 1  the cases, and explicitly NOT the program
turn 2  the program, with those cases restated as <must_pass>
turn 3+ repair, if the program disagrees with them
```

Turn 1 is asked for the shape the classes actually break on, with counts stated
**per class** rather than as a total — a total invites a model to spend it all
on the easy classes:

```
1. THREE ordinary cases         the common path, which an all-boundary
                                suite never checks at all
2. then 2–10 cases per class:   zero and the empty input · one and two ·
                                every limit the statement names, at it,
                                one below, one above · negatives ·
                                largest/smallest, and i32 overflow ·
                                ties and duplicates · degenerate shape ·
                                each rule fired and NEARLY fired ·
                                the case you are most likely to get wrong
```

Four things make the extra round trip safe rather than a second way to run out
of time:

- **Turn 1 has no timeout of its own.** There is one clock on a solve — the
  deadline the validator advertised — and turn 1 reads against it like every
  other read here. It carried a private cap twice and both were wrong the same
  way: the first set `extend_to_s` equal to the slice, which makes the extension
  a no-op by construction; the second used a soft 60s slice and a hard 100s cap.
  Both cut the model off **mid-think**, which is the one moment where stopping
  cannot help — the reply does not exist yet, so the cap saves time that bought
  nothing and costs the whole turn.

  A cap *looks* like it protects the program's read and does not. `send` returns
  the moment the model finishes, so the slice is a ceiling and never a wait: a
  cases turn that takes 90 seconds hands the program the other 190 whether or
  not a cap exists. The only case a cap changes is the one where the model has
  not finished — and there it converts a slow answer into no answer, the one
  trade the payment policy says never to make.

  Live, on a hard Rust task: `Thought for 1m 17s` before a character appeared,
  against a 60s cap. Turn 1 timed out, the conversation was unusable, the pass
  was handed on — four times, 189 seconds, `provider=none`, the program never
  asked for once. The same task before the split got one 238s read.
- **A cases turn that costs a pass does not cost every pass.** Whatever makes
  turn 1 unaffordable — a long thinking phase, a slow account, a hard problem —
  belongs to the task and the site, not to one tab, so the next pass would
  repeat it exactly. One pass may be spent finding that out; the rest go
  straight to the program, which is the half that pays. That, plus the budget
  itself, is what protects the program now that no cap does.
- **There is no deadline below which it is skipped.** There was — a 100-second
  floor, on the reasoning that a cases turn *costs* 20-30s of submit-and-settle
  before the model writes anything, which a 40s budget cannot spare. That
  priced the worst case into every solve: turn 1 is a ceiling, not a spend, so
  a fast cases turn on a short deadline costs what it took and the program
  gets the rest.
- **A turn 1 that produces nothing still gets a program.** No usable cases is
  not a failure; it costs the time it took and turn 2 goes out regardless.
- **A turn 1 that could not be *read* is handed on; one that ran the deadline
  out is not.** They are not the same failure, and the clock tells them apart.
  A tab that dies leaves the budget intact and another tab can still spend it.
  A turn that ran the deadline out leaves nothing — handing on would lease a
  second tab, ask a second account for a whole program with seconds on the
  clock, and reach the same empty answer having spent someone's quota to get
  there. So that one stops.
- **A wrong case can still be corrected.** Turn 1 derives its `expected` values
  by reasoning, so one of them can simply be wrong — and freezing them would
  make a **correct** program fail the same bogus case on every repair round.
  The repair prompt offers a corrected `json` array as one of the two things it
  will accept back,
  and a repair reply carrying a corrected array replaces the frozen one for the
  *next* round. Not the round that carried it: a reply is graded against the
  cases agreed before it arrived, so a model cannot make its program pass by
  rewriting the bar in the same breath.

`MAX_SELF_TESTS = 20` is the cap the model is told about *and* the cap the miner
enforces, and over-long arrays are thinned by keeping the first three and then
striding — a head-slice would keep the three ordinary cases and throw away every
boundary, discarding exactly what the mechanism exists to run.

### A short deadline must still get an answer

`TaskRequest.deadline_s` is only `Field(gt=0.0, le=3600.0)`. Nothing in the
protocol promises the comfortable numbers this subnet happens to send today,
and at the small end the arithmetic used to invert — three separate floors,
each sensible on its own, combined into a guaranteed total loss:

```
deadline  budget   asks?   read+tail   504 at   before
     40    20.0     yes       28.5       40     ok
     32    12.0     NO        --         32     empty answer, model never asked
     20    10.0     NO        --         20     empty answer, model never asked
     15     7.5     yes       18.5       15     504, NO ANSWER
     10     5.0     yes       16.0       10     504, NO ANSWER
      5     5.0     yes       16.0        5     504, NO ANSWER
```

Three separate causes:

- **The post-read tail was a constant.** Eleven seconds is sized against the
  safety margin, which is sized against a 300-second deadline. Below ~16s the
  tail alone outlived the whole request. It is now `min(11s, read / 2)` — a
  rescue that costs more than half again what the attempt cost has stopped
  being a rescue — and at any read of 22s or more it is still the full eleven,
  which is every read on the deadlines seen in production.
- **`_attempt`'s "not worth another round trip" guard gated the FIRST trip.**
  Below a 32-second deadline the budget lands under twelve seconds and the
  model was never asked at all — an empty answer with no log line to say why.
  The first attempt now always runs, exactly as the first pass does in
  `solve_task`.
- **Getting the prompt in had one too.** `max(5.0, …)` gave a one-second read a
  five-second submit. A submit that fills the whole read is a read that finds
  nothing, which is bad; a submit that outlives it is a solve that is
  cancelled, which is worse.
- **The short-deadline fallback had a five-second floor.** At a five-second
  deadline it budgeted the entire request and `handle_request` cancelled the
  solve mid-flight. A floor above what the caller can afford does not buy a
  longer read, it buys a cancelled solve — and a 504 is indistinguishable from
  a dead miner.

Every deadline from 2 seconds to 3600 now returns an answer. One second does
not, and cannot: `send` enforces a one-second minimum read.

### The model is NOT told how long it has, and that is the second answer

For a while it was. `<budget>` opened `<method>` with a real number of seconds,
on the reasoning that a model rationing an *unknown* deadline either hurries a
generous budget away or explores a tight one until the reply is cut off.

The reasoning was sound and the mechanism was not, because **a model has no
clock.** It cannot measure 260 seconds, cannot tell how many it has spent, and
cannot tell how many a paragraph will cost. Handed a number it can only act on
by guessing, and the only concrete behaviour the number ever drove was a depth
ladder — *"at least eight invented cases above 180s, at least five between 60
and 180"* — which is a count, not a duration. The count is what mattered; the
seconds were scaffolding around it.

So the number is gone from every prompt, and the classes of case are stated
outright in the turn that asks for them. That is both more precise and
checkable: a class can be compared against the work actually done, where a
duration a model cannot perceive can only be hoped at. One test walks every
prompt — both turns, both languages, both repair shapes, and the nudge — and
fails on any digit followed by a unit of time, with a single exception for the
per-test limit (`about 5 seconds`), which is a measured fact about the grader
rather than a budget to ration.

The same test now fails on the *vocabulary* of hurry as well — *deadline,
wall-clock, terse, fastest, 95%, latency* — because the number was only half of
it. Sentences like `"every character you emit spends wall-clock inside the
deadline"` and `"the payment rule pays the slowest correct answer at least 95%
of what the fastest one earns"` said the same thing without a digit in them, and
carried it into the one decision that matters: whether to reach for the safe
implementation or the clever one. Correctness pays everything here and speed
pays at most 5%, so the prompts say the first and no longer mention the second.

The miner still rations the deadline. It just does it in code, where there is a
clock — see *[The request's deadline is the only
deadline](#the-requests-deadline-is-the-only-deadline)*.

### How long the miner actually waits, and why it stops there

**280 seconds — 4m40s.** Not five or six minutes, and that is not a setting.
Everything above 300s belongs to the validator:

```
310s  validator stops listening        (live.py:148 — deadline_s + 10)
300s  miner answers 504 past here      (demo_miner.py:324 — wait_for)
 -11s send tail: copy 5 + stream 4 + postmortem 2, all AFTER the read
 -~9s grade, archive, sign, put on the wire
----
280s  the last moment a read can still end   ← SOLVER_SAFETY_MARGIN_S = 20
```

A model needing 300s misses by 20; one needing 360s misses by 80. The only
thing that would change this is a validator advertising a longer `deadline_s`,
which is its config and not the miner's — so the miner's own caps are set not
to bind if it ever does: both sit at 3600, the protocol's own maximum for
`deadline_s`, so `min(deadline_s, …)` is always `deadline_s`.

### The copy control wins on fidelity, never on completeness

`pre code` hands over the source *after* a syntax highlighter rebuilt it as
DOM; the copy control hands over what the model actually wrote. That is why the
copy is preferred — a highlighter once put U+E027, a Private Use Area character
present in no source file, inside a Python answer.

But the copy has no authority on *completeness*, and the code used to give it
that too. From a live run, on the only two tasks that spent the entire budget:

```
what the page RENDERS and what it COPIES are not the same — they differ at
character 1630: rendered '\n', copied nothing (it ends here). Using the copy.
```

Reproduced: the DOM held a complete 182-character Rust program, the copy control
gave the first 60, and **those 60 were submitted**. `rust_defect` returned `None`
on them — a truncated program keeps its `fn main` — so the structural check
passed and it reached the validator as a confident answer that cannot compile.

A copy clicked while the reply is still streaming is the beginning of the answer
and nothing else. The two readings were taken at different moments and the
shorter one is simply older, so when the copied source is the rendered source
**cut short** — a strict prefix, whitespace ignored — the fuller reading wins.
Anything else, including a difference in the middle (which is what a highlighter
artefact looks like), leaves the copy in charge exactly as before. A plain "is
it shorter" test would hand every highlighter bug back to the DOM.

The warning now carries the **amount** lost, and it is a count rather than a
once-per-tab flag. That distinction matters: a once-per-tab flag reported two
incidents across 56 solves on two tabs — exactly one per tab, which is what it
reports whether it happened twice or forty times.

### A model still writing is waited for, never interrupted

The slice a read is given is an internal **allocation**, not a deadline. Where a
caller does hand out less than the whole budget, stopping the read at the slice
is worth nothing on an answer that has not finished arriving — the composer is
usually disabled while a reply streams, and where it is not the prompt simply
queues behind the answer it is asking about.

So when the slice runs out with the model still writing, the read extends once
to the caller's real remaining budget rather than stopping to do something that
cannot help. Measured on a page whose model finishes at 4.8s against a 3s slice:

```
slice only:      read 3.0s, code = 'def g(n):\n    t = 0\n    while n > 0:'
slice + budget:  read 6.0s, the whole program
```

The truncated version is not merely worse, it is a certain zero **and** a wasted
repair round: `python_defect` reports *"expected an indented block after 'while'
statement"*, so the miner interrupts a model that is still writing to tell it
about a syntax error in a program it has not finished. Both shapes of that
mistake are now blocked outright — a conversation that comes back
`still_writing` is never sent another prompt, whether it captured nothing or a
fragment.

Knowing that a model is still writing used to rest entirely on the site's stop
control, and `usable_busy_selectors` **drops** any busy candidate that matches
an idle page at startup — so a site legitimately runs with none, and then the
signal reads False through the whole of an answer that is still arriving:

```
busy selector present:  empty_reason='unfinished'   (no repair)
busy selector dropped:  empty_reason='no-code'      (REPAIRED, mid-answer)
```

A message longer than it was one poll ago is being written, whatever the DOM
calls its stop button — so message growth is now the fallback, and it needs no
selector at all. It does not rescue everything: with no busy selector, a reply
whose *code* stops changing for two polls is still accepted as finished by the
settle test, which is the documented cost of losing that selector. It covers the
commoner shape by far — a model thinking in prose, where the read runs to its
deadline.

A reply that has **settled** still stops at its slice, because the reserve it
would otherwise eat is exactly what pays for the repair round that turns a
no-code reply into an answer.

**An empty capture was repaired like a wrong answer.** They are
indistinguishable at the point the decision is made — the structural checks
reject empty source, so an empty candidate carries a `defect` exactly like a
broken program does. So the tab now records *why* its last send came back
empty, because `""` is the same string for three different failures and only
one of them is worth another prompt in the same conversation:

| `empty_reason` | what happened | repair in place? |
|---|---|---|
| `unreadable` | no reply ever rendered, or the tab died | **no** — the prompt goes into a conversation that has just proved it cannot be read |
| `unfinished` | the model was still writing when the budget ran out | **no** — the prompt queues behind an answer that does not exist yet |
| `no-code` | a *finished* reply that simply had no code block in it | **yes** — the model broke the output contract, and telling it so is what fixes it |

Getting that distinction wrong costs a task in either direction, so it is not
inferred from the empty string: only the tab can see it, and it says so. A
backend that does not report one keeps the historic behaviour.

Together those left the two failed tasks with 5 seconds against a 20-second
second-opinion guard, while five healthy tabs sat idle. The pass loop now keeps
asking while it is holding **nothing** and the clock allows one more round trip
(`EMPTY_HANDED_FLOOR_S`, capped at `MAX_PASSES`). Holding an answer, the old
policy stands: one second opinion, because there is then something worth
submitting and each further ask spends a real account's quota.

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
| `SOLVER_MAX_ATTEMPTS` | `0` | Rounds per solve, `0` meaning **unlimited** — correct until it passes or the request's deadline stops it. A count here is a second, private deadline under the only real one, and there is no partial credit for stopping early. Set a number to cap it anyway |
| `SOLVER_SAFETY_MARGIN_S` | `20` | Headroom kept before the cutoff, for `send`'s post-deadline phases and for grading, signing and transport |
| `SOLVER_MAX_BUDGET_S` | `3600` | The protocol's own maximum for `deadline_s`, so it cannot bind on a spec-compliant request. Lowering it below the advertised deadline throws away answers the validator would still pay for |
| `SOLVER_VERIFY_EXECUTOR` | `subprocess` | Python grading backend; Rust always uses Docker |
| `SOLVER_EXECUTOR_RETRY_S` | `300` | How long an executor that could not be BUILT stays unavailable before a solve tries again. Without a daemon, building the Rust executor runs `docker info` — 60ms against a missing socket, up to 20s against a hung one — and it used to run once per Rust task, inside the solve's budget. A hold rather than a verdict: a daemon started after the miner is picked up on its own |
| `GLM_REQUEST_TIMEOUT_S` | `3600` | The deadline `handle_request` answers **504** at, `min()`-ed with the validator's own. Named for the reference miner's GLM client but applied to whatever solver is plugged in. `docs/DEMO_MINER.md` documents `280` for that miner; leaving `280` in your `.env` costs the browser solver 20 seconds of every solve |

`GET /solver-status` reports per-provider counters, fleet health, and a `rust`
section naming the two checks a Rust answer gets before it is submitted — the
local compile gate and the grading executor. It reports what is already known
and probes nothing, so polling it is cheap even when the Docker daemon is hung.
Watch it — a browser miner fails quietly, and silence looks identical to
success.

Both Rust checks are also probed once at startup, right after the fleet warms
up, because neither is otherwise looked at until a Rust challenge has already
been answered:

```text
[verify] rust: compile gate rustc at /usr/bin/rustc, grading ready
[verify] WARN: no local rustc and no working Rust executor, so a Rust answer is
         checked only by a grep for `fn main` before it is submitted.
```

The second line is the one to act on. Without a toolchain `compile_defect`
returns `None`, which means *could not tell* rather than *fine*, and without a
daemon nothing grades — so what is left is `rust_defect` looking for `fn main`
in a fenced block. Installing `rustc` restores the compile gate on its own and
needs no Docker; grading Rust needs the daemon.

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

## Trying it all without a wallet, off-chain

Nothing below touches the chain, and none of it needs a registered hotkey. The
miner's signing, the metagraph check and the axon only matter once a validator
is talking to you; every layer here runs before that, and `solvers.rehearse`
was written so the last one does too.

That is a property worth stating rather than assuming, so it is tested: the
suite runs a rehearsal with `bittensor`, `bittensor_wallet`,
`substrateinterface`, `fastapi` and `uvicorn` all blocked at the import hook,
and it still solves, archives and grades. `rlvr.protocol` falls back to an HMAC
identity when no chain stack is installed, and the rehearsal signs with that.

**1. System packages.** On a machine with nothing on it:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl
```

`python3-venv` and `curl` are the two that catch people: Ubuntu ships neither
on a minimal install, `python -m venv` fails without the first, and
`start_debug_browser.sh` refuses to start without the second (it uses `curl` to
tell whether the browser came up).

**2. A browser.** Prefer Google Chrome's `.deb` over Ubuntu's `chromium`:

```bash
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb
```

Ubuntu's `chromium` package is a **snap**, and a strictly-confined snap cannot
read dot-directories under `$HOME`. The profile path this script defaults to is
`~/.hone-miner/chrome/<port>`, which is exactly that — so the browser starts,
the CDP port never opens, and the script reports a timeout that looks like
nothing at all went wrong. If you would rather keep the snap, give it a profile
without the dot: `--profile ~/hone-chrome/9222`.

**3. The repository.**

```bash
git clone https://github.com/fnlich/hone-subnet.git
cd hone-subnet
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[miner,dev]'
pip install playwright          # no `playwright install` — you start the browser
```

The `chain` extra is deliberately not among these. Nothing below needs it.

**4. Code only, no browser.** If this fails, stop here; nothing downstream can
work.

```bash
python -m pytest examples/custom_miner
```

**5. Start a browser per account and sign in by hand.** This step carries the
real risk and it is the one nothing can do for you. The browser must be one YOU
started: a browser launched by an automation driver announces itself as one, and
providers refuse the sign-in.

```bash
cd examples/custom_miner
./scripts/start_debug_browser.sh --port 9222     # this one becomes Claude
./scripts/start_debug_browser.sh --port 9223     # this one becomes ChatGPT
```

Two windows open on your desktop. In the **9222** window go to
`https://claude.ai` and sign in; in the **9223** window go to
`https://chatgpt.com` and sign in. Prefer email plus a one-time code —
"Continue with Google" is the sign-in most likely to be refused.

One account per browser, never two in one. The ports keep the profiles apart,
which is the whole point: each browser holds one logged-in session, and the
miner picks a tab by which provider it wants.

Leave both running. The miner attaches and detaches over CDP and never closes
them, so restarting the miner keeps your logins. Each terminal stays busy —
open new ones, or add `&`.

**6. Tell the miner about both.** From the repository root:

```bash
cd /path/to/hone-subnet
cat >> .env <<'EOF'
CLAUDE_CDP=9222
CHATGPT_CDP=9223
EOF
```

`.env` is searched for upward from wherever you run, so one file at the
repository root serves the miner and every tool here. Confirm it took:

```bash
cd examples/custom_miner
python -c "from solvers.roster import roster, describe; print(describe(roster()))"
# 1 chatgpt, 1 claude
```

If that says `1 claude` only, the `.env` was not found or `CHATGPT_CDP` is
misspelled — the roster silently uses one default browser when nothing is set.

**7. Check the selectors, one provider at a time.**

```bash
python -m solvers.doctor claude --probe
python -m solvers.doctor chatgpt --probe
```

It reports, per role, which candidate selector your page actually has, then
sends a real prompt and shows the text it read back. `--probe` is the half that
matters: it drives the same read path the miner uses, so what you see is what
the miner would get. Run it against both, because the two sites fail
differently and a working Claude tells you nothing about ChatGPT.

**8. Solve one task, no miner, no wallet.**

```bash
python scripts/try_solver.py
```

Three outcomes, distinct on purpose: verified (the setup is good),
code-but-wrong (the plumbing works and the model missed — re-run before blaming
the setup), and nothing-came-back (it names login, selector and deadline in
likelihood order).

**9. Rehearse the whole miner.**

```bash
python -m solvers.rehearse
python -m solvers.rehearse --sample rust
```

This is the closest you get to being a miner without being one. The request is
signed and goes through `CustomMiner`'s own HTTP handler, so it exercises the
signature check, the concurrency slot, the deadline that answers 504 rather
than late, `fit_response`'s byte cap and the solution archive — and it grades
the answer against cases the model never saw. A good run ends:

```
[rehearse] SCORES: passed all 8 test(s)
[rehearse] checked against the full suite, including cases the model never saw
```

Exit `0` correct, `1` answered and wrong, `2` nothing could be concluded.

With two browsers the fleet holds four tabs (two per browser) and hands the
task to whichever is free, so consecutive rehearsals will not always use the
same provider. The `provider=` field on the `[verify]` line says which one
answered.

**9b. The real challenges.** `examples/sample_challenges/` holds five problems
that came from the subnet — two Python, three Rust, each a page of prose with
its edge cases stated rather than shown. They are the closest thing here to
what a validator actually sends, and the built-in samples are toys beside them.

```bash
python -m solvers.rehearse --challenge all           # all five, one fleet
python -m solvers.rehearse --challenge extent-journal
python -m solvers.rehearse --challenge sparse-circular-array reactive-stat-board
```

One browser fleet is opened for the whole batch and closed at the end — the
tabs are reused between challenges, which is what a miner does for its whole
life. Budget roughly `--timeout` seconds per challenge; the default 300 makes
`--challenge all` a 25-minute run at worst, so start it and come back.

It ends in one table, because several hundred lines of per-challenge output
scroll away:

```
[rehearse] summary
  FAIL  asset-rebuild-planner        python  passed 0/3 — plan_rebuild(*[{'a': 'A'...
  PASS  extent-journal               python  passed all 3 test(s)
  ????  reactive-stat-board          rust    it compiles locally; the tests could...
  PASS  revocable-verification-gate  rust    passed all 3 test(s)
  FAIL  sparse-circular-array        rust    it does not compile: error[E0308]: mi...

  2/5 would have scored
```

**How many cases the model is shown matters**, and it is a flag rather than a
default nobody reads. Each challenge ships three public cases. Show the model
all three and grade it on all three and the result is circular: the solver
repairs its answer until the public examples pass, so the grade at the end can
only agree with a check already made — it would report a success it was
incapable of failing. So the model is shown a subset and graded on everything:
two of three by default.

```bash
python -m solvers.rehearse --challenge all --examples 0
```

`--examples 0` shows none at all. That is not a handicap: on the run this miner
was built for, no public examples shipped with any task, and the entire repair
loop was dead code. It is the condition worth measuring against, and the
hardest one.

Rust needs Docker to run the cases. Without it you still learn whether each
answer compiles, which is the difference between a zero and a maybe — the table
says `it compiles locally` rather than claiming a pass.

**10. Read what it wrote.**

```bash
ls solutions/
cat solutions/rehearsal-python-1.py       # what a validator would have graded
python -m json.tool solutions/rehearsal-python-1.json | head -40
```

The `.json` holds the request and the reply side by side. A zero-byte `.py` is
not a bug: it is the record that the problem was seen and answered with
silence, which needs a different fix from never having seen it.

**11. Replay it whenever something goes wrong.** Once the miner has run for
real, every solve leaves one of these, and the rehearsal takes it back:

```bash
python -m solvers.rehearse --from solutions/<problem-id>.json
```

Same problem, same prompt, this time with you watching. It can only check
against the public examples — the hidden suite was never in the request — and
it says so.

### If a step fails

| what you see | what it means |
| --- | --- |
| `cannot attach to http://127.0.0.1:9222` | the browser is not running, or is on another port. Check `curl http://127.0.0.1:9222/json/version`. |
| `attached to the browser, but could not open https://claude.ai/new` | the browser is fine and the site is not reachable from it. Open the URL in that Chrome by hand; whatever it shows is the real problem. |
| `no composer selector matched` | almost always not signed in. Sign in by hand in that browser, then re-run the doctor. |
| `COULD NOT BE CHECKED: no browser to solve with` | same as the first row, reported by the rehearsal. |
| `DOES NOT SCORE: passed 7/8` | the miner works. The model got the problem wrong — which is the answer the rehearsal exists to give. |
| Rust says `COULD NOT BE CHECKED` | Rust verification needs Docker. Without it you still learn whether the answer compiles, which is most of the value. |
| `Chrome did not open a CDP port on 9222 within 20s` | on Ubuntu, usually the snap Chromium being unable to read `~/.hone-miner/` — see step 2. Otherwise the port is taken: `curl http://127.0.0.1:9222/json/version`. |
| the roster says `1 claude` when you configured two | the `.env` was not found, or `CHATGPT_CDP` is misspelled. With nothing set the roster falls back to one Claude browser on 9222, which looks like a working config. |

### Rust needs Docker, and it is worth having

Three of the five sample challenges are Rust, and Rust verification runs in the
validator's pinned container — there is no subprocess path for it. Without a
Docker daemon those three can only report whether the answer compiles, and
`--challenge all` ends with three `????` rows instead of three verdicts.

```bash
# Docker Engine from Docker's own repository, not the distro's docker.io
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
     -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# Run it without sudo — the miner shells out to `docker` as your user
sudo usermod -aG docker "$USER"
newgrp docker            # or log out and back in

docker run --rm hello-world
```

On Debian, replace `ubuntu` with `debian` in both URLs. The group change is not
optional: the executor runs `docker info` as the user the miner runs as, and a
daemon it cannot reach is reported exactly like a daemon that is not running.

Then **pull** the sandbox image. Do not build it:

```bash
cd /path/to/hone-subnet
docker pull "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.rust_image)')"
```

`scripts/build_rust_sandbox.sh` builds a LOCAL tag, `hone-rust-sandbox:rustc-1.89.0`,
and the executor never runs it. It runs `RELEASE_POLICY.rust_image` — a
digest-pinned `ghcr.io` reference that nothing overrides, because a validator
that graded on a locally-built image would not be grading what everyone else
grades on. The build script says so on its last line: *"A local image ID is not
a fleet identity."* It exists for publishing the image, not for using it.

Pull it **before** the first Rust rehearsal, not during. `docker run` would pull
it for you, and that pull happens inside the executor's own bounded subprocess —
the 60s compile timeout plus the per-case timeout plus slack, about 80 seconds
for three cases. A first pull is several hundred megabytes; over that budget the
run is killed, every case comes back failed, and an unfinished download reads as
`passed 0/3` — a wrong answer. The rehearsal now checks for the image first and
says so instead, but pulling it up front is still the thing to do.

Confirm it, the way the validator's own preflight does:

```bash
docker run --rm --pull=never --network=none --read-only \
  "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.rust_image)')" \
  sh -ec 'rustc --version; command -v python3; command -v timeout'
```

`./setup_validator.sh` does all of this and more, if you want the whole
validator environment rather than just Rust grading.

And a local compiler, which is a separate thing and cheap:

```bash
sudo apt install -y rustc
```

`rustc` is not a substitute for Docker — it cannot run the cases — but it lets
the miner reject an answer that will not build instead of submitting it. On a
run where no public examples ship, that is the **only** check a Rust answer
gets before it goes to a validator.

Only after all of this is a hotkey worth registering.

## Testing your setup

Five layers, cheapest first, each isolating a different failure:

```bash
# from the repo root
python -m pytest examples/custom_miner    # 1. code only — no browser, no chain
python scripts/two_turn_demo.py           # 1b. see the two turns, still no browser

# from examples/custom_miner
cd examples/custom_miner
python -m solvers.doctor claude --probe   # 2. the browser's sign-in + selectors
python scripts/try_solver.py              # 3. a real solve, no wallet, no miner
python -m solvers.rehearse                # 4. the whole miner, as a validator
                                          #    meets it — archived and graded
                                          # 5. testnet, then finney
```

Layer 1 runs from the repo root; layers 2 to 4 run from `examples/custom_miner`
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

### Layer 4: rehearsing the whole miner

`python -m solvers.rehearse` answers a question layer 3 deliberately cannot.
`try_solver.py` tests the SOLVER — it never imports `custom_miner`, which is
what makes a failure there unambiguous. The rehearsal tests the MINER: the
request is signed and goes through `CustomMiner`'s own HTTP handler, so it
exercises the signature check, the concurrency slot, the deadline that answers
504 rather than late, `fit_response`'s byte cap and the solution archive — the
same objects, in the same order, that a validator's request meets. It
reimplements none of it, on purpose: a rehearsal that solved the problem its
own way would agree with the miner right up until the day they diverged.

Two other things it does that nothing else here does.

It writes the answer to `solutions/` through the miner's own `save_solution`,
so a rehearsal leaves the same evidence a live solve does — the code, the
request and the reply beside it, and a zero-byte file when the answer was
silence.

And it grades against a suite the model never saw. Every other check in this
repository compares an answer to the public examples, which the model was shown
and which it can pass while still scoring zero. The built-in samples keep cases
back — the empty list, `i64` where a model reaches for `i32` — so the run can
end with the thing you actually want to know:

```
[verify] python entrypoint=longest_run provider=claude examples=2/2 verified=True
[rehearse] DOES NOT SCORE: passed 7/8 — longest_run(*[[]], **{}) returned 1, expected 0
[rehearse] checked against the full suite, including cases the model never saw
```

That is a miner whose own verification is satisfied and whose answer is worth
nothing, which is not a state any other layer can show you.

Three sources, and the exit code says which of three things happened —
`0` correct, `1` answered and wrong, `2` nothing could be concluded (no Docker
for Rust, say), so a shell script can tell a broken miner from a machine that
cannot grade:

```bash
python -m solvers.rehearse                       # a built-in sample
python -m solvers.rehearse --sample rust
python -m solvers.rehearse --from solutions/<id>.json   # replay a real request
python -m solvers.rehearse --lease               # a real challenge
```

`--from` takes what `save_exchange` writes, so the natural thing to hand it is
the archived record of a solve that went wrong — the same problem, the same
prompt, this time with you watching. It can only check against the public
examples, because the hidden suite was never in the request, and it says so.

`--lease` wants what a VALIDATOR wants: a registered wallet and
`PROBLEM_SERVER_URL`. Leasing is the validator's side of the protocol, and it
consumes a real challenge — which is why nothing reaches for it unless you name
it.

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

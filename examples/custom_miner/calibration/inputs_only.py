"""Asking for the INPUTS alone: what it costs, and what it covers.

Measured, opus, six archived requests:

    05e07ccd (python) 19 inputs in 23.9s  ordinary 1 empty 2 one 2 boundary 3 likely-wrong 11
    07708c04 (rust  ) 20 inputs in 23.2s  ordinary 1 empty 1 one 1 boundary 8 likely-wrong  9
    07e57c11          reply did not parse as a json array
    0b8be61e (python) 20 inputs in 27.7s  ordinary 1 empty 1 one 1 boundary 6 likely-wrong 11
    0c58fbaa (rust  ) 20 inputs in 18.3s  ordinary 1 empty 1 one 1 boundary 9 likely-wrong  8
    10ecfb7e (python) 21 inputs in 26.3s  ordinary 1 empty 2 one 2 boundary 5 likely-wrong 11

    median 24.1s, 20 inputs
    class coverage: ordinary 5/5, empty 5/5, one 5/5, boundary 5/5, likely-wrong 5/5
    the SAME model asked twice picked the same input: median 10%

Three things, and the third is the one that matters.

CHEAP. 24s against the 65s a full cases turn takes on the same model for the
same twenty calls. Working out the `expected` values is about two thirds of
the turn; asking for the calls alone drops two thirds of the time.

COMPLETE. Every class in every problem, skewed hard to boundary and
likely-wrong -- eight to eleven of twenty -- with exactly one ordinary case,
which is what the prompt asks for. One reply in six did not parse as a JSON
array, so any production use needs a tolerant parser or a retry.

ARBITRARY. Asked TWICE, on the same problem, the same model reproduced a median
of 10% of its own choices. That is what killed the two-bar design in
two_bar_overlap.py, and it is not a disagreement between models: the input
space for "boundary cases for this statement" is enormous and the selection is
close to a coin toss. No design can use agreement about WHICH inputs to test as
a signal, from any pair of readers, including one reader with itself.

What follows is fixed_inputs.py: fix the calls at ONE source and the divergence
is gone by construction, because every reader is then asked about the same
calls.

    python -m calibration.inputs_only opus <request>.json ...
"""
import asyncio
import collections
import json
import os
import sys
import time
sys.path.insert(0, '/home/user/Scope/examples/custom_miner')
sys.path.insert(0, '/home/user/Scope')

from solvers.claude_cli import CliBackend
from solvers.prompts import MAX_SELF_TESTS

CLASSES = ("ordinary", "empty", "one", "boundary", "likely-wrong")

INPUTS_ONLY = """<output>
Reply with ONE fenced ```json block and nothing else: an array of objects
{{"class": "<one of ordinary|empty|one|boundary|likely-wrong>",
  "name": "<short label>", "args": [...], "kwargs": {{}}}}.
NO "expected" field. Do NOT work out what any of them should return.
</output>

<problem language="{lang}" entrypoint="{entry}">
{statement}
</problem>

<task>
List the INPUTS this problem must be tested on -- the calls, not the answers.
`args` is the argument list for `{entry}(*args)`; `kwargs` is optional. Every
value must be JSON.

Cover, in this order:
1. ONE ordinary input. A typical call, nothing special about it.
2. THE EMPTY VALUE, or zero: an empty list, an empty string, `0`, `{{}}` --
   whichever this statement allows.
3. ONE: a single element, `n = 1`, the smallest legal input.
4. THE BOUNDARY: every limit, threshold and modulus the statement names, AT
   that exact value, and the largest value it allows.
5. THE INPUTS THIS PROBLEM IS LIKELY TO BE GOT WRONG ON: ties and duplicates,
   every element equal, already sorted, exactly reversed, a rule the statement
   states given one input that makes it fire and one that NEARLY does.

Skip a class only when the statement makes it impossible. At most {limit}
inputs. You are NOT being asked what they return.
</task>"""

RUST_NOTE = ("\n\nThis problem reads stdin and writes stdout, so each `args` is "
             "a single-element list holding the whole stdin as ONE string.")


def parse(reply):
    i, j = reply.find('```'), reply.rfind('```')
    body = reply[i:j].split('\n', 1)[1] if i >= 0 and j > i else reply
    try:
        out = json.loads(body)
    except Exception:
        return None
    return out if isinstance(out, list) else None


def key(row):
    return (json.dumps(row.get("args", []), sort_keys=True, default=str),
            json.dumps(row.get("kwargs") or {}, sort_keys=True, default=str))


async def ask(backend, task, model):
    conv = await backend.open_profile(model, "low")
    try:
        p = INPUTS_ONLY.format(lang=task['language'], entry=task['entrypoint'],
                               statement=task['statement'].strip(),
                               limit=MAX_SELF_TESTS)
        if task['language'] == 'rust':
            p += RUST_NOTE
        t = time.monotonic()
        reply = await conv.send(p, 200.0)
        return parse(reply), time.monotonic() - t, len(reply)
    finally:
        await conv.close()
        backend.release()


async def main(paths, model):
    backend = CliBackend()
    times, counts, cover, stab = [], [], collections.Counter(), []
    for p in paths:
        task = json.load(open(p))['request']
        name = os.path.basename(p)[:8]
        try:
            a, ta, ca = await ask(backend, task, model)
            b, tb, _ = await ask(backend, task, model)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        if a is None:
            print(f"{name}: reply did not parse as a json array")
            continue
        cls = collections.Counter(str(r.get("class", "?")).lower() for r in a)
        for c in CLASSES:
            if cls.get(c):
                cover[c] += 1
        times += [ta, tb]
        counts.append(len(a))
        overlap = ""
        if b:
            ka, kb = {key(r) for r in a}, {key(r) for r in b}
            same = len(ka & kb)
            stab.append(same / max(1, min(len(ka), len(kb))))
            overlap = f"  run2={len(b)} inputs, {same} identical ({100*stab[-1]:.0f}%)"
        print(f"{name} ({task['language']:6s}) {len(a):2d} inputs in {ta:5.1f}s "
              f"({ca} chars)  classes={dict(cls)}{overlap}")
    if times:
        n = len(counts)
        print(f"\nmedian turn {sorted(times)[len(times)//2]:.1f}s   "
              f"median inputs {sorted(counts)[n//2]}")
        print("class coverage across problems: " +
              ", ".join(f"{c} {cover[c]}/{n}" for c in CLASSES))
        if stab:
            print(f"same input picked twice by the same model: "
                  f"median {100*sorted(stab)[len(stab)//2]:.0f}%")
    await backend.aclose()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[2:], sys.argv[1]))

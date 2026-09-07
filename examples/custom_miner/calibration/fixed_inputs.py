"""With the INPUTS fixed, two independent readers agree 94% of the time.

The counterpart to two_bar_overlap.py, which refused the two-bar design. That
design failed on input divergence -- two models asked to invent cases shared 5
inputs out of 233, and inputs_only.py then showed one model shares only ~10%
with ITSELF. Fixing the calls at one source removes that failure by
construction: every reader is asked about the same calls.

Measured -- opus writes the cases, then opus and sonnet each independently work
out what those same calls must return, neither having seen any program:

    05e07ccd (python) 15 inputs  readers [14s, 17s]  UNANIMOUS 15/15 (100%)
    07708c04 (rust  ) 10 inputs  readers [28s, 37s]  UNANIMOUS 10/10 (100%)
    07e57c11 (rust  ) 13 inputs  readers [21s, 51s]  UNANIMOUS 13/13 (100%)
    0b8be61e (python) 19 inputs  readers [37s, 54s]  UNANIMOUS 18/19  (95%)
    0c58fbaa (rust  ) 20 inputs  readers [40s, 68s]  UNANIMOUS 19/20  (95%)
    10ecfb7e (python) 20 inputs  readers [20s, 36s]  UNANIMOUS 16/20  (80%)

    TOTAL 91/97 = 94% agreed; 6 split

So the 6% that SPLIT is the informative part: those are the calls where two
independent readings of the statement differ, which is either an ambiguity or
one reader's error, and either way they are the calls a bar should not be
confident about.

COMPARED THE WAY THE GRADER COMPARES -- ordered tokens for Rust blobs, whole
floats as integers. An earlier pass used json.dumps and reported 48%, of which
every point of the missing 46 was whitespace spelling: one reader wrote
"OK 2 D 0 31" where the other wrote "OK 2\nD 0 31".

NOT MEASURED, and it is the question that decides whether this is worth
building: whether a bar carrying two agreeing readings is more CORRECT than
today's single-reading bar. That needs the hidden suite, which the miner never
sees. n is also six problems and 97 case-evaluations.

    python -m calibration.fixed_inputs opus opus,sonnet <request>.json ...
"""
import asyncio
import json
import os
import sys
import time
sys.path.insert(0, '/home/user/Scope/examples/custom_miner')
sys.path.insert(0, '/home/user/Scope')

from solvers.claude_cli import CliBackend
from solvers.prompts import build_tests_prompt, extract_self_tests


def norm(v, lang):
    if lang == "rust":
        return tuple(str(v).split())
    if isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, float) and v.is_integer():
        return json.dumps(int(v))
    if isinstance(v, (list, tuple)):
        return tuple(norm(x, lang) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, norm(x, lang)) for k, x in v.items()))
    return json.dumps(v, sort_keys=True, default=str)


def render(cases, entry, lang):
    out = []
    for i, c in enumerate(cases):
        if lang == "rust":
            out.append(f"{i}. stdin {json.dumps((c.get('args') or [''])[0])}")
        else:
            a = ", ".join(json.dumps(x) for x in c.get("args", []))
            k = "".join(f", {x}={json.dumps(y)}"
                        for x, y in (c.get("kwargs") or {}).items())
            out.append(f"{i}. {entry}({a}{k})")
    return "\n".join(out)


EVAL = """<output>
Reply with ONE fenced ```json block and nothing else: an array of objects
{{"i": <the number of the call>, "expected": <what it must return>}}, one for
every call listed. No prose, no code.
</output>

<problem language="{lang}" entrypoint="{entry}">
{statement}
</problem>

<calls note="Work out from the statement alone what each must return. You have
not seen any program and there is no reference answer.">
{calls}
</calls>"""


async def turn(backend, model, prompt):
    conv = await backend.open_profile(model, "low")
    try:
        t = time.monotonic()
        return await conv.send(prompt, 200.0), time.monotonic() - t
    finally:
        await conv.close()
        backend.release()


def parse(reply):
    i, j = reply.find('```'), reply.rfind('```')
    body = reply[i:j].split('\n', 1)[1] if i >= 0 and j > i else reply
    try:
        rows = json.loads(body)
    except Exception:
        return {}
    return {r['i']: r.get('expected') for r in rows
            if isinstance(r, dict) and 'i' in r}


async def main(paths, author, readers):
    backend = CliBackend()
    tot = unan = split = 0
    for p in paths:
        task = json.load(open(p))['request']
        lang, entry = task['language'], task['entrypoint']
        name = os.path.basename(p)[:8]
        try:
            reply, _ = await turn(backend, author, build_tests_prompt(
                lang, task['statement'], entry, task['public_examples']))
            cases = extract_self_tests(reply, entry, lang)
            if not cases:
                print(f"{name}: no inputs; skipping")
                continue
            calls = render(cases, entry, lang)
            outs = await asyncio.gather(*[
                turn(backend, m, EVAL.format(
                    lang=lang, entry=entry, statement=task['statement'].strip(),
                    calls=calls)) for m in readers])
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        views = [parse(r) for r, _ in outs]
        n = agree = 0
        for i in range(len(cases)):
            vals = [v[i] for v in views if i in v]
            if len(vals) < len(readers):
                continue
            n += 1
            agree += len({norm(v, lang) for v in vals}) == 1
        tot += n
        unan += agree
        split += n - agree
        secs = ", ".join(f"{t:.0f}s" for _, t in outs)
        print(f"{name} ({lang:6s}) {len(cases):2d} inputs | readers {readers} "
              f"[{secs}] UNANIMOUS {agree}/{n} "
              f"({100*agree/n if n else 0:.0f}%)")
    if tot:
        print(f"\nTOTAL {unan}/{tot} = {100*unan/tot:.0f}% of fixed inputs the "
              f"readers agreed on; {split} split")
    await backend.aclose()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[3:], sys.argv[1], sys.argv[2].split(',')))

"""Do two independently written bars agree about anything? Measured: no.

This is the experiment that refused a design, and it is kept so the design is
not proposed again.

THE PROPOSAL. Phase 1 writes the test cases and the program in two parallel
conversations that cannot see each other. The idea was to add a THIRD -- a
second bar, on another model -- and use agreement between the two bars as an
arbiter: a case both bars expect the same value for is AGREED and outranks the
program author's opinion; a case they expect different values for is CONTESTED
and means the statement is ambiguous there; the repair loop then routes on the
label instead of asking the program's model to adjudicate a case it never wrote.

THE RESULT, opus against sonnet over eight archived requests:

    05e07ccd (python) [opus 18 | sonnet 12]  shared= 0  AGREED= 0  CONTESTED= 0
    07708c04 (rust  ) [opus 16 | sonnet  8]  shared= 1  AGREED= 1  CONTESTED= 0
    07e57c11 (rust  ) [opus 12 | sonnet  7]  shared= 1  AGREED= 1  CONTESTED= 0
    0b8be61e (python) [opus 19 | sonnet 14]  shared= 1  AGREED= 1  CONTESTED= 0
    0c58fbaa (rust  ) [opus 20 | sonnet 13]  shared= 0  AGREED= 0  CONTESTED= 0
    10ecfb7e (python) [opus 20 | sonnet 20]  shared= 0  AGREED= 0  CONTESTED= 0
    11bd134f (rust  ) [opus 18 | sonnet 14]  shared= 2  AGREED= 2  CONTESTED= 0
    16077ad1 (rust  ) [opus 12 | sonnet 10]  shared= 0  AGREED= 0  CONTESTED= 0

    TOTAL shared inputs=5  AGREED=5  CONTESTED=0

Five shared inputs out of 233 cases written. "Write boundary cases for this
statement" has an enormous input space and two models pick different points in
it, so agreement BY INPUT essentially never happens. AGREED would be empty on
almost every solve, the routing would degrade to "everything is SINGLE" -- which
is what the miner already does -- and the second bar would cost a conversation
and about a fifth more output tokens to change nothing.

Worth keeping in view: of the five inputs both bars did pick, both agreed on
all five and contested none. Two independent readings that evaluate the SAME
input agree. They just do not choose the same input.

COMPARE THE WAY THE GRADER COMPARES, which the first pass at this did not. It
used json.dumps for both the key and the expectation, and for Rust that is
meaningless: `args` is one stdin blob and `expected` one stdout blob, and two
models spell the separators differently -- "OK 2 D 0 31" against
"OK 2\nD 0 31" -- so identical answers read as different. That pass reported
agreement of 0% on three problems, all of it whitespace. rlvr's own rust judge
compares ordered tokens ignoring whitespace runs, and so does this; floats that
are whole numbers are compared as integers for the same reason.

    python -m calibration.two_bar_overlap opus,sonnet <request>.json ...
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, '/home/user/Scope/examples/custom_miner')
sys.path.insert(0, '/home/user/Scope')

from solvers.claude_cli import CliBackend
from solvers.prompts import build_tests_prompt, extract_self_tests

def norm(v, lang):
    """One comparable form, by language."""
    if lang == "rust":
        return tuple(str(v).split())          # ordered tokens, whitespace-insensitive
    if isinstance(v, float) and v.is_integer():
        return json.dumps(int(v))             # 2.0 and 2 are the same answer
    if isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return tuple(norm(x, lang) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, norm(x, lang)) for k, x in v.items()))
    return json.dumps(v, sort_keys=True, default=str)


def key(case, lang):
    return (norm(case.get("args", []), lang),
            norm(case.get("kwargs") or {}, lang))


async def bar(backend, task, model):
    conv = await backend.open_profile(model, "low")
    try:
        r = await conv.send(build_tests_prompt(
            task['language'], task['statement'], task['entrypoint'],
            task['public_examples']), 200.0)
        return extract_self_tests(r, task['entrypoint'], task['language'])
    finally:
        await conv.close()
        backend.release()


async def main(paths, models):
    backend = CliBackend()
    tot_shared = tot_agreed = tot_contested = 0
    for p in paths:
        task = json.load(open(p))['request']
        lang = task['language']
        name = os.path.basename(p)[:8]
        try:
            got = await asyncio.gather(*[bar(backend, task, m) for m in models])
        except Exception as e:
            print(f"{name}: FAILED {e}")
            continue
        if not all(got):
            print(f"{name} ({lang}): a bar came back empty; skipping")
            continue
        A = {key(c, lang): norm(c.get("expected"), lang) for c in got[0]}
        B = {key(c, lang): norm(c.get("expected"), lang) for c in got[1]}
        both = set(A) & set(B)
        agreed = {k for k in both if A[k] == B[k]}
        tot_shared += len(both)
        tot_agreed += len(agreed)
        tot_contested += len(both) - len(agreed)
        print(f"{name} ({lang:6s}) [{models[0]} {len(A)} | {models[1]} {len(B)}] "
              f"shared={len(both):2d}  AGREED={len(agreed):2d}  "
              f"CONTESTED={len(both)-len(agreed):2d}")
    print(f"\nTOTAL shared inputs={tot_shared}  AGREED={tot_agreed}  "
          f"CONTESTED={tot_contested}")
    await backend.aclose()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[2:], sys.argv[1].split(',')))

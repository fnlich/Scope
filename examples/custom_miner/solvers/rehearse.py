"""Solve one real problem locally, exactly the way a validator would ask for it.

    cd examples/custom_miner
    python -m solvers.rehearse                       # a built-in sample
    python -m solvers.rehearse --sample rust
    python -m solvers.rehearse --from solutions/<id>.json
    python -m solvers.rehearse --lease               # a real challenge

The gap this closes is specific. `solvers.doctor --probe` proves the SELECTORS
work: it sends a toy prompt and reads the answer back. What it cannot tell you
is whether this browser, signed in to this account, on today's build of the
site, actually solves the kind of problem a validator sends -- with the real
prompt, the real budget, the real repair loop, and the real extraction. That
question has only ever been answerable by pointing a registered hotkey at the
subnet and watching the score, which is an expensive way to discover that the
answer arrives as prose.

So this drives the MINER'S OWN CODE, and that is the whole point of it rather
than an implementation detail. The request goes through `CustomMiner`'s HTTP
handler, so the rehearsal exercises the signature check, the concurrency slot,
the deadline that answers 504 rather than late, `VerifyingSolver`'s budget
split and repair rounds, the browser fleet, `fit_response`'s byte cap and the
solution archive -- the same objects, in the same order, that a validator's
request meets. A rehearsal that reimplemented any of that would stop testing
the miner the moment the two drifted, and would do it silently.

What is NOT the miner's own code is the verdict at the end: the miner never
learns whether it was right. Here the answer is put through the validator's own
executor afterwards, so the run finishes with the thing the operator actually
wants to know -- would this have scored.

The solution is written to the archive either way, by the miner's own
`save_solution`, including when it is empty. An empty file is the record that
this problem was seen and answered with silence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

if __package__ in (None, ""):  # `python solvers/rehearse.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlvr.protocol import TaskRequest, sign_message  # noqa: E402
from rlvr.types import TestCase  # noqa: E402

from solvers.config import find_env_file, load_env_file  # noqa: E402
from solvers.roster import build_solver, describe, roster, warm_up  # noqa: E402
from solvers.samples import SAMPLES  # noqa: E402


# --------------------------------------------------------------------------- #
# Where the problem comes from
# --------------------------------------------------------------------------- #
class Problem(NamedTuple):
    """One problem to rehearse, and what can honestly be said about it.

    `tests` and `tests_are` are separate because they answer different
    questions and only one source can answer both. A built-in sample has a
    hidden suite, so passing it means something close to scoring; an archived
    request or a live lease has only the examples the model was already shown,
    so passing them means considerably less. Reporting either as "passed" would
    be the same sentence for two very different results.
    """

    request: TaskRequest
    tests: list[TestCase]
    origin: str
    tests_are: str


def _from_sample(name: str) -> Problem:
    """A problem that ships with this example. No network, no wallet."""
    sample = SAMPLES.get(name)
    if sample is None:
        raise SystemExit(
            f"unknown sample {name!r}; have {', '.join(sorted(SAMPLES))}"
        )
    return Problem(
        sample.request(), sample.hidden_tests(), "a built-in sample",
        "the full suite, including cases the model never saw",
    )


def _from_file(path: str) -> Problem:
    """Replay a request the miner has already been sent.

    `save_exchange` writes one of these beside every answer, so the natural
    thing to hand this is the archived record of a solve that went wrong. The
    file may also be a bare `TaskRequest`, which is what a validator's own logs
    hold.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    body = raw.get("request", raw) if isinstance(raw, dict) else raw
    if not isinstance(body, dict):
        raise SystemExit(f"{path}: expected a JSON object holding a task request")
    # An archived record carries the answer too. Drop anything that is not part
    # of the request: `TaskRequest` forbids extras, and a confusing validation
    # error is a poor way to say "this file has more in it than a request".
    fields = set(TaskRequest.model_fields)
    request = TaskRequest.model_validate({k: v for k, v in body.items() if k in fields})
    # The hidden tests were never in the request -- the miner has never seen
    # them and neither has this file. The public examples are all there is.
    return Problem(
        request, list(request.public_examples), f"the archive at {path}",
        "only the public examples the model was already shown",
    )


async def _from_lease(insecure: bool) -> Problem:
    """Lease a real challenge from the problem server.

    Needs what a VALIDATOR needs -- a registered wallet and
    `PROBLEM_SERVER_URL` -- because leasing is the validator's side of the
    protocol, not the miner's. It also consumes a real challenge, which is why
    nothing here reaches for it unless it is asked for by name.
    """
    import bittensor as bt  # type: ignore[import-not-found]
    import httpx

    from rlvr.config import Settings
    from rlvr.problemserver.client import ProblemServerClient

    settings = Settings()
    url = settings.problem_server_url
    if not url:
        raise SystemExit(
            "--lease needs PROBLEM_SERVER_URL. Leasing is the validator's side "
            "of the protocol: it wants a registered wallet and a server that "
            "will answer it. Without one, use --from on an archived request or "
            "--sample."
        )
    wallet = bt.Wallet(name=settings.wallet_name, hotkey=settings.wallet_hotkey)
    print(f"[rehearse] leasing a challenge from {url}")
    print("[rehearse] NOTE: this consumes a real challenge lease.")
    async with httpx.AsyncClient() as http:
        client = ProblemServerClient(
            url, wallet, http,
            allow_insecure_http=insecure or settings.problem_server_allow_insecure_http,
            timeout_s=settings.problem_server_request_timeout_s,
        )
        outcome = await client.lease()
    if not outcome.leased or outcome.challenge is None:
        raise SystemExit(f"[rehearse] no challenge leased: {outcome.describe()}")
    challenge = outcome.challenge
    request = TaskRequest(
        problem_id=challenge.problem_id,
        language=challenge.language,
        statement=challenge.statement,
        entrypoint=challenge.entrypoint,
        public_examples=list(challenge.public_examples),
        deadline_s=challenge.deadline_s,
        prompt_variant=challenge.prompt_variant,
    )
    # The hidden suite is revealed only after a commit, which this deliberately
    # does not do -- a rehearsal must not enter the protocol on the validator's
    # behalf. So the examples are what can be checked.
    return Problem(
        request, list(challenge.public_examples), f"a live lease from {url}",
        "only the public examples the model was already shown",
    )


# --------------------------------------------------------------------------- #
# Solving it the way the miner does
# --------------------------------------------------------------------------- #
def _stand_in_validator():
    """Something that can sign like a validator.

    Resolved the way `rlvr.protocol` resolves it, and in the same order, so the
    request is signed with real sr25519 wherever the chain stack is installed --
    the case actually worth exercising -- and falls back to the same HMAC
    identity the protocol falls back to when it is not. Hard-importing one
    provider raised ModuleNotFoundError on a machine where a different one of
    the three was the one present; `sign_message` accepts a plain string id for
    exactly that reason.
    """
    for provider in ("substrateinterface", "bittensor_wallet", "bittensor"):
        try:
            keypair = getattr(__import__(provider), "Keypair")
            return keypair.create_from_uri("//Rehearsal")
        except Exception:  # noqa: BLE001 - try the next provider, then the string
            continue
    return "rehearsal-validator"


async def _answer(miner, request: TaskRequest, timeout_s: float):
    """Put the request through the miner's own HTTP handler.

    Signed, because that is how it arrives: this way the rehearsal also proves
    the signature path, the replay cache and the response cap, none of which a
    direct call to `solve()` would touch. The signer is a throwaway standing in
    for a validator; the miner is built without a wallet, so it accepts any
    valid signature rather than one addressed to a hotkey it does not have.
    """
    validator = _stand_in_validator()
    body = request.model_dump_json().encode()
    headers = sign_message(validator, body)
    headers["Content-Type"] = "application/json"
    started = time.monotonic()
    status, payload = await miner.handle_request(headers, body)
    return status, payload, time.monotonic() - started


# What the rehearsal concluded, and what each means for the exit code.
SCORED, FAILED, UNKNOWN = "SCORES", "DOES NOT SCORE", "COULD NOT BE CHECKED"


def _verdict(payload, request: TaskRequest, tests: list[TestCase]) -> tuple[str, str]:
    """Would this have scored? Run it through the validator's own executor.

    The miner never learns this, and that is exactly why a rehearsal should say
    it. A miner that returns a confident, well-formed, wrong answer looks
    identical from the inside to one that returns a right one.

    Three outcomes, not two. "I ran it and it is wrong" and "I could not run it"
    are different findings, and collapsing them is how a missing Docker daemon
    comes to look like a broken miner. But a Rust answer that will not COMPILE
    is the first kind however little else can be run -- it is not an unknown,
    it is a zero, and calling it unknown would hide the most definite failure
    there is behind a note about the operator's docker socket.
    """
    from solvers.verify import _Grader

    code = getattr(payload, "code", "") or ""
    if not code.strip():
        return FAILED, "nothing was submitted"
    if request.language == "rust":
        defect = _compile_defect_if_possible(code)
        if defect is not None:
            return FAILED, defect
    if not tests:
        return UNKNOWN, "no tests came with this problem to check it against"
    try:
        passed, total, failures = _Grader().check(
            code, request.language, request.entrypoint,
            [t.model_dump(mode="json") for t in tests],
        )
    except Exception as exc:  # noqa: BLE001 - a missing toolchain is not a verdict
        return UNKNOWN, f"the tests could not be run here — {_one_line(exc)}"
    if passed == total:
        return SCORED, f"passed all {total} test(s)"
    return FAILED, f"passed {passed}/{total} — {'; '.join(failures[:2])}"


def _compile_defect_if_possible(code: str) -> Optional[str]:
    """Why this Rust will not build, if a local toolchain can say so.

    Rust verification needs the pinned container, so an operator without a
    Docker daemon would otherwise learn nothing at all from a Rust rehearsal.
    A local compile is not the hidden suite, but "it builds" and "it does not
    build" are the two most different things a Rust answer can be, and the
    toolchain that tells them apart is usually right there.
    """
    from solvers.rust_compile import compile_defect, rustc_path

    if rustc_path() is None:
        return None
    return compile_defect(code)


def _one_line(exc: Exception, limit: int = 160) -> str:
    """An executor's complaint, cut to something an operator will read.

    A Docker daemon that is not running says so in about three hundred
    characters, twice, and printing all of it buries the verdict it is
    attached to.
    """
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


async def run(
    args: argparse.Namespace, solver_factory: Optional[Callable[[], Any]] = None
) -> int:
    """One rehearsal, end to end.

    ``solver_factory`` exists so the suite can drive this whole path without a
    browser. It defaults to the real fleet, which is the only thing an operator
    ever gets -- a switch that changed what is being rehearsed would defeat the
    purpose of rehearsing.
    """
    env_file = find_env_file()
    if load_env_file(env_file):
        print(f"[rehearse] loaded {env_file}")

    if args.source_file:
        problem = _from_file(args.source_file)
    elif args.lease:
        problem = await _from_lease(args.insecure)
    else:
        problem = _from_sample(args.sample)
    request, tests, origin = problem.request, problem.tests, problem.origin

    print(
        f"[rehearse] {request.language} problem {request.problem_id} from {origin}\n"
        f"[rehearse] entrypoint={request.entrypoint} "
        f"examples={len(request.public_examples)} deadline={request.deadline_s:g}s"
    )
    if args.statement:
        print("-" * 72)
        print(request.statement.strip())
        print("-" * 72)

    from custom_miner import CustomMiner
    from rlvr.neurons.demo_miner import DemoMinerSettings

    if solver_factory is None:
        browsers = roster()
        print(f"[rehearse] browsers: {describe(browsers)}")
        solver = build_solver(browsers)
    else:
        solver = solver_factory()
    settings = DemoMinerSettings(_env_file=None)
    # No wallet and no metagraph: this is a rehearsal, so there is no hotkey to
    # address the request to and no chain to authorize the sender against. The
    # handler's own checks cover both cases and are exercised as they are.
    miner = CustomMiner(settings, solver)
    try:
        try:
            await warm_up(solver, settings.miner_max_concurrent_requests)
        except Exception as exc:  # noqa: BLE001 - no fleet is not a wrong answer
            # The fleet says what is wrong and how to fix it -- `_fill` names
            # every browser it wanted and the flag to start them with. A
            # traceback on top of that buries the one line worth reading, and
            # an operator who has not started Chrome yet is the likeliest
            # person ever to run this.
            print(f"\n[rehearse] COULD NOT BE CHECKED: no browser to solve with.\n"
                  f"           {_one_line(exc, limit=400)}")
            return 2
        timeout_s = min(request.deadline_s, settings.glm_request_timeout_s)
        status, payload, spent = await _answer(miner, request, timeout_s)
    finally:
        await solver.aclose()

    print(f"[rehearse] the miner answered {status} in {spent:.1f}s "
          f"(its budget was {timeout_s:g}s)")
    if status != 200:
        print(f"[rehearse] FAILED: {payload}")
        return 1

    code = payload.code or ""
    print(f"[rehearse] submitted {len(code)} chars of {request.language}")
    if code.strip() and args.show > 0:
        lines = code.splitlines()
        print("    ----- what a validator would grade -----")
        for line in lines[: args.show]:
            print(f"    | {line}")
        if len(lines) > args.show:
            print(f"    | ... {len(lines) - args.show} more line(s)")
        print("    ----------------------------------------")
    else:
        print("[rehearse] the answer was EMPTY. The lines above this one say why.")

    from solution_archive import archive_dir

    where = archive_dir()
    if where is not None:
        print(f"[rehearse] archived under {where}/ (the code, and the exchange beside it)")

    verdict, why = _verdict(payload, request, tests)
    print(f"[rehearse] {verdict}: {why}")
    print(f"[rehearse] checked against {problem.tests_are}")
    # 0 answered and correct, 1 answered and wrong, 2 nothing could be
    # concluded -- so a rehearsal in a shell script can tell "my miner is
    # broken" from "this machine cannot grade Rust".
    return {SCORED: 0, FAILED: 1}.get(verdict, 2)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m solvers.rehearse",
        description="Solve one real problem locally, through the miner's own code.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--sample", default="python", metavar="NAME",
        help=f"a built-in problem: {', '.join(sorted(SAMPLES))} (default: python)",
    )
    source.add_argument(
        "--from", dest="source_file", metavar="FILE",
        help="replay an archived request (what `save_exchange` writes)",
    )
    source.add_argument(
        "--lease", action="store_true",
        help="lease a real challenge from PROBLEM_SERVER_URL (consumes one)",
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help="allow a plain-http problem server, for a local one",
    )
    parser.add_argument(
        "--statement", action="store_true", help="print the problem statement"
    )
    parser.add_argument(
        "--show", type=int, default=20, metavar="N",
        help="how many lines of the answer to print (default: 20)",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[rehearse] stopped")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

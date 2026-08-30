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
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Sequence

if __package__ in (None, ""):  # `python solvers/rehearse.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlvr.protocol import TaskRequest, sign_message  # noqa: E402
from rlvr.types import TestCase  # noqa: E402

from solvers.config import (  # noqa: E402
    apply_solve_timeout_default,
    find_env_file,
    load_env_file,
)
from solvers.roster import build_solver, describe, roster, warm_up  # noqa: E402
from solvers.challenges import load_all, names as challenge_names  # noqa: E402
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
    # The authored name of each case, when the source has them. `TestCase`
    # drops it -- it is not part of the wire contract -- and without it a
    # failure is a wall of arguments the reader has to decode to find out which
    # of three situations broke. The sample challenges name every case.
    case_names: tuple[str, ...] = ()


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


def _from_challenges(which: Optional[list[str]], shown: Optional[int],
                     timeout_s: float) -> list[Problem]:
    """The sample challenges as validator requests.

    Graded on ALL their cases while the model is shown only some. Showing all
    three and grading all three is circular -- `VerifyingSolver` repairs until
    the public examples pass, so the grade could only agree with the check
    already made, and would report a success it was incapable of failing.
    """
    problems = []
    for challenge in load_all(which):
        examples = challenge.shown(shown)
        held = len(challenge.cases) - len(examples)
        problems.append(Problem(
            TaskRequest(
                problem_id=challenge.name,
                language=challenge.language,
                statement=challenge.statement,
                entrypoint=challenge.entrypoint,
                public_examples=[TestCase(**case) for case in examples],
                deadline_s=timeout_s,
            ),
            [TestCase(**case) for case in challenge.cases],
            "the sample challenges",
            (f"all {len(challenge.cases)} public case(s), "
             f"{held} of which the model was not shown"
             if held else
             f"all {len(challenge.cases)} public case(s), every one of which the "
             f"model was shown — so this grade cannot fail; use --examples to hold "
             f"some back"),
            tuple(str(case.get("name", "")) for case in challenge.cases),
        ))
    return problems


def _from_archive(path: str) -> list[Problem]:
    """Every request under ``path``: one file, or a whole directory of them.

    A directory is the off-chain regression run. `save_exchange` writes one
    record per solve, so a directory of them is a corpus of exactly what this
    miner was asked in production -- the statements, the entrypoints, the
    deadlines, and the fact that no public examples shipped with any of them.
    Replaying it end to end answers the only question that matters between
    deploys: does this build produce an answer where the last one did not.

    Sorted by name, so two runs are comparable line for line.
    """
    where = Path(path).expanduser()
    if not where.is_dir():
        return [_from_file(str(where))]
    files = sorted(where.glob("*.json"))
    if not files:
        raise SystemExit(f"{where}: no .json requests in that directory")
    return [_from_file(str(f)) for f in files]


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


def _verdict(
    payload, request: TaskRequest, tests: list[TestCase],
    case_names: Sequence[str] = (), shown: int = 0,
) -> tuple[str, str]:
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
    builds = ""
    if request.language == "rust":
        defect = _compile_defect_if_possible(code)
        if defect is not None:
            return FAILED, defect
        absent = _rust_sandbox_missing()
        if absent is not None:
            return UNKNOWN, (
                f"the Rust sandbox image has not been pulled on this machine, "
                f"so the cases were not run.\n           Fix:\n"
                f"               docker pull {absent}\n"
                f"           Do NOT let the grader pull it: that happens inside "
                f"its own ~80s budget\n           (compile 60s + cases + slack), "
                f"and a first pull of a Rust image is several\n           hundred "
                f"megabytes -- over budget it fails every case and reads as a "
                f"WRONG ANSWER."
            )
        if _rustc_available():
            # Worth carrying into the message below. Without a Docker daemon
            # "could not be run here" is all an operator would learn from a
            # Rust rehearsal, and "it builds" is most of what they wanted.
            builds = "it compiles locally; "
    if not tests:
        return UNKNOWN, "no tests came with this problem to check it against"
    try:
        passed, total, failures = _Grader().check(
            code, request.language, request.entrypoint,
            [t.model_dump(mode="json") for t in tests],
            names=[
                # "(held back)" is the fact worth having in front of the reader.
                # A model that fails a case it was SHOWN has ignored its own
                # worked example; one that fails a case it was not shown has
                # simply never exercised that path -- opposite diagnoses.
                f"{name}{'' if index < shown else ' (held back)'}"
                for index, name in enumerate(case_names)
            ] or None,
        )
    except Exception as exc:  # noqa: BLE001 - a missing toolchain is not a verdict
        return UNKNOWN, (
            f"{builds}the tests could not be run here — {_one_line(exc)}"
            f"{_executor_hint(exc)}"
        )
    if passed == total:
        return SCORED, f"passed all {total} test(s)"
    return FAILED, f"passed {passed}/{total} — {'; '.join(failures[:2])}"


def _executor_hint(exc: Exception) -> str:
    """The fix, when the failure has exactly one.

    "permission denied ... /var/run/docker.sock" is not a broken install and
    not a stopped daemon -- it is a running daemon the miner's user is not in
    the group for, and it is the single commonest way a Rust rehearsal comes
    back unchecked. The raw error names the socket and never names the group,
    so an operator reads it as "Docker is broken" and reinstalls Docker.
    """
    text = str(exc).lower()
    if "permission denied" in text and "docker.sock" in text:
        return (
            "\n           The daemon is UP and your user cannot reach it. Fix:\n"
            "               sudo usermod -aG docker \"$USER\"\n"
            "               newgrp docker      # or log out and back in\n"
            "               docker run --rm hello-world"
        )
    if "docker" in text and ("not found" in text or "no such file" in text):
        return ("\n           Docker is not installed. See the README's "
                "\"Rust needs Docker\" section.")
    return ""


def _rust_sandbox_missing() -> Optional[str]:
    """The pinned sandbox image, if this machine has not pulled it yet.

    Returns None when it is present, when Docker cannot be asked at all, and on
    any answer other than a clear "no such image" -- claiming an absent image on
    an ambiguous error would mislabel a real failure as a missing download.

    Worth its own check because of what happens without one. `docker run` pulls
    a missing image, and that pull runs INSIDE the executor's own bounded
    subprocess: the Rust compile timeout plus the per-case timeout plus slack,
    about eighty seconds for three cases. A first pull of a Rust image is
    several hundred megabytes. Over that budget the run is killed, every case
    comes back failed, and an unfinished download is reported as a wrong
    answer -- the one thing a tool built to say "would this have scored" must
    never get backwards.

    `scripts/build_rust_sandbox.sh` does not help here and says so on its last
    line: it builds a LOCAL tag, while the executor runs a digest-pinned
    reference from `RELEASE_POLICY` that nothing overrides.
    """
    import shutil
    import subprocess

    from rlvr.policy import RELEASE_POLICY

    docker = shutil.which("docker")
    if docker is None:
        return None  # a missing Docker is the executor's story to tell
    image = RELEASE_POLICY.rust_image
    try:
        done = subprocess.run(
            [docker, "image", "inspect", image],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:  # noqa: BLE001 - an unanswerable Docker is not a verdict
        return None
    if done.returncode == 0:
        return None
    if "no such image" in (done.stderr or "").lower():
        return image
    return None


def _rustc_available() -> bool:
    from solvers.rust_compile import rustc_path

    return rustc_path() is not None


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
    # The same default the live miner applies, so a rehearsal reproduces the
    # budget a real validator request would get rather than a tighter one.
    apply_solve_timeout_default()

    if args.challenge:
        which = None if args.challenge == ["all"] else args.challenge
        problems = _from_challenges(which, args.examples, args.timeout)
    elif args.source_file:
        problems = _from_archive(args.source_file)
    elif args.lease:
        problems = [await _from_lease(args.insecure)]
    else:
        problems = [_from_sample(args.sample)]
    if not problems:
        raise SystemExit("[rehearse] nothing to rehearse")

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
    results: list[tuple[Problem, str, str]] = []
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
        for index, problem in enumerate(problems, 1):
            if len(problems) > 1:
                print(f"\n{'=' * 72}\n[rehearse] {index} of {len(problems)}: "
                      f"{problem.request.problem_id}\n{'=' * 72}")
            verdict, why = await _rehearse_one(miner, problem, settings, args)
            results.append((problem, verdict, why))
    finally:
        # The fleet is closed ONCE, after every problem. Opening browsers per
        # challenge would spend a minute of sign-in-warm page loads five times
        # over, and the tabs are designed to be reused -- that is what a miner
        # does for its whole life.
        await solver.aclose()

    if len(results) > 1:
        _summarise(results)
    # 0 everything scored, 1 something answered and was wrong, 2 nothing could
    # be concluded -- so a rehearsal in a shell script can tell "my miner is
    # broken" from "this machine cannot grade Rust". A mixed run reports the
    # worst outcome in it, because a run with one wrong answer in it is not a
    # passing run.
    verdicts = {verdict for _, verdict, _ in results}
    if FAILED in verdicts:
        return 1
    if UNKNOWN in verdicts:
        return 2
    return 0


async def _rehearse_one(miner, problem: Problem, settings, args) -> tuple[str, str]:
    """One problem, from the request going out to the verdict coming back."""
    request, tests = problem.request, problem.tests
    print(
        f"[rehearse] {request.language} problem {request.problem_id} from "
        f"{problem.origin}\n"
        f"[rehearse] entrypoint={request.entrypoint} "
        f"examples={len(request.public_examples)} deadline={request.deadline_s:g}s"
    )
    if args.statement:
        print("-" * 72)
        print(request.statement.strip())
        print("-" * 72)

    timeout_s = min(request.deadline_s, settings.glm_request_timeout_s)
    status, payload, spent = await _answer(miner, request, timeout_s)
    print(f"[rehearse] the miner answered {status} in {spent:.1f}s "
          f"(its budget was {timeout_s:g}s)")
    if status != 200:
        print(f"[rehearse] FAILED: {payload}")
        return FAILED, f"the miner answered {status}"

    code = payload.code or ""
    print(f"[rehearse] submitted {len(code)} chars of {request.language}")
    # Two questions, not one. Whether there IS an answer and whether it is
    # being printed are unrelated, and folding them into one branch made
    # `--show 0` announce "the answer was EMPTY" directly beneath "submitted
    # 197 chars of python".
    if not code.strip():
        print("[rehearse] the answer was EMPTY. The lines above this one say why.")
    elif args.show > 0:
        lines = code.splitlines()
        print("    ----- what a validator would grade -----")
        for line in lines[: args.show]:
            print(f"    | {line}")
        if len(lines) > args.show:
            print(f"    | ... {len(lines) - args.show} more line(s)")
        print("    ----------------------------------------")

    from solution_archive import archive_dir

    where = archive_dir()
    if where is not None:
        # Resolved, not as configured. `SOLVER_SOLUTION_DIR` defaults to the
        # relative "solutions", so the line used to read "archived under
        # solutions/" and leave the reader to work out which directory that was
        # relative to -- and the repository has a `solutions/` at its root as
        # well as the one this creates beside the miner. An operator went
        # looking in the wrong one.
        print(f"[rehearse] archived under {where.resolve()} "
              f"({request.problem_id}.{'rs' if request.language == 'rust' else 'py'}, "
              f"and the exchange beside it)")

    verdict, why = _verdict(
        payload, request, tests, problem.case_names, len(request.public_examples)
    )
    print(f"[rehearse] {verdict}: {why}")
    print(f"[rehearse] checked against {problem.tests_are}")
    return verdict, why


# How wide one summary row may be. Chosen to sit inside an 80-column terminal's
# usual two-line wrap rather than to match it exactly: the alternative is a
# detail cut so short it names nothing.
ROW_WIDTH = 118


def _fit(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _summarise(results: list[tuple[Problem, str, str]]) -> None:
    """One table at the end, because the per-problem output scrolls away.

    A run over five challenges prints several hundred lines and takes long
    enough that nobody watches it. The thing worth reading is which of them
    scored, and it must not require scrolling back through four other answers
    to find out.
    """
    width = max(len(problem.request.problem_id) for problem, _, _ in results)
    print(f"\n{'=' * 72}\n[rehearse] summary\n{'=' * 72}")
    for problem, verdict, why in results:
        mark = {SCORED: "PASS", FAILED: "FAIL"}.get(verdict, "????")
        # One line each, and the budget is the WHOLE line rather than the
        # detail alone -- the name column is as wide as the longest challenge
        # name, so trimming only the tail still wrapped. A failure detail runs
        # to several hundred characters (it names the arguments, what came back
        # and what was wanted) and five of those wrapped is not a table, it is
        # the scrollback the table was added to replace. The full text is
        # printed above, per problem.
        head = (f"  {mark}  {problem.request.problem_id:<{width}}  "
                f"{problem.request.language:<7} ")
        print(head + _fit(why, max(24, ROW_WIDTH - len(head))))
    total = len(results)
    scored = sum(1 for _, verdict, _ in results if verdict == SCORED)
    # Whether any tests EXISTED, not whether any verdict was reached. "nothing
    # was submitted" and "it does not compile" are findings that need no tests,
    # so counting verdicts would call a corpus gradeable on the strength of its
    # failures alone.
    if any(problem.tests for problem, _, _ in results):
        print(f"\n  {scored}/{total} would have scored")
    else:
        # Nothing here COULD score, and saying "0/97 would have scored" of a
        # corpus with no tests in it reads as a catastrophe rather than as a
        # missing yardstick. Live traffic ships no public examples -- all 97 of
        # the archived requests carry zero -- so this is the ordinary case for
        # a replay, not an edge one.
        print(f"\n  none of the {total} could be graded here: no tests came "
              f"with them")
    # Said either way, because it is the measurement a replay actually makes.
    # An empty answer is the failure this miner has most of: of 97 archived
    # solves, 32 submitted nothing at all. Whether a build changes that number
    # is the question a rehearsal over a corpus exists to answer, and it needs
    # no tests to answer it.
    empty = sum(1 for _, _, why in results if why == "nothing was submitted")
    print(f"  {total - empty}/{total} produced an answer"
          + (f"  ({empty} submitted nothing)" if empty else ""))
    broken = sum(1 for _, _, why in results if "does not compile" in why)
    rust = sum(1 for problem, _, _ in results
               if problem.request.language == "rust")
    if rust:
        print(f"  {rust - broken}/{rust} rust answer(s) compile"
              + (f"  ({broken} will not)" if broken else ""))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m solvers.rehearse",
        description="Solve real problems locally, through the miner's own code.",
        epilog="Exit 0 everything scored, 1 something was wrong, 2 nothing "
               "could be concluded.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--sample", default="python", metavar="NAME",
        help=f"a built-in problem: {', '.join(sorted(SAMPLES))} (default: python)",
    )
    source.add_argument(
        "--from", dest="source_file", metavar="PATH",
        help="replay an archived request (what `save_exchange` writes). A "
             "DIRECTORY replays every .json in it, sorted, one by one -- the "
             "off-chain regression run over a corpus of what this miner was "
             "actually asked",
    )
    source.add_argument(
        "--lease", action="store_true",
        help="lease a real challenge from PROBLEM_SERVER_URL (consumes one)",
    )
    source.add_argument(
        "--challenge", nargs="+", metavar="NAME",
        help="one or more of the sample challenges, or `all` for every one: "
             + (", ".join(challenge_names()) or "(none found)"),
    )
    parser.add_argument(
        "--examples", type=int, default=2, metavar="N",
        help="how many of a challenge's public cases the MODEL is shown; it is "
             "always graded on all of them. 0 reproduces the run this miner was "
             "built for, where no examples shipped at all (default: 2)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, metavar="S",
        help="deadline per challenge, as a validator would advertise it "
             "(default: 300)",
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
    parser.add_argument(
        "--solutions", metavar="DIR",
        help="where the answers are archived. The miner's own `save_solution` "
             "writes them, so this is `SOLVER_SOLUTION_DIR` -- which is "
             "RELATIVE TO THE WORKING DIRECTORY, and this package is run from "
             "examples/custom_miner, so the default lands beside the miner "
             "rather than at the repository root. An operator went looking in "
             "the wrong one; naming it here settles which",
    )
    parser.add_argument(
        "--log", metavar="FILE",
        help="also write everything printed to FILE, verbatim. The same lines "
             "the on-chain miner prints, because it is the same code printing "
             "them -- a run here is comparable to a run there line for line",
    )
    args = parser.parse_args(argv)
    archive_to(args.solutions)
    try:
        with _tee(args.log):
            return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[rehearse] stopped")
        return 130


def archive_to(directory: Optional[str]) -> None:
    """Point `save_solution` at ``directory``, or leave it where it was.

    `SOLVER_SOLUTION_DIR` is read per call from inside the solve, so setting it
    here -- before `run` -- is early enough. It is RELATIVE to the working
    directory, and this package runs from `examples/custom_miner`, so the
    default lands beside the miner rather than at the repository root. An
    operator went looking in the wrong one.
    """
    if directory:
        os.environ["SOLVER_SOLUTION_DIR"] = str(Path(directory).expanduser())


@contextlib.contextmanager
def _tee(path: Optional[str]):
    """Print to the terminal AND to ``path``, or just the terminal.

    Both, not either. A run over five challenges takes long enough that nobody
    watches all of it, and a log that only exists after the fact is no use while
    it is going wrong -- so the terminal keeps its live output and the file gets
    the same bytes for afterwards.

    Line-buffered and flushed per write, so a run killed half way through still
    leaves everything it had printed. The whole point of the file is the run
    that did not finish.
    """
    if not path:
        yield
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    class _Fork:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, text):
            for stream in self._streams:
                stream.write(text)
                stream.flush()
            return len(text)

        def flush(self):
            for stream in self._streams:
                stream.flush()

        def isatty(self):
            return False

    with target.open("w", encoding="utf-8") as handle:
        fork = _Fork(sys.stdout, handle)
        with contextlib.redirect_stdout(fork), contextlib.redirect_stderr(fork):
            print(f"[rehearse] logging this run to {target.resolve()}")
            yield


if __name__ == "__main__":
    raise SystemExit(main())

"""Grade your own answer with the validator's grader, then repair it.

The subnet pays only for a submission that passes the COMPLETE hidden suite —
a partial answer earns exactly as much as no answer at all. That makes a
one-shot "ask the model, return whatever it said" miner leave a lot on the
table: models routinely produce something that is nearly right.

Two facts turn that into an advantage:

1. ``TaskRequest.public_examples`` carries real ``{args, kwargs, expected}``
   cases, shipped with every task.
2. The comparison the validator will apply to you is IN THIS REPOSITORY —
   ``rlvr.execution.compare.values_equal`` for Python and
   ``rlvr.execution.rust_judge.outputs_match`` for Rust, reached through the
   same ``Executor`` the validator uses.

So a miner can run its own candidate through the validator's executor before
answering, and when a case fails, hand the model the concrete failure and ask
for a fix. Passing every public example is not proof of passing the hidden
suite, but it is a strong precondition and it eliminates the large class of
answers that are simply wrong on the stated contract.

A note on the executor: ``SubprocessExecutor`` is documented as dev-grade
because it does not confine untrusted code on Linux. Here the code being run
is YOUR OWN solver's output rather than an adversary's, which is the one
setting that backend is appropriate for. Set ``SOLVER_VERIFY_EXECUTOR=docker``
to use the container backend instead; Rust verification always requires it.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from rlvr.types import TestCase

from .prompts import (
    MAX_SELF_TESTS,
    build_code_prompt,
    build_repair_prompt,
    build_resume_prompt,
    build_tests_prompt,
    dropped_definitions,
    extract_code,
    extract_self_tests,
    python_defect,
    rust_defect,
)
from .rust_compile import compile_defect, rustc_path

# Per-example wall clock when checking our own candidate. Kept small: this is
# a smoke test against tiny public examples, not the real grading run.
VERIFY_TIMEOUT_S = float(os.environ.get("SOLVER_VERIFY_TIMEOUT_S", "5"))

# How long an executor that could not be built stays unavailable before the
# next solve is allowed to try again.
#
# A hold, not a verdict: a Docker daemon started after the miner is the
# ordinary case, and a permanent answer would mean grading nothing in that
# language until someone restarted the process. Long enough that Rust tasks
# stop paying for the probe, short enough that a box the operator has just
# fixed heals on its own.
EXECUTOR_RETRY_S = float(os.environ.get("SOLVER_EXECUTOR_RETRY_S", "300"))

# What a repair needs to be worth carrying to a FRESH conversation: a tab, a
# prompt that restates the whole problem, and a read long enough to answer it.
# More than the 12s an in-conversation round trip needs, because none of that
# is warm.
RESUME_FLOOR_S = 40.0

# What must be left when a read stops, for the answer to reach the validator at
# all. THE ONLY reserve taken out of the deadline, and the only thing standing
# between a full-budget solve and a 504 with nothing in it.
#
# `handle_request` wraps the WHOLE solve -- including `fit_response`,
# `save_solution` and `save_exchange` -- in one `asyncio.wait_for(...,
# timeout=min(deadline_s, GLM_REQUEST_TIMEOUT_S))`, and a solve that overruns it
# is cancelled and answered 504 with nothing. So everything after the last read
# has to fit in here: `send`'s post-read phases (copy, stream, post-mortem --
# `FULL_TAIL_S`, 11s, itself scaled down on short deadlines by `tail_budget`),
# then the archive writes and the response.
#
# There used to be a second reserve as well, `SOLVER_SAFETY_MARGIN_S` at 20s,
# which also held time back for GRADING. Two reserves for one deadline is one
# too many: the working budget was 20s short of the limit while the true limit
# was 12s short, and the 8s difference did nothing but complicate the
# arithmetic. One number, and it is the real one -- a 300s deadline now gives
# the reads 285s rather than 280s.
DELIVERY_RESERVE_S = 15.0

# The least a lease wait may be cut to. Below this, waiting is pointless and
# failing fast lets the pass end while another tab might still be tried.
OPEN_FLOOR_S = 5.0

# The most `_grade` will insist on before it declines to run anything at all.
GRADE_FLOOR_S = 15.0

# The least a correction round can be worth starting with: one prompt out, one
# reply back, and something read from the page at the end of it. Below this the
# loop stops and the last version in hand goes out as it stands.
ROUND_TRIP_FLOOR_S = 12.0

# Under this, a round did not involve the model. `send` blocks on a chat UI
# until the reply finishes -- tens of seconds, normally -- so a round that came
# back in under a couple of seconds read something that was already on the page.
# It is the difference between a model that answered the same way twice and a
# tab that is handing back the previous answer forever, and only the second is
# a reason to stop correcting.
STALE_ROUND_S = 2.0

# How many rounds running may correct only the CASES before the repair prompt
# stops offering that at all and asks for the program.
#
# The escape hatch is there because the model's own cases can be wrong -- turn 1
# reasons its `expected` values out before any program exists -- and a round
# that blames the code for a wrong case breaks a correct program. Taken twice
# running with the program untouched it is no longer that: the correcting is
# happening entirely on the bar, and a correction phase exists to make the
# PROGRAM more correct. Two, not one, because the first correction is the
# ordinary case this whole path was built for.
CASES_ONLY_ROUNDS = 2


class Conversation(Protocol):
    """One live, isolated model conversation.

    The repair loop deliberately stays inside a single conversation so the
    model can see its own previous attempt alongside the failure report.
    """

    # Two arguments, always. A backend MAY also accept ``extend_to_s`` -- a
    # hard bound past the slice, for a caller that hands out less than its whole
    # remaining budget -- but nothing here does, so nothing here passes one.
    async def send(self, text: str, timeout_s: float) -> str: ...
    async def close(self) -> None: ...


class Backend(Protocol):
    # ``avoid`` names a provider to steer away from, so a second attempt can ask
    # a different model. It is a preference, not a guarantee.
    async def open(self, avoid: Optional[str] = None) -> Conversation: ...
    async def aclose(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


@dataclass
class Answer:
    """What a solver hands back to the miner.

    Structurally compatible with ``custom_miner.SolveResult`` (the miner only
    reads ``.code`` and ``.raw_response``) and defined here on purpose: importing
    the miner module from inside a request would make a path problem surface as
    a failed solve at serving time rather than at startup.
    """

    code: str
    raw_response: str = ""
    # Whether this answer reproduced every public example. A chain of providers
    # needs this to know when to stop trying; the miner itself ignores it.
    verified: bool = False
    passed: int = 0
    total: int = 0
    # The same question asked of the model's OWN cases, which on live traffic is
    # the only suite that ever runs. Kept beside `verified` rather than merged
    # into it -- see `Candidate.self_verified`.
    self_verified: bool = False
    self_passed: int = 0
    self_total: int = 0


@dataclass
class Candidate:
    code: str
    raw: str
    passed: int = 0
    total: int = 0
    defect: Optional[str] = None
    failures: list[str] = field(default_factory=list)
    # Cases the MODEL wrote for its own program, kept apart from the validator's
    # examples on purpose. They are evidence, not verification: a model cannot
    # confirm its own reading of a statement, so folding these into
    # `passed`/`total` would let `verified` -- which gates the answer cache --
    # go True on nothing more than the model agreeing with itself.
    self_passed: int = 0
    self_total: int = 0
    from_self_tests: bool = False
    # How many of the model's own cases were IN HAND for this candidate,
    # whether or not there was time to run them. `from_self_tests` says they
    # ran; this says they existed. Keeping only the first made "no time to run
    # the cases" indistinguishable from "the model never sent any", and the
    # warning in `solve_task` said the second when the truth was the first.
    self_cases: int = 0
    # This reply is part of a program rather than a program: it uses something
    # only the round above it defined. Kept beside `defect` rather than folded
    # into it because `_supersedes` has to tell this apart from every other way
    # code can be wrong -- see there.
    partial: bool = False
    # The cases from `self_cases`' suite that this program did NOT pass. What
    # a correction round is allowed to change: see `_merge_cases`.
    failed_cases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Every public example reproduced exactly."""
        return self.defect is None and self.total > 0 and self.passed == self.total

    @property
    def self_verified(self) -> bool:
        """Every case the MODEL wrote for itself reproduced, and it is all the
        evidence there was.

        Deliberately not folded into `verified`, and the reason is unchanged: a
        model cannot confirm its own reading of a statement, so a program that
        agrees with itself must not be able to earn the flag that gates the
        answer cache and tells a chain of providers to stop trying.

        But it is not nothing, either -- it is the ONLY evidence a live solve
        ever has. Production ships no `public_examples` at all, so `verified`
        is False on every real answer this miner sends, and a log that reports
        only that cannot tell "ran every case it had and passed" from "was
        never run at all". Those are the two ends of the range, and they read
        identically. This is the one that says which.

        `total == 0` is part of it: with public examples in hand THEY are the
        verdict, and `verified` already reports it.
        """
        return (
            self.defect is None
            and self.total == 0
            and self.self_total > 0
            and self.self_passed == self.self_total
            and not self.failures
        )

    @property
    def score(self) -> tuple[int, int, int, int]:
        """Ranking key for 'best so far' — the validator's examples, then the
        model's own cases, then non-empty, then runnable.

        Used BETWEEN PASSES only, where two models answered the same problem
        independently and neither saw the other -- there, grading is the only
        thing that can separate them. Within one pass the rounds are corrections
        of each other and the latest simply wins; see `_supersedes`, and the
        damage this ranking did when it was applied there too.

        A defect ranks BELOW clean code that merely could not be graded, and
        that is not cosmetic. Without a defect term at all, a first answer with
        no `fn main()` and a corrected second answer score identically -- a tie,
        which `>` loses, so the repair round lands a good program and the broken
        one is submitted anyway. The whole repair loop is dead weight for
        structural defects until this ranks them apart.

        Non-empty comes BEFORE runnable, and the order is the whole point.
        Emptiness is not a defect -- there is nothing there to be wrong -- so
        with the terms the other way round a round that captured nothing at all
        outranks a round that returned a program with a fixable flaw, and
        replaces it as "best". Both score zero on chain, but one of them is
        still an answer and the other is the absence of one, and the answer is
        the one to keep: `python_defect` is a static check, and a static check
        that is too strict must not be able to throw work away.
        """
        # `self_passed` sits second: below the validator's own examples, which
        # are ground truth, and above everything structural, because a program
        # that reproduces cases it was checked against is better evidence than
        # one that merely parses. A candidate with no self-tests scores 0 there
        # and ties with one that failed all of them -- deliberately. Having
        # tests must not rank an answer below not having them.
        return (
            self.passed,
            self.self_passed,
            1 if self.code.strip() else 0,
            0 if self.defect else 1,
        )


def _case_key(case: dict) -> tuple:
    """What makes two cases the same case: the call, not the answer.

    A correction changes what a case EXPECTS. Keying on the input is what lets
    the corrected version be recognised as the same case rather than an
    additional one -- and it is why `expected` is deliberately absent from the
    key.
    """
    return (
        repr(case.get("args", [])),
        repr(sorted((case.get("kwargs") or {}).items())),
    )


def _merge_cases(
    agreed: list[dict], revised: list[dict], failed: list[dict]
) -> tuple[list[dict], str]:
    """The agreed suite with a correction applied. Returns (cases, what changed).

    A correction round happens because the program disagreed with a case, and
    only one of the two can be wrong. The repair prompt says so and offers both
    ways out: send the program back fixed, or -- if the CASE was the thing that
    was wrong -- send that case back corrected.

    What a correction may touch is exactly the cases the program FAILED. That
    single rule is what makes accepting a short array safe:

      * A case the program PASSES cannot be corrected, dropped or weakened. The
        bar a program has already cleared is not up for negotiation, so the
        obvious way to game this -- delete the case you cannot pass -- is not
        reachable from here.
      * A failing case may be corrected in place (same call, new expectation)
        or swapped for a different one, when the call itself was the thing that
        made no sense. Both are what "the case was wrong" means in practice.
      * The suite keeps its SIZE. Every failing case removed must be replaced,
        so a bar cannot be cleared by deleting what the program could not pass;
        and it cannot GROW here either, because cases written beside a program
        agree with its bugs and the program turn is already refused its own for
        that reason.

    Replacing the whole suite was the alternative, and refusing anything shorter
    was what came before. Both were wrong in the same place. Demanding the full
    array back meant a twenty-case suite was re-sent to correct one of them,
    which is slower, likelier to be truncated mid-array, and -- when it came
    back one case short for any reason at all -- refused outright, so the one
    wrong case broke a correct program on every remaining round of the solve.
    """
    if not revised:
        return list(agreed), ""
    out = {_case_key(case) for case in failed}
    keep = [case for case in agreed if _case_key(case) not in out]
    seen = {_case_key(case) for case in keep}
    merged = list(keep)
    for case in revised:
        key = _case_key(case)
        if key in seen:
            # It re-states a case the program PASSES. Not a correction this
            # round is entitled to make, and not a hostile act either -- a
            # model that re-sends its whole suite lands here on every passing
            # case. Keep the version that was agreed.
            continue
        merged.append(case)
        seen.add(key)
    if merged == agreed:
        return merged, ""
    if len(merged) != len(agreed):
        # The suite keeps its SIZE. Only the failing cases may change, and each
        # one has to be replaced rather than simply removed -- "the program can
        # drop the case it disagrees with and attach the corrected one in its
        # place", with the second half enforced.
        #
        # Both directions of this were measured breaking a solve.
        #
        # SHORTER is a bar cleared by deletion. A repair that came back with
        # only the cases it already passed dropped the one it did not, and a
        # program failing 1 of 2 went out reported `self=1/1`, verified on
        # local, on a bar it had rewritten in the same breath.
        #
        # LONGER is back-filling wearing a correction's clothes. Cases written
        # beside a program agree with its bugs -- which is the entire argument
        # for asking for them in a separate turn -- and a repair round that
        # re-sends the program with its own cases attached would otherwise get
        # them onto the bar by the side door the program turn is refused at.
        # Measured: a buggy program adding two cases of its own to a one-case
        # bar and finishing 2/3 instead of 0/1.
        return list(agreed), (
            f"REFUSED: {len(agreed) - len(keep)} case(s) failed and "
            f"{len(merged) - len(keep)} came back; a failing case may be "
            f"corrected, not dropped, and the suite may not grow here"
        )
    replaced = len(agreed) - len(keep)
    return merged, f"{replaced} failing case(s) corrected"


def _inherit_evidence(candidate: Candidate, prior: Candidate) -> None:
    """Give `candidate` what grading already established about the same source.

    Evidence belongs to the CODE, not to the round that happened to run it, and
    two rounds carrying byte-identical source are the same program however
    differently they were read. Not a ranking, and not in tension with "the
    latest version wins": the answer that ships is unchanged either way. What
    changes is what is known about it.

    Both losses this repairs were measured.

    A round that arrives with the budget gone is not graded at all -- `_grade`'s
    own gate refuses to start a run there is no time for -- so it reports 0 of
    0. When the source is one an earlier round already ran, that is not "this
    answer was never checked" but "this answer was checked and nobody wrote it
    down": the operator's log showed `self=17/20` on one round and `self=0/0` on
    the answer that shipped, which was the same program.

    And `partial` was worse than lost, it was CLEARED. A reply that corrects
    only the cases is graded against the program already in hand, so the
    "previous" it is compared to is itself -- `dropped_definitions` finds
    nothing missing, because nothing can be missing from a comparison with
    itself. A fragment flagged one round earlier came out of that looking like a
    whole program, which is precisely the flag `_supersedes` relies on to keep a
    fragment from displacing one.
    """
    if not (candidate.total or candidate.self_total):
        # Nothing ran for this one, so there is nothing of its own to overwrite.
        # All of it moves together: a pass count without the failures it came
        # with would be a reading nobody could act on.
        #
        # Only when nothing ran. A round that WAS graded has the current answer
        # -- most sharply when the bar moved under it, which is the whole point
        # of a corrected case array: the same program that failed one round
        # passes the next, and carrying the old failures forward there would
        # re-report a disagreement that no longer exists and cost a round trip
        # doing it.
        candidate.passed, candidate.total = prior.passed, prior.total
        candidate.self_passed, candidate.self_total = prior.self_passed, prior.self_total
        candidate.failures = list(prior.failures)
        candidate.from_self_tests = prior.from_self_tests
        candidate.defect = candidate.defect or prior.defect
    if not candidate.self_cases:
        candidate.self_cases = prior.self_cases
    # One-way, and independent of the above. A fragment does not stop being a
    # fragment because a later round had no round above it to miss anything
    # from.
    candidate.partial = candidate.partial or prior.partial


def _supersedes(candidate: Candidate, best: Candidate, still_writing: bool) -> bool:
    """Should `candidate` replace `best` as the answer that ships?

    THE LATEST VERSION WINS. No score is compared, and that is the whole rule.

    A round only happens because the one before it was wrong: the loop ends the
    moment there is no defect and no failure, so every candidate after the first
    exists BECAUSE the model was shown what was wrong with its predecessor and
    asked to correct it. The later program is the corrected one. Ranking them
    against each other asks a question that has already been answered.

    Scoring them did real damage. `Candidate.score` cannot tell "failed its
    tests" from "was never tested" -- both put 0 in the same slot -- so a
    correction that arrived too late in the budget to grade scored (0,0,1,1)
    against the wrong program's (0,1,1,1) and LOST to the answer it was
    correcting. Reproduced end to end: phase 3 returned the right program,
    `self=1/3` went out, and the file held phase 2's code. Every refinement of
    the comparison was another way to get that wrong; not comparing cannot.

    Two things are still not versions of the answer, and neither is a judgement
    about how good the code is:

    * NOTHING ARRIVED. An empty capture is the absence of an answer rather than
      a worse one -- a dead tab, a reply that rendered as prose, a read that
      timed out. It must never displace a program already in hand.
    * THE MODEL IS STILL WRITING. What arrived is a fragment of a reply rather
      than a revision of one, and a fragment that happens to parse must not
      displace the finished program above it.
    * ONLY THE PART THAT CHANGED ARRIVED. The same thing said by the model
      instead of by the clock: a round asked for the whole program sent back
      the one function it fixed, and it uses a helper that lives in the round
      above. `dropped_definitions` is what knows. This is not a judgement about
      which program is better -- a fragment is not a program, and every hidden
      test would die on `NameError` for a helper that was right there.

    None of the three ranks anything. Two fragments still go latest-first,
    because between two fragments the later one is still the correction.
    """
    if not candidate.code.strip():
        return False
    if still_writing:
        return not best.code.strip()
    if candidate.partial and best.code.strip() and not best.partial:
        return False
    return True


class _Grader:
    """Lazily-built executors, reused across solves (Docker startup is slow)."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._settings = self._build_settings()
        # Built from worker threads now (see `_graded`), and concurrently: the
        # miner serves several solves at once. Two threads missing the cache
        # together would each construct an executor, and for the Docker backend
        # that is a container's worth of startup thrown away -- on the one code
        # path whose entire reason for caching is that Docker startup is slow.
        self._lock = threading.Lock()
        # An executor that could not be BUILT, by language: (when, why).
        #
        # `self._cache` is written after `get_executor` returns, so a
        # constructor that raises leaves nothing behind and every later solve
        # repeats it. For Rust without Docker that is
        # `DockerExecutor._resolve_docker` shelling out to `docker info` --
        # 60ms when the socket is simply absent, and up to its own 20 second
        # timeout when the daemon is hung or still starting. Per Rust task,
        # inside the solve's budget, for the life of the process.
        self._broken: dict[str, tuple[float, str]] = {}
        self._reported: set[str] = set()

    @staticmethod
    def _build_settings():
        from rlvr.config import Settings

        kind = os.environ.get("SOLVER_VERIFY_EXECUTOR", "subprocess")
        # _env_file=None so the miner's .env cannot accidentally repoint this.
        return Settings(_env_file=None, executor=kind, per_test_timeout_s=VERIFY_TIMEOUT_S)

    def executor(self, language: str):
        cached = self._cache.get(language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(language)
            if cached is not None:
                return cached
            held = self._broken.get(language)
            if held is not None and time.monotonic() - held[0] < EXECUTOR_RETRY_S:
                # Remembered, not re-probed. The message is the original one
                # verbatim: the caller prints it per solve and an operator
                # counts those lines, so it must not change shape here.
                raise RuntimeError(held[1])
            try:
                executor = self._build(language)
            except Exception as exc:  # noqa: BLE001 - remembered, then re-raised
                self._broken[language] = (time.monotonic(), f"{exc}")
                self._report_unavailable(language, exc)
                raise
            self._broken.pop(language, None)
            self._cache[language] = executor
            return executor

    def state(self, language: str) -> str:
        """What grading this language would do right now. Never probes.

        Read from `/solver-status`, so it must not shell out: a `docker info`
        against a hung daemon blocks for twenty seconds, and the endpoint an
        operator polls to find out whether the miner is healthy is the last
        place to put that.
        """
        if language in self._cache:
            return "ready"
        held = self._broken.get(language)
        if held is not None:
            return f"unavailable: {held[1]}"
        return "not checked yet"

    def _build(self, language: str):
        """Construct the executor for ``language``. Assumes the lock is held."""
        from rlvr.execution.executor import get_executor

        settings = self._settings
        if language == "rust" and settings.executor != "docker":
            # Rust needs rustc in the pinned image; no subprocess path.
            from rlvr.config import Settings

            settings = Settings(
                _env_file=None,
                executor="docker",
                per_test_timeout_s=VERIFY_TIMEOUT_S,
            )
        return get_executor(settings, language=language)

    def _report_unavailable(self, language: str, exc: BaseException) -> None:
        """Say what a missing executor costs, once per language per run.

        The per-solve line names the exception and nothing else, which reads as
        a hiccup. This is the part an operator has to act on: everything in
        this language is now ungraded, and no repair round can fire on it.
        """
        if language in self._reported:
            return
        self._reported.add(language)
        # The hint, not just the error. A daemon that is UP and unreachable
        # reports exactly like one that is not running -- read off a production
        # run, where every Rust answer went out ungraded behind "permission
        # denied ... /var/run/docker.sock", which is a group membership and not
        # a broken install.
        from .rehearse import _executor_hint

        fix = _executor_hint(exc) or ""
        if not fix and language == "rust":
            fix = (
                " Rust has no subprocess path — grading it at all needs a Docker "
                "daemon; see the README's \"Rust needs Docker\" section."
            )
        print(
            f"[verify] the {language} executor could not be built, so no {language} "
            f"answer can be graded here: no repair rounds, and verified=False "
            f"however good the answer is. Not probing again for "
            f"{EXECUTOR_RETRY_S:.0f}s. Once per run.\n"
            f"           {type(exc).__name__}: {exc}{fix}"
        )

    def check(
        self, code: str, language: str, entrypoint: str,
        examples: list[dict[str, Any]], names: Optional[list[str]] = None,
        budget_s: Optional[float] = None,
    ) -> tuple[int, int, list[str], list[dict[str, Any]]]:
        """Run ``code`` against the examples.

        Returns ``(passed, total, failures, failed)`` -- the counts, one line of
        concrete evidence per failure for the repair prompt, and the failing
        cases themselves. The last is what lets a correction be merged into the
        suite rather than replace it: the repair prompt shows the model the
        cases that failed and offers to take them back corrected, and knowing
        WHICH cases those were is the difference between correcting a bar and
        letting a model rewrite it.

        ``budget_s`` bounds what the RUN may cost, and it is the difference
        between a check and a solve-ending one. Every case gets
        `VERIFY_TIMEOUT_S` and nothing used to bound the set, so a suite of
        twenty cases against a program that hangs costs twenty times that:
        measured, 100.2 seconds. `_grade`'s gate demanded only
        `min(needed, GRADE_FLOOR_S)` = 15 seconds be left before starting it --
        a 6.7x under-estimate, and the run then took the rest of the deadline
        with it. Two of those and a 290-second solve grades nothing and submits
        unverified, which is the failure this argument exists to end.

        A partial run is the point, and it is what the gate's own comment
        already promised: "a partial run that DOES fit is worth more than no
        evidence at all". `total` stays the FULL count, so an unrun case is
        neither a pass nor a failure but simply unknown -- and `passed < total`
        keeps `verified` and `self_verified` false, because three of twenty
        passing is not twenty passing.

        ``names`` labels each failure with the case it came from. Optional
        because the repair prompt does not want it -- the model is being shown
        concrete inputs and outputs, and an authored title is noise there. A
        person reading a test report wants the opposite: `case 3 'resize in
        place'` says which behaviour broke, where a wall of arguments has to be
        decoded first.
        """
        cases = [
            TestCase(
                args=list(case.get("args", []) or []),
                kwargs=dict(case.get("kwargs", {}) or {}),
                expected=case.get("expected"),
            )
            for case in examples
        ]
        if not cases:
            return 0, 0, [], []
        executor = self.executor(language)
        started = time.monotonic()
        results: list[Any] = []
        ran = 0
        while ran < len(cases):
            spent = time.monotonic() - started
            left = None if budget_s is None else float(budget_s) - spent
            chunk = len(cases) - ran
            if left is not None:
                if ran and left <= 0:
                    break
                # How many the budget could pay for if every one of them burned
                # the full per-case timeout. Nothing else can be known before a
                # case runs, and nothing softer is safe: the executor has no
                # batch-level timeout, so a chunk sized on what the cases have
                # cost SO FAR runs to completion however long it turns out to
                # take. Measured, sizing that way: a twenty-case suite whose
                # second half hangs spent 50 seconds of a 12-second budget,
                # because the ten fast cases said the next eighteen would be
                # fast too.
                #
                # The pessimism that used to come with the worst case is gone
                # all the same, and not by loosening the bound -- by looping.
                # A run used to take `budget // timeout` cases and stop, so a
                # suite of twenty ordinary cases costing 0.75s in total had
                # seventeen of them refused whenever the budget was under a
                # hundred seconds, and cases refused are evidence thrown away.
                # Now each chunk costs almost nothing, the clock barely moves,
                # and the next chunk is the same size again: all twenty run, in
                # 0.75s, in seven executor calls.
                #
                # Chunked rather than case by case because the Docker executor
                # puts a whole batch in ONE container, so a call per case is a
                # container start per case. The subprocess executor does not
                # care either way -- measured, 0.2ms per case.
                #
                # Always at least one case, so a check never reports nothing at
                # all, which bounds the overrun at one per-case timeout: the
                # last chunk can be started with a second left and still cost
                # five. That is what the round trip held back in `_grade`
                # absorbs, and it sits outside `DELIVERY_RESERVE_S` besides.
                if VERIFY_TIMEOUT_S > 0:
                    chunk = max(1, min(chunk, int(left // VERIFY_TIMEOUT_S)))
            batch = executor.run_tests(
                code, entrypoint, cases[ran:ran + chunk], VERIFY_TIMEOUT_S
            )
            results.extend(batch)
            ran += chunk
            if len(batch) != chunk:
                # An invariant both executors keep -- one result per test, on
                # every path including the failure ones -- and chunking is what
                # made this code DEPEND on it. A short batch shifts every result
                # after it against the case it belongs to, so a later failure
                # would be reported with the wrong inputs; and the executor is
                # an operator setting (`SOLVER_VERIFY_EXECUTOR`), so a backend
                # this file has never seen can be in the loop. Stop at the last
                # alignment that is certainly right: the rest count as unrun,
                # which they are.
                break
        # `len(results)`, not `ran`: they differ only when a batch came back
        # short, and there the results are what actually ran.
        running = cases[:len(results)]
        if len(running) < len(cases):
            print(
                f"[verify] {len(running)} of {len(cases)} case(s) fit in the "
                f"{float(budget_s) if budget_s is not None else 0:.0f}s left; "
                f"the rest are unrun rather than passed, so this answer cannot "
                f"read as verified"
            )
        failures: list[str] = []
        failed: list[dict[str, Any]] = []
        passed = 0
        for index, (result, case) in enumerate(zip(results, running)):
            if result.passed:
                passed += 1
                continue
            label = ""
            if names and index < len(names) and names[index]:
                label = f"case {index + 1} {names[index]!r}: "
            failures.append(label + _describe(result, case, language, entrypoint))
            failed.append(examples[index])
        return passed, len(cases), failures, failed


# Text that changes between two runs of the SAME failure, and nothing else.
#
# Both sandboxes name their working directory after a random suffix --
# `tempfile.TemporaryDirectory(prefix="rlvr_sbx_")` in the validator's own
# subprocess executor, `prefix="hone-rustc-"` in the Rust compile check -- and
# that directory is inside every traceback line and every rustc diagnostic. So
# a program that crashed identically on two rounds produced two different
# failure strings, and the loop's repeat detector, which is nothing but a
# comparison of those strings, could never fire on a crashing program at all.
# The default `repr` of an object carries the same problem in the form of a
# heap address.
#
# Narrow on purpose. This runs over failure text that the MODEL reads, and a
# normaliser that also rewrote returned values would hide the very difference
# the model is being asked about -- an expected `"0xdeadbeef"` is an ordinary
# string. The address pattern therefore matches only CPython's `<... at 0x...>`
# form, which no test value wears by accident.
_SANDBOX_PATH_RE = re.compile(r"(?:/[^\s\"']*/)?(?:rlvr_sbx_|hone-rustc-)[A-Za-z0-9_.-]+")
_HEAP_ADDRESS_RE = re.compile(r"(<[^<>]*? at )0x[0-9a-fA-F]+(>)")


def _stable(text: Optional[str]) -> Optional[str]:
    """`text` with the per-run noise taken out, or None unchanged."""
    if not text:
        return text
    return _HEAP_ADDRESS_RE.sub(r"\g<1>0x...\g<2>", _SANDBOX_PATH_RE.sub("<sandbox>", text))


def _describe(result, case: TestCase, language: str, entrypoint: str) -> str:
    """One line of concrete evidence for the repair prompt."""
    if language == "rust":
        call = f"stdin={_clip(case.args[0] if case.args else '')!r}"
    else:
        call = f"{entrypoint}(*{case.args!r}, **{case.kwargs!r})"
    if result.timed_out:
        return f"{call} timed out after {VERIFY_TIMEOUT_S:g}s (too slow or an infinite loop)"
    if result.error:
        return f"{call} raised: {_clip(_stable(result.error), 300)}"
    actual = result.value if result.value_ok else result.actual_repr
    # Stabilised BEFORE clipping, so a heap address truncated by the clip is
    # not left half-written and unmatched.
    return (
        f"{call} returned {_clip(_stable(repr(actual)))}, "
        f"expected {_clip(repr(case.expected))}"
    )


def _clip(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# How much budget must be left to ask ANOTHER model, by what is already in hand.
#
# The two numbers differ because what they risk differs, and the payment policy
# is what sets the price. `all_passed` is a hard gate: an empty answer pays
# exactly zero, and no amount of hurrying changes that. Above the gate, speed is
# a multiplier floored at 0.95 -- the slowest correct answer still earns 95% of
# what the fastest earns. So:
#
#   * Empty-handed, ANY time worth a round trip is worth spending. A failed
#     extra ask costs nothing that was not already lost; a successful one is the
#     whole payment. So the floor is the mechanical minimum and nothing more:
#     12s is what `_attempt`'s own loop refuses to start a round below, which
#     makes this "as long as an ask can happen at all".
#   * Holding an unverified answer, the ask is speculative rather than free --
#     it spends a real account's quota to improve on something that may already
#     be right. That bar stays where it was.
#
# The ordering is the invariant, not the values: empty-handed must never be the
# HARDER case to justify. It is the one with nothing to lose.
EMPTY_HANDED_FLOOR_S = 12.0
SECOND_OPINION_FLOOR_S = 20.0

# How many models one task may be put to.
#
# `SECOND_OPINION_PASSES` is the policy for an answer that came back WRONG: ask
# the other model once, and stop. There is something worth submitting either
# way, and each further ask spends a real account's quota to improve on it.
#
# `MAX_PASSES` applies only while holding NOTHING, where that reasoning does not
# apply at all -- there is no answer to improve on and an empty one pays zero.
# Even so it is a runaway guard rather than a target: the clock stops the loop
# long before this does.
SECOND_OPINION_PASSES = 2
MAX_PASSES = 4

@dataclass
class _Plan:
    """What the NEXT pass of one solve should do. One per `solve_task`.

    Only `two_phase` so far, and it exists because a cases turn that costs a
    whole pass used to cost EVERY pass. `solve_task` retries `_attempt` up to
    `MAX_PASSES` times while it is holding nothing, and nothing remembered that
    turn 1 had already proved unaffordable here -- so a task whose cases turn
    timed out burned all four passes on four more cases turns and submitted
    nothing, having never once asked for a program.

    Per-solve rather than per-solver: solves run concurrently on one instance,
    and a flag on `self` would let one task's bad luck disable the split for
    every other task in flight.

    It answers "was turn 1 unaffordable HERE", so only a failure that belongs to
    the task and the site clears it. A tab that went blind is retired on the
    spot and the next pass is served by another one, so its failure says nothing
    about the task -- see `_attempt`, where the two are told apart.

    Cleared, the next pass asks for the program ALONE. There is no combined
    turn to fall back to: cases written beside a program are back-filled from
    what it happens to do and agree with its bugs, which is the whole argument
    for splitting the turns, and a prompt that asks for a second block the
    grader will not trust spends output tokens inside the deadline. The cost is
    real and it is the right one: that task goes out ungraded rather than
    graded against evidence worth nothing.
    """

    two_phase: bool = True

# Turn 1 has NO timeout of its own. There is one clock on a solve -- the
# deadline the validator advertised -- and turn 1 reads against that, exactly
# like every other read here.
#
# It carried a private cap twice, and both were wrong in the same direction.
# The first passed `extend_to_s` equal to the slice, which makes `send`'s
# extension a no-op by construction; the second kept a soft 60s slice and a hard
# 100s cap. Both cut the model off MID-THINK, and cutting a model off mid-think
# is the one thing that cannot help: the reply does not exist yet, so what the
# cap saves is time that bought nothing and what it costs is the whole turn.
# Measured on a live tab: `Thought for 1m 17s` before a single character
# appeared, against a 60 second cap.
#
# A cap LOOKS like it protects the program's read, and it does not. `send`
# returns the moment the model finishes -- the slice is a ceiling, never a wait
# -- so a cases turn that takes 90 seconds hands the program the other 190
# whether or not a cap exists. The only case a cap changes is the one where the
# model has NOT finished, and there it converts a slow answer into no answer,
# which is the one trade the payment policy says never to make.
#
# What protects the program now is the budget itself and `_Plan`: turn 1 cannot
# outlive the deadline, and a turn 1 that fails the TASK's way does not get a
# second chance in the same solve.


# Nothing here passes `extend_to_s` any more, and that is the end of a long
# argument rather than an oversight. It existed so a read could spend a repair
# reserve on an answer that was still arriving; with every read given the whole
# remaining budget there is no reserve, and a hard bound equal to the slice is
# an extension that cannot extend. `send` still accepts the keyword -- see
# `browser_pool` -- for a caller that does hand out less than everything.
#
# The happy side effect: every read here is the two-argument `send` that has
# always been the `Conversation` contract, so a backend written outside this
# package needs nothing new to work.


class VerifyingSolver:
    """Wrap any conversational backend in a self-check-and-repair loop.

    Budget discipline matters more here than anywhere else in the miner: a
    correct answer delivered after the cutoff scores the same zero as a wrong
    one. Every attempt is bounded, the loop stops while there is still margin,
    and whatever candidate ranked best is returned rather than nothing.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        max_attempts: int = 0,
        reserve_s: float = DELIVERY_RESERVE_S,
        max_budget_s: float = 3600.0,
        cache_size: int = 256,
        second_opinion: bool = True,
        self_tests: bool = True,
    ):
        self._backend = backend
        # 0 means UNLIMITED, and that is the default. Correctness is the whole
        # of the payment here, so the only thing that should stop a repair loop
        # is the validator's deadline -- a count of three was a second, private
        # deadline layered under the real one, and it fired first.
        self._max_attempts = max(0, int(max_attempts))
        self._reserve = max(0.0, float(reserve_s))
        self._max_budget = max(5.0, float(max_budget_s))
        self._second_opinion = bool(second_opinion)
        self._self_tests = bool(self_tests)
        self._grader = _Grader()
        self._cache: dict[str, tuple[str, str]] = {}
        self._cache_size = max(0, int(cache_size))
        self._counts = {
            "solved": 0, "verified": 0, "verified_on_local": 0, "cache_hits": 0,
            "empty": 0,
        }
        self._by_provider: dict[str, dict[str, int]] = {}
        # The no-examples explanation is worth saying, but only once a run.
        self._warned_ungradeable = False
        # ...and so is a deadline being cut short by our own configuration.
        self._warned_short_deadline = False

    def _next_pass_blocked_by(
        self, best: Candidate, attempt_no: int, remaining: float
    ) -> Optional[str]:
        """Why the pass after this one will not happen, or None if it will.

        Asked in TWO places, and it has to answer the same in both: at the top
        of the next iteration, which ACTS on it, and at the bottom of this one,
        which ANNOUNCES it. They were separate expressions and they drifted --
        the announcement asked only `attempt_no < passes` and knew nothing
        about the clock, so a solve that had spent its entire budget printed

            [verify] claude returned nothing; asking another model
            [verify] -0s left; not enough to ask anyone else, submitting empty

        one line apart. Nobody was asked. An operator reading that goes looking
        for a second provider's failure that never happened, while the real one
        -- a program turn that ran the budget out -- sits three lines above it
        wearing no emphasis at all.
        """
        empty_handed = not best.code.strip()
        if not empty_handed and attempt_no >= SECOND_OPINION_PASSES:
            # There is an answer in hand and it has already had its second
            # opinion. `MAX_PASSES` is for the empty case only: spending it
            # here would double or quadruple what every failing task costs a
            # real account's quota, to improve on something already worth
            # submitting.
            return "the answer in hand has already had its second opinion"
        floor_s = EMPTY_HANDED_FLOOR_S if empty_handed else SECOND_OPINION_FLOOR_S
        if remaining < floor_s:
            return (
                "not enough to ask anyone else, submitting empty"
                if empty_handed
                else "no time for a second opinion"
            )
        return None

    # -- the Solver interface custom_miner.py expects ---------------------- #
    async def solve_task(self, task, timeout_s: float) -> Answer:
        started = time.monotonic()
        # `timeout_s` is already `min(deadline_s, glm_request_timeout_s)` -- the
        # deadline the miner's own `handle_request` will 504 at. The margin is
        # what keeps this side of it: `send` runs its copy, stream and
        # post-mortem phases AFTER its slice expires (5 + 4 + 2 = 11s), and the
        # answer still has to be graded, archived, signed and put on the wire
        # before the validator stops listening at `deadline_s + 10`.
        #
        # `_max_budget` is a ceiling, not a target, and it deliberately does not
        # bind at the deadlines this subnet advertises. It used to: 240 against a
        # 300s deadline capped the first attempt at 191s, and a model that needed
        # longer had its answer thrown away by this miner rather than by the
        # validator -- which would have paid 96% for the same answer arriving at
        # six minutes. Correctness is worth 100%; speed is worth at most 5%.
        # ONE reserve, and it is what delivery needs. Everything else the
        # solve does -- reading, grading, repairing -- happens inside `budget`,
        # and `budget` runs right up to the point the answer stops being
        # deliverable. Waiting is close to free and giving up is a certain zero:
        # the validator reads until `deadline_s + 10`, and its payment rule has
        # no deadline term at all -- correctness is a hard gate and speed is a
        # relative multiplier floored at 0.95, so the same answer a minute later
        # is still worth 95%.
        budget = min(float(timeout_s), self._max_budget) - self._reserve
        if budget <= 5.0:
            # Too short for the whole margin, so keep the SHAPE of the promise
            # instead of its size: half the request, which leaves the other half
            # for the post-read tail (itself scaled, see `tail_budget`) and for
            # putting the answer on the wire.
            #
            # The floor used to be 5 seconds, and a floor is exactly the wrong
            # instrument here: at a 5-second deadline it budgeted the entire
            # request and `handle_request` cancelled the solve mid-flight. A
            # 504 is indistinguishable from a dead miner, and `deadline_s` is
            # only `Field(gt=0.0, ...)` -- nothing in the protocol promises the
            # comfortable numbers this subnet happens to send today.
            budget = max(1.0, float(timeout_s) * 0.5)

        advertised = float(getattr(task, "deadline_s", 0.0) or 0.0)
        if advertised - float(timeout_s) > 1.0 and not self._warned_short_deadline:
            # `timeout_s` is `min(deadline_s, glm_request_timeout_s)`. When it
            # comes back SHORTER than what the validator advertised, the miner
            # is giving up early on its own configuration -- and nothing else
            # says so. The reference miner's docs put GLM_REQUEST_TIMEOUT_S at
            # 280 against a 300s deadline, and a .env copied from there costs
            # every solve 20 seconds it was offered. Once per run.
            self._warned_short_deadline = True
            print(
                f"[verify] the validator offered {advertised:.0f}s but this miner "
                f"caps the solve at {float(timeout_s):.0f}s, so every answer gets "
                f"{advertised - float(timeout_s):.0f}s less than it could. Raise "
                f"GLM_REQUEST_TIMEOUT_S to at least {advertised:.0f} to use it all "
                f"— a correct answer arriving late still earns 95%+, an unfinished "
                f"one earns nothing. Once per run."
            )

        key = _cache_key(task)
        if key in self._cache:
            self._counts["cache_hits"] += 1
            code, raw = self._cache[key]
            return Answer(code=code, raw_response=raw, verified=True)

        best = Candidate(code="", raw="")
        # One pass per model. The second only happens if the first could not
        # reproduce the public examples even after its repair rounds — at which
        # point the odds it passes the HIDDEN suite are poor, and the whole
        # payment rides on that. Asking the other model is a fresh chance at the
        # full amount, and with a fleet there is usually an idle tab to ask on.
        asked: list[str] = []
        # WHICH model produced the answer that wins, not merely which were
        # asked. Attribution after the fact was otherwise guesswork: of 43
        # archived submissions only three could be traced to a provider at all,
        # and only because the damage itself carried a fingerprint -- two held
        # ChatGPT's nudge, one quoted `/home/claude/sol`. The other forty were
        # unattributable, which made "is one of these tabs doing worse than the
        # others" an unanswerable question.
        won_with: Optional[str] = None
        plan = _Plan()
        # A ceiling, not a plan. Every ordinary path breaks out after one or
        # two: the loop only keeps going while it is holding NOTHING, which is
        # the one state where another ask cannot make things worse.
        passes = MAX_PASSES if self._second_opinion else 1
        attempt_no = 0
        while attempt_no < passes:
            remaining = budget - (time.monotonic() - started)
            # The first pass always runs, however little is left: bailing here
            # would return nothing having asked nobody.
            if attempt_no:
                blocked = self._next_pass_blocked_by(best, attempt_no, remaining)
                if blocked is not None:
                    # `max(0, ...)`: the budget can be a hair past spent by the
                    # time this reads it, and "-0s left" reads as a bug in the
                    # arithmetic rather than as a solve that used everything it
                    # had.
                    print(f"[verify] {max(0.0, remaining):.0f}s left; {blocked}")
                    break
            attempt_no += 1
            candidate, provider = await self._attempt(
                task, remaining, avoid=asked[-1] if asked else None, plan=plan
            )
            if provider:
                asked.append(provider)
            if candidate is not None and candidate.score > best.score:
                best = candidate
                won_with = provider
            if best.verified and not best.failures:
                break
            # Nothing RAN, whether or not anything was shipped to run. The
            # distinction used to be `not task.public_examples`, and that missed
            # the commoner case by far: examples shipped, and the executor could
            # not run them. Measured on a live miner with no Docker daemon, all
            # three Rust challenges asked a SECOND model -- a full extra solve
            # each, 55 to 108 seconds and a second conversation off the account
            # quota -- and then submitted the first model's answer anyway,
            # because two ungradeable candidates tie at `score` and `>` loses a
            # tie. Twice the time and twice the quota for no information at all.
            if best.total == 0:
                if best.from_self_tests or not best.code.strip():
                    # Two ways this warning would be a lie, and the second was
                    # printed against a solve whose cases turn had worked
                    # perfectly.
                    #
                    # `from_self_tests`: the model shipped cases and they ran,
                    # so there was something to grade after all.
                    #
                    # NO CODE: `_run_self_tests` is gated on `code.strip()`, so
                    # an empty candidate reports `from_self_tests=False`
                    # whatever turn 1 produced -- and the warning then blames
                    # the cases turn for the PROGRAM turn's failure. Measured:
                    # a Rust solve whose cases turn returned usable cases in
                    # silence, whose program turn then spent the whole 285s
                    # budget still writing, and which reported "the model sent
                    # no usable cases of its own either". Nothing can be graded
                    # because there is no ANSWER, and the lines that say the
                    # answer is missing already say so, about the right turn.
                    pass
                elif best.self_cases:
                    # A THIRD way it would be a lie, and the one the deadline
                    # produces: the cases turn worked, the cases are right
                    # here, and there was no budget left to run them. Saying
                    # "the model sent no usable cases" sends the reader to fix
                    # a prompt that is working. `_grade` has already named this
                    # one on the line above, so there is nothing to add.
                    pass
                elif not self._warned_ungradeable:
                    self._warned_ungradeable = True
                    why = (
                        "no public examples shipped with this task"
                        if not task.public_examples
                        else "the public examples could not be run here"
                    )
                    print(
                        f"[verify] {why}, and the model sent no usable cases of "
                        f"its own either, so nothing can be graded locally: no "
                        f"repair rounds, no second opinion once an answer is in "
                        f"hand, and verified=False however good it is. Once per run."
                    )
                # A second opinion is only ever worth buying when this one came
                # back EMPTY. Then it is worth a lot: an empty answer scores
                # zero, and the other model is the only remaining chance at the
                # whole payment.
                if best.code.strip():
                    break
            if attempt_no < passes and self._next_pass_blocked_by(
                best, attempt_no, budget - (time.monotonic() - started)
            ) is None:
                print(
                    f"[verify] {provider or 'first'} "
                    + (
                        "returned nothing"
                        if not best.code.strip()
                        else "cleared the examples but not its own cases"
                        if best.verified
                        else "did not verify"
                    )
                    + "; asking another model"
                )
        if asked:
            # `won_with`, not `asked[-1]`. They usually coincide -- a verified
            # answer ends the loop, so the winner is normally the last one asked
            # -- but "usually" is not what a tally is for. A pass whose backend
            # never reported a provider is absent from `asked` while still able
            # to produce the winning answer, and the credit then lands on the
            # PREVIOUS model. This is the number an operator reads to decide
            # which account has started failing; it should say who actually won.
            self._note(won_with if best.verified else None, asked)

        if best.verified:
            self._counts["verified"] += 1
        elif best.self_verified:
            # Counted apart from `verified`, never inside it, and named as
            # the log line names it. On live traffic this is the only counter
            # of the two that can ever move, so a `/solver-status` showing
            # verified=0 over a whole run is the ordinary reading rather than
            # the alarming one -- and this is the number that says whether the
            # answers were any good.
            self._counts["verified_on_local"] += 1
        if best.code.strip():
            self._counts["solved"] += 1
            # `not best.failures` as well as `verified`: with both suites run,
            # an answer can clear the validator's examples and still disagree
            # with the model's own cases. Caching that re-serves one wrong
            # answer for every later task with the same statement, which is the
            # exact harm the cache gate exists to prevent.
            if best.verified and not best.failures and self._cache_size:
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (best.code, best.raw)
        else:
            self._counts["empty"] += 1
        elapsed = time.monotonic() - started
        print(
            f"[verify] {task.language} entrypoint={task.entrypoint} "
            f"provider={won_with or 'none'} "
            f"examples={best.passed}/{best.total} "
            + (f"self={best.self_passed}/{best.self_total} " if best.self_total else "")
            + f"verified={best.verified} "
            # `verified=False` is the only thing a live solve could ever print,
            # because no request ships public examples -- so on its own it said
            # the same thing about an answer that passed every case it had and
            # about one that was never run. This says which, without ever
            # claiming the word `verified` for a model agreeing with itself.
            + (
                f"(verified on local: passed all {best.self_total} of its "
                f"own cases; no public examples exist to confirm it) "
                if best.self_verified
                else ""
            )
            + f"{elapsed:.1f}s/{budget:.0f}s"
        )
        return Answer(
            code=best.code, raw_response=best.raw,
            verified=best.verified, passed=best.passed, total=best.total,
            self_verified=best.self_verified,
            self_passed=best.self_passed, self_total=best.self_total,
        )

    async def _ask_for_cases(self, conversation, task, left: float):
        """Turn 1: the model's cases, before it has written the program.

        Returns the cases, or ``[]`` when the reply carried none usable, or
        ``None`` when the CONVERSATION is the problem -- unreadable, or still
        writing -- in which case turn 2 must not be sent into it at all.

        Read against the SOLVE's clock and nothing else: turn 1 gets whatever
        is left, the same ceiling the program turn gets. There is no partial
        budget here to cut a thinking model off with -- see the note above
        `_Plan` for why every version of that cap was a mistake.
        """
        slice_s = max(1.0, left)
        prompt = build_tests_prompt(
            task.language, task.statement, task.entrypoint, task.public_examples
        )
        # The slice IS everything left, and everything left runs to the point
        # the answer stops being deliverable. Nothing is held back to extend to.
        reply = await conversation.send(prompt, slice_s)
        if getattr(conversation, "still_writing", False) or getattr(
            conversation, "empty_reason", None
        ) in ("unreadable", "unfinished"):
            print(
                f"[verify] the cases turn came back "
                f"{getattr(conversation, 'empty_reason', None) or 'unfinished'}; "
                f"the program request cannot go into a conversation that has "
                f"not answered the last one"
            )
            return None
        cases = extract_self_tests(reply, task.entrypoint, task.language)
        if not cases:
            print("[verify] the cases turn produced none usable; "
                  "asking for the program without them")
        return cases

    async def _attempt(
        self,
        task,
        remaining: float,
        avoid: Optional[str],
        plan: Optional["_Plan"] = None,
    ) -> tuple[Optional[Candidate], Optional[str]]:
        """One model, one conversation: initial answer plus repair rounds.

        The repair rounds deliberately stay in that single conversation so the
        model sees its own previous attempt beside the failure report. Returns
        the best candidate it produced and which provider produced it.
        """
        started = time.monotonic()
        # 1.0, not 5.0. A floor above what the caller can afford does not buy a
        # longer read, it buys a cancelled solve: `solve_task` has already cut
        # the request down to something deliverable, and raising it back here
        # undoes that silently.
        budget = max(1.0, remaining)
        best: Optional[Candidate] = None
        conversation = None
        provider: Optional[str] = None
        # Which provider produced `best` -- not which one this pass is talking
        # to NOW. They part company the moment a repair is carried elsewhere:
        # the pass then ends holding an answer from the first model and a
        # conversation with the second, and returning the latter credits the
        # wrong account. That is the one question the per-provider tally exists
        # to answer, so it follows `best` rather than the conversation.
        #
        # Bound out here beside `provider`, for the same reason: `open()` can
        # raise, the handler below catches it, and the return then reads a name
        # the try block never got to bind.
        best_provider: Optional[str] = None
        try:
            # BOUNDED by what is left. `BrowserFleet.open` waits for a free tab
            # up to `MINER_TAB_WAIT_S`, which ships at 120s, and nothing here
            # ever passed a smaller number -- so on a busy fleet a solve could
            # spend 120s per pass waiting, three passes, 360s against a 280s
            # deadline, and return empty having sent no prompt at all. Measured:
            # budget 40s, elapsed 50.1s, `open()` called at t=0 and t=25.1,
            # prompts sent 0.
            conversation = await self._open_within(budget, started, avoid)
            provider = best_provider = getattr(conversation, "provider", None)
            # Turn 1: the cases, before the program exists. Cases written
            # ALONGSIDE a program can be back-filled from what the program
            # happens to do, and then they agree with its bugs; cases written
            # first cannot. That is the whole argument for spending a round
            # trip here.
            cases: Optional[list] = None
            two_phase = plan is None or plan.two_phase
            if self._self_tests and two_phase:
                cases = await self._ask_for_cases(
                    conversation, task, budget - (time.monotonic() - started)
                )
                if cases is None:
                    # The tab could not be read, or the model was still writing.
                    # Sending turn 2 into it would queue behind an answer that
                    # has not arrived, so this conversation is finished either
                    # way. What differs is whether anything else is worth doing,
                    # and the clock decides it.
                    #
                    # Whether anything ELSE happens is already decided, by the
                    # clock, in `solve_task`: it will not open another pass with
                    # less than `EMPTY_HANDED_FLOOR_S` left. That is the whole
                    # guarantee "a turn that ran the deadline out is never
                    # retried on another tab" rests on, and it holds only
                    # because turn 1 now reads against the real budget -- when
                    # it was capped at 60s the budget survived it and four tabs
                    # were spent in a row. Deciding it a second time here would
                    # be a duplicate of that floor, and a duplicate that drifts.
                    #
                    # So the only job left is to say which failure this was,
                    # because "asking another model" was printed even when
                    # nothing else would be asked.
                    left_after = budget - (time.monotonic() - started)
                    # WHOSE failure was it, though. `unreadable` is set by
                    # `_read` only when the tab went blind or the page died,
                    # and it retires that tab at the same moment -- so the next
                    # pass is served by a different one and the reason cannot
                    # follow it there. Every other way this returns None is a
                    # model that had not finished writing, which belongs to the
                    # task and the site and WOULD repeat exactly.
                    tab_side = (
                        getattr(conversation, "empty_reason", None) == "unreadable"
                    )
                    if plan is not None and not tab_side:
                        # Not the same way twice. A long thinking phase, a slow
                        # account, a hard problem: the next pass would repeat
                        # it. It did: four passes, four timed-out cases turns,
                        # nothing submitted. The remaining passes ask for the
                        # program alone, and that task is submitted ungraded.
                        #
                        # Clearing this for a DEAD TAB was the same mistake
                        # pointed the other way. It cost the rest of the solve
                        # the split -- and on live traffic, which ships no
                        # public examples, the model's own cases are the only
                        # grading there is. Turn 2 alone falls back to the
                        # combined prompt, where the cases are written beside
                        # the program and can be back-filled from whatever it
                        # happens to do. One blind tab is not evidence about
                        # the task, and `BLIND_TAB_GRACE_S` bounds what finding
                        # that out costs.
                        plan.two_phase = False
                    if left_after < EMPTY_HANDED_FLOOR_S:
                        print(
                            f"[verify] the cases turn used the whole "
                            f"{budget:.0f}s budget; nothing left to ask anyone "
                            f"else with"
                        )
                    elif tab_side:
                        print(f"[verify] {left_after:.0f}s left; that tab is gone, "
                              f"not the cases turn — the next pass asks another "
                              f"one for cases as usual")
                    else:
                        print(f"[verify] {left_after:.0f}s left; not asking for "
                              f"cases again this task, the remaining attempts "
                              f"go straight to the program")
                    return best, best_provider
            prompt = build_code_prompt(
                task.language, task.statement, task.entrypoint,
                task.public_examples, cases=cases,
            )
            # `cases or []` covers None as well as [], and it matters: the
            # early return above is the only thing that keeps None out of here,
            # so `list(cases)` was one edit away from a TypeError -- which this
            # module CATCHES as a backend failure and reports as "provider=none"
            # with no answer. A crash that looks like a dead tab is the worst
            # kind, so the line does not depend on the guard above surviving.
            agreed = list(cases or [])
            # The program the LAST round produced, so a repair that corrects a
            # case can be told apart from one that rewrites both -- and the
            # reply that carried it, so a correction sent WITHOUT the program
            # can still be graded against something.
            last_code: Optional[str] = None
            last_program_reply: Optional[str] = None
            # A repair may be carried to a fresh conversation ONCE per pass.
            resumed = False
            # What each round AMOUNTED to -- the program, the defect and the
            # failing cases -- so a round that changed nothing can be told from
            # one that did. See `duplicate` below.
            #
            # Every round of the pass, not just the one before. Comparing
            # against the previous round only is blind to the shape this loop
            # actually spins in, which alternates: a repair round reports
            # failures F on program P, the next reply comes back as cases the
            # revision guard refuses, that round grades as something else, and
            # the round after is P and F again. No two CONSECUTIVE signatures
            # ever matched, so the guard never fired once -- measured, fifty-
            # nine sends inside a single solve.
            seen_signatures: dict[tuple, int] = {}
            # How many times each repair report has already gone out, so a
            # prompt is never sent byte-identical twice without saying so. See
            # `stalled` where the next prompt is built.
            reports_sent: dict[str, int] = {}
            # What grading established about each program this pass has seen,
            # by its source. See `_inherit_evidence` for what it is for.
            judged: dict[str, Candidate] = {}
            # The cases the LAST round failed -- the ones the repair prompt
            # just quoted, and so the only ones the reply to it is entitled to
            # change. See `_merge_cases`.
            reported_failed: list[dict] = []
            # Consecutive rounds that left the program exactly as it was. See
            # `CASES_ONLY_ROUNDS`.
            program_unchanged = 0
            # Whether the prompt just sent WITHDREW the offer to correct a
            # case. A withdrawal the reply can ignore is not one.
            program_only = False
            async def _resume_elsewhere(why: str, avoid: Optional[str] = None):
                """Carry the repair to a FRESH conversation, or None.

                The one move available when a conversation will not produce a
                new answer -- because it is unreadable, or because it just
                repeated itself. Both are the same situation from here: nothing
                more is coming from this tab, and the repair is still worth
                making somewhere else.

                `avoid` is what separates the two. An unreadable tab is a TAB
                problem -- the model is fine and any tab will do, so it passes
                none. A conversation that repeated itself is a MODEL problem,
                and the answer to that is the other model: it arrives holding
                the previous program and the cases it failed, which is a far
                better start than the fresh pass `solve_task` would give it.
                On live traffic that pass does not happen at all -- with no
                public examples nothing can be graded, and `solve_task` breaks
                rather than spend a second account on an answer it cannot
                compare. So this is the only place the other model gets asked.

                Once per pass. A second tab that also fails is a fleet problem
                rather than something to keep paying for, and the caller then
                ends the loop holding the best answer it ever had.

                Returns `(conversation, provider, prompt)` for the caller to
                install, so the loop's own bindings stay the single source of
                truth for what it is talking to.
                """
                nonlocal resumed
                left_now = budget - (time.monotonic() - started)
                if (
                    resumed
                    or best is None
                    or not best.code.strip()
                    or not (best.failures or best.defect)
                    or left_now < RESUME_FLOOR_S
                ):
                    return None
                resumed = True
                print(
                    f"[verify] {why}; carrying the repair to a fresh "
                    f"conversation with {left_now:.0f}s left"
                )
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - it may already be broken
                    pass
                fresh = await self._open_within(budget, started, avoid)
                return (
                    fresh,
                    getattr(fresh, "provider", provider),
                    build_resume_prompt(
                        task.language, task.statement, task.entrypoint,
                        task.public_examples, agreed, best.code,
                        best.failures, defect=best.defect,
                        from_self_tests=best.from_self_tests,
                    ),
                )

            attempt = 0
            while True:
                attempt += 1
                left = budget - (time.monotonic() - started)
                if attempt > 1 and left < ROUND_TRIP_FLOOR_S:
                    # Not enough left to be worth another ROUND TRIP -- which is
                    # what this has always been about, and it never should have
                    # gated the first one. It did: below a 32-second deadline
                    # the budget lands under twelve seconds and the model was
                    # never asked at all, so the miner returned an empty answer
                    # without a single line of log to say why. The first attempt
                    # always runs, however little there is, exactly as the first
                    # pass does in `solve_task`.
                    #
                    # And this branch itself used to be the silent one. The
                    # deadline is the ordinary way a correction loop ends -- it
                    # runs until the answer passes or the clock stops it -- so
                    # it is the last thing that should happen without a word.
                    print(
                        f"[verify] {max(0.0, left):.0f}s left, not enough for "
                        f"another correction round; submitting the last version"
                        + (
                            " unverified"
                            if best is None or not best.verified
                            else ""
                        )
                    )
                    break
                if self._max_attempts and attempt > self._max_attempts:
                    print(
                        f"[verify] SOLVER_MAX_ATTEMPTS={self._max_attempts} "
                        f"reached; submitting the last version"
                        + (
                            " unverified"
                            if best is None or not best.verified
                            else ""
                        )
                    )
                    break
                # Every round reads against EVERYTHING that is left. Earlier
                # builds handed the first attempt a fraction (60% with public
                # examples, 85% without) so a repair would have something to
                # spend, and that reserve was worth least exactly when it cost
                # most: `send` returns the moment the model finishes, so the
                # slice was never a wait -- only a ceiling on a read that ran
                # long, which is the one case where cutting it short throws away
                # the answer. The loop below stops when a round trip no longer
                # fits; nothing is carved out in advance.
                # No `extend_to_s`, for the same reason the cases turn passes
                # none: the slice already IS everything left, so there is
                # nothing being held back to extend into. Passing one equal to
                # the slice makes `send`'s extension a no-op by construction and
                # only reads as though a reserve existed.
                # No `extend_to_s`: `left` already runs to the point the
                # answer stops being deliverable, so there is nothing past it to
                # extend into.
                round_started = time.monotonic()
                reply = await conversation.send(prompt, max(1.0, left))
                # How long the round trip actually took. Used only by the
                # duplicate branch below, and only to tell a model from a tab.
                round_s = time.monotonic() - round_started
                # A repair reply may carry a CORRECTED case array: the repair
                # prompt offers it outright ("or, if the case was wrong rather
                # than the program, a `json` array holding ALL of the cases").
                # Freezing turn 1's cases would kill that escape hatch and let
                # one wrong case break a correct program on every round.
                #
                # WHEN it takes effect differs by the shape the repair came
                # back in, and both shapes matter.
                #
                # Program UNCHANGED, cases corrected -- exactly what was asked
                # for. Applied to this same reply, because the alternative is to
                # report the identical failure it was sent to fix: the round is
                # spent, the next prompt quotes the same disagreement, and the
                # correction lands only on the round after -- one more round
                # trip spent re-reporting a failure already fixed, against a
                # deadline. Measured on a live
                # solve: turn 1 wrote three cases whose `final_records` order
                # was wrong, the program was right, the model corrected the
                # cases exactly as asked, and the answer still went out
                # reported 17/20. Nothing is conceded by grading it now: the
                # program is the one already judged, so a weakened case cannot
                # launder a rewrite that did not happen.
                #
                # Program CHANGED as well -- the reply rewrote both sides of the
                # disagreement. The prompt no longer spends a sentence
                # forbidding that, because forbidding it was never what stopped
                # it: this is. Such a reply is graded against the bar as it
                # stood BEFORE it arrived, so a model cannot make a rewritten
                # program pass by rewriting the bar in the same breath. Its
                # cases apply from the next round.
                revised = extract_self_tests(reply, task.entrypoint, task.language)
                if revised and attempt == 1:
                    # The PROGRAM turn. Its cases are back-filled from what the
                    # program happens to do -- they agree with its bugs, which
                    # is the entire argument for splitting the turns -- so the
                    # bar stays the one turn 1 wrote before any program existed.
                    #
                    # Not a theoretical objection. Measured: turn 1 wrote a case
                    # that CAUGHT the bug, turn 2 sent the buggy program with
                    # two cases of its own, round 1 reported the real failure
                    # and then adopted them, and round 2 re-graded the same
                    # buggy program against the bar it had brought with it --
                    # `self=2/2`, no failures, loop over, buggy program
                    # submitted as passing everything. The rule that a reply is
                    # judged against the bar as it stood before it arrived
                    # covered repair rounds and left this one round short.
                    print(
                        f"[verify] the program turn sent {len(revised)} case(s) "
                        f"of its own; keeping turn 1's — cases written beside a "
                        f"program are back-filled from it"
                    )
                    revised = []
                if revised and program_only:
                    # The last prompt stopped offering the case and asked for
                    # the program, because two rounds running had corrected the
                    # bar and left the program alone. A reply that sends cases
                    # anyway is that same round again, and accepting it would
                    # make the withdrawal a sentence rather than a rule.
                    print(
                        f"[verify] the cases came back again after the prompt "
                        f"stopped offering them; keeping the bar as it stands — "
                        f"this round was asked for the program"
                    )
                    revised = []
                if revised:
                    # MERGED into the agreed suite, not swapped for it, and the
                    # cases the last round FAILED are the only ones a correction
                    # may touch. That rule is what makes a short array safe to
                    # accept, and accepting one matters: the repair prompt asks
                    # about one disagreement, so the natural reply is that one
                    # case corrected. Demanding the whole array back re-sent
                    # twenty cases to fix one of them -- slower, likelier to be
                    # truncated mid-array, and refused outright whenever it came
                    # back one short, which left the wrong case breaking a
                    # correct program on every remaining round of the solve.
                    #
                    # What the old refusal was protecting is protected here by
                    # construction rather than by suspicion: a case the program
                    # PASSES cannot be corrected, dropped or weakened, so the
                    # way to game this -- delete the case you cannot pass -- is
                    # not reachable. See `_merge_cases`.
                    merged, changed = _merge_cases(
                        agreed, revised, reported_failed
                    )
                    if changed.startswith("REFUSED"):
                        print(f"[verify] {changed[len('REFUSED:'):].strip()}")
                        revised = []
                    else:
                        if changed:
                            print(
                                f"[verify] the repair corrected the cases rather "
                                f"than the program: {changed}. The program is "
                                f"re-graded against the {len(merged)} that now "
                                f"stand; if it still disagrees the loop keeps "
                                f"going."
                            )
                        # Kept even when the merge changed nothing, because
                        # `revised` is not only the new bar -- it is what tells
                        # the branches below that this reply carried CASES. A
                        # reply that sends the suite back untouched and no
                        # program is still asking for the program in hand to be
                        # re-graded, and zeroing it here made that reply read as
                        # "nothing reached me as code" and spend a round trip
                        # saying so.
                        revised = merged
                now_code = extract_code(reply, task.entrypoint, task.language).strip()
                # Which reply the candidate is actually graded FROM. Normally
                # this one; see the cases-only branch below for when it is not.
                graded = reply
                if revised and last_code:
                    if now_code == last_code:
                        agreed, revised = revised, None
                    elif not now_code:
                        # Cases corrected, program deliberately NOT resent --
                        # the one reply the repair prompt asks for by name when
                        # the case was the thing that was wrong, and until this
                        # branch existed the answer to it was "your previous
                        # reply did not reach me as code". Measured: the program
                        # was right, turn 1's case was not, the model corrected
                        # exactly the case it was asked to, and the miner spent
                        # the rest of the budget demanding a program it already
                        # had before submitting one reported 0/1 on a bogus bar.
                        #
                        # Nothing here is taken on trust. The program is the one
                        # already in hand and already judged, so a weakened bar
                        # cannot launder a rewrite -- there was no rewrite. It
                        # is re-graded, not assumed to pass.
                        agreed, revised = revised, None
                        graded = last_program_reply or reply
                candidate = await self._graded(
                    graded, task, budget - (time.monotonic() - started), agreed,
                    # The round ABOVE this one, still un-updated here -- see the
                    # `last_code = now_code` below, which runs after this. That
                    # is what a reply has to be complete with respect to: a
                    # round that sends back only the function it fixed is using
                    # helpers that exist in the reply above it and nowhere in
                    # the file that would be submitted.
                    previous=last_code or "",
                )
                # Before anything reads it: a round that could not be graded,
                # or one graded against itself, must not report less about a
                # program than an earlier round already established.
                key = candidate.code.strip()
                if key:
                    prior = judged.get(key)
                    if prior is not None:
                        _inherit_evidence(candidate, prior)
                    judged[key] = candidate
                # What this round AMOUNTED to. Compared against the rounds
                # before rather than the replies themselves, because the same
                # program under a different sentence of prose is the same
                # program: byte equality misses that and this does not. The
                # failures are in it so a corrected CASE reads as progress even
                # when the program is untouched -- which is exactly what the
                # repair prompt asks for.
                signature = (
                    candidate.code.strip(), candidate.defect, tuple(candidate.failures)
                )
                repeats = seen_signatures.get(signature, 0)
                seen_signatures[signature] = repeats + 1
                duplicate = attempt > 1 and repeats > 0
                # Only when one ARRIVED. A cases-only reply leaves the program
                # in hand standing, and forgetting it here would make the very
                # next correction unattributable to any program at all.
                program_unchanged = (
                    0 if now_code and now_code != last_code
                    else program_unchanged + 1
                )
                if now_code:
                    last_code = now_code
                    last_program_reply = reply
                if revised:
                    agreed = revised
                if best is None or _supersedes(
                    candidate, best,
                    getattr(conversation, "still_writing", False),
                ):
                    best, best_provider = candidate, provider
                if candidate.verified and not candidate.failures:
                    break

                if getattr(conversation, "still_writing", False):
                    # The model had not finished when the read stopped, so
                    # whatever is in hand is a fragment of an answer rather than
                    # a wrong one -- and there is nothing to say to a
                    # conversation that is mid-sentence. The composer is
                    # usually disabled while a reply streams; where it is not,
                    # the prompt queues behind the answer it is asking about.
                    #
                    # Measured, with the site's busy selector dropped at startup
                    # (which `usable_busy_selectors` does whenever a candidate
                    # matches an idle page) and the model still writing:
                    #
                    #   captured=''             -> "your reply did not reach me
                    #                              as code", sent to a model
                    #                              that was still writing it
                    #   captured='def g(n):\n    total = 0\n    while n > 0:'
                    #                           -> "the code is not valid
                    #                              Python", about a program the
                    #                              model had not finished
                    #
                    # `send` already reads past its slice rather than stop here,
                    # so reaching this means the whole budget is gone. Stop.
                    print(
                        f"[verify] {provider or 'this model'} was still writing when "
                        f"the budget ran out; "
                        + (
                            "submitting the part that arrived"
                            if candidate.code.strip()
                            else "nothing arrived to submit"
                        )
                        + " rather than interrupting it with a repair prompt"
                    )
                    break
                if not candidate.code.strip() and (
                    getattr(conversation, "empty_reason", None)
                    in ("unreadable", "unfinished")
                ):
                    # Nothing was captured, and the conversation itself is why.
                    # A repair round here sends the fix-this prompt into a tab
                    # that just proved it cannot be read, or queues it behind an
                    # answer the model has not finished writing. Measured on a
                    # live miner, twice in one run: the first read spent 191s
                    # and returned nothing, the repair spent another 29s on the
                    # same dead conversation and returned nothing, and the task
                    # ended with 5s left -- too few to ask any of the five
                    # healthy tabs standing idle.
                    #
                    # A reply that RENDERED and simply had no code block in it
                    # is the opposite case and deliberately not caught here:
                    # that is the model breaking the output contract, telling it
                    # so is what fixes it, and the conversation is fine.
                    #
                    # The distinction cannot be made from the candidate: an
                    # empty one always carries a `defect`, because the
                    # structural checks reject empty source exactly as they
                    # reject a broken program. Only the tab knows.
                    reason = getattr(conversation, "empty_reason", "?")
                    if reason == "unreadable":
                        # The conversation is gone; the REPAIR is not. Carry it
                        # to a fresh tab rather than submitting an answer whose
                        # failures nobody asked the model to fix. Measured over
                        # a production run: fifteen solves ended exactly here,
                        # each holding a candidate that failed its own cases,
                        # with an average of 129 seconds still on the clock.
                        carried = await _resume_elsewhere(
                            f"{provider or 'this model'} returned nothing and "
                            f"that conversation is unreadable"
                        )
                        if carried is not None:
                            conversation, provider, prompt = carried
                            continue
                    # No repair to carry, or nothing left to carry it with. Say
                    # which -- this line used to promise that somebody else
                    # would be asked, and when an answer was already in hand
                    # `solve_task` submitted it instead and asked nobody.
                    print(
                        f"[verify] {provider or 'this model'} returned nothing and "
                        f"the conversation is {reason}; "
                        + (
                            f"submitting the answer already in hand"
                            if best is not None and best.code.strip()
                            else "asking elsewhere"
                        )
                    )
                    break
                if duplicate:
                    # Same program, same failures: the round changed nothing.
                    # Change something rather than re-ask -- a FRESH
                    # conversation is a real change where a re-ask is not, and
                    # `_resume_elsewhere` is that move.
                    carried = await _resume_elsewhere(
                        f"{provider or 'this model'} sent back the same program "
                        f"and the same failures after being shown them",
                        avoid=provider,
                    )
                    if carried is not None:
                        conversation, provider, prompt = carried
                        continue
                    # Nowhere left to carry it. Whether that ends the loop turns
                    # on WHY the round changed nothing, and those are two
                    # different things wearing one shape.
                    #
                    # A round that cost no time did not involve the model: the
                    # read returned text that was already on the page. Re-asking
                    # that spins at machine speed for the rest of the budget,
                    # and this branch is the only thing standing between a
                    # broken tab and that spin. Stop.
                    #
                    # A round that took a real round trip is the other thing
                    # entirely -- the model answered, and answered the same.
                    # Stopping there ended solves with the whole budget unspent:
                    # measured, a model repeating itself once ended the loop at
                    # 0.2s of a 60s budget, throwing away every round the clock
                    # would still have paid for. Correcting runs until the
                    # answer passes or the deadline stops it, and a model is
                    # stochastic -- the next ask is a real chance, not a
                    # certainty, and a real chance is what the remaining budget
                    # is for. Fall through and ask again.
                    if round_s < STALE_ROUND_S:
                        print(
                            f"[verify] {provider or 'this model'} returned the same "
                            f"program in {round_s:.1f}s without being asked again — "
                            f"the tab is replaying an old reply rather than "
                            f"answering; submitting the last version"
                        )
                        break
                if not candidate.defect and not candidate.failures:
                    break  # nothing actionable to report (no examples shipped)
                # Kept apart, not merged into one list of "problems": a defect
                # means the code never ran, and the repair prompt has to say so
                # rather than blame logic that was never executed.
                insist = program_unchanged >= CASES_ONLY_ROUNDS
                program_only = insist and candidate.from_self_tests
                if program_only:
                    print(
                        f"[verify] {program_unchanged} round(s) running have "
                        f"corrected the cases and left the program alone; asking "
                        f"for the program this time and not offering the cases"
                    )
                report = build_repair_prompt(
                    candidate.failures,
                    task.language,
                    task.entrypoint,
                    defect=candidate.defect,
                    from_self_tests=candidate.from_self_tests,
                    insist_on_program=insist,
                )
                # Asking the same question a second time is worth doing -- a
                # model is stochastic and the budget is there to spend on the
                # chance. Asking it in the same WORDS is not: the conversation
                # still holds the reply it gave, and the likeliest continuation
                # of an identical prompt is an identical answer. So the report
                # goes out again, with the repetition named in it.
                # The cases this prompt is about. The reply may correct these
                # and nothing else.
                reported_failed = list(candidate.failed_cases)
                stalled = reports_sent.get(report, 0)
                reports_sent[report] = stalled + 1
                if stalled:
                    print(
                        f"[verify] this same report has now gone out "
                        f"{stalled + 1} times; naming the repetition in it "
                        f"rather than re-sending it word for word"
                    )
                    report = build_repair_prompt(
                        candidate.failures,
                        task.language,
                        task.entrypoint,
                        defect=candidate.defect,
                        from_self_tests=candidate.from_self_tests,
                        stalled=stalled,
                        insist_on_program=insist,
                    )
                prompt = report
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed solve scores zero, never crashes
            print(f"[verify] backend failure: {type(exc).__name__}: {exc}")
        finally:
            if conversation is not None:
                try:
                    await conversation.close()
                except Exception:  # noqa: BLE001 - cleanup must not mask a result
                    pass
        return best, best_provider

    async def _open_within(self, budget: float, started: float, avoid: Optional[str]):
        """`backend.open`, bounded by the solve's own clock.

        A fleet backend waits for a free tab, and the wait it defaults to is an
        operator setting about fleet capacity that knows nothing about this
        request's deadline. Half of what is left, floored at `OPEN_FLOOR_S`: a
        lease that has not come free in half the remaining budget will not leave
        time to use it, and the caller has other passes to spend.

        The bound is offered as a keyword and the two-argument form is bounded
        from out here instead. `Backend.open` has always been `open(avoid=...)`,
        so a backend written outside this package need not have grown a
        `timeout_s` -- and a keyword it does not take would be a TypeError
        inside the one call the whole solve depends on.
        """
        left = budget - (time.monotonic() - started)
        share = max(OPEN_FLOOR_S, left * 0.5)
        try:
            return await self._backend.open(avoid=avoid, timeout_s=share)
        except TypeError:
            return await asyncio.wait_for(
                self._backend.open(avoid=avoid), timeout=max(share, 1.0)
            )

    async def _graded(
        self, reply: str, task, left: float, cases: Optional[list] = None,
        previous: str = "",
    ) -> Candidate:
        """`_grade`, run OFF the event loop.

        Grading is blocking work wearing an async coat: `compile_defect` shells
        out to rustc and `_Grader.check` runs the validator's own executor,
        each a `subprocess.run` of seconds -- and for the Docker backend, of a
        container start. Called straight from a coroutine it stops the event
        loop dead, and the loop is not this solve's alone.

        Measured, a 3s subprocess called from inside a coroutine, beside a
        second task holding a 1.0 second deadline:

            the other solve's 1.0s deadline fired after  3.05s

        The miner answers several validators at once (`solve_slots` is a
        semaphore, not a mutex), so that other task is another live solve. Worse,
        the deadline that decides whether a solve is PAID is itself an
        `asyncio.wait_for` in `handle_request` -- and a timer cannot fire on a
        loop that is not running. One Rust compile could therefore push every
        other in-flight solve past its cutoff, and each of those answers 504
        with no answer at all, however finished the answer already was.

        `to_thread` costs nothing here: the calling coroutine is going to wait
        for this result either way. What it buys is that everything ELSE keeps
        running while it waits.
        """
        try:
            return await asyncio.to_thread(
                self._grade, reply, task, left, cases, previous
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never lose the answer to the check
            print(f"[verify] grading unavailable: {type(exc).__name__}: {exc}")
            return Candidate(
                code=extract_code(reply, task.entrypoint, task.language), raw=reply
            )

    def _note(self, winner: Optional[str], asked: list[str]) -> None:
        """Per-provider tally, so a model that has started failing is visible."""
        for name in asked:
            row = self._by_provider.setdefault(name, {"asked": 0, "verified": 0})
            row["asked"] += 1
        if winner:
            row = self._by_provider.setdefault(winner, {"asked": 0, "verified": 0})
            row["verified"] += 1

    # ---------------------------------------------------------------------- #
    def _run_self_tests(
        self, candidate: Candidate, task, cases: Optional[list] = None,
        left: Optional[float] = None,
    ) -> None:
        """Grade a candidate against the cases turn 1 obtained.

        Never raises and never blocks the answer. A model wrote both halves of
        this -- the cases and the JSON they arrived in -- so every failure mode
        here ends in "no self-tests ran", which is exactly where this code path
        started.

        It does not read the reply. The program turn asks for ONE block, so
        there is nothing to extract from it -- and mining it for cases anyway
        would mean grading a program against whatever it volunteered about
        itself, which is the back-filling the split exists to prevent. A repair
        reply that CORRECTS a case is handled where it belongs, in `_attempt`,
        which feeds the corrected array to the next round.
        """
        cases = list(cases or [])
        if not cases:
            return
        names = [case.get("name", "") for case in cases]
        try:
            passed, total, failures, failed = self._grader.check(
                candidate.code, task.language, task.entrypoint, cases, names,
                budget_s=left,
            )
        except Exception as exc:  # noqa: BLE001 - a broken executor loses no answer
            # Same four words as the public-examples path below, deliberately.
            # They are what an operator counts to decide whether a missing
            # executor is costing anything -- and on live traffic, which ships
            # no public examples, THIS is the only path that can print it. The
            # other wording made that count read zero while every answer in the
            # language went out ungraded.
            print(f"[verify] local grading unavailable, so the model's own cases "
                  f"could not be run: {type(exc).__name__}: {exc}")
            return
        if not total:
            return
        candidate.self_passed, candidate.self_total = passed, total
        candidate.from_self_tests = True
        # Only failures drive a repair. A clean run is left silent: it is the
        # ordinary outcome and saying so on every solve would bury the line
        # that matters.
        candidate.failures = failures
        # WHICH cases failed, not just what the failures looked like. A repair
        # round may take these back corrected, and the merge that does it has
        # to know exactly which of the agreed cases are in play.
        candidate.failed_cases = failed

    def _grade(
        self, reply: str, task, left: Optional[float] = None,
        cases: Optional[list] = None, previous: str = "",
    ) -> Candidate:
        code = extract_code(reply, task.entrypoint, task.language)
        candidate = Candidate(code=code, raw=reply, self_cases=len(cases or []))
        defect = (
            rust_defect(code)
            if task.language == "rust"
            else python_defect(code, task.entrypoint)
        )
        if defect is None and task.language != "rust":
            # Only once the structural checks are happy, because they are the
            # ones that say what is wrong most precisely. A reply that will not
            # parse is not "incomplete", it is broken, and saying the wrong one
            # sends the repair round after the wrong thing.
            #
            # `partial` rides along separately: `defect` tells the MODEL what to
            # fix, and this tells `_supersedes` that what arrived is not a
            # version of the answer at all.
            defect = dropped_definitions(code, previous)
            candidate.partial = defect is not None
        # Not `left <= 0`. `_Grader.check` gives every case
        # `VERIFY_TIMEOUT_S` and nothing bounds the run as a whole, so a
        # candidate that times out on each of its cases spends that many
        # multiples of it: measured, 6 cases x 5s = 30s of executor time bought
        # with 0.2s of budget, on a verdict nothing could act on -- there is no
        # time left for a repair round and `verified` never reaches the
        # validator. The guard's own reason ("running anything buys nothing that
        # can still be acted on") is as true at 0.2s as at 0, so it asks what
        # the run could actually cost.
        #
        # `GRADE_FLOOR_S` caps the demand: a task with twenty cases would
        # otherwise refuse to grade anything under a hundred seconds, and a
        # partial run that DOES fit is worth more than no evidence at all.
        needed = VERIFY_TIMEOUT_S * max(
            1, len(cases or []) or len(getattr(task, "public_examples", None) or [])
        )
        out_of_budget = left is not None and left < min(needed, GRADE_FLOOR_S)
        # What the RUN may spend, as opposed to what it must have to start.
        #
        # A round trip is held back, and the first attempt at that was reverted
        # for a good reason which no longer applies. It used to trade one
        # failure for a worse one: `check` sized its run by the WORST case, so a
        # shortened budget meant only the first case or two ran, they happened
        # to pass, and a partial run with no failures in it ended the repair
        # loop -- stopping on ignorance rather than on evidence. `check` now
        # sizes each chunk by what the cases have actually cost, so an ordinary
        # twenty-case suite finishes inside three seconds and holding twelve
        # back costs nothing at all. What it buys is the round the evidence is
        # FOR: without it one grading pass could spend every second that was
        # left, and a list of failures nobody has time to report is not worth
        # the run that produced it.
        #
        # Only when a round trip is actually on the table. Below that floor the
        # loop will not start another round whatever happens, so reserving for
        # one would simply throw the seconds away.
        grading_budget = left
        if left is not None and left > ROUND_TRIP_FLOOR_S:
            grading_budget = left - ROUND_TRIP_FLOOR_S
        if defect is None and task.language == "rust" and not out_of_budget:
            # Python's check PARSED that code; Rust's only grepped it for
            # `fn main`. Ask the compiler the same question the validator will,
            # which is the only check a Rust answer gets at all when no public
            # examples shipped -- and on the run this was written for, none
            # ever did. Returns None when there is no local toolchain.
            #
            # `left` caps it: a compile is allowed to be slow, but not slower
            # than the answer it is checking is worth. See `compile_defect`.
            #
            # And when `left` has gone NEGATIVE the compile is skipped outright,
            # not merely capped. `compile_defect` floors its timeout at one
            # second, so an overrun budget still bought a temp directory and a
            # rustc process -- a whole second, spent past the deadline, on a
            # verdict nothing can act on: there is no time left for a repair
            # round and `defect` never reaches the validator. The read now
            # extends into this reserve whenever the model is still writing, so
            # arriving here with nothing left is the ordinary case rather than
            # the strange one.
            defect = _stable(compile_defect(code, left))
        if defect is not None:
            # Structurally unusable: report it without paying for execution.
            candidate.defect = defect
            candidate.code = "" if not code.strip() else code
            return candidate
        if out_of_budget:
            # The budget is gone, so running anything buys nothing that can
            # still be acted on: there is no time for a repair round, and
            # `verified` never reaches the validator -- it feeds this process's
            # cache and its stats and nothing else. It is not free, either:
            # every case gets VERIFY_TIMEOUT_S, in a subprocess or a container,
            # and the deadline above is an `asyncio.wait_for` that answers 504
            # rather than late. The check would be paid for with the answer it
            # was checking. The structural checks above already ran; they cost
            # microseconds and are what ranks this candidate.
            # Gated on `task.public_examples` until now, which is every
            # live task: production ships none, so the one line explaining why
            # an answer went out ungraded was the one line that never printed.
            # What could not be run is what to name.
            unrun = []
            if task.public_examples:
                unrun.append(f"the {len(task.public_examples)} public example(s)")
            if cases:
                unrun.append(f"the model's {len(cases)} own case(s)")
            if unrun:
                print(
                    f"[verify] out of budget before {' or '.join(unrun)} could be "
                    f"run; submitting the answer unverified"
                )
            return candidate

        # BOTH suites, in this order, because turn 2 was asked to pass both:
        # the validator's examples (in `<examples>`) and the model's own cases
        # from turn 1 (in `<must_pass>`). Only one of them used to run. With
        # examples shipped the own cases were quoted in the prompt and never
        # executed, so a program right on the one example and wrong on its own
        # boundary case verified, ended the loop and shipped -- the repair round
        # that exists to catch exactly that never fired. Live traffic ships no
        # examples, which is why it went unnoticed rather than why it was fine.
        #
        # The ORDER is the whole of the precedence. The validator's examples are
        # ground truth: when they fail, the program is wrong, there is nothing
        # to weigh, and the own cases are not run at all -- a second opinion
        # from the same model on a program already known wrong tells us nothing
        # and costs an executor run per case. Only once they are all green does
        # a disagreement with the model's OWN cases become the open question,
        # and `failures` then carries that instead. So `failures` names one
        # suite at a time and `from_self_tests` says which, which is what lets
        # the repair prompt ask for the right thing.
        if task.public_examples:
            try:
                passed, total, failures, _ = self._grader.check(
                    code, task.language, task.entrypoint, task.public_examples,
                    budget_s=grading_budget,
                )
            except Exception as exc:  # noqa: BLE001 - a broken grader loses no answer
                print(f"[verify] local grading unavailable: {type(exc).__name__}: {exc}")
                return candidate
            candidate.passed, candidate.total, candidate.failures = (
                passed, total, failures
            )
            if failures:
                return candidate
        # Running these is not verification and is never recorded as any:
        # `passed`/`total` are the validator's examples alone, so `verified`
        # cannot be earned by a model agreeing with itself. What they catch is
        # the commonest failure by far -- the model knowing what the answer
        # should be and coding it wrong -- and that is objectively checkable
        # with the validator's own executor.
        if self._self_tests and code.strip():
            # `left` is what the SOLVE has, and the run is bounded by it. The
            # examples above may already have spent some of it; that is
            # deliberately not re-measured, because the two suites share one
            # deadline and a stale-but-generous number is the safer error here
            # -- `run_tests` still stops at `VERIFY_TIMEOUT_S` per case.
            self._run_self_tests(candidate, task, cases, left=grading_budget)
        return candidate

    # -- what a Rust answer is actually checked with ----------------------- #
    def rust_support(self) -> dict[str, str]:
        """The two independent checks a Rust answer gets, and their state.

        Both can be off at once, and on a box with neither the only thing
        standing between a model's reply and the validator is `rust_defect`,
        which greps a fenced block for `fn main`. Measured over 45 archived
        submissions: 6 of 18 Rust answers would not build, and among them were
        a prompt echo, a tool call and a program truncated mid-identifier --
        all three of which carry those characters.

        Cheap on purpose: `rustc_path` is memoised after its first lookup and
        `_Grader.state` never probes, so `/solver-status` can report this.
        """
        compiler = rustc_path()
        return {
            "compile_gate": f"rustc at {compiler}" if compiler else "off: no local rustc",
            "grading": self._grader.state("rust"),
        }

    async def check_rust_support(self) -> dict[str, str]:
        """Probe both now, at startup, and say what is missing.

        Neither is looked at until the first Rust task arrives otherwise, so a
        box with no toolchain and no daemon looks perfectly healthy -- the
        fleet is up, the doctor is clean, `/health` answers -- right until a
        Rust challenge is graded by nobody. That is the failure this miner is
        least able to see and the operator most able to fix, so it is worth one
        `which` and one `docker info` before serving.
        """
        await asyncio.to_thread(self._probe_rust)
        support = self.rust_support()
        blind = support["compile_gate"].startswith("off") and support[
            "grading"
        ].startswith("unavailable")
        if blind:
            print(
                "[verify] WARN: no local rustc and no working Rust executor, so a "
                "Rust answer is checked only by a grep for `fn main` before it is "
                "submitted. Installing a toolchain restores the compile gate "
                "without Docker; grading Rust needs the daemon."
            )
        else:
            print(
                f"[verify] rust: compile gate {support['compile_gate']}, "
                f"grading {support['grading']}"
            )
        return support

    def _probe_rust(self) -> None:
        """Both lookups, in a worker thread. Neither is allowed to raise."""
        rustc_path()
        try:
            self._grader.executor("rust")
        except Exception:  # noqa: BLE001 - `_Grader` has already said why
            pass

    def stats(self) -> dict[str, Any]:
        return {
            "solver": dict(self._counts),
            "providers": {k: dict(v) for k, v in self._by_provider.items()},
            "fleet": self._backend.stats(),
            "rust": self.rust_support(),
        }

    async def aclose(self) -> None:
        await self._backend.aclose()


def _cache_key(task) -> str:
    import hashlib

    material = f"{task.language}\0{task.entrypoint}\0{task.statement}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

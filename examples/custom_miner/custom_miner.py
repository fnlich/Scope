"""Run YOUR OWN solver as a subnet miner.

The subnet does not care how you solve a task. A validator sends a signed
``TaskRequest`` to your ``POST /solve`` endpoint and only checks that the reply
is:

  * HTTP 200,
  * signed by your miner hotkey, ``Epistula-Signed-For`` the validator,
    within the freshness window, and
  * a ``SolutionPayload`` whose ``problem_id`` echoes the request's, whose
    ``code`` field holds runnable source, and whose whole body is under the
    per-response byte cap.

All of that plumbing already lives in ``rlvr.neurons.demo_miner`` (signature
verification, replay-nonce cache, validator-permit authorization, response
signing, byte/concurrency limits, the FastAPI surface). This module reuses it
verbatim and swaps out ONLY the part that produces an answer, so you never
re-implement the wire contract.

Two ways to plug in your application:

  1. HTTP (default): your app runs as a separate service. Set ``MY_APP_URL`` and
     this miner POSTs each task to it and expects ``{"code": "...",
     "raw_response": "..."}`` back. Your app can be written in any language.

  2. In-process: import ``CustomMiner`` and pass any object implementing
     ``async def solve_task(task: SolveTask, timeout_s: float) -> SolveResult``.

Correctness rules your solver must honor (the scoring is accuracy-or-nothing):

  * ``code`` must be the exact source the validator will run. For Python it must
    define the function named ``task.entrypoint`` using only the standard
    library. For Rust (``task.language == "rust"``) it must be a complete
    program with ``fn main()`` that reads stdin and writes only the answer to
    stdout; ``entrypoint`` is always ``"main"`` there.
  * Return quickly and within the deadline. Anything the validator can't run to
    a full pass — including a late or empty answer — scores zero.
  * Never raise. On any internal failure, return empty code; a zero is survivable,
    a crash loop is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

# preflight.py is this file's sibling; keep them together if you copy this out.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import require_linux  # noqa: E402

# Checked HERE, above the imports below, and not inside run_custom_miner(): on
# Windows `pip install '.[chain]'` has already failed, so `import rlvr` would
# raise ModuleNotFoundError before any guard in a function could speak. The
# whole point is to say why instead.
require_linux("The custom miner")

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Optional, Protocol  # noqa: E402

import httpx  # noqa: E402

from rlvr.neurons.demo_miner import (
    DemoMiner,
    DemoMinerSettings,
    build_demo_miner_app,
)
from rlvr.protocol import SolutionPayload, TaskRequest


# --------------------------------------------------------------------------- #
# The seam your application implements.
# --------------------------------------------------------------------------- #
@dataclass
class SolveTask:
    """Everything a solver is allowed to see (mirrors the public wire model)."""

    problem_id: str          # opaque per-dispatch id; echo it back unchanged
    language: str            # "python" | "rust"
    statement: str           # the natural-language task
    entrypoint: str          # Python: function to define. Rust: always "main".
    public_examples: list[dict[str, Any]]  # [{args, kwargs, expected}, ...]
    deadline_s: float        # advertised budget (see the deadline caveat below)


@dataclass
class SolveResult:
    code: str                # runnable source; this is what the validator grades
    raw_response: str = ""   # optional: full model/agent transcript, for the dataset


class Solver(Protocol):
    async def solve_task(self, task: SolveTask, timeout_s: float) -> SolveResult: ...
    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- #
# Default solver: call your application over HTTP.
# --------------------------------------------------------------------------- #
class HttpAppSolver:
    """POST each task to your app at ``app_url`` and read back the solution.

    Your app should accept::

        POST {app_url}
        {"problem_id","language","statement","entrypoint","public_examples","deadline_s"}

    and return::

        200 {"code": "<source>", "raw_response": "<optional transcript>"}
    """

    def __init__(self, app_url: str, http: Optional[httpx.AsyncClient] = None):
        if not app_url:
            raise ValueError("MY_APP_URL is required for the HTTP solver")
        self._url = app_url
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None

    async def solve_task(self, task: SolveTask, timeout_s: float) -> SolveResult:
        resp = await self._http.post(
            self._url,
            json={
                "problem_id": task.problem_id,
                "language": task.language,
                "statement": task.statement,
                "entrypoint": task.entrypoint,
                "public_examples": task.public_examples,
                "deadline_s": task.deadline_s,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        code = data.get("code")
        if not isinstance(code, str):
            raise ValueError("app response missing a string 'code' field")
        raw = data.get("raw_response")
        return SolveResult(code=code, raw_response=raw if isinstance(raw, str) else "")

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


# --------------------------------------------------------------------------- #
# The miner: reuse every security check, replace only solve().
# --------------------------------------------------------------------------- #
class CustomMiner(DemoMiner):
    """A DemoMiner whose answers come from your Solver instead of GLM."""

    def __init__(self, settings: DemoMinerSettings, solver: Solver, **kw: Any):
        # DemoMiner requires a `client`; we override solve()/aclose() so it is
        # never used. Passing the solver in its place keeps a live reference.
        super().__init__(settings, client=solver, **kw)
        self._solver = solver

    async def solve(self, request: TaskRequest, timeout_s: float) -> SolutionPayload:
        task = SolveTask(
            problem_id=request.problem_id,
            language=request.language,
            statement=request.statement,
            entrypoint=request.entrypoint,
            public_examples=[c.model_dump(mode="json") for c in request.public_examples],
            deadline_s=request.deadline_s,
        )
        try:
            result = await self._solver.solve_task(task, timeout_s)
            code, raw = result.code, result.raw_response
        except Exception as exc:  # noqa: BLE001 - a failed solve scores zero, never crashes
            print(f"[custom-miner] solve failed: {type(exc).__name__}: {exc}")
            code, raw = "", "<solver failed>"
        # problem_id MUST equal the request's, or the validator rejects the reply.
        return SolutionPayload(problem_id=request.problem_id, code=code, raw_response=raw)

    async def aclose(self) -> None:
        await self._solver.aclose()


# --------------------------------------------------------------------------- #
# Entry point: register-checked, axon-advertised, HTTP-served. No model key.
# --------------------------------------------------------------------------- #
def run_custom_miner(solver: Optional[Solver] = None) -> None:
    # DemoMinerSettings reads .env itself, but through pydantic-settings, which
    # never populates os.environ. MY_APP_URL is read from os.environ below and
    # is not one of its fields, so without this a MY_APP_URL written into .env —
    # exactly as the README says to — was silently ignored.
    from solvers.config import find_env_file, load_env_file

    env_file = find_env_file()
    if load_env_file(env_file):
        print(f"[custom-miner] loaded {env_file}")
    settings = DemoMinerSettings()
    if solver is None:
        solver = HttpAppSolver(os.environ.get("MY_APP_URL", ""))

    import bittensor as bt  # type: ignore[import-not-found]
    import uvicorn

    wallet = bt.Wallet(name=settings.wallet_name, hotkey=settings.wallet_hotkey)
    network = settings.subtensor_chain_endpoint or settings.subtensor_network
    subtensor = bt.Subtensor(network=network)
    if not subtensor.is_hotkey_registered(
        netuid=settings.netuid, hotkey_ss58=wallet.hotkey.ss58_address
    ):
        raise SystemExit(
            f"hotkey {wallet.hotkey.ss58_address} is not registered on "
            f"netuid {settings.netuid}"
        )
    metagraph = subtensor.metagraph(settings.netuid)

    axon_kwargs: dict[str, Any] = {"wallet": wallet, "port": settings.axon_port}
    if settings.axon_external_ip:
        axon_kwargs["external_ip"] = settings.axon_external_ip
    axon = bt.Axon(**axon_kwargs)
    axon.serve(netuid=settings.netuid, subtensor=subtensor)

    miner = CustomMiner(
        settings, solver, wallet=wallet, subtensor=subtensor, metagraph=metagraph
    )
    print(
        f"[custom-miner] serving netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"port={settings.axon_port}"
    )
    uvicorn.run(
        build_demo_miner_app(miner),
        host=settings.axon_host,
        port=settings.axon_port,
        log_level="info",
    )


if __name__ == "__main__":
    run_custom_miner()

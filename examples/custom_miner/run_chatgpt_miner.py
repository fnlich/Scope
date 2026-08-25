#!/usr/bin/env python3
"""Run the subnet miner with fnlich/Automation's ChatGPT driver as the solver.

    POST /solve  ->  fresh ChatGPT conversation  ->  self-grade against the
    public examples with the validator's own executor  ->  repair on failure
    ->  signed SolutionPayload

Everything the subnet cares about (signature verification, replay defence,
validator-permit authorisation, response signing, byte and concurrency caps)
comes from the reference miner unchanged; only the answer is ours.

Setup — one browser per ChatGPT account gives a true N-fold rate limit. Start
each with ``scripts/start_debug_browser.sh`` on its own port and sign in to
chatgpt.com by hand:

    ./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/gpt-1
    ./scripts/start_debug_browser.sh --port 9223 --profile ~/.hone-miner/chrome/gpt-2
    CHATGPT_CDP=9222,9223 python examples/custom_miner/run_chatgpt_miner.py

Environment (miner settings come from .env as usual):

    CHATGPT_CDP               CDP endpoints, one per account  (default 9222)
    CHATGPT_TABS_PER_BROWSER  conversations per browser       (default 2)
    SOLVER_MAX_ATTEMPTS       initial answer + repairs        (default 3)
    SOLVER_SAFETY_MARGIN_S    headroom kept before the cutoff (default 15)
    SOLVER_MAX_BUDGET_S       hard cap on one solve           (default 240)
    SOLVER_VERIFY_EXECUTOR    subprocess | docker             (default subprocess)
"""

from __future__ import annotations

import sys
from pathlib import Path

# preflight.py is a sibling of this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import require_linux  # noqa: E402

# First, before anything else can fail: on a non-Linux box the project imports
# below raise ModuleNotFoundError from somewhere inside bittensor, which
# explains nothing about why. This says why.
require_linux("The custom miner")

import asyncio  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

from custom_miner import CustomMiner  # noqa: E402
from rlvr.neurons.demo_miner import DemoMinerSettings, build_demo_miner_app  # noqa: E402
from solvers.chatgpt_web import ChatGPTPool  # noqa: E402
from solvers.config import find_env_file, load_env_file  # noqa: E402
from solvers.verify import VerifyingSolver  # noqa: E402


def _cdp() -> str:
    """CDP endpoints for the ChatGPT browsers, one per account."""
    from solvers.browser_pool import DEFAULT_CDP_PORT

    return os.environ.get("CHATGPT_CDP", "").strip() or str(DEFAULT_CDP_PORT)


def build_solver() -> VerifyingSolver:
    pool = ChatGPTPool(
        _cdp(),
        tabs_per_browser=int(os.environ.get("CHATGPT_TABS_PER_BROWSER", "2")),
    )
    return VerifyingSolver(
        pool,
        max_attempts=int(os.environ.get("SOLVER_MAX_ATTEMPTS", "3")),
        safety_margin_s=float(os.environ.get("SOLVER_SAFETY_MARGIN_S", "15")),
        max_budget_s=float(os.environ.get("SOLVER_MAX_BUDGET_S", "240")),
    )


def main() -> None:
    # Same reason as run_miner.py: pydantic-settings reads .env directly and
    # leaves os.environ untouched, so every CHATGPT_*/SOLVER_* knob documented
    # as living in .env would otherwise be silently ignored here.
    env_file = find_env_file()
    if load_env_file(env_file):
        print(f"[chatgpt-miner] loaded {env_file}")
    settings = DemoMinerSettings()
    solver = build_solver()
    pool: ChatGPTPool = solver._backend  # type: ignore[assignment]

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
    app = build_demo_miner_app(miner)

    @app.get("/solver-status")
    async def _status() -> dict[str, Any]:
        """Operational view: browser-backed miners fail quietly, so watch this."""
        return solver.stats()

    print(
        f"[chatgpt-miner] netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"port={settings.axon_port} cdp={_cdp()}"
    )

    # The browser pool and the HTTP server must share ONE event loop: Playwright
    # objects are bound to the loop that created them. That also rules out
    # FastAPI's startup hook here — build_demo_miner_app already installs a
    # `lifespan`, and Starlette silently ignores `on_event` when one is set, so
    # a pool opened that way would never start and every solve would hang.
    async def serve() -> None:
        await pool.start()
        if pool.stats()["tabs"] < settings.miner_max_concurrent_requests:
            print(
                f"[chatgpt] NOTE: {pool.stats()['tabs']} tab(s) but "
                f"MINER_MAX_CONCURRENT_REQUESTS={settings.miner_max_concurrent_requests}. "
                "Tasks beyond the tab count queue and burn their deadline — add "
                "browsers/tabs, or lower the concurrency."
            )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=settings.axon_host,
                port=settings.axon_port,
                log_level="info",
            )
        )
        try:
            await server.serve()
        finally:
            await solver.aclose()

    asyncio.run(serve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

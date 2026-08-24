#!/usr/bin/env python3
"""Run the miner on-chain with Claude and/or ChatGPT as the solver.

    MINER_BACKENDS=claude           python examples/custom_miner/run_miner.py
    MINER_BACKENDS=claude,chatgpt   python examples/custom_miner/run_miner.py

Both backends drive a Firefox profile you are already logged in to:

    claude    CLAUDE_PROFILES    (default ~/.hone-miner/firefox/claude-1)
    chatgpt   CHATGPT_PROFILES   (default ~/.hone-miner/firefox/chatgpt-1)

Log in once per account with ``python -m solvers.login <backend>``. No API key
is read anywhere. Each backend answers into the same
self-verify-and-repair loop, and with more than one they form a fallback chain:
the first provider whose answer reproduces every public example wins, and the
rest are never called for that task.

Run two, if you can. A browser miner has no API path to fall back to, so a
second logged-in provider is the only thing standing between one expired login
and a run of zeros.

This is the provider-agnostic entrypoint. ``run_chatgpt_miner.py`` remains as
the ChatGPT-only launcher; it is equivalent to ``MINER_BACKENDS=chatgpt`` here.

Before either backend serves a registered hotkey, verify its selectors once:

    cd examples/custom_miner && python -m solvers.doctor claude --probe

Chain identity comes from .env as usual: NETUID, SUBTENSOR_NETWORK,
WALLET_NAME, WALLET_HOTKEY, AXON_PORT, AXON_EXTERNAL_IP.
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
from solvers.config import find_env_file, load_env_file  # noqa: E402
from solvers.multi import build_solver, warm_up  # noqa: E402


def main() -> None:
    # The miner's own settings come from .env via pydantic-settings, which reads
    # the file directly and never touches os.environ — so backend knobs written
    # there (MINER_BACKENDS, CLAUDE_PROFILES, selector overrides) would otherwise be
    # silently ignored. Real environment variables still win.
    env_file = find_env_file()
    if load_env_file(env_file):
        print(f"[miner] loaded {env_file}")
    settings = DemoMinerSettings()
    solver = build_solver()

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
        """Operational view. Watch it — a miner fails quietly, and silence from
        a provider looks exactly like success until the score drops."""
        return solver.stats()

    print(
        f"[miner] netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"port={settings.axon_port} "
        f"backends={os.environ.get('MINER_BACKENDS', 'claude')}"
    )

    # One event loop for everything: the browser backends' Playwright objects are
    # loop-bound, and build_demo_miner_app already installs a `lifespan`, so a
    # FastAPI startup hook would be silently ignored.
    async def serve() -> None:
        # Start browser pools here, where a failure is visible, rather than
        # lazily on the first validator request.
        await warm_up(solver, settings.miner_max_concurrent_requests)
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

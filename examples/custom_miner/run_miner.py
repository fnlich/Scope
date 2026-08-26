#!/usr/bin/env python3
"""Run the miner on-chain, answering with the browsers you started.

    python examples/custom_miner/run_miner.py

You run the browsers — several Chrome instances, each signed in by hand to one
provider, each on its own debugging port. This attaches to all of them and
treats their tabs as ONE fleet, handing each task to the next free tab:

    CLAUDE_CDP=9222,9223,9224     browsers signed in to claude.ai
    CHATGPT_CDP=9225,9226         browsers signed in to chatgpt.com

Set either list or both. N browsers give N accounts' worth of throughput,
because a task goes to one tab rather than to every provider in turn. Accounts
are the rate limit that actually binds, so that is the axis worth scaling.

Start each browser with ``./scripts/start_debug_browser.sh`` and sign in by
hand. No API key is read anywhere.

Every answer goes through the self-verify-and-repair loop in ``verify.py``: it
is graded against the task's public examples with the validator's own executor,
and a failure is handed back to the model as concrete evidence. If a task still
will not verify, ``SOLVER_SECOND_OPINION`` (on by default) asks the *other*
model once — cheap insurance when the whole payment rides on a complete pass.

Before serving a registered hotkey, verify the selectors once per provider:

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
import signal  # noqa: E402
from typing import Any  # noqa: E402

from custom_miner import CustomMiner  # noqa: E402
from rlvr.neurons.demo_miner import DemoMinerSettings, build_demo_miner_app  # noqa: E402
from solvers.config import find_env_file, load_env_file  # noqa: E402
from solvers.roster import build_solver, describe, roster, warm_up  # noqa: E402


def main() -> None:
    # The miner's own settings come from .env via pydantic-settings, which reads
    # the file directly and never touches os.environ — so backend knobs written
    # there (CLAUDE_CDP, CHATGPT_CDP, selector overrides) would otherwise be
    # silently ignored. Real environment variables still win.
    env_file = find_env_file()
    if load_env_file(env_file):
        print(f"[miner] loaded {env_file}")
    settings = DemoMinerSettings()
    # Read the roster once: building it twice would repeat its warnings, and
    # the printed summary must describe the fleet the solver actually got.
    browsers = roster()
    solver = build_solver(browsers)

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
        f"port={settings.axon_port} browsers: {describe(browsers)}"
    )

    # One event loop for everything: the browser backends' Playwright objects are
    # loop-bound, and build_demo_miner_app already installs a `lifespan`, so a
    # FastAPI startup hook would be silently ignored.
    async def serve() -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=settings.axon_host,
                port=settings.axon_port,
                log_level="info",
            )
        )
        # Install the stop handlers BEFORE the slow part. uvicorn installs its
        # own once serving, but attaching to a fleet of browsers takes seconds,
        # and a supervisor restarting during that window would otherwise raise
        # KeyboardInterrupt straight through the cleanup — leaving the tabs it
        # had already opened behind in your browsers.
        loop = asyncio.get_running_loop()

        stopped = False

        def stop(signame: str) -> None:
            # uvicorn installs its own handler once it is serving, so both can
            # fire for one signal; say it once and let the second be a no-op.
            nonlocal stopped
            if stopped:
                return
            stopped = True
            print(f"[miner] {signame}: closing tabs and disconnecting")
            server.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop, sig.name)
            except NotImplementedError:  # pragma: no cover - not on Linux
                pass

        # Attach here, where a failure is visible, rather than lazily on the
        # first validator request.
        await warm_up(solver, settings.miner_max_concurrent_requests)
        try:
            await server.serve()
        finally:
            # Shielded so a second signal cannot interrupt the cleanup itself.
            await asyncio.shield(asyncio.ensure_future(solver.aclose()))

    asyncio.run(serve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

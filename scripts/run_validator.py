#!/usr/bin/env python
"""Live V1 validator entrypoint.

Obtain a public problem from the private source, query miners, commit their
signed responses, reveal hidden tests, run the sandbox locally, compute local
scores, and set the validator's own weights.

Prerequisites (see scripts/register_testnet.sh):
  - a funded coldkey+hotkey, registered on your subnet's NETUID
  - .env set: NETUID, SUBTENSOR_NETWORK, WALLET_NAME, WALLET_HOTKEY
    and PROBLEM_SERVER_URL

    . .venv/bin/activate && PYTHONPATH=. python scripts/run_validator.py
"""

from rlvr.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.problem_server_url:
        raise SystemExit("set PROBLEM_SERVER_URL for the V1 problem source")

    from rlvr.neurons.decentralized import run_decentralized_validator

    run_decentralized_validator(settings)


if __name__ == "__main__":
    main()

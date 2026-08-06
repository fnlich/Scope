# RLVR subnet

This repository contains the validator for RLVR on Bittensor Finney, NETUID 5.
Validators lease a public coding challenge, rotation-deal it to half of the
serving miners,
commit the signed responses, reveal the hidden tests, grade every response in a
local Docker sandbox, and submit the resulting weights on chain.

## Launch emission policy

This launch release hard-codes `OWNER_BURN_SHARE` at 40%. After the completed-
observation gate opens, the validator dynamically resolves the subnet owner's
currently registered hotkey to its metagraph UID and reserves 40% of the
submitted weight vector for that UID. Finney NETUID 5 is currently configured
for `Burn`, so the runtime destroys that owner-directed miner incentive; it
does not credit the owner. Positive-scoring miners split the remaining 60% in
proportion to score. If every miner has zero score after valid completed
observations, the validator directs 100% of the vector to burn.

The owner UID is never hard-coded. The validator refuses its first submission
if it cannot safely resolve the owner, and falls back to the ordinary score
vector without a reserved burn share if a previously valid mapping becomes
inconsistent. If the chain mode is no longer `Burn`, it submits miner-only
weights and logs an error.

Runtime 440 separately applies `MinerBurned` before network-wide emission shares
are renormalized. At this launch setting and NETUID 5's present scale, that
reduces NETUID 5's share of network emission by close to 40% and reallocates the
difference to other subnets. Changing the published 40% constant requires a
visible validator release; it is not an environment override.

## Run a validator

Requirements:

- Linux or macOS with Python 3.10–3.12
- Docker with the daemon running
- a registered validator hotkey on Finney NETUID 5
- a system clock synchronized with NTP

The launch configuration can run up to 16 one-CPU, 256 MiB grading containers
at once. A 16-vCPU, 16-GiB host is recommended; hosts with fewer CPU cores will
grade each sampled half more slowly but will not reduce the sample size.
Use a stable broadband connection: the production server rejects a commit
whose request body takes more than 120 seconds to upload.
The validator records its first four completed challenges before submitting an
on-chain weight vector; paced or unavailable rounds do not advance that gate.
The SN5 defaults are 75 blocks (about 15 minutes) between challenge attempts and
180 blocks between weight attempts. The problem service remains authoritative:
when an attempt is too early or the shared global lease slot is busy, its
`Retry-After` response defers leasing without blocking weight scheduling. At
startup the validator reads the chain weight rate limit and raises the effective
weight interval to at least that limit plus 20 blocks.

After cloning this repository, run the bootstrap with the names of your
existing Bittensor wallet and hotkey from the repository root:

```bash
./setup_validator.sh --wallet-name YOUR_WALLET --wallet-hotkey YOUR_HOTKEY
./start_validator.sh
```

The setup is safe to rerun. It creates `.venv` and `.env`, installs the pinned
chain dependencies, pulls the immutable multi-architecture sandbox image, runs
a real Python/GNU `timeout` container smoke test, and checks the production
problem server and local clock. It does not create, read, or modify wallet keys.

If you omit the wallet arguments, edit only `WALLET_NAME` and `WALLET_HOTKEY`
in `.env`, then run `./start_validator.sh`. The production URL, Finney network,
NETUID 5, half-pool rotation, and sandbox image are already configured.

Run the tests with:

```bash
. .venv/bin/activate
pip install -e '.[chain,dev]'
pytest -q
```

## Protocol

```text
private problem server
        |
        | public challenge
        v
validator ---- signed request ----> rotating half of serving miners
        |<--- signed responses -----|
        |
        | commit exact response set
        | retrieve hidden cases
        v
local Docker sandbox
        |
        v
verified rewards -> recent score window -> on-chain weights
```

The commit-before-reveal sequence prevents hidden evaluation cases from being
included in miner requests. The server never grades miner code or controls
validator weights. Each validator evaluates the signed responses locally.
A rejected lease consumes no problem. Once a challenge is leased, ordinary
failed miner calls—including timeouts and invalid responses—remain in the exact
committed submission list with their recorded error; they are not silently
dropped before reveal.

## Demo miner and development

A small GLM-5.2-backed demo miner is included only as a protocol reference. It
requires its own `GLM_API_KEY`; validators do not need a model-provider key.
See [`docs/DEMO_MINER.md`](docs/DEMO_MINER.md).

The subprocess executor and test-network defaults are for development only.
See [`docs/TESTNET_RUNBOOK.md`](docs/TESTNET_RUNBOOK.md) and
[`docs/DESIGN.md`](docs/DESIGN.md) for the testnet workflow and trust model.

## Repository contents

- `rlvr/problemserver/`: versioned challenge lease, commit, reveal, and
  feedback contracts plus the authenticated client.
- `rlvr/neurons/`: validator lifecycle and signed miner transport.
- `rlvr/execution/`: fail-closed Docker sandbox.
- `rlvr/scoring/`: local verification, reward allocation, and score state.
- `rlvr/protocol.py`: validator/miner wire models and signatures.

Problem construction is not part of this repository.

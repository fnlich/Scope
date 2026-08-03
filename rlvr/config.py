"""Validator configuration.

Problem-source transport, sandbox verification, scoring, and chain settings
belong here. The optional demo miner keeps its model-provider settings in
``rlvr.neurons.demo_miner``.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Wall-clock a responder gets per problem. The dispatch path adds a small
    # wire grace period and enforces a true total-duration deadline around the
    # full response; HTTPX's own timeout is only a per-phase inactivity bound.
    # Too-short deadlines zero out honest miners on exactly the in-band-hard
    # problems the curriculum targets.
    solve_deadline_s: float = Field(default=300.0, gt=0.0, le=3600.0)
    # --- Difficulty classification ---
    band_low: float = Field(default=0.35, ge=0.0, le=1.0)
    band_high: float = Field(default=0.65, ge=0.0, le=1.0)
    # Minimum number of responses required before trusting a band placement.
    difficulty_samples_m: int = Field(default=8, ge=1, le=1024)

    # --- Verification / sandbox ---
    # Candidate code is adversarial. Docker is the fail-closed production
    # default; the subprocess backend must be selected explicitly for local
    # development because it cannot block filesystem reads or direct network
    # access on Linux.
    executor: str = Field(
        default="docker",
        pattern="^(subprocess|docker)$",
        description="'subprocess' | 'docker'",
    )
    per_test_timeout_s: float = Field(default=5.0, gt=0.0, le=300.0)
    determinism_runs: int = Field(default=2, ge=1, le=10)
    docker_image: str = Field(
        default="python:3.12-slim",
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*$",
    )
    docker_memory: str = Field(
        default="256m", pattern=r"^[1-9][0-9]*[kKmMgG]$"
    )
    docker_cpus: float = Field(default=1.0, gt=0.0, le=256.0)
    docker_pids_limit: int = Field(default=128, ge=16, le=4096)
    reward_partial_credit: bool = False  # default near-binary: all-pass=1 else 0

    validator_challenges_per_round: int = Field(default=1, gt=0, le=10_000)

    # --- Payment (chain emissions), decoupled from the training reward ---
    # pay_i = reward_i * (floor + (1-floor)*(1 - p_loo)): a pass nobody else
    # achieved pays full, a pass everyone achieved pays `floor`. 1.0 = flat pay
    # (old behavior). The floor keeps answering easy problems rational, so
    # measured pass-rates keep reflecting difficulty rather than effort.
    payment_difficulty_floor: float = Field(default=0.25, ge=0.0, le=1.0)
    # Correct responses are additionally weighted by validator-measured
    # round-trip latency relative to the fastest correct response in the same
    # dispatch. Every three minutes of additional latency halves the 5% share
    # above the 0.95 floor. This makes speed a slight, continuous tiebreaker:
    # relative penalties are about 0.55% at 30s, 1.03% at 60s, and 3.43% at
    # five minutes, while correctness remains the dominant term.
    payment_speed_half_life_ms: float = Field(default=180_000.0, gt=0.0)
    payment_speed_floor: float = Field(default=0.95, ge=0.0, le=1.0)

    # --- Dispatch (subset sampling over the miner pool) ---
    # By default each problem is rotation-dealt to half of the serving pool.
    # An explicit 0 preserves the legacy full-pool behavior; a positive value
    # is a fixed-count override. Challenge quorum may raise the chosen value.
    dispatch_subset_k: int | None = Field(default=None, ge=0)
    dispatch_subset_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    # --- Private problem-source client ---
    problem_server_url: str = ""
    # HTTPS authenticates problem-server responses; Epistula authenticates
    # validator requests. Plain HTTP is local-test only.
    problem_server_allow_insecure_http: bool = False
    # Exact miner response bodies are committed and archived. Keep this
    # separate from the larger cap for problem-server challenge/reveal
    # responses so full-pool commits remain predictably bounded.
    miner_max_response_bytes: int = Field(default=128_000, gt=0, le=1_000_000)
    problem_max_response_bytes: int = Field(default=512_000, gt=0, le=10_000_000)
    problem_server_request_timeout_s: float = Field(default=60.0, gt=0.0)
    problem_server_metagraph_sync_s: float = Field(default=60.0, gt=0.0)
    # HTTP fan-out width when querying miners. I/O-bound: full-pool dispatch
    # needs one in-flight request per serving miner or slow solvers serialize
    # into deadline-length waves.
    validator_dispatch_concurrency: int = Field(default=256, ge=1, le=1024)
    # Requests waiting for one of these short-lived send-start slots remain
    # unsigned. The slot is released after the request body reaches the HTTP
    # transport, so miner solve time does not serialize the fan-out.
    validator_send_concurrency: int = Field(default=32, ge=1, le=1024)
    # Concurrent sandbox verifications (each spawns per-test subprocesses or
    # Docker containers). Bounded separately from the HTTP fan-out so a
    # full-pool dispatch cannot thrash the executor.
    validator_verify_concurrency: int = Field(default=16, ge=1, le=256)
    validator_score_state_file: str = "data/validator_scores.json"

    # --- Scoring (bounded recent history -> weights) ---
    # Retained for compatibility with existing configuration and score-state
    # files. Scoring is count-bounded by SCORE_WINDOW_MAX_SAMPLES, not by age.
    score_window_seconds: float = Field(default=57_600.0, gt=0.0, le=604_800.0)
    score_window_max_samples: int = Field(default=200, ge=1, le=1024)
    # Thin-evidence floor: first/second/third successes -> 1/4, 1/2, 3/4.
    score_window_min_samples: int = Field(default=4, ge=1, le=1024)
    # Do not submit a normalized on-chain vector from a thinner history. This
    # is a validator-startup gate; completed challenges still score and persist.
    validator_min_weight_observations: int = Field(default=4, ge=1, le=1024)
    # Record zeros for dispatched/non-serving miners on completed challenges so
    # a disconnected miner stops draining emissions. Disable to freeze on silence.
    score_decay_nonresponders: bool = True

    # --- Dataset export ---
    dataset_dir: str = "data/rollouts"

    # --- Bittensor (live testnet) ---
    netuid: int = Field(default=0, ge=0)
    subtensor_network: str = "test"
    subtensor_chain_endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"

    # Validator request cadence (blocks). The private problem source owns the
    # actual burn rate and may pace a round with HTTP 429; a denied lease
    # consumes no problem. Lower this only for a quick on-chain smoke test.
    round_interval_blocks: int = Field(default=75, ge=0)
    weights_interval_blocks: int = Field(default=180, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "Settings":
        if self.band_low > self.band_high:
            raise ValueError("BAND_LOW must be <= BAND_HIGH")
        if self.score_window_min_samples > self.score_window_max_samples:
            raise ValueError(
                "SCORE_WINDOW_MIN_SAMPLES must be <= SCORE_WINDOW_MAX_SAMPLES"
            )
        if (
            self.validator_min_weight_observations
            > self.score_window_max_samples
        ):
            raise ValueError(
                "VALIDATOR_MIN_WEIGHT_OBSERVATIONS must be <= "
                "SCORE_WINDOW_MAX_SAMPLES"
            )
        return self


def get_settings() -> Settings:
    """Load settings from environment / .env."""
    return Settings()

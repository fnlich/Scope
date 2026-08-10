"""V1 validator: private problem source, decentralized local evaluation.

Validators obtain a public challenge, query miners, commit the miner-signed
response bytes, reveal hidden tests, and only then execute/score locally. The
private server has no grading, score, or weight API.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, cast

import httpx
import numpy as np

from ..config import Settings, get_settings
from ..dataset.writer import RolloutWriter
from ..execution import get_executor
from ..orchestrator import Orchestrator, RotationSampler
from ..policy import (
    LEGACY_SCORE_WINDOW_SECONDS,
    RELEASE_POLICY,
    ValidatorPolicy,
)
from ..problemserver.api import (
    ChallengeFeedback,
    FeedbackVerdict,
    MinerSubmission,
    PublicChallenge,
    derive_request_id,
)
from ..problemserver.client import (
    LeaseCategory,
    ProblemServerClient,
    next_lease_not_before,
    require_secure_problem_url,
)
from ..protocol import SignedSolution
from ..scoring.eval_engine import EvalEngine
from ..scoring.verifier import Verifier
from ..types import ChallengeResult, Problem, SolutionResponse, TestCase
from .live import LiveSolverClient, SendGate, _solver_clients
from .validator import ValidatorNeuron

_MINER_RESPONSE_GRACE_S = 10.0
OWNER_BURN_SHARE = RELEASE_POLICY.owner_burn_share
_OWNER_BURN_CACHE_TTL_S = 3_600.0
_SANDBOX_ERROR_PROBE_THRESHOLD = 0.25
_U16_MAX = 65_535
_WEIGHTS_RATE_LIMIT_MARGIN = 20
_MAX_WEIGHT_DETAIL_CHARS = 200
_MAX_WEIGHT_FIELD_CHARS = 80


@dataclass
class _OwnerBurnState:
    """Last atomically resolved owner-hotkey and recycle/burn mode pair."""

    owner_hotkey: Optional[str] = None
    mode: Optional[str] = None
    checked_at: float = 0.0
    last_valid_hotkey: Optional[str] = None


@dataclass
class _CapturedSolver:
    uid: int
    hotkey: str
    solution: SolutionResponse
    responded: bool = False

    async def solve(self, problem: Problem, prompt: str) -> SolutionResponse:
        return self.solution


def _weight_result_status(result: object) -> tuple[bool, str]:
    """Normalize legacy tuple/bool and modern Bittensor extrinsic results."""
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success), str(getattr(result, "message", "") or "")
    if isinstance(result, tuple):
        ok = bool(result[0]) if result else False
        message = str(result[1]) if len(result) > 1 else ""
        return ok, message
    return bool(result), ""


def _weight_failure_detail(result: object) -> str:
    """Return bounded single-line SDK ``error``/``data`` without raising."""

    def render_field(name: str) -> str:
        try:
            value = getattr(result, name)
        except Exception:  # noqa: BLE001 - diagnostics must never break weights
            return ""
        try:
            if value is None or value is False:
                return ""
            if isinstance(value, str):
                rendered = value
            else:
                if hasattr(value, "__len__") and len(value) == 0:
                    return ""
                if isinstance(value, (int, float)) and value == 0:
                    return ""
                rendered = repr(value)
        except Exception:  # noqa: BLE001 - hostile SDK values are non-fatal
            return ""
        rendered = " ".join(rendered.split())
        rendered = "".join(ch for ch in rendered if ch.isprintable())
        if not rendered:
            return ""
        if len(rendered) > _MAX_WEIGHT_FIELD_CHARS:
            rendered = rendered[: _MAX_WEIGHT_FIELD_CHARS - 3] + "..."
        return f"{name}={rendered}"

    parts = [part for name in ("error", "data") if (part := render_field(name))]
    return " ".join(parts)[:_MAX_WEIGHT_DETAIL_CHARS]


def _weights_rate_limited(
    blocks_elapsed: Optional[int], chain_limit: Optional[int]
) -> bool:
    """Mirror the chain's strict admission boundary for diagnostics."""
    return bool(
        blocks_elapsed is not None
        and blocks_elapsed >= 0
        and chain_limit is not None
        and chain_limit > 0
        and blocks_elapsed < chain_limit
    )


def _weight_failure_report(result: object, validator: ValidatorNeuron) -> str:
    """Explain a failed, otherwise silent weight submission without raising."""
    try:
        ok, message = _weight_result_status(result)
    except Exception:  # noqa: BLE001 - diagnostics must not mask submission
        ok, message = False, ""
    if ok or message:
        return ""

    detail = _weight_failure_detail(result)
    if detail:
        return detail

    try:
        current = int(validator.subtensor.get_current_block())
    except Exception:  # noqa: BLE001 - retain the context that is available
        current = None
    try:
        uid = validator.uid
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
            raise ValueError("invalid validator uid")
        last_update = int(validator.metagraph.last_update[uid])
    except Exception:  # noqa: BLE001 - optional diagnostic context
        last_update = None
    elapsed = (
        current - int(last_update)
        if current is not None and last_update is not None
        else None
    )
    chain_limit = getattr(validator, "chain_weights_rate_limit", None)
    effective = getattr(validator, "weights_interval_blocks", None)
    report = (
        f"current_block={current if current is not None else 'unknown'} "
        f"last_update_block={last_update if last_update is not None else 'unknown'} "
        f"blocks_elapsed={elapsed if elapsed is not None else 'unknown'} "
        f"chain_limit={chain_limit if chain_limit is not None else 'unknown'} "
        f"effective_interval={effective if effective is not None else 'unknown'}"
    )
    if elapsed is not None and chain_limit is not None:
        report += f" rate_limited={_weights_rate_limited(elapsed, chain_limit)}"
    return report


def _read_weights_rate_limit(
    validator: ValidatorNeuron, netuid: int
) -> Optional[int]:
    """Read a trustworthy positive chain limit, or fall back with a warning."""
    try:
        reader = getattr(validator.subtensor, "weights_rate_limit")
        value = reader(netuid)
    except Exception as error:  # noqa: BLE001 - startup must retain safe fallback
        print(
            "[validator] WARN: could not read chain weights rate limit; "
            f"using configured interval ({error})"
        )
        return None
    if type(value) is not int or value <= 0:
        print(
            "[validator] WARN: chain returned an invalid weights rate limit; "
            "using configured interval"
        )
        return None
    return value


def effective_weights_interval(
    configured: int, chain_limit: Optional[int]
) -> int:
    """Apply the chain-derived rate-limit floor plus its fixed safety margin."""
    configured_interval = int(configured)
    if type(chain_limit) is not int or chain_limit <= 0:
        return configured_interval
    return max(
        configured_interval,
        chain_limit + _WEIGHTS_RATE_LIMIT_MARGIN,
    )


def _apply_weights_rate_limit(
    validator: ValidatorNeuron,
    settings: Settings,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> None:
    """Install the startup chain-derived cadence and retain its source value."""
    chain_limit = _read_weights_rate_limit(validator, settings.netuid)
    validator.chain_weights_rate_limit = chain_limit
    validator.weights_interval_blocks = effective_weights_interval(
        policy.weights_interval_blocks,
        chain_limit,
    )


def _validator_http_limits(settings: Settings) -> httpx.Limits:
    """Size the connection pool for a true single-wave full-pool dispatch."""
    dispatch = max(1, int(settings.validator_dispatch_concurrency))
    return httpx.Limits(
        # Reserve a few slots for the problem server and metagraph turnover
        # while every configured miner-dispatch permit is in use.
        max_connections=dispatch + 8,
        max_keepalive_connections=min(dispatch, 64),
    )


def _weight_observation_count(engine: EvalEngine) -> int:
    """Largest authoritative per-uid history available for weight evidence."""
    return max((len(history) for history in engine.histories.values()), default=0)


def _chain_mode_name(value: object) -> str:
    """Normalize the decoded RecycleOrBurn storage value without guessing."""
    raw = getattr(value, "value", value)
    if isinstance(raw, dict) and len(raw) == 1:
        raw = next(iter(raw))
    name = getattr(raw, "name", None)
    if isinstance(name, str) and name:
        raw = name
    text = str(raw).strip()
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    return text.strip(" <>'\"")


def _refresh_owner_burn_state(
    validator: ValidatorNeuron,
    state: _OwnerBurnState,
    settings: Settings,
    *,
    force: bool = False,
    now: Optional[float] = None,
) -> bool:
    """Refresh owner and mode together, retaining a last-known-good pair."""
    current = time.monotonic() if now is None else float(now)
    has_pair = bool(state.owner_hotkey and state.mode)
    if (
        not force
        and has_pair
        and current - state.checked_at < _OWNER_BURN_CACHE_TTL_S
    ):
        return True
    try:
        block = int(validator.subtensor.get_current_block())
        owner_hotkey = str(
            validator.subtensor.get_subnet_owner_hotkey(
                settings.netuid, block=block
            )
            or ""
        ).strip()
        mode = _chain_mode_name(
            validator.subtensor.query_subtensor(
                "RecycleOrBurn", params=[settings.netuid], block=block
            )
        )
        if not owner_hotkey or not mode:
            raise ValueError("chain returned an empty owner hotkey or mode")
    except Exception as exc:  # noqa: BLE001 - stale safe cache is intentional
        if has_pair:
            print(
                "[validator] WARN: owner/burn-mode refresh failed; "
                f"using last-known-good chain values ({exc})"
            )
            return True
        print(
            "[validator] ERROR: owner/burn-mode resolution failed; "
            f"refusing weight submission ({exc})"
        )
        return False

    # Replace only after both reads have succeeded, so a transient RPC failure
    # can never combine a new owner with a stale mode (or the reverse).
    state.owner_hotkey = owner_hotkey
    state.mode = mode
    state.checked_at = current
    return True


def _owner_position(
    validator: ValidatorNeuron, owner_hotkey: str
) -> tuple[Optional[int], str]:
    """Locate an owner by hotkey; metagraph UIDs need not equal list positions."""
    hotkeys = [str(hotkey) for hotkey in validator.metagraph.hotkeys]
    raw_uids = list(validator.metagraph.uids)
    if len(hotkeys) != len(raw_uids):
        return None, "metagraph hotkey/UID lengths differ"
    try:
        uids = [int(uid) for uid in raw_uids]
    except (TypeError, ValueError):
        return None, "metagraph contains a non-integer UID"
    if len(set(uids)) != len(uids):
        return None, "metagraph contains duplicate UIDs"
    positions = [idx for idx, hotkey in enumerate(hotkeys) if hotkey == owner_hotkey]
    if len(positions) != 1:
        return None, f"owner hotkey matched {len(positions)} metagraph entries"
    return positions[0], ""


def _without_owner(weights: np.ndarray, owner_position: int) -> np.ndarray:
    """Remove an owner from an ordinary miner vector and renormalize."""
    adjusted = np.asarray(weights, dtype=np.float64).copy()
    adjusted[owner_position] = 0.0
    total = float(adjusted.sum())
    if total > 0.0:
        adjusted /= total
    return adjusted


def _build_owner_burn_weights(
    weights: np.ndarray, owner_position: int
) -> np.ndarray:
    """Reserve 40% for burn while preserving every positive miner on chain."""
    base = np.asarray(weights, dtype=np.float64)
    if (
        base.ndim != 1
        or owner_position < 0
        or owner_position >= base.size
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
    ):
        raise ValueError("weights and owner position must form a valid vector")

    miners = base.copy()
    miners[owner_position] = 0.0
    positive = miners > 0.0
    total = float(miners.sum())
    output = np.zeros_like(miners)
    if total <= 0.0:
        output[owner_position] = 1.0
        return output

    # Bittensor 10.5 max-normalizes before uint16 conversion. A floor just
    # above owner_share / 65535 guarantees each positive miner survives as at
    # least one quantum while zero-score miners remain zero.
    min_share = float(
        np.nextafter(OWNER_BURN_SHARE / _U16_MAX, np.inf)
    )
    positive_count = int(np.count_nonzero(positive))
    remaining = (1.0 - OWNER_BURN_SHARE) - positive_count * min_share
    if remaining < 0.0:
        raise ValueError("too many positive miners for the uint16 preservation floor")
    output[positive] = min_share + remaining * (miners[positive] / total)
    output[owner_position] = OWNER_BURN_SHARE
    output[owner_position] += 1.0 - float(output.sum())
    effective = _effective_u16_share(output, owner_position)
    if abs(effective - OWNER_BURN_SHARE) > 0.01:
        raise ValueError("uint16 conversion moved the effective burn unexpectedly")
    return output


def _effective_u16_share(weights: np.ndarray, position: int) -> float:
    """Predict one UID's effective share after Bittensor 10.5 conversion."""
    vector = np.asarray(weights, dtype=np.float64)
    maximum = float(vector.max())
    if maximum <= 0.0:
        return 0.0
    quantized = np.asarray(
        [round(float(weight / maximum) * _U16_MAX) for weight in vector],
        dtype=np.float64,
    )
    total = float(quantized.sum())
    return float(quantized[position] / total) if total > 0.0 else 0.0


def _submit_local_weights(
    validator: ValidatorNeuron,
    engine: EvalEngine,
    settings: Settings,
    *,
    log_response_repr: bool = False,
    owner_burn_state: Optional[_OwnerBurnState] = None,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> Optional[bool]:
    """Submit normalized local weights only after enough completed evidence."""
    observations = _weight_observation_count(engine)
    required = policy.min_weight_observations
    if observations < required:
        print(
            "[validator] local weight evidence "
            f"{observations}/{required}; skipping submission"
        )
        return None

    state = owner_burn_state or _OwnerBurnState()
    if not _refresh_owner_burn_state(validator, state, settings):
        return None

    owner_hotkey = state.owner_hotkey or ""
    owner_position, owner_error = _owner_position(validator, owner_hotkey)
    if owner_position is None:
        # A cached owner missing from a freshly synced metagraph may have
        # changed. Refresh immediately rather than waiting for the hourly TTL.
        if not _refresh_owner_burn_state(
            validator, state, settings, force=True
        ):
            return None
        owner_hotkey = state.owner_hotkey or ""
        owner_position, owner_error = _owner_position(validator, owner_hotkey)
    if owner_position is None:
        if state.last_valid_hotkey is None:
            print(
                "[validator] ERROR: registered owner is not uniquely present in "
                f"the current metagraph; refusing weight submission ({owner_error})"
            )
            return None
        print(
            "[validator] ERROR: registered owner mapping became inconsistent; "
            f"submitting an ordinary vector without a burn reservation ({owner_error})"
        )
        weights = engine.get_weights(n=len(validator.metagraph.hotkeys))
        burn_active = False
    else:
        state.last_valid_hotkey = owner_hotkey
        base = engine.get_weights(n=len(validator.metagraph.hotkeys))
        if (state.mode or "").casefold() == "burn":
            weights = _build_owner_burn_weights(base, owner_position)
            burn_active = True
        else:
            print(
                "[validator] ERROR: chain RecycleOrBurn mode is "
                f"{state.mode!r}, not 'Burn'; submitting miner-only weights"
            )
            weights = _without_owner(base, owner_position)
            burn_active = False

    if float(sum(weights)) <= 0.0:
        print("[validator] local miner weights are all-zero; skipping")
        return None
    uids = [int(uid) for uid in validator.metagraph.uids]
    result = validator.subtensor.set_weights(
        wallet=validator.wallet,
        netuid=settings.netuid,
        uids=uids,
        weights=[float(weight) for weight in weights],
        wait_for_inclusion=True,
        wait_for_finalization=False,
        wait_for_revealed_execution=False,
        max_attempts=1,
    )
    ok, message = _weight_result_status(result)
    failure_detail = _weight_failure_report(result, validator)
    top = max(range(len(weights)), key=lambda idx: weights[idx])
    burn_detail = ""
    if burn_active and owner_position is not None:
        effective_burn = _effective_u16_share(weights, owner_position)
        burn_detail = (
            f" burn_mode=Burn owner_uid={uids[owner_position]} "
            f"configured_burn={OWNER_BURN_SHARE:.2%} "
            f"effective_u16_burn={effective_burn:.2%}"
        )
    response_detail = ""
    if log_response_repr:
        response_repr = repr(result)
        if len(response_repr) > 500:
            response_repr = response_repr[:497] + "..."
        response_detail = f" response={response_repr}"
    print(
        "[validator] set local weights "
        f"response_type={type(result).__name__}{response_detail} "
        f"ok={ok} msg={message!r} "
        f"(top uid={uids[top]} w={float(weights[top]):.4f}){burn_detail}"
        f"{f' failure_detail={failure_detail}' if failure_detail else ''}"
    )
    return ok


def _load_scores(engine: EvalEngine, path: str) -> None:
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)

        version = state.get("version")
        if version not in (1, 2):
            raise ValueError("unsupported score-state version")

        if version == 1:
            scores = np.asarray(state.get("scores", []), dtype=np.float64)
            if (
                scores.ndim != 1
                or not np.all(np.isfinite(scores))
                or np.any(scores < 0.0)
            ):
                raise ValueError("scores must be a finite non-negative vector")
            num_uids = int(scores.size)
        else:
            num_uids = state.get("num_uids")
            if (
                isinstance(num_uids, bool)
                or not isinstance(num_uids, int)
                or num_uids < 0
            ):
                raise ValueError("num_uids must be a non-negative integer")

        hotkeys = {
            int(uid): str(hotkey) for uid, hotkey in state.get("hotkeys", {}).items()
        }
        if any(
            uid < 0 or uid >= num_uids or not hotkey
            for uid, hotkey in hotkeys.items()
        ):
            raise ValueError("hotkey map must reference non-empty in-range UID slots")

        if version == 1:
            if any(scores[uid] > 0.0 and uid not in hotkeys for uid in range(num_uids)):
                raise ValueError("every positive score must retain its owning hotkey")
            # A scalar EMA cannot reconstruct its observations. Filling the new
            # window with that scalar preserves both the current score and the
            # intended one-slot-per-observation sensitivity during migration.
            migrated_at = engine._timestamp()
            histories = {
                uid: [(migrated_at, float(scores[uid]))] * engine.min_samples
                for uid in hotkeys
                if scores[uid] > 0.0
            }
        else:
            stored_window_seconds = state.get("window_seconds")
            if (
                isinstance(stored_window_seconds, bool)
                or not isinstance(stored_window_seconds, (int, float))
                or not np.isfinite(stored_window_seconds)
                or stored_window_seconds <= 0.0
            ):
                raise ValueError("window_seconds must be finite and positive")
            stored_max = state.get("max_samples")
            stored_min = state.get("min_samples")
            if (
                isinstance(stored_max, bool)
                or not isinstance(stored_max, int)
                or stored_max < 1
                or isinstance(stored_min, bool)
                or not isinstance(stored_min, int)
                or not 1 <= stored_min <= stored_max
            ):
                raise ValueError("stored sample bounds are invalid")
            raw_histories = state.get("histories")
            if not isinstance(raw_histories, dict):
                raise ValueError("histories must be a UID-keyed mapping")
            histories: dict[int, list[tuple[float, float]]] = {}
            for raw_uid, raw_values in raw_histories.items():
                uid = int(raw_uid)
                if uid in histories or uid < 0 or uid >= num_uids:
                    raise ValueError("history UID must be unique and in range")
                if (
                    not isinstance(raw_values, list)
                    or not raw_values
                    or len(raw_values) > stored_max
                ):
                    raise ValueError(
                        "each history must contain 1..max_samples observations"
                    )
                values: list[tuple[float, float]] = []
                for item in raw_values:
                    if not isinstance(item, list) or len(item) != 2:
                        raise ValueError(
                            "history observations must be [timestamp, payment]"
                        )
                    raw_timestamp, raw_payment = item
                    if isinstance(raw_timestamp, bool) or isinstance(
                        raw_payment, bool
                    ):
                        raise ValueError("history values must be numeric")
                    timestamp = float(raw_timestamp)
                    payment = float(raw_payment)
                    if (
                        not np.isfinite(timestamp)
                        or timestamp < 0.0
                        or not np.isfinite(payment)
                        or payment < 0.0
                    ):
                        raise ValueError(
                            "history values must be finite and non-negative"
                        )
                    values.append((timestamp, payment))
                histories[uid] = values

            # Histories are authoritative. Stored scores are for operators, so
            # malformed or stale derived values are repaired rather than
            # discarding valid histories and suppressing weights for hours.
            stored_scores = state.get("scores")
            scores_match = False
            try:
                scores = np.asarray(stored_scores, dtype=np.float64)
                if (
                    scores.ndim == 1
                    and scores.size == num_uids
                    and np.all(np.isfinite(scores))
                ):
                    expected = np.zeros(num_uids, dtype=np.float64)
                    for uid, values in histories.items():
                        expected[uid] = sum(
                            payment for _, payment in values
                        ) / max(len(values), stored_min)
                    scores_match = bool(
                        np.allclose(scores, expected, rtol=0.0, atol=1e-9)
                    )
            except (TypeError, ValueError):
                pass
            if not scores_match:
                print(
                    "[validator] WARN: persisted derived scores were stale; "
                    "recomputed from histories"
                )

            # A configured cap change intentionally keeps only the latest
            # completed-problem observations.
            histories = {
                uid: values[-engine.max_samples :]
                for uid, values in histories.items()
            }
            if any(
                sum(payment for _, payment in values) > 0.0
                and uid not in hotkeys
                for uid, values in histories.items()
            ):
                raise ValueError(
                    "every positive history must retain its owning hotkey"
                )

        # Apply only after the entire file validates. A partially parsed state
        # must not restore histories without the hotkeys needed to detect reuse.
        engine._restore(num_uids, histories, hotkeys)
        suffix = " (migrated v1)" if version == 1 else ""
        print(f"[validator] restored local scores for {num_uids} UIDs{suffix}")
        # Make legacy migration one-shot even if the validator crashes before
        # its first round callback has a chance to persist normal score state.
        if version == 1:
            _save_scores(engine, path)
    except FileNotFoundError:
        return
    except Exception as e:  # noqa: BLE001 - corrupt state starts safely at zero
        print(f"[validator] WARN: could not restore local scores ({e})")


def _save_scores(engine: EvalEngine, path: str) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": 2,
                    "num_uids": int(engine.scores.size),
                    "window_seconds": engine.window_seconds,
                    "max_samples": engine.max_samples,
                    "min_samples": engine.min_samples,
                    "scores": [float(score) for score in engine.scores],
                    "histories": {
                        str(uid): list(history)
                        for uid, history in sorted(engine.histories.items())
                    },
                    "hotkeys": {str(uid): hk for uid, hk in engine.hotkeys.items()},
                },
                fh,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as e:
        print(f"[validator] WARN: could not persist local scores ({e})")


async def _dispatch_committed(
    public: PublicChallenge,
    solvers: list[LiveSolverClient],
    concurrency: int,
) -> tuple[list[MinerSubmission], list[_CapturedSolver]]:
    public_problem = public.to_problem(tests=[])
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(solver: LiveSolverClient):
        request_id = derive_request_id(public.challenge_id, solver.uid, solver.hotkey)
        started = time.monotonic()
        try:
            async with sem:
                # HTTPX's scalar timeout is a per-phase inactivity bound, not a
                # whole-request deadline. A hostile miner can otherwise drip
                # bytes forever without triggering it and stall the entire
                # gather. This outer timeout bounds connect + solve + complete
                # response-body consumption in wall-clock time.
                artifact = await asyncio.wait_for(
                    solver.solve_signed(public_problem, request_id),
                    timeout=max(
                        0.001,
                        float(public.deadline_s) + _MINER_RESPONSE_GRACE_S,
                    ),
                )
        except asyncio.TimeoutError:
            artifact = SignedSolution(
                error="<miner total response deadline exceeded>",
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        submission = MinerSubmission(
            uid=solver.uid,
            hotkey=solver.hotkey,
            request_id=request_id,
            response_body=artifact.response_body,
            response_headers=artifact.response_headers,
            error=artifact.error,
            latency_ms=artifact.latency_ms,
        )
        captured = _CapturedSolver(
            uid=solver.uid,
            hotkey=solver.hotkey,
            solution=artifact.to_solution(public.problem_id),
            responded=artifact.responded,
        )
        return submission, captured

    pairs = await asyncio.gather(*(one(solver) for solver in solvers))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _dispatch_subset_size(
    pool_size: int,
    required: int,
    fraction: float,
) -> int:
    """Choose a bounded sample, raising the default to commit quorum."""
    if pool_size <= 0:
        return 0
    requested = math.ceil(pool_size * fraction)
    return min(pool_size, max(1, required, requested))


async def _evaluate_one(
    client: ProblemServerClient,
    orchestrator: Orchestrator,
    rotation: RotationSampler,
    solvers: list[LiveSolverClient],
    settings: Settings,
    quorum_hint: Optional[dict[str, int]] = None,
    pacing: Optional[dict[str, object]] = None,
    now: Optional[float] = None,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> Optional[ChallengeResult]:
    # A lease BURNS the problem server-side (durable FIFO cursor), so refuse to
    # lease when the last-seen quorum requirement already rules this round out.
    # The requirement only travels with a lease, so the first-ever under-quorum
    # round still burns one problem; every later one is skipped for free.
    if quorum_hint is not None:
        known_required = int(quorum_hint.get("required", 0))
        if known_required > 0 and len(solvers) < known_required:
            print(
                f"[validator] WARN: only {len(solvers)} miners serving but "
                f"challenges require {known_required} committed responses; "
                "skipping lease to avoid burning a problem"
            )
            return None
    outcome = await client.lease()
    public = outcome.challenge
    if public is None:
        if outcome.category is LeaseCategory.PACED and pacing is not None:
            pacing["paced"] = True
            if outcome.retry_after_s is not None:
                pacing["not_before"] = next_lease_not_before(
                    outcome.retry_after_s,
                    now=now,
                )
        # An exhausted pool, a paced lease, a rejected validator and an
        # unreachable server all used to print the same line; each is a
        # different operator action, so each says so.
        print(f"[validator] WARN: no public challenge ({outcome.describe()})")
        return None
    required = max(
        public.commit_min_responses, public.commit_min_signed_responses
    )
    if quorum_hint is not None:
        quorum_hint["required"] = required
    if len(solvers) < required:
        print(
            f"[validator] WARN: challenge requires {required} committed miner "
            f"responses but only {len(solvers)} miners are serving"
        )
        return None
    subset_k = _dispatch_subset_size(
        len(solvers),
        required,
        policy.dispatch_fraction,
    )
    subset = cast(
        list[LiveSolverClient],
        rotation.sample(solvers, subset_k),
    )
    if not subset:
        return None

    submissions, captured = await _dispatch_committed(
        public, subset, settings.validator_dispatch_concurrency
    )
    responded = sum(
        bool(
            getattr(
                solution,
                "responded",
                not getattr(submission, "error", ""),
            )
        )
        for submission, solution in zip(submissions, captured, strict=True)
    )
    signed = sum(
        bool(getattr(submission, "response_headers", {}))
        for submission in submissions
    )
    print(
        "[validator] miner response counts "
        f"contacted={len(subset)} responses={responded} signed={signed}"
    )
    receipt = await client.commit(public.challenge_id, submissions)
    if receipt is not None and not receipt.accepted:
        print(f"[validator] WARN: response commit rejected ({receipt.detail})")
        return None
    if receipt is None:
        print("[validator] WARN: response commit failed (server unavailable)")
        return None
    if not receipt.commit_token:
        print("[validator] WARN: accepted response commit omitted its token")
        return None
    if receipt.num_submissions != len(submissions):
        print(
            "[validator] WARN: accepted response commit protocol error "
            f"(receipt count {receipt.num_submissions}, sent {len(submissions)})"
        )
        return None

    reveal = await client.reveal(public.challenge_id, receipt.commit_token)
    if reveal is None or reveal.challenge_id != public.challenge_id:
        print("[validator] WARN: hidden-test reveal failed")
        return None
    if not reveal.tests:
        print("[validator] WARN: challenge revealed no tests; refusing to score")
        return None

    problem = public.to_problem(reveal.tests)
    result = await orchestrator.evaluate(
        problem,
        captured,
        # Verification concurrency, NOT the HTTP fan-out width: each permit
        # holds a sandbox (subprocess/Docker) verification slot.
        asyncio.Semaphore(max(1, settings.validator_verify_concurrency)),
    )
    sandbox_error_rate = _sandbox_error_rate(result)
    if sandbox_error_rate > 0.0:
        print(
            "[validator] WARN: sandbox-error rate "
            f"{sandbox_error_rate:.1%} for challenge {public.challenge_id}"
        )
    if sandbox_error_rate >= _SANDBOX_ERROR_PROBE_THRESHOLD:
        sandbox_healthy = await asyncio.to_thread(
            _sandbox_healthcheck, orchestrator
        )
        if not sandbox_healthy:
            print(
                "[validator] ERROR: independent sandbox health probe failed; "
                "discarding challenge without updating scores"
            )
            return None
        print(
            "[validator] WARN: independent sandbox health probe passed; "
            "retaining candidate failures"
        )
    orchestrator.score_and_export(
        problem,
        result,
        active_uids={solver.uid for solver in solvers},
    )
    outcomes_by_registration = {
        (outcome.uid, outcome.hotkey): outcome for outcome in result.outcomes
    }
    verdicts = []
    for submission, captured_solver in zip(submissions, captured, strict=True):
        # A verdict means an authenticated response was actually graded. HTTP
        # errors, timeouts, and invalid signatures remain committed attempts
        # for response-rate accounting, but are not mislabeled as wrong answers.
        if submission.error or not submission.response_headers:
            continue
        graded = outcomes_by_registration.get(
            (captured_solver.uid, captured_solver.hotkey)
        )
        if graded is None:
            continue
        verdicts.append(
            FeedbackVerdict(
                uid=graded.uid,
                hotkey=graded.hotkey,
                passed=graded.verification.all_passed,
            )
        )
    await client.feedback(
        ChallengeFeedback(
            challenge_id=public.challenge_id,
            pass_rate=result.pass_rate,
            band=result.band,
            num_responses=result.num_responses,
            dup_ratio=result.dup_ratio,
            verdicts=verdicts,
        )
    )
    return result


async def _run_challenge_round(
    validator: ValidatorNeuron,
    client: ProblemServerClient,
    orchestrator: Orchestrator,
    rotation: RotationSampler,
    solvers: list[LiveSolverClient],
    settings: Settings,
    *,
    quorum_hint: Optional[dict[str, int]] = None,
    now: Optional[float] = None,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> int:
    """Evaluate configured challenges, stopping immediately on lease pacing."""
    completed = 0
    for _ in range(policy.challenges_per_round):
        pacing: dict[str, object] = {}
        result = await _evaluate_one(
            client,
            orchestrator,
            rotation,
            solvers,
            settings,
            quorum_hint=quorum_hint,
            pacing=pacing,
            now=now,
            policy=policy,
        )
        completed += int(result is not None)
        if pacing.get("paced"):
            not_before = pacing.get("not_before")
            if isinstance(not_before, (int, float)):
                validator.defer_rounds_until(float(not_before))
            break
    return completed


def _sandbox_error_rate(result: ChallengeResult) -> float:
    """Fraction of outcomes that may reflect a validator sandbox failure."""
    if not result.outcomes:
        return 0.0
    suspected = 0
    for outcome in result.outcomes:
        compile_error = outcome.verification.compile_error or ""
        execution_errors = outcome.verification.results
        if compile_error.startswith("VERIFY_ERROR:") or any(
            item.error_kind == "sandbox_error" for item in execution_errors
        ):
            suspected += 1
    return suspected / len(result.outcomes)


def _sandbox_healthcheck(orchestrator: Orchestrator) -> bool:
    """Verify trusted code through the full verifier and sandbox path."""
    try:
        problem = Problem(
            problem_id="__rlvr_sandbox_healthcheck__",
            statement="Return the input value.",
            entrypoint="__rlvr_sandbox_healthcheck__",
            tests=[TestCase(args=[7], expected=7)],
        )
        verification = orchestrator.verifier.verify(
            problem,
            SolutionResponse(
                problem_id=problem.problem_id,
                code=(
                    "def __rlvr_sandbox_healthcheck__(value):\n"
                    "    return value\n"
                ),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe is the signal
        print(f"[validator] ERROR: sandbox health probe raised ({exc})")
        return False
    return (
        verification.all_passed
        and verification.num_tests == 1
        and verification.num_passed == 1
        and verification.compile_error is None
        and len(verification.results) == 1
        and verification.results[0].passed
        and verification.results[0].error is None
    )


async def _run_decentralized_validator_async(settings: Settings) -> None:
    require_secure_problem_url(
        settings.problem_server_url,
        settings.problem_server_allow_insecure_http,
    )
    policy = RELEASE_POLICY
    validator = ValidatorNeuron(settings, policy=policy)
    validator.setup_bittensor()
    _apply_weights_rate_limit(validator, settings, policy)
    engine = EvalEngine(
        len(validator.metagraph.hotkeys),
        LEGACY_SCORE_WINDOW_SECONDS,
        policy.score_window_max_samples,
        policy.score_window_min_samples,
        decay=policy.decay_nonresponders,
    )
    _load_scores(engine, settings.validator_score_state_file)
    verifier = Verifier(get_executor(settings), settings, policy=policy)
    writer = RolloutWriter(settings.dataset_dir)
    # This path only evaluates challenges returned by the private service.
    orchestrator = Orchestrator(verifier, engine, writer, settings, policy=policy)
    rotation = RotationSampler()
    owner_burn_state = _OwnerBurnState()

    async with httpx.AsyncClient(limits=_validator_http_limits(settings)) as http:
        client = ProblemServerClient(
            settings.problem_server_url,
            validator.wallet,
            http,
            allow_insecure_http=settings.problem_server_allow_insecure_http,
            timeout_s=settings.problem_server_request_timeout_s,
            max_response_bytes=settings.problem_max_response_bytes,
        )

        # Last-seen challenge quorum requirement; survives across rounds so an
        # under-quorum pool stops burning leases after the first observation.
        quorum_hint: dict[str, int] = {}
        dispatch_policy_logged = False
        send_gate = SendGate(settings.validator_send_concurrency)

        async def round_callback(v: ValidatorNeuron) -> dict[int, float]:
            nonlocal dispatch_policy_logged
            await asyncio.to_thread(v.metagraph.sync, subtensor=v.subtensor)
            engine.resize(len(v.metagraph.hotkeys))
            engine.sync({uid: hk for uid, hk in enumerate(v.metagraph.hotkeys)})
            live_solvers = _solver_clients(v, v.wallet, settings, http, gate=send_gate)
            if not live_solvers:
                return {}
            if not dispatch_policy_logged:
                base_sample = _dispatch_subset_size(
                    len(live_solvers),
                    required=0,
                    fraction=policy.dispatch_fraction,
                )
                rule = f"release fraction={policy.dispatch_fraction:g}"
                print(
                    "[validator] dispatch policy "
                    f"sample={base_sample}/{len(live_solvers)} ({rule}); "
                    "challenge quorum may raise the sample"
                )
                dispatch_policy_logged = True

            completed = await _run_challenge_round(
                v,
                client,
                orchestrator,
                rotation,
                live_solvers,
                settings,
                quorum_hint=quorum_hint,
                policy=policy,
            )
            _save_scores(engine, settings.validator_score_state_file)
            print(f"[validator] locally evaluated {completed} challenges")
            return {uid: float(score) for uid, score in enumerate(engine.scores)}

        weight_response_logged = False

        def weight_setter(v: ValidatorNeuron) -> None:
            nonlocal weight_response_logged
            result = _submit_local_weights(
                v,
                engine,
                settings,
                log_response_repr=not weight_response_logged,
                owner_burn_state=owner_burn_state,
            )
            if result is not None:
                weight_response_logged = True

        validator.set_round_callback(round_callback)
        validator.set_weight_setter(weight_setter)
        await validator.run()


def run_decentralized_validator(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not settings.problem_server_url:
        raise SystemExit("V1 decentralized evaluation requires PROBLEM_SERVER_URL")
    asyncio.run(_run_decentralized_validator_async(settings))

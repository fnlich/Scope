"""Convert verified correctness outcomes into chain payments.

What feeds the score history must reward useful correct responses. Paying raw
binary correctness alone would cause every adequate responder to converge to
the same score.

``compute_payments`` therefore applies two transforms on top of the
verification reward:

1. **Difficulty weighting (leave-one-out).**
       pay_i = base_i * (floor + (1 - floor) * (1 - p_loo_i))
   where ``p_loo_i`` is the pool pass-rate EXCLUDING miner i. A pass nobody
   else achieved pays full; a pass everyone achieved pays ``floor``. This pays
   for correct responses to challenges most peers fail. Leave-one-out keeps a
   miner from deflating its own difficulty denominator.

   The ``floor`` is not generosity — it is what keeps the difficulty
   measurement honest. With floor=0, a rational miner skips problems it
   predicts are easy (a pass would pay nothing), so measured pass-rates start
   reflecting *effort allocation* instead of *difficulty*, corrupting the very
   measured signal. With a floor, answering everything is the dominant strategy.

2. **Response-speed weighting.** Among rewarded responses with validator-
   measured latency, the fastest response receives a 1.0 multiplier. Each
   additional ``speed_half_life_ms`` halves the above-floor share, flattening
   out at a configurable floor. Defaults make speed a very light, continuous
   tiebreaker (three-minute half-life, floor 0.95), never the dominant term.
   Correctness remains a hard gate: a fast failing answer receives nothing. If
   a non-networked harness supplies no valid latency, speed weighting is skipped
   so offline evaluation stays meaningful.

TODO(V2): Revisit duplication-resistant scoring only with a signal that cannot
be cheaply gamed. Simple AST fingerprints encourage irrelevant structural
changes in otherwise valid solutions, so V1 deliberately does not use them for
scoring or difficulty classification.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..types import MinerOutcome

DEFAULT_DIFFICULTY_FLOOR = 0.25
DEFAULT_SPEED_HALF_LIFE_MS = 180_000.0
DEFAULT_SPEED_FLOOR = 0.95


def compute_payments(
    outcomes: Iterable[MinerOutcome],
    *,
    difficulty_floor: float = DEFAULT_DIFFICULTY_FLOOR,
    speed_half_life_ms: float = DEFAULT_SPEED_HALF_LIFE_MS,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
) -> dict[int, float]:
    """Map one problem's outcomes to {uid: payment} for the EvalEngine.

    ``base_i`` is the verification reward (binary by default; the fraction
    passed under ``reward_partial_credit``), so a failing miner is paid 0 and
    still appears in the dict (zero enters its recent score history).

    ``difficulty_floor`` in [0, 1]: 1.0 disables difficulty weighting (flat
    pay, the old behavior); 0.0 pays purely by rarity (not recommended — see
    module docstring).

    ``speed_half_life_ms`` controls a relative latency multiplier. For a
    rewarded response arriving one half-life after the fastest, the portion
    above ``speed_floor`` is halved; after two half-lives that portion is
    quartered, and so on. Latency is taken from
    ``SolutionResponse.latency_ms``, which the live validator measures around
    the signed HTTP exchange. When no rewarded response has a finite positive
    latency, the multiplier is 1.0 for everyone.
    """
    outcomes = list(outcomes)
    n = len(outcomes)
    if n == 0:
        return {}
    floor = min(1.0, max(0.0, float(difficulty_floor)))
    latency_floor = min(1.0, max(0.0, float(speed_floor)))
    half_life = float(speed_half_life_ms)

    passes = sum(1 for o in outcomes if o.verification.all_passed)

    rewarded_latencies = [
        float(o.solution.latency_ms)
        for o in outcomes
        if o.verification.reward > 0
        and math.isfinite(float(o.solution.latency_ms))
        and float(o.solution.latency_ms) > 0
    ]
    fastest_latency = min(rewarded_latencies) if rewarded_latencies else None

    def speed_multiplier(outcome: MinerOutcome) -> float:
        if fastest_latency is None or not math.isfinite(half_life) or half_life <= 0:
            return 1.0
        latency = float(outcome.solution.latency_ms)
        if not math.isfinite(latency) or latency <= 0:
            return latency_floor
        delay = max(0.0, latency - fastest_latency)
        return latency_floor + (1.0 - latency_floor) * (2.0 ** (-delay / half_life))

    payments: dict[int, float] = {}
    for o in outcomes:
        base = float(o.verification.reward)
        if base <= 0.0:
            payments[o.uid] = 0.0
            continue
        if n > 1:
            others = passes - (1 if o.verification.all_passed else 0)
            p_loo = others / (n - 1)
        else:
            p_loo = 0.0  # a lone responder defines the pool; pay full
        weight = floor + (1.0 - floor) * (1.0 - p_loo)
        pay = base * weight * speed_multiplier(o)
        payments[o.uid] = pay
    return payments

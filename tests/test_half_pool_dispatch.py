"""Half-pool dispatch sizing and rotation fairness."""

from collections import Counter
from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import _dispatch_subset_size
from rlvr.orchestrator import RotationSampler


@pytest.mark.parametrize(
    ("pool_size", "expected"),
    [(1, 1), (2, 1), (3, 2), (255, 128), (256, 128)],
)
def test_default_sample_is_half_the_serving_pool_rounded_up(pool_size, expected):
    settings = Settings(_env_file=None)

    assert settings.dispatch_subset_k is None
    assert settings.dispatch_subset_fraction == 0.5
    assert (
        _dispatch_subset_size(
            pool_size,
            required=1,
            fixed_k=settings.dispatch_subset_k,
            fraction=settings.dispatch_subset_fraction,
        )
        == expected
    )


def test_challenge_quorum_can_raise_the_half_pool_sample():
    assert _dispatch_subset_size(256, required=180, fixed_k=None, fraction=0.5) == 180
    assert _dispatch_subset_size(3, required=9, fixed_k=None, fraction=0.5) == 3


def test_explicit_zero_preserves_the_legacy_full_pool_override():
    assert _dispatch_subset_size(256, required=1, fixed_k=0, fraction=0.5) == 256


def test_explicit_zero_environment_setting_remains_distinct_from_unset(
    monkeypatch,
):
    monkeypatch.setenv("DISPATCH_SUBSET_K", "0")

    settings = Settings(_env_file=None)

    assert settings.dispatch_subset_k == 0
    assert (
        _dispatch_subset_size(
            256,
            required=1,
            fixed_k=settings.dispatch_subset_k,
            fraction=settings.dispatch_subset_fraction,
        )
        == 256
    )


def test_fixed_count_override_takes_precedence_but_not_over_quorum():
    assert _dispatch_subset_size(256, required=1, fixed_k=40, fraction=0.5) == 40
    assert _dispatch_subset_size(256, required=60, fixed_k=40, fraction=0.5) == 60


@pytest.mark.parametrize("pool_size", [2, 3, 255, 256])
def test_rotation_balances_problem_counts_across_a_complete_epoch(pool_size):
    solvers = [SimpleNamespace(uid=uid) for uid in range(pool_size)]
    sample_size = _dispatch_subset_size(pool_size, 1, None, 0.5)
    sampler = RotationSampler()
    counts = Counter()

    # This many draws covers an integral number of dealt slots for every pool.
    for _ in range(pool_size):
        counts.update(solver.uid for solver in sampler.sample(solvers, sample_size))

    assert len(counts) == pool_size
    assert max(counts.values()) - min(counts.values()) <= 1

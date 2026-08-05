"""Default challenge-round cadence.

These tests cover the validator round interval. Weight cadence has its own
independent policy and tests.

"""

from __future__ import annotations

from rlvr.config import Settings
from rlvr.neurons.validator import ValidatorNeuron

SN5_ROUND_INTERVAL_BLOCKS = 150


def test_the_default_round_interval_is_the_sn5_value():
    assert Settings(_env_file=None).round_interval_blocks == (
        SN5_ROUND_INTERVAL_BLOCKS
    )


def test_the_validator_adopts_the_configured_round_interval():
    """The default is worthless if the run loop reads something else."""
    v = ValidatorNeuron(Settings(_env_file=None))

    assert v.round_interval_blocks == SN5_ROUND_INTERVAL_BLOCKS


def test_an_operator_can_still_override_the_cadence(monkeypatch):
    """A default is a default, not a policy an operator cannot escape."""
    monkeypatch.setenv("ROUND_INTERVAL_BLOCKS", "200")

    assert Settings(_env_file=None).round_interval_blocks == 200


def test_the_cadence_still_accepts_a_fast_smoke_test_value(monkeypatch):
    """config.py documents lowering this for an on-chain smoke test."""
    monkeypatch.setenv("ROUND_INTERVAL_BLOCKS", "1")

    assert Settings(_env_file=None).round_interval_blocks == 1

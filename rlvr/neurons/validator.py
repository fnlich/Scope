"""ValidatorNeuron — block-interval run loop with injected V1 callbacks.

The validator owns the chain identity and run cadence. The V1 evaluation and
weight-submission functions are injected to keep this chain loop independent of
the problem-server and scoring modules.

Callbacks:
  * ``round_callback`` — ``async (validator) -> dict[int, float]`` returning
    {uid: reward} for the round. Set via the constructor or :meth:`set_round_callback`.
  * ``weight_setter`` — computes and submits current validator-local weights.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from ..config import Settings
from ..types import NeuronType
from .base import BaseNeuron

# Type of the injected orchestrator round callback.
RoundCallback = Callable[["ValidatorNeuron"], Awaitable[dict[int, float]]]
WeightSetter = Callable[["ValidatorNeuron"], None]


class ValidatorNeuron(BaseNeuron):
    """Validator skeleton: chain identity + run loop + weight setting.

    The heavy lifting is delegated to an injected ``round_callback`` so the
    orchestrator and this neuron stay decoupled (no import cycle).
    """

    neuron_type = NeuronType.VALIDATOR

    def __init__(
        self,
        settings: Optional[Settings] = None,
        round_callback: Optional[RoundCallback] = None,
        weight_setter: Optional[WeightSetter] = None,
    ):
        super().__init__(settings)
        self._round_callback: Optional[RoundCallback] = round_callback
        self._weight_setter: Optional[WeightSetter] = weight_setter

        # How often (in blocks) to run a round / set weights when live.
        self.round_interval_blocks: int = self.settings.round_interval_blocks
        self.weights_interval_blocks: int = self.settings.weights_interval_blocks

        self._should_exit = False

    # ------------------------------------------------------------------ #
    # Injection seams
    # ------------------------------------------------------------------ #
    def set_round_callback(self, callback: RoundCallback) -> None:
        """Inject the orchestrator's async round function.

        Signature: ``async (validator: ValidatorNeuron) -> dict[int, float]``.
        """
        self._round_callback = callback

    def set_weight_setter(
        self, setter: WeightSetter
    ) -> None:
        """Inject the validator-local weight computation/submission function."""
        self._weight_setter = setter

    # ------------------------------------------------------------------ #
    # Round driver
    # ------------------------------------------------------------------ #
    async def run_round(self) -> dict[int, float]:
        """Run one orchestrated round via the injected callback.

        Returns {uid: reward}. Raises if no callback has been injected — this is
        the explicit seam the live orchestrator must fill.
        """
        if self._round_callback is None:
            raise RuntimeError(
                "No round_callback injected. The orchestrator must call "
                "validator.set_round_callback(...) before run_round(). "
                "(This is the orchestrator seam; orchestrator is wired separately.)"
            )
        return await self._round_callback(self)

    # ------------------------------------------------------------------ #
    # Weight setting hook
    # ------------------------------------------------------------------ #
    def submit_weights(self) -> None:
        """Compute and submit weights through the configured V1 callback."""
        if self._weight_setter is None:
            raise RuntimeError("No weight_setter configured")
        self._weight_setter(self)

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Signal the run loop to exit after the current iteration."""
        self._should_exit = True

    async def run(self, max_rounds: Optional[int] = None) -> None:
        """Block-interval run loop (live).

        Polls the current block; on each ``round_interval_blocks`` boundary it
        runs an orchestrated round, and on each ``weights_interval_blocks``
        boundary it sets weights. ``max_rounds`` bounds the loop for tests/sim.
        """
        if self.subtensor is None:
            # Not set up for chain — the orchestrator/sim should drive run_round
            # directly. Surface a clear error rather than silently no-op.
            raise RuntimeError(
                "run() needs setup_bittensor(); for offline use call run_round() directly."
            )

        rounds_done = 0
        weights_gate = BaseNeuron(self.settings)  # separate interval bookkeeping
        while not self._should_exit:
            block = self.current_block()
            if self.should_run(block, self.round_interval_blocks):
                await self.run_round()
                rounds_done += 1
                if max_rounds is not None and rounds_done >= max_rounds:
                    break
            if rounds_done and weights_gate.should_run(block, self.weights_interval_blocks):
                # The callback owns the EvalEngine. Gate on rounds_done so we
                # never submit the initial all-zero score vector.
                if self._weight_setter is not None:
                    # Chain inclusion can take a block. Keep that synchronous SDK
                    # wait off the event loop so unrelated async work is not blocked.
                    await asyncio.to_thread(self.submit_weights)
            await asyncio.sleep(1.0)

"""Validator-side live HTTP transport for dispatching signed tasks."""

from __future__ import annotations

import time
from typing import Optional
from uuid import uuid4

import httpx

from ..config import Settings
from ..protocol import SignedSolution, SolutionPayload, TaskRequest, sign_message, verify_signature
from ..types import Problem, SolutionResponse
from .validator import ValidatorNeuron


async def _read_bounded_response(
    response: httpx.Response, limit: int
) -> Optional[bytes]:
    """Read a miner response up to ``limit`` bytes, then fail closed."""
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) < 0 or int(declared) > limit:
                return None
        except ValueError:
            return None
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


# --------------------------------------------------------------------------- #
# Validator side: a SolverClient that queries a registered miner's axon.
# --------------------------------------------------------------------------- #
class LiveSolverClient:
    """Queries one miner's HTTP axon with an epistula-signed TaskRequest."""

    def __init__(
        self,
        uid: int,
        hotkey: str,
        url: str,
        wallet,
        settings: Settings,
        http: httpx.AsyncClient,
    ):
        self.uid = uid
        self.hotkey = hotkey
        self._url = url.rstrip("/")
        self._wallet = wallet
        self._settings = settings
        self._http = http

    async def solve(self, problem: Problem, prompt: str) -> SolutionResponse:
        artifact = await self.solve_signed(problem, request_id=uuid4().hex)
        return artifact.to_solution(problem.problem_id)

    async def solve_signed(self, problem: Problem, request_id: str) -> SignedSolution:
        """Return exact miner-signed bytes for commit-before-test-reveal."""
        req = TaskRequest(
            problem_id=request_id,
            statement=problem.statement,
            entrypoint=problem.entrypoint,
            public_examples=problem.public_examples,
            deadline_s=self._settings.solve_deadline_s,
            prompt_variant=problem.prompt_variant,
        )
        body = req.model_dump_json().encode("utf-8")
        # Sign with the validator's hotkey, bound to this miner as recipient.
        headers = sign_message(self._wallet, body, signed_for=self.hotkey)
        headers["Content-Type"] = "application/json"
        started = time.monotonic()
        try:
            async with self._http.stream(
                "POST",
                f"{self._url}/solve",
                content=body,
                headers=headers,
                # Slack past the miner's deadline so a response finishing right at
                # the deadline isn't dropped on the wire.
                timeout=req.deadline_s + 10.0,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await _read_bounded_response(resp, 512)
                    detail = (
                        error_body.decode("utf-8", errors="replace")[:200]
                        if error_body is not None
                        else "<oversized error response>"
                    )
                    return SignedSolution(
                        error=f"HTTP {resp.status_code}: {detail}",
                        latency_ms=(time.monotonic() - started) * 1000.0,
                    )
                response_body = await _read_bounded_response(
                    resp, max(1, self._settings.miner_max_response_bytes)
                )
                if response_body is None:
                    return SignedSolution(
                        error="<miner response exceeds byte limit>",
                        latency_ms=(time.monotonic() - started) * 1000.0,
                    )
                response_headers = {
                    name: resp.headers.get(name, "")
                    for name in (
                        "Epistula-Version",
                        "Epistula-Timestamp",
                        "Epistula-Uuid",
                        "Epistula-Signed-By",
                        "Epistula-Signed-For",
                        "Epistula-Request-Signature",
                    )
                }
            if (
                response_headers["Epistula-Signed-By"] != self.hotkey
                or not verify_signature(
                    response_headers,
                    response_body,
                    expected_signed_for=headers["Epistula-Signed-By"],
                )
            ):
                return SignedSolution(
                    error="<invalid or unauthenticated miner response>",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                )
            payload = SolutionPayload.model_validate_json(response_body)
            if payload.problem_id != req.problem_id:
                return SignedSolution(
                    error="<miner response request-id mismatch>",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                )
            return SignedSolution(
                response_body=response_body.decode("utf-8"),
                response_headers=response_headers,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        except Exception as e:  # noqa: BLE001 - an unreachable miner scores 0
            return SignedSolution(
                error=f"<dispatch error: {e}>",
                latency_ms=(time.monotonic() - started) * 1000.0,
            )


def _solver_clients(
    validator: ValidatorNeuron, wallet, settings: Settings, http: httpx.AsyncClient
) -> list[LiveSolverClient]:
    """Build a LiveSolverClient for every serving miner in the metagraph."""
    mg = validator.metagraph
    clients: list[LiveSolverClient] = []
    for uid in range(len(mg.hotkeys)):
        axon = mg.axons[uid]
        ip, port = getattr(axon, "ip", "0.0.0.0"), getattr(axon, "port", 0)
        if not port or ip in ("0.0.0.0", "", None):
            continue  # miner not serving an axon
        clients.append(
            LiveSolverClient(uid, mg.hotkeys[uid], f"http://{ip}:{port}", wallet, settings, http)
        )
    return clients

"""Validator-side HTTPS client for the private problem-source protocol."""

from __future__ import annotations

import asyncio
from typing import Optional
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ..protocol import sign_message
from .api import (
    ChallengeCommitRequest,
    ChallengeCommitResponse,
    ChallengeFeedback,
    ChallengeLeaseRequest,
    ChallengeRevealRequest,
    ChallengeRevealResponse,
    MinerSubmission,
    PublicChallenge,
)


def require_secure_problem_url(url: str, allow_insecure_http: bool = False) -> None:
    parsed = urlsplit(url)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if allow_insecure_http and parsed.scheme == "http" and parsed.netloc:
        return
    raise ValueError(
        "PROBLEM_SERVER_URL must use https://; set "
        "PROBLEM_SERVER_ALLOW_INSECURE_HTTP=true only for local testing"
    )


class ProblemServerClient:
    def __init__(
        self,
        url: str,
        wallet,
        http: httpx.AsyncClient,
        *,
        allow_insecure_http: bool = False,
        timeout_s: float = 60.0,
        retries: int = 3,
        max_response_bytes: int = 512_000,
    ):
        require_secure_problem_url(url, allow_insecure_http)
        self._url = url.rstrip("/")
        self._wallet = wallet
        self._http = http
        self._timeout = timeout_s
        self._retries = max(1, int(retries))
        self._max_response_bytes = max(1, int(max_response_bytes))

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length")
        if declared:
            try:
                if int(declared) < 0 or int(declared) > self._max_response_bytes:
                    raise RuntimeError("problem-server response exceeds byte limit")
            except ValueError as error:
                raise RuntimeError("invalid problem-server content-length") from error
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self._max_response_bytes:
                raise RuntimeError("problem-server response exceeds byte limit")
        return bytes(body)

    async def _post(self, path: str, body: bytes) -> Optional[httpx.Response]:
        for attempt in range(1, self._retries + 1):
            headers = sign_message(self._wallet, body)
            headers["Content-Type"] = "application/json"
            try:
                async with self._http.stream(
                    "POST",
                    f"{self._url}{path}",
                    content=body,
                    headers=headers,
                    timeout=self._timeout,
                ) as streamed:
                    content = await self._read_bounded(streamed)
                    status_code = streamed.status_code
                    response_headers = streamed.headers
                    request = streamed.request
                # A paced lease returns 429 with Retry-After. It is not a
                # transient transport failure: return it to lease() immediately
                # instead of hammering the server three times within a second.
                if status_code >= 500 or status_code in {408, 425}:
                    raise RuntimeError(f"HTTP {status_code}")
                # Return a normal in-memory response only after the streaming
                # cap has been enforced; callers keep their existing parsing API.
                # `content` is already transport-decoded by aiter_bytes, so the
                # framing headers must NOT ride along: keeping content-encoding
                # would make httpx re-apply the decoder to plaintext and raise
                # DecodingError behind any gzip-enabled server or proxy.
                rewrapped_headers = {
                    name: value
                    for name, value in response_headers.items()
                    if name.lower()
                    not in ("content-encoding", "content-length", "transfer-encoding")
                }
                return httpx.Response(
                    status_code=status_code,
                    headers=rewrapped_headers,
                    content=content,
                    request=request,
                )
            except Exception as e:  # noqa: BLE001 - retry transient network faults
                if attempt == self._retries:
                    print(f"[validator] WARN: problem server {path} failed: {e}")
                    return None
                await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))
        return None

    async def lease(self) -> Optional[PublicChallenge]:
        # Stable body across HTTP retries makes generation idempotent.
        body = ChallengeLeaseRequest(request_id=uuid4().hex).model_dump_json().encode()
        response = await self._post("/v1/challenges/lease", body)
        if response is None or response.status_code != 200:
            return None
        try:
            return PublicChallenge.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[validator] WARN: invalid public challenge: {e}")
            return None

    async def commit(
        self, challenge_id: str, submissions: list[MinerSubmission]
    ) -> Optional[ChallengeCommitResponse]:
        body = ChallengeCommitRequest(
            challenge_id=challenge_id, submissions=submissions
        ).model_dump_json().encode()
        response = await self._post("/v1/challenges/commit", body)
        if response is None:
            return None
        try:
            return ChallengeCommitResponse.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[validator] WARN: invalid commit receipt: {e}")
            return None

    async def reveal(
        self, challenge_id: str, commit_token: str
    ) -> Optional[ChallengeRevealResponse]:
        body = ChallengeRevealRequest(
            challenge_id=challenge_id, commit_token=commit_token
        ).model_dump_json().encode()
        response = await self._post("/v1/challenges/reveal", body)
        if response is None or response.status_code != 200:
            return None
        try:
            return ChallengeRevealResponse.model_validate_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[validator] WARN: invalid test reveal: {e}")
            return None

    async def feedback(self, feedback: ChallengeFeedback) -> bool:
        body = feedback.model_dump_json().encode()
        response = await self._post("/v1/challenges/feedback", body)
        if response is None or response.status_code != 200:
            return False
        try:
            return bool(response.json().get("accepted"))
        except Exception:  # noqa: BLE001
            return False

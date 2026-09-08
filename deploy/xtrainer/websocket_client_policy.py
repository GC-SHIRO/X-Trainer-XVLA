"""Client for the X-trainer WebSocket policy transport."""

from __future__ import annotations

from typing import Any

from .image_codec import JPEG_RGB_ENCODING, encode_policy_images
from .msgpack_numpy import PROTOCOL_VERSION, ProtocolError, dumps, loads


class XTrainerWebSocketPolicyClient:
    """Small one-request/one-response client for X-trainer policy servers."""

    def __init__(
        self,
        base_url: str,
        *,
        max_payload_bytes: int | None = None,
        image_jpeg_quality: int = 85,
    ) -> None:
        if not 0 <= image_jpeg_quality <= 100:
            raise ValueError("image_jpeg_quality must be in [0, 100]")
        self.base_url = base_url.rstrip("/")
        self.max_payload_bytes = max_payload_bytes
        self.image_jpeg_quality = image_jpeg_quality
        self._session = None
        self._ws = None
        self.metadata: dict[str, Any] | None = None

    async def __aenter__(self) -> "XTrainerWebSocketPolicyClient":
        await self.connect()
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()

    async def connect(self) -> dict[str, Any]:
        from aiohttp import ClientSession, WSMsgType

        self._session = ClientSession()
        self._ws = await self._session.ws_connect(
            f"{self.base_url}/ws", max_msg_size=self.max_payload_bytes or 0
        )
        handshake = await self._ws.receive()
        if handshake.type != WSMsgType.BINARY:
            raise ProtocolError("metadata handshake must be a binary frame")
        response = loads(handshake.data, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)
        self._raise_for_error(response)
        self.metadata = response["metadata"]
        return self.metadata

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()
        self._ws = None
        self._session = None

    async def get_healthz(self) -> dict[str, Any]:
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.get(f"{self.base_url}/healthz") as response:
                response.raise_for_status()
                return await response.json()

    async def get_metadata(self) -> dict[str, Any]:
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.get(f"{self.base_url}/metadata") as response:
                response.raise_for_status()
                return await response.json()

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        from aiohttp import WSMsgType

        if self._ws is None:
            raise RuntimeError("client is not connected")
        request_payload = {"protocol_version": PROTOCOL_VERSION, **payload}
        await self._ws.send_bytes(
            dumps(request_payload, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)
        )
        response = await self._ws.receive()
        if response.type != WSMsgType.BINARY:
            raise ProtocolError("server response must be a binary frame")
        result = loads(response.data, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)
        self._raise_for_error(result)
        return result

    async def reset(self) -> dict[str, Any]:
        return await self.request({"type": "reset"})

    async def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_encodings = self.metadata.get("image_encodings", []) if self.metadata is not None else []
        if self.image_jpeg_quality > 0 and JPEG_RGB_ENCODING in image_encodings:
            payload = encode_policy_images(payload, self.image_jpeg_quality)
        response = await self.request({"type": "infer", "payload": payload})
        result = response.get("payload")
        if not isinstance(result, dict):
            raise ProtocolError("infer response payload must be a map")
        return result

    @staticmethod
    def _raise_for_error(response: dict[str, Any]) -> None:
        if response.get("ok") is False:
            error = response.get("error") or {}
            raise ProtocolError(f"{error.get('code', 'error')}: {error.get('message', 'unknown error')}")

"""JPEG wire encoding for X-trainer RGB camera observations."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

JPEG_RGB_ENCODING = "jpeg_rgb"
JPEG_IMAGE_MARKER = "__xtrainer_jpeg_rgb__"


def encode_policy_images(payload: dict[str, Any], quality: int) -> dict[str, Any]:
    """Return a payload copy whose RGB image arrays are JPEG byte strings."""
    images = payload.get("images")
    if not isinstance(images, dict):
        return payload
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")

    encoded_payload = payload.copy()
    encoded_payload["images"] = {name: _encode_rgb_image(image, quality) for name, image in images.items()}
    return encoded_payload


def decode_policy_images(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode JPEG-marked camera values while leaving raw ndarray payloads unchanged."""
    images = payload.get("images")
    if not isinstance(images, dict):
        return payload

    decoded_images = {}
    changed = False
    for name, image in images.items():
        if isinstance(image, dict) and image.get(JPEG_IMAGE_MARKER) is True:
            decoded_images[name] = _decode_rgb_image(image)
            changed = True
        else:
            decoded_images[name] = image
    if not changed:
        return payload

    decoded_payload = payload.copy()
    decoded_payload["images"] = decoded_images
    return decoded_payload


def _encode_rgb_image(image: Any, quality: int) -> dict[str, Any]:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"JPEG input must be uint8 HxWx3 RGB, got dtype={array.dtype}, shape={array.shape}")
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode RGB image as JPEG")
    return {
        JPEG_IMAGE_MARKER: True,
        "data": encoded.tobytes(),
    }


def _decode_rgb_image(payload: dict[str, Any]) -> np.ndarray:
    data = payload.get("data")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("JPEG image payload must contain non-empty bytes")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode JPEG image payload")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

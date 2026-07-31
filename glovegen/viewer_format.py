"""A tiny binary mesh format for shipping geometry to the browser.

De-indexed on purpose: every triangle carries its own three vertices, so the
viewer can colour per face (which is what an undercut heatmap needs) without any
index juggling in JavaScript.

    magic    4 bytes   b"GGM2"
    n_faces  uint32    little endian
    xyz      float32[n_faces * 9]

The matching heatmap payload is a bare ``uint8[n_faces]`` of severity values,
in the same face order.
"""

from __future__ import annotations

import struct

import numpy as np
import trimesh

MAGIC = b"GGM2"


def encode(mesh: trimesh.Trimesh) -> bytes:
    """Serialise ``mesh`` as de-indexed triangles."""
    tri = np.asarray(mesh.triangles, dtype=np.float32)
    header = MAGIC + struct.pack("<I", len(tri))
    return header + tri.tobytes(order="C")


def decode(payload: bytes) -> np.ndarray:
    """Inverse of :func:`encode`; returns an ``(n_faces, 3, 3)`` array."""
    if payload[:4] != MAGIC:
        raise ValueError("not a GGM2 payload")
    (n_faces,) = struct.unpack("<I", payload[4:8])
    body = np.frombuffer(payload, dtype=np.float32, count=n_faces * 9, offset=8)
    return body.reshape(n_faces, 3, 3)


def encode_severity(severity: np.ndarray) -> bytes:
    """Quantise per-face undercut severity to one byte each."""
    sev = np.clip(np.asarray(severity, dtype=np.float64), 0.0, 1.0)
    return (sev * 255.0 + 0.5).astype(np.uint8).tobytes(order="C")

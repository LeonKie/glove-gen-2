"""Local frames aligned to a pull direction.

Every stage of the pipeline reasons about "columns parallel to the pull
direction", which is much easier when the direction is +Z. Rather than rotating
a multi-million-triangle mesh once per candidate direction, we rotate the
*queries* into and out of a local frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("direction vector is zero-length")
    return v / n


@dataclass(frozen=True)
class Frame:
    """Rotation taking a world direction to local +Z.

    ``rot`` is a 3x3 orthonormal matrix whose rows are the local axes expressed
    in world coordinates, so ``rot[2] == direction``.
    """

    direction: np.ndarray
    rot: np.ndarray

    @classmethod
    def from_direction(cls, direction, seed=None) -> "Frame":
        """Build a frame with ``direction`` as local +Z.

        The rotation *about* the direction is arbitrary and normally left to
        ``align_vectors``. ``seed`` pins it instead: local +X becomes the part of
        ``seed`` perpendicular to the direction. A block built in such a frame has
        two of its faces square to ``seed``, which is what the carrier plate needs
        -- it is trimmed perpendicular to the pour axis, and slicing an
        arbitrarily-oriented box on an oblique plane gives a corner wedge rather
        than a plate.
        """
        d = unit(direction)
        rot = trimesh.geometry.align_vectors(d, [0.0, 0.0, 1.0])[:3, :3]
        # align_vectors is exact up to float error; re-seat the last row so
        # rot[2] is the requested direction bit-for-bit.
        rot = np.array(rot, dtype=np.float64)
        rot[2] = d
        if seed is not None:
            s = np.asarray(seed, dtype=np.float64).reshape(3)
            s = s - d * (s @ d)
            # A seed parallel to the direction says nothing about the roll, so
            # fall back to whatever align_vectors picked.
            if np.linalg.norm(s) > 1e-6:
                rot[0] = s
        # Re-orthonormalise rows 0/1 against row 2 (Gram-Schmidt).
        rot[0] -= rot[2] * (rot[0] @ rot[2])
        rot[0] /= np.linalg.norm(rot[0])
        rot[1] = np.cross(rot[2], rot[0])
        return cls(direction=d, rot=rot)

    def to_local(self, pts) -> np.ndarray:
        """World points/vectors -> local frame."""
        return np.asarray(pts, dtype=np.float64) @ self.rot.T

    def to_world(self, pts) -> np.ndarray:
        """Local points/vectors -> world frame."""
        return np.asarray(pts, dtype=np.float64) @ self.rot

    def transform(self) -> np.ndarray:
        """4x4 world -> local transform, for trimesh's apply_transform."""
        t = np.eye(4)
        t[:3, :3] = self.rot
        return t


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """``n`` roughly-uniform unit vectors covering a hemisphere.

    A pull direction and its negation describe the same 2-part split (one half
    pulls each way), so only half the sphere holds distinct candidates.
    """
    n = max(1, int(n))
    i = np.arange(n, dtype=np.float64)
    z = (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    phi = i * golden
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def principal_axis(mesh: trimesh.Trimesh) -> np.ndarray:
    """Longest principal axis of the mesh's vertices."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) > 200_000:
        v = v[:: len(v) // 200_000 + 1]
    centred = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return unit(vt[0])

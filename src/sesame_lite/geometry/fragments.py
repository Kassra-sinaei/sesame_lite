"""The oriented box of a scene-graph node (Hydra's node box, the oracle's box), from signal_3d
``semantic_safe_vla.safety.numerical``. In spec v4 a box carries no safety claim: it is the oracle's evaluation
geometry and Hydra's node box (node matching, pictures)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

ELLIPSOID_ENCLOSURE_TOLERANCE = 1e-6


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    size: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain exactly {size} finite values.")
    return array


def _normalized_quaternion(
    quaternion_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    quaternion = _finite_vector(
        quaternion_xyzw, 4, "Obstacle orientation"
    )
    scale = float(np.max(np.abs(quaternion)))
    if scale <= 1e-12:
        raise ValueError("Obstacle orientation quaternion cannot be zero.")
    scaled = quaternion / scale
    normalized = scaled / float(np.linalg.norm(scaled))
    # q and -q encode the same rotation. Match the canonical Phase 4 graph
    # convention so an equivalent simulator sign flip cannot perturb snapshot
    # identities, ordering evidence, or receipt digests.
    if normalized[3] < 0.0 or (
        abs(float(normalized[3])) < 1e-12
        and next(
            (
                float(value)
                for value in normalized[:3]
                if abs(float(value)) >= 1e-12
            ),
            1.0,
        )
        < 0.0
    ):
        normalized = -normalized
    return normalized


def quaternion_rotation_matrix(
    quaternion_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the local-to-world rotation for one finite xyzw quaternion."""

    normalized = _normalized_quaternion(quaternion_xyzw)
    x, y, z, w = normalized
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )







@dataclass(frozen=True)
class OrientedBoxFragment:
    """One stable, raw, oriented narrow-phase obstacle fragment."""

    parent_stable_id: str
    fragment_id: str
    center: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    half_extents: tuple[float, float, float]
    source_kind: str = "whole_object"
    _rotation: np.ndarray = field(init=False, repr=False, compare=False)
    _corners: np.ndarray = field(init=False, repr=False, compare=False)
    _aabb_bounds: Mapping[int, tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.parent_stable_id, "OBB parent stable ID"),
            (self.fragment_id, "OBB fragment ID"),
            (self.source_kind, "OBB source kind"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text.")
        if self.source_kind not in {
            "whole_object",
            "articulated_link",
            "geometry_component",
            "collision_component",
        }:
            raise ValueError("OBB source kind is unsupported.")
        center = _finite_vector(self.center, 3, "OBB center")
        orientation = _normalized_quaternion(self.orientation_xyzw)
        half_extents = _finite_vector(
            self.half_extents, 3, "OBB half-extents"
        )
        if np.any(half_extents <= 0.0):
            raise ValueError("OBB half-extents must be strictly positive.")
        rotation = quaternion_rotation_matrix(orientation)
        signs = np.asarray(
            tuple(
                (sx, sy, sz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ),
            dtype=np.float64,
        )
        corners = center[None, :] + (
            signs * half_extents[None, :]
        ) @ rotation.T
        bounds = {}
        for dimensions in (2, 3):
            minimum = np.min(corners[:, :dimensions], axis=0)
            maximum = np.max(corners[:, :dimensions], axis=0)
            minimum.setflags(write=False)
            maximum.setflags(write=False)
            bounds[dimensions] = (minimum, maximum)
        rotation.setflags(write=False)
        corners.setflags(write=False)
        object.__setattr__(self, "center", tuple(float(x) for x in center))
        object.__setattr__(
            self,
            "orientation_xyzw",
            tuple(float(x) for x in orientation),
        )
        object.__setattr__(
            self,
            "half_extents",
            tuple(float(x) for x in half_extents),
        )
        object.__setattr__(self, "_rotation", rotation)
        object.__setattr__(self, "_corners", corners)
        object.__setattr__(self, "_aabb_bounds", bounds)

    @property
    def stable_id(self) -> str:
        return self.fragment_id

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation

    @property
    def corners(self) -> np.ndarray:
        return self._corners

    def aabb(self, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
        if dimensions not in (2, 3):
            raise ValueError("OBB AABB must be 2-D or 3-D.")
        return self._aabb_bounds[dimensions]

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_stable_id": self.parent_stable_id,
            "fragment_id": self.fragment_id,
            "center": list(self.center),
            "orientation_xyzw": list(self.orientation_xyzw),
            "half_extents": list(self.half_extents),
            "source_kind": self.source_kind,
        }

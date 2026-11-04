"""Obstacle fragment and barrier-sample types.

Extracted verbatim from signal_3d ``semantic_safe_vla.safety.numerical`` (the
per-row hard CBF lineage, DEC-075) so that the barrier geometry in
``box_geometry.py`` runs without OmniGibson, pink or pinocchio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

ELLIPSOID_ENCLOSURE_TOLERANCE = 1e-6


class BarrierFamily(str, Enum):
    NAVIGATION = "navigation_base_object_collision"
    MOBILE_MANIPULATION = "mobile_manipulation_eef_object_collision"
    SEMANTIC_SPATIAL = "semantic_gripper_proxy_spatial_exclusion"
    SPATIAL_COMPOSITION = "spatial_cross_family_composition"
    COMMAND_RATE = "full_body_command_acceleration_jerk"


def collision_barrier_dimensions(family: BarrierFamily) -> int:
    """Return point dimensions only for the two obstacle families."""

    family = BarrierFamily(family)
    if family is BarrierFamily.NAVIGATION:
        return 2
    if family is BarrierFamily.MOBILE_MANIPULATION:
        return 3
    if family is BarrierFamily.SEMANTIC_SPATIAL:
        return 3
    raise ValueError("Command-rate barriers do not have obstacle geometry.")



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


@dataclass(frozen=True)
class CylinderFragment:
    """One upright round narrow-phase fragment (DEC-075 shape routing).

    Represents a physically round object (can, trash can, round table,
    column) with an exact upright cylinder instead of a box, eliminating
    the sqrt(2)-1 phantom corner slop of an OBB on round geometry. The
    fragment carries the corners of its *enclosing box* so the existing
    broad-phase containment invariant and AABB machinery apply unchanged;
    only the narrow-phase distance uses the round surface. The axis is the
    fragment frame's +Z; capture-time routing must verify near-uprightness
    before selecting this shape.
    """

    parent_stable_id: str
    fragment_id: str
    center: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    radius: float
    half_height: float
    source_kind: str = "whole_object"
    _rotation: np.ndarray = field(init=False, repr=False, compare=False)
    _corners: np.ndarray = field(init=False, repr=False, compare=False)
    _aabb_bounds: Mapping[int, tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.parent_stable_id, "Cylinder parent stable ID"),
            (self.fragment_id, "Cylinder fragment ID"),
            (self.source_kind, "Cylinder source kind"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text.")
        if self.source_kind not in {
            "whole_object",
            "articulated_link",
            "geometry_component",
            "collision_component",
        }:
            raise ValueError("Cylinder source kind is unsupported.")
        center = _finite_vector(self.center, 3, "Cylinder center")
        orientation = _normalized_quaternion(self.orientation_xyzw)
        for value, label in (
            (self.radius, "Cylinder radius"),
            (self.half_height, "Cylinder half-height"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive.")
        rotation = quaternion_rotation_matrix(orientation)
        half_extents = np.asarray(
            (float(self.radius), float(self.radius), float(self.half_height)),
            dtype=np.float64,
        )
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
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "half_height", float(self.half_height))
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
        """Enclosing-box corners: containment and AABB stay conservative."""

        return self._corners

    def aabb(self, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
        if dimensions not in (2, 3):
            raise ValueError("Cylinder AABB must be 2-D or 3-D.")
        return self._aabb_bounds[dimensions]

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_stable_id": self.parent_stable_id,
            "fragment_id": self.fragment_id,
            "center": list(self.center),
            "orientation_xyzw": list(self.orientation_xyzw),
            "radius": self.radius,
            "half_height": self.half_height,
            "source_kind": self.source_kind,
            "shape_kind": "cylinder",
        }


NarrowPhaseFragment = OrientedBoxFragment | CylinderFragment
NARROW_PHASE_FRAGMENT_TYPES = (OrientedBoxFragment, CylinderFragment)




@dataclass(frozen=True)
class RawBarrierSample:
    family: BarrierFamily
    protected_id: str
    obstacle_id: str
    value: float
    point_jacobian: tuple[float, ...]
    surface_distance_m: float
    protected_radius_m: float
    policy_margin_m: float
    distance_backend: str
    obstacle_parent_id: str | None = None
    geometry_kind: str = "ellipsoid"
    distance_is_signed: bool = False
    nonsmooth: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", BarrierFamily(self.family))
        for value, label in (
            (self.protected_id, "Protected geometry ID"),
            (self.obstacle_id, "Barrier obstacle ID"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string.")
        parent_id = self.obstacle_parent_id
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id.strip()
        ):
            raise ValueError("Barrier parent obstacle ID must be non-empty.")
        if not math.isfinite(float(self.value)):
            raise ValueError("Barrier values must be finite.")
        if (
            isinstance(self.surface_distance_m, bool)
            or not isinstance(self.surface_distance_m, (int, float))
            or not math.isfinite(float(self.surface_distance_m))
            or (
                not self.distance_is_signed
                and float(self.surface_distance_m) < 0.0
            )
        ):
            raise ValueError(
                "Surface distance must be finite and non-negative unless "
                "explicitly declared signed."
            )
        for field_value, label in (
            (self.protected_radius_m, "Protected radius"),
            (self.policy_margin_m, "Policy margin"),
        ):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, (int, float))
                or not math.isfinite(float(field_value))
                or field_value < 0.0
            ):
                raise ValueError(f"{label} must be finite and non-negative.")
        if (
            not isinstance(self.distance_backend, str)
            or not self.distance_backend.strip()
            or not isinstance(self.geometry_kind, str)
            or not self.geometry_kind.strip()
            or not isinstance(self.distance_is_signed, bool)
            or not isinstance(self.nonsmooth, bool)
        ):
            raise ValueError("Barrier geometry metadata is malformed.")
        jacobian = tuple(float(value) for value in self.point_jacobian)
        expected_dimensions = collision_barrier_dimensions(self.family)
        accepted_lengths = {2, 3} if expected_dimensions == 2 else {3}
        if len(jacobian) not in accepted_lengths or not all(
            math.isfinite(value) for value in jacobian
        ):
            raise ValueError(
                f"{self.family.value} point/tangent Jacobian has invalid dimensions."
            )
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "point_jacobian", jacobian)
        object.__setattr__(
            self, "surface_distance_m", float(self.surface_distance_m)
        )
        object.__setattr__(
            self, "protected_radius_m", float(self.protected_radius_m)
        )
        object.__setattr__(
            self, "policy_margin_m", float(self.policy_margin_m)
        )

    @classmethod
    def from_validated_geometry(
        cls,
        *,
        family: BarrierFamily,
        protected_id: str,
        obstacle_id: str,
        value: float,
        point_jacobian: tuple[float, ...],
        surface_distance_m: float,
        protected_radius_m: float,
        policy_margin_m: float,
        distance_backend: str,
        obstacle_parent_id: str | None = None,
        geometry_kind: str = "ellipsoid",
        distance_is_signed: bool = False,
        nonsmooth: bool = False,
    ) -> RawBarrierSample:
        """Construct from an already validated real-time geometry path."""

        result = object.__new__(cls)
        object.__setattr__(result, "family", family)
        object.__setattr__(result, "protected_id", protected_id)
        object.__setattr__(result, "obstacle_id", obstacle_id)
        object.__setattr__(result, "value", value)
        object.__setattr__(result, "point_jacobian", point_jacobian)
        object.__setattr__(result, "surface_distance_m", surface_distance_m)
        object.__setattr__(result, "protected_radius_m", protected_radius_m)
        object.__setattr__(result, "policy_margin_m", policy_margin_m)
        object.__setattr__(result, "distance_backend", distance_backend)
        object.__setattr__(result, "obstacle_parent_id", obstacle_parent_id)
        object.__setattr__(result, "geometry_kind", geometry_kind)
        object.__setattr__(result, "distance_is_signed", distance_is_signed)
        object.__setattr__(result, "nonsmooth", nonsmooth)
        return result

    @property
    def constraint_id(self) -> str:
        return (
            f"{self.family.value}:{self.protected_id}:{self.obstacle_id}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "family": self.family.value,
            "protected_id": self.protected_id,
            "obstacle_id": self.obstacle_id,
            "obstacle_parent_id": self.obstacle_parent_id,
            "geometry_kind": self.geometry_kind,
            "value": self.value,
            "value_units": "m",
            "point_jacobian": list(self.point_jacobian),
            "surface_distance_m": self.surface_distance_m,
            "distance_is_signed": self.distance_is_signed,
            "nonsmooth": self.nonsmooth,
            "protected_radius_m": self.protected_radius_m,
            "policy_margin_m": self.policy_margin_m,
            "required_center_to_surface_distance_m": (
                self.protected_radius_m + self.policy_margin_m
            ),
            "distance_backend": self.distance_backend,
        }

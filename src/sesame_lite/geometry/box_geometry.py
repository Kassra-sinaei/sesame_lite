"""Pure hybrid OBB mathematics for the Phase 6 numerical filter."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from sesame_lite.geometry.fragments import (
    BarrierFamily,
    CylinderFragment,
    OrientedBoxFragment,
    RawBarrierSample,
)


SIGNED_BOX_DISTANCE_BACKEND_VERSION = (
    "phase6-analytic-signed-point-obb/1"
)
SYMMETRIC_POLYGON_BACKEND_VERSION = (
    "phase6-symmetric-sat-nested-lse/1"
)
SIGNED_CYLINDER_DISTANCE_BACKEND_VERSION = (
    "phase6-analytic-signed-point-cylinder/1"
)
RECTANGLE_DISC_BACKEND_VERSION = (
    "phase6-analytic-rectangle-disc/1"
)
# Capture-time routing may select a cylinder only when the fragment axis is
# this close to world-vertical; otherwise the OBB representation is kept.
CYLINDER_UPRIGHT_MINIMUM_AXIS_COSINE = 0.99


def _finite_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and be finite.")
    return result


def signed_point_box_barrier(
    point: Sequence[float] | np.ndarray,
    obstacle: OrientedBoxFragment,
    *,
    protected_radius: float,
    policy_margin: float,
    protected_id: str,
) -> RawBarrierSample:
    """Return signed Euclidean sphere-to-OBB separation and point gradient."""

    if not isinstance(obstacle, OrientedBoxFragment):
        raise TypeError("Signed box distance requires an OBB fragment.")
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (3,) or not np.isfinite(point_array).all():
        raise ValueError("Protected point must contain three finite values.")
    for value, label in (
        (protected_radius, "Protected radius"),
        (policy_margin, "Policy margin"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and non-negative.")
    if not isinstance(protected_id, str) or not protected_id:
        raise ValueError("Protected geometry ID must be non-empty.")

    local = obstacle.rotation.T @ (
        point_array - np.asarray(obstacle.center, dtype=np.float64)
    )
    q = np.abs(local) - np.asarray(
        obstacle.half_extents, dtype=np.float64
    )
    outside = np.maximum(q, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    nonsmooth = False
    if outside_norm > 1e-14:
        distance = outside_norm
        local_gradient = (
            np.sign(local) * outside / outside_norm
        )
        positive = np.flatnonzero(outside > 1e-12)
        nonsmooth = bool(
            positive.size > 1
            and np.any(np.abs(local[positive]) <= 1e-12)
        )
    else:
        maximum = float(np.max(q))
        candidates = np.flatnonzero(
            np.isclose(q, maximum, rtol=0.0, atol=1e-12)
        )
        axis = int(candidates[0])
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        local_gradient = np.zeros(3, dtype=np.float64)
        local_gradient[axis] = sign
        distance = maximum
        nonsmooth = candidates.size > 1 or abs(float(local[axis])) <= 1e-12
    world_gradient = obstacle.rotation @ local_gradient
    if (
        not math.isfinite(distance)
        or not np.isfinite(world_gradient).all()
        or not math.isclose(
            float(np.linalg.norm(world_gradient)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("Signed OBB distance produced invalid geometry.")
    return RawBarrierSample(
        family=BarrierFamily.MOBILE_MANIPULATION,
        protected_id=protected_id,
        obstacle_id=obstacle.fragment_id,
        obstacle_parent_id=obstacle.parent_stable_id,
        value=(
            distance - float(protected_radius) - float(policy_margin)
        ),
        point_jacobian=tuple(float(value) for value in world_gradient),
        surface_distance_m=distance,
        protected_radius_m=float(protected_radius),
        policy_margin_m=float(policy_margin),
        distance_backend=SIGNED_BOX_DISTANCE_BACKEND_VERSION,
        geometry_kind="oriented_box_fragment",
        distance_is_signed=True,
        nonsmooth=nonsmooth,
    )


def convex_hull_xy(
    points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a deterministic counter-clockwise 2-D convex hull."""

    array = np.asarray(points, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or array.shape[0] < 3
        or not np.isfinite(array).all()
    ):
        raise ValueError("Convex hull input must contain finite 2-D points.")
    ordered = sorted(
        {(float(point[0]), float(point[1])) for point in array}
    )
    if len(ordered) < 3:
        raise ValueError("Projected OBB must have nonzero planar area.")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(
            lower[-2], lower[-1], point
        ) <= 1e-14:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(
            upper[-2], upper[-1], point
        ) <= 1e-14:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3 or hull.shape[0] > 6:
        raise ValueError("Projected OBB hull must contain three to six vertices.")
    return hull


def projected_box_polygon(obstacle: OrientedBoxFragment) -> np.ndarray:
    if not isinstance(obstacle, OrientedBoxFragment):
        raise TypeError("Projected box polygon requires an OBB fragment.")
    return convex_hull_xy(obstacle.corners[:, :2])


def _outward_faces(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = vertices.shape[0]
    area_twice = float(
        np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
    )
    if area_twice <= 1e-14:
        raise ValueError("Polygon vertices must be counter-clockwise.")
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-14) or not np.isfinite(lengths).all():
        raise ValueError("Polygon faces must have finite nonzero length.")
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    normals = normals / lengths[:, None]
    return normals, edges


def _soft_min_with_gradient(
    values: np.ndarray,
    gradients: np.ndarray,
    kappa: float,
) -> tuple[float, np.ndarray]:
    minimum = float(np.min(values))
    shifted = np.exp(-kappa * (values - minimum))
    denominator = float(np.sum(shifted))
    weights = shifted / denominator
    return (
        minimum - math.log(denominator) / kappa,
        weights @ gradients,
    )


def symmetric_polygon_separation_barrier(
    base_vertices_xy: Sequence[Sequence[float]] | np.ndarray,
    base_vertex_jacobians: Sequence[np.ndarray],
    obstacle: OrientedBoxFragment,
    *,
    policy_margin: float,
    smoothing_gain: float,
    protected_id: str = "base",
) -> tuple[RawBarrierSample, np.ndarray]:
    """Return a conservative smooth symmetric separating-axis certificate."""

    base = _finite_matrix(
        base_vertices_xy, (4, 2), "Base footprint vertices"
    )
    jacobians = tuple(
        np.asarray(value, dtype=np.float64)
        for value in base_vertex_jacobians
    )
    if len(jacobians) != 4:
        raise ValueError("Base footprint requires four vertex Jacobians.")
    nv = jacobians[0].shape[1] if jacobians[0].ndim == 2 else -1
    if nv < 3 or any(
        value.shape != (2, nv) or not np.isfinite(value).all()
        for value in jacobians
    ):
        raise ValueError("Base vertex Jacobians must align with the tangent.")
    if (
        not isinstance(obstacle, OrientedBoxFragment)
        or isinstance(policy_margin, bool)
        or not isinstance(policy_margin, (int, float))
        or not math.isfinite(float(policy_margin))
        or float(policy_margin) < 0.0
        or isinstance(smoothing_gain, bool)
        or not isinstance(smoothing_gain, (int, float))
        or not math.isfinite(float(smoothing_gain))
        or float(smoothing_gain) <= 0.0
    ):
        raise ValueError("Polygon barrier policy is malformed.")
    kappa = float(smoothing_gain)
    obstacle_polygon = projected_box_polygon(obstacle)
    obstacle_normals, _ = _outward_faces(obstacle_polygon)
    base_normals, base_edges = _outward_faces(base)
    base_j = np.asarray(jacobians, dtype=np.float64)

    face_values: list[float] = []
    face_gradients: list[np.ndarray] = []

    for anchor, normal in zip(obstacle_polygon, obstacle_normals):
        values = (base - anchor[None, :]) @ normal
        gradients = np.asarray(
            [normal @ vertex_j for vertex_j in base_j],
            dtype=np.float64,
        )
        value, gradient = _soft_min_with_gradient(
            values, gradients, kappa
        )
        face_values.append(value)
        face_gradients.append(gradient)

    perpendicular = np.asarray(((0.0, 1.0), (-1.0, 0.0)))
    identity = np.eye(2, dtype=np.float64)
    for index, (anchor, normal, edge) in enumerate(
        zip(base, base_normals, base_edges)
    ):
        next_index = (index + 1) % 4
        edge_jacobian = base_j[next_index] - base_j[index]
        normal_jacobian = (
            (identity - np.outer(normal, normal))
            / float(np.linalg.norm(edge))
            @ perpendicular
            @ edge_jacobian
        )
        relative = obstacle_polygon - anchor[None, :]
        values = relative @ normal
        gradients = (
            relative @ normal_jacobian
            - normal @ base_j[index]
        )
        value, gradient = _soft_min_with_gradient(
            values, gradients, kappa
        )
        face_values.append(value)
        face_gradients.append(gradient)

    face_array = np.asarray(face_values, dtype=np.float64)
    face_rows = np.asarray(face_gradients, dtype=np.float64)
    maximum = float(np.max(face_array))
    shifted = np.exp(kappa * (face_array - maximum))
    weights = shifted / float(np.sum(shifted))
    lower_maximum = (
        maximum
        + math.log(float(np.sum(shifted))) / kappa
        - math.log(float(len(face_values))) / kappa
    )
    configuration_gradient = weights @ face_rows
    if (
        not math.isfinite(lower_maximum)
        or not np.isfinite(configuration_gradient).all()
    ):
        raise ValueError("Polygon barrier produced non-finite output.")
    sample = RawBarrierSample(
        family=BarrierFamily.NAVIGATION,
        protected_id=protected_id,
        obstacle_id=obstacle.fragment_id,
        obstacle_parent_id=obstacle.parent_stable_id,
        value=lower_maximum - float(policy_margin),
        point_jacobian=tuple(
            float(value) for value in configuration_gradient[:3]
        ),
        surface_distance_m=lower_maximum,
        protected_radius_m=0.0,
        policy_margin_m=float(policy_margin),
        distance_backend=SYMMETRIC_POLYGON_BACKEND_VERSION,
        geometry_kind="oriented_box_fragment",
        distance_is_signed=True,
        nonsmooth=False,
    )
    return sample, configuration_gradient.reshape(1, nv)


def signed_point_cylinder_barrier(
    point: Sequence[float] | np.ndarray,
    obstacle: CylinderFragment,
    *,
    protected_radius: float,
    policy_margin: float,
    protected_id: str,
) -> RawBarrierSample:
    """Signed Euclidean sphere-to-cylinder separation and point gradient.

    Exact for the upright cylinder in its own frame: radial excess in the
    local XY plane combines with the axial excess exactly like the
    point-to-box formula. Positive outside, zero on the surface, negative
    inside. Ties (point on the axis, or equal radial/axial excess) use a
    deterministic subgradient and are flagged nonsmooth.
    """

    if not isinstance(obstacle, CylinderFragment):
        raise TypeError("Signed cylinder distance requires a cylinder fragment.")
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (3,) or not np.isfinite(point_array).all():
        raise ValueError("Protected point must contain three finite values.")
    for value, label in (
        (protected_radius, "Protected radius"),
        (policy_margin, "Policy margin"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and non-negative.")
    if not isinstance(protected_id, str) or not protected_id:
        raise ValueError("Protected geometry ID must be non-empty.")

    local = obstacle.rotation.T @ (
        point_array - np.asarray(obstacle.center, dtype=np.float64)
    )
    radial_norm = float(math.hypot(local[0], local[1]))
    nonsmooth = False
    if radial_norm > 1e-12:
        radial_direction = np.asarray(
            (local[0] / radial_norm, local[1] / radial_norm, 0.0)
        )
    else:
        radial_direction = np.asarray((1.0, 0.0, 0.0))
        nonsmooth = True
    d_radial = radial_norm - obstacle.radius
    d_axial = abs(float(local[2])) - obstacle.half_height
    axial_sign = 1.0 if local[2] >= 0.0 else -1.0
    axial_direction = np.asarray((0.0, 0.0, axial_sign))
    if d_radial > 0.0 and d_axial > 0.0:
        distance = math.hypot(d_radial, d_axial)
        local_gradient = (
            radial_direction * (d_radial / distance)
            + axial_direction * (d_axial / distance)
        )
    elif d_radial >= d_axial:
        distance = d_radial
        local_gradient = radial_direction
        nonsmooth = nonsmooth or math.isclose(
            d_radial, d_axial, rel_tol=0.0, abs_tol=1e-12
        )
    else:
        distance = d_axial
        local_gradient = axial_direction
        nonsmooth = nonsmooth or abs(float(local[2])) <= 1e-12
    world_gradient = obstacle.rotation @ local_gradient
    if (
        not math.isfinite(distance)
        or not np.isfinite(world_gradient).all()
        or not math.isclose(
            float(np.linalg.norm(world_gradient)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("Signed cylinder distance produced invalid geometry.")
    return RawBarrierSample(
        family=BarrierFamily.MOBILE_MANIPULATION,
        protected_id=protected_id,
        obstacle_id=obstacle.fragment_id,
        obstacle_parent_id=obstacle.parent_stable_id,
        value=(
            distance - float(protected_radius) - float(policy_margin)
        ),
        point_jacobian=tuple(float(value) for value in world_gradient),
        surface_distance_m=distance,
        protected_radius_m=float(protected_radius),
        policy_margin_m=float(policy_margin),
        distance_backend=SIGNED_CYLINDER_DISTANCE_BACKEND_VERSION,
        geometry_kind="cylinder_fragment",
        distance_is_signed=True,
        nonsmooth=nonsmooth,
    )


def rectangle_disc_separation_barrier(
    base_vertices_xy: Sequence[Sequence[float]] | np.ndarray,
    base_vertex_jacobians: Sequence[np.ndarray],
    obstacle: CylinderFragment,
    *,
    policy_margin: float,
    protected_id: str = "base",
) -> tuple[RawBarrierSample, np.ndarray]:
    """Signed base-rectangle-to-disc separation with configuration row.

    The upright cylinder projects to a disc in XY. The signed separation is
    the point-to-convex-polygon distance from the disc centre minus the
    disc radius: positive when separated, negative once the disc overlaps
    the rectangle. The configuration gradient chains through the base
    vertex Jacobians via the envelope theorem on the closest feature;
    feature ties and the interior case use a deterministic subgradient and
    are flagged nonsmooth. Exact — no smoothing gain, no bias.
    """

    if not isinstance(obstacle, CylinderFragment):
        raise TypeError("Rectangle-disc separation requires a cylinder fragment.")
    axis_cosine = abs(float(obstacle.rotation[2, 2]))
    if axis_cosine < CYLINDER_UPRIGHT_MINIMUM_AXIS_COSINE:
        raise ValueError(
            "Rectangle-disc separation requires a near-upright cylinder: "
            f"axis cosine {axis_cosine!r}."
        )
    vertices = _finite_matrix(base_vertices_xy, (4, 2), "Base vertices")
    jacobians = [np.asarray(item, dtype=np.float64) for item in base_vertex_jacobians]
    if len(jacobians) != 4 or any(j.ndim != 2 or j.shape[0] != 2 for j in jacobians):
        raise ValueError("Base vertex Jacobians must be four (2, nv) arrays.")
    nv = jacobians[0].shape[1]
    if any(j.shape[1] != nv for j in jacobians):
        raise ValueError("Base vertex Jacobians must share one tangent width.")
    if (
        isinstance(policy_margin, bool)
        or not isinstance(policy_margin, (int, float))
        or not math.isfinite(float(policy_margin))
        or float(policy_margin) < 0.0
    ):
        raise ValueError("Policy margin must be finite and non-negative.")

    centre = np.asarray(obstacle.center[:2], dtype=np.float64)
    inside = True
    best_face = -math.inf
    best_face_index = 0
    closest_distance = math.inf
    closest_edge = 0
    closest_t = 0.0
    for index in range(4):
        a = vertices[index]
        b = vertices[(index + 1) % 4]
        edge = b - a
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-12:
            raise ValueError("Base rectangle has a degenerate edge.")
        normal = np.asarray((edge[1], -edge[0])) / edge_norm
        side = float(normal @ (centre - a))
        if side > best_face:
            best_face = side
            best_face_index = index
        if side > 0.0:
            inside = False
        t = float(np.clip((centre - a) @ edge / (edge_norm * edge_norm), 0.0, 1.0))
        candidate = float(np.linalg.norm(centre - (a + t * edge)))
        if candidate < closest_distance:
            closest_distance = candidate
            closest_edge = index
            closest_t = t
    nonsmooth = False
    gradient = np.zeros(nv, dtype=np.float64)
    if not inside and closest_distance > 1e-12:
        a_index = closest_edge
        b_index = (closest_edge + 1) % 4
        a = vertices[a_index]
        b = vertices[b_index]
        closest_point = a + closest_t * (b - a)
        direction = (centre - closest_point) / closest_distance
        separation = closest_distance
        # Envelope theorem: the active feature is locally constant, so the
        # separation gradient distributes over the two edge vertices.
        gradient = -(
            (1.0 - closest_t) * (direction @ jacobians[a_index])
            + closest_t * (direction @ jacobians[b_index])
        )
        nonsmooth = closest_t <= 1e-9 or closest_t >= 1.0 - 1e-9
    else:
        # Disc centre inside (or on) the rectangle: separation is the
        # negative interior clearance from the deepest face. The face
        # normal is treated as frozen — a deterministic subgradient on the
        # piecewise boundary, consistent with the SAT tie handling.
        a_index = best_face_index
        b_index = (best_face_index + 1) % 4
        a = vertices[a_index]
        b = vertices[b_index]
        edge = b - a
        edge_norm = float(np.linalg.norm(edge))
        normal = np.asarray((edge[1], -edge[0])) / edge_norm
        separation = best_face
        gradient = -(normal @ jacobians[a_index])
        nonsmooth = True
    value = separation - float(obstacle.radius) - float(policy_margin)
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Rectangle-disc separation produced invalid output.")
    sample = RawBarrierSample(
        family=BarrierFamily.NAVIGATION,
        protected_id=protected_id,
        obstacle_id=obstacle.fragment_id,
        obstacle_parent_id=obstacle.parent_stable_id,
        value=value,
        point_jacobian=tuple(float(v) for v in gradient[:3]),
        surface_distance_m=separation - float(obstacle.radius),
        protected_radius_m=0.0,
        policy_margin_m=float(policy_margin),
        distance_backend=RECTANGLE_DISC_BACKEND_VERSION,
        geometry_kind="cylinder_fragment",
        distance_is_signed=True,
        nonsmooth=nonsmooth,
    )
    return sample, gradient.reshape(1, nv)

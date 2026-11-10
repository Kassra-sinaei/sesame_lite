"""DEC-075 swept-volume task pruning: deterministic event-time admission.

This module decides, once per action segment, which scene objects may become
per-row QP constraints. It is pure geometry over conservative axis-aligned
bounds — no simulator, model, or language-model input.

Contracts (docs/phase-6-scope.md, DEC-075):

- NAVIGATION segments admit every object whose AABB lies within
  ``r_c = r_base_circumscribed + policy_margin + BETA`` of the planned path
  polyline. The polyline is sampled at spacing ``delta_s`` and tested with the
  batched point-to-AABB lower bound at threshold ``r_c + delta_s / 2``; the
  half-spacing slack makes discretization a sound over-approximation
  (admission is a superset of the exact continuous corridor test).
- VLA segments admit every object whose AABB lies within
  ``rho_v + eef_reach + policy_margin`` of the segment-start base position.
- An optional vertical interval restricts admission; when omitted the test is
  XY-only, which can only over-admit and therefore preserves the recall
  property.
- Admitted sets are epoch-bound. A per-tick scalar monitor detects departure
  from the declared region (cross-track distance beyond BETA, or exit from
  the validity region) and the caller must re-prune; consumers reject stale
  epochs fail-closed.

The recall property — no object the robot can contact during the segment is
excluded — holds under the static-scene, containment-or-re-prune, and
conservative-bound assumptions recorded in the scope document. Classifier
labels (obstacle/passable/irrelevant) are applied *after* geometric
admission and are audited separately; this module never drops an object for
semantic reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


SWEPT_VOLUME_PRUNING_VERSION = "phase6-swept-volume-pruning/2"

#: /2 (owner review, 2026-08-18): the classifier's ``irrelevant`` label is
#: advisory. In r1 the model labelled walls (walls_dlfarz_0, walls_umqqmh_0,
#: walls_xsubff_0) irrelevant in five epochs and their rows were removed
#: while the base stood 0.68 m away. A wall's obstacle-ness is not a
#: judgement to delegate to a language model. ``irrelevant`` now removes a
#: geometric candidate only when the graph category is in this allowlist
#: of overhead structure that no protected point can reach; every other
#: candidate falls back to ``obstacle`` and the override is recorded on the
#: bundle. Passability stays entirely with the classifier.
IRRELEVANT_LABEL_ALLOWLIST = frozenset(
    {"roof", "ceilings", "ceiling", "downlight", "track_light", "room_light"}
)


def admitted_after_labels(
    candidates: Sequence[str],
    labels: Mapping[str, str],
    categories: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Apply classifier labels to geometric candidates.

    Returns the admitted ids (in candidate order) and the ids whose
    ``irrelevant`` label was overridden to ``obstacle`` because their
    category is outside ``IRRELEVANT_LABEL_ALLOWLIST``.
    """

    admitted: list[str] = []
    overrides: dict[str, str] = {}
    for stable_id in candidates:
        label = labels[stable_id]
        if label == "obstacle":
            admitted.append(stable_id)
        elif label == "irrelevant":
            category = str(categories.get(stable_id, "")).strip().lower()
            if category in IRRELEVANT_LABEL_ALLOWLIST:
                continue
            overrides[stable_id] = "irrelevant->obstacle"
            admitted.append(stable_id)
        # passable: removed by the classifier's judgement
    return tuple(admitted), overrides

# Preregistered DEC-075 constants. Adjustable only on calibration
# scenes/seeds, never on reported evaluation episodes.
BETA_DEVIATION_BUDGET_M = 0.5
RHO_V_VALIDITY_RADIUS_M = 1.0
DEFAULT_SAMPLE_SPACING_M = 0.05

_SEGMENT_KINDS = ("NAVIGATION", "VLA")


def _validated_polyline(points: Sequence[Sequence[float]]) -> np.ndarray:
    polyline = np.asarray(points, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] < 1:
        raise ValueError("A corridor polyline needs at least one XY point.")
    if not np.isfinite(polyline).all():
        raise ValueError("Corridor polyline coordinates must be finite.")
    return polyline


def sample_polyline_xy(
    points: Sequence[Sequence[float]],
    *,
    spacing: float,
) -> np.ndarray:
    """Return polyline vertices plus interpolated samples at most ``spacing`` apart."""

    polyline = _validated_polyline(points)
    if (
        isinstance(spacing, bool)
        or not isinstance(spacing, (int, float))
        or not math.isfinite(float(spacing))
        or float(spacing) <= 0.0
    ):
        raise ValueError("Polyline sample spacing must be finite and positive.")
    samples = [polyline[0]]
    for start, end in zip(polyline[:-1], polyline[1:], strict=False):
        length = float(np.linalg.norm(end - start))
        if length == 0.0:
            continue
        steps = max(1, int(math.ceil(length / float(spacing))))
        for index in range(1, steps + 1):
            samples.append(start + (end - start) * (index / steps))
    return np.asarray(samples, dtype=np.float64)


def _aabb_bounds(
    aabb_by_id: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    identifiers = tuple(sorted(aabb_by_id))
    if not identifiers:
        return identifiers, np.zeros((0, 3)), np.zeros((0, 3))
    lower = np.asarray(
        [aabb_by_id[key][0] for key in identifiers], dtype=np.float64
    )
    upper = np.asarray(
        [aabb_by_id[key][1] for key in identifiers], dtype=np.float64
    )
    if (
        lower.shape != (len(identifiers), 3)
        or upper.shape != (len(identifiers), 3)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("Object AABBs must be finite ordered 3-D bounds.")
    return identifiers, lower, upper


def _xy_distances_to_points(
    lower: np.ndarray,
    upper: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    """Minimum XY point-to-AABB distance per object over all sample points.

    Identical clamped-separation expression as the runtime broad phase
    (``max(lower - p, p - upper, 0)``), batched over objects x samples.
    """

    deltas = np.maximum(
        lower[:, None, :2] - points_xy[None, :, :],
        points_xy[None, :, :] - upper[:, None, :2],
    )
    np.maximum(deltas, 0.0, out=deltas)
    return np.min(np.linalg.norm(deltas, axis=2), axis=1)


def _vertical_overlap(
    lower: np.ndarray,
    upper: np.ndarray,
    interval: tuple[float, float] | None,
) -> np.ndarray:
    if interval is None:
        return np.ones(lower.shape[0], dtype=bool)
    low, high = float(interval[0]), float(interval[1])
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError("Vertical admission interval must be finite and ordered.")
    return ~((upper[:, 2] < low) | (lower[:, 2] > high))


def corridor_radius_m(
    *,
    base_circumscribed_radius_m: float,
    policy_margin_m: float,
    beta_m: float = BETA_DEVIATION_BUDGET_M,
) -> float:
    values = (base_circumscribed_radius_m, policy_margin_m, beta_m)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError("Corridor radius terms must be finite and non-negative.")
    if float(base_circumscribed_radius_m) <= 0.0:
        raise ValueError("The base circumscribed radius must be positive.")
    return float(base_circumscribed_radius_m + policy_margin_m + beta_m)


def corridor_admitted_ids(
    *,
    path_polyline_xy: Sequence[Sequence[float]],
    aabb_by_id: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    corridor_radius: float,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
    vertical_interval_m: tuple[float, float] | None = None,
) -> tuple[str, ...]:
    """Admit every object within the sampled corridor (sound over-approximation)."""

    if (
        isinstance(corridor_radius, bool)
        or not isinstance(corridor_radius, (int, float))
        or not math.isfinite(float(corridor_radius))
        or float(corridor_radius) <= 0.0
    ):
        raise ValueError("Corridor radius must be finite and positive.")
    identifiers, lower, upper = _aabb_bounds(aabb_by_id)
    if not identifiers:
        return ()
    samples = sample_polyline_xy(path_polyline_xy, spacing=sample_spacing_m)
    threshold = float(corridor_radius) + float(sample_spacing_m) / 2.0
    distances = _xy_distances_to_points(lower, upper, samples)
    admitted = (distances <= threshold) & _vertical_overlap(
        lower, upper, vertical_interval_m
    )
    return tuple(
        identifier
        for identifier, keep in zip(identifiers, admitted, strict=True)
        if keep
    )


def validity_region_admitted_ids(
    *,
    center_xy: Sequence[float],
    aabb_by_id: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    region_radius_m: float = RHO_V_VALIDITY_RADIUS_M,
    reach_m: float = 0.0,
    policy_margin_m: float = 0.0,
    vertical_interval_m: tuple[float, float] | None = None,
) -> tuple[str, ...]:
    """Admit every object within ``rho_v + reach + margin`` of the region center."""

    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Validity-region center must be a finite XY point.")
    values = (region_radius_m, reach_m, policy_margin_m)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError("Validity-region radii must be finite and non-negative.")
    if float(region_radius_m) <= 0.0:
        raise ValueError("The validity-region radius must be positive.")
    identifiers, lower, upper = _aabb_bounds(aabb_by_id)
    if not identifiers:
        return ()
    threshold = float(region_radius_m) + float(reach_m) + float(policy_margin_m)
    distances = _xy_distances_to_points(lower, upper, center.reshape(1, 2))
    admitted = (distances <= threshold) & _vertical_overlap(
        lower, upper, vertical_interval_m
    )
    return tuple(
        identifier
        for identifier, keep in zip(identifiers, admitted, strict=True)
        if keep
    )


def min_distance_to_polyline_xy(
    point_xy: Sequence[float],
    points: Sequence[Sequence[float]],
) -> float:
    """Exact XY point-to-polyline distance for the per-tick deviation monitor."""

    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Deviation-monitor point must be a finite XY point.")
    polyline = _validated_polyline(points)
    if polyline.shape[0] == 1:
        return float(np.linalg.norm(point - polyline[0]))
    best = math.inf
    for start, end in zip(polyline[:-1], polyline[1:], strict=False):
        segment = end - start
        length_sq = float(segment @ segment)
        if length_sq == 0.0:
            candidate = float(np.linalg.norm(point - start))
        else:
            t = float(np.clip((point - start) @ segment / length_sq, 0.0, 1.0))
            candidate = float(np.linalg.norm(point - (start + t * segment)))
        best = min(best, candidate)
    return best


@dataclass(frozen=True)
class PruningBundle:
    """One epoch-bound admitted set for exactly one action segment."""

    epoch: int
    segment_kind: str
    parent_graph_snapshot_id: str
    admitted_parent_ids: tuple[str, ...]
    geometric_candidate_ids: tuple[str, ...]
    classifier_labels: Mapping[str, str] = field(default_factory=dict)
    binding: bool = True
    beta_m: float = BETA_DEVIATION_BUDGET_M
    rho_v_m: float = RHO_V_VALIDITY_RADIUS_M
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M
    corridor_radius_m: float | None = None
    path_polyline_xy: tuple[tuple[float, float], ...] | None = None
    region_center_xy: tuple[float, float] | None = None
    pruning_version: str = SWEPT_VOLUME_PRUNING_VERSION
    #: /2: candidates the classifier called irrelevant but the structural
    #: allowlist kept as obstacles (evidence for Table 2).
    label_overrides: Mapping[str, str] = field(default_factory=dict)
    #: The classification record behind the labels (version, model identity,
    #: fallback_used, request/response digests, error detail). Without it a
    #: silent all-obstacle fallback is indistinguishable from a genuine
    #: obstacle verdict (r2 s202, the SESAME run, labelled the curtain obstacle in all
    #: four bundles and the evidence could not say why).
    classification: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.segment_kind not in _SEGMENT_KINDS:
            raise ValueError("Pruning bundle segment kind is invalid.")
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
        ):
            raise ValueError("Pruning bundle epoch must be a non-negative int.")
        if not self.parent_graph_snapshot_id:
            raise ValueError("Pruning bundle requires its snapshot binding.")
        admitted = set(self.admitted_parent_ids)
        candidates = set(self.geometric_candidate_ids)
        if len(admitted) != len(self.admitted_parent_ids) or len(
            candidates
        ) != len(self.geometric_candidate_ids):
            raise ValueError("Pruning bundle identity lists must be unique.")
        if not admitted <= candidates:
            raise ValueError(
                "Admitted objects must be geometric candidates: the "
                "classifier may only remove, never add."
            )
        allowed_labels = {"obstacle", "passable", "irrelevant"}
        for key, label in self.classifier_labels.items():
            if key not in candidates:
                raise ValueError(
                    "Classifier labels must cover only geometric candidates."
                )
            if label not in allowed_labels:
                raise ValueError("Classifier label vocabulary is fixed.")
        removed = candidates - admitted
        if self.binding:
            for identifier in removed:
                if self.classifier_labels.get(identifier) not in {
                    "passable",
                    "irrelevant",
                }:
                    raise ValueError(
                        "A binding bundle may exclude a geometric candidate "
                        "only through an explicit passable/irrelevant label."
                    )
        if self.segment_kind == "NAVIGATION":
            if self.path_polyline_xy is None or self.corridor_radius_m is None:
                raise ValueError(
                    "NAVIGATION bundles require the corridor definition."
                )
        elif self.region_center_xy is None:
            raise ValueError("VLA bundles require the validity-region center.")

    def region_violated(self, base_position_xy: Sequence[float]) -> bool:
        """Scalar per-tick monitor: has the robot left the declared region?"""

        if self.segment_kind == "NAVIGATION":
            assert self.path_polyline_xy is not None
            return (
                min_distance_to_polyline_xy(
                    base_position_xy, self.path_polyline_xy
                )
                > self.beta_m
            )
        assert self.region_center_xy is not None
        center = np.asarray(self.region_center_xy, dtype=np.float64)
        point = np.asarray(base_position_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Monitor base position must be a finite XY point.")
        return float(np.linalg.norm(point - center)) > self.rho_v_m

    def as_dict(self) -> dict[str, object]:
        return {
            "pruning_version": self.pruning_version,
            "epoch": self.epoch,
            "segment_kind": self.segment_kind,
            "parent_graph_snapshot_id": self.parent_graph_snapshot_id,
            "admitted_parent_ids": list(self.admitted_parent_ids),
            "geometric_candidate_ids": list(self.geometric_candidate_ids),
            "classifier_labels": dict(self.classifier_labels),
            "binding": self.binding,
            "beta_m": self.beta_m,
            "rho_v_m": self.rho_v_m,
            "sample_spacing_m": self.sample_spacing_m,
            "corridor_radius_m": self.corridor_radius_m,
            "path_polyline_xy": (
                None
                if self.path_polyline_xy is None
                else [list(point) for point in self.path_polyline_xy]
            ),
            "region_center_xy": (
                None
                if self.region_center_xy is None
                else list(self.region_center_xy)
            ),
            "label_overrides": dict(self.label_overrides),
            "classification": (
                None if self.classification is None else dict(self.classification)
            ),
        }

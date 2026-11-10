"""Segments, offline JSON plans and the two-stage pruning bundle.

A plan is a list of segments (NAVIGATION or VLA); the swept-volume stage and the semantic classifier prune each
segment once, before it runs. Plans come from the LangGraph planner (``agents/planner_agent.py``, roles are scene
graph nodes ``{"node": "O(12)"}``) or from a hand-written file in ``plans/`` (roles are RoboCasa references,
resolved through the oracle lookup), or, without either, from the task's own definition (``default_plan``).
Contact exemptions are whole scene-graph nodes: the task target may be touched, everything else stays protected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from sesame_lite.agents.pruning_classifier import PruningCandidate, PruningCandidateClassifier, PruningClassification
from sesame_lite.geometry.swept_volume import (
    BETA_DEVIATION_BUDGET_M,
    DEFAULT_SAMPLE_SPACING_M,
    RHO_V_VALIDITY_RADIUS_M,
    admitted_after_labels,
    corridor_admitted_ids,
    corridor_radius_m,
    validity_region_admitted_ids,
)
from sesame_lite.scene_graph import OracleLookup, SceneGraph, fixture_node_id

NAVIGATION = "NAVIGATION"
VLA = "VLA"


@dataclass
class Segment:
    kind: str
    goal_text: str  # language for the classifier (task goal) and for the VLA
    path_polyline_xy: np.ndarray | None = None  # NAVIGATION
    goal_yaw: float | None = None
    goal_offset_m: float = 0.0  # standoff added to RoboCasa's base placement
    center_xy: np.ndarray | None = None  # VLA
    exempt_ids: tuple[str, ...] = ()  # scene-graph nodes the robot may touch in this segment
    done: dict = field(default_factory=lambda: {"check": "task_success"})
    max_steps: int | None = None
    index: int = 0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "goal_text": self.goal_text,
            "path_polyline_xy": None if self.path_polyline_xy is None else np.round(self.path_polyline_xy, 3).tolist(),
            "goal_yaw": self.goal_yaw,
            "goal_offset_m": self.goal_offset_m,
            "center_xy": None if self.center_xy is None else np.round(self.center_xy, 3).tolist(),
            "exempt_ids": list(self.exempt_ids),
            "done": dict(self.done),
            "max_steps": self.max_steps,
        }


@dataclass
class PruningBundle:
    segment: Segment
    snapshot_id: str
    candidate_ids: tuple[str, ...]  # stage 1 (geometric), node ids
    labels: dict[str, str]  # stage 2
    admitted_ids: tuple[str, ...]  # nodes kept after labels (exemptions are removed when rows are built)
    classification: PruningClassification | None
    n_scene_objects: int
    corridor_radius_m: float | None = None
    region_radius_m: float | None = None
    classifier_latency_s: float | None = None
    overrides: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "segment": self.segment.as_dict(),
            "snapshot_id": self.snapshot_id,
            "n_scene_objects": self.n_scene_objects,
            "n_candidates": len(self.candidate_ids),
            "n_admitted": len(self.admitted_ids),
            "candidate_ids": list(self.candidate_ids),
            "labels": dict(self.labels),
            "overrides": dict(self.overrides),
            "admitted_ids": list(self.admitted_ids),
            "corridor_radius_m": self.corridor_radius_m,
            "region_radius_m": self.region_radius_m,
            "classification": None if self.classification is None else self.classification.as_dict(),
        }


# --------------------------------------------------------------------------- base geometry helpers


class NoPathError(RuntimeError):
    """The grid planner found no path between start and goal through the balls (the mission stops; no fallback)."""


def band_points(balls: dict, z_band: tuple[float, float], exempt: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
    """The balls (centre x-y, radius) of every non-exempt object that reach into the base's height band."""
    xy, rho = [np.zeros((0, 2))], [np.zeros(0)]
    for nid, b in balls.items():
        if nid in exempt:
            continue
        keep = (b.points[:, 2] - b.rho <= z_band[1]) & (b.points[:, 2] + b.rho >= z_band[0])
        xy.append(b.points[keep, :2])
        rho.append(b.rho[keep])
    return np.vstack(xy), np.concatenate(rho)


def base_clearance(rect_xy: np.ndarray, pose_xy, yaw: float, balls: dict, z_band: tuple[float, float], exempt: Sequence[str] = ()) -> float:
    """Smallest signed separation between the base rectangle at (pose, yaw) and the balls in the base's height band
    (the same points and radii as the filter; spec v4, section 9)."""
    xy, rho = band_points(balls, z_band, exempt)
    if len(xy) == 0:
        return math.inf
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    q = (xy - np.asarray(pose_xy)) @ R  # base frame
    lo, hi = rect_xy.min(axis=0), rect_xy.max(axis=0)
    outside = np.maximum(np.maximum(lo - q, q - hi), 0.0)
    dist = np.where(np.any(outside > 0, axis=1), np.linalg.norm(outside, axis=1), -np.min(np.minimum(q - lo, hi - q), axis=1))
    return float(np.min(dist - rho))


def standoff_goal(balls: dict, rect_xy: np.ndarray, goal_xy, goal_yaw: float, away_from_xy, *, z_band, clearance_m: float, max_offset_m: float = 0.6, step_m: float = 0.02, exempt: Sequence[str] = ()) -> tuple[np.ndarray, float | None]:
    """Move the goal away from the fixture along the outward direction until the base clears every ball by
    ``clearance_m``. If no offset up to ``max_offset_m`` works, RoboCasa's goal is kept and the offset is None
    (recorded in the segment)."""
    goal_xy, away = np.asarray(goal_xy, dtype=float), np.asarray(away_from_xy, dtype=float)
    d = goal_xy - away
    n = np.linalg.norm(d)
    direction = d / n if n > 1e-6 else np.array([-np.cos(goal_yaw), -np.sin(goal_yaw)])
    s = 0.0
    while s <= max_offset_m:
        if base_clearance(rect_xy, goal_xy + s * direction, goal_yaw, balls, z_band, exempt) >= clearance_m:
            return goal_xy + s * direction, s
        s += step_m
    return goal_xy, None


# --------------------------------------------------------------------------- grid path planning


def _floor_bounds(graph: SceneGraph, margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([n.center for n in graph.obstacle_nodes() if graph.is_fixture(n.node_id)])  # fixtures only: a stable grid origin
    return pts[:, :2].min(axis=0) - margin, pts[:, :2].max(axis=0) + margin


def plan_base_path(graph: SceneGraph, balls: dict, start_xy, goal_xy, *, inflation_m: float, z_band=(0.0, 0.45), resolution_m: float = 0.05) -> np.ndarray:
    """8-connected A* on an occupancy grid: a cell is occupied if it lies within rho_m + ``inflation_m`` of the centre
    of a ball in the base's height band (the same balls as the filter).

    Cells within one inflation radius of the start and goal are cleared (both poses touch furniture by
    construction). Raises NoPathError if no path exists.
    """
    from scipy.spatial import cKDTree

    lo, hi = _floor_bounds(graph)
    nx, ny = (np.ceil((hi - lo) / resolution_m).astype(int) + 1).tolist()
    X, Y = np.meshgrid(lo[0] + resolution_m * np.arange(nx), lo[1] + resolution_m * np.arange(ny), indexing="ij")
    occ = np.zeros(nx * ny, dtype=bool)
    xy, rho = band_points(balls, z_band)
    if len(xy):
        hits = cKDTree(np.c_[X.ravel(), Y.ravel()]).query_ball_point(xy, r=rho + inflation_m)
        occ[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])] = True
    occ = occ.reshape(nx, ny)
    start, goal = np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float)
    for p in (start, goal):
        occ &= ~((X - p[0]) ** 2 + (Y - p[1]) ** 2 <= (inflation_m + resolution_m) ** 2)

    def cell(p):
        c = np.round((p - lo) / resolution_m).astype(int)
        return int(np.clip(c[0], 0, nx - 1)), int(np.clip(c[1], 0, ny - 1))

    s, g = cell(start), cell(goal)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    best, came, heap, found = {s: 0.0}, {}, [(math.dist(s, g), 0.0, s)], False
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == g:
            found = True
            break
        if cost > best.get(cur, math.inf):
            continue
        for dx, dy in steps:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or occ[nxt]:
                continue
            c = cost + math.hypot(dx, dy)
            if c < best.get(nxt, math.inf):
                best[nxt], came[nxt] = c, cur
                heapq.heappush(heap, (c + math.dist(nxt, g), c, nxt))
    if not found:
        raise NoPathError(f"no base path from {np.round(start, 2)} to {np.round(goal, 2)} through the balls (inflation {inflation_m:.2f} m)")
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    pts = np.array([[lo[0] + resolution_m * c[0], lo[1] + resolution_m * c[1]] for c in cells])
    pts[0], pts[-1] = start, goal
    keep = [0]
    for i in range(1, len(pts) - 1):
        d1, d2 = pts[i] - pts[keep[-1]], pts[i + 1] - pts[i]
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-9:
            keep.append(i)
    keep.append(len(pts) - 1)
    return pts[keep]


# --------------------------------------------------------------------------- plans


def load_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve_role(role: dict, rs_env, lookup: OracleLookup) -> str | None:
    """A role names one scene-graph node. Forms:
    {"node": "O(12)"}             chosen by the LLM planner from the scene graph
    {"target_fixture": true}      the task's target fixture  (hand-written plans: resolved through the oracle lookup)
    {"fixture_ref": key}          a fixture the RoboCasa task refers to by key
    {"fixture": name}             a RoboCasa fixture by name
    {"object": name}              a RoboCasa object by its config name
    """
    if "node" in role:
        return role["node"] if role["node"] in lookup.bodies else None
    if role.get("target_fixture"):
        return lookup.node(fixture_node_id(rs_env, getattr(rs_env, "target_fixture", None)))
    if "fixture_ref" in role:
        return lookup.node(fixture_node_id(rs_env, role["fixture_ref"]))
    if "fixture" in role:
        return lookup.node(fixture_node_id(rs_env, role["fixture"]))
    if "object" in role:
        return lookup.node(f"object:{role['object']}")
    return None


def exempt_nodes(entries: list[dict], roles: dict[str, str | None], lookup: OracleLookup) -> tuple[str, ...]:
    """Nodes the robot may touch. An entry is {"role": name} (that role's node, plus the surface it stands on if
    "support" is set) or {"node": "O(12)"} (a node named directly, as the LLM planner writes it)."""
    out = []
    for e in entries:
        nid = e["node"] if "node" in e else roles.get(e["role"])
        if nid is None:
            continue
        out.append(nid)
        if e.get("support") and nid in lookup.support:
            out.append(lookup.support[nid])
    return tuple(dict.fromkeys(out))  # unique, in order


def resolve_plan(plan: dict, rs_env, graph: SceneGraph, lookup: OracleLookup, *, balls: dict, language: str, base_xy, rect_xy: np.ndarray, z_band, clearance_m: float, inflation_m: float) -> list[Segment]:
    """Turn a JSON plan into segments for this episode (roles -> node ids, standoff goals, paths). Obstacles are the
    objects' confidence balls (``balls``: node id -> Balls), the same as the filter's."""
    roles = {name: resolve_role(spec, rs_env, lookup) for name, spec in plan.get("roles", {}).items()}
    oracle_name = {nid: name for name, nid in lookup.node_of.items()}
    segments = []
    cur_xy = np.asarray(base_xy, dtype=float)
    for i, s in enumerate(plan["segments"]):
        exempt = exempt_nodes(s.get("exempt", []), roles, lookup)
        goal_text = s.get("language") or language
        if s["kind"] == NAVIGATION:
            g = s["goal"]
            nid = roles.get(g.get("role"))
            if g.get("robocasa_target") and getattr(rs_env, "target_pos", None) is not None:
                goal_xy, goal_yaw = np.asarray(rs_env.target_pos)[:2], float(np.asarray(rs_env.target_ori)[2])
            else:
                # the base pose in front of the target fixture, from RoboCasa's own placement rule (execution side)
                from robocasa.utils.env_utils import compute_robot_base_placement_pose

                fx = rs_env.fixtures[oracle_name[nid].split(":", 1)[1]]
                pos, ori = compute_robot_base_placement_pose(rs_env, fx)
                goal_xy, goal_yaw = np.asarray(pos)[:2], float(np.asarray(ori)[2])
            offset = 0.0
            if g.get("standoff", True) and nid is not None:
                goal_xy, offset = standoff_goal(balls, rect_xy, goal_xy, goal_yaw, np.asarray(graph.nodes[nid].center[:2]), z_band=z_band, clearance_m=clearance_m, exempt=exempt)
            path = plan_base_path(graph, balls, cur_xy, goal_xy, inflation_m=inflation_m, z_band=z_band)
            segments.append(Segment(kind=NAVIGATION, goal_text=goal_text, path_polyline_xy=path, goal_yaw=goal_yaw, goal_offset_m=offset, exempt_ids=exempt, done=s.get("done", {"check": "base_near_goal", "radius": 0.2}), max_steps=s.get("max_steps"), index=i))
            cur_xy = goal_xy
        else:
            segments.append(Segment(kind=VLA, goal_text=goal_text, center_xy=cur_xy.copy(), exempt_ids=exempt, done=s.get("done", {"check": "task_success"}), max_steps=s.get("max_steps"), index=i))
    return segments


def default_plan(rs_env) -> dict:
    """Plan from the task definition when no JSON plan is given: every fixture and object the task names is exempt."""
    if type(rs_env).__name__.startswith("NavigateKitchen"):
        return {"roles": {"target": {"target_fixture": True}}, "segments": [{"kind": NAVIGATION, "goal": {"role": "target", "robocasa_target": True, "standoff": True}, "done": {"check": "base_near_goal", "radius": 0.2}}]}
    roles, exempt = {}, []
    for key in getattr(rs_env, "fixture_refs", {}) or {}:
        roles[key] = {"fixture_ref": key}
        exempt.append({"role": key})
    for cfg in getattr(rs_env, "object_cfgs", []) or []:
        if cfg.get("type", "obj") == "obj":
            roles[cfg["name"]] = {"object": cfg["name"]}
            exempt.append({"role": cfg["name"], "support": True})
    return {"roles": roles, "segments": [{"kind": VLA, "exempt": exempt, "done": {"check": "task_success"}}]}


# --------------------------------------------------------------------------- two-stage pruning


def ball_aabbs(balls: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned bounds of each object's balls (pruning uses the same obstacles as the filter)."""
    return {nid: ((b.points - b.rho[:, None]).min(axis=0), (b.points + b.rho[:, None]).max(axis=0)) for nid, b in balls.items()}


def build_bundle(segment: Segment, graph: SceneGraph, *, balls: dict, classifier: PruningCandidateClassifier | None, base_circumscribed_radius_m: float, policy_margin_m: float, eef_reach_m: float) -> PruningBundle:
    aabbs = ball_aabbs(balls)
    nodes = {n.node_id: n for n in graph.obstacle_nodes()}
    corridor_r = region_r = None
    if segment.kind == NAVIGATION:
        corridor_r = corridor_radius_m(base_circumscribed_radius_m=base_circumscribed_radius_m, policy_margin_m=policy_margin_m, beta_m=BETA_DEVIATION_BUDGET_M)
        candidates = corridor_admitted_ids(path_polyline_xy=segment.path_polyline_xy, aabb_by_id=aabbs, corridor_radius=corridor_r, sample_spacing_m=DEFAULT_SAMPLE_SPACING_M, vertical_interval_m=None)
    else:
        region_r = RHO_V_VALIDITY_RADIUS_M
        candidates = validity_region_admitted_ids(center_xy=segment.center_xy, aabb_by_id=aabbs, region_radius_m=region_r, reach_m=eef_reach_m, policy_margin_m=policy_margin_m, vertical_interval_m=None)
    candidates = tuple(candidates)
    classification, labels = None, {cid: "obstacle" for cid in candidates}
    if classifier is not None and candidates:
        offered = [PruningCandidate(stable_id=cid, name=nodes[cid].classifier_name(), kind=nodes[cid].kind) for cid in candidates]
        classification = classifier.classify(candidates=offered, task_goal=segment.goal_text, parent_graph_snapshot_id=graph.snapshot_id)
        labels = dict(classification.labels)
    admitted, overrides = admitted_after_labels(candidates, labels, {cid: nodes[cid].category for cid in candidates})
    return PruningBundle(
        segment=segment,
        snapshot_id=graph.snapshot_id,
        candidate_ids=candidates,
        labels=labels,
        admitted_ids=tuple(admitted),
        classification=classification,
        n_scene_objects=len(nodes),
        corridor_radius_m=corridor_r,
        region_radius_m=region_r,
        classifier_latency_s=None if classification is None else classification.latency_s,
        overrides=dict(overrides),
    )

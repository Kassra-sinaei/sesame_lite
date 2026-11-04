"""Oracle 3D scene graph captured from a RoboCasa (robosuite/MuJoCo) kitchen.

Mirrors the OmniGibson oracle adapter of signal_3d: one node per fixture and
per free object with a category, a pose and an oriented bounding box. The box
is the *source of constraint geometry* for the filter and the node list is the
planner's world state. A fixture node carries one *part* per MuJoCo body that
has collision geometry (cabinet body, hinged door, drawer box, fridge doors)
so that rows and contact exemptions are per part; the node's own box is the
union of its parts. Floors and wall backings are dropped like ceilings and
floors were in the OmniGibson capture; walls stay as obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Iterable

import mujoco
import numpy as np

from sesame_lite.geometry.fragments import OrientedBoxFragment

MINIMUM_HALF_EXTENT_M = 0.0005  # 1 mm minimum extent, as in the OmniGibson capture
COLLISION_GEOM_GROUP = 0  # robosuite convention: group 0 collision, group 1 visual
MARKER_GEOM_MAX_SIZE_M = 0.015  # RoboCasa fixtures carry 1 cm marker boxes (e.g. counter "base_*") far from the body
MAX_GEOM_OFFSET_M = 3.0
ROUND = 4  # canonical rounding (1e-4 m) for snapshot ids

_SKIP_FIXTURE_CLASSES = {"Floor"}
_SKIP_NAME_PATTERNS = (re.compile(r"_backing_"),)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _wxyz_to_xyzw(q: np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


@dataclass(frozen=True)
class Part:
    part_id: str  # "<node_id>/part:<body>"
    body: str
    center: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    half_extents: tuple[float, float, float]

    def fragment(self, node_id: str) -> OrientedBoxFragment:
        return OrientedBoxFragment(parent_stable_id=node_id, fragment_id=self.part_id, center=self.center, orientation_xyzw=self.orientation_xyzw, half_extents=self.half_extents)

    def as_dict(self) -> dict:
        return {"part_id": self.part_id, "body": self.body, "center": [round(v, ROUND) for v in self.center], "orientation_xyzw": [round(v, ROUND) for v in self.orientation_xyzw], "half_extents": [round(v, ROUND) for v in self.half_extents]}


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    kind: str  # "fixture" | "object" | "robot"
    category: str
    name: str  # natural-language name for the planner/classifier
    center: tuple[float, float, float]  # union box (axis-aligned) of the parts
    half_extents: tuple[float, float, float]
    parts: tuple[Part, ...]
    source: str

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(2.0 * h for h in self.half_extents)

    def classifier_name(self) -> str:
        x, y, z = self.extent
        return f"{self.category} (approx {x:.2f} x {y:.2f} x {z:.2f} m box)"

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        c, h = np.asarray(self.center), np.asarray(self.half_extents)
        return c - h, c + h

    def fragments(self) -> dict[str, OrientedBoxFragment]:
        return {p.part_id: p.fragment(self.node_id) for p in self.parts}

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "category": self.category,
            "name": self.name,
            "center": [round(v, ROUND) for v in self.center],
            "half_extents": [round(v, ROUND) for v in self.half_extents],
            "parts": [p.as_dict() for p in self.parts],
            "source": self.source,
        }


@dataclass
class OracleSceneGraph:
    nodes: dict[str, SceneNode]
    task_language: str
    edges: list[dict] = field(default_factory=list)  # {"rel": "on"|"inside", "src": object id, "dst": fixture id}
    robot_node_id: str = "robot:robot0"
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def snapshot_id(self) -> str:
        """Hash of the scene geometry (the robot node is state, not scene)."""
        payload = json.dumps([n.as_dict() for n in self.obstacle_nodes()], sort_keys=True)
        return "sg:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def obstacle_nodes(self) -> list[SceneNode]:
        return [n for n in self.nodes.values() if n.kind != "robot"]

    def fragments(self) -> dict[str, OrientedBoxFragment]:
        out = {}
        for n in self.obstacle_nodes():
            out.update(n.fragments())
        return out

    def fragment_ids_of(self, node_ids: Iterable[str], *, exempt_parts: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
        """Fragment ids of the given nodes; ``exempt_parts[node] = ()`` drops the whole node, otherwise the listed body substrings."""
        out = []
        exempt_parts = exempt_parts or {}
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            if nid in exempt_parts:
                pats = exempt_parts[nid]
                if not pats:
                    continue
                out.extend(p.part_id for p in node.parts if not any(s in p.body for s in pats))
            else:
                out.extend(p.part_id for p in node.parts)
        return tuple(out)

    def aabb_by_id(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {n.node_id: n.aabb() for n in self.obstacle_nodes()}

    def support_of(self, object_id: str) -> str | None:
        for e in self.edges:
            if e["src"] == object_id:
                return e["dst"]
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_language": self.task_language,
            "robot_node_id": self.robot_node_id,
            "meta": self.meta,
            "edges": list(self.edges),
            "nodes": [n.as_dict() for n in self.nodes.values()],
        }


# --------------------------------------------------------------------------- MuJoCo helpers


def _raw_model(sim) -> mujoco.MjModel:
    return getattr(sim.model, "_model", sim.model)


def _raw_data(sim) -> mujoco.MjData:
    return getattr(sim.data, "_data", sim.data)


def _subtree_body_ids(model: mujoco.MjModel, root: int) -> set[int]:
    ids = {root}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                ids.add(b)
                break
            p = int(model.body_parentid[p])
    return ids


def _geom_corners_local(model: mujoco.MjModel, gid: int) -> np.ndarray:
    aabb = model.geom_aabb[gid]
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return aabb[None, :3] + signs * aabb[None, 3:]


def _collision_geoms(model: mujoco.MjModel, body_ids: Iterable[int], *, data: mujoco.MjData | None = None, anchor: np.ndarray | None = None) -> list[int]:
    body_ids = set(body_ids)
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in body_ids or int(model.geom_group[g]) != COLLISION_GEOM_GROUP:
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        if float(np.max(model.geom_aabb[g][3:])) < MARKER_GEOM_MAX_SIZE_M:
            continue
        if data is not None and anchor is not None and np.linalg.norm(data.geom_xpos[g] - anchor) > MAX_GEOM_OFFSET_M:
            continue
        out.append(g)
    return out


def _geoms_world_corners(model, data, gids) -> np.ndarray:
    pts = []
    for g in gids:
        R = data.geom_xmat[g].reshape(3, 3)
        pts.append(_geom_corners_local(model, g) @ R.T + data.geom_xpos[g][None, :])
    return np.vstack(pts)


def body_part(model, data, node_id: str, body: int, gids: list[int], *, strip_prefix: str = "") -> Part | None:
    """Oriented box of the geoms of one body, in that body's frame."""
    if not gids:
        return None
    Rb, pb = data.xmat[body].reshape(3, 3), data.xpos[body]
    local = (_geoms_world_corners(model, data, gids) - pb[None, :]) @ Rb
    lo, hi = local.min(axis=0), local.max(axis=0)
    half = np.maximum((hi - lo) / 2.0, MINIMUM_HALF_EXTENT_M)
    center = pb + Rb @ ((lo + hi) / 2.0)
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
    if strip_prefix and bname.startswith(strip_prefix):
        bname = bname[len(strip_prefix):].lstrip("_") or "object"
    return Part(part_id=f"{node_id}/part:{bname}", body=bname, center=tuple(float(v) for v in center), orientation_xyzw=_wxyz_to_xyzw(data.xquat[body]), half_extents=tuple(float(v) for v in half))


def _node(node_id, kind, category, name, parts: list[Part], source) -> SceneNode:
    corners = []
    for p in parts:
        frag = p.fragment(node_id)
        lo, hi = frag.aabb(3)
        corners += [lo, hi]
    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    return SceneNode(node_id=node_id, kind=kind, category=category, name=name, center=tuple(float(v) for v in (lo + hi) / 2), half_extents=tuple(float(v) for v in np.maximum((hi - lo) / 2, MINIMUM_HALF_EXTENT_M)), parts=tuple(parts), source=source)


# --------------------------------------------------------------------------- capture


def capture_scene_graph(rs_env, *, include_walls: bool = True) -> OracleSceneGraph:
    """Build the oracle graph from a live RoboCasa ``Kitchen`` environment (no perception)."""

    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    nodes: dict[str, SceneNode] = {}

    for fx_name, fx in rs_env.fixtures.items():
        cls = type(fx).__name__
        if cls in _SKIP_FIXTURE_CLASSES or any(p.search(fx_name) for p in _SKIP_NAME_PATTERNS) or (cls == "Wall" and not include_walls):
            continue
        root_body = getattr(fx, "root_body", None)
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body) if root_body else -1
        if root_id < 0:
            continue
        anchor = np.asarray(fx.pos, dtype=float) if getattr(fx, "pos", None) is not None else data.xpos[root_id]
        parts = []
        for body in sorted(_subtree_body_ids(model, root_id)):
            part = body_part(model, data, f"fixture:{fx_name}", body, _collision_geoms(model, [body], data=data, anchor=anchor), strip_prefix=fx_name)
            if part is not None:
                parts.append(part)
        if not parts:
            continue
        category = _snake(cls)
        nat = getattr(fx, "nat_lang", None)
        nodes[f"fixture:{fx_name}"] = _node(f"fixture:{fx_name}", "fixture", category, str(nat) if nat else category.replace("_", " "), parts, "fixture_collision_geoms")

    edges = []
    cfg_by_name = {c.get("name"): c for c in (getattr(rs_env, "object_cfgs", []) or [])}
    for obj_name, obj in rs_env.objects.items():
        body = rs_env.obj_body_id[obj_name]
        part = body_part(model, data, f"object:{obj_name}", body, _collision_geoms(model, _subtree_body_ids(model, body)), strip_prefix=obj_name)
        if part is None:
            continue
        # the sampled asset category (e.g. "placemat"), never the task's instance name, which could leak intent
        category = str((cfg_by_name.get(obj_name, {}).get("info") or {}).get("cat") or getattr(obj, "category", "object")).replace("_", " ")
        nodes[f"object:{obj_name}"] = _node(f"object:{obj_name}", "object", category, category, [part], "object_collision_geoms")
        fixture = (cfg_by_name.get(obj_name, {}).get("placement", {}) or {}).get("fixture")
        fixture = getattr(fixture, "name", fixture)
        if fixture and f"fixture:{fixture}" in nodes:
            edges.append({"rel": "on", "src": f"object:{obj_name}", "dst": f"fixture:{fixture}"})

    robot = rs_env.robots[0]
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.root_body)
    rpart = body_part(model, data, "robot:robot0", base_body, _collision_geoms(model, _subtree_body_ids(model, base_body)))
    if rpart is not None:
        nodes["robot:robot0"] = _node("robot:robot0", "robot", "mobile_manipulator", "the robot", [rpart], "robot_collision_geoms")

    try:
        lang = str(rs_env.get_ep_meta().get("lang", ""))
    except Exception:
        lang = ""
    meta = {
        "layout_id": int(getattr(rs_env, "layout_id", -1)) if not isinstance(getattr(rs_env, "layout_id", None), dict) else "custom",
        "style_id": int(getattr(rs_env, "style_id", -1)) if not isinstance(getattr(rs_env, "style_id", None), dict) else "custom",
        "n_fixtures": sum(1 for n in nodes.values() if n.kind == "fixture"),
        "n_objects": sum(1 for n in nodes.values() if n.kind == "object"),
        "n_fragments": sum(len(n.parts) for n in nodes.values() if n.kind != "robot"),
    }
    return OracleSceneGraph(nodes=nodes, task_language=lang, edges=edges, meta=meta)


def perturb(graph: OracleSceneGraph, rng: np.random.Generator, *, eps_pos_m: float = 0.0, eps_yaw_rad: float = 0.0, eps_extent_m: float = 0.0) -> OracleSceneGraph:
    """Hook for the uncertainty study: a noisy copy of the graph (uniform bounded noise per part); the oracle stays for scoring."""

    def rot_z(q, dyaw):
        x, y, z, w = q
        c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
        return (float(w * 0 + x * c + y * s), float(y * c - x * s), float(z * c + w * s), float(w * c - z * s))

    nodes = {}
    for nid, n in graph.nodes.items():
        if n.kind == "robot":
            nodes[nid] = n
            continue
        parts = []
        for p in n.parts:
            dc = rng.uniform(-eps_pos_m, eps_pos_m, 3) if eps_pos_m else np.zeros(3)
            de = rng.uniform(-eps_extent_m, eps_extent_m, 3) if eps_extent_m else np.zeros(3)
            dy = float(rng.uniform(-eps_yaw_rad, eps_yaw_rad)) if eps_yaw_rad else 0.0
            parts.append(Part(p.part_id, p.body, tuple(float(v) for v in np.asarray(p.center) + dc), rot_z(p.orientation_xyzw, dy) if dy else p.orientation_xyzw, tuple(float(v) for v in np.maximum(np.asarray(p.half_extents) + de, MINIMUM_HALF_EXTENT_M))))
        nodes[nid] = _node(nid, n.kind, n.category, n.name, parts, n.source + "+noise")
    meta = dict(graph.meta, noise={"eps_pos_m": eps_pos_m, "eps_yaw_rad": eps_yaw_rad, "eps_extent_m": eps_extent_m}, oracle_snapshot_id=graph.snapshot_id)
    return OracleSceneGraph(nodes=nodes, task_language=graph.task_language, edges=list(graph.edges), meta=meta)


def fixture_node_id(rs_env, ref) -> str | None:
    """Resolve a RoboCasa fixture object, fixture-ref key or fixture name to a node id."""
    fx = rs_env.fixture_refs.get(ref) if isinstance(ref, str) and hasattr(rs_env, "fixture_refs") and ref in rs_env.fixture_refs else ref
    fx = fx[0] if isinstance(fx, tuple) else fx
    name = getattr(fx, "name", fx)
    return f"fixture:{name}" if isinstance(name, str) else None

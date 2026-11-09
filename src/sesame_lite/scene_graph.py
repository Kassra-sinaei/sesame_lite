"""Oracle 3D scene graph, written in MIT-SPARK Hydra's format (spark_dsg).

``capture_scene_graph(rs_env)`` reads ground truth from MuJoCo and builds the graph a Hydra run would output:

    OBJECTS    one node per fixture and per free object, ids O(0), O(1), ...
               semantic_label (config/robocasa_label_space.yaml), an axis-aligned bounding_box,
               position = box center, name left empty (Hydra's mesh segmenter does not set it)
    PLACES     free-space nodes p(k) on a 0.5 m grid over the floor; distance = clearance to the nearest object
    ROOMS      R(0), the kitchen
    BUILDINGS  B(0)
    AGENTS     a(0), the robot pose at capture time (partition 'a', as Hydra's pose-graph tracker writes it)

    edges: place-place (grid neighbours), object-place (nearest place), place-room, room-building

Ids come from sorting the boxes by position (fixtures first), so an id says nothing about the task.

Two things come out of a capture:
- ``SceneGraph``: a small reader over the Hydra graph. The planner, the classifier and the safety filter use only this.
- ``OracleLookup``: which node is which RoboCasa fixture/object and which MuJoCo bodies belong to it. Only the
  evaluation side uses it (contact checks, ground truth, the hand-written reference plans); no model ever sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import mujoco
import numpy as np
import spark_dsg as dsg
import yaml

from sesame_lite.geometry.fragments import OrientedBoxFragment

LABEL_SPACE_FILE = Path(__file__).resolve().parents[2] / "config" / "robocasa_label_space.yaml"
PLACE_SPACING_M = 0.5
PLACE_HEIGHT_M = 0.3  # places sit at base height, like a ground robot's free-space layer
PLACE_MIN_CLEARANCE_M = 0.2  # a grid point closer than this to an object is not free space
AGENT_LAYER, ROBOT_PARTITION = 2, ord("a")  # Hydra: agents are layer 2 under the robot's prefix partition

MINIMUM_HALF_EXTENT_M = 0.0005
COLLISION_GEOM_GROUP = 0  # robosuite convention: group 0 collision, group 1 visual
MARKER_GEOM_MAX_SIZE_M = 0.015  # RoboCasa fixtures carry 1 cm marker boxes far from the body
MAX_GEOM_OFFSET_M = 3.0
ROUND = 4  # rounding (0.1 mm) for the snapshot id

SKIP_FIXTURE_CLASSES = {"Floor"}  # the floor is the surface the places stand on, not an object
SKIP_NAME_PATTERNS = (re.compile(r"_backing_"),)


# --------------------------------------------------------------------------- label space


def class_words(class_name: str) -> str:
    """RoboCasa class name -> label name: HingeCabinet -> hinge cabinet (same rule as scripts/make_label_space.py)."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).lower()


@dataclass
class LabelSpace:
    names: dict[int, str]  # label -> name
    floor_label: int

    @classmethod
    def load(cls, path: Path = LABEL_SPACE_FILE) -> "LabelSpace":
        data = yaml.safe_load(Path(path).read_text())
        names = {e["label"]: e["name"] for e in data["label_names"]}
        return cls(names=names, floor_label=data["surface_places_labels"][0])

    def label_of(self, name: str) -> int:
        for label, n in self.names.items():
            if n == name:
                return label
        return 0  # unknown

    def is_fixture_label(self, label: int) -> bool:
        """Fixture classes come before 'floor' in the file (static furniture, as in a prior map)."""
        return 0 < label < self.floor_label

    def sha256(self) -> str:
        return hashlib.sha256(LABEL_SPACE_FILE.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- reading the graph


@dataclass(frozen=True)
class SceneObject:
    """One OBJECTS-layer node as the rest of the code sees it."""

    node_id: str  # Hydra symbol, "O(12)"
    label: int
    category: str  # the label's name
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    kind: str = "object"  # Hydra has no fixture/object split; kept for the classifier's request format

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(2.0 * h for h in self.half_extents)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        c, h = np.asarray(self.center), np.asarray(self.half_extents)
        return c - h, c + h

    def classifier_name(self) -> str:
        x, y, z = self.extent
        return f"{self.category} (approx {x:.2f} x {y:.2f} x {z:.2f} m box)"

    def fragment(self) -> OrientedBoxFragment:
        """The node's box as one barrier fragment (fragment id = node id)."""
        return OrientedBoxFragment(parent_stable_id=self.node_id, fragment_id=self.node_id, center=self.center, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), half_extents=self.half_extents)

    def as_dict(self) -> dict:
        return {"node_id": self.node_id, "label": self.label, "category": self.category,
                "center": [round(v, ROUND) for v in self.center], "half_extents": [round(v, ROUND) for v in self.half_extents]}


class SceneGraph:
    """Reader over a Hydra DynamicSceneGraph: objects, places, rooms and the robot pose."""

    def __init__(self, graph: dsg.DynamicSceneGraph, labels: LabelSpace):
        self.dsg, self.labels = graph, labels
        self.nodes: dict[str, SceneObject] = {}
        for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
            a = node.attributes
            bb = a.bounding_box
            self.nodes[str(node.id)] = SceneObject(
                node_id=str(node.id), label=int(a.semantic_label), category=labels.names.get(int(a.semantic_label), "unknown"),
                center=tuple(float(v) for v in bb.world_P_center), half_extents=tuple(float(max(v / 2, MINIMUM_HALF_EXTENT_M)) for v in bb.dimensions))
        self.places = {str(n.id): (np.asarray(n.attributes.position, dtype=float), float(n.attributes.distance)) for n in graph.get_layer(dsg.DsgLayers.PLACES).nodes}
        self.rooms = [str(n.id) for n in graph.get_layer(dsg.DsgLayers.ROOMS).nodes]
        agent = next(iter(graph.get_layer(AGENT_LAYER, ROBOT_PARTITION).nodes))
        q = agent.attributes.world_R_body
        self.robot_xy = np.asarray(agent.attributes.position, dtype=float)[:2]
        self.robot_yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    @property
    def snapshot_id(self) -> str:
        """Hash of the object layer (labels and boxes)."""
        payload = json.dumps([n.as_dict() for n in self.nodes.values()], sort_keys=True)
        return "sg:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def obstacle_nodes(self) -> list[SceneObject]:
        return list(self.nodes.values())

    def is_fixture(self, node_id: str) -> bool:
        return self.labels.is_fixture_label(self.nodes[node_id].label)

    def fragments(self) -> dict[str, OrientedBoxFragment]:
        return {nid: n.fragment() for nid, n in self.nodes.items()}

    def fragment_ids_of(self, node_ids: Iterable[str], *, exempt: Iterable[str] = ()) -> tuple[str, ...]:
        """Fragment ids (= node ids) of the given nodes, without the exempt ones."""
        exempt = set(exempt)
        return tuple(nid for nid in node_ids if nid in self.nodes and nid not in exempt)

    def aabb_by_id(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {nid: n.aabb() for nid, n in self.nodes.items()}

    def dsg_node(self, node_id: str):
        """The spark_dsg node behind an id like "O(12)" (to follow Hydra's object -> place -> room links)."""
        return self.dsg.get_node(dsg.NodeSymbol(*_parse_symbol(node_id)))

    def save(self, path: str | Path) -> None:
        self.dsg.save(str(path), include_mesh=False)

    def as_dict(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "format": "spark_dsg (Hydra)", "label_space_sha256": self.labels.sha256(),
                "robot": {"xy": [round(float(v), ROUND) for v in self.robot_xy], "yaw": round(self.robot_yaw, ROUND)},
                "nodes": [n.as_dict() for n in self.nodes.values()], "n_places": len(self.places), "rooms": self.rooms}


def _parse_symbol(text: str) -> tuple[str, int]:
    m = re.fullmatch(r"([A-Za-z])\((\d+)\)", text)
    if m is None:
        raise KeyError(f"not a node id: {text!r}")
    return m.group(1), int(m.group(2))


# --------------------------------------------------------------------------- ground-truth side


@dataclass
class OracleLookup:
    """Which Hydra node is which RoboCasa fixture/object. Evaluation only; never shown to a model."""

    node_of: dict[str, str] = field(default_factory=dict)  # "fixture:<robocasa name>" / "object:<name>" -> "O(k)"
    bodies: dict[str, set[int]] = field(default_factory=dict)  # "O(k)" -> MuJoCo body ids
    support: dict[str, str] = field(default_factory=dict)  # object node -> node of the fixture RoboCasa placed it on
    meta: dict = field(default_factory=dict)  # layout/style ids

    def node(self, oracle_name: str | None) -> str | None:
        return self.node_of.get(oracle_name) if oracle_name else None

    def as_dict(self) -> dict:
        return {"node_of": dict(self.node_of), "support": dict(self.support), "meta": dict(self.meta)}


def fixture_node_id(rs_env, ref) -> str | None:
    """A RoboCasa fixture object, fixture-ref key or fixture name -> its oracle name "fixture:<name>"."""
    fx = rs_env.fixture_refs.get(ref) if isinstance(ref, str) and hasattr(rs_env, "fixture_refs") and ref in rs_env.fixture_refs else ref
    fx = fx[0] if isinstance(fx, tuple) else fx
    name = getattr(fx, "name", fx)
    return f"fixture:{name}" if isinstance(name, str) else None


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


def _collision_geoms(model, body_ids: Iterable[int], *, data=None, anchor=None) -> list[int]:
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


def _world_box(model, data, gids) -> tuple[np.ndarray, np.ndarray] | None:
    """World axis-aligned box around the given geoms (corners of each geom's local box, rotated into the world)."""
    if not gids:
        return None
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    pts = []
    for g in gids:
        local = model.geom_aabb[g][None, :3] + signs * model.geom_aabb[g][None, 3:]
        pts.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g][None, :])
    pts = np.vstack(pts)
    return pts.min(axis=0), pts.max(axis=0)


def _distance_to_box(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


# --------------------------------------------------------------------------- capture


def capture_scene_graph(rs_env, labels: LabelSpace | None = None) -> tuple[SceneGraph, OracleLookup]:
    """Read the kitchen from MuJoCo and write it as a Hydra scene graph."""
    labels = labels or LabelSpace.load()
    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)

    # 1. one box per fixture and per free object, from its collision geoms
    found = []  # (oracle name, label, lo, hi, body ids)
    for name, fx in rs_env.fixtures.items():
        cls = type(fx).__name__
        if cls in SKIP_FIXTURE_CLASSES or any(p.search(name) for p in SKIP_NAME_PATTERNS):
            continue
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fx.root_body) if getattr(fx, "root_body", None) else -1
        if root < 0:
            continue
        bodies = _subtree_body_ids(model, root)
        anchor = np.asarray(fx.pos, dtype=float) if getattr(fx, "pos", None) is not None else data.xpos[root]
        box = _world_box(model, data, _collision_geoms(model, bodies, data=data, anchor=anchor))
        if box is not None:
            found.append((f"fixture:{name}", labels.label_of(class_words(cls)), *box, bodies))
    cfg_by_name = {c.get("name"): c for c in (getattr(rs_env, "object_cfgs", []) or [])}
    for name in rs_env.objects:
        bodies = _subtree_body_ids(model, rs_env.obj_body_id[name])
        box = _world_box(model, data, _collision_geoms(model, bodies))
        category = (cfg_by_name.get(name, {}).get("info") or {}).get("cat") or getattr(rs_env.objects[name], "category", "")
        if box is not None:
            found.append((f"object:{name}", labels.label_of(str(category).replace("_", " ")), *box, bodies))

    # 2. ids by position, so the numbering carries no task information; fixtures first (they never move, so a
    #    fixture keeps its id when objects are moved and the graph is captured again)
    found.sort(key=lambda f: (not labels.is_fixture_label(f[1]), *np.round((f[2] + f[3]) / 2, 2), f[1]))

    graph = dsg.DynamicSceneGraph()
    lookup = OracleLookup()
    boxes = []
    for k, (oracle_name, label, lo, hi, bodies) in enumerate(found):
        attrs = dsg.ObjectNodeAttributes()
        attrs.semantic_label = label
        attrs.position = (lo + hi) / 2
        attrs.bounding_box = dsg.BoundingBox(np.maximum(hi - lo, 2 * MINIMUM_HALF_EXTENT_M).astype(np.float32), ((lo + hi) / 2).astype(np.float32))
        color = dsg.distinct_150_color(label)  # spark_dsg's label colormap
        attrs.color = np.array([color.r, color.g, color.b], dtype=np.uint8)
        graph.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", k), attrs)
        node_id = str(dsg.NodeSymbol("O", k))
        lookup.node_of[oracle_name] = node_id
        lookup.bodies[node_id] = bodies
        boxes.append((node_id, lo, hi))

    # 3. places: free grid points over the floor, linked to their grid neighbours
    lo_xy = np.min([b[1][:2] for b in boxes], axis=0)
    hi_xy = np.max([b[2][:2] for b in boxes], axis=0)
    xs = np.arange(lo_xy[0] + PLACE_SPACING_M / 2, hi_xy[0], PLACE_SPACING_M)
    ys = np.arange(lo_xy[1] + PLACE_SPACING_M / 2, hi_xy[1], PLACE_SPACING_M)
    place_at, place_xy = {}, {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y, PLACE_HEIGHT_M])
            clearance = min(_distance_to_box(p, lo, hi) for _, lo, hi in boxes)
            if clearance < PLACE_MIN_CLEARANCE_M:
                continue
            attrs = dsg.PlaceNodeAttributes()
            attrs.position, attrs.distance = p, clearance
            k = len(place_at)
            graph.add_node(dsg.DsgLayers.PLACES, dsg.NodeSymbol("p", k), attrs)
            place_at[(i, j)] = dsg.NodeSymbol("p", k)
            place_xy[k] = p[:2]
    for (i, j), sym in place_at.items():
        for nb in ((i + 1, j), (i, j + 1)):
            if nb in place_at:
                graph.insert_edge(sym, place_at[nb])

    # 4. each object hangs off its nearest place; places belong to the one room, the room to the building
    for node_id, lo, hi in boxes:
        center = (lo[:2] + hi[:2]) / 2
        nearest = min(place_xy, key=lambda k: float(np.linalg.norm(place_xy[k] - center)))
        graph.insert_edge(dsg.NodeSymbol(*_parse_symbol(node_id)), dsg.NodeSymbol("p", nearest))
    room = dsg.RoomNodeAttributes()
    room.position = np.r_[(lo_xy + hi_xy) / 2, PLACE_HEIGHT_M]
    graph.add_node(dsg.DsgLayers.ROOMS, dsg.NodeSymbol("R", 0), room)
    building = dsg.SemanticNodeAttributes()
    building.position = room.position
    graph.add_node(dsg.DsgLayers.BUILDINGS, dsg.NodeSymbol("B", 0), building)
    for sym in place_at.values():
        graph.insert_edge(sym, dsg.NodeSymbol("R", 0))
    graph.insert_edge(dsg.NodeSymbol("R", 0), dsg.NodeSymbol("B", 0))

    # 5. the robot pose, as Hydra's agent layer
    robot = rs_env.robots[0]
    sid = rs_env.sim.model.site_name2id(robot.robot_model.base.correct_naming("center"))
    agent = dsg.AgentNodeAttributes()
    agent.position = np.array(rs_env.sim.data.site_xpos[sid])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array(rs_env.sim.data.site_xmat[sid]).reshape(9))
    agent.world_R_body = dsg.Quaternion(*quat)  # (w, x, y, z)
    graph.add_node(AGENT_LAYER, dsg.NodeSymbol("a", 0), agent, ROBOT_PARTITION)

    graph.set_labelspace(dsg.Labelspace(labels.names), dsg.DsgLayers.OBJECTS)

    # ground-truth side: what RoboCasa placed each object on, and the layout ids
    for name in rs_env.objects:
        fixture = (cfg_by_name.get(name, {}).get("placement", {}) or {}).get("fixture")
        fixture = getattr(fixture, "name", fixture)
        if f"object:{name}" in lookup.node_of and f"fixture:{fixture}" in lookup.node_of:
            lookup.support[lookup.node_of[f"object:{name}"]] = lookup.node_of[f"fixture:{fixture}"]
    for key in ("layout_id", "style_id"):
        value = getattr(rs_env, key, None)
        lookup.meta[key] = int(value) if isinstance(value, (int, np.integer)) else "custom"
    return SceneGraph(graph, labels), lookup


def perturb(graph: SceneGraph, rng: np.random.Generator, *, eps_pos_m: float = 0.0, eps_extent_m: float = 0.0, eps_yaw_rad: float = 0.0) -> SceneGraph:
    """Hook for the uncertainty study: a copy of the graph with bounded uniform noise on every object box.
    Boxes are axis-aligned (as in Hydra's AABB output), so yaw noise is not representable."""
    if eps_yaw_rad:
        raise ValueError("eps_yaw_rad: the scene graph's boxes are axis-aligned")
    noisy = graph.dsg.clone()
    for node in noisy.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        bb = node.attributes.bounding_box
        center = np.asarray(bb.world_P_center) + rng.uniform(-eps_pos_m, eps_pos_m, 3)
        dims = np.maximum(np.asarray(bb.dimensions) + 2 * rng.uniform(-eps_extent_m, eps_extent_m, 3), 2 * MINIMUM_HALF_EXTENT_M)
        node.attributes.bounding_box = dsg.BoundingBox(dims.astype(np.float32), center.astype(np.float32))
        node.attributes.position = center
    return SceneGraph(noisy, graph.labels)

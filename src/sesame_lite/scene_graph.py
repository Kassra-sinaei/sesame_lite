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

``HydraSceneGraph`` reads the graph Hydra itself built from the mapping pass (``sesame_lite.mapping``): oriented boxes
with per-face standard deviations. ``IdBridge`` is the one crossing point between its node ids and the oracle's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
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
class Face:
    """One face of a node's box as a half-space a . x <= offset (spec 4.6)."""

    normal: tuple  # a_k, unit, world: +-R[:,0], +-R[:,1], +-R[:,2]
    offset: float  # b_hat_k = max over the node's mesh vertices of a_k . p
    sigma: float  # std of the offset
    support_vertex: tuple  # the vertex that attains the max
    observed: bool  # evaluation only (true supporting point near an observed voxel); the filter never reads it


@dataclass(frozen=True)
class SceneObject:
    """One OBJECTS-layer node as the rest of the code sees it."""

    node_id: str  # Hydra symbol, "O(12)"
    label: int
    category: str  # the label's name
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    kind: str = "object"  # Hydra has no fixture/object split; kept for the classifier's request format
    rotation: tuple | None = None  # world_R_box (3x3, row-major tuples); None = axis-aligned (the oracle)
    faces: tuple[Face, ...] = ()  # Hydra nodes: the six faces with their std
    label_confidence: float | None = None  # hook for a label-confidence gate (not used yet)
    confidence: dict | None = None  # the confidence box (SceneGraph.with_confidence): center, half_extents, kappa_sigma, inflation

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(2.0 * h for h in self.half_extents)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the box the filter enforces (the confidence box once it is set)."""
        if self.rotation is None and self.confidence is None:
            c, h = np.asarray(self.center), np.asarray(self.half_extents)
            return c - h, c + h
        lo, hi = self.fragment().aabb(3)
        return np.array(lo), np.array(hi)

    def classifier_name(self) -> str:
        x, y, z = self.extent
        return f"{self.category} (approx {x:.2f} x {y:.2f} x {z:.2f} m box)"

    def fragment(self) -> OrientedBoxFragment:
        """The box the filter enforces as one barrier fragment (fragment id = node id): the confidence box once it is
        set, else the estimated box. ``center`` / ``half_extents`` stay the estimated box (what the planner shows)."""
        c, h = (self.confidence["center"], self.confidence["half_extents"]) if self.confidence else (self.center, self.half_extents)
        return OrientedBoxFragment(parent_stable_id=self.node_id, fragment_id=self.node_id, center=tuple(c), orientation_xyzw=self.orientation_xyzw, half_extents=tuple(h))

    @property
    def orientation_xyzw(self) -> tuple[float, float, float, float]:
        if self.rotation is None:
            return (0.0, 0.0, 0.0, 1.0)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(self.rotation, dtype=np.float64).reshape(9))  # (w, x, y, z)
        return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))

    def as_dict(self) -> dict:
        d = {"node_id": self.node_id, "label": self.label, "category": self.category,
             "center": [round(v, ROUND) for v in self.center], "half_extents": [round(v, ROUND) for v in self.half_extents]}
        if self.rotation is not None:
            d["rotation"] = [[round(v, 6) for v in row] for row in self.rotation]
        if self.faces:
            d["faces"] = [{"normal": [round(v, 6) for v in f.normal], "offset": round(f.offset, 5), "sigma": round(f.sigma, 6)} for f in self.faces]
        return d


class SceneGraph:
    """Reader over a Hydra DynamicSceneGraph: objects, places, rooms and the robot pose."""

    voxel_size_m = 0.0  # map resolution behind the boxes (the oracle has none: no discretisation margin m_r)
    label = "oracle"

    def __init__(self, graph: dsg.DynamicSceneGraph, labels: LabelSpace, *, robot_pose: tuple[np.ndarray, float] | None = None):
        self.dsg, self.labels = graph, labels
        self.nodes: dict[str, SceneObject] = {}
        for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
            obj = self._read_object(node)
            if obj is not None:
                self.nodes[obj.node_id] = obj
        self.places = {str(n.id): (np.asarray(n.attributes.position, dtype=float), float(n.attributes.distance)) for n in graph.get_layer(dsg.DsgLayers.PLACES).nodes}
        self.rooms = [str(n.id) for n in graph.get_layer(dsg.DsgLayers.ROOMS).nodes]
        if robot_pose is not None:  # from the simulator at capture time
            self.robot_xy, self.robot_yaw = np.asarray(robot_pose[0], dtype=float)[:2], float(robot_pose[1])
        else:  # the oracle graph stores it in the agent layer
            agent = next(iter(graph.get_layer(AGENT_LAYER, ROBOT_PARTITION).nodes))
            q = agent.attributes.world_R_body
            self.robot_xy = np.asarray(agent.attributes.position, dtype=float)[:2]
            self.robot_yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _read_object(self, node) -> SceneObject | None:
        a = node.attributes
        bb = a.bounding_box
        return SceneObject(
            node_id=str(node.id), label=int(a.semantic_label), category=self.labels.names.get(int(a.semantic_label), "unknown"),
            center=tuple(float(v) for v in bb.world_P_center), half_extents=tuple(float(max(v / 2, MINIMUM_HALF_EXTENT_M)) for v in bb.dimensions))

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

    def with_confidence(self, *, kappa: float, m_d: float, m_r_voxels: float = 0.25) -> "SceneGraph":
        """A copy whose boxes are the confidence boxes (spec 5.1): every face pushed out by kappa*sigma + m_r + m_d,
        m_r = m_r_voxels * voxel_size. A node without faces (the oracle) has sigma = 0 and m_r = 0: only m_d is added."""
        from sesame_lite.uncertainty import box_axes, box_from_offsets

        m_r = self.voxel_size_m * m_r_voxels
        out = copy.copy(self)
        out.nodes = {}
        for nid, n in self.nodes.items():
            R = np.asarray(n.rotation if n.rotation is not None else np.eye(3), dtype=float)
            A = box_axes(R)
            if n.faces:
                b_hat = np.array([f.offset for f in n.faces])
                ks = kappa * np.array([f.sigma for f in n.faces])
                mr = m_r
            else:
                c, h = np.asarray(n.center), np.asarray(n.half_extents)
                b_hat = A @ c + np.repeat(h, 2)
                ks, mr = np.zeros(6), 0.0
            inflation = ks + mr + m_d
            center, half = box_from_offsets(A, b_hat + inflation, R)
            out.nodes[nid] = replace(n, confidence={"center": tuple(float(v) for v in center), "half_extents": tuple(float(max(v, MINIMUM_HALF_EXTENT_M)) for v in half),
                                                    "kappa_sigma": tuple(float(v) for v in ks), "inflation": tuple(float(v) for v in inflation), "kappa": kappa, "m_r": mr, "m_d": m_d})
        return out

    def confidence_as_dict(self) -> dict:
        return {nid: {k: (list(v) if isinstance(v, tuple) else v) for k, v in n.confidence.items()} for nid, n in self.nodes.items() if n.confidence}

    def fragment_ids_of(self, node_ids: Iterable[str], *, exempt: Iterable[str] = ()) -> tuple[str, ...]:
        """Fragment ids (= node ids) of the given nodes, without the exempt ones."""
        exempt = set(exempt)
        return tuple(nid for nid in node_ids if nid in self.nodes and nid not in exempt)

    def aabb_by_id(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {nid: n.aabb() for nid, n in self.nodes.items()}

    def dsg_node(self, node_id: str):
        """The spark_dsg node behind an id like "O(12)" (to follow Hydra's object -> place -> room links)."""
        return self.dsg.get_node(dsg.NodeSymbol(*_parse_symbol(node_id)))

    def _parent(self, node):
        parent = node.get_parent()
        return self.dsg.get_node(parent) if parent is not None and self.dsg.has_node(parent) else None

    def room_of(self, node_id: str) -> str:
        """Hydra's hierarchy object -> place -> room; "unknown" where Hydra built no such link (rooms are never invented)."""
        place = self._parent(self.dsg_node(node_id))
        room = self._parent(place) if place is not None else None
        return str(room.id) if room is not None else "unknown"

    def place_clearance(self, node_id: str) -> float | None:
        """Free-space clearance of the object's parent place, None if it has none."""
        place = self._parent(self.dsg_node(node_id))
        return float(place.attributes.distance) if place is not None and hasattr(place.attributes, "distance") else None

    @property
    def robot_room(self) -> str:
        return self.rooms[0] if self.rooms else "unknown"

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


class HydraSceneGraph(SceneGraph):
    """Reader over the graph Hydra built in the mapping pass (outputs/maps/<mission>_s<seed>_<n>/).

    Labels are instance labels (the panoptic stand-in); ``label`` and ``category`` are the instance's category,
    through the instance -> category table of the map, so no instance or RoboCasa name reaches a model. Boxes are
    Hydra's oriented boxes; each node carries its six faces with their std (faces.json). Places and rooms are
    Hydra's own (there may be no room, and an object may have no parent place). The robot pose comes from the
    simulator at capture time: Hydra's agent layer holds the mapping camera's poses.
    """

    def __init__(self, map_dir: str | Path, labels: LabelSpace, *, robot_pose: tuple[np.ndarray, float]):
        map_dir = Path(map_dir)
        self.map_dir = map_dir
        self.meta = json.loads((map_dir / "meta.json").read_text())
        self.category_of_label = {int(k): int(v) for k, v in self.meta["category_of_label"].items()}
        self.voxel_size_m = float(self.meta["params"]["voxel_size_m"])
        self.label = "Hydra"
        self._faces = {n["node_id"]: n for n in json.loads((map_dir / "faces.json").read_text())["nodes"]}
        super().__init__(dsg.DynamicSceneGraph.load(str(map_dir / "dsg.json")), labels, robot_pose=robot_pose)

    def _read_object(self, node) -> SceneObject | None:
        node_id = str(node.id)
        record = self._faces.get(node_id)
        if record is None:  # a node without mesh vertices has no faces: nothing to protect or show
            return None
        bb = node.attributes.bounding_box
        category = self.category_of_label.get(int(node.attributes.semantic_label), 0)
        faces = tuple(Face(normal=tuple(f["normal"]), offset=float(f["offset"]), sigma=float(f["sigma"]) if f["sigma"] is not None else float("nan"),
                           support_vertex=tuple(f["support_vertex"]), observed=bool(f["observed"])) for f in record["faces"])
        R = np.asarray(bb.world_R_center, dtype=float)
        return SceneObject(
            node_id=node_id, label=category, category=self.labels.names.get(category, "unknown"),
            center=tuple(float(v) for v in bb.world_P_center), half_extents=tuple(float(max(v / 2, MINIMUM_HALF_EXTENT_M)) for v in bb.dimensions),
            rotation=tuple(tuple(float(v) for v in row) for row in R), faces=faces)

    def instance_of(self, node_id: str) -> int:
        """The node's instance label (evaluation side: the bridge to the oracle)."""
        return int(self.dsg_node(node_id).attributes.semantic_label)

    def n_vertices(self, node_id: str) -> int:
        return int(self._faces[node_id]["n_vertices"])


@dataclass
class IdBridge:
    """The one crossing point between filter-side (Hydra) ids and evaluation-side (oracle) ids.

    Built from the instance labels of the map: hydra node -> instance -> oracle node (many to one). With the oracle
    graph on the filter side it is the identity. Every conversion checks that both nodes have the same category.
    """

    oracle_of: dict[str, str]  # filter-side id -> oracle id
    hydra_of: dict[str, list[str]]  # oracle id -> filter-side ids of that instance
    category: dict[str, int]  # "H:<id>" / "O:<id>" -> category label (for the check)
    prefix: str  # "H:" when the filter side is Hydra, "O:" when it is the oracle itself

    @classmethod
    def identity(cls, oracle_graph: SceneGraph) -> "IdBridge":
        cat = {f"O:{oid}": n.label for oid, n in oracle_graph.nodes.items()}
        return cls(oracle_of={oid: oid for oid in oracle_graph.nodes}, hydra_of={oid: [oid] for oid in oracle_graph.nodes}, category=cat, prefix="O:")

    @classmethod
    def from_instances(cls, hydra_graph: HydraSceneGraph, oracle_graph: SceneGraph) -> "IdBridge":
        oracle_of_label = {int(k): v for k, v in hydra_graph.meta["oracle_of_label"].items()}
        oracle_of, hydra_of, cat = {}, {}, {f"O:{oid}": n.label for oid, n in oracle_graph.nodes.items()}
        for hydra_id, n in hydra_graph.nodes.items():
            oracle_ids = oracle_of_label.get(hydra_graph.instance_of(hydra_id), [])
            cat[f"H:{hydra_id}"] = n.label
            if len(oracle_ids) == 1:  # instance labels: exactly one oracle node per label
                oracle_of[hydra_id] = oracle_ids[0]
                hydra_of.setdefault(oracle_ids[0], []).append(hydra_id)
        for oid in hydra_of:  # largest node first (node_of picks it)
            hydra_of[oid].sort(key=lambda h: -hydra_graph.n_vertices(h))
        return cls(oracle_of=oracle_of, hydra_of=hydra_of, category=cat, prefix="H:")

    def to_oracle(self, filter_id: str) -> str | None:
        oracle_id = self.oracle_of.get(filter_id)
        if oracle_id is not None:
            assert self.category[f"{self.prefix}{filter_id}"] == self.category[f"O:{oracle_id}"], f"id mix-up: {self.prefix}{filter_id} -> O:{oracle_id}"
        return oracle_id

    def instance_nodes(self, filter_id: str) -> list[str]:
        """Every filter-side node of the same instance (a role's exemption covers all of them)."""
        oracle_id = self.to_oracle(filter_id)
        return list(self.hydra_of.get(oracle_id, [filter_id])) if oracle_id is not None else [filter_id]

    def unmapped(self, oracle_graph: SceneGraph) -> list[str]:
        """Oracle nodes the map has no node for (never seen, or too small)."""
        return [oid for oid in oracle_graph.nodes if oid not in self.hydra_of]

    def as_dict(self) -> dict:
        return {"prefix": self.prefix, "oracle_of": dict(self.oracle_of)}


def hydra_lookup(oracle_lookup: "OracleLookup", bridge: IdBridge) -> "OracleLookup":
    """OracleLookup for the Hydra graph (evaluation and execution side: resolving the hand-written plans' RoboCasa
    references). ``node_of`` picks the largest Hydra node of an instance; an instance with no Hydra node has none."""
    out = OracleLookup(meta=dict(oracle_lookup.meta))
    for name, oracle_id in oracle_lookup.node_of.items():
        if bridge.hydra_of.get(oracle_id):
            out.node_of[name] = bridge.hydra_of[oracle_id][0]
    for oracle_id, hydra_ids in bridge.hydra_of.items():
        for hydra_id in hydra_ids:
            out.bodies[hydra_id] = oracle_lookup.bodies[oracle_id]
    for obj, fixture in oracle_lookup.support.items():
        if bridge.hydra_of.get(obj) and bridge.hydra_of.get(fixture):
            for hydra_id in bridge.hydra_of[obj]:
                out.support[hydra_id] = bridge.hydra_of[fixture][0]
    return out


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


def perturb(graph: SceneGraph, rng: np.random.Generator, *, face_std_m: float, voxel_size_m: float = 0.05) -> SceneGraph:
    """Fallback of spec section 7 (synthetic face noise, not Hydra): a copy of the graph whose box faces are drawn
    from a Gaussian around the true faces, b_hat_k ~ N(b_k, face_std_m^2), with that std attached as the face sigma.
    The estimated box is the box of the drawn faces. ``voxel_size_m`` is the resolution the copy stands in for (it
    sets m_r = voxel / 4 of the confidence boxes)."""
    from sesame_lite.uncertainty import box_axes, box_from_offsets

    noisy = copy.copy(graph)
    noisy.nodes = {}
    noisy.voxel_size_m = voxel_size_m
    noisy.label = "synthetic face noise, not Hydra"
    A = box_axes(np.eye(3))
    for nid in sorted(graph.nodes, key=lambda k: _parse_symbol(k)[1]):  # fixed draw order
        n = graph.nodes[nid]
        c, h = np.asarray(n.center), np.asarray(n.half_extents)
        b_hat = A @ c + np.repeat(h, 2) + rng.normal(0.0, face_std_m, 6)
        center, half = box_from_offsets(A, b_hat, np.eye(3))
        faces = tuple(Face(normal=tuple(A[k]), offset=float(b_hat[k]), sigma=float(face_std_m), support_vertex=(float("nan"),) * 3, observed=True) for k in range(6))
        noisy.nodes[nid] = replace(n, center=tuple(float(v) for v in center), half_extents=tuple(float(max(v, MINIMUM_HALF_EXTENT_M)) for v in half), rotation=None, faces=faces)
    return noisy

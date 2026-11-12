"""Evaluation scenes (EVAL_SPEC.md, section 4) and the environment the evaluation runs in.

A scene is one (instruction, seed). ``prepare_scene`` builds it once and writes outputs/eval/scenes/<scene>/:

    scene.json   kitchen, start, the shared plan (path, standoff, roles), clutter poses, snapshot ids, rule constants
    map_plan/    full-pass map WITHOUT the six clutter objects (they are hidden from the camera like the robot):
                 the base path is planned on it once and shared by every method and map kind (Kasra, 2026-11-12)
    map_full/    full-pass map of the final scene
    map_look/    look-around map of the final scene, from the start position

Rules (fixed before any run; every number is in ``RULE``):

1. Kitchen: RoboCasa resets the seed's kitchen with the robot at the task's own base pose; that pose is the navigation
   goal, moved outward by the standoff rule (planner.standoff_goal on the planning balls of map_plan).
2. Start: the oracle places over the floor with clearance >= 0.6 m, in a seeded order; the first whose planned path to
   the standoff is >= 2.5 m long, where the robot touches nothing, and where all three floor obstacles can be placed.
   The robot faces along the first path segment.
3. Floor obstacles (flour bag, basket, boxed food), at path fractions 0.3 / 0.5 / 0.7 on alternating sides (left,
   right, left). The near edge lies ``base half-width - 0.06 m`` from the centreline, so the base on the path clips it
   by 6 cm (checked: the robot on the path touches it); a way past must exist (checked: the robot shifted 0.11 m away
   from it touches nothing). The obstacle touches nothing, lies over the floor and >= 0.6 m from start and goal.
   Fractions tried in order f, f+-0.05, f+-0.1, f+-0.15, first on its side, then on the other side.
4. Counter objects (3, categories drawn by RoboCasa from the small ``EVAL_COUNTER_CATEGORIES``): spots on a 2 cm grid at x-y
   distance 0.25-0.6 m from the target point (the point of the target's box nearest the standoff) and <= 0.9 m from
   the standoff, in a seeded order, seeded yaw; the object stands on a counter top (a Counter's or a Sink unit's top at
   >= 0.8 m, the first thing a downward ray from 1.3 m hits; centre and the four footprint
   corners hit a counter top within 1 cm of its height), touches nothing, keeps >= 8 cm (x-y) from every other free object and
   >= 12 cm from the other counter clutter.
A seed whose scene cannot be built by these rules is skipped (logged), and the next seed is taken.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from sesame_lite.env import SafetyFilterEnv
from sesame_lite.mapping import build_map, floor_test
from sesame_lite.planner import NAVIGATION, VLA, NoPathError, Segment, plan_base_path, resolve_role, standoff_goal
from sesame_lite.scene_graph import (COLLISION_GEOM_GROUP, HydraSceneGraph, IdBridge, _collision_geoms, _raw_data, _raw_model, _subtree_body_ids,
                                     capture_scene_graph, fixture_node_id, hydra_lookup)

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "outputs" / "eval"
SCENES_DIR = EVAL_DIR / "scenes"

# roles and VLA exemptions per instruction ("target": the role the counter clutter is placed around)
INSTRUCTIONS = {
    "open_drawer": dict(task="EvalOpenDrawer", roles={"drawer": {"fixture_ref": "drawer"}}, exempt=["drawer"], target="drawer"),
    "close_fridge": dict(task="EvalCloseFridge", roles={"fridge": {"fixture_ref": "fxtr"}}, exempt=["fridge"], target="fridge"),
    "close_cabinet": dict(task="EvalCloseCabinet", roles={"cabinet": {"fixture_ref": "fxtr"}}, exempt=["cabinet"], target="cabinet"),
    "sink_to_counter": dict(task="EvalPickPlaceSinkToCounter", roles={"sink": {"fixture_ref": "sink"}, "object": {"object": "obj"}, "container": {"object": "container"}},
                            exempt=["sink", "object", "container"], target="object"),
    "counter_to_cabinet": dict(task="EvalPickPlaceCounterToCabinet", roles={"cabinet": {"fixture_ref": "cab"}, "object": {"object": "obj"}},
                               exempt=["cabinet", "object"], target="object"),
    "counter_to_sink": dict(task="EvalPickPlaceCounterToSink", roles={"sink": {"fixture_ref": "sink"}, "object": {"object": "obj"}}, exempt=["sink", "object"], target="object"),
    "counter_to_stove": dict(task="EvalPickPlaceCounterToStove", roles={"stove": {"fixture_ref": "stove"}, "object": {"object": "obj"}, "container": {"object": "container"},
                                                                     "plate": {"object": "obj_container"}}, exempt=["stove", "object", "container", "plate"], target="object"),
}
ENV_KWARGS = {"split": "pretrain", "clutter_mode": 1}
HORIZON = 1200

RULE = dict(start_min_clearance_m=0.6, min_path_m=2.5, floor_fractions=(0.3, 0.5, 0.7), floor_sides=(1, -1, 1), clip_m=0.06, pass_extra_m=0.05,
            keep_out_m=0.6, fraction_offsets=(0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15), counter_ring_m=(0.25, 0.6), counter_reach_m=0.9,
            counter_gap_objects_m=0.08, counter_gap_clutter_m=0.12, counter_grid_m=0.02, counter_level_m=0.01, counter_min_height_m=0.8, counter_ray_from_z=1.3, drop_m=0.002)
BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw")
FLOOR_CLUTTER = ("clutter_floor_0", "clutter_floor_1", "clutter_floor_2")
COUNTER_CLUTTER = ("clutter_counter_0", "clutter_counter_1", "clutter_counter_2")


class ScenePrepError(RuntimeError):
    """The rules cannot build this seed's scene (the seed is skipped)."""


def scene_name(instruction: str, seed: int) -> str:
    return f"{instruction}_s{seed}"


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


# --------------------------------------------------------------------------- MuJoCo helpers


def set_base_pose(env: SafetyFilterEnv, xy, yaw: float) -> None:
    """Teleport the base through its joints (two slides along the reset frame, then the yaw joint) to (x, y, yaw)."""
    model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in BASE_JOINTS]
    adr, dof = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    data.qpos[adr[2]] += _wrap(yaw - env.base_yaw())
    mujoco.mj_forward(model, data)
    axes = data.xaxis[jid[:2]][:, :2].T  # world directions of the two slides
    data.qpos[adr[:2]] += np.linalg.solve(axes, np.asarray(xy, dtype=float) - env.robot_pose()[0][:2])
    data.qvel[dof] = 0.0
    mujoco.mj_forward(model, data)


def _free_adr(model, body: int) -> tuple[int, int]:
    j = next(j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE))
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def _corners(model, data, gids) -> np.ndarray:
    """World corners of the geoms' local boxes."""
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.vstack([(model.geom_aabb[g][None, :3] + signs * model.geom_aabb[g][None, 3:]) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g in gids])


def place_body(model, data, body: int, gids, xy, turn: float, z_bottom: float) -> np.ndarray:
    """Put a free body with its collision geoms' centre at x-y, turned by ``turn`` about world z from its current
    orientation, its lowest point ``drop_m`` above ``z_bottom``. Returns the world corners."""
    adr, dof = _free_adr(model, body)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)]), data.qpos[adr + 3:adr + 7].copy())
    data.qpos[adr + 3:adr + 7] = q
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    pts = _corners(model, data, gids)
    data.qpos[adr:adr + 2] += np.asarray(xy, dtype=float) - (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2
    data.qpos[adr + 2] += z_bottom + RULE["drop_m"] - pts[:, 2].min()
    mujoco.mj_forward(model, data)
    return _corners(model, data, gids)


def shift_body(model, data, body: int, gids, dxy) -> np.ndarray:
    adr, _ = _free_adr(model, body)
    data.qpos[adr:adr + 2] += np.asarray(dxy, dtype=float)
    mujoco.mj_forward(model, data)
    return _corners(model, data, gids)


def touching(model, data, bodies: set[int], ignore: set[int] = frozenset()) -> set[int]:
    """Bodies (outside ``bodies`` and ``ignore``) in contact (penetration or touch) with ``bodies``."""
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > 0:
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        for a, b in ((b1, b2), (b2, b1)):
            if a in bodies and b not in bodies and b not in ignore:
                out.add(b)
    return out


def _gap_xy(lo1, hi1, lo2, hi2) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lo1[:2] - hi2[:2], lo2[:2] - hi1[:2]), 0.0)))


def _fixture_bodies(rs, model, cls: str) -> set[int]:
    out = set()
    for fx in rs.fixtures.values():
        if type(fx).__name__ == cls and getattr(fx, "root_body", None):
            out |= _subtree_body_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fx.root_body))
    return out


# --------------------------------------------------------------------------- scene preparation


class _Prep:
    """State of one scene preparation (see the module docstring for the rules)."""

    def __init__(self, env: SafetyFilterEnv, instruction: str, seed: int, out_dir: Path):
        self.env, self.instruction, self.seed, self.out = env, instruction, seed, out_dir
        self.spec = INSTRUCTIONS[instruction]
        self.rs = env.rs
        self.model, self.data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.robot_bodies = _subtree_body_ids(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.rs.robots[0].robot_model.root_body))
        self.floor_bodies = _fixture_bodies(self.rs, m, "Floor")
        self.counter_bodies = _fixture_bodies(self.rs, m, "Counter") | _fixture_bodies(self.rs, m, "Sink")  # a sink unit's top is counter top
        self.floor_z = max(_corners(m, self.data, [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.floor_bodies])[:, 2])
        self.clutter = {}
        for name in FLOOR_CLUTTER + COUNTER_CLUTTER:
            body = self.rs.obj_body_id[name]
            bodies = _subtree_body_ids(m, body)
            self.clutter[name] = dict(body=body, bodies=bodies, gids=_collision_geoms(m, bodies))
        self.over_floor = floor_test(self.rs)

    def robot_touches(self) -> set[int]:
        return touching(self.model, self.data, self.robot_bodies, self.floor_bodies)

    def park(self, names) -> None:
        for name in names:  # far outside the kitchen (each its own spot), while the others are placed
            c = self.clutter[name]
            place_body(self.model, self.data, c["body"], c["gids"], self.park_xy(name), 0.0, 0.0)

    def park_xy(self, name: str) -> tuple[float, float]:
        return 60.0 + 3 * list(self.clutter).index(name), 60.0

    # ---- floor obstacles

    def place_floor(self, path: np.ndarray, start_xy, goal_xy) -> list[dict] | None:
        seg = np.diff(path, axis=0)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
        half_w = float(self.env.rect[:, 1].max() - self.env.rect[:, 1].min()) / 2
        start_pose = (self.env.robot_pose()[0][:2].copy(), self.env.base_yaw())
        placed = []
        for name, frac, side in zip(FLOOR_CLUTTER, RULE["floor_fractions"], RULE["floor_sides"]):
            info = None
            for s in (side, -side):
                for off in RULE["fraction_offsets"]:
                    info = self._try_floor(name, frac + off, s, path, seg, cum, half_w, start_xy, goal_xy)
                    if info is not None:
                        break
                if info is not None:
                    break
            set_base_pose(self.env, *start_pose)
            if info is None:
                return None
            placed.append(info)
        return placed

    def _try_floor(self, name, frac, side, path, seg, cum, half_w, start_xy, goal_xy) -> dict | None:
        m, d, c = self.model, self.data, self.clutter[name]
        s = frac * cum[-1]
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        t = seg[i] / np.linalg.norm(seg[i])
        p = path[i] + (s - cum[i]) * t
        n = np.array([-t[1], t[0]]) * side  # towards the obstacle's side
        yaw = float(np.arctan2(t[1], t[0]))
        adr, _ = _free_adr(m, c["body"])
        q0 = d.qpos[adr + 3:adr + 7].copy()
        pts = place_body(m, d, c["body"], c["gids"], p, yaw, self.floor_z)
        near = float(((pts[:, :2] - p) @ n).min())  # near edge, along n (< 0 while centred on the path)
        pts = shift_body(m, d, c["body"], c["gids"], (half_w - RULE["clip_m"] - near) * n)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = (lo[:2] + hi[:2]) / 2
        ok = (all(self.over_floor(xy) for xy in pts[:, :2]) and min(np.linalg.norm(centre - start_xy), np.linalg.norm(centre - goal_xy)) >= RULE["keep_out_m"]
              and not touching(m, d, c["bodies"]))
        clipped = passable = False
        if ok:
            set_base_pose(self.env, p, yaw)
            clipped = bool(self.robot_touches() & c["bodies"])
            set_base_pose(self.env, p - (RULE["clip_m"] + RULE["pass_extra_m"]) * n, yaw)
            passable = not self.robot_touches()
        if ok and clipped and passable:
            return {"name": name, "category": self._category(name), "path_fraction": round(frac, 3), "side": "left" if side > 0 else "right",
                    "near_edge_from_centreline_m": round(half_w - RULE["clip_m"], 4), "clipped_on_path": True, "way_past": True}
        d.qpos[adr + 3:adr + 7] = q0  # undo the turn before the next try
        self.park([name])
        return None

    # ---- counter objects

    def place_counter(self, target_xy: np.ndarray, standoff_xy: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        m, d = self.model, self.data
        r0, r1 = RULE["counter_ring_m"]
        g = RULE["counter_grid_m"]
        xs = np.arange(-r1, r1 + 1e-9, g)
        grid = np.array([(target_xy[0] + x, target_xy[1] + y) for x in xs for y in xs])
        dist = np.linalg.norm(grid - target_xy, axis=1)
        grid = grid[(dist >= r0) & (dist <= r1) & (np.linalg.norm(grid - standoff_xy, axis=1) <= RULE["counter_reach_m"])]
        grid = grid[self.rng.permutation(len(grid))]
        placed, boxes = [], []
        for name in COUNTER_CLUTTER:
            c = self.clutter[name]
            turn = float(self.rng.uniform(-np.pi, np.pi))
            far = self.park_xy(name)
            pts = place_body(m, d, c["body"], c["gids"], far, turn, 0.0)  # turned, parked: its footprint, and no ray hits it
            half = (pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)) / 2
            info = None
            for xy in grid:
                top = self._counter_top(xy)
                if top is None:
                    continue
                lo, hi = np.r_[xy - half, 0.0], np.r_[xy + half, 0.0]
                corners = [(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])]
                if any((h := self._counter_top(np.array(xy_c))) is None or abs(h[1] - top[1]) > RULE["counter_level_m"] for xy_c in corners):
                    continue
                if any(_gap_xy(lo, hi, a, b) < RULE["counter_gap_objects_m"] for a, b in others) or any(_gap_xy(lo, hi, a, b) < RULE["counter_gap_clutter_m"] for a, b in boxes):
                    continue
                place_body(m, d, c["body"], c["gids"], xy, 0.0, top[1])
                if touching(m, d, c["bodies"]):
                    place_body(m, d, c["body"], c["gids"], far, 0.0, 0.0)
                    continue
                info = {"name": name, "category": self._category(name), "xy": [round(float(v), 4) for v in xy], "turn": round(turn, 4),
                        "distance_to_target_m": round(float(np.linalg.norm(xy - target_xy)), 4), "distance_to_standoff_m": round(float(np.linalg.norm(xy - standoff_xy)), 4)}
                boxes.append((lo, hi))
                break
            if info is None:
                raise ScenePrepError(f"no counter spot for {name} (placed {len(placed)} of 3)")
            placed.append(info)
        return placed

    def _counter_top(self, xy) -> tuple[int, float] | None:
        """(geom, height) under x-y: the first collision geom a downward ray hits, if it belongs to a counter."""
        group = np.zeros(6, np.uint8)
        group[COLLISION_GEOM_GROUP] = 1
        hit = np.array([-1], np.int32)
        dist = mujoco.mj_ray(self.model, self.data, np.array([xy[0], xy[1], RULE["counter_ray_from_z"]]), np.array([0.0, 0.0, -1.0]), group, 1, -1, hit)
        if hit[0] < 0 or int(self.model.geom_bodyid[hit[0]]) not in self.counter_bodies or RULE["counter_ray_from_z"] - float(dist) < RULE["counter_min_height_m"]:
            return None
        return int(hit[0]), RULE["counter_ray_from_z"] - float(dist)

    def _category(self, name: str) -> str:
        cfg = next((c for c in self.rs.object_cfgs if c.get("name") == name), {})
        return (cfg.get("info") or {}).get("cat") or getattr(self.rs.objects[name], "category", "")


def _load(cache: Path, oracle_graph, oracle_lookup, robot_pose, policy):
    graph = HydraSceneGraph(cache, oracle_graph.labels, robot_pose=robot_pose)
    bridge = IdBridge.from_instances(graph, oracle_graph)
    lookup = hydra_lookup(oracle_lookup, bridge)
    balls = graph.balls(n_admitted=len(graph.nodes), delta=policy.delta, m_d=policy.m_d)
    return graph, bridge, lookup, balls


def prepare_scene(env: SafetyFilterEnv, instruction: str, seed: int, out_dir: Path, exclude: set[tuple[int, int]] = frozenset()) -> dict:
    """Build the scene of (instruction, seed) by the module's rules and write ``out_dir`` (raises ScenePrepError;
    a kitchen (layout, style) in ``exclude`` is not built)."""
    spec = INSTRUCTIONS[instruction]
    env.env.reset(seed=seed)
    rs = env.rs
    meta = rs.get_ep_meta()
    kitchen = (int(meta["layout_id"]), int(meta["style_id"]))
    if kitchen in exclude:
        raise ScenePrepError(f"kitchen {kitchen} is excluded")
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = _Prep(env, instruction, seed, out_dir)
    m, d = prep.model, prep.data
    goal_xy, goal_yaw = env.robot_pose()[0][:2].copy(), env.base_yaw()  # RoboCasa's base pose for the task
    env._base_geometry()

    # 1. the map without the clutter, and the planning balls on it
    oracle0, lookup0 = capture_scene_graph(rs)
    hidden = set().union(*(c["bodies"] for c in prep.clutter.values()))
    build_map(rs, oracle0, lookup0, out_dir / "map_plan", seed=seed, hide_bodies=hidden)
    bp, R, _ = env.robot_pose()
    graph, bridge, lookup, balls = _load(out_dir / "map_plan", oracle0, lookup0, (bp, float(np.arctan2(R[1, 0], R[0, 0]))), env.policy)
    nav_oracle_name = fixture_node_id(rs, rs.init_robot_base_ref)
    nav_node = lookup.node(nav_oracle_name)
    if nav_node is None:
        raise ScenePrepError(f"the navigation target {nav_oracle_name} is not in map_plan")
    standoff_xy, offset = standoff_goal(balls, env.rect, goal_xy, goal_yaw, np.asarray(graph.nodes[nav_node].center[:2]), z_band=env.z_band, clearance_m=env.policy.goal_standoff_m)

    # 2. start, path and floor obstacles
    prep.park(list(prep.clutter))
    places = [p for p, clearance in oracle0.places.values() if clearance >= RULE["start_min_clearance_m"] and prep.over_floor(p[:2])]
    order = prep.rng.permutation(len(places))
    tried, chosen = 0, None
    for k in order:
        start = np.asarray(places[k][:2], dtype=float)
        tried += 1
        try:
            path = plan_base_path(graph, balls, start, standoff_xy, inflation_m=env.inflation_m, z_band=env.z_band)
        except NoPathError:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        if length < RULE["min_path_m"]:
            continue
        yaw = float(np.arctan2(*(path[1] - path[0])[::-1]))
        set_base_pose(env, start, yaw)
        if prep.robot_touches():
            continue
        floor = prep.place_floor(path, start, standoff_xy)
        if floor is None:
            continue
        chosen = dict(start=start, yaw=yaw, path=path, length=length, floor=floor)
        break
    if chosen is None:
        raise ScenePrepError(f"no start place satisfies the start and floor-obstacle rules ({tried} tried)")

    # 3. counter objects around the target
    target_node = lookup0.node_of.get(_oracle_name(rs, spec["roles"][spec["target"]]))
    lo, hi = oracle0.nodes[target_node].aabb()
    target_xy = np.clip(standoff_xy, lo[:2], hi[:2])  # the point of the target's box nearest the standoff
    task_objects = [n for n in rs.objects if n not in prep.clutter]
    others = []
    for name in task_objects:
        pts = _corners(m, d, _collision_geoms(m, _subtree_body_ids(m, rs.obj_body_id[name])))
        others.append((pts.min(axis=0), pts.max(axis=0)))
    counter = prep.place_counter(target_xy, standoff_xy, others)
    mujoco.mj_forward(m, d)
    if prep.robot_touches():
        raise ScenePrepError("the robot touches something at the start of the final scene")

    # 4. the final scene, its two maps and the record
    oracle, lookup_f = capture_scene_graph(rs)
    build_map(rs, oracle, lookup_f, out_dir / "map_full", seed=seed)
    build_map(rs, oracle, lookup_f, out_dir / "map_look", seed=seed, look_from=chosen["start"])
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in BASE_JOINTS]
    scene = {
        "scene": scene_name(instruction, seed), "instruction": instruction, "task": spec["task"], "seed": seed, "env_kwargs": ENV_KWARGS, "horizon": HORIZON,
        "language": env.env.task_description, "kitchen": {"layout": kitchen[0], "style": kitchen[1]},
        "task_pose": {"xy": goal_xy.round(4).tolist(), "yaw": round(goal_yaw, 5)},
        "standoff": {"xy": np.round(standoff_xy, 4).tolist(), "yaw": round(goal_yaw, 5), "offset_m": offset},
        "start": {"xy": chosen["start"].round(4).tolist(), "yaw": round(chosen["yaw"], 5), "places_tried": tried, "path_length_m": round(chosen["length"], 3)},
        "plan": {"roles": spec["roles"], "nav_target": nav_oracle_name, "path_xy": np.round(chosen["path"], 4).tolist(), "exempt": spec["exempt"], "target": spec["target"]},
        "base_qpos": [float(d.qpos[m.jnt_qposadr[j]]) for j in jid],
        "clutter_qpos": {name: d.qpos[_free_adr(m, c["body"])[0]:_free_adr(m, c["body"])[0] + 7].tolist() for name, c in prep.clutter.items()},
        "floor_obstacles": chosen["floor"], "counter_objects": counter, "target_point_xy": np.round(target_xy, 4).tolist(),
        "snapshot_id": oracle.snapshot_id, "snapshot_id_map_plan": oracle0.snapshot_id, "rule": RULE,
    }
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


def _oracle_name(rs, role: dict) -> str | None:
    if "fixture_ref" in role:
        return fixture_node_id(rs, role["fixture_ref"])
    return f"object:{role['object']}" if "object" in role else None


def apply_scene(env: SafetyFilterEnv, scene: dict) -> None:
    """After ``env.env.reset(seed)``: the stored base joints and clutter poses (velocities zero)."""
    model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
    for j, v in zip(BASE_JOINTS, scene["base_qpos"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        data.qpos[model.jnt_qposadr[jid]], data.qvel[model.jnt_dofadr[jid]] = v, 0.0
    for name, q in scene["clutter_qpos"].items():
        adr, dof = _free_adr(model, env.rs.obj_body_id[name])
        data.qpos[adr:adr + 7], data.qvel[dof:dof + 6] = q, 0.0
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------- the evaluation environment


class EvalFilterEnv(SafetyFilterEnv):
    """SafetyFilterEnv on a prepared scene: the shared plan (never replanned) and one of its two maps.

    Exemptions come from the plan's roles: on the filter side the role's map nodes (a role the map has no node for
    is logged as an unmapped task object), on the evaluation side the role's oracle node (always)."""

    def __init__(self, env, *, map_kind: str = "full", **kwargs):
        super().__init__(env, max_replans=0, **kwargs)
        if map_kind not in ("full", "look"):
            raise ValueError(map_kind)
        self.map_kind = map_kind

    def reset(self, seed=None, *, scene_dir: Path, **kwargs):
        scene = json.loads((Path(scene_dir) / "scene.json").read_text())
        obs, info = self.env.reset(seed=scene["seed"])
        apply_scene(self, scene)
        self.scene, self.seed, self.mission, self.n_capture = scene, scene["seed"], scene["instruction"], 0
        self.oracle_graph, self.oracle_lookup = capture_scene_graph(self.rs)
        cache = Path(scene_dir) / f"map_{self.map_kind}"
        for sid in (self.oracle_graph.snapshot_id, json.loads((cache / "meta.json").read_text())["oracle_snapshot_id"]):
            if sid != scene["snapshot_id"]:
                raise RuntimeError(f"scene {scene['scene']}: snapshot {sid} != {scene['snapshot_id']}")
        self._load_map(cache)
        self._base_geometry()
        self.plan = scene["plan"]
        self.plan_kwargs = dict(inflation_m=self.inflation_m)
        self.segments = self._eval_segments()
        return self._finish_reset(obs, info)

    def _eval_segments(self) -> list[Segment]:
        plan, sc = self.plan, self.scene
        self.unmapped_roles, filter_exempt, self.exempt_by_segment = [], [], {0: set(), 1: set()}
        for role in plan["exempt"]:
            spec = plan["roles"][role]
            node = resolve_role(spec, self.rs, self.lookup)
            if node is None:
                self.unmapped_roles.append(role)
            else:
                filter_exempt += self.bridge.instance_nodes(node)
            oracle_node = resolve_role(spec, self.rs, self.oracle_lookup)
            if oracle_node is not None:
                self.exempt_by_segment[1].add(oracle_node)
        lang = self.env.task_description
        return [Segment(kind=NAVIGATION, goal_text=lang, path_polyline_xy=np.asarray(plan["path_xy"]), goal_yaw=sc["standoff"]["yaw"], goal_offset_m=sc["standoff"]["offset_m"] or 0.0,
                        done={"check": "base_near_goal", "radius": 0.2}, index=0),
                Segment(kind=VLA, goal_text=lang, center_xy=np.asarray(sc["standoff"]["xy"]), exempt_ids=tuple(dict.fromkeys(filter_exempt)), done={"check": "task_success"}, index=1)]

    def _exempt_oracle_ids(self, seg) -> set[str]:
        return set(self.exempt_by_segment[seg.index])

    def _record(self, rec, action, q, bar) -> None:
        super()._record(rec, action, q, bar)
        g = self.ground_truth[-1]
        g["touched_not_admitted"] = [o for o in g["touched"] if o[2:] not in self.row_oracle]

    def summary(self, success: bool, stalled: bool) -> dict:
        out = super().summary(success, stalled)
        gt, dist = self.ground_truth, out["disturbance"]
        obj = resolve_role({"object": "obj"}, self.rs, self.oracle_lookup) if "obj" in self.rs.objects else None
        out.update(
            scene=self.scene["scene"], instruction=self.scene["instruction"], map_kind=self.map_kind, horizon=self.env._max_episode_steps,
            collision_free=out["contact_ticks"] == 0, safe_success=bool(success and out["contact_ticks"] == 0),
            contacts_not_admitted=sorted({o for g in gt for o in g["touched_not_admitted"]}),
            relaxed_share=out["feasibility"]["relaxed_ticks"] / max(out["steps"], 1),
            tracking_error_max=(dist.get("norm") or {}).get("max"), unmapped_task_objects=self.unmapped_roles,
            support_of_object=f"O:{self.oracle_lookup.support[obj]}" if obj in self.oracle_lookup.support else None,  # for the open support-exemption choice
            snapshot_ok=self.oracle_graph.snapshot_id == self.scene["snapshot_id"])
        return out

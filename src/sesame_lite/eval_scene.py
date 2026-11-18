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
   right, left). The near edge lies ``base half-width - 0.03 m`` from the centreline, so the base on the path clips it
   by 3 cm (checked: the robot on the path touches it); a way past must exist (checked: the robot shifted 0.15 m away
   from it touches nothing, so at the ~9 cm the filter needs it keeps ~6 cm on both sides). (Round 1 of the tuning,
   with 6 cm and 0.11 m, deadlocked every filtered run at an obstacle; Kasra, 2026-11-13.) The obstacle touches nothing, lies over the floor and >= 0.6 m from start and goal.
   Fractions tried in order f, f+-0.05, f+-0.1, f+-0.15, first on its side, then on the other side.
4. Counter objects (3, categories drawn by RoboCasa from the small ``EVAL_COUNTER_CATEGORIES``): spots on a 2 cm grid at x-y
   distance 0.25-0.6 m from the target point (the point of the target's box nearest the standoff) and <= 0.9 m from
   the standoff, in a seeded order, seeded yaw; the object stands on a counter top (a Counter's or a Sink unit's top at
   >= 0.8 m, the first thing a downward ray from 1.3 m hits; centre and the four footprint
   corners hit a counter top within 1 cm of its height), touches nothing, keeps >= 8 cm (x-y) from every other free object and
   >= 12 cm from the other counter clutter.
   As many of the three as fit are placed (fridges and stoves have little or no counter top within reach; Kasra,
   2026-11-13 and 2026-11-15); the others are stored 5 m under the floor (gravity compensated, out of every view).
5. Path: the clutter is placed along the path planned on map_plan (rules 2-3); the shared path of the scene is then
   planned once on map_full, which contains the clutter (Kasra, 2026-11-13), with the two-layer planner (base; rigid
   column and shoulder), and is never replanned.
A seed whose scene cannot be built by these rules is skipped (logged), and the next seed is taken.
"""

from __future__ import annotations

import json
from dataclasses import replace
import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np

from sesame_lite.env import SafetyFilterEnv
from sesame_lite.mapping import MappingParams, build_map, floor_test
from sesame_lite.planner import NAVIGATION, VLA, NoPathError, Segment, plan_base_path, resolve_role, standoff_goal
from sesame_lite.scene_graph import (COLLISION_GEOM_GROUP, HydraSceneGraph, IdBridge, _collision_geoms, _raw_data, _raw_model, _subtree_body_ids,
                                     capture_scene_graph, fixture_node_id, hydra_lookup)

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "outputs" / "eval"  # only output the paper can reference (Kasra, 2026-11-13)
DEBUG_DIR = REPO / "outputs" / "eval_debug"  # logs, raw tuning runs and other working files
SCENES_DIR = EVAL_DIR / "scenes"

# roles and VLA exemptions per instruction ("target": the role the counter clutter is placed around)
INSTRUCTIONS = {
    "open_drawer": dict(task="EvalOpenDrawer", roles={"drawer": {"fixture_ref": "drawer"}}, exempt=["drawer"], target="drawer"),
    "close_fridge": dict(task="EvalCloseFridge", roles={"fridge": {"fixture_ref": "fxtr"}}, exempt=["fridge"], target="fridge"),
    "close_cabinet": dict(task="EvalCloseCabinet", roles={"cabinet": {"fixture_ref": "fxtr"}}, exempt=["cabinet"], target="cabinet"),
    "sink_to_counter": dict(task="EvalPickPlaceSinkToCounter", roles={"sink": {"fixture_ref": "sink"}, "object": {"object": "obj"}, "container": {"object": "container"}},
                            exempt=["sink", "object", "container"], target="object", patch="object"),
    "counter_to_cabinet": dict(task="EvalPickPlaceCounterToCabinet", roles={"cabinet": {"fixture_ref": "cab"}, "object": {"object": "obj"}},
                               exempt=["cabinet", "object"], target="object", patch="object"),
    "counter_to_sink": dict(task="EvalPickPlaceCounterToSink", roles={"sink": {"fixture_ref": "sink"}, "object": {"object": "obj"}}, exempt=["sink", "object"], target="object", patch="object"),
    "open_cabinet": dict(task="EvalOpenCabinet", roles={"cabinet": {"fixture_ref": "fxtr"}}, exempt=["cabinet"], target="cabinet"),
    "slide_dishwasher_rack": dict(task="EvalSlideDishwasherRack", roles={"dishwasher": {"fixture_ref": "dishwasher"}}, exempt=["dishwasher"], target="dishwasher"),
    "stove_to_counter": dict(task="EvalPickPlaceStoveToCounter", roles={"stove": {"fixture_ref": "stove"}, "object": {"object": "obj"}, "pan": {"object": "obj_container"},
                                                                     "container": {"object": "container"}}, exempt=["stove", "object", "pan", "container"], target="object", patch="object"),
    "counter_to_stove": dict(task="EvalPickPlaceCounterToStove", roles={"stove": {"fixture_ref": "stove"}, "object": {"object": "obj"}, "container": {"object": "container"},
                                                                     "plate": {"object": "obj_container"}}, exempt=["stove", "object", "container", "plate"], target="object", patch="object"),
}
ENV_KWARGS = {"split": "pretrain", "clutter_mode": 1}
HORIZON = 1800  # ticks, every method (Kasra, 2026-11-16; the finished filtered tuning runs took 588-1582 ticks)

RULE = dict(start_min_clearance_m=0.6, min_path_m=2.5, floor_fractions=(0.3, 0.5, 0.7), floor_sides=(1, -1, 1), clip_m=0.03, pass_extra_m=0.12,
            keep_out_m=0.6, fraction_offsets=(0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15), counter_ring_m=(0.25, 0.6), counter_reach_m=0.9,
            counter_gap_objects_m=0.08, counter_gap_clutter_m=0.12, counter_grid_m=0.02, counter_level_m=0.01, counter_min_height_m=0.8, counter_ray_from_z=1.3, drop_m=0.002,
            store_below_floor_m=5.0)
# the support of a picked object is not exempt; only the finger spheres against its points within this x-y radius of
# the object (filter side: the object's map node centre; truth: the object's position when manipulation starts), during
# the manipulation segment (Kasra, 2026-11-13); every method is scored the same way
FINGER_PATCH_RADIUS_M = 0.15
# the nominal base follower's look-ahead along the path (pure pursuit; same for every method); stage 0b round 3 sets it
FOLLOWER_LOOKAHEAD_M = 0.2
FINGER_LINKS = ("gripper0_right_leftfinger", "gripper0_right_rightfinger")
# pair-level pruning (Kasra, 2026-11-13, round 5): robot groups, and the groups whose pairs with the task's objects are
# dropped from the QP during manipulation (whole object); the base and the arm (links 0-6) keep every pair
ROBOT_GROUPS = ("base", "arm", "hand", "fingers")
PAIR_DROP_GROUPS = ("hand", "fingers")


def robot_group(name: str) -> str:
    """The robot group of a body or sphere link: base (mobilebase0_*), arm (robot0_link0-6), fingers (finger bodies),
    hand (the hand, the gripper's base and the wrist flange, link 7, which sits on top of the hand; Kasra, 2026-11-15)."""
    if name.startswith("mobilebase0"):
        return "base"
    if "finger" in name:
        return "fingers"
    if name == "robot0_link7":
        return "hand"
    if name.startswith("robot0_link") or name == "robot0_base":
        return "arm"
    return "hand"
BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw")
FLOOR_CLUTTER = ("clutter_floor_0", "clutter_floor_1", "clutter_floor_2")
COUNTER_CLUTTER = ("clutter_counter_0", "clutter_counter_1", "clutter_counter_2")
OBSTACLE_CANDIDATES = ("obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3")  # tasks.EVAL_OBSTACLE_CATEGORIES


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

    def place_counter(self, target_xy: np.ndarray, standoff_xy: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict], list[str]]:
        m, d = self.model, self.data
        r0, r1 = RULE["counter_ring_m"]
        g = RULE["counter_grid_m"]
        xs = np.arange(-r1, r1 + 1e-9, g)
        grid = np.array([(target_xy[0] + x, target_xy[1] + y) for x in xs for y in xs])
        dist = np.linalg.norm(grid - target_xy, axis=1)
        grid = grid[(dist >= r0) & (dist <= r1) & (np.linalg.norm(grid - standoff_xy, axis=1) <= RULE["counter_reach_m"])]
        grid = grid[self.rng.permutation(len(grid))]
        placed, boxes, unplaced = [], [], []
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
            if info is None:  # no spot for this one: it is left out of the scene (stored under the floor by the caller)
                unplaced.append(name)
                continue
            placed.append(info)
        return placed, unplaced

    def counter_within_reach(self, standoff_xy) -> bool:
        """Is any counter top (as ``_counter_top``) within the reach radius of the standoff (2 cm grid)?"""
        r, g = RULE["counter_reach_m"], RULE["counter_grid_m"]
        xs = np.arange(-r, r + 1e-9, g)
        return any(self._counter_top(standoff_xy + np.array([x, y])) is not None for x in xs for y in xs if x * x + y * y <= r * r)

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
        cfg = next((c for c in list(self.rs.object_cfgs) + list(getattr(self.rs, "obstacle_cfgs", [])) if c.get("name") == name), {})
        return (cfg.get("info") or {}).get("cat") or getattr(self.rs.objects[name], "category", "")


def _load(cache: Path, oracle_graph, oracle_lookup, robot_pose, policy):
    graph = HydraSceneGraph(cache, oracle_graph.labels, robot_pose=robot_pose)
    bridge = IdBridge.from_instances(graph, oracle_graph)
    lookup = hydra_lookup(oracle_lookup, bridge)
    balls = graph.balls(n_admitted=len(graph.nodes), delta=policy.delta, m_d=policy.m_d)
    return graph, bridge, lookup, balls


def prepare_scene(env: SafetyFilterEnv, instruction: str, seed: int, out_dir: Path, exclude: set[tuple[int, int]] = frozenset(), params: MappingParams | None = None) -> dict:
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
    mp = {} if params is None else {"params": params}  # the maps' parameters (default: MappingParams())
    build_map(rs, oracle0, lookup0, out_dir / "map_plan", seed=seed, hide_bodies=hidden, **mp)
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
            path = plan_base_path(graph, balls, start, standoff_xy, inflation_m=env.inflation_m, z_band=env.z_band, upper=env.upper)
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
    if target_node is None:
        raise ScenePrepError(f"the target ({spec['target']}) has no node in the scene graph")
    lo, hi = oracle0.nodes[target_node].aabb()
    target_xy = np.clip(standoff_xy, lo[:2], hi[:2])  # the point of the target's box nearest the standoff
    task_objects = [n for n in rs.objects if n not in prep.clutter]
    others = []
    for name in task_objects:
        geoms = _collision_geoms(m, _subtree_body_ids(m, rs.obj_body_id[name]))
        if not len(geoms):
            raise ScenePrepError(f"the task object {name} has no collision geometry")
        pts = _corners(m, d, geoms)
        others.append((pts.min(axis=0), pts.max(axis=0)))
    # as many of the three counter objects as fit by the rule (Kasra, 2026-11-13 and 2026-11-15: fridges and stoves have
    # little or no counter top within reach); the others are stored under the floor
    counter, stored = prep.place_counter(target_xy, standoff_xy, others) if prep.counter_within_reach(standoff_xy) else ([], list(COUNTER_CLUTTER))
    for k, name in enumerate(stored):
        c = prep.clutter[name]
        place_body(m, d, c["body"], c["gids"], standoff_xy + np.array([0.5 * k, 0.0]), 0.0, prep.floor_z - RULE["store_below_floor_m"])
    hold_still(m, stored_bodies(rs, stored))
    mujoco.mj_forward(m, d)
    if prep.robot_touches():
        raise ScenePrepError("the robot touches something at the start of the final scene")

    # 4. the final scene, its two maps and the record
    oracle, lookup_f = capture_scene_graph(rs)
    build_map(rs, oracle, lookup_f, out_dir / "map_full", seed=seed, **mp)
    build_map(rs, oracle, lookup_f, out_dir / "map_look", seed=seed, look_from=chosen["start"], **mp)
    # the shared path: planned once on the full-pass map that contains the clutter (Kasra, 2026-11-13), two layers
    bp, R, _ = env.robot_pose()
    graph_f, _, _, balls_f = _load(out_dir / "map_full", oracle, lookup_f, (bp, float(np.arctan2(R[1, 0], R[0, 0]))), env.policy)
    try:
        path = plan_base_path(graph_f, balls_f, chosen["start"], standoff_xy, inflation_m=env.inflation_m, z_band=env.z_band, upper=env.upper)
    except NoPathError as exc:
        raise ScenePrepError(f"no path on the map with the clutter: {exc}") from exc
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in BASE_JOINTS]
    scene = {
        "scene": scene_name(instruction, seed), "instruction": instruction, "task": spec["task"], "seed": seed, "env_kwargs": ENV_KWARGS, "horizon": HORIZON,
        "language": env.env.task_description, "kitchen": {"layout": kitchen[0], "style": kitchen[1]},
        "task_pose": {"xy": goal_xy.round(4).tolist(), "yaw": round(goal_yaw, 5)},
        "standoff": {"xy": np.round(standoff_xy, 4).tolist(), "yaw": round(goal_yaw, 5), "offset_m": offset},
        "start": {"xy": chosen["start"].round(4).tolist(), "yaw": round(chosen["yaw"], 5), "places_tried": tried,
                  "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 3), "path_length_without_clutter_m": round(chosen["length"], 3)},
        "plan": {"roles": spec["roles"], "nav_target": nav_oracle_name, "path_xy": np.round(path, 4).tolist(), "path_xy_without_clutter": np.round(chosen["path"], 4).tolist(),
                 "exempt": spec["exempt"], "target": spec["target"],
                 "patch": spec.get("patch")},
        "base_qpos": [float(d.qpos[m.jnt_qposadr[j]]) for j in jid],
        "clutter_qpos": {name: d.qpos[_free_adr(m, c["body"])[0]:_free_adr(m, c["body"])[0] + 7].tolist() for name, c in prep.clutter.items()},
        "floor_obstacles": chosen["floor"], "counter_objects": counter, "stored_below_floor": stored, "target_point_xy": np.round(target_xy, 4).tolist(),
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
    hold_still(model, stored_bodies(env.rs, scene.get("stored_below_floor", [])))
    mujoco.mj_forward(model, data)


def stored_bodies(rs, names) -> set[int]:
    model = _raw_model(rs.sim)
    return set().union(set(), *(_subtree_body_ids(model, rs.obj_body_id[n]) for n in names))


def hold_still(model, bodies) -> None:
    """Objects stored under the floor: gravity compensated, so they stay where they are (nothing is near them)."""
    for b in bodies:
        model.body_gravcomp[b] = 1.0


# --------------------------------------------------------------------------- the evaluation environment


class SharedDecisions:
    """The language model's decisions of one scene and map kind (object labels per segment, dropped pairs), asked once
    and stored in the scene folder (model_<map>.json), so that every method of the scene runs with the same ones."""

    def __init__(self, path: Path, classifier=None):
        self.path, self.classifier = path, classifier

    def load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def store(self, key: str, value: dict) -> None:
        data = self.load() | {key: value}
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)

    def must_ask(self, key: str) -> None:
        """A store marked complete (scripts/eval_decide.py asked everything before the campaign) must answer every
        request: a miss then means the run differs from the decided scene, and it stops instead of asking again."""
        if self.load().get("complete"):
            raise RuntimeError(f"{self.path}: no stored decision for {key}, although the store is complete")

    def classify(self, *, candidates, task_goal: str, parent_graph_snapshot_id: str):
        """The wrapped object classifier, answered from the store when this request was asked before."""
        from sesame_lite.agents.pruning_classifier import PruningClassification, request_payload

        key = "labels:" + hashlib.sha256(request_payload(tuple(candidates), task_goal).encode()).hexdigest()[:16]
        data = self.load()
        hit = data.get(key)
        if hit is None and data.get("complete"):
            # another method of the scene with slightly other ball radii (no ISSf margin) offers a slightly other list:
            # its objects take the labels of the stored request that contains them all (the smallest such); an object no
            # stored request contains stays an obstacle (the model can only remove constraints)
            ids = [c.stable_id for c in candidates]
            stored = [v for k, v in data.items() if k.startswith("labels:")]
            cover = sorted((v for v in stored if all(i in v["labels"] for i in ids)), key=lambda v: len(v["labels"]))
            src = cover[0]["labels"] if cover else {i: next((v["labels"][i] for v in stored if i in v["labels"]), "obstacle") for i in ids}
            hit = {"labels": {i: src[i] for i in ids}, "classifier_version": stored[0]["classifier_version"] if stored else "none", "fallback_used": not cover, "latency_s": None,
                   "detail": "labels taken from the scene's stored requests"}
        if hit is None:
            res = self.classifier.classify(candidates=candidates, task_goal=task_goal, parent_graph_snapshot_id=parent_graph_snapshot_id)
            hit = {"labels": dict(res.labels), "classifier_version": res.classifier_version, "fallback_used": res.fallback_used, "latency_s": res.latency_s, "detail": res.detail}
            self.store(key, hit)
        return PruningClassification(labels=hit["labels"], parent_graph_snapshot_id=parent_graph_snapshot_id, classifier_version=hit["classifier_version"],
                                     fallback_used=hit["fallback_used"], latency_s=hit["latency_s"], detail=hit["detail"])


class EvalFilterEnv(SafetyFilterEnv):
    """SafetyFilterEnv on a prepared scene: the shared plan (never replanned) and one of its two maps.

    Exemptions come from the plan's roles: on the filter side the role's map nodes (a role the map has no node for
    is logged as an unmapped task object), on the evaluation side the role's oracle node (always)."""

    def __init__(self, env, *, map_kind: str = "full", pair_pruning: str | None = None, **kwargs):
        """``pair_pruning``: "rule" or "model" (see ``_pairs_setup``): pairs of robot groups and objects are dropped
        instead of exempting whole objects and instead of the finger patch."""
        super().__init__(env, max_replans=0, **kwargs)
        if map_kind not in ("full", "look"):
            raise ValueError(map_kind)
        self.map_kind, self.pair_pruning = map_kind, pair_pruning

    def reset(self, seed=None, *, scene_dir: Path, **kwargs):
        scene = json.loads((Path(scene_dir) / "scene.json").read_text())
        obs, info = self.env.reset(seed=scene["seed"])
        apply_scene(self, scene)
        self.scene, self.seed, self.mission, self.n_capture = scene, scene["seed"], scene["instruction"], 0
        if (self.pair_pruning == "model" or self.classifier_kind == "qwen") and not hasattr(self, "shared"):
            self.shared = SharedDecisions(Path(scene_dir) / f"model_{self.map_kind}.json", self._classifier)
            if self._classifier is not None:  # (filter off, Nominal: no object labels are needed)
                self._classifier = self.shared
        self.oracle_graph, self.oracle_lookup = capture_scene_graph(self.rs)
        ob = scene.get("safety_obstacle") or {}
        self.obstacle_oracle = self.oracle_lookup.node_of.get(f"object:{ob['name']}") if ob.get("placed") else None
        cache = Path(scene_dir) / f"map_{self.map_kind}"
        for sid in (self.oracle_graph.snapshot_id, json.loads((cache / "meta.json").read_text())["oracle_snapshot_id"]):
            if sid != scene["snapshot_id"]:
                raise RuntimeError(f"scene {scene['scene']}: snapshot {sid} != {scene['snapshot_id']}")
        self._load_map(cache)
        self._base_geometry()
        self.plan = scene["plan"]
        self.plan_kwargs = dict(inflation_m=self.inflation_m)
        self.segments = self._eval_segments()
        self._patch_setup()
        self._pairs_setup()
        return self._finish_reset(obs, info)

    def _eval_segments(self) -> list[Segment]:
        plan, sc = self.plan, self.scene
        self.unmapped_roles, filter_exempt, self.exempt_by_segment = [], [], {0: set(), 1: set()}
        for role in plan["exempt"]:
            spec = plan["roles"][role]
            node = resolve_role(spec, self.rs, self.lookup)
            if node is None:
                self.unmapped_roles.append(role)
            elif not self.pair_pruning:  # pair-level pruning: no object is exempt as a whole
                filter_exempt += self.bridge.instance_nodes(node)
            oracle_node = resolve_role(spec, self.rs, self.oracle_lookup)
            if oracle_node is not None and not self.pair_pruning:
                self.exempt_by_segment[1].add(oracle_node)
        lang = self.env.task_description
        return [Segment(kind=NAVIGATION, goal_text=lang, path_polyline_xy=np.asarray(plan["path_xy"]), goal_yaw=sc["standoff"]["yaw"], goal_offset_m=sc["standoff"]["offset_m"] or 0.0,
                        done={"check": "base_near_goal", "radius": 0.2}, index=0),
                Segment(kind=VLA, goal_text=lang, center_xy=np.asarray(sc["standoff"]["xy"]), exempt_ids=tuple(dict.fromkeys(filter_exempt)), done={"check": "task_success"}, index=1)]

    def _exempt_oracle_ids(self, seg) -> set[str]:
        return set(self.exempt_by_segment[seg.index])

    # ---- the finger patch on the picked object's support (FINGER_PATCH_RADIUS_M)

    def _patch_setup(self) -> None:
        """The support (RoboCasa's placement surface of the picked object) unless it is exempt anyway (the sink)."""
        self.patch, self.patch_target, self.patch_contact = None, None, False
        role = self.plan.get("patch")
        if not role or self.pair_pruning:
            return
        spec = self.plan["roles"][role]
        oracle_obj = resolve_role(spec, self.rs, self.oracle_lookup)
        oracle_sup = self.oracle_lookup.support.get(oracle_obj)
        if oracle_sup is None or oracle_sup in self.exempt_by_segment[1]:
            return
        map_obj = resolve_role(spec, self.rs, self.lookup)
        map_sup = self.lookup.support.get(map_obj) if map_obj is not None else None
        self.patch = {"oracle_support": oracle_sup, "object_body": self.rs.obj_body_id[spec["object"]], "map_object": map_obj,
                      "map_support": self.bridge.instance_nodes(map_sup) if map_sup is not None else [], "radius_m": FINGER_PATCH_RADIUS_M}
        model = _raw_model(self.rs.sim)
        self.finger_bodies = set().union(*(_subtree_body_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)) for b in FINGER_LINKS))

    # ---- pair-level pruning (round 5)

    def _pairs_setup(self) -> None:
        """Pairs dropped from the QP during manipulation (robot group, object), for the whole object. ``"rule"``: a fixed
        rule, the hand and the fingers may touch the task's objects (the plan's roles: target, container, the sink of a
        sink task) and what each of these objects stands on (A8). ``"model"``: Qwen chooses them once per manipulation
        step from the instruction, that step's object list and the four groups (``_ask_model``). Every other pair stays."""
        self.pair_groups, self.pair_map_groups, self.dropped_pairs, self.pair_reply = {}, {}, [], None
        # task objects of the plan that the map has no node for: the filter has no constraint against them and the model
        # cannot name them, so a gripper contact with one is scored as a contact on an exempt pair (Kasra, 2026-11-15)
        self.unmapped_task_oracle = set()
        if self.pair_pruning:
            for role in self.plan["exempt"]:
                oid = resolve_role(self.plan["roles"][role], self.rs, self.oracle_lookup)
                if oid is not None and not self.bridge.hydra_of.get(oid):
                    self.unmapped_task_oracle.add(oid)
        if self.pair_pruning != "rule":
            return
        objects = [(role, self.plan["roles"][role]) for role in self.plan["exempt"]]
        for role in self.plan["exempt"] if self.plan.get("patch") else ():  # what the plan's objects stand on (RoboCasa's placement surfaces)
            obj = resolve_role(self.plan["roles"][role], self.rs, self.oracle_lookup)
            if obj in self.oracle_lookup.support:  # e.g. the counter under the plate that holds the picked object (A8)
                objects.append(("support", {"oracle": self.oracle_lookup.support[obj]}))
        for role, spec in objects:
            oid = spec.get("oracle") or resolve_role(spec, self.rs, self.oracle_lookup)
            if oid is not None and oid not in self.pair_groups:
                for g in PAIR_DROP_GROUPS:
                    self._drop_pair(g, oid, self.bridge.hydra_of.get(oid, []), role)

    def _drop_pair(self, group: str, oracle_id: str | None, nodes, role: str) -> None:
        names = {v: k for k, v in self.oracle_lookup.node_of.items()}
        if oracle_id is not None:
            self.pair_groups.setdefault(oracle_id, set()).add(group)
        for nid in nodes:
            self.pair_map_groups.setdefault(nid, set()).add(group)
        self.dropped_pairs.append({"segment": 1, "group": group, "role": role, "object": f"O:{oracle_id}" if oracle_id else None,
                                   "name": names.get(oracle_id, "").split(":", 1)[-1], "map_nodes": list(nodes),
                                   "category": self.graph.nodes[nodes[0]].category if nodes else (self.oracle_graph.nodes[oracle_id].category if oracle_id else None)})

    def _ask_model(self) -> None:
        """Once per manipulation step: the model's pairs over the step's candidate objects (stage 1 of the pruning)."""
        from sesame_lite.agents.pruning_classifier import PruningCandidate, QwenPairClassifier

        cands = [PruningCandidate(stable_id=nid, name=self.graph.nodes[nid].classifier_name(), kind=self.graph.nodes[nid].kind) for nid in self.bundle.candidate_ids]
        # where each object is, from the map, relative to the plan's standoff pose (the same for every method of the scene)
        xy, yaw = np.asarray(self.scene["standoff"]["xy"]), self.scene["standoff"]["yaw"]
        where = {}
        for c in cands:
            lo, hi = self.graph.nodes[c.stable_id].aabb()
            d = np.clip(xy, lo[:2], hi[:2]) - xy  # to the nearest point of the object's box
            centre = (lo[:2] + hi[:2]) / 2 - xy
            bearing = np.degrees(_wrap(np.arctan2(centre[1], centre[0]) - yaw))
            direction = "ahead" if abs(bearing) <= 30 else "behind" if abs(bearing) >= 150 else ("left" if bearing > 0 else "right") if abs(bearing) >= 60 else ("ahead-left" if bearing > 0 else "ahead-right")
            where[c.stable_id] = {"distance_m": round(float(np.linalg.norm(d)), 1), "direction": direction, "height_m": round(float((lo[2] + hi[2]) / 2), 1)}
        reply = self.shared.load().get("pairs")
        if reply is None:  # asked once per scene and map kind; every method of the scene uses this reply
            self.shared.must_ask("pairs")
            res = QwenPairClassifier().classify_pairs(candidates=cands, task_goal=self.segment.goal_text, where=where)
            reply = {"latency_s": round(res.latency_s, 3), "error": res.error, "raw_reply": res.raw_reply, "n_objects": len(cands),
                     "pairs_returned": [list(p) for p in res.pairs], "pairs_not_applied": [list(p) for p in res.pairs if p[0] not in PAIR_DROP_GROUPS]}
            self.shared.store("pairs", reply)
        # the model chooses the objects; the robot part is fixed (Kasra, 2026-11-15): an object the model names in a pair
        # with the hand or the fingers loses its pairs with the whole gripper (fingers, hand, wrist flange). Pairs the
        # model returns with the base or the arm are not applied (they keep every pair) and are logged.
        self.pair_reply = reply
        chosen = list(dict.fromkeys(nid for group, nid in reply["pairs_returned"] if group in PAIR_DROP_GROUPS and nid in self.graph.nodes))
        for nid in chosen:
            for group in PAIR_DROP_GROUPS:
                self._drop_pair(group, self.bridge.to_oracle(nid), self.bridge.instance_nodes(nid), "model")

    def _pairs_on(self) -> bool:
        return bool(self.pair_pruning) and self.segment.kind == VLA

    def dropped_sphere_mask(self) -> np.ndarray:
        """Spheres with at least one pair dropped by the pair pruning now (for the video)."""
        groups = set().union(set(), *self.pair_map_groups.values()) if self._pairs_on() else set()
        return np.array([robot_group(link) in groups for link in self.robot.links])

    def _patch_on(self) -> bool:
        return self.patch is not None and self.patch_target is not None and self.segment.kind == VLA

    def begin_segment(self, index: int) -> None:
        if getattr(self, "patch", None) is not None and self.segments[index].kind == VLA:
            self.patch_target = _raw_data(self.rs.sim).xpos[self.patch["object_body"]][:2].copy()  # truth: where manipulation starts
        super().begin_segment(index)

    def _filter_objects(self, seg):
        """The support's points outside the patch become a separate object with rows for the finger spheres only; the
        full support keeps the rows of every other sphere."""
        balls, masks = super()._filter_objects(seg)
        if self.pair_pruning:
            if seg.kind != VLA:
                return balls, masks
            if self.pair_pruning == "model" and self.pair_reply is None:
                self._ask_model()
            groups = [robot_group(link) for link in self.robot.links]
            return balls, {nid: np.array([g not in gs for g in groups]) for nid, gs in self.pair_map_groups.items() if nid in balls}
        p = self.patch
        if seg.kind != VLA or p is None or p["map_object"] is None:
            return balls, masks
        balls, masks = dict(balls), {}
        finger = np.array([link in FINGER_LINKS for link in self.robot.links])
        centre = np.asarray(self.graph.nodes[p["map_object"]].center[:2])
        for nid in p["map_support"]:
            if nid not in balls:
                continue
            b = balls[nid]
            out = np.linalg.norm(b.points[:, :2] - centre, axis=1) > p["radius_m"]
            masks[nid] = ~finger
            if out.any():
                balls[f"{nid}~fingers"] = replace(b, points=b.points[out], rho=b.rho[out], capped=b.capped[out])
                masks[f"{nid}~fingers"] = finger
        return balls, masks

    def _touched_oracle_ids(self) -> set[str]:
        """Every robot-object contact, logged with its robot group (``self.tick_contacts``) and whether it is on an
        exempt pair (``self.tick_exempt_contacts``): an exempt object (patch method: the plan's roles), a finger contact
        with the support inside the patch, or a pair dropped by the pair pruning. Returns the objects of the contacts
        that are not exempt."""
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        if not hasattr(self, "body_group"):
            self.body_group = {b: robot_group(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in self.robot_bodies}
        vla = self.segment.kind == VLA
        out, contacts, exempt, points = set(), set(), set(), []
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if a in self.robot_bodies and b in self.oracle_id_of_body:
                    oid, grp = self.oracle_id_of_body[b], self.body_group[a]
                    contacts.add((grp, f"O:{oid}"))
                    if vla and self.pair_pruning and (grp in self.pair_groups.get(oid, ()) or (oid in self.unmapped_task_oracle and grp in PAIR_DROP_GROUPS)):
                        exempt.add((grp, f"O:{oid}"))
                        continue
                    if vla and oid in self.exempt_oracle_ids:
                        exempt.add((grp, f"O:{oid}"))
                        continue
                    if self._patch_on() and oid == self.patch["oracle_support"] and a in self.finger_bodies and np.linalg.norm(c.pos[:2] - self.patch_target) <= self.patch["radius_m"]:
                        self.patch_contact = True
                        exempt.add((grp, f"O:{oid}"))
                        continue
                    out.add(oid)
                    points.append([grp, f"O:{oid}", *np.round(c.pos, 4).tolist()])
        self.tick_contacts, self.tick_exempt_contacts, self.tick_contact_points = contacts, exempt, points
        return out

    def _truth(self, centres: np.ndarray, radii: np.ndarray) -> dict:
        """As the base class, but finger spheres over the patch are left out of the clearance to the support; with pair
        pruning, the hand and finger spheres are left out of the clearance to the objects of their dropped pairs."""
        out = super()._truth(centres, radii)
        if self._pairs_on() and self.pair_groups:
            data, groups = _raw_data(self.rs.sim), [robot_group(link) for link in self.robot.links]
            for key, owners in (("clearance", self.eval_set), ("clearance_rows", self.row_oracle)):
                pair = owners & set(self.pair_groups)
                if not pair:
                    continue
                parts = [self.true_geometry.clearance(data, centres, radii, owners - pair)]
                for gs in {frozenset(self.pair_groups[o]) for o in pair}:  # objects with the same dropped groups together
                    keep = np.flatnonzero([g not in gs for g in groups])
                    v, x = self.true_geometry.clearance(data, centres[keep], radii[keep], {o for o in pair if frozenset(self.pair_groups[o]) == gs})
                    parts.append((v, None if x is None else [int(keep[x[0]]), x[1]]))
                best, pr = min(((v, x) for v, x in parts if v is not None), default=(None, None), key=lambda x: x[0])
                out[key] = best
                if key == "clearance":
                    out["pair"] = None if pr is None else [self.robot.links[pr[0]], f"O:{pr[1]}"]
            return out
        if not self._patch_on():
            return out
        finger = np.array([link in FINGER_LINKS for link in self.robot.links])
        over = finger & (np.linalg.norm(centres[:, :2] - self.patch_target, axis=1) <= self.patch["radius_m"])
        sup = self.patch["oracle_support"]
        if not over.any():
            return out
        data, keep = _raw_data(self.rs.sim), np.flatnonzero(~over)
        for key, owners in (("clearance", self.eval_set), ("clearance_rows", self.row_oracle)):
            if sup not in owners:
                continue
            a, pa = self.true_geometry.clearance(data, centres, radii, owners - {sup})
            b, pb = self.true_geometry.clearance(data, centres[keep], radii[keep], {sup})
            pb = None if pb is None else [int(keep[pb[0]]), pb[1]]
            best, pair = min(((v, pr) for v, pr in ((a, pa), (b, pb)) if v is not None), default=(None, None), key=lambda x: x[0])
            out[key] = best
            if key == "clearance":
                out["pair"] = None if pair is None else [self.robot.links[pair[0]], f"O:{pair[1]}"]
        return out

    def _record(self, rec, action, q, bar) -> None:
        self.patch_contact = False
        super()._record(rec, action, q, bar)
        g = self.ground_truth[-1]
        g["touched_not_admitted"] = [o for o in g["touched"] if o[2:] not in self.row_oracle]
        g["finger_patch_contact"] = self.patch_contact
        g["contacts"] = sorted(f"{grp}|{o}" for grp, o in self.tick_contacts)  # every contact, exempt pairs included
        g["exempt_pair_contacts"] = sorted(f"{grp}|{o}" for grp, o in self.tick_exempt_contacts)
        g["contact_points"] = self.tick_contact_points  # counted contacts: robot group, object, world position
        if getattr(self, "obstacle_oracle", None) is not None:  # true clearance of the robot's spheres to the safety obstacle
            g["clearance_obstacle"] = self.true_geometry.clearance(_raw_data(self.rs.sim), bar.centres, bar.radii, {self.obstacle_oracle})[0]

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
            support_of_object=f"O:{self.oracle_lookup.support[obj]}" if obj in self.oracle_lookup.support else None,
            finger_patch={k: v for k, v in (self.patch or {}).items() if k != "object_body"} or None,
            finger_patch_contact_ticks=sum(bool(g.get("finger_patch_contact")) for g in gt),
            pair_pruning=self.pair_pruning, dropped_pairs=self.dropped_pairs, pair_model_reply=self.pair_reply,
            contact_ticks_all=sum(bool(g["contacts"]) for g in gt), exempt_pair_contact_ticks=sum(bool(g["exempt_pair_contacts"]) for g in gt),
            contacts_by_pair={k: sum(k in g["contacts"] for g in gt) for k in sorted({k for g in gt for k in g["contacts"]})},
            exempt_contacts_by_pair={k: sum(k in g["exempt_pair_contacts"] for g in gt) for k in sorted({k for g in gt for k in g["exempt_pair_contacts"]})},
            snapshot_ok=self.oracle_graph.snapshot_id == self.scene["snapshot_id"])
        ob = getattr(self, "obstacle_oracle", None)
        out["safety_obstacle"] = None if ob is None else {
            **self.scene["safety_obstacle"], "oracle_id": f"O:{ob}", "contact_ticks": sum(any(c.endswith(f"|O:{ob}") for c in g["contacts"]) for g in gt),
            "clearance_min_m": min((g["clearance_obstacle"] for g in gt if g.get("clearance_obstacle") is not None), default=None)}
        return out

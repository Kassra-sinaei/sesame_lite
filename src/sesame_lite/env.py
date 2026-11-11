"""SESAME-3D on RoboCasa: lerobot's ``RoboCasaEnv`` subclassed for pinned, non-resetting episodes, a gym wrapper
that runs the safety filter of spec v4 on Hydra's map, and the nominal policies.

What runs is set by two switches (defaults: both on, the SESAME-3D method). Every run of a seed shares the scene
(pinned episode metadata), the nominal policy, the barrier definition and the ground-truth evaluation set:

  pruning   on: rows only for the objects admitted by the swept-volume stage and the semantic classifier
            off: TODO (spec v4, section 11): out of scope for this version
  filter    on: the QP of eq. (9) sets the base and arm joint velocities
            off: the rows are evaluated, the nominal joint velocities drive the robot (no classifier call)

Every tick: u_nom (base: the command's joint velocities; arm: robosuite's OSC response, joint_drive.py), the filter
(filter.py), then the base command for the filtered base velocities and the arm drive's joint velocity command.
Ground truth (evaluation only): every robot sphere against the collision geometry of the non-exempt objects
(truth.py), and contacts; the oracle boxes are kept next to it for comparison only.
``MissionRunner`` sequences the plan's segments. One process per episode (scripts/run_episode.py).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
from lerobot.envs.robocasa import RoboCasaEnv, convert_action

from sesame_lite.agents.pruning_classifier import (
    CategoryPriorClassifier,
    ClaudeCodeClassifier,
    ConservativeFallbackClassifier,
    QwenClassifier,
    StickyPassabilityClassifier,
)
from sesame_lite.filter import (
    BASE_SLICE,
    CONTROL_DT_S,
    EE_POS_SLICE,
    EE_ROT_SLICE,
    MODE_INDEX,
    TORSO_INDEX,
    BaseModel,
    FilterPolicy,
    SphereFilter,
)
from sesame_lite.geometry.swept_volume import BETA_DEVIATION_BUDGET_M
from sesame_lite.joint_drive import JointVelocityDrive, pd_mean_velocity
from sesame_lite.planner import NAVIGATION, VLA, NoPathError, build_bundle, default_plan, resolve_plan
from sesame_lite.robot_spheres import RobotSpheres
from sesame_lite.truth import CollisionGeometry
from sesame_lite.determinism import SAMPLER_SEED, begin_episode, seed_fallback_generators
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _collision_geoms, _raw_data, _raw_model, _subtree_body_ids, capture_scene_graph, hydra_lookup
import sesame_lite.render_fix  # noqa: F401,E402  (stops freed render contexts from corrupting camera frames)

# The checkpoint's own camera assignment (policy_preprocessor.json); lerobot's docs swap camera2/camera3.
RENAME_MAP = {
    "observation.images.robot0_agentview_left": "observation.images.camera1",
    "observation.images.robot0_agentview_right": "observation.images.camera2",
    "observation.images.robot0_eye_in_hand": "observation.images.camera3",
}
POLICY_PATH = "lerobot/smolvla_robocasa"
CLASSIFIERS = {
    "none": lambda: None,
    "prior": CategoryPriorClassifier,
    "claude": lambda: StickyPassabilityClassifier(ConservativeFallbackClassifier(ClaudeCodeClassifier())),
    "qwen": lambda: StickyPassabilityClassifier(ConservativeFallbackClassifier(QwenClassifier())),
}
STALL_HOLD_TICKS = 50
STALL_PROGRESS_TICKS = 100
STALL_SPEED_MPS = 0.02
MAX_REPLANS = 3


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------- lerobot env subclass




class SesameRoboCasaEnv(RoboCasaEnv):
    """lerobot's RoboCasaEnv with kitchen kwargs forwarded, seeded scene generation and no auto-reset on success.

    With the placement samplers seeded and the robot's initialization noise off, the kitchen generated for a
    seed is identical in every process; ``summarize.py`` checks this through the graph snapshot id.
    """

    def __init__(self, task: str, *, env_kwargs: dict | None = None, **kwargs):
        env_kwargs = {"initialization_noise": None, **(env_kwargs or {})}  # no random joint noise: every run of a seed starts identically
        for key in ("split", "obj_registries"):  # lerobot RoboCasaEnv arguments, not Kitchen kwargs
            if key in env_kwargs:
                kwargs[key] = env_kwargs.pop(key)
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel"))  # both packs are installed
        super().__init__(task, **kwargs)
        self.env_kwargs = env_kwargs
        seed_fallback_generators()

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

        # create_env overrides layout_ids/style_ids whenever a split is given, so explicit scene ids need split=None;
        # the robosuite rng is seeded at construction (create_env(seed=...)) because the wrapper's first, implicit
        # reset would otherwise draw an unseeded scene and leave per-task references behind.
        explicit_scene = any(k in self.env_kwargs for k in ("layout_ids", "style_ids", "layout_and_style_ids"))
        split = None if explicit_scene else (self.split if self.split is not None else "all")
        self._env = RoboCasaGymEnv(
            env_name=self.task,
            camera_widths=self.observation_width,
            camera_heights=self.observation_height,
            split=split,
            obj_registries=self.obj_registries,
            seed=self._scene_seed,
            **self.env_kwargs,
        )
        self.task_description = self._env.env.get_ep_meta().get("lang", self.task)

    @property
    def rs(self):
        return self._env.env  # the robosuite Kitchen

    def reset(self, seed=None, **kwargs):
        self._scene_seed = int(seed or 0)
        begin_episode(self._scene_seed)
        self._ensure_env()
        obs, info = super().reset(seed=seed)  # hard reset: the kitchen is rebuilt from the seed, identically every time
        return obs, info

    def step(self, action: np.ndarray):
        self._ensure_env()
        raw_obs, reward, done, truncated, info = self._env.step(convert_action(np.asarray(action, dtype=np.float32)))
        self.last_raw_obs = raw_obs  # kept so the video shows exactly the frames the policy received
        is_success = bool(info.get("success", False))
        info.update({"task": self.task, "done": done, "is_success": is_success})
        return self._format_raw_obs(raw_obs), reward, bool(done or is_success), truncated, info


# ---------------------------------------------------------------------- safety filter wrapper


class UnmappedRoleError(RuntimeError):
    """A plan role names an object the Hydra map has no node for: the mission stops (nothing is filled in from the oracle)."""


class SafetyFilterEnv(gym.Wrapper):
    """Semantic safety filter in front of ``SesameRoboCasaEnv`` on the Hydra map; segments come from a plan.

    Obstacles are the objects' confidence balls (PDF eq. 4). Planning, the standoff goals and pruning use the balls
    with delta split over every node of the map (``plan_balls``); the rows of a segment use delta split over that
    segment's admitted objects (``row_balls``).
    """

    def __init__(self, env: SesameRoboCasaEnv, *, pruning: bool = True, filter: bool = True, classifier: str = "prior", policy: FilterPolicy = FilterPolicy()):
        super().__init__(env)
        if not pruning:
            raise NotImplementedError("TODO (spec v4, section 11): runs with --no-pruning are out of scope for this version")
        self.pruning, self.filter_on, self.classifier_kind, self.policy = pruning, filter, classifier, policy
        self.tag = run_tag(pruning, filter)
        self._classifier = CLASSIFIERS[classifier]() if (pruning and filter) else None  # the classifier only runs when it can change something

    # ------------------------------------------------------------------ robot geometry

    @property
    def rs(self):
        return self.env.rs

    def _base_box(self):
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mobilebase0_pedestal_feet_col")
        return data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]

    def robot_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base position, base rotation and end-effector position read from MuJoCo data (robosuite's observation
        buffer is only refreshed on steps, so it is stale right after a reset or a state restore)."""
        robot, data = self.rs.robots[0], self.rs.sim.data
        sid = self.rs.sim.model.site_name2id(robot.robot_model.base.correct_naming("center"))
        return np.array(data.site_xpos[sid]), np.array(data.site_xmat[sid]).reshape(3, 3), np.array(data.site_xpos[robot.eef_site_id["right"]])

    def base_yaw(self) -> float:
        R = self.robot_pose()[1]
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _touched_oracle_ids(self) -> set[str]:
        """Oracle nodes the robot touches now (MuJoCo contacts map to oracle ids only)."""
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        out = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if b1 in self.robot_bodies and b2 in self.oracle_id_of_body:
                out.add(self.oracle_id_of_body[b2])
            elif b2 in self.robot_bodies and b1 in self.oracle_id_of_body:
                out.add(self.oracle_id_of_body[b1])
        return out

    # ------------------------------------------------------------------ episode set-up

    def reset(self, seed=None, *, plan: dict | None = None, **kwargs):
        obs, info = self.env.reset(seed=seed)
        rs, model = self.rs, _raw_model(self.rs.sim)
        self.seed, self.mission = int(seed or 0), (plan or {}).get("mission", self.env.task)
        self.n_capture = 0
        self._capture()
        bp, R, _ = self.robot_pose()
        pg, Rg, size = self._base_box()
        signs = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)
        local = ((pg[None, :2] + (signs * size[:2]) @ Rg[:2, :2].T) - bp[None, :2]) @ R[:2, :2]
        lo, hi = local.min(axis=0), local.max(axis=0)
        self.rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])  # base footprint (planning)
        self.z_band = (float(pg[2] - size[2]), float(pg[2] + size[2]))
        self.base_radius = float(np.max(np.linalg.norm(self.rect, axis=1)))
        self.plan = plan or default_plan(rs)
        self.plan_kwargs = dict(base_xy=bp[:2], rect_xy=self.rect, z_band=self.z_band, clearance_m=self.policy.goal_standoff_m, inflation_m=float(hi[1] - lo[1]) / 2 + 0.10)
        self.segments = self._resolve()
        if self.plan.get("injections") and self.segments[0].kind == NAVIGATION:
            # floor objects are dropped next to the planned path, then the map is built again (the path planner
            # only knows the first map, as a navigation stack's static map would; the objects enter through the 3DSG)
            from sesame_lite.tasks import place_on_path

            place_on_path(rs, self.segments[0].path_polyline_xy)
            first = self.segments
            self._capture()  # objects are renumbered by position: resolve the roles again on the new graph
            self.segments = self._resolve()
            for old, new in zip(first, self.segments):  # but keep the path the objects were placed along
                new.path_polyline_xy, new.goal_yaw, new.goal_offset_m, new.center_xy = old.path_polyline_xy, old.goal_yaw, old.goal_offset_m, old.center_xy
        # contact bookkeeping (MuJoCo body -> oracle node), from the oracle lookup
        self.oracle_id_of_body = {b: oracle_id for oracle_id, bodies in self.oracle_lookup.bodies.items() for b in bodies}
        self.robot_bodies = _subtree_body_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rs.robots[0].robot_model.root_body))
        self.true_geometry = CollisionGeometry(model, {oid: _collision_geoms(model, bodies) for oid, bodies in self.oracle_lookup.bodies.items()})
        robot = rs.robots[0]
        self.robot = RobotSpheres(model, _raw_data(rs.sim))
        self.drive = JointVelocityDrive.install(robot)
        self.base_ctrl, self.torso_ctrl = robot.composite_controller.part_controllers["base"], robot.composite_controller.part_controllers["torso"]
        self.filter = SphereFilter(self.robot, self.policy)
        self.base_model = BaseModel()
        arm_j = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        self.arm_q_lo, self.arm_q_hi = model.jnt_range[arm_j, 0].copy(), model.jnt_range[arm_j, 1].copy()
        self.receipts, self.ground_truth, self.bundles, self.ball_log, self.q_log, self.sphere_h_log = [], [], [], [], [], []
        self.tick, self.epoch, self.hold_streak, self.stop_reason = 0, 0, 0, None
        self.replans, self.segment_index = 0, -1
        self.begin_segment(0)
        self.start = self._start_check()
        return obs, info

    def _capture(self) -> None:
        """Hydra's graph from the mapping pass (filter side) and the oracle graph (evaluation side).

        The map is cached in outputs/maps/<mission>_s<seed>_<n>/ (n = 0 at reset, 1 after the injections) and built
        here if the cache is missing; the cache must belong to this exact scene.
        """
        from sesame_lite.mapping import build_map, map_dir

        self.oracle_graph, self.oracle_lookup = capture_scene_graph(self.rs)
        cache = map_dir(self.mission, self.seed, self.n_capture)
        if not (cache / "points.npz").exists():
            build_map(self.rs, self.oracle_graph, self.oracle_lookup, cache, seed=self.seed)
        cached_id = json.loads((cache / "meta.json").read_text())["oracle_snapshot_id"]
        if cached_id != self.oracle_graph.snapshot_id:
            raise RuntimeError(f"map cache {cache} belongs to another scene ({cached_id} != {self.oracle_graph.snapshot_id})")
        bp, R, _ = self.robot_pose()
        self.graph = HydraSceneGraph(cache, self.oracle_graph.labels, robot_pose=(bp, float(np.arctan2(R[1, 0], R[0, 0]))))
        self.bridge = IdBridge.from_instances(self.graph, self.oracle_graph)
        self.lookup = hydra_lookup(self.oracle_lookup, self.bridge)
        self.n_capture += 1
        # planning, standoff and pruning: kappa for every node admitted (Kasra, 2026-11-10), so these balls are never
        # smaller than the ones the rows use once a segment's admitted count is known
        self.plan_balls = self.graph.balls(n_admitted=len(self.graph.nodes), delta=self.policy.delta, m_d=self.policy.m_d)

    def _resolve(self):
        """Plan roles -> filter-side nodes; a role's exemption covers every node of its instance. A role that resolves
        to no node of the map (the target was never mapped) stops the mission: it is never filled in from the oracle."""
        from sesame_lite.planner import resolve_role

        missing = [name for name, spec in self.plan.get("roles", {}).items() if resolve_role(spec, self.rs, self.lookup) is None]
        if missing:
            raise UnmappedRoleError(f"role(s) {missing} resolve to no node of the Hydra map {self.graph.map_dir}")
        segments = resolve_plan(self.plan, self.rs, self.graph, self.lookup, balls=self.plan_balls, language=self.env.task_description, **self.plan_kwargs)
        for seg in segments:
            seg.exempt_ids = tuple(dict.fromkeys(h for hydra_id in seg.exempt_ids for h in self.bridge.instance_nodes(hydra_id)))
        return segments

    def begin_segment(self, index: int) -> None:
        """Prune for one segment (an epoch) and fix its rows: the admitted, non-exempt objects and their balls."""
        if index != self.segment_index:
            self.replans = 0
        self.segment_index = index
        seg = self.segments[index]
        self.bundle = build_bundle(seg, self.graph, balls=self.plan_balls, classifier=self._classifier, base_circumscribed_radius_m=self.base_radius, policy_margin_m=0.0, eef_reach_m=1.0 * CONTROL_DT_S)
        self.bundles.append(self.bundle)
        self.row_ids = self.graph.fragment_ids_of(self.bundle.admitted_ids, exempt=seg.exempt_ids)
        # delta split over the objects that get rows (eq. 4: delta_j = delta / |A|)
        balls = self.graph.balls(n_admitted=len(self.row_ids), delta=self.policy.delta, m_d=self.policy.m_d)
        self.row_balls = {nid: balls[nid] for nid in self.row_ids}
        # ground truth: every non-exempt node of the oracle graph, the same for every switch setting
        self.exempt_oracle_ids = {self.bridge.to_oracle(hydra_id) for hydra_id in seg.exempt_ids} - {None}
        self.eval_ids = self.oracle_graph.fragment_ids_of([n.node_id for n in self.oracle_graph.obstacle_nodes()], exempt=self.exempt_oracle_ids)
        boxes = [self.oracle_graph.nodes[o].aabb() for o in self.eval_ids]  # oracle boxes are axis-aligned
        self.eval_lo, self.eval_hi = np.array([lo for lo, _ in boxes]).reshape(-1, 3), np.array([hi for _, hi in boxes]).reshape(-1, 3)
        row_oracle = {self.bridge.to_oracle(nid) for nid in self.row_ids} - {None}
        self.eval_set, self.row_oracle = set(self.eval_ids), row_oracle
        self.filter.set_objects(self.row_balls)
        ks = np.concatenate([b.kappa * np.minimum(self.graph.nodes[nid].std, b.s_max) for nid, b in self.row_balls.items()] or [np.zeros(0)])
        self.ball_log.append({"segment": index, "epoch": self.epoch + 1, "n_admitted": len(self.row_ids), "m_r": self.graph.m_r, "m_d": self.policy.m_d,
                              "kappa": {nid: b.kappa for nid, b in self.row_balls.items()}, "s_max": {nid: b.s_max for nid, b in self.row_balls.items()},
                              "capped": {nid: int(b.capped.sum()) for nid, b in self.row_balls.items()}, "points": {nid: len(b.points) for nid, b in self.row_balls.items()},
                              "kappa_s_bar_quantiles_m": dict(zip(["min", "50%", "90%", "max"], np.percentile(ks, [0, 50, 90, 100]).round(5).tolist())) if len(ks) else None})
        self.epoch += 1
        self.segment_start_tick = self.tick

    @property
    def segment(self):
        return self.segments[self.segment_index]

    @property
    def unmapped_oracle_ids(self) -> list[str]:
        return self.bridge.unmapped(self.oracle_graph)

    # ------------------------------------------------------------------ robot state

    def base_theta(self) -> float:
        """theta of robosuite's base controller: base yaw now minus at episode start (turns the command onto the slides)."""
        import robosuite.utils.transform_utils as T

        _, ori = self.base_ctrl.get_base_pose()
        return float(T.mat2euler(ori)[2] - T.mat2euler(self.base_ctrl.init_ori)[2])

    def _start_check(self) -> dict:
        """The start pose against the balls planning uses (kappa for every node admitted; spec v4, section 9), and the
        truth. Nothing is done about it: the QP acts from the first tick and the receipts show what happens."""
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        mujoco.mj_forward(model, data)
        check = SphereFilter(self.robot, self.policy)
        check.set_objects({nid: b for nid, b in self.plan_balls.items() if nid not in self.segments[0].exempt_ids})
        bar = check.barrier(self.robot.q(data))
        i, j = np.unravel_index(int(np.argmin(bar.h)), bar.h.shape)
        truth = self._truth(bar.centres, bar.radii)
        return {"h_min": float(bar.h[i, j]), "pair": [int(i), self.robot.links[i], check.object_ids[j]], "pairs_below_zero": int((bar.h < 0).sum()),
                "spheres_below_zero": sorted({self.robot.links[k] for k in np.flatnonzero(bar.h.min(axis=1) < 0)}),
                "objects_below_zero": sorted({check.object_ids[k] for k in np.flatnonzero(bar.h.min(axis=0) < 0)}), "true_clearance": truth["clearance"], "true_pair": truth["pair"]}

    def _truth(self, centres: np.ndarray, radii: np.ndarray) -> dict:
        """True clearance (spec 6.4): the smallest signed distance of every robot sphere to the collision geometry of
        the non-exempt objects (truth.py), and over the rows' objects only (to check the barrier). Next to it, for
        comparison only, the same against the oracle boxes (one axis-aligned box per object), the obstacle model before
        spec v4. None: nothing within truth.HORIZON_M."""
        data = _raw_data(self.rs.sim)
        clear, pair = self.true_geometry.clearance(data, centres, radii, self.eval_set)
        clear_rows, _ = self.true_geometry.clearance(data, centres, radii, self.row_oracle)
        out = {"clearance": clear, "pair": None if pair is None else [self.robot.links[pair[0]], f"O:{pair[1]}"], "clearance_rows": clear_rows,
               "box_clearance": None, "box_pair": None}
        if len(self.eval_lo):
            c = centres[:, None, :]
            outside = np.maximum(np.maximum(self.eval_lo[None] - c, c - self.eval_hi[None]), 0.0)
            dist = np.linalg.norm(outside, axis=2)
            depth = np.min(np.minimum(c - self.eval_lo[None], self.eval_hi[None] - c), axis=2)
            sd = np.where(dist > 0, dist, -depth) - radii[:, None]
            i, j = np.unravel_index(int(np.argmin(sd)), sd.shape)
            out.update(box_clearance=float(sd[i, j]), box_pair=[self.robot.links[i], f"O:{self.eval_ids[j]}"])
        return out

    # ------------------------------------------------------------------ gym API

    def step(self, action):
        a = np.asarray(action, dtype=float).copy()
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        mujoco.mj_forward(model, data)  # derived quantities of the current state (step1 of the next step recomputes the same)
        q, qd, theta = self.robot.q(data), self.robot.qdot(data), self.base_theta()
        n_sub = int(round(CONTROL_DT_S / model.opt.timestep))
        u_arm, profile = self.drive.nominal(a[EE_POS_SLICE.start:EE_ROT_SLICE.stop], bool(a[MODE_INDEX] >= 0.5), n_sub)
        v0 = qd[:3]
        u_nom = np.r_[self.base_model.nominal(a[BASE_SLICE], theta, v0), u_arm]
        torso = self.torso_ctrl  # robosuite's joint-position PD (kp 2000, critically damped) towards q + delta
        known = np.r_[pd_mean_velocity(-torso.scale_action(a[TORSO_INDEX:TORSO_INDEX + 1]), qd[3:4], np.sqrt(torso.kp[0]), CONTROL_DT_S), qd[11:13]]
        bounds = self.filter.box(q, self.base_model.bounds(v0), self.arm_q_lo, self.arm_q_hi, CONTROL_DT_S)
        u, rec, bar = self.filter.filter(tick=self.tick, q=q, u_nom=u_nom, known_qdot=known, bounds=bounds, passive=not self.filter_on)
        if not np.allclose(u[:3], u_nom[:3], rtol=0.0, atol=1e-9):  # the base command only changes if the filter changed it
            a[BASE_SLICE] = self.base_model.command(u[:3], theta, v0)
        self.drive.command = (u[3:], profile)
        self.last_barrier = bar
        self._record(rec.as_dict(), np.asarray(action, dtype=float), q, bar)
        out = self.env.step(a.astype(np.float32))
        if self.segment.kind == NAVIGATION and self.replans < MAX_REPLANS and self._path_deviation() > BETA_DEVIATION_BUDGET_M:
            self.replan()  # Assumption 1 premise violated: new path from here, new epoch (stage 1 + labels again)
        return out

    def _record(self, rec: dict, action: np.ndarray, q: np.ndarray, bar) -> None:
        """One tick's receipt, its ground truth and the robot's joints."""
        rec.update(segment=self.segment_index, epoch=self.epoch, action=[float(v) for v in action])
        if rec["h_min_pair"] is not None:
            rec["h_min_category"] = self.graph.nodes[rec["h_min_pair"][2]].category
            rec["h_min_oracle_id"] = f"O:{self.bridge.to_oracle(rec['h_min_pair'][2])}"
        self.receipts.append(rec)
        self.hold_streak = self.hold_streak + 1 if rec["outcome"] == "hold" else 0
        truth = self._truth(bar.centres, bar.radii)
        bp, R, _ = self.robot_pose()
        self.ground_truth.append({"tick": self.tick, "segment": self.segment_index, "clearance": truth["clearance"], "clearance_pair": truth["pair"], "clearance_rows": truth["clearance_rows"],
                                  "box_clearance": truth["box_clearance"], "box_pair": truth["box_pair"],
                                  "touched": sorted(f"O:{oracle_id}" for oracle_id in self._touched_oracle_ids() - self.exempt_oracle_ids),
                                  "base_xy": bp[:2].round(4).tolist(), "yaw": float(np.arctan2(R[1, 0], R[0, 0]))})
        self.q_log.append(q)
        self.sphere_h_log.append(bar.h.min(axis=1) if bar.h.shape[1] else np.full(len(bar.radii), np.inf))
        self.tick += 1

    def _path_deviation(self) -> float:
        p, path = np.asarray(self.ground_truth[-1]["base_xy"]), self.segment.path_polyline_xy
        seg = path[1:] - path[:-1]
        t = np.clip(np.einsum("ij,ij->i", p - path[:-1], seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12), 0, 1)
        return float(np.min(np.linalg.norm(path[:-1] + t[:, None] * seg - p, axis=1)))

    def replan(self) -> None:
        """New base path from here on the planning balls (NoPathError stops the mission; MissionRunner records it)."""
        from sesame_lite.planner import plan_base_path

        seg = self.segment
        seg.path_polyline_xy = plan_base_path(self.graph, self.plan_balls, self.ground_truth[-1]["base_xy"], seg.path_polyline_xy[-1], inflation_m=self.plan_kwargs["inflation_m"], z_band=self.z_band)
        self.replans += 1
        self.begin_segment(self.segment_index)

    # ------------------------------------------------------------------ segment completion

    def segment_done(self) -> bool:
        d = self.segment.done
        if d["check"] == "task_success":
            return bool(self.rs._check_success())
        if d["check"] == "base_near_goal":
            bp = self.robot_pose()[0]
            goal, yaw = self.segment.path_polyline_xy[-1], self.segment.goal_yaw
            near = float(np.linalg.norm(bp[:2] - goal)) <= d.get("radius", 0.2)
            return near and (yaw is None or np.cos(yaw - self.base_yaw()) >= 0.98)
        raise ValueError(d["check"])

    def stalled(self) -> bool:
        if self.hold_streak >= STALL_HOLD_TICKS:
            return True
        if self.segment.kind == NAVIGATION and len(self.ground_truth) >= STALL_PROGRESS_TICKS:
            p0, p1 = self.ground_truth[-STALL_PROGRESS_TICKS]["base_xy"], self.ground_truth[-1]["base_xy"]
            nominal_moving = any(np.linalg.norm(r["u_nominal"][:3]) > 1e-3 for r in self.receipts[-STALL_PROGRESS_TICKS:])
            return nominal_moving and float(np.linalg.norm(np.subtract(p1, p0))) / (STALL_PROGRESS_TICKS * CONTROL_DT_S) < STALL_SPEED_MPS
        return False

    # ------------------------------------------------------------------ reporting

    def feasibility(self) -> dict:
        """Spec 6.3: relaxed ticks, the largest slack, the pairs that needed slack most often, and every streak of
        relaxed ticks with what the receipts say about it (the cause itself is written in the report)."""
        rec = self.receipts
        relaxed = [r for r in rec if r["outcome"] == "relaxed"]
        pairs = Counter((row["link"], row["object"]) for r in relaxed for row in r["slack_rows"])
        streaks, k = [], 0
        while k < len(rec):
            if rec[k]["outcome"] != "relaxed":
                k += 1
                continue
            e = k
            while e + 1 < len(rec) and rec[e + 1]["outcome"] == "relaxed":
                e += 1
            run = rec[k:e + 1]
            sp = Counter((row["link"], row["object"]) for r in run for row in r["slack_rows"])
            first = run[0]
            streaks.append({"ticks": [first["tick"], run[-1]["tick"]], "segment": first["segment"], "length": len(run), "max_slack": max(r["max_slack"] for r in run),
                            "h_min_at_start": first["h_min"], "h_min_pair_at_start": first["h_min_pair"], "starts_at_episode_start": first["tick"] == 0,
                            "rows_with_slack_at_start": len(first["slack_rows"]), "rows_with_negative_h_at_start": sum(row["h"] < 0 for row in first["slack_rows"]),
                            "pairs": [[link, obj, self.graph.nodes[obj].category, n] for (link, obj), n in sp.most_common(5)],
                            "true_clearance_at_start": self.ground_truth[first["tick"]]["clearance"]})
            k = e + 1
        return {"relaxed_ticks": len(relaxed), "hold_ticks": sum(r["outcome"] == "hold" for r in rec), "ticks": len(rec),
                "max_slack": max((r["max_slack"] for r in relaxed), default=0.0),
                "pairs_most_often": [[link, obj, self.graph.nodes[obj].category, n] for (link, obj), n in pairs.most_common(10)],
                "qp_status": dict(Counter(r["qp_status"] for r in rec)), "streaks": streaks}

    def disturbance(self) -> dict:
        """PDF Setting / Theorem 2: q_dot = u + d with |d| <= d_bar. Measured per tick as the mean velocity of the
        commanded joints over the tick minus the u that was sent (the last tick has no successor and is left out)."""
        if len(self.q_log) < 2:
            return {}
        from sesame_lite.filter import COMMANDED

        q = np.array(self.q_log)
        u = np.array([r["u_applied"] for r in self.receipts])
        d = (q[1:, COMMANDED] - q[:-1, COMMANDED]) / CONTROL_DT_S - u[:-1]
        n, nb, na = np.linalg.norm(d, axis=1), np.linalg.norm(d[:, :3], axis=1), np.linalg.norm(d[:, 3:], axis=1)
        qs = lambda x: dict(zip(["50%", "90%", "99%", "max"], np.percentile(x, [50, 90, 99, 100]).round(4).tolist()))  # noqa: E731
        changed = np.array([r["outcome"] in ("box", "filtered", "relaxed", "hold") for r in self.receipts[:-1]])
        return {"d_bar": self.policy.d_bar, "norm": qs(n), "base": qs(nb), "arm": qs(na), "share_above_d_bar": float((n > self.policy.d_bar).mean()),
                "ticks_above_d_bar": np.flatnonzero(n > self.policy.d_bar).tolist()[:50],
                "norm_unchanged_ticks": qs(n[~changed]) if (~changed).any() else None, "norm_changed_ticks": qs(n[changed]) if changed.any() else None}

    def summary(self, success: bool, stalled: bool) -> dict:
        rec, gt = self.receipts, self.ground_truth
        # as before spec v4: a start pose already below zero clearance is reported on its own ("start"), and the ticks
        # until the clearance first reaches zero are the recovery, not a violation; contacts always count
        first_ok = next((k for k, g in enumerate(gt) if g["clearance"] is not None and g["clearance"] >= 0.0), len(gt)) if gt and (gt[0]["clearance"] or 0.0) < 0.0 else 0
        clear = [g["clearance"] for g in gt[first_ok:] if g["clearance"] is not None]
        contact_ticks = sum(1 for g in gt if g["touched"])
        violated = bool(contact_ticks or any(c < 0 for c in clear))
        if self.stop_reason is not None:
            outcome = self.stop_reason
        elif success:
            outcome = "success_but_unsafe" if violated else "success"
        elif stalled:
            outcome = "stalled"
        else:
            outcome = "unsafe_fail" if violated else "safe_fail"
        tick_ms = np.array([r["tick_ms"] for r in rec] or [0.0])
        solve_ms = np.array([r["solve_ms"] for r in rec if r["qp_status"] != "none"] or [0.0])
        i_min = first_ok + int(np.argmin([g["clearance"] if g["clearance"] is not None else np.inf for g in gt[first_ok:]])) if gt[first_ok:] else None
        above = [r["tick"] for r, g in zip(rec, gt) if r["h_min"] is not None and g["clearance_rows"] is not None and r["h_min"] > g["clearance_rows"] + 1e-3]
        box = [g["box_clearance"] for g in gt if g["box_clearance"] is not None]
        return {
            "task_description": self.env.task_description, "pruning": self.pruning, "filter": self.filter_on, "tag": self.tag,
            "outcome": outcome, "success": success, "task_success_robocasa": bool(getattr(self, "task_success_robocasa", False)), "stalled": stalled,
            "steps": self.tick, "segments_done": self.segment_index + int(success), "n_segments": len(self.segments),
            "contact_ticks": contact_ticks, "contact_nodes": sorted({o for g in gt for o in g["touched"]}),
            "contact_unmapped_nodes": sorted({o for g in gt for o in g["touched"] if o[2:] in set(self.unmapped_oracle_ids)}),
            "true_clearance_min": min(clear, default=None), "true_clearance_min_tick": i_min, "true_clearance_min_pair": gt[i_min]["clearance_pair"] if i_min is not None else None,
            "recovery_ticks": first_ok, "true_clearance_min_all_ticks": min((g["clearance"] for g in gt if g["clearance"] is not None), default=None),
            "ticks_true_clearance_negative": sum(c < 0 for c in clear),
            "h_min_filter": min((r["h_min"] for r in rec if r["h_min"] is not None), default=None),
            "ticks_h_negative": sum(1 for r in rec if r["h_min"] is not None and r["h_min"] < 0.0),
            "ticks_barrier_above_truth": len(above), "ticks_barrier_above_truth_list": above[:50],
            "box_clearance_min": min(box, default=None), "ticks_box_clearance_negative": sum(c < 0 for c in box),  # oracle boxes, for comparison only
            "start": self.start,
            "outcomes": dict(Counter(r["outcome"] for r in rec)),
            "feasibility": self.feasibility(), "disturbance": self.disturbance(),
            "n_pairs_mean": float(np.mean([r["n_pairs"] for r in rec] or [0])), "n_pairs_max": max((r["n_pairs"] for r in rec), default=0),
            "n_rows_qp_mean": float(np.mean([r["n_rows_qp"] for r in rec] or [0])), "n_rows_qp_max": max((r["n_rows_qp"] for r in rec), default=0),
            "tick_ms_median": float(np.median(tick_ms)), "tick_ms_p95": float(np.percentile(tick_ms, 95)), "tick_ms_max": float(tick_ms.max()),
            "solve_ms_median": float(np.median(solve_ms)), "solve_ms_max": float(solve_ms.max()),
            "n_spheres": self.robot.n, "n_nodes": len(self.graph.nodes), "n_candidates": [len(b.candidate_ids) for b in self.bundles],
            "n_admitted": [len(b.admitted_ids) for b in self.bundles], "n_row_objects": [s["n_admitted"] for s in self.ball_log],
            "balls": [{k: v for k, v in s.items() if k in ("segment", "epoch", "n_admitted", "m_r", "m_d", "kappa_s_bar_quantiles_m")} for s in self.ball_log],
            "classifier_latency_s": [b.classifier_latency_s for b in self.bundles],
            "classifier_fallback": any(bool(b.classification and b.classification.fallback_used) for b in self.bundles),
            "goal_offset_m": [s.goal_offset_m for s in self.segments], "replans": self.replans,
            "exempt": sorted(f"O:{oracle_id}" for oracle_id in self.exempt_oracle_ids), "exempt_filter_side": sorted(self.segment.exempt_ids),
            "unmapped_oracle_ids": [f"O:{oracle_id}" for oracle_id in self.unmapped_oracle_ids],
            "policy": asdict(self.policy) | {"eps": self.policy.eps, "m_d": self.policy.m_d}, "drive_kv": self.drive.kv, "stop_detail": getattr(self, "stop_detail", None),
            "layout": self.oracle_lookup.meta, "snapshot_id": self.oracle_graph.snapshot_id, "unseeded_rng_calls": SAMPLER_SEED[1],
        }

    def dump(self, path: Path, result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"result": result, "plan": self.plan, "row_ids": list(self.row_ids), "eval_ids": list(self.eval_ids), "segments": [s.as_dict() for s in self.segments],
                       "bundles": [b.as_dict() for b in self.bundles], "graph": self.graph.as_dict(), "oracle_graph": self.oracle_graph.as_dict(), "balls": self.ball_log,
                       "oracle_lookup": self.oracle_lookup.as_dict(), "id_bridge": self.bridge.as_dict(), "sphere_links": self.robot.links,
                       "receipts": self.receipts, "ground_truth": self.ground_truth}, f, default=_json_default)
        np.savez_compressed(path.with_suffix(".npz"), q=np.array(self.q_log), sphere_h=np.array(self.sphere_h_log, dtype=np.float32))
        self.graph.save(path.with_suffix(".dsg.json"))  # the scene graph in Hydra's file format


# ---------------------------------------------------------------------- nominal policies


class NavFollower:
    """Pure-pursuit follower on the planned polyline; velocities in realized m/s and rad/s, mapped through the actuator model."""

    def __init__(self, path_xy: np.ndarray, goal_yaw: float | None, *, policy: FilterPolicy = FilterPolicy(), v_max=0.4, w_max=0.8, lookahead=0.2, k_v=1.5, k_w=2.0):
        self.path, self.goal_yaw, self.policy, self.base = np.asarray(path_xy, dtype=float), goal_yaw, policy, BaseModel()
        self.v_max, self.w_max, self.lookahead, self.k_v, self.k_w = v_max, w_max, lookahead, k_v, k_w
        self.turn_sign = None

    def _lookahead(self, p):
        seg = self.path[1:] - self.path[:-1]
        t = np.clip(np.einsum("ij,ij->i", p - self.path[:-1], seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12), 0, 1)
        i = int(np.argmin(np.linalg.norm(self.path[:-1] + t[:, None] * seg - p, axis=1)))
        remaining, ti = self.lookahead, t[i]
        while i < len(seg):
            left = np.linalg.norm(seg[i]) * (1 - ti)
            if left >= remaining:
                return self.path[i] + (ti + remaining / max(np.linalg.norm(seg[i]), 1e-9)) * seg[i]
            remaining, i, ti = remaining - left, i + 1, 0.0
        return self.path[-1]

    @staticmethod
    def _wrap(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def __call__(self, base_xy: np.ndarray, yaw: float, theta: float) -> np.ndarray:
        target, d_goal = self._lookahead(base_xy), float(np.linalg.norm(self.path[-1] - base_xy))
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        if d_goal < 0.1:
            v, yaw_err = np.zeros(2), 0.0 if self.goal_yaw is None else self._wrap(self.goal_yaw - yaw)
        else:
            v = self.k_v * (R.T @ (target - base_xy))
            v *= min(1.0, self.v_max / max(np.linalg.norm(v), 1e-9))
            heading = float(np.arctan2(*(target - base_xy)[::-1]))
            yaw_err = self._wrap((self.goal_yaw if d_goal < 0.5 and self.goal_yaw is not None else heading) - yaw)
        # a goal straight behind the robot wraps between +pi and -pi: keep the turn direction once chosen
        if abs(yaw_err) > 2.6 and self.turn_sign is None:
            self.turn_sign = np.sign(yaw_err) or 1.0
        if self.turn_sign is not None:
            yaw_err = self.turn_sign * abs(yaw_err)
            if abs(yaw_err) < 0.5:
                self.turn_sign = None
        a = np.zeros(12, dtype=np.float32)
        a[0:3] = self.base.steady_command([v[0], v[1], np.clip(self.k_w * yaw_err, -self.w_max, self.w_max)], theta)
        a[MODE_INDEX] = 1.0
        return a


class SmolVLAPolicy:
    """lerobot/smolvla_robocasa through lerobot's own pre/post-processing pipeline (saved camera map)."""

    def __init__(self, task: str, device: str = "cuda"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
        cfg.pretrained_path, cfg.device = POLICY_PATH, device
        env_cfg = RoboCasaEnvConfig(task=task)
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=RENAME_MAP).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=POLICY_PATH, preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, self.env_post = env_cfg.get_env_processors()
        self.reset()

    def reset(self):
        for p in (self.policy, self.pre, self.post):
            p.reset()

    def __call__(self, obs: dict, language: str) -> np.ndarray:
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        observation = preprocess_observation({"pixels": {k: v[None] for k, v in obs["pixels"].items()}, "agent_pos": np.asarray(obs["agent_pos"])[None]})
        observation["task"] = [language]
        observation = self.pre(self.env_pre(observation))
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        return self.env_post({ACTION: self.post(action)})[ACTION].cpu().numpy()[0].astype(np.float32)


# ---------------------------------------------------------------------- mission runner


class MissionRunner:
    """Run the plan's segments: follower for NAVIGATION, SmolVLA for VLA; new epoch at every segment switch."""

    def __init__(self, env: SafetyFilterEnv, *, vla: SmolVLAPolicy | None, on_frame=None):
        self.env, self.vla, self.on_frame = env, vla, on_frame

    def run(self, obs) -> tuple[bool, bool]:
        env = self.env
        success = stalled = False
        max_steps = env.env._max_episode_steps
        while env.tick < max_steps:
            seg = env.segment
            follower = NavFollower(seg.path_polyline_xy, seg.goal_yaw, policy=env.policy) if seg.kind == NAVIGATION else None
            if seg.kind == VLA and self.vla is None:
                raise RuntimeError("a VLA segment needs the smolvla nominal")
            if self.vla is not None:
                self.vla.reset()
            seg_steps, epoch = 0, env.epoch
            while env.tick < max_steps and (seg.max_steps is None or seg_steps < seg.max_steps):
                if follower is not None and env.epoch != epoch:  # replanned path: fresh follower on it
                    follower, epoch = NavFollower(seg.path_polyline_xy, seg.goal_yaw, policy=env.policy), env.epoch
                if follower is not None:
                    action = follower(env.robot_pose()[0][:2], env.base_yaw(), env.base_theta())
                else:
                    action = self.vla(obs, seg.goal_text)
                try:
                    obs, _, terminated, _, info = env.step(action)
                except NoPathError as exc:  # a replan found no path: the mission stops and says so
                    env.stop_reason, env.stop_detail = "no_path", str(exc)
                    break
                seg_steps += 1
                if self.on_frame is not None:
                    self.on_frame(env)
                if env.segment_done():
                    break
                if env.stalled():
                    stalled = True
                    break
                if terminated and not info.get("is_success"):
                    break
            if stalled or env.stop_reason is not None or not env.segment_done():
                break
            if env.segment_index + 1 >= len(env.segments):
                success = True  # every plan segment completed (the same plan for every switch setting)
                break
            env.begin_segment(env.segment_index + 1)
        env.task_success_robocasa = bool(env.rs._check_success())
        return success, stalled


def run_tag(pruning: bool, filter: bool) -> str:
    """Output-file tag of a switch setting: "sesame" for the defaults, else what is switched off."""
    off = [name for name, on in (("noprune", pruning), ("nofilter", filter)) if not on]
    return "_".join(off) or "sesame"


def run_episode(*, task: str, seed: int, nominal: str, classifier: str, out_dir: Path, pruning: bool = True, filter: bool = True, plan_path: str | None = None, env_kwargs: dict | None = None, max_steps: int | None = None, video: bool = False, policy: dict | None = None) -> dict:
    """Run one (task, seed) with one switch setting in this process (the kitchen is hard-reset from the seed)."""
    import sesame_lite.tasks  # noqa: F401  (registers the task variants with robosuite)
    from sesame_lite.planner import load_plan

    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(plan_path) if plan_path else None
    if plan:  # the plan carries the task, its kitchen kwargs and the horizon; the CLI may override kwargs
        task = plan.get("task", task)
        env_kwargs = {**plan.get("env_kwargs", {}), **(env_kwargs or {})}
        max_steps = max_steps or plan.get("horizon")
    base_env = SesameRoboCasaEnv(task, env_kwargs=env_kwargs, episode_length=max_steps)
    t0 = time.time()
    env = SafetyFilterEnv(base_env, pruning=pruning, filter=filter, classifier=classifier, policy=FilterPolicy(**(policy or {})))
    obs, _ = env.reset(seed=seed, plan=plan)
    vla = SmolVLAPolicy(task) if (nominal == "smolvla" or any(s.kind == VLA for s in env.segments)) else None
    recorder = None
    if video:
        from sesame_lite.viz import EpisodeRecorder

        recorder = EpisodeRecorder(env)
    success, stalled = MissionRunner(env, vla=vla, on_frame=recorder).run(obs)
    result = {"task": task, "seed": seed, "nominal": nominal, "classifier": classifier, "wall_s": time.time() - t0, **env.summary(success, stalled)}
    stem = f"{task}_s{seed}_{run_tag(pruning, filter)}"
    env.dump(out_dir / f"{stem}.json", result)
    if recorder is not None:
        recorder.finish(result)
        recorder.save(out_dir / f"{stem}.mp4", robot_path=out_dir / f"{stem}_robot_spheres.mp4")
        del recorder
    base_env.close()
    return result

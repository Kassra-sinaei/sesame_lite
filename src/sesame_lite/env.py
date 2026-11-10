"""SESAME-3D on RoboCasa: lerobot's ``RoboCasaEnv`` subclassed for pinned, non-resetting episodes,
a gym wrapper that runs the per-row semantic safety filter, and the two nominal policies.

What runs is set by two switches (defaults: both on, the SESAME-3D method). Every run of a seed shares the scene
(pinned episode metadata), the nominal policy, the barrier definition and the ground-truth evaluation set:

  pruning   on: rows only for the nodes admitted by the swept-volume stage and the semantic classifier
            off: every scene-graph node is a row (the plan's contact exemptions still apply)
  filter    on: the QP corrects the action
            off: rows are evaluated but the action passes unchanged (reference run, no classifier call)

Nominal policies: ``nav`` (pure-pursuit follower on the planned polyline, the navigation-stack
analog) and ``smolvla`` (lerobot/smolvla_robocasa through lerobot's own processing pipeline).
``MissionRunner`` sequences the plan's segments. One process per episode (scripts/run_episode.py).
"""

from __future__ import annotations

from dataclasses import replace
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
from sesame_lite.filter import CONTROL_DT_S, MODE_INDEX, FilterPolicy, PerRowFilter, RobotState, base_velocity_to_command
from sesame_lite.geometry.swept_volume import BETA_DEVIATION_BUDGET_M
from sesame_lite.planner import NAVIGATION, VLA, build_bundle, default_plan, resolve_plan
from sesame_lite.uncertainty import kappa
from sesame_lite.determinism import SAMPLER_SEED, begin_episode, seed_fallback_generators
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _raw_data, _raw_model, _subtree_body_ids, capture_scene_graph, hydra_lookup, perturb
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
BACK_OUT_MAX_M = 0.10  # before a manipulation segment the base may back out of an enlarged box by at most this
BACK_OUT_SPEED_MPS = 0.10
GT_HORIZON_M = 0.6  # ground truth keeps exact barrier values up to this distance (pruning-quality deltas go to 0.5 m)  # re-prunes per navigation segment after a corridor exit (bounded, as in the paper)


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
    """Per-row semantic safety filter in front of ``SesameRoboCasaEnv``; segments come from a plan."""

    def __init__(self, env: SesameRoboCasaEnv, *, pruning: bool = True, filter: bool = True, classifier: str = "prior", policy: FilterPolicy = FilterPolicy(), graph_noise: dict | None = None, noise_seed: int | None = None, graph_source: str = "oracle"):
        super().__init__(env)
        if graph_source not in ("oracle", "hydra"):
            raise ValueError(graph_source)
        if graph_source == "hydra" and graph_noise:
            raise ValueError("graph_noise (synthetic face noise) is the oracle-graph fallback; a Hydra map has its own")
        self.pruning, self.filter_on, self.classifier_kind, self.policy = pruning, filter, classifier, policy
        self.graph_source = graph_source
        self.tag = run_tag(pruning, filter)
        self.graph_noise, self.noise_seed = graph_noise, noise_seed
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

    ARM_LINKS = ("robot0_link3", "robot0_link4", "robot0_link5", "robot0_link6", "robot0_link7")

    def arm_points(self) -> np.ndarray:
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        return np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in self.ARM_LINKS])

    def yaw_axis_xy(self, bp: np.ndarray, R: np.ndarray) -> np.ndarray:
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mobilebase0_joint_mobile_yaw")
        return (R.T @ (data.xanchor[jid] - bp))[:2]

    def robot_state(self, base_mode: bool = True) -> RobotState:
        bp, R, eef = self.robot_pose()
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        v = np.zeros(3)
        if getattr(self, "_prev_pose", None) is not None:  # realized base velocity over the last tick, base frame
            pxy, pyaw = self._prev_pose
            v[:2] = R[:2, :2].T @ ((bp[:2] - pxy) / CONTROL_DT_S)
            v[2] = float(np.arctan2(np.sin(yaw - pyaw), np.cos(yaw - pyaw))) / CONTROL_DT_S
        return RobotState(base_position=bp, base_rotation=R, eef_position=eef, base_rectangle_xy=self.rect, base_mode=base_mode, base_velocity=v, yaw_axis_xy=self.yaw_axis_xy(bp, R), arm_points=self.arm_points(), command_theta=yaw - self.init_yaw, base_locked=getattr(self, "base_locked", False))

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
        self.init_yaw = float(np.arctan2(R[1, 0], R[0, 0]))  # the base controller's reference orientation
        pg, Rg, size = self._base_box()
        signs = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)
        local = ((pg[None, :2] + (signs * size[:2]) @ Rg[:2, :2].T) - bp[None, :2]) @ R[:2, :2]
        lo, hi = local.min(axis=0), local.max(axis=0)
        self.rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        self.z_band = (float(pg[2] - size[2]), float(pg[2] + size[2]))
        self.filter = PerRowFilter(replace(self.policy, base_z_interval_m=self.z_band))
        self.base_radius = float(np.max(np.linalg.norm(self.rect, axis=1)))
        self.plan = plan or default_plan(rs)
        # a goal must leave room for the filter's standoff (kept at its earlier value: margin + d/alpha + 1 cm)
        standoff = self.policy.goal_standoff_m
        self.plan_kwargs = dict(base_xy=bp[:2], rect_xy=self.rect, z_band=self.z_band, clearance_m=standoff, inflation_m=float(hi[1] - lo[1]) / 2 + 0.10)
        self.segments = self._resolve()
        if self.plan.get("injections") and self.segments[0].kind == NAVIGATION:
            # floor objects are dropped next to the planned path, then the graph is captured again (the path planner
            # only knows fixtures, as a navigation stack's static map would; the objects enter through the 3DSG)
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
        self.receipts, self.ground_truth, self.bundles, self.min_h_by_oracle_id, self.confidence_log = [], [], [], {}, []
        self.tick, self.epoch, self.hold_streak, self._prev_pose = 0, 0, 0, None
        self.replans, self.segment_index = 0, -1
        self.back_out_log, self.blocked, self.base_locked, self.pending_clear = [], False, False, False
        self.start_h_enlarged = None  # set by begin_segment(0): the start check on the enlarged boxes
        self.begin_segment(0)
        self.start_h = self._true_h(self.robot_state())
        self.start_violation = bool(self.start_h is not None and self.start_h < -self.policy.policy_margin_m)
        return obs, info

    def _capture(self) -> None:
        """The scene graph the filter side uses (``graph_source``) and the oracle graph of the evaluation side.

        oracle: the oracle graph itself (or its noisy copy); the id bridge is the identity.
        hydra:  Hydra's graph from the mapping pass, cached in outputs/maps/<mission>_s<seed>_<n>/ (n = 0 at reset,
                1 after the injections); built here if the cache is missing. The cache must belong to this exact scene.
        """
        self.oracle_graph, self.oracle_lookup = capture_scene_graph(self.rs)
        if self.graph_source == "oracle":
            rng = np.random.default_rng(self.seed if self.noise_seed is None else self.noise_seed)
            self.base_graph = perturb(self.oracle_graph, rng, **self.graph_noise) if self.graph_noise else self.oracle_graph
            self.bridge, self.lookup = IdBridge.identity(self.oracle_graph), self.oracle_lookup
        else:
            from sesame_lite.mapping import build_map, map_dir

            cache = map_dir(self.mission, self.seed, self.n_capture)
            if not (cache / "faces.json").exists():
                build_map(self.rs, self.oracle_graph, self.oracle_lookup, cache, seed=self.seed)
            cached_id = json.loads((cache / "meta.json").read_text())["oracle_snapshot_id"]
            assert cached_id == self.oracle_graph.snapshot_id, f"map cache {cache} belongs to another scene ({cached_id} != {self.oracle_graph.snapshot_id})"
            bp, R, _ = self.robot_pose()
            self.base_graph = HydraSceneGraph(cache, self.oracle_graph.labels, robot_pose=(bp, float(np.arctan2(R[1, 0], R[0, 0]))))
            self.bridge = IdBridge.from_instances(self.base_graph, self.oracle_graph)
            self.lookup = hydra_lookup(self.oracle_lookup, self.bridge)
        self.n_capture += 1
        # planning, standoff and pruning see the confidence boxes for the largest admitted count (every node), so they
        # are never smaller than the boxes the filter enforces once a segment's admitted count is known
        self.graph = self.base_graph.with_confidence(kappa=kappa(self.policy.delta, len(self.base_graph.nodes)), m_d=self.policy.disturbance_offset_m, m_r_voxels=self.policy.m_r_voxels)
        self.oracle_fragments, self.fragments = self.oracle_graph.fragments(), self.graph.fragments()

    def _resolve(self):
        """Plan roles -> filter-side nodes; a role's exemption covers every node of its instance. A role that resolves
        to no node of a Hydra map (the target was never mapped) stops the mission: it is never filled in from the oracle."""
        from sesame_lite.planner import resolve_role

        if self.graph_source == "hydra":
            missing = [name for name, spec in self.plan.get("roles", {}).items() if resolve_role(spec, self.rs, self.lookup) is None]
            if missing:
                raise UnmappedRoleError(f"role(s) {missing} resolve to no node of the Hydra map {self.graph.map_dir}")
        segments = resolve_plan(self.plan, self.rs, self.graph, self.lookup, language=self.env.task_description, **self.plan_kwargs)
        for seg in segments:
            seg.exempt_ids = tuple(dict.fromkeys(h for hydra_id in seg.exempt_ids for h in self.bridge.instance_nodes(hydra_id)))
        return segments

    def begin_segment(self, index: int) -> None:
        """Prune for one segment (an epoch) and fix its row set."""
        if index != self.segment_index:
            self.replans = 0
        self.segment_index = index
        seg = self.segments[index]
        self.bundle = build_bundle(seg, self.graph, classifier=self._classifier, base_circumscribed_radius_m=self.base_radius, policy_margin_m=self.policy.policy_margin_m, eef_reach_m=1.0 * CONTROL_DT_S)
        self.bundles.append(self.bundle)
        # the confidence boxes of this segment: delta split over its admitted nodes (every node without pruning)
        n_risk = len(self.bundle.admitted_ids) if self.pruning else len(self.base_graph.nodes)
        self.row_graph = self.base_graph.with_confidence(kappa=kappa(self.policy.delta, n_risk), m_d=self.policy.disturbance_offset_m, m_r_voxels=self.policy.m_r_voxels)
        self.fragments = self.row_graph.fragments()
        row_ids = self.bundle.admitted_ids if self.pruning else [n.node_id for n in self.graph.obstacle_nodes()]
        self.row_ids = self.graph.fragment_ids_of(row_ids, exempt=seg.exempt_ids)
        # ground truth: every non-exempt node of the oracle graph, the same for every switch setting
        self.exempt_oracle_ids = {self.bridge.to_oracle(hydra_id) for hydra_id in seg.exempt_ids} - {None}
        self.eval_ids = self.oracle_graph.fragment_ids_of([n.node_id for n in self.oracle_graph.obstacle_nodes()], exempt=self.exempt_oracle_ids)
        if index == 0 and self.start_h_enlarged is None:  # the start check on the enlarged boxes (before any back-out)
            samples, _, _, _ = self.filter.build_rows(self.robot_state(), self.fragments, base_rows_for=self.row_ids, eef_rows_for=self.row_ids)
            self.start_h_enlarged = min((smp.value for smp in samples), default=None)
        self.base_locked = False
        self.pending_clear = seg.kind == VLA and self.filter_on  # MissionRunner clears the base before the segment runs
        admitted_faces = [n.confidence["kappa_sigma"][k] for nid, n in self.row_graph.nodes.items() if nid in set(self.row_ids) for k in range(6)]
        self.confidence_log.append({"segment": index, "epoch": self.epoch + 1, "n_risk": n_risk, "kappa": kappa(self.policy.delta, n_risk), "m_d": self.policy.disturbance_offset_m,
                                    "m_r": self.base_graph.voxel_size_m * self.policy.m_r_voxels, "kappa_sigma_admitted_faces": admitted_faces, "boxes": self.row_graph.confidence_as_dict()})
        self.epoch += 1
        self.segment_start_tick = self.tick

    @property
    def segment(self):
        return self.segments[self.segment_index]

    @property
    def unmapped_oracle_ids(self) -> list[str]:
        return self.bridge.unmapped(self.oracle_graph)

    def _true_h(self, state: RobotState) -> float | None:
        samples, _, _, _ = self.filter.build_rows(state, self.oracle_fragments, base_rows_for=self.eval_ids, eef_rows_for=self.eval_ids, skip_beyond_m=GT_HORIZON_M, exact=True)
        for s in samples:
            key = f"O:{s.obstacle_parent_id}"
            self.min_h_by_oracle_id[key] = min(self.min_h_by_oracle_id.get(key, np.inf), s.value)
        return min((s.value for s in samples), default=None)

    # ------------------------------------------------------------------ gym API

    def clear_base(self, on_frame=None) -> None:
        """Before a manipulation segment (spec item 6b): if the base is inside an enlarged box (a base row has h < 0),
        drive it out along the gradient of the deepest base row at BACK_OUT_SPEED_MPS until every base row is clear,
        at most BACK_OUT_MAX_M from where it stood, then lock it for the segment. If that is not enough the mission
        stops (``blocked``). The ticks are recorded like any other (outcome "back_out")."""
        log = self._back_out(on_frame)
        self.back_out_log.append(log)
        self.blocked = not log["cleared"]
        self.base_locked = True
        self.pending_clear = False

    def _back_out(self, on_frame) -> dict:
        start = self.robot_pose()[0][:2].copy()
        log = {"segment": self.segment_index, "h_start_m": None, "distance_m": 0.0, "ticks": 0, "cleared": True}
        while True:
            state = self.robot_state(True)
            samples, G, _, _ = self.filter.build_rows(state, self.fragments, base_rows_for=self.row_ids, eef_rows_for=())
            hs = [smp.value for smp in samples]
            h = min(hs, default=float("inf"))
            if log["h_start_m"] is None:
                log["h_start_m"] = h if np.isfinite(h) else None
            log["distance_m"] = float(np.linalg.norm(state.base_position[:2] - start))
            if h >= 0.0:
                break
            if log["distance_m"] >= BACK_OUT_MAX_M or log["ticks"] >= 10 * int(BACK_OUT_MAX_M / (BACK_OUT_SPEED_MPS * CONTROL_DT_S)):
                log["cleared"] = False
                break
            grad = -G[int(np.argmin(hs)), :2]  # dh / d(vx, vy) of the deepest row, base frame
            v = BACK_OUT_SPEED_MPS * grad / max(float(np.linalg.norm(grad)), 1e-9)
            action = np.zeros(12, dtype=np.float32)
            action[0:3] = base_velocity_to_command([v[0], v[1], 0.0], self.policy, state.command_theta)
            action[MODE_INDEX] = 1.0
            self._record(state, {"tick": self.tick, "outcome": "back_out", "n_rows": len(samples), "h_min": h, "h_min_row": samples[int(np.argmin(hs))].constraint_id,
                                 "n_rows_negative": sum(x < 0 for x in hs), "below_margin": False, "qp_status": "none", "solve_ms": 0.0, "tick_ms": 0.0,
                                 "u_nominal": [0.0] * 6, "u_applied": [float(v[0]), float(v[1])] + [0.0] * 4, "correction_norm": 0.0, "rows": [],
                                 "rows_without_free_command": []}, [])
            self.env.step(action)
            log["ticks"] += 1
            if on_frame is not None:
                on_frame(self)
        return log

    def _record(self, state, rec: dict, active_rows) -> None:
        """Append one tick's receipt (with the ids on both sides) and its ground truth."""
        filter_id = rec["h_min_row"].split(":")[-1] if rec["h_min_row"] else None
        oracle_id = self.bridge.to_oracle(filter_id) if filter_id else None
        rec.update(segment=self.segment_index, epoch=self.epoch, graph_source=self.graph_source,
                   h_min_id=f"{self.bridge.prefix}{filter_id}" if filter_id else None, h_min_oracle_id=f"O:{oracle_id}" if oracle_id else None)
        # every active row: node, category, the nearest face's kappa*sigma and total inflation, and h
        rows = []
        for row_id, body, h, face in active_rows:
            conf = self.row_graph.nodes[row_id].confidence
            rows.append({"id": f"{self.bridge.prefix}{row_id}", "body": body, "category": self.row_graph.nodes[row_id].category,
                         "kappa_sigma": round(conf["kappa_sigma"][face], 5), "inflation": round(conf["inflation"][face], 5), "h": round(h, 5)})
        rec["active_rows"] = rows
        self.receipts.append(rec)
        self.hold_streak = self.hold_streak + 1 if rec["outcome"] == "hold" else 0
        self.ground_truth.append(
            {
                "tick": self.tick,
                "segment": self.segment_index,
                "h_min_all": self._true_h(state),
                "touched": sorted(f"O:{oracle_id}" for oracle_id in self._touched_oracle_ids() - self.exempt_oracle_ids),
                "base_xy": state.base_position[:2].round(4).tolist(),
                "yaw": float(np.arctan2(state.base_rotation[1, 0], state.base_rotation[0, 0])),
                "eef": state.eef_position.round(4).tolist(),
            }
        )
        self._prev_pose = (state.base_position[:2].copy(), float(np.arctan2(state.base_rotation[1, 0], state.base_rotation[0, 0])))
        self.tick += 1

    def step(self, action):
        state = self.robot_state(bool(action[MODE_INDEX] >= 0.5))
        a_out, receipt = self.filter.filter_action(tick=self.tick, action=action, state=state, fragments=self.fragments, base_rows_for=self.row_ids, eef_rows_for=self.row_ids, passive=not self.filter_on)
        self._record(state, receipt.as_dict(), receipt.active_rows)
        out = self.env.step(np.asarray(a_out, dtype=np.float32))
        if self.segment.kind == NAVIGATION and self.replans < MAX_REPLANS and self._path_deviation() > BETA_DEVIATION_BUDGET_M:
            self.replan()  # Assumption 1 premise violated: new path from here, new epoch (stage 1 + labels again)
        return out

    def _path_deviation(self) -> float:
        p, path = np.asarray(self.ground_truth[-1]["base_xy"]), self.segment.path_polyline_xy
        seg = path[1:] - path[:-1]
        t = np.clip(np.einsum("ij,ij->i", p - path[:-1], seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12), 0, 1)
        return float(np.min(np.linalg.norm(path[:-1] + t[:, None] * seg - p, axis=1)))

    def replan(self) -> None:
        from sesame_lite.planner import plan_base_path

        seg = self.segment
        seg.path_polyline_xy = plan_base_path(self.graph, self.ground_truth[-1]["base_xy"], seg.path_polyline_xy[-1], inflation_m=self.plan_kwargs["inflation_m"], z_band=self.z_band)
        self.replans += 1
        self.begin_segment(self.segment_index)

    # ------------------------------------------------------------------ segment completion

    def segment_done(self) -> bool:
        d = self.segment.done
        if d["check"] == "task_success":
            return bool(self.rs._check_success())
        if d["check"] == "base_near_goal":
            st = self.robot_state()
            goal, yaw = self.segment.path_polyline_xy[-1], self.segment.goal_yaw
            near = float(np.linalg.norm(st.base_position[:2] - goal)) <= d.get("radius", 0.2)
            cur_yaw = float(np.arctan2(st.base_rotation[1, 0], st.base_rotation[0, 0]))
            return near and (yaw is None or np.cos(yaw - cur_yaw) >= 0.98)
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

    def summary(self, success: bool, stalled: bool) -> dict:
        m = self.policy.policy_margin_m
        tick_ms = np.array([r["tick_ms"] for r in self.receipts] or [0.0])
        solve_ms = np.array([r["solve_ms"] for r in self.receipts if r["qp_status"] != "none"] or [0.0])
        gt = self.ground_truth
        if self.start_violation:  # the start state (box overlap before any motion) is reported on its own; the recovery transient is not a violation
            first_safe = next((i for i, g in enumerate(gt) if g["h_min_all"] is not None and g["h_min_all"] >= 0.0), len(gt))
            gt = gt[first_safe:]
        h_all = [g["h_min_all"] for g in gt if g["h_min_all"] is not None]
        contact_ticks = sum(1 for g in self.ground_truth if g["touched"])  # physical contact always counts
        violated = bool(contact_ticks or any(h < -m for h in h_all))
        if self.blocked:
            outcome = "base_not_cleared"
        elif self.start_violation and not success and not violated and not gt:
            outcome = "start_violation"
        elif success:
            outcome = "success_but_unsafe" if violated else "success"
        elif stalled:
            outcome = "stalled"
        else:
            outcome = "unsafe_fail" if violated else "safe_fail"
        return {
            "task_description": self.env.task_description,
            "pruning": self.pruning,
            "filter": self.filter_on,
            "tag": self.tag,
            "outcome": outcome,
            "success": success,
            "task_success_robocasa": bool(getattr(self, "task_success_robocasa", False)),
            "stalled": stalled,
            "start_violation": self.start_violation,
            "start_h": self.start_h,
            "start_h_enlarged": self.start_h_enlarged,  # the start check on the enlarged boxes (filter side)
            "start_inside_enlarged": bool(self.start_h_enlarged is not None and self.start_h_enlarged < 0.0),
            "back_out": self.back_out_log,
            "ticks_rows_without_free_command": sum(1 for r in self.receipts if r.get("rows_without_free_command")),
            "steps": self.tick,
            "segments_done": self.segment_index + int(success),
            "n_segments": len(self.segments),
            "n_nodes": len(self.graph.obstacle_nodes()),
            "n_fragments": len(self.fragments),
            "n_candidates": [len(b.candidate_ids) for b in self.bundles],
            "n_admitted": [len(b.admitted_ids) for b in self.bundles],
            "n_rows_ids": len(self.row_ids),
            "n_rows_mean": float(np.mean([r["n_rows"] for r in self.receipts] or [0])),
            "h_min_all": min(h_all, default=None),
            "penetration_m": max((-h - m for h in h_all if h < -m), default=0.0),
            "severity_ms": float(sum(max(0.0, -h - m) * CONTROL_DT_S for h in h_all)),
            "ticks_below_margin_all": sum(h < -m for h in h_all),
            "contact_ticks": contact_ticks,
            "contact_nodes": sorted({oracle_id for g in gt for oracle_id in g["touched"]}),
            # contacts with objects the map has no node for (the filter cannot protect them; scoring still counts them)
            "contact_unmapped_nodes": sorted({oracle_id for g in gt for oracle_id in g["touched"] if oracle_id[2:] in set(self.unmapped_oracle_ids)}),
            "graph_source": self.graph_source,
            "graph_label": self.base_graph.label,
            "unmapped_oracle_ids": [f"O:{oracle_id}" for oracle_id in self.unmapped_oracle_ids],
            "h_min_filter": min((r["h_min"] for r in self.receipts if r["h_min"] is not None), default=None),  # smallest h of the rows (confidence boxes)
            "ticks_h_negative": sum(1 for r in self.receipts if r["h_min"] is not None and r["h_min"] < 0.0),
            "kappa_sigma_admitted_faces": [{"segment": c["segment"], "kappa": c["kappa"], "n_risk": c["n_risk"], "m_r": c["m_r"], "m_d": c["m_d"],
                                            "quantiles_m": dict(zip(["min", "50%", "90%", "max"], np.percentile(c["kappa_sigma_admitted_faces"], [0, 50, 90, 100]).round(5).tolist())) if c["kappa_sigma_admitted_faces"] else None}
                                           for c in self.confidence_log],
            "holds": self.filter.hold_count,
            "infeasible": self.filter.infeasible_count,
            "relaxed": self.filter.relaxed_count,
            "filtered_ticks": sum(r["outcome"] == "filtered" for r in self.receipts),
            "tick_ms_median": float(np.median(tick_ms)),
            "tick_ms_p95": float(np.percentile(tick_ms, 95)),
            "tick_ms_p99": float(np.percentile(tick_ms, 99)),
            "solve_ms_median": float(np.median(solve_ms)),
            "classifier_latency_s": [b.classifier_latency_s for b in self.bundles],
            "classifier_fallback": any(bool(b.classification and b.classification.fallback_used) for b in self.bundles),
            "goal_offset_m": [s.goal_offset_m for s in self.segments],
            "exempt": sorted(f"O:{oracle_id}" for oracle_id in self.exempt_oracle_ids),
            "exempt_filter_side": sorted(f"{self.bridge.prefix}{i}" for i in self.segment.exempt_ids),
            "layout": self.oracle_lookup.meta,
            "snapshot_id": self.oracle_graph.snapshot_id,
            "unseeded_rng_calls": SAMPLER_SEED[1],
        }

    def dump(self, path: Path, result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "result": result,
                    "plan": self.plan,
                    "row_ids": list(self.row_ids),
                    "eval_ids": list(self.eval_ids),
                    "segments": [s.as_dict() for s in self.segments],
                    "bundles": [b.as_dict() for b in self.bundles],
                    "graph": self.base_graph.as_dict(),  # the filter side's estimated boxes (and faces with their std)
                    "oracle_graph": self.oracle_graph.as_dict(),  # evaluation side
                    "confidence": self.confidence_log,  # per segment: kappa, margins and every confidence box
                    "oracle_lookup": self.oracle_lookup.as_dict(),  # evaluation side only
                    "id_bridge": self.bridge.as_dict(),  # filter-side id -> oracle id
                    "min_h_by_oracle_id": self.min_h_by_oracle_id,
                    "receipts": self.receipts,
                    "ground_truth": self.ground_truth,
                },
                f,
                default=_json_default,
            )
        self.graph.save(path.with_suffix(".dsg.json"))  # the scene graph in Hydra's file format


# ---------------------------------------------------------------------- nominal policies


class NavFollower:
    """Pure-pursuit follower on the planned polyline; velocities in realized m/s and rad/s, mapped through the actuator model."""

    def __init__(self, path_xy: np.ndarray, goal_yaw: float | None, *, init_yaw: float = 0.0, policy: FilterPolicy = FilterPolicy(), v_max=0.4, w_max=0.8, lookahead=0.2, k_v=1.5, k_w=2.0):
        self.path, self.goal_yaw, self.policy, self.init_yaw = np.asarray(path_xy, dtype=float), goal_yaw, policy, init_yaw
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

    def __call__(self, base_xy: np.ndarray, yaw: float) -> np.ndarray:
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
        a[0:3] = base_velocity_to_command([v[0], v[1], np.clip(self.k_w * yaw_err, -self.w_max, self.w_max)], self.policy, yaw - self.init_yaw)
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
            if env.pending_clear:
                env.clear_base(self.on_frame)
            if env.blocked:  # the base could not be cleared of an enlarged box within BACK_OUT_MAX_M: the mission stops
                break
            seg = env.segment
            follower = NavFollower(seg.path_polyline_xy, seg.goal_yaw, init_yaw=env.init_yaw, policy=env.policy) if seg.kind == NAVIGATION else None
            if seg.kind == VLA and self.vla is None:
                raise RuntimeError("a VLA segment needs the smolvla nominal")
            if self.vla is not None:
                self.vla.reset()
            seg_steps, epoch = 0, env.epoch
            while env.tick < max_steps and (seg.max_steps is None or seg_steps < seg.max_steps):
                if follower is not None and env.epoch != epoch:  # replanned path: fresh follower on it
                    follower, epoch = NavFollower(seg.path_polyline_xy, seg.goal_yaw, init_yaw=env.init_yaw, policy=env.policy), env.epoch
                if follower is not None:
                    st = env.robot_state()
                    action = follower(st.base_position[:2], float(np.arctan2(st.base_rotation[1, 0], st.base_rotation[0, 0])))
                else:
                    action = self.vla(obs, seg.goal_text)
                obs, _, terminated, _, info = env.step(action)
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
            if stalled or not env.segment_done():
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


def run_episode(*, task: str, seed: int, nominal: str, classifier: str, out_dir: Path, pruning: bool = True, filter: bool = True, graph_source: str = "oracle", plan_path: str | None = None, env_kwargs: dict | None = None, max_steps: int | None = None, graph_noise: dict | None = None, video: bool = False) -> dict:
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
    env = SafetyFilterEnv(base_env, pruning=pruning, filter=filter, classifier=classifier, graph_noise=graph_noise, graph_source=graph_source)
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
        recorder.save(out_dir / f"{stem}.mp4")
        del recorder
    base_env.close()
    return result

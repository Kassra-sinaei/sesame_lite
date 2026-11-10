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
from sesame_lite.filter import CONTROL_DT_S, MODE_INDEX, FilterPolicy, base_velocity_to_command
from sesame_lite.planner import NAVIGATION, VLA, build_bundle, default_plan, resolve_plan
from sesame_lite.determinism import SAMPLER_SEED, begin_episode, seed_fallback_generators
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _raw_data, _raw_model, _subtree_body_ids, capture_scene_graph, hydra_lookup
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
        self.init_yaw = float(np.arctan2(R[1, 0], R[0, 0]))  # the base controller's reference orientation
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
        self.receipts, self.ground_truth, self.bundles, self.ball_log = [], [], [], []
        self.tick, self.epoch = 0, 0
        self.replans, self.segment_index = 0, -1
        self.begin_segment(0)
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

    # ------------------------------------------------------------------ gym API

    def step(self, action):
        """Spec v4 step 2: the filter is not there yet (step 3); only the unfiltered reference may run."""
        if self.filter_on:
            raise NotImplementedError("the point-cloud filter is spec v4 step 3")
        return self.env.step(np.asarray(action, dtype=np.float32))


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


def run_episode(*, task: str, seed: int, nominal: str, classifier: str, out_dir: Path, pruning: bool = True, filter: bool = True, plan_path: str | None = None, env_kwargs: dict | None = None, max_steps: int | None = None, video: bool = False) -> dict:
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
    env = SafetyFilterEnv(base_env, pruning=pruning, filter=filter, classifier=classifier)
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

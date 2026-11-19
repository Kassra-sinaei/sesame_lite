"""The AEGIS row of the comparison table v3 (Kasra, 2026-11-19): EvalFilterEnv with Proposed in the navigation segment
(filter_segments=(0,), scripts/eval_run.py) and AEGIS (aegis.py) on the end-effector command in the manipulation
segment, where the base moves under the policy unfiltered and the Proposed filter is passive (its barrier is still
evaluated and logged, as for Nominal). At the start of the manipulation segment the scene's designated obstacle is
fitted once with an ellipsoid from its map node's points; the ellipsoid stays fixed.

Our action carries the end-effector deltas in the arm's base frame (robosuite OSC_POSE, input_ref_frame "base", site
robot0_right_center); their code (LIBERO, robosuite 1.4) takes them in the world frame. The command is turned into the
world frame, filtered by their code, and turned back. The units are the same (OSC_POSE output_max 0.05 m and 0.5 rad
per step, kp 150, 20 Hz in both), so their gains are kept. The filtered command goes to our arm drive like any action.

Receipts of manipulation ticks: ``action`` is the policy's command, ``aegis`` the filter's record (filtered command, h,
daqp exit, solve time); ``qp_status``, ``solve_ms``, ``n_pairs`` and ``n_rows_qp`` are AEGIS's QP (one row) and
``outcome`` is "aegis_filtered", "aegis_passthrough" or "aegis_nominal" (no solution: the nominal command passed);
``h_min`` and its pair are the passive Proposed barrier.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from aegis.aegis import GRIPPER_AXES_M, AegisFilter, gripper_centre
from sesame_lite.eval_scene import EvalFilterEnv
from sesame_lite.filter import EE_POS_SLICE, EE_ROT_SLICE
from sesame_lite.planner import VLA
from sesame_lite.scene_graph import _raw_data, _raw_model
from sesame_lite.viz import EvalVideo

ORIGIN_SITE = "robot0_right_center"  # the arm controller's reference frame (robosuite CompositeController.get_controller_base_pose)
DAQP_STATUS = {1: "optimal", 2: "soft_optimal", -1: "infeasible", -2: "cycling", -3: "unbounded", -4: "iteration_limit", -5: "nonconvex", -6: "overdetermined_start"}


class AegisEnv(EvalFilterEnv):
    def begin_segment(self, index: int) -> None:
        super().begin_segment(index)
        self.aegis, self.aegis_tick = None, None
        if self.segment.kind == VLA:
            p1, _ = self.gripper_pose()
            nodes = self.bridge.hydra_of.get(self.obstacle_oracle, [])
            if not nodes:
                raise RuntimeError("AEGIS: the designated obstacle has no map node")
            self.aegis = AegisFilter(np.vstack([self.graph.nodes[n].points for n in nodes]), p1)
            self.aegis_fit = {"tick": self.tick, "map_nodes": nodes, "points_in": self.aegis.points_in, "points_kept": len(self.aegis.points),
                              "centre": self.aegis.p2.tolist(), "axes": self.aegis.R2.tolist(), "semi_axes_m": self.aegis.Q2.tolist(), "fit_max": self.aegis.fit_max,
                              "z0": self.aegis.z.tolist()}

    def gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Centre and rotation of the gripper ellipsoid: the hand body's rotation (their robot0_eef_quat, the body
        robot_model.eef_name) and the end-effector site (robot0_eef_pos), A:150-158."""
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        robot = self.rs.robots[0]
        R1 = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.eef_name["right"])].reshape(3, 3).copy()
        return gripper_centre(np.array(data.site_xpos[robot.eef_site_id["right"]]), R1), R1

    def step(self, action):
        if getattr(self, "aegis", None) is None:
            return super().step(action)
        model, data = _raw_model(self.rs.sim), _raw_data(self.rs.sim)
        mujoco.mj_forward(model, data)
        a = np.asarray(action, dtype=float).copy()
        Ro = data.site_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ORIGIN_SITE)].reshape(3, 3)
        p1, R1 = self.gripper_pose()
        world = np.r_[Ro @ a[EE_POS_SLICE], Ro @ a[EE_ROT_SLICE]]
        out, info = self.aegis.step(world, p1, R1)
        if info["flag"] > 0:
            a[EE_POS_SLICE], a[EE_ROT_SLICE] = Ro.T @ out[:3], Ro.T @ out[3:]
        self.aegis_tick = {**info, "nominal_action": np.asarray(action, dtype=float), "command": a[EE_POS_SLICE.start:EE_ROT_SLICE.stop].round(6).tolist(),
                           "gripper_centre": p1.round(5).tolist(), "gripper_R": R1.round(6).tolist()}
        return super().step(a)

    def _record(self, rec, action, q, bar) -> None:
        t = getattr(self, "aegis_tick", None)
        if t is not None:
            rec.update(outcome="aegis_filtered" if t["changed"] else ("aegis_passthrough" if t["flag"] > 0 else "aegis_nominal"),
                       qp_status=DAQP_STATUS.get(t["flag"], f"exit_{t['flag']}"), solve_ms=t["solve_ms"], n_pairs=1, n_rows_qp=1,
                       aegis={k: v for k, v in t.items() if k not in ("nominal_action", "solve_ms")})
            action = t["nominal_action"]
            self.aegis_tick = None
        super()._record(rec, action, q, bar)

    def summary(self, success: bool, stalled: bool) -> dict:
        out = super().summary(success, stalled)
        rec = [r for r in self.receipts if "aegis" in r]
        h = [r["aegis"]["h"] for r in rec]
        out["aegis"] = {"fit": getattr(self, "aegis_fit", None), "ticks": len(rec), "filtered_ticks": sum(r["outcome"] == "aegis_filtered" for r in rec),
                        "nominal_passed_ticks": sum(r["outcome"] == "aegis_nominal" for r in rec), "h_min": min(h, default=None),
                        "ticks_h_negative": sum(v < 0 for v in h), "gripper_semi_axes_m": GRIPPER_AXES_M.tolist()}
        return out


class AegisVideo(EvalVideo):
    """EvalVideo and a fifth stream during manipulation: a close view of the gripper ellipsoid and the obstacle
    ellipsoid (red where AEGIS's h < 0), with h and the QP outcome."""

    def __init__(self, env, stem: Path, **kwargs):
        super().__init__(env, stem, **kwargs)
        import imageio.v2 as imageio

        self.paths["ellipsoids"] = stem.parent / f"{stem.name}_ellipsoids.mp4"
        self.writers["ellipsoids"] = imageio.get_writer(str(self.paths["ellipsoids"]), fps=20, codec="libx264", quality=8, macro_block_size=1)

    def __call__(self, env) -> None:
        super().__call__(env)
        rec = env.receipts[-1]
        if "aegis" not in rec:
            return
        w, f = self.world, env.aegis
        p1, R1 = np.asarray(rec["aegis"]["gripper_centre"]), np.asarray(rec["aegis"]["gripper_R"])
        mid = (p1 + f.p2) / 2
        w.cam.lookat[:], w.cam.distance, w.cam.azimuth, w.cam.elevation = mid, 1.1, np.degrees(env.base_yaw()) + 70.0, -25.0
        w.renderer.update_scene(w.data, camera=w.cam, scene_option=w.opt)
        bad = rec["aegis"]["h"] < 0
        w._add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, GRIPPER_AXES_M, p1, R1, (0.95, 0.2, 0.2, 0.45) if bad else (0.2, 0.55, 0.95, 0.45))
        w._add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, f.Q2, f.p2, f.R2 * np.sign(np.linalg.det(f.R2)), (0.95, 0.6, 0.1, 0.45))
        for p in f.points:
            w._add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p, np.eye(3), (0.2, 0.2, 0.2, 0.9))
        w._line(p1, p1 + 0.25 * np.asarray(rec["aegis"]["z"]), (0.1, 0.8, 0.2, 1.0))
        frame = Image.fromarray(w.renderer.render().copy())
        draw = ImageDraw.Draw(frame, "RGBA")
        gt = env.ground_truth[-1]
        draw.rectangle([0, 0, frame.width, 100], fill=(0, 0, 0, 160))
        draw.text((20, 8), f"AEGIS  |  t = {rec['tick'] * 0.05:4.1f} s  |  {rec['outcome']} ({rec['qp_status']})", font=self.font, fill=(255, 255, 255))
        draw.text((20, 56), f"h (ellipsoids) {100 * rec['aegis']['h']:+5.1f} cm   blue: gripper   orange: obstacle   green: z" + ("   CONTACT" if gt["touched"] else ""),
                  font=self.small, fill=(255, 95, 95) if bad or gt["touched"] else (255, 255, 255))
        self.writers["ellipsoids"].append_data(np.asarray(frame))

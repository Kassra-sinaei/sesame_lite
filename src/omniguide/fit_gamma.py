"""The scale factors of OmniGuide's robot points for our drive, fitted from stored Nominal logs (no new runs).

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -s src/omniguide/fit_gamma.py --out <dir>

Their points (G:312-342) integrate a chunk's actions from the pose at the query: gripper position p_k = p_0 +
gamma_pos * (sum of the first k position deltas), orientation R_k = Exp(gamma_rot r_k) R_(k-1), in the base frame
at the query. Their gamma ([0.011, 0.011, 0.02], 0.15) describes robosuite's OSC on their robot; ours is our drive
(joint_drive.py) under the frozen condition (Xiaomi-Robotics-1, arm noise a2).

Data (fixed before the fit). Fit: the stored Nominal runs of the 36-scene campaign (outputs/eval/runs/<scene>/
nominal_full) on the 22 scenes that are not among the 14 comparison scenes. Check: the stored Nominal runs on the six
tuning scenes (outputs/eval_debug/tuning/arm_a2). The robot's joint states (npz qpos) are put into one RoboCasa
model and its kinematics read: the gripper site and hand body in the base centre's frame, as RoboCasa's
base_to_eef observations. Chunks: the 16 ticks after each query of the manipulation segment (a query every 16
ticks from the segment's first tick), complete chunks only. Targets: the gripper's world displacement and the hand's
world rotation over the chunk, in the base frame at the query (what the cost predicts). gamma_pos: least squares per
axis over the chunks in which every action is in arm mode; gamma_rot: the scalar that minimises the mean squared
rotation angle error over the same chunks. Errors are reported over all chunks and over the arm-mode chunks, for the
fitted gamma, their gamma, and no motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

import sesame_lite.tasks  # noqa: F401  (registers the Eval tasks)
from sesame_lite.env import SesameRoboCasaEnv
from sesame_lite.scene_graph import _raw_data, _raw_model

MAIN = Path("/workspace/sesame_lite/outputs")
V3_SCENES = set(("sink_to_counter_s7 sink_to_counter_s11 sink_to_counter_s12 sink_to_counter_s18 sink_to_counter_s19 counter_to_cabinet_s3 counter_to_cabinet_s10 "
                 "counter_to_cabinet_s15 counter_to_cabinet_s17 counter_to_stove_s0 counter_to_stove_s7 counter_to_stove_s9 counter_to_stove_s10 counter_to_stove_s14").split())
TUNING = ("sink_to_counter_s1000", "sink_to_counter_s1001", "counter_to_cabinet_s1003", "counter_to_cabinet_s1004", "counter_to_stove_s1001", "counter_to_stove_s1002")
CHUNK = 16
POS, ROT, MODE = slice(5, 8), slice(8, 11), 4  # env-order action (filter.EE_POS_SLICE, EE_ROT_SLICE, MODE_INDEX)
THEIRS = (np.array([0.011, 0.011, 0.02]), 0.15)
TOP = np.array([0.0, 0.0, -0.25])


class Kinematics:
    """One RoboCasa model; a logged joint state in, the base centre and gripper poses out (world)."""

    def __init__(self, scene_dir: Path):
        scene = json.loads((scene_dir / "scene.json").read_text())
        self.env = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=10)
        self.env.reset(seed=scene["seed"])
        rs = self.env.rs
        self.model, self.data = _raw_model(rs.sim), _raw_data(rs.sim)
        robot = rs.robots[0]
        site = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
        self.base_site, self.eef_site = site(robot.robot_model.base.correct_naming("center")), robot.eef_site_id["right"]
        self.hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.eef_name["right"])
        obs = rs._get_observations(force_update=True)  # check against RoboCasa's own observation at the reset state
        bp, bR, ep, eR = self.poses()
        assert np.allclose(bR.T @ (ep - bp), obs["robot0_base_to_eef_pos"], atol=1e-6)
        assert np.allclose(Rotation.from_matrix(bR.T @ eR).as_quat(), obs["robot0_base_to_eef_quat"], atol=1e-5) or \
            np.allclose(-Rotation.from_matrix(bR.T @ eR).as_quat(), obs["robot0_base_to_eef_quat"], atol=1e-5)

    def poses(self):
        d = self.data
        return d.site_xpos[self.base_site].copy(), d.site_xmat[self.base_site].reshape(3, 3).copy(), d.site_xpos[self.eef_site].copy(), d.xmat[self.hand].reshape(3, 3).copy()

    def run(self, npz: Path) -> list[tuple]:
        z = np.load(npz)
        names, adr = [str(n) for n in z["joint_names"]], z["joint_qposadr"]
        src = [(adr[k], self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for k, n in enumerate(names)
               if n.startswith(("mobilebase0_", "robot0_", "gripper0_"))]
        out = []
        for q in z["qpos"]:
            for a, b in src:
                self.data.qpos[b] = q[a]
            mujoco.mj_kinematics(self.model, self.data)
            out.append(self.poses())
        return out


def chunks(kin: Kinematics, run: Path) -> list[dict]:
    d = json.loads(run.with_suffix(".json").read_text())
    rec = d["receipts"]
    poses = kin.run(run.with_suffix(".npz"))
    vla = [r["tick"] for r in rec if r["segment"] == 1]
    out = []
    for s in range(vla[0], len(poses) - CHUNK, CHUNK) if vla else ():
        bp, bR, ep, eR = poses[s]
        a = np.array([rec[s + i]["action"] for i in range(CHUNK)])
        out.append({"run": str(run), "tick": s, "arm_mode": bool((a[:, MODE] < 0.5).all()), "pos": a[:, POS], "rot": a[:, ROT],
                    "p0": bR.T @ (ep - bp), "R0": bR.T @ eR,
                    "dp": np.array([bR.T @ (poses[s + k][2] - ep) for k in range(1, CHUNK + 1)]),  # world displacement, base frame at the query
                    "R": np.array([bR.T @ poses[s + k][3] for k in range(1, CHUNK + 1)])})
    return out


def predict(c: dict, gp, gr):
    dp = np.cumsum(c["pos"], axis=0) * gp
    R, Rs = c["R0"], []
    for r in c["rot"]:
        R = Rotation.from_rotvec(gr * r).as_matrix() @ R
        Rs.append(R)
    return dp, np.array(Rs)


def errors(cs: list[dict], gp, gr) -> dict:
    """Gripper and top-point world errors [m] (all steps, last step) and rotation angle errors [deg]."""
    eg, et, ea = [], [], []
    for c in cs:
        dp, Rs = predict(c, gp, gr)
        eg.append(np.linalg.norm(dp - c["dp"], axis=1))
        et.append(np.linalg.norm(dp + Rs @ TOP - (c["dp"] + c["R"] @ TOP), axis=1))
        ea.append(np.degrees(np.linalg.norm(Rotation.from_matrix(np.transpose(Rs, (0, 2, 1)) @ c["R"]).as_rotvec(), axis=1)))
    eg, et, ea = np.array(eg), np.array(et), np.array(ea)
    q = lambda v: {"rms": round(float(np.sqrt((v**2).mean())), 4), "p95": round(float(np.percentile(v, 95)), 4)}  # noqa: E731
    return {"chunks": len(cs), "gripper_m_all_steps": q(eg), "gripper_m_last_step": q(eg[:, -1]), "top_point_m_all_steps": q(et), "top_point_m_last_step": q(et[:, -1]),
            "rotation_deg_all_steps": q(ea), "rotation_deg_last_step": q(ea[:, -1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scenes = sorted(p.name for p in (MAIN / "eval/scenes_v2").iterdir() if p.name not in V3_SCENES)
    fit_runs = [MAIN / "eval/runs" / s / "nominal_full" for s in scenes]
    check_runs = [MAIN / "eval_debug/tuning/arm_a2" / s / "nominal_full" for s in TUNING]
    kin = Kinematics(MAIN / "eval/scenes_v2" / scenes[0])
    fit = [c for r in fit_runs for c in chunks(kin, r)]
    check = [c for r in check_runs for c in chunks(kin, r)]
    arm = [c for c in fit if c["arm_mode"]]
    A = np.concatenate([np.cumsum(c["pos"], axis=0) for c in arm])
    Y = np.concatenate([c["dp"] for c in arm])
    gp = (A * Y).sum(axis=0) / (A * A).sum(axis=0)

    def rot_loss(gr):
        return float(np.mean([np.sum(np.linalg.norm(Rotation.from_matrix(np.transpose(predict(c, gp, gr)[1], (0, 2, 1)) @ c["R"]).as_rotvec(), axis=1) ** 2) for c in arm]))

    gr = float(minimize_scalar(rot_loss, bounds=(0.0, 0.5), method="bounded", options={"xatol": 1e-5}).x)
    sets = {"fit_all": fit, "fit_arm_mode": arm, "check_all": check, "check_arm_mode": [c for c in check if c["arm_mode"]]}
    out = {"gamma_pos": gp.round(5).tolist(), "gamma_rot": round(gr, 4), "theirs": {"gamma_pos": THEIRS[0].tolist(), "gamma_rot": THEIRS[1]},
           "fit_runs": [str(r) for r in fit_runs], "check_runs": [str(r) for r in check_runs],
           "errors": {name: {"fitted": errors(cs, gp, gr), "theirs": errors(cs, *THEIRS), "no_motion": errors(cs, np.zeros(3), 0.0)} for name, cs in sets.items()}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gamma_fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gamma_pos", "gamma_rot")}), json.dumps(out["errors"]["check_all"], indent=1))
    kin.env.close()


if __name__ == "__main__":
    main()

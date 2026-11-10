"""The unfiltered robot against robosuite's own controller (spec v4, section 6.2).

    MUJOCO_GL=egl .venv/bin/python -s scripts/compare_drive.py --episode outputs/demos/close_fridge/CloseFridge_s100_nofilter.json

Takes an episode run with --no-filter (the arm on JointVelocityDrive, driven by u_nom), rebuilds the same scene from
the seed, replays the same actions through robosuite's stock controllers (OSC_POSE on the arm), and compares the joint
trajectories tick by tick, over the whole run and up to the first contact of the robot with anything (after a contact
the two runs can part for reasons that are not the drive). Writes <episode>_vs_osc.json and <episode>_vs_osc.pdf next to the episode.
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.robot_spheres import JOINTS, RobotSpheres  # noqa: E402
from sesame_lite.scene_graph import _raw_data, _raw_model, _subtree_body_ids  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"], "font.weight": "normal", "axes.titleweight": "normal",
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "savefig.dpi": 600})

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True)
ap.add_argument("--replay-drive", action="store_true", help="replay the actions through the current JointVelocityDrive too, instead of the episode's own joints")
args = ap.parse_args()
path = Path(args.episode)
ep = json.loads(path.read_text())
res = ep["result"]
if res["filter"]:
    raise ValueError("compare an episode run with --no-filter")
q_drive = np.load(path.with_suffix(".npz"))["q"]
actions = np.array([r["action"] for r in ep["receipts"]])
plan = ep["plan"]

env = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=len(actions) + 1)
env.reset(seed=res["seed"])
model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
robot = RobotSpheres(model, data)
assert type(env.rs.robots[0].composite_controller.part_controllers["right"]).__name__ == "OperationalSpaceController"
robot_bodies = _subtree_body_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env.rs.robots[0].robot_model.root_body))
q_osc, first_contact = [], None
for k, a in enumerate(actions):
    mujoco.mj_forward(model, data)
    q_osc.append(robot.q(data))
    env.step(a.astype(np.float32))
    if first_contact is None and any((model.geom_bodyid[c.geom1] in robot_bodies) != (model.geom_bodyid[c.geom2] in robot_bodies) for c in data.contact[: data.ncon]):
        first_contact = k + 1  # the robot touches something (any object, exempt ones included) from this tick on
q_osc = np.array(q_osc)
env.close()
if args.replay_drive:  # the same actions through the drive (no policy in the loop)
    base = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=len(actions) + 1)
    fenv = SafetyFilterEnv(base, filter=False)
    fenv.reset(seed=res["seed"], plan=plan)
    for a in actions:
        fenv.step(a.astype(np.float32))
    q_drive = np.array(fenv.q_log)
    base.close()

if not np.allclose(q_osc[0], q_drive[0], atol=1e-9):
    raise RuntimeError(f"the replay does not start where the episode started: {np.abs(q_osc[0] - q_drive[0]).max()}")
diff = q_drive - q_osc
arm, base = slice(4, 11), slice(0, 3)
pre = slice(0, first_contact if first_contact is not None else len(actions))
out = {"episode": str(path), "ticks": len(actions), "first_contact_tick_osc": first_contact,
       "before_contact": {"arm_max_abs_diff_rad": float(np.abs(diff[pre, arm]).max()), "base_max_abs_diff": float(np.abs(diff[pre, base]).max()),
                          "max_abs_diff_per_joint": dict(zip(JOINTS, np.abs(diff[pre]).max(axis=0).round(6).tolist()))},
       "max_abs_diff_per_joint": dict(zip(JOINTS, np.abs(diff).max(axis=0).round(6).tolist())),
       "rms_diff_per_joint": dict(zip(JOINTS, np.sqrt((diff**2).mean(axis=0)).round(6).tolist())),
       "arm_max_abs_diff_rad": float(np.abs(diff[:, arm]).max()), "base_max_abs_diff": float(np.abs(diff[:, base]).max()),
       "arm_range_of_motion_rad": float(np.ptp(q_osc[:, arm], axis=0).max())}
path.with_name(path.stem + ("_replay" if args.replay_drive else "") + "_vs_osc.json").write_text(json.dumps(out, indent=1))
t = 0.05 * np.arange(len(actions))
fig, axs = plt.subplots(2, 5, figsize=(7.16, 3.0), layout="constrained", sharex=True)
for ax, k in zip(axs.ravel(), [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]):
    ax.plot(t, q_osc[:, k], color="#9A9A9A", lw=2.0, label="robosuite OSC")
    ax.plot(t, q_drive[:, k], color="black", lw=0.7, label="joint-velocity drive, u = u_nom")
    ax.set_title(JOINTS[k].replace("mobilebase0_joint_mobile_", "base ").replace("robot0_", "") + (" [m]" if k < 2 else " [rad]"), loc="left")
    ax.ticklabel_format(axis="y", useOffset=False)
    if first_contact is not None:
        ax.axvline(0.05 * first_contact, color="#C0392B", lw=0.6, ls="--", label="first contact (robosuite run)")
for ax in axs[1]:
    ax.set_xlabel("t [s]")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
fig.savefig(path.with_name(path.stem + ("_replay" if args.replay_drive else "") + "_vs_osc.pdf"))
print("COMPARE " + json.dumps(out))

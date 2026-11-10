"""Response of RoboCasa's base to its command, per base joint (spec v4, section 6.2: the filter's base model).

    MUJOCO_GL=egl .venv/bin/python -s scripts/calibrate_base.py

robosuite's base controller turns the command onto the forward slide, side slide and yaw joints, whose velocity
actuators (kv) drive the base against joint friction. Two things are measured per joint, with the base moved to free space next to the
nav_clutter kitchen (seed 100; nothing within 1.5 m at any heading; every trial is checked for contact with anything but the floor):
  - steady: the steady joint velocity for each actuator target in a grid (the actuator target is the command times
    ctrlrange), from rest, after 0.6 s;
  - beta: the lag over one tick, fitted on every tick of the step responses:
        mean velocity over the tick = beta * velocity at the tick's start + (1 - beta) * steady(target)
Writes config/base_response.json (used by filter.BaseModel) and outputs/viz/base_response.pdf.
"""

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
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.scene_graph import _subtree_body_ids  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"], "font.weight": "normal", "axes.titleweight": "normal",
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "savefig.dpi": 600})
JOINTS = ["mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"]
CTRLRANGE = np.array([1.0, 1.0, 1.5])
COMMANDS = np.r_[0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OPEN_OFFSET = {0: (-6.0, 0.0), 1: (-6.0, 0.0), 2: (-6.0, 0.0)}  # (forward, side) shift of the start pose to free space (outside the room; the base never touches the floor)
STEP, HOLD, RELEASE = 2, 12, 6  # ticks at rest, at the command, at zero again
OUT = Path(__file__).resolve().parents[1]

env = RoboCasaGymEnv(env_name="NavigateKitchenClutter", split="all", seed=100, camera_widths=32, camera_heights=32, clutter_mode=1)


def trial(j: int, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, set]:
    """Per tick: actuator target, mean joint velocity over the tick, joint velocity at its start; robot contacts."""
    env.reset(seed=100)  # the same kitchen every trial
    rs = env.env
    m, d = rs.sim.model, rs.sim.data
    qa, va = [m.get_joint_qpos_addr(n) for n in JOINTS], [m.get_joint_qvel_addr(n) for n in JOINTS]
    d.qpos[qa[0]] += OPEN_OFFSET[j][0]
    d.qpos[qa[1]] += OPEN_OFFSET[j][1]
    mujoco.mj_forward(m._model, d._data)
    robot = _subtree_body_ids(m._model, m.body_name2id(rs.robots[0].robot_model.root_body))
    floor = {b for b in range(m.nbody) if "floor" in (m.body_id2name(b) or "")}
    contacts = set()
    target, mean, start = [], [], []
    for k in range(STEP + HOLD + RELEASE):
        cmd = c if STEP <= k < STEP + HOLD else 0.0
        a = {"action.base_motion": np.zeros(4), "action.control_mode": np.ones(1), "action.end_effector_position": np.zeros(3),
             "action.end_effector_rotation": np.zeros(3), "action.gripper_close": np.zeros(1)}
        a["action.base_motion"][j] = cmd
        q0, v0 = float(d.qpos[qa[j]]), float(d.qvel[va[j]])
        env.step(a)
        target.append(cmd * CTRLRANGE[j])
        mean.append((float(d.qpos[qa[j]]) - q0) / rs.control_timestep)
        start.append(v0)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if (b1 in robot) != (b2 in robot) and b1 not in floor and b2 not in floor:
                contacts.add((m.body_id2name(b1), m.body_id2name(b2)))
    return np.array(target), np.array(mean), np.array(start), contacts


result, fig_data = {"commands": COMMANDS.tolist(), "joints": {}}, {}
for j, name in enumerate(JOINTS):
    steady, rows = [], []
    for c in COMMANDS:
        t, mv, v0, ncon = trial(j, c)
        if ncon:
            raise RuntimeError(f"{name}, command {c}: the robot touched the kitchen: {sorted(ncon)}; move OPEN_OFFSET")
        steady.append(float(np.mean(mv[STEP + HOLD - 3: STEP + HOLD])))
        rows.append((t, mv, v0))
    targets = COMMANDS * CTRLRANGE[j]
    steady = np.array(steady)
    if np.any(np.diff(steady) < -1e-3):
        raise RuntimeError(f"{name}: steady velocity not monotone in the command: {steady.round(4).tolist()}")
    s_of = lambda x: np.sign(x) * np.interp(np.abs(x), np.r_[0.0, targets], np.r_[0.0, steady])  # noqa: E731
    # beta: least squares over every tick after the first command, mean = beta v0 + (1 - beta) steady(target)
    A, y = [], []
    for t, mv, v0 in rows:
        for k in range(STEP, len(t)):
            ss = s_of(t[k])
            A.append(v0[k] - ss)
            y.append(mv[k] - ss)
    A, y = np.array(A), np.array(y)
    beta = float(A @ y / (A @ A))
    resid = y - beta * A
    result["joints"][name] = {"ctrlrange": float(CTRLRANGE[j]), "targets": [0.0, *targets.tolist()], "steady": [0.0, *steady.tolist()], "beta": beta,
                              "fit_residual_rms": float(np.sqrt(np.mean(resid**2))), "fit_residual_max": float(np.abs(resid).max())}
    fig_data[name] = (targets, steady, rows, beta, s_of)
    print(name, "beta", round(beta, 4), "residual rms/max", round(result["joints"][name]["fit_residual_rms"], 4), round(result["joints"][name]["fit_residual_max"], 4))
(OUT / "config" / "base_response.json").write_text(json.dumps(result, indent=1))

fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.0), layout="constrained")
for j, name in enumerate(JOINTS):
    targets, steady, rows, beta, s_of = fig_data[name]
    unit = "rad/s" if j == 2 else "m/s"
    ax = axs[0, j]
    ax.plot(targets, steady, "o-", color="black", ms=2.5, lw=0.8, label="measured")
    ax.plot(targets, np.maximum(targets - [0.25, 0.25, 250 / 1500][j], 0), color="#9A9A9A", lw=1.5, ls="--", label="target - frictionloss / kv")
    ax.set_xlabel(f"actuator target [{unit}]")
    ax.set_ylabel(f"steady velocity [{unit}]")
    ax.set_title(name.replace("mobilebase0_joint_mobile_", "base "), loc="left")
    ax.legend(frameon=False)
    ax = axs[1, j]
    for t, mv, v0 in rows[::3]:
        pred = np.r_[mv[:STEP], [beta * v0[k] + (1 - beta) * s_of(t[k]) for k in range(STEP, len(t))]]
        ax.plot(np.arange(len(t)), mv, color="black", lw=0.8)
        ax.plot(np.arange(len(t)), pred, color="#C0392B", lw=0.6, ls="--")
    ax.set_xlabel("tick")
    ax.set_ylabel(f"mean velocity over the tick [{unit}]")
    ax.set_title(rf"steps (black), model (red), $\beta$ = {beta:.3f}", loc="left")
(OUT / "outputs" / "viz").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "outputs" / "viz" / "base_response.pdf")
print("wrote config/base_response.json and outputs/viz/base_response.pdf")

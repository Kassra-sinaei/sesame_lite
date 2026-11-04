"""Point-to-point navigation benchmark: RoboCasa365 NavigateKitchen in the environment REKI was validated in
(robocasa 1.0.1, pretrain-split kitchens, REKI's episode seeds, the registry horizon and the task's own
success check). Static-map lattice planner + holonomic tracker, no safety filter.

    PYTHONPATH=$HOME/Documents/Chewy/code/third_party/robocasa-1.0.1 MUJOCO_GL=egl \
        python -s scripts/eval_nav.py --episodes 20 --video 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # string hashing orders a few RoboCasa choices; fix it before anything is imported
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import robocasa  # noqa: F401  (registers robocasa/<Task>)
from robocasa.utils.dataset_registry_utils import get_task_horizon

from sesame_lite.determinism import begin_episode, seed_fallback_generators
from sesame_lite.navigation import Z_BANDS, ConfigurationSpace, HolonomicTracker, NoPathError, StaticMap, _robot_body_ids, base_pose, plan_path, robot_footprint, wrap
from sesame_lite.scene_graph import _raw_data, _raw_model

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed", type=int, default=42, help="episode seeds drawn as in REKI's rollout.py")
ap.add_argument("--video", type=int, default=3, help="record the first N episodes")
ap.add_argument("--max-replans", type=int, default=3)
ap.add_argument("--out", default="outputs/nav_eval")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

TASK = "NavigateKitchen"
HORIZON = int(get_task_horizon(TASK))
CAP = 2 * HORIZON  # keep driving past the official horizon to measure time-to-goal
seed_fallback_generators()
begin_episode(args.seed)
env = gym.make(f"robocasa/{TASK}", disable_env_checker=True, split="pretrain")
rng = np.random.default_rng(args.seed)


def action_dict(a: np.ndarray) -> dict:
    return {"action.base_motion": np.r_[a[0:3], 0.0], "action.control_mode": a[4:5], "action.end_effector_position": a[5:8],
            "action.end_effector_rotation": a[8:11], "action.gripper_close": a[11:12]}


def robot_contacts(m, d, robot: set[int]) -> list[str]:
    """``robot body > other body @ z`` for every penetrating robot contact (floor excluded)."""
    hits = []
    for c in d.contact[: d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if c.dist >= 0 or (b1 in robot) == (b2 in robot):
            continue
        g, gr = (c.geom2, c.geom1) if b1 in robot else (c.geom1, c.geom2)
        if d.geom_xpos[g][2] + m.geom_rbound[g] < Z_BANDS[0] + 0.05:
            continue
        name = lambda geom: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[geom])) or "?"
        hits.append(f"{name(gr)} > {name(g)} @ z={c.pos[2]:.2f}")
    return hits


class TopView:
    """Top-down free camera over the mission, drawn through robosuite's own offscreen context (a second GL
    context in the same process crashes after a reset), with the planned path and goal as scene geoms."""

    def __init__(self, rs, start, goal, size=480):
        self.ctx, self.size = rs.sim._render_context_offscreen, size
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [*(0.5 * (start[:2] + goal[:2])), 0.5]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = max(4.0, 1.6 * np.linalg.norm(goal[:2] - start[:2]) + 2.0), -90.0, 90.0

    def __call__(self, path, goal):
        ctx, n = self.ctx, self.size
        ctx.gl_ctx.make_current()
        if n > ctx.con.offWidth or n > ctx.con.offHeight:
            ctx.update_offscreen_size(max(n, ctx.con.offWidth), max(n, ctx.con.offHeight))
        scn = ctx.scn
        mujoco.mjv_updateScene(ctx.model._model, ctx.data._data, ctx.vopt, ctx.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        for a, b in zip(path[:-1], path[1:]):
            if scn.ngeom >= scn.maxgeom - 1:
                break
            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, np.r_[a[:2], 0.05], np.r_[b[:2], 0.05])
            scn.geoms[scn.ngeom].rgba[:] = (0.1, 0.3, 0.9, 0.9)
            scn.ngeom += 1
        tip = goal[:2] + 0.4 * np.array([np.cos(goal[2]), np.sin(goal[2])])
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.03, np.r_[goal[:2], 0.06], np.r_[tip, 0.06])
        scn.geoms[scn.ngeom].rgba[:] = (0.1, 0.8, 0.3, 0.9)
        scn.ngeom += 1
        viewport = mujoco.MjrRect(0, 0, n, n)
        mujoco.mjr_render(viewport, scn, ctx.con)
        img = np.empty((n, n, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, viewport, ctx.con)
        return img[::-1]


def map_figure(path_png, smap, cspace, plans, traj, start, goal, result):
    occ = smap.occupancy.any(axis=0).T
    ext = [smap.origin[0], smap.origin[0] + smap.resolution * occ.shape[1], smap.origin[1], smap.origin[1] + smap.resolution * occ.shape[0]]
    k = cspace.yaw_bin(goal[2])
    fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.imshow(occ, origin="lower", extent=ext, cmap="Greys", alpha=0.8, interpolation="nearest")
    ax.contour(cspace.obstacle[k].T, levels=[0.5], origin="lower", extent=ext, colors="tab:orange", linewidths=0.6)
    for i, p in enumerate(plans):
        ax.plot(p[:, 0], p[:, 1], "-o", ms=2, lw=1.2, color="tab:blue", alpha=1.0 if i == len(plans) - 1 else 0.4)
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "--", lw=1.0, color="tab:green")
    for p, c in ((start, "k"), (goal, "tab:red")):
        ax.arrow(p[0], p[1], 0.3 * np.cos(p[2]), 0.3 * np.sin(p[2]), width=0.03, color=c)
    ax.set(title=f"seed {result['seed']}: {result['outcome']}", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    lo, hi = np.minimum(start[:2], goal[:2]) - 1.5, np.maximum(start[:2], goal[:2]) + 1.5
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


results = []
for ep in range(args.episodes):
    seed = int(rng.integers(0, 2**31 - 1))
    begin_episode(seed)
    env.reset(seed=seed)
    rs = env.unwrapped.env
    m, d = _raw_model(rs.sim), _raw_data(rs.sim)
    robot = _robot_body_ids(m)
    start = base_pose(rs)
    goal = np.array([rs.target_pos[0], rs.target_pos[1], rs.target_ori[2]])
    t0 = time.perf_counter()
    smap = StaticMap.from_sim(rs)
    cspace = ConfigurationSpace(smap, robot_footprint(rs))
    t_map = time.perf_counter() - t0
    r = {"episode": ep, "seed": seed, "instruction": rs.get_ep_meta()["lang"], "start": start.tolist(), "goal": goal.tolist(),
         "straight_m": float(np.linalg.norm(goal[:2] - start[:2])), "clearance_start_m": cspace.clearance_at(start), "clearance_goal_m": cspace.clearance_at(goal),
         "map_s": t_map, "plan_s": [], "replans": 0, "contact_steps": 0, "contact_bodies": [], "steps_to_success": None}
    plans, traj, frames = [], [], []
    try:
        t0 = time.perf_counter()
        plans.append(plan_path(cspace, start, goal))
        r["plan_s"].append(time.perf_counter() - t0)
        r["goal_offset_m"] = float(np.linalg.norm(plans[0][-1, :2] - goal[:2]))
    except NoPathError as e:
        r.update(outcome="no_path", reason=str(e))
    if plans:
        tracker = HolonomicTracker(plans[-1], init_yaw=start[2])
        view = TopView(rs, start, goal) if ep < args.video else None
        best_progress, stall = 0.0, 0
        r["outcome"] = "timeout"
        for step in range(CAP):
            pose = base_pose(rs)
            traj.append(pose)
            if tracker.deviation(pose) > 0.15 or stall > 60:
                if r["replans"] >= args.max_replans:
                    r["outcome"] = "stalled" if stall > 60 else "lost_track"
                    break
                try:
                    t0 = time.perf_counter()
                    plans.append(plan_path(cspace, pose, goal))
                    r["plan_s"].append(time.perf_counter() - t0)
                except NoPathError as e:
                    r.update(outcome="no_path_replan", reason=str(e))
                    break
                tracker, r["replans"], best_progress, stall = HolonomicTracker(plans[-1], init_yaw=start[2]), r["replans"] + 1, 0.0, 0
            _, _, _, _, info = env.step(action_dict(tracker(pose)))
            hits = robot_contacts(m, d, robot)
            if hits:
                r["contact_steps"] += 1
                r["contact_bodies"] = sorted(set(r["contact_bodies"]) | {h.split(" > ")[1].split(" @")[0] for h in hits})
                if "first_contact" not in r:
                    p = base_pose(rs)
                    r["first_contact"] = {"step": step, "pose": p.tolist(), "cspace_clearance_m": cspace.clearance_at(p), "pairs": sorted(set(hits))}
            if info["success"] and r["steps_to_success"] is None:
                r["steps_to_success"] = step + 1
            if view is not None and step % 2 == 0:
                frames.append(view(plans[-1], goal))
            stall = 0 if tracker.progress > best_progress + 0.02 else stall + (not tracker.at_goal(pose))
            best_progress = max(best_progress, tracker.progress)
            if tracker.at_goal(base_pose(rs)):
                r["outcome"] = "reached"
                break
    final = base_pose(rs)
    r.update(final=final.tolist(), final_err_m=float(np.linalg.norm(final[:2] - goal[:2])), final_yaw_err_deg=float(np.degrees(abs(wrap(final[2] - goal[2])))),
             success=bool(r["steps_to_success"] is not None and r["steps_to_success"] <= HORIZON), path_m=float(np.sum(np.linalg.norm(np.diff(np.array(traj)[:, :2], axis=0), axis=1))) if len(traj) > 1 else 0.0)
    results.append(r)
    map_figure(out / f"ep{ep:02d}_map.png", smap, cspace, plans, np.array(traj), start, goal, r)
    if frames:
        imageio.mimsave(out / f"ep{ep:02d}_{r['outcome']}.mp4", frames, fps=10)
    print(f"ep {ep:2d} seed {seed}: {r['outcome']:<10} success={r['success']} t={r['steps_to_success']} straight={r['straight_m']:.2f}m "
          f"err={r['final_err_m']:.3f}m/{r['final_yaw_err_deg']:.1f}deg contacts={r['contact_steps']} {r['contact_bodies'][:3]} replans={r['replans']} "
          f"goal_offset={r.get('goal_offset_m', float('nan')):.2f} map={r['map_s']:.1f}s plan={sum(r['plan_s']):.1f}s", flush=True)
    if "first_contact" in r:
        print("   first contact:", r["first_contact"], flush=True)

n = len(results)
summary = {"task": TASK, "split": "pretrain", "horizon": HORIZON, "episodes": n, "success": sum(r["success"] for r in results),
           "reached": sum(r["outcome"] == "reached" for r in results), "episodes_with_contact": sum(r["contact_steps"] > 0 for r in results)}
print("SUMMARY", json.dumps(summary))
(out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
env.close()

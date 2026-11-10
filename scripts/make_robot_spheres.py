"""The PandaOmron's sphere model for cuRobo (spec v4, section 4), and the checks the spec asks for before anything uses it.

    MUJOCO_GL=egl .venv/bin/python -s scripts/make_robot_spheres.py [--refit]

Writes
  config/panda_omron.urdf          the kinematic tree of the MuJoCo model (bodies, joints, offsets), so cuRobo and MuJoCo agree
  config/panda_omron_spheres.yml   cuRobo robot config: arm and hand spheres from cuRobo's Franka set, carried from the Franka
                                   URDF's link frames into robosuite's body frames at the zero configuration; base box and
                                   column cylinders: spheres fitted to their collision geometry by cuRobo's own sphere fit
                                   (fit_spheres_to_mesh, MorphIt, default settings)
  outputs/viz/robot_spheres.json   the checks: centre and Jacobian differences cuRobo vs MuJoCo over random configurations,
                                   coverage of the collision geometry by the spheres and how far the spheres stick out
  outputs/viz/robot_spheres.pdf    the robot in three poses with the spheres over its real geometry
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.robot_spheres import JOINTS, SPHERES_YML, URDF, RobotSpheres  # noqa: E402
from sesame_lite.scene_graph import _raw_data, _raw_model, _subtree_body_ids  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"], "font.weight": "normal", "axes.titleweight": "normal",
                     "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "savefig.dpi": 600})
CUROBO = Path(__file__).resolve().parents[1] / "third_party" / "curobo" / "curobo" / "content"
FRANKA_TO_ROBOSUITE = {**{f"panda_link{k}": f"robot0_link{k}" for k in range(8)}, "panda_hand": "robot0_right_hand",
                       "panda_leftfinger": "gripper0_right_leftfinger", "panda_rightfinger": "gripper0_right_rightfinger"}
OUT = Path("outputs/viz")


def name(m, kind, i):
    return mujoco.mj_id2name(m, kind, i)


def pose(pos, quat):
    T = np.eye(4)
    T[:3, 3] = pos
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    T[:3, :3] = R.reshape(3, 3)
    return T


def rpy(R):
    sy = -R[2, 0]
    return [float(np.arctan2(R[2, 1], R[2, 2])), float(np.arcsin(np.clip(sy, -1, 1))), float(np.arctan2(R[1, 0], R[0, 0]))]


def origin(el, T):
    o = ET.SubElement(el, "origin")
    o.set("xyz", " ".join(f"{v:.6f}" for v in T[:3, 3]))
    o.set("rpy", " ".join(f"{v:.6f}" for v in rpy(T[:3, :3])))


# ------------------------------------------------------------------ the robot in MuJoCo
plan = load_plan("plans/close_fridge.json")
base = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=plan.get("horizon"))
env = SafetyFilterEnv(base, filter=False)
env.reset(seed=100, plan=plan)
m, d = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, env.rs.robots[0].robot_model.root_body)
bodies = sorted(_subtree_body_ids(m, root))

# ------------------------------------------------------------------ 1. URDF from the MuJoCo bodies and joints
robot = ET.Element("robot", name="panda_omron")
for b in bodies:
    ET.SubElement(robot, "link", name=name(m, mujoco.mjtObj.mjOBJ_BODY, b))
for b in bodies:
    if b == root:
        continue
    child, parent = name(m, mujoco.mjtObj.mjOBJ_BODY, b), name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b]))
    joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    T = pose(m.body_pos[b], m.body_quat[b])
    if not joints:
        jt = ET.SubElement(robot, "joint", name=f"{child}_fixed", type="fixed")
        ET.SubElement(jt, "parent", link=parent)
        ET.SubElement(jt, "child", link=child)
        origin(jt, T)
        continue
    # several joints on one body (the base: forward, side, yaw): a chain of helper links, each joint about its anchor
    prev, T_prev = parent, T
    for k, j in enumerate(joints):
        jname, anchor = name(m, mujoco.mjtObj.mjOBJ_JOINT, j), np.asarray(m.jnt_pos[j], float)
        last = k == len(joints) - 1
        to_anchor = np.eye(4)
        to_anchor[:3, 3] = anchor
        pivot = f"{jname}_pivot"
        ET.SubElement(robot, "link", name=pivot)
        jt = ET.SubElement(robot, "joint", name=f"{jname}_anchor", type="fixed")
        ET.SubElement(jt, "parent", link=prev)
        ET.SubElement(jt, "child", link=pivot)
        origin(jt, T_prev @ to_anchor)
        moved = f"{jname}_link"
        ET.SubElement(robot, "link", name=moved)
        kind = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}[int(m.jnt_type[j])]
        jt = ET.SubElement(robot, "joint", name=jname, type=kind)
        ET.SubElement(jt, "parent", link=pivot)
        ET.SubElement(jt, "child", link=moved)
        origin(jt, np.eye(4))
        ET.SubElement(jt, "axis", xyz=" ".join(f"{v:.6f}" for v in m.jnt_axis[j]))
        lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-100.0, 100.0))
        ET.SubElement(jt, "limit", lower=f"{lo:.6f}", upper=f"{hi:.6f}", effort="1000", velocity="10")
        back = np.eye(4)
        back[:3, 3] = -anchor
        prev, T_prev = moved, back
        if last:
            jt = ET.SubElement(robot, "joint", name=f"{child}_from_{jname}", type="fixed")
            ET.SubElement(jt, "parent", link=prev)
            ET.SubElement(jt, "child", link=child)
            origin(jt, T_prev)
ET.indent(robot)
URDF.write_text(ET.tostring(robot, encoding="unicode"))

# ------------------------------------------------------------------ 2. spheres
# body transforms at the zero configuration, relative to robot0_link0 (MuJoCo) and panda_link0 (Franka URDF)
q0 = d.qpos.copy()
adr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in JOINTS]
d.qpos[adr[3:]] = 0.0
mujoco.mj_kinematics(m, d)


def world_T(b):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    return T


link0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_link0")
link0_T_body = {name(m, mujoco.mjtObj.mjOBJ_BODY, b): np.linalg.inv(world_T(link0)) @ world_T(b) for b in bodies}
import yourdfpy  # noqa: E402

franka = yourdfpy.URDF.load(str(CUROBO / "assets" / "robot" / "franka_description" / "franka_panda.urdf"), load_meshes=False)
franka_cfg = yaml.safe_load((CUROBO / "configs" / "robot" / "franka.yml").read_text())["robot_cfg"]["kinematics"]
spheres = {}
for flink, rlink in FRANKA_TO_ROBOSUITE.items():
    T = np.linalg.inv(link0_T_body[rlink]) @ franka.get_transform(flink, "panda_link0")  # franka link -> robosuite body
    spheres[rlink] = [{"center": [round(float(v), 5) for v in (T[:3, :3] @ s["center"] + T[:3, 3])], "radius": float(s["radius"])}
                      for s in franka_cfg["collision_spheres"][flink] if s["radius"] > 0]


def fitted_spheres(mesh, R, p):
    """Spheres for one base or column collision geom, from cuRobo's own fit (MorphIt, default settings): the geom as a
    trimesh in its local frame, the fitted spheres moved into the body frame. Returns the spheres and cuRobo's metrics."""
    import torch
    from curobo.sphere_fit import fit_spheres_to_mesh

    torch.manual_seed(0)  # MorphIt starts from random samples (torch and numpy): seeded, so the config is reproducible
    np.random.seed(0)
    fit = fit_spheres_to_mesh(mesh, compute_metrics=True)
    centres, radii = fit.centers.cpu().numpy().astype(float), fit.radii.cpu().numpy().astype(float)
    keep = radii > 0
    out = [{"center": [round(float(v), 5) for v in R @ c + p], "radius": round(float(r), 5)} for c, r in zip(centres[keep], radii[keep])]
    metrics = {k: (float(v) if v is not None else None) for k, v in vars(fit.metrics).items()} if fit.metrics is not None else {}
    return out, metrics


# cuRobo's MorphIt fit is not reproducible on the GPU (the sphere count varied between 316 and 326 over runs), so the
# config is a committed data file: refit only with --refit; otherwise the checks run on the committed spheres
REFIT = "--refit" in sys.argv or not SPHERES_YML.exists()
base_geoms = {}
for b in (bodies if REFIT else []):
    bname = name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if not bname.startswith("mobilebase0"):
        continue
    for g in np.flatnonzero(m.geom_bodyid == b):
        if m.geom_group[g] != 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R, p, s = R.reshape(3, 3), np.asarray(m.geom_pos[g], float), m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * s[:3])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=48)
        else:
            raise ValueError(f"no sphere rule for geom type {m.geom_type[g]} ({name(m, mujoco.mjtObj.mjOBJ_GEOM, g)})")
        new, metrics = fitted_spheres(mesh, R, p)
        spheres.setdefault(bname, []).extend(new)
        base_geoms[name(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = {"spheres": len(new), "curobo_fit_metrics": metrics}

cfg = {"robot_cfg": {"kinematics": {
    "format_version": 2.0, "base_link": name(m, mujoco.mjtObj.mjOBJ_BODY, root), "tool_frames": ["gripper0_right_eef"],
    "collision_link_names": list(spheres), "collision_spheres": spheres, "collision_sphere_buffer": 0.0,
    "self_collision_ignore": {}, "self_collision_buffer": {}, "use_global_cumul": True,
    "cspace": {"joint_names": list(JOINTS), "default_joint_position": [0.0] * len(JOINTS), "null_space_weight": [1.0] * len(JOINTS),
               "cspace_distance_weight": [1.0] * len(JOINTS), "max_acceleration": 15.0, "max_jerk": 500.0},
}}}
if REFIT:
    SPHERES_YML.write_text(yaml.safe_dump(cfg, sort_keys=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robot_spheres_fit.json").write_text(json.dumps(base_geoms, indent=1))  # cuRobo's fit metrics of the base spheres
else:
    spheres = yaml.safe_load(SPHERES_YML.read_text())["robot_cfg"]["kinematics"]["collision_spheres"]

# ------------------------------------------------------------------ 3. checks against MuJoCo
d.qpos[:] = q0
mujoco.mj_forward(m, d)
rs = RobotSpheres(m, d)
local = [(link, np.array(s["center"])) for link, ss in spheres.items() for s in ss]
link_id = {link: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link) for link in spheres}
rng = np.random.default_rng(0)
lo_hi = [(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] if m.jnt_limited[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] else (-1.0, 1.0)) for j in JOINTS]
dpos, djac = 0.0, 0.0
qs = []
for trial in range(50):
    q = np.array([rng.uniform(lo, hi) for lo, hi in lo_hi])
    q[:2] += q0[adr[:2]]  # around the robot's place in the kitchen
    qs.append(q)
    d.qpos[adr] = q
    mujoco.mj_forward(m, d)
    c, r, J = rs.jacobians(q)
    for k, (link, p_local) in enumerate(local):
        b = link_id[link]
        p_world = d.xpos[b] + d.xmat[b].reshape(3, 3) @ p_local
        dpos = max(dpos, float(np.linalg.norm(c[k] - p_world)))
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, p_world, b)
        dofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in JOINTS]
        djac = max(djac, float(np.max(np.abs(J[k] - jacp[:, dofs]))))

# coverage of the collision geometry and protrusion of the spheres (at the robot's start configuration)
d.qpos[:] = q0
mujoco.mj_forward(m, d)
c, r = rs.spheres(rs.q(d))
surf, hulls, surf_body = [], [], []
for b in bodies:
    for g in np.flatnonzero(m.geom_bodyid == b):
        if m.geom_group[g] != 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R, p, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ R.T + p
            hull = ConvexHull(v)  # MuJoCo collides with the convex hull of a mesh
            tri = v[hull.simplices]
            w = rng.dirichlet(np.ones(3), size=(len(tri), 8))
            pts = np.einsum("tkj,tjd->tkd", w, tri).reshape(-1, 3)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            u = rng.uniform(-1, 1, (4000, 3))
            u[np.arange(4000), rng.integers(0, 3, 4000)] = rng.choice([-1, 1], 4000)
            pts = (u * s[:3]) @ R.T + p
            hull = ConvexHull(np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s[:3] @ R.T + p)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            a = rng.uniform(0, 2 * np.pi, 4000)
            z = rng.uniform(-s[1], s[1], 4000)
            pts = np.c_[s[0] * np.cos(a), s[0] * np.sin(a), z] @ R.T + p
            ring = np.array([[s[0] * np.cos(t2), s[0] * np.sin(t2), zz] for t2 in np.linspace(0, 2 * np.pi, 32, endpoint=False) for zz in (-s[1], s[1])])
            hull = ConvexHull(ring @ R.T + p)
        else:
            continue
        surf.append(pts)
        hulls.append(hull)
        surf_body.append(np.full(len(pts), b))
surf, surf_body = np.vstack(surf), np.concatenate(surf_body)
gap = np.min(np.linalg.norm(surf[:, None, :] - c[None], axis=2) - r[None], axis=1)  # > 0: this geometry point lies outside every sphere, by gap
inside = gap <= 0
uncovered_by_body = {name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)): round(float(np.max(gap[surf_body == b])), 4) for b in np.unique(surf_body)}
coverage_by_body = {name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)): round(float(inside[surf_body == b].mean()), 3) for b in np.unique(surf_body)}
dirs = rng.normal(size=(200, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
shell = (c[:, None, :] + r[:, None, None] * dirs[None]).reshape(-1, 3)
outside = np.ones(len(shell), bool)
for hull in hulls:
    outside &= np.any(shell @ hull.equations[:, :3].T + hull.equations[:, 3] > 1e-9, axis=1)
stick_out = cKDTree(surf).query(shell[outside])[0] if np.any(outside) else np.zeros(1)
check = {"n_spheres": int(len(r)), "spheres_per_link": {k: len(v) for k, v in spheres.items()},
         "base_sphere_fit": json.loads((OUT / "robot_spheres_fit.json").read_text()),
         "max_centre_difference_m": dpos, "max_jacobian_difference": djac, "configurations": len(qs),
         "coverage_of_collision_surface": float(inside.mean()), "coverage_by_body": coverage_by_body,
         "geometry_outside_spheres_m": {"50%": float(np.percentile(np.maximum(gap, 0), 50)), "90%": float(np.percentile(np.maximum(gap, 0), 90)), "99%": float(np.percentile(np.maximum(gap, 0), 99)), "max": float(gap.max())},
         "geometry_outside_spheres_max_by_body_m": uncovered_by_body, "sphere_surface_outside_geometry_share": float(outside.mean()),
         "stick_out_m": {"50%": float(np.percentile(stick_out, 50)), "90%": float(np.percentile(stick_out, 90)), "max": float(stick_out.max())}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "robot_spheres.json").write_text(json.dumps(check, indent=1))
print("CHECK " + json.dumps(check))

# ------------------------------------------------------------------ 4. figure: three poses, spheres over the real geometry
model_cam = mujoco.MjvCamera()
model_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1
m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 800), max(m.vis.global_.offheight, 800)
poses = [q0[adr], qs[3], qs[11]]
robot_bodies = set(bodies)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), layout="constrained")
with mujoco.Renderer(m, 700, 700, max_geom=20000) as rend:
    for ax, q, title in zip(axs, poses, ("start configuration", "random configuration 1", "random configuration 2")):
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        c, r = rs.spheres(q)
        model_cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_link0")] - np.array([0.0, 0.0, 0.25])
        model_cam.distance, model_cam.azimuth, model_cam.elevation = 3.0, 135.0, -18.0
        rend.update_scene(d, camera=model_cam, scene_option=opt)
        scn = rend.scene
        for i in range(scn.ngeom):  # hide the kitchen: only the robot's visual geoms stay
            gg = scn.geoms[i]
            if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and int(m.geom_bodyid[gg.objid]) not in robot_bodies:
                gg.rgba[3] = 0.0
        for ci, ri in zip(c, r):
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([ri, 0, 0]), ci, np.eye(3).ravel(), np.array([0.9, 0.35, 0.1, 0.35], np.float32))
            scn.ngeom += 1
        ax.imshow(rend.render())
        ax.axis("off")
        ax.set_title(title)
fig.suptitle(f"PandaOmron: {len(r)} cuRobo collision spheres (orange) over the robot's geometry", fontsize=8)
fig.savefig(OUT / "robot_spheres.pdf")
base.close()

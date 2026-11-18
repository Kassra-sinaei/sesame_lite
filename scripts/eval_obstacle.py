"""The safety obstacle on Nominal's path (tuning procedure of 2026-11-16, stage 3).

    MUJOCO_GL=egl python -s scripts/eval_obstacle.py --scene counter_to_sink_s1000 --nominal RUN.json [--out-root DIR] [--none]

Rule (OBSTACLE_RULE below), for the pick-and-place instructions; the door instructions get no obstacle:
- window: Nominal's run (no obstacle) from the start of the manipulation until the picked object arrives within 10 cm of
  where it goes (the destination object, else the destination fixture), or until the end of the run; if Nominal never
  lifts the object, until the hand's closest approach to it. This covers the approach and the carry over the counter;
- the volume swept in the window by the spheres of the arm, the hand and the fingers;
- the obstacle: the first candidate (pitcher, kettle, thermos; order drawn by the seed) at least 18 cm high and 7 cm wide,
  turned by one of 4 angles 45 degrees apart (order drawn by the seed), standing on the counter top the picked object stands on (sink to counter: the
  plate), at least 5 cm from the picked object, the plate under it and the destination object, touching nothing, clear
  of the robot at its start and stop poses;
- its position: g is drawn by the seed from [-3, +1] cm; among the positions whose signed distance to the swept volume
  lies in [-3, +1] cm (negative: it cuts into Nominal's path by that depth), the ones closest to g are tried first;
- counter to stove: the stove top counts as a support too (a kettle on another burner);
- the maps are built again with the obstacle and must show it (a map node with points up to 5 cm below its top), and the
  base path is planned again; otherwise the next position is tried (at most 3 map builds).
The scene's start, floor obstacles and counter objects are unchanged; the kitchen is checked against Nominal's run
(every fixture and object within 1 mm). ``--none``: the scene is only saved again with the candidates stored (no
obstacle). Writes <out-root or scenes>/<scene>/scene.json and the maps; in place the first scene.json is kept as
scene_first_campaign.json. Prints one OBSTACLE line.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import (BASE_JOINTS, OBSTACLE_CANDIDATES, RULE, SCENES_DIR, EvalFilterEnv, _corners, _free_adr, _gap_xy, _load, _Prep, apply_scene,  # noqa: E402
                                    hold_still, place_body, robot_group, set_base_pose, stored_bodies, touching)
from sesame_lite.mapping import MappingParams, build_map  # noqa: E402
from sesame_lite.planner import NoPathError, plan_base_path  # noqa: E402
from sesame_lite.robot_spheres import RobotSpheres  # noqa: E402
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _collision_geoms, _raw_data, _raw_model, _subtree_body_ids, capture_scene_graph  # noqa: E402

OBSTACLE_RULE = dict(min_height_m=0.18, min_width_m=0.07, depth_range_m=(-0.03, 0.01), target_gap_m=0.05, turns=4,
                     support_level_m=0.03, search_radius_m=0.7, grid_m=0.02, lift_m=0.02, plate_reach_m=0.10, mapped_top_m=0.05, map_tries=3)
PICK_FROM_COUNTER = ("counter_to_sink", "counter_to_cabinet", "counter_to_stove")

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--nominal", default=None, help="Nominal's run file of the scene without obstacle")
ap.add_argument("--out-root", default=None)
ap.add_argument("--none", action="store_true")
ap.add_argument("--voxel", type=float, default=None, help="voxel size of the maps [m] (default: MappingParams)")
args = ap.parse_args()
src = SCENES_DIR / args.scene
first = src / "scene_first_campaign.json"
scene = json.loads((first if first.exists() else src / "scene.json").read_text())
out = Path(args.out_root) / args.scene if args.out_root else src
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, filter=False)
env.env.reset(seed=scene["seed"])
apply_scene(env, scene)
rs = env.rs
m, d = _raw_model(rs.sim), _raw_data(rs.sim)
oracle, lookup = capture_scene_graph(rs)
line = {"scene": args.scene}
nom = json.loads(Path(args.nominal).read_text()) if args.nominal else None

# the same kitchen as Nominal's run (the candidates are new objects of the task)
if nom is not None:
    centre = {n["node_id"]: np.array(n["center"]) for n in nom["oracle_graph"]["nodes"]}
    shift = max(float(np.abs(centre[oid] - np.asarray(oracle.nodes[lookup.node_of[name]].center)).max()) if name in lookup.node_of else np.inf
                for name, oid in nom["oracle_lookup"]["node_of"].items() if not name.startswith("object:obstacle_"))  # (the candidates are placed by this tool)
    line["kitchen_shift_m"] = shift
    if shift > 1e-3:
        line.update(status="failed", reason="the kitchen differs from Nominal's run")
        print("OBSTACLE " + json.dumps(line, default=float), flush=True)
        sys.exit(1)

prep = _Prep(env, scene["instruction"], scene["seed"], out)
if scene["instruction"] == "counter_to_stove":  # a kettle or pitcher may stand on the stove top
    from sesame_lite.eval_scene import _fixture_bodies
    prep.counter_bodies |= _fixture_bodies(rs, m, "Stove") | _fixture_bodies(rs, m, "Stovetop")
env._base_geometry()
env.robot = RobotSpheres(m, d)  # the robot's sphere model (Nominal's swept volume)
cand = {}
for name in OBSTACLE_CANDIDATES:
    body = rs.obj_body_id[name]
    bodies = _subtree_body_ids(m, body)
    cand[name] = dict(body=body, bodies=bodies, gids=_collision_geoms(m, bodies))
park = lambda name, k: place_body(m, d, cand[name]["body"], cand[name]["gids"], (70.0 + 3 * k, 70.0), 0.0, 0.0)  # noqa: E731
for k, name in enumerate(cand):
    park(name, k)
jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in BASE_JOINTS]
base_q = d.qpos[m.jnt_qposadr[jid]].copy()
standoff_xy, standoff_yaw = np.asarray(scene["standoff"]["xy"], float), float(scene["standoff"]["yaw"])
rng = np.random.default_rng([scene["seed"], 7])
params = MappingParams() if args.voxel is None else MappingParams(voxel_size_m=args.voxel)


def box_of(names):
    pts = np.vstack([_corners(m, d, _collision_geoms(m, _subtree_body_ids(m, rs.obj_body_id[n]))) for n in names])
    return pts.min(axis=0), pts.max(axis=0)


def sweep(nominal, instruction):
    """Sphere centres and radii (arm, hand, fingers) over Nominal's window, and the window."""
    z = np.load(Path(args.nominal).with_suffix(".npz"))
    names = [str(x) for x in z["object_names"]]
    seg = np.array([g["segment"] for g in nominal["ground_truth"]])
    if not (seg == 1).any() or "obj" not in names:
        return None, "Nominal did not reach the manipulation"
    t0 = int(np.argmax(seg == 1))
    n = len(z["q"])
    p = z["object_pos"][:n, names.index("obj")]
    lifted = np.flatnonzero(p[t0:, 2] - p[t0, 2] > OBSTACLE_RULE["lift_m"]) + t0
    groups = np.array([robot_group(link) for link in env.robot.links])
    keep = np.isin(groups, ("arm", "hand", "fingers"))
    if not len(lifted):  # never grasped: up to the closest approach of the hand to the object
        hand = np.array([env.robot.spheres(z["q"][t])[0][groups == "hand"].mean(axis=0) for t in range(t0, n)])
        window = (t0, t0 + int(np.argmin(np.linalg.norm(hand - p[t0:n], axis=1))))
    else:  # until the object arrives where it goes: the destination object, else the destination fixture
        if "container" in names:
            dest = z["object_pos"][t0, names.index("container")][:2]
        else:
            node = {x["node_id"]: x for x in nominal["oracle_graph"]["nodes"]}
            fx = next(x for x in nominal["result"]["dropped_pairs"] if x["role"] not in ("object", "support", "container", "plate"))
            dest = np.array(node[fx["object"][2:]]["center"][:2])
        arrive = np.flatnonzero(np.linalg.norm(p[lifted[0]:, :2] - dest, axis=1) < OBSTACLE_RULE["plate_reach_m"])
        window = (t0, int(lifted[0] + arrive[0]) if len(arrive) else n - 1)
    c, r = [], []
    for t in range(window[0], window[1] + 1):
        cc, rr = env.robot.spheres(z["q"][t])
        c.append(cc[keep]), r.append(rr[keep])
    return (np.vstack(c), np.concatenate(r), groups[keep].tolist() * (window[1] - window[0] + 1), window), ""


def signed(lo, hi, c, r):
    """Smallest signed distance between the box [lo, hi] and the spheres (centres c, radii r)."""
    q = np.maximum(np.maximum(lo - c, c - hi), 0.0)
    return float((np.linalg.norm(q, axis=1) - r).min())


def try_place(instruction):
    if instruction not in PICK_FROM_COUNTER + ("sink_to_counter",):
        return None, "no obstacle for this instruction"
    if nom is None:
        return None, "no Nominal run given"
    sw, why = sweep(nom, instruction)
    if sw is None:
        return None, why
    centres, radii, sgroups, window = sw
    keep_off = [n for n in ("obj", "obj_container", "container") if n in rs.objects]  # the obstacle keeps 8 cm from each of these
    boxes = [box_of([n]) for n in keep_off]
    support = float(box_of(["container"] if instruction == "sink_to_counter" else [n for n in ("obj", "obj_container") if n in rs.objects])[0][2])
    tlo, thi = box_of(["container"] if instruction == "sink_to_counter" else ["obj"])
    t_c = (tlo[:2] + thi[:2]) / 2
    g = float(rng.uniform(*OBSTACLE_RULE["depth_range_m"]))
    closest, fails = np.inf, []
    order = [OBSTACLE_CANDIDATES[i] for i in rng.permutation(len(OBSTACLE_CANDIDATES))]
    turns = [np.pi / 4 * i for i in rng.permutation(OBSTACLE_RULE["turns"])]
    for k, (name, turn) in enumerate((n, t) for n in order for t in turns):
        c = cand[name]
        park(name, OBSTACLE_CANDIDATES.index(name))
        adr = _free_adr(m, c["body"])[0]
        d.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)  # the candidate's own orientation, then this turn
        pts = place_body(m, d, c["body"], c["gids"], (70.0 + 3 * OBSTACLE_CANDIDATES.index(name), 70.0), turn, 0.0)
        size = pts.max(axis=0) - pts.min(axis=0)
        if size[2] < OBSTACLE_RULE["min_height_m"] or min(size[:2]) < OBSTACLE_RULE["min_width_m"]:
            continue
        half = size[:2] / 2
        R, s = OBSTACLE_RULE["search_radius_m"], OBSTACLE_RULE["grid_m"]
        xs = np.arange(-R, R + 1e-9, s)
        grid = np.array([t_c + (x, y) for x in xs for y in xs if x * x + y * y <= R * R])
        rows = []
        for xy in grid:
            lo, hi = xy - half, xy + half
            if min(_gap_xy(lo, hi, a, b) for a, b in boxes) < OBSTACLE_RULE["target_gap_m"]:
                continue
            tops = [prep._counter_top(np.array(p)) for p in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]), tuple(xy))]
            if any(t is None for t in tops) or max(t[1] for t in tops) - min(t[1] for t in tops) > RULE["counter_level_m"] or abs(tops[-1][1] - support) > OBSTACLE_RULE["support_level_m"]:
                continue
            z0 = tops[-1][1]
            rows.append((xy, z0, signed(np.r_[lo, z0], np.r_[hi, z0 + size[2]], centres, radii)))
        lo_d, hi_d = OBSTACLE_RULE["depth_range_m"]
        near = sorted((x for x in rows if lo_d <= x[2] <= hi_d), key=lambda x: (abs(x[2] - g), rng.random()))
        if not near:
            closest = min([closest] + [min(abs(x[2] - lo_d), abs(x[2] - hi_d)) for x in rows])
            continue
        tried = 0
        for xy, z0, depth in near:
            place_body(m, d, c["body"], c["gids"], xy, 0.0, z0)
            if touching(m, d, c["bodies"]):
                continue
            ok = True
            for pose in (None, (standoff_xy, standoff_yaw)):
                if pose is not None:
                    set_base_pose(env, *pose)
                ok &= not (prep.robot_touches() & c["bodies"])
            d.qpos[m.jnt_qposadr[jid]] = base_q
            mujoco.mj_forward(m, d)
            if not ok:
                continue
            tried += 1
            res = finish(name, dict(name=name, category=prep._category(name), mjcf=str((next(x for x in rs.obstacle_cfgs if x["name"] == name)["info"] or {}).get("mjcf_path", "")),
                                    height_m=round(float(size[2]), 3), width_m=[round(float(v), 3) for v in size[:2]], xy=[round(float(v), 4) for v in xy], turn=round(turn, 4),
                                    depth_drawn_m=round(g, 4), depth_m=round(depth, 4), gap_to_target_m=round(_gap_xy(xy - half, xy + half, tlo, thi), 4),
                                    nominal_window=list(window), nominal_run=str(args.nominal)))
            if isinstance(res, dict):
                return res, ""
            fails.append(res)
            if tried >= OBSTACLE_RULE["map_tries"]:
                return None, f"map or path checks failed in 3 positions: {'; '.join(fails)}"
    return None, f"no position with a depth in [-3, +1] cm" + (f" (the closest was {100 * closest:.1f} cm outside)" if np.isfinite(closest) else "") + (f"; map or path checks: {'; '.join(fails)}" if fails else "")


def finish(name, info):
    """Store the other candidates, build the maps, check the obstacle is mapped, plan the path; the new scene or None."""
    stored = list(scene.get("stored_below_floor", []))
    unused = [n for n in OBSTACLE_CANDIDATES if n != name]
    for k, n in enumerate(unused, start=len(stored)):
        place_body(m, d, cand[n]["body"], cand[n]["gids"], standoff_xy + np.array([0.5 * k, 0.0]), 0.0, prep.floor_z - RULE["store_below_floor_m"])
    hold_still(m, stored_bodies(rs, stored + unused))
    mujoco.mj_forward(m, d)
    if prep.robot_touches():
        return "the robot touches something at the start"
    orc, lk = capture_scene_graph(rs)
    out.mkdir(parents=True, exist_ok=True)
    build_map(rs, orc, lk, out / "map_full", seed=scene["seed"], params=params)
    bp, Rm, _ = env.robot_pose()
    graph, _, _, balls = _load(out / "map_full", orc, lk, (bp, float(np.arctan2(Rm[1, 0], Rm[0, 0]))), env.policy)
    if name is not None:
        oid = lk.node_of[f"object:{name}"]
        nodes = IdBridge.from_instances(HydraSceneGraph(out / "map_full", orc.labels, robot_pose=(bp, 0.0)), orc).hydra_of.get(oid, [])
        top = max((float(graph.nodes[n].points[:, 2].max()) for n in nodes if n in graph.nodes), default=None)
        true_top = float(box_of([name])[1][2])
        info.update(map_nodes=len(nodes), mapped_top_below_true_m=None if top is None else round(true_top - top, 3))
        if top is None or true_top - top > OBSTACLE_RULE["mapped_top_m"]:
            place_body(m, d, cand[name]["body"], cand[name]["gids"], (70.0, 70.0), 0.0, 0.0)
            return "not in the map" if top is None else f"map top {100 * (true_top - top):.0f} cm below the true top"
    try:
        path = plan_base_path(graph, balls, np.asarray(scene["start"]["xy"], float), standoff_xy, inflation_m=env.inflation_m, z_band=env.z_band, upper=env.upper)
    except NoPathError:
        if name is not None:
            place_body(m, d, cand[name]["body"], cand[name]["gids"], (70.0, 70.0), 0.0, 0.0)
        return "no base path"
    build_map(rs, orc, lk, out / "map_look", seed=scene["seed"], params=params, look_from=scene["start"]["xy"])
    new = json.loads(json.dumps(scene))
    new["plan"]["path_xy_before_obstacle"], new["plan"]["path_xy"] = scene["plan"]["path_xy"], np.round(path, 4).tolist()
    new["start"]["path_length_m"] = round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 3)
    names = list(scene["clutter_qpos"]) + list(OBSTACLE_CANDIDATES)
    new["clutter_qpos"] = {n: d.qpos[_free_adr(m, rs.obj_body_id[n])[0]:_free_adr(m, rs.obj_body_id[n])[0] + 7].tolist() for n in names}
    new["stored_below_floor"] = stored + unused
    new["snapshot_id_first_campaign"], new["snapshot_id"] = scene["snapshot_id"], orc.snapshot_id
    new.update(safety_obstacle={"placed": True, **info} if name is not None else {"placed": False, "reason": info.get("reason")}, obstacle_rule=OBSTACLE_RULE)
    if out == src and not first.exists():
        shutil.copy(src / "scene.json", first)
    (out / "scene.json").write_text(json.dumps(new, indent=1))
    return new


if args.none:
    new, why = finish(None, {"reason": "no obstacle in this condition"}), ""
else:
    new, why = try_place(scene["instruction"])
    if new is None:  # no obstacle for this scene: saved again with the candidates stored
        new = finish(None, {"reason": why})
new = new if isinstance(new, dict) else None
line.update(status="ok" if new is not None else "failed", obstacle=(new or {}).get("safety_obstacle"), reason=why)
print("OBSTACLE " + json.dumps(line, default=float), flush=True)
base.close()

"""Map study (EVAL_SPEC.md, section 8; no robot runs): does the observed true surface of every mapped object lie inside
the union of its balls (main.pdf, Lemma 1), and how large must one fixed radius be for the same?

    MUJOCO_GL=egl python -s scripts/eval_map_study.py --scene close_fridge_s3 --kind full --noise 0
    python -s scripts/eval_map_study.py --combine

Maps: the test scenes of the Table II instructions, both map kinds, three depth-noise seeds (NOISE_SEEDS), mapped again
here (the frames are needed). A sample is one object of one map that has map points and observed surface.
- True surface: points sampled on the object's visual geometry (triangles, by area, ~1.5 cm apart).
- Observed: a surface point that at least one mapping frame saw (uncertainty.point_visibility: its pixel carries the
  object's label at its depth).
- Capped exclusion (as in the paper: Lemma 1 covers the surface of the points with sigma <= sigma_max): an observed surface
  point whose nearest map point of the object is capped is left out. Reported: the share left out, and the failure share
  with and without the exclusion.
- A sample fails if an observed surface point lies more than 1 mm outside every ball of the object.
- Two definitions of the observed surface are measured (column ``surface``):
  "seen in a frame": every true surface point a frame saw (agreed with Kasra on 2026-11-12);
  "reconstructed": for points spread over the triangles of the object's mesh in the map, the nearest point of the true
  surface, i.e. the true surface that the map's mesh represents (main.pdf, Lemma 1 is stated for it). Added on 2026-11-16
  after the first numbers of "seen in a frame" (95 % of the objects of the full-pass maps failed, by surface that has no
  mesh at all at 5 cm voxels: thin panels, edges, far regions); both are reported.
- Proposed radius: rho_k = beta_j min(sigma_k, sigma_max) + m_r (m_d = 0, the map claim alone), delta_j in DELTAS,
  beta_j from the object's own number of points. Fixed radius: the same radius for every point, FIXED_RADII_M (on all
  observed surface points; there is no cap to exclude).
- Point error: the distance of every map point to the object's true surface (exact, to the triangles), over sigma_k.
Writes outputs/eval/map_study/<scene>_<kind>_n<k>.json (+ .npz); --combine writes results/map_study.csv and
results/map_study_stats.json.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402

NOISE_SEEDS = (101, 202, 303)
DELTAS = (0.2, 0.1, 0.05, 0.01)
FIXED_RADII_M = [round(0.01 + 0.005 * k, 3) for k in range(23)]  # 1 cm to 12 cm in steps of 0.5 cm
TOL_M = 1e-3
STUDY = Path(__file__).resolve().parents[1] / "outputs" / "eval" / "map_study"
RESULTS = STUDY.parent / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--scene")
ap.add_argument("--kind", choices=("full", "look"))
ap.add_argument("--noise", type=int, choices=(0, 1, 2))
ap.add_argument("--combine", action="store_true")
args = ap.parse_args()

if args.combine:
    import csv

    maps = [json.loads(f.read_text()) for f in sorted(STUDY.glob("*.json"))]
    rows, stats = [], {}
    for kind in ("full", "look"):
        ms = [m for m in maps if m["kind"] == kind]
        objs = [o for m in ms for o in m["objects"]]
        for surface, pick in (("reconstructed", lambda o: o["recon"]), ("seen in a frame", lambda o: o)):
            src = [pick(o) for o in objs if pick(o) is not None]
            for d in DELTAS:
                recs = [o["proposed"][str(d)] for o in src]
                kept = [r for r in recs if r["fail_excl"] is not None]
                rows.append({"map_kind": kind, "surface": surface, "radius_rule": "proposed", "delta_j": d, "fixed_radius_m": "", "maps": len(ms), "samples": len(recs),
                             "failure_share": sum(r["fail_excl"] for r in kept) / max(len(kept), 1), "samples_with_surface_left": len(kept),
                             "failure_share_without_exclusion": sum(r["fail_all"] for r in recs) / max(len(recs), 1),
                             "surface_share_excluded_at_capped_points": sum(r["n_excluded"] for r in recs) / max(sum(r["n_observed"] for r in recs), 1),
                             "surface_share_outside_the_balls": float(np.mean([r["share_of_surface_outside"] for r in recs])) if recs else "",
                             "worst_gap_m_median": float(np.median([r["worst_outside_m"] for r in recs])) if recs else "",
                             "mean_radius_m": sum(r["rho_sum"] for r in recs) / max(sum(r["n_points"] for r in recs), 1),
                             "capped_point_share": sum(r["n_capped"] for r in recs) / max(sum(r["n_points"] for r in recs), 1)})
            for k, r in enumerate(FIXED_RADII_M):
                rows.append({"map_kind": kind, "surface": surface, "radius_rule": "fixed", "delta_j": "", "fixed_radius_m": r, "maps": len(ms), "samples": len(src),
                             "failure_share": sum(o["fixed_fail"][k] for o in src) / max(len(src), 1), "samples_with_surface_left": len(src),
                             "failure_share_without_exclusion": sum(o["fixed_fail"][k] for o in src) / max(len(src), 1),
                             "surface_share_excluded_at_capped_points": 0.0, "surface_share_outside_the_balls": "", "worst_gap_m_median": "", "mean_radius_m": r, "capped_point_share": ""})
        arr = [np.load(STUDY / f"{m['scene']}_{kind}_n{m['noise']}.npz") for m in ms]
        cat = {k: np.concatenate([a[k] for a in arr]) for k in ("beta_sigma_bar", "rho", "capped", "err_over_sigma", "err")} if arr else {}
        q = lambda x: dict(zip(["10%", "50%", "90%", "99%", "max"], np.percentile(x, [10, 50, 90, 99, 100]).round(5).tolist()))  # noqa: E731
        stats[kind] = {"maps": len(ms), "objects_with_points_and_observed_surface": len(objs), "mapped_objects_without_observed_surface": sum(m["objects_no_observed_surface"] for m in ms),
                       "objects_without_map_points": sum(m["objects_unmapped"] for m in ms), "points": int(len(cat.get("rho", []))),
                       "beta_j_sigma_bar_m (delta_j 0.01)": q(cat["beta_sigma_bar"]), "rho_m (delta_j 0.01)": q(cat["rho"]), "capped_point_share (delta_j 0.01)": float(cat["capped"].mean()),
                       "m_r_m": q(np.array([m["m_r"] for m in ms])), "point_error_m": q(cat["err"]),
                       "std_of_point_error_over_sigma (rms)": float(np.sqrt(np.mean(cat["err_over_sigma"] ** 2))), "median_of_point_error_over_sigma": float(np.median(cat["err_over_sigma"])),
                       "mapping_seconds_median": float(np.median([m["mapping_s"] for m in ms]))} if arr else {"maps": 0}
    with open(RESULTS / "map_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "map_study_stats.json").write_text(json.dumps({"noise_seeds": NOISE_SEEDS, "deltas": DELTAS, "tolerance_m": TOL_M, **stats}, indent=1))
    for r in rows:
        if r["radius_rule"] == "proposed" or r["fixed_radius_m"] in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12):
            print(r["map_kind"], "|", r["surface"], "|", r["radius_rule"], r["delta_j"] or r["fixed_radius_m"], "samples", r["samples"], "fail %.3f" % r["failure_share"], "fail w/o exclusion %.3f" % r["failure_share_without_exclusion"],
                  "excluded %.3f" % r["surface_share_excluded_at_capped_points"], "mean radius %.3f" % r["mean_radius_m"])
    sys.exit(0)

import mujoco  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import SCENES_DIR, EvalFilterEnv, apply_scene  # noqa: E402
from sesame_lite.mapping import MappingParams, build_map, instance_table  # noqa: E402
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _raw_data, _raw_model, capture_scene_graph  # noqa: E402
from sesame_lite.truth import _closest_on_triangles  # noqa: E402
from sesame_lite.uncertainty import balls, point_visibility  # noqa: E402

out = STUDY / f"{args.scene}_{args.kind}_n{args.noise}.json"
if out.exists():
    sys.exit(0)
STUDY.mkdir(parents=True, exist_ok=True)
scene = json.loads((SCENES_DIR / args.scene / "scene.json").read_text())
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, filter=False)
base.reset(seed=scene["seed"])
apply_scene(env, scene)
oracle, lookup = capture_scene_graph(env.rs)
assert oracle.snapshot_id == scene["snapshot_id"]
model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
params = MappingParams()
tmp = STUDY / f"tmp_{args.scene}_{args.kind}_n{args.noise}"
views, t0 = [], time.time()
build_map(env.rs, oracle, lookup, tmp, seed=NOISE_SEEDS[args.noise], params=params, look_from=scene["start"]["xy"] if args.kind == "look" else None, views_out=views)
mapping_s = time.time() - t0
bp, R, _ = env.robot_pose()
graph = HydraSceneGraph(tmp, oracle.labels, robot_pose=(bp, float(np.arctan2(R[1, 0], R[0, 0]))))
bridge = IdBridge.from_instances(graph, oracle)
label_of = {oid: k + 1 for k, oid in enumerate(oracle.nodes)}  # as mapping.instance_table
assert instance_table(env.rs, oracle, lookup, oracle.labels, params.labels)["oracle_of_label"].get(1) == [next(iter(oracle.nodes))]


def triangles(bodies) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World triangles of the visual geoms (meshes and boxes) of the bodies."""
    A, B, C = [], [], []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies or int(model.geom_group[g]) != 1 or model.geom_rgba[g][3] == 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            f = model.mesh_face[model.mesh_faceadr[m]: model.mesh_faceadr[m] + model.mesh_facenum[m]]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]
            v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
            f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
        else:
            continue
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        A.append(w[f[:, 0]]); B.append(w[f[:, 1]]); C.append(w[f[:, 2]])  # noqa: E702
    if not A:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    a, b, c = np.vstack(A), np.vstack(B), np.vstack(C)
    ok = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-10  # no degenerate triangles
    return a[ok], b[ok], c[ok]


def sample(a, b, c, rng, spacing=0.015, n_min=300, n_max=6000) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(np.clip(area.sum() / spacing**2, n_min, n_max))
    t = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[t] + u[:, None] * (b[t] - a[t]) + v[:, None] * (c[t] - a[t])


def closest_on_triangle(p, a, b, c) -> np.ndarray:
    """Closest point of each triangle (a, b, c) [n, 3] to its point p [n, 3] (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        q = a + (vb / den)[:, None] * ab + (vc / den)[:, None] * ac
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None], b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b), q)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + (d2 / (d2 - d6))[:, None] * ac, q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + (d1 / (d1 - d3))[:, None] * ab, q)
        q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
        q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)
    return q


def nearest_on_surface(points, a, b, c) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each point to the triangles and the closest surface point (exact; candidates: the 64 triangles
    nearest by centroid and every large triangle)."""
    cen = (a + b + c) / 3
    big = np.flatnonzero(np.max(np.linalg.norm(np.stack([b - a, c - b, a - c]), axis=2), axis=0) > 0.2)
    near = cKDTree(cen).query(points, k=min(64, len(cen)))[1].reshape(len(points), -1)
    dist, where = np.empty(len(points)), np.empty((len(points), 3))
    for s in range(0, len(points), 128):
        idx = np.union1d(np.unique(near[s:s + 128]), big)
        P = points[s:s + 128]
        t = idx[np.argmin(_closest_on_triangles(P, a[idx], b[idx], c[idx]), axis=1)]
        q = closest_on_triangle(P, a[t], b[t], c[t])
        dist[s:s + 128], where[s:s + 128] = np.linalg.norm(P - q, axis=1), q
    return dist, where


mesh_v = np.asarray(graph.dsg.mesh.get_vertices(), dtype=np.float64)[:3].T
mesh_f = np.asarray(graph.dsg.mesh.get_faces(), dtype=np.int64).T


def coverage(So, P, nodes) -> tuple[dict, list]:
    """Per delta_j and per fixed radius: is every point of So inside the balls of the object's points P?"""
    D = np.linalg.norm(So[:, None, :] - P[None], axis=2)
    nearest, dmin = D.argmin(axis=1), D.min(axis=1)
    out = {}
    for d in DELTAS:
        bs = [balls(graph.nodes[n].points, graph.nodes[n].std, delta_j=d, voxel_size=graph.voxel_size_m, m_r=graph.m_r, m_d=0.0) for n in nodes]
        rho, capped = np.concatenate([x.rho for x in bs]), np.concatenate([x.capped for x in bs])
        gap = (D - rho[None]).min(axis=1)
        excl = capped[nearest]
        out[str(d)] = {"fail_all": bool((gap > TOL_M).any()), "fail_excl": bool((gap[~excl] > TOL_M).any()) if (~excl).any() else None, "n_observed": int(len(So)), "n_excluded": int(excl.sum()),
                       "rho_sum": float(rho.sum()), "n_points": int(len(P)), "n_capped": int(capped.sum()), "beta_j": [x.kappa for x in bs], "worst_outside_m": float(gap.max()),
                       "share_of_surface_outside": float((gap > TOL_M).mean())}
    return out, [bool((dmin > r + TOL_M).any()) for r in FIXED_RADII_M]


objects, acc = [], {k: [] for k in ("beta_sigma_bar", "rho", "capped", "err_over_sigma", "err")}
no_surface = unmapped = 0
for k, oid in enumerate(oracle.nodes):
    nodes = bridge.hydra_of.get(oid, [])
    if not nodes:
        unmapped += 1
        continue
    a, b, c = triangles(lookup.bodies[oid])
    if len(a) == 0:
        continue
    S = sample(a, b, c, np.random.default_rng([scene["seed"], k]))
    seen = point_visibility(S, np.full(len(S), label_of[oid]), views, params)[0] >= 1
    if not seen.any():
        no_surface += 1
        continue
    So = S[seen]
    P = np.vstack([graph.nodes[n].points for n in nodes])
    std = np.concatenate([graph.nodes[n].std for n in nodes])
    rec = {"oracle_id": oid, "category": oracle.nodes[oid].category, "n_points": int(len(P)), "n_surface_samples": int(len(S)), "n_observed": int(len(So))}
    rec["proposed"], rec["fixed_fail"] = coverage(So, P, nodes)
    # the reconstructed surface: points on the object's mesh triangles in the map, moved to the nearest true surface point
    conn = np.concatenate([np.asarray(sorted(graph.dsg_node(n).attributes.mesh_connections), dtype=np.int64) for n in nodes])
    tri = mesh_f[np.all(np.isin(mesh_f, conn), axis=1)]
    rec["recon"] = None
    if len(tri):
        Q = sample(mesh_v[tri[:, 0]], mesh_v[tri[:, 1]], mesh_v[tri[:, 2]], np.random.default_rng([scene["seed"], k, 1]), n_min=100, n_max=3000)
        prop, fixed = coverage(nearest_on_surface(Q, a, b, c)[1], P, nodes)
        rec["recon"] = {"proposed": prop, "fixed_fail": fixed, "n_mesh_samples": int(len(Q))}
    bs = [balls(graph.nodes[n].points, graph.nodes[n].std, delta_j=0.01, voxel_size=graph.voxel_size_m, m_r=graph.m_r, m_d=0.0) for n in nodes]
    acc["rho"].append(np.concatenate([x.rho for x in bs])); acc["capped"].append(np.concatenate([x.capped for x in bs]))  # noqa: E702
    acc["beta_sigma_bar"].append(np.concatenate([x.kappa * np.minimum(graph.nodes[n].std, x.s_max) for x, n in zip(bs, nodes)]))
    err = nearest_on_surface(P, a, b, c)[0]
    acc["err"].append(err); acc["err_over_sigma"].append(err / np.maximum(std, 1e-9))  # noqa: E702
    objects.append(rec)
np.savez_compressed(out.with_suffix(".npz"), **{k: np.concatenate(v) if v else np.zeros(0) for k, v in acc.items()})
out.write_text(json.dumps({"scene": args.scene, "kind": args.kind, "noise": args.noise, "noise_seed": NOISE_SEEDS[args.noise], "frames": len(views), "mapping_s": round(mapping_s, 1),
                           "m_r": graph.m_r, "voxel_m": graph.voxel_size_m, "objects_unmapped": unmapped, "objects_no_observed_surface": no_surface, "objects": objects}))
shutil.rmtree(tmp)
print("STUDY " + json.dumps({"scene": args.scene, "kind": args.kind, "noise": args.noise, "samples": len(objects), "fail_0.01": sum(bool(o["proposed"]["0.01"]["fail_excl"]) for o in objects),
                             "fail_0.01_without_exclusion": sum(o["proposed"]["0.01"]["fail_all"] for o in objects), "unmapped": unmapped, "no_observed_surface": no_surface, "mapping_s": round(mapping_s, 1)}), flush=True)
base.close()

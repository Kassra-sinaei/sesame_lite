"""OmniGuide's distance field, built on the simulation side from our map (Kasra, 2026-11-19).

Their field (collision/esdf.py:280-341, NVIDIA License: rewritten here, not copied) is an unsigned distance: the
camera points are rounded to a 1 cm voxel grid, the occupied voxels get 0 and every other voxel the Euclidean
distance to the nearest occupied voxel (scipy EDT). Ours takes the points of our map (map_full/points.npz) instead
of their camera points: every map point except those of the objects the plan names (its roles, every map node of
each role's instance). The map is static, so the field is built once per scene, in a fixed box around the plan's
standoff, and sent to the policy server once (omniguide/server.py samples it).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from sesame_lite.planner import resolve_role

VOXEL_M = 0.01  # their voxel_size (esdf.py:282, the default used at G:507)
BOX_HALF_XY_M = 2.5  # ours: the box around the standoff (their grid is the camera cloud's box plus 0.3 m, G:507)
BOX_Z_M = (-0.1, 2.3)  # ours: floor to above the highest reach of the gripper


def task_nodes(env) -> dict[str, list[str]]:
    """Map nodes of each plan role (the role's whole instance); a role the map has no node for has none."""
    out = {}
    for role, spec in env.plan["roles"].items():
        node = resolve_role(spec, env.rs, env.lookup)
        out[role] = [] if node is None else list(env.bridge.instance_nodes(node))
    return out


def build_field(env) -> dict:
    """The unsigned distance field [m] on a 1 cm grid (float32, axes x, y, z; voxel i at origin + i * voxel)."""
    t0 = time.time()
    pts = np.load(env.graph.map_dir / "points.npz")
    roles = task_nodes(env)
    dropped = {n for nodes in roles.values() for n in nodes}
    off, ids = pts["offsets"], [str(n) for n in pts["node_ids"]]
    keep = [pts["points"][off[k]:off[k + 1]] for k, nid in enumerate(ids) if nid not in dropped]
    points = np.concatenate(keep) if keep else np.zeros((0, 3))
    centre = np.asarray(env.segments[-1].center_xy, dtype=float)  # the plan's standoff (manipulation segment)
    lo = np.r_[centre - BOX_HALF_XY_M, BOX_Z_M[0]]
    hi = np.r_[centre + BOX_HALF_XY_M, BOX_Z_M[1]]
    dims = np.round((hi - lo) / VOXEL_M).astype(int) + 1
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    idx = np.round((points[inside] - lo) / VOXEL_M).astype(int)  # nearest voxel, as esdf.py:306-314
    idx = np.minimum(idx, dims - 1)
    occupied = np.zeros(dims, dtype=bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    sdf = (distance_transform_edt(~occupied) * VOXEL_M).astype(np.float32)  # as esdf.py:331-334 (no inside part)
    return {"sdf": sdf, "origin": lo.astype(np.float32), "voxel_m": VOXEL_M,
            "meta": {"roles_removed": roles, "points_total": int(len(pts["points"])), "points_removed": int(len(pts["points"]) - len(points)),
                     "points_in_box": int(inside.sum()), "occupied_voxels": int(occupied.sum()), "dims": dims.tolist(), "origin": lo.round(4).tolist(),
                     "box_half_xy_m": BOX_HALF_XY_M, "box_z_m": list(BOX_Z_M), "build_s": round(time.time() - t0, 2)}}

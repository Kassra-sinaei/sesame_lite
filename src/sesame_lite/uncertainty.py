"""Per-face standard deviation of Hydra's object boxes (spec 4.6; derivation in uncertainty_cvar.pdf, eqs. 1-3).

    voxel weight w  ->  Var[d] = sigma0_sq / w                      (Hydra's stock weight is 1 / variance, up to sigma0_sq)
    mesh vertex     ->  s^2 = sigma0_sq ((1 - lam)^2 / w_a + lam^2 / w_b)   on the grid edge a-b it was interpolated on
    box face k      ->  sigma_k = max(|a_k . n|, 0.2) s   at the face's supporting vertex (the outermost one along a_k)

The face offsets b_hat_k = max over the node's vertices of a_k . p reproduce Hydra's own box (checked when the faces
are written). Each face also gets two flags:
  viewed_within_60  map side: some mapping frame saw the face's supporting vertex (a pixel within one of it carries the
                    node's label at the vertex's depth) from within 60 degrees of the face normal; best_view_angle_deg
  own_near_truth    evaluation only: a vertex of this node lies within 1.5 voxels of the true supporting point (the
                    outermost point of the instance's visual geoms along a_k)
and ``observed`` = both. A face that is not observed gets no claim in the coverage check. The filter reads neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import spark_dsg as dsg

from sesame_lite.scene_graph import _raw_data, _raw_model

SIGMA_NORMAL_FLOOR = 0.2  # a grazing normal must not give a zero face std
LATTICE_TOL = 1e-3  # in voxels: a coordinate this close to a voxel centre is on the lattice
OBSERVED_RADIUS_VOXELS = 1.5


class VoxelGrid:
    """The TSDF voxels from the TsdfSink (outputs/maps/.../voxels.npz), looked up by global index."""

    def __init__(self, path: Path):
        v = np.load(path)
        self.index, self.distance, self.weight = v["index"].astype(np.int64), v["distance"], v["weight"]
        self.voxel_size, self.origin = float(v["voxel_size"]), float(v["origin"])
        self._keys = self._key(self.index)
        self._order = np.argsort(self._keys)
        self._sorted = self._keys[self._order]

    @staticmethod
    def _key(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64) + (1 << 20)
        return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]

    def lookup(self, idx: np.ndarray) -> np.ndarray:
        """Row of each global index (-1 if the voxel was never observed near a surface)."""
        keys = self._key(idx)
        pos = np.clip(np.searchsorted(self._sorted, keys), 0, len(self._sorted) - 1)
        return np.where(self._sorted[pos] == keys, self._order[pos], -1)

    def lattice(self, points: np.ndarray) -> np.ndarray:
        """Point coordinates in voxel units, voxel centres at integers (centre of g is (g + 0.5) * voxel_size)."""
        return np.asarray(points, dtype=np.float64) / self.voxel_size - 0.5


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from the adjacent mesh triangles (sign does not matter here)."""
    n = np.zeros_like(vertices, dtype=np.float64)
    if len(faces):
        tri = vertices[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(n, faces[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norm > 0, n / np.maximum(norm, 1e-12), 0.0)


def vertex_std(points: np.ndarray, grid: VoxelGrid, sigma0_sq: float) -> tuple[np.ndarray, np.ndarray]:
    """Std of each vertex along its normal (PDF eq. 2) and how it was obtained: 0 grid edge, 1 bound, 2 none."""
    u = grid.lattice(points)
    r = np.round(u)
    on = np.abs(u - r) < LATTICE_TOL
    s2 = np.full(len(points), np.nan)
    how = np.full(len(points), 2, dtype=np.int8)
    # (a) a marching-cubes vertex: two coordinates on the lattice, the third between voxels a and b
    edge = on.sum(axis=1) >= 2
    if np.any(edge):
        ue, re, oe = u[edge], r[edge], on[edge]
        axis = np.where(oe.sum(axis=1) == 3, 0, np.argmin(oe, axis=1))  # the off-lattice axis (any, if all three are on)
        rows = np.arange(len(ue))
        a = re.copy()
        a[rows, axis] = np.floor(ue[rows, axis] + LATTICE_TOL * (oe.sum(axis=1) == 3))
        lam = np.clip(ue[rows, axis] - a[rows, axis], 0.0, 1.0)
        b = a.copy()
        b[rows, axis] += 1
        ia, ib = grid.lookup(a.astype(np.int64)), grid.lookup(b.astype(np.int64))
        ok = (ia >= 0) & (ib >= 0)
        da, db = np.where(ok, grid.distance[ia], np.nan), np.where(ok, grid.distance[ib], np.nan)
        wa, wb = np.where(ok, grid.weight[ia], np.nan), np.where(ok, grid.weight[ib], np.nan)
        ok &= ((da < 0) & (db >= 0)) | ((db < 0) & (da >= 0))  # the edge still crosses the surface in the final TSDF
        val = sigma0_sq * ((1 - lam) ** 2 / wa + lam**2 / wb)
        idx = np.flatnonzero(edge)[ok]
        s2[idx], how[idx] = val[ok], 0
    # (b) otherwise the bound sigma0_sq / min(w) over the observed voxels within one voxel of the vertex
    rest = np.flatnonzero(how == 2)
    if len(rest):
        offsets = np.array([(i, j, k) for i in (-1, 0, 1, 2) for j in (-1, 0, 1, 2) for k in (-1, 0, 1, 2)])
        base = np.floor(u[rest]).astype(np.int64)
        cand = base[:, None, :] + offsets[None, :, :]
        near = np.all(np.abs(cand - u[rest][:, None, :]) <= 1.0, axis=2)
        rows = grid.lookup(cand)
        w = np.where((rows >= 0) & near, grid.weight[np.maximum(rows, 0)], np.inf)
        wmin = w.min(axis=1)
        found = np.isfinite(wmin)
        s2[rest[found]], how[rest[found]] = sigma0_sq / wmin[found], 1
    return np.sqrt(s2), how


def box_axes(R: np.ndarray) -> np.ndarray:
    """The six outward face normals of a box with rotation R: +R0, -R0, +R1, -R1, +R2, -R2."""
    R = np.asarray(R, dtype=np.float64)
    return np.stack([R[:, 0], -R[:, 0], R[:, 1], -R[:, 1], R[:, 2], -R[:, 2]])


def box_from_offsets(A: np.ndarray, b: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and half extents of the box {A x <= b} whose faces are +-R columns (rows of A as from box_axes)."""
    half = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    center = sum(np.asarray(R)[:, i] * (b[2 * i] - b[2 * i + 1]) / 2 for i in range(3))
    return center, half


def kappa(delta: float, n_admitted: int, n_faces: int = 6) -> float:
    """Gaussian quantile of the confidence box (spec 5.1): the risk delta is split over the admitted nodes, then over
    their faces (union bound, no independence needed)."""
    from scipy.stats import norm

    return float(norm.ppf(1.0 - (delta / max(int(n_admitted), 1)) / n_faces))


def confidence_box(faces, R: np.ndarray, *, kappa_value: float, m_r: float, m_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """b_tilde_k = b_hat_k + kappa sigma_k + m_r + m_d per face; returns the box's centre, half extents and b_tilde."""
    A = np.array([f.normal for f in faces], dtype=np.float64)
    b = np.array([f.offset + kappa_value * f.sigma + m_r + m_d for f in faces], dtype=np.float64)
    center, half = box_from_offsets(A, b, R)
    return center, half, b


# --------------------------------------------------------------------------- ground truth for the coverage check


def visual_support(model, data, geoms: list[int], directions: np.ndarray) -> np.ndarray:
    """For each direction a, the point of the given visual geoms that is outermost along a (support point)."""
    best_val = np.full(len(directions), -np.inf)
    best_pt = np.zeros((len(directions), 3))
    for g in geoms:
        c, Rg, s, t = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g], int(model.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            m = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            pts = v @ Rg.T + c
            val = directions @ pts.T
            k = np.argmax(val, axis=1)
            cand, cval = pts[k], val[np.arange(len(directions)), k]
        else:
            loc = directions @ Rg  # directions in the geom frame
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                p_loc = np.sign(loc) * s[None, :3]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                p_loc = loc * s[0]
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                xy = loc[:, :2]
                n = np.linalg.norm(xy, axis=1, keepdims=True)
                p_loc = np.c_[np.where(n > 1e-12, xy / np.maximum(n, 1e-12), 0.0) * s[0], np.sign(loc[:, 2]) * s[1]]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                p_loc = loc * s[0] + np.c_[np.zeros((len(loc), 2)), np.sign(loc[:, 2]) * s[1]]
            elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                q = loc * s[None, :3]
                p_loc = q * s[None, :3] / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            else:
                continue  # planes and height fields have no finite support
            cand = p_loc @ Rg.T + c
            cval = np.einsum("ij,ij->i", directions, cand)
        better = cval > best_val
        best_val[better], best_pt[better] = cval[better], cand[better]
    return best_pt


def instance_visual_geoms(model, bodies: set[int]) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and int(model.geom_group[g]) == 1 and float(model.geom_rgba[g][3]) > 0]


# --------------------------------------------------------------------------- faces.json


def best_view_angles(points: np.ndarray, normals: np.ndarray, label: int, views: list, params) -> np.ndarray:
    """Per point (a face's supporting vertex with its face normal): the smallest angle, over the mapping frames that saw
    it, between the face normal and the direction to the camera (inf if no frame saw it). A frame saw the point if it
    projects into the image within range and a pixel within one of it has the node's label and a depth within
    3 sigma_z + one voxel of the point's depth (not hidden behind something else)."""
    best = np.full(len(points), np.inf)
    if not views:
        return best
    pos = np.array([v[0] for v in views])
    Rs = np.array([v[1] for v in views])
    for i, (p, a) in enumerate(zip(points, normals)):
        pc = np.einsum("nji,nj->ni", Rs, p[None, :] - pos)  # optical frame: R^T (p - pos)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(params.fx * pc[:, 0] / z + params.cx).astype(np.int64)
            v = np.round(params.fy * pc[:, 1] / z + params.cy).astype(np.int64)
        ok = (z > params.min_range_m) & (z < params.max_range_m) & (u >= 1) & (u < params.width - 1) & (v >= 1) & (v < params.height - 1)
        to_cam = pos - p
        cosang = to_cam @ a / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-9)
        for n in np.flatnonzero(ok & (cosang > 0)):
            depth, labels = views[n][2], views[n][3]
            win = (slice(v[n] - 1, v[n] + 2), slice(u[n] - 1, u[n] + 2))
            tol = 3 * params.depth_noise_c * z[n] ** 2 + params.voxel_size_m
            hit = (labels[win] == label) & (np.abs(depth[win].astype(np.float32) - z[n]) <= tol)
            if np.any(hit):
                best[i] = min(best[i], float(np.degrees(np.arccos(np.clip(cosang[n], -1, 1)))))
    return best


def write_faces(map_dir: Path, rs_env, oracle_graph, lookup, *, views: list | None = None, params=None) -> dict:
    """faces.json for one cached Hydra map: per object node its category, box and six faces (see the module docstring).
    ``views`` are the mapping frames (pose, depth, labels); without them the view angles are unknown (not observed)."""
    meta = json.loads((map_dir / "meta.json").read_text())
    sigma0_sq = meta["params"]["sigma0_sq"]
    graph = dsg.DynamicSceneGraph.load(str(map_dir / "dsg.json"))
    mesh = graph.mesh
    verts = np.asarray(mesh.get_vertices(), dtype=np.float64)[:3].T  # spark_dsg: rows x, y, z, r, g, b
    tri = np.asarray(mesh.get_faces(), dtype=np.int64).T  # (3, n_faces) -> (n_faces, 3)
    normals = vertex_normals(verts, tri)
    grid = VoxelGrid(map_dir / "voxels.npz")
    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    category_of = {int(k): v for k, v in meta["category_of_label"].items()}
    oracle_of = {int(k): v for k, v in meta["oracle_of_label"].items()}
    nodes, how_counts, box_err = [], np.zeros(3, dtype=int), 0.0
    for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        a = node.attributes
        conn = np.asarray(list(a.mesh_connections), dtype=np.int64)
        if len(conn) == 0:
            continue
        bb = a.bounding_box
        R = np.asarray(bb.world_R_center, dtype=np.float64)
        A = box_axes(R)
        P = verts[conn]
        proj = P @ A.T  # (n_vertices, 6)
        k_sup = np.argmax(proj, axis=0)
        b_hat = proj[k_sup, np.arange(6)]
        center, half = box_from_offsets(A, b_hat, R)
        box_err = max(box_err, float(np.max(np.abs(center - np.asarray(bb.world_P_center)))), float(np.max(np.abs(2 * half - np.asarray(bb.dimensions)))))
        s, how = vertex_std(P[k_sup], grid, sigma0_sq)
        n = normals[conn[k_sup]]
        sigma = np.maximum(np.abs(np.einsum("ij,ij->i", A, n)), SIGMA_NORMAL_FLOOR) * s
        np.add.at(how_counts, how, 1)
        # evaluation only: truth along each face normal, from the instance's visual geoms
        label = int(a.semantic_label)
        oracle_ids = oracle_of.get(label, [])
        bodies = set().union(*(lookup.bodies[o] for o in oracle_ids)) if oracle_ids else set()
        truth_pts = visual_support(model, data, instance_visual_geoms(model, bodies), A) if bodies else np.full((6, 3), np.nan)
        truth = np.einsum("ij,ij->i", A, truth_pts)
        own_near = [bool(np.isfinite(truth[f]) and np.min(np.linalg.norm(P - truth_pts[f], axis=1)) <= OBSERVED_RADIUS_VOXELS * grid.voxel_size) for f in range(6)]
        angle = best_view_angles(P[k_sup], A, label, views or [], params) if params is not None else np.full(6, np.inf)
        max_angle = params.observed_view_angle_deg if params is not None else 0.0
        nodes.append({
            "node_id": str(node.id),
            "label": label,
            "category": category_of.get(label, 0),
            "n_vertices": int(len(conn)),
            "box": {"center": np.round(np.asarray(bb.world_P_center, dtype=float), 5).tolist(), "dimensions": np.round(np.asarray(bb.dimensions, dtype=float), 5).tolist(), "rotation": np.round(R, 6).tolist()},
            "faces": [{"normal": np.round(A[f], 6).tolist(), "offset": round(float(b_hat[f]), 5), "sigma": None if not np.isfinite(sigma[f]) else round(float(sigma[f]), 6),
                       "support_vertex": np.round(P[k_sup[f]], 5).tolist(), "vertex_std_source": ["grid_edge", "bound", "none"][int(how[f])],
                       "best_view_angle_deg": None if not np.isfinite(angle[f]) else round(float(angle[f]), 1), "viewed_within_60": bool(angle[f] <= max_angle),
                       "own_near_truth": own_near[f], "observed": bool(own_near[f] and angle[f] <= max_angle),
                       "truth": None if not np.isfinite(truth[f]) else round(float(truth[f]), 5), "truth_point": None if not np.isfinite(truth[f]) else np.round(truth_pts[f], 5).tolist()} for f in range(6)],
        })
    out = {"sigma0_sq": sigma0_sq, "vertex_std_source_counts": {"grid_edge": int(how_counts[0]), "bound": int(how_counts[1]), "none": int(how_counts[2])},
           "max_box_reproduction_error_m": box_err, "nodes": nodes}
    (map_dir / "faces.json").write_text(json.dumps(out, indent=1))
    return out

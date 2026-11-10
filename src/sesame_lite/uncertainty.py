"""Obstacle uncertainty from Hydra's TSDF, per mesh point (uncertainty_cvar.pdf, eqs. 1-4; spec v4, section 5).

    voxel weight w  ->  Var[d] = sigma0_sq / w                          (Hydra's stock weight is 1 / variance, up to sigma0_sq)
    mesh vertex m   ->  s_m^2 = sigma0_sq ((1 - lam)^2 / w_a + lam^2 / w_b)   on the grid edge a-b it was interpolated on (eq. 2)
    capped std      ->  s_bar_m = min(s_m, s_max)                                                               (eq. 3)
    ball radius     ->  rho_m = kappa_j s_bar_m + m_r + m_d,  kappa_j = Phi^-1(1 - delta_j / (2 |P_j|)),
                        m_r = e_max / sqrt(3) (covering radius of the mesh)                                     (eq. 4)

``write_points`` writes, per Hydra map, points.npz: every object node's mesh vertices with their std s_m (a vertex
whose std cannot be recovered is dropped and counted), e_max and m_r, and, for the pictures, how many mapping frames
saw each point and from how far. The per-segment radii (``Balls``) need the admitted count, so they are made where the
filter knows it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import spark_dsg as dsg

LATTICE_TOL = 1e-3  # in voxels: a coordinate this close to a voxel centre is on the lattice


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


def kappa_j(delta_j: float, n_points: int) -> float:
    """Gaussian quantile of eq. (4): the object's risk delta_j split over 2 |P_j| tails (union bound over its points)."""
    from scipy.stats import norm

    return float(norm.isf(delta_j / (2.0 * max(int(n_points), 1))))


@dataclass
class Balls:
    """The confidence balls of one object for one segment (eq. 4): centres p_m, radii rho_m and which points are capped."""

    points: np.ndarray  # [n, 3]
    rho: np.ndarray  # [n]
    capped: np.ndarray  # [n] bool, s_m > s_max
    kappa: float
    s_max: float


def balls(points: np.ndarray, std: np.ndarray, *, delta_j: float, voxel_size: float, m_r: float, m_d: float) -> Balls:
    """rho_m = kappa_j min(s_m, s_max) + m_r + m_d with s_max = voxel_size / kappa_j (PDF, Numbers)."""
    k = kappa_j(delta_j, len(points))
    s_max = voxel_size / k
    capped = std > s_max
    return Balls(points=points, rho=k * np.minimum(std, s_max) + m_r + m_d, capped=capped, kappa=k, s_max=s_max)


def point_visibility(points: np.ndarray, labels_of_points: np.ndarray, views: list, params) -> tuple[np.ndarray, np.ndarray]:
    """Per point: how many mapping frames saw it (it projects into the image within range and that pixel carries its
    node's label at its depth, within 3 sigma_z + one voxel) and the mean distance from those cameras."""
    n_views = np.zeros(len(points), dtype=np.int32)
    dist_sum = np.zeros(len(points))
    for pos, R, depth, labels in views:
        pc = (points - pos) @ R  # optical frame
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(params.fx * pc[:, 0] / z + params.cx).astype(np.int64)
            v = np.round(params.fy * pc[:, 1] / z + params.cy).astype(np.int64)
        ok = (z > params.min_range_m) & (z < params.max_range_m) & (u >= 0) & (u < params.width) & (v >= 0) & (v < params.height)
        idx = np.flatnonzero(ok)
        seen = (labels[v[idx], u[idx]] == labels_of_points[idx]) & (np.abs(depth[v[idx], u[idx]].astype(np.float32) - z[idx]) <= 3 * params.depth_noise_c * z[idx] ** 2 + params.voxel_size_m)
        hit = idx[seen]
        n_views[hit] += 1
        dist_sum[hit] += np.linalg.norm(points[hit] - pos, axis=1)
    return n_views, np.where(n_views > 0, dist_sum / np.maximum(n_views, 1), np.nan)


def write_points(map_dir: Path, *, views: list | None = None, params=None) -> dict:
    """points.npz for one cached Hydra map (see the module docstring)."""
    meta = json.loads((map_dir / "meta.json").read_text())
    sigma0_sq = meta["params"]["sigma0_sq"]
    graph = dsg.DynamicSceneGraph.load(str(map_dir / "dsg.json"))
    verts = np.asarray(graph.mesh.get_vertices(), dtype=np.float64)[:3].T  # spark_dsg: rows x, y, z, r, g, b
    tri = np.asarray(graph.mesh.get_faces(), dtype=np.int64).T  # (n_faces, 3)
    grid = VoxelGrid(map_dir / "voxels.npz")
    node_ids, points, std, source, labels, offsets, dropped, e_max = [], [], [], [], [], [0], {}, 0.0
    for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        conn = np.asarray(sorted(node.attributes.mesh_connections), dtype=np.int64)
        if len(conn) == 0:
            continue
        s, how = vertex_std(verts[conn], grid, sigma0_sq)
        keep = how != 2  # 'none': no std can be recovered, the point is dropped (spec v4, section 9)
        dropped[str(node.id)] = int((~keep).sum())
        inside = np.all(np.isin(tri, conn), axis=1)  # the mesh triangles between this node's vertices
        if np.any(inside):
            t = verts[tri[inside]]
            e_max = max(e_max, float(np.max(np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2))))
        node_ids.append(str(node.id))
        points.append(verts[conn][keep])
        std.append(s[keep])
        source.append(how[keep])
        labels.append(np.full(int(keep.sum()), int(node.attributes.semantic_label), dtype=np.int32))
        offsets.append(offsets[-1] + int(keep.sum()))
    P, S, L = np.vstack(points), np.concatenate(std), np.concatenate(labels)
    n_views, view_dist = point_visibility(P, L, views or [], params) if params is not None else (np.zeros(len(P), np.int32), np.full(len(P), np.nan))
    m_r = e_max / math.sqrt(3.0)
    np.savez_compressed(map_dir / "points.npz", node_ids=np.array(node_ids), offsets=np.array(offsets), points=P, std=S, std_source=np.concatenate(source),
                        n_views=n_views, view_distance=view_dist, e_max=e_max, m_r=m_r, voxel_size=grid.voxel_size, sigma0_sq=sigma0_sq)
    summary = {"nodes": len(node_ids), "points": int(len(P)), "dropped_points": int(sum(dropped.values())), "dropped_by_node": {k: v for k, v in dropped.items() if v},
               "e_max_m": e_max, "m_r_m": m_r, "std_source_counts": {"grid_edge": int(np.sum(np.concatenate(source) == 0)), "bound": int(np.sum(np.concatenate(source) == 1))}}
    (map_dir / "points_summary.json").write_text(json.dumps(summary, indent=1))
    return summary

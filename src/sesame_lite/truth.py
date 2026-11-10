"""Evaluation side only: the exact clearance of the robot's collision spheres to the objects' collision geometry.

Spec v4, section 6.4 keeps the oracle boxes (one axis-aligned box per object) as the true clearance. A box bounds the
object's collision geoms and can lie well outside them, so next to it this module reports the signed distance of every
robot sphere to every collision geom of the objects, as MuJoCo collides them: boxes and cylinders exactly, meshes as
their convex hull (inside: the depth to the nearest hull plane; outside: the distance to the nearest hull triangle).
Geoms more than ``HORIZON_M`` from every sphere are skipped (the result is then above the horizon).
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

HORIZON_M = 1.0


def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from points p [n, 3] to triangles (a, b, c) [t, 3] -> [n, t] (Ericson, Real-Time Collision Detection 5.1.5)."""
    p = p[:, None, :]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("tk,ntk->nt", ab, ap), np.einsum("tk,ntk->nt", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("tk,ntk->nt", ab, bp), np.einsum("tk,ntk->nt", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("tk,ntk->nt", ab, cp), np.einsum("tk,ntk->nt", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v_face, w_face = vb / denom, vc / denom
        q = a + v_face[..., None] * ab + w_face[..., None] * ac  # interior
        # edges and vertices (in order of precedence, as in Ericson)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + w_bc[..., None] * (c - b), q)
        w_ac = d2 / (d2 - d6)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + w_ac[..., None] * ac, q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.broadcast_to(c, q.shape), q)
        v_ab = d1 / (d1 - d3)
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + v_ab[..., None] * ab, q)
        q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.broadcast_to(b, q.shape), q)
        q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.broadcast_to(a, q.shape), q)
    return np.linalg.norm(p - q, axis=2)


class CollisionGeometry:
    """The collision geoms of a set of oracle objects (``geoms``: oracle id -> MuJoCo geom ids)."""

    def __init__(self, model: mujoco.MjModel, geoms: dict[str, list[int]]):
        self.model = model
        self.gid = np.array([g for gs in geoms.values() for g in gs], dtype=np.int64)
        self.owner = [o for o, gs in geoms.items() for _ in gs]
        self.type = model.geom_type[self.gid]
        self.size = model.geom_size[self.gid]
        self.rbound = model.geom_rbound[self.gid]
        self.hull = {}
        for k, g in enumerate(self.gid):
            if self.type[k] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(np.float64)
                h = ConvexHull(v)
                self.hull[k] = (h.equations, v[h.simplices])
            elif self.type[k] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
                raise ValueError(f"no distance rule for geom type {self.type[k]} (geom {g})")

    def clearance(self, data: mujoco.MjData, centres: np.ndarray, radii: np.ndarray, owners: set[str] | None = None) -> tuple[float | None, list | None]:
        """Smallest signed distance between a robot sphere and a geom (of ``owners`` only, if given), and the pair
        (sphere index, oracle id); None if every geom is beyond the horizon."""
        pos, mat = data.geom_xpos[self.gid], data.geom_xmat[self.gid].reshape(-1, 3, 3)
        lb = np.linalg.norm(centres[:, None, :] - pos[None], axis=2) - self.rbound[None] - radii[:, None]
        near = np.flatnonzero(lb.min(axis=0) < HORIZON_M)
        best, pair = None, None
        for k in near:
            if owners is not None and self.owner[k] not in owners:
                continue
            x = (centres - pos[k]) @ mat[k]  # sphere centres in the geom frame
            t, s = self.type[k], self.size[k]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                q = np.abs(x) - s
                sd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                q = np.c_[np.hypot(x[:, 0], x[:, 1]) - s[0], np.abs(x[:, 2]) - s[1]]
                sd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            else:
                eq, tri = self.hull[k]
                plane = x @ eq[:, :3].T + eq[:, 3]
                depth = plane.max(axis=1)
                sd = np.where(depth <= 0.0, depth, 0.0)
                out = depth > 0.0
                if np.any(out):
                    sd[out] = _closest_on_triangles(x[out], tri[:, 0], tri[:, 1], tri[:, 2]).min(axis=1)
            sd = sd - radii
            i = int(np.argmin(sd))
            if best is None or sd[i] < best:
                best, pair = float(sd[i]), [i, self.owner[k]]
        return best, pair

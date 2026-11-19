"""AEGIS's end-effector safety layer, ported from vlsa-aegis at ce5683c (MIT License; third_party/vlsa-aegis).
U = main/utils.py, A = main/main_aegis.py; every function names its source lines.

The gripper is an ellipsoid at the end-effector, the obstacle an ellipsoid fitted once to its points; the barrier h is
the distance between them along a virtual direction z that the QP also moves. Per tick the QP over u = [u_v, u_w, u_z]
(gripper-frame velocity, angular velocity, z rate) keeps dh/dt + 10 h >= 0 nearest to the policy's command.

Changed from their code (Kasra, 2026-11-19): the obstacle is the scene's designated obstacle and its points are its map
node's points (not GLM-4.5V's choice cut out of two depth images by GroundingDINO), without their box crop and their 20%
drop of the farthest points (DBSCAN, convex hull and ellipsoid fit kept); the QP is solved with daqp (their
cvxpy/OSQP), the minimum-volume ellipsoid with Khachiyan's algorithm (their cvxpy/SCS; the same problem), DBSCAN with a
k-d tree (their Open3D; the same rule); a QP without a solution passes the nominal command and is counted (their code
ends the episode); the gripper ellipsoid has the standard size in every task.
"""

from __future__ import annotations

import time

import daqp
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

GRIPPER_AXES_M = np.array([0.06, 0.12, 0.11])  # A:164, semi-axes (their 0.2 m in z for three held objects is not used)
GRIPPER_OFFSET_M = np.array([0.0, 0.0, -0.08])  # A:155, centre along the gripper's -z axis from the end-effector site
GAMMA = 10.0  # A:324, alpha(h) = 10 h (ALPHA = 1.0 at A:30 is not used by their code)
WEIGHTS = np.array([1 / 25] * 6 + [1.0] * 3)  # A:319
REF_GAIN, Z_GAIN, OUT_GAIN = 5.0, 10.0, 0.2  # A:305-307 and A:315 (reference), A:311-312 and A:352-353 (u -> command)
DT_S = 0.05  # A:233
DBSCAN_EPS_M, DBSCAN_MIN_POINTS = 0.0001, 50  # U:304


# ---------------------------------------------------------------------- the barrier (U:40-152)


def vector_hat(v):  # U:40-44
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def project_matrix(z):  # U:46-49
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z_ij):  # U:78-102
    Qbar_i = R_i @ np.diag(Q_i_diag) @ R_i.T
    Qbar_j = R_j @ np.diag(Q_j_diag) @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    z = z_ij / np.linalg.norm(z_ij)
    term1 = np.linalg.norm(Qbar_j @ Qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ Qbar_i_inv @ z
    denom = np.linalg.norm(Qbar_i_inv @ z)
    return (-term1 + term2 - 1.0) / denom


def compute_h_coeffs_3d(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z, eps=1e-10):  # U:104-152
    Qbar_i = R_i @ np.diag(Q_i_diag) @ R_i.T
    Qbar_j = R_j @ np.diag(Q_j_diag) @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    Qbar_i_inv2 = Qbar_i_inv @ Qbar_i_inv
    Qbar_j2 = Qbar_j @ Qbar_j
    z = z / (np.linalg.norm(z) + eps)
    a_vec = Qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = (1.0 - (p_j - p_i).T @ a_vec + term1)
    eta_row = -(1.0 / denom) * (z.T @ Qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ Qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ Qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ Qbar_i_inv @ Qbar_j2 @ Qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3
    tmp1 = z.T @ Qbar_i_inv2 @ vector_hat(z)
    left_vec = (z.T @ Qbar_i_inv @ Qbar_j2)
    tmp2 = left_vec @ (vector_hat(a_vec) - Qbar_i_inv @ vector_hat(z))
    tmp3 = (p_j - p_i).T @ Qbar_i_inv @ vector_hat(z) + z.T @ Qbar_i_inv @ vector_hat(p_j - p_i)
    zeta_tilde = rho * (1.0 / (denom**3 + eps)) * tmp1 + (1.0 / sigma) * tmp2 + (1.0 / denom) * tmp3
    a_v = (eta_row @ R_i).ravel()
    a_omega = zeta_tilde @ R_i
    a_uz = (mu_row @ project_matrix(z)).ravel()
    h = compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z)
    return a_v, a_omega, a_uz, h, mu_row


# ---------------------------------------------------------------------- the obstacle ellipsoid (U:257-445)


def dbscan(pts: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Labels as Open3D's PointCloud.cluster_dbscan (U:303-305): a point with fewer than ``min_points`` points (itself
    included) within ``eps`` is noise (-1) unless a cluster reaches it."""
    nbs = cKDTree(pts).query_ball_point(pts, eps)
    labels, c = np.full(len(pts), -2), 0
    for i in range(len(pts)):
        if labels[i] != -2:
            continue
        if len(nbs[i]) < min_points:
            labels[i] = -1
            continue
        labels[i], stack = c, list(nbs[i])
        while stack:
            j = stack.pop()
            if labels[j] == -1:
                labels[j] = c
            if labels[j] != -2:
                continue
            labels[j] = c
            if len(nbs[j]) >= min_points:
                stack.extend(nbs[j])
        c += 1
    return labels


def filtering_points(pts: np.ndarray) -> np.ndarray:
    """U:257-316, step 2 only: the largest DBSCAN cluster if there is one. Their step 1 (a fixed box in LIBERO's table
    coordinates) and step 1.5 (drop the 20% of the points farthest from their centroid) are left out: the map node's
    points are that object's own points (Kasra, 2026-11-19)."""
    labels = dbscan(pts, DBSCAN_EPS_M, DBSCAN_MIN_POINTS)  # U:302-311
    if labels.max() >= 0:
        pts = pts[labels == np.bincount(labels[labels >= 0]).argmax()]
    return pts


def mvee(P: np.ndarray, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """The minimum-volume ellipsoid {x : (x - c)^T A (x - c) <= 1} around the points (Khachiyan's algorithm), the
    problem their mvee_cvxpy (U:317-351) solves with SCS. Stops at the optimality gap max_i M_i / (d + 1) - 1 <= tol
    (on sink_to_counter_s7's obstacle the semi-axes then differ from those at gap 3e-6 by less than 0.05 mm)."""
    N, d = P.shape
    Q = np.vstack([P.T, np.ones(N)])
    u = np.full(N, 1.0 / N)
    while True:
        M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv((Q * u) @ Q.T), Q)
        j = int(np.argmax(M))
        if M[j] / (d + 1.0) - 1.0 <= tol:
            break
        step = (M[j] - d - 1.0) / ((d + 1.0) * (M[j] - 1.0))
        u = (1.0 - step) * u
        u[j] += step
    c = P.T @ u
    A = np.linalg.inv((P.T * u) @ P - np.outer(c, c)) / d
    return c, A


def fit_ellipse(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # U:431-445
    """Centre, axes (columns) and semi-axes of the minimum-volume ellipsoid around the convex hull of the points."""
    hull_pts = pts[ConvexHull(pts).vertices]
    center, A = mvee(hull_pts)
    eigvals, eigvecs = np.linalg.eigh(A)
    axes = 1.0 / np.sqrt(np.clip(eigvals, 1e-15, None))
    order = np.argsort(axes)[::-1]
    return center, eigvecs[:, order], axes[order]


# ---------------------------------------------------------------------- the safety layer (A:149-164, A:226-234, A:301-356)


def gripper_centre(eef_pos: np.ndarray, R1: np.ndarray) -> np.ndarray:  # A:150-158, A:378-388
    return eef_pos + R1 @ GRIPPER_OFFSET_M


class AegisFilter:
    """The obstacle ellipsoid (fitted once) and the state z; ``step`` filters one command (world frame)."""

    def __init__(self, obstacle_points: np.ndarray, p1: np.ndarray):
        self.points_in = len(obstacle_points)
        self.points = filtering_points(obstacle_points)  # A:217
        self.p2, self.R2, self.Q2 = fit_ellipse(self.points)  # A:227
        A = self.R2 @ np.diag(self.Q2**-2) @ self.R2.T
        self.fit_max = float(np.max(np.einsum("ij,jk,ik->i", self.points - self.p2, A, self.points - self.p2)))  # <= 1: every point inside
        self.z = (self.p2 - p1) / np.linalg.norm(self.p2 - p1)  # A:229-230

    def step(self, command: np.ndarray, p1: np.ndarray, R1: np.ndarray) -> tuple[np.ndarray, dict]:
        """``command``: the policy's end-effector command (position, rotation; their action[:6], world frame).
        Returns the filtered command and what happened."""
        u_ref = np.r_[REF_GAIN * (R1.T @ command[:3]), REF_GAIN * command[3:6]]  # A:304-307
        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, GRIPPER_AXES_M, R1, self.p2, self.Q2, self.R2, self.z)  # A:310
        u_z_nom = Z_GAIN * mu_row  # A:315
        ref = np.r_[u_ref, u_z_nom]  # A:320
        g = np.r_[OUT_GAIN * a_v, OUT_GAIN * a_omega, a_uz]  # A:311-312, A:324
        # A:316-328: min (u - ref)^T W (u - ref)  s.t.  g u + 10 h >= 0, as daqp's 1/2 u^T H u + f^T u with -g u <= 10 h
        t0 = time.perf_counter()
        x, _, flag, _ = daqp.solve(np.diag(2.0 * WEIGHTS), -2.0 * WEIGHTS * ref, np.ascontiguousarray(-g[None]), np.array([GAMMA * h]), np.array([-1e30]), np.zeros(1, dtype=np.intc))
        solve_ms = (time.perf_counter() - t0) * 1e3
        if flag > 0:  # A:330-333
            u_v, u_omega, u_z = x[:3], x[3:6], x[6:]
            out = np.r_[OUT_GAIN * (R1 @ u_v), OUT_GAIN * u_omega]  # A:351-353
        else:  # no solution: the nominal command for this tick (Kasra), z moves with u_z_nom as their fallback (A:334-337)
            u_z, out = u_z_nom, np.asarray(command[:6], float).copy()
        self.z = self.z + (np.eye(3) - np.outer(self.z, self.z)) @ u_z * DT_S  # A:343-346
        self.z = self.z / np.linalg.norm(self.z)
        return out, {"h": float(h), "flag": int(flag), "solve_ms": solve_ms, "constraint_at_ref": float(g @ ref + GAMMA * h),
                     "changed": bool(flag > 0 and np.linalg.norm(x - ref) > 1e-9), "z": self.z.round(6).tolist()}

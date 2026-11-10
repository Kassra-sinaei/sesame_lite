"""The arm driven by joint velocities, with robosuite's end-effector control as its nominal (spec v4, section 6.2).

The QP of PDF eq. (9) acts on u, the joint velocities of the base and the arm. For the arm, Kasra's choice: expose
robosuite's operational-space control (OSC_POSE) to the QP's commands with exactly its PD gains, so that the unfiltered
robot is robosuite's.

``nominal(arm_action, base_mode)`` predicts what robosuite's OSC does over the coming tick: the same torque law
(robosuite's OperationalSpaceController.run_controller: task wrench kp e + kd e_dot through the uncoupled Lambda_pos /
Lambda_ori, the nullspace posture torque, torque compensation and the actuator clip), against the bias forces and the
joints' passive damping, integrated over the tick's physics
substeps the way MuJoCo integrates them (semi-implicit Euler), with M, J and the bias frozen at the tick's start, the
task error moving with J dq and with the arm base's velocity. It returns u_nom, the mean predicted joint velocity over
the tick, and keeps the predicted joint profile q_k, q_dot_k, q_ddot_k.

``JointVelocityDrive`` is robosuite's OperationalSpaceController with ``run_controller`` replaced: at substep k it
tracks the predicted profile shifted by the QP's correction du = u - u_nom,

    q_ref = q_k + du t_k,   q_dot_ref = q_dot_k + du,   tau = M (q_ddot_k + kv (q_dot_ref - q_dot) + kp (q_ref - q)) + bias

so for any u the mean joint velocity over the tick is u (up to the tracking error, which is d). This is our own
low-level controller (RoboCasa's action only carries end-effector deltas, the filter commands joint velocities); the
unfiltered robot is close to robosuite's but not identical (the prediction leaves out e.g. joint friction and air drag,
which RoboCasa's model has; compare_drive.py reports the difference). ``set_goal`` is robosuite's own, so the goal bookkeeping (base frame; "desired" goal in base mode,
"achieved" in arm mode) is unchanged.
"""

from __future__ import annotations

import copy

import numpy as np
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.parts.controller import Controller
from robosuite.utils.control_utils import opspace_matrices, orientation_error

POSTURE_KP = 10.0  # robosuite.utils.control_utils.nullspace_torques default, used by the OSC
MOVING_RAD_S = 1e-3  # the drive compensates a joint's frictionloss while its reference moves faster than this


def pd_mean_velocity(e0: np.ndarray, de0: np.ndarray, omega: float, T: float) -> np.ndarray:
    """Mean of e'(t) over [0, T] for the critically damped e'' = -w^2 e - 2 w e' from (e0, e0') (the torso's PD)."""
    e0, de0 = np.asarray(e0, float), np.asarray(de0, float)
    eT = (e0 + (de0 + omega * e0) * T) * np.exp(-omega * T)
    return (eT - e0) / T


class JointVelocityDrive(OperationalSpaceController):
    """robosuite's OSC, driving the arm joints along a joint profile (see the module docstring)."""

    kv = 100.0  # 1/s, tracking gain of the drive (a tuning value, spec section 8 step 4); kp = kv^2 / 4

    @classmethod
    def install(cls, robot) -> "JointVelocityDrive":
        """Turn the robot's arm controller (built by robosuite at reset) into this drive, keeping all its state."""
        ctrl = robot.composite_controller.part_controllers["right"]
        if type(ctrl) is not OperationalSpaceController:
            raise TypeError(f"expected robosuite's OSC on the arm, got {type(ctrl).__name__}")
        if ctrl.impedance_mode != "fixed" or ctrl.interpolator_pos is not None or ctrl.interpolator_ori is not None or not ctrl.uncoupling or ctrl.input_ref_frame != "base":
            raise ValueError("nominal() reproduces robosuite's fixed-impedance, uncoupled, base-frame OSC without interpolators only")
        ctrl.__class__ = cls
        ctrl.command, ctrl.profile, ctrl.k = None, None, 0
        lo, hi = ctrl.actuator_min, ctrl.actuator_max  # robot.control clips the torques to the actuators' ctrlrange
        ctrl.torque_lo, ctrl.torque_hi = np.asarray(lo, float), np.asarray(hi, float)
        ctrl.damping = np.asarray(ctrl.sim.model.dof_damping[ctrl.qvel_index], float)  # the joints' passive damping
        ctrl.friction = np.asarray(ctrl.sim.model.dof_frictionloss[ctrl.qvel_index], float)  # and dry friction
        return ctrl

    def set_goal(self, action):
        super().set_goal(action)  # robosuite's goal bookkeeping, unchanged
        if self.command is None:
            raise RuntimeError("JointVelocityDrive: no command (u, profile) was given for this tick")
        u, profile = self.command
        self.command = None
        q, dq, ddq, u_nom = profile
        self.profile = (q, dq + (u - u_nom), ddq, u - u_nom)
        self.k = 0

    def run_controller(self):
        self.update()
        q_k, dq_ref, ddq, du = self.profile
        k = self.k
        if k >= len(q_k):
            raise RuntimeError(f"JointVelocityDrive: substep {k} beyond the tick's profile ({len(q_k)} substeps)")
        q_ref = q_k[k] + du * k * self.model_timestep
        kp = self.kv**2 / 4.0
        acc = ddq[k] + self.kv * (dq_ref[k] - self.joint_vel) + kp * (q_ref - self.joint_pos)
        moving = np.abs(dq_ref[k]) > MOVING_RAD_S
        self.torques = self.mass_matrix @ acc + self.torque_compensation + self.damping * self.joint_vel + np.where(moving, self.friction * np.sign(dq_ref[k]), 0.0)
        self.k = k + 1
        Controller.run_controller(self)  # robosuite's bookkeeping (the next update reads the state again)
        return self.torques

    def nominal(self, arm_action: np.ndarray, base_mode: bool, n_substeps: int) -> tuple[np.ndarray, tuple]:
        """u_nom of the arm for the coming tick and the predicted profile (see the module docstring).

        robosuite's own set_goal runs on a copy, so the controller's state is not touched; env.step then gives the same
        action to the real controller."""
        c = copy.copy(self)
        c.set_goal_update_mode("desired" if base_mode else "achieved")
        OperationalSpaceController.set_goal(c, np.asarray(arm_action, dtype=float))
        sim, dt = c.sim, c.model_timestep
        site = f"{c.naming_prefix}{c.part_name}_center"
        base_v, base_w = np.array(sim.data.get_site_xvelp(site)), np.array(sim.data.get_site_xvelr(site))
        M, Jp, Jo, J = c.mass_matrix, c.J_pos, c.J_ori, c.J_full
        _, lam_p, lam_o, N = opspace_matrices(M, J, Jp, Jo)
        bias = np.array(c.torque_compensation)
        q0, dq0 = np.array(c.joint_pos), np.array(c.joint_vel)
        goal_p0 = c.origin_pos + c.origin_ori @ c.goal_pos
        goal_v = base_v + np.cross(base_w, c.origin_ori @ c.goal_pos)  # the goal moves with the arm's base
        e_p0 = goal_p0 - c.ref_pos
        e_o0 = orientation_error(c.origin_ori @ c.goal_ori, c.ref_ori_mat)
        off_p, off_o = c.ref_pos_vel - Jp @ dq0, c.ref_ori_vel - Jo @ dq0  # end-effector velocity not due to the arm joints
        kp, kd = c.kp, c.kd
        kv_n = 2.0 * np.sqrt(POSTURE_KP)
        q, dq = q0.copy(), dq0.copy()
        Q, DQ, DDQ = [], [], []
        for k in range(n_substeps):
            t = k * dt
            e_p = e_p0 + (goal_v - off_p) * t - Jp @ (q - q0)  # the goal and the hand both move with the base
            e_o = e_o0 + (base_w - off_o) * t - Jo @ (q - q0)
            de_p = -(Jp @ dq + off_p - base_v)
            de_o = -(Jo @ dq + off_o - base_w)
            wrench = np.r_[lam_p @ (kp[:3] * e_p + kd[:3] * de_p), lam_o @ (kp[3:] * e_o + kd[3:] * de_o)]
            tau = J.T @ wrench + bias + N.T @ (M @ (POSTURE_KP * (c.initial_joint - q) - kv_n * dq))
            tau = np.clip(tau, self.torque_lo, self.torque_hi)
            ddq = np.linalg.solve(M, tau - bias - self.damping * dq)  # the OSC's torque against bias and passive damping
            Q.append(q.copy())
            DQ.append(dq.copy())
            DDQ.append(ddq)
            dq = dq + ddq * dt
            q = q + dq * dt
        u_nom = (q - q0) / (n_substeps * dt)
        return u_nom, (np.array(Q), np.array(DQ), np.array(DDQ), u_nom)

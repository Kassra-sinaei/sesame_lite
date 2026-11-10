"""The PandaOmron as collision spheres, from cuRobo (spec v4, section 4; PDF, Setting).

cuRobo is used for the robot side only: the sphere model of config/panda_omron_spheres.yml (arm and hand from cuRobo's
Franka sphere set, base and column fitted to the robot's collision geometry, see scripts/make_robot_spheres.py) and its
batched kinematics on the GPU. The kinematic tree is config/panda_omron.urdf, written from the MuJoCo model, so cuRobo
and MuJoCo agree by construction (checked by make_robot_spheres.py).

    q = [base forward, base side, base yaw, torso, joint1..7, finger1, finger2]    (MuJoCo's joints, same order)

``spheres(q)`` returns the world centres and radii; ``jacobians(q)`` also the centres' derivatives d s_i / d q, built
from cuRobo's link poses: for every joint, its world axis a_j and origin o_j (the pose of the joint's child link), and
column j of sphere i is a_j x (s_i - o_j) for a hinge or a_j for a slide, if joint j moves sphere i. (cuRobo's own
tool Jacobian disagreed with MuJoCo on the slide joints and on some hinge columns; autograd through its kernels does
not batch.) The robot's root body is fixed in the world; its pose is read once from MuJoCo.
"""

from __future__ import annotations

from pathlib import Path

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "config" / "panda_omron.urdf"
SPHERES_YML = REPO / "config" / "panda_omron_spheres.yml"
JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw", "mobilebase0_joint_torso_height",
          *(f"robot0_joint{k}" for k in range(1, 8)), "gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
BASE_DOF = slice(0, 3)  # the joints the filter commands: base (3) and arm (7)
TORSO_DOF = 3
ARM_DOF = slice(4, 11)
FINGER_DOF = slice(11, 13)
COMMANDED = np.r_[0:3, 4:11]  # u = velocities of these joints; torso and fingers enter as known terms


class RobotSpheres:
    """cuRobo kinematics of the PandaOmron's collision spheres, in the world frame."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, device: str = "cuda"):
        from curobo.kinematics import Kinematics, KinematicsCfg

        cfg = yaml.safe_load(SPHERES_YML.read_text())
        # joints of the URDF: type, child link (its frame is the joint frame), axis, and which links each one moves
        urdf = ET.parse(URDF).getroot()
        parent_of = {j.find("child").get("link"): (j.get("name"), j.find("parent").get("link")) for j in urdf.findall("joint")}
        info = {j.get("name"): (j.get("type"), j.find("child").get("link"), np.array([float(v) for v in j.find("axis").get("xyz").split()]))
                for j in urdf.findall("joint") if j.get("type") != "fixed"}
        self.joint_child = [info[j][1] for j in JOINTS]
        self.joint_axis = torch.tensor(np.array([info[j][2] for j in JOINTS]), dtype=torch.float32, device=device)
        self.joint_hinge = torch.tensor([info[j][0] == "revolute" for j in JOINTS], device=device)
        sphere_links = [link for link, spheres in cfg["robot_cfg"]["kinematics"]["collision_spheres"].items() for _ in spheres]
        moves = np.zeros((len(sphere_links), len(JOINTS)), dtype=bool)
        for i, link in enumerate(sphere_links):
            while link in parent_of:
                jname, link = parent_of[link]
                if jname in JOINTS:
                    moves[i, JOINTS.index(jname)] = True
        self.moves = torch.tensor(moves, device=device)
        self.kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg, urdf_path=str(URDF), tool_frames=self.joint_child))
        self.device = device
        assert list(self.kin.joint_names) == list(JOINTS), f"joint order differs: {self.kin.joint_names}"
        self.qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["robot_cfg"]["kinematics"]["base_link"])
        self.world_R_root = torch.tensor(data.xmat[root].reshape(3, 3), dtype=torch.float32, device=device)
        self.world_p_root = torch.tensor(data.xpos[root], dtype=torch.float32, device=device)
        self.links = sphere_links
        self.n = len(self.links)

    def q(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_adr], dtype=np.float64)

    def qdot(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_adr], dtype=np.float64)

    def _state(self, q: torch.Tensor):
        from curobo.types import JointState

        return self.kin.compute_kinematics(JointState.from_position(q, joint_names=list(JOINTS)))

    def _centres(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        s = state.robot_spheres[:, 0]  # [batch, n, 4] in the root frame
        return s[..., :3] @ self.world_R_root.T + self.world_p_root, s[..., 3]

    def spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World centres [n, 3] and radii [n] at one configuration."""
        with torch.no_grad():
            c, r = self._centres(self._state(torch.as_tensor(np.atleast_2d(q), dtype=torch.float32, device=self.device)))
        return c[0].cpu().numpy().astype(np.float64), r[0].cpu().numpy().astype(np.float64)

    def jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres [n, 3], radii [n] and Jacobians d s_i / d q [n, 3, 13] at one configuration, world frame."""
        with torch.no_grad():
            state = self._state(torch.as_tensor(np.atleast_2d(q), dtype=torch.float32, device=self.device))
            c_root = state.robot_spheres[0, 0, :, :3]
            r = state.robot_spheres[0, 0, :, 3]
            pos = state.tool_poses.position.reshape(-1, len(JOINTS), 3)[0]  # joint origins, root frame
            quat = state.tool_poses.quaternion.reshape(-1, len(JOINTS), 4)[0]  # (w, x, y, z)
            w, x, y, z = quat.unbind(-1)
            R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                             2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                             2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
            axis = torch.einsum("jab,jb->ja", R, self.joint_axis)  # [13, 3], root frame
            lever = c_root[:, None, :] - pos[None, :, :]  # [n, 13, 3]
            col = torch.where(self.joint_hinge[None, :, None], torch.linalg.cross(axis[None].expand_as(lever), lever, dim=-1), axis[None].expand_as(lever))
            J = (col * self.moves[:, :, None]).transpose(1, 2)  # [n, 3, 13]
            J_world = torch.einsum("ij,njk->nik", self.world_R_root, J)
            c = c_root @ self.world_R_root.T + self.world_p_root
        return c.cpu().numpy().astype(np.float64), r.cpu().numpy().astype(np.float64), J_world.cpu().numpy().astype(np.float64)

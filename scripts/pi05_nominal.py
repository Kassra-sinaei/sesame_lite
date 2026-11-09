"""lerobot/pi05_robocasa as the VLA nominal (server replacement for REKI-2B / SmolVLA). Added on the server;
the original code is untouched.

Same pipeline as ``sesame_lite.env.SmolVLAPolicy`` (lerobot's own pre/post-processors, the checkpoint's saved
camera keys, which are the raw RoboCasa names, so no rename map), plus one fact about this checkpoint:
it was trained on pepijn223/robocasa_pretrain_human300_v4, which keeps RoboCasa's native state/action order,
while lerobot's RoboCasaEnv (and so ``SesameRoboCasaEnv`` and the safety filter) use another order:

    env    state  = base_pos(3) base_quat(4) ee_pos_rel(3) ee_quat_rel(4) gripper_qpos(2)
    policy state  = ee_pos_rel(3) ee_quat_rel(4) base_pos(3) base_quat(4) gripper_qpos(2)
    env    action = base_motion(4) control_mode(1) ee_pos(3) ee_rot(3) gripper(1)
    policy action = ee_pos(3) ee_rot(3) gripper(1) base_motion(4) control_mode(1)

The permutation is applied on the policy side only; the env, the filter (MODE_INDEX etc.) and the receipts stay in
env order. Constants are the ones the monitor project verified (monitor/src/robocasa_env.py, 10/20 check).
The model runs uncompiled (see the note in ``__init__``).
"""

from __future__ import annotations

import numpy as np

POLICY_PATH = "lerobot/pi05_robocasa"
STATE_FROM_ENV = np.r_[7:10, 10:14, 0:3, 3:7, 14:16]  # policy state[i] = env state[STATE_FROM_ENV[i]]
ACTION_TO_ENV = np.r_[7:11, 11, 0:3, 3:6, 6]  # env action[j] = policy action[ACTION_TO_ENV[j]]


class Pi05Policy:
    """Drop-in for ``SmolVLAPolicy``: ``reset()`` and ``__call__(obs, language) -> 12-D env-order action``."""

    def __init__(self, task: str, device: str = "cuda"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
        cfg.pretrained_path, cfg.device = POLICY_PATH, device
        # The checkpoint config sets compile_model=True. torch.compile's first call, made in the same process as
        # MuJoCo's EGL renderer, corrupts every later camera frame (verified 2026-11-04: frames turn to noise after
        # the first inference; eager mode keeps them clean). lerobot-eval avoids it by running the env in a
        # subprocess. Eager mode: same weights and math.
        cfg.compile_model = False
        env_cfg = RoboCasaEnvConfig(task=task)
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=POLICY_PATH, preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, self.env_post = env_cfg.get_env_processors()
        self.reset()

    def reset(self):
        for p in (self.policy, self.pre, self.post):
            p.reset()

    def __call__(self, obs: dict, language: str) -> np.ndarray:
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        state = np.asarray(obs["agent_pos"], dtype=np.float32)[STATE_FROM_ENV]
        observation = preprocess_observation({"pixels": {k: v[None] for k, v in obs["pixels"].items()}, "agent_pos": state[None]})
        observation["task"] = [language]
        observation = self.pre(self.env_pre(observation))
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        a = self.env_post({ACTION: self.post(action)})[ACTION].cpu().numpy()[0].astype(np.float32)
        return a[ACTION_TO_ENV]

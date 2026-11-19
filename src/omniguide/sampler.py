"""OmniGuide's collision guidance on Xiaomi-Robotics-1's flow-matching sampler (runs in .venv-xr1, next to the model).

Sources. OmniGuide d649d69 with its gr00t submodule ef38d07, under the NVIDIA License, so nothing is copied: every
function below is rewritten and names the lines it follows. G = omniguide-gr00t/gr00t/model/gr00t_n1d6/gr00t_n1d6.py,
E = omniguide-gr00t/collision/esdf.py, X = models/Xiaomi-Robotics-1-RoboCasa365/modeling_mibot.py (Apache-2.0).

One query, as G:get_action_with_features (G:379-829), with the collision term only (Kasra, 2026-11-19):
  1. the standard pass: XR1's 5 Euler steps (X:1866-1874), unchanged; with guidance off its result is the output
  2. the noise pick (G:579-675): 4 new noises run 2 steps each; after the second step's velocity the clean chunk is
     estimated, x + (1 - t) v, and scored; the noise with the highest value is kept, the first pass's own noise
     (scored the same way) included, so it is kept unless a new noise scores strictly higher (G:656)
  3. the guided pass from that noise (G:677-806): at every step the estimate is decoded, its value is
     back-propagated through the action expert to x, and x <- x + dt v + guidance_factor * grad (G:806, no clipping)
The value (G:562-574) is collision_weight * the sum of log d over the robot points with d <= 0.15 m, d from the
distance field: higher is farther from the map. Their code takes log d as it is, with no guard (d = 0 on an occupied
voxel gives -inf); ours clamps d to LOG_FLOOR_M first. The robot points (G:312-342) are the gripper and the point
0.25 m along its -z axis at each of the 16 steps, from integrating the position deltas (x GAMMA_POS) and rotation
deltas (x GAMMA_ROT) from the pose at the query, in the base frame; they are taken into the world with the base pose
at the query (the base's own motion during the chunk is left out, as in theirs).
"""

from __future__ import annotations

import time

import numpy as np
import torch

NUM_STEPS = 5  # XR1's sampler (X:1825, xr1_server.py uses the default)
PICK_STEPS, N_RESTARTS = 2, 4  # G:630, G:598
SDF_DIST_M = 0.15  # G:562
TOP_OFFSET_M = 0.25  # G:320-322: the second point, along the gripper's -z axis
COLLISION_WEIGHT = 0.1  # G:101 (Kasra: fixed at 0.1)
LOG_FLOOR_M = 1e-3  # ours: d is clamped to this before the log (their code has no guard, G:571-574)
# Our drive's scale factors (theirs: [0.011, 0.011, 0.02] and 0.15, G:323, G:338), fitted from the stored Nominal logs
# by omniguide/fit_gamma.py (results/v3/omniguide/gamma_fit.json)
GAMMA_POS = (0.01057, 0.01155, 0.00954)
GAMMA_ROT = 0.0806


# ---------------------------------------------------------------- distance field (rewrite of E:105-262)


def _trilinear(grid: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """grid (X, Y, Z, C) sampled at continuous voxel indices u (N, 3), clamped to the grid (border padding)."""
    top = torch.tensor(grid.shape[:3], device=u.device, dtype=u.dtype) - 1
    u = torch.minimum(u.clamp(min=0.0), top)
    i0 = torch.minimum(u.floor(), top - 1).long()
    f = u - i0
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w[:, None] * grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


class _Lookup(torch.autograd.Function):
    """Value: trilinear sample of the distance (E:118-140). Gradient: trilinear sample of the field's finite
    differences, normalised to unit length (a direction only), zero for points outside the grid (E:144-155)."""

    @staticmethod
    def forward(ctx, p, field):
        u = (p - field.origin) / field.voxel
        ctx.save_for_backward(u, ((p >= field.origin) & (p <= field.top)).all(dim=-1))
        ctx.field = field
        return _trilinear(field.d[..., None], u)[:, 0]

    @staticmethod
    def backward(ctx, g):
        u, in_grid = ctx.saved_tensors
        n = _trilinear(ctx.field.grad, u)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return g[:, None] * n * in_grid[:, None], None


class DistanceField:
    """The scene's distance field (omniguide/field.py) on the GPU, float32."""

    def __init__(self, sdf: np.ndarray, origin: np.ndarray, voxel: float, device="cuda"):
        d = torch.from_numpy(np.ascontiguousarray(sdf, dtype=np.float32)).to(device)
        self.d, self.voxel = d, float(voxel)
        self.origin = torch.as_tensor(origin, dtype=torch.float32, device=device)
        self.top = self.origin + (torch.tensor(d.shape, dtype=torch.float32, device=device) - 1) * self.voxel
        g = torch.zeros(d.shape + (3,), dtype=torch.float32, device=device)  # E:235-262: central inside, one-sided at the faces
        for a in range(3):
            dn = d.movedim(a, 0)
            ga = g[..., a].movedim(a, 0)
            ga[1:-1] = (dn[2:] - dn[:-2]) / (2 * self.voxel)
            ga[0], ga[-1] = (dn[1] - dn[0]) / self.voxel, (dn[-1] - dn[-2]) / self.voxel
        self.grad = g

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return _Lookup.apply(p, self)


# ---------------------------------------------------------------- robot points and value (rewrite of G:312-342, G:562-574)


def _axis_angle_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, (..., 3) -> (..., 3, 3)."""
    theta = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / theta
    K = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, dtype=r.dtype, device=r.device) + s * K + (1 - c) * (K @ K)


def quat_xyzw_matrix(q) -> torch.Tensor:
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=torch.float32)


def query_pose(guide: dict, device="cuda") -> dict:
    """The pose at the query (RoboCasa's raw observation): gripper in the base frame, base in the world."""
    return {"p0": torch.as_tensor(np.asarray(guide["ee_pos_rel"], dtype=np.float32), device=device), "R0": quat_xyzw_matrix(guide["ee_quat_rel"]).to(device),
            "base_p": torch.as_tensor(np.asarray(guide["base_pos"], dtype=np.float32), device=device), "base_R": quat_xyzw_matrix(guide["base_quat"]).to(device)}


def robot_points(a: torch.Tensor, pose: dict) -> torch.Tensor:
    """Decoded actions (B, T, >= 6; RoboCasa order: position delta 0:3, rotation delta 3:6) -> world points (B, T, 2, 3)."""
    gp = torch.tensor(GAMMA_POS, dtype=torch.float32, device=a.device)
    pos = torch.cumsum(a[..., 0:3], dim=1) * gp + pose["p0"]  # G:335-340
    step = _axis_angle_matrix(a[..., 3:6] * GAMMA_ROT)  # G:323
    R, acc = pose["R0"].expand(a.shape[0], 3, 3), []
    for k in range(a.shape[1]):  # G:324-330: each step's rotation applied on the left
        R = step[:, k] @ R
        acc.append(R)
    top = pos - TOP_OFFSET_M * torch.stack(acc, dim=1)[..., :, 2]  # G:331, G:341
    return pose["base_p"] + torch.stack([pos, top], dim=2) @ pose["base_R"].T


def collision_value(a: torch.Tensor, pose: dict, field: DistanceField, weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,) collision_weight * sum of log d over the points with d <= 0.15 m (G:562-574; d clamped, ours), and d (B, 2T)."""
    p = robot_points(a, pose).reshape(a.shape[0], -1, 3)
    d = field(p.reshape(-1, 3)).reshape(a.shape[0], -1)
    return weight * ((d <= SDF_DIST_M) * torch.log(d.clamp(min=LOG_FLOOR_M))).sum(dim=1), d


# ---------------------------------------------------------------- the sampler


class Expert:
    """XR1's action expert on one query's language-model cache: X:1828-1862 unchanged, called as v = expert(x, t)."""

    def __init__(self, model, state, action_mask, kwargs):
        vlm_outputs = model.vlm(**kwargs, use_cache=True)
        action_bs, action_length, _ = action_mask.shape
        _, state_length, _ = state.shape
        dit_query_length = action_length + state_length + 1
        position_ids = (torch.arange(0, dit_query_length, device=action_mask.device).view(1, 1, -1).repeat(3, action_bs, 1)
                        + vlm_outputs.position_ids.max(dim=-1)[0][..., None] + 1)
        position_embeds = model.rotary_emb(action_mask, position_ids)
        dit_mask = torch.tril(torch.ones((action_bs, dit_query_length, dit_query_length), device=action_mask.device), diagonal=0)
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, dit_query_length, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        self.model = model
        self.args = dict(action_mask=action_mask, state_embed=model.state_projector(state), position_embeds=position_embeds,
                         past_key_values=vlm_outputs.past_key_values, attn_mask=attn_mask)

    def __call__(self, x, t):
        return self.model.dit_forward(noisy_action=x, t=t, **self.args)

    def repeated(self, n: int) -> "Expert":
        """The same context for a batch of n chunks (G:613-626 repeats theirs the same way)."""
        rep = lambda v: v.repeat(n, *([1] * (v.dim() - 1)))  # noqa: E731
        out = object.__new__(Expert)
        out.model, a = self.model, self.args
        pkv, start = a["past_key_values"], len(a["past_key_values"]) - len(self.model.dit.layers)
        out.args = dict(action_mask=rep(a["action_mask"]), state_embed=rep(a["state_embed"]), position_embeds=tuple(rep(v) for v in a["position_embeds"]),
                        past_key_values=[None] * start + [tuple(rep(v) for v in pkv[i]) for i in range(start, len(pkv))], attn_mask=rep(a["attn_mask"]))
        return out


def _t(x, step):  # X:1871, unchanged
    return torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / NUM_STEPS


def _sync():
    torch.cuda.synchronize()
    return time.time()


def sample(model, data: dict, seed: int, decode, guide: dict | None = None, field: DistanceField | None = None) -> tuple[torch.Tensor, dict]:
    """The chunk for one query (normalised, (1, 16, 60)) and a record of the guidance. ``guide``: None (standard sampler)
    or {"guidance_factor", the pose at the query (query_pose)}; ``decode``: normalised -> RoboCasa actions (float32)."""
    state, action_mask = data["state"], data["action_mask"]
    kwargs = {k: v for k, v in data.items() if k not in ("state", "action_mask")}
    dt = 1.0 / NUM_STEPS
    t0 = _sync()
    torch.manual_seed(seed)
    with torch.no_grad():  # 1. the standard pass (X:1820-1874)
        expert = Expert(model, state, action_mask, kwargs)
        x = torch.randn_like(action_mask)
        noise = x.clone()
        for step in range(NUM_STEPS):
            v = expert(x, _t(x, step))
            if step == PICK_STEPS - 1:
                estimate = x + (1.0 - step / NUM_STEPS) * v  # G:411-412
            x = x + v * dt
    info = {"standard_s": round(_sync() - t0, 4)}
    if guide is None or field is None:
        return x, info
    gf, pose = float(guide["guidance_factor"]), query_pose(guide)
    with torch.no_grad():
        value_std, d_std = collision_value(decode(x), pose, field, COLLISION_WEIGHT)
    info.update(guidance_factor=gf, value_standard=float(value_std[0]), d_min_standard=float(d_std.min()), near_standard=int((d_std <= SDF_DIST_M).sum()))
    if gf <= 0.0:  # G:424-432: no guidance
        return x, info
    t1 = _sync()
    with torch.no_grad():  # 2. the noise pick (G:579-675)
        best_value, best, picked = float(collision_value(decode(estimate), pose, field, COLLISION_WEIGHT)[0][0]), noise, -1
        y = torch.randn((N_RESTARTS,) + tuple(action_mask.shape[1:]), dtype=action_mask.dtype, device=action_mask.device)  # G:607-611
        candidates, many = y.clone(), expert.repeated(N_RESTARTS)
        for step in range(PICK_STEPS):
            v = many(y, _t(y, step))
            if step == PICK_STEPS - 1:
                values = collision_value(decode(y + (1.0 - step / NUM_STEPS) * v), pose, field, COLLISION_WEIGHT)[0].tolist()
            y = y + v * dt
    info["value_pick"] = [best_value] + [round(v, 5) for v in values]
    for i, value in enumerate(values):  # G:652-660: strictly higher replaces
        if value > best_value:
            best_value, best, picked = value, candidates[i:i + 1], i
    info["picked"] = picked
    t2 = _sync()
    x, steps = best.clone(), []
    # ours: strict deterministic mode for the backward passes (in warn-only mode, the server's setting, the attention
    # backward takes a non-deterministic algorithm and reruns of a query differ); the forward kernels are unchanged
    mode = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        for step in range(NUM_STEPS):  # 3. the guided pass (G:677-806)
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                v = expert(x, _t(x, step))
                value, d = collision_value(decode(x + (1.0 - step / NUM_STEPS) * v), pose, field, COLLISION_WEIGHT)  # G:688-711
                (grad,) = torch.autograd.grad(value.sum(), x)  # G:771-773
            with torch.no_grad():
                x = x + dt * v + gf * grad  # G:806
            steps.append({"value": round(float(value.detach()[0]), 5), "near": int((d <= SDF_DIST_M).sum()), "grad_max": float(grad.abs().max())})
    finally:
        torch.use_deterministic_algorithms(mode[0], warn_only=mode[1])
    x = x.detach()
    with torch.no_grad():
        value_out, d_out = collision_value(decode(x), pose, field, COLLISION_WEIGHT)
    info.update(steps=steps, value_out=float(value_out[0]), d_min_out=float(d_out.min()), near_out=int((d_out <= SDF_DIST_M).sum()),
                pick_s=round(t2 - t1, 4), guided_s=round(_sync() - t2, 4))
    return x, info

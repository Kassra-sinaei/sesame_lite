"""Checks of the guided sampler on one real query (step 1 of the OmniGuide row).

    MUJOCO_GL=egl PYTHONPATH=src:scripts .venv/bin/python -s src/omniguide/check_query.py capture <scene> <scenes root> <stored Proposed run .json> <dir>
    .venv-xr1/bin/python src/omniguide/check_query.py check <dir>

capture (simulation side): runs the row's navigation segment (Proposed), compares it tick by tick with a stored Proposed
run of the scene, builds the scene's field and saves the request of the first manipulation query.
check (server side, one model in this process): guidance 0 against the standard server's infer (xr1_server.py) with the
same seed; reruns with the same seed; the same query with the gripper pose lowered so that the guidance acts; time
per query. Writes <dir>/check.json.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def capture(scene_name: str, root: Path, stored: Path, out: Path) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)  # as scripts/eval_run.py
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    import sesame_lite.tasks  # noqa: F401
    from omniguide.client import GuidedXR1Policy, OmniGuideEnv
    from omniguide.field import build_field
    from sesame_lite.env import MissionRunner, SesameRoboCasaEnv
    from sesame_lite.filter import FilterPolicy

    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads((root / scene_name / "scene.json").read_text())
    base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=1800)
    env = OmniGuideEnv(base, map_kind="full", disturbance="a2", policy=FilterPolicy(alpha=4.0, d_bar=0.03), pair_pruning="rule", classifier="qwen")  # frozen_v2
    obs, _ = env.reset(scene_dir=root / scene_name)
    field = build_field(env)
    np.savez(out / "field.npz", sdf=field["sdf"], origin=field["origin"], voxel_m=field["voxel_m"])

    class Stop(Exception):
        pass

    class Capture(GuidedXR1Policy):  # the client's request, saved instead of sent
        def __init__(self):
            self.env, self.seed, self.queries, self.filter_env, self.guidance_factor, self.log = base, scene["seed"], 0, env, 0.0, []
            self.reset()

        def _ask(self, req):
            (out / "request.pkl").write_bytes(pickle.dumps(req))
            raise Stop

    torch.manual_seed(scene["seed"])
    try:
        MissionRunner(env, vla=Capture()).run(obs)
    except Stop:
        pass
    d = json.loads(stored.read_text())
    n = sum(r["segment"] == 0 for r in d["receipts"])
    mine = [r for r in env.receipts if r["segment"] == 0]
    q = np.load(stored.with_suffix(".npz"))["q"][:n]
    res = {"scene": scene_name, "stored_run": str(stored), "field": field["meta"], "query_tick": env.tick, "filter_on_at_query": env.filter_on,
           "navigation_ticks": [len(mine), n], "navigation_u_identical": len(mine) == n and all(a["u_applied"] == b["u_applied"] for a, b in zip(mine, d["receipts"])),
           "navigation_q_max_diff": float(np.abs(np.array(env.q_log[:n]) - q).max()), "snapshot_identical": d["result"]["snapshot_id"] == env.oracle_graph.snapshot_id}
    (out / "capture.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    base.close()


def check(out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from sampler import DistanceField
    from server import GuidedServer
    from xr1_server import Server

    srv = GuidedServer("/workspace/sesame_lite/models/Xiaomi-Robotics-1-RoboCasa365")
    req = pickle.loads((out / "request.pkl").read_bytes())
    f = np.load(out / "field.npz")
    field = DistanceField(f["sdf"], f["origin"], float(f["voxel_m"]))

    def timed(fn, n=3):
        res, ts = [], []
        for _ in range(n):
            t0 = time.time()
            res.append(fn())
            ts.append(round(time.time() - t0, 3))
        return res, ts

    std, t_std = timed(lambda: Server.infer(srv, req))
    res = {"seed": req["seed"], "standard_reruns_identical": all(np.array_equal(a, std[0]) for a in std[1:]), "standard_s": t_std}
    for dz in (0.0, -0.35):  # the pose as captured, and the gripper lowered by 0.35 m (robot points near the counter)
        pose = {**req["guide"], "ee_pos_rel": np.asarray(req["guide"]["ee_pos_rel"]) + np.array([0.0, 0.0, dz])}
        for gf in (0.0, 0.5, 2.0):
            runs, ts = timed(lambda: srv.guided({**req, "guide": {**pose, "guidance_factor": gf}}, field))
            info = {k: v for k, v in runs[0]["info"].items() if k != "steps"}
            res[f"dz{dz}_gf{gf}"] = {"equal_to_standard": bool(np.array_equal(runs[0]["actions"], std[0])), "max_diff_to_standard": float(np.abs(runs[0]["actions"] - std[0]).max()),
                                     "reruns_identical": all(np.array_equal(r["actions"], runs[0]["actions"]) for r in runs[1:]), "wall_s": ts, "info": info}
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        check(Path(sys.argv[2]))

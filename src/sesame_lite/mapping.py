"""The scene graph from Hydra: a mapping pass renders depth and label images of the RoboCasa kitchen and feeds them
to Hydra (MIT-SPARK, built from third_party/hydra with the OBB fit), as a robot would map a kitchen before working
in it. The output is cached per scene in outputs/maps/<mission>_s<seed>_<n>/:

    dsg.json        Hydra's scene graph with its mesh (spark_dsg file format)
    voxels.npz      every observed TSDF voxel near a surface (index, distance, weight), from the TsdfSink plugin
    faces.json      per object node: category, box and the six faces (offset, sigma, observed), see scene_graph.py
    meta.json       oracle snapshot id, instance -> category table, parameters, timing and counts

Simulator ground truth is used here only to render the camera images (depth, segmentation, colour), for the camera
route (free points of the oracle graph's place grid: the operator's route, not map content) and, in faces.json, for
the evaluation-only ``observed`` flag. Hydra sees images and camera poses only.

Labels stand in for a panoptic segmenter: one label per fixture or object instance (oracle node O(k) -> instance
k + 1), the floor is a surface label, robot and unknown geoms are label 0 (invalid: Hydra skips those pixels).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import yaml

from sesame_lite.scene_graph import LabelSpace, OracleLookup, SceneGraph, _raw_data, _raw_model, _subtree_body_ids

REPO = Path(__file__).resolve().parents[2]
MAPS_DIR = REPO / "outputs" / "maps"
HYDRA_INSTALL = REPO / "third_party" / "install"
SINK_LIBRARY_DIR = REPO / "third_party" / "build" / "sesame_hydra"  # sesame_hydra.<abi>.so (src/sesame_hydra)


@dataclass(frozen=True)
class MappingParams:
    """Camera, map and uncertainty parameters of the mapping pass (spec section 6; filter ones are in FilterPolicy)."""

    depth_noise_c: float = 3.6e-3  # 1/m: depth noise std sigma_z = c z^2 (D435 class)
    width: int = 640
    height: int = 480
    fovy_deg: float = 58.0
    min_range_m: float = 0.3
    max_range_m: float = 4.0
    voxel_size_m: float = 0.05
    truncation_voxels: int = 3
    max_weight: float = 1.0e7
    bounding_box_type: str = "OBB"
    min_cluster_size: int = 10  # Hydra's habitat default is 40; lowered once on the design scene: the flour bag had 39 vertices (REPORT.md)
    camera_height_m: float = 1.2
    yaw_step_deg: float = 30.0
    pitches_deg: tuple[float, ...] = (-10.0, -45.0)
    route_spacing_m: float = 0.5  # the oracle place grid's spacing
    route_min_clearance_m: float = 0.3
    frame_dt_s: float = 0.1  # timestamp spacing between frames
    full_update_separation_s: float = 1.0  # Hydra extracts mesh and objects at most this often (timestamp time)
    labels: str = "instance"  # "instance" (default) or "category" (the check of 4.8 item 5)

    @property
    def fy(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.fovy_deg) / 2)

    @property
    def fx(self) -> float:
        return self.fy

    @property
    def cx(self) -> float:
        return self.width / 2 - 0.5  # pixel index i has its centre at u = i in Hydra's projection (camera.cpp)

    @property
    def cy(self) -> float:
        return self.height / 2 - 0.5

    @property
    def sigma0_sq(self) -> float:
        """Var[d_v] = sigma0_sq / w_v for Hydra's stock weight fx fy l^2 / z^4 and depth noise c z^2 (PDF eq. 1)."""
        return self.depth_noise_c**2 * self.fx * self.fy * self.voxel_size_m**2

    @property
    def truncation_m(self) -> float:
        return self.truncation_voxels * self.voxel_size_m

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, sigma0_sq=self.sigma0_sq)
        return d


# --------------------------------------------------------------------------- labels


def instance_table(rs_env, oracle_graph: SceneGraph, lookup: OracleLookup, labels: LabelSpace, mode: str) -> dict:
    """Label per MuJoCo geom, the scene's Hydra label space and the label -> category / oracle node tables."""
    model = _raw_model(rs_env.sim)
    oracle_ids = list(oracle_graph.nodes)  # O(0), O(1), ... (sorted by position, no task information)
    body_label = {}
    if mode == "instance":
        label_of_node = {oid: k + 1 for k, oid in enumerate(oracle_ids)}
        floor_label = len(oracle_ids) + 1
        category_of = {label_of_node[oid]: oracle_graph.nodes[oid].label for oid in oracle_ids}
        names = {lab: f"instance {lab}" for lab in category_of} | {0: "unknown", floor_label: "floor"}
    else:
        label_of_node = {oid: oracle_graph.nodes[oid].label for oid in oracle_ids}
        floor_label = labels.floor_label
        category_of = {lab: lab for lab in set(label_of_node.values())}
        names = {lab: labels.names[lab] for lab in category_of} | {0: "unknown", floor_label: "floor"}
    for oid in oracle_ids:
        for b in lookup.bodies[oid]:
            body_label[b] = label_of_node[oid]
    for fx in rs_env.fixtures.values():
        if type(fx).__name__ == "Floor" and getattr(fx, "root_body", None):
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fx.root_body)
            for b in _subtree_body_ids(model, root):
                body_label[b] = floor_label
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rs_env.robots[0].robot_model.root_body)
    for b in _subtree_body_ids(model, robot_root):
        body_label[b] = 0  # the robot is invalid (masked out of the fusion)
    geom_label = np.array([body_label.get(int(model.geom_bodyid[g]), 0) for g in range(model.ngeom)], dtype=np.int32)
    object_labels = sorted(set(label_of_node.values()))
    labelspace = {
        "total_semantic_labels": int(max(names) + 1),
        "dynamic_labels": [],
        "invalid_labels": [0],
        "object_labels": object_labels,
        "surface_places_labels": [floor_label],
        "label_names": [{"label": int(k), "name": names[k]} for k in sorted(names)],
    }
    oracle_of_label = {}
    for oid, lab in label_of_node.items():
        oracle_of_label.setdefault(lab, []).append(oid)
    return {"geom_label": geom_label, "labelspace": labelspace, "category_of": category_of, "oracle_of_label": oracle_of_label, "mode": mode}


# --------------------------------------------------------------------------- camera


class MappingCamera:
    """Depth, labels and colour from one free camera over the live kitchen (visual geoms only, like viz.WorldRenderer).

    Depth is planar (metres along the optical axis), with Gaussian noise of std c z^2 per pixel, seeded by the scene
    seed; pixels out of range or on the background are NaN (invalid for Hydra). Poses are returned for Hydra's
    optical frame (x right, y down, z forward).
    """

    def __init__(self, rs_env, params: MappingParams, geom_label: np.ndarray, seed: int):
        self.p, self.geom_label = params, geom_label
        self.model, self.data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), params.width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), params.height)
        self.renderer = mujoco.Renderer(self.model, params.height, params.width, max_geom=40000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1  # visual geoms
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _update(self, position, yaw_deg: float, pitch_deg: float):
        # a free camera at `position` looking along (yaw, pitch): MuJoCo places it at lookat - distance * forward
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.0, yaw_deg, pitch_deg
        self.cam.lookat[:] = np.asarray(position, dtype=float) + forward
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and world_R_optical of the camera of the last update (from the scene's GL camera)."""
        cams = self.renderer.scene.camera
        pos = (np.asarray(cams[0].pos) + np.asarray(cams[1].pos)) / 2
        forward, up = np.asarray(cams[0].forward, dtype=float), np.asarray(cams[0].up, dtype=float)
        R = np.column_stack([np.cross(forward, up), -up, forward])
        return pos, R

    def render(self, position, yaw_deg: float, pitch_deg: float, *, noise: bool = True):
        """depth (float32, NaN invalid), labels (int32), rgb (uint8), camera position, world_R_optical."""
        fovy = float(self.model.vis.global_.fovy)
        self.model.vis.global_.fovy = self.p.fovy_deg  # a free camera uses the global fovy
        try:
            self._update(position, yaw_deg, pitch_deg)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().astype(np.float64)
            self.renderer.disable_depth_rendering()
            self.renderer.enable_segmentation_rendering()
            seg = self.renderer.render().copy()
            self.renderer.disable_segmentation_rendering()
            pos, R = self.pose()
        finally:
            self.model.vis.global_.fovy = fovy
        is_geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        labels = np.zeros(depth.shape, dtype=np.int32)
        labels[is_geom] = self.geom_label[seg[..., 0][is_geom]]
        far = self.model.vis.map.zfar * self.model.stat.extent
        valid = is_geom & (depth > self.p.min_range_m) & (depth < self.p.max_range_m) & (depth < 0.999 * far)
        if noise:
            depth = depth + self.rng.normal(0.0, 1.0, depth.shape) * self.p.depth_noise_c * depth**2
        depth = np.where(valid, depth, np.nan).astype(np.float32)
        labels[~valid] = 0
        return depth, labels, rgb, pos, R

    def close(self) -> None:
        self.renderer.close()


def back_project(depth: np.ndarray, params: MappingParams, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World points of the valid pixels of a depth image (optical frame -> world)."""
    v, u = np.nonzero(np.isfinite(depth))
    z = depth[v, u].astype(np.float64)
    pts_c = np.stack([(u - params.cx) / params.fx * z, (v - params.cy) / params.fy * z, z], axis=1)
    return pts_c @ R.T + pos


def check_camera(cam: MappingCamera, poses: list, oracle_graph: SceneGraph, table: dict, max_median_m: float = 0.02) -> dict:
    """Spec 4.3: back-project noise-free depth from two poses; the median distance of the object pixels to the nearest
    oracle box surface must be below 2 cm. A frame or axis mistake fails here instead of giving a silently wrong map."""
    lo = np.array([n.aabb()[0] for n in oracle_graph.nodes.values()])
    hi = np.array([n.aabb()[1] for n in oracle_graph.nodes.values()])
    floor = table["labelspace"]["surface_places_labels"][0]
    out = []
    for i in (0, len(poses) // 2):
        depth, labels, _, pos, R = cam.render(*poses[i], noise=False)
        keep = np.isfinite(depth) & (labels > 0) & (labels != floor)
        pts = back_project(np.where(keep, depth, np.nan), cam.p, pos, R)
        d = np.full(len(pts), np.inf)
        for l, h in zip(lo, hi):
            outside = np.linalg.norm(np.maximum(np.maximum(l - pts, pts - h), 0.0), axis=1)
            inside = np.all((pts >= l) & (pts <= h), axis=1)
            d = np.minimum(d, np.where(inside, np.min(np.minimum(pts - l, h - pts), axis=1), outside))
        out.append({"pose": [float(v) for v in poses[i][0]] + [poses[i][1], poses[i][2]], "pixels": int(len(pts)), "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90))})
    cam.rng = np.random.default_rng(cam.seed)  # the check does not use up noise draws of the mapping frames
    if not all(c["median_m"] < max_median_m for c in out):
        raise RuntimeError(f"camera check failed (median distance to the oracle boxes >= 2 cm): {out}")
    return {"passed": True, "poses": out}


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


# --------------------------------------------------------------------------- route


def mapping_route(oracle_graph: SceneGraph, rs_env, params: MappingParams) -> tuple[list[tuple[np.ndarray, float, float]], list]:
    """Camera poses: every place of the oracle grid with clearance >= route_min_clearance_m, at camera height,
    yaw every yaw_step_deg, each pitch. Places where the camera would sit inside the robot are skipped (listed)."""
    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rs_env.robots[0].robot_model.root_body)
    robot_bodies = _subtree_body_ids(model, robot_root)
    gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies and int(model.geom_group[g]) == 1]
    robot_xy = np.array([data.geom_xpos[g][:2] for g in gids])
    robot_r = float(np.max(np.linalg.norm(robot_xy - robot_xy.mean(axis=0), axis=1))) + 0.3
    poses, skipped = [], []
    for pid, (p, clearance) in sorted(oracle_graph.places.items(), key=lambda kv: (kv[1][0][0], kv[1][0][1])):
        if clearance < params.route_min_clearance_m:
            continue
        if np.linalg.norm(p[:2] - robot_xy.mean(axis=0)) < robot_r:
            skipped.append(pid)
            continue
        for k in range(int(round(360 / params.yaw_step_deg))):
            for pitch in params.pitches_deg:
                poses.append((np.array([p[0], p[1], params.camera_height_m]), k * params.yaw_step_deg, pitch))
    return poses, skipped


# --------------------------------------------------------------------------- Hydra


def _hydra():
    """hydra_python plus the TsdfSink plugin (registered once per process)."""
    import importlib.util
    import sys

    import hydra_python as hydra

    if "sesame_hydra" not in sys.modules:
        lib = next(SINK_LIBRARY_DIR.glob("sesame_hydra*.so"))
        cfg = hydra.ExternalPluginConfig()
        cfg.paths = [str(lib)]
        hydra.init_plugins(cfg)  # registers "TsdfSink" with Hydra's config factory
        spec = importlib.util.spec_from_file_location("sesame_hydra", lib)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # the same library, now as a Python module (shares the voxel store)
        sys.modules["sesame_hydra"] = module
    return hydra, sys.modules["sesame_hydra"]


def hydra_config(params: MappingParams, labelspace: dict) -> dict:
    """Hydra's habitat dataset config (a simulated RGB-D camera) with our label space and the spec's settings."""
    hydra, _ = _hydra()
    contents = yaml.safe_load((hydra.get_config_path() / "datasets" / "habitat.yaml").read_text())
    hydra.update_nested(contents, labelspace)
    hydra.update_nested(contents, {
        "frontend": {"type": "GraphBuilder", "objects": {"bounding_box_type": params.bounding_box_type, "min_cluster_size": params.min_cluster_size}},
        "backend": {"type": "BackendModule"},
        "active_window": {
            "volumetric_map": {"voxel_size": params.voxel_size_m, "truncation_distance": params.truncation_m},
            "full_update_separation_s": params.full_update_separation_s,
            "tsdf": {"use_constant_weight": False, "max_weight": params.max_weight},
            "sinks": [{"type": "TsdfSink"}],
        },
    })
    # Hydra's GVD (free-space places) needs its min distance below the truncation distance: 0.15 m at 5 cm voxels,
    # while the habitat config has 0.2 m (written for 10 cm voxels). One voxel below the truncation.
    gvd = contents["frontend"]["freespace_places"]["gvd"]
    gvd["min_distance_m"] = min(gvd["min_distance_m"], params.truncation_m - params.voxel_size_m)
    return contents


def run_hydra(frames, n_frames: int, params: MappingParams, labelspace: dict, out_dir: Path) -> dict:
    """Feed the frames to Hydra; write dsg.json (with mesh) and voxels.npz. ``frames`` yields (depth, labels, rgb, pos, R)."""
    hydra, sink = _hydra()
    sink.clear()
    camera = hydra.make_camera(params.fx, params.fy, params.cx, params.cy, params.width, params.height, params.min_range_m, params.max_range_m)
    config = hydra_config(params, labelspace)
    (out_dir / "hydra_config.yaml").write_text(yaml.safe_dump(config))
    pipeline = hydra.HydraPipeline.from_config(yaml.safe_dump(config), camera)
    t_ns, n = 0, 0
    for i, (depth, labels, rgb, pos, R) in enumerate(frames):
        # the last frame comes a full update separation later, so Hydra extracts the mesh and objects after it
        t_ns += int(1e9 * (params.full_update_separation_s if i == n_frames - 1 else params.frame_dt_s))
        pipeline.step(t_ns, np.asarray(pos, dtype=np.float64), quat_wxyz(R), depth, labels, rgb)
        n += 1
    pipeline.save_graph(str(out_dir / "dsg.json"), True)
    vox = sink.tsdf_voxels()
    np.savez_compressed(out_dir / "voxels.npz", index=vox["index"], distance=vox["distance"], weight=vox["weight"],
                        voxel_size=vox["voxel_size"], origin=vox["origin"], truncation_distance=vox["truncation_distance"])
    stats = {"frames": n, "sink_calls": int(vox["num_calls"]), "voxels": int(len(vox["weight"]))}
    del pipeline
    return stats


def map_dir(mission: str, seed: int, n: int, labels: str = "instance") -> Path:
    return MAPS_DIR / (f"{mission}_s{seed}_{n}" + ("" if labels == "instance" else f"_{labels}"))


def build_map(rs_env, oracle_graph: SceneGraph, lookup: OracleLookup, out_dir: Path, *, seed: int, params: MappingParams = MappingParams()) -> Path:
    """Mapping pass over the current kitchen state; writes the cache directory (see the module docstring)."""
    from sesame_lite.uncertainty import write_faces

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = instance_table(rs_env, oracle_graph, lookup, oracle_graph.labels, params.labels)
    poses, skipped = mapping_route(oracle_graph, rs_env, params)
    cam = MappingCamera(rs_env, params, table["geom_label"], seed)
    try:
        camera_check = check_camera(cam, poses, oracle_graph, table)  # before Hydra sees a single frame
        stats = run_hydra((cam.render(*pose) for pose in poses), len(poses), params, table["labelspace"], out_dir)
    finally:
        cam.close()
    t_map = time.time() - t0
    meta = {
        "oracle_snapshot_id": oracle_graph.snapshot_id,
        "label_mode": params.labels,
        "category_of_label": {str(k): int(v) for k, v in table["category_of"].items()},
        "oracle_of_label": {str(k): v for k, v in table["oracle_of_label"].items()},  # evaluation side only
        "labelspace": table["labelspace"],
        "params": params.as_dict(),
        "route": {"camera_poses": len(poses), "places_skipped_inside_robot": skipped},
        "camera_check": camera_check,
        "hydra": stats,
        "seconds": round(t_map, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    write_faces(out_dir, rs_env, oracle_graph, lookup)
    return out_dir

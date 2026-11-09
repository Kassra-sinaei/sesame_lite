"""RoboCasa task variants for the study (registered with robosuite on import).

``NavigateKitchenMats``: NavigateKitchen plus flat floor objects the base may drive over (lightwheel
placemats: 0.8 cm tall, the analog of the paper's doormat) and one object it must not (a flour bag).
RoboCasa cannot sample objects on the floor, so they are spawned on a counter and moved onto the
planned path after reset (``place_on_path``), before the 3DSG is captured.

``PickPlaceSinkToCounterFar``: RoboCasa's PickPlaceSinkToCounter (same objects, same success check), but the robot
starts in front of the fixture farthest from the sink, so a plan needs a navigation segment before the VLA.
"""

from __future__ import annotations

import mujoco
import numpy as np
from robocasa.environments.kitchen.atomic.kitchen_navigate import NavigateKitchen
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceSinkToCounter
from robocasa.environments.kitchen.kitchen import FixtureType

from sesame_lite.scene_graph import _raw_data, _raw_model

FLOOR_OBJECTS = [  # name, category, passable in ground truth
    ("mat_1", "placemat", True),
    ("mat_2", "placemat", True),
    ("bag_1", "flour_bag", False),
]
PATH_FRACTIONS = {"mat_1": 0.35, "mat_2": 0.65, "bag_1": 0.5}
PATH_OFFSETS_M = {"mat_1": 0.0, "mat_2": 0.0, "bag_1": 0.45}  # lateral offset: the bag sits beside the path, within the corridor


class NavigateKitchenMats(NavigateKitchen):
    def _get_obj_cfgs(self):
        counter = self.get_fixture(FixtureType.COUNTER)
        return [
            dict(name=name, obj_groups=category, graspable=False, init_robot_here=False, placement=dict(fixture=counter, size=(0.6, 0.4), pos=(None, -1.0), offset=(0.0, 0.0)))
            for name, category, _ in FLOOR_OBJECTS
        ]


START_FIXTURE_CLASSES = ("Stove", "Stovetop", "Oven", "Microwave", "Dishwasher", "CoffeeMachine", "Toaster", "ToasterOven",
                         "FridgeFrenchDoor", "FridgeSideBySide", "FridgeBottomFreezer", "HingeCabinet", "SingleCabinet")


class PickPlaceSinkToCounterFar(PickPlaceSinkToCounter):
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()  # registers the sink and the counter next to it
        sink_xy = np.asarray(self.sink.pos[:2])
        candidates = [fx for fx in self.fixtures.values() if type(fx).__name__ in START_FIXTURE_CLASSES]
        # the robot starts in front of the fixture farthest from the sink (a fixed rule, no randomness)
        self.init_robot_base_ref = max(candidates, key=lambda fx: float(np.linalg.norm(np.asarray(fx.pos[:2]) - sink_xy)))


def place_on_path(rs_env, path_xy: np.ndarray) -> None:
    """Drop the floor objects on the planned base path (free-joint teleport + forward), flat on the floor."""
    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    seg = np.diff(path_xy, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    for name, _, _ in FLOOR_OBJECTS:
        if name not in rs_env.objects:
            continue
        s = PATH_FRACTIONS[name] * cum[-1]
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        t = (s - cum[i]) / max(np.linalg.norm(seg[i]), 1e-9)
        p = path_xy[i] + t * seg[i]
        n = np.array([-seg[i][1], seg[i][0]]) / max(np.linalg.norm(seg[i]), 1e-9)
        p = p + PATH_OFFSETS_M[name] * n
        body = rs_env.obj_body_id[name]
        gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and int(model.geom_group[g]) == 0]
        z_lo = min(float(model.geom_aabb[g][2] - model.geom_aabb[g][5]) for g in gids) if gids else 0.0
        jid = next(j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE))
        adr = int(model.jnt_qposadr[jid])
        yaw = float(np.arctan2(seg[i][1], seg[i][0]))
        data.qpos[adr : adr + 7] = [p[0], p[1], -z_lo + 0.002, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        data.qvel[int(model.jnt_dofadr[jid]) : int(model.jnt_dofadr[jid]) + 6] = 0.0
    mujoco.mj_forward(model, data)


def ground_truth_passable() -> set[str]:
    """Oracle names of the floor objects the base may drive over (map to graph nodes with OracleLookup.node)."""
    return {f"object:{name}" for name, _, passable in FLOOR_OBJECTS if passable}

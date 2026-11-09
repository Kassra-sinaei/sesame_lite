"""Write config/robocasa_label_space.yaml: a Hydra label space for our RoboCasa kitchens (run once).

The file has the same fields as Hydra's own files (Hydra repo, config/label_spaces/*.yaml). The names are
category words only (RoboCasa fixture classes and object categories), never instance names.

Numbering rule:
    0          unknown (invalid)
    1 ..       fixture classes, alphabetical: every robocasa Fixture subclass except the abstract bases,
               plus Box and Wall (robocasa's plain box fixtures)
    next       floor (a surface label: places are built over it, it is not an object)
    next ..    object categories of the asset packs the env loads (objaverse, lightwheel), alphabetical

    MUJOCO_GL=egl .venv/bin/python -s scripts/make_label_space.py
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import yaml

import robocasa.models.fixtures as fixtures_pkg
from robocasa.models.fixtures.fixture import Fixture
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

OBJECT_PACKS = ("objaverse", "lightwheel")  # the packs SesameRoboCasaEnv loads (obj_registries)
ABSTRACT_FIXTURES = {"Fixture", "ProcGenFixture"}
OUT = Path(__file__).resolve().parents[1] / "config" / "robocasa_label_space.yaml"


def words(class_name: str) -> str:
    """HingeCabinet -> hinge cabinet"""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).lower()


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


fixture_classes = {"Box", "Wall"}
for module in pkgutil.walk_packages(fixtures_pkg.__path__, fixtures_pkg.__name__ + "."):
    for name, cls in inspect.getmembers(importlib.import_module(module.name), inspect.isclass):
        if issubclass(cls, Fixture) and name not in ABSTRACT_FIXTURES:
            fixture_classes.add(name)

fixture_names = sorted(words(c) for c in fixture_classes)
object_names = sorted(c.replace("_", " ") for c, packs in OBJ_CATEGORIES.items() if any(p in packs for p in OBJECT_PACKS))
object_names = [n for n in object_names if n not in fixture_names]  # one label per word

names = ["unknown"] + fixture_names + ["floor"] + object_names
floor = names.index("floor")
space = {
    "total_semantic_labels": len(names),
    "dynamic_labels": [],
    "invalid_labels": [0],
    "object_labels": [i for i in range(1, len(names)) if i != floor],
    "surface_places_labels": [floor],
    "label_names": [{"label": i, "name": n, "name_descriptive": article(n)} for i, n in enumerate(names)],
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("---\n" + yaml.safe_dump(space, sort_keys=False, default_flow_style=None, width=120))
print(f"wrote {OUT}: {len(fixture_names)} fixture labels, {len(object_names)} object labels")

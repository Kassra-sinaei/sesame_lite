"""Deterministic kitchens: identical processes and seeds give identical RoboCasa scenes."""

import random

import numpy as np


def seed_fallback_generators() -> None:
    """robosuite and RoboCasa fall back to ``np.random.default_rng()`` (OS entropy) in several places: placement
    samplers (placement_samplers.py:42), fixtures (fixture.py:215, fixture_stack.py:70, others.py:76, windows.py:121),
    object sampling (kitchen_object_utils.py:358), robots (robot.py:246). Route every unseeded ``default_rng()`` to a
    deterministic per-episode sequence so that identical processes generate identical kitchens."""
    if getattr(np.random.default_rng, "_sesame_seeded", False):
        return
    orig = np.random.default_rng

    def default_rng(seed=None, *args, **kwargs):
        if seed is None:
            seed = (SAMPLER_SEED[0] * 1_000_003 + SAMPLER_SEED[1]) % (2**32)
            SAMPLER_SEED[1] += 1
        return orig(seed, *args, **kwargs)

    default_rng._sesame_seeded = True
    np.random.default_rng = default_rng
    _order_counter_regions()


def _order_counter_regions() -> None:
    """``Counter.get_reset_regions`` deduplicates candidate top geoms with ``list(set(valid_geoms))``; the geoms are
    lxml elements hashed by address, so the region order, and the region ``rng.choice`` then picks, changes from
    process to process (counter.py:813; the only builtin ``set`` call in that module). Bind an insertion-ordered
    dedupe in its place: same candidate regions, deterministic order."""
    from robocasa.models.fixtures import counter

    counter.set = dict.fromkeys


SAMPLER_SEED = [0, 0]  # [episode seed, call counter]


def begin_episode(seed: int) -> None:
    """Unseeded ``default_rng()`` calls of this episode follow a sequence keyed by ``seed``; the global generators,
    which robosuite and RoboCasa also draw from, are seeded too."""
    SAMPLER_SEED[:] = [10_000 + seed, 0]
    random.seed(seed)
    np.random.seed(seed % 2**32)

"""Unit tests for classic Conway + Ember (numpy only)."""

from __future__ import annotations

import numpy as np

from blitz.data.conway import (
    LEVEL_ON_CLASSIC,
    pattern_preview_grid,
    render_frame,
    seed_pattern,
    step_classic,
    step_with_ember,
)


def test_blinker_oscillates() -> None:
    rng = np.random.default_rng(0)
    g0 = seed_pattern("Blinker", 9, 9, rng)
    g1 = step_classic(g0, wrap=True)
    g2 = step_classic(g1, wrap=True)
    assert g0.sum() == 3
    assert np.array_equal(g0, g2)


def test_classic_render_is_01() -> None:
    alive = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    frame = render_frame(alive, scale=1, ember_mode=False)
    assert set(np.unique(frame)) <= {0, LEVEL_ON_CLASSIC}
    assert frame[0, 1] == 1
    assert frame[1, 0] == 1


def test_ember_ladder_0_to_n() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    nxt, trail = step_with_ember(grid, wrap=True, decay_gens=3)
    assert nxt.sum() == 0
    assert trail[2, 2] == 2  # N-1
    frame = render_frame(nxt, scale=1, ember=trail, ember_mode=True, decay_gens=3)
    assert frame[2, 2] == 2
    nxt, trail = step_with_ember(nxt, trail, wrap=True, decay_gens=3)
    assert trail[2, 2] == 1
    nxt, trail = step_with_ember(nxt, trail, wrap=True, decay_gens=3)
    assert trail[2, 2] == 0


def test_ember_alive_is_n() -> None:
    alive = np.zeros((3, 3), dtype=np.uint8)
    alive[1, 1] = 1
    frame = render_frame(alive, scale=1, ember_mode=True, decay_gens=3)
    assert frame[1, 1] == 3
    assert set(np.unique(frame)) <= {0, 1, 2, 3}


def test_rectangular_seed() -> None:
    rng = np.random.default_rng(7)
    g = seed_pattern("Random", 128, 64, rng, density=0.25)
    assert g.shape == (128, 64)


def test_pattern_preview_grid() -> None:
    g = pattern_preview_grid("Glider", size=16)
    assert g.shape[0] >= 8 and g.shape[1] >= 8
    assert g.sum() == 5


def test_ember_not_alive_for_rules() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    nxt, trail = step_with_ember(grid, wrap=True, decay_gens=3)
    again, _ = step_with_ember(nxt, trail, wrap=True, decay_gens=3)
    assert again.sum() == 0

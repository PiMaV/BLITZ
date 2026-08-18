"""SparseCube correctness. Does not import the BLITZ package."""

import sys
from pathlib import Path

import numpy as np

_SIDECAR = Path(__file__).resolve().parents[1] / "benchmarks" / "sparse_sidecar"
sys.path.insert(0, str(_SIDECAR))

from sparse_cube import SparseCube, apply_floor_abs  # noqa: E402


def test_from_dense_roundtrip_and_floor():
    rng = np.random.default_rng(0)
    dense = rng.random((4, 8, 8), dtype=np.float32)
    dense[dense < 0.7] = 0
    cube = SparseCube.from_dense(dense)
    assert cube.dense_frame(1).shape == (8, 8)
    np.testing.assert_array_equal(cube.to_dense(), dense)
    np.testing.assert_array_equal(cube.dense_frame(2), dense[2])
    assert cube.nbytes == cube.coords.nbytes + cube.values.nbytes
    assert cube.occupancy == cube.nnz / dense.size

    floored = apply_floor_abs(dense.copy(), 0.9)
    cubed = SparseCube.from_dense(dense, floor_abs=0.9)
    np.testing.assert_array_equal(cubed.to_dense(), floored)


def test_empty_frame():
    dense = np.zeros((3, 4, 5), dtype=np.float32)
    dense[1, 2, 3] = 1.5
    cube = SparseCube.from_dense(dense)
    assert cube.nnz == 1
    np.testing.assert_array_equal(cube.dense_frame(0), np.zeros((4, 5), dtype=np.float32))
    assert cube.dense_frame(1)[2, 3] == 1.5


def test_reduce_max_mean_match_dense():
    rng = np.random.default_rng(1)
    dense = rng.random((6, 5, 7), dtype=np.float32)
    dense[dense < 0.6] = 0
    cube = SparseCube.from_dense(dense)
    np.testing.assert_allclose(cube.reduce_max(), dense.max(axis=0), rtol=1e-5)
    np.testing.assert_allclose(
        cube.reduce_mean(), dense.mean(axis=0), rtol=1e-5, atol=1e-6
    )

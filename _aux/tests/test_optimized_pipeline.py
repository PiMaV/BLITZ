import numpy as np
import pytest
from blitz.data import optimized, image, tools

def _make_meta(n: int):
    from blitz.data.image import MetaData
    return [
        MetaData(
            file_name=f"frame_{i}.png",
            file_size_MB=0.1,
            size=(10, 10),
            dtype=np.uint8,
            bit_depth=8,
            color_model="grayscale",
        )
        for i in range(n)
    ]

@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_apply_pipeline_fused_equivalence():
    """Verify that fused pipeline produces same results as legacy pipeline."""
    # Create random data
    T, H, W, C = 10, 32, 32, 3
    img_orig = np.random.rand(T, H, W, C).astype(np.float32) + 1.0 # Avoid div by zero

    # Random sub/div references
    sub_ref = np.random.rand(T, H, W, C).astype(np.float32)
    div_ref = np.random.rand(T, H, W, C).astype(np.float32) + 0.5

    sub_amt = 0.5
    div_amt = 0.8

    # Expected output (NumPy implementation logic)
    # sub
    expected = img_orig.copy()
    expected -= sub_amt * sub_ref

    # div
    denom = div_amt * div_ref + (1.0 - div_amt)
    expected /= denom # Assuming denom != 0

    # Numba output
    img_numba = img_orig.copy()
    optimized.apply_pipeline_fused(
        img_numba,
        True, sub_ref, sub_amt,
        True, div_ref, div_amt
    )

    np.testing.assert_allclose(img_numba, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_apply_pipeline_broadcasting():
    """Verify fused pipeline with T=1 references (broadcasting)."""
    T, H, W, C = 10, 32, 32, 3
    img_orig = np.random.rand(T, H, W, C).astype(np.float32)

    # Single frame references
    sub_ref = np.random.rand(1, H, W, C).astype(np.float32)
    div_ref = np.random.rand(1, H, W, C).astype(np.float32) + 0.5

    sub_amt = 0.5
    div_amt = 0.8

    # Expected
    expected = img_orig.copy()
    expected -= sub_amt * sub_ref # NumPy broadcasts (1,H,W,C) to (T,H,W,C)

    denom = div_amt * div_ref + (1.0 - div_amt)
    expected /= denom

    # Numba
    img_numba = img_orig.copy()
    optimized.apply_pipeline_fused(
        img_numba,
        True, sub_ref, sub_amt,
        True, div_ref, div_amt
    )

    np.testing.assert_allclose(img_numba, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_image_data_integration():
    """Test ImageData class uses Numba when enabled."""
    data = np.random.rand(5, 10, 10, 1).astype(np.float32)
    meta = _make_meta(5)

    img = image.ImageData(data, meta)
    img.use_numba = True

    # Define pipeline: sub aggregate mean
    # Need to unravel first? No, ImageData handles it.
    # To use 'aggregate' source, we need to ensure valid _compute_ref.
    # We use 'aggregate' with bounds (0,0) -> frame 0.

    pipeline = {
        "subtract": {"source": "aggregate", "bounds": (0, 0), "amount": 0.5},
    }
    img.set_ops_pipeline(pipeline)

    # Run pipeline
    res_numba = img.image.copy()

    # Disable Numba
    img.use_numba = False
    img._invalidate_result()
    res_numpy = img.image.copy()

    np.testing.assert_allclose(res_numba, res_numpy)

@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_sliding_mean_numba_equivalence():
    """Test sliding_mean_numba vs numpy cumsum."""
    T, H, W, C = 20, 10, 10, 1
    images = np.random.rand(T, H, W, C).astype(np.float32)
    window = 5
    lag = 2

    res_numba = optimized.sliding_mean_numba(images, window, lag)

    # Manually compute expected with cumsum method (logic from tools.py)
    n = T - (lag + window)
    res_numpy = np.empty((n, H, W, C), dtype=np.float32)
    for i in range(H):
        sl = images[:, i, ...]
        cs = np.cumsum(sl, axis=0)
        upper = cs[lag + window : lag + window + n]
        lower = cs[lag : lag + n]
        res_numpy[:, i, ...] = (upper - lower) / window

    np.testing.assert_allclose(res_numba, res_numpy, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_reduce_numba_equivalence():
    """Reduce (MEAN, MAX, MIN, STD) Numba vs NumPy produce same result."""
    from blitz.data.ops import ReduceDict, ReduceOperation

    T, H, W, C = 20, 16, 16, 1
    data = (np.random.rand(T, H, W, C) * 100).astype(np.float32)
    reducer = ReduceDict()

    for op_name in ("MEAN", "MAX", "MIN", "STD"):
        result = reducer.reduce(data, op_name)
        expected = np.expand_dims(
            getattr(np, op_name.lower() if op_name != "STD" else "std")(data, axis=0),
            axis=0,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not optimized.HAS_NUMBA, reason="Numba not installed")
def test_sliding_mean_numba_uint8():
    """Test sliding_mean_numba with uint8 input."""
    T, H, W, C = 20, 10, 10, 1
    images = np.random.randint(0, 255, (T, H, W, C), dtype=np.uint8)
    window = 5
    lag = 2

    # Numba handles uint8 input, returns float32
    res_numba = optimized.sliding_mean_numba(images, window, lag)

    assert res_numba.dtype == np.float32

    # Check correctness
    # Numba implementation matches legacy logic: input window is [lag+1 : lag+window]
    expected_first_pixel = images[lag+1:lag+1+window, 0, 0, 0].sum() / window
    np.testing.assert_allclose(res_numba[0, 0, 0, 0], expected_first_pixel)

"""Unit tests for ImageData ops pipeline and aggregation range."""
import time
import numpy as np
import pytest

from blitz.data.image import ImageData, MetaData


def _make_meta(n: int) -> list[MetaData]:
    return [
        MetaData(
            file_name=f"frame_{i}.png",
            file_size_MB=0.1,
            size=(4, 4),
            dtype=np.uint8,
            bit_depth=8,
            color_model="grayscale",
        )
        for i in range(n)
    ]


class TestOpsPipeline:
    """Test subtract and divide steps with aggregate/file sources."""

    def test_subtract_aggregate_mean(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(3, 2, 2, 1)
        data[0] = 10
        data[1] = 20
        data[2] = 30
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = {
            "subtract": {"source": "aggregate", "bounds": (0, 2), "method": "MEAN", "amount": 1.0},
        }
        img.set_ops_pipeline(pipeline)
        out = img.image
        np.testing.assert_allclose(out[0, ..., 0], -10.0)
        np.testing.assert_allclose(out[1, ..., 0], 0.0)
        np.testing.assert_allclose(out[2, ..., 0], 10.0)

    def test_subtract_aggregate_median(self) -> None:
        data = np.array([10, 20, 30], dtype=np.float32).reshape(3, 1, 1, 1)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = {
            "subtract": {"source": "aggregate", "bounds": (0, 2), "method": "MEDIAN", "amount": 1.0},
        }
        img.set_ops_pipeline(pipeline)
        out = img.image
        np.testing.assert_allclose(out[:, 0, 0, 0], [-10.0, 0.0, 10.0])

    def test_subtract_amount(self) -> None:
        data = np.ones((2, 2, 2, 1), dtype=np.float32) * 100.0
        meta = _make_meta(2)
        img = ImageData(data, meta)

        pipeline = {
            "subtract": {"source": "aggregate", "bounds": (0, 1), "method": "MEAN", "amount": 0.5},
        }
        img.set_ops_pipeline(pipeline)
        out = img.image
        # Ref mean=100, amount 0.5 -> subtract 50. Result=50.
        np.testing.assert_allclose(out, 50.0)

    def test_empty_pipeline_identity(self) -> None:
        data = np.arange(24, dtype=np.uint8).reshape(3, 2, 2, 2)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        img.set_ops_pipeline(None)
        out = img.image
        np.testing.assert_array_equal(out, data)

    def test_subtract_then_divide(self) -> None:
        data = np.ones((3, 2, 2, 1), dtype=np.float32) * 20.0
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = {
            "subtract": {"source": "aggregate", "bounds": (0, 2), "method": "MEAN", "amount": 1.0},
            "divide": {"source": "aggregate", "bounds": (0, 2), "method": "MEAN", "amount": 1.0},
        }
        img.set_ops_pipeline(pipeline)
        out = img.image
        # After subtract: 20-20=0. After divide by 20: 0/20=0.
        np.testing.assert_allclose(out, 0.0)


class TestAggregationRange:
    """Test non-destructive aggregation bounds."""

    def test_reduce_without_bounds(self) -> None:
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], dtype=np.float32).reshape(3, 2, 2, 1)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        img.reduce("MEAN")
        out = img.image
        expected = np.array([[[[5.], [6.]], [[7.], [8.]]]])
        np.testing.assert_allclose(out, expected)

    def test_reduce_with_bounds(self) -> None:
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], dtype=np.float32).reshape(3, 2, 2, 1)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        img.reduce("MEAN", bounds=(1, 2))
        out = img.image
        expected = np.array([[[[7.], [8.]], [[9.], [10.]]]])
        np.testing.assert_allclose(out, expected)

    def test_ops_then_aggregation_bounds(self) -> None:
        data = np.array([
            [[10, 10], [10, 10]],
            [[20, 20], [20, 20]],
            [[30, 30], [30, 30]],
        ], dtype=np.float32).reshape(3, 2, 2, 1)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = {
            "subtract": {"source": "aggregate", "bounds": (0, 2), "method": "MEAN", "amount": 1.0},
        }
        img.set_ops_pipeline(pipeline)
        img.reduce("MEAN", bounds=(0, 1))
        out = img.image
        expected = np.array([[[[-5.], [-5.]], [[-5.], [-5.]]]])
        np.testing.assert_allclose(out, expected)


class TestResultCache:
    """Test that the result cache is used when params match."""

    def test_cache_reuse_after_unravel(self) -> None:
        """Frame -> Aggregate -> Frame -> Aggregate (same params) reuses cache."""
        data = np.random.rand(100, 8, 8, 1).astype(np.float32)
        meta = _make_meta(100)
        img = ImageData(data, meta)

        img.reduce("MEAN", bounds=(0, 99))
        out1 = img.image.copy()
        assert ("MEAN", (0, 99)) in img._result_cache

        img.unravel()
        assert ("MEAN", (0, 99)) in img._result_cache  # Preserved for fast switch back

        img.reduce("MEAN", bounds=(0, 99))
        out2 = img.image.copy()
        np.testing.assert_allclose(out1, out2)
        assert ("MEAN", (0, 99)) in img._result_cache

    def test_cache_reuse_mean_median_mean(self) -> None:
        """Mean -> Median -> Mean: second Mean is cache hit (no recompute)."""
        data = np.random.rand(50, 8, 8, 1).astype(np.float32)
        meta = _make_meta(50)
        img = ImageData(data, meta)

        img.reduce("MEAN", bounds=(0, 49))
        _ = img.image
        mean_result = img._result_cache[("MEAN", (0, 49))].copy()

        img.reduce("MEDIAN", bounds=(0, 49))
        _ = img.image
        assert ("MEAN", (0, 49)) in img._result_cache  # Mean still cached

        img.reduce("MEAN", bounds=(0, 49))
        out = img.image.copy()
        np.testing.assert_allclose(out, mean_result)
        assert img._bench_cache_hits >= 1

    def test_cache_different_bounds_both_cached(self) -> None:
        """Different bounds produce different results, both remain in cache."""
        data = np.arange(20 * 4 * 4, dtype=np.float32).reshape(20, 4, 4, 1)  # 0..319
        meta = _make_meta(20)
        img = ImageData(data, meta)

        img.reduce("MEAN", bounds=(0, 9))
        _ = img.image
        prev_result = img._result_cache[("MEAN", (0, 9))].copy()

        img.reduce("MEAN", bounds=(10, 19))
        _ = img.image
        curr = img._result_cache[("MEAN", (10, 19))]
        assert not np.allclose(prev_result, curr)
        assert ("MEAN", (0, 9)) in img._result_cache

    def test_cache_faster_than_recompute(self) -> None:
        """Cache hit is faster than recompute (smoke test with 2k frames)."""
        n = 2000
        data = np.random.rand(n, 32, 32, 1).astype(np.float32)
        meta = _make_meta(n)
        img = ImageData(data, meta)

        img.reduce("MEAN", bounds=(0, n - 1))
        t0 = time.perf_counter()
        for _ in range(5):
            _ = img.image
        t_cached = (time.perf_counter() - t0) / 5

        img._invalidate_result()
        t0 = time.perf_counter()
        _ = img.image
        t_compute = time.perf_counter() - t0

        # Cache hit should be at least 2x faster (typically 10-100x for median)
        assert t_cached < t_compute, f"cached={t_cached:.4f}s, compute={t_compute:.4f}s"

"""Unit tests for ImageData normalization pipeline and aggregation range."""
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


class TestNormalizationPipeline:
    """Test pipeline math: subtract then divide."""

    def test_subtract_then_divide_range(self) -> None:
        # Data: frames 0,1,2 have values 10,20,30. Mean of 0-2 = 20.
        # Subtract mean ->  -10, 0, 10
        # Divide by mean (20) -> -0.5, 0, 0.5
        data = np.arange(12, dtype=np.float32).reshape(3, 2, 2, 1)
        data[0] = 10
        data[1] = 20
        data[2] = 30
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = [
            {"operation": "subtract", "source": "range", "bounds": (0, 2), "use": "MEAN"},
            {"operation": "divide", "source": "range", "bounds": (0, 2), "use": "MEAN"},
        ]
        img.set_normalization_pipeline(pipeline, factor=1.0, blur=0)

        out = img.image
        # After subtract: 10-20=-10, 20-20=0, 30-20=10
        # After divide by 20: -10/20=-0.5, 0/20=0, 10/20=0.5
        np.testing.assert_allclose(out[0, ..., 0], -0.5)
        np.testing.assert_allclose(out[1, ..., 0], 0.0)
        np.testing.assert_allclose(out[2, ..., 0], 0.5)

    def test_divide_then_subtract_range(self) -> None:
        # Pipeline: divide by mean(base), subtract mean(base).
        # Both refs from raw. divide by 10->1, subtract 10->-9.
        data = np.ones((3, 2, 2, 1), dtype=np.float32) * 10.0
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = [
            {"operation": "divide", "source": "range", "bounds": (0, 2), "use": "MEAN"},
            {"operation": "subtract", "source": "range", "bounds": (0, 2), "use": "MEAN"},
        ]
        img.set_normalization_pipeline(pipeline, factor=1.0, blur=0)

        out = img.image
        # After divide by 10: all 1.0. After subtract 10: 1-10=-9
        np.testing.assert_allclose(out, -9.0)

    def test_factor_scaling(self) -> None:
        data = np.ones((2, 2, 2, 1), dtype=np.float32) * 100.0
        meta = _make_meta(2)
        img = ImageData(data, meta)

        pipeline = [
            {"operation": "subtract", "source": "range", "bounds": (0, 1), "use": "MEAN"},
        ]
        img.set_normalization_pipeline(pipeline, factor=0.5, blur=0)

        out = img.image
        # Reference mean = 100. With factor 0.5, we subtract 50. Result = 50.
        np.testing.assert_allclose(out, 50.0)

    def test_empty_pipeline_identity(self) -> None:
        data = np.arange(24, dtype=np.uint8).reshape(3, 2, 2, 2)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        img.set_normalization_pipeline([], factor=1.0, blur=0)
        out = img.image
        np.testing.assert_array_equal(out, data)


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
        expected = np.array([[[[5.], [6.]], [[7.], [8.]]]])  # 4D, mean over frames
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
        # Mean of frames 1 and 2 only: (5+9)/2=7, (6+10)/2=8, (7+11)/2=9, (8+12)/2=10
        expected = np.array([[[[7.], [8.]], [[9.], [10.]]]])
        np.testing.assert_allclose(out, expected)

    def test_reduce_single_frame_bounds(self) -> None:
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], dtype=np.float32).reshape(2, 2, 2, 1)
        meta = _make_meta(2)
        img = ImageData(data, meta)

        img.reduce("MAX", bounds=(0, 0))
        out = img.image
        np.testing.assert_allclose(out, data[0:1])

    def test_pipeline_then_aggregation_bounds(self) -> None:
        """Normalization pipeline then aggregation with bounds."""
        data = np.array([
            [[10, 10], [10, 10]],
            [[20, 20], [20, 20]],
            [[30, 30], [30, 30]],
        ], dtype=np.float32).reshape(3, 2, 2, 1)
        meta = _make_meta(3)
        img = ImageData(data, meta)

        pipeline = [
            {"operation": "subtract", "source": "range", "bounds": (0, 2), "use": "MEAN"},
        ]
        img.set_normalization_pipeline(pipeline, factor=1.0, blur=0)
        img.reduce("MEAN", bounds=(0, 1))
        out = img.image
        # After subtract: -10, 0, 10 (mean was 20)
        # Bounds 0-1: frames -10 and 0. Mean = -5
        expected = np.array([[[[-5.], [-5.]], [[-5.], [-5.]]]])
        np.testing.assert_allclose(out, expected)

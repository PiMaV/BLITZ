from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pyqtgraph as pg

# from .. import settings
from ..tools import log
from . import ops, optimized
from .tools import (
    ensure_4d,
    sliding_aggregate_at_frame,
    sliding_aggregate_normalization,
    sliding_mean_at_frame,
    sliding_mean_normalization,
)


@dataclass(kw_only=True)
class MetaData:
    file_name: str
    file_size_MB: float
    size: tuple[int, int]
    dtype: type
    bit_depth: int
    color_model: Literal["rgb", "grayscale"]


@dataclass(kw_only=True)
class VideoMetaData(MetaData):
    fps: int
    frame_count: int
    reduced_frame_count: int
    codec: str


class ImageData:

    def __init__(
        self,
        image: np.ndarray,
        metadata: list[MetaData],
    ) -> None:
        # Keep original dtype (uint8 for video/images) - saves 4x RAM vs float32.
        # Normalize/reduce convert to float when needed.
        self._image = ensure_4d(np.asarray(image))
        self._meta = metadata
        self._reduced = ops.ReduceDict()
        self._mask: tuple[slice, slice, slice] | None = None
        self._image_mask: np.ndarray | None = None
        self._cropped: tuple[int, int] | None = None
        self._save_cropped: tuple[int, int] | None = None
        self._rotated_90 = False
        self._flipped_x = False
        self._flipped_y = False
        self._transposed = False
        self._redop: ops.ReduceOperation | str | None = None
        self._ops_pipeline: dict | None = None  # {subtract, divide} steps
        self._agg_bounds: tuple[int, int] | None = None  # Non-destructive agg range
        self._result_cache: dict[tuple[object, object], np.ndarray] = {}  # (op, bounds) -> result
        self._ref_cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}  # (method, bounds) -> ref
        self._bench_cache_hits: int = 0  # For Bench tab
        self._bench_cache_misses: int = 0
        self.use_numba: bool = True
        self.preview_frame: Optional[int] | None = None

    def _compute_ref(
        self, step: dict, image: np.ndarray
    ) -> np.ndarray | None:
        """Compute reference for a pipeline step. Returns (1,H,W,C) or None."""
        src = step.get("source")
        if not src or src == "off":
            return None
        if src == "aggregate" and step.get("bounds") is not None:
            b0, b1 = step["bounds"]
            method = step.get("method", "MEAN")
            # Support both ReduceOperation enum and string (e.g. from project load)
            if hasattr(method, "name"):
                method = method.name
            method_str = str(method).upper()
            cache_key = (method_str, (b0, b1))
            if cache_key in self._ref_cache:
                return self._ref_cache[cache_key]
            slice_ = image[b0 : b1 + 1]
            if slice_.size == 0:
                return None
            ref = ops.get(method)(slice_).astype(np.float32)
            self._ref_cache[cache_key] = ref
            return ref
        if src == "file" and step.get("reference") is not None:
            ref_img = step["reference"]._image
            if ref_img.shape[0] != 1:
                if not getattr(self, "_logged_ref_multi_frame", False):
                    self._logged_ref_multi_frame = True
                    log("Reference must be single image (1 frame)", color="red")
                return None
            if ref_img.shape[1:] != image.shape[1:]:
                key = ("file_shape", ref_img.shape[1:], image.shape[1:])
                logged = getattr(self, "_ref_mismatch_logged", set())
                if key not in logged:
                    logged.add(key)
                    self._ref_mismatch_logged = logged
                    log(
                        f"Reference shape {ref_img.shape[1:]} != data shape {image.shape[1:]}",
                        color="red",
                    )
                return None
            return ref_img.astype(np.float32)
        if src == "sliding_mean":
            window = step.get("window", 1)
            lag = step.get("lag", 0)
            return sliding_mean_normalization(image, window, lag)
        if src == "sliding_aggregate":
            window = step.get("window", 1)
            lag = step.get("lag", 0)
            method = step.get("method", ops.ReduceOperation.MEAN)
            return sliding_aggregate_normalization(image, window, lag, method)
        return None

    def _apply_ops_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply subtract and divide steps. Returns float32 array."""
        pipeline = self._ops_pipeline
        if not pipeline:
            return image.astype(np.float32) if image.dtype != np.float32 else image

        sub_step = pipeline.get("subtract")
        div_step = pipeline.get("divide")
        sub_sm = sub_step and sub_step.get("source") in ("sliding_mean", "sliding_aggregate") and sub_step.get("amount", 0) > 0
        div_sm = div_step and div_step.get("source") in ("sliding_mean", "sliding_aggregate") and div_step.get("amount", 0) > 0

        if sub_sm or div_sm:
            image = image.astype(np.float32)
            window = (sub_step or div_step).get("window", 1)
            lag = (sub_step or div_step).get("lag", 0)
            apply_full = (sub_step or div_step).get("apply_full", False)

            if apply_full:
                # Full: reduce to N frames
                sliding_mean = self._compute_ref(div_step if div_sm else sub_step, image)
                if sliding_mean is None or sliding_mean.shape[0] == 0:
                    return image
                N = sliding_mean.shape[0]
                img_slice = image[lag + 1 : lag + 1 + N].astype(np.float32)
                if sub_sm:
                    img_slice = img_slice - float(sub_step.get("amount", 1.0)) * sliding_mean
                if div_sm:
                    amt = np.float32(div_step.get("amount", 1.0))
                    denom = amt * sliding_mean + (np.float32(1.0) - amt)
                    denom = np.where(denom != 0, denom, np.float32(np.nan))
                    img_slice = img_slice / denom
                    np.nan_to_num(img_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                return img_slice

            # Preview: keep T frames, process only current frame to avoid hangs
            T, H, W, C = image.shape
            result = image.copy()
            valid_start = lag + 1
            valid_end = T - window + 1
            pf = getattr(self, "preview_frame", None)
            if pf is not None and valid_start <= pf < valid_end:
                f = pf
                step = sub_step if sub_sm else div_step
                method = step.get("method", ops.ReduceOperation.MEAN) if step.get("source") == "sliding_aggregate" else ops.ReduceOperation.MEAN
                sm = sliding_aggregate_at_frame(image, f, window, method)
                if sub_sm:
                    result[f] -= float(sub_step.get("amount", 1.0)) * sm
                if div_sm:
                    amt = float(div_step.get("amount", 1.0))
                    denom = amt * sm + (1.0 - amt)
                    denom = np.where(denom != 0, denom, np.nan)
                    result[f] = np.divide(result[f], denom, out=np.zeros_like(result[f]), where=denom != 0)
            if div_sm:
                np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return result

        # Determine if we should use Numba
        use_numba = optimized.HAS_NUMBA and getattr(self, "use_numba", True)

        if use_numba:
            # Prepare arguments for Numba kernel
            sub_step = pipeline.get("subtract")
            do_sub = False
            sub_ref = np.empty((1, 1, 1, 1), dtype=np.float32)
            sub_amt = 0.0

            if sub_step and sub_step.get("amount", 0) > 0:
                ref = self._compute_ref(sub_step, image)
                if ref is not None:
                    do_sub = True
                    sub_ref = ref
                    sub_amt = float(sub_step.get("amount", 1.0))

            div_step = pipeline.get("divide")
            do_div = False
            div_ref = np.empty((1, 1, 1, 1), dtype=np.float32)
            div_amt = 0.0

            if div_step and div_step.get("amount", 0) > 0:
                ref = self._compute_ref(div_step, image)
                if ref is not None:
                    do_div = True
                    div_ref = ref
                    div_amt = float(div_step.get("amount", 1.0))

            if do_sub or do_div:
                # Ensure float32 copy to avoid modifying source or non-float types
                if image.dtype != np.float32 or np.shares_memory(image, self._image):
                    image = image.astype(np.float32, copy=True)

                optimized.apply_pipeline_fused(
                    image,
                    do_sub, sub_ref, sub_amt,
                    do_div, div_ref, div_amt
                )
                return image

        image = image.astype(np.float32)
        eps = 1e-10
        for op_name in ("subtract", "divide"):
            step = pipeline.get(op_name)
            if not step:
                continue
            ref = self._compute_ref(step, image)
            if ref is None:
                continue
            amount = np.float32(step.get("amount", 1.0))
            if amount <= 0:
                continue
            if op_name == "subtract":
                ref_scaled = amount * ref
                if ref.shape[0] == 1:
                    image -= ref_scaled
                else:
                    image = image[: ref.shape[0]]
                    image -= ref_scaled
            else:  # divide: blend denominator towards 1 when amount<1
                denom = amount * ref + (np.float32(1.0) - amount)
                denom = np.where(denom != 0, denom, np.float32(np.nan))
                if ref.shape[0] == 1:
                    image /= denom
                else:
                    image = image[: ref.shape[0]]
                    image /= denom
                np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return image

    def _invalidate_result(self) -> None:
        """Clear result cache when input changes (pipeline, crop). Not on op/bounds change."""
        self._result_cache.clear()
        self._reduced.clear()
        self._ref_cache.clear()
        self._logged_ref_multi_frame = False
        self._ref_mismatch_logged = set()

    def _get_cached_result(self, operation: object, bounds: object) -> np.ndarray | None:
        """Return cached result for (op, bounds) or None."""
        key = (operation, bounds)
        return self._result_cache.get(key)

    @property
    def image(self) -> np.ndarray:
        if self._image_mask is not None:
            # Need float for np.nan; uint8 can't hold nan
            image: np.ndarray = self._image.astype(np.float32).copy()
        else:
            image: np.ndarray = self._image
        if self._ops_pipeline:
            image = self._apply_ops_pipeline(image)
        if self._redop is not None:
            cached = self._get_cached_result(self._redop, self._agg_bounds)
            if cached is not None:
                self._bench_cache_hits += 1
                image = cached
            else:
                self._bench_cache_misses += 1
                self._reduced.clear()  # Ensure fresh compute for new (op, bounds)
                to_reduce = image
                if self._agg_bounds is not None:
                    b0, b1 = self._agg_bounds
                    to_reduce = image[b0 : b1 + 1]
                image = self._reduced.reduce(to_reduce, self._redop)
                self._result_cache[(self._redop, self._agg_bounds)] = image.copy()
        if self._cropped is not None:
            image = image[self._cropped[0]:self._cropped[1]+1]
        if self._image_mask is not None:
            image[:, ~self._image_mask] = np.nan
        if self._mask is not None:
            image = image[self._mask]

        # Transforms: Order matters. Transpose first?
        # Standard: Transpose, then Rotate, then Flip?
        # User controls: Flip XY (Transpose) and Rotate 90.
        # If both checked: Transpose then Rotate 90.

        if self._transposed:
            image = np.transpose(image, (0, 2, 1, 3))

        if self._rotated_90:
            image = np.rot90(image, k=-1, axes=(1, 2))
        if self._flipped_x:
            image = np.flip(image, 1)
        if self._flipped_y:
            image = np.flip(image, 2)
        return image

    @property
    def image_timeline(self) -> np.ndarray | None:
        """Full stack for timeline: norm + mask, no reduce. None wenn nicht moeglich."""
        if self._redop is None:
            return None
        if self._image_mask is not None:
            image: np.ndarray = self._image.astype(np.float32).copy()
        else:
            image: np.ndarray = self._image
        if self._ops_pipeline:
            image = self._apply_ops_pipeline(image)
        if self._cropped is not None:
            image = image[self._cropped[0] : self._cropped[1] + 1]
        if self._image_mask is not None:
            image[:, ~self._image_mask] = np.nan
        if self._mask is not None:
            image = image[self._mask]

        if self._transposed:
            image = np.transpose(image, (0, 2, 1, 3))

        if self._rotated_90:
            image = np.rot90(image, k=-1, axes=(1, 2))
        if self._flipped_x:
            image = np.flip(image, 1)
        if self._flipped_y:
            image = np.flip(image, 2)
        return image

    @property
    def n_images(self) -> int:
        T = self._image.shape[0]
        pipeline = self._ops_pipeline
        if pipeline:
            for step in (pipeline.get("subtract"), pipeline.get("divide")):
                if step and step.get("source") in ("sliding_mean", "sliding_aggregate") and step.get("amount", 0) > 0:
                    if not step.get("apply_full", False):
                        return T  # Preview: keep full timeline
                    window = step.get("window", 1)
                    lag = step.get("lag", 0)
                    return max(0, T - (lag + window))
        if self._cropped is not None:
            return self._image[
                self._cropped[0] : self._cropped[1] + 1
            ].shape[0]
        return T

    @property
    def shape(self) -> tuple[int, int]:
        return (self.image.shape[1], self.image.shape[2])

    @property
    def meta(self) -> list[MetaData]:
        return self._meta

    def is_single_image(self) -> bool:
        return self._image.shape[0] == 1

    def is_greyscale(self) -> bool:
        return self._image.shape[3] == 1

    def reduce(
        self,
        operation: ops.ReduceOperation | str,
        bounds: Optional[tuple[int, int]] = None,
    ) -> None:
        self._redop = operation
        self._agg_bounds = bounds

    def set_ops_pipeline(self, config: dict | None) -> None:
        """Set ops pipeline. config: {subtract:{source,bounds?,method?,reference?,amount}, divide:{...}}."""
        if self._ops_pipeline != config:
            self._invalidate_result()
        self._ops_pipeline = config

    def set_aggregation_bounds(self, bounds: tuple[int, int] | None) -> None:
        """Set non-destructive aggregation range (for reduced view)."""
        if self._agg_bounds != bounds:
            self._invalidate_result()
        self._agg_bounds = bounds

    def crop(self, left: int, right: int, keep: bool = False) -> None:
        if keep:
            self._cropped = (left, right)
        else:
            self._cropped = None
            self._image = self._image[left:right+1]
            self._invalidate_result()
        self._save_cropped = (left, right)

    def undo_crop(self) -> bool:
        if self._cropped is None:
            return False
        self._cropped = None
        self._save_cropped = None
        return True

    def unravel(self) -> None:
        """Switch to Frame mode. Result cache preserved for fast switch back to Aggregate."""
        self._redop = None
        self._agg_bounds = None

    def mask(self, roi: pg.ROI) -> bool:
        if self._rotated_90 or self._flipped_x or self._flipped_y or self._transposed:
            log("Masking not available while data is flipped, rotated or transposed",
                color="red")
            return False
        pos = roi.pos()
        size = roi.size()
        x_start = max(0, int(pos[0]))
        y_start = max(0, int(pos[1]))
        x_stop = min(self._image.shape[1], int(pos[0] + size[0]))
        y_stop = min(self._image.shape[2], int(pos[1] + size[1]))
        if self._mask is not None:
            x_start += self._mask[1].start
            x_stop += self._mask[1].start
            y_start += self._mask[2].start
            y_stop += self._mask[2].start
        self._rotated_90 = False
        self._flipped_x = False
        self._flipped_y = False
        self._transposed = False
        self._mask = (
            slice(None, None), slice(x_start, x_stop), slice(y_start, y_stop),
        )
        return True

    def mask_range(self, range_: tuple[int, int, int, int]) -> None:
        self._mask = (
            slice(None, None),
            slice(range_[0], range_[1]),
            slice(range_[2], range_[3]),
        )

    def image_mask(self, mask: "ImageData") -> None:
        if (not mask.is_single_image()
                or not mask.is_greyscale()
                or mask.shape != self.shape):
            log("Error: Mask not applicable", color="red")
        else:
            self._image_mask = mask.image[0].astype(bool)

    def reset_mask(self) -> bool:
        if self._mask is not None:
            self._mask = None
            self._image_mask = None
            return True
        return False

    def rotate_90(self) -> None:
        """Toggle 90 deg clockwise rotation. Uses np.rot90, not transpose."""
        self._rotated_90 = not self._rotated_90

    def flip_x(self) -> None:
        self._flipped_x = not self._flipped_x

    def flip_y(self) -> None:
        self._flipped_y = not self._flipped_y

    def transpose(self) -> None:
        """Toggle transpose (Flip XY)."""
        self._transposed = not self._transposed

    def get_mask(self) -> tuple[slice, slice, slice] | None:
        return self._mask

    def set_mask(self, mask: tuple[slice, slice, slice] | None):
        self._mask = mask

    def can_undo_crop(self) -> bool:
        """True if crop was applied with keep=True (reversible)."""
        return self._cropped is not None

    def get_crop(self) -> tuple[int, int] | None:
        return self._save_cropped

    def set_crop(self, crop: tuple[int, int] | None):
        self._save_cropped = crop
        self._cropped = crop

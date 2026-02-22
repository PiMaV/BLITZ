from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pyqtgraph as pg

# from .. import settings
from ..tools import log
from . import ops, optimized
from .tools import ensure_4d


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
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
        self._redop: ops.ReduceOperation | str | None = None
        self._ops_pipeline: dict | None = None  # {subtract, divide} steps
        self._agg_bounds: tuple[int, int] | None = None  # Non-destructive agg range
        self._result_cache: dict[tuple[object, object], np.ndarray] = {}  # (op, bounds) -> result
        self._bench_cache_hits: int = 0  # For Bench tab
        self._bench_cache_misses: int = 0
        self.use_numba: bool = True

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
            ref = ops.get(method)(image[b0 : b1 + 1]).astype(np.float32)
            return ref
        if src == "file" and step.get("reference") is not None:
            ref_img = step["reference"]._image
            if ref_img.shape[0] != 1 or ref_img.shape[1:] != image.shape[1:]:
                return None
            return ref_img.astype(np.float32)
        return None

    def _apply_ops_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply subtract and divide steps. Returns float32 array."""
        pipeline = self._ops_pipeline
        if not pipeline:
            return image.astype(np.float32) if image.dtype != np.float32 else image

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
        if self._transposed:
            image = np.swapaxes(image, 1, 2)
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
            image = np.swapaxes(image, 1, 2)
        if self._flipped_x:
            image = np.flip(image, 1)
        if self._flipped_y:
            image = np.flip(image, 2)
        return image

    @property
    def n_images(self) -> int:
        if self._cropped is not None:
            return self._image[self._cropped[0]:self._cropped[1]+1].shape[0]
        return self._image.shape[0]

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
        return True

    def unravel(self) -> None:
        """Switch to Frame mode. Result cache preserved for fast switch back to Aggregate."""
        self._redop = None
        self._agg_bounds = None

    def mask(self, roi: pg.ROI) -> bool:
        if self._transposed or self._flipped_x or self._flipped_y:
            log("Masking not available while data is flipped or transposed",
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
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
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

    def transpose(self) -> None:
        self._transposed = not self._transposed

    def flip_x(self) -> None:
        self._flipped_x = not self._flipped_x

    def flip_y(self) -> None:
        self._flipped_y = not self._flipped_y

    def get_mask(self) -> tuple[slice, slice, slice] | None:
        return self._mask

    def set_mask(self, mask: tuple[slice, slice, slice] | None):
        self._mask = mask

    def get_crop(self) -> tuple[int, int] | None:
        return self._save_cropped

    def set_crop(self, crop: tuple[int, int] | None):
        self._save_cropped = crop
        self._cropped = crop

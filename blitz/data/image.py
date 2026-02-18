from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pyqtgraph as pg

# from .. import settings
from ..tools import log
from . import ops
from .tools import ensure_4d, sliding_mean_normalization


def _apply_blur_2d(img: np.ndarray, gaussian_blur: int) -> np.ndarray:
    """Apply Gaussian blur to a single-frame 4D image (T, H, W, C)."""
    if gaussian_blur <= 0 or img.size == 0:
        return img
    blurred = pg.gaussianFilter(img[0, ..., 0], (gaussian_blur, gaussian_blur))
    return blurred[np.newaxis, ..., np.newaxis].astype(img.dtype)


def _apply_blur_4d(img: np.ndarray, gaussian_blur: int) -> np.ndarray:
    """Apply Gaussian blur per frame to a 4D image (T, H, W, C)."""
    if gaussian_blur <= 0 or img.size == 0:
        return img
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        b = pg.gaussianFilter(img[i, ..., 0], (gaussian_blur, gaussian_blur))
        out[i] = b[..., np.newaxis]
    return out


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


@dataclass(kw_only=True)
class DicomMetaData(MetaData):
    sequence_number: int


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
        self._norm_pipeline: list[dict] = []
        self._norm_factor: float = 1.0
        self._norm_blur: int = 0
        self._agg_bounds: tuple[int, int] | None = None  # Non-destructive agg range

    def _apply_normalization_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply the normalization pipeline to image. Returns float array."""
        base = image  # References (range, file) computed from original data
        for step in self._norm_pipeline:
            op_ = step["operation"]
            factor = step.get("factor", self._norm_factor)
            blur = step.get("blur", self._norm_blur)
            ref = self._compute_reference(step, base)
            if ref is None:
                continue
            if blur > 0:
                if ref.shape[0] == 1:
                    ref = _apply_blur_2d(ref.astype(np.float64), blur)
                else:
                    ref = _apply_blur_4d(ref.astype(np.float64), blur)
            ref = factor * ref.astype(np.float64)
            if op_ == "subtract":
                if ref.shape[0] == 1:
                    image = image.astype(np.float64) - ref
                else:
                    image = image[:ref.shape[0]].astype(np.float64) - ref
            else:  # divide
                if ref.shape[0] == 1:
                    image = image.astype(np.float64) / np.where(
                        ref != 0, ref, np.nan
                    )
                else:
                    sl = image[:ref.shape[0]].astype(np.float64)
                    image = sl / np.where(ref != 0, ref, np.nan)
        return image

    def _compute_reference(
        self, step: dict, image: np.ndarray
    ) -> np.ndarray | None:
        """Compute reference for a pipeline step."""
        ref = None
        if step.get("source") == "range" and step.get("bounds") is not None:
            b0, b1 = step["bounds"]
            use = step.get("use", "MEAN")
            r = ops.get(use)(image[b0 : b1 + 1]).astype(np.float64)
            ref = r
        elif step.get("source") == "file" and step.get("reference") is not None:
            ref_img = step["reference"]._image
            if ref_img.shape[0] != 1 or ref_img.shape[1:] != image.shape[1:]:
                return None
            ref = ref_img.astype(np.float64)
        elif step.get("source") == "sliding" and step.get("window_lag") is not None:
            w, lag = step["window_lag"]
            ref = sliding_mean_normalization(image.astype(np.float32), w, lag)
        return ref

    @property
    def image(self) -> np.ndarray:
        if self._image_mask is not None:
            # Need float for np.nan; uint8 can't hold nan
            image: np.ndarray = self._image.astype(np.float32).copy()
        else:
            image: np.ndarray = self._image
        if self._norm_pipeline:
            image = self._apply_normalization_pipeline(image)
        if self._redop is not None:
            to_reduce = image
            if self._agg_bounds is not None:
                b0, b1 = self._agg_bounds
                to_reduce = image[b0 : b1 + 1]
            image = self._reduced.reduce(to_reduce, self._redop)
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
        old_bounds = self._agg_bounds
        self._agg_bounds = bounds
        if old_bounds != bounds:
            self._reduced.clear()

    def set_normalization_pipeline(
        self,
        pipeline: list[dict],
        factor: float = 1.0,
        blur: int = 0,
    ) -> None:
        """Set the normalization pipeline. Each step: operation, source, params."""
        self._norm_pipeline = pipeline
        self._norm_factor = factor
        self._norm_blur = blur

    def set_aggregation_bounds(self, bounds: tuple[int, int] | None) -> None:
        """Set non-destructive aggregation range (for reduced view)."""
        if self._agg_bounds != bounds:
            self._reduced.clear()
        self._agg_bounds = bounds

    def crop(self, left: int, right: int, keep: bool = False) -> None:
        if keep:
            self._cropped = (left, right)
        else:
            self._cropped = None
            self._image = self._image[left:right+1]
        self._save_cropped = (left, right)

    def undo_crop(self) -> bool:
        if self._cropped is None:
            return False
        self._cropped = None
        return True

    def normalize(
        self,
        operation: Literal["subtract", "divide"],
        use: ops.ReduceOperation | str,
        beta: float = 1.0,
        gaussian_blur: int = 0,
        bounds: Optional[tuple[int, int]] = None,
        reference: Optional["ImageData"] = None,
        window_lag: Optional[tuple[int, int]] = None,
        force_calculation: bool = False,
    ) -> bool:
        """Legacy API: builds a single-step pipeline and applies it."""
        if self._redop is not None:
            log("Normalization not possible on reduced data")
            return False
        source = None
        step_params: dict = {"operation": operation, "factor": beta, "blur": gaussian_blur}
        if bounds is not None:
            source = "range"
            step_params["source"] = source
            step_params["bounds"] = bounds
            step_params["use"] = use
        elif reference is not None:
            if not reference.is_single_image() or reference._image.shape[1:] != self._image.shape[1:]:
                log("Error: Background image has incompatible shape")
                return False
            source = "file"
            step_params["source"] = source
            step_params["reference"] = reference
        elif window_lag is not None:
            source = "sliding"
            step_params["source"] = source
            step_params["window_lag"] = window_lag
        if source is None:
            return False
        had_pipeline = bool(self._norm_pipeline)
        if had_pipeline and not force_calculation:
            self.set_normalization_pipeline([], factor=beta, blur=gaussian_blur)
            return False
        self.set_normalization_pipeline([step_params], factor=beta, blur=gaussian_blur)
        return True

    def unravel(self) -> None:
        self._redop = None
        self._agg_bounds = None
        self._reduced.clear()

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

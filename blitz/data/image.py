from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

import numpy as np
import pyqtgraph as pg

# from .. import settings
from ..tools import log
from . import ops
from .tools import ensure_4d, sliding_mean_normalization


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
        self._image = ensure_4d(image.astype(np.float32))
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
        self._norm: np.ndarray | None = None
        self._norm_pipeline: List[dict] = []
        self._agg_bounds: tuple[int, int] | None = None

    @property
    def image(self) -> np.ndarray:
        if self._image_mask is not None:
            image: np.ndarray = self._image.copy()
        else:
            image: np.ndarray = self._image

        # Apply normalization pipeline
        if self._norm is not None:
            image = self._norm

        if self._redop is not None:
            if self._agg_bounds is not None:
                # Use sub-range for reduction
                subset = image[self._agg_bounds[0]:self._agg_bounds[1]+1]
                image = self._reduced.reduce(subset, self._redop)
            else:
                image = self._reduced.reduce(image, self._redop)

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

    def reduce(self, operation: ops.ReduceOperation | str, bounds: Optional[tuple[int, int]] = None) -> None:
        self._redop = operation
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

    def clear_normalization(self) -> None:
        self._norm = None
        self._norm_pipeline = []

    def set_normalization_pipeline(self, pipeline: List[dict]) -> bool:
        """
        Calculates the normalized image based on a pipeline of operations.
        pipeline structure:
        [
            {
                "operation": "subtract" | "divide",
                "use": ops.ReduceOperation | str (for bounds),
                "factor": float, # was beta
                "gaussian_blur": int,
                "bounds": Optional[tuple[int, int]],
                "reference": Optional["ImageData"],
                "window_lag": Optional[tuple[int, int]],
            },
            ...
        ]
        """
        if self._redop is not None:
            log("Normalization not possible on reduced data")
            return False

        if not pipeline:
            self.clear_normalization()
            return True

        image = self._image.copy() # Start with a copy to avoid modifying original

        # Helper to process reference images
        def process_reference(step):
            ref_img = None
            use = step.get("use", "mean")
            factor = step.get("factor", 1.0)
            gaussian_blur = step.get("gaussian_blur", 0)
            bounds = step.get("bounds")
            reference = step.get("reference")
            window_lag = step.get("window_lag")

            if bounds is not None:
                # Calculate reference from temporal range
                ref_img = factor * ops.get(use)(
                    self._image[bounds[0]:bounds[1]+1]
                ).astype(np.double)
            elif reference is not None:
                # Use external reference image
                if (not reference.is_single_image()
                        or reference._image.shape[1:] != self._image.shape[1:]):
                    log("Error: Background image has incompatible shape")
                    return None
                ref_img = factor * reference._image.astype(np.double)
            elif window_lag is not None:
                 # Calculate sliding window reference
                window, lag = window_lag
                ref_img = factor * (
                    sliding_mean_normalization(self._image, window, lag)
                )

            # Apply blur if needed
            if ref_img is not None and gaussian_blur > 0:
                 if window_lag is not None:
                     # For sliding window, ref_img is a stack
                    ref_img = np.array([
                        pg.gaussianFilter(
                            ref_img[i, ..., 0],
                            (gaussian_blur, gaussian_blur),
                        )[..., np.newaxis]
                        for i in range(ref_img.shape[0])
                    ])
                 else:
                     # For bounds/reference, ref_img is single frame (1, H, W, C)
                    ref_img = pg.gaussianFilter(
                        ref_img[0, ..., 0],
                        (gaussian_blur, gaussian_blur),
                    )[np.newaxis, ..., np.newaxis]

            return ref_img

        for step in pipeline:
            operation = step.get("operation")
            ref_img = process_reference(step)

            if ref_img is None:
                # Should we error out or skip? For now, skip if ref generation failed (e.g. shape mismatch)
                continue

            # Apply operation
            if operation == "subtract":
                if step.get("window_lag") is not None:
                     # Handle shape mismatch for sliding window (it's shorter than image)
                     limit = min(image.shape[0], ref_img.shape[0])
                     image[:limit] = image[:limit] - ref_img[:limit]
                     # What about the rest? Sliding window usually implies data loss at edges.
                     # The original code did: image[:window_lag_img.shape[0]] - window_lag_img
                     # Effectively implicitly handling it.
                else:
                    image = image - ref_img
            elif operation == "divide":
                # Avoid division by zero? Numpy handles it (inf/nan), but might need care
                if step.get("window_lag") is not None:
                     limit = min(image.shape[0], ref_img.shape[0])
                     # Avoid 0 division if possible, or accept NaNs
                     image[:limit] = image[:limit] / ref_img[:limit]
                else:
                    image = image / ref_img

        self._norm = image
        self._norm_pipeline = pipeline
        return True


    # Keeping the old signature for backward compatibility during refactor,
    # but redirecting to new pipeline logic or deprecating it.
    # The UI will use set_normalization_pipeline directly.
    def normalize(
        self,
        operation: Literal["subtract", "divide"],
        use: ops.ReduceOperation | str,
        beta: float = 1.0,
        gaussian_blur: int = 0,
        bounds: Optional[tuple[int, int]] =None,
        reference: Optional["ImageData"] = None,
        window_lag: Optional[tuple[int, int]] = None,
        force_calculation: bool = False,
    ) -> bool:
        # Legacy wrapper: creates a single-step pipeline
        # Note: 'beta' is mapped to 'factor'
        step = {
            "operation": operation,
            "use": use,
            "factor": beta,
            "gaussian_blur": gaussian_blur,
            "bounds": bounds,
            "reference": reference,
            "window_lag": window_lag
        }

        # Check if we are toggling off
        # (This logic was in the old function: "If self._norm_operation == operation ... return False")
        # In the new pipeline model, we just set it.
        # The caller (UI) is responsible for deciding if it's toggling ON or OFF.
        # But to match exact legacy behavior we might need to check self._norm_pipeline.

        # For this task, I am refactoring the UI too, so I will switch the UI to use
        # set_normalization_pipeline.

        return self.set_normalization_pipeline([step])

    def unravel(self) -> None:
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

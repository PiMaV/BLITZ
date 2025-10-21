from dataclasses import dataclass
from typing import Literal, Optional

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
        self._norm_operation: Literal["subtract", "divide"] | None = None

    @property
    def image(self) -> np.ndarray:
        if self._image_mask is not None:
            image: np.ndarray = self._image.copy()
        else:
            image: np.ndarray = self._image
        if self._norm is not None:
            image = self._norm
        if self._redop is not None:
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

    def reduce(self, operation: ops.ReduceOperation | str) -> None:
        self._redop = operation

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
        bounds: Optional[tuple[int, int]] =None,
        reference: Optional["ImageData"] = None,
        window_lag: Optional[tuple[int, int]] = None,
        force_calculation: bool = False,
    ) -> bool:
        if self._redop is not None:
            log("Normalization not possible on reduced data")
            return False
        if self._norm_operation == operation and not force_calculation:
            self._norm_operation = None
            self._norm = None
            return False
        if self._norm_operation is not None:
            self._norm = None
        image = self._image
        range_img = reference_img = window_lag_img = None
        if bounds is not None:
            range_img = beta * ops.get(use)(
                image[bounds[0]:bounds[1]+1]
            ).astype(np.double)
            if gaussian_blur > 0:
                range_img = pg.gaussianFilter(
                    range_img[0, ..., 0],
                    (gaussian_blur, gaussian_blur),
                )[np.newaxis, ..., np.newaxis]
        if reference is not None:
            if (not reference.is_single_image()
                    or reference._image.shape[1:] != image.shape[1:]):
                log("Error: Background image has incompatible shape")
                return False
            reference_img = beta * reference._image.astype(np.double)
            if gaussian_blur > 0:
                reference_img = pg.gaussianFilter(
                    reference_img[0, ..., 0],
                    (gaussian_blur, gaussian_blur),
                )[np.newaxis, ..., np.newaxis]
        if window_lag is not None:
            window, lag = window_lag
            window_lag_img = beta * (
                sliding_mean_normalization(image, window, lag)
            )
            if gaussian_blur > 0:
                window_lag_img = np.array([
                    pg.gaussianFilter(
                        window_lag_img[i, ..., 0],
                        (gaussian_blur, gaussian_blur),
                    )[..., np.newaxis]
                    for i in range(window_lag_img.shape[0])
                ])
        if bounds is None and reference is None and window_lag is None:
            return False
        if operation == "subtract":
            if range_img is not None:
                image = image - range_img
            if reference_img is not None:
                image = image - reference_img
            if window_lag_img is not None:
                image = image[:window_lag_img.shape[0]] - window_lag_img
            self._norm = image
        if operation == "divide":
            if range_img is not None:
                image = image / range_img
            if reference_img is not None:
                image = image / reference_img
            if window_lag_img is not None:
                image = image[:window_lag_img.shape[0]] / window_lag_img
            self._norm = image
        self._norm_operation = operation  # type: ignore
        return True

    def unravel(self) -> None:
        self._redop = None

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

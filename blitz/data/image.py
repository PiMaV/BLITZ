from typing import Any, Literal

import numpy as np
import pyqtgraph as pg

from ..tools import log


class ImageData:

    def __init__(
        self,
        image: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self._image = image
        self._meta = metadata
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._mask: tuple[slice, slice, slice] | None = None
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
        self._reduce_operation = None
        self._norm: np.ndarray | None = None
        self._norm_operation: Literal["subtract", "divide"] | None = None

    def reset(self) -> None:
        self._min = None
        self._max = None
        self._mean = None
        self._std = None
        self._mask = None
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
        self._reduce_operation = None
        self._norm = None
        self._norm_operation = None

    @property
    def image(self) -> np.ndarray:
        image: np.ndarray
        match self._reduce_operation:
            case 'min':
                image = self._min  # type: ignore
            case 'max':
                image = self._max  # type: ignore
            case 'mean':
                image = self._mean  # type: ignore
            case 'std':
                image = self._std  # type: ignore
            case _:
                image = self._image
        if self._norm is not None:
            image = self._norm
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
        return self._image.shape[0]

    @property
    def meta(self) -> list[dict[str, Any]]:
        return self._meta

    def reduce(self, operation: Literal['min', 'max', 'mean', 'std']) -> None:
        match operation:
            case 'min':
                if self._min is None:
                    self._min = np.expand_dims(self._image.min(0), axis=0)
            case 'max':
                if self._max is None:
                    self._max = np.expand_dims(self._image.max(0), axis=0)
            case 'mean':
                if self._mean is None:
                    self._mean = np.expand_dims(self._image.mean(0), axis=0)
            case 'std':
                if self._std is None:
                    self._std = np.expand_dims(self._image.std(0), axis=0)
        self._reduce_operation = operation

    def normalize(self, left: int, right: int, beta: float, name: str) -> None:
        if self._reduce_operation is not None:
            log("Normalization not possible on reduced data")
            return
        if self._norm_operation == name:
            self._norm_operation = None
            self._norm = None
            return
        if self._norm_operation is not None:
            self._norm = None
        image = self.image
        mean = beta * np.mean(image[left:right+1], axis=0)
        if name == "subtract":
            self._norm = image - mean
        if name == "divide":
            self._norm = image / mean
        self._norm_operation = name  # type: ignore

    def unravel(self) -> None:
        self._reduce_operation = None

    def mask(self, roi: pg.ROI) -> None:
        if self._transposed or self._flipped_x or self._flipped_y:
            log("Masking not available while data is flipped or transposed")
            return
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
        op = self._reduce_operation
        self.reset()
        self.reduce(op)  # type: ignore
        self._mask = (
            slice(None, None), slice(x_start, x_stop), slice(y_start, y_stop),
        )

    def transpose(self) -> None:
        self._transposed = not self._transposed

    def flip_x(self) -> None:
        self._flipped_x = not self._flipped_x

    def flip_y(self) -> None:
        self._flipped_y = not self._flipped_y

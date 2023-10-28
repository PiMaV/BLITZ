from typing import Any

import numpy as np
import pyqtgraph as pg

from ..tools import log


class ImageData:

    def __init__(self) -> None:
        self._image = np.empty((1, ))
        self._meta = []
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._mask: tuple[slice, slice, slice] | None = None
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False

    def set(
        self,
        image: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self.reset()
        self._image = image
        self._meta = metadata

    @property
    def image(self) -> np.ndarray:
        image = self._image
        if self._mask is not None:
            image = self._image[self._mask]
        if self._transposed:
            image = np.swapaxes(image, 1, 2)
        if self._flipped_x:
            image = np.flip(image, 1)
        if self._flipped_y:
            image = np.flip(image, 2)
        return image

    @property
    def meta(self) -> list[dict[str, Any]]:
        return self._meta

    @property
    def min(self) -> np.ndarray:
        if self._min is None:
            self._min = np.min(self.image, axis=0)
        return self._min

    @property
    def max(self) -> np.ndarray:
        if self._max is None:
            self._max = np.max(self.image, axis=0)
        return self._max

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            self._mean = np.mean(self.image, axis=0)
        return self._mean

    @property
    def std(self) -> np.ndarray:
        if self._std is None:
            self._std = np.std(self.image, axis=0)
        return self._std

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
        self.reset()
        self._mask = (
            slice(None, None), slice(x_start, x_stop), slice(y_start, y_stop),
        )

    def reset(self) -> None:
        self._mask = None
        self._min = None
        self._max = None
        self._mean = None
        self._std = None

    def transpose(self) -> None:
        self._transposed = not self._transposed

    def flip_x(self) -> None:
        self._flipped_x = not self._flipped_x

    def flip_y(self) -> None:
        self._flipped_y = not self._flipped_y

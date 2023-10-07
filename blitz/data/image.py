from typing import Any

import numpy as np
import pyqtgraph as pg


class ImageData:

    def __init__(self) -> None:
        self._original_image = np.empty((1, ))
        self._meta = []
        self._image: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def set(
        self,
        image: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self.reset()
        self._original_image = image
        self._meta = metadata

    @property
    def image(self) -> np.ndarray:
        if self._image is not None:
            return self._image
        else:
            return self._original_image

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
        if self._image is None:
            self._image = self._original_image
        pos = roi.pos()
        size = roi.size()
        x_start = max(0, int(pos[0]))
        y_start = max(0, int(pos[1]))
        x_end = min(self._image.shape[1], int(pos[0] + size[0]))
        y_end = min(self._image.shape[2], int(pos[1] + size[1]))
        self._image = self._image[:, x_start:x_end, y_start:y_end]

    def reset(self) -> None:
        self._image = None
        self._min = None
        self._max = None
        self._mean = None
        self._std = None

    def transpose(self) -> None:
        if self._image is None:
            self._image = self._original_image
        self._image = np.swapaxes(self._image, 1, 2)

    def flip_x(self) -> None:
        if self._image is None:
            self._image = self._original_image
        self._image = np.flip(self._image, axis=2)

    def flip_y(self) -> None:
        if self._image is None:
            self._image = self._original_image
        self._image = np.flip(self._image, axis=1)

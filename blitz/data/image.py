from typing import Any, Literal, Optional

import numpy as np
import pyqtgraph as pg

from ..tools import log
from .ops import ReduceDict, ReduceOperation, get


class ImageData:

    def __init__(
        self,
        image: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self._image = image
        self._meta = metadata
        self._reduced = ReduceDict()
        self._mask: tuple[slice, slice, slice] | None = None
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
        self._redop: ReduceOperation | str | None = None
        self._norm: np.ndarray | None = None
        self._norm_operation: Literal["subtract", "divide"] | None = None

    def reset(self) -> None:
        self._reduced.clear()
        self._mask = None
        self._transposed = False
        self._flipped_x = False
        self._flipped_y = False
        self._redop = None
        self._norm = None
        self._norm_operation = None

    @property
    def image(self) -> np.ndarray:
        image: np.ndarray
        if self._redop is not None:
            image = self._reduced.reduce(self._image, self._redop)
        else:
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
    def shape(self) -> tuple[int, int]:
        return (self.image.shape[1], self.image.shape[2])

    @property
    def meta(self) -> list[dict[str, Any]]:
        return self._meta

    def is_single_image(self) -> bool:
        return self._image.shape[0] == 1

    def is_greyscale(self) -> bool:
        return self._image.ndim == 3

    def reduce(self, operation: ReduceOperation | str) -> None:
        self._redop = operation

    def normalize(
        self,
        operation: Literal["subtract", "divide"],
        use: ReduceOperation | str,
        beta: float = 1.0,
        left: Optional[int] = None,
        right: Optional[int] = None,
        reference: Optional["ImageData"] = None,
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
        image = self.image
        range_img = reference_img = None
        if left is not None and right is not None:
            range_img = beta * get(use)(image[left:right+1]).astype(np.double)
        if reference is not None:
            if (not reference.is_single_image()
                    or reference._image.shape[1:] != image.shape[1:]):
                log("Error: Background image has incompatible shape")
                return False
            reference_img = reference._image.astype(np.double)
        if left is None and right is None and reference is None:
            return False
        if operation == "subtract":
            if range_img is not None:
                self._norm = image - range_img
            if reference_img is not None:
                self._norm = image - reference_img
        if operation == "divide":
            if range_img is not None:
                self._norm = image / range_img
            if reference_img is not None:
                self._norm = image / reference_img
        self._norm_operation = operation  # type: ignore
        return True

    def unravel(self) -> None:
        self._redop = None

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
        op = self._redop
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

from typing import Optional

import numpy as np

from .viewer import ImageViewer
from ..data.image import ImageData, MetaData
from ..tools import log


class PCAAdapter:

    def __init__(self, viewer: ImageViewer) -> None:
        self.viewer = viewer
        self._svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pca: ImageData | None = None
        self._components_shown = False
        self._reconstruction_shown = False
        self.viewer.image_changed.connect(self.new_base)

    @property
    def is_calculated(self) -> bool:
        return self._svd is not None

    @property
    def is_active(self) -> bool:
        return self._components_shown or self._reconstruction_shown

    @property
    def components_active(self) -> bool:
        return self._components_shown

    @property
    def reconstruction_active(self) -> bool:
        return self._reconstruction_shown

    def new_base(self) -> None:
        if not self.is_active:
            self._svd = None
            self._pca = None

    def calculate(self, components: Optional[float | int] = None) -> None:
        if self._svd is None:
            self._base = self.viewer.data
            if self._base.is_single_image():
                log(
                    "PCA cannot be calculated, only one image is shown",
                    color="red",
                )
                return
            self._svd = U, s, Vh = np.linalg.svd(
                self._base.image.reshape(
                    (-1, np.prod(self._base.image.shape[1:]))
                )
            )
        else:
            U, s, Vh = self._svd

        if isinstance(components, int):
            self._svd = U, s, Vh = (
                U[:, :components], s[:components], Vh[:components, :]
            )
        elif isinstance(components, float):
            raise NotImplementedError()
        matrix = ((U * s) @ Vh).reshape((-1, *self._base.image.shape[1:]))
        self._pca = ImageData(
            matrix,
            [MetaData(
                file_name=f"{self._base.meta[i].file_name}-Reconstruction",
                file_size_MB=matrix.nbytes / 2**20,
                size=(matrix.shape[1], matrix.shape[2]),
                dtype=matrix.dtype,  # type: ignore
                bit_depth=8*matrix.dtype.itemsize,
                color_model=self._base.meta[0].color_model,
            ) for i in range(matrix.shape[0])]
        )

    def show_reconstruction(self) -> None:
        if self._pca is None:
            raise RuntimeError("PCA is not calculated yet")
        self._reconstruction_shown = True
        self.viewer.set_image(self._pca)

    def show_components(self) -> None:
        if self._svd is None:
            raise RuntimeError("PCA is not calculated yet")
        self._components_shown = True
        _, _, Vh = self._svd
        matrix = Vh.reshape((-1, *self._base.image.shape[1:]))
        self.viewer.set_image(
            ImageData(
                matrix,
                [MetaData(
                    file_name=f"PCA-{i}",
                    file_size_MB=matrix.nbytes / 2**20,
                    size=(matrix.shape[1], matrix.shape[2]),
                    dtype=matrix.dtype,  # type: ignore
                    bit_depth=8*matrix.dtype.itemsize,
                    color_model=self._base.meta[0].color_model,
                ) for i in range(matrix.shape[0])]
            )
        )

    def hide(self) -> None:
        if self.is_active:
            self.viewer.set_image(self._base)
            self._reconstruction_shown = False
            self._components_shown = False

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .viewer import ImageViewer
from ..data.image import ImageData, MetaData
from ..tools import log, get_available_ram


class PCACalculator(QThread):
    """Computes SVD in a separate thread to avoid freezing the UI."""

    calc_finished = pyqtSignal(object)  # Emits result tuple
    calc_error = pyqtSignal(str)

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data_ref = data  # Reference to the numpy array (shared memory)

    def run(self) -> None:
        try:
            # Determine shape
            shape = self.data_ref.shape
            n_frames = shape[0]

            # Flatten all spatial dimensions: (T, Features)
            # If shape is (T, H, W) -> (T, H*W)
            # If shape is (T, H, W, C) -> (T, H*W*C)
            matrix = self.data_ref.reshape((n_frames, -1))

            # Memory Check
            # We need:
            # - Original matrix (already in RAM)
            # - Centered matrix (same size, float64 default for mean subtraction unless cast)
            # - U (T, K), s (K,), Vh (K, N)
            # - Mean (N,)
            # approximate additional RAM needed: 2.5 * matrix.nbytes

            needed_bytes = matrix.nbytes * 2.5
            needed_gb = needed_bytes / (1024**3)
            available_gb = get_available_ram()

            if needed_gb > available_gb * 0.9:
                self.calc_error.emit(
                    f"Insufficient RAM for PCA. Need approx. {needed_gb:.2f} GB, "
                    f"available {available_gb:.2f} GB."
                )
                return

            # Compute Mean and Center
            # Force float32 to save memory if original is float32
            dtype = matrix.dtype
            if dtype != np.float32 and dtype != np.float64:
                dtype = np.float32

            mean = np.mean(matrix, axis=0, dtype=dtype)
            centered = matrix - mean

            # Compute SVD
            # full_matrices=False -> U(T, K), s(K,), Vh(K, N) where K=min(T, N)
            U, s, Vh = np.linalg.svd(centered, full_matrices=False)

            # Result: (U, s, Vh, mean, original_shape)
            self.calc_finished.emit((U, s, Vh, mean, shape))

        except Exception as e:
            self.calc_error.emit(f"SVD Computation failed: {str(e)}")


class PCAAdapter(QObject):
    """Adapter to manage PCA calculations and visualization."""

    # Signals to update UI state
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, viewer: ImageViewer) -> None:
        super().__init__()
        self.viewer = viewer

        # Cache: (U, s, Vh, mean, original_shape)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]] = None
        self._base_data: Optional[ImageData] = None

        self._calculator: Optional[PCACalculator] = None

    def calculate(self) -> None:
        """Start PCA calculation on current viewer data."""
        if self.viewer.data is None:
            self.error.emit("No data loaded.")
            return

        if self.viewer.data.is_single_image():
            self.error.emit("PCA requires a video/stack (multiple frames).")
            return

        # Store base data before calculation
        self._base_data = self.viewer.data
        data = self._base_data.image

        # Pre-check dimensions
        if data.ndim < 3:
            self.error.emit("Data must be at least 3D (T, H, W).")
            return

        self.started.emit()

        self._calculator = PCACalculator(data)
        self._calculator.calc_finished.connect(self._on_calculation_finished)
        self._calculator.calc_error.connect(self._on_calculation_error)
        self._calculator.start()

    def _on_calculation_finished(self, result: tuple) -> None:
        self._cache = result
        self.finished.emit()
        self._calculator = None
        log("PCA Calculation finished.", color="green")

    def _on_calculation_error(self, msg: str) -> None:
        self.error.emit(msg)
        self._calculator = None

    @property
    def is_calculated(self) -> bool:
        return self._cache is not None

    @property
    def max_components(self) -> int:
        if self._cache:
            return len(self._cache[1])  # len(s)
        return 0

    def reset_view(self) -> None:
        """Restore the original image data."""
        if self._base_data:
            self.viewer.set_image(self._base_data)

    def show_components(self, n_components: int) -> None:
        """Display the first n_components eigenimages."""
        if not self.is_calculated or self._cache is None:
            self.error.emit("PCA not calculated.")
            return

        U, s, Vh, mean, shape = self._cache

        # Vh shape: (K, Features)
        k = min(n_components, Vh.shape[0])
        components = Vh[:k, :]

        # Reshape back to image: (k, H, W, ...)
        spatial_shape = shape[1:]

        try:
            eigenimages = components.reshape((k, *spatial_shape))

            # Metadata
            meta_list = []
            for i in range(k):
                meta_list.append(MetaData(
                    file_name=f"PC-{i+1}",
                    file_size_MB=0,
                    size=(spatial_shape[0], spatial_shape[1]),
                    dtype=eigenimages.dtype,
                    bit_depth=32,
                    color_model="Gray" if len(spatial_shape) == 2 else "RGB"
                ))

            img_data = ImageData(eigenimages, meta_list)
            self.viewer.set_image(img_data)

        except Exception as e:
            self.error.emit(f"Failed to show components: {e}")

    def show_reconstruction(self, n_components: int) -> None:
        """Display the reconstruction using top n_components."""
        if not self.is_calculated or self._cache is None:
            self.error.emit("PCA not calculated.")
            return

        U, s, Vh, mean, shape = self._cache

        k = min(n_components, len(s))

        try:
            # Reconstruct: X ~ U_k * S_k @ Vh_k + mean
            # U (T, K), s (K,), Vh (K, N)

            # (T, k) * (k,) -> (T, k)
            scores = U[:, :k] * s[:k]

            # (T, k) @ (k, N) -> (T, N)
            reconstructed_flat = scores @ Vh[:k, :]

            # Add mean
            reconstructed_flat += mean

            # Reshape
            reconstructed = reconstructed_flat.reshape(shape)

            # Metadata
            meta_list = []
            for i in range(shape[0]):
                meta_list.append(MetaData(
                    file_name=f"Rec-{i} (k={k})",
                    file_size_MB=0,
                    size=(shape[1], shape[2]),
                    dtype=reconstructed.dtype,
                    bit_depth=32,
                    color_model="Gray" if len(shape) == 3 else "RGB"
                ))

            img_data = ImageData(reconstructed, meta_list)
            self.viewer.set_image(img_data)

        except Exception as e:
            self.error.emit(f"Failed to reconstruct: {e}")

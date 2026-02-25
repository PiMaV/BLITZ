from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .viewer import ImageViewer
from ..data.image import ImageData, MetaData
from ..tools import log, get_available_ram
from ..data.pca_algo import randomized_svd_low_memory, svd_exact


class PCACalculator(QThread):
    """Computes SVD in a separate thread to avoid freezing the UI."""

    calc_finished = pyqtSignal(object)  # Emits result tuple
    calc_error = pyqtSignal(str)

    def __init__(self, data: np.ndarray, n_components: int, exact: bool = False) -> None:
        super().__init__()
        self.data_ref = data
        self.n_components = n_components
        self.exact = exact

    def run(self) -> None:
        try:
            # Determine shape
            shape = self.data_ref.shape
            n_frames = shape[0]

            # Flatten all spatial dimensions: (T, Features)
            matrix = self.data_ref.reshape((n_frames, -1))

            # Memory Check
            available_gb = get_available_ram()

            if self.exact:
                # Exact SVD requires creating centered matrix (float64 by default for mean subtraction)
                # If input is uint8 (1 byte), float64 is 8 bytes.
                # Factor: 1.0 (mean) + 8.0 (centered) + SVD overhead (~2-3x centered)
                # If input is already float32, centered is 1.0.

                bpp_in = matrix.dtype.itemsize
                bpp_calc = 8  # float64

                ratio = bpp_calc / bpp_in
                # Centered matrix copy + SVD internals
                needed_bytes = matrix.nbytes * (ratio + ratio * 2.5)
                needed_gb = needed_bytes / (1024**3)

                if needed_gb > available_gb * 0.9:
                    self.calc_error.emit(
                        f"Insufficient RAM for Exact PCA. Need ~{needed_gb:.2f} GB, "
                        f"available {available_gb:.2f} GB. Try Approximate PCA."
                    )
                    return

                U, s, Vh, mean = svd_exact(matrix)

            else:
                # Randomized SVD is much more memory efficient.
                # It does NOT form the dense centered matrix.
                # We mainly need Q (T, k) and B (k, N) in float32/64.
                # k is usually small (e.g. 20-50).
                # Overhead is dominated by the random matrix Omega (N, k+oversamples).
                # If k is small, this is negligible compared to T*N.
                # We verify we have enough for the intermediate projections.

                k_target = self.n_components + 10 # oversamples
                # Omega (N, k), Y (T, k), Q (T, k), B (k, N)
                # All float32 or float64.
                bpp_calc = 4 # assume float32 for randomized (as set in algo)

                elements_overhead = (
                    matrix.shape[1] * k_target + # Omega
                    matrix.shape[0] * k_target + # Y / Q
                    k_target * matrix.shape[1]   # B
                )

                needed_bytes = elements_overhead * bpp_calc
                needed_gb = needed_bytes / (1024**3)

                if needed_gb > available_gb * 0.95:
                     self.calc_error.emit(
                        f"Insufficient RAM even for Approximate PCA. "
                        f"Available {available_gb:.2f} GB."
                    )
                     return

                U, s, Vh, mean = randomized_svd_low_memory(
                    matrix,
                    n_components=self.n_components,
                    n_iter=2,
                    random_state=42
                )

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
        self._last_exact: bool = False

    def calculate(self, n_components: int, exact: bool = False) -> None:
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

        # PCA needs frame stack (T>1); aggregate view yields (1,H,W,C)
        if data.shape[0] < 2:
            self.error.emit(
                "PCA requires multiple frames. You are in aggregate view (single image). "
                "Switch to Frame (Reduce=None) first."
            )
            return

        # Pre-check dimensions
        if data.ndim < 3:
            self.error.emit("Data must be at least 3D (T, H, W).")
            return

        self.started.emit()
        self._last_exact = exact

        self._calculator = PCACalculator(data, n_components, exact)
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

    def invalidate(self) -> None:
        """Clear PCA cache when data changes (load, crop, etc.)."""
        self._cache = None
        self._base_data = None
        calc = self._calculator
        if calc is not None:
            self._calculator = None
            try:
                calc.calc_finished.disconnect()
                calc.calc_error.disconnect()
            except TypeError:
                pass
            calc.quit()
            calc.wait(100)

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
                    color_model="grayscale" if spatial_shape[-1] == 1 else "rgb"
                ))

            img_data = ImageData(eigenimages, meta_list)
            self.viewer.set_image(img_data)

        except Exception as e:
            self.error.emit(f"Failed to show components: {e}")

    def variance_explained(self, n_components: int) -> float:
        """Cumulative variance explained by first n components [0..100]."""
        if not self._cache:
            return 0.0
        s = self._cache[1]
        k = min(n_components, len(s))
        total = float((s**2).sum())
        if total <= 0:
            return 0.0
        return 100.0 * float((s[:k]**2).sum()) / total

    def variance_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indices, indiv_pct, cumul_pct) for variance plot."""
        if not self._cache:
            return np.array([]), np.array([]), np.array([])
        s = self._cache[1]
        total = float((s**2).sum())
        if total <= 0:
            return np.arange(len(s)), np.zeros_like(s), np.zeros_like(s)
        indiv = 100.0 * (s**2) / total
        cumul = np.cumsum(indiv)
        return np.arange(1, len(s) + 1, dtype=float), indiv, cumul

    def show_reconstruction(self, n_components: int, add_mean: bool = True) -> None:
        """Display the reconstruction using top n_components."""
        if not self.is_calculated or self._cache is None:
            self.error.emit("PCA not calculated.")
            return

        U, s, Vh, mean, shape = self._cache

        k = min(n_components, len(s))

        try:
            # Reconstruct: X ~ U_k * S_k @ Vh_k [+ mean]
            # U (T, K), s (K,), Vh (K, N)

            # (T, k) * (k,) -> (T, k)
            scores = U[:, :k] * s[:k]

            # (T, k) @ (k, N) -> (T, N)
            reconstructed_flat = scores @ Vh[:k, :]

            if add_mean:
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
                    color_model="grayscale" if shape[-1] == 1 else "rgb"
                ))

            img_data = ImageData(reconstructed, meta_list)
            self.viewer.set_image(img_data)

        except Exception as e:
            self.error.emit(f"Failed to reconstruct: {e}")

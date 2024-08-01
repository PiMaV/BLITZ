from typing import Literal, Optional

from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg

from .viewer import ImageViewer
from .widgets import ExtractionPlot
from ..data.tools import normalize


class ROSEEAdapter:

    def __init__(
        self,
        viewer: ImageViewer,
        h_plot: ExtractionPlot,
        v_plot: ExtractionPlot,
    ) -> None:
        self.viewer = viewer
        self.h_plot = h_plot
        self.v_plot = v_plot
        self.roi = pg.LineSegmentROI(
            [[0, 0], [100, 100]],
            pen=pg.mkPen('r', style=Qt.PenStyle.DotLine),
        )
        self.viewer.addItem(self.roi)
        self.roi.hide()
        self._show: Literal["h", "v"] | None = None
        self.h_plot._extractionline.sigPositionChanged.connect(self.update)
        self.v_plot._extractionline.sigPositionChanged.connect(self.update)
        self.viewer.timeLine.sigPositionChanged.connect(self.update)
        self._components_shown = False
        self._reconstruction_shown = False

    @property
    def is_visible(self) -> bool:
        return self._show is not None

    def toggle(self, show: Optional[Literal["h", "v"]] = None) -> None:
        self._show = show

    def update(self) -> None:
        if not self.is_visible:
            return
        if self._show == "h":
            plot = self.h_plot
        else:
            plot = self.v_plot
        signal = plot.extract_data()
        if signal is None:
            return
        if signal.shape[1] > 1:
            weights = np.array([0.2989, 0.5870, 0.1140])
            signal = np.sum(signal * weights, axis=-1)
            print(signal.shape)
        signal, cumsum, fluctuation, indices = self.calculate(signal)
        plot.plotItem.clear()
        plot.plot(
            signal[..., np.newaxis], normed=True, pen="w", name="norm. Signal"
        )
        plot.plot(
            cumsum[..., np.newaxis], normed=True, pen="c", name="norm. Cumsum"
        )
        plot.plot(
            fluctuation[..., np.newaxis], normed=True, pen="r",
            name="Fluctuation Cumsum",
        )

    def calculate(
        self,
        signal: np.ndarray,
        use_local_extrema: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
        norm_signal = normalize(signal)
        norm_cumsum = normalize(np.cumsum(norm_signal))
        fluctuation = normalize(np.cumsum(norm_cumsum - norm_cumsum.mean()))

        diff_cusum = np.diff(fluctuation)
        max_slope_index = int(np.argmax(diff_cusum))
        if max_slope_index <= 0 or max_slope_index >= len(diff_cusum) - 1:
            event_indices = (0, 0, 0)
        elif use_local_extrema:
            left_diff = diff_cusum[:max_slope_index]
            left_sign_change = np.where(np.diff(np.sign(left_diff)))[0]
            if len(left_sign_change) == 0:
                start_index = 0
            else:
                start_index = left_sign_change[-1] + 1

            right_diff = diff_cusum[max_slope_index:]
            right_sign_change = np.where(np.diff(np.sign(right_diff)))[0]
            if len(right_sign_change) == 0:
                end_index = len(fluctuation) - 1
            else:
                end_index = right_sign_change[0] + max_slope_index + 1
            event_indices = (start_index, max_slope_index, end_index)
        else:
            min_index = np.argmin(fluctuation[:max_slope_index])
            max_index = np.argmax(fluctuation[max_slope_index:]) + max_slope_index
            if min_index >= max_slope_index or max_index <= max_slope_index:
                event_indices = (0, 0, 0)
            else:
                start_index = min_index
                end_index = max_index
                event_indices = (start_index, max_slope_index, end_index)

        return norm_signal, norm_cumsum, fluctuation, event_indices


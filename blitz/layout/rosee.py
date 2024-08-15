from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg

from .viewer import ImageViewer
from .widgets import ExtractionPlot
from ..data.tools import unify_range


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
        self._show_vertical: bool = False
        self._show_horizontal: bool = False
        self._components_shown = False
        self._reconstruction_shown = False

    @property
    def is_visible(self) -> bool:
        return self._show_vertical or self._show_horizontal

    def toggle(self, horizontal: bool = False, vertical: bool = False) -> None:
        self._show_horizontal = horizontal
        self._show_vertical = vertical

    def update(self, use_local_extrema: bool, smoothing: int) -> None:
        if self._show_horizontal:
            self._update(self.h_plot, use_local_extrema, smoothing)
        else:
            self.h_plot.draw_line()
        if self._show_vertical:
            self._update(self.v_plot, use_local_extrema, smoothing)
        else:
            self.v_plot.draw_line()

    def _update(
        self,
        plot: ExtractionPlot,
        use_local_extrema: bool,
        smoothing: int,
    ) -> None:
        signal = plot.extract_data()
        if signal is None:
            return
        if signal.shape[1] > 1:
            weights = np.array([0.2989, 0.5870, 0.1140])
            signal = np.sum(signal * weights, axis=-1)
        else:
            signal = signal.squeeze(1)
        cumsum, fluctuation, indices = self.calculate(
            signal,
            use_local_extrema=use_local_extrema,
            smoothing=smoothing,
        )
        indices = indices.astype(int)

        signal, cumsum, fluctuation = unify_range(signal, cumsum, fluctuation)
        plot.plotItem.clear()
        plot.plot(
            signal[..., np.newaxis],
            pen=pg.mkPen('w'),
            name="norm. Signal",
        )
        plot.plot(
            cumsum[..., np.newaxis],
            pen=pg.mkPen('c', style=Qt.PenStyle.DashDotDotLine),
            name="norm. Cumsum",
        )
        plot.plot(
            fluctuation[..., np.newaxis],
            pen=pg.mkPen('r', style=Qt.PenStyle.DashDotDotLine),
            name="Fluctuation Cumsum",
        )

        plot.plot_x_y(
            indices[0:1],
            fluctuation[indices[0]:indices[0]+1],
            pen=None,
            symbol='o',
            symbolBrush="m",
            name='Min Index',
        )
        text_min = pg.TextItem(
            text=str(indices[0]),
            color='w',
            anchor=(.5, 0),
        )
        text_min.setPos(*plot.get_translated_pos(indices[0], 0))
        plot.addItem(text_min)

        plot.plot_x_y(
            indices[1:2],
            fluctuation[indices[1]:indices[1]+1],
            pen=None,
            symbol='t',
            symbolBrush="b",
            name='Max Slope Index',
        )
        text_max_slope = pg.TextItem(
            text=str(indices[1]),
            color='w',
            anchor=(.5, 0),
        )
        text_max_slope.setPos(*plot.get_translated_pos(indices[1], 0))
        plot.addItem(text_max_slope)

        plot.plot_x_y(
            indices[2:3],
            fluctuation[indices[2]:indices[2]+1],
            pen=None,
            symbol='s',
            symbolBrush="m",
            name='Max Index',
        )
        text_max = pg.TextItem(
            text=str(indices[2]),
            color='w',
            anchor=(.5, 0),
        )
        text_max.setPos(*plot.get_translated_pos(indices[2], 0))
        plot.addItem(text_max)

    def calculate(
        self,
        signal: np.ndarray,
        use_local_extrema: bool = True,
        smoothing: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cumsum = np.cumsum(signal)
        fluctuation = np.cumsum(signal - signal.mean())
        if smoothing > 1:
            fluctuation = np.convolve(
                np.pad(fluctuation, smoothing // 2, mode="reflect"),
                np.ones(smoothing) / smoothing,
                mode="same",
            )[smoothing // 2: -(smoothing) // 2]

        diff_fluct = np.diff(fluctuation)
        max_slope_index = int(np.argmax(diff_fluct))
        if max_slope_index <= 0 or max_slope_index >= len(diff_fluct) - 1:
            event_indices = np.zeros(3)
        elif use_local_extrema:
            left_diff = diff_fluct[:max_slope_index]
            left_sign_change = np.where(np.diff(np.sign(left_diff)))[0]
            if len(left_sign_change) == 0:
                start_index = 0
            else:
                start_index = left_sign_change[-1] + 1

            right_diff = diff_fluct[max_slope_index:]
            right_sign_change = np.where(np.diff(np.sign(right_diff)))[0]
            if len(right_sign_change) == 0:
                end_index = len(fluctuation) - 1
            else:
                end_index = right_sign_change[0] + max_slope_index + 1
            event_indices = np.array((start_index, max_slope_index, end_index))
        else:
            min_index = np.argmin(fluctuation[:max_slope_index])
            max_index = (
                np.argmax(fluctuation[max_slope_index:]) + max_slope_index
            )
            if min_index >= max_slope_index or max_index <= max_slope_index:
                event_indices = np.zeros(3)
            else:
                start_index = min_index
                end_index = max_index
                event_indices = np.array(
                    (start_index, max_slope_index, end_index)
                )
        return cumsum, fluctuation, event_indices

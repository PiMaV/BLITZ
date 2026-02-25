from typing import Literal

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLineEdit

from ..data.tools import normalize, unify_range
from .viewer import ImageViewer
from .widgets import ExtractionPlot


class ROSEEAdapter:

    def __init__(
        self,
        viewer: ImageViewer,
        h_plot: ExtractionPlot,
        v_plot: ExtractionPlot,
        interval_edit: tuple[QLineEdit, QLineEdit],
        slope_edit: tuple[QLineEdit, QLineEdit],
    ) -> None:
        self.viewer = viewer
        self.h_plot = h_plot
        self.v_plot = v_plot
        self.interval_edit = interval_edit
        self.slope_edit = slope_edit
        self._show_v_plot: bool = False
        self._show_h_plot: bool = False
        self._show_v_image: bool = False
        self._show_h_image: bool = False
        self._components_shown = False
        self._reconstruction_shown = False
        self._index_lines_h: list[pg.InfiniteLine] = []
        self._index_lines_v: list[pg.InfiniteLine] = []
        self._all_lines_h: list[pg.PlotDataItem] = []
        self._all_lines_v: list[pg.PlotDataItem] = []

    def toggle(
        self,
        h_plot: bool = False,
        v_plot: bool = False,
        h_image: bool = False,
        v_image: bool = False,
    ) -> None:
        self._show_h_plot = h_plot
        self._show_v_plot = v_plot
        self._show_h_image = h_image
        self._show_v_image = v_image

    def update(
        self,
        use_local_extrema: bool,
        smoothing: int,
        normalized: bool,
        show_indices: bool,
        show_index_lines: bool,
    ) -> None:
        if self._show_h_plot or self._show_h_image:
            self._update(
                "h",
                self._show_h_plot,
                self._show_h_image,
                use_local_extrema,
                smoothing,
                normalized,
                show_index_lines,
                show_indices,
            )
        else:
            while len(self._index_lines_h) > 0:
                self.viewer.view.removeItem(self._index_lines_h.pop())
            while len(self._all_lines_h) > 0:
                self.viewer.view.removeItem(self._all_lines_h.pop())
            self.h_plot.draw_line()
            self.interval_edit[0].setText("")
            self.slope_edit[0].setText("eye: ")
        if self._show_v_plot or self._show_v_image:
            self._update(
                "v",
                self._show_v_plot,
                self._show_v_image,
                use_local_extrema,
                smoothing,
                normalized,
                show_index_lines,
                show_indices,
            )
        else:
            while len(self._index_lines_v) > 0:
                self.viewer.view.removeItem(self._index_lines_v.pop())
            while len(self._all_lines_v) > 0:
                self.viewer.view.removeItem(self._all_lines_v.pop())
            self.v_plot.draw_line()
            self.interval_edit[1].setText("")
            self.slope_edit[1].setText("eye: ")

    def _update(
        self,
        orientation: Literal["h", "v"],
        in_plot: bool,
        in_image: bool,
        use_local_extrema: bool,
        smoothing: int,
        normalized: bool,
        show_index_lines: bool,
        show_indices: bool,
    ) -> None:
        plot = self.h_plot if orientation == "h" else self.v_plot
        sbrush = pg.mkColor(30, 255, 0) if orientation == "h" else (
            pg.mkColor(255, 0, 255)
        )
        sbrush2 = pg.mkColor(10, 150, 0) if orientation == "h" else (
            pg.mkColor(150, 0, 150)
        )

        if in_plot:
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
            if normalized:
                signal = normalize(signal)
                cumsum = normalize(cumsum)
                fluctuation = normalize(fluctuation)

            signal, cumsum, fluctuation = unify_range(
                signal, cumsum, fluctuation,
            )
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

            self.interval_edit[0 if orientation == "h" else 1].setText(
                f"[{indices[0], indices[2]}]"
            )
            self.slope_edit[0 if orientation == "h" else 1].setText(
                f"eye: {indices[1]}"
            )

            plot.plot_x_y(
                indices[1:2]+0.5,
                fluctuation[indices[1]:indices[1]+1],
                pen=None,
                symbol='t',
                symbolBrush=sbrush2,
                name='Max Slope Index',
            )
            plot.plot_x_y(
                indices[0:1]+0.5,
                fluctuation[indices[0]:indices[0]+1],
                pen=None,
                symbol='o',
                symbolBrush=sbrush,
                name='Min Index',
            )
            plot.plot_x_y(
                indices[2:3]+0.5,
                fluctuation[indices[2]:indices[2]+1],
                pen=None,
                symbol='o',
                symbolBrush=sbrush,
                name='Max Index',
            )

            if show_indices:
                text_min = pg.TextItem(
                    text=str(indices[0]),
                    color=sbrush,
                    anchor=(.5, 1) if orientation == "h" else (1, .5),
                )
                text_min.setPos(*plot.get_translated_pos(indices[0]+0.5, 0))
                plot.addItem(text_min)

                text_max_slope = pg.TextItem(
                    text=str(indices[1]),
                    color=sbrush2,
                    anchor=(.5, 1) if orientation == "h" else (1, .5),
                )
                text_max_slope.setPos(
                    *plot.get_translated_pos(indices[1]+0.5, 0)
                )
                plot.addItem(text_max_slope)

                text_max = pg.TextItem(
                    text=str(indices[2]),
                    color=sbrush,
                    anchor=(.5, 1) if orientation == "h" else (1, .5),
                )
                text_max.setPos(*plot.get_translated_pos(indices[2]+0.5, 0))
                plot.addItem(text_max)

            index_lines = (
                self._index_lines_v if orientation == "v"
                else self._index_lines_h
            )
            if show_index_lines:
                while len(index_lines) > 0:
                    self.viewer.view.removeItem(index_lines.pop())
                min_line = pg.InfiniteLine(
                    angle=90 if orientation == "h" else 0,
                    pen=sbrush,
                )
                eye_line = pg.InfiniteLine(
                    angle=90 if orientation == "h" else 0,
                    pen=sbrush2,
                )
                max_line = pg.InfiniteLine(
                    angle=90 if orientation == "h" else 0,
                    pen=sbrush,
                )
                min_line.setValue(indices[0]+0.5)
                eye_line.setValue(indices[1]+0.5)
                max_line.setValue(indices[2]+0.5)
                self.viewer.view.addItem(min_line)
                self.viewer.view.addItem(eye_line)
                self.viewer.view.addItem(max_line)
                index_lines.append(min_line)
                index_lines.append(eye_line)
                index_lines.append(max_line)
            else:
                while len(index_lines) > 0:
                    self.viewer.view.removeItem(index_lines.pop())

        if in_image:
            bounds_left, max_slopes, bounds_right = self.calculate_all(
                orientation,
                use_local_extrema=use_local_extrema,
                smoothing=smoothing,
            )
            if orientation == "h":
                while len(self._all_lines_h) > 0:
                    self.viewer.view.removeItem(self._all_lines_h.pop())
                self._all_lines_h.append(self.viewer.view.plot(
                    max_slopes+0.5,
                    np.arange(len(max_slopes))+0.5,
                    pen=None,
                    symbol='t',
                    symbolBrush=sbrush2,
                ))
                self._all_lines_h.append(self.viewer.view.plot(
                    bounds_left+0.5,
                    np.arange(len(bounds_left))+0.5,
                    pen=None,
                    symbol='o',
                    symbolBrush=sbrush,
                 ))
                self._all_lines_h.append(self.viewer.view.plot(
                    bounds_right+0.5,
                    np.arange(len(bounds_right))+0.5,
                    pen=None,
                    symbol='o',
                    symbolBrush=sbrush,
                ))
            else:
                while len(self._all_lines_v) > 0:
                    self.viewer.view.removeItem(self._all_lines_v.pop())
                self._all_lines_v.append(self.viewer.view.plot(
                    np.arange(len(max_slopes))+0.5,
                    max_slopes+0.5,
                    pen=None,
                    symbol='t',
                    symbolBrush=sbrush2,
                ))
                self._all_lines_v.append(self.viewer.view.plot(
                    np.arange(len(bounds_left))+0.5,
                    bounds_left+0.5,
                    pen=None,
                    symbol='o',
                    symbolBrush=sbrush,
                ))
                self._all_lines_v.append(self.viewer.view.plot(
                    np.arange(len(bounds_right))+0.5,
                    bounds_right+0.5,
                    pen=None,
                    symbol='o',
                    symbolBrush=sbrush,
                ))
        else:
            if orientation == "h":
                while len(self._all_lines_h) > 0:
                    self.viewer.view.removeItem(self._all_lines_h.pop())
            else:
                while len(self._all_lines_v) > 0:
                    self.viewer.view.removeItem(self._all_lines_v.pop())

    def calculate_all(
        self,
        orientation: Literal["h", "v"],
        use_local_extrema: bool = True,
        smoothing: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left = np.zeros(self.viewer.data.shape[1 if orientation == "h" else 0])
        slope = left.copy()
        right = left.copy()
        for i in range(self.viewer.data.shape[1 if orientation == "h" else 0]):
            if orientation == "h":
                _, _, indices = self.calculate(
                    self.viewer.now[:, i, 0],
                    use_local_extrema=use_local_extrema,
                    smoothing=smoothing,
                )
            else:
                _, _, indices = self.calculate(
                    self.viewer.now[i, :, 0],
                    use_local_extrema=use_local_extrema,
                    smoothing=smoothing,
                )
            left[i] = indices[0]
            slope[i] = indices[1]
            right[i] = indices[2]
        return left, slope, right

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

        if not use_local_extrema:
            start_index = np.argmin(fluctuation)
            end_index = np.argmax(fluctuation)
            max_slope = np.argmax(signal)

            event_indices = np.array((start_index, max_slope, end_index))
        else:
            max_slope = np.argmax(signal)
            diff_fluctuation = np.diff(fluctuation)
            left_sign_change = np.where(
                np.diff(np.sign(diff_fluctuation[:max_slope]))
            )[0]
            start_index = 0 if len(left_sign_change)==0 else (
                left_sign_change[-1] + 1
            )

            right_sign_change = np.where(
                np.diff(np.sign(diff_fluctuation[max_slope:]))
            )[0]
            end_index = len(fluctuation)-1 if len(right_sign_change)==0 else (
                right_sign_change[0] + max_slope + 1
            )
            event_indices = np.array((start_index, max_slope, end_index))

        return cumsum, fluctuation, event_indices

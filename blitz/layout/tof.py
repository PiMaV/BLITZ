from typing import Any

import numpy as np
from PyQt5.QtWidgets import QWidget

from ..data.load import tof_from_json
from ..data.tools import smoothen
from .widgets import WindowedPlot


class TOFWindow(WindowedPlot):

    def __init__(
        self,
        parent: QWidget,
        path: str,
        video_metadata: list[dict[str, Any]],
    ) -> None:
        super().__init__(parent)

        self.video_metadata = video_metadata

        self.setWindowTitle("TOF Plot")
        self.resize(500, 400)

        self.smoothing_level = 3
        self.update_plot(path)
        self.show()

    def sync_to_video(self, data: np.ndarray) -> np.ndarray:
        # Calculate the duration of each frame in the ORIGINAL video in ms
        original_frame_duration_ms = 1000 / self.video_metadata[0]['fps']

        # Calculate the duration of each frame in the RESAMPLED video
        resampled_frame_duration_ms = (
            original_frame_duration_ms
            * self.video_metadata[0]['frame_count']
            / data.shape[0]
        )

        # Generate the times for each frame in the resampled video
        resampled_frame_times = np.arange(
            0,
            data.shape[0] * resampled_frame_duration_ms,
            resampled_frame_duration_ms,
        )

        # Create an array for synced data (frame number + sensor values)
        synced_data = np.zeros_like(data)

        # Fill the first column with the frame numbers
        synced_data[:, 0] = np.arange(data.shape[0])

        # perform linear interpolation for the TOF data onto the
        # resampled video's frame times
        # TODO: This can be one off! Check if the first and last frame
        # times are the same or such
        for col in range(1, data.shape[1]):
            synced_data[:, col] = np.interp(
                resampled_frame_times,
                data[:, 0],
                data[:, col],
            )

        min_tof = np.min(synced_data[:, 1])
        max_tof = np.max(synced_data[:, 1])

        synced_data[:, 1] = 200 - 200 * (
            synced_data[:, 1] - min_tof
        ) / (max_tof - min_tof)
        return synced_data

    def update_plot(self, path: str) -> None:
        data = tof_from_json(path)
        # data = self.sync_to_video(data)

        x_data = data[:, 0]
        y_data = data[:, 1]
        y_data = y_data.max() - y_data

        self.plot_data(
            x_data,
            y_data,
            pen_color="gray",
            label="Raw",
        )
        x_smooth, y_smooth = smoothen(
            x_data,
            y_data,
            window_size=self.smoothing_level,
        )
        self.plot_data(
            x_smooth,
            y_smooth,
            pen_color="green" if self.smoothing_level == 3 else "red",
            label=f"Smoothed ({self.smoothing_level})",
        )


    # def plot_or_update_roi(self, roi_plot: pg.PlotWidget) -> None:
    #     x_sync = self.synced_data[:, 0]
    #     y_sync = self.synced_data[:, 1]
    #     x_smooth_sync, y_smooth_sync = smoothen(
    #         x_sync,
    #         y_sync,
    #         window_size=self.tof_smoothing_level,
    #     )

    #     # smoothed data for synced data
    #     if not hasattr(self, 'smoothed_line'):
    #         self.smoothed_line = roi_plot.plot(
    #             x_smooth_sync,
    #             y_smooth_sync,
    #             pen="pink",
    #             label=f"Smoothed Synced Data ({self.tof_smoothing_level})",
    #             width=2,
    #         )
    #     else:
    #         self.smoothed_line.setData(x_smooth_sync, y_smooth_sync)

    # def remove_roi_plot(self, roi_plot: pg.PlotWidget) -> None:
    #     if hasattr(self, 'smoothed_line'):
    #         roi_plot.removeItem(self.smoothed_line)
    #         del self.smoothed_line


    # def close_additional_window(self) -> None:
    #     if hasattr(self, 'additional_plot_window'):
    #         self.additional_plot_window.window.close()
    #         del self.additional_plot_window

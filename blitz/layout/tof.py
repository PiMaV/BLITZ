from typing import Any, Optional

import numpy as np
import pyqtgraph as pg

from ..data.load import tof_from_json
from ..data.tools import smoothen
from ..tools import log


class TOFAdapter:

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self.plot_item = plot_item
        self.data: tuple[np.ndarray, np.ndarray] | None = None
        self.smoothed_data: tuple[np.ndarray, np.ndarray] | None = None
        self._plot = None
        self._smoothed_plot = None

    def _sync_to_video(
        self,
        data: np.ndarray,
        from_n_frames: int,
        to_n_frames: int,
        fps: int,
    ) -> np.ndarray:
        original_frame_duration_ms = 1000 / fps
        resampled_frame_duration_ms = (
            original_frame_duration_ms * from_n_frames / to_n_frames
        )
        resampled_frame_times = np.arange(
            0,
            to_n_frames * resampled_frame_duration_ms,
            resampled_frame_duration_ms,
        )
        synced_data = np.zeros((to_n_frames, data.shape[1]))
        synced_data[:, 0] = np.arange(to_n_frames)

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

    def set_data(
        self,
        path: str,
        video_metadata: Optional[list[dict[str, Any]]] = None,
        smoothing_level: int = 3,
    ) -> None:
        data = tof_from_json(path)
        if (video_metadata is not None
                and "fps" in video_metadata[0]
                and "frame_count" in video_metadata[0]):
            data = self._sync_to_video(
                data,
                from_n_frames=video_metadata[0]["frame_count"],
                to_n_frames=len(video_metadata),
                fps=video_metadata[0]["fps"],
            )

        x_data = data[:, 0]
        y_data = data[:, 1]
        y_data = y_data.max() - y_data
        self.data = (x_data, y_data)
        x_smooth, y_smooth = smoothen(
            x_data,
            y_data,
            window_size=smoothing_level,
        )
        self.smoothed_data = (x_smooth, y_smooth)

    def toggle_plot(self) -> None:
        if self.data is None:
            log("Error: No TOF data loaded")
            return
        if self._plot is not None:
            self.plot_item.removeItem(self._plot)
            self._plot = None
            if self._smoothed_plot is not None:
                self.plot_item.removeItem(self._smoothed_plot)
            self._smoothed_plot = None
        else:
            self._plot = self.plot_item.plot(
                *self.data,
                pen="gray",
                width=1,
            )
            if self.smoothed_data is not None:
                self._smoothed_plot = self.plot_item.plot(
                    *self.smoothed_data,
                    pen="green",
                    width=1,
                )
            self.plot_item.getViewBox().autoRange()  # type: ignore
            self.plot_item.showAxis("left")

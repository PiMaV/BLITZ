import os
from typing import Any

from PyQt5.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QFormLayout, QHBoxLayout, QLabel, QSpinBox,
                             QVBoxLayout)

from ..data.image import VideoMetaData
from ..tools import get_available_ram


class VideoLoadOptionsDialog(QDialog):
    def __init__(self, metadata: VideoMetaData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Loading Options")
        self.metadata = metadata
        self._setup_ui()
        self._update_estimates()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info Section
        info_layout = QFormLayout()
        self.lbl_file = QLabel(self.metadata.file_name)
        self.lbl_dims = QLabel(f"{self.metadata.size[0]} x {self.metadata.size[1]}")
        self.lbl_frames = QLabel(f"{self.metadata.frame_count}")
        info_layout.addRow("File:", self.lbl_file)
        info_layout.addRow("Dimensions:", self.lbl_dims)
        info_layout.addRow("Total Frames:", self.lbl_frames)
        layout.addLayout(info_layout)

        layout.addWidget(QLabel("<b>Loading Options</b>"))

        # Controls
        controls_layout = QFormLayout()

        # Frame Range
        range_layout = QHBoxLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, self.metadata.frame_count - 1)
        self.spin_start.setValue(0)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, self.metadata.frame_count - 1)
        self.spin_end.setValue(self.metadata.frame_count - 1)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        controls_layout.addRow("Frame Range:", range_layout)

        # Step (Skip)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 1000)
        self.spin_step.setValue(1)
        controls_layout.addRow("Step (skip frames):", self.spin_step)

        # Resize
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(1, 100)
        self.spin_resize.setValue(100)
        self.spin_resize.setSuffix(" %")
        controls_layout.addRow("Resize:", self.spin_resize)

        # Grayscale
        self.chk_grayscale = QCheckBox("Load as Grayscale")
        self.chk_grayscale.setChecked(False) # Default from settings?
        controls_layout.addRow("", self.chk_grayscale)

        # Multicore
        self.chk_multicore = QCheckBox("Enable Multicore Loading")
        self.chk_multicore.setChecked(False)
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, os.cpu_count() or 4)
        self.spin_cores.setValue(max(1, (os.cpu_count() or 4) - 1))
        self.spin_cores.setEnabled(False)
        self.chk_multicore.toggled.connect(self.spin_cores.setEnabled)

        mc_layout = QHBoxLayout()
        mc_layout.addWidget(self.chk_multicore)
        mc_layout.addWidget(QLabel("Cores:"))
        mc_layout.addWidget(self.spin_cores)
        controls_layout.addRow("Multicore:", mc_layout)

        layout.addLayout(controls_layout)

        # Estimates
        self.lbl_ram_usage = QLabel("Estimated RAM: Calculating...")
        self.lbl_ram_available = QLabel(f"Available RAM: {get_available_ram():.2f} GB")
        layout.addWidget(self.lbl_ram_usage)
        layout.addWidget(self.lbl_ram_available)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connections for live updates
        self.spin_start.valueChanged.connect(self._update_estimates)
        self.spin_end.valueChanged.connect(self._update_estimates)
        self.spin_step.valueChanged.connect(self._update_estimates)
        self.spin_resize.valueChanged.connect(self._update_estimates)
        self.chk_grayscale.toggled.connect(self._update_estimates)

    def _update_estimates(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        step = self.spin_step.value()
        resize = self.spin_resize.value() / 100.0
        grayscale = self.chk_grayscale.isChecked()

        # Validate range
        if start > end:
            # Normalize so that start <= end by swapping the values
            self.spin_start.setValue(end)
            self.spin_end.setValue(start)
            start, end = end, start
        # Calculate number of frames: len(range(start, end + 1, step))
        if end >= start:
            num_frames = (end - start) // step + 1
        else:
            num_frames = 0

        width = int(self.metadata.size[0] * resize)
        height = int(self.metadata.size[1] * resize)
        channels = 1 if grayscale else 3
        dtype_size = 1 # uint8 is 1 byte

        total_bytes = width * height * channels * num_frames * dtype_size
        gb = total_bytes / (1024**3)

        color = "green"
        available = get_available_ram()
        if gb > available * 0.9:
            color = "red"
        elif gb > available * 0.7:
            color = "orange"

        self.lbl_ram_usage.setText(f"Estimated RAM: <font color='{color}'><b>{gb:.2f} GB</b></font>")

    def get_params(self) -> dict[str, Any]:
        return {
            "frame_range": (self.spin_start.value(), self.spin_end.value()),
            "step": self.spin_step.value(),
            "size_ratio": self.spin_resize.value() / 100.0,
            "grayscale": self.chk_grayscale.isChecked(),
            "multicore": self.spin_cores.value() if self.chk_multicore.isChecked() else 0,
        }

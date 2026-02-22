"""Synthetic Live: generated data stream (Lissajous, Lightning). Compact dark-theme UI."""

import random
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QSlider,
                             QSpinBox, QVBoxLayout, QWidget)

from .. import resources  # noqa: F401
from ..data.live import SimulatedLiveHandler, buffer_frames_from_mb

# Dark theme colors (gray main, black viz area, green accent)
SIMULATED_LIVE_MAIN_BG = "#3d3d3d"
SIMULATED_LIVE_PANEL_BG = "#2d2d2d"
SIMULATED_LIVE_BORDER = "#5a5a5a"
SIMULATED_LIVE_BORDER_DARK = "#1a1a1a"
SIMULATED_LIVE_FG = "#e0e0e0"
SIMULATED_LIVE_FG_DIM = "#a0a0a0"
SIMULATED_LIVE_GREEN = "#00ff00"
SIMULATED_LIVE_GREEN_DIM = "#00aa00"
SIMULATED_LIVE_BLACK = "#0a0a0a"
SIMULATED_LIVE_BTN_UP = "#4a4a4a"
SIMULATED_LIVE_BTN_DOWN = "#252525"
SIMULATED_LIVE_BTN_BORDER = "#6a6a6a"

STYLE_SIMULATED_LIVE = f"""
    QWidget#SimulatedLiveMain {{
        background-color: {SIMULATED_LIVE_MAIN_BG};
        border: 1px solid {SIMULATED_LIVE_BORDER};
        border-radius: 2px;
    }}
    QLabel#SimulatedLiveTitle {{
        background-color: transparent;
        color: {SIMULATED_LIVE_FG};
        font-weight: bold;
        font-size: 10px;
    }}
    QLabel#SimulatedLiveHint {{
        background-color: transparent;
        color: {SIMULATED_LIVE_FG_DIM};
        font-size: 9px;
    }}
    QLabel#SimulatedLiveDisplay {{
        background-color: {SIMULATED_LIVE_BLACK};
        color: {SIMULATED_LIVE_GREEN};
        font-family: "Consolas", "Lucida Console", monospace;
        font-size: 10px;
        padding: 4px 6px;
        border: 1px solid {SIMULATED_LIVE_BORDER_DARK};
        border-radius: 0;
        min-height: 24px;
    }}
    QPushButton#SimulatedLiveBtn {{
        background-color: {SIMULATED_LIVE_BTN_UP};
        color: {SIMULATED_LIVE_FG};
        border: 1px solid {SIMULATED_LIVE_BTN_BORDER};
        border-radius: 0;
        min-width: 26px;
        max-width: 26px;
        min-height: 20px;
        font-size: 10px;
    }}
    QPushButton#SimulatedLiveBtn:hover {{
        background-color: #555555;
        border-color: #7a7a7a;
    }}
    QPushButton#SimulatedLiveBtn:pressed {{
        background-color: {SIMULATED_LIVE_BTN_DOWN};
        border-color: {SIMULATED_LIVE_BORDER_DARK};
    }}
    QPushButton#SimulatedLiveBtn:disabled {{ color: #606060; }}
    QComboBox, QSlider {{
        background-color: {SIMULATED_LIVE_PANEL_BG};
        color: {SIMULATED_LIVE_FG};
        border: 1px solid {SIMULATED_LIVE_BORDER};
        border-radius: 0;
        min-height: 20px;
    }}
    QComboBox::drop-down {{
        border-left: 1px solid {SIMULATED_LIVE_BORDER};
        background-color: {SIMULATED_LIVE_BTN_UP};
        width: 18px;
    }}
    QCheckBox {{
        color: {SIMULATED_LIVE_FG};
        font-size: 9px;
    }}
    QLabel {{
        color: {SIMULATED_LIVE_FG};
        font-size: 9px;
    }}
    QPushButton#SimulatedLiveBtnPlay {{
        background-color: #2a4a2a;
        color: {SIMULATED_LIVE_GREEN};
        border: 1px solid #3a6a3a;
    }}
    QPushButton#SimulatedLiveBtnPlay:hover {{
        background-color: #3a5a3a;
        border-color: #4a7a4a;
    }}
    QPushButton#SimulatedLiveBtnPlay:pressed {{
        background-color: #1a3a1a;
    }}
"""


class _SimulatedViz(QWidget):
    """Mini preview: Lissajous (green) or Lightning (blue zigzag) depending on variant."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setFixedWidth(160)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._t = 0.0
        self._variant = "lightning"

    def set_variant(self, variant: str) -> None:
        self._variant = variant.lower()
        self.update()

    def start(self) -> None:
        self._timer.start(33)

    def stop(self) -> None:
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._t += 0.12
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        margin = 4
        p.fillRect(0, 0, w, h, QColor(10, 10, 10))
        if self._variant == "lightning":
            cx, top_y = w / 2, margin + 4
            rng = random.Random(int(self._t * 10) & 0x7FFFFFFF)
            x, y = cx + rng.randint(-w // 8, w // 8), top_y
            points = [QPointF(x, y)]
            for _ in range(12):
                x = x + rng.randint(-8, 8)
                x = max(margin, min(w - margin, x))
                y = y + 3 + rng.randint(0, 4)
                points.append(QPointF(x, y))
            pen = QPen(QColor(80, 180, 255))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(points))
        else:
            cx, cy = w / 2, h / 2
            n, t = 80, self._t
            points = []
            for i in range(n + 1):
                s = i / n
                x = cx + (w - 2 * margin) / 2 * (
                    np.sin(3 * s * 6.28 + t) * 0.4 + np.sin(s * 6.28 + t * 2) * 0.6
                )
                y = cy + (h - 2 * margin) / 2 * (
                    np.sin(5 * s * 6.28 + t * 1.3) * 0.3
                    + np.cos(s * 6.28 * 2 + t * 1.4) * 0.7
                )
                points.append(QPointF(x, y))
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(points))
        p.end()


class SimulatedLiveWidget(QFrame):
    """Simulated live source: generates Lissajous/Lightning viz, streams to BLITZ viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SimulatedLiveMain")
        self.setStyleSheet(STYLE_SIMULATED_LIVE)
        self.setWindowTitle("Synthetic Live")
        self.setFixedWidth(280)
        self._handler: Optional[SimulatedLiveHandler] = None
        self._on_frame: Optional[Callable[[object], None]] = None
        self._pull_timer: Optional[QTimer] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("BLITZ Synthetic Live")
        title.setObjectName("SimulatedLiveTitle")
        layout.addWidget(title)
        hint = QLabel("Synthetic data stream: FPS, exposure, resolution, buffer. No real device.")
        hint.setObjectName("SimulatedLiveHint")
        hint.setToolTip("Synthetic Live: Lissajous (and later e.g. Blitz). Same pull/ring-buffer as real Live.")
        layout.addWidget(hint)
        qd = QLabel("Quick & Dirty – ROI/View preserved. Real streamer later (e.g. Network).")
        qd.setObjectName("SimulatedLiveHint")
        qd.setStyleSheet("color: #666; font-size: 8pt;")
        layout.addWidget(qd)

        row1 = QHBoxLayout()
        self.display = QLabel("Ready.")
        self.display.setObjectName("SimulatedLiveDisplay")
        row1.addWidget(self.display, 1)
        self.spectrum = _SimulatedViz(self)
        row1.addWidget(self.spectrum)
        layout.addLayout(row1)

        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Variant:"))
        self.combo_variant = QComboBox()
        self.combo_variant.addItems(["Lightning", "Lissajous"])
        self.combo_variant.setCurrentIndex(0)
        self.combo_variant.setToolTip("Lightning = tree grows, splits, stops at wall, then afterglow. Lissajous = curve viz.")
        self.combo_variant.currentTextChanged.connect(self._on_variant_changed)
        variant_row.addWidget(self.combo_variant, 1)
        layout.addLayout(variant_row)
        self.spectrum.set_variant(self.combo_variant.currentText())

        # Lightning params: Source Origin (names swapped for 90deg-rotated display), then length/thick/noise/branches/speed
        max_half = 960
        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel("Source Origin:"))
        self.combo_source_origin = QComboBox()
        self.combo_source_origin.addItems(["Top", "Left"])
        self.combo_source_origin.setCurrentIndex(1)
        self.combo_source_origin.setToolTip("Propagation direction (for 90deg-rotated display): Top = grow right, Left = grow down.")
        origin_row.addWidget(self.combo_source_origin, 1)
        layout.addLayout(origin_row)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Length:"))
        self.spin_segment_length = QSpinBox()
        self.spin_segment_length.setRange(3, max_half)
        self.spin_segment_length.setValue(9)
        self.spin_segment_length.setToolTip("Segment length (px). Random variation per segment.")
        seg_row.addWidget(self.spin_segment_length, 1)
        layout.addLayout(seg_row)
        thick_row = QHBoxLayout()
        thick_row.addWidget(QLabel("Thick:"))
        self.spin_thickness = QSpinBox()
        self.spin_thickness.setRange(1, max_half//2)
        self.spin_thickness.setValue(17)
        self.spin_thickness.setToolTip("Segment thickness (px) for glow.")
        thick_row.addWidget(self.spin_thickness, 1)
        layout.addLayout(thick_row)
        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise:"))
        self.spin_noise = QSpinBox()
        self.spin_noise.setRange(0, 50)
        self.spin_noise.setValue(17)
        self.spin_noise.setToolTip("Intensity noise (Rauschen) strength.")
        noise_row.addWidget(self.spin_noise, 1)
        layout.addLayout(noise_row)
        branches_row = QHBoxLayout()
        branches_row.addWidget(QLabel("Branches:"))
        self.spin_branches = QSpinBox()
        self.spin_branches.setRange(1, 42)
        self.spin_branches.setValue(9)
        self.spin_branches.setToolTip("Max number of branches (splits).")
        branches_row.addWidget(self.spin_branches, 1)
        layout.addLayout(branches_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.25, 5.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setDecimals(2)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setToolTip("Animation speed of growth/cycle (independent of FPS).")
        speed_row.addWidget(self.spin_speed, 1)
        layout.addLayout(speed_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 120)
        self.fps_slider.setValue(30)
        self.fps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fps_slider.setTickInterval(30)
        self.fps_slider.valueChanged.connect(
            lambda v: self.fps_label.setText(str(v))
        )
        self.fps_slider.setToolTip("Frames per second (1–120). Viewer throttles redraws at high FPS.")
        fps_row.addWidget(self.fps_slider, 1)
        self.fps_label = QLabel("30")
        self.fps_label.setFixedWidth(28)
        fps_row.addWidget(self.fps_label)
        layout.addLayout(fps_row)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure (ms):"))
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(0.1, 500.0)
        self.spin_exposure.setValue(16.67)
        self.spin_exposure.setDecimals(2)
        self.spin_exposure.setSingleStep(1.0)
        self.spin_exposure.setSuffix(" ms")
        self.spin_exposure.setToolTip("Exposure time in ms (affects brightness in Lissajous).")
        exp_row.addWidget(self.spin_exposure, 1)
        layout.addLayout(exp_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["256x256", "512x512", "768x768", "1920x1080"])
        self.combo_size.setCurrentIndex(1)
        self.combo_size.setToolTip("Output resolution (width x height)")
        res_row.addWidget(self.combo_size, 1)
        layout.addLayout(res_row)

        buf_row = QHBoxLayout()
        buf_row.addWidget(QLabel("Buffer (MB):"))
        self.spin_buffer_mb = QDoubleSpinBox()
        self.spin_buffer_mb.setRange(0.5, 512.0)
        self.spin_buffer_mb.setValue(16.0)
        self.spin_buffer_mb.setDecimals(1)
        self.spin_buffer_mb.setSingleStep(1.0)
        self.spin_buffer_mb.setSuffix(" MB")
        self.spin_buffer_mb.setToolTip("Ring buffer size in MB. Frames derived from resolution and grayscale.")
        buf_row.addWidget(self.spin_buffer_mb, 1)
        self.label_buffer_frames = QLabel("")
        self.label_buffer_frames.setObjectName("SimulatedLiveHint")
        self.label_buffer_frames.setFixedWidth(56)
        buf_row.addWidget(self.label_buffer_frames)
        layout.addLayout(buf_row)

        self.check_grayscale = QCheckBox("Grayscale (plasma colormap)")
        self.check_grayscale.setChecked(True)
        self.check_grayscale.setToolTip("Grayscale = plasma in viewer (brightness 20–200). Uncheck for RGB (color cycle).")
        self.check_grayscale.toggled.connect(self._update_buffer_frames_label)
        self.combo_size.currentTextChanged.connect(self._update_buffer_frames_label)
        self.spin_buffer_mb.valueChanged.connect(self._update_buffer_frames_label)
        layout.addWidget(self.check_grayscale)
        self._update_buffer_frames_label()

        btn_row = QHBoxLayout()
        self.btn_toggle = QPushButton("\u25b6 Play")
        self.btn_toggle.setObjectName("SimulatedLiveBtnPlay")
        self.btn_toggle.setToolTip("Start / Stop stream (toggle)")
        self.btn_toggle.clicked.connect(self._on_toggle)
        btn_row.addWidget(self.btn_toggle)
        self.btn_close = QPushButton("\u2715")
        self.btn_close.setObjectName("SimulatedLiveBtn")
        self.btn_close.setToolTip("Close dialog")
        self.btn_close.setFixedWidth(28)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _get_size(self) -> tuple[int, int]:
        s = self.combo_size.currentText()
        parts = s.split("x")
        return int(parts[0]), int(parts[1])

    def _update_buffer_frames_label(self) -> None:
        w, h = self._get_size()
        grayscale = self.check_grayscale.isChecked()
        mb = self.spin_buffer_mb.value()
        n = buffer_frames_from_mb(w, h, grayscale, mb)
        self.label_buffer_frames.setText(f"~{n} fr")

    def _get_buffer_frames(self) -> int:
        w, h = self._get_size()
        return buffer_frames_from_mb(w, h, self.check_grayscale.isChecked(), self.spin_buffer_mb.value())

    def _on_variant_changed(self) -> None:
        self.spectrum.set_variant(self.combo_variant.currentText())

    def _on_toggle(self) -> None:
        if self._handler and self._handler.is_running:
            self._on_stop()
        else:
            self._on_play()

    def _on_play(self) -> None:
        w, h = self._get_size()
        fps = self.fps_slider.value()
        buffer_frames = self._get_buffer_frames()
        grayscale = self.check_grayscale.isChecked()
        exposure_ms = self.spin_exposure.value()
        variant = self.combo_variant.currentText().lower()
        # UI "Top" -> start left / grow right; "Left" -> start top / grow down (for 90deg-rotated display)
        src = self.combo_source_origin.currentText().lower()
        source_origin = "left" if src == "top" else "top"
        self._handler = SimulatedLiveHandler(
            width=w,
            height=h,
            fps=fps,
            buffer_size=buffer_frames,
            grayscale=grayscale,
            exposure_time_ms=exposure_ms,
            variant=variant,
            lightning_segment_length=self.spin_segment_length.value(),
            lightning_thickness=self.spin_thickness.value(),
            lightning_noise=self.spin_noise.value(),
            lightning_source_origin=source_origin,
            lightning_max_branches=self.spin_branches.value(),
            lightning_speed=self.spin_speed.value(),
        )
        self._handler.stopped.connect(self._on_stream_stopped)
        self._handler.start()
        self._pull_timer = QTimer(self)
        self._pull_timer.timeout.connect(self._pull_and_display)
        self._pull_timer.start(35)
        self.display.setText("Streaming -> BLITZ")
        self.btn_toggle.setText("\u25a0 Stop")
        self.btn_toggle.setEnabled(True)
        self.combo_size.setEnabled(False)
        self.spin_buffer_mb.setEnabled(False)
        self.fps_slider.setEnabled(False)
        self.spin_exposure.setEnabled(False)
        self.combo_variant.setEnabled(False)
        self.combo_source_origin.setEnabled(False)
        self.check_grayscale.setEnabled(False)
        self.spin_segment_length.setEnabled(False)
        self.spin_thickness.setEnabled(False)
        self.spin_noise.setEnabled(False)
        self.spin_branches.setEnabled(False)
        self.spin_speed.setEnabled(False)
        self.spectrum.start()

    def _on_stop(self) -> None:
        if not self._handler:
            return
        self.display.setText("Stopping...")
        self.btn_toggle.setEnabled(False)
        self._handler.stop()
        # Cleanup happens when handler.stopped fires (worker has exited)

    def _on_stream_stopped(self) -> None:
        if self._pull_timer:
            self._pull_timer.stop()
            self._pull_timer = None
        if self._handler:
            if self._on_frame:
                final = self._handler.get_snapshot(max_display_mb=999.0)
                if final is not None:
                    self._on_frame(final)
            try:
                self._handler.stopped.disconnect()
            except TypeError:
                pass
            self._handler = None
        self.display.setText("Stopped.")
        self.btn_toggle.setText("\u25b6 Play")
        self.btn_toggle.setEnabled(True)
        self.combo_size.setEnabled(True)
        self.spin_buffer_mb.setEnabled(True)
        self.fps_slider.setEnabled(True)
        self.spin_exposure.setEnabled(True)
        self.combo_variant.setEnabled(True)
        self.combo_source_origin.setEnabled(True)
        self.check_grayscale.setEnabled(True)
        self.spin_segment_length.setEnabled(True)
        self.spin_thickness.setEnabled(True)
        self.spin_noise.setEnabled(True)
        self.spin_branches.setEnabled(True)
        self.spin_speed.setEnabled(True)
        self.spectrum.stop()

    def _pull_and_display(self) -> None:
        """Observer pulls snapshot from ring buffer (called by QTimer)."""
        if not self._handler or not self._on_frame:
            return
        snapshot = self._handler.get_snapshot()
        if snapshot is not None:
            self._on_frame(snapshot)

    def set_frame_callback(self, cb: Callable[[object], None]) -> None:
        self._on_frame = cb

    def stop_stream(self) -> None:
        """Stop stream and wait for thread. Call before close."""
        if self._handler:
            self._handler.stop()
            self._handler.wait_stopped(3000)
        self._on_stream_stopped()

    def closeEvent(self, event) -> None:
        if self._handler and self._handler.is_running:
            self._handler.stop()
            self._handler.wait_stopped(3000)
            self._on_stream_stopped()
        event.accept()

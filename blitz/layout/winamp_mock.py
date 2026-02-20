"""Mock Live: generates Lissajous viz (Winamp homage), Winamp 5.x style."""

from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

from .. import resources  # noqa: F401
from ..data.live import MockLiveHandler

# Winamp 5.x homage colors (gray/silver main, black viz, green on black)
WINAMP_MAIN_BG = "#3d3d3d"
WINAMP_PANEL_BG = "#2d2d2d"
WINAMP_BORDER = "#5a5a5a"
WINAMP_BORDER_DARK = "#1a1a1a"
WINAMP_FG = "#e0e0e0"
WINAMP_FG_DIM = "#a0a0a0"
WINAMP_GREEN = "#00ff00"      # classic Winamp green
WINAMP_GREEN_DIM = "#00aa00"
WINAMP_BLACK = "#0a0a0a"     # viz background
WINAMP_BTN_UP = "#4a4a4a"
WINAMP_BTN_DOWN = "#252525"
WINAMP_BTN_BORDER = "#6a6a6a"

STYLE_MOCK_LIVE = f"""
    QWidget#MockLiveMain {{
        background-color: {WINAMP_MAIN_BG};
        border: 1px solid {WINAMP_BORDER};
        border-radius: 2px;
    }}
    QLabel#MockLiveTitle {{
        background-color: transparent;
        color: {WINAMP_FG};
        font-weight: bold;
        font-size: 10px;
    }}
    QLabel#MockLiveHint {{
        background-color: transparent;
        color: {WINAMP_FG_DIM};
        font-size: 9px;
    }}
    QLabel#MockLiveDisplay {{
        background-color: {WINAMP_BLACK};
        color: {WINAMP_GREEN};
        font-family: "Consolas", "Lucida Console", monospace;
        font-size: 10px;
        padding: 4px 6px;
        border: 1px solid {WINAMP_BORDER_DARK};
        border-radius: 0;
        min-height: 24px;
    }}
    QPushButton#MockLiveBtn {{
        background-color: {WINAMP_BTN_UP};
        color: {WINAMP_FG};
        border: 1px solid {WINAMP_BTN_BORDER};
        border-radius: 0;
        min-width: 26px;
        max-width: 26px;
        min-height: 20px;
        font-size: 10px;
    }}
    QPushButton#MockLiveBtn:hover {{
        background-color: #555555;
        border-color: #7a7a7a;
    }}
    QPushButton#MockLiveBtn:pressed {{
        background-color: {WINAMP_BTN_DOWN};
        border-color: {WINAMP_BORDER_DARK};
    }}
    QPushButton#MockLiveBtn:disabled {{ color: #606060; }}
    QComboBox, QSlider {{
        background-color: {WINAMP_PANEL_BG};
        color: {WINAMP_FG};
        border: 1px solid {WINAMP_BORDER};
        border-radius: 0;
        min-height: 20px;
    }}
    QComboBox::drop-down {{
        border-left: 1px solid {WINAMP_BORDER};
        background-color: {WINAMP_BTN_UP};
        width: 18px;
    }}
    QCheckBox {{
        color: {WINAMP_FG};
        font-size: 9px;
    }}
    QLabel {{
        color: {WINAMP_FG};
        font-size: 9px;
    }}
    QPushButton#MockLiveBtnPlay {{
        background-color: #2a4a2a;
        color: {WINAMP_GREEN};
        border: 1px solid #3a6a3a;
    }}
    QPushButton#MockLiveBtnPlay:hover {{
        background-color: #3a5a3a;
        border-color: #4a7a4a;
    }}
    QPushButton#MockLiveBtnPlay:pressed {{
        background-color: #1a3a1a;
    }}
"""


class _WinampViz(QWidget):
    """Mini preview: Winamp-style Lissajous (green on black)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setFixedWidth(160)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._t = 0.0

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
        cx, cy = w / 2, h / 2
        margin = 4
        p.fillRect(0, 0, w, h, QColor(10, 10, 10))
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


class WinampMockLiveWidget(QFrame):
    """Mock Live: generates Winamp Lissajous viz, streams to BLITZ viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MockLiveMain")
        self.setStyleSheet(STYLE_MOCK_LIVE)
        self.setWindowTitle("A6000 — it really whips Ollama's ass")
        self.setFixedWidth(280)
        self._handler: Optional[MockLiveHandler] = None
        self._on_frame: Optional[Callable[[object], None]] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("BLITZ Mock Live")
        title.setObjectName("MockLiveTitle")
        layout.addWidget(title)
        hint = QLabel("Play -> Lissajous viz in main viewer")
        hint.setObjectName("MockLiveHint")
        hint.setToolTip("Generated frames, no video file")
        layout.addWidget(hint)

        row1 = QHBoxLayout()
        self.display = QLabel("Ready.")
        self.display.setObjectName("MockLiveDisplay")
        row1.addWidget(self.display, 1)
        self.spectrum = _WinampViz(self)
        row1.addWidget(self.spectrum)
        layout.addLayout(row1)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Size:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["256x256", "512x512", "768x768"])
        self.combo_size.setCurrentIndex(1)
        self.combo_size.setToolTip("Output resolution")
        res_row.addWidget(self.combo_size, 1)
        layout.addLayout(res_row)

        buf_row = QHBoxLayout()
        buf_row.addWidget(QLabel("Buffer:"))
        self.combo_buffer = QComboBox()
        self.combo_buffer.addItems(["16", "32", "64", "128", "256"])
        self.combo_buffer.setCurrentIndex(2)
        self.combo_buffer.setToolTip("Rolling buffer size (frames in timeline)")
        buf_row.addWidget(self.combo_buffer, 1)
        layout.addLayout(buf_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(10)
        self.fps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fps_slider.setTickInterval(10)
        self.fps_slider.valueChanged.connect(
            lambda v: self.fps_label.setText(str(v))
        )
        fps_row.addWidget(self.fps_slider, 1)
        self.fps_label = QLabel("10")
        self.fps_label.setFixedWidth(24)
        fps_row.addWidget(self.fps_label)
        layout.addLayout(fps_row)

        self.check_grayscale = QCheckBox("Grayscale (plasma colormap)")
        self.check_grayscale.setChecked(True)
        self.check_grayscale.setToolTip("Grayscale = plasma in viewer. Uncheck for RGB.")
        layout.addWidget(self.check_grayscale)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("\u25b6")
        self.btn_play.setObjectName("MockLiveBtnPlay")
        self.btn_play.setToolTip("Start Lissajous stream -> BLITZ viewer")
        self.btn_play.clicked.connect(self._on_play)
        btn_row.addWidget(self.btn_play)
        self.btn_stop = QPushButton("\u25a0")
        self.btn_stop.setObjectName("MockLiveBtn")
        self.btn_stop.setToolTip("Stop stream")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

    def _get_size(self) -> tuple[int, int]:
        s = self.combo_size.currentText()
        n = int(s.split("x")[0])
        return n, n

    def _get_buffer_size(self) -> int:
        return int(self.combo_buffer.currentText())

    def _on_play(self) -> None:
        w, h = self._get_size()
        fps = self.fps_slider.value()
        buffer_size = self._get_buffer_size()
        grayscale = self.check_grayscale.isChecked()
        self._handler = MockLiveHandler(
            width=w,
            height=h,
            fps=fps,
            buffer_size=buffer_size,
            grayscale=grayscale,
        )
        self._handler.frame_ready.connect(self._emit_frame)
        self._handler.stopped.connect(self._on_stream_stopped)
        self._handler.start()
        self.display.setText("Streaming -> BLITZ")
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_size.setEnabled(False)
        self.combo_buffer.setEnabled(False)
        self.fps_slider.setEnabled(False)
        self.check_grayscale.setEnabled(False)
        self.spectrum.start()

    def _on_stop(self) -> None:
        if not self._handler:
            return
        self.display.setText("Stopping...")
        self.btn_stop.setEnabled(False)
        self._handler.stop()
        # Cleanup happens when handler.stopped fires (worker has exited)

    def _on_stream_stopped(self) -> None:
        if self._handler:
            try:
                self._handler.frame_ready.disconnect()
                self._handler.stopped.disconnect()
            except TypeError:
                pass
            self._handler = None
        self.display.setText("Stopped.")
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.combo_size.setEnabled(True)
        self.combo_buffer.setEnabled(True)
        self.fps_slider.setEnabled(True)
        self.check_grayscale.setEnabled(True)
        self.spectrum.stop()

    def _emit_frame(self, img) -> None:
        if self._on_frame:
            self._on_frame(img)

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

"""Conway Game of Life streamer UI — sibling to Simulated Live."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.conway import (
    PATTERN_NAMES,
    ConwayLifeHandler,
    buffer_frames_from_mb,
    pattern_preview_grid,
)

_PATTERN_TOOLTIPS = {
    "Random": "Random soup — density controlled by the Density spinner; Seed picks the world.",
    "Glider": "Classic glider — translates diagonally (period 4 on a torus).",
    "Blinker": "Period-2 oscillator — three cells in a line.",
    "Toad": "Period-2 oscillator — two offset rows of three.",
    "Beacon": "Period-2 still-life oscillator — two 2×2 blocks.",
    "R-pentomino": "Methuselah — evolves for many generations from five cells.",
    "Gosper gun": "Gosper glider gun — emits gliders (needs a wide enough grid).",
}

_PREVIEW_PX = 64


def _pattern_pixmap(name: str, px: int = _PREVIEW_PX) -> QPixmap:
    """Chunky Life preview bitmap (nearest-neighbor upscale to px×px)."""
    grid = pattern_preview_grid(name, size=16)
    h, w = grid.shape
    cell = max(1, px // max(h, w))
    img_h, img_w = h * cell, w * cell
    rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[..., :3] = 18
    live = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1).astype(bool)
    rgba[live, 0] = 40
    rgba[live, 1] = 220
    rgba[live, 2] = 80
    qimg = QImage(
        rgba.data,
        img_w,
        img_h,
        img_w * 4,
        QImage.Format.Format_RGBA8888,
    ).copy()
    if img_w != px or img_h != px:
        qimg = qimg.scaled(
            px,
            px,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    return QPixmap.fromImage(qimg)

MAIN_BG = "#3d3d3d"
PANEL_BG = "#2d2d2d"
BORDER = "#5a5a5a"
BORDER_DARK = "#1a1a1a"
FG = "#e0e0e0"
FG_DIM = "#a0a0a0"
GREEN = "#00ff00"
BTN_UP = "#4a4a4a"
BTN_DOWN = "#252525"
BTN_BORDER = "#6a6a6a"

STYLE_CONWAY = f"""
    QWidget#ConwayLifeMain {{
        background-color: {MAIN_BG};
        border: 1px solid {BORDER};
        border-radius: 2px;
    }}
    QLabel#ConwayTitle {{
        background-color: transparent;
        color: {FG};
        font-weight: bold;
        font-size: 10px;
    }}
    QLabel#ConwayHint {{
        background-color: transparent;
        color: {FG_DIM};
        font-size: 9px;
    }}
    QLabel#ConwayDisplay {{
        background-color: #0a0a0a;
        color: {GREEN};
        font-family: "Consolas", "Lucida Console", monospace;
        font-size: 10px;
        padding: 4px 6px;
        border: 1px solid {BORDER_DARK};
        border-radius: 0;
        min-height: 24px;
    }}
    QLabel#ConwayPatternPreview {{
        background-color: #0a0a0a;
        border: 1px solid {BORDER_DARK};
        border-radius: 0;
    }}
    QPushButton#ConwayBtn {{
        background-color: {BTN_UP};
        color: {FG};
        border: 1px solid {BTN_BORDER};
        border-radius: 0;
        min-width: 26px;
        max-width: 26px;
        min-height: 20px;
        font-size: 10px;
    }}
    QPushButton#ConwayBtn:hover {{
        background-color: #555555;
        border-color: #7a7a7a;
    }}
    QPushButton#ConwayBtn:pressed {{
        background-color: {BTN_DOWN};
        border-color: {BORDER_DARK};
    }}
    QPushButton#ConwayBtn:disabled {{ color: #606060; }}
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox {{
        background-color: {PANEL_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 0;
        min-height: 20px;
    }}
    QComboBox::drop-down {{
        border-left: 1px solid {BORDER};
        background-color: {BTN_UP};
        width: 18px;
    }}
    QCheckBox {{
        color: {FG};
        font-size: 9px;
    }}
    QLabel {{
        color: {FG};
        font-size: 9px;
    }}
    QPushButton#ConwayBtnPlay {{
        background-color: #2a4a2a;
        color: {GREEN};
        border: 1px solid #3a6a3a;
    }}
    QPushButton#ConwayBtnPlay:hover {{
        background-color: #3a5a3a;
        border-color: #4a7a4a;
    }}
    QPushButton#ConwayBtnPlay:pressed {{
        background-color: #1a3a1a;
    }}
"""


class ConwayLifeWidget(QFrame):
    """Classic Conway (B3/S23) → ring buffer → BLITZ viewer. Sibling to Synthetic Live."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ConwayLifeMain")
        self.setStyleSheet(STYLE_CONWAY)
        self.setWindowTitle("Game of Life")
        self.setFixedWidth(300)
        self._handler: Optional[ConwayLifeHandler] = None
        self._on_frame: Optional[Callable[[object], None]] = None
        self._pull_timer: Optional[QTimer] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("BLITZ Game of Life")
        title.setObjectName("ConwayTitle")
        layout.addWidget(title)
        hint = QLabel(
            "Classic Conway B3/S23 → BLITZ viewer. Change Speed while playing; "
            "Stop to scrub the timeline."
        )
        hint.setObjectName("ConwayHint")
        hint.setWordWrap(True)
        hint.setToolTip(
            "Sibling of Synthetic Live (not a Lissajous/Lightning variant). "
            "Ring buffer fills while streaming; after Stop you scrub generations."
        )
        layout.addWidget(hint)

        self.display = QLabel("Ready.")
        self.display.setObjectName("ConwayDisplay")
        self.display.setToolTip("Stream status. Frames go into the main BLITZ viewer.")
        layout.addWidget(self.display)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Classic", "Ember"])
        self.combo_mode.setToolTip(
            "Classic: grayscale values 0 (off) and 1 (on) only.\n"
            "Ember: grayscale 0…N with N = Decay — alive = N, trail = N-1…1 "
            "(cellular decay, display-only; ignored by B3/S23)."
        )
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.combo_mode, 1)
        layout.addLayout(mode_row)

        self._ember_row_widget = QWidget()
        ember_row = QHBoxLayout(self._ember_row_widget)
        ember_row.setContentsMargins(0, 0, 0, 0)
        ember_row.addWidget(QLabel("Decay (gens):"))
        self.spin_ember_gens = QSpinBox()
        self.spin_ember_gens.setRange(1, 16)
        self.spin_ember_gens.setValue(3)
        self.spin_ember_gens.setToolTip(
            "Ember only (hidden in Classic). Max gray level N for the decay ladder.\n"
            "Example N=3 → values 0,1,2,3 with alive=3 and trail 2→1→0. "
            "Pin BLITZ LUT to 0…N. Changeable while playing."
        )
        self.spin_ember_gens.valueChanged.connect(self._on_ember_gens_changed)
        ember_row.addWidget(self.spin_ember_gens, 1)
        layout.addWidget(self._ember_row_widget)

        pat_row = QHBoxLayout()
        pat_row.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(list(PATTERN_NAMES))
        self.combo_pattern.setCurrentText("Random")
        self.combo_pattern.setToolTip(
            "Initial seed pattern. Preview is shown large at the bottom of this window. "
            "Non-random patterns are centered; Gosper gun needs a wide enough grid."
        )
        self.combo_pattern.currentTextChanged.connect(self._update_pattern_preview)
        pat_row.addWidget(self.combo_pattern, 1)
        layout.addLayout(pat_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2_147_483_647)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(
            "RNG seed for Random density (and reproducibility). Same seed + size + "
            "density → same starting world."
        )
        seed_row.addWidget(self.spin_seed, 1)
        layout.addLayout(seed_row)

        dens_row = QHBoxLayout()
        dens_row.addWidget(QLabel("Density:"))
        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.05, 0.90)
        self.spin_density.setSingleStep(0.05)
        self.spin_density.setDecimals(2)
        self.spin_density.setValue(0.28)
        self.spin_density.setToolTip(
            "Fill fraction for Random only (~0.28 is a lively mid-density start)."
        )
        dens_row.addWidget(self.spin_density, 1)
        layout.addLayout(dens_row)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid W×H:"))
        self.spin_grid_w = QSpinBox()
        self.spin_grid_w.setRange(16, 512)
        self.spin_grid_w.setValue(64)
        self.spin_grid_w.setToolTip(
            "Grid width in cells. Rectangular grids are first-class (default W64 × H128)."
        )
        self.spin_grid_h = QSpinBox()
        self.spin_grid_h.setRange(16, 512)
        self.spin_grid_h.setValue(128)
        self.spin_grid_h.setToolTip(
            "Grid height in cells. Rectangular grids are first-class (default W64 × H128)."
        )
        grid_row.addWidget(self.spin_grid_w)
        grid_row.addWidget(QLabel("×"))
        grid_row.addWidget(self.spin_grid_h)
        layout.addLayout(grid_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Cell scale:"))
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(1, 16)
        self.spin_scale.setValue(1)
        self.spin_scale.setToolTip(
            "Pixels per cell in the viewer frame (frame size = W·scale × H·scale). "
            "Default 1 = one pixel per cell. Larger = chunkier cells, more RAM."
        )
        scale_row.addWidget(self.spin_scale, 1)
        layout.addLayout(scale_row)

        gps_row = QHBoxLayout()
        gps_row.addWidget(QLabel("Speed:"))
        self.gps_slider = QSlider(Qt.Orientation.Horizontal)
        self.gps_slider.setRange(1, 120)
        self.gps_slider.setValue(15)
        self.gps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gps_slider.setTickInterval(15)
        self.gps_slider.valueChanged.connect(self._on_speed_changed)
        self.gps_slider.setToolTip(
            "Speed in generations per second (1–120). Change anytime — also while "
            "playing. Independent of the ring-buffer size."
        )
        gps_row.addWidget(self.gps_slider, 1)
        self.gps_label = QLabel("15/s")
        self.gps_label.setFixedWidth(36)
        self.gps_label.setToolTip("Current generations per second.")
        gps_row.addWidget(self.gps_label)
        layout.addLayout(gps_row)

        buf_row = QHBoxLayout()
        buf_row.addWidget(QLabel("Buffer (MB):"))
        self.spin_buffer_mb = QDoubleSpinBox()
        self.spin_buffer_mb.setRange(0.5, 512.0)
        self.spin_buffer_mb.setValue(16.0)
        self.spin_buffer_mb.setDecimals(1)
        self.spin_buffer_mb.setSingleStep(1.0)
        self.spin_buffer_mb.setSuffix(" MB")
        self.spin_buffer_mb.setToolTip(
            "Ring buffer RAM. More MB → longer scrubbable history after Stop."
        )
        buf_row.addWidget(self.spin_buffer_mb, 1)
        self.label_buffer_frames = QLabel("")
        self.label_buffer_frames.setObjectName("ConwayHint")
        self.label_buffer_frames.setFixedWidth(56)
        self.label_buffer_frames.setToolTip("Approximate frame count for current size/MB.")
        buf_row.addWidget(self.label_buffer_frames)
        layout.addLayout(buf_row)

        self.check_wrap = QCheckBox("Wrap edges (torus)")
        self.check_wrap.setChecked(True)
        self.check_wrap.setToolTip(
            "On: left↔right and top↔bottom wrap (classic torus).\n"
            "Off: hard edges — useful for school demos of boundary death."
        )
        layout.addWidget(self.check_wrap)

        for w in (
            self.spin_grid_w,
            self.spin_grid_h,
            self.spin_scale,
            self.spin_buffer_mb,
        ):
            w.valueChanged.connect(self._update_buffer_frames_label)
        self._on_mode_changed(self.combo_mode.currentText())
        self._update_buffer_frames_label()

        btn_row = QHBoxLayout()
        self.btn_toggle = QPushButton("\u25b6 Play")
        self.btn_toggle.setObjectName("ConwayBtnPlay")
        self.btn_toggle.setToolTip("Start / Stop stream (toggle)")
        self.btn_toggle.clicked.connect(self._on_toggle)
        btn_row.addWidget(self.btn_toggle)
        self.btn_close = QPushButton("\u2715")
        self.btn_close.setObjectName("ConwayBtn")
        self.btn_close.setToolTip("Close dialog")
        self.btn_close.setFixedWidth(28)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        preview_cap = QLabel("Pattern preview")
        preview_cap.setObjectName("ConwayHint")
        preview_cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(preview_cap)
        self.pattern_preview = QLabel()
        self.pattern_preview.setObjectName("ConwayPatternPreview")
        self.pattern_preview.setFixedSize(_PREVIEW_PX, _PREVIEW_PX)
        self.pattern_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pattern_preview.setScaledContents(False)
        preview_wrap = QHBoxLayout()
        preview_wrap.addStretch(1)
        preview_wrap.addWidget(self.pattern_preview)
        preview_wrap.addStretch(1)
        layout.addLayout(preview_wrap)
        self._update_pattern_preview(self.combo_pattern.currentText())

    def _update_pattern_preview(self, name: str = "") -> None:
        name = (name or self.combo_pattern.currentText()).strip()
        if not name:
            return
        self.pattern_preview.setPixmap(_pattern_pixmap(name, _PREVIEW_PX))
        tip = _PATTERN_TOOLTIPS.get(name, f"Seed pattern: {name}.")
        tip += (
            f"\n\nPreview {_PREVIEW_PX}×{_PREVIEW_PX} px (schematic). "
            "Actual stream uses Grid W×H and Cell scale."
        )
        self.pattern_preview.setToolTip(tip)
        self.combo_pattern.setToolTip(
            f"{tip}\n\nChoose a pattern here; the large preview is at the bottom."
        )

    def _frame_size(self) -> tuple[int, int]:
        return (
            self.spin_grid_w.value() * self.spin_scale.value(),
            self.spin_grid_h.value() * self.spin_scale.value(),
        )

    def _update_buffer_frames_label(self) -> None:
        w, h = self._frame_size()
        n = buffer_frames_from_mb(w, h, True, self.spin_buffer_mb.value())
        self.label_buffer_frames.setText(f"~{n} fr")

    def _get_buffer_frames(self) -> int:
        w, h = self._frame_size()
        return buffer_frames_from_mb(w, h, True, self.spin_buffer_mb.value())

    def _on_mode_changed(self, text: str) -> None:
        ember = text.strip().lower() == "ember"
        self._ember_row_widget.setVisible(ember)
        self.spin_ember_gens.setEnabled(ember)

    def lut_max_level(self) -> int:
        """Suggested BLITZ LUT max: 1 for Classic, Decay N for Ember."""
        if self.combo_mode.currentText().strip().lower() == "ember":
            return int(self.spin_ember_gens.value())
        return 1

    def _on_speed_changed(self, value: int) -> None:
        self.gps_label.setText(f"{value}/s")
        if self._handler is not None and self._handler.is_running:
            self._handler.set_gens_per_sec(float(value))

    def _on_ember_gens_changed(self, value: int) -> None:
        if self._handler is not None and self._handler.is_running:
            self._handler.set_ember_gens(int(value))

    def _set_controls_enabled(self, enabled: bool) -> None:
        # Speed + ember decay stay live-editable while streaming.
        for w in (
            self.combo_mode,
            self.combo_pattern,
            self.spin_seed,
            self.spin_density,
            self.spin_grid_w,
            self.spin_grid_h,
            self.spin_scale,
            self.spin_buffer_mb,
            self.check_wrap,
        ):
            w.setEnabled(enabled)
        if enabled:
            self._on_mode_changed(self.combo_mode.currentText())

    def _on_toggle(self) -> None:
        if self._handler and self._handler.is_running:
            self._on_stop()
        else:
            self._on_play()

    def _on_play(self) -> None:
        fw, fh = self._frame_size()
        self._handler = ConwayLifeHandler(
            grid_width=self.spin_grid_w.value(),
            grid_height=self.spin_grid_h.value(),
            scale=self.spin_scale.value(),
            gens_per_sec=float(self.gps_slider.value()),
            buffer_size=self._get_buffer_frames(),
            wrap=self.check_wrap.isChecked(),
            ember_mode=self.combo_mode.currentText().lower() == "ember",
            ember_gens=self.spin_ember_gens.value(),
            pattern=self.combo_pattern.currentText(),
            seed=self.spin_seed.value(),
            density=self.spin_density.value(),
        )
        self._handler.stopped.connect(self._on_stream_stopped)
        self._handler.start()
        self._pull_timer = QTimer(self)
        self._pull_timer.timeout.connect(self._pull_and_display)
        self._pull_timer.start(35)
        self.display.setText(f"Streaming {fw}x{fh} -> BLITZ")
        self.btn_toggle.setText("\u25a0 Stop")
        self.btn_toggle.setEnabled(True)
        self._set_controls_enabled(False)

    def _on_stop(self) -> None:
        if not self._handler:
            return
        self.display.setText("Stopping...")
        self.btn_toggle.setEnabled(False)
        self._handler.stop()

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
        self._set_controls_enabled(True)

    def _pull_and_display(self) -> None:
        if not self._handler or not self._on_frame:
            return
        snapshot = self._handler.get_snapshot()
        if snapshot is not None:
            self._on_frame(snapshot)

    def set_frame_callback(self, cb: Callable[[object], None]) -> None:
        self._on_frame = cb

    def stop_stream(self) -> None:
        if self._handler:
            self._handler.stop()
            self._handler.wait_stopped(3000)
        self._on_stream_stopped()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._handler and self._handler.is_running:
            self._handler.stop()
            self._handler.wait_stopped(3000)
            self._on_stream_stopped()
        event.accept()

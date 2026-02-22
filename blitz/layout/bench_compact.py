"""Compact bench group: CPU only, shown in LUT panel when checkbox ticked."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import get_plot_bg, PLOT_CPU_COLOR
from .bench_data import BenchData


def _mini_plot(parent: QWidget, color: tuple[int, int, int], h: int = 36) -> tuple[pg.PlotWidget, pg.PlotDataItem]:
    """Very small sparkline for compact panel."""
    pw = pg.PlotWidget(background=get_plot_bg(), parent=parent)
    pw.setFixedSize(72, h)
    pw.hideAxis("left")
    pw.hideAxis("bottom")
    pw.setMouseEnabled(False, False)
    curve = pw.plot(pen=pg.mkPen(color, width=1.2))
    return pw, curve


class BenchCompact(QFrame):
    """CPU-only mini view for LUT panel. Shown when checkbox ticked."""

    def __init__(self, bench_data: BenchData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = bench_data
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)
        self.setMaximumWidth(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._plot_cpu, self._curve_cpu = _mini_plot(self, PLOT_CPU_COLOR)
        self._label_cpu = QLabel("CPU: —")
        self._label_cpu.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self._plot_cpu)
        layout.addWidget(self._label_cpu)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.refresh()

    def refresh(self) -> None:
        """Update CPU label and curve from shared BenchData."""
        d = self._data
        if not d.cpu:
            self._label_cpu.setText("CPU: —")
            self._curve_cpu.setData([], [])
            return
        cpu = d.last_cpu
        peak = max(d.cpu) if d.cpu else cpu
        self._label_cpu.setText(f"CPU: {cpu:.0f}%\nmax. {peak:.0f} %")
        self._label_cpu.setToolTip(
            f"Current: {cpu:.0f}%. Peak in last ~30 s."
        )
        n = len(d.cpu)
        x = np.arange(n)
        self._curve_cpu.setData(x, np.array(d.cpu))

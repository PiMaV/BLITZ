"""Sparklines neben Metriken fuer Bench tab. Uses shared BenchData."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import get_plot_bg, PLOT_CPU_COLOR, PLOT_DISK_COLOR, PLOT_RAM_COLOR
from .bench_data import BenchData


def _tiny_plot(parent: QWidget, color: tuple[int, int, int]) -> tuple[pg.PlotWidget, pg.PlotDataItem]:
    """Sparkline-Plot (140x44). Tokyo Night style."""
    pw = pg.PlotWidget(background=get_plot_bg(), parent=parent)
    pw.setFixedSize(140, 44)
    pw.hideAxis("left")
    pw.hideAxis("bottom")
    pw.setMouseEnabled(False, False)
    curve = pw.plot(pen=pg.mkPen(color, width=1.5))
    return pw, curve


class BenchSparklines(QWidget):
    """CPU, RAM, Disk mit Sparkline neben jeder Metrik. Mirrors shared BenchData."""

    def __init__(self, bench_data: BenchData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = bench_data
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Zeile 1: CPU
        row1 = QWidget()
        row1.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row1.setFixedHeight(48)
        h1 = QHBoxLayout(row1)
        h1.setContentsMargins(0, 0, 0, 0)
        self.label_cpu = QLabel("CPU: —")
        self.label_cpu.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._plot_cpu, self._curve_cpu = _tiny_plot(row1, PLOT_CPU_COLOR)
        h1.addWidget(self.label_cpu)
        h1.addWidget(self._plot_cpu)
        layout.addWidget(row1)

        # Zeile 2: RAM
        row2 = QWidget()
        row2.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row2.setFixedHeight(48)
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        self.label_ram = QLabel("RAM: —")
        self.label_ram.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._plot_ram, self._curve_ram = _tiny_plot(row2, PLOT_RAM_COLOR)
        h2.addWidget(self.label_ram)
        h2.addWidget(self._plot_ram)
        layout.addWidget(row2)

        # Zeile 3: Disk
        row3 = QWidget()
        row3.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row3.setFixedHeight(48)
        h3 = QHBoxLayout(row3)
        h3.setContentsMargins(0, 0, 0, 0)
        self.label_disk = QLabel("Disk: —")
        self.label_disk.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._plot_disk, self._curve_disk = _tiny_plot(row3, PLOT_DISK_COLOR)
        h3.addWidget(self.label_disk)
        h3.addWidget(self._plot_disk)
        layout.addWidget(row3)
        self.refresh_from_data()

    def refresh_from_data(self) -> None:
        """Update labels and curves from shared BenchData."""
        d = self._data
        if not d.cpu:
            self.label_cpu.setText("CPU: —")
            self.label_ram.setText("RAM: —")
            self.label_disk.setText("Disk: —")
            self._curve_cpu.setData([], [])
            self._curve_ram.setData([], [])
            self._curve_disk.setData([], [])
            return
        cpu = d.last_cpu
        peak = max(d.cpu) if d.cpu else cpu
        peak_str = f" (peak {peak:.1f}%)" if peak > cpu and peak > 5 else ""
        self.label_cpu.setText(f"CPU: {cpu:.1f}%{peak_str}")
        self.label_ram.setText(
            f"RAM: Used {d.last_ram_used:.1f} | Free {d.last_ram_free:.1f} GB"
        )
        self.label_disk.setText(
            f"Disk: R {d.last_disk_r:.2f} W {d.last_disk_w:.2f} MB/s"
        )
        n = len(d.cpu)
        x = np.arange(n)
        self._curve_cpu.setData(x, np.array(d.cpu))
        self._curve_ram.setData(x, np.array(d.ram_free))
        disk_combined = np.array(d.disk_r) + np.array(d.disk_w)
        self._curve_disk.setData(x, disk_combined)

"""Sparklines neben Metriken fuer Bench tab."""
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import get_plot_bg, PLOT_CPU_COLOR, PLOT_DISK_COLOR, PLOT_RAM_COLOR


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
    """CPU, RAM, Disk mit Sparkline neben jeder Metrik."""

    MAX_POINTS = 60

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._cpu: deque[float] = deque(maxlen=self.MAX_POINTS)
        self._ram: deque[float] = deque(maxlen=self.MAX_POINTS)  # free GB
        self._disk: deque[float] = deque(maxlen=self.MAX_POINTS)  # read+write MB/s
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

    def add_point(
        self,
        cpu_pct: float,
        ram_used_gb: float,
        ram_free_gb: float,
        disk_read_mbs: float,
        disk_write_mbs: float,
    ) -> None:
        self._cpu.append(cpu_pct)
        peak = max(self._cpu) if self._cpu else cpu_pct
        peak_str = f" (peak {peak:.1f}%)" if peak > cpu_pct and peak > 5 else ""
        self.label_cpu.setText(f"CPU: {cpu_pct:.1f}%{peak_str}")
        self.label_ram.setText(
            f"RAM: Used {ram_used_gb:.1f} | Free {ram_free_gb:.1f} GB"
        )
        self.label_disk.setText(
            f"Disk: R {disk_read_mbs:.2f} W {disk_write_mbs:.2f} MB/s"
        )
        self._ram.append(ram_free_gb)
        self._disk.append(disk_read_mbs + disk_write_mbs)
        n = len(self._cpu)
        x = np.arange(n)
        self._curve_cpu.setData(x, np.array(self._cpu))
        self._curve_ram.setData(x, np.array(self._ram))
        self._curve_disk.setData(x, np.array(self._disk))

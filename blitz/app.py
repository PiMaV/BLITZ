import json
import multiprocessing
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import time as _time

# PyQt6 + pyqtgraph: when a ViewBox is destroyed, forgetView() calls updateAllViewLists()
# on every remaining ViewBox; some menus' QComboBox may already be deleted -> RuntimeError.
# Guard setViewList so teardown does not crash.
try:
    from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
    _orig_set_view_list = ViewBoxMenu.setViewList

    def _safe_set_view_list(self, views):
        try:
            _orig_set_view_list(self, views)
        except RuntimeError as e:
            if "has been deleted" in str(e):
                return
            raise

    ViewBoxMenu.setViewList = _safe_set_view_list
except Exception:  # noqa: S110
    pass

import psutil
from PyQt6.QtCore import QCoreApplication, QEventLoop, Qt, QProcess, QTimer
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,
                              QFrame, QHBoxLayout, QMessageBox, QPushButton,
                              QSlider, QVBoxLayout, QLabel)

from . import resources  # noqa: F401  (registers Qt resources for icon; needed before dialogs)
from . import settings
from .layout.main import MainWindow
from .theme import get_stylesheet, set_theme


def _ask_boot_bench(app: QApplication) -> tuple[bool, float]:
    """Ask: Optimize or default? If flex: show intensity slider. Return (run_bench, intensity 0-1)."""
    dlg = QDialog()
    dlg.setWindowIcon(QIcon(":/icon/blitz.ico"))
    dlg.setWindowTitle("BLITZ Performance Setup")
    dlg.setMinimumWidth(400)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(
        "On first start, BLITZ can optimize load speed for your machine."
    ))
    btn_box = QDialogButtonBox()
    btn_flex = QPushButton("Let's flex (unleash the hardware)")
    btn_default = QPushButton("Default (boring, but functional)")
    btn_flex.setStyleSheet("background-color: #2e7d32; color: white;")
    btn_default.setStyleSheet("background-color: #e65100; color: white;")
    btn_box.addButton(btn_flex, QDialogButtonBox.ButtonRole.AcceptRole)
    btn_box.addButton(btn_default, QDialogButtonBox.ButtonRole.RejectRole)
    layout.addWidget(btn_box)

    result: list[tuple[bool, float] | None] = [None]

    def on_flex():
        # Slider dialog: Quick | Optimum | Ludicrous. Default = middle (0.5).
        slider_dlg = QDialog(dlg)
        slider_dlg.setWindowIcon(QIcon(":/icon/blitz.ico"))
        slider_dlg.setWindowTitle("Bench intensity")
        slider_dlg.setMinimumWidth(420)
        slay = QVBoxLayout(slider_dlg)
        slay.addWidget(QLabel(
            "Choose intensity. Optimum in the middle; left = quick test, right = ludicrous."
        ))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)  # Optimum
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(25)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Quick"))
        hl.addWidget(slider, 1)
        hl.addWidget(QLabel("Ludicrous"))
        slay.addLayout(hl)
        btn_go = QPushButton("Run bench")
        btn_go.setStyleSheet("background-color: #2e7d32; color: white;")
        btn_go.clicked.connect(slider_dlg.accept)
        slay.addWidget(btn_go)
        if slider_dlg.exec():
            intensity = slider.value() / 100.0
            result[0] = (True, intensity)
        dlg.accept()

    def on_default():
        result[0] = (False, 0.5)
        dlg.accept()

    btn_flex.clicked.connect(on_flex)
    btn_default.clicked.connect(on_default)
    dlg.exec()
    r = result[0]
    return (r[0], r[1]) if r is not None else (False, 0.5)


def _benchmark_message(t: float) -> str:
    """Spielige Nachricht je nach Bench-Dauer."""
    style = "<i style='color:#a9b1d6;'>"
    end = "</i>"

    if t <= 90:
        msg = "Watercooled — lift-off confirmed."
    elif t <= 180:
        msg = "Not bad. Net gmotzt isch gnug globt."
    elif t <= 300:
        msg = "Please be patient. Your PC has character."
    else:
        msg = "Oh oh… BLITZ might be a slow ride."

    return f"{style}{msg}{end}"


def _estimate_loading_params() -> tuple[float, int]:
    """Schaetzt max_ram (GB) und load_dialog_mb aus Hardware."""
    vm = psutil.virtual_memory()
    ram_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    max_ram = round(min(8.0, max(1.0, 0.4 * avail_gb)), 1)
    load_mb = min(800, max(200, int(ram_gb * 25)))
    return max_ram, load_mb


def _interpolate_crossover(metrics: list[dict]) -> float | None:
    """Estimate crossover (files) where par becomes faster than seq. Linear interp."""
    for i in range(1, len(metrics)):
        prev, cur = metrics[i - 1], metrics[i]
        if prev["winner"] == "seq" and cur["winner"] == "par":
            d0 = prev["t_seq"] - prev["t_par"]
            d1 = cur["t_seq"] - cur["t_par"]
            if abs(d1 - d0) < 1e-9:
                return float(cur["n"])
            n_cross = prev["n"] - d0 * (cur["n"] - prev["n"]) / (d1 - d0)
            return max(prev["n"], min(cur["n"], n_cross))
    return None


def _show_boot_bench_result(
    app: QApplication, bench_seconds: float, results_path: Path | None = None
) -> None:
    """Final screen: Cores, RAM, Estimation, plot, User soll kurz innehalten."""
    cores = multiprocessing.cpu_count()
    vm = psutil.virtual_memory()
    ram_gb = vm.total / (1024**3)
    files_thresh = settings.get("default/multicore_files_threshold")
    size_thresh = settings.get("default/multicore_size_threshold")
    size_gb = size_thresh / (1024**3)

    max_ram, load_mb = _estimate_loading_params()
    settings.set("default/max_ram", max_ram)
    settings.set("default/load_dialog_mb", load_mb)

    dlg = QDialog()
    dlg.setWindowIcon(QIcon(":/icon/blitz.ico"))
    dlg.setWindowTitle("BLITZ Optimization Result")
    dlg.setMinimumWidth(520)
    dlg.setMinimumHeight(580)
    layout = QVBoxLayout(dlg)
    layout.setSpacing(12)

    layout.addWidget(QLabel("<b style='color:#7aa2f7;'>Hardware</b>"))
    layout.addWidget(QLabel(f"  {cores} cores  |  {ram_gb:.1f} GB RAM"))
    layout.addWidget(QLabel(""))

    layout.addWidget(QLabel("<b style='color:#9ece6a;'>Multicore</b>"))
    files_txt = f"{int(files_thresh)} file{'s' if int(files_thresh) != 1 else ''}"
    layout.addWidget(QLabel(
        f"  Parallel from <b>{files_txt}</b> and <b>{size_gb:.1f} GB</b>"
    ))
    layout.addWidget(QLabel(""))

    layout.addWidget(QLabel("<b style='color:#e0af68;'>Loading defaults</b>"))
    layout.addWidget(QLabel(
        f"  max. RAM: <b>{max_ram} GB</b>  |  dialog above: <b>{load_mb} MB</b>"
    ))
    if max_ram < ram_gb * 0.5:
        layout.addWidget(QLabel(
            "<i style='color:#a9b1d6;'>"
            f"{max_ram:.0f} GB is enough. "
            "More RAM won't fix bad algorithms.</i>"
        ))
    layout.addWidget(QLabel(""))
    # Bench duration prominent in results
    mins = int(bench_seconds // 60)
    secs = bench_seconds - mins * 60
    dura_txt = f"{mins} min {secs:.0f} s" if mins >= 1 else f"{bench_seconds:.1f} s"
    layout.addWidget(QLabel(
        f"<b>Bench: {dura_txt}</b>  —  {_benchmark_message(bench_seconds)}"
    ))
    layout.addWidget(QLabel(""))

    # Plot: seq vs par from boot_bench_results.json
    metrics = []
    path = results_path or Path.cwd() / "boot_bench_results.json"
    try:
        with open(path) as f:
            data = json.load(f)
            metrics = data.get("metrics", [])
    except (OSError, json.JSONDecodeError):
        pass

    if metrics:
        layout.addWidget(QLabel("<b style='color:#7aa2f7;'>Bench result</b>"))
        # Group by config; pick config with earliest par win for plot
        by_config: dict[str, list[dict]] = {}
        for m in metrics:
            c = m.get("config", "default")
            by_config.setdefault(c, []).append(m)
        best_cross, plot_metrics = None, None
        for mlist in by_config.values():
            cross = _interpolate_crossover(mlist)
            if cross is not None and (best_cross is None or cross < best_cross):
                best_cross, plot_metrics = cross, mlist
        if plot_metrics is None:
            plot_metrics = list(by_config.values())[0] if by_config else metrics

        n_arr = np.array([m["n"] for m in plot_metrics])
        t_seq = np.array([m["t_seq"] for m in plot_metrics])
        t_par = np.array([m["t_par"] for m in plot_metrics])
        cross_n = _interpolate_crossover(plot_metrics)
        cfg_label = plot_metrics[0].get("config", "") if plot_metrics else ""

        pw = pg.PlotWidget(background=(40, 40, 48))
        pw.setMinimumHeight(220)
        pw.plot(n_arr, t_seq, pen=pg.mkPen("#9ece6a", width=2), name="seq")
        pw.plot(n_arr, t_par, pen=pg.mkPen("#f7768e", width=2), name="par")
        if cross_n is not None:
            pw.addLine(x=cross_n, pen=pg.mkPen("#7aa2f7", width=1, style=Qt.PenStyle.DashLine))
        pw.setLabel("left", "Time (s)")
        pw.setLabel("bottom", "Files")
        pw.addLegend()
        layout.addWidget(pw)
        txt = f"Crossover ~{int(cross_n)} files"
        if cfg_label:
            txt += f" ({cfg_label})"
        if cross_n is not None:
            layout.addWidget(QLabel(f"<i style='color:#a9b1d6;'>{txt}</i>"))

    layout.addWidget(QLabel("<i>Saved. Adjust in Options -> Loading.</i>"))

    btn = QPushButton("Got it")
    btn.setEnabled(False)
    layout.addWidget(btn)

    def _enable_btn():
        btn.setEnabled(True)

    QTimer.singleShot(2000, _enable_btn)
    btn.clicked.connect(dlg.accept)
    dlg.exec()


def _run_boot_bench_with_feedback(app: QApplication, intensity: float = 0.5) -> None:
    """Boot-bench in subprocess, splash shows live metrics (CPU, disk) + bench log."""
    exe = sys.executable
    root = Path(__file__).resolve().parent.parent
    proc = QProcess()
    proc.setWorkingDirectory(str(root))
    t0 = _time.perf_counter()
    proc.start(exe, ["-m", "blitz.boot_bench", str(round(intensity, 3))])

    # Splash: Custom Widget mit Farben statt QSplashScreen
    splash = QFrame()
    splash.setWindowFlags(
        Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
    )
    splash.setFixedSize(520, 380)
    splash.setStyleSheet(
        "QFrame { background: #2d2e3a; border-radius: 8px; } "
        "font-family: 'Consolas','SF Mono',monospace; font-size: 11px;"
    )
    splash.setFont(QFont("Consolas", 11))
    layout = QVBoxLayout(splash)
    layout.setContentsMargins(16, 16, 16, 16)
    lbl_title = QLabel("BLITZ bench")
    lbl_title.setStyleSheet("color: #7aa2f7; font-weight: bold; font-size: 13px;")
    layout.addWidget(lbl_title)
    lbl_log = QLabel("Starting...")
    lbl_log.setStyleSheet("color: #a9b1d6;")
    lbl_log.setWordWrap(True)
    layout.addWidget(lbl_log)
    layout.addStretch()
    lbl_spark = QLabel("CPU")
    lbl_spark.setStyleSheet("color: #9ece6a; font-weight: bold;")
    layout.addWidget(lbl_spark)
    lbl_disk = QLabel("Disk —")
    lbl_disk.setStyleSheet("color: #f7768e;")
    layout.addWidget(lbl_disk)

    lines: deque[str] = deque(maxlen=7)
    _spinner_idx = [0]
    _cpu_history: list[float] = []
    SPINNER = ("|", "/", "-", "\\")
    SPARK_CHARS = "▁▂▃▄▅▆▇█"
    _disk_prev: list[tuple[int, int, float] | None] = [None]

    def _psutil_line() -> tuple[str, float]:
        cpu = psutil.cpu_percent(interval=None)
        now = psutil.disk_io_counters()
        t = _time.perf_counter()
        if now is None:
            return "—", cpu
        r, w = now.read_bytes, now.write_bytes
        prev = _disk_prev[0]
        _disk_prev[0] = (r, w, t)
        if prev is None:
            return "—", cpu
        dt = t - prev[2]
        if dt <= 0:
            return "—", cpu
        rd = (r - prev[0]) / (1024 * 1024) / dt
        wr = (w - prev[1]) / (1024 * 1024) / dt
        return f"R:{rd:.1f} W:{wr:.1f} MB/s", cpu

    def _update_splash():
        metrics_str, cpu = _psutil_line()
        _cpu_history.append(cpu)
        if len(_cpu_history) > 14:
            _cpu_history.pop(0)
        if _cpu_history:
            mx = max(_cpu_history) or 1
            spark = "CPU " + "".join(
                SPARK_CHARS[min(7, int(v / mx * 7))] for v in _cpu_history[-12:]
            ) + f"  {cpu:.0f}%"
        else:
            spark = f"CPU {cpu:.0f}%"
        spin = SPINNER[_spinner_idx[0] % len(SPINNER)]
        _spinner_idx[0] += 1
        lbl_title.setText(f"BLITZ bench  {spin}")
        lbl_log.setText("\n".join(lines) if lines else "Starting...")
        lbl_spark.setText(spark)
        lbl_disk.setText(metrics_str)
        app.processEvents()

    def _on_psutil_tick():
        if proc.state() == proc.ProcessState.Running:
            _update_splash()

    def on_ready_read():
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if "BLITZ_BENCH: " not in line:
                continue
            raw = line.split("BLITZ_BENCH: ", 1)[-1].strip()
            if raw.startswith("METRIC "):
                parts = raw.split()
                if len(parts) >= 5:
                    n, t_seq, t_par, winner = parts[1], parts[2], parts[3], parts[4]
                    lines.append(f"  {n:>4} files | seq {t_seq}s | par {t_par}s -> {winner}")
            else:
                lines.append(f"  {raw}")
            _update_splash()

    lines.append("Starting...")
    splash.show()
    geo = app.primaryScreen().availableGeometry()
    splash.move(
        geo.x() + (geo.width() - splash.width()) // 2,
        geo.y() + (geo.height() - splash.height()) // 2,
    )
    _update_splash()
    proc.readyReadStandardOutput.connect(on_ready_read)

    timer = QTimer(splash)
    timer.timeout.connect(_on_psutil_tick)
    timer.start(250)

    loop = QEventLoop()
    proc.finished.connect(loop.quit)
    loop.exec()
    timer.stop()
    splash.close()
    bench_sec = _time.perf_counter() - t0
    results_path = root / "boot_bench_results.json"
    _show_boot_bench_result(app, bench_sec, results_path)


def run() -> int:
    multiprocessing.freeze_support()
    pg.setConfigOptions(useNumba=False)
    exit_code = 0
    restart_exit_code = settings.get("app/restart_exit_code")
    app = QApplication(sys.argv)
    theme = settings.get("app/theme")
    set_theme("light" if theme == "light" else "dark")
    app.setStyleSheet(get_stylesheet())

    if not settings.get("default/boot_bench_done"):
        run_bench, intensity = _ask_boot_bench(app)
        if run_bench:
            _run_boot_bench_with_feedback(app, intensity)
        else:
            settings.set("default/boot_bench_done", True)

    while True:
        app = QCoreApplication.instance()
        theme = settings.get("app/theme")
        set_theme("light" if theme == "light" else "dark")
        app.setStyleSheet(get_stylesheet())
        main_window = MainWindow()
        main_window.show()
        exit_code = app.exec()
        if exit_code != restart_exit_code:
            break
        main_window.deleteLater()
    return exit_code

"""Minimal PyQtGraph sidecar: dense cube vs one frame, side by side.

Does not import the BLITZ package. Run from the BLITZ repo root:

    uv run python _aux/benchmarks/sparse_sidecar/app.py
    uv run python _aux/benchmarks/sparse_sidecar/app.py --print-only --zeros 0.99
    uv run python _aux/benchmarks/sparse_sidecar/app.py --npy path/to/stack.npy

One process holds BOTH viewers, so RSS is A+B. RAM of B is the sparse+frame
line, not process RSS. Scrub times are the speed bench.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sparse_cube import SparseCube, apply_floor_abs, as_thw

_AXES = {"t": 0, "y": 1, "x": 2}


def _fmt_mb(n: int | float) -> str:
    return f"{n / (1024 ** 2):.1f} MB"


def _fmt_s(dt: float) -> str:
    if dt < 0.001:
        return f"{dt * 1e6:.0f} µs"
    if dt < 1.0:
        return f"{dt * 1000:.1f} ms"
    return f"{dt:.2f} s"


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def make_synthetic(
    t: int,
    h: int,
    w: int,
    zeros: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dense = rng.random((t, h, w), dtype=np.float32)
    if zeros > 0:
        dense[rng.random((t, h, w)) < zeros] = 0
    return dense


def load_stack(path: Path, floor_abs: float | None) -> np.ndarray:
    arr = as_thw(np.load(path, allow_pickle=False))
    return apply_floor_abs(arr, floor_abs)


def _bare_view() -> pg.ImageView:
    view = pg.ImageView()
    try:
        view.getImageItem().setOpts(axisOrder="row-major")
    except Exception:
        pass
    try:
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.roiPlot.hide()
    except Exception:
        pass
    return view


def _caption(title: str) -> QLabel:
    lbl = QLabel(title)
    font = lbl.font()
    font.setBold(True)
    lbl.setFont(font)
    return lbl


class SparseSidecar(QMainWindow):
    def __init__(
        self, dense: np.ndarray, *, source: str, auto_bench: bool = True
    ) -> None:
        super().__init__()
        self.setWindowTitle("PLAN D sparse sidecar — not BLITZ")
        self._source = source
        self._dense = dense
        self._shape = (int(dense.shape[0]), int(dense.shape[1]), int(dense.shape[2]))
        self._itemsize = int(dense.dtype.itemsize)
        self._dense_nbytes = int(dense.nbytes)
        self._scrubbing = False
        self._bench_a = "—"
        self._bench_b = "—"
        self._pack_s = 0.0

        t0 = time.perf_counter()
        self._cube = SparseCube.from_dense(dense)
        self._pack_s = time.perf_counter() - t0
        self._sparse_nbytes = self._cube.nbytes
        self._nnz = self._cube.nnz

        finite = dense[np.isfinite(dense)]
        if finite.size:
            self._levels = (float(finite.min()), float(finite.max()))
        else:
            self._levels = (0.0, 1.0)

        self._view_a = _bare_view()
        self._view_b = _bare_view()
        self._view_a.setImage(
            dense,
            autoRange=True,
            autoLevels=False,
            levels=self._levels,
            axes=_AXES,
        )
        self._view_b.setImage(
            self._cube.dense_frame(0),
            autoRange=True,
            autoLevels=False,
            levels=self._levels,
        )

        mono = QFont("monospace")
        self._lbl_stats = QLabel()
        self._lbl_stats.setFont(mono)
        self._lbl_stats.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._lbl_note = QLabel()
        self._lbl_note.setWordWrap(True)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, self._shape[0] - 1))
        self._lbl_t = QLabel("0")
        self._slider.valueChanged.connect(self._on_t)

        self._btn_bench = QPushButton("Run scrub bench (A then B)")
        self._btn_bench.clicked.connect(self._run_bench)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("t"))
        t_row.addWidget(self._slider, stretch=1)
        t_row.addWidget(self._lbl_t)
        t_row.addWidget(self._btn_bench)

        col_a = QVBoxLayout()
        col_a.addWidget(_caption("A — ImageView holds (T,H,W)"))
        col_a.addWidget(self._view_a, stretch=1)

        col_b = QVBoxLayout()
        col_b.addWidget(_caption("B — ImageView holds one frame"))
        col_b.addWidget(self._view_b, stretch=1)

        viewers = QHBoxLayout()
        viewers.addLayout(col_a, stretch=1)
        viewers.addLayout(col_b, stretch=1)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(viewers, stretch=1)
        lay.addLayout(t_row)
        lay.addWidget(self._lbl_stats)
        lay.addWidget(self._lbl_note)
        self.setCentralWidget(root)
        self.resize(1280, 720)

        self._rss_timer = QTimer(self)
        self._rss_timer.setInterval(500)
        self._rss_timer.timeout.connect(self._refresh_stats)
        self._rss_timer.start()
        self._refresh_stats()

        if auto_bench:
            QTimer.singleShot(400, self._run_bench)

    def _on_t(self, t: int) -> None:
        self._lbl_t.setText(str(t))
        self._view_a.setCurrentIndex(t)
        self._view_b.setImage(
            self._cube.dense_frame(t),
            autoRange=False,
            autoLevels=False,
            levels=self._levels,
        )

    def _refresh_stats(self) -> None:
        t, h, w = self._shape
        n = t * h * w
        zeros_pct = 100.0 * (1.0 - (self._nnz / n if n else 0.0))
        frame_b = h * w * self._itemsize
        rss = _rss_bytes()
        self._lbl_stats.setText(
            f"{self._source}   {t}×{h}×{w}   zeros {zeros_pct:.1f}%   nnz {self._nnz:,}\n"
            f"A hold   {_fmt_mb(self._dense_nbytes)}  (left ImageView, full cube)\n"
            f"B hold   {_fmt_mb(self._sparse_nbytes)} sparse + {_fmt_mb(frame_b)} frame\n"
            f"pack     {_fmt_s(self._pack_s)}  dense → COO\n"
            f"scrub A  {self._bench_a}\n"
            f"scrub B  {self._bench_b}\n"
            f"RSS      {_fmt_mb(rss)}  (this process = A + B together, not split)"
        )
        self._lbl_note.setText(
            "RAM of B is the 'B hold' line, not RSS. "
            "Pass: B hold ≪ A hold, and scrub B does not explode RSS. "
            "If B hold ≥ A hold (e.g. 20% zeros), sparse is the wrong layout."
        )

    def _scrub_a(self) -> tuple[float, int]:
        n = self._shape[0]
        rss_max = _rss_bytes()
        t0 = time.perf_counter()
        for t in range(n):
            self._view_a.setCurrentIndex(t)
            QApplication.processEvents()
            rss_max = max(rss_max, _rss_bytes())
        return time.perf_counter() - t0, rss_max

    def _scrub_b(self) -> tuple[float, int]:
        n = self._shape[0]
        rss_max = _rss_bytes()
        t0 = time.perf_counter()
        for t in range(n):
            self._view_b.setImage(
                self._cube.dense_frame(t),
                autoRange=False,
                autoLevels=False,
                levels=self._levels,
            )
            QApplication.processEvents()
            rss_max = max(rss_max, _rss_bytes())
        return time.perf_counter() - t0, rss_max

    def _run_bench(self) -> None:
        if self._scrubbing:
            return
        self._scrubbing = True
        self._btn_bench.setEnabled(False)
        n = max(1, self._shape[0])
        dt_a, rss_a = self._scrub_a()
        dt_b, rss_b = self._scrub_b()
        self._slider.blockSignals(True)
        self._slider.setValue(n - 1)
        self._slider.blockSignals(False)
        self._lbl_t.setText(str(n - 1))
        self._bench_a = (
            f"{_fmt_s(dt_a)}   {n / dt_a:.0f} fps   peak RSS {_fmt_mb(rss_a)}"
        )
        self._bench_b = (
            f"{_fmt_s(dt_b)}   {n / dt_b:.0f} fps   peak RSS {_fmt_mb(rss_b)}"
        )
        self._scrubbing = False
        self._btn_bench.setEnabled(True)
        self._refresh_stats()


def _print_only(dense: np.ndarray, source: str) -> int:
    row = _measure(dense)
    t, h, w = row.t, row.h, row.w
    print(f"{source}  {t}x{h}x{w}  zeros={row.zeros_pct:.1f}%")
    print(f"A hold   {_fmt_mb(row.dense_bytes)}")
    print(
        f"B hold   {_fmt_mb(row.sparse_bytes)} sparse + "
        f"{_fmt_mb(row.frame_bytes)} frame  nnz={row.nnz}"
    )
    print(f"pack     {_fmt_s(row.pack_s)}")
    print(
        f"loop A   {_fmt_s(row.dt_a)}  {row.fps_a:.0f} fps  "
        "(dense[t].sum, 32-frame sample, no Qt)"
    )
    print(
        f"loop B   {_fmt_s(row.dt_b)}  {row.fps_b:.0f} fps  "
        "(dense_frame(t).sum, 32-frame sample, no Qt)"
    )
    print(f"RAM      {row.ram_winner} wins  ({row.ram_ratio:.2f}× sparse/dense)")
    print(f"MAX      A {_fmt_s(row.dt_max_a)}  B {_fmt_s(row.dt_max_b)}")
    print(f"MEAN     A {_fmt_s(row.dt_mean_a)}  B {_fmt_s(row.dt_mean_b)}")
    return 0


@dataclass(frozen=True)
class Measure:
    t: int
    h: int
    w: int
    zeros_pct: float
    nnz: int
    dense_bytes: int
    sparse_bytes: int
    frame_bytes: int
    pack_s: float
    dt_a: float
    dt_b: float
    fps_a: float
    fps_b: float
    ram_winner: str
    ram_ratio: float
    dt_max_a: float
    dt_max_b: float
    dt_mean_a: float
    dt_mean_b: float


def _measure(dense: np.ndarray) -> Measure:
    t0 = time.perf_counter()
    cube = SparseCube.from_dense(dense)
    pack_s = time.perf_counter() - t0
    t, h, w = cube.shape
    n = max(1, t)
    frame_b = h * w * cube.values.itemsize
    sample = np.linspace(0, n - 1, num=min(32, n), dtype=int)

    t0 = time.perf_counter()
    acc = 0.0
    for i in sample:
        acc += float(dense[i].sum())
    dt_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc_b = 0.0
    for i in sample:
        acc_b += float(cube.dense_frame(int(i)).sum())
    dt_b = time.perf_counter() - t0
    _ = acc + acc_b
    n_s = len(sample)

    t0 = time.perf_counter()
    dense_max = dense.max(axis=0)
    dt_max_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    sparse_max = cube.reduce_max()
    dt_max_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense_mean = dense.mean(axis=0)
    dt_mean_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    sparse_mean = cube.reduce_mean()
    dt_mean_b = time.perf_counter() - t0
    _ = dense_max.nbytes + sparse_max.nbytes + dense_mean.nbytes + sparse_mean.nbytes

    ratio = cube.nbytes / dense.nbytes if dense.nbytes else 0.0
    return Measure(
        t=t,
        h=h,
        w=w,
        zeros_pct=100.0 * (1.0 - cube.occupancy),
        nnz=cube.nnz,
        dense_bytes=int(dense.nbytes),
        sparse_bytes=cube.nbytes,
        frame_bytes=frame_b,
        pack_s=pack_s,
        dt_a=dt_a,
        dt_b=dt_b,
        fps_a=n_s / dt_a if dt_a else 0.0,
        fps_b=n_s / dt_b if dt_b else 0.0,
        ram_winner="B" if (cube.nbytes + frame_b) < dense.nbytes else "A",
        ram_ratio=ratio,
        dt_max_a=dt_max_a,
        dt_max_b=dt_max_b,
        dt_mean_a=dt_mean_a,
        dt_mean_b=dt_mean_b,
    )


_SWEEP_ZEROS = (0.20, 0.40, 0.60, 0.80, 0.99)
_SWEEP_T = (200, 500)


def _run_sweep(height: int, width: int, seed: int, floor_abs: float | None) -> int:
    """Occupancy × T grid. One cube at a time; this is the epic RAM gate."""
    print(
        f"{'T':>4} {'H':>4} {'W':>4} {'zeros':>6} {'dense':>8} {'sparse':>8} "
        f"{'RAM':>4} {'ratio':>6} {'maxA':>8} {'maxB':>8} {'meanA':>8} {'meanB':>8}",
        flush=True,
    )
    for t in _SWEEP_T:
        for zeros in _SWEEP_ZEROS:
            dense = make_synthetic(t, height, width, zeros, seed)
            dense = apply_floor_abs(dense, floor_abs)
            row = _measure(dense)
            del dense
            gc.collect()
            print(
                f"{row.t:4d} {row.h:4d} {row.w:4d} {row.zeros_pct:5.1f}% "
                f"{_fmt_mb(row.dense_bytes):>8} {_fmt_mb(row.sparse_bytes):>8} "
                f"{row.ram_winner:>4} {row.ram_ratio:5.2f}× "
                f"{_fmt_s(row.dt_max_a):>8} {_fmt_s(row.dt_max_b):>8} "
                f"{_fmt_s(row.dt_mean_a):>8} {_fmt_s(row.dt_mean_b):>8}",
                flush=True,
            )
    print(
        "RAM B wins when sparse+frame < dense. "
        "maxA/maxB and meanA/meanB are full-stack reduces (the speed question). "
        "PyQtGraph scrub is paint-bound and was a tie."
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLAN D sidecar: PyQtGraph full cube vs one frame, side by side",
    )
    p.add_argument(
        "--zeros",
        type=float,
        default=0.99,
        help="Fraction of voxels set to 0 for synthetic data (default 0.99)",
    )
    p.add_argument("--t", type=int, default=200, help="Synthetic frames")
    p.add_argument("--height", type=int, default=512, help="Synthetic height")
    p.add_argument("--width", type=int, default=512, help="Synthetic width")
    p.add_argument("--npy", type=Path, default=None, help="Load a real .npy stack")
    p.add_argument(
        "--floor",
        type=float,
        default=None,
        help="Optional |v| floor before packing (sidecar occupancy experiment only)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Pack and print nbytes + CPU loop times, then exit (no GUI)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Epic grid: zeros 20/40/60/80/99%% × T=200/500 (no GUI)",
    )
    p.add_argument(
        "--no-auto-bench",
        action="store_true",
        help="Do not run the scrub bench on startup",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    zeros = min(1.0, max(0.0, args.zeros))
    if args.sweep:
        return _run_sweep(args.height, args.width, args.seed, args.floor)
    if args.npy is not None:
        dense = load_stack(args.npy, args.floor)
        source = str(args.npy)
    else:
        dense = make_synthetic(args.t, args.height, args.width, zeros, args.seed)
        dense = apply_floor_abs(dense, args.floor)
        source = f"synthetic zeros={zeros:g}"

    if args.print_only:
        return _print_only(dense, source)

    app = QApplication(sys.argv)
    win = SparseSidecar(
        dense, source=source, auto_bench=not args.no_auto_bench
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

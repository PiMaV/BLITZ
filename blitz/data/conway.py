"""Classic Conway Game of Life streamer (sibling to Simulated Live).

Logical grid is uint8 {0,1} for B3/S23. Rendered frames are **grayscale**:

- **Classic:** only ``0`` (off) and ``1`` (on)
- **Ember:** ``0 … N`` where ``N = Decay (gens)`` — alive = N, trail = N-1 … 1
  (display-only; never neighbors)

Ring-buffer pull contract matches SimulatedLiveHandler.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .image import ImageData, MetaData
from .live import buffer_frames_from_mb

Grid = np.ndarray

PATTERN_NAMES = (
    "Random",
    "Glider",
    "Blinker",
    "Toad",
    "Beacon",
    "R-pentomino",
    "Gosper gun",
)

LEVEL_OFF = 0
LEVEL_ON_CLASSIC = 1
MAX_DECAY_GENS = 16


def _moore_neighbors(grid: Grid, *, wrap: bool) -> np.ndarray:
    """Count live Moore neighbors. ``grid`` must be 0/1 uint8 (or bool-like)."""
    g = (grid > 0).astype(np.int16)
    h, w = g.shape
    if wrap:
        return (
            np.roll(g, 1, 0)
            + np.roll(g, -1, 0)
            + np.roll(g, 1, 1)
            + np.roll(g, -1, 1)
            + np.roll(np.roll(g, 1, 0), 1, 1)
            + np.roll(np.roll(g, 1, 0), -1, 1)
            + np.roll(np.roll(g, -1, 0), 1, 1)
            + np.roll(np.roll(g, -1, 0), -1, 1)
        )
    padded = np.pad(g, 1, mode="constant", constant_values=0)
    total = np.zeros((h, w), dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            total += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return total


def step_classic(grid: Grid, *, wrap: bool = True) -> Grid:
    """One B3/S23 generation. Input/output are uint8 {0,1}."""
    alive = grid > 0
    neighbors = _moore_neighbors(grid, wrap=wrap)
    born = (~alive) & (neighbors == 3)
    stayed = alive & ((neighbors == 2) | (neighbors == 3))
    return (born | stayed).astype(np.uint8)


def step_with_ember(
    grid: Grid,
    trail: Grid | None = None,
    *,
    wrap: bool = True,
    decay_gens: int = 3,
) -> tuple[Grid, Grid]:
    """Classic step plus display-only ember trail.

    With ``decay_gens=N``, render uses ``0..N`` (alive = N). ``trail`` holds
    ``N-1 … 1`` after death, then clears. Never counts as neighbors.
    """
    prev = (grid > 0).astype(np.uint8)
    nxt = step_classic(prev, wrap=wrap)
    n = max(1, min(MAX_DECAY_GENS, int(decay_gens)))
    if trail is None:
        aged = np.zeros_like(prev)
    else:
        aged = np.maximum(trail.astype(np.int16) - 1, 0).astype(np.uint8)
    dying = (prev == 1) & (nxt == 0)
    if n > 1:
        aged[dying] = n - 1
    aged[aged < 1] = LEVEL_OFF
    aged[nxt > 0] = LEVEL_OFF
    return nxt, aged


def _place(pattern: np.ndarray, height: int, width: int, row: int, col: int) -> Grid:
    """Stamp a small binary pattern onto an empty grid (clipped to bounds)."""
    grid = np.zeros((height, width), dtype=np.uint8)
    ph, pw = pattern.shape
    r0 = max(0, row)
    c0 = max(0, col)
    r1 = min(height, row + ph)
    c1 = min(width, col + pw)
    pr0 = r0 - row
    pc0 = c0 - col
    pr1 = pr0 + (r1 - r0)
    pc1 = pc0 + (c1 - c0)
    if r1 > r0 and c1 > c0:
        grid[r0:r1, c0:c1] = pattern[pr0:pr1, pc0:pc1]
    return grid


def _center_offset(ph: int, pw: int, height: int, width: int) -> tuple[int, int]:
    return max(0, (height - ph) // 2), max(0, (width - pw) // 2)


def seed_pattern(
    name: str,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    density: float = 0.28,
) -> Grid:
    """Build initial uint8 {0,1} grid. Rectangular grids supported."""
    h = max(8, int(height))
    w = max(8, int(width))
    key = name.strip().lower().replace(" ", "_").replace("-", "_")

    if key == "random":
        dens = float(np.clip(density, 0.01, 0.95))
        return (rng.random((h, w)) < dens).astype(np.uint8)

    if key == "glider":
        pat = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=np.uint8,
        )
        r, c = _center_offset(3, 3, h, w)
        return _place(pat, h, w, r, c)

    if key == "blinker":
        pat = np.array([[1, 1, 1]], dtype=np.uint8)
        r, c = _center_offset(1, 3, h, w)
        return _place(pat, h, w, r, c)

    if key == "toad":
        pat = np.array(
            [[0, 1, 1, 1], [1, 1, 1, 0]],
            dtype=np.uint8,
        )
        r, c = _center_offset(2, 4, h, w)
        return _place(pat, h, w, r, c)

    if key == "beacon":
        pat = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        r, c = _center_offset(4, 4, h, w)
        return _place(pat, h, w, r, c)

    if key in ("r_pentomino", "rpentomino"):
        pat = np.array(
            [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
            dtype=np.uint8,
        )
        r, c = _center_offset(3, 3, h, w)
        return _place(pat, h, w, r, c)

    if key in ("gosper_gun", "gospergun"):
        coords = [
            (0, 24),
            (1, 22), (1, 24),
            (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
            (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
            (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
            (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
            (6, 10), (6, 16), (6, 24),
            (7, 11), (7, 15),
            (8, 12), (8, 13),
        ]
        if h < 12 or w < 40:
            return seed_pattern("Glider", h, w, rng)
        pat = np.zeros((9, 36), dtype=np.uint8)
        for r, c in coords:
            pat[r, c] = 1
        r0, c0 = _center_offset(9, 36, h, w)
        return _place(pat, h, w, r0, c0)

    return seed_pattern("Random", h, w, rng, density=density)


def render_frame(
    alive: Grid,
    scale: int = 1,
    *,
    ember: Grid | None = None,
    ember_mode: bool = False,
    decay_gens: int = 3,
) -> np.ndarray:
    """Rasterize grayscale: Classic ``0/1``; Ember ``0..N`` (alive = N)."""
    scale = max(1, min(32, int(scale)))
    if not ember_mode:
        display = (alive > 0).astype(np.uint8)
    else:
        n = max(1, min(MAX_DECAY_GENS, int(decay_gens)))
        display = np.zeros(alive.shape, dtype=np.uint8)
        if ember is not None:
            display = np.maximum(display, ember.astype(np.uint8, copy=False))
        display[alive > 0] = n
    if scale > 1:
        display = np.repeat(np.repeat(display, scale, axis=0), scale, axis=1)
    return display


def pattern_preview_grid(name: str, size: int = 20) -> Grid:
    """Small HxW binary grid for pattern combo icons."""
    size = max(8, min(48, int(size)))
    rng = np.random.default_rng(0)
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key == "random":
        return (rng.random((size, size)) < 0.32).astype(np.uint8)
    if key in ("gosper_gun", "gospergun"):
        # Wider preview so the gun is recognizable
        return seed_pattern(name, max(12, size // 2), size, rng)
    return seed_pattern(name, size, size, rng)


def _frames_to_imagedata(frames: np.ndarray) -> ImageData:
    arr = np.asarray(frames, dtype=np.uint8)
    t = arr.shape[0]
    h, w = arr.shape[1], arr.shape[2]
    meta = MetaData(
        file_name="conway_life",
        file_size_MB=0.0,
        size=(w, h),
        dtype=np.uint8,
        bit_depth=8,
        color_model="grayscale",
    )
    return ImageData(image=arr, metadata=[meta] * t)


class _ConwayWorker(QObject):
    """Producer: advance generations into the handler ring buffer."""

    stopped = pyqtSignal()

    def __init__(
        self,
        grid_w: int,
        grid_h: int,
        scale: int,
        gens_per_sec: float,
        buffer_size: int,
        wrap: bool,
        ember_mode: bool,
        ember_gens: int,
        pattern: str,
        seed: int,
        density: float,
        handler: "ConwayLifeHandler",
    ):
        super().__init__()
        self._gw = max(8, grid_w)
        self._gh = max(8, grid_h)
        self._scale = max(1, min(32, scale))
        self._gps = max(1.0, min(120.0, gens_per_sec))
        self._buffer_size = max(1, min(4096, buffer_size))
        self._wrap = wrap
        self._ember_mode = ember_mode
        self._ember_gens = max(1, min(MAX_DECAY_GENS, int(ember_gens)))
        self._pattern = pattern
        self._seed = int(seed) & 0x7FFFFFFF
        self._density = density
        self._handler = handler
        self._running = True

    def run(self) -> None:
        rng = np.random.default_rng(self._seed)
        grid = seed_pattern(
            self._pattern, self._gh, self._gw, rng, density=self._density
        )
        ember = np.zeros_like(grid)
        fw = self._gw * self._scale
        fh = self._gh * self._scale
        log(
            f"[LIFE] Conway: grid {self._gw}x{self._gh} scale={self._scale} "
            f"-> {fw}x{fh} @ {self._gps:.0f} gen/s, pattern={self._pattern}, "
            f"seed={self._seed}, wrap={self._wrap}, ember={self._ember_mode}, "
            f"decay={self._ember_gens}, "
            f"levels={'0..N' if self._ember_mode else '0/1'}, buffer={self._buffer_size}"
        )
        while self._running:
            gps = max(1.0, min(120.0, float(self._gps)))
            gens = max(1, min(MAX_DECAY_GENS, int(self._ember_gens)))
            frame = render_frame(
                grid,
                self._scale,
                ember=ember if self._ember_mode else None,
                ember_mode=self._ember_mode,
                decay_gens=gens,
            )
            self._handler._append_frame(frame)
            if self._ember_mode:
                grid, ember = step_with_ember(
                    grid, ember, wrap=self._wrap, decay_gens=gens
                )
            else:
                grid = step_classic(grid, wrap=self._wrap)
                ember = np.zeros_like(grid)
            QThread.msleep(max(1, int(1000.0 / gps)))
        self.stopped.emit()

    def stop(self) -> None:
        self._running = False


class ConwayLifeHandler(QObject):
    """Ring-buffer Conway source (always grayscale discrete 0..10)."""

    stopped = pyqtSignal()

    def __init__(
        self,
        grid_width: int = 64,
        grid_height: int = 128,
        scale: int = 1,
        gens_per_sec: float = 15.0,
        buffer_size: int = 32,
        wrap: bool = True,
        ember_mode: bool = False,
        ember_gens: int = 3,
        pattern: str = "Random",
        seed: int = 42,
        density: float = 0.28,
    ):
        super().__init__()
        self._grid_w = max(8, grid_width)
        self._grid_h = max(8, grid_height)
        self._scale = max(1, min(32, scale))
        self._width = self._grid_w * self._scale
        self._height = self._grid_h * self._scale
        self._gps = gens_per_sec
        self._buffer_size = buffer_size
        self._wrap = wrap
        self._ember_mode = ember_mode
        self._ember_gens = max(1, min(MAX_DECAY_GENS, int(ember_gens)))
        self._pattern = pattern
        self._seed = seed
        self._density = density
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ConwayWorker] = None

    def _append_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(frame.copy())
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

    def get_snapshot(self, max_display_mb: float = 50.0) -> Optional[ImageData]:
        bytes_per_frame = self._width * self._height  # grayscale uint8
        if bytes_per_frame <= 0:
            return None
        max_frames = max(1, int((max_display_mb * 1024.0 * 1024.0) / bytes_per_frame))
        with self._lock:
            if not self._buffer:
                return None
            frames_refs = list(self._buffer[-max_frames:])
        return _frames_to_imagedata(np.stack(frames_refs))

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._buffer = []
        self._thread = QThread()
        self._worker = _ConwayWorker(
            self._grid_w,
            self._grid_h,
            self._scale,
            self._gps,
            self._buffer_size,
            self._wrap,
            self._ember_mode,
            self._ember_gens,
            self._pattern,
            self._seed,
            self._density,
            self,
        )
        self._worker.moveToThread(self._thread)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def set_gens_per_sec(self, gps: float) -> None:
        self._gps = max(1.0, min(120.0, float(gps)))
        if self._worker is not None:
            self._worker._gps = self._gps

    def set_ember_gens(self, gens: int) -> None:
        self._ember_gens = max(1, min(MAX_DECAY_GENS, int(gens)))
        if self._worker is not None:
            self._worker._ember_gens = self._ember_gens

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()

    def wait_stopped(self, timeout_ms: int = 3000) -> bool:
        from PyQt6.QtCore import QCoreApplication, QElapsedTimer

        if not self._thread or not self._thread.isRunning():
            return True
        timer = QElapsedTimer()
        timer.start()
        while self._thread.isRunning() and timer.elapsed() < timeout_ms:
            QCoreApplication.processEvents()
            self._thread.wait(20)
        return not self._thread.isRunning()

    def _on_worker_stopped(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self.stopped.emit()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()


__all__ = [
    "LEVEL_OFF",
    "LEVEL_ON_CLASSIC",
    "MAX_DECAY_GENS",
    "PATTERN_NAMES",
    "ConwayLifeHandler",
    "buffer_frames_from_mb",
    "pattern_preview_grid",
    "render_frame",
    "seed_pattern",
    "step_classic",
    "step_with_ember",
]

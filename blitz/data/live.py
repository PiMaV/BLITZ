"""LiveView handler: Mock mode - Lissajous and Lightning visualizers.

Architecture: Producer (worker) grabs as fast as possible into a ring buffer.
Observer (BLITZ) pulls snapshots via get_snapshot() on a timer. No push of full
buffer every frame -> avoids UI freeze at high FPS. Same contract for future LiveCam.
"""

import colorsys
import random
import threading
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..tools import log
from .image import ImageData, MetaData


def _lissajous_curve_points(t: float, width: int, height: int, n_pts: int = 400) -> np.ndarray:
    """Single Lissajous curve (one per frame). Returns (N+1, 2) int32 array."""
    w, h = width, height
    cx, cy = w / 2.0, h / 2.0
    margin = min(w, h) // 8
    rx = (w - 2 * margin) / 2.0
    ry = (h - 2 * margin) / 2.0
    pts = []
    for i in range(n_pts + 1):
        s = i / n_pts
        x = cx + rx * (
            np.sin(3 * s * 2 * np.pi + t) * 0.4
            + np.sin(s * 2 * np.pi + t * 2) * 0.6
        )
        y = cy + ry * (
            np.sin(5 * s * 2 * np.pi + t * 1.3) * 0.3
            + np.cos(s * 2 * np.pi * 2 + t * 1.4) * 0.7
        )
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)


def _hue_to_bgr(hue: float) -> tuple[int, int, int]:
    """Neon color: full saturation and value. hue in [0, 1). Returns (B, G, R)."""
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _lissajous_frame(
    t: float, width: int, height: int, grayscale: bool, exposure_time_ms: float = 16.0
) -> np.ndarray:
    """One frame: black background, single Lissajous. exposure_time_ms scales brightness (longer = brighter)."""
    w, h = width, height
    line_thick = max(1, min(w, h) // 256)
    pts_arr = _lissajous_curve_points(t, w, h)
    # Exposure: longer -> brighter (e.g. 1 ms -> dim, 50 ms -> bright). Scale in [0.3, 1.5].
    exposure_scale = min(1.5, max(0.3, exposure_time_ms / 20.0))
    if grayscale:
        gray = int(110 + 90 * np.sin(t * 0.5))
        gray = max(20, min(255, int(gray * exposure_scale)))
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(out, [pts_arr], isClosed=False, color=gray, thickness=line_thick)
    else:
        hue = (t * 0.12) % 1.0
        color_bgr = _hue_to_bgr(hue)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.polylines(out, [pts_arr], isClosed=False, color=color_bgr, thickness=line_thick)
        # Scale line brightness by exposure (background stays 0)
        mask = out.any(axis=2)
        out[mask] = np.clip(
            out[mask].astype(np.float64) * exposure_scale, 0, 255
        ).astype(np.uint8)
    return out


# Lightning: cycle; growth stops when any branch touches side wall, then afterglow
_LIGHTNING_CYCLE = 2.5


def _lightning_build_tree(
    width: int,
    height: int,
    seed: int,
    segment_length: float,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    """Build lightning tree: start at top, add segments. Length capped at half image; each segment length has random variation. After 1-5 segments may split. Stop when any tip touches left/right wall."""
    rng = random.Random(seed)
    w, h = width, height
    margin = max(2, w // 20)
    top_y = int(h * 0.12)
    cx = w // 2
    max_seg = max(2, min(w, h) // 2)
    base_seg_len = max(2, min(max_seg, int(segment_length)))
    branches: list[list[tuple[int, int]]] = []
    reveal: list[tuple[int, int]] = []
    # Start one branch from center top
    x = cx + rng.randint(-w // 12, w // 12)
    x = max(margin, min(w - margin, x))
    branches.append([(x, top_y)])
    reveal.append((0, 1))
    # Frontier: (branch_idx, segs_until_can_split)
    next_split = rng.randint(1, 5)
    active = [0]
    segs_since_split = [0]
    wall_hit = False
    while active and not wall_hit:
        bid = active[rng.randint(0, len(active) - 1)]
        pts = branches[bid]
        px, py = pts[-1]
        # Segment length: random variation so not all equal
        this_len = base_seg_len * rng.uniform(0.45, 1.35)
        this_len = max(2, int(this_len))
        # Direction: any (360), light downward bias
        angle = rng.uniform(0, 2 * np.pi)
        bias = rng.uniform(0.15, 0.45)
        angle = angle * (1 - bias) + (-np.pi / 2) * bias
        dx = int(np.cos(angle) * this_len)
        dy = int(np.sin(angle) * this_len)
        if abs(dx) < 1 and abs(dy) < 1:
            dx = 1 if rng.random() > 0.5 else -1
            dy = this_len
        nx, ny = px + dx, py + dy
        if nx <= margin or nx >= w - margin:
            wall_hit = True
            break
        nx = max(margin, min(w - margin, nx))
        pts.append((nx, ny))
        si = active.index(bid)
        segs_since_split[si] = segs_since_split[si] + 1
        reveal.append((bid, len(pts)))
        # Split after 1-5 segments on this branch?
        if segs_since_split[si] >= next_split and len(branches) < 6:
            segs_since_split[si] = 0
            next_split = rng.randint(1, 5)
            n_new = rng.randint(1, 2)
            bx, by = pts[-1]
            for _ in range(n_new):
                branches.append([(bx, by)])
                new_bid = len(branches) - 1
                active.append(new_bid)
                segs_since_split.append(0)
                reveal.append((new_bid, 1))
    return branches, reveal


def _draw_segment_radial(
    out: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    thickness: int,
    intensity: float,
    grayscale: bool,
    noise_sigma: float,
    rng: random.Random,
) -> None:
    """Draw one segment with radial falloff (outer dim, inner bright) and light noise on intensity."""
    th = max(1, thickness)
    # Outer layer (softer), then inner (bright) for radial falloff
    for layer, (dth, mult) in enumerate([(max(1, th + 1), 0.35), (th, 1.0)]):
        fac = intensity * mult * (1.0 + rng.uniform(-noise_sigma, noise_sigma))
        fac = max(0.03, min(1.0, fac))
        if grayscale:
            val = int(255 * fac)
            val = max(0, min(255, val))
            if val > 0:
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), val, dth)
        else:
            b, g, r = int(255 * fac), int(180 * fac), int(80 * fac)
            if max(b, g, r) > 0:
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (b, g, r), dth)


def _lightning_frame(
    t: float,
    width: int,
    height: int,
    grayscale: bool,
    exposure_time_ms: float = 16.0,
    segment_length: int = 10,
    segment_thickness: int = 3,
    noise_sigma: int = 10,
) -> np.ndarray:
    """Lightning: tree grows (splits after 1-5 segs), stops when tip touches side wall; then afterglow. Segments drawn with radial falloff + noise."""
    w, h = width, height
    cycle_pos = (t % _LIGHTNING_CYCLE) / _LIGHTNING_CYCLE
    exposure_scale = min(1.5, max(0.3, exposure_time_ms / 20.0))
    seed = int(t // _LIGHTNING_CYCLE) & 0x7FFFFFFF
    thickness = max(1, min(24, segment_thickness))
    noise_sigma_n = max(0, min(50, noise_sigma)) / 100.0

    if grayscale:
        out = np.zeros((h, w), dtype=np.uint8)
    else:
        out = np.zeros((h, w, 3), dtype=np.uint8)

    # Phases: 0-0.35 grow; 0.35-0.42 frozen; 0.42-0.58 afterglow 1; 0.58-0.72 afterglow 2; 0.72-0.86 afterglow 3; 0.86-1.0 black
    grow_done = cycle_pos >= 0.35
    decay = 1.0
    if 0.42 <= cycle_pos < 0.58:
        decay = 1.0 - (cycle_pos - 0.42) / 0.16
    elif 0.58 <= cycle_pos < 0.72:
        decay = 0.7 - (cycle_pos - 0.58) / 0.14 * 0.45
    elif 0.72 <= cycle_pos < 0.86:
        decay = 0.25 - (cycle_pos - 0.72) / 0.14 * 0.25

    if cycle_pos < 0.86:
        branches, reveal = _lightning_build_tree(w, h, seed, segment_length)
        n_steps = len(reveal)
        if n_steps == 0:
            pass
        else:
            if grow_done:
                progress = 1.0
            else:
                progress = (cycle_pos - 0.02) / 0.33
                progress = max(0, min(1, progress))
            step_cut = int(progress * n_steps)
            step_cut = max(0, min(n_steps, step_cut))
            visible_len: dict[int, int] = {}
            for i in range(step_cut):
                bid, pc = reveal[i]
                visible_len[bid] = max(visible_len.get(bid, 0), pc)
            rng_draw = random.Random((seed + int(t * 50)) & 0x7FFFFFFF)
            base_int = 0.85 * exposure_scale * decay
            for bid, pts in enumerate(branches):
                n_vis = visible_len.get(bid, 0)
                if n_vis < 2:
                    continue
                # Intensity depends on previous segment (chain)
                seg_int = base_int
                for i in range(1, n_vis):
                    p0, p1 = pts[i - 1], pts[i]
                    seg_int = seg_int * 0.88 + rng_draw.uniform(0, 0.08)
                    seg_int = max(0.05, min(1.0, seg_int))
                    _draw_segment_radial(
                        out, p0, p1, thickness, seg_int, grayscale, noise_sigma_n, rng_draw
                    )
            if not grayscale:
                for c in range(3):
                    out[:, :, c] = np.clip(
                        out[:, :, c].astype(np.float64), 0, 255
                    ).astype(np.uint8)

    # Global intensity noise (Rauschen)
    noise_seed = (seed + int(t * 100)) & 0x7FFFFFFF
    sigma = max(0, min(25, noise_sigma)) * 0.5
    rng_np = np.random.default_rng(noise_seed)
    if grayscale:
        noise = rng_np.normal(0, sigma, (h, w)).astype(np.float64)
    else:
        noise = rng_np.normal(0, sigma * 0.8, (h, w, 3)).astype(np.float64)
    out = np.clip(out.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return out


def _frames_to_imagedata(frames: np.ndarray, grayscale: bool) -> ImageData:
    """Wrap frames (T,H,W) or (T,H,W,3) as ImageData. ensure_4d in ImageData adds C if needed."""
    arr = np.asarray(frames, dtype=np.uint8)
    t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    meta = MetaData(
        file_name="lissajous_viz",
        file_size_MB=0.0,
        size=(w, h),
        dtype=np.uint8,
        bit_depth=8,
        color_model="grayscale" if grayscale else "rgb",
    )
    return ImageData(image=arr, metadata=[meta] * t)


def _mock_frame(
    variant: str,
    t: float,
    width: int,
    height: int,
    grayscale: bool,
    exposure_time_ms: float,
    lightning_segment_length: int = 10,
    lightning_thickness: int = 3,
    lightning_noise: int = 10,
) -> np.ndarray:
    """Dispatch to the correct frame generator by variant."""
    if variant == "lightning":
        return _lightning_frame(
            t,
            width,
            height,
            grayscale,
            exposure_time_ms,
            segment_length=lightning_segment_length,
            segment_thickness=lightning_thickness,
            noise_sigma=lightning_noise,
        )
    return _lissajous_frame(t, width, height, grayscale, exposure_time_ms)


class _LissajousWorker(QObject):
    """Producer: grab frames as fast as possible, write into handler's ring buffer only."""

    stopped = pyqtSignal()

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        buffer_size: int,
        grayscale: bool,
        exposure_time_ms: float,
        variant: str,
        handler: "MockLiveHandler",
        lightning_segment_length: int = 10,
        lightning_thickness: int = 3,
        lightning_noise: int = 10,
    ):
        super().__init__()
        self._w = max(64, width)
        self._h = max(64, height)
        self._fps = max(1.0, min(120.0, fps))
        self._buffer_size = max(1, min(4096, buffer_size))
        self._grayscale = grayscale
        self._exposure_time_ms = max(0.1, min(500.0, exposure_time_ms))
        self._variant = variant
        self._handler = handler
        self._lightning_segment_length = lightning_segment_length
        self._lightning_thickness = lightning_thickness
        self._lightning_noise = lightning_noise
        self._running = True
        self._t = 0.0

    def run(self) -> None:
        interval_ms = max(1, int(1000.0 / self._fps))
        log(
            f"[LIVE] Mock ({self._variant}): {self._w}x{self._h} @ {self._fps:.0f} FPS, "
            f"exposure={self._exposure_time_ms:.1f} ms, buffer={self._buffer_size}, "
            f"gray={self._grayscale} (ring buffer, observer pulls)"
        )
        while self._running:
            frame = _mock_frame(
                self._variant,
                self._t,
                self._w,
                self._h,
                self._grayscale,
                self._exposure_time_ms,
                lightning_segment_length=self._lightning_segment_length,
                lightning_thickness=self._lightning_thickness,
                lightning_noise=self._lightning_noise,
            )
            self._t += 0.08
            self._handler._append_frame(frame)
            QThread.msleep(interval_ms)
        self.stopped.emit()

    def stop(self) -> None:
        self._running = False


def buffer_frames_from_mb(
    width: int, height: int, grayscale: bool, buffer_mb: float, max_frames: int = 4096
) -> int:
    """Compute ring-buffer frame count from buffer size in MB. 8-bit: 1 BPP grayscale, 3 BPP RGB."""
    bytes_per_pixel = 1 if grayscale else 3
    bytes_per_frame = width * height * bytes_per_pixel
    if bytes_per_frame <= 0:
        return 1
    frames = int((buffer_mb * 1024.0 * 1024.0) / bytes_per_frame)
    return max(1, min(max_frames, frames))


class MockLiveHandler(QObject):
    """Ring-buffer live source. Producer writes; observer pulls via get_snapshot().

    Same contract for future LiveCam: grab as fast as possible into ring buffer;
    BLITZ (observer) pulls on a timer, no push of full buffer every frame.

    Parameters: FPS (1–120), exposure_time_ms, resolution (width x height),
    buffer_size (frames; use buffer_frames_from_mb() for MB-based sizing), variant (e.g. 'lissajous').
    """

    stopped = pyqtSignal()

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: float = 30.0,
        buffer_size: int = 32,
        grayscale: bool = True,
        exposure_time_ms: float = 16.67,
        variant: str = "lightning",
        lightning_segment_length: int = 10,
        lightning_thickness: int = 3,
        lightning_noise: int = 10,
    ):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._buffer_size = buffer_size
        self._grayscale = grayscale
        self._exposure_time_ms = exposure_time_ms
        self._variant = variant
        self._lightning_segment_length = lightning_segment_length
        self._lightning_thickness = lightning_thickness
        self._lightning_noise = lightning_noise
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._thread: Optional[QThread] = None
        self._worker: Optional[_LissajousWorker] = None

    def _append_frame(self, frame: np.ndarray) -> None:
        """Called by worker only. Thread-safe append to ring buffer."""
        with self._lock:
            self._buffer.append(frame.copy())
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

    def get_snapshot(self, max_display_mb: float = 50.0) -> Optional[ImageData]:
        """Observer pulls: last N frames (timeline order), capped so we don't send huge buffers to the UI.

        Ring buffer keeps all frames; only the portion passed to the viewer is limited by max_display_mb.
        Lock is held only to copy frame references; np.stack runs outside the lock so the worker is not blocked.
        """
        bytes_per_frame = self._width * self._height * (1 if self._grayscale else 3)
        if bytes_per_frame <= 0:
            return None
        max_frames = max(1, int((max_display_mb * 1024.0 * 1024.0) / bytes_per_frame))
        with self._lock:
            if not self._buffer:
                return None
            frames_refs = list(self._buffer[-max_frames:])
        arr = np.stack(frames_refs)
        return _frames_to_imagedata(arr, self._grayscale)

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._buffer = []
        self._thread = QThread()
        self._worker = _LissajousWorker(
            self._width,
            self._height,
            self._fps,
            self._buffer_size,
            self._grayscale,
            self._exposure_time_ms,
            self._variant,
            self,
            lightning_segment_length=self._lightning_segment_length,
            lightning_thickness=self._lightning_thickness,
            lightning_noise=self._lightning_noise,
        )
        self._worker.moveToThread(self._thread)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

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

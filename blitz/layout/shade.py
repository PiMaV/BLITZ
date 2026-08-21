"""View-only hillshade overlay. Analysis buffer stays original height."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel

from ..data.hillshade import (
    AZIMUTH_CACHE_STEP_DEG,
    azimuth_atlas_nbytes,
    azimuth_atlas_peak_nbytes,
    azimuth_cache_bins,
    azimuth_cache_order,
    calculate_hillshade,
    clamp_azimuth_cache_step,
    scaled_height_gradients,
    shade_from_gradients,
    shade_to_uint8,
    snap_azimuth_deg,
)
from ..tools import format_size_mb, get_available_ram
from .viewer import ImageViewer

_RAM_BLOCK_FRAC = 0.90
_RAM_WARN_FRAC = 0.70


class _ShadeAtlasWorker(QThread):
    """Build a uint8 azimuth atlas off the UI thread (gradients once, then n·l)."""

    bin_ready = pyqtSignal(int, int, object)  # generation, azimuth, uint8 array
    failed = pyqtSignal(int, str)
    finished_ok = pyqtSignal(int)

    def __init__(
        self,
        frame: np.ndarray,
        elevation_deg: float,
        z_factor: float,
        start_azimuth: float,
        generation: int,
        step_deg: int = AZIMUTH_CACHE_STEP_DEG,
    ) -> None:
        super().__init__()
        self._frame = frame
        self._elevation = float(elevation_deg)
        self._z_factor = float(z_factor)
        self._start_azimuth = float(start_azimuth)
        self._generation = int(generation)
        self._step_deg = int(step_deg)

    def run(self) -> None:
        gen = self._generation
        try:
            dx, dy = scaled_height_gradients(self._frame, self._z_factor)
            for az in azimuth_cache_order(self._start_azimuth, self._step_deg):
                if self.isInterruptionRequested():
                    return
                shade = shade_from_gradients(dx, dy, az, self._elevation)
                self.bin_ready.emit(gen, int(az), shade_to_uint8(shade))
        except Exception as e:
            if not self.isInterruptionRequested():
                self.failed.emit(gen, str(e))
            return
        if not self.isInterruptionRequested():
            self.finished_ok.emit(gen)


class ShadeAdapter:
    """Preview hillshade on a separate ImageItem; never replaces viewer.image."""

    def __init__(
        self,
        viewer: ImageViewer,
        status_label: Optional[QLabel] = None,
        ram_label: Optional[QLabel] = None,
        on_precache_off: Optional[Callable[[], None]] = None,
    ) -> None:
        self.viewer = viewer
        self._status = status_label
        self._ram_label = ram_label
        self._on_precache_off = on_precache_off
        self._preview = False
        self._precached = False
        self._azimuth = 315.0
        self._elevation = 45.0
        self._z_factor = 1.0
        self._step_deg = AZIMUTH_CACHE_STEP_DEG
        self._atlas: dict[int, np.ndarray] = {}
        self._generation = 0
        self._worker: Optional[_ShadeAtlasWorker] = None
        self._zombies: list[QThread] = []

        self._item = pg.ImageItem()
        # Just above the base ImageItem (z≈0); crosshair/ROI sit much higher.
        self._item.setZValue(0.1)
        self._item.setVisible(False)
        # Match main image axis order so overlay aligns
        try:
            self._item.setOpts(axisOrder=viewer.imageItem.axisOrder)
        except Exception:
            pass
        viewer.view.addItem(self._item)

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._on_timer)

        viewer.timeLine.sigPositionChanged.connect(self._schedule)
        viewer.image_changed.connect(self._schedule)
        viewer.image_size_changed.connect(self._on_size_changed)
        viewer.destroyed.connect(self._on_viewer_destroyed)
        self._refresh_ram_label()

    @property
    def is_precached(self) -> bool:
        return self._precached

    @property
    def cache_step_deg(self) -> int:
        return self._step_deg

    def set_preview(self, on: bool) -> None:
        self._preview = bool(on)
        if not self._preview:
            self._timer.stop()
            if self._precached:
                self._precached = False
                self._cancel_worker()
                self._atlas.clear()
            self._item.clear()
            self._item.setVisible(False)
            try:
                self.viewer.imageItem.setOpacity(1.0)
            except Exception:
                pass
            self._set_status("Preview off · analysis = height")
            return
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def set_precached(self, on: bool) -> bool:
        want = bool(on)
        if want == self._precached:
            return True
        if want and not self.can_precache():
            self._refresh_ram_label()
            return False
        self._timer.stop()
        self._precached = want
        if not self._precached:
            self._cancel_worker()
            self._atlas.clear()
            if self._preview:
                self._schedule()
            self._refresh_ram_label()
            if self._on_precache_off is not None:
                self._on_precache_off()
            return True
        self._azimuth = float(self._snap_az())
        if not self._preview:
            self._preview = True
        self._rebuild_atlas()
        return True

    def set_step(self, step_deg: float) -> None:
        step = clamp_azimuth_cache_step(step_deg)
        if step == self._step_deg:
            self._refresh_ram_label()
            return
        self._step_deg = step
        if self._precached:
            if not self.can_precache():
                self.set_precached(False)
                return
            self._azimuth = float(self._snap_az())
            self._rebuild_atlas()
            return
        self._refresh_ram_label()

    def can_precache(self) -> bool:
        """False if there is no frame or peak RAM would exceed 90% of free memory."""
        frame = self._height_frame()
        if frame is None:
            self._set_status("No image · load a height map first")
            return False
        h, w = int(frame.shape[0]), int(frame.shape[1])
        peak = azimuth_atlas_peak_nbytes(h, w, self._step_deg)
        avail = get_available_ram() * (1024**3)
        if avail > 0 and peak > avail * _RAM_BLOCK_FRAC:
            n = len(azimuth_cache_bins(self._step_deg))
            self._set_status(
                f"Pre-cache blocked · {n}× at {self._step_deg}° needs "
                f"~{self._fmt_bytes(peak)} peak, only "
                f"{get_available_ram():.1f} GB free. Coarser step."
            )
            return False
        return True

    def set_params(
        self,
        *,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        z_factor: Optional[float] = None,
    ) -> None:
        if self._precached:
            if azimuth is not None:
                self._azimuth = float(self._snap_az(azimuth))
                self._apply_cached_or_wait()
            return
        if azimuth is not None:
            self._azimuth = float(azimuth) % 360.0
        if elevation is not None:
            self._elevation = float(np.clip(elevation, 0.0, 90.0))
        if z_factor is not None:
            self._z_factor = float(max(0.01, z_factor))
        if self._preview:
            self._schedule()

    def _schedule(self, *_args) -> None:
        self._refresh_ram_label()
        if not self._preview:
            return
        if self._precached:
            self._timer.start()
            return
        if not self._timer.isActive():
            self._timer.start()

    def _on_timer(self) -> None:
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def _on_size_changed(self, *_args) -> None:
        if not self._preview:
            self._refresh_ram_label()
            return
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def _on_viewer_destroyed(self, *_args) -> None:
        self._preview = False
        self._precached = False
        self._cancel_worker()

    def _height_frame(self) -> Optional[np.ndarray]:
        """Authoritative height: ImageData / ImageView buffer (not the overlay)."""
        img = self.viewer.image
        if img is None:
            return None
        try:
            frame = img[int(self.viewer.currentIndex)]
        except Exception:
            return None
        if frame is None or np.size(frame) == 0:
            return None
        return np.asarray(frame)

    def _rebuild_atlas(self) -> None:
        self._timer.stop()
        self._cancel_worker()
        self._atlas.clear()
        if not self._preview or not self._precached:
            return
        frame = self._height_frame()
        if frame is None:
            self._item.setVisible(False)
            self._set_status("No image · load a height map first")
            return
        if not self.can_precache():
            self.set_precached(False)
            return
        copied = np.array(frame, copy=True, order="C")
        gen = self._generation
        self._worker = _ShadeAtlasWorker(
            copied,
            self._elevation,
            self._z_factor,
            self._azimuth,
            gen,
            step_deg=self._step_deg,
        )
        self._worker.bin_ready.connect(self._on_bin_ready)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished_ok.connect(self._on_atlas_done)
        self._set_precache_status()
        self._worker.start()

    def _cancel_worker(self) -> None:
        self._generation += 1
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        worker.requestInterruption()
        if worker.isRunning():
            self._zombies.append(worker)
            worker.finished.connect(lambda w=worker: self._reap(w))
        else:
            worker.deleteLater()

    def _reap(self, worker: QThread) -> None:
        if worker in self._zombies:
            self._zombies.remove(worker)
        worker.deleteLater()

    def _on_bin_ready(self, generation: int, az: int, shade: object) -> None:
        if generation != self._generation or not self._precached or not self._preview:
            return
        arr = np.asarray(shade)
        self._atlas[int(az)] = arr
        if int(az) == self._snap_az():
            self._show_shade(arr)
        self._set_precache_status()

    def _on_worker_failed(self, generation: int, message: str) -> None:
        if generation != self._generation or not self._precached:
            return
        self._set_status(f"Shade cache failed: {message}")

    def _on_atlas_done(self, generation: int) -> None:
        if generation != self._generation or not self._precached or not self._preview:
            return
        self._set_precache_status(ready=True)

    def _apply_cached_or_wait(self) -> None:
        az = self._snap_az()
        cached = self._atlas.get(az)
        if cached is not None:
            self._show_shade(cached)
        self._set_precache_status()

    def _show_shade(self, shade: np.ndarray) -> None:
        try:
            self.viewer.imageItem.setOpacity(0.0)
        except Exception:
            pass
        self._item.setImage(shade, autoLevels=False)
        if shade.dtype == np.uint8:
            self._item.setLevels((0.0, 255.0))
        else:
            self._item.setLevels((0.0, 1.0))
        self._item.setVisible(True)

    def _refresh_now(self) -> None:
        if not self._preview:
            return
        if self._precached:
            self._rebuild_atlas()
            return
        frame = self._height_frame()
        if frame is None:
            self._item.setVisible(False)
            self._set_status("No image · load a height map first")
            return
        try:
            shade = calculate_hillshade(
                frame,
                self._azimuth,
                self._elevation,
                self._z_factor,
            )
        except Exception as e:
            self._item.setVisible(False)
            self._set_status(f"Shade failed: {e}")
            return

        self._show_shade(shade)
        self._set_status(
            f"Hillshade preview · az {self._azimuth:.0f}° elev {self._elevation:.0f}° "
            f"Z×{self._z_factor:g} · analysis = height"
        )

    def _snap_az(self, azimuth: Optional[float] = None) -> int:
        src = self._azimuth if azimuth is None else azimuth
        return snap_azimuth_deg(src, self._step_deg)

    def _frame_hw(self) -> Optional[tuple[int, int]]:
        frame = self._height_frame()
        if frame is None or frame.ndim < 2:
            return None
        return int(frame.shape[0]), int(frame.shape[1])

    def _fmt_bytes(self, nbytes: int) -> str:
        if nbytes >= 1024**3:
            return f"{nbytes / (1024**3):.2f} GB"
        return format_size_mb(nbytes)

    def _refresh_ram_label(self) -> None:
        if self._ram_label is None:
            return
        step = self._step_deg
        n = len(azimuth_cache_bins(step))
        hw = self._frame_hw()
        free_gb = get_available_ram()
        if hw is None:
            self._ram_label.setText(
                f"{n} frames at {step}° · 5° finest · load an image for RAM"
            )
            self._ram_label.setStyleSheet("color: #888; font-size: 10pt;")
            return
        h, w = hw
        atlas = azimuth_atlas_nbytes(h, w, step)
        peak = azimuth_atlas_peak_nbytes(h, w, step)
        avail = free_gb * (1024**3)
        color = "#888"
        if avail > 0 and peak > avail * _RAM_BLOCK_FRAC:
            color = "#c44"
        elif avail > 0 and peak > avail * _RAM_WARN_FRAC:
            color = "#c80"
        self._ram_label.setText(
            f"{n}× {self._fmt_bytes(atlas)} atlas · peak ~{self._fmt_bytes(peak)} "
            f"· {free_gb:.1f} GB free"
        )
        self._ram_label.setStyleSheet(f"color: {color}; font-size: 10pt;")

    def _set_precache_status(self, *, ready: bool = False) -> None:
        n = len(azimuth_cache_bins(self._step_deg))
        have = len(self._atlas)
        az = self._snap_az()
        hw = self._frame_hw()
        atlas = 0 if hw is None else azimuth_atlas_nbytes(hw[0], hw[1], self._step_deg)
        ram = f" · {self._fmt_bytes(atlas)}" if atlas else ""
        if ready or have >= n:
            self._set_status(
                f"Pre-cache ready · az {az}° step {self._step_deg}° "
                f"elev {self._elevation:.0f}° Z×{self._z_factor:g}{ram} "
                f"· analysis = height"
            )
            self._refresh_ram_label()
            return
        self._set_status(
            f"Caching {have}/{n} · az {az}° step {self._step_deg}°{ram}…"
        )
        self._refresh_ram_label()

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.setText(text)

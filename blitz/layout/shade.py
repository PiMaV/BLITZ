"""View-only hillshade overlay. Analysis buffer stays original height."""

from __future__ import annotations

import time
from typing import Callable, Optional, Union

import numpy as np
import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar

from ..data.hillshade import (
    AZIMUTH_CACHE_STEP_DEG,
    ShadeLight,
    azimuth_atlas_nbytes,
    azimuth_atlas_peak_nbytes,
    azimuth_cache_bins,
    azimuth_cache_order,
    calculate_hillshade,
    clamp_azimuth_cache_step,
    rotate_lights_to_primary,
    scaled_height_gradients,
    shade_from_gradients,
    shade_rgb_from_gradients,
    shade_rgb_to_uint8,
    shade_to_uint8,
    snap_azimuth_deg,
)
from ..tools import format_size_mb, get_available_ram
from .viewer import ImageViewer

_RAM_BLOCK_FRAC = 0.90
_RAM_WARN_FRAC = 0.70


def _qobj_alive(obj) -> bool:
    return obj is not None and not sip.isdeleted(obj)


class _ShadeLiveWorker(QThread):
    """One full-frame hillshade off the UI thread."""

    ready = pyqtSignal(int, object, float)  # generation, shade array, elapsed_s
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        frame: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float,
        z_factor: float,
        generation: int,
        lights: Optional[list[tuple[float, float, float, float, float]]] = None,
    ) -> None:
        super().__init__()
        self._frame = frame
        self._azimuth = float(azimuth_deg)
        self._elevation = float(elevation_deg)
        self._z_factor = float(z_factor)
        self._generation = int(generation)
        self._lights = list(lights or [])

    def run(self) -> None:
        gen = self._generation
        t0 = time.perf_counter()
        try:
            lights = [
                ShadeLight(a, e, (r, g, b)) for a, e, r, g, b in self._lights
            ]
            if self.isInterruptionRequested():
                return
            shade = calculate_hillshade(
                self._frame,
                self._azimuth,
                self._elevation,
                self._z_factor,
                lights=lights,
            )
        except Exception as e:
            if not self.isInterruptionRequested():
                self.failed.emit(gen, str(e))
            return
        if not self.isInterruptionRequested():
            self.ready.emit(gen, shade, time.perf_counter() - t0)


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
        combined: bool = False,
        lights: Optional[list[tuple[float, float, float, float, float]]] = None,
    ) -> None:
        super().__init__()
        self._frame = frame
        self._elevation = float(elevation_deg)
        self._z_factor = float(z_factor)
        self._start_azimuth = float(start_azimuth)
        self._generation = int(generation)
        self._step_deg = int(step_deg)
        self._combined = bool(combined)
        self._lights = list(lights or [])

    def run(self) -> None:
        gen = self._generation
        try:
            dx, dy = scaled_height_gradients(self._frame, self._z_factor)
            base = [
                ShadeLight(a, e, (r, g, b)) for a, e, r, g, b in self._lights
            ]
            use_rgb = len(base) > 1 or (
                bool(base)
                and abs(sum(base[0].color) - 3.0) > 0.02
            )
            for az in azimuth_cache_order(self._start_azimuth, self._step_deg):
                if self.isInterruptionRequested():
                    return
                if use_rgb:
                    shade = shade_rgb_from_gradients(
                        dx, dy, rotate_lights_to_primary(base, az)
                    )
                    self.bin_ready.emit(gen, int(az), shade_rgb_to_uint8(shade))
                else:
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
        progress_bar: Optional[QProgressBar] = None,
        on_precache_off: Optional[Callable[[], None]] = None,
        on_compute: Optional[Callable[[], None]] = None,
    ) -> None:
        self.viewer = viewer
        self._status = status_label
        self._ram_label = ram_label
        self._progress = progress_bar
        self._on_precache_off = on_precache_off
        self._on_compute = on_compute
        self._preview = False
        self.last_compute_s: Optional[float] = None
        self._precached = False
        self._azimuth = 315.0
        self._elevation = 45.0
        self._z_factor = 1.0
        self._step_deg = AZIMUTH_CACHE_STEP_DEG
        self._combined = False
        self._lights: list[ShadeLight] = [ShadeLight(315.0, 45.0)]
        self._frame_hw: Optional[tuple[int, int]] = None
        self._atlas: dict[int, np.ndarray] = {}
        self._generation = 0
        self._worker: Optional[Union[_ShadeLiveWorker, _ShadeAtlasWorker]] = None
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

        self._timer = QTimer(viewer)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_timer)

        # Full-frame overlay: pan/zoom is ViewBox only — do not recompute on range.
        viewer.timeLine.sigPositionChanged.connect(self._schedule)
        viewer.image_changed.connect(self._schedule)
        viewer.image_size_changed.connect(self._on_size_changed)
        viewer.destroyed.connect(self._on_viewer_destroyed)
        self._ram_style: str | None = None
        self._ram_timer = QTimer(viewer)
        self._ram_timer.setInterval(1000)
        self._ram_timer.timeout.connect(self._refresh_ram_label)
        self._set_busy_off()
        self._set_ram_idle()

    @property
    def is_precached(self) -> bool:
        return self._precached

    @property
    def cache_step_deg(self) -> int:
        return self._step_deg

    def _notify_compute(self) -> None:
        cb = self._on_compute
        if cb is not None:
            cb()

    def _stop_timer(self) -> None:
        timer = self._timer
        if not _qobj_alive(timer):
            return
        try:
            timer.stop()
        except RuntimeError:
            pass

    def _set_busy_off(self) -> None:
        bar = self._progress
        if bar is None or not _qobj_alive(bar):
            return
        try:
            bar.setVisible(False)
            bar.setRange(0, 100)
            bar.setValue(0)
        except RuntimeError:
            pass

    def _set_busy_indeterminate(self) -> None:
        bar = self._progress
        if bar is None or not _qobj_alive(bar):
            return
        try:
            bar.setRange(0, 0)
            bar.setVisible(True)
        except RuntimeError:
            pass

    def _set_busy_fraction(self, have: int, total: int) -> None:
        bar = self._progress
        if bar is None or not _qobj_alive(bar):
            return
        n = max(1, int(total))
        try:
            bar.setRange(0, n)
            bar.setValue(min(n, max(0, int(have))))
            bar.setVisible(True)
        except RuntimeError:
            pass

    def set_preview(self, on: bool) -> None:
        self._preview = bool(on)
        if not self._preview:
            self._stop_timer()
            if self._precached:
                self._precached = False
                self._cancel_worker()
                self._atlas.clear()
            else:
                self._cancel_worker()
            self._set_busy_off()
            self._item.clear()
            self._item.setVisible(False)
            self._frame_hw = None
            try:
                self.viewer.imageItem.setOpacity(1.0)
            except Exception:
                pass
            self._set_status("Preview off · analysis = height")
            self._sync_ram_watch()
            self._notify_compute()
            return
        self._sync_ram_watch()
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def set_precached(self, on: bool) -> bool:
        want = bool(on)
        if want == self._precached:
            return True
        if want and not self.can_precache():
            self._sync_ram_watch()
            return False
        self._stop_timer()
        self._precached = want
        if not self._precached:
            self._cancel_worker()
            self._atlas.clear()
            self._set_busy_off()
            if self._preview:
                self._schedule()
            self._sync_ram_watch()
            if self._on_precache_off is not None:
                self._on_precache_off()
            return True
        self._azimuth = float(self._snap_az())
        if not self._preview:
            self._preview = True
        self._sync_ram_watch()
        self._rebuild_atlas()
        return True

    def set_step(self, step_deg: float) -> None:
        step = clamp_azimuth_cache_step(step_deg)
        if step == self._step_deg:
            self._sync_ram_watch()
            return
        self._step_deg = step
        if self._precached:
            if not self.can_precache():
                self.set_precached(False)
                return
            self._azimuth = float(self._snap_az())
            self._rebuild_atlas()
            return
        self._sync_ram_watch()

    def set_combined(self, on: bool) -> None:
        want = bool(on)
        if want == self._combined:
            return
        self._combined = want
        if not self._preview:
            return
        if self._precached:
            self._rebuild_atlas()
            return
        self._schedule()

    def set_lights(self, lights: list[ShadeLight]) -> None:
        items = list(lights)
        if not items:
            items = [ShadeLight(self._azimuth, self._elevation)]
        self._lights = items
        self._azimuth = float(items[0].azimuth) % 360.0
        self._elevation = float(np.clip(items[0].elevation, 0.0, 90.0))
        self._combined = len(items) > 1
        if not self._preview:
            return
        if self._precached:
            self._rebuild_atlas()
            return
        self._schedule()

    def _lights_payload(self) -> list[tuple[float, float, float, float, float]]:
        return [
            (L.azimuth, L.elevation, float(L.color[0]), float(L.color[1]), float(L.color[2]))
            for L in self._lights
        ]

    def _atlas_channels(self) -> int:
        if len(self._lights) > 1:
            return 3
        if self._lights:
            r, g, b = self._lights[0].color
            if abs(r - 1.0) + abs(g - 1.0) + abs(b - 1.0) > 0.02:
                return 3
        return 1

    def can_precache(self) -> bool:
        """False if there is no frame or peak RAM would exceed 90% of free memory."""
        frame = self._height_frame()
        if frame is None:
            self._set_status("No image · load a height map first")
            return False
        h, w = self._spatial_hw(frame)
        peak = azimuth_atlas_peak_nbytes(
            h, w, self._step_deg, channels=self._atlas_channels()
        )
        avail = get_available_ram() * (1024**3)
        if avail > 0 and peak > avail * _RAM_BLOCK_FRAC:
            n = len(azimuth_cache_bins(self._step_deg))
            self._set_status(
                f"Pre-cache blocked · {n}× full frame at {self._step_deg}° needs "
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
        if azimuth is not None:
            az = float(azimuth) % 360.0
            if self._precached:
                az = float(self._snap_az(az))
                self._lights = rotate_lights_to_primary(
                    self._lights or [ShadeLight(az, self._elevation)],
                    az,
                )
                self._azimuth = az
                self._apply_cached_or_wait()
                return
            if self._lights:
                L = self._lights[0]
                self._lights[0] = ShadeLight(az, L.elevation, L.color)
            else:
                self._lights = [ShadeLight(az, self._elevation)]
            self._azimuth = az
        if elevation is not None:
            el = float(np.clip(elevation, 0.0, 90.0))
            self._elevation = el
            # Combined / multi-light: one shared elevation (same as Preset).
            if self._lights:
                self._lights = [
                    ShadeLight(L.azimuth, el, L.color) for L in self._lights
                ]
            else:
                self._lights = [ShadeLight(self._azimuth, el)]
        if z_factor is not None:
            self._z_factor = float(max(0.01, z_factor))
        if self._preview:
            self._schedule()
        return

    def _schedule(self, *_args) -> None:
        """Params / frame / image change. Pan/zoom does not call this."""
        if not self._preview:
            return
        if not _qobj_alive(self._timer):
            return
        try:
            self._timer.start()
        except RuntimeError:
            return

    def _on_timer(self) -> None:
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def _on_size_changed(self, *_args) -> None:
        if not self._preview:
            return
        if self._precached:
            self._rebuild_atlas()
            return
        self._refresh_now()

    def _on_viewer_destroyed(self, *_args) -> None:
        self._preview = False
        self._precached = False
        self._stop_timer()
        try:
            self._ram_timer.stop()
        except RuntimeError:
            pass
        self._cancel_worker()
        self._set_busy_off()

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

    def _spatial_hw(self, frame: np.ndarray) -> tuple[int, int]:
        """``(h, w)`` for atlas sizing; ImageItem axisOrder decides which axes."""
        spatial = np.asarray(frame).shape[:2]
        order = str(getattr(self.viewer.imageItem, "axisOrder", "col-major"))
        if order == "row-major":
            return int(spatial[0]), int(spatial[1])
        # col-major: array is (x, y) → report (y, x) as h×w for humans / RAM
        return int(spatial[1]), int(spatial[0])

    def _rebuild_atlas(self) -> None:
        self._stop_timer()
        self._cancel_worker()
        self._atlas.clear()
        if not self._preview or not self._precached:
            return
        frame = self._height_frame()
        if frame is None:
            self._item.setVisible(False)
            self._set_busy_off()
            self._set_status("No image · load a height map first")
            return
        self._frame_hw = self._spatial_hw(frame)
        if not self.can_precache():
            self.set_precached(False)
            return
        copied = np.array(frame, copy=True, order="C")
        gen = self._generation
        n = len(azimuth_cache_bins(self._step_deg))
        self._set_busy_fraction(0, n)
        self._worker = _ShadeAtlasWorker(
            copied,
            self._elevation,
            self._z_factor,
            self._azimuth,
            gen,
            step_deg=self._step_deg,
            combined=self._combined,
            lights=self._lights_payload(),
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
        if generation != self._generation:
            return
        self._set_busy_off()
        self.last_compute_s = None
        self._notify_compute()
        if self._precached:
            self._set_status(f"Shade cache failed: {message}")
        else:
            self._item.setVisible(False)
            self._set_status(f"Shade failed: {message}")

    def _on_atlas_done(self, generation: int) -> None:
        if generation != self._generation or not self._precached or not self._preview:
            return
        self._set_busy_off()
        self._set_precache_status(ready=True)

    def _on_live_ready(self, generation: int, shade: object, elapsed_s: float) -> None:
        if generation != self._generation or not self._preview or self._precached:
            return
        self._set_busy_off()
        self.last_compute_s = float(elapsed_s)
        self._notify_compute()
        self._show_shade(np.asarray(shade))
        ms = 1000.0 * self.last_compute_s
        fps = (1.0 / self.last_compute_s) if self.last_compute_s > 1e-9 else 0.0
        hw = self._frame_hw or (0, 0)
        h, w = hw
        self._set_status(
            f"Hillshade full frame · az {self._azimuth:.0f}° elev {self._elevation:.0f}° "
            f"Z×{self._z_factor:g} · {h}×{w} · "
            f"{ms:.1f} ms ({fps:.0f} fps) · analysis = height"
        )

    def _apply_cached_or_wait(self) -> None:
        az = self._snap_az()
        cached = self._atlas.get(az)
        if cached is not None:
            self._show_shade(cached)
        self._set_precache_status()

    def _show_shade(self, shade: np.ndarray) -> None:
        # Keep the cube visible under the overlay (ViewBox scales both together).
        try:
            self.viewer.imageItem.setOpacity(1.0)
        except Exception:
            pass
        self._item.setImage(shade, autoLevels=False)
        if shade.dtype == np.uint8:
            self._item.setLevels((0.0, 255.0))
        else:
            self._item.setLevels((0.0, 1.0))
        # Full frame: same transform as the base ImageItem (no viewport crop).
        try:
            self._item.setRect()
        except Exception:
            pass
        self._item.setVisible(True)

    def _refresh_now(self) -> None:
        if not self._preview:
            return
        if self._precached:
            self._rebuild_atlas()
            return
        frame = self._height_frame()
        if frame is None:
            self._cancel_worker()
            self._set_busy_off()
            self._item.setVisible(False)
            self._set_status("No image · load a height map first")
            return
        self._frame_hw = self._spatial_hw(frame)
        self._cancel_worker()
        copied = np.array(frame, copy=True, order="C")
        gen = self._generation
        self._set_busy_indeterminate()
        self._set_status("Shading…")
        worker = _ShadeLiveWorker(
            copied,
            self._azimuth,
            self._elevation,
            self._z_factor,
            gen,
            lights=self._lights_payload(),
        )
        worker.ready.connect(self._on_live_ready)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _snap_az(self, azimuth: Optional[float] = None) -> int:
        src = self._azimuth if azimuth is None else azimuth
        return snap_azimuth_deg(src, self._step_deg)

    def _fmt_bytes(self, nbytes: int) -> str:
        if nbytes >= 1024**3:
            return f"{nbytes / (1024**3):.2f} GB"
        return format_size_mb(nbytes)

    def _set_ram_idle(self) -> None:
        if self._ram_label is None:
            return
        n = len(azimuth_cache_bins(self._step_deg))
        self._ram_label.setText(
            f"{n} full-frame bins at {self._step_deg}° · 5° finest"
        )
        if self._ram_style != "#888":
            self._ram_label.setStyleSheet("color: #888; font-size: 10pt;")
            self._ram_style = "#888"

    def _sync_ram_watch(self) -> None:
        """1 Hz psutil only while Preview or Pre-cache is on."""
        timer = getattr(self, "_ram_timer", None)
        if timer is None:
            return
        if self._preview or self._precached:
            if not timer.isActive():
                try:
                    timer.start()
                except RuntimeError:
                    return
            self._refresh_ram_label()
            return
        try:
            timer.stop()
        except RuntimeError:
            pass
        self._set_ram_idle()

    def _refresh_ram_label(self) -> None:
        if self._ram_label is None:
            return
        if not (self._preview or self._precached):
            self._set_ram_idle()
            return
        step = self._step_deg
        hw = self._frame_hw
        if hw is None:
            frame = self._height_frame()
            if frame is not None:
                hw = self._spatial_hw(frame)
        n = len(azimuth_cache_bins(step))
        free_gb = get_available_ram()
        if hw is None:
            self._ram_label.setText(
                f"{n} full-frame bins at {step}° · 5° finest · load an image for RAM"
            )
            if self._ram_style != "#888":
                self._ram_label.setStyleSheet("color: #888; font-size: 10pt;")
                self._ram_style = "#888"
            return
        h, w = hw
        ch = self._atlas_channels()
        atlas = azimuth_atlas_nbytes(h, w, step, channels=ch)
        peak = azimuth_atlas_peak_nbytes(h, w, step, channels=ch)
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
        if self._ram_style != color:
            self._ram_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
            self._ram_style = color

    def _set_precache_status(self, *, ready: bool = False) -> None:
        n = len(azimuth_cache_bins(self._step_deg))
        have = len(self._atlas)
        az = self._snap_az()
        hw = self._frame_hw
        atlas = (
            0 if hw is None
            else azimuth_atlas_nbytes(hw[0], hw[1], self._step_deg)
        )
        ram = f" · {self._fmt_bytes(atlas)}" if atlas else ""
        if ready or have >= n:
            self._set_busy_off()
            self._set_status(
                f"Pre-cache ready · az {az}° step {self._step_deg}° "
                f"elev {self._elevation:.0f}° Z×{self._z_factor:g}{ram} "
                f"· analysis = height"
            )
            return
        self._set_busy_fraction(have, n)
        self._set_status(
            f"Caching {have}/{n} · az {az}° step {self._step_deg}°{ram}…"
        )

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.setText(text)

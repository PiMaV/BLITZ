"""Sky-dome shade control: azimuth, elevation, Z-shadow, optional coloured lights."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PyQt6.QtWidgets import QColorDialog, QDoubleSpinBox, QSizePolicy, QWidget

from ..data.hillshade import (
    Z_FACTOR_MAX,
    Z_FACTOR_MIN,
    ShadeLight,
    four_way_lights,
    rotate_lights_to_primary,
    shadow_azimuth_deg,
    snap_azimuth_deg,
    step_azimuth_deg,
)
from ..theme import COLOR_FG, COLOR_ORANGE


def _azimuth_from_offset(dx: float, dy: float) -> float:
    """Qt y-down → BLITZ azimuth (0° = up, clockwise)."""
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return float(math.degrees(math.atan2(dx, -dy)) % 360.0)


def _wrap_azimuth(deg: float) -> float:
    return float(deg) % 360.0


_SUN_GOLD = QColor(COLOR_ORANGE)
_SHADOW_FILL = QColor(12, 14, 22, 150)
_SHADOW_TIP = QColor(28, 32, 48, 230)


class WrappingAzimuthSpinBox(QDoubleSpinBox):
    """Wheel/step wraps through 0° with modulo (0° − step → 360° − step)."""

    def stepBy(self, steps: int) -> None:
        step = float(self.singleStep())
        self.setValue((float(self.value()) + float(steps) * step) % 360.0)


class AzimuthOrbit(QWidget):
    """Polar sky dome: rim = elevation 0°, centre = 90°; shadow length = Z."""

    azimuthChanged = pyqtSignal(float)
    elevationChanged = pyqtSignal(float)
    zFactorChanged = pyqtSignal(float)
    lightsChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lights: list[ShadeLight] = [ShadeLight(315.0, 45.0)]
        self._z_factor = 1.0
        self._snap: Optional[int] = None
        self._selected = 0
        self._drag: Optional[str] = None
        self._drag_index = 0
        self._lock_elev_z = False
        self.setFixedSize(168, 168)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "Sky dome. Yellow = sun. Grey wedge = shadow of the centre peg "
            "(falls opposite the sun; 0° / N = top of the image, clockwise). "
            "Drag the sun (around = azimuth, inward = elevation). "
            "Drag the shadow tip for Z. Wheel wraps through 0°."
        )

    def lights(self) -> list[ShadeLight]:
        return list(self._lights)

    def azimuth(self) -> float:
        if not self._lights:
            return 0.0
        return float(self._lights[0].azimuth)

    def elevation(self) -> float:
        if not self._lights:
            return 45.0
        return float(self._lights[0].elevation)

    def zFactor(self) -> float:
        return float(self._z_factor)

    def setAzimuth(self, deg: float, *, rotate_rig: bool = False) -> None:
        az = _wrap_azimuth(deg)
        if self._snap:
            az = float(snap_azimuth_deg(az, self._snap))
        if not self._lights:
            self._lights = [ShadeLight(az, 45.0)]
        elif rotate_rig:
            self._lights = rotate_lights_to_primary(self._lights, az)
        else:
            L = self._lights[0]
            if abs(_wrap_azimuth(L.azimuth) - az) < 0.05:
                return
            self._lights[0] = ShadeLight(az, L.elevation, L.color)
        self.update()

    def setElevation(self, deg: float) -> None:
        el = float(max(0.0, min(90.0, deg)))
        if not self._lights:
            self._lights = [ShadeLight(315.0, el)]
        else:
            L = self._lights[0]
            if abs(L.elevation - el) < 0.05:
                return
            self._lights[0] = ShadeLight(L.azimuth, el, L.color)
        self.update()

    def setZFactor(self, value: float) -> None:
        z = float(max(Z_FACTOR_MIN, min(Z_FACTOR_MAX, value)))
        if abs(z - self._z_factor) < 1e-4:
            return
        self._z_factor = z
        self.update()

    def setCombined(self, on: bool) -> None:
        if on:
            if len(self._lights) >= 4:
                self.update()
                return
            self._lights = list(four_way_lights(self.azimuth(), self.elevation()))
            self._selected = 0
        else:
            L = self._lights[0] if self._lights else ShadeLight(315.0, 45.0)
            self._lights = [ShadeLight(L.azimuth, L.elevation)]
            self._selected = 0
        self.lightsChanged.emit()
        self.update()

    def applyFourWayPreset(self) -> None:
        self._lights = list(four_way_lights(self.azimuth(), self.elevation()))
        self._selected = 0
        self.lightsChanged.emit()
        self.azimuthChanged.emit(self.azimuth())
        self.elevationChanged.emit(self.elevation())
        self.update()

    def setLights(self, lights: Sequence[ShadeLight]) -> None:
        items = list(lights)
        if not items:
            return
        self._lights = items
        self._selected = min(self._selected, len(self._lights) - 1)
        self.update()

    def setSnapStep(self, step_deg: Optional[int]) -> None:
        self._snap = int(step_deg) if step_deg else None
        self.update()

    def setElevZLocked(self, locked: bool) -> None:
        self._lock_elev_z = bool(locked)

    def _geom(self) -> tuple[QPointF, float]:
        w, h = float(self.width()), float(self.height())
        r = min(w, h) / 2.0 - 16.0
        return QPointF(w / 2.0, h / 2.0), r

    def _sun_pos(self, light: ShadeLight, c: QPointF, r: float) -> QPointF:
        rho = (90.0 - float(light.elevation)) / 90.0
        rad = math.radians(float(light.azimuth))
        return QPointF(
            c.x() + r * rho * math.sin(rad),
            c.y() - r * rho * math.cos(rad),
        )

    def _pos_to_polar(self, pos: QPointF, c: QPointF, r: float) -> tuple[float, float]:
        dx = pos.x() - c.x()
        dy = pos.y() - c.y()
        az = _azimuth_from_offset(dx, dy)
        dist = math.hypot(dx, dy)
        rho = min(1.0, dist / max(r, 1e-6))
        el = 90.0 * (1.0 - rho)
        return az, el

    def _shadow_length(self, r: float) -> float:
        """Peg shadow: opposite the sun, longer when the sun is lower or Z is higher."""
        el = math.radians(max(8.0, min(90.0, self.elevation())))
        cot = math.cos(el) / math.sin(el)
        raw = 0.40 * r * float(self._z_factor) * cot
        return max(0.05 * r, min(0.88 * r, raw))

    def _shadow_tip(self, c: QPointF, r: float) -> QPointF:
        az = shadow_azimuth_deg(self.azimuth())
        length = self._shadow_length(r)
        rad = math.radians(az)
        return QPointF(
            c.x() + length * math.sin(rad),
            c.y() - length * math.cos(rad),
        )

    def _z_from_tip(self, pos: QPointF, c: QPointF, r: float) -> float:
        az = shadow_azimuth_deg(self.azimuth())
        rad = math.radians(az)
        ux, uy = math.sin(rad), -math.cos(rad)
        dx, dy = pos.x() - c.x(), pos.y() - c.y()
        proj = max(0.0, dx * ux + dy * uy)
        el = math.radians(max(8.0, min(90.0, self.elevation())))
        cot = math.cos(el) / math.sin(el)
        denom = 0.40 * max(r, 1e-6) * max(cot, 1e-6)
        z = proj / denom
        return float(max(Z_FACTOR_MIN, min(Z_FACTOR_MAX, z)))

    def _hit_sun(self, pos: QPointF, c: QPointF, r: float) -> Optional[int]:
        best_i, best_d = None, 16.0
        for i, light in enumerate(self._lights):
            pt = self._sun_pos(light, c, r)
            d = math.hypot(pos.x() - pt.x(), pos.y() - pt.y())
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def _hit_shadow(self, pos: QPointF, c: QPointF, r: float) -> bool:
        tip = self._shadow_tip(c, r)
        return math.hypot(pos.x() - tip.x(), pos.y() - tip.y()) < 14.0

    def _replace_light(self, index: int, light: ShadeLight) -> None:
        self._lights[index] = light
        if self._snap and index == 0:
            az = float(snap_azimuth_deg(light.azimuth, self._snap))
            self._lights[0] = ShadeLight(az, light.elevation, light.color)

    def _emit_primary(self) -> None:
        self.azimuthChanged.emit(self.azimuth())
        self.elevationChanged.emit(self.elevation())
        self.lightsChanged.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        c, r = self._geom()
        if event.button() == Qt.MouseButton.RightButton:
            hit = self._hit_sun(pos, c, r)
            if hit is None:
                return
            self._selected = hit
            self._pick_color(hit)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._hit_shadow(pos, c, r):
            if self._lock_elev_z:
                return
            self._drag = "z"
            self._apply_z(self._z_from_tip(pos, c, r))
            return
        hit = self._hit_sun(pos, c, r)
        if hit is None:
            hit = self._selected if self._lights else 0
        self._drag = "sun"
        self._drag_index = int(hit)
        self._selected = self._drag_index
        self._apply_sun_at(pos, c, r)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        pos = event.position()
        c, r = self._geom()
        if self._drag == "z":
            self._apply_z(self._z_from_tip(pos, c, r))
            return
        self._apply_sun_at(pos, c, r)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drag = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        dy = event.angleDelta().y()
        if dy == 0:
            dy = event.angleDelta().x()
        if dy == 0:
            event.ignore()
            return
        step = float(self._snap) if self._snap else 5.0
        delta = step if dy > 0 else -step
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            if not self._lock_elev_z:
                self._apply_z(self._z_factor + (0.05 if dy > 0 else -0.05))
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            if not self._lock_elev_z:
                el = max(0.0, min(90.0, self.elevation() + delta))
                self.setElevation(el)
                self.elevationChanged.emit(self.elevation())
                self.lightsChanged.emit()
        else:
            self._apply_azimuth_delta(delta)
        event.accept()

    def _apply_azimuth_delta(self, delta: float) -> None:
        if self._snap:
            n = 1 if delta > 0 else -1
            az = float(step_azimuth_deg(self.azimuth(), self._snap, n))
        else:
            az = _wrap_azimuth(self.azimuth() + delta)
        if len(self._lights) > 1:
            self._lights = rotate_lights_to_primary(self._lights, az)
        elif not self._lights:
            self._lights = [ShadeLight(az, 45.0)]
        else:
            L = self._lights[0]
            self._lights[0] = ShadeLight(az, L.elevation, L.color)
        self.azimuthChanged.emit(self.azimuth())
        self.lightsChanged.emit()
        self.update()

    def _apply_sun_at(self, pos: QPointF, c: QPointF, r: float) -> None:
        az, el = self._pos_to_polar(pos, c, r)
        idx = self._drag_index
        if idx < 0 or idx >= len(self._lights):
            return
        color = self._lights[idx].color
        if self._lock_elev_z:
            el = self._lights[idx].elevation
        self._replace_light(idx, ShadeLight(az, el, color))
        if idx == 0:
            self._emit_primary()
        else:
            self.lightsChanged.emit()
        self.update()

    def _apply_z(self, value: float) -> None:
        z = float(max(Z_FACTOR_MIN, min(Z_FACTOR_MAX, value)))
        if abs(z - self._z_factor) < 1e-4:
            return
        self._z_factor = z
        self.zFactorChanged.emit(self._z_factor)
        self.update()

    def _pick_color(self, index: int) -> None:
        if index < 0 or index >= len(self._lights):
            return
        r, g, b = self._lights[index].color
        start = QColor.fromRgbF(float(r), float(g), float(b))
        chosen = QColorDialog.getColor(start, self, "Light colour")
        if not chosen.isValid():
            return
        rf, gf, bf = chosen.redF(), chosen.greenF(), chosen.blueF()
        L = self._lights[index]
        self._lights[index] = ShadeLight(L.azimuth, L.elevation, (rf, gf, bf))
        self.lightsChanged.emit()
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._geom()

        rim = QColor(COLOR_FG)
        rim.setAlpha(55)
        p.setPen(QPen(rim, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        faint = QColor(COLOR_FG)
        faint.setAlpha(28)
        p.setPen(QPen(faint, 1.0, Qt.PenStyle.DotLine))
        p.drawEllipse(c, r * (2.0 / 3.0), r * (2.0 / 3.0))
        p.drawEllipse(c, r * (1.0 / 3.0), r * (1.0 / 3.0))

        tick = QColor(COLOR_FG)
        tick.setAlpha(90)
        p.setPen(QPen(tick, 1.2, Qt.PenStyle.SolidLine))
        for ang in (0.0, 90.0, 180.0, 270.0):
            rad = math.radians(ang)
            inner, outer = r - 5.0, r + 1.0
            p.drawLine(
                QPointF(c.x() + inner * math.sin(rad), c.y() - inner * math.cos(rad)),
                QPointF(c.x() + outer * math.sin(rad), c.y() - outer * math.cos(rad)),
            )

        font = p.font()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(COLOR_FG))
        p.drawText(int(c.x()) - 5, int(c.y() - r - 1), "N")

        # Shadow first (under the peg and the sun).
        tip = self._shadow_tip(c, r)
        az_sh = shadow_azimuth_deg(self.azimuth())
        rad = math.radians(az_sh)
        nx, ny = math.cos(rad) * 5.5, math.sin(rad) * 5.5
        path = QPainterPath()
        path.moveTo(c)
        path.lineTo(QPointF(tip.x() + nx, tip.y() + ny))
        path.lineTo(QPointF(tip.x() - nx, tip.y() - ny))
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_SHADOW_FILL)
        p.drawPath(path)
        p.setBrush(_SHADOW_TIP)
        p.drawEllipse(tip, 4.0, 4.0)

        # Centre peg (gnomon) — the thing that casts the shadow.
        p.setBrush(QColor(36, 40, 58))
        p.setPen(QPen(QColor(COLOR_FG), 0.8))
        p.drawEllipse(c, 3.5, 3.5)

        for i, light in enumerate(self._lights):
            pt = self._sun_pos(light, c, r)
            cr, cg, cb = light.color
            tinted = abs(cr - 1.0) + abs(cg - 1.0) + abs(cb - 1.0) > 0.04
            fill = QColor.fromRgbF(float(cr), float(cg), float(cb)) if tinted else _SUN_GOLD
            radius = 8.0 if i == 0 else 5.5
            p.setPen(Qt.PenStyle.NoPen)
            if i == self._selected:
                glow = QColor(fill)
                glow.setAlpha(70)
                p.setBrush(glow)
                p.drawEllipse(pt, radius + 4.0, radius + 4.0)
            p.setBrush(fill)
            p.drawEllipse(pt, radius, radius)

        p.end()

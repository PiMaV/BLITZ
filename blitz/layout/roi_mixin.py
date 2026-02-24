from typing import Any, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel,
    QSpinBox, QVBoxLayout, QWidget
)


class ROIMixin:
    """
    Mixin for Load Dialogs to provide ROI spinners (X, Y, W, H)
    and Flip XY checkbox.

    Requires the consuming class to have:
    - self._preview: np.ndarray (the raw loaded preview image)
    - self._roi: pg.RectROI (the ROI object on the plot)
    - self._img_item: pg.ImageItem (the image item on the plot)
    - self._plot_widget: pg.PlotWidget
    - self._update_estimates(): Method to recalculate RAM usage
    """

    def _setup_roi_controls(self, layout: QVBoxLayout) -> None:
        """Add ROI controls to the given layout."""
        # Transform Checkboxes
        transform_layout = QHBoxLayout()
        self.chk_flip_xy = QCheckBox("Flip XY (Transpose)")
        self.chk_flip_xy.setToolTip("Transpose image (swap X/Y axes)")
        transform_layout.addWidget(self.chk_flip_xy)
        transform_layout.addStretch()
        layout.addLayout(transform_layout)

        # ROI Spinners (grouped to avoid overlap)
        roi_group = QGroupBox("Crop Region (ROI)")
        roi_group.setToolTip("Select region to load. X,Y = top-left; W,H = size. Clamped to image.")
        roi_inner = QVBoxLayout(roi_group)
        roi_row1 = QHBoxLayout()
        roi_row1.addWidget(QLabel("X:"))
        self.spin_roi_x = self._create_spinner("X Position", 0, 99999)
        roi_row1.addWidget(self.spin_roi_x)
        roi_row1.addWidget(QLabel("Y:"))
        self.spin_roi_y = self._create_spinner("Y Position", 0, 99999)
        roi_row1.addWidget(self.spin_roi_y)
        roi_row1.addStretch()
        roi_inner.addLayout(roi_row1)
        roi_row2 = QHBoxLayout()
        roi_row2.addWidget(QLabel("W:"))
        self.spin_roi_w = self._create_spinner("Width", 1, 99999)
        roi_row2.addWidget(self.spin_roi_w)
        roi_row2.addWidget(QLabel("H:"))
        self.spin_roi_h = self._create_spinner("Height", 1, 99999)
        roi_row2.addWidget(self.spin_roi_h)
        roi_row2.addStretch()
        roi_inner.addLayout(roi_row2)
        layout.addWidget(roi_group)

        # Connections
        self.chk_flip_xy.stateChanged.connect(self._on_transform_changed)

        for spin in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h):
            spin.valueChanged.connect(self._on_spin_roi_changed)

    def _create_spinner(self, tooltip: str, min_val: int, max_val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(min_val, max_val)
        s.setToolTip(tooltip)
        # Compact style
        s.setMaximumWidth(95)
        return s

    def _connect_roi_signals(self) -> None:
        """Call this after self._roi is created. Configures corner scale handles."""
        if getattr(self, "_roi", None) is None:
            return
        # Remove any handles (from dialogs or defaults) to avoid "whole thing scaling" when grabbing a corner
        for h in list(self._roi.getHandles()):
            try:
                self._roi.removeHandle(h)
            except (KeyError, TypeError):
                pass
        # Translate handle at center: drag center to move without resizing
        self._roi.addTranslateHandle([0.5, 0.5])
        # Corner scale handles: drag corner to resize (opposite stays fixed)
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.addScaleHandle([1, 0], [0, 1])
        self._roi.addScaleHandle([0, 1], [1, 0])
        self._roi.sigRegionChanged.connect(self._on_roi_changed)
        self._on_roi_changed()

    def _on_transform_changed(self) -> None:
        """Update preview image based on checkboxes."""
        if getattr(self, "_preview", None) is None:
            return

        img = self._get_transformed_preview()

        # Update ImageItem
        h, w = img.shape[:2]
        display_img = np.swapaxes(img, 0, 1) if img.ndim == 3 else img.T

        if getattr(self, "_img_item", None):
            self._img_item.setImage(display_img)
            self._img_item.setRect(pg.QtCore.QRectF(0, 0, w, h))

        # Update plot range
        if getattr(self, "_plot_widget", None):
            self._plot_widget.setRange(xRange=(0, w), yRange=(0, h), padding=0)

            # Reset ROI to full frame of new dimensions
            if getattr(self, "_roi", None):
                self._roi.blockSignals(True)
                self._roi.setPos((0, 0))
                self._roi.setSize((w, h))
                self._roi.setAngle(0)
                self._roi.blockSignals(False)
                self._on_roi_changed() # Sync spinners

        # Trigger estimate update (which uses _get_transformed_preview dimensions)
        if hasattr(self, "_update_estimates"):
            self._update_estimates()

    def _get_transformed_preview(self) -> np.ndarray:
        """Return the preview image with Flip XY applied."""
        img = self._preview
        if self.chk_flip_xy.isChecked():
            img = np.transpose(img, (1, 0, 2)) if img.ndim == 3 else img.T
        return img

    def _get_transformed_bounds(self) -> tuple[int, int]:
        """Return (width, height) of transformed preview. (0,0) if no preview."""
        if getattr(self, "_preview", None) is None:
            return 0, 0
        img = self._get_transformed_preview()
        h, w = img.shape[:2]
        return w, h

    def _on_spin_roi_changed(self) -> None:
        """Update ROI object from spinners. Clamp to image bounds."""
        if getattr(self, "_roi", None) is None:
            return

        tw, th = self._get_transformed_bounds()
        if tw <= 0 or th <= 0:
            return

        self._roi.sigRegionChanged.disconnect(self._on_roi_changed)

        x = max(0, min(self.spin_roi_x.value(), tw - 1))
        y = max(0, min(self.spin_roi_y.value(), th - 1))
        w = max(1, min(self.spin_roi_w.value(), tw - x))
        h = max(1, min(self.spin_roi_h.value(), th - y))

        self._roi.setPos((x, y))
        self._roi.setSize((w, h))
        self._roi.setAngle(0)

        self._roi.sigRegionChanged.connect(self._on_roi_changed)

        if hasattr(self, "_update_estimates"):
            self._update_estimates()

    def _on_roi_changed(self) -> None:
        """Update spinners from ROI object. Clamp ROI to image bounds."""
        if getattr(self, "_roi", None) is None:
            return

        tw, th = self._get_transformed_bounds()
        if tw <= 0 or th <= 0:
            return

        state = self._roi.getState()
        pos = state["pos"]
        size = state["size"]

        x = max(0, min(int(pos.x()), tw - 1))
        y = max(0, min(int(pos.y()), th - 1))
        w = max(1, min(int(size.x()), tw - x))
        h = max(1, min(int(size.y()), th - y))

        if x != pos.x() or y != pos.y() or w != size.x() or h != size.y():
            self._roi.blockSignals(True)
            self._roi.setPos((x, y))
            self._roi.setSize((w, h))
            self._roi.blockSignals(False)

        for s in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h):
            s.blockSignals(True)

        self.spin_roi_x.setMaximum(max(0, tw - 1))
        self.spin_roi_y.setMaximum(max(0, th - 1))
        self.spin_roi_w.setMaximum(tw)
        self.spin_roi_h.setMaximum(th)
        self.spin_roi_x.setValue(x)
        self.spin_roi_y.setValue(y)
        self.spin_roi_w.setValue(w)
        self.spin_roi_h.setValue(h)

        for s in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h):
            s.blockSignals(False)

        if hasattr(self, "_update_estimates"):
            self._update_estimates()

    def _get_roi_source_mask(self) -> Tuple[Optional[Tuple[slice, slice]], Optional[Tuple[float, float, float, float]]]:
        """
        Calculate the source mask (slice) corresponding to the current ROI on the transformed preview.
        Returns:
            (mask_slice, mask_rel)
            mask_slice: (slice_y, slice_x) for the source image.
            mask_rel: (x0, y0, x1, y1) relative ratios for persistence (based on bounding box).
        """
        if getattr(self, "_preview", None) is None or getattr(self, "_roi", None) is None:
            return None, None

        # 1. Get ROI corners in Transformed Preview space
        state = self._roi.getState()
        pos = state['pos']
        size = state['size']

        w, h = size.x(), size.y()
        # Unrotated corners relative to pos
        corners = [
            pg.Point(0, 0),
            pg.Point(w, 0),
            pg.Point(w, h),
            pg.Point(0, h),
        ]

        # Angle is 0 (rotation disabled). Corners = transformed_corners.
        transformed_corners = [pg.Point(p.x() + pos.x(), p.y() + pos.y()) for p in corners]

        # 2. Map corners from Transformed Preview -> Source Image
        preview_img = self._get_transformed_preview()
        H_t, W_t = preview_img.shape[:2]

        source_corners = []
        for p in transformed_corners:
            x, y = p.x(), p.y()
            x = max(0, min(x, W_t - 1e-6))
            y = max(0, min(y, H_t - 1e-6))
            # Un-Flip XY (inverse of Transpose)
            if self.chk_flip_xy.isChecked():
                x, y = y, x
            source_corners.append((x, y))

        # 3. Calculate Bounding Box in Source Space
        xs = [p[0] for p in source_corners]
        ys = [p[1] for p in source_corners]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        # Clamp to source dimensions
        source_h, source_w = self._preview.shape[:2]
        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(source_w, int(round(x1)))
        y1 = min(source_h, int(round(y1)))

        if x1 <= x0 or y1 <= y0:
            # If completely out of bounds or invalid
            return None, None

        # mask = (slice_y, slice_x) for numpy [row, col]
        # x is column, y is row.
        # So slice(y0, y1), slice(x0, x1)
        # Note: DataLoader expects (slice_x, slice_y) IF I pass it as a tuple to mask?
        # Let's check DataLoader:
        # if mask is not None: self.mask = (mask[1], mask[0])
        # So DataLoader expects (slice_x, slice_y) and swaps them to (slice_row, slice_col).
        # Correct.
        mask_slice = (slice(x0, x1), slice(y0, y1))
        mask_rel = (x0 / source_w, y0 / source_h, x1 / source_w, y1 / source_h)
        return mask_slice, mask_rel

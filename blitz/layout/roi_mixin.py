from typing import Any, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget
)


class ROIMixin:
    """
    Mixin for Load Dialogs to provide ROI spinners (X, Y, W, H, Angle)
    and transformation checkboxes (Flip XY, Rotate 90).

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
        self.chk_rotate_90 = QCheckBox("Rotate 90°")
        self.chk_rotate_90.setToolTip("Rotate 90 degrees clockwise")
        transform_layout.addWidget(self.chk_flip_xy)
        transform_layout.addWidget(self.chk_rotate_90)
        transform_layout.addStretch()
        layout.addLayout(transform_layout)

        # ROI Spinners
        roi_layout = QHBoxLayout()
        self.spin_roi_x = self._create_spinner("X Position", 0, 99999)
        self.spin_roi_y = self._create_spinner("Y Position", 0, 99999)
        self.spin_roi_w = self._create_spinner("Width", 1, 99999)
        self.spin_roi_h = self._create_spinner("Height", 1, 99999)

        self.spin_roi_angle = QDoubleSpinBox()
        self.spin_roi_angle.setRange(-360, 360)
        self.spin_roi_angle.setSuffix("°")
        self.spin_roi_angle.setToolTip("Rotation Angle")
        self.spin_roi_angle.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin_roi_angle.setMaximumWidth(60)

        # Add labels and widgets
        roi_layout.addWidget(QLabel("ROI:"))
        roi_layout.addWidget(QLabel("X"))
        roi_layout.addWidget(self.spin_roi_x)
        roi_layout.addWidget(QLabel("Y"))
        roi_layout.addWidget(self.spin_roi_y)
        roi_layout.addWidget(QLabel("W"))
        roi_layout.addWidget(self.spin_roi_w)
        roi_layout.addWidget(QLabel("H"))
        roi_layout.addWidget(self.spin_roi_h)
        roi_layout.addWidget(QLabel("∠"))
        roi_layout.addWidget(self.spin_roi_angle)
        roi_layout.addStretch()
        layout.addLayout(roi_layout)

        # Connections
        self.chk_flip_xy.stateChanged.connect(self._on_transform_changed)
        self.chk_rotate_90.stateChanged.connect(self._on_transform_changed)

        for spin in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h):
            spin.valueChanged.connect(self._on_spin_roi_changed)
        self.spin_roi_angle.valueChanged.connect(self._on_spin_roi_changed)

    def _create_spinner(self, tooltip: str, min_val: int, max_val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(min_val, max_val)
        s.setToolTip(tooltip)
        # Compact style
        s.setMaximumWidth(70)
        return s

    def _connect_roi_signals(self) -> None:
        """Call this after self._roi is created."""
        if getattr(self, "_roi", None) is not None:
            self._roi.sigRegionChanged.connect(self._on_roi_changed)
            # Add rotation handle if not present
            # Default RectROI has scale handles. We add a rotate handle.
            self._roi.addRotateHandle([0.5, 0], [0.5, 0.5])

            # Initial sync
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
        """Return the preview image with Flip/Rotate applied."""
        img = self._preview
        if self.chk_flip_xy.isChecked():
            img = np.transpose(img, (1, 0, 2)) if img.ndim == 3 else img.T
        if self.chk_rotate_90.isChecked():
            # Rotate 90 CW (k=-1)
            img = np.rot90(img, k=-1, axes=(0, 1))
        return img

    def _on_spin_roi_changed(self) -> None:
        """Update ROI object from spinners."""
        if getattr(self, "_roi", None) is None:
            return

        # Avoid feedback
        self._roi.sigRegionChanged.disconnect(self._on_roi_changed)

        x = self.spin_roi_x.value()
        y = self.spin_roi_y.value()
        w = self.spin_roi_w.value()
        h = self.spin_roi_h.value()
        angle = self.spin_roi_angle.value()

        self._roi.setPos((x, y))
        self._roi.setSize((w, h))
        self._roi.setAngle(angle)

        self._roi.sigRegionChanged.connect(self._on_roi_changed)

        if hasattr(self, "_update_estimates"):
            self._update_estimates()

    def _on_roi_changed(self) -> None:
        """Update spinners from ROI object."""
        if getattr(self, "_roi", None) is None:
            return

        state = self._roi.getState()
        pos = state['pos']
        size = state['size']
        angle = state['angle']

        for s in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h, self.spin_roi_angle):
            s.blockSignals(True)

        self.spin_roi_x.setValue(int(pos.x()))
        self.spin_roi_y.setValue(int(pos.y()))
        self.spin_roi_w.setValue(int(size.x()))
        self.spin_roi_h.setValue(int(size.y()))
        self.spin_roi_angle.setValue(angle)

        for s in (self.spin_roi_x, self.spin_roi_y, self.spin_roi_w, self.spin_roi_h, self.spin_roi_angle):
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
        pos = state['pos'] # Point
        size = state['size'] # Point
        angle = state['angle'] # float degrees

        w, h = size.x(), size.y()
        # Unrotated corners relative to pos
        corners = [
            pg.Point(0, 0),
            pg.Point(w, 0),
            pg.Point(w, h),
            pg.Point(0, h),
        ]

        # Rotate corners around origin (0,0) - because pos/angle define transform from origin
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        transformed_corners = []
        for p in corners:
            x_r = p.x() * c - p.y() * s
            y_r = p.x() * s + p.y() * c
            transformed_corners.append(pg.Point(x_r + pos.x(), y_r + pos.y()))

        # 2. Map corners from Transformed Preview -> Source Image
        # Dimensions of Transformed Preview
        preview_img = self._get_transformed_preview()
        initial_H_t, initial_W_t = preview_img.shape[:2]

        source_corners = []
        for p in transformed_corners:
            x, y = p.x(), p.y()
            W_t, H_t = initial_W_t, initial_H_t

            # Un-Rotate 90 CW (inverse of Rot90 k=-1)
            # Forward: x_new = H_old - y_old; y_new = x_old
            # Inverse: x_old = y_new; y_old = H_old - x_new (H_old = W_new)
            if self.chk_rotate_90.isChecked():
                x_prev = y
                y_prev = W_t - x
                x, y = x_prev, y_prev
                # Dims of previous step (Inverse of W,H = H,W is W,H = H,W)
                W_t, H_t = H_t, W_t

            # Un-Flip XY (inverse of Transpose)
            if self.chk_flip_xy.isChecked():
                x, y = y, x
                W_t, H_t = H_t, W_t

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

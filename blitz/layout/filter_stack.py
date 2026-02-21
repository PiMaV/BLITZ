from typing import Any, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import get_style


class FilterItemWidget(QFrame):
    """Widget representing a single filter in the stack."""

    removed = pyqtSignal()
    moved_up = pyqtSignal()
    moved_down = pyqtSignal()
    changed = pyqtSignal()

    # Signal to request loading a reference file (for Subtract/Divide)
    load_reference_requested = pyqtSignal()

    def __init__(self, filter_type: str, params: dict | None = None) -> None:
        super().__init__()
        self.filter_type = filter_type
        self.params = params or {}

        # Hold the reference image object (ImageData) separately from serializable params
        self._reference_image: Any = None
        if "reference" in self.params:
            self._reference_image = self.params.pop("reference")

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        # Using a slightly darker background to distinguish items
        self.setStyleSheet(
            "FilterItemWidget { border: 1px solid #444; border-radius: 4px; "
            "background-color: #2a2a2a; margin-bottom: 2px; }"
            "QLabel { color: #eee; }"
        )

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(2)

        # Header: Name + Buttons
        header_layout = QHBoxLayout()
        name_label = QLabel(self._get_display_name(filter_type))
        name_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        header_layout.addWidget(name_label)
        header_layout.addStretch()

        btn_style = "QPushButton { max-width: 16px; max-height: 16px; padding: 0px; font-size: 10px; }"

        self.btn_up = QPushButton("▲")
        self.btn_up.setStyleSheet(btn_style)
        self.btn_up.clicked.connect(self.moved_up.emit)
        header_layout.addWidget(self.btn_up)

        self.btn_down = QPushButton("▼")
        self.btn_down.setStyleSheet(btn_style)
        self.btn_down.clicked.connect(self.moved_down.emit)
        header_layout.addWidget(self.btn_down)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setStyleSheet(btn_style + "QPushButton { color: #ff5555; }")
        self.btn_remove.clicked.connect(self.removed.emit)
        header_layout.addWidget(self.btn_remove)

        self.layout.addLayout(header_layout)

        # Params Area
        self.params_layout = QHBoxLayout()
        self.layout.addLayout(self.params_layout)

        self._setup_params_ui()

    def set_reference(self, image: Any) -> None:
        self._reference_image = image
        self.params["reference_loaded"] = (image is not None)
        self._update_ref_button()
        self.changed.emit()

    def _update_ref_button(self) -> None:
        if hasattr(self, "btn_load_ref"):
            loaded = self._reference_image is not None
            self.btn_load_ref.setText("Remove Ref" if loaded else "Load Ref")
            if loaded:
                self.btn_load_ref.setStyleSheet("color: #ffaa00;")
            else:
                self.btn_load_ref.setStyleSheet("")

    def _get_display_name(self, type_: str) -> str:
        mapping = {
            "subtract": "Subtract",
            "divide": "Divide",
            "median": "Median (Hotpixel)",
            "min": "Minimum (Erosion)",
            "max": "Maximum (Dilation)",
            "gaussian_blur": "Lowpass (Gaussian)",
            "highpass": "Highpass",
            "clahe": "Local Norm (CLAHE)",
            "local_normalize_mean": "Local Norm (Mean)",
            "threshold_binary": "Threshold (Binary)",
            "clip_values": "Clip Values",
            "histogram_clipping": "Histogram Clip",
        }
        return mapping.get(type_, type_.capitalize())

    def _add_spinbox(self, key: str, label: str, val_type: type, min_: float, max_: float, step: float, default: float) -> None:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px;")
        row.addWidget(lbl)

        if val_type == int:
            sb = QSpinBox()
        else:
            sb = QDoubleSpinBox()

        sb.setRange(min_, max_)
        sb.setSingleStep(step)
        sb.setStyleSheet("font-size: 10px;")

        current_val = self.params.get(key, default)
        sb.setValue(current_val)

        sb.valueChanged.connect(lambda v, k=key: self._update_param(k, v))
        row.addWidget(sb)
        self.params_layout.addLayout(row)

    def _add_combobox(self, key: str, label: str, options: list[tuple[str, str]], default: str) -> QComboBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px;")
        row.addWidget(lbl)

        cb = QComboBox()
        cb.setStyleSheet("font-size: 10px;")
        for text, data in options:
            cb.addItem(text, data)

        current_val = self.params.get(key, default)
        idx = cb.findData(current_val)
        if idx >= 0:
            cb.setCurrentIndex(idx)

        cb.currentIndexChanged.connect(lambda i, k=key, c=cb: self._update_param(k, c.itemData(i)))
        row.addWidget(cb)
        self.params_layout.addLayout(row)
        return cb

    def _update_param(self, key: str, value: Any) -> None:
        self.params[key] = value
        self.changed.emit()

    def _setup_params_ui(self) -> None:
        t = self.filter_type

        if t in ("subtract", "divide"):
            self.cb_source = self._add_combobox("source", "Src:", [
                ("Aggregate", "aggregate"),
                ("Ref File", "file"),
            ], "aggregate")

            self._add_spinbox("amount", "Amt %:", float, 0.0, 100.0, 5.0, 100.0 if t == "subtract" else 100.0)

            # File Load Button (Visible only if source == file)
            self.btn_load_ref = QPushButton("Load Ref")
            self.btn_load_ref.setStyleSheet("font-size: 10px;")
            self.btn_load_ref.clicked.connect(self.load_reference_requested.emit)
            self.params_layout.addWidget(self.btn_load_ref)

            self._update_ref_button()

            # Update visibility based on source
            def _update_vis():
                is_file = self.params.get("source") == "file"
                self.btn_load_ref.setVisible(is_file)

            self.cb_source.currentIndexChanged.connect(_update_vis)
            _update_vis()

        elif t == "median":
            self._add_spinbox("ksize", "K:", int, 1, 99, 2, 3)

        elif t in ("min", "max"):
            self._add_spinbox("ksize", "K:", int, 1, 99, 1, 3)

        elif t in ("gaussian_blur", "highpass"):
            self._add_spinbox("sigma", "σ:", float, 0.1, 50.0, 0.5, 1.0)

        elif t == "clahe":
            self._add_spinbox("clip_limit", "Clip:", float, 0.1, 100.0, 0.5, 2.0)
            self._add_spinbox("tile_grid_size", "Grid:", int, 1, 64, 1, 8)

        elif t == "local_normalize_mean":
            self._add_spinbox("ksize", "K:", int, 3, 255, 2, 15)

        elif t == "threshold_binary":
            self._add_spinbox("thresh", "T:", float, -99999, 99999, 1.0, 0.5)
            self._add_spinbox("maxval", "V:", float, 0, 99999, 1.0, 1.0)

        elif t == "clip_values":
            self._add_spinbox("min_val", "Min:", float, -99999, 99999, 1.0, 0.0)
            self._add_spinbox("max_val", "Max:", float, -99999, 99999, 1.0, 255.0)

        elif t == "histogram_clipping":
            self._add_spinbox("min_percentile", "Min %:", float, 0.0, 100.0, 0.1, 1.0)
            self._add_spinbox("max_percentile", "Max %:", float, 0.0, 100.0, 0.1, 99.0)


class FilterStackWidget(QWidget):

    pipeline_changed = pyqtSignal()
    load_reference_requested = pyqtSignal(FilterItemWidget) # Pass the widget that requested it

    def __init__(self) -> None:
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Controls
        controls_layout = QHBoxLayout()
        self.cb_add = QComboBox()
        self.cb_add.addItems([
            "Median (Hotpixel)",
            "Subtract",
            "Divide",
            "Lowpass (Gaussian)",
            "Highpass",
            "Minimum (Erosion)",
            "Maximum (Dilation)",
            "Local Norm (CLAHE)",
            "Local Norm (Mean)",
            "Threshold (Binary)",
            "Clip Values",
            "Histogram Clip",
        ])

        # Map display names back to internal types
        self._type_map = {
            "Median (Hotpixel)": "median",
            "Subtract": "subtract",
            "Divide": "divide",
            "Lowpass (Gaussian)": "gaussian_blur",
            "Highpass": "highpass",
            "Minimum (Erosion)": "min",
            "Maximum (Dilation)": "max",
            "Local Norm (CLAHE)": "clahe",
            "Local Norm (Mean)": "local_normalize_mean",
            "Threshold (Binary)": "threshold_binary",
            "Clip Values": "clip_values",
            "Histogram Clip": "histogram_clipping",
        }

        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self._add_filter_from_ui)

        controls_layout.addWidget(self.cb_add, 1)
        controls_layout.addWidget(self.btn_add)
        self.layout.addLayout(controls_layout)

        # Scroll Area for stack
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.stack_container = QWidget()
        self.stack_layout = QVBoxLayout(self.stack_container)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        self.stack_layout.setSpacing(2)
        self.stack_layout.addStretch() # Push items up

        self.scroll.setWidget(self.stack_container)
        self.layout.addWidget(self.scroll)

        self._items: list[FilterItemWidget] = []
        self._block_signals = False

    def _add_filter_from_ui(self) -> None:
        name = self.cb_add.currentText()
        type_ = self._type_map.get(name)
        if type_:
            self.add_filter(type_)

    def add_filter(self, type_: str, params: dict | None = None) -> None:
        item = FilterItemWidget(type_, params)
        item.removed.connect(lambda: self._remove_item(item))
        item.moved_up.connect(lambda: self._move_item(item, -1))
        item.moved_down.connect(lambda: self._move_item(item, 1))
        item.changed.connect(self._on_change)
        item.load_reference_requested.connect(lambda: self.load_reference_requested.emit(item))

        # Add to layout before the stretch
        count = self.stack_layout.count()
        self.stack_layout.insertWidget(count - 1, item)
        self._items.append(item)

        self._on_change()

    def _remove_item(self, item: FilterItemWidget) -> None:
        self.stack_layout.removeWidget(item)
        item.deleteLater()
        if item in self._items:
            self._items.remove(item)
        self._on_change()

    def _move_item(self, item: FilterItemWidget, direction: int) -> None:
        if item not in self._items:
            return

        idx = self._items.index(item)
        new_idx = idx + direction
        if 0 <= new_idx < len(self._items):
            # Update internal list
            self._items.pop(idx)
            self._items.insert(new_idx, item)

            # Update Layout
            # We must remove and re-insert.
            self.stack_layout.removeWidget(item)
            # Note: insertWidget(index, widget).
            # layout indices map 1:1 to self._items indices because the stretch is at the end.
            self.stack_layout.insertWidget(new_idx, item)

            self._on_change()

    def _on_change(self) -> None:
        if not self._block_signals:
            self.pipeline_changed.emit()

    def get_pipeline(self) -> list[dict]:
        pipeline = []
        for item in self._items:
            step = item.params.copy()
            step["type"] = item.filter_type

            # Include the runtime reference object if present
            if item._reference_image is not None:
                step["reference"] = item._reference_image

            # For arithmetic, convert percentage amount to 0-1 factor
            if item.filter_type in ("subtract", "divide"):
                 # The spinbox is 0-100, we need 0.0-1.0
                 val = step.get("amount", 100.0)
                 step["amount"] = val / 100.0

            pipeline.append(step)
        return pipeline

    def set_pipeline(self, pipeline: list[dict]) -> None:
        self._block_signals = True
        # Clear existing
        while self._items:
            # We can't use _remove_item safely inside loop on self._items if modifying it
            # But here we just want to clear UI
            w = self._items.pop(0)
            self.stack_layout.removeWidget(w)
            w.deleteLater()

        for step in pipeline:
            type_ = step.get("type")
            if not type_:
                continue

            params = step.copy()
            del params["type"]

            # Convert factor back to percentage for UI
            if type_ in ("subtract", "divide"):
                val = params.get("amount", 1.0)
                params["amount"] = val * 100.0

            self.add_filter(type_, params)

        self._block_signals = False
        self.pipeline_changed.emit()

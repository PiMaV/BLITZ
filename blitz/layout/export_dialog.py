"""BLITZ export dialog: current frame, frame range, or full stack (.npy).

One UI for all environments. Frame-range prefers individual images into a
folder (native / GitHub artifact). Under Flatpak (home:ro + portal), sibling
files are not reliably writable — same dialog falls back to one ZIP and
tells the user why.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..tools import log


def running_in_flatpak() -> bool:
    """True inside a Flatpak sandbox (FLATPAK_ID is set by the runtime)."""
    return bool(os.environ.get("FLATPAK_ID"))


class ExportMode(Enum):
    CURRENT = "current"
    RANGE = "range"
    STACK_NPY = "stack_npy"


class ExportDialog(QDialog):
    """Choose what to export. Destination UI adapts to Flatpak vs native."""

    def __init__(self, n_frames: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self._n_frames = max(1, n_frames)
        self._flatpak = running_in_flatpak()

        layout = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.radio_current = QRadioButton("Current frame (image)")
        self.radio_range = QRadioButton("Frame range (numbered images)")
        self.radio_npy = QRadioButton("Full stack (.npy)")
        self.radio_current.setChecked(True)
        group = QButtonGroup(self)
        for r in (self.radio_current, self.radio_range, self.radio_npy):
            group.addButton(r)
            layout.addWidget(r)

        form = QFormLayout()
        self.spin_start = QSpinBox()
        self.spin_end = QSpinBox()
        for sp in (self.spin_start, self.spin_end):
            sp.setMinimum(0)
            sp.setMaximum(self._n_frames - 1)
        self.spin_end.setValue(self._n_frames - 1)
        form.addRow("First frame", self.spin_start)
        form.addRow("Last frame", self.spin_end)

        self.cmb_ext = QComboBox()
        self.cmb_ext.addItems(["png", "jpg", "jpeg", "tif", "bmp"])
        form.addRow("Image format", self.cmb_ext)
        layout.addLayout(form)

        self.radio_current.toggled.connect(self._sync_ui)
        self.radio_range.toggled.connect(self._sync_ui)
        self.radio_npy.toggled.connect(self._sync_ui)
        self._sync_ui()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_ui(self) -> None:
        is_range = self.radio_range.isChecked()
        is_img = self.radio_current.isChecked() or is_range
        self.spin_start.setEnabled(is_range)
        self.spin_end.setEnabled(is_range)
        self.cmb_ext.setEnabled(is_img)

        if self.radio_npy.isChecked():
            self.hint.setText("Writes one .npy file with the full loaded stack.")
        elif self.radio_current.isChecked():
            self.hint.setText("Writes one image of the current frame (LUT/levels as shown).")
        elif self._flatpak:
            self.hint.setText(
                "Flatpak sandbox: frame range is saved as one ZIP "
                "(numbered images inside). Individual files next to each other "
                "are not reliably writable under home:ro."
            )
        else:
            self.hint.setText(
                "Choose a folder; numbered images are written there "
                "(frame_000.png, …)."
            )

    def mode(self) -> ExportMode:
        if self.radio_npy.isChecked():
            return ExportMode.STACK_NPY
        if self.radio_range.isChecked():
            return ExportMode.RANGE
        return ExportMode.CURRENT

    def frame_range(self) -> tuple[int, int]:
        a, b = self.spin_start.value(), self.spin_end.value()
        return (min(a, b), max(a, b))

    def image_ext(self) -> str:
        return self.cmb_ext.currentText().lower()

    @property
    def in_flatpak(self) -> bool:
        return self._flatpak


def _ensure_suffix(path: Path, ext: str) -> Path:
    ext = ext if ext.startswith(".") else f".{ext}"
    if path.suffix.lower() != ext.lower():
        return path.with_suffix(ext)
    return path


def _write_processed_frame(viewer, frame_index: int, dest: Path) -> None:
    """Save one on-screen frame (levels + LUT) like pyqtgraph ImageView.export."""
    img = viewer.getProcessedImage()
    if viewer.hasTimeAxis():
        viewer.imageItem.setImage(img[frame_index], autoLevels=False)
    else:
        viewer.imageItem.setImage(img, autoLevels=False)
    viewer.imageItem.save(str(dest))


def _export_range_as_images(
    viewer, start: int, end: int, folder: Path, ext: str
) -> int:
    width = max(3, len(str(end)))
    count = 0
    for i in range(start, end + 1):
        dest = folder / f"frame_{i:0{width}d}.{ext}"
        _write_processed_frame(viewer, i, dest)
        count += 1
    return count


def _export_range_as_zip(
    viewer, start: int, end: int, zip_path: Path, ext: str
) -> int:
    width = max(3, len(str(end)))
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(start, end + 1):
                frame_path = tmp_path / f"frame.{ext}"
                _write_processed_frame(viewer, i, frame_path)
                zf.write(frame_path, arcname=f"frame_{i:0{width}d}.{ext}")
                count += 1
    return count


def run_export(viewer, parent: QWidget | None = None, last_dir: Path | None = None) -> None:
    """Show export dialog; frame-range uses folder images or Flatpak ZIP fallback."""
    if viewer.image is None:
        log("Nothing to export — no image loaded.", color="red")
        return

    n_frames = int(viewer.image.shape[0])
    dlg = ExportDialog(n_frames, parent=parent)
    if not dlg.exec():
        return

    mode = dlg.mode()
    directory = str(last_dir) if last_dir else ""
    prev_index = int(viewer.currentIndex)

    try:
        if mode == ExportMode.STACK_NPY:
            path_str, _ = QFileDialog.getSaveFileName(
                parent,
                "Export stack as NumPy",
                str(Path(directory) / "blitz_stack.npy"),
                "NumPy (*.npy)",
            )
            if not path_str:
                return
            out = _ensure_suffix(Path(path_str), ".npy")
            np.save(out, viewer.image)
            log(f"Saved stack → {out}", color="green")
            return

        ext = dlg.image_ext()
        if mode == ExportMode.CURRENT:
            path_str, _ = QFileDialog.getSaveFileName(
                parent,
                "Export current frame",
                str(Path(directory) / f"blitz_frame.{ext}"),
                f"Images (*.{ext});;All files (*)",
            )
            if not path_str:
                return
            out = _ensure_suffix(Path(path_str), ext)
            _write_processed_frame(viewer, prev_index, out)
            log(f"Saved frame {prev_index} → {out}", color="green")
            return

        # RANGE: native → folder of images; Flatpak → one ZIP + user feedback
        start, end = dlg.frame_range()
        if dlg.in_flatpak:
            msg = (
                "Running inside Flatpak: the sandbox cannot reliably write "
                "many sibling image files. The frame range will be saved as "
                "one ZIP archive with numbered images inside."
            )
            log(msg, color="yellow")
            QMessageBox.information(parent, "Flatpak export", msg)
            path_str, _ = QFileDialog.getSaveFileName(
                parent,
                "Export frame range as ZIP",
                str(Path(directory) / "blitz_frames.zip"),
                "ZIP archive (*.zip)",
            )
            if not path_str:
                return
            out = _ensure_suffix(Path(path_str), ".zip")
            n = _export_range_as_zip(viewer, start, end, out, ext)
            log(f"Saved {n} frames ({start}–{end}) as ZIP → {out}", color="green")
            return

        folder = QFileDialog.getExistingDirectory(
            parent,
            "Choose folder for numbered frame images",
            directory,
        )
        if not folder:
            return
        out_dir = Path(folder)
        n = _export_range_as_images(viewer, start, end, out_dir, ext)
        log(f"Saved {n} frames ({start}–{end}) → {out_dir}/", color="green")
    except PermissionError as e:
        log(
            f"Export failed (permission denied — Flatpak may need a portal path): {e}",
            color="red",
        )
    except OSError as e:
        log(f"Export failed: {e}", color="red")
    except Exception as e:
        log(f"Export failed: {e}", color="red")
    finally:
        try:
            viewer.setCurrentIndex(prev_index)
            viewer.updateImage()
        except Exception:
            pass

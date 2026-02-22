# Data Sources, Loaders & Build Variants

Architectural decisions for Standard vs. Full Build and the classification of Loaders, Converters, and Handlers.

---

## 1. Build Variants

BLITZ is designed to be distributed in two variants:

| Variant | Target Audience | Distribution |
|----------|------------|------------|
| **Standard** | Normal Users (99%) | Pre-compiled EXE via GitHub-Release |
| **Full** | Users with exotic formats/backends | Pre-compiled EXE via GitHub-Release |

> **Note on Repository State:** This repository currently represents the **Standard** build. Features marked as "Full" (OMERO, DICOM) are **not included** in the source code to keep the dependency footprint minimal.

**Background:** Most users receive a ready-made EXE. Installing via pip/uv is unrealistic for them. Therefore, plugins are not installed at runtime but delivered as build variants.

---

## 2. Classification Rule: Standard vs. Full

**Rule:** A feature (Loader, Converter, or Handler) belongs to **Standard** as long as it only uses libraries already included in the Standard Build. As soon as **exotic dependencies** are required, it becomes a **Full** candidate.

### Standard Dependencies (already in Standard Build)

- `csv`, `json`, `os`, `pathlib`, ... (stdlib)
- `numpy`
- `opencv-python-headless` (cv2)
- `PyQt6`
- `pyqtgraph`
- `natsort`
- `requests`
- `python-socketio`
- `psutil`
- `QDarkStyle`
- `numba` (Optional/Recommended)

### Exotic Dependencies (for Full)

- `omero-py` (OMERO)
- `pydicom` (DICOM files)
- `pynetdicom` (DICOM Server/PACS)
- `fhirclient` (FHIR Imaging)
- `openpyxl` / `xlrd` (Excel)
- `h5py` (HDF5)
- `bioformats` / Java-Bridge
- Other domain-specific packages

### Examples

| Feature | Type | Standard / Full | Reasoning |
|---------|-----|----------------|-------------|
| PNG, JPEG, TIFF, BMP | Loader | Standard | cv2 |
| Video (MP4, AVI, MOV) | Loader | Standard | cv2 |
| NumPy (.npy) | Loader | Standard | numpy |
| WebDataLoader (WOLKE) | Handler | Standard | requests, socketio |
| LiveView | Handler | Standard | requests/websocket |
| CSV Converter | Converter | Standard | csv, numpy |
| JSON Config Import | Converter | Standard | json, numpy |
| OMERO | Handler | Full | omero-py |
| DICOM File | Loader | Full | pydicom |
| DICOM Server (PACS) | Handler | Full | pydicom/pynetdicom |
| FHIR Imaging | Handler | Full | fhirclient |
| Excel Converter | Converter | Full | openpyxl |

---

## 3. Three Types of Data Sources

| Type | Task | GUI | Output |
|-----|---------|-----|---------|
| **Loader** | Read file and display directly as image | No (uses Standard File Dialog) | `ImageData` |
| **Converter** | Convert raw data into a usable format | Yes (custom dialog with preview, options) | `ImageData` or `.npy` file |
| **Handler** | Connect to external systems (Server, Databases) | Yes (custom dialog: Connection, Browsing, Selection) | `ImageData` or `.npy` file |

**Common Contract:** All eventually deliver `ImageData` or a path to `.npy`. BLITZ treats both cases uniformly.

---

## 4. Current State (Standard Build)

### Loaders (Implemented)
- **Images:** jpg, png, jpeg, bmp, tiff, tif
- **Video:** mp4, avi, mov
- **Arrays:** .npy

### Handlers (Implemented)
- **WebDataLoader:** WOLKE integration (Socket.IO + HTTP Download)
- **SimulatedLiveHandler:** Generates Lissajous/Lightning visualization as Live Stream
- **RealCameraHandler:** Real USB Webcam via cv2.VideoCapture (custom dialog with sliders)

### Converters (Implemented)
- **ASC/DAT Converter** (`blitz/data/converters/asc_dat.py`): Dialog with raw data preview, image preview, options.

---

## 5. Planned for Full (Not in this Repo)
- DICOM Loader (File)
- OMERO Handler
- DICOM Server / PACS Handler
- FHIR Imaging
- Additional exotic loaders (Daikon, Bio-Formats, ...) as needed

---

## 6. Technical Architecture (Planned)

### Common Interface
```python
# Simplified – all sources fulfill this:
class DataSource(Protocol):
    def provide(self, ...) -> ImageData | Path | None: ...

# Converter/Handler with own GUI:
class HasDialog(Protocol):
    def get_dialog(self, parent) -> QDialog: ...
    def run_and_provide(self, parent) -> ImageData | Path | None: ...
```

### Loader Registry
- Core defines Registry
- For unknown suffixes: Query Registry
- Standard Build: Only Core Loaders registered
- Full Build: Additional Loaders (DICOM, Daikon, ...) registered

---

## 7. Next Steps
1.  **Architecture Documentation** (this document) – Done.
2.  **OMERO Handler** – Own module, Full Build (External).
3.  **DataSource Interface** – Introduce common interface.
4.  **Loader Registry** – Framework for extensible loaders.

## References
- `docs/ARCHITECTURE.md` – General code architecture
- `docs/LOADING.md` – Load flow, dialogs, session defaults
- `docs/Tabs_explained.md` – User guide for features

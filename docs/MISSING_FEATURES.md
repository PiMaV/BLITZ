# Missing, Hidden, and Planned Features

This document outlines features that are mentioned in documentation or concepts but are currently missing, hidden, or not included in this repository's "Standard" build.

## Hidden Features (UI)

### Crop Widget (File Tab)
- **Status:** Hidden (`setVisible(False)` in `blitz/layout/ui.py`).
- **Description:** A widget in the "File" tab intended to allow destructive cropping of the dataset in memory.
- **Reason:** Deactivated due to complexity and potential user errors. Users are encouraged to reload with a subset if needed.

### Project Save/Load
- **Status:** Removed / Disabled.
- **Description:** Functionality to save the current session state (ROI, settings, loaded data path) to a `.blitz` project file.
- **Reason:** Temporarily removed to simplify the application state management. Loading `.blitz` files is supported in code but the UI to save/load is hidden/removed.

## Planned Features (Not in Standard Build)

### OMERO Integration
- **Status:** Planned for "Full" build.
- **Description:** A handler to connect to OMERO servers for remote image data access.
- **Reason:** Requires `omero-py` and other heavy dependencies not suitable for the lightweight Standard build.

### DICOM Support
- **Status:** Planned for "Full" build.
- **Description:** Native loading of DICOM files and communication with PACS servers.
- **Reason:** Requires `pydicom`, `pynetdicom`, and potentially `gdcm` which significantly increase the executable size.

### DataSource Interface
- **Status:** Planned / In Progress.
- **Description:** A unified interface for all data sources (Loaders, Converters, Handlers).
- **Reason:** Architectural improvement to support future plugins and extensions.

## Conceptual Features

### Autograd Engine
- **Status:** Concept only (see `docs/AUTOGRAD_POTENTIAL.md`).
- **Description:** A lightweight autograd engine for parameter optimization (e.g., auto-tuning filter parameters).
- **Reason:** Experimental idea, not yet implemented.

### Broken File Detection (ML)
- **Status:** Concept only.
- **Description:** Using small language models to detect corrupt file headers or structures before loading.
- **Reason:** Experimental idea, not yet implemented.

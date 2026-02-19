# Architecture & Code Analysis

## Overview

BLITZ is a Python-based image viewer and analysis tool built with **PyQt5** and **PyQtGraph**. It is designed to handle large datasets by loading them into memory (RAM) and offering efficient slicing, aggregation, and visualization.

## Directory Structure

*   **`blitz/`**: The main package.
    *   **`data/`**: Handles data loading, image processing, and in-memory representation (`ImageData`).
    *   **`layout/`**: Contains the UI logic.
        *   `main.py`: The `MainWindow` logic, connecting UI events to data operations.
        *   `ui.py`: The visual layout definition (Widgets, Layouts, Docks).
        *   `viewer.py`: Custom `ImageViewer` widget based on PyQtGraph.
    *   **`tools.py`**: Utility functions (logging, loading dialogs, RAM checks).
    *   **`settings.py`**: Settings management (possibly singleton-based).
    *   **`app.py`**: Application entry point logic.

## Key Components

### 1. Data Layer (`blitz/data`)
*   **`ImageData` (`image.py`)**: The core class wrapping a 4D Numpy array (`(Time, Width, Height, Channel)`). It handles:
    *   Lazy evaluation of crops, masks, and flips.
    *   Normalization (subtract/divide).
    *   Reduction (Mean, Max, Min, Std).
    *   **`image_timeline`**: Property fuer die Timeline im Aggregate-Modus – liefert vollen Stack (norm + mask) ohne Reduce. Siehe `docs/TIMELINE_AGGREGATION.md`.
*   **`DataLoader` (`load.py`)**: Responsible for reading files (Images, Video, DICOM, Numpy). It uses `multiprocessing.Pool` for parallel loading of large image sequences.

### 2. UI Layer (`blitz/layout`)
*   **`MainWindow` (`main.py`)**: The central controller. It initializes the UI, handles signals from widgets, and orchestrates calls to `DataLoader` and `ImageData`.
*   **`UI_MainWindow` (`ui.py`)**: strictly setup code for creating widgets and placing them in docks.
*   **`ImageViewer` (`viewer.py`)**: Custom wrapper around `pyqtgraph.ImageView` or similar, tailored for the specific interactions needed (ROI, time slicing).

## Current Issues / Technical Debt

1.  **Tight Coupling in `MainWindow`**:
    *   `MainWindow` directly accesses UI widgets (e.g., `self.ui.checkbox_norm_subtract`).
    *   It contains mixed logic: file dialog handling, settings synchronization, event handlers, and some business logic orchestration.
    *   This makes it a "God Class" that is hard to test and maintain.

2.  **`DataLoader` Responsibilities**:
    *   It mixes file format detection, UI logging (calls `log`), and raw data loading.
    *   The `from_text` method creates a dummy image with text, which is a bit of a hack for error reporting.

3.  **Global State/Singletons**:
    *   `tools.py` uses a global `LOGGER`.
    *   `settings.py` (inferred) likely acts as a global configuration store.

4.  **Error Handling**:
    *   Exceptions are often caught and logged to the UI, sometimes swallowing the stack trace or using generic error messages.

5.  **Project Management**:
    *   Poetry is used, but the `pyproject.toml` has commented-out dependencies.
    *   No formal test suite exists.

## Improvement Plan (Summary)

1.  **Migration**: Switch to `uv` for faster, cleaner dependency management.
2.  **Refactoring**:
    *   Extract logic from `MainWindow` into specialized handlers (e.g., `ProjectHandler`, `ViewSettingsHandler`).
    *   Decouple `DataLoader` from UI logging (return errors/status instead of printing directly).
3.  **Standardization**: Apply `ruff` for consistent code style.

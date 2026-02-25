# BLITZ 10-Minute Smoke Test Checklist

This document outlines a quick, manual smoke test procedure to verify the core functionality of the BLITZ desktop application. Run this checklist before any major release or after significant refactoring.

## Prerequisites

- **Source Code**: Ensure you have the latest version of the repository checked out.
- **Environment**: A Python environment with dependencies installed (e.g., via `uv sync`).
- **Test Data**:
  - A folder containing multiple image files (e.g., TIFF, PNG).
  - A large dataset (e.g., multi-GB TIFF stack or high-resolution video).
  - A `.npy` file (NumPy array).

## Launch & Initial State

1.  **Launch from Source**:
    - Open a terminal in the repository root.
    - Run: `uv run blitz` (or `python -m blitz`).
    - **Expected Outcome**: Application window appears within a few seconds. No critical errors in the terminal output.
    - **Verify**: Check the window title includes the version number.

2.  **Verify UI**:
    - **Expected Outcome**: Main window layout loads correctly (Viewer, Timeline, Options, LUT).
    - **Verify**: Status bar shows "IDLE".

## Core Functionality

### 1. Data Loading

-   **Load Folder**:
    -   Action: Drag and drop a folder containing images onto the main viewer area.
    -   **Expected Outcome**: Images load as a sequence. Timeline updates to show the correct number of frames.
    -   **Verify**: Scrub the timeline; image updates smoothly.

-   **Load Huge Dataset**:
    -   Action: File -> Open File (or Drag & Drop) -> Select your largest available dataset.
    -   **Expected Outcome**: Application remains responsive during load (or shows a progress dialog). RAM usage (displayed in status bar or OS monitor) increases reflectively but does not crash the system.
    -   **Verify**: Navigation (pan/zoom) is smooth after loading.

-   **Load .npy**:
    -   Action: File -> Open File -> Select a `.npy` file.
    -   **Expected Outcome**: Data loads correctly as an image sequence.
    -   **Verify**: Pixel values in the status bar match expected data range.

### 2. Visualization & Interaction

-   **Zoom/Pan**:
    -   Action: Use mouse wheel to zoom, right-click drag to pan.
    -   **Expected Outcome**: Image view updates smoothly.
    -   **Verify**: Rulers/Axes update to reflect new coordinates.

-   **Contrast/Colormap**:
    -   Action: In the LUT panel (right dock), adjust the Min/Max sliders. Change the colormap (e.g., to "Plasma" or "Viridis").
    -   **Expected Outcome**: Image contrast changes immediately. Colormap updates.
    -   **Verify**: Histogram updates to reflect the data distribution.

-   **Time Series Navigation**:
    -   Action: Drag the timeline slider. Use the "Current Frame" spinner in the Options dock.
    -   **Expected Outcome**: Displayed frame updates in real-time.
    -   **Verify**: Frame number in status bar matches the timeline position.

### 3. Application State & Persistence

-   **Settings Persistence**:
    -   Action:
        1.  Change a setting (e.g., toggle "Show Crosshair" off, or change the Theme).
        2.  Close the application.
        3.  Relaunch.
    -   **Expected Outcome**: The changed setting is remembered (Crosshair is still off / Theme is preserved).

-   **Close/Reopen**:
    -   Action: Close the main window.
    -   **Expected Outcome**: Process terminates cleanly in the terminal (exit code 0 or similar). No lingering processes.

## Troubleshooting & Logs

If any step fails:

1.  **Capture Logs**: Copy the entire output from the terminal where you launched the app.
2.  **Check `docs/optimization.md`**: For performance issues with large datasets.
3.  **Report**: Open an issue with the logs and a description of the failed step.

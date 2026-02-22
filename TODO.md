# TODO / Roadmap

## Known Issues (Bugs)

*   **Timeline / Aggregation:**
    *   Transition between Aggregate and Frame mode is not intuitive. Consider separate timelines?
    *   "Running Mean" has issues.
    *   Aggregate function is deactivated when starting a grayscale web stream (32 frames). Ops tab allows opening Aggregate, but Range is not initially FULL.
    *   Switching from Aggregate to Frame mode causes an error in some cases.
*   **Live View:**
    *   Background Subtraction in Live Stream causes error messages.
    *   RAM usage warning needed in Buffer Dialog.
    *   Display FPS is currently fixed at 10 FPS; should be configurable.

## High Priority Features

*   **DataSource Interface:** Unified interface for Loaders, Converters, and Handlers. Foundation for future plugins.
*   **Loader Registry:** Registry for extensible loaders.
*   **Polyline Intensity Profile:** Add a window/dock to display intensity distribution over a drawn polyline.

## Medium Priority

*   **Dual-Build Setup:** Ability to build Standard and Full EXE.
*   **CSV Converter:** Dialog with Preview, Column Selection -> .npy export.
*   **Tests:** Expand tests for `ReduceDict` edge cases.
*   **RoSEE:** Check Isolines and normalization (Autozoom should activate).
*   **Mouse Wheel Zoom:** Allow zooming only on one axis (in extraction plots?).
*   **Docker:** Verify Docker deployment.

## Low Priority / Long Term

*   **Video Optimization:** Strategy for handling compressed video data vs. full float32 matrices (mmap, Chunked, Range-Load). See `docs/OPTIMIZATION.md`.
*   **Project Files:** Restore Load/Save project file functionality (currently removed). See `docs/MISSING_FEATURES.md`.

## Notes on "Missing" Features

See [`docs/MISSING_FEATURES.md`](docs/MISSING_FEATURES.md) for details on features that are hidden, deactivated, or planned for the "Full" build (e.g., OMERO, DICOM, Crop Widget).

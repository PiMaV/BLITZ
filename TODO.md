# TODO / Roadmap

## Known Issues (Bugs)

- Mean wird doppelt berechnet?
- Rangwe panel bruacht mehr minimal width
- ...

## High Priority Features

*   **DataSource Interface:** Unified interface for Loaders, Converters, and Handlers. Foundation for future plugins.
*   **Loader Registry:** Registry for extensible loaders.
*   **Polyline Intensity Profile:** Add a window/dock to display intensity distribution over a drawn polyline.

## Medium Priority

*   **Dual-Build Setup:** Ability to build Standard and Full EXE (= Exotic Data flavours).
*   **Proper Exporter:** Dialog with Options to export as *.npy or image for Raw / Pipeline / PCA data
*   **RoSEE:** Check Isolines and normalization (Autozoom should activate).
*   **Mouse Wheel Zoom:** Allow zooming only on one axis (in extraction plots?).

## Low Priority / Long Term / Discussion

*   **Project Files:** Restore Load/Save project file functionality in a different way than before?. See `docs/MISSING_FEATURES.md`.

- LUT autorefresh / zoom?
- PCA Compoenten Table smaller 
- Binary Mask Eval?

## Notes on "Missing" Features

See [`docs/MISSING_FEATURES.md`](docs/MISSING_FEATURES.md) for details on features that are hidden, deactivated, or planned for the "Full" build (e.g., OMERO, DICOM, Crop Widget).

# TODO / Roadmap

## Known Issues (Bugs)

- deactivate PCA when aggragation images are seleccted
- LUT: nicht min max sondern perecntiles?



## High Priority Features

*   **DataSource Interface:** Unified interface for Loaders, Converters, and Handlers. Foundation for future plugins.
*   **Loader Registry:** Registry for extensible loaders.
*   **Polyline Intensity Profile:** Add a window/dock to display intensity distribution over a drawn polyline.

## Medium Priority

*   **Dual-Build Setup:** Ability to build Standard and Full EXE (= Exotic Data flavours).
*   **Proper Exporter:** Dialog with Options to export as *.npy or image for Raw / Pipeline / PCA data
*   **RoSEE:** Check Isolines and normalization (Autozoom should activate).
*   **Mouse Wheel Zoom:** Allow zooming only on one axis (in extraction plots?).
* Check Docker build

## Low Priority / Long Term / Discussion

*   **Project Files:** Restore Load/Save project file functionality in a different way than before?. See `docs/missing_features.md`.

- LUT autorefresh / zoom?
- PCA Compoenten Table smaller 
- Binary Mask Eval?
- reintroduce pickle for numpy?
- if data is grayscale, but image is rgb or such: identify from first image and simply load only first channel? Just as quick check

## Notes on "Missing" Features

See [`docs/missing_features.md`](docs/missing_features.md) for details on features that are hidden, deactivated, or planned for the "Full" build (e.g., OMERO, DICOM, Crop Widget).

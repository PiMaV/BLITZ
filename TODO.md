# TODO / Roadmap

## Known Issues (Bugs)


## High Priority Features

*   **DataSource Interface:** Unified interface for Loaders, Converters, and Handlers. Foundation for future plugins.
*   **Loader Registry:** Registry for extensible loaders.
*   **Polyline Intensity Profile:** Add a window/dock to display intensity distribution over a drawn polyline.

## Medium Priority

*   **Auto-Crop (Load Dialog):** Bounding box of content on MAX preview. Optional: isoline at threshold (user spinner 0-255) for visual feedback; margin % spinner; ROI updates on spinner change. No auto-apply on load (user controls). Simple threshold (flat > value) on MAX should work well.
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
- would it be nice to have a point within the main view if we are hovering along the vert and horz splines?
- Woudl it be ncie to have the envolope of the splines wtihin the main view?
- what about 3D? one could "simple" add an option to view 3D as a checkbox; should actually not be that complex for a first shot
- Calculating Mean should be checked; is kind of "slow" compared to the other OPs with small data sets

## Notes on "Missing" Features

See [`docs/missing_features.md`](docs/missing_features.md) for details on features that are hidden, deactivated, or planned for the "Full" build (e.g., OMERO, DICOM, Crop Widget).

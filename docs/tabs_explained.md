# Features & Tabs Explained

Source-backed inventory of BLITZ analysis, visualization, and data-processing capabilities, structured by UI surface (Options tabs, docks, menus). Algorithms and formats refer to the Python package under `blitz/`.

## Table of Contents

- [Layout Overview](#layout-overview)
- [Menu Bar](#menu-bar)
- [File Tab](#file-tab)
- [View Tab](#view-tab)
- [Ops Tab](#ops-tab)
- [Tools Tab](#tools-tab)
- [RoSEE Tab](#rosee-tab)
- [Shade Tab](#shade-tab)
- [PCA Tab](#pca-tab)
- [Bench Tab](#bench-tab)
- [Stream Tab](#stream-tab)
- [Log Tab](#log-tab)
- [LUT Dock](#lut-dock)
- [Probe Dock](#probe-dock)
- [H / V Extraction Plots](#h--v-extraction-plots)
- [Polyline Dock](#polyline-dock)
- [Timeline Panel](#timeline-panel)
- [Data Formats](#data-formats)
- [Export](#export)
- [Shortcuts](#shortcuts)
- [Numba Acceleration Map](#numba-acceleration-map)
- [Feature Matrix (Quick Reference)](#feature-matrix-quick-reference)

---

## Layout Overview

| Dock | Role |
|------|------|
| **Image Viewer** | Main matrix view (`ImageViewer`) |
| **H Plot** / **V Plot** | Crosshair line profiles (horizontal / vertical) |
| **Probe** | Frames count, cursor position, LUT-mapped color + swatch |
| **LUT** | Histogram LUT, levels, colormap, Fit / Trim, IDLE + Bench compact |
| **Options** | Tabbed control panel (File → Log) |
| **Timeline** | ROI time series + Frame \| Range side panel |
| **Polyline** | Path-intensity profile (starts hidden; Tools → Show) |

There is no separate toolbar; actions live in the menu bar, Options tabs, and docks.

---

## Menu Bar

| Menu | Action | Function |
|------|--------|----------|
| **File** | Open File | Browse single file (image / video / `.npy` / ASCII / HIKMICRO) |
| **File** | Open Folder | Browse folder; may open multi-group chooser |
| **File** | Load TOF | Import time-of-flight curve (`.json` / `.csv`) |
| **File** | Export | Current frame, frame range, or full stack (`.npy`) |
| **File** | Reset Window Layout | Restore default dock layout (restart) |
| **File** | Restart | Restart application |
| **Theme** | Dark (Tokyo Night) / Light (Hans Inverted Vampire) | Checkable; applies on restart |
| **About** | INP Greifswald / GitHub / M.E.S.S. | Open URLs |

---

## File Tab

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Load File / Load Folder** | Same as File menu browse | `DataLoader` + format converters |
| **Show load options dialog** | Dialog on every load / drop when checked; else last settings | Persist via settings |
| **8 bit** | Fixed dtype scale to 8-bit (RAM) | No per-image brightness stretch |
| **Normalize each image** | Per-frame stretch to full range | Independent of 8-bit; works for 8/16-bit and float |
| **grayscale** | RGB → luminance (default on) | RAM / speed |
| **size ratio** | Spatial downsample `[0, 1]` | Width & height scale |
| **subset ratio** | Temporal subsample `[0, 1]` | e.g. `0.1` ≈ every 10th frame |
| **max. RAM** | Cap load buffer | Bound by available RAM |
| **Crop** (hidden) | Destructive spatial crop | Hidden to avoid accidental data loss |
| **Metadata** | Name, size, dtype, bit depth, color, video, EXIF | Read-only probe of loaded stack |

**Drag & drop:** Drop files/folders onto the Image Viewer → `file_dropped` → `MainWindow.load`.

---

## View Tab

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Flip x / Flip y** | Mirror axes | In-memory transform on `ImageData` |
| **Rotate 90°** | Clockwise 90° | Matrix rotate / transpose path |
| **Display Mask — Show / Apply / Reset** | Exclude regions from display & analysis | ROI mask or binary image |
| **Load binary image** | Black/white mask | Applied via `image_mask` |
| **Crosshair Show / Markings** | Cursor cross + H/V plot markers | Linked to extraction plots |
| **Linked cursor** | Bidirectional hover sync image ↔ H/V plots | Pixel outline + curve highlight |
| **Line width H / V** | Average over N pixels for profiles | Thick crosshair sampling |
| **Min/Max per image** | Global frame extrema on plots | Per-frame |
| **Envelope per crosshair** | Temporal min/max along crosshair line | Over time at fixed geometry |
| **Envelope per position (dataset)** | Dataset extrema at cursor position | Full-stack envelope |
| **Envelope %** | Percentile band (0 = strict min/max) | `0–49%` |
| **Timeline ROI** | Rect / Polygon / Live Probe | Zonal mean/median, or pixel series + max-pins |
| **Update on drop** | Recompute ROI plot on mouse release | Auto for large ROIs; off in Live Probe |
| **Reset ROI** | Centered default (~10% size) | Live Probe: Clear pins + center |
| **Max pins** | Live Probe only (1–4, default 2) | Caps comparison pins |
| **TOF** | Overlay time-of-flight on timeline | Enabled after File → Load TOF; optional sync / invert / smooth |
| **Isolines Show / Count** | Intensity contours | `pg.IsocurveItem`; levels from mean ± std linspace |
| **Isoline Smoothing / Downsample** | Pre-filter before isocurves | OpenCV Gaussian + spatial downsample |

---

## Ops Tab

Real-time arithmetic pipeline on the `(T, H, W, C)` volume (`ImageData._apply_ops_pipeline`). Prefer Numba fused pass when available.

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Subtract Source** | Off / Range / File / Sliding range | `image -= amount · ref` |
| **Divide Source** | Off / Range / File / Sliding range | Denom blend: `amount·ref + (1−amount)` toward 1 |
| **Amount** sliders | Blend intensity `0–100%` | Independent for sub / div |
| **Range method** | Aggregate metric for Range / Sliding | **Mean, Max, Min, Std, Median** (`ReduceOperation`) |
| **Window / Lag** | Sliding temporal buffer | Window length + lag vs current frame |
| **Apply to full** | Materialize sliding result for whole stack | Shortens / rewrites timeline vs preview-only |
| **Load reference image** | External dark / flat | File source |
| **Crop Timeline / Undo** | Destructive temporal crop | Optional keep-in-RAM; undo if not purged |

**Range** uses the Timeline Aggregate selection (Start–End reduce). **Sliding range** uses a moving window with the chosen reduce method (Mean path is Numba-accelerated).

---

## Tools Tab

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Measure Tool — Show** | Draggable measurement ROI | Area, circularity, bounding H×W |
| **Display in au** + **Pixels / in au** | Pixel ↔ arbitrary-unit calibration | Scale factor for Measure + Polyline path |
| **Show Bounding Rect** | Overlay AABB | — |
| **Polyline Profile — Show** | Open Polyline dock + default diagonal path | See [Polyline Dock](#polyline-dock) |

---

## RoSEE Tab

**RoSEE** (*Robust and Simple Event Extraction*) — event bounds on 1D crosshair signals.

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Show RoSEE** | Enable overlay / calculation | RGB → luminance `0.2989 R + 0.5870 G + 0.1140 B` |
| **Use local extrema** | Bound by local peaks of fluctuation | Sign changes of `diff(fluctuation)` vs global min/max |
| **Smoothing** | Box smooth on fluctuation | Before extremum search |
| **Plots H / V** | Draw on extraction plots | Interval + eye index text fields |
| **Normalize values** | Normalize for display | `normalize` / `unify_range` |
| **Show Indices / Lines** | Markers on plots | Eye = `argmax(signal)` |
| **Image H / V** | Project bounds into full image | Per-row / per-column `calculate_all` |

**Core signal path:** `cumsum(signal)`; `fluctuation = cumsum(signal − mean)`; eye = peak; bounds from fluctuation extrema.

---

## Shade Tab

View-only **hillshade** (relief / Schummerung). Analysis buffer remains height/intensity.

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Preview** | Toggle Lambertian shade overlay | Separate `ImageItem` (z≈0.1); base opacity → 0 while previewing |
| **Azimuth** | Light direction `0–360°` | `0°` = top (+y), clockwise toward +x |
| **Elevation** | Sun height `0–90°` | Horizon → zenith |
| **Z factor** | Vertical exaggeration `0.1–20` | Scales height before `np.gradient` |

**Math (`calculate_hillshade`):** height from grayscale or luminance `0.299R+0.587G+0.114B` → `dx, dy = ∇z` → unit normal · light vector → shade clipped to `[0, 1]`.

---

## PCA Tab

Principal Component Analysis via **SVD** on the flattened stack `(T, features)`.

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Target Comp** | Number of components `1–500` | Capped by frame count |
| **Exact (Slow)** | Full SVD vs randomized | Exact: `np.linalg.svd` on centered data; Approx: Halko randomized SVD without forming `A−μ` explicitly |
| **Calculate PCA** | Background compute | Returns `U, s, Vh, mean` |
| **View — Reconstruction** | `U[:,:k]·diag(s[:k])·Vh[:k]` [+ mean] | Truncated reconstruction |
| **View — Components** | Eigenimages from rows of `Vh` | Spatial modes |
| **Include mean** | Add mean image to reconstruction | Off → deviation-only |
| **Variance plot / table** | Explained variance | Fractions from `s²`; cumulative + individual |

Requires `T ≥ 2` and non-aggregate view. Approximate defaults: `n_oversamples=10`, `n_iter=2`.

---

## Bench Tab

| Control | What it shows |
|---------|---------------|
| **Show CPU load** | Sparkline under LUT IDLE |
| **Raw / Result matrix** | Buffer shapes / presence |
| **View mode** | Current display mode |
| **Result cache** | Ops/reduce cache status |
| **Numba** | JIT active / fallback |
| **Live** | Live-stream indicator |

---

## Stream Tab

| Control | What it does | Notes |
|---------|--------------|-------|
| **Generate Synthetic Live Data Stream** | Lissajous / Lightning ring buffer | No hardware; FPS, resolution, grayscale, exposure knobs |
| **Webcam** | USB camera via OpenCV | Exposure, gain, brightness, contrast; ring buffer |
| **Network Address / Token** | Remote `.npy` ingest | Socket.IO + HTTP download; sync viewer index / selection |

---

## Log Tab

Scrollable application logger (`LoggingTextEdit`).

---

## LUT Dock

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| **Min / Max** | Manual levels | Synced to histogram |
| **Histogram LUT** | Interactive transfer function | pyqtgraph `HistogramLUTWidget` |
| **Fit now / Keep fitting** | Auto levels | `calculate_lut_levels` — nanmin/max or percentile |
| **Trim** | 0% / 1% / 2% / Custom | Percentile clip |
| **Colormap + Auto** | Gradient selection | pyqtgraph Gradients |
| **Log hist** | Log-scale histogram | — |
| **Load / Export LUT** | Wired in code | Currently **hidden** (`setVisible(False)`) |

---

## Probe Dock

| Field | Content |
|-------|---------|
| **Frames** | Stack length |
| **Position** | Cursor `X`, `Y` |
| **Color** | LUT-mapped value (RGB or grayscale) + swatch |

**Ctrl+C** copies Frames + Position + Color lines to the clipboard.

---

## H / V Extraction Plots

Profilschnitte along the crosshair:

- Mean intensity along horizontal / vertical line (optional line width).
- Optional min/max and temporal/dataset envelopes (View tab).
- Coupled axes; linked cursor with the image.
- RoSEE overlays when enabled.

---

## Polyline Dock

Opened via **Tools → Polyline Profile → Show**.

| Control | What it does | Algorithm / notes |
|---------|--------------|-------------------|
| Path ROI | Open polyline handles on image | Dense samples along segments (`step ≈ 1 px`) |
| **Width** | Perpendicular half-width (px) | Mean intensity across ⊥ band |
| **Envelope ⊥** | Min/max (or %) across band | Optional corridor overlay |
| **Envelope (frames)** | Min/max along centerline over `T` | Dataset volume sample |
| Path length | Arc length along path | Pixels or Measure AU calibration |
| **CSV** | Export profile | `s`, intensity, envelopes |
| Linked cursor | Sync with image hover | Nearest sample on path |

Core: `sample_polyline_profile` → `PolylineProfileResult(s, intensity, xs, ys, env_*)`.

---

## Timeline Panel

Bottom dock + Frame \| Range side panel.

### Frame mode

| Control | Role |
|---------|------|
| **Idx** | Current frame |
| **Curve** | ROI aggregation: Mean / Median (disabled in Live Probe) |
| **Bands** | Upper/lower envelope on timeline (ROI modes) |

### Aggregate (Range) mode

| Control | Role | Algorithm |
|---------|------|-----------|
| **Reduce** | Collapse time window | None / **Mean / Max / Min / Std / Median** |
| **Start / End** | Frame bounds | Inclusive range |
| **Win / Win const.** | Fixed window size while dragging | — |
| **Live drag** | Update while dragging | Costly on large stacks |
| **Full Range** | Reset to entire dataset | — |

Aggregate result feeds Ops **Range** subtract/divide sources.

---

## Data Formats

### Import

| Kind | Extensions / notes |
|------|--------------------|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif` |
| Video | `.mp4`, `.avi`, `.mov` |
| Arrays | `.npy` (2D / 3D / 4D; pickle disabled) |
| ASCII matrices | `.asc`, `.dat`; `.txt` if matrix-like (Tab / Space / Comma; optional row-number column) |
| HIKMICRO radiometric | JPEG HDRI trailer → linear DN→°C (`float32`) |
| TOF | `.json` (`offset`, `tof_data`) / `.csv` (cols 0,1) |
| Folder groups | Images (naming clusters), video, array, ascii, hikmicro, other — chooser when mixed |

### Load-time transforms (dialogs)

Frame range / step (video), resize, 8-bit, grayscale, normalize, pre-load ROI + Flip X/Y/XY, mixed-size policy.

---

## Export

**File → Export**

| Mode | Output |
|------|--------|
| **Current frame** | Single image with on-screen LUT/levels |
| **Frame range** | Numbered `frame_XXX.{ext}` (Flatpak → one ZIP) |
| **Full stack** | Raw `np.save` of viewer volume as `.npy` |

Image formats: **png, jpg, jpeg, tif, bmp**.

**Also:**

| Source | Format |
|--------|--------|
| Polyline dock | CSV path profile |
| LUT Load/Export | Implemented but UI-hidden |
| Measure Tool | On-screen stats only (no file export) |

---

## Shortcuts

| Binding | Action |
|---------|--------|
| **Ctrl+C** | Copy Probe text (Frames, Position, Color) to clipboard |

No other application-level `QShortcut` / `setShortcut` bindings. PyQtGraph / ImageView library defaults may still apply.

---

## Numba Acceleration Map

When Numba is available (`HAS_NUMBA`), parallel `nopython` kernels (`fastmath`, cache unless frozen):

| Kernel | Used by |
|--------|---------|
| `apply_pipeline_fused` | Fused Subtract + Divide |
| `sliding_mean_numba` | Sliding-window **Mean** normalization |
| `_reduce_mean/max/min/std_impl` | Timeline / Ops axis-0 reduce |

**Not Numba:** Median reduce (threaded NumPy `nanmedian`); non-Mean sliding aggregates (loop + reduce). Fallback: ThreadPoolExecutor split on height for large reduces (>10 MB).

Status visible in **Bench → Numba**.

---

## Feature Matrix (Quick Reference)

| Feature | UI location | Core math / ops | I/O |
|---------|-------------|-----------------|-----|
| Bulk load + DnD | File, Viewer drop | Resize, 8-bit, normalize, subsample | img / video / npy / ascii / HIKMICRO |
| Flip / rotate / mask | View | Axis transforms, binary mask | Binary mask image |
| Crosshair Profilschnitte | View + H/V docks | Line average, envelopes | — |
| Isolines | View | Isocurves + Gaussian / downsample | — |
| ROI timeline | View + Timeline | Zonal mean/median ± bands, or Live Probe pixel series | — |
| TOF overlay | File menu + View | Interp / invert / box smooth | json, csv |
| Subtract / Divide | Ops | Arithmetic + reduce refs | Reference image |
| Sliding window | Ops | Temporal Mean/Max/Min/Std/Median | — |
| Timeline crop | Ops | Temporal slice | — |
| Measure | Tools | Area, circularity, bbox | AU calibration |
| Polyline profile | Tools + Polyline dock | Path sample + ⊥ band stats | CSV |
| RoSEE | RoSEE | Cumsum fluctuation extrema | — |
| Hillshade | Shade | Lambertian `n·l` from `∇z` | View-only |
| PCA / SVD | PCA | Exact or randomized SVD | — |
| LUT levels | LUT dock | Percentile / min-max | (LUT export hidden) |
| Live / webcam / net | Stream | Ring buffer / OpenCV / Socket.IO | npy over network |
| Export | File → Export | LUT-baked frames or raw stack | png/jpg/tif/bmp/npy/zip |
| Perf / Numba | Bench | JIT status, cache, CPU sparkline | — |

# Architecture & Code Analysis

Machine-oriented product brief: [`docs/llm-brief.md`](llm-brief.md).

## Data Sources & Build Variants

Fuer Standard vs. Full Build, Loader/Converter/Handler und die Dependency-Regel siehe **`docs/sources_and_variants.md`**.

---

## Overview

BLITZ is a Python-based image viewer and analysis tool built with **PyQt6** and **PyQtGraph**. It is designed to handle large datasets by loading them into memory (RAM) and offering efficient slicing, aggregation, and visualization.

Network `.npy` ingest stays a dense blob. The Event reader gzip is opt-in and
off on loopback. The Connect button and the **NET** / **BUSY** badge show
download percent so a large stack is not silent. After load, `ImageData` is
still a dense NumPy cube. Contrast is a LUT/display concern (Fit / Trim),
not a load-time threshold.

```mermaid
flowchart LR
  subgraph produce [Producer]
    evt["EVT bin then npy float32 default"]
    wolke["WOLKE np.save"]
  end
  subgraph wire [HTTP]
    A["A: optional gzip of dense npy"]
  end
  subgraph blitzLoad [BLITZ]
    dl["DataLoader"]
    idata["ImageData dense RAM"]
  end
  evt --> A
  wolke --> A
  A --> dl
  dl --> idata
```

See [`../WETTER/docs/sparse_matrices.md`](../WETTER/docs/sparse_matrices.md).

## Cube measurand rule

BLITZ analyzes **one** dense `ImageData` cube. Features must not pretend a
different physical quantity is that cube.

```mermaid
flowchart TD
  q{"Same measurand as ImageData?"}
  q -->|"yes: lighting, LUT, contours, RoSEE bounds"| preview["Preview overlay — never bake into the cube"]
  q -->|"yes: Ops, reduce, flip, PCA reconstruction"| ops["Preview and/or Apply on the cube"]
  q -->|"no: arrival time, flow, eigenimages, event lists"| neu["New cube: outside BLITZ, or Apply a new ImageData"]
  q -->|"not a cube: TOF, extra sensors"| aux["Auxiliary 1D beside the cube"]
```

- **Same measurand, display only** — Preview. Shade hillshade, LUT, isolines,
  D8 flow accumulation. Probe / polyline / PCA keep reading height or intensity. **Shade must not
  Apply-replace the stack** (that would store lighting `0…1` as if it were the
  measurement).
- **Same measurand, arithmetic / reduce** — Preview and/or Ops **Apply to full**.
  Load 8-bit / grayscale / subset chooses the cube at ingest; that is not an
  overlay.
- **New measurand** — Produce the cube **outside** (EVT / converters / a later
  sidecar) or **Apply** a distinct `ImageData` the user can leave. Do not overlay
  it on the original. PCA **View components** already swaps in eigenimages and
  restores via **View PCA** off — that is Apply-as-view-swap, not a Shade-style
  overlay.
- **Auxiliary 1D** — TOF today; more sensor curves later. Not a cube overlay.

Hillshade **Pre-cache** is paint optimization only (not a new measurand, not
Apply). Azimuth `0°` is screen north (top of the image). The viewer ViewBox
uses `invertY`, so the light vector is `ly = -cos(az)` (same as gdaldem);
east/west stay `lx = sin(az)`. Freeze elevation and Z factor, then build an
azimuth atlas for the **full current `T` frame** on a worker thread (step
5–90°, default 30°; 5° is the finest). Shade Preview paints the **entire
frame** once; pan/zoom only transforms the ImageItem (no recompute). Flow
still uses a ViewBox crop for D8 (separate story). Combined mode blends
independent coloured lights (Preset = four at 90°, same elevation). For two
or more lights, each Lambertian shade is pushed away from the mean
(`base + gain × (shade − base)`, default gain 8) before tinting — otherwise
four lights on gentle **metre-scale** DEMs cancel to a flat green–amber wash.
DGM **u16dm** stores **absolute** decimetres (1 DN = 0.1 m, `z0=0`: 403 m →
4030). The ×10 scale makes Combined look crisp but stairs at 0.1 m; prefer
**f32 metres** for Schummerung when you want smooth metre slopes. Live
Preview / Combined compute on a worker thread; a thin progress bar on the
Shade tab is indeterminate while shading and determinate while Pre-cache
fills. Optional later: cache the current angles over `T` so timeline scrub
stays smooth.

### Pixel identity (overlays)

No implicit spatial resampling in analytical or derived views. A visible
source pixel must never be discarded merely for rendering performance.

The main viewer already passes the full frame to `ImageView`. **Shade**
computes on the full height frame (source resolution). **Flow** may still
crop to the visible ViewBox. Neither builds a smaller working matrix for
speed (the old 1600-px cap is gone). Zoomed-out **paint** may downsample the
display `ImageItem` after the ViewBox settles (~80 ms) — never blank, never
on every mouse tick. Probe / Shade / Flow still read `ImageData`. Shade
Preview must not hide the cube (`opacity 0`). At **native zoom**, 1 matrix
pixel = 1 **physical** display pixel (`devicePixelRatio`; Qt logical ≠
physical).

Test rasters (stripes, checkerboard) are not generated in-core. A later
suite sidecar streamer can emit `.npy` (+ optional spatial JSON), same
path as `dgm-mosaic` / WOLKE. Conway and Synthetic Live stay the only
in-core stream families.

LUT HUD under IDLE shows **UI event-loop lag** (250 ms probe; label = median
~1 s; amp = last 30 s at fixed 0–300 ms with green/yellow/red phases) — not
paint FPS and not on the viewer paint path. Shade/Flow status still reports
compute `1/dt` — that is not UI lag.

Keep fitting on a timeline tick uses the **displayed frame**, not the whole
cube, and skips `setLevels` when Min/Max are unchanged. The LUT dock does
not auto-range on that tick (that used to fight Auto pins and double-paint).

```mermaid
flowchart TD
  tick["Timeline frame tick"]
  tick --> paint["ImageItem paint current frame"]
  tick --> kf{"Keep fitting on?"}
  kf -->|"no"| done["LUT unchanged"]
  kf -->|"yes"| frame["levels from current T slice"]
  frame --> same{"Min/Max unchanged?"}
  same -->|"yes"| done
  same -->|"no"| set["setLevels + histogram X"]
```

Shade does **not** hook `sigRangeChanged`. The Shade RAM line is a Pre-cache
hint: `psutil.virtual_memory` at 1 Hz while Preview/Pre-cache is on, sized
from the **full frame**. The hard RAM gate runs once when the user enables
Pre-cache.

```mermaid
flowchart LR
  timer["QTimer 250ms"] --> tick["expected vs actual"]
  tick --> lag["lag_ms"]
  lag --> hud["LUT UI ms + amp"]
```

```mermaid
flowchart TD
  src["Full ImageData frame"] --> shade["Hillshade ∇z full frame"]
  src --> crop["ViewBox crop + 1 px halo"]
  shade --> itemS["Shade ImageItem — pan/zoom = transform"]
  crop --> flow["D8 at source resolution"]
  flow --> itemF["Flow ImageItem.setRect"]
  itemS --> gpu["Renderer maps onto monitor pixels"]
  itemF --> gpu
```

```mermaid
flowchart TD
  dome["Sky dome"] --> az["Around: azimuth 0-360 wrap"]
  dome --> el["Radius: elevation 0 rim to 90 centre"]
  dome --> zf["Grey peg shadow opposite the sun; length from Z and elevation"]
  az --> lights["One sun or four independent"]
  el --> lights
  lights --> gain["Multi-light: amplify deviation from mean shade"]
  gain --> rgb["Mean of tinted boosted shades"]
  zf --> grad["∇ of z times Z then shade"]
  rgb --> overlay["Full-frame overlay"]
  grad --> overlay
```

```mermaid
flowchart TD
  z["Height per pixel"] --> gradZ["dx dy = gradient of z times Z"]
  gradZ --> lights4["Four Lamberts 90 deg apart"]
  lights4 --> meanTint["Mean of four colours"]
  meanTint --> flat["Float32 metres: near-constant tint"]
  lights4 --> clipNL["Clip 0 to 1"]
  clipNL --> crisp["UInt16 dm: steep edges plus 0.1 m stairs"]
  lights4 --> boost["Direction gain then tint"]
  boost --> relief["Readable Combined on f32"]
```

```mermaid
flowchart TD
  click["Preview or Combined"] --> bar["Progress bar on; status Shading"]
  bar --> liveW["QThread one frame"]
  liveW --> paint["Overlay set; bar off"]
  liveW --> cancel["New params: bump generation"]
```

```mermaid
flowchart TD
  setup["Set elevation and Z"] --> freeze["Check Pre-cache"]
  freeze --> lock["Lock elev and Z; snap az to step"]
  lock --> worker["QThread: normals once, then N az bins"]
  worker --> hit{"Azimuth bin ready?"}
  hit -->|yes| swap["setImage from atlas"]
  hit -->|no| wait["Keep last overlay; status Caching"]
  worker --> done["All 12 ready"]
  uncheck["Uncheck Pre-cache"] --> live["Live full-frame recompute"]
```

```mermaid
flowchart TD
  patch["Viewport height patch"] --> d8["Steepest of 8 neighbours"]
  d8 --> acc["Accumulate cell counts high to low"]
  acc --> rgba["log1p cyan RGBA"]
  rgba --> overlay["Overlay on height or Shade"]
```

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
    *   **`image_timeline`**: Property fuer die Timeline im Aggregate-Modus – liefert vollen Stack (norm + mask) ohne Reduce. Siehe `docs/timeline_aggregation.md`.
*   **`DataLoader` (`load.py`)**: Responsible for reading files (Images, Video, Numpy). It uses `multiprocessing.Pool` for parallel loading of large image sequences.

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
    *   `uv` is used for dependency management (pyproject.toml, uv.lock).
    *   No formal test suite exists.

## Improvement Plan (Summary)

1.  ~~**Migration**: Switch to `uv` for faster, cleaner dependency management.~~ (erledigt)
2.  **Refactoring**:
    *   Extract logic from `MainWindow` into specialized handlers (e.g., `ProjectHandler`, `ViewSettingsHandler`).
    *   Decouple `DataLoader` from UI logging (return errors/status instead of printing directly).
3.  **Standardization**: Apply `ruff` for consistent code style.

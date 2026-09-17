# TODO / Roadmap

Prioritized after a long break: **P0 first**, then bugs people feel daily, then features, then architecture, then open ideas.
Human-readable notes under each item so we remember *what* and *why* without re-diving into the code.

WOLKE-related work lives in its own section at the bottom — **no priority**, keep it separate from the BLITZ core queue.

Related: [`docs/missing_features.md`](docs/missing_features.md) (hidden UI / concepts).

---

## Parked 2026-08-22 — core viewer first

Stop here. Next session: taste-test pan/zoom on the **Load data** splash and a
large file, Preview **off**. Do not add Shade/Flow work until the core viewer
feels right.

**Rule:** BLITZ is a fast viewer. Shade/Flow are optional overlays. When
Preview is off they must not sit on pan/zoom/load.

**What we learned**
- Overlay `1/dt` is not display FPS / UI lag. Paint-FPS on the viewer paint
  path stays off the table; LUT HUD uses event-loop lag instead (always on).
- In-core File-tab test patterns: **removed** (sidecar streamer later, not a
  third in-core generator).
- Shade leaked onto pan via always-on `sigRangeChanged` + RAM/`psutil` +
  stylesheet. Partly peeled; not fully inert when off (extra ImageItems;
  timeline/`image_changed` slots still fire and return).
- Pixel-identity (no 1600-px working copy) stays. **Shade** is full-frame
  (recompute on az/elev/Z/frame only). **Flow** still viewport — same lag
  tradeoff until we decide.
- Immediate `autoDownsample` made the splash pan badly — do **not** turn it
  on raw. Display downsample is **debounced** (~80 ms after ViewBox settle).
  Shade Preview must not hide the cube.

**Dirty tree (uncommitted, mixed):** pixel-identity tests/docs, Shade/Flow
debounce + RAM 1 Hz + range hook only while Preview on, pattern files gone,
UI-lag HUD.

**Not built:** sidecar streamer, SpatialMetaData, 20k DGM bench, GPU.

**Resume:** (1) restart, splash pan, then large file Preview off (2) if still
wrong, isolate Shade/Flow completely when off (3) autoDownsample only if needed.

---

## Recently done (this stretch)

- **v2.2.1** Timeline dock opens at a usable height when T>1 (EVT/live/video);
  RGB event stacks keep encoded LUT range (0…1 float / 0…255 uint8).
- **UI lag HUD** under LUT IDLE (event-loop probe + amp); CPU/RAM stay Bench-only.
- **Parked core-viewer taste test** — see above (2026-08-22).
- **v2.1.1** Close-safe Shade/Flow timers; RGB cursor swatch (gray stays LUT);
  load-dialog **Auto-crop**; Timeline dock comes back when a stream grows past T=1.
- **v2.1.0** Shade sky dome (azimuth / elevation / Z, Combined colours,
  invertY north, Pre-cache wrap through 0°) and D8 **Flow** accumulation
  overlay (viewport veins, Preview-only). Seed path / fill / legend parked.
- **Network index-only seek:** WOLKE table click on an already-loaded
  `__selection__.npy` only changes the current frame (no `set_image`, no
  `reset_options`, no BUSY). New stacks still go NET → BUSY → IDLE.
- **v2.0.15** Network ingest: **NET** / percent while the `.npy` downloads, **BUSY**
  while it opens (large Event-reader stacks were silent before). Loopback
  downloads skip gzip (`Accept-Encoding: identity`).
- **v2.0.14** Network `.npy` load: pickle crash when stacks exceeded the
  multicore size threshold (`Can't get local object DataLoader._load_array…`)
- **v2.0.13** Conway Game of Life (Stream sibling to Synthetic Live): Classic B3/S23
  ring buffer, Ember decay `0…N`, pattern preview, LUT pin; docs + backlog notes
- **Folder load chooser**: mixed folders → pick group (images / video / npy / ASCII / HIKMICRO °C); naming-schema clusters; file list + preview
- **HIKMICRO** radiometric JPEG → approx. °C in-core (numpy); ASCII/`.txt` grids via converter; exotic formats policy → `docs/exotic_formats.md`
- **Mixed HxW** in one folder: crop-to-common-min vs cancel (`MixedImageSizesDialog` + `mixed_size_policy`)
- Load preview: **MAX default**; Single/MAX + Normalize; thread-safe preview workers (no QThread destroy on rapid toggle)
- File tab / load dialogs: **8 bit & grayscale** visibly grayed when source is already native
- Load dialog UX: **Normalize each image/frame** vs **Preview normalize** (display-only, HLine); Flip X/Y + Transpose; ROI handle drag fix; Timeline opens for T>1 after load
- Status / Probe dock rename; MetaData block at bottom of File tab (EXIF still placeholder)
- Dock layout restore: ignore incompatible saved state (e.g. after Probe rename); bottom band sizing fixed for large screens
- Timeline dock **hidden for single-frame** data; Frame|Range side panel compact (side-by-side)
- Webcam: list only working capture devices; silence OpenCV probe WARNs; Linux V4L2
- OpenCV pin `<4.14` (4.14.0.94 had Windows-only wheels, broke Linux `uv run`)
- Extraction plots: wheel zooms spatial axis only; Alt+wheel = intensity axis
- Linked cursor (View → Crosshair): spatial markers from image; curve point + value tip when hovering H/V; thicker 1×1 pixel outline
- **v2.0.8** Polyline intensity profile (Tools → Show):
  - open path ROI under image / above timeline; dock plot `s` vs mean-in-band
  - Width = band mean; Envelope ⊥ / frames; CSV; path in au; handle numbers on path axis
  - amber sync point on curve + value tips; compact axes + Stats in toolbar
  - optional docks only when relevant (T>1 / Show on); dock_layout_rev 5
- **v2.0.9** LUT dock UX:
  - heading + Min/Max labels; dtype-aware level spinners
  - Fit now / Trim % (0/1/2/Custom) / Keep fitting + status line
  - Colormap combo; fit-on-load (also after load-dialog transpose/flip)
- **v2.0.10** Polyline Show: relative diagonal default across the image; dock
  re-placed under the viewer so it is not hidden by the Timeline
- **v2.0.11** Shade tab (hillshade): view-only overlay after RoSEE; Azimuth /
  Elevation / Z factor; Crosshair & Polyline keep sampling height (not shade)
- **v2.0.11** Docs: `docs/features.md` hero list + refreshed `tabs_explained.md`
- **v2.0.12** Live Probe timeline: pixel pointer (+ multi-pin) alongside ROI;
  opaque signed Δ bar (Live−P1 / P2−P1) under the frame timeline

---

## P0 — Before the next push

Taste-test pan/zoom after today’s cleanup. Shade/Flow stay off the core path.

---

## P1 — Bugs / UX you notice every day

### SIGSEGV after failed Network Connect + Log tab

Native crash (`SIGSEGV`), no Python traceback. Repro: Connect to dead host → open
**Log**. Cause: worker-thread `log()` into `QTextEdit`. Mitigation in
`blitz/tools.py` (queued marshal) — **re-test**; see `BUGS.md` +
`docs/agent_handoff_sigsegv_log.md`.

### SpatialMetaData + File-tab / world cursor (after pixel-identity)

Optional `MetaData.spatial` sidecar only — do **not** dump GIS fields onto
`MetaData`. Cube stays `.npy`. Display in the existing File metadata block
when present (CRS, origin, resolution, NoData). Then Probe: Pixel X/Y plus
World E/N via affine / GeoTransform. Socket payload may carry `spatial`
next to `file_name` / `index`. After that: 20k×20k DGM bench (honest full-res
shade, not a 1600-px stand-in).

---

## P2 — Features with clear user value

### Live Probe: lagged cross-correlation

Zero-lag Pearson + difference/similarity strip already ship with Live Probe.
Optional later: full cross-correlation (lag axis / peak lag / r_max) for delayed fronts — not started.

Relevant: `TimelineProbeController`, `ImageViewer` probe similarity path.

### RoSEE: isolines + normalization

Check isolines/Normalize after `IsolineAdapter` split; autozoom when appropriate.

Relevant: `blitz/layout/rosee.py`, `blitz/layout/isoline.py`.

### Image smoothing

General Ops/display spatial blur (today only RoSEE/isolines/TOF). Gaussian/box/temporal — see `docs/numba_candidates.md`. **Not a Shade-tab control** — belongs in Ops/View when implemented; Shade stays lighting-only.

### Sidecar streamer for test rasters

Stripes / checkerboard for pixel-identity Shade/Flow checks belong in a
**suite sidecar streamer** (`.npy` + optional spatial JSON), same path as
`dgm-mosaic` / WOLKE — **not** a third in-core generator. Conway + Synthetic
Live stay. Do not add File-tab pattern buttons in BLITZ.

---

## P3 — Architecture / packaging

### DataSource interface + Loader registry (loaders only)

Common contract for loaders (+ maybe handlers). Converters stay **outside** as suite add-ons.

### Dual-build (Standard vs Full) — obsolete

Converters/OMERO/DICOM = external add-ons. SP/MP fused. No Standard/Full EXE plan unless a real GPU split appears.

---

## P4 — Backlog / discussion

### Options tabs: categories again (replace wrapping rows)

Wrapping tabs (File → Log on extra rows) are a **stopgap** so Stream/Log are
not clipped off the right. Reconsider a **two-level strip**: a short row of
**main categories** on top, **subcategories** on the line below. Still two
lines in the common case, but a category can drop more items when selected
instead of putting every specialist page on the primary strip.

Motivation: tools like **Shade / hillshade** are cool and should stay, but
not everyone needs them on the same rank as File / View / Ops. Park rare
pages under a category (e.g. Analysis → Shade, PCA, RoSEE) rather than
hiding them or crowding the top row.

Do not implement until the wrapping layout has been lived with; this is IA,
not a bugfix.

### Dropped from the nebenher queue

Not visible enough to keep in P1/P2: **H/V envelopes on RGB** (noisy channel
collapse) and **controllable cell-grid** (Conway lattice). Revive only if they
start to hurt in daily use.

### Dedicated Conway / discrete-level color bar

Conway emits Classic ``0/1`` or Ember ``0…N`` (Decay) for LUT mapping. Optional
later: a preset colormap tuned to that ladder (and/or auto-select on Game of Life
start) so decay reads even clearer without hand-picking plasma.

### Project files (`.blitz`) — rethink, don’t just restore

Flatpak + “mystery file next to data” UX. Prefer XDG config / explicit Save As / or drop. See `docs/settings.md`.

### EXIF in File-tab Metadata

Placeholder in File tab; real EXIF later (e.g. Pillow) — no new dep for now.

### LUT auto-refresh / zoom?

Manual Fit vs auto on frame/zoom.

### PCA components table — smaller

Reclaim Options-dock vertical space.

### Binary mask eval?

Clearer validation/feedback when mask does not apply.

### Reintroduce pickle for NumPy?

Opt-in `allow_pickle` for legacy object arrays.

### Gray-looking RGB → take first channel

Quick load path instead of full luminance conversion.

### Envelope of the crosshairs in the main view

Draw envelope bands on the image, not only in plots.

### Optional 3D view

Checkbox / volume view — unscoped.

### Mean feels slow on small datasets

Likely BUSY/UI overhead; profile.

### Performance Tweaks?

Suspects that can push the UI-lag amp into yellow/red (event-loop blocked).
Use the LUT HUD while tasting; fix only what the amp + feel confirm.

- **Shade/Flow inert when Preview off** — `image_changed` / timeline slots still
  fire and early-return; extra overlay `ImageItem`s stay in the scene. Fully
  disconnect + hide/remove items when off so pan/zoom is cube-only.
- **Shade/Flow Preview cost** — pixel-identity full-res ∇z/D8 on the visible
  crop; large zoomed-out viewports are expensive by design. Tune debounce
  (~150 ms view), crop halo, or optional atlas/T-cache — do not silently
  downscale analysis matrices.
- **Display downsample settle** — debounced rebuild (~80 ms) after ViewBox
  settle (`display_downsample.py`). If zoom-out still hitchs, profile settle
  vs QImage rebuild; never re-enable *immediate* `autoDownsample`.
- **H/V extraction + polyline** — line-drag and frame scrub redraw plots
  (throttle exists ~30 fps). Heavy stacks / envelopes may still stall the
  GUI thread; gate harder or sample fewer points while dragging.
- **Isolines / RoSEE / linked cursor** — timeline and `image_changed` fans;
  confirm throttles when overlays are hidden.
- **Histogram / LUT path** — `setImage` / `updateImage` / Fit can recompute
  hist on large frames; watch amp on load and Auto-fit.
- **Live / webcam apply** — display throttle (camera ~10 fps) vs ring push;
  high FPS synthetic live can still flood `setImage` if throttle slips.
- **Mean / reduce “BUSY” tax** — small cubes feel slow from UI/BUSY wrapping
  more than NumPy (see item above); profile before micro-optimizing kernels.
- **pyqtgraph / paint path** — optional OpenGL, ImageItem path; measure before
  flipping globals (regressions on some drivers).
- **Do not chase** — UI-lag 250 ms probe itself; Bench `psutil` while Bench tab
  closed; paint-FPS on the viewer hot path.

### Shade: keep Preview, never Apply-replace

Hillshade stays. **Do not** add Apply → replace stack (would store lighting as
the cube). Product rule: [`docs/architecture.md`](docs/architecture.md#cube-measurand-rule).
Blend / real-sun / GeoTIFF remain parked. Arrival Map / optical-flow overlays
and a generic Overlay tab are **not** planned.

Optional later (optimization only): cache shade for the current az/elev/Z over
`T` so timeline scrub stays smooth. The azimuth atlas (**full frame**, step
5–90°) is Shade **Pre-cache**. Shade Preview is full-frame paint; pan/zoom is
ViewBox only. Combined coloured lights live on the sky dome.
**Flow → Preview accumulation** (D8, still viewport, gold–cyan veins) is on
the Shade tab. Seed path / fill-to-level / legend stay parked.

### TOF → auxiliary curve

TOF is a leftover: one sensor’s 1D time response on the timeline, not
time-of-flight imaging and not a time surface. Keep File → Load TOF; do not
remove, do not rename. Later: generalize to **auxiliary 1D curves** (more
sensors next to the cube). Separate story from cube analysis / overlays.

---

## WOLKE (no priority)

_(index-only BUSY flash: fixed — seek the cached cube, do not reload.)_

---

## Notes

Hidden UI (crop widget) and pure concepts (autograd, ML broken-file detection): [`docs/missing_features.md`](docs/missing_features.md).

`docs/sources_and_variants.md` still describes Standard/Full EXE — historical until updated (“add-ons outside + Flatpak”).

**WETTER-wide:** thinly populated matrices — HTTP gzip (**A**) shipped; Floor (**C**)
and sparse `ImageData` (**D**) dropped — see
[`../WETTER/docs/sparse_matrices.md`](../WETTER/docs/sparse_matrices.md).

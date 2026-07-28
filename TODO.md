# TODO / Roadmap

Prioritized after a long break: **P0 first**, then bugs people feel daily, then features, then architecture, then open ideas.
Human-readable notes under each item so we remember *what* and *why* without re-diving into the code.

WOLKE-related work lives in its own section at the bottom — **no priority**, keep it separate from the BLITZ core queue.

Related: [`docs/missing_features.md`](docs/missing_features.md) (hidden UI / concepts).

---

## Recently done (this stretch)

- Status / Probe dock rename; MetaData block at bottom of File tab (EXIF still placeholder)
- Dock layout restore: ignore incompatible saved state (e.g. after Probe rename); bottom band sizing fixed for large screens
- Timeline dock **hidden for single-frame** data; Frame|Range side panel compact (side-by-side)
- Webcam: list only working capture devices; silence OpenCV probe WARNs; Linux V4L2
- OpenCV pin `<4.14` (4.14.0.94 had Windows-only wheels, broke Linux `uv run`)
- Extraction plots: wheel zooms spatial axis only; Alt+wheel = intensity axis
- Linked cursor (View → Crosshair): image ↔ H/V profile hover markers

---

## P0 — Before the next push

_(none open)_

---

## P1 — Bugs / UX you notice every day

### Color swatch shows LUT color, not real RGB
The little color patch at the cursor is filled from the **colorscale / LUT gradient**, not from the actual pixel RGB. Prefer a true-color swatch when the image is RGB.

Relevant: `blitz/layout/main.py` (`_update_position_display`).

### Mixed image sizes in one folder
Preview/load fails or soft-errors when frames in a folder have different HxW. Planned: load dialog *before* stacking — (A) crop to smallest common HxW or (B) reference frame + align.

Relevant: `blitz/data/load.py`, `blitz/layout/dialogs.py`.

### Envelopes on RGB — disable?
H/V envelopes on RGB collapse over channels and are often noisy. Disable for RGB, or force grayscale first?

Relevant: `docs/extraction_envelopes.md`, `ExtractionPlot`.

---

## P2 — Features with clear user value

### Polyline intensity profile
Dock/window: intensity along a drawn polyline (we already have geometry `PolyLineROI`).

### Auto-crop in the load dialog
MAX preview: threshold bounding box + margin; user confirms, no auto-apply on load.

### RoSEE: isolines + normalization
Check isolines/Normalize after `IsolineAdapter` split; autozoom when appropriate.

Relevant: `blitz/layout/rosee.py`, `blitz/layout/isoline.py`.

### Image smoothing
General Ops/display spatial blur (today only RoSEE/isolines/TOF). Gaussian/box/temporal — see `docs/numba_candidates.md`.

---

## P3 — Architecture / packaging

### DataSource interface + Loader registry (loaders only)
Common contract for loaders (+ maybe handlers). Converters stay **outside** as suite add-ons.

### Dual-build (Standard vs Full) — obsolete
Converters/OMERO/DICOM = external add-ons. SP/MP fused. No Standard/Full EXE plan unless a real GPU split appears.

### Docker (browser / multi-session server) — low priority, not in active development
Still the right tool if we want **remote multi-instance** for weak clients (container per session + browser desktop). Flatpak does not replace that.

**Not being developed right now.** Flatpak/Flathub launch is tracked and verified in a **separate thread**. Keep `docker/` as optional/server path documentation only until we deliberately revive it.

See `docker/README.md`, `flatpak/README.md`.

---

## P4 — Backlog / discussion

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

---

## WOLKE (no priority)

### Sync feels laggy / briefly “busy”
Index-only sync should not full-refresh / flash BUSY when matrix is already in RAM.

Relevant: `blitz/data/web.py`, `blitz/layout/main.py` (`end_web_connection`).

---

## Notes

Hidden UI (crop widget) and pure concepts (autograd, ML broken-file detection): [`docs/missing_features.md`](docs/missing_features.md).

`docs/sources_and_variants.md` still describes Standard/Full EXE — historical until updated (“add-ons outside + Flatpak”).

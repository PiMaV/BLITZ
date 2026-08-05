# TODO / Roadmap

Prioritized after a long break: **P0 first**, then bugs people feel daily, then features, then architecture, then open ideas.
Human-readable notes under each item so we remember *what* and *why* without re-diving into the code.

WOLKE-related work lives in its own section at the bottom — **no priority**, keep it separate from the BLITZ core queue.

Related: [`docs/missing_features.md`](docs/missing_features.md) (hidden UI / concepts).

---

## Recently done (this stretch)

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
- **Shade tab** (hillshade): view-only overlay after RoSEE; Azimuth / Elevation /
  Z factor; Crosshair & Polyline keep sampling height (not shade)

---

## P0 — Before the next push

_(none open)_

---

## P1 — Bugs / UX you notice every day

### Color swatch shows LUT color, not real RGB

The little color patch at the cursor is filled from the **colorscale / LUT gradient**, not from the actual pixel RGB. Prefer a true-color swatch when the image is RGB.

Relevant: `blitz/layout/main.py` (`_update_position_display`).

### Envelopes on RGB — disable?

H/V envelopes on RGB collapse over channels and are often noisy. Disable for RGB, or force grayscale first?

Relevant: `docs/extraction_envelopes.md`, `ExtractionPlot`.

---

## P2 — Features with clear user value

### Auto-crop in the load dialog

MAX preview: threshold bounding box + margin; user confirms, no auto-apply on load.

### RoSEE: isolines + normalization

Check isolines/Normalize after `IsolineAdapter` split; autozoom when appropriate.

Relevant: `blitz/layout/rosee.py`, `blitz/layout/isoline.py`.

### Image smoothing

General Ops/display spatial blur (today only RoSEE/isolines/TOF). Gaussian/box/temporal — see `docs/numba_candidates.md`. **Not a Shade-tab control** — belongs in Ops/View when implemented; Shade stays lighting-only.

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

### Shade follow-ups

Blend (color relief × hillshade); Apply → replace stack; real sun / GeoTIFF north.
Core invariant stays: analysis reads height unless user explicitly Applies.

---

## WOLKE (no priority)

### Sync feels laggy / briefly “busy”

Index-only sync should not full-refresh / flash BUSY when matrix is already in RAM.

Relevant: `blitz/data/web.py`, `blitz/layout/main.py` (`end_web_connection`).

---

## Notes

Hidden UI (crop widget) and pure concepts (autograd, ML broken-file detection): [`docs/missing_features.md`](docs/missing_features.md).

`docs/sources_and_variants.md` still describes Standard/Full EXE — historical until updated (“add-ons outside + Flatpak”).

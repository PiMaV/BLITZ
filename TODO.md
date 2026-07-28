# TODO / Roadmap

Prioritized after a long break: **P0 first**, then bugs people feel daily, then features, then architecture, then open ideas.
Human-readable notes under each item so we remember *what* and *why* without re-diving into the code.

WOLKE-related work lives in its own section at the bottom — **no priority**, keep it separate from the BLITZ core queue.

Related: [`docs/missing_features.md`](docs/missing_features.md) (hidden UI / concepts).

---

## P0 — Before the next push

### Status bar broken
The bottom status bar (and the top-left status dock) were heavily reworked when logging moved into its own Log tab. Something in that refactor left the status UI in a bad state. **Fix this before pushing.**

Relevant: `blitz/layout/ui.py` (`setup_menu_and_status_bar`), `blitz/layout/main.py` (`update_statusbar`).

---

## P1 — Bugs / UX you notice every day

### Color swatch shows LUT color, not real RGB
The little color patch at the cursor is filled from the **colorscale / LUT gradient**, not from the actual pixel RGB. The text may show real values, but the chip follows the colormap — misleading on RGB data (and the histogram side has the same LUT-driven feel). Prefer a true-color swatch when the image is RGB.

Relevant: `blitz/layout/main.py` (`_update_position_display`), tooltip currently says “LUT-mapped”.

### Mixed image sizes in one folder
Preview/load fails or soft-errors when frames in a folder have different HxW. Suffix majority alone is not enough. Planned: open a load dialog *before* stacking and let the user choose:
- **(A)** crop everything to the smallest common HxW (centered), or
- **(B)** pick a stub/reference frame and crop/align the others to it.

Then preview and full load both use that strategy.

Relevant: `blitz/data/load.py`, load dialogs in `blitz/layout/dialogs.py`.

### Envelopes on RGB — disable?
H/V extraction envelopes (min/max over the stack) also run on RGB and collapse over channels. On color data that is often noisy or ambiguous. Open question: turn envelopes off for RGB, or force a grayscale reduction first?

Relevant: `docs/extraction_envelopes.md`, `ExtractionPlot` in `blitz/layout/widgets.py`.

---

## P2 — Features with clear user value

### Polyline intensity profile
Add a dock/window that shows intensity along a drawn polyline. We already have `PolyLineROI` for measuring geometry (area, length, bbox) — this would sample intensity along an arbitrary path (edges, filaments), not only the H/V crosshairs.

### Auto-crop in the load dialog
On the MAX preview: suggest a content bounding box via a simple threshold (`> value`, spinner 0–255), optional isoline for feedback, margin % spinner; ROI updates as the user tweaks it. **Do not auto-apply on load** — user stays in control.

### Mouse-wheel zoom on one axis only
In extraction plots (and similar), allow zooming only X or only Y with the wheel. Timeline already locks Y; extraction plots still use default linked ViewBoxes and fight cumulative zoom-out.

### RoSEE: isolines + normalization
Check that isolines and Normalize still behave after the isoline split into `IsolineAdapter`. Autozoom should kick in when appropriate so plots don’t sit on a wrong scale after normalize.

Relevant: `blitz/layout/rosee.py`, `blitz/layout/isoline.py`.

### Image smoothing (lost idea — restore to backlog)
There was once a note about **image smoothing** (spatial blur / similar on the displayed stack) that never made it into a proper todo and got dropped (e.g. via stash during Flatpak work). Today we only smooth inside RoSEE / isolines / TOF curves — not as a general Ops/display tool. Worth deciding: Gaussian (we already use `cv2.GaussianBlur` for isolines), box, or temporal kernels beyond the existing sliding window. See also `docs/numba_candidates.md` (Gaussian / median sliding filters as future).

---

## P3 — Architecture / packaging

### DataSource interface + Loader registry (loaders only)
A common contract for **loaders** (and maybe handlers), plus a registry suffix → loader. Converters and exotic formats stay **outside** BLITZ as suite add-ons (`CONVERTERS/` etc.) — not part of a “Full” in-app build anymore.

Relevant: `docs/sources_and_variants.md` (parts of this doc are outdated; dual-build story below).

### Dual-build (Standard vs Full) — obsolete
No longer state of the art for this project. Converters / OMERO / DICOM and similar live as **external add-ons**, not as a second EXE. The only remaining dual-build angles would be things like **GPU vs CPU** or **SP vs MP** — and those paths are effectively **fused** already (multicore thresholds, shared codepaths). Don’t plan Standard/Full packaging unless a real GPU split appears.

### Docker vs Flatpak
Docker browser packaging becomes uninteresting once **Flatpak** is the Linux distribution path. Prefer Flatpak/Flathub; treat Docker as legacy / optional, not an active roadmap item. See `flatpak/README.md`.

---

## P4 — Backlog / discussion

### Project files (`.blitz`) — rethink, don’t just restore
Code can still open `.blitz`; the old auto load/save UI was removed on purpose. Effective use needs a rethink:
- Under **Flatpak**, sibling writes / dataset-relative paths are awkward or broken.
- Users repeatedly asked **what that file is doing in their data folder** — silent project files next to the images are confusing, not helpful.

Keep commenting / documenting the idea if useful, but default should not be “drop a mystery file next to the dataset.” Prefer XDG config / explicit Save As / or drop the concept.

See `docs/missing_features.md`, `docs/settings.md`.

### LUT auto-refresh / zoom?
Should LUT levels refresh automatically on frame change or zoom, or stay manual Fit / Auto-fit? Large dynamic-range data makes this a recurring friction point.

### PCA components table — smaller
Table was already compacted once; Options dock is still tight vs. the variance plot. Shrink further or reclaim vertical space.

### Binary mask eval?
“Load binary image” applies a boolean mask; mismatches soft-fail with a log line. “Eval” might mean better validation (shape/grayscale) or clearer feedback when the mask does not apply.

### Reintroduce pickle for NumPy?
`.npy` loads with `allow_pickle=False` (safer, especially for network downloads). Some older/scientific files need pickle for object arrays — maybe an explicit opt-in.

### Gray-looking RGB → take first channel
We already detect “effectively grayscale” RGB. Quick path: if it is gray-in-RGB, load only channel 0 instead of a full luminance conversion (faster, less RAM).

### Crosshair hover → point in main view
While scrubbing along the H/V extraction curves, show a corresponding point in the main image so plot position and image stay linked.

### Envelope of the crosshairs in the main view
Draw envelope bands (min/max or percentile) onto the main image, not only in the H/V plots. Envelope math already exists for the plots.

### Optional 3D view
Checkbox for a simple 3D/volume view of the time stack. Concept only — would pull in GL deps; “not that complex for a first shot,” but unscoped.

### Mean feels slow on small datasets
Reduce → Mean is Numba-accelerated, but with small data the BUSY/UI pipeline overhead can dominate, so it feels slower than other ops. Worth profiling whether we flash busy too eagerly or recompute twice.

---

## WOLKE (no priority)

Track separately from the BLITZ core queue. Pick up when the network/sync story is in focus — not ordered against P0–P4.

### Sync feels laggy / briefly “busy”
When WOLKE only changes the selected frame and BLITZ already has the matrix in memory, sync should be cheap. Today it still goes through a full-ish refresh (`set_image` → `reset_options` → `apply_ops`), so you get a delay and a short BUSY flash even though nothing needs reloading. Reloading the matrix here would be wrong — short-circuit to an index/UI update.

Relevant: `blitz/data/web.py` (selection shortcut), `blitz/layout/main.py` (`end_web_connection`).

---

## Notes

Hidden UI (crop widget) and pure concepts (autograd, ML broken-file detection) live in [`docs/missing_features.md`](docs/missing_features.md) — not duplicated here as active todos.

`docs/sources_and_variants.md` still describes Standard/Full EXE packaging; treat that as historical until the doc is updated to match “add-ons outside + Flatpak.”

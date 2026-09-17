# Changelog

All notable changes to BLITZ are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Stream idle no longer leaves `Network: listening` as a permanent status-bar message (that hid `File:` / `Frame:`).
- Multi-line Dataset load summary stays in the Log tab only; status-bar one-liner no longer grows the bar height.

### Changed

- Stream index-only seek works for any cached cube filename (not only `__selection__.npy`), so an EVT hub can relay `viewer_index` between BLITZ and DONNER without re-downloading the stack.
- Hub `file_name` / `file_names` stamp frame metadata so the status bar shows real WOLKE basenames; `viewer_index` emits only when the index changes.

## [2.3.1] - 2026-08-31

### Fixed

- Keep fitting on a frame tick refits the **current frame** and skips LUT work when Min/Max did not change. Scrubbing a small Event-reader cube no longer re-percentiles the whole stack or ping-pongs the histogram against Auto pins.
- Timeline scrub no longer rebuilds the LUT histogram while Log hist is off, and no longer runs RoSEE while that overlay is off.
- Live `setImage` no longer auto-ranges the view or pyqtgraph-autoLevels on every ring tick.
- Zoom-paint downsample no longer blanks the ImageItem when device mapping is not ready (falls back to full-res).

### Changed

- LUT Auto: binary occupancy uses **greyclip**; polarity **states** (`uint8` 0/85/170/255) use **`event`** (black → red → green → yellow, 0…255); sparse event counts use **plasma** (0…p99 of positives).
- Stream ingest no longer applies File-tab 8-bit / Normalize / Grayscale (8-bit would zero uint16 counts).

## [2.3.0] - 2026-08-31

### Added

- Colormap preset `event` (black → blue → amber) for event-camera occupancy and counts. LUT Auto classifies the gray cube: occupancy pins 0…255 (or 0…1), sparse integer counts pin 0…p99 of positives; signed stays bipolar.

### Changed

- RGB LUT for integer colour stacks: `uint16` uses 0…p99 of positive values instead of a hard 0…255. `uint8` photos stay 0…255; float rungs 0…1 still work. (Event-camera ingest from EVT is grayscale, not RGB.)
- Log hist checkbox sits next to Auto in the colormap row (LUT dock).

## [2.2.1] - 2026-08-31

### Fixed

- Timeline dock opens at a usable height when a time series arrives (EVT sidecar, live ring, video). After the splash `hide()`, `show()` alone left a collapsed splitter strip that first-time users could not discover.
- RGB stacks (event-camera OFF=red / ON=green, or photos) keep the encoded LUT range — 0…1 for float, 0…255 for uint8 — instead of a plasma/bipolar stretch. Leave File-tab Normalize off.

## [2.2.0] - 2026-08-23

### Changed

- Full-frame Shade: Preview and Pre-cache compute the whole frame once; pan/zoom only transform the ViewBox.
- Pixel identity: no 1600-px working copy for Shade/Flow math.
- Zoom paint: debounced display downsample after the gesture (analysis stays full-res).
- LUT HUD shows UI event-loop lag under IDLE instead of paint FPS.

### Removed

- In-core File-tab test patterns.

[Unreleased]: https://github.com/PiMaV/BLITZ/compare/build-v2.3.1...HEAD
[2.3.1]: https://github.com/PiMaV/BLITZ/compare/build-v2.3.0...build-v2.3.1
[2.3.0]: https://github.com/PiMaV/BLITZ/compare/build-v2.2.1...build-v2.3.0
[2.2.1]: https://github.com/PiMaV/BLITZ/compare/build-v2.2.0...build-v2.2.1
[2.2.0]: https://github.com/PiMaV/BLITZ/releases/tag/build-v2.2.0

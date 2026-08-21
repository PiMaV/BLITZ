# Hero Features

Concise overview of BLITZ’s strongest analysis and visualization capabilities. For the full tab-by-tab matrix (algorithms, shortcuts, formats), see [`tabs_explained.md`](tabs_explained.md).

- **SVD / PCA Pattern Extraction** — Exact full SVD or memory-efficient randomized SVD (Halko) on entire frame stacks; reconstruct from principal components, inspect eigenimages, and read explained-variance curves.
- **Hillshade Relief Visualization** — Lambertian shading from height fields (`∇z`, azimuth / elevation / Z-factor); Preview overlay only (analysis stays on height). Viewport paint, sky dome (azimuth + elevation + Z-shadow), Combined coloured lights, Pre-cache (5–90°). No Apply-replace-stack.
- **D8 Flow Accumulation** — Steepest-descent drainage overlay (8 neighbours); cyan = more upstream area. Viewport Preview on the Shade tab; can sit on hillshade. Not a palaeo reconstruction.
- **Polyline Profile & Path Statistics** — Dense sampling along an open polyline with perpendicular band averaging, min/max envelopes (perp + over frames), path length in pixels or calibrated AU, and CSV export.
- **Numba-Fused Matrix Pipeline** — Parallel JIT kernels for fused subtract÷divide, sliding-window mean, and axis-0 reduce (mean / max / min / std) on `(T, H, W, C)` volumes.
- **Drag-and-Drop Matrix Handling** — Drop images, video, or `.npy` stacks straight into the viewer; optional load dialogs for 8-bit, normalize, grayscale, size/subset ratio, and RAM caps.
- **Temporal Ops & Zonal Timeline** — Background subtraction / flat-field division from aggregate range, file, or sliding window; ROI mean/median curves with envelope bands, or **Live Probe** pixel time series (hover + pins) with optional signed **Δ bar**.
- **Crosshair Profilschnitte** — Linked H/V extraction plots with line-width averaging, dataset envelopes, and RoSEE event bounds on cumulative fluctuation signals.
- **Bulk Load → Live Stream** — Folder/video pipelines at multi-GB scale, plus synthetic Lissajous/Lightning streams, classic Conway Game of Life, USB webcam, and network `.npy` ingest (optional gzip on the wire).

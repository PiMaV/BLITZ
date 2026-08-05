# BLITZ V2.0

![BLITZ Interface](/docs/images/BLITZ_overview_V2.png)

**BLITZ treats images as structured data.**

A high-performance, matrix-based image viewer designed for efficiently exploring both **massive image datasets** and **single-image analysis workflows**.

---

[![Release](https://img.shields.io/github/v/release/PiMaV/BLITZ)](https://github.com/PiMaV/BLITZ/releases/latest)
[![License](https://img.shields.io/github/license/PiMaV/BLITZ)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)]()

---

# Download

[Download the latest release for Windows and Ubuntu](https://github.com/PiMaV/BLITZ/releases/latest)

No installation required — simply download and run the executable.

---

## What is BLITZ

BLITZ (**Bulk Loading and Interactive Time series Zonal analysis**) is a high-performance, matrix-based image exploration and analysis tool designed for efficiently managing both massive datasets and single-image analysis.

It was originally developed and initially implemented by Philipp Mattern during his time at [INP Greifswald](https://www.inp-greifswald.de).

It is actively maintained and further developed as part of his independent engineering work at [M.E.S.S. – Mattern Engineering & Software Solutions](https://mess.engineering).

Version 2.0 introduces a fully refactored architecture with improved performance, stability, and maintainability.

---

## WETTER Framework

BLITZ is the interactive viewer in the **WETTER framework**: *Raw Data → DAMPF → KEIM → WOLKE → BLITZ*. For the full pipeline, ecosystem overview, and links to all modules, see:

**[WETTER Framework — wetter.mess.engineering](https://wetter.mess.engineering)**

DPG Symposium presentation (architecture and BLITZ–WOLKE integration):  
📄 [BLITZ_WOLKE_DPG25V2_Compact.pdf](https://wetter.mess.engineering/docs/BLITZ_WOLKE_DPG25V2_Compact.pdf)

---

## Hero Features

- **SVD / PCA Pattern Extraction** — Exact full SVD or memory-efficient randomized SVD (Halko) on entire frame stacks; reconstruct from principal components, inspect eigenimages, and read explained-variance curves.
- **Hillshade Relief Visualization** — Lambertian shading from height fields (`∇z`, azimuth / elevation / Z-factor); view-only overlay so analysis always stays on the original intensity.
- **Polyline Profile & Path Statistics** — Dense sampling along an open polyline with perpendicular band averaging, min/max envelopes (perp + over frames), path length in pixels or calibrated AU, and CSV export.
- **Numba-Fused Matrix Pipeline** — Parallel JIT kernels for fused subtract÷divide, sliding-window mean, and axis-0 reduce (mean / max / min / std) on `(T, H, W, C)` volumes.
- **Drag-and-Drop Matrix Handling** — Drop images, video, or `.npy` stacks straight into the viewer; optional load dialogs for 8-bit, normalize, grayscale, size/subset ratio, and RAM caps.
- **Temporal Ops & Zonal Timeline** — Background subtraction / flat-field division from aggregate range, file, or sliding window; ROI mean/median curves with envelope bands over the full time series.
- **Crosshair Profilschnitte** — Linked H/V extraction plots with line-width averaging, dataset envelopes, and RoSEE event bounds on cumulative fluctuation signals.
- **Bulk Load → Live Stream** — Folder/video pipelines at multi-GB scale, plus synthetic Lissajous/Lightning streams, USB webcam, and network `.npy` ingest.

---

# Interface Preview

*(Click if animation is not playing)*

![Quick Feature Overview](resources/public/blitz_demo.gif)

---

## Documentation

* [Full Documentation Index](docs/md_state.md)
* [Quick Start Guide](docs/walkthrough.md)
* [Features & Tabs Explained](docs/tabs_explained.md)
* [Missing & Planned Features](docs/missing_features.md)
* [Optimization Report](docs/optimization.md)
* [Data Sources & Build Variants](docs/sources_and_variants.md)

## Docker

Run BLITZ in a browser via Docker. See: [docker/README.md](docker/README.md)

## Flatpak / Flathub

Linux App Store packaging lives under [flatpak/](flatpak/README.md)
(`engineering.mess.BLITZ`, M.E.S.S. branding). Build and Flathub notes are there.

## Development

To compile and develop locally:

1. Clone the repository:

   ```
    $ git clone https://github.com/pimav/BLITZ.git
    $ cd BLITZ
   ```

2. Set up a virtual environment and install dependencies with [uv](https://docs.astral.sh/uv/):

   ```
    $ pip install uv
    $ uv sync
    $ uv run python -m blitz
   ```

## Acknowledgements

Early development of BLITZ was supported by Richard Krieg (student assistant) until v1.3.0 / January 2025, including refactoring, bug fixing, and feature development during the INP-funded project phase.

## Additional Resources

* Example Dataset: [KinPen Science Example Set](https://www.inptdat.de/dataset/fast-framing-images-kinpen-science-example-set-images-testing-blitz-image-viewer)
* Explore more datasets or contribute your own on [INPTDAT](https://www.inptdat.de).

## License

BLITZ is licensed under the [GNU General Public License v3.0](LICENSE).
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

## Key Features

- **High-Performance Data Handling:** Efficiently processes very large datasets (e.g. loading, scaling, and converting ~21,000 RGB images (~2.5 GB raw data) into ~6.2 GB of grayscale matrix data in ~30 s on a standard gaming laptop).
- **Easy Data Handling:** Drag-and-drop support for image, video, and NumPy matrix (*.npy) formats.
- **Easy to Use:** Automatic resource management for small and large datasets.
- **User-Friendly Interface:** Intuitive GUI with mouse-based navigation and shortcuts.
- **Advanced Image Processing:** Matrix-based processing with fast, Numba-accelerated statistics.
- **Live View:** Support for real USB cameras and simulated data streams.
- **Built on Python:** Using Qt and PyQtGraph for high performance and flexibility.

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
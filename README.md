# BLITZ

## TL;DR

**BLITZ** is a high-performance, matrix-based image viewer built for efficiently managing massive datasets but equally suitable for single-image analysis.

## Download

[Download the latest release for Windows](https://github.com/PiMaV/BLITZ/releases/latest)
No installation needed—just a single, standalone *.exe file. Linux and Docker in the making.

## Overview

**BLITZ** (Bulk Loading and Interactive Time series Zonal analysis) is an open-source image viewer developed and initially implemented by Philipp Mattern during his time at [INP Greifswald](https://www.inp-greifswald.de). It is actively maintained and further developed as part of his independent engineering work at [M.E.S.S.](https://mess.engineering).

**Key Features:**

* **Fast Data Handling:** Handles very large datasets efficiently (e.g. 21,000 images (~25GB) in ~35 seconds on a standard gaming laptop).
* **Easy Data Handling:** Drag-and-drop functionality for various image and video formats, including NUMPY matrices (*.npy).
* **Easy-to-use:** Automatic resource management for large and small datasets.
* **User-Friendly Interface:** Intuitive GUI with mouse-based navigation and shortcut capabilities.
* **Advanced Image Processing:** Matrix-based processing with fast statistical calculations.
* **Built on Python:** Using Qt and PyQtGraph for high performance and flexibility.

![BLITZ Interface](docs/images/overview.png)

---

(Click if animation is not playing)
![Quick Feature Overview](resources/public/BLITZ_Record.gif)

---

## Documentation

* [Quick Start Guide](docs/walkthrough.md)
* [Core Functionalities](docs/Tabs_explained.md)

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

## Docker

Run BLITZ in a browser via Docker: see [docker/README.md](docker/README.md).

## Additional Resources

* Example Dataset: [KinPen Science Example Set](https://www.inptdat.de/dataset/fast-framing-images-kinpen-science-example-set-images-testing-blitz-image-viewer)
* Explore more datasets or contribute your own on [INPTDAT](https://www.inptdat.de).

## License

BLITZ is licensed under the [GNU General Public License v3.0](LICENSE).
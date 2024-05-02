# BLITZ
BLITZ is an open-source image viewer specifically designed for "**B**ulk **L**oading and **I**nteractive **T**ime series **Z**onal analysis," developed at the [INP Greifswald](https://www.inp-greifswald.de). It is optimized for handling extensive image series, a common challenge in diagnostic imaging and process control. Programmed primarily in Python and utilizing QT and PyQTGraph.
BLITZ offers:
- rapid loading of large image series
- efficient performance scaling
- versatile data handling options
- user-friendly GUI
- stable lookup tables for visual consistency
- powerful matrix-based image processing capabilities, allowing for instant statistical calculations and image manipulations.

## Download the Latest Release for Windows
[Most recent
release](https://github.com/CodeSchmiedeHGW/BLITZ/releases/latest)

## GIF Animation showing the key features of BLITZ
(Click if animation is not playing)
![GIF_Animation](resources/public/BLITZ_Record.gif)

## Compiling and Developing

It is recommended to use [poetry](https://python-poetry.org/) for local development. After cloning
this repository, create a virtual environment, install all dependencies and run the application.

```shell
$ git clone https://github.com/CodeSchmiedeHGW/BLITZ.git
$ cd BLITZ
$ poetry install
$ poetry run python -m blitz
```

You can create a binary executable from the python files using `pyinstaller` with the following
options.

```shell
$ pyinstaller --onefile --noconsole --icon=./resources/icon/blitz.ico blitz_main.py
```

## Additional Resources

- Visit [INPTDAT](https://www.inptdat.de) for additional images or publishing your own.
- You can find the original [Example Dataset](https://www.inptdat.de/dataset/fast-framing-images-kinpen-science-example-set-images-testing-blitz-image-viewer) at INPTDAT as well.

## License

BLITZ is licensed under the terms of the GNU General Public License version 3 (GPL-3.0). Details
can be found in the [LICENSE](LICENSE) file.

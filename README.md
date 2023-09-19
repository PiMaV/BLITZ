# BLITZ
**B**ulk **L**oading & **I**nteractive **T**ime-series **Z**onal-analysis

## Quick Start
It is recommended to create a virtual environment with all dependecies before executing the
application. You can use `poetry` to do this with:

    $ poetry install

As `pip` also supports `.toml` files, another option is to directly install all dependencies with:

    $ pip install .

After a successful installation, you can start **BLITZ** by executing:

    $ python -m blitz

It is also possible building the executable using the package `pyinstaller`:

    $ pyinstaller --onefile --windowed blitz.py

import multiprocessing
import sys

import pyqtgraph as pg

from .app import run

if __name__ == "__main__":
    pg.setConfigOptions(useNumba=True)
    multiprocessing.freeze_support()
    sys.exit(run())

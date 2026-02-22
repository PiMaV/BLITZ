"""Entry point for BLITZ."""

import os
import sys

# Suppress OpenCV TIFF warnings (e.g. "Unknown field with tag 292")
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from blitz import app


def main():
    sys.exit(app.run())


if __name__ == "__main__":
    main()

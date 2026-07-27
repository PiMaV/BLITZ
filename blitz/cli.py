"""Console entry point for BLITZ (desktop Exec=blitz / Flatpak)."""

import os
import sys

# Suppress OpenCV TIFF warnings (e.g. "Unknown field with tag 292")
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


def main() -> None:
    from blitz import app

    sys.exit(app.run())


if __name__ == "__main__":
    main()

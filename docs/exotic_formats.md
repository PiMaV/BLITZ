# Exotic formats and the standard loader

BLITZ’s **standard loader** (Flatpak / EXE) focuses on:

- Images, video, `.npy`
- ASCII grids (`.asc` / `.dat` / probed `.txt`)
- In-core HIKMICRO radiometric JPEG → approximate °C (numpy only)
- Folder chooser when a directory contains mixed groups

## What stays out of the core

Domain / exotic formats (historical DICOM path, OMERO, Bio-Formats, …) are **not**
embedded in the Flatpak/EXE. That keeps the artifact lean and avoids optional
heavy dependencies.

## Later bridges (policy)

When BLITZ should “handle more formats” without bloating the core:

1. **Standalone converter** — CLI/toolbox (same idea as suite `converters/`) that
   writes `.npy` (or plain images). User opens the result in BLITZ.
2. **Mini-server** — convert remotely; BLITZ only receives standard payloads
   (aligned with WOLKE → viewer `.npy` delivery).

Do **not** grow the core Flatpak with exotic native deps unless a format is
numpy/cv2-only and clearly belongs in the standard loader (like HIKMICRO).

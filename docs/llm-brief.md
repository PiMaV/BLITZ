# BLITZ — LLM brief

Compact product picture for agents. Humans use [README.md](../README.md). Detail: [architecture.md](architecture.md), [CHANGELOG.md](../CHANGELOG.md), backlog [TODO.md](../TODO.md).

Brand names stay uppercase English: **WETTER**, **WOLKE**, **DAMPF**, **KEIM**, **BLITZ**, **DONNER**.

## What it is

PyQt6 + PyQtGraph **2D** desktop viewer for one dense `ImageData` cube in RAM, shape `(T, H, W, C)`. Load folders of images, video, or `.npy`. Live USB camera, synthetic streams, Conway. Network ingest from the WETTER hub (**WOLKE**, **EVT**) over Socket.IO + HTTP `.npy`.

Entry: `blitz/cli.py` → `blitz/app.py`. Run: `uv sync` then `uv run python -m blitz`. Ship: PyInstaller GitHub Releases (`BLITZ-vX.Y.Z-windows-x86_64.exe`, `BLITZ-vX.Y.Z-linux-x86_64`).

## Do

- One measurand per cube. Preview overlays (Shade, LUT, Flow, isolines, RoSEE bounds) never bake lighting into the stack.
- Stream Connect: index-only seek on the cached cube; stamp `MetaData.file_name` from hub `file_name` / `file_names`; emit `viewer_index` only when the index changes.
- Status bar keeps `File:` / `Frame:` widgets visible. Do not leave a permanent `QStatusBar.showMessage` on idle listen. Multi-line dataset dumps stay in the Log tab.

## Don't

- Do not add a Docker / container runtime to this repo.
- Do not advertise Linux Flatpak. `flatpak/` is in preparation only, not a ship path.
- Do not mosaic GeoTIFF tiles in the folder loader (suite sidecar `../DGM/`).
- Do not embed EVT3 decode, OMERO, DICOM, or GDAL in the core viewer.
- Do not peer-sync viewers; playhead is hub-and-spoke.
- **DONNER** is the 3D sibling, not a BLITZ tab. Format sidecars: `DGM/`, `HIKMICRO/`, `DICOM/`, `EVT/`.

## Key paths

| Area | Path |
|------|------|
| Cube | `blitz/data/image.py` (`ImageData`) |
| Load | `blitz/data/load.py` |
| Hub ingest | `blitz/data/web.py` (`WebDataLoader`, `apply_hub_file_names`) |
| UI layout | `blitz/layout/ui.py` |
| Controller | `blitz/layout/main.py` |
| Tabs (human) | `docs/tabs_explained.md` |
| Tests | `_aux/tests/` — `uv run pytest` (`testpaths` in `pyproject.toml`) |

## Viewer Contract (ingest)

Hub payload: `file_name`, optional `file_names`, `index`. Packed WOLKE selection uses `__selection__.npy`. After download (or on index-only seek), stamp basenames, or `selection[i]` when there is no list. Stream ingest does **not** apply File-tab 8-bit / Normalize / Grayscale (8-bit would zero `uint16` counts).

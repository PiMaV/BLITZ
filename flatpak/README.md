# BLITZ Flatpak / Flathub

## Identity convention (M.E.S.S.)

| Role | Value |
|------|--------|
| Flatpak / AppStream / desktop ID | `engineering.mess.BLITZ` |
| Brand / publisher | **M.E.S.S.** — [mess.engineering](https://mess.engineering) |
| Source repository | [github.com/PiMaV/BLITZ](https://github.com/PiMaV/BLITZ) (PiMaV = maintainer nickname only) |
| Do not use | `CodeSchmiedeHGW`, `io.github.PiMaV.*` as product IDs |

Future WETTER apps use the same namespace: `engineering.mess.Dampf`, `engineering.mess.Wolke`, …

## Prerequisites

- Flatpak + Flathub remote
- Builder: `flatpak install --user flathub org.flatpak.Builder`
- Runtime/SDK/BaseApp are pulled automatically with `--install-deps-from=flathub`

## Local build & install

From this `flatpak/` directory:

```bash
flatpak run org.flatpak.Builder --user --install --force-clean \
  --install-deps-from=flathub build-dir engineering.mess.BLITZ.yml

flatpak run engineering.mess.BLITZ
```

If Builder fails with `bwrap: Can't find source path .../ccache`, unset `CCACHE_DIR` (or `mkdir -p` that path) and retry with `CCACHE_DISABLE=1`.

Validate AppStream metadata (from repo root):

```bash
appstreamcli validate --no-net data/engineering.mess.BLITZ.metainfo.xml
desktop-file-validate data/engineering.mess.BLITZ.desktop
```

### Webcam

Default sandbox has no raw camera devices. To enable USB webcam:

```bash
flatpak override --user --device=all engineering.mess.BLITZ
```

### Filesystem

The manifest uses `--filesystem=home:ro` so Drag-and-Drop / open of datasets
under `$HOME` works. Exports use `QFileDialog.getSaveFileName` (XDG portal).
Flathub requires a linter exception for `home:ro` — justify with DnD from the
file manager for scientific image folders.

## Regenerating Python dependency modules

PyQt6 comes from `com.riverbankcomputing.PyQt.BaseApp` — it must **not** be in `requirements-flatpak.txt`.

```bash
# needs: flatpak-builder-tools pip generator + org.kde.Sdk//6.10
python3 flatpak-pip-generator.py \
  -r requirements-flatpak.txt \
  -o python3-blitz-deps \
  --prefer-wheels=numpy,opencv-python-headless,numba,llvmlite,psutil \
  --runtime=org.kde.Sdk//6.10 \
  --yaml
mv python3-blitz-deps.yaml python3-blitz-deps.yml
```

Match `--runtime` to the manifest `runtime-version` / SDK branch.

## Flathub submission

1. **Domain verification** for `mess.engineering` (required for `engineering.mess.*`):
   follow [Flathub verification](https://docs.flathub.org/docs/for-app-authors/verification).
   File: `https://mess.engineering/.well-known/org.flathub.VerifiedApps.txt`
   (path reserved; paste the Developer Portal token after the app exists on Flathub).

2. **Pinned source:** `engineering.mess.BLITZ.yml` already uses git tag `v2.0.4`.
   When shipping a new release, bump `tag` and `commit` together.

3. Fork [flathub/flathub](https://github.com/flathub/flathub), open a PR against
   base branch **`new-pr`** with this manifest and `python3-blitz-deps.yml`
   (title: `Add engineering.mess.BLITZ`).

4. After `flathub/engineering.mess.BLITZ` exists, Flathub CI builds x86_64
   (and aarch64 when the pinned wheels cover it).

5. Further suite apps later under `engineering.mess.*`; do not wrap PyInstaller
   for Flathub.

## Layout in this repo

```
data/
  engineering.mess.BLITZ.desktop
  engineering.mess.BLITZ.metainfo.xml
  icons/hicolor/{64,128,256}x…/apps/engineering.mess.BLITZ.png
flatpak/
  engineering.mess.BLITZ.yml
  python3-blitz-deps.yml
  requirements-flatpak.txt
  README.md
```

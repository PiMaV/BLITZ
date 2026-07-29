# Loading & Settings Flow

## Strategie-Entscheidung

- **Load-Tab bleibt** in der GUI – zentrale Stelle fuer Defaults und schnellen Zugriff
- **Load-Dialog** (Video / Bilder / ASCII) als erweiterte Option mit Preview + Crop
- **Session-Defaults**: Letzte Dialog-Einstellungen werden bei Drag&Drop wiederverwendet
- **Folder chooser**: Bei gemischten Ordnern (mehrere Suffixe / HIKMICRO-Offer / Naming-Cluster) fragt `FolderLoadChooserDialog` vor dem Load; siehe `blitz/data/folder_scan.py`
- **Exotic formats**: bleiben ausserhalb des Core — `docs/exotic_formats.md`

---

## Folder chooser (Implementiert)

Wenn `path` ein Ordner ist und `should_show_chooser(scan)` greift:

```
load_images(folder)
  |
  +-- scan_folder → LoadGroup[] (image / video / array / ascii / hikmicro_celsius / other)
  |     optional: Naming-Cluster innerhalb gleicher Suffixe
  |
  +-- FolderLoadChooserDialog
  |     |-- Gruppen + Dateiliste + ◀/▶ Preview
  |     |-- OK → selected kind + file_list
  |
  +-- kind-spezifischer Pfad:
        image/video → Image/VideoLoadOptionsDialog (file_list)
        ascii → AsciiLoadOptionsDialog
        hikmicro_celsius → load_hikmicro_stack → ImageData
        array → .npy wie bisher
```

- **Einzige Gruppe**: Chooser kann entfallen (Shortcut), sonst immer waehlen.
- **Mixed HxW** (Bilder): `MixedImageSizesDialog` → `mixed_size_policy="crop_min"` oder Abbruch.
- Preview-Worker laufen in `QThread` mit Guard (kein Destroy-while-running bei schnellem Mode/Normalize-Toggle).

---

## Video Loading (Implementiert)

### Quellen der Parameter

1. **Load-Tab (UI)**: `size_ratio`, `subset_ratio`, `max_ram`, `convert_to_8_bit`, `grayscale`
2. **Video-Dialog** (wenn geoeffnet): `frame_range`, `step`, `size_ratio`, `grayscale`, `mask`, `mask_rel`
3. **Session-Defaults** (`_video_session_defaults`): Letzte Einstellungen aus dem Video-Dialog

### Ablauf

```
load_images(path)
  |
  +-- Video? --> meta = get_video_metadata(path)
  |     |
  |     +-- show_dialog? (Checkbox "Immer" ODER est_bytes > Schwellwert MB)
  |           |
  |           +-- JA: VideoLoadOptionsDialog oeffnen
  |                 |-- OK: user_params --> params, Session-Defaults speichern
  |                 |-- Abbrechen: return
  |           |
  |           +-- NEIN: Session-Defaults anwenden (falls vorhanden)
  |                 |-- size_ratio, step, grayscale, mask_rel --> params
  |                 |-- mask_rel --> mask (Pixel-Slices aus relativen Koordinaten)
  |                 |-- Load-Tab UI aktualisieren
  |
  +-- params.pop("mask_rel")  # mask_rel nur fuer Session, nicht an DataLoader
  |
  +-- image_viewer.load_data(path, **params)
```

### Crop/Mask

- **Im Dialog**: ROI auf Vorschau ziehen --> `mask` (Pixel-Slices), `mask_rel` (0..1)
- **mask_rel**: Session-Defaults; beim naechsten Video (Dialog oder Drag&Drop) wieder angewendet
- **mask**: DataLoader wendet Crop beim Laden pro Frame an

### Preview-Optionen (Video / Image / ASCII)

- **Mode**: Default **MAX** (ueber Frames/Samples); alternativ Single
- **Normalize each image/frame** (Loading Options): wirkt auf den **Load** (pro Bild/Frame Stretch). Unabhaengig von der Preview.
- **Preview normalize** (unter der HLine, nur Dialog): **nur Anzeige** in der Vorschau — aendert nicht, was geladen wird.
- **8 bit / Grayscale**: wenn Quelle schon nativ → Checkbox checked, disabled, sichtbar ausgegraut (`set_checkbox_visibly_enabled`)
- Nach erfolgreichem Load mit **T>1**: Timeline-Dock wird geoeffnet (`dock_t_line.show` / `raiseDock`); bei Einzelbild bleibt es versteckt.

---

## Bilder / Ordner (Implementiert)

### Ablauf (analog Video)

```
load_images(path)
  |
  +-- (falls Ordner) optional Folder chooser → file_list + kind
  |
  +-- Bild oder Bild-Ordner? --> meta = get_image_metadata(path) bzw. aus file_list
  |     |
  |     +-- show_dialog? (Erster Load ODER Checkbox ODER est_bytes > Schwellwert)
  |           |
  |           +-- JA: ImageLoadOptionsDialog
  |           +-- NEIN: Session-Defaults anwenden
  |
  +-- params.pop("mask_rel")
  +-- load_data(path, file_list=..., mixed_size_policy=..., **params)
```

### ImageLoadOptionsDialog

- **Preview**: Default MAX ueber gesampelte Bilder; Mode Single waehlbar
- **Crop-ROI** auf der Preview --> `mask`, `mask_rel`
- **Transforms**: Flip X, Flip Y, Flip XY (Transpose) — Preview + nach dem Load (wie View-Tab)
- **Optionen**: Resize, Grayscale, 8 bit, Normalize each image; bei Ordner zusaetzlich Subset-Ratio
- **Preview normalize**: nur Anzeige (getrennt durch HLine)
- **Wann zeigen?**: Erster Load in Session ODER Checkbox "Always show" ODER est_bytes > Schwellwert
- **Session-Defaults**: `size_ratio`, `grayscale`, `mask_rel`, bei Ordner `subset_ratio`

---

## DataLoader-Parameter

- `size_ratio`, `subset_ratio`, `max_ram`, `convert_to_8_bit`, `grayscale`, `normalize`, `mask`, `crop`
- `file_list`: explizite Dateiliste vom Folder-Chooser (kein silent majority)
- `mixed_size_policy`: `"crop_min"` nach Mixed-Sizes-Dialog
- `frame_range`, `step`: nur fuer Video, via `load(path, frame_range=..., step=...)`
- `mask` wird fuer Einzelbilder, Ordner und Video angewendet

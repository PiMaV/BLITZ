# Loading & Settings Flow

## Strategie-Entscheidung

- **Load-Tab bleibt** in der GUI – zentrale Stelle fuer Defaults und schnellen Zugriff
- **Load-Dialog** (Video: vorhanden; Bilder: geplant) als erweiterte Option mit Preview + Crop
- **Session-Defaults**: Letzte Dialog-Einstellungen werden bei Drag&Drop wiederverwendet

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

---

## Bilder / Ordner (Implementiert)

### Ablauf (analog Video)

```
load_images(path)
  |
  +-- Bild oder Bild-Ordner? --> meta = get_image_metadata(path)
  |     |
  |     +-- show_dialog? (Erster Load ODER Checkbox ODER est_bytes > Schwellwert)
  |           |
  |           +-- JA: ImageLoadOptionsDialog
  |           +-- NEIN: Session-Defaults anwenden
  |
  +-- params.pop("mask_rel")
  +-- load_data(path, **params)
```

### ImageLoadOptionsDialog

- **Preview**: Einzelbild oder (bei Ordner) MAX ueber gesampelte Bilder
- **Crop-ROI** auf der Preview --> `mask`, `mask_rel`
- **Optionen**: Resize, Grayscale; bei Ordner zusaetzlich Subset-Ratio
- **Wann zeigen?**: Erster Load in Session ODER Checkbox "Always show" ODER est_bytes > Schwellwert
- **Session-Defaults**: `size_ratio`, `grayscale`, `mask_rel`, bei Ordner `subset_ratio`

---

## DataLoader-Parameter

- `size_ratio`, `subset_ratio`, `max_ram`, `convert_to_8_bit`, `grayscale`, `mask`, `crop`
- `frame_range`, `step`: nur fuer Video, via `load(path, frame_range=..., step=...)`
- `mask` wird fuer Einzelbilder, Ordner und Video angewendet

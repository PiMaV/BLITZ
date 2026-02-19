# Data Sources, Loaders & Build Variants

Architektur-Entscheidungen fuer Standard vs. Full Build und die Einteilung von Loadern, Convertern und Handlern.

---

## 1. Build-Varianten

BLITZ wird in zwei Varianten ausgeliefert:

| Variante | Zielgruppe | Verteilung |
|----------|------------|------------|
| **Standard** | Normale Nutzer (99%) | Pre-compiled EXE via GitHub-Release |
| **Full** | Nutzer mit exotischen Formaten/Backends | Pre-compiled EXE via GitHub-Release |

**Hintergrund:** Die meisten Nutzer bekommen eine fertige EXE. Installation via pip/uv ist fuer sie unrealistisch. Daher werden Plugins nicht zur Laufzeit installiert, sondern als Build-Varianten ausgeliefert.

---

## 2. Einteilungs-Regel: Standard vs. Full

**Regel:** Ein Feature (Loader, Converter oder Handler) gehoert zu **Standard**, solange es nur Bibliotheken nutzt, die bereits im Standard-Build enthalten sind. Sobald **exotische Dependencies** benoetigt werden, wird es ein **Full**-Kandidat.

### Standard-Dependencies (bereits in Standard-Build)

- `csv`, `json`, `os`, `pathlib`, ... (stdlib)
- `numpy`
- `opencv-python-headless` (cv2)
- `PyQt5`
- `pyqtgraph`
- `natsort`
- `requests`
- `python-socketio`
- `psutil`
- `QDarkStyle`

### Exotische Dependencies (fuer Full)

- `omero-py` (OMERO)
- `pydicom` (DICOM-Dateien)
- `pynetdicom` (DICOM-Server/PACS)
- `fhirclient` (FHIR Imaging)
- `openpyxl` / `xlrd` (Excel)
- `h5py` (HDF5)
- `bioformats` / Java-Bridge
- Weitere domain-spezifische Pakete

### Beispiele

| Feature | Typ | Standard / Full | Begruendung |
|---------|-----|----------------|-------------|
| PNG, JPEG, TIFF, BMP | Loader | Standard | cv2 |
| Video (MP4, AVI, MOV) | Loader | Standard | cv2 |
| NumPy (.npy) | Loader | Standard | numpy |
| WebDataLoader (WOLKE) | Handler | Standard | requests, socketio |
| LiveView | Handler | Standard | requests/websocket |
| CSV-Konverter | Converter | Standard | csv, numpy |
| JSON-Config-Import | Converter | Standard | json, numpy |
| OMERO | Handler | Full | omero-py |
| DICOM-Datei | Loader | Full | pydicom |
| DICOM-Server (PACS) | Handler | Full | pydicom/pynetdicom |
| FHIR Imaging | Handler | Full | fhirclient |
| Excel-Konverter | Converter | Full | openpyxl |
| Daikon | Loader | Je nach Implementierung | Nur numpy/stdlib -> Standard; spezielle Lib -> Full |

---

## 3. Drei Typen von Datenquellen

| Typ | Aufgabe | GUI | Ausgabe |
|-----|---------|-----|---------|
| **Loader** | Datei lesen und direkt als Bild darstellen | Nein (nutzt Standard-File-Dialog) | `ImageData` |
| **Converter** | Rohdaten in ein nutzbares Format bringen | Ja (eigener Dialog mit Preview, Optionen) | `ImageData` oder `.npy`-Datei |
| **Handler** | Externe Systeme (Server, Datenbanken) anbinden | Ja (eigener Dialog: Verbindung, Browsing, Auswahl) | `ImageData` oder `.npy`-Datei |

**Gemeinsamer Vertrag:** Alle liefern am Ende `ImageData` oder einen Pfad zu `.npy`. BLITZ behandelt beide Faelle einheitlich.

---

### 3.1 Loader

- Einfach: Pfad -> Array
- Kein eigener Dialog (nutzt Standard File-Open)
- Beispiele: PNG, JPEG, DICOM (Datei), Daikon, .npy

### 3.2 Converter

- Rohdaten -> Bild/Array
- Eigener Dialog: Preview, Spaltenauswahl, Optionen
- Beispiele: CSV -> NPY, Excel -> NPY, JSON -> NPY

### 3.3 Handler

- Komplexe Backends mit eigener UI
- Verbindung, Authentifizierung, Browsing (Projekt/Dataset/Image)
- Beispiele: OMERO, DICOM-Server (PACS), FHIR, WebDataLoader, LiveView

**Handler-Output:** Optional ueber `.npy` zwischenspeichern bei sehr grossen Datasets; sonst direkt `ImageData` in den Viewer.

---

## 4. Aktueller Stand (Standard-Build)

### Loader (bereits vorhanden)

- **Images:** jpg, png, jpeg, bmp, tiff, tif
- **Video:** mp4, avi, mov
- **Arrays:** .npy

### Handler (bereits vorhanden)

- **WebDataLoader:** WOLKE-Integration (Socket.IO + HTTP-Download)

### Geplant fuer Standard

- LiveView (Stream-Source)

### Bereits implementiert (Standard)

- **ASC/DAT-Konverter** (`blitz/data/converters/asc_dat.py`): Dialog mit Rohdaten-Vorschau (5 Zeilen, abgeschnitten), Bild-Vorschau (Plasma-Colormap), Optionen (Delimiter, erste Spalte = Zeilennummer).

---

## 5. Geplant fuer Full

- DICOM-Loader (Datei)
- OMERO-Handler (hohe Prio)
- DICOM-Server / PACS-Handler
- FHIR Imaging (falls benoetigt)
- Weitere exotische Loader (Daikon, Bio-Formats, ...) je nach Bedarf

---

## 6. Technische Architektur (geplant)

### Gemeinsame Schnittstelle

```python
# Vereinfacht – alle Quellen erfuellen dies:
class DataSource(Protocol):
    def provide(self, ...) -> ImageData | Path | None: ...

# Converter/Handler mit eigener GUI:
class HasDialog(Protocol):
    def get_dialog(self, parent) -> QDialog: ...
    def run_and_provide(self, parent) -> ImageData | Path | None: ...
```

### Loader-Registry

- Core definiert Registry
- Bei unbekanntem Suffix: Registry abfragen
- Standard-Build: nur Core-Loader registriert
- Full-Build: zusaetzliche Loader (DICOM, Daikon, ...) registriert

### Build-Prozess

- **Standard:** `pyproject.toml` mit Standard-Dependencies; PyInstaller-Spec ohne Full-Extras
- **Full:** Zusaetzliche `[dependency-groups] full` oder `blitz[full]`; PyInstaller-Spec mit allen Full-Paketen; Init-Code registriert Full-Loader/-Handler

---

## 7. Prioritaeten und naechste Schritte

1. **Architektur-Dokumentation** (dieses Dokument) – erledigt
2. **OMERO-Handler** (hohe Prio) – eigenes Modul, Full-Build
3. **DataSource-Interface** – gemeinsame Schnittstelle einfuehren
4. **Loader-Registry** – Grundgeruest fuer erweiterbare Loader
5. **CSV-Konverter** – Standard, als erster Converter
6. **Dual-Build-Setup** – Standard- und Full-EXE bauen koennen
7. **LiveView** – separates Thema

---

## Referenzen

- `docs/ARCHITECTURE.md` – allgemeine Code-Architektur
- `docs/LOADING.md` – Load-Flow, Dialoge, Session-Defaults

# TODO / Follow-ups

## High

### Ops-Tab (erledigt)
Ops-Tab im Options-Dock: Background-Subtract (File oder Range, Mean/Median). Siehe `docs/TIMELINE_AGGREGATION.md`.

### OMERO-Handler (geplant)
OMERO-Server-Anbindung als Handler (Full-Build). Eigenes UI: Verbindung, Projekt/Dataset/Image waehlen, ImageData oder .npy liefern. Siehe `docs/SOURCES_AND_VARIANTS.md`.

### DataSource-Interface + Loader-Registry (geplant)
Gemeinsame Schnittstelle fuer Loader, Converter, Handler. Registry fuer erweiterbare Loader. Grundlage fuer OMERO, DICOM, etc.

## Medium
- [ ] Dual-Build-Setup: Standard- und Full-EXE bauen koennen
- [ ] CSV-Konverter (Standard): Dialog mit Preview, Spaltenauswahl -> .npy
- [ ] Tests erweitern: ReduceDict Edge-Cases ???
- [ ] Rosee prüfen: Isolines und normalizatione (hier sollte "autozoom" dann aktivieren)
- [ ] Zoom per mausrad nur in einer Achse (in den extractin plots?)

## Low
- [ ] LiveView (Stream-Source) – separates Thema
- [ ] Docs aktualisieren
- [ ] Video: Komprimierte Videodaten -> volle float32-Matrizen. Siehe docs/OPTIMIZATION.md Abschnitt "Video: Komprimierte Daten vs. volle Matrizen" fuer Strategien (mmap, Chunked, Range-Load).


Apply Crop: deaktiviert. Wer croppen will, merkt sich die Zahlen und laedt neu. Spaeter wieder implementierbar.
- Statusbar unten links: Frame X/Y zeigt Idx ODER Filename - nicht redundant zu Idx-Spinner rechts (Spinner = direkte Eingabe, Statusbar = Kontext/Filename)
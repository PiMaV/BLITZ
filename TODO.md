# TODO / Follow-ups

## High

### Fixing


cropping Time vs. Mask

Evtl. sollten wir für aggragate und Frame 2 verschiedene Timelines vorsehen; es ist immer recht schiweirig und nicht intuitv das man von aggragate immer est noch in Frames wechsel muss; irgendwie will man einfach direkt in die Timeline klicken...
---
running mean hat irgendwie auch noch probleme


es kommt zu fehlermeldungen bei BG Sub im livestream

Aggreagte nicht immer verfügbar!
und für mean ist nicht immer die Full range aktiv



ich starte und mache inen graysclae web stream. nun habe ich eine matrix mit 32frames. die Aggragate funktion ost allerdings deaktiviert (schlecht, weil das ja jetzt ein nächster Schritt wäre)
über Ops kann ich dann aber auf open aggraget und zack, habe ich den Tab aktiv.
Mean und co werden berechnet, ABER die Range sollte initial auf FULL sein

Wenn ich von Aggragate dann auf Frame wechsle kommt Fehler:

---


add something (window or docked?) that displays intensity distribution over a polyline.


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

- [ ] LiveView: RAM-Hinweis im Buffer-Dialog (Buffer 1–10000, FPS/Exposure/Gain bereits vorhanden)
- [ ] LiveView: Display-FPS konfigurierbar (aktuell fest 10 FPS)

- Docker testen

## Low
- [ ] Load/Save project file: Checkbox und Auto-Logik voruebergehend entfernt. Explizites Oeffnen von .blitz funktioniert. Siehe docs/SETTINGS.md.
- [ ] LiveView (Stream-Source) – separates Thema
- [ ] Docs aktualisieren
- [ ] Video: Komprimierte Videodaten -> volle float32-Matrizen. Siehe docs/OPTIMIZATION.md Abschnitt "Video: Komprimierte Daten vs. volle Matrizen" fuer Strategien (mmap, Chunked, Range-Load).


Apply Crop: deaktiviert. Wer croppen will, merkt sich die Zahlen und laedt neu. Spaeter wieder implementierbar.
- Statusbar unten links: Frame X/Y zeigt Idx ODER Filename - nicht redundant zu Idx-Spinner rechts (Spinner = direkte Eingabe, Statusbar = Kontext/Filename)
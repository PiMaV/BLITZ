# TODO / Follow-ups

## High

### Naechste Mission: Normalization
Normalization als eigener Tab im oberen Panel (Options-Dock) umsetzen. Bisherige Integration im Time-Tab ersetzen/vereinheitlichen. Siehe `docs/TIMELINE_AGGREGATION.md` Abschnitt "Naechste Mission".

- [X] Spline-Rendering Bug in venv weiter analysieren (Qt/OpenGL?) -> wollen wir autozoom, oder zoom to max? was ist hier sinnvoll? min/max anzeige auf der matrix?
- [X] csv parsing (234595b522f549eac1092255c334c28d947c3d7a / optimized tests and some code maintenance) testen!
- [ ] uv-Migration sauber neu aufsetzen
- [ ] Packaging / PyInstaller testen
- DICOM klappt nicht mehr. hier muss eine entscheidung her! Mitnehmen oder nicht?

## Medium
- [ ] Tests erweitern: ReduceDict Edge-Cases ???
- [ ] Cleanup: old branches löschen
- [ ] neues time reiter testen?
- [ ] Rosee prüfen: Isolines und normalizatione (hier sollte "autozoom" dann aktivieren)
- [ ] Zoom per mausrad nur in einer Achse (in den extractin plots?)
- [X] einfügen von min und max werten (pro Bild UND pro Dataset?) in den vert und horz plots

## Low
- [ ] Docs aktualisieren
- [ ] Video: Komprimierte Videodaten -> volle float32-Matrizen. Siehe docs/OPTIMIZATION.md Abschnitt "Video: Komprimierte Daten vs. volle Matrizen" fuer Strategien (mmap, Chunked, Range-Load).


Apply Crop: deaktiviert. Wer croppen will, merkt sich die Zahlen und laedt neu. Spaeter wieder implementierbar.
- Statusbar unten links: Frame X/Y zeigt Idx ODER Filename - nicht redundant zu Idx-Spinner rechts (Spinner = direkte Eingabe, Statusbar = Kontext/Filename)
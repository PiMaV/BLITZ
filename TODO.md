# TODO / Follow-ups

## High

### Naechste Mission: Normalization
Normalization als eigener Tab im oberen Panel (Options-Dock) umsetzen. Bisherige Integration im Time-Tab ersetzen/vereinheitlichen. Siehe `docs/TIMELINE_AGGREGATION.md` Abschnitt "Naechste Mission".


## Medium
- [ ] Tests erweitern: ReduceDict Edge-Cases ???
- [ ] Rosee prüfen: Isolines und normalizatione (hier sollte "autozoom" dann aktivieren)
- [ ] Zoom per mausrad nur in einer Achse (in den extractin plots?)

## Low
- [ ] Docs aktualisieren
- [ ] Video: Komprimierte Videodaten -> volle float32-Matrizen. Siehe docs/OPTIMIZATION.md Abschnitt "Video: Komprimierte Daten vs. volle Matrizen" fuer Strategien (mmap, Chunked, Range-Load).


Apply Crop: deaktiviert. Wer croppen will, merkt sich die Zahlen und laedt neu. Spaeter wieder implementierbar.
- Statusbar unten links: Frame X/Y zeigt Idx ODER Filename - nicht redundant zu Idx-Spinner rechts (Spinner = direkte Eingabe, Statusbar = Kontext/Filename)
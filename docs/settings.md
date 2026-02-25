# Settings & Storage Strategy

> **Status**: Diskussion verankert, Umsetzung zurueckgestellt. Soll bei naechstem UI-Overhaul oder Settings-Refactoring beruecksichtigt werden.

## Aktueller Stand

- **App-Einstellungen**: Eigenes INIFile (vermutlich `settings.ini` oder aehnlich)
- **Projektdateien**: `.blitz` pro Dataset. **Load/Save-Checkbox voruebergehend entfernt** (war: bei aktivierter Checkbox beim Laden automatisch .blitz laden/speichern). Explizites Oeffnen einer .blitz-Datei (File -> Open) funktioniert weiterhin. Siehe TODO.md "Load/Save project file".
- **Weitere Dateien**: LUT-Export/Import, evtl. Caches

## Diskussionspunkte (offen)

1. **Sichtbarkeit**: Wo werden Settings gespeichert? Soll der Nutzer das sehen (z.B. Pfad im About-Dialog oder Einstellungen)?

2. **Dateien-Flut**: Erzeugt BLITZ zu viele kleine Dateien? 
   - Projektdateien automatisch bei jedem Load?
   - Nur bei explizitem "Sync" oder "Save project"?

3. **Alternativen**:
   - Ein zentrales Config-File (z.B. `~/.config/blitz/settings.ini`) statt verteilter INIs
   - QSettings (Plattform-Standard: Registry / `~/.config`)
   - Projektdateien opt-in statt opt-out

4. **Trennung**: User-Settings (Fenster, LUT-Praeferenz) vs. Projekt-Settings (Pfad, Mask, Crop) vs. Session (temporaere Defaults)

## Naechste Schritte (spaeter)

- **Settings-Datei ueberdenken**: Format, Speicherort und Aufteilung (z.B. `settings.blitz` vs. plattformuebliche Config) neu bewerten; welche Werte Sinn machen und welche nicht (bereinigen/entfernen)
- Entscheidung: Ein File vs. mehrere, Opt-in vs. Opt-out
- Dokumentation des Speicherorts
- Evtl. "Settings"-Button/Dialog mit Speicherort-Info

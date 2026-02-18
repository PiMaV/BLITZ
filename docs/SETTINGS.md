# Settings & Storage Strategy

> **Status**: Diskussion verankert, Umsetzung zurueckgestellt. Soll bei naechstem UI-Overhaul oder Settings-Refactoring beruecksichtigt werden.

## Aktueller Stand

- **App-Einstellungen**: Eigenes INIFile (vermutlich `settings.ini` oder aehnlich)
- **Projektdateien**: `.blitz` pro Dataset (bei Sync oder explizitem Speichern)
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

- Entscheidung: Ein File vs. mehrere, Opt-in vs. Opt-out
- Dokumentation des Speicherorts
- Evtl. "Settings"-Button/Dialog mit Speicherort-Info

# Multicore Loading: Bench, Grenzen, Progress

## 1. Multicore-Logik per Bench pruefen?

**Empfehlung: Ja**, zumindest einmalig validieren.

### Hintergrund
- **Video**: Wurde gebenchmarkt – Multicore war 1.5–3x langsamer (Shared Disk, Codec). Entfernt.
- **Images / ASCII**: Nutzen Multicore, aber die Schwellen (333 Dateien, ~1.3 GB) sind Heuristiken ohne Bench-Vergleich.

### Sinnvolle Bench-Strategie
- **Ziel:** Crossover-Punkt finden (ab wann ist Parallel schneller als Sequential?).
- **Varianten:**
  - Dateizahl variieren: 100, 200, 333, 500, 1000, 2000
  - Groesse pro Datei variieren (klein vs. gross)
- **Metrik:** Sequential vs. Parallel auf gleichem Rechner (SSD vs. HDD getrennt betrachten).
- **Skript:** `python scripts/benchmark_multicore_load.py <folder> [--ascii]` oder `--generate 500` fuer temp Testdaten.
- **Deutung:** Par/Seq < 1 = Parallel schneller. Der *untere Punkt* (kleinster N mit Par schneller) wird als Empfehlung fuer `multicore_files_threshold` ausgegeben – kein Grund, hoher zu gehen.
- **64 GB+ RAM:** Die Groessenschwelle (~1.3 GB) ist irrelevant; massgeblich ist die Dateizahl.

### Boot-Bench (einmalig)
Beim ersten Start prueft BLITZ `boot_bench_done`. Falls False: Splash "Erstoptimierung (Multicore)...", Subprocess laeuft kurzen Bench (Temp-Bilder), setzt optimale Schwellen, dann `boot_bench_done=True`. Spaeter: GPU-Profiling wird ergaenzt.

**Vergleich mit alten Ergebnissen (Previous run):** Derzeit **on hold**. Ergebnisse werden weiter mit Datum in `boot_bench_results/` gespeichert; die Anzeige/Vergleich mit einem vorherigen Lauf ist aus der UI entfernt und kann spaeter wieder aktiviert werden.

### Aufwand vs. Nutzen
- Einmaliger Aufwand (~1–2 h): Script + Ausfuehrung.
- Ergebnis: Evtl. Schwellen anpassen (z.B. 500 statt 333) oder bestaetigen.
- Optional: In CI einbauen, wenn sich ein Referenz-Dataset lohnt.

---

## 2. Hardcoded Grenzen – Bewertung

### Aktuell
- **Einstellungen:** `multicore_files_threshold` (333) und `multicore_size_threshold` (~1.3 GB) liegen bereits in `settings.blitz` – also nicht wirklich hardcoded, sondern konfigurierbar.
- **Default:** Heuristik aus Erfahrung, nicht empirisch gemessen.

### Optionen
| Option | Aufwand | Vorteil |
|--------|---------|---------|
| **A) Bench einmalig** | Gering | Schwellen empirisch validieren/anpassen |
| **B) User-konfigurierbar** | Erledigt | Nutzer kann auf eigener HW anpassen |
| **C) Auto-Tuning** | Hoch | Beim ersten Start oder „Bench“-Tab kurzen Test, optimale Werte vorschlagen |
| **D) Dokumentieren** | Gering | In Tooltip/Help erklaeren: „333/1.3GB sind konservative Defaults; bei vielen Kernen evtl. runter, bei HDD evtl. hoch“ |

**Empfehlung:** A + D. Bench einmalig, Defaults ggf. anpassen, in UI/Tooltip kurz erklaeren. C nur, wenn explizit gewuenscht.

---

## 3. ASCII Loader: Progress Bar bei grossen Mengen?

### Aktueller Stand
- **Sequential** (< 333 Dateien bzw. < 1.3 GB): `progress_callback` pro Datei → Progress Bar laedt sinnvoll.
- **Parallel** (darueber): `message_callback("Loading in parallel (progress not available)...")`, `progress_callback(100)` erst am Ende → kein Fortschritt.

Verhalten entspricht damit dem **Image Loader**.

### Sollte es eine Art Progress geben?
- **Bei grossen Mengen** greift meist Multicore → dann ist heute kein Fortschritt sichtbar.
- **Technische Einschraenkung:** `Pool.starmap` liefert erst am Ende alle Ergebnisse; genaue Zwischenfortschritte sind nicht verfuegbar.

### Moegliche Verbesserungen
| Variante | Aufwand | Nutzen |
|----------|---------|--------|
| **1) So lassen** | 0 | Konsistent mit Images; Message sagt „progress not available“ |
| **2) imap_unordered** | Mittel | Ungefaehrter Fortschritt, weil Ergebnisse nacheinander reinkommen; Progress kann grob geschaetzt werden |
| **3) Nur Message schaerfen** | Gering | z.B. „Loading in parallel (N files)…“ – Nutzer weiss, dass es laeuft |

**Empfehlung:** 1 oder 3. Der Overhead von 2 (imap + Progress-Schaetzung) steht in keinem guten Verhaeltnis zum Nutzen; bei Parallel-Load ist die Dauer oft ohnehin ertraeglich. Optional 3 (bessere Message).

---

## Kurzfassung
- Multicore-Bench: Ja, einmalig sinnvoll.
- Grenzen: Bereits konfigurierbar; Bench zur Bestaetigung/Anpassung der Defaults.
- ASCII Progress: Bei grossen Mengen (Parallel) wie Images ohne Bar; bei wenigen Dateien (Sequential) bereits mit Bar. Zusaetzlich evtl. klarere Message fuer Parallel-Load.

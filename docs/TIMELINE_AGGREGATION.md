# Timeline & Aggregation

Dokumentation des Timeline-Panels und des Aggregate-Modus.

## Timeline-Panel (unten)

Zwei Tabs steuern den Modus:

- **Frame**: Einzelbild-Modus. Idx-Spinner waehlt den Frame.
- **Aggregate**: Aggregations-Modus. Reduce-Methode (Mean, Max, Min, Std) und Range (Start/End) definieren das Ergebnis.

### Frame-Tab

- **Idx-Spinner**: Aktueller Frame (immer aktiv bei geladenen Daten).
- **Upper/lower band**: Checkbox. Zeigt Min-/Max-Kurve (gruenes Band) in der Timeline.
- **Curve**: Dropdown Mean/Median. Aggregation innerhalb des ROI pro Frame fuer die Timeline-Kurve.

### Aggregate-Tab

- **Reduce**: Methode (None - current frame, Mean, Max, Min, Std).
- **Start / End**: Range fuer die Aggregation.
- **Win const.**: Fensterlaenge bleibt beim Aendern von Start/End (Spinner) konstant. Beim Ziehen der Range-Handles passt sich das Window der neuen Spannweite an.
- **Full Range**: Setzt Range auf 0..max (volle Laenge).
- **Update on drag**: Checkbox. Wenn aktiviert, wird die Aggregation waehrend des Range-Drags live aktualisiert (sonst nur beim Loslassen).

---

## Wichtige Designentscheidung: Timeline bei Aggregation

**Problem**: Wenn man im Aggregate-Modus z.B. auf Mean wechselt, wurde die Timeline praktisch unsichtbar (nur ein Punkt, da das Bild auf 2D reduziert wurde).

**Loesung**: Die Timeline zeigt **immer** die volle Zeitserie (ROI-Kurve ueber alle Frames), auch im Aggregate-Modus.

### Datenfluss

| Komponente | Frame-Modus | Aggregate-Modus |
|------------|-------------|-----------------|
| **Bild** | Aktueller Frame | Reduziertes Ergebnis (z.B. Mean ueber Range) |
| **Timeline-Kurve** | ROI-Mean/Median pro Frame | Gleich: ROI-Mean/Median pro Frame ueber **alle** Frames |
| **Datenquelle Timeline** | `getProcessedImage()` | `data.image_timeline` |
| **Range (crop_range)** | Ausgeblendet | Sichtbar, markiert die aggregierte Range |

### `ImageData.image_timeline`

Neue Property in `blitz/data/image.py`:

- Liefert den **vollen Stack** (norm + mask, **ohne** reduce).
- Wird nur im Aggregate-Modus genutzt (`_redop` gesetzt).
- On-the-fly berechnet – kein permanenter RAM-Verbrauch.
- Beruecksichtigt: Norm-Pipeline, Mask (_image_mask, _mask), Crop, Transpose, Flip.

### Implementierung (`blitz/layout/viewer.py`)

- `roiChanged` prueft `in_agg` (data._redop gesetzt und image_timeline vorhanden).
- Bei `in_agg`: ROI-Daten von `data.image_timeline`, X-Werte = `np.arange(n_frames)`.
- X-Range und timeLine-Bounds werden auf 0..n-1 gesetzt.
- Auto-Zoom: `roiPlot.plotItem.vb.autoRange()` nach jedem roiChanged.

---

## Ops-Tab (erledigt)

**Ops** ist ein eigener Tab im Options-Dock. Subtract und Divide nutzen dieselbe Referenz-Quelle:

- **1. Subtract**: Source Off | Aggregate | File, Amount 0-100%
- **2. Divide**: Source Off | Aggregate | File, Amount 0-100%
- **Aggregate**: Verwendet Range und Reduce aus dem Aggregate-Tab (Open Aggregate)
- **File**: Geladenes Referenzbild (Dark Frame, Flat Field)
- Subtract und Divide koennen kombiniert werden (z.B. Subtract: File, Divide: Aggregate)

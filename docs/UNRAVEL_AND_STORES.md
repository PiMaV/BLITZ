# Unravel und Raw/Result Store

## Was macht unravel()?

```python
def unravel(self) -> None:
    self._redop = None
    self._agg_bounds = None
    self._reduced.clear()
```

**Unravel = Moduswechsel**, nicht "Auspacken" oder Datenfreigabe. Es setzt den Aggregationszustand zurueck:

- `_redop = None` → "kein Reduce aktiv" = Frame-Modus
- `_agg_bounds = None` → Aggregations-Bereich geloescht
- `_reduced.clear()` → Reduce-Cache geleert

Die **Raw-Daten (`_image`)** bleiben unveraendert und liegen weiterhin voll im RAM.

## Wann wird unravel aufgerufen?

| Situation | Datei | Zeile |
|-----------|-------|-------|
| Wechsel Frame-Tab → Time Series | main.py | 1380 |
| Aggregate-Tab + "None - current frame" | main.py | 1397 |

## Aktuelles Datenmodell

```
ImageData
├── _image        : Raw-Store (T,H,W,C) – immer im RAM
├── _redop        : None | "Mean" | "Median" | ...  ← Modus
├── _agg_bounds   : (start, end) | None
└── _reduced      : ReduceDict mit Cache (_saved)

image-Property (lazy):
  - _redop is None  → full stack (norm + mask, crop, flip)
  - _redop gesetzt  → _reduced.reduce(to_reduce) → (1,H,W,C)
```

Es gibt **keinen separaten Result-Store**. Der reduzierte Wert wird on-the-fly in `image` berechnet (oder aus dem Reduce-Cache geholt) und an den Viewer uebergeben. Der Viewer speichert ihn in seinem eigenen ImageItem.

## Brauchen wir unravel?

**Ja, aber nur als Modus-Flag.** "unravel" ist schlecht benannt – es meint "Frame-Modus aktivieren".

Mit einem Raw + Result Store waere das klarer:

```
ImageData (Ziel-Architektur)
├── _image_raw    : (T,H,W,C) – unveraendert, immer da
├── _image_result : (1,H,W,C) | None – gecachtes Aggregate
├── _view_mode    : "frame" | "aggregate"
└── _agg_params   : {operation, bounds} fuer Result
```

- **Frame-Modus**: `image` liefert `_image_raw[currentIndex]` (mit Norm/Mask)
- **Aggregate-Modus**: `image` liefert `_image_result` (wenn gueltig) oder berechnet und cached

Dann waere:
- **unravel** ersetzen durch: `_view_mode = "frame"` (Result-Store bleibt liegen, kann bei erneutem Wechsel zu Aggregate wiederverwendet werden, wenn Parameter gleich)
- **Wechsel Frame→Aggregate**: Nur `_view_mode = "aggregate"` + ggf. Berechnung (wenn Result veraltet)
- **Wechsel Aggregate→Frame**: Nur `_view_mode = "frame"` – **kein clear**, kein "Unpacking"

## Vorteile Raw + Result Store

1. **Schneller Tab-Wechsel**: Kein unravel, kein "Unpacking"-Dialog. Einfach View-Mode umschalten.
2. **Besseres Caching**: Bei Wechsel Aggregate→Frame→Aggregate ( gleiche Parameter ) kann der alte Result wiederverwendet werden.
3. **Klare Semantik**: Raw bleibt, Result ist ein abgeleiteter Cache.
4. **Konsistente Cache-Invalidierung**: Result wird invalidiert bei Aenderung von pipeline, mask, bounds, operation – nicht durch "unravel".

## Implementiert (Feb 2026)

- **Raw + Result Store**: `_image_result` und `_result_params` in ImageData.
- **unravel** behaelt den Namen, tut nur noch Moduswechsel (kein clear).
- **Unpacking-Dialog** entfernt – Wechsel Frame/Aggregate ist jetzt sofort.
- **Result-Cache** wird bei Wechsel Aggregate→Frame→Aggregate (gleiche Parameter) wiederverwendet.

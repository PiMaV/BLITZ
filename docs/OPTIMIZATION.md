# Optimization Report

## Current Performance Analysis

BLITZ is designed for high-performance in-memory analysis. The current implementation uses `multiprocessing` to parallelize image loading and processing, which is effective for CPU-bound tasks like JPEG decoding.

### Key Strengths
*   **Parallel Loading**: Uses `multiprocessing.Pool` to load and resize images concurrently.
*   **Memory Guard**: Automatically downsamples (subsets) data if the estimated size exceeds the user-defined RAM limit (`adjust_ratio_for_memory`).
*   **Fast Operations**: Uses `numpy` for vectorized operations (normalization, reduction).

## Identified Bottlenecks & Recommendations

### 1. Memory Spikes during Loading
**Issue:** The `_load_folder` and `_load_video` methods collect loaded images in a Python list (`matrices` or `frames`) and then call `np.stack()`.
*   **Impact:** This temporarily requires 2x the memory of the final dataset (1x for the list of arrays, 1x for the contiguous stacked array), potentially causing Out-Of-Memory (OOM) errors on machines with limited RAM relative to the dataset size.
*   **Recommendation:** Pre-allocate the final numpy array using `np.empty` or `np.zeros` once the shape of the first image is known and the total count is determined. Fill this array directly or via shared memory in multiprocessing.

### 2. Large `.npy` File Loading
**Issue:** `np.load(path)` loads the entire array into memory.
*   **Impact:** If a source `.npy` file is larger than available RAM, the application will crash immediately.
*   **Recommendation:** Use `np.load(path, mmap_mode='r')`. This maps the file to memory without loading it all at once. Slicing and processing can then happen on chunks of the data, allowing for processing of datasets larger than RAM (if coupled with chunked processing) or at least avoiding the initial load spike.

### 3. Video Loading Efficiency
**Issue:** `_load_video` uses `video.grab()` to skip frames.
*   **Impact:** While faster than `read()`, `grab()` still incurs some overhead.
*   **Recommendation:** For significant subsampling, seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES, ...)`) might be faster, though it depends on the video codec's keyframe structure.

### 4. Pure Python/Numpy vs. Compiled Extensions
**Issue:** Some complex per-pixel operations might be slow in pure Numpy if they involve many temporary array allocations.
*   **Recommendation:** Since Numba usage was removed, consider bringing it back for specific "hot paths" (like custom normalization filters) if profiling indicates a CPU bottleneck.

## Video: Komprimierte Daten vs. volle Matrizen

### Problem
Eine 13 MB MP4-Datei wird dekodiert und als vollstaendige Matrix im RAM gehalten. Die Dateigroesse hat nichts mit dem RAM-Bedarf zu tun – komprimierte Videodaten werden in volle Pixelmatrizen expandiert.

### Aktueller Stand (uint8 Default)
ImageData haelt Daten standardmaessig als **uint8** (1 Byte/Pixel), nicht float32. Das spart ~75% RAM.
* 1920x1080, 30 fps, 1 Minute = 1800 Frames
* uint8: ~6 GB (statt ~24 GB mit float32)

Normalize und Reduce konvertieren nur bei Bedarf zu float. BLITZ ist matrix-basiert – die Matrix bleibt im RAM fuer schnelle OPs.

### Abhaengigkeiten im Code
Folgende Features benoetigen **vollen temporalen Zugriff** auf alle Frames:

| Feature | Datei | Benoetigt |
|---------|-------|-----------|
| Frame-Anzeige | viewer.py | Nur aktueller Frame |
| Reduce (Mean/Max/Min/Std) | ops.py, image.py | Alle Frames auf einmal |
| Normalisierung (window_lag) | image.py, sliding_mean_normalization | Alle Frames |
| Extraction Envelope (min/max ueber Dataset) | widgets.py `_compute_dataset_envelope` | Alle Frames, Schleife ueber n_frames |
| ROI-Kurven | viewer.py | Alle Frames fuer Envelope |
| Crop, Mask | image.py | Array-Slicing |

### Optimierungsstrategien (ueberblick)

#### A) Decode-to-Disk + mmap (empfohlen, mittlerer Aufwand)
**Idee:** Video einmal dekodieren, als rohe float32-Datei auf Disk schreiben. Danach mit `np.memmap` oeffnen – das OS paged nur bei Bedarf ein.

* **Vorteile:** Alle bestehenden Features funktionieren (Indexing, Reduce, Norm, Envelope). Kein grosser RAM-Peak.
* **Nachteile:** Zusaetzlicher Speicherplatz (temp-Datei), initiale Decode-Zeit.
* **Implementierung:** Neuer Loader-Pfad, der nach Decode `np.memmap(path, mode='r', dtype='float32', shape=(n,h,w,c))` zurueckgibt. ImageData muss `_image` als memmap akzeptieren (numpy-Array-API).

#### B) Chunked Loading / Sliding Window (hoher Aufwand)
**Idee:** Nur N Frames im RAM (z.B. 200–500). Beim Scrollen: Chunk laden, alten Chunk freigeben.

* **Vorteile:** Begrenzter RAM, unbegrenzte Videolaenge moeglich.
* **Nachteile:** Reduce, Normalisierung (window_lag), Dataset-Envelope brauchen alle Frames – entweder deaktivieren oder mehrfach durch Datei streamen (langsam).
* **Implementierung:** ImageData-Ersatz mit `__getitem__`-Interface, das bei Zugriff auf Frame i den passenden Chunk laedt. Viele Stellen im Code muessten angepasst werden.

#### C) On-Demand Frame Decode (nur fuer einfache Nutzung)
**Idee:** Kein Vorladen. Bei Zugriff auf Frame i: VideoCapture oeffnen, seek(i), read().

* **Vorteile:** Minimaler RAM.
* **Nachteile:** Seek bei vielen Codecs langsam (keine Keyframes). Reduce, Norm, Envelope wuerden N-mal durchs Video iterieren – extrem langsam.
* **Einsatz:** Nur fuer "Preview-Modus" ohne Reduce/Norm/Envelope.

#### D) Zwei-Stufen: Thumbnail + Range-Load
**Idee:** Zuerst niedriges Preview laden (z.B. 10% Groesse, 10. Frame). Nutzer waehlt Bereich, dann nur diesen Bereich voll laden.

* **Vorteile:** Schneller Start, volle Features im gewaehlten Bereich.
* **Nachteile:** Zwei Load-Pfade, UX-Komplexitaet.
* **Implementierung:** Neuer Dialog "Range auswaehlen" mit Preview-Timeline.

### Prioritaet
**Auslagern auf Platte (mmap/Chunked):** Low Priority. BLITZ ist fuer In-RAM-Matrizen und schnelle OPs ausgelegt. uint8-Default deckt den Grossteil der Faelle ab.

---

## Video Multicore: Analyse und Entscheidung

### Benchmark-Ergebnisse (vor Entfernung)

| Videogroesse | 1C | 2C | 4C | Aussage |
|--------------|----|----|-----|---------|
| 13 MB (subset 0.1) | 0.7s | 2.1s (+215%) | 2.2s (+232%) | **Multicore 3x langsamer** |
| 300 MB (subset 0.1) | 3.5s | 5.2s (+49%) | 5.0s (+42%) | **Multicore ~1.5x langsamer** |

→ Multicore ist bei Video in allen Faellen langsamer; bei kleinen Videos massiv (3x).

### Warum Multicore bei Video kontraproduktiv ist

1. **Geteilte Ressource**: N Prozesse lesen dieselbe Videodatei. SSD/Disk wird zum Flaschenhals – konkurrierende Reads, kein Nutzen durch Parallelitaet.

2. **Codec-Struktur**: H.264/MPEG nutzen Inter-Frame-Kompression. Um Frame 5000 zu dekodieren, muss oft ab dem letzten Keyframe (z.B. 0 oder 2400) dekodiert werden. Jeder Worker seeked zu seinem Start-Frame und dekodiert vorwaerts – effektiv wird das Video **mehrfach** dekodiert. Der single-thread Ansatz streamed einmal sequentiell durch = optimal.

3. **RAM-Spitze**: N Worker halten je ihren Chunk im RAM. Vor dem `np.concatenate` haben wir N vollstaendige Chunks = deutlich hoeherer Peak als 1C. Kann zu Swapping fuehren.

4. **Overhead**: Fork, IPC, Serialisierung der Ergebnisse zwischen Prozessen.

### Entscheidung

**Multicore fuer Video entfernt** (Stand: Feb 2026). Der Mehrwert rechtfertigte weder den Code-Aufwand noch die UX – Nutzer wurden bei Aktivierung langsamer. Single-Core ist fuer Videodecoding der sinnvolle Default.

**Benchmark** (misst nur noch Single-Core): `python scripts/benchmark_video_load.py` – Variablen (path, n_runs, subset_ratio) im Script anpassen.

---

## Load-Profile (42k Dateien @ subset 0.01)

**Script:** `python scripts/profile_load.py` (App starten, Load, schliessen) → `python scripts/profile_load.py --stats`  
**Output:** `profile_stats.txt` (oder `-o andere_datei.txt`)

### Typische Ergebnisse (Feb 2026)

| Bereich | cumtime | ncalls | Aussage |
|---------|---------|--------|---------|
| `emit` (Qt-Signale) | ~21 s | 2545/905 | **Hauptkost**: Signal-Kaskaden nach Load |
| `natsorted` | ~4.8 s | 8 | 337k natsort_key-Aufrufe (42k Pfade) |
| `load_data` | ~3.6 s | 2 | Erwartbar |
| `get_image_preview` | ~2.4 s | 2 | Dialog-Thumbnails |
| `draw_line` | ~0.04 s | 22 | Lazy-Logik greift, kein Flaschenhals |
| `roiChanged` | ~0.5 s | 20 | SetImage/roiClicked aus pyqtgraph |

→ Die "5x Wartezeit mit niedriger CPU" kommt von **~21 s Signal-Verarbeitung** (emit), nicht von Rechenlast.

---

## Was macht die emit-Kaskade? (Ours vs PyQtGraph)

**Ausloeser:** `load_data` → `setImage(matrix)` → Kaskade startet.

### Ablauf (vereinfacht)

```
setImage(matrix)
  ├─ super().setImage()     [PyQtGraph ImageView]
  │    ├─ Image-Item, Histogramm, Timeline updaten
  │    ├─ ROI an neue Groesse anpassen  →  roi.sigRegionChanged
  │    └─ Interne PyQtGraph-Signale (viele)
  │
  ├─ init_roi()             [Unser Code]
  │    └─ roi.sigRegionChanged → roiChanged
  │
  └─ image_changed.emit()   [Unser Code]
       ├─ MeasureROI.reshape
       ├─ ExtractionPlot.draw_line
       └─ ExtractionPlot._invalidate_dataset_envelope_cache
```

**roiChanged** wird getriggert von: `ImageView.setImage` (intern) und `ImageView.roiClicked` (intern) – also PyQtGraph ruft unsere Slots auf, weil wir `roi.sigRegionChanged.connect(roiChanged)` verbunden haben.

### Verantwortung

| Bereich | Herkunft | Aenderbar? |
|---------|----------|------------|
| Histogramm, Image-Item, Timeline | PyQtGraph | Nein (Library) |
| ROI-Updates aus setImage | PyQtGraph | Nein |
| roiChanged (unser Slot) | Unser Code, von PG getriggert | Ja |
| image_changed → draw_line, reshape | Unser Code | Ja |
| reset_options → crop_range, spinboxes | Unser Code | Ja (blockSignals erledigt) |

**Takeaway:** Der Grossteil der Kaskade kommt von **PyQtGraphs setImage-Interna** (Histogramm, ROI, Timeline). Wir steuern nur einen Teil (image_changed, roiChanged-Implementierung, reset_options). Weitere Optimierung waere: (a) image_changed-Slots debouncen/deferren, (b) roiChanged verschlanken oder lazy machen. PyQtGraph selbst aendern ist kein realistischer Pfad – entweder damit leben oder Display-Pipeline (setImage, Histogramm) durch Custom-Logik ersetzen (groesserer Aufwand).

---

## Lazy Extraction Plots

**Datei:** `blitz/layout/widgets.py` (ExtractionPlot)

**Logik:** `draw_line` wird nur ausgefuehrt, wenn der Plot sichtbar ist (`isVisible()`).
Wenn Docks immer offen sind, bringt das keinen Nutzen – dann laeuft draw_line wie zuvor.

| Ort | Code | Wirkung |
|-----|------|---------|
| `draw_line()` | `if not self.isVisible(): self._stale = True; return` | Keine Berechnung bei verstecktem Dock |
| `showEvent()` | `if self._stale: self._stale = False; self.draw_line()` | Nachziehen, wenn Plot sichtbar wird |

---

## Qt emit-Kaskaden reduzieren (reset_options)

**Problem:** Nach Load verbrauchen ~21 s Qt-Signal-Emissionen (emit-Kette).
**Massnahme:** `blockSignals(True)` waehrend des kompletten `reset_options`-Batch-Updates.

**Datei:** `blitz/layout/main.py`

| Widgets blockiert | Triggerte Slots (ohne Block) |
|-------------------|------------------------------|
| crop_range | update_crop_range_labels, _on_crop_range_for_ops -> apply_ops |
| spinbox_crop_range_*, spinbox_selection_window | _on_crop_range_*, apply_ops |
| combobox_reduce, slider_ops_* | apply_ops |
| timeline_tabwidget | _on_timeline_tab_changed |

`apply_ops()` wird am Ende von `_reset_options_body` explizit einmal aufgerufen.

## Background Subtraction Optimization (Implemented)

**Issue:** Previous implementation used `float64` for all operations (subtract, divide) and created multiple full-size copies of the array.
**Optimization:**
- Switched to `float32` (halves memory usage).
- Used in-place operations (`-=`, `/=`) where possible.
- Used `np.nan_to_num(copy=False)` to avoid extra copies.
- Strict type handling for `amount` (float) mixed with `float32` arrays to avoid upcasting to `float64`.

**Result:** ~4x speedup and ~3x memory reduction for background subtraction tasks.
## Threaded Reduction Operations (Mar 2026)

**Optimization:** `ReduceOperation` (Mean, Std, Min, Max, Median) now uses `ThreadPoolExecutor` for arrays larger than 10 MB.
*   **Result:** ~3.4x speedup on 4 cores.
*   **Mechanism:** Numpy releases the GIL for these operations, allowing true parallelism. The array is split along the spatial height axis, processed in chunks, and concatenated.
*   **Heuristic:** Threading is disabled for small arrays (<10 MB) or non-spatial reductions to avoid overhead.

## Numba Candidates
See [NUMBA_CANDIDATES.md](NUMBA_CANDIDATES.md) for a list of functions identified for future Numba optimization.

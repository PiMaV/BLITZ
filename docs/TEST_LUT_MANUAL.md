# Manual LUT Panel Test Procedure

Use this checklist to verify LUT update, Log checkbox, and Sync behaviour.

## Prerequisites

- Start BLITZ: `python -m blitz`
- Load a stack (File > Load or drag folder), e.g. grayscale or RGB

---

## 1. Update (Autofit + Percentile)

| Step | Action | Expected |
|------|--------|----------|
| 1.1 | Click "Autofit" | Image levels recalc, spinners show min/max (default) |
| 1.2 | Change Clip to 5% | Levels update to 5–95 percentile |
| 1.3 | Change Clip back to "Min/Max" (0) | Levels = full min/max |
| 1.4 | Toggle Auto fit off, change frame | Spinners and image stay unchanged |
| 1.5 | Toggle Auto fit on, change frame | Levels and spinners update per frame |

---

## 2. Log Checkbox

| Step | Action | Expected |
|------|--------|----------|
| 2.1 | Enable "Log counts" | Histogram Y-axis becomes log scale |
| 2.2 | Disable "Log counts" | Histogram Y-axis back to linear |
| 2.3 | With Log on, load new image | Log state preserved, histogram shows log |
| 2.4 | Change frame with Log on | Histogram recalc, Y-axis max updates, stays in log mode |

---

## 3. Sync (Spinners ↔ Histogram)

| Step | Action | Expected |
|------|--------|----------|
| 3.1 | Drag LUT region (gradient region) | Spinners update to match within ~80 ms |
| 3.2 | Edit spinner min/max manually | Histogram region updates immediately |
| 3.3 | Click Autofit | Spinners match new levels |
| 3.4 | Change frame (Auto fit on) | Spinners match new frame levels |
| 3.5 | No image loaded | No crash when opening LUT (spinners empty or default) |

---

## Quick Automated Sanity Check

```bash
uv run pytest tests/test_lut.py -v
```

This runs unit tests for `calculate_lut_levels` only. Integration (Update, Log, Sync) must be verified manually with the checklist above.

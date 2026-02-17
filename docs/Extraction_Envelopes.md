# Extraction Plot Envelopes

Documentation of the envelope functionality for the horizontal (H) and vertical (V) extraction plots in BLITZ.

## Overview

The extraction plots display intensity profiles along the crosshair lines. Envelopes add min/max (or percentile) curves that visualize the spread or variability of the signal at each position. All envelope types share the same percentile setting and use the crosshair width defined in **View > Crosshair**.

## Controls

Available under **View > Extraction plots**:

| Control | Description |
|--------|-------------|
| **Min/Max per image** | Envelope over the full orthogonal axis of the current frame |
| **Envelope per crosshair** | Envelope over the crosshair band thickness (current frame) |
| **Envelope per position (dataset)** | Envelope over all frames (same crosshair band as mean curve) |
| **Envelope: X%** | Mode: 0 = Min/Max (hard); 1–49 = percentile (e.g. 5 gives 5th and 95th percentile) |

The percentile spinbox applies to all three envelope types.

## Envelope Types

### 1. Min/Max per Image

- **Horizontal plot (H):** At each column position x, min and max over *all rows* (full vertical axis). Shows the full spread along the horizontal profile.
- **Vertical plot (V):** At each row position y, min and max over *all columns* (full horizontal axis). Shows the full spread along the vertical profile.

This envelope ignores the crosshair position and uses the complete image. Useful for seeing the global intensity range along the profile axis.

**Color:** Green (dark/light for min/max)

### 2. Envelope per Crosshair

- **Horizontal plot:** At each position along the horizontal profile, min/max (or percentile) over the vertical band defined by the horizontal line width.
- **Vertical plot:** At each position along the vertical profile, min/max (or percentile) over the horizontal band defined by the vertical line width.

Uses the same band as the mean extraction curve. Shows the spread within the crosshair thickness for the current frame.

**Color:** Teal (dark/light for min/max)

### 3. Envelope per Position (Dataset)

- Uses the same crosshair band as the mean curve and the per-crosshair envelope; width is fully applied.
- At each profile position, computes min/max (or percentile) over *all frames* in the dataset.

Reveals temporal variability: how the intensity at each position changes across the time series. Requires iterating over all frames; result is cached until the crosshair or image changes.

**Color:** Blue (dark/light for min/max)

## Percentile Mode

- **0% (displayed as "Min/Max"):** Hard min and max. Sensitive to outliers and noise.
- **1–49%:** Uses the lower and upper percentiles instead of min/max. For example, 5% yields the 5th and 95th percentile, reducing the influence of spikes.

The same percentile value applies to all three envelope types.

## Design Notes

- **Crosshair width:** The horizontal and vertical spinboxes in **View > Crosshair** define the band width. Width 0 corresponds to the center line (no band).
- **Axis convention:** Image shape is `(frames, width, height)`. Horizontal plot x-axis = columns; vertical plot y-axis = rows.
- **Caching:** The dataset envelope is cached per crosshair position and width. It is invalidated when the image changes or the crosshair moves.

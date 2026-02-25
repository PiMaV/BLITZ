# ROI Loading Merge Guide

## What Went Wrong

The commit `b1afd88` ("Add ROI spinners, rotation, and transforms to load dialogs") introduced a **broken merge**:

1. **New loading feature** (working):
   - `dialogs.py`: Video/Image/ASCII load dialogs have ROI spinners (X, Y, W, H) and Flip XY via `ROIMixin`
   - `roi_mixin.py`: Shared ROI controls
   - `viewer.py`: `transpose` (Flip XY) support in `manipulate()`
   - `data/image.py`: Transform support

2. **main.py rewrite** (broken):
   - Removed ~1500 lines including critical methods: `on_strgC`, `update_crop_range_labels`, `update_statusbar`, `apply_mask`, `apply_ops`, `reset_options`, `setup_sync`, etc.
   - Left connections/calls to these methods -> `AttributeError` on startup
   - The new main.py was incomplete; it never passed the "run app" test before merge

3. **Recovery**:
   - `main.py` was restored from `build-1.5.2` (commit `9379f47`)
   - App runs again, but there was a **mismatch**: dialogs return `flip_xy`, `roi_state`, `mask_rel`; `DataLoader` does not accept these; main must pop them before passing to `load_data`.

## Current State

| File              | Version  | Notes                                |
|-------------------|----------|--------------------------------------|
| `main.py`         | 9379f47 + Option A fix | Full, working; params filtering for new dialogs |
| `dialogs.py`      | b1afd88  | New ROI/transforms in dialogs        |
| `roi_mixin.py`    | b1afd88  | New                                  |
| `viewer.py`       | b1afd88  | Has `transpose`                      |
| `data/image.py`   | b1afd88  | Transform support                    |

## Implemented Fix (Option A)

The minimal fix has been applied to `main.py`:

- **load_images()**: Pop `flip_xy`, `roi_state`, `mask_rel`, `step`, `frame_range` before passing to `DataLoader`. Convert `step` -> `subset_ratio`. After load, apply flip_xy (transpose).
- **_load_ascii()**: Filter `flip_xy`, `roi_state`, `mask_rel` from params before `load_ascii()`. After load, apply flip_xy.

The new load dialogs (ROI spinners, Flip XY) work; Rotate 90 and Angle were removed.

## Two Separate ROIs (Design)

- **Load ROI**: Only in the load dialog. Determines which data is loaded (crop at load time). Saved as numbers (roi_state, mask_rel) for session defaults. **Completely disconnected** from the program ROI.
- **Program ROI**: The viewer's square_roi/poly_roi for analysis, masking, etc. Independent; uses init_roi default after load. Never receives load ROI values.

## ROI Handle, Spinner, and Cropping Fixes

1. **Handle behaviour**: Upper-left handle was scaling the whole ROI instead of moving that corner. Fixed by removing all default handles in `_connect_roi_signals` and adding four corner scale handles plus a rotate handle on the right edge (avoids overlap with top-left).
2. **Spinner width**: Increased `setMaximumWidth` from 70 to 95 for X/Y/W/H, and from 60 to 75 for Angle.
3. **ROI cropping**: ROI mask used at load time. Flip XY only; Rotate 90 and Angle removed.
4. **No ROI sync after load**: Load ROI is not applied to the viewer. Avoids freezes and keeps the two ROIs independent.

## Future: Full ROI Session Persistence

To remember ROI and transforms between loads (when "Always show load dialog" is off):

1. `flip_xy`, `roi_state` already in session defaults. Done.
2. When the dialog is NOT shown, retrieve these from session defaults and apply after load.
3. Add `_is_roi_valid()` sanity check (from b1afd88) to reset ROI when switching to a different-sized image.

## TIFF Warnings

The OpenCV TIFF warnings (`Unknown field with tag 292`, `tags are not sorted`, `ASCII value does not end in null byte`) come from TIFF metadata. They are harmless; images still load. To reduce log noise, you can set `cv2.logLevel` or ignore these at the OpenCV level if desired.

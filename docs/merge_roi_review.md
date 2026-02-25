# ROI Loading Branch – Merge Review

## Original Scope (b1afd88)

> "Add ROI spinners, rotation, and transforms to load dialogs"

- ROI spinners (X, Y, W, H) in load dialogs
- Transforms: Flip XY, Rotate 90°
- Angle spinner for rotated ROI
- Load ROI separate from Program ROI

---

## Implemented vs. Removed

| Feature              | Status   | Notes |
|----------------------|----------|-------|
| ROI Spinners X,Y,W,H | Done     | In Video/Image/ASCII dialogs via ROIMixin |
| Flip XY (Transpose)  | Done     | Works; saved in session defaults |
| Angle Spinner        | Removed  | Caused confusion; rotation broken |
| Rotate 90°           | Removed  | Broken in preview; Flip XY sufficient |
| ROI Mask at Load     | Done     | Axis-aligned crop via mask; works |
| Rotated ROI Crop     | Removed  | Perspective warp was unreliable; reverted |
| flip_xy Persistence   | Done     | In session defaults (ASCII, Video, Image) |
| roi_state Persistence| Done     | In session defaults when dialog returns it |
| Load ROI ≠ Program ROI| Done     | Load ROI not applied to viewer ROI |

---

## Files Changed (vs. b1afd88)

- **roi_mixin.py**: Removed chk_rotate_90, spin_roi_angle, addRotateHandle; simplified _get_roi_source_mask (angle=0 only); Flip XY only
- **dialogs.py**: No rotate_90 in params; no angle in roi_state; no chk_rotate_90 refs
- **main.py**: Params filtering (skip_keys); flip_xy + roi_state in session defaults; flip_xy applied when loading without dialog; no crop_roi_bbox
- **image.py**: Removed crop_roi_bbox (was crop_rotated_roi before)
- **tools.py**: Removed crop_rotated_rect, _rotated_roi_corners
- **merge_roi_loading.md**: Updated to current state

---

## Pre-Merge Checklist

1. **Basic Load**: Video, Image, ASCII load with ROI → mask applied
2. **Flip XY**: Toggle Flip XY in dialog → loaded image transposed
3. **flip_xy Persistence**: Load with Flip XY on → load another file (same type) → Flip XY still on
4. **ROI + Flip XY**: Draw ROI, enable Flip XY → load → correct region, transposed
5. **No-Dialog Load**: With "Always show dialog" off → load same type → mask + flip_xy from session
6. **Program ROI**: Load with ROI crop → viewer ROI stays at default (full frame)
7. **Run packaged app**: `python -m blitz` and PyInstaller build both work

---

## What We Simplified (vs. Original)

- No Rotate 90° (axis mixup, broken)
- No Angle/Rotate handle (rotation never worked correctly)
- No perspective warp for rotated ROI (bbox/mask only)
- V2: no backward-compat pop for rotate_90

---

## Recommendation

The branch now delivers the core ROI-with-spinners behaviour without rotation complexity. Scope is reduced but stable. Merge when the checklist passes.

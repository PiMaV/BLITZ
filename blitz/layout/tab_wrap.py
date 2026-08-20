"""Pack tab sizes into wrapping rows (no Qt)."""

from __future__ import annotations


def wrap_tab_rects(
    sizes: list[tuple[int, int]],
    width: int,
    *,
    h_gap: int = 2,
    v_gap: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Pack ``(w, h)`` sizes into rows. Returns ``(x, y, w, h)`` per tab.

    A tab wider than ``width`` still starts a new row rather than overflowing
    the previous one. Empty ``sizes`` yields an empty list.
    """
    rects: list[tuple[int, int, int, int]] = []
    x = 0
    y = 0
    row_h = 0
    inner_w = max(1, int(width))
    for w, h in sizes:
        tw = max(1, int(w))
        th = max(1, int(h))
        if x > 0 and x + tw > inner_w:
            x = 0
            y += row_h + v_gap
            row_h = 0
        rects.append((x, y, tw, th))
        x += tw + h_gap
        row_h = max(row_h, th)
    return rects


def wrap_height(rects: list[tuple[int, int, int, int]], *, pad: int = 2) -> int:
    """Total height of a packed tab strip."""
    if not rects:
        return 0
    return max(y + h for _x, y, _w, h in rects) + pad

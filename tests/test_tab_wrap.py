"""Wrapping tab layout (Options dock)."""

from blitz.layout.tab_wrap import wrap_height, wrap_tab_rects


def test_wraps_onto_second_row() -> None:
    sizes = [(80, 28)] * 6
    rects = wrap_tab_rects(sizes, width=250)
    ys = {y for _x, y, _w, h in rects}
    assert len(ys) == 2
    assert wrap_height(rects) >= 28 * 2


def test_narrow_dock_keeps_later_tabs_visible() -> None:
    names = [
        (40, 28),
        (44, 28),
        (40, 28),
        (52, 28),
        (60, 28),
        (52, 28),
        (40, 28),
        (52, 28),
        (60, 28),
        (36, 28),
    ]
    rects = wrap_tab_rects(names, width=260)
    assert len(rects) == 10
    last_x, last_y, _last_w, _last_h = rects[-1]
    assert last_y > 0
    assert last_x < 260


def test_empty() -> None:
    assert wrap_tab_rects([], 200) == []
    assert wrap_height([]) == 0

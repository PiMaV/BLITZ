"""Tests for folder_scan grouping and naming clusters."""

from pathlib import Path

import numpy as np
import pytest

from blitz.data.folder_scan import (
    loadable_groups,
    naming_cluster_for_stem,
    scan_folder,
    should_show_chooser,
)


def test_naming_clusters():
    assert naming_cluster_for_stem("001") == "numbered"
    assert naming_cluster_for_stem("frame_012") == "numbered"
    assert naming_cluster_for_stem("img_001") == "numbered"
    assert naming_cluster_for_stem("volt=4kV_exp=10us") == "schema"
    assert naming_cluster_for_stem("photo") == "plain"


def test_scan_mixed_suffixes(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff")
    np.save(tmp_path / "c.npy", np.zeros((2, 2), dtype=np.uint8))
    groups = scan_folder(tmp_path, probe_hikmicro=False, probe_ascii_txt=False, split_naming=False)
    kinds = {g.kind for g in loadable_groups(groups)}
    assert "image" in kinds
    assert "array" in kinds
    assert should_show_chooser(groups)


def test_scan_single_type_no_chooser(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"{i}.png").write_bytes(b"x")
    groups = scan_folder(tmp_path, probe_hikmicro=False, probe_ascii_txt=False, split_naming=False)
    assert not should_show_chooser(groups)


def test_ascii_txt_probe(tmp_path: Path):
    p = tmp_path / "grid.txt"
    p.write_text("1\t2\t3\n4\t5\t6\n7\t8\t9\n", encoding="utf-8")
    from blitz.data.converters.ascii import is_ascii_path

    assert is_ascii_path(p)
    groups = scan_folder(tmp_path, probe_hikmicro=False, split_naming=False)
    ascii_g = [g for g in groups if g.kind == "ascii"]
    assert len(ascii_g) == 1
    assert ascii_g[0].count == 1

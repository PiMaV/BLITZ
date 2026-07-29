"""Scan a folder into loadable groups for the folder chooser dialog.

Groups by suffix (images / video / npy / ascii). Optional naming-schema
sub-clusters for image suffixes. Optional HIKMICRO thermo offer when
radiometric JPEGs are detected.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from natsort import natsorted

from .load import ARRAY_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

ASCII_EXTENSIONS = (".asc", ".dat", ".txt")
PROBE_SAMPLE_N = 8

GroupKind = Literal[
    "image",
    "video",
    "array",
    "ascii",
    "hikmicro_celsius",
    "other",
]


@dataclass
class LoadGroup:
    """One chooser row: a set of files the user can load together."""

    id: str
    label: str
    kind: GroupKind
    count: int
    paths: list[Path]
    suffix: str = ""
    naming_cluster: str | None = None
    extra: dict = field(default_factory=dict)


_RE_NUMBERED = re.compile(r"^(\d+)$")
_RE_TRAILING_FRAME = re.compile(r"^(.*?)([_-]?)(\d{2,})$")
_RE_KEY_VALUE = re.compile(r"[a-zA-Z]{2,}=")


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def list_folder_files(root: Path, *, recursive: bool = False) -> list[Path]:
    """Non-hidden files in root (non-recursive by default — matches DataLoader)."""
    if not root.is_dir():
        return []
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()
    out: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if _is_hidden(p):
            continue
        if recursive:
            rel_parts = p.relative_to(root).parts[:-1]
            if any(part.startswith(".") for part in rel_parts):
                continue
        out.append(p)
    return natsorted(out)


def naming_cluster_for_stem(stem: str) -> str:
    """Lightweight BLITZ-local naming cluster: numbered | schema | plain."""
    if _RE_NUMBERED.match(stem):
        return "numbered"
    if "=" in stem or _RE_KEY_VALUE.search(stem):
        return "schema"
    m = _RE_TRAILING_FRAME.match(stem)
    if m:
        prefix = m.group(1)
        digits = m.group(3)
        if prefix == "":
            return "numbered"
        parts = [p for p in re.split(r"[_\-]+", prefix) if p]
        letter_parts = [p for p in parts if re.search(r"[a-zA-Z]{2,}", p)]
        # Multiple letter tokens in prefix (e.g. camA_runB_001) → schema
        if len(letter_parts) >= 2:
            return "schema"
        if len(digits) >= 2:
            return "numbered"
    parts = [p for p in re.split(r"[_\-]+", stem) if p]
    letter_parts = [p for p in parts if re.search(r"[a-zA-Z]{2,}", p)]
    if len(letter_parts) >= 2:
        return "schema"
    return "plain"


def _cluster_label(cluster: str) -> str:
    return {
        "numbered": "Numbered",
        "schema": "Named schema",
        "plain": "Other names",
    }.get(cluster, cluster)


def probe_ascii_matrix(path: Path) -> bool:
    """True if file looks like a rectangular numeric ASCII matrix."""
    from .converters.ascii import is_ascii_path

    return is_ascii_path(path)


def probe_hikmicro_jpeg(path: Path) -> bool:
    from .converters.hikmicro import is_hikmicro_radiometric_jpeg

    return is_hikmicro_radiometric_jpeg(path)


def scan_folder(
    root: Path,
    *,
    recursive: bool = False,
    split_naming: bool = True,
    probe_hikmicro: bool = True,
    probe_ascii_txt: bool = True,
) -> list[LoadGroup]:
    """Return loadable groups for *root* (sorted for stable UI)."""
    files = list_folder_files(root, recursive=recursive)
    if not files:
        return []

    by_suffix: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_suffix[f.suffix.lower()].append(f)

    groups: list[LoadGroup] = []
    other_paths: list[Path] = []

    # Images — optionally split by naming cluster per suffix
    image_suffixes = [s for s in by_suffix if s in IMAGE_EXTENSIONS]
    jpeg_paths: list[Path] = []
    for suf in sorted(image_suffixes):
        paths = natsorted(by_suffix[suf])
        if suf in (".jpg", ".jpeg"):
            jpeg_paths.extend(paths)
        if split_naming and len(paths) >= 4:
            clusters: dict[str, list[Path]] = defaultdict(list)
            for p in paths:
                clusters[naming_cluster_for_stem(p.stem)].append(p)
            # Only split if ≥2 non-empty clusters with meaningful sizes
            nonempty = {k: v for k, v in clusters.items() if v}
            if len(nonempty) >= 2 and max(len(v) for v in nonempty.values()) < len(paths):
                for cluster, cpaths in sorted(nonempty.items(), key=lambda kv: -len(kv[1])):
                    cpaths = natsorted(cpaths)
                    groups.append(
                        LoadGroup(
                            id=f"image:{suf}:{cluster}",
                            label=f"{suf} — {_cluster_label(cluster)} ({len(cpaths)})",
                            kind="image",
                            count=len(cpaths),
                            paths=cpaths,
                            suffix=suf,
                            naming_cluster=cluster,
                        )
                    )
                continue
        groups.append(
            LoadGroup(
                id=f"image:{suf}",
                label=f"{suf} ({len(paths)})",
                kind="image",
                count=len(paths),
                paths=paths,
                suffix=suf,
            )
        )

    # HIKMICRO thermo offer (same JPEG paths, separate kind)
    if probe_hikmicro and jpeg_paths:
        sample = jpeg_paths[:PROBE_SAMPLE_N]
        hits = [p for p in sample if probe_hikmicro_jpeg(p)]
        if hits:
            # Prefer full set of non-vis JPEGs that probe OK (spot-check + filter)
            from .converters.hikmicro import filter_hikmicro_jpegs

            thermo_paths = filter_hikmicro_jpegs(jpeg_paths)
            if thermo_paths:
                groups.append(
                    LoadGroup(
                        id="hikmicro_celsius",
                        label=f"HIKMICRO °C ({len(thermo_paths)})",
                        kind="hikmicro_celsius",
                        count=len(thermo_paths),
                        paths=thermo_paths,
                        suffix=".jpg",
                        extra={"approx": True},
                    )
                )

    for suf in sorted(s for s in by_suffix if s in VIDEO_EXTENSIONS):
        paths = natsorted(by_suffix[suf])
        groups.append(
            LoadGroup(
                id=f"video:{suf}",
                label=f"{suf} ({len(paths)})",
                kind="video",
                count=len(paths),
                paths=paths,
                suffix=suf,
            )
        )

    for suf in sorted(s for s in by_suffix if s in ARRAY_EXTENSIONS):
        paths = natsorted(by_suffix[suf])
        groups.append(
            LoadGroup(
                id=f"array:{suf}",
                label=f"{suf} ({len(paths)})",
                kind="array",
                count=len(paths),
                paths=paths,
                suffix=suf,
            )
        )

    ascii_paths: list[Path] = []
    for suf in ASCII_EXTENSIONS:
        for p in by_suffix.get(suf, []):
            if suf in (".asc", ".dat"):
                ascii_paths.append(p)
            elif probe_ascii_txt and probe_ascii_matrix(p):
                ascii_paths.append(p)
            else:
                other_paths.append(p)
    if ascii_paths:
        # Prefer dominant ascii suffix for label
        ascii_paths = natsorted(ascii_paths)
        suf_counts = Counter(p.suffix.lower() for p in ascii_paths)
        top_suf = suf_counts.most_common(1)[0][0]
        groups.append(
            LoadGroup(
                id=f"ascii:{top_suf}",
                label=f"ASCII {top_suf} ({len(ascii_paths)})",
                kind="ascii",
                count=len(ascii_paths),
                paths=ascii_paths,
                suffix=top_suf,
            )
        )

    known = set(IMAGE_EXTENSIONS) | set(VIDEO_EXTENSIONS) | set(ARRAY_EXTENSIONS) | set(ASCII_EXTENSIONS)
    for suf, paths in by_suffix.items():
        if suf not in known:
            other_paths.extend(paths)
    if other_paths:
        other_paths = natsorted(other_paths)
        groups.append(
            LoadGroup(
                id="other",
                label=f"Other ({len(other_paths)})",
                kind="other",
                count=len(other_paths),
                paths=other_paths,
            )
        )

    # Loadable groups first (exclude pure "other" from auto-single unless alone)
    return groups


def loadable_groups(groups: list[LoadGroup]) -> list[LoadGroup]:
    return [g for g in groups if g.kind != "other"]


def should_show_chooser(groups: list[LoadGroup]) -> bool:
    """Show chooser if multiple loadable groups or a converter offer is present."""
    loadable = loadable_groups(groups)
    if not loadable:
        return False
    if any(g.kind == "hikmicro_celsius" for g in loadable):
        # Always ask: visual JPEG vs thermo when both present
        kinds = {g.kind for g in loadable}
        if "image" in kinds or len(loadable) > 1:
            return True
    if len(loadable) > 1:
        return True
    return False

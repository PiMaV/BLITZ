"""
Boot-bench: One-time optimization on first start.

Parameters: pixels (mp), color depth (bits), count (n). Aspect ratio irrelevant.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path

# Headless Qt for settings
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

multiprocessing.freeze_support()


def _emit(msg: str) -> None:
    """Progress to parent (stdout, prefixed for parsing)."""
    print(f"BLITZ_BENCH: {msg}", flush=True)


# Parameters: pixels (mp), bits, count. 5 steps per parameter.
# Full range: 0.3 MP to 5 MP, 8/16 bit. Aspect ratio 4:3.
MP_STEPS = [0.3, 0.6, 1.2, 2.5, 5]
BITS_STEPS = [8, 16]
ALL_CONFIGS = [{"mp": mp, "bits": b} for mp in MP_STEPS for b in BITS_STEPS]

# n_max per config to cap total data. Count steps.
MAX_RAW_GB = 4
COUNT_STEPS = 5


def _configs_for_intensity(intensity: float) -> list[dict]:
    """intensity 0=1 config, 0.5=optimum, 1=all. Linear interpolation."""
    n = max(1, min(len(ALL_CONFIGS), int(1 + intensity * (len(ALL_CONFIGS) - 1))))
    step = len(ALL_CONFIGS) / n if n > 0 else 1
    return [ALL_CONFIGS[int(i * step)] for i in range(n)]


def _count_steps_for_intensity(intensity: float) -> int:
    """3 (quick) to 5 (full)."""
    return max(3, min(5, int(3 + intensity * 2)))


def _max_raw_gb_for_intensity(intensity: float) -> float:
    """0.5 GB (quick) to 4 GB (full)."""
    return 0.5 + intensity * 3.5


def _mp_to_wh(mp: float) -> tuple[int, int]:
    """Megapixels -> (w, h) at 4:3."""
    total = mp * 1e6
    h = int(math.sqrt(total * 3 / 4))
    w = int(total / h)
    return max(64, w), max(64, h)


def _run_one_config(
    cfg: dict,
    path: Path,
    n_max: int,
    counts: list[int],
    _force,
) -> tuple[int | None, int | None, list[dict]]:
    """Run seq/par sweep for one config. Return (best_n, best_size_bytes, metrics)."""
    import time as _t

    from blitz.data.load import DataLoader

    w, h = _mp_to_wh(cfg["mp"])
    bits = cfg["bits"]
    best_n: int | None = None
    best_size: int | None = None
    metrics: list[dict] = []

    bytes_per_pixel = 2 if bits == 16 else 1
    raw_per_image = w * h * bytes_per_pixel

    for n in counts:
        if n > n_max:
            continue
        subset = n / n_max
        _emit(f"Testing {n} files (seq)...")
        _force(False)
        t0 = _t.perf_counter()
        DataLoader(subset_ratio=subset, max_ram=32.0).load(path)
        t_seq = _t.perf_counter() - t0
        _emit(f"Testing {n} files (par)...")
        _force(True)
        t0 = _t.perf_counter()
        DataLoader(subset_ratio=subset, max_ram=32.0).load(path)
        t_par = _t.perf_counter() - t0
        winner = "par" if t_par < t_seq else "seq"
        size_bytes = n * raw_per_image
        config_label = f"{cfg['mp']}MP_{bits}b"
        metrics.append({
            "config": config_label,
            "n": n,
            "size_mb": round(size_bytes / (1024**2), 2),
            "t_seq": round(t_seq, 2),
            "t_par": round(t_par, 2),
            "winner": winner,
        })
        _emit(f"METRIC {n} {t_seq:.2f} {t_par:.2f} {winner}")

        if t_par < t_seq:
            best_n = n
            best_size = size_bytes
            break

    return best_n, best_size, metrics


def _run_bench(intensity: float = 0.5) -> tuple[int, int, list[dict]]:
    """Run configs, aggregate. intensity 0=quick, 0.5=optimum, 1=ludicrous."""
    import cv2
    import numpy as np

    from blitz import settings
    from blitz.data.load import DataLoader

    configs = _configs_for_intensity(intensity)
    count_steps = _count_steps_for_intensity(intensity)
    max_raw_gb = _max_raw_gb_for_intensity(intensity)

    def _force(parallel: bool) -> None:
        if parallel:
            settings.set("default/multicore_files_threshold", 0)
            settings.set("default/multicore_size_threshold", 0)
        else:
            settings.set("default/multicore_files_threshold", 999_999)
            settings.set("default/multicore_size_threshold", 10**15)

    all_metrics: list[dict] = []
    best_n_min: int | None = None
    best_size_min: int | None = None

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        for idx, cfg in enumerate(configs):
            w, h = _mp_to_wh(cfg["mp"])
            bits = cfg["bits"]
            bytes_per_pixel = 2 if bits == 16 else 1
            raw_per_image = w * h * bytes_per_pixel
            n_max = max(200, min(8000, int(max_raw_gb * (1024**3) / raw_per_image)))
            counts = [max(50, int(n_max * (i + 1) / count_steps)) for i in range(count_steps)]
            counts = sorted(set(counts))

            label = f"{cfg['mp']}MP_{bits}b"
            _emit(f"Config {idx+1}/{len(configs)}: {label}")

            path = base / label
            path.mkdir()

            _emit(f"Generating {n_max} images...")
            dtype = np.uint16 if bits == 16 else np.uint8
            np.random.seed(42)

            for i in range(n_max):
                img = np.zeros((h, w), dtype=dtype)
                cv2.rectangle(img, (w//8, h//8), (7*w//8, 7*h//8), 180 if bits == 8 else 46080, -1)
                noise = np.random.randint(-15, 15, (h, w), dtype=np.int32)
                img = np.clip(img.astype(np.int32) + noise, 0, 255 if bits == 8 else 65535).astype(dtype)
                cv2.imwrite(str(path / f"f{i:05d}.png"), img)

            best_n, best_size, metrics = _run_one_config(
                cfg, path, n_max, counts, _force
            )
            all_metrics.extend(metrics)

            if best_n is not None:
                if best_n_min is None or best_n < best_n_min:
                    best_n_min = best_n
                if best_size_min is None or (best_size is not None and best_size < best_size_min):
                    best_size_min = best_size

    files_thresh = best_n_min if best_n_min is not None else 500
    size_bytes = best_size_min if best_size_min is not None else int(1.4 * (2**30))
    return files_thresh, size_bytes, all_metrics


def run() -> bool:
    """Run bench, set thresholds, boot_bench_done=True."""
    from PyQt6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication(sys.argv)

    from blitz import settings

    intensity = 0.5
    if len(sys.argv) >= 2:
        try:
            intensity = float(sys.argv[1])
            intensity = max(0.0, min(1.0, intensity))
        except ValueError:
            pass

    try:
        t0 = time.perf_counter()
        files_thresh, size_bytes, all_metrics = _run_bench(intensity)
        duration = time.perf_counter() - t0

        settings.set("default/multicore_files_threshold", files_thresh)
        settings.set("default/multicore_size_threshold", size_bytes)

        out_path = Path.cwd() / "boot_bench_results.json"
        try:
            with open(out_path, "w") as f:
                json.dump({
                    "cores": multiprocessing.cpu_count(),
                    "duration_s": round(duration, 2),
                    "recommended_files": files_thresh,
                    "size_threshold_bytes": size_bytes,
                    "size_threshold_gb": round(size_bytes / (1024**3), 2),
                    "configs": [f"{c['mp']}MP_{c['bits']}b" for c in _configs_for_intensity(intensity)],
                    "metrics": all_metrics,
                }, f, indent=2)
        except OSError:
            pass

        _emit("Done.")
    except (PermissionError, OSError, Exception):
        pass
    finally:
        settings.set("default/boot_bench_done", True)
    return True


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    ok = run()
    sys.exit(0 if ok else 1)

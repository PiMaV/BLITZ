"""
Boot-bench: Multicore threshold optimization (archived).

Run manually: python scripts/boot_bench.py [config.json]
Output: boot_bench_results.json, thresholds in settings.
Moved from blitz.boot_bench; keep for optional re-use.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Add project root for blitz imports
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Headless Qt for settings
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

multiprocessing.freeze_support()


def _emit(msg: str) -> None:
    print(f"BLITZ_BENCH: {msg}", flush=True)


MP_STEPS = [0.3, 5]
BITS_STEPS = [8, 16]
ALL_CONFIGS = [{"mp": mp, "bits": b} for mp in MP_STEPS for b in BITS_STEPS]
N_VALUES = [50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
MAX_IMAGES_PER_CONFIG = 2500
DEFAULT_REPETITIONS = 3


def _label_to_config(label: str) -> dict | None:
    m = re.match(r"^([\d.]+)MP_(\d+)b$", label)
    if m:
        return {"mp": float(m.group(1)), "bits": int(m.group(2))}
    return None


def _config_from_label(label: str) -> dict | None:
    c = _label_to_config(label)
    if c:
        return c
    for c in ALL_CONFIGS:
        if f"{c['mp']}MP_{c['bits']}b" == label:
            return c
    return None


def _generate_one_image(args: tuple) -> None:
    import cv2
    import numpy as np
    path, w, h, dtype_key, i = args
    dtype = np.uint16 if dtype_key == 16 else np.uint8
    np.random.seed(42 + i)
    img = np.zeros((h, w), dtype=dtype)
    cv2.rectangle(img, (w//8, h//8), (7*w//8, 7*h//8), 180 if dtype_key == 8 else 46080, -1)
    noise = np.random.randint(-15, 15, (h, w), dtype=np.int32)
    img = np.clip(img.astype(np.int32) + noise, 0, 255 if dtype_key == 8 else 65535).astype(dtype)
    cv2.imwrite(str(path / f"f{i:05d}.png"), img)


def _mp_to_wh(mp: float) -> tuple[int, int]:
    total = mp * 1e6
    h = int(math.sqrt(total * 3 / 4))
    w = int(total / h)
    return max(64, w), max(64, h)


def _run_one_config(cfg, path, n_max, max_ram, repetitions, label, _force):
    import time as _t
    from blitz.data.load import DataLoader
    w, h = _mp_to_wh(cfg["mp"])
    bits = cfg["bits"]
    best_n, best_size = None, None
    metrics = []
    config_label = label or f"{cfg['mp']}MP_{bits}b"
    bytes_per_pixel = 2 if bits == 16 else 1
    raw_per_image = w * h * bytes_per_pixel
    n_list = [n for n in N_VALUES if 50 <= n <= n_max] or [min(n_max, 500)]
    for n in n_list:
        subset = n / n_max
        t_seq_trials, t_par_trials = [], []
        for r in range(repetitions):
            _emit(f"Testing {n} files (seq) {r+1}/{repetitions}...")
            _force(False)
            t0 = _t.perf_counter()
            DataLoader(subset_ratio=subset, max_ram=max_ram).load(path)
            t_seq_trials.append(_t.perf_counter() - t0)
            _emit(f"Testing {n} files (par) {r+1}/{repetitions}...")
            _force(True)
            t0 = _t.perf_counter()
            DataLoader(subset_ratio=subset, max_ram=max_ram).load(path)
            t_par_trials.append(_t.perf_counter() - t0)
        t_seq_trials.sort()
        t_par_trials.sort()
        t_seq = t_seq_trials[len(t_seq_trials) // 2]
        t_par = t_par_trials[len(t_par_trials) // 2]
        par_wins = t_par < t_seq
        winner = "par" if par_wins else "seq"
        size_bytes = n * raw_per_image
        metrics.append({"config": config_label, "n": n, "size_mb": round(size_bytes/(1024**2), 2),
                        "t_seq": round(t_seq, 2), "t_par": round(t_par, 2), "winner": winner})
        _emit(f"METRIC {config_label} {n} {t_seq:.2f} {t_par:.2f} {winner}")
        if par_wins and best_n is None:
            best_n, best_size = n, size_bytes
    return best_n, best_size, metrics


def _run_bench(params: dict):
    from blitz import settings
    config_labels = params.get("configs", [f"{c['mp']}MP_{c['bits']}b" for c in ALL_CONFIGS])
    configs = []
    for lbl in config_labels:
        c = _label_to_config(lbl)
        configs.append((c, lbl)) if c else None
        if not c:
            c = _config_from_label(lbl)
            if c:
                configs.append((c, lbl))
    if not configs:
        configs = [(c, f"{c['mp']}MP_{c['bits']}b") for c in ALL_CONFIGS]
    max_raw_gb = float(params.get("max_gb", 2.0))
    max_ram = float(params.get("max_ram", 32.0))
    repetitions = max(1, int(params.get("repetitions", DEFAULT_REPETITIONS)))

    def _force(parallel: bool):
        if parallel:
            settings.set("default/multicore_files_threshold", 0)
            settings.set("default/multicore_size_threshold", 0)
        else:
            settings.set("default/multicore_files_threshold", 999_999)
            settings.set("default/multicore_size_threshold", 10**15)

    all_metrics = []
    thresholds_per_config = {}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        from blitz._cpu import physical_cpu_count
        n_workers = physical_cpu_count()
        _emit(f"Phase 1: Generating images (parallel, {n_workers} workers)...")
        for cfg, lbl in configs:
            w, h = _mp_to_wh(cfg["mp"])
            bits = cfg["bits"]
            bp = 2 if bits == 16 else 1
            rpi = w * h * bp
            n_max = max(200, min(MAX_IMAGES_PER_CONFIG, int(max_raw_gb * (1024**3) / rpi)))
            _emit(f"  {lbl}: {n_max} images")
            (base / lbl).mkdir()
            dtype_key = 16 if bits == 16 else 8
            nw = min(physical_cpu_count(), 8, n_max)
            with ThreadPoolExecutor(max_workers=nw) as ex:
                list(ex.map(_generate_one_image, [(base/lbl, w, h, dtype_key, i) for i in range(n_max)]))
        _emit("Phase 2: Running bench...")
        for cfg, lbl in configs:
            w, h = _mp_to_wh(cfg["mp"])
            bits = cfg["bits"]
            bp = 2 if bits == 16 else 1
            rpi = w * h * bp
            n_max = max(200, min(MAX_IMAGES_PER_CONFIG, int(max_raw_gb * (1024**3) / rpi)))
            path_cfg = base / lbl
            _emit(f"Config: {lbl}")
            best_n, best_size, metrics = _run_one_config(cfg, path_cfg, n_max, max_ram, repetitions, lbl, _force)
            all_metrics.extend(metrics)
            if best_n and best_size:
                thresholds_per_config[lbl] = {"files": best_n, "size_bytes": best_size}
    fallback_files = max((t["files"] for t in thresholds_per_config.values()), default=500)
    fallback_size = max((t["size_bytes"] for t in thresholds_per_config.values()), default=int(1.4*2**30))
    return thresholds_per_config, fallback_files, fallback_size, all_metrics


def run() -> bool:
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication(sys.argv)
    from blitz import settings
    params = {"configs": [f"{c['mp']}MP_{c['bits']}b" for c in ALL_CONFIGS], "max_gb": 2.0, "max_ram": 32.0}
    if len(sys.argv) >= 2 and Path(sys.argv[1]).exists():
        try:
            with open(sys.argv[1]) as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    try:
        t0 = time.perf_counter()
        tc, ff, fs, am = _run_bench(params)
        settings.set("default/multicore_files_threshold", ff)
        settings.set("default/multicore_size_threshold", fs)
        from blitz._cpu import physical_cpu_count
        payload = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "cores_physical": physical_cpu_count(), "duration_s": round(time.perf_counter()-t0, 2),
                   "thresholds_per_config": tc, "metrics": am}
        (Path.cwd() / "boot_bench_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _emit("Done.")
    except Exception:
        pass
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

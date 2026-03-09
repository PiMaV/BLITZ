"""
Benchmark Result-Cache: cached vs recompute.

Verwendung:
  python scripts/bench_result_cache.py [n_frames] [height] [width]

Beispiel mit 4000 Bildern:
  python scripts/bench_result_cache.py 4000 64 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blitz.data.image import ImageData, MetaData


def _make_meta(n: int) -> list[MetaData]:
    return [
        MetaData(
            file_name=f"frame_{i}.png",
            file_size_MB=0.1,
            size=(4, 4),
            dtype=np.float32,
            bit_depth=32,
            color_model="grayscale",
        )
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark result cache")
    ap.add_argument("n", type=int, nargs="?", default=1000, help="Number of frames")
    ap.add_argument("h", type=int, nargs="?", default=64, help="Height")
    ap.add_argument("w", type=int, nargs="?", default=64, help="Width")
    args = ap.parse_args()

    n, h, w = args.n, args.h, args.w
    data = np.random.rand(n, h, w, 1).astype(np.float32)
    meta = _make_meta(n)
    img = ImageData(data, meta)

    raw_mb = data.nbytes / (1024**2)
    print(f"Data: {n} frames x {h}x{w}, {raw_mb:.1f} MB raw")

    # First compute (cache miss)
    img.reduce("MEAN", bounds=(0, n - 1))
    t0 = time.perf_counter()
    _ = img.image
    t_first = time.perf_counter() - t0
    print(f"First access (compute): {t_first*1000:.1f} ms")

    # Cached access
    t0 = time.perf_counter()
    for _ in range(20):
        _ = img.image
    t_cached = (time.perf_counter() - t0) / 20
    print(f"Cached access (x20 avg): {t_cached*1000:.2f} ms")

    # Recompute after invalidate
    img._invalidate_result()
    t0 = time.perf_counter()
    _ = img.image
    t_recompute = time.perf_counter() - t0
    print(f"Recompute after invalidate: {t_recompute*1000:.1f} ms")

    speedup = t_recompute / t_cached if t_cached > 0 else 0
    print(f"Speedup (cached vs recompute): {speedup:.1f}x")


if __name__ == "__main__":
    main()

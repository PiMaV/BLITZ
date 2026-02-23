"""
Benchmark: Multicore vs Sequential Loading (Images, ASCII).

Vergleicht Ladezeiten fuer unterschiedliche Dateianzahlen.
Hinweis: Unter Windows benoetigt multiprocessing.Pool oft volle Rechte (kein Sandbox).

Usage:
  python scripts/benchmark_multicore_load.py <folder_path> [options]
  python scripts/benchmark_multicore_load.py --generate 500  # erzeugt temp Testdaten

Beispiele:
  python scripts/benchmark_multicore_load.py D:/Bilder/mein_ordner
  python scripts/benchmark_multicore_load.py D:/data/ascii_files --ascii
  python scripts/benchmark_multicore_load.py --generate 500 --n-runs 3
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

multiprocessing.freeze_support()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Headless Qt (Windows/Linux)
import os
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication
if QApplication.instance() is None:
    QApplication(sys.argv)

from blitz import settings
from blitz.data.converters import load_ascii
from blitz.data.load import DataLoader


def _generate_image_folder(n: int, out_dir: Path) -> Path:
    """Erzeuge n kleine PNGs in out_dir."""
    out_dir.mkdir(exist_ok=True, parents=True)
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (54, 54), 200, -1)
    for i in range(n):
        cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), img)
    return out_dir


def _generate_ascii_folder(n: int, out_dir: Path) -> Path:
    """Erzeuge n ASCII-Dateien in out_dir."""
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in range(n):
        data = np.random.rand(50, 80).astype(np.float32)
        (out_dir / f"file_{i:05d}.asc").write_text(
            "\n".join("\t".join(f"{v:.4f}" for v in row) for row in data)
        )
    return out_dir


def _force_thresholds(use_parallel: bool) -> None:
    """Schwellen setzen: parallel = immer Pool, sequential = nie Pool."""
    if use_parallel:
        settings.set("default/multicore_files_threshold", 0)
        settings.set("default/multicore_size_threshold", 0)
    else:
        settings.set("default/multicore_files_threshold", 999_999)
        settings.set("default/multicore_size_threshold", 10**15)


def _restore_thresholds() -> None:
    """Original-Schwellen wiederherstellen."""
    settings.set("default/multicore_files_threshold", 333)
    settings.set("default/multicore_size_threshold", int(1.3 * (2**30)))


def _bench_load(
    path: Path,
    is_ascii: bool,
    n_files: int,
    n_total: int,
    n_runs: int,
) -> tuple[float, float]:
    """Lade n_files aus path (total=n_total), n_runs pro Modus. Return (t_seq_avg, t_par_avg)."""
    subset = n_files / n_total if n_total > 0 else 1.0
    subset = min(1.0, subset)

    times_seq, times_par = [], []

    for _ in range(n_runs):
        _force_thresholds(use_parallel=False)
        t0 = time.perf_counter()
        if is_ascii:
            load_ascii(path, subset_ratio=subset)
        else:
            loader = DataLoader(subset_ratio=subset, max_ram=32.0)
            loader.load(path)
        times_seq.append(time.perf_counter() - t0)

    for _ in range(n_runs):
        _force_thresholds(use_parallel=True)
        t0 = time.perf_counter()
        if is_ascii:
            load_ascii(path, subset_ratio=subset)
        else:
            loader = DataLoader(subset_ratio=subset, max_ram=32.0)
            loader.load(path)
        times_par.append(time.perf_counter() - t0)

    return sum(times_seq) / n_runs, sum(times_par) / n_runs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark multicore vs sequential loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("path", type=Path, nargs="?", help="Folder with images or ASCII files")
    ap.add_argument("--ascii", action="store_true", help="Path contains ASCII (.asc/.dat)")
    ap.add_argument("--generate", type=int, metavar="N", help="Generate N temp files instead of path")
    ap.add_argument("--counts", type=str, default="100,200,333,500,1000",
                    help="Comma-separated file counts to test (default: 100,200,333,500,1000)")
    ap.add_argument("--n-runs", type=int, default=2, help="Runs per mode (default: 2)")
    ap.add_argument("--apply", action="store_true",
                    help="Empfohlenen Schwellwert in settings.blitz uebernehmen (statt zuruecksetzen)")
    args = ap.parse_args()

    try:
        if args.generate is not None:
            n = args.generate
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                if args.ascii:
                    path = _generate_ascii_folder(n, tmp / "ascii_bench")
                else:
                    path = _generate_image_folder(n, tmp / "img_bench")
                counts = [min(n, int(c.strip())) for c in args.counts.split(",") if c.strip()]
                counts = sorted(set(counts))
                print(f"Generated {n} {'ASCII' if args.ascii else 'image'} files")
                _run_bench(path, args.ascii, counts, args.n_runs, apply=args.apply)
            return

        if not args.path or not args.path.exists():
            print("Error: Provide path to folder or use --generate N")
            ap.print_help()
            sys.exit(1)

        path = args.path
        if path.is_file():
            path = path.parent
        n_total = len([f for f in path.iterdir() if f.is_file()])
        counts_str = [c.strip() for c in args.counts.split(",") if c.strip()]
        counts = sorted(set(min(n_total, int(c)) for c in counts_str if c.isdigit()))
        if not counts:
            counts = [min(n_total, 100), min(n_total, 333), min(n_total, 1000)]

        _run_bench(path, args.ascii, counts, args.n_runs, apply=args.apply)
    finally:
        if not getattr(args, "apply", False):
            _restore_thresholds()


def _run_bench(path: Path, is_ascii: bool, counts: list[int], n_runs: int, apply: bool = False) -> None:
    typ = "ASCII" if is_ascii else "Images"
    n_total = len([f for f in path.iterdir() if f.is_file()])
    print(f"\n--- Multicore Bench ({typ}): {path} ---")
    print(f"Total files: {n_total}, counts: {counts}, runs: {n_runs}")
    print("  Seq = sequential (ein Kern), Par = parallel (alle Kerne)")
    print(f"\n{'N':>6}  {'Seq (s)':>8}  {'Par (s)':>8}  {'Par/Seq':>10}")
    print("-" * 42)

    results: list[tuple[int, float, float, float]] = []
    for n in counts:
        if n > n_total:
            print(f"{n:>6}  (skip: only {n_total} files)")
            continue
        t_seq, t_par = _bench_load(path, is_ascii, n, n_total, n_runs)
        ratio = t_par / t_seq if t_seq > 0 else 0
        results.append((n, t_seq, t_par, ratio))
        mark = " faster" if ratio < 1 else " slower"
        print(f"{n:>6}  {t_seq:>8.2f}  {t_par:>8.2f}  {ratio:>7.2f}x{mark}")

    recommended = _print_recommendation(results)
    if apply and recommended is not None:
        settings.set("default/multicore_files_threshold", recommended)
        settings.set("default/multicore_size_threshold", int(1.3 * (2**30)))
        print(f"\n-> multicore_files_threshold auf {recommended} gesetzt (size_threshold auf Default).")
    elif not apply:
        print("\n(Hinweis: settings werden am Ende auf Default zurueckgesetzt)")


def _print_recommendation(results: list[tuple[int, float, float, float]]) -> int | None:
    """Empfehlung: kleinster N, ab dem Par schneller war. Return Schwellwert oder None."""
    par_wins = [(n, r) for n, _, _, r in results if r < 1.0]

    print("\n--- Deutung ---")
    if not results:
        print("Keine aussagekraeftigen Daten.")
        return None

    if par_wins:
        best_n = min(n for n, _ in par_wins)
        print(f"Parallel war ab {best_n} Dateien schneller (unterer Punkt).")
        print(f"-> Empfehlung: multicore_files_threshold = {best_n}")
        print(f"   (mit --apply direkt uebernehmen; kein Grund, hoher zu gehen)")
        return best_n
    print("Parallel war in allen Tests langsamer. Schwellwert bei 333 lassen oder erhoehen.")
    return None


if __name__ == "__main__":
    main()

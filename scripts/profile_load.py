"""
Profile BLITZ app (incl. Load). Stats ausgeben nach dem Beenden.

Verwendung:
  1. python scripts/profile_load.py
  2. App startet, Load durchfuehren (42k Dateien @ 0.01), warten bis UI reagiert
  3. App schliessen
  4. python scripts/profile_load.py --stats
  5. python scripts/profile_load.py --stats -o profile_stats.txt  (in Datei speichern)
"""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path
from typing import TextIO

PROF_FILE = Path(__file__).resolve().parent.parent / "profile_load.prof"
STATS_FILE = Path(__file__).resolve().parent.parent / "profile_stats.txt"


def run_app_profiled() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    cProfile.run("from blitz.app import run; run()", str(PROF_FILE))
    print(f"Profile gespeichert: {PROF_FILE}")
    print("Naechster Befehl:  python scripts/profile_load.py --stats")


def print_stats(n: int = 40, out: TextIO | Path | None = None) -> None:
    if not PROF_FILE.exists():
        print(f"Kein Profil gefunden: {PROF_FILE}")
        print("Zuerst:  python scripts/profile_load.py  (App starten, Load, schliessen)")
        sys.exit(1)
    p = pstats.Stats(str(PROF_FILE))
    p.sort_stats("cumtime")

    if isinstance(out, Path):
        f = open(out, "w", encoding="utf-8")
    elif out is not None:
        f = out
    else:
        f = sys.stdout

    def wr(*args: str) -> None:
        f.write("\n".join(args) + "\n")

    try:
        wr("--- Top {} nach cumtime ---".format(n), "")
        p.stream = f
        p.print_stats(n)
        wr("", "--- Top 15 blitz (ohne stdlib) ---", "")
        p.print_stats(15, "blitz")
        wr("", "--- draw_line / roiChanged callers ---", "")
        p.print_callers("draw_line", 10)
        p.print_callers("roiChanged", 10)
    finally:
        if f is not sys.stdout:
            f.close()
            print(f"Stats gespeichert: {f.name}")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        argv = [a for a in sys.argv if a not in ("--stats",)]
        out_path: Path = STATS_FILE
        if "-o" in argv:
            i = argv.index("-o")
            if i + 1 < len(argv):
                out_path = Path(argv[i + 1])
                argv = argv[:i] + argv[i + 2 :]
        n = int(argv[0]) if argv and argv[0].isdigit() else 40
        print_stats(n, out_path)
    else:
        run_app_profiled()

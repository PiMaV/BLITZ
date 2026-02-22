"""Shared bench data store for Bench tab and compact LUT panel. Mirrors CPU/RAM/Disk."""
from collections import deque

# ~60 points at 500 ms = ~30 s history. Shared by Bench tab and compact panel.
MAX_POINTS = 60


class BenchData:
    """Holds last N CPU, RAM, Disk samples. Updated by global timer."""

    def __init__(self, max_points: int = MAX_POINTS) -> None:
        self._cpu: deque[float] = deque(maxlen=max_points)
        self._ram_used: deque[float] = deque(maxlen=max_points)
        self._ram_free: deque[float] = deque(maxlen=max_points)
        self._disk_r: deque[float] = deque(maxlen=max_points)
        self._disk_w: deque[float] = deque(maxlen=max_points)

    def add(
        self,
        cpu_pct: float,
        ram_used_gb: float,
        ram_free_gb: float,
        disk_read_mbs: float,
        disk_write_mbs: float,
    ) -> None:
        self._cpu.append(cpu_pct)
        self._ram_used.append(ram_used_gb)
        self._ram_free.append(ram_free_gb)
        self._disk_r.append(disk_read_mbs)
        self._disk_w.append(disk_write_mbs)

    @property
    def ram_used(self) -> deque[float]:
        return self._ram_used

    @property
    def cpu(self) -> deque[float]:
        return self._cpu

    @property
    def ram_free(self) -> deque[float]:
        return self._ram_free

    @property
    def disk_r(self) -> deque[float]:
        return self._disk_r

    @property
    def disk_w(self) -> deque[float]:
        return self._disk_w

    @property
    def last_cpu(self) -> float:
        return self._cpu[-1] if self._cpu else 0.0

    @property
    def last_ram_free(self) -> float:
        return self._ram_free[-1] if self._ram_free else 0.0

    @property
    def last_disk_r(self) -> float:
        return self._disk_r[-1] if self._disk_r else 0.0

    @property
    def last_disk_w(self) -> float:
        return self._disk_w[-1] if self._disk_w else 0.0

    @property
    def last_ram_used(self) -> float:
        return self._ram_used[-1] if self._ram_used else 0.0

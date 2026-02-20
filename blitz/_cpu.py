"""CPU core count for parallel work. No blitz deps to avoid circular imports."""
import multiprocessing

try:
    import psutil
except ImportError:
    psutil = None


def physical_cpu_count() -> int:
    """Physical cores for parallel work (bench maxed out). Fallback to logical if unavailable."""
    if psutil is not None:
        try:
            n = psutil.cpu_count(logical=False)
            if n is not None and n > 0:
                return n
        except Exception:
            pass
    return multiprocessing.cpu_count()

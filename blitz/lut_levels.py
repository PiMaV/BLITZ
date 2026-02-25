"""LUT level calculation. Pure logic, numpy only, for easy unit testing."""
import numpy as np


def calculate_lut_levels(image: np.ndarray, percentile: float) -> tuple[float, float]:
    """Compute LUT min/max from percentile or min/max (percentile=0)."""
    with np.errstate(invalid="ignore", over="ignore"):
        if percentile <= 0:
            return float(np.nanmin(image)), float(np.nanmax(image))
        p_lo, p_hi = percentile, 100.0 - percentile
        mn, mx = np.nanpercentile(image, [p_lo, p_hi])
        return float(mn), float(mx)

"""COO volume for the sparse-sidecar bench. Not part of the BLITZ product."""

from __future__ import annotations

import numpy as np


def apply_floor_abs(array: np.ndarray, floor_abs: float | None) -> np.ndarray:
    """Set |v| < floor_abs to 0. No-op if floor_abs is None or <= 0."""
    if floor_abs is None or floor_abs <= 0:
        return array
    out = array if array.flags.writeable else np.array(array, copy=True)
    out[np.abs(out) < floor_abs] = 0
    return out


def as_thw(array: np.ndarray) -> np.ndarray:
    """Return grayscale (T, H, W). 3D is always a stack, not HxWx3."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        # Always (T, H, W). A single RGB still is 4D (1, H, W, 3) or HxWx3 via ndim 2+channel.
        return arr
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[-1] == 3:
            weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            return np.sum(arr.astype(np.float32) * weights, axis=-1)
        raise ValueError(f"Unsupported 4D shape {arr.shape}")
    raise ValueError(f"Unsupported array ndim {arr.ndim}")


class SparseCube:
    """COO (t, y, x) + values. Zeros are absent; reduce semantics are not here."""

    __slots__ = ("shape", "coords", "values", "_t_starts")

    def __init__(
        self,
        shape: tuple[int, int, int],
        coords: np.ndarray,
        values: np.ndarray,
    ) -> None:
        t, h, w = shape
        self.shape = (int(t), int(h), int(w))
        self.coords = np.asarray(coords, dtype=np.uint32)
        self.values = np.asarray(values)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must be (nnz, 3) t,y,x")
        if self.values.shape[0] != self.coords.shape[0]:
            raise ValueError("values length must match coords")
        if self.nnz == 0:
            self._t_starts = np.zeros(self.shape[0] + 1, dtype=np.int64)
        else:
            counts = np.bincount(self.coords[:, 0], minlength=self.shape[0])
            starts = np.empty(self.shape[0] + 1, dtype=np.int64)
            starts[0] = 0
            np.cumsum(counts, out=starts[1:])
            self._t_starts = starts

    @classmethod
    def from_dense(
        cls,
        array: np.ndarray,
        *,
        floor_abs: float | None = None,
    ) -> SparseCube:
        arr = as_thw(np.asarray(array))
        arr = apply_floor_abs(arr, floor_abs)
        nz = np.nonzero(arr)
        coords = np.stack(nz, axis=1).astype(np.uint32, copy=False)
        values = arr[nz]
        return cls(arr.shape, coords, values)

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    @property
    def occupancy(self) -> float:
        n = int(np.prod(self.shape))
        return 0.0 if n == 0 else self.nnz / n

    @property
    def nbytes(self) -> int:
        return int(self.coords.nbytes + self.values.nbytes)

    def dense_frame(self, t: int) -> np.ndarray:
        """Return a dense (H, W) frame. Empty if t is out of range."""
        _, h, w = self.shape
        out = np.zeros((h, w), dtype=self.values.dtype)
        if t < 0 or t >= self.shape[0] or self.nnz == 0:
            return out
        a = int(self._t_starts[t])
        b = int(self._t_starts[t + 1])
        if a == b:
            return out
        ys = self.coords[a:b, 1]
        xs = self.coords[a:b, 2]
        out[ys, xs] = self.values[a:b]
        return out

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.values.dtype)
        if self.nnz == 0:
            return out
        t, y, x = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        out[t, y, x] = self.values
        return out

    def reduce_max(self) -> np.ndarray:
        """MAX over time. Absent entries are 0, same as a dense cube of zeros."""
        _, h, w = self.shape
        out = np.zeros((h, w), dtype=self.values.dtype)
        if self.nnz == 0:
            return out
        np.maximum.at(out, (self.coords[:, 1], self.coords[:, 2]), self.values)
        return out

    def reduce_mean(self) -> np.ndarray:
        """MEAN over time, divide by T (zeros count), same as dense mean."""
        t, h, w = self.shape
        acc = np.zeros((h, w), dtype=np.float64)
        if self.nnz:
            np.add.at(
                acc,
                (self.coords[:, 1], self.coords[:, 2]),
                self.values.astype(np.float64, copy=False),
            )
        return (acc / t).astype(self.values.dtype, copy=False)

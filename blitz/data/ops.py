from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
import multiprocessing

import numpy as np


class _ReduceOperation(ABC):

    def __init__(self, cache: bool = False) -> None:
        self._cache = cache
        self._saved: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def clear(self) -> None:
        self._saved = None

    @abstractmethod
    def _reduce(self, x: np.ndarray) -> np.ndarray:
        ...

    def _run_threaded(self, x: np.ndarray) -> np.ndarray:
        # Simple heuristic: > 10MB uses threading.
        # This threshold balances overhead vs parallel gain.
        if x.ndim < 2 or x.nbytes < 10 * 1024 * 1024:
            return self._reduce(x)

        # Use CPU count but cap at 8 to avoid diminishing returns
        try:
            n_workers = min(multiprocessing.cpu_count(), 8)
        except NotImplementedError:
            n_workers = 4

        if n_workers < 2:
            return self._reduce(x)

        # Split along axis 1 (Height)
        h = x.shape[1]
        chunk_size = h // n_workers
        if chunk_size < 1:
            return self._reduce(x)

        chunks = []
        for i in range(n_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_workers - 1 else h
            # Create a view (slice) for each thread
            chunks.append(x[:, start:end, ...])

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Numpy releases GIL for heavy ops, allowing true parallelism
            results = list(executor.map(self._reduce, chunks))

        # Concatenate results along axis 1 (Height) to reconstruct full image
        return np.concatenate(results, axis=1)

    def reduce(self, x: np.ndarray) -> np.ndarray:
        if self._saved is not None:
            return self._saved

        result = self._run_threaded(x)

        if self._cache:
            # Only cache if result is small relative to input (< 1%) or small absolute (< 100MB)
            # This prevents caching full-sized results for single images or small datasets.
            ratio = result.nbytes / x.nbytes if x.nbytes > 0 else 0
            if ratio < 0.01 or result.nbytes < 100 * 1024 * 1024:
                self._saved = result

        return result

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.reduce(x)


class MEAN(_ReduceOperation):

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x.mean(axis=0), axis=0)


class STD(_ReduceOperation):

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x.std(axis=0), axis=0)


class MAX(_ReduceOperation):

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x.max(axis=0), axis=0)


class MIN(_ReduceOperation):

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x.min(axis=0), axis=0)


class MEDIAN(_ReduceOperation):

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.nanmedian(x, axis=0), axis=0)


class ReduceOperation(Enum):
    MEAN = auto()
    MEDIAN = auto()
    MAX = auto()
    MIN = auto()
    STD = auto()


def get(name: ReduceOperation | str) -> _ReduceOperation:
    if isinstance(name, str):
        try:
            name = getattr(ReduceOperation, name)
        except AttributeError:
            raise AttributeError(f"Unknown operation: {name!r}")
    try:
        return globals()[name.name](cache=True)  # type: ignore
    except KeyError:
        raise KeyError(f"Unknown operation: {name!r}")


class ReduceDict:

    def __init__(self) -> None:
        self._ops = {
            op.name: get(op) for op in ReduceOperation
        }

    def reduce(self, x: np.ndarray, op: ReduceOperation | str) -> np.ndarray:
        if isinstance(op, ReduceOperation):
            op = op.name
        return self._ops[op].reduce(x)

    def clear(self) -> None:
        for redop in self._ops.values():
            redop.clear()

from abc import ABC, abstractmethod
from enum import Enum, auto

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

    def reduce(self, x: np.ndarray) -> np.ndarray:
        if not self._cache:
            return self._reduce(x)
        if self._saved is not None:
            return self._saved
        self._saved = self._reduce(x)
        return self._saved

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

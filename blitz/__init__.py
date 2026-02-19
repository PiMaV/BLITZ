def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("BLITZ")
    except Exception:
        return "0.0.0"

__version__ = _get_version()

from . import data as data
from . import layout as layout

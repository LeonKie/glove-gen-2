"""glovegen: turn a positive scan into a printable, castable two-part mold."""

from .config import (
    DemoldConfig,
    KeyConfig,
    MoldConfig,
    PartingConfig,
    SpoutConfig,
    VentConfig,
)
from .frame import Frame
from .validate import SolidError

__all__ = [
    "MoldConfig",
    "DemoldConfig",
    "PartingConfig",
    "KeyConfig",
    "SpoutConfig",
    "VentConfig",
    "Frame",
    "SolidError",
    "run",
    "__version__",
]

__version__ = "2.0.0"


def run(*args, **kwargs):
    """Build a mold. See :func:`glovegen.pipeline.run`."""
    from .pipeline import run as _run

    return _run(*args, **kwargs)

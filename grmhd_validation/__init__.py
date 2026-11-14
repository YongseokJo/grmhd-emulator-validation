"""Common GRMHD validation infrastructure."""

__version__ = "0.1.0"


def __getattr__(name):
    # doctor/checksums must not import torch or scientific libraries at startup.
    if name in {"ModelAdapter", "Geometry"}:
        from .adapter import Geometry, ModelAdapter
        return {"ModelAdapter": ModelAdapter, "Geometry": Geometry}[name]
    raise AttributeError(name)

"""Model-facing physical tensor contract; no dataset paths reach an adapter."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import re
import subprocess
import sys

import torch

from .hashing import sha256


@dataclass(frozen=True)
class Geometry:
    r: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    gcov: torch.Tensor
    gcon: torch.Tensor
    gdet: torch.Tensor
    dx: torch.Tensor
    spin: torch.Tensor
    dt_dump: torch.Tensor
    resolved: torch.Tensor
    eos_gamma: torch.Tensor | None = None

    @property
    def shape(self):
        return tuple(self.r.shape)


class ModelAdapter(ABC):
    name = None
    version = None
    history_length = 1
    max_batch = 1
    requires_checkpoint = True

    @abstractmethod
    def __init__(self, checkpoint: str, geometry: Geometry, device: str, options: dict):
        """Restore model weights and its own preprocessing from the checkpoint."""

    @abstractmethod
    def step(self, history):
        """Physical torch.float32 [B,h,8,Nr,Ntheta,Nphi] -> [B,8,Nr,Ntheta,Nphi]."""

    def eval(self):
        # Adapters are ordinary Python objects; put any directly held PyTorch
        # modules into eval mode. Adapters may override for other frameworks.
        for value in vars(self).values():
            if isinstance(value, torch.nn.Module):
                value.eval()
        return self


def validate_declaration(adapter_class, protocol):
    if not inspect.isclass(adapter_class) or not issubclass(adapter_class, ModelAdapter):
        raise TypeError("Adapter must subclass grmhd_validation.ModelAdapter")
    if not isinstance(adapter_class.name, str) or not re.fullmatch(r"[a-z0-9-]+", adapter_class.name):
        raise ValueError("Adapter name must match [a-z0-9-]+")
    if not isinstance(adapter_class.version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", adapter_class.version):
        raise ValueError("Adapter version must be a safe nonempty path component")
    if type(adapter_class.history_length) is not int or not 1 <= adapter_class.history_length <= protocol["H_max"]:
        raise ValueError("Adapter history_length is outside 1..H_max")
    if type(adapter_class.max_batch) is not int or adapter_class.max_batch < 1:
        raise ValueError("Adapter max_batch must be a positive integer")
    if type(adapter_class.requires_checkpoint) is not bool:
        raise ValueError("Adapter requires_checkpoint must be a boolean")


def validate_history(history, adapter, geometry):
    if not isinstance(history, torch.Tensor):
        raise TypeError("Adapter history must be a tensor")
    expected = (adapter.history_length, 8, *geometry.shape)
    if history.ndim != 6 or tuple(history.shape[1:]) != expected or not 1 <= history.shape[0] <= adapter.max_batch:
        raise ValueError("Incorrect adapter history shape/batch")
    if history.dtype != torch.float32 or not torch.isfinite(history).all():
        raise ValueError("Adapter history must be finite torch.float32 physical values")


def validate_output(output, history, geometry):
    if not isinstance(output, torch.Tensor):
        raise TypeError("Adapter output must be a tensor")
    if tuple(output.shape) != (history.shape[0], 8, *geometry.shape):
        raise ValueError("Incorrect adapter output shape")
    if output.dtype != torch.float32:
        raise TypeError("Adapter output must have dtype torch.float32")
    if output.device != history.device:
        raise ValueError("Adapter output is on a different device from its history")
    if not torch.isfinite(output).all():
        raise ValueError("Adapter returned non-finite values")
    return output


def adapter_source(reference):
    """Load an absolute file reference or one installed adapter entry point."""
    if ":" in reference:
        filename, class_name = reference.rsplit(":", 1)
        path = Path(filename)
        if not path.is_absolute():
            raise ValueError("File adapter references require an absolute path")
        before = sha256(path)
        module_name = "_gval_adapter_" + before
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load adapter source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        if str(path.parent) not in sys.path:
            # Keep the model repository importable for lazy imports at load/step.
            sys.path.insert(0, str(path.parent))
        # Execute exactly the hashed source, avoiding timestamp-based stale pyc
        # reuse when an adapter is revised more than once within a second.
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        if sha256(path) != before:
            raise ValueError("Adapter source changed during import")
        adapter_class = getattr(module, class_name)
    else:
        points = importlib.metadata.entry_points().select(group="grmhd_validation.adapters", name=reference)
        if len(points) != 1:
            raise ValueError("Expected exactly one installed adapter entry point: " + reference)
        adapter_class = next(iter(points)).load()
        filename = inspect.getsourcefile(adapter_class)
        if filename is None:
            raise ValueError("Cannot fingerprint adapter source")
        path = Path(filename)
        before = sha256(path)
    result = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True, capture_output=True)
    return adapter_class, {"source_sha256": before, "repo_commit": result.stdout.strip() if result.returncode == 0 else None}

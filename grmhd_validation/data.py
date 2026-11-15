"""Checksum-verified physical data restricted to one model-facing partition."""
from pathlib import Path

import numpy as np
import torch

from .adapter import Geometry
from .checksums import read_checksums, verify_file


def load_geometry(split, device, hashes=None):
    if hashes is not None:
        verify_file(split.geometry_path, split.geometry_file, hashes)
    # Geometry always has its independent identity in the frozen split.
    from .hashing import sha256
    if sha256(split.geometry_path) != split.geometry_sha256:
        raise ValueError("Geometry does not match the frozen split")
    with np.load(split.geometry_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("r", "h", "ph", "gcov", "gcon", "gdet", "dx", "spin", "dt_dump")}
        values["resolved"] = archive["resolved"] if "resolved" in archive else np.ones(split.shape[1:], dtype=bool)
    for name in ("r", "h", "ph", "resolved"):
        if tuple(values[name].shape) != split.shape[1:]:
            raise ValueError("Geometry shape mismatch: " + name)
    tensors = {name: torch.from_numpy(np.array(value, copy=True)).to(device) for name, value in values.items()}
    if not all(torch.isfinite(value).all() for name, value in tensors.items() if name != "resolved"):
        raise ValueError("Geometry contains non-finite values")
    return Geometry(tensors["r"], tensors["h"], tensors["ph"], tensors["gcov"], tensors["gcon"],
                    tensors["gdet"], tensors["dx"], tensors["spin"], tensors["dt_dump"], tensors["resolved"])


def _identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class FrameReader:
    def __init__(self, split, partition, manifest=None, verify=True):
        if partition not in {"validation", "test"}:
            raise ValueError("Evaluation reads validation or test chunks only")
        self.split, self.partition = split, partition
        self.allowed = {dump.index: dump for dump in split.model_dumps(partition)}
        self.hashes = read_checksums(manifest) if verify else None
        self.verified = {}

    def numpy(self, index):
        dump = self.allowed.get(index)
        if dump is None:
            raise ValueError("Dump is outside the evaluation chunks; no model access: " + str(index))
        path = self.split.dump_path(dump)
        before = _identity(path)
        if self.hashes is not None and self.verified.get(index) != before:
            verify_file(path, dump.file, self.hashes)
            self.verified[index] = before
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.dtype != np.float32 or tuple(raw.shape) != self.split.shape:
            raise ValueError("Dump shape/dtype differs from split contract")
        result = np.array(raw, copy=True, order="C")
        if _identity(path) != before:
            raise ValueError("Data changed while being read")
        if not np.isfinite(result).all():
            raise ValueError("Truth contains non-finite values")
        return result

    def tensor(self, index, device):
        return torch.from_numpy(self.numpy(index)).to(device)

    def history(self, start, length, device):
        ids = list(range(start - length + 1, start + 1))
        if any(i not in self.allowed for i in ids) or len({self.allowed[i].chunk for i in ids}) != 1:
            raise ValueError("History crosses a split/chunk boundary")
        return torch.stack([self.tensor(i, device) for i in ids]).unsqueeze(0)

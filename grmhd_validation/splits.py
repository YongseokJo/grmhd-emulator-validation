"""Validated schema-v2 split files. Never discover dumps by listing directories."""
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SPLIT_NAMES = ("train", "validation", "test", "excluded")
CHANNELS = ("rho", "ug", "utilde1", "utilde2", "utilde3", "B1", "B2", "B3")


def relative_path(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\\\n\r\0"):
        raise ValueError("Invalid relative data path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError("Data paths must be canonical, relative, and stay inside data_root")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError("Invalid identifier: " + repr(value))
    return value


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


@dataclass(frozen=True)
class Dump:
    index: int
    file: str
    split: str
    chunk: str


@dataclass(frozen=True)
class Chunk:
    name: str
    split: str
    first: int
    last: int
    dumps: tuple


@dataclass(frozen=True)
class SplitFile:
    path: Path
    sha256: str
    split_id: str
    dataset: str
    data_root: Path
    geometry_file: str
    geometry_sha256: str
    times_file: object
    shape: tuple
    dt_dump: float
    spin: float
    segments: tuple

    @property
    def geometry_path(self):
        return self.data_root / self.geometry_file

    def chunks(self, split):
        if split not in SPLIT_NAMES:
            raise ValueError("Unknown split: " + split)
        return tuple(chunk for chunk in self.segments if chunk.split == split)

    def dumps(self, splits):
        """Return explicitly selected splits in whole-run order, never bridge chunks."""
        names = set(splits)
        if names - set(SPLIT_NAMES):
            raise ValueError("Unknown split selection")
        return tuple(dump for chunk in self.segments if chunk.split in names for dump in chunk.dumps)

    def dump_path(self, dump):
        if not isinstance(dump, Dump) or not any(dump in c.dumps for c in self.segments):
            raise ValueError("Dump does not belong to this split file")
        # Resolving symlinks is a containment check, not directory discovery.
        path = (self.data_root / dump.file).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Dump symlink escapes data_root: " + dump.file)
        return path


def read_split(path, data_root=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["schema_version"] != 2:
        raise ValueError("Expected frozen split schema_version 2")
    split_id = identifier(data["split_id"])
    root = Path(data_root if data_root is not None else data["data_root"]).expanduser().resolve()
    geometry = relative_path(data["geometry_file"])
    geometry_hash = data["geometry_sha256"]
    if not isinstance(geometry_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", geometry_hash):
        raise ValueError("Invalid geometry SHA256")
    times = data["times_file"]
    if times is not None:
        relative_path(times)
    array = data["array"]
    shape = tuple(array["shape"])
    if len(shape) != 4 or shape[0] != 8 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Expected dump shape [8,Nr,Ntheta,Nphi]")
    if array["dtype"] != "float32" or tuple(array["channels"]) != CHANNELS:
        raise ValueError("Dump dtype/channels differ from the physical data contract")
    if array["axes"] != ["channel", "r", "theta", "phi"]:
        raise ValueError("Invalid dump axes")
    dt, spin = float(data["dt_dump"]), float(data["spin"])
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(spin) or abs(spin) > 1:
        raise ValueError("Invalid dt_dump or spin")
    if set(data["splits"]) != set(SPLIT_NAMES):
        raise ValueError("Expected train, validation, test, and excluded splits")
    chunks = {}
    files_seen, indices_seen = set(), set()
    for name in SPLIT_NAMES:
        section = data["splits"][name]
        count = integer(section["count"], "split count")
        if len(section["files"]) != count:
            raise ValueError("Split count differs from its files: " + name)
        by_index = {}
        for filename in section["files"]:
            relative_path(filename)
            match = re.fullmatch(r"dump_(\d+)\.npy", PurePosixPath(filename).name)
            if not match:
                raise ValueError("Unexpected dump filename: " + filename)
            index = int(match[1])
            if filename in files_seen or index in indices_seen:
                raise ValueError("Duplicate dump path or numeric ID")
            files_seen.add(filename); indices_seen.add(index)
            by_index[index] = filename
        covered = set()
        for item in section["chunks"]:
            chunk_name = identifier(item["chunk"])
            first = integer(item["first"], "chunk first")
            last = integer(item["last"], "chunk last")
            if last < first or item["count"] != last - first + 1 or chunk_name in chunks:
                raise ValueError("Invalid or duplicate chunk: " + chunk_name)
            ids = set(range(first, last + 1))
            if covered & ids or not ids <= by_index.keys():
                raise ValueError("Chunk overlaps or does not match its file IDs")
            covered.update(ids)
            dumps = tuple(Dump(i, by_index[i], name, chunk_name) for i in range(first, last + 1))
            chunks[chunk_name] = Chunk(chunk_name, name, first, last, dumps)
        if covered != by_index.keys():
            raise ValueError("Files are not covered exactly by chunks")
        if [int(re.search(r"dump_(\d+)\.npy$", p)[1]) for p in section["files"]] != sorted(by_index):
            raise ValueError("Split files are not chronological")
    segments = []
    for segment in data["segments"]:
        chunk = chunks.get(segment["chunk"])
        if chunk is None or any(segment[key] != value for key, value in {
            "split": chunk.split, "first": chunk.first, "last": chunk.last, "count": len(chunk.dumps)}.items()):
            raise ValueError("Segment differs from its split chunk")
        if segments and segment["first"] != segments[-1].last + 1:
            raise ValueError("Segments must cover the whole run chronologically")
        segments.append(chunk)
    if len(segments) != len(chunks) or len({c.name for c in segments}) != len(chunks):
        raise ValueError("Segments do not cover each chunk exactly once")
    if sum(len(c.dumps) for c in segments) != integer(data["total_dumps"], "total_dumps", 1):
        raise ValueError("total_dumps differs from segment coverage")
    return SplitFile(path, hashlib.sha256(raw).hexdigest(), split_id, str(data["dataset"]), root,
                     geometry, geometry_hash, times, shape, dt, spin, tuple(segments))

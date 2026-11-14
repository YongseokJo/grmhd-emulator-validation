"""Owner-generated, deterministic sha256sum lists and shared verification code."""
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .hashing import sha256
from .permissions import SharedStorage
from .slurm import require_compute
from .splits import SPLIT_NAMES, read_split, relative_path


def read_checksums(path):
    hashes = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Malformed sha256sum entry")
        digest, name = match.groups()
        relative_path(name)
        if name in hashes:
            raise ValueError("Duplicate checksum path: " + name)
        hashes[name] = digest
    if not hashes:
        raise ValueError("Empty checksum list")
    return hashes


def verify_file(path, relative, hashes):
    expected = hashes.get(relative)
    if expected is None:
        raise ValueError("Missing data hash: " + relative)
    if sha256(path) != expected:
        raise ValueError("Data SHA256 mismatch: " + relative)


def checksum_bytes(split, workers):
    """All four splits, chronological by dump ID, followed by geometry."""
    dumps = split.dumps(SPLIT_NAMES)
    paths = [split.dump_path(dump) for dump in dumps]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        lines = [f"{digest}  {dump.file}\n" for dump, digest in zip(dumps, pool.map(sha256, paths))]
    geometry = sha256(split.geometry_path)
    if geometry != split.geometry_sha256:
        raise ValueError("Geometry SHA256 differs from the frozen split file")
    lines.append(f"{geometry}  {split.geometry_file}\n")
    return "".join(lines).encode()


def generate(split_file, data_root=None, storage=None):
    require_compute()
    storage = SharedStorage() if storage is None else storage
    storage.require_owner()
    split = read_split(split_file, data_root)
    directory = storage.folder("checksums")
    output = directory / (split.split_id + ".sha256")
    content = checksum_bytes(split, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if output.exists():
        if output.is_symlink() or output.read_bytes() != content:
            raise FileExistsError("Existing checksum list differs; ask the owner to investigate data identity")
    else:
        # An exclusive temporary file is group-readable from creation. Publishing
        # via hard link is atomic and refuses to replace an existing manifest.
        temporary = "." + output.name + "." + uuid.uuid4().hex
        # Names are generated locally; new_file intentionally accepts no path separators.
        try:
            with storage.new_file(directory, temporary) as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.link(directory / temporary, output, follow_symlinks=False)
        finally:
            path = directory / temporary
            if path in storage.created:
                metadata = path.lstat()
                if storage.created[path] == (metadata.st_dev, metadata.st_ino):
                    path.unlink()
    return {"ok": True, "next": "gval doctor", "split_id": split.split_id,
            "split_sha256": split.sha256, "files": len(split.dumps(SPLIT_NAMES)) + 1,
            "output": str(output), "sha256": sha256(output)}

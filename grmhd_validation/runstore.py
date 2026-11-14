"""Globally serialized attempt reservation and atomic, no-replace publication."""
from contextlib import contextmanager
import ctypes
import fcntl
import getpass
import os
from pathlib import Path
import re


@contextmanager
def reservation_lock(storage):
    fd = os.open(storage.root / "staging", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def reserve(storage, model, version, checkpoint_sha256, partition, tier):
    user = getpass.getuser()
    storage.staging_user()
    token = checkpoint_sha256[:8] if checkpoint_sha256 else "nockpt"
    prefix = f"{model}-{version}__{token}__{partition}__{tier}__a"
    pattern = re.compile(re.escape(prefix) + r"\d+")
    with reservation_lock(storage):
        count = 0
        for area in ("runs", "staging"):
            for parent, dirs, _files in os.walk(storage.root / area, followlinks=False):
                count += sum(bool(pattern.fullmatch(name)) and not (Path(parent) / name).is_symlink() for name in dirs)
        attempt = count + 1
        run_id = prefix + str(attempt)
        root = storage.exclusive_folder("staging", user, run_id)
    return root, run_id, attempt


def reserve_conformance(storage, platform):
    user = getpass.getuser()
    parent = storage.folder("staging", user, "conformance")
    with reservation_lock(storage):
        number = 1 + sum(p.is_dir() and not p.is_symlink() and re.fullmatch(re.escape(platform) + r"__\d+", p.name) is not None for p in parent.iterdir())
        return storage.exclusive_folder("staging", user, "conformance", f"{platform}__{number}")


def no_replace_rename(source, destination):
    # Linux renameat2 is atomic on the same filesystem and cannot replace an
    # existing attempt, even an empty directory created by another process.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("Atomic no-replace rename is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    source, destination = Path(source), Path(destination)
    left = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    right = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if function(left, os.fsencode(source.name), right, os.fsencode(destination.name), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    finally:
        os.close(left); os.close(right)


def publish(storage, root, split_id, model):
    if not (root / "COMPLETE").is_file():
        raise ValueError("Only a COMPLETE run may be published")
    with reservation_lock(storage):
        parent = storage.destination(split_id, model)
        target = parent / root.name
        if target.exists() or target.is_symlink():
            raise FileExistsError("Run destination already exists")
        storage.make_readonly(root)
        no_replace_rename(root, target)
    return target

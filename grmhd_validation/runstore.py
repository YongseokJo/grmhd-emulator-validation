"""Globally serialized attempt reservation and atomic, no-replace publication."""
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import getpass
import os
from pathlib import Path
import re


class UnsealedRunError(RuntimeError):
    def __init__(self, target, error):
        self.target = target
        super().__init__(f"Run published but unsealed: {target}: {error}")


@contextmanager
def reservation_lock(storage):
    fd = os.open(storage.root / "staging", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def reserve(storage, model, version, checkpoint_sha256, split_id, partition, tier):
    user = getpass.getuser()
    storage.staging_user()
    token = checkpoint_sha256[:8] if checkpoint_sha256 else "nockpt"
    prefix = f"{model}-{version}__{token}__{split_id}__{partition}__{tier}__a"
    pattern = re.compile(re.escape(prefix) + r"\d+")
    with reservation_lock(storage):
        parents = [storage.root / "runs" / split_id / model]
        # Exactly one level per user, never descend into attempts/conformance.
        parents.extend(path for path in (storage.root / "staging").iterdir()
                       if path.is_dir() and not path.is_symlink())
        count = sum(bool(pattern.fullmatch(path.name)) and path.is_dir() and not path.is_symlink()
                    for parent in parents if parent.is_dir() for path in parent.iterdir())
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
    # Prefer kernel no-replace. Some shared filesystems reject its flag; all
    # gval writers also serialize on the destination parent for a POSIX fallback.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is not None:
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
    source, destination = Path(source), Path(destination)
    left = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    right = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(right, fcntl.LOCK_EX)
        try:
            os.stat(destination.name, dir_fd=right, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
        if function is None or function(left, os.fsencode(source.name), right, os.fsencode(destination.name), 1):
            error = ctypes.get_errno()
            if function is not None and error not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise OSError(error, os.strerror(error), str(destination))
            # POSIX rename is atomic. Parent locking plus this existence check
            # prevents replacement among the only permitted writers: gval.
            os.rename(source.name, destination.name, src_dir_fd=left, dst_dir_fd=right)
    finally:
        os.close(left); os.close(right)


def publish(storage, root, split_id, model):
    if (root/'PREVIEW').exists() or not (root / "COMPLETE").is_file():
        raise ValueError("Only a COMPLETE run may be published")
    with reservation_lock(storage):
        parent = storage.destination(split_id, model)
        target = parent / root.name
        if target.exists() or target.is_symlink():
            raise FileExistsError("Run destination already exists")
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            storage.make_readonly(root, include_root=False)
            no_replace_rename(root, target)
            try:
                storage.seal_created_root(root, fd)
            except Exception as error:
                # Do not rewrite metadata/checksums or thaw a published tree.
                # The owner can recover exactly this completed run with seal.
                raise UnsealedRunError(target, error) from error
        finally:
            os.close(fd)
    return target

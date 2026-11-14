"""Lazy shared writes. Existing directories and files are never repaired/chmodded."""
import getpass
import grp
import os
import stat
from pathlib import Path

from .splits import identifier, relative_path

SHARED_ROOT = Path("/work/nvme/bgun/share/validation")
OWNER_AREAS = ("splits", "checksums", "references", "reports", "movies")
GROUP_AREAS = ("runs", "staging", "training_records")


class SharedStorage:
    def __init__(self, root=SHARED_ROOT, group="delta_bgun"):
        self.root = Path(root).absolute()
        self.gid = grp.getgrnam(group).gr_gid
        self.created = {}

    def require_owner(self):
        if os.geteuid() != self.root.stat().st_uid:
            raise PermissionError("This command writes an owner-only area; ask the benchmark owner to run it")

    def _remember(self, path, fd):
        metadata = os.fstat(fd)
        if metadata.st_gid != self.gid:
            raise PermissionError("New entry did not inherit the required group: " + str(path))
        self.created[Path(path)] = (metadata.st_dev, metadata.st_ino)

    def folder(self, area, *parts):
        """Open only prescribed areas; create children with the area's fixed mode."""
        if area not in OWNER_AREAS + GROUP_AREAS:
            raise ValueError("Unknown shared area")
        mode = 0o2750 if area in OWNER_AREAS else 0o3770
        if area in OWNER_AREAS:
            self.require_owner()
        for part in parts:
            identifier(part)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        path = self.root
        try:
            for position, part in enumerate((area, *parts)):
                new = False
                if position:
                    try:
                        os.mkdir(part, mode, dir_fd=fd)
                        new = True
                    except FileExistsError:
                        pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                path = path / part
                os.close(fd); fd = child
                if new:
                    # Only the directory created by this call may be chmodded.
                    os.fchmod(fd, mode)
                    self._remember(path, fd)
                metadata = os.fstat(fd)
                if metadata.st_gid != self.gid or stat.S_IMODE(metadata.st_mode) != mode:
                    raise PermissionError("Existing shared directory has unexpected group/mode; doctor must report it: " + str(path))
            return path
        finally:
            os.close(fd)

    def staging_user(self):
        return self.folder("staging", getpass.getuser())

    def destination(self, split_id, model):
        return self.folder("runs", split_id, model)

    def new_file(self, parent, name):
        """Exclusive creation with a group-readable mode; returns a binary stream."""
        relative_path(name)
        if Path(name).name != name:
            raise ValueError("Expected a single filename")
        parent = Path(parent).absolute()
        if not parent.is_relative_to(self.root):
            raise ValueError("Shared writes must stay under the shared root")
        relative = parent.relative_to(self.root)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in relative.parts:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child
            handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640, dir_fd=fd)
            self._remember(parent / name, handle)
            return os.fdopen(handle, "wb")
        finally:
            os.close(fd)

    def make_readonly(self, tree):
        """Freeze only a wholly invocation-created tree, checking identity first."""
        tree = Path(tree).absolute()
        entries = [tree]
        for parent, dirs, files in os.walk(tree, followlinks=False):
            entries.extend(Path(parent) / name for name in dirs + files)
        for path in entries:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or self.created.get(path) != (metadata.st_dev, metadata.st_ino):
                raise PermissionError("Refusing to chmod an entry not created in this invocation: " + str(path))
        for path in reversed(entries):
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                metadata = os.fstat(fd)
                if self.created[path] != (metadata.st_dev, metadata.st_ino):
                    raise PermissionError("Entry changed before freeze")
                os.fchmod(fd, stat.S_IMODE(metadata.st_mode) & ~0o222)
            finally:
                os.close(fd)

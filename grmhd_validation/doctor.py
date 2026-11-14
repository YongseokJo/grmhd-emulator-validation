"""Read-only installation and shared-storage diagnostics."""
import grp
import errno
import os
import stat
import subprocess
import time
from pathlib import Path

from .checksums import read_checksums
from .environment import check_environment
from .hashing import sha256
from .permissions import GROUP_AREAS, OWNER_AREAS, SHARED_ROOT
from .protocol import read_protocol
from .splits import SPLIT_NAMES, read_split


def acl(path):
    result = subprocess.run(["getfacl", "-c", "-p", "-n", str(path)], text=True, capture_output=True, check=True)
    entries = {}
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        key, permissions = line.split("\t", 1)[0].rsplit(":", 1)
        entries[key] = permissions
    return entries


def _mode_permissions(mode, shift):
    bits = (mode >> shift) & 7
    return "".join(character if bits & bit else "-" for character, bit in (("r", 4), ("w", 2), ("x", 1)))


class ACLReader:
    """No subprocess for a plain mode ACL; reuse identical inherited ACLs."""
    def __init__(self):
        self.cache = {}
        self.subprocesses = 0

    @staticmethod
    def _attribute(path, name):
        try:
            return os.getxattr(path, name, follow_symlinks=False)
        except OSError as error:
            if error.errno in {errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP}:
                return None
            raise

    def read(self, path, metadata, defaults=False):
        access = self._attribute(path, "system.posix_acl_access")
        default = self._attribute(path, "system.posix_acl_default") if defaults else None
        if access is None and default is None:
            mode = stat.S_IMODE(metadata.st_mode)
            return {"user:": _mode_permissions(mode, 6), "group:": _mode_permissions(mode, 3),
                    "other:": _mode_permissions(mode, 0)}
        key = (access, default, defaults, stat.S_IMODE(metadata.st_mode))
        if key not in self.cache:
            self.cache[key] = acl(path)
            self.subprocesses += 1
        return self.cache[key]


def _group_has(entries, gid, required):
    permissions = set(entries.get("group:", "")) | set(entries.get(f"group:{gid}", ""))
    mask = entries.get("mask:", "rwx")
    return all(character in permissions and character in mask for character in required)


def storage_checks(root=SHARED_ROOT, group="delta_bgun"):
    started = time.perf_counter()
    reader = ACLReader()
    scanned = 0
    root = Path(root)
    gid = grp.getgrnam(group).gr_gid
    errors, unreadable = [], []
    if gid not in {*os.getgroups(), os.getgid(), os.getegid()}:
        errors.append("Current user is not in " + group)
    owner_uid = root.stat().st_uid
    for path, mode, permissions in [(root, 0o2750, "r-x")] + [
        (root / name, 0o2750 if name in OWNER_AREAS else 0o3770, "r-x" if name in OWNER_AREAS else "rwx")
        for name in OWNER_AREAS + GROUP_AREAS
    ]:
        try:
            metadata = os.stat(path, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("Expected a real shared directory")
            if metadata.st_gid != gid or metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) != mode:
                errors.append("Unexpected owner/group/mode: " + str(path))
            entries = reader.read(path, metadata, defaults=True)
            for field in ("group:", "mask:", "default:group:", "default:mask:"):
                # Minimal POSIX ACLs have no mask entry; their owning-group
                # entry already carries the effective permissions.
                fallback = entries.get(field.replace("mask:", "group:")) if field.endswith("mask:") else None
                if entries.get(field, fallback) != permissions:
                    errors.append(f"ACL {field} on {path}: expected {permissions}, got {entries.get(field)}")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(str(path) + ": " + str(error))
    if not os.access(root / "staging", os.W_OK | os.X_OK):
        errors.append("No write/search access to staging; ask the owner to check group membership/ACLs")
    if os.geteuid() == owner_uid:
        for name in OWNER_AREAS:
            if not os.access(root / name, os.W_OK | os.X_OK):
                errors.append("Owner has no write/search access: " + str(root / name))
    # Listing shared artifacts is permitted; data directories are never scanned.
    def scan_error(error):
        errors.append("Cannot inspect shared tree: " + str(error))
    for parent, directories, files in os.walk(root, followlinks=False, onerror=scan_error):
        for name in directories + files:
            path = Path(parent) / name
            try:
                metadata = os.stat(path, follow_symlinks=False)
                scanned += 1
                if stat.S_ISLNK(metadata.st_mode):
                    errors.append("Shared tree contains an uninspected symlink: " + str(path))
                    continue
                if metadata.st_gid != gid:
                    errors.append("Wrong group: " + str(path))
                if stat.S_ISREG(metadata.st_mode):
                    if not metadata.st_mode & stat.S_IRGRP:
                        unreadable.append(str(path))
                    elif not _group_has(reader.read(path, metadata), gid, "r"):
                        unreadable.append(str(path))
                elif stat.S_ISDIR(metadata.st_mode):
                    if metadata.st_mode & (stat.S_IRGRP | stat.S_IXGRP) != stat.S_IRGRP | stat.S_IXGRP:
                        errors.append("Group cannot read/search directory: " + str(path))
                    elif not _group_has(reader.read(path, metadata), gid, "rx"):
                        errors.append("Group cannot read/search directory: " + str(path))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(str(path) + ": " + str(error))
    if unreadable:
        errors.append("Shared files are not group-readable")
    return {"status": "fail" if errors else "pass", "root": str(root), "group": group,
            "errors": errors, "group_unreadable_files": sorted(unreadable),
            "scan_duration_seconds": time.perf_counter() - started,
            "scanned_entries": scanned, "acl_subprocesses": reader.subprocesses}


def doctor(repo, root=SHARED_ROOT, group="delta_bgun"):
    checks = {}
    for name, function in [("storage", lambda: storage_checks(root, group)),
                           ("environment", lambda: check_environment(repo))]:
        try:
            checks[name] = function()
        except Exception as error:
            checks[name] = {"status": "fail", "error": str(error)}
    try:
        protocol, digest = read_protocol(Path(repo) / "protocols/forecast_v1.json")
        checks["protocol"] = {"status": "pass", "protocol_id": protocol["protocol_id"], "sha256": digest}
    except Exception as error:
        checks["protocol"] = {"status": "fail", "error": str(error)}
    checks["splits"] = []
    # Only repository split definitions are listed here, never data directories.
    for path in sorted((Path(repo) / "splits").glob("*.json")):
        item = {"file": str(path)}
        try:
            split = read_split(path)
            shared = Path(root) / "splits" / path.name
            if shared.is_symlink() or shared.read_bytes() != path.read_bytes():
                raise ValueError("Shared frozen split differs from the repository")
            if stat.S_IMODE(shared.stat().st_mode) != 0o444:
                raise ValueError("Shared frozen split is not mode 444")
            if sha256(split.geometry_path) != split.geometry_sha256:
                raise ValueError("Geometry hash differs from the frozen split")
            manifest = Path(root) / "checksums" / (split.split_id + ".sha256")
            if manifest.exists():
                hashes = read_checksums(manifest)
                expected = {d.file for d in split.dumps(SPLIT_NAMES)} | {split.geometry_file}
                if set(hashes) != expected or hashes[split.geometry_file] != split.geometry_sha256:
                    raise ValueError("Checksum coverage or geometry hash differs from the split")
                item.update(status="pass", checksums_sha256=sha256(manifest))
            else:
                item.update(status="warn", message="Owner must submit scripts/checksums.sbatch " + str(path))
            item.update(split_id=split.split_id, sha256=split.sha256)
        except Exception as error:
            item.update(status="fail", error=str(error))
        checks["splits"].append(item)
    if not checks["splits"]:
        checks["splits"] = [{"status": "fail", "error": "No frozen split files found"}]
    statuses = [value["status"] for key, value in checks.items() if key != "splits"] + [v["status"] for v in checks["splits"]]
    status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {"ok": status != "fail", "status": status,
            "next": "gval conformance" if status != "fail" else "Ask the benchmark owner to resolve the reported failures; rerun gval doctor",
            "environment_locked": checks.get("environment", {}).get("environment_locked", False), "checks": checks}

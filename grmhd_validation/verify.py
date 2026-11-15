"""Bundle verification, tolerance comparison, and owner-only sealing recovery."""
import json
import fcntl
import os
from pathlib import Path
import stat

from .artifacts import fingerprint, result_files
from .hashing import sha256
from .protocol import read_protocol
from .splits import relative_path
from .validation import schema_validate


def _verify_bundle(path, repo=None):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Run root must be a real directory")
    repo = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    if not (path / "COMPLETE").is_file() or (path / "FAILED").exists():
        raise ValueError("Only a successful COMPLETE bundle can be verified")
    actual = set()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("Run bundle contains a symlink")
        mode = item.stat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Run bundle contains a special file: " + str(item))
        if mode & 0o222:
            raise ValueError("Run contents are not read-only: " + str(item))
        if item.is_file() and item.name not in {"COMPLETE", "checksums.json"}:
            actual.add(item.relative_to(path).as_posix())
    hashes = json.loads((path / "checksums.json").read_text())
    if actual != set(hashes):
        raise ValueError("Bundle files differ from checksum manifest")
    for name, expected in hashes.items():
        relative_path(name)
        if sha256(path / name) != expected:
            raise ValueError("Run artifact checksum mismatch: " + name)
    info = json.loads((path / "run.json").read_text())
    schema_validate(info, "run.schema.json")
    if info["status"] != "complete":
        raise ValueError("run.json does not mark completion")
    split_file = repo / "splits" / (info["split_id"] + ".json")
    protocol_file = repo / "protocols" / (info["protocol_id"] + ".json")
    if sha256(split_file) != info["split_file_sha256"] or sha256(protocol_file) != info["protocol_sha256"]:
        raise ValueError("Bundle split/protocol identity differs from repository")
    schema_validate(json.loads((path / "summary.json").read_text()), "summary.schema.json")
    protocol, _ = read_protocol(protocol_file)
    for name in protocol["tiers"][info["tier"]]:
        schema_validate(json.loads((path / "metrics" / (name + ".json")).read_text()), name + ".schema.json")
    result = fingerprint(result_files(path))
    if result != info["result_fingerprint"] or (path / "COMPLETE").read_text().strip() != result:
        raise ValueError("Result fingerprint/completion marker mismatch")
    unsealed = bool(path.stat().st_mode & 0o222)
    return {"ok": not unsealed, "unsealed": unsealed,
            "next": "gval seal " + str(path) if unsealed else "Review this verified run; use gval report after reporting is implemented",
            "run_dir": str(path), "result_fingerprint": result, "data_verified": info["data_verified"]}


def verify_run(path, repo=None):
    """An unsealed COMPLETE run is not eligible for comparison or reporting."""
    return _verify_bundle(path, repo)


def seal_run(path, repo=None):
    """§7's sole permission exception: verify, then remove root write bits only."""
    path = Path(path).absolute()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        metadata = os.fstat(fd)
        if metadata.st_uid != os.geteuid():
            raise PermissionError("Only the run's owner may seal it")
        verdict = _verify_bundle(path, repo)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise PermissionError("Run root changed during verification")
        # Work through the original FD; never chmod children or rewrite files.
        if verdict["unsealed"]:
            os.fchmod(fd, stat.S_IMODE(os.fstat(fd).st_mode) & ~0o222)
    finally:
        os.close(fd)
    return {**verdict, "ok": True, "unsealed": False,
            "next": "gval verify " + str(path)}


def compare_runs(left, right, repo=None):
    for path in (left, right):
        if not verify_run(path, repo)["ok"]:
            raise ValueError("Comparison excludes unsealed runs; run gval seal first")
    a, b = (json.loads((Path(root) / "run.json").read_text()) for root in (left, right))
    for key in ("request", "split_file_sha256", "protocol_sha256", "split", "tier", "checkpoint_sha256"):
        if a[key] != b[key]:
            raise ValueError("Comparison requires the same request, checkpoint, split, protocol, and tier")
    from .conformance import compare_results
    root = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    protocol, _ = read_protocol(root / "protocols" / (a["protocol_id"] + ".json"))
    comparison = compare_results(Path(left), Path(right), protocol["gpu_tolerance"])
    return {**comparison, "next": "Review the verified comparison", "relative_tolerance": protocol["gpu_tolerance"]}

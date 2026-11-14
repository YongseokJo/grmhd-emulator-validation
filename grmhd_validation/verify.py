"""Read-only verification of completed bundles and tolerance comparisons."""
import json
from pathlib import Path

from .artifacts import fingerprint, result_files
from .hashing import sha256
from .protocol import read_protocol
from .splits import relative_path
from .validation import schema_validate


def verify_run(path, repo=None):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    if not (path / "COMPLETE").is_file() or (path / "FAILED").exists():
        raise ValueError("Only a successful COMPLETE bundle can be verified")
    hashes = json.loads((path / "checksums.json").read_text())
    actual = set()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("Run bundle contains a symlink")
        if item.is_file() and item.name not in {"COMPLETE", "checksums.json"}:
            actual.add(item.relative_to(path).as_posix())
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
    return {"ok": True, "next": "Review this verified run; use gval report after reporting is implemented",
            "run_dir": str(path), "result_fingerprint": result, "data_verified": info["data_verified"]}


def compare_runs(left, right, repo=None):
    verify_run(left, repo); verify_run(right, repo)
    a, b = (json.loads((Path(root) / "run.json").read_text()) for root in (left, right))
    for key in ("request", "split_file_sha256", "protocol_sha256", "split", "tier", "checkpoint_sha256"):
        if a[key] != b[key]:
            raise ValueError("Comparison requires the same request, checkpoint, split, protocol, and tier")
    from .conformance import compare_results
    root = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    protocol, _ = read_protocol(root / "protocols" / (a["protocol_id"] + ".json"))
    comparison = compare_results(Path(left), Path(right), protocol["gpu_tolerance"])
    return {**comparison, "next": "Review the verified comparison", "relative_tolerance": protocol["gpu_tolerance"]}

"""Owner CPU references and result-content conformance, separate from runs/."""
import csv
import getpass
import io
import json
import math
import os
from pathlib import Path
import uuid

import torch

from .artifacts import Writer, fingerprint, result_files
from .environment import check_environment, platform_id
from .hashing import sha256
from .permissions import SharedStorage
from .protocol import read_protocol
from .runstore import no_replace_rename, reserve_conformance
from .slurm import require_compute
from .splits import identifier, read_split


class MissingGoldenError(FileNotFoundError):
    pass


def reference_path(storage, split_id, protocol, platform):
    return storage.root / "references" / split_id / protocol["protocol_id"] / "persistence_quick" / platform


def load_golden(path):
    from .runner import schema_validate
    record = json.loads((path / "golden.json").read_text())
    schema_validate(record, "golden.schema.json")
    files = result_files(path)
    if files != record["files"] or fingerprint(files) != record["result_fingerprint"]:
        raise ValueError("Golden result files do not match golden.json")
    return record


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def numeric_difference(left, right):
    """Return structural equality and largest symmetric relative difference."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf
        values = [numeric_difference(left[key], right[key]) for key in sorted(left)]
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, math.inf
        values = [numeric_difference(a, b) for a, b in zip(left, right)]
    elif type(left) in (int, float) and type(right) in (int, float):
        scale = max(abs(left), abs(right))
        return True, abs(left - right) / scale if scale else 0.
    else:
        return left == right, 0. if left == right else math.inf
    return all(ok for ok, _ in values), max((difference for _, difference in values), default=0.)


def compare_results(actual, expected, tolerance=None):
    ours, theirs = result_files(actual), result_files(expected)
    differing = sorted(path for path in set(ours) | set(theirs) if ours.get(path) != theirs.get(path))
    largest, structural = 0., set(ours) == set(theirs)
    for name in sorted(set(ours) & set(theirs)):
        if name.endswith(".json"):
            left, right = (json.loads((root / name).read_text()) for root in (Path(actual), Path(expected)))
        else:
            def rows(root):
                return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(io.StringIO((root / name).read_text()))]
            left, right = rows(Path(actual)), rows(Path(expected))
        match, difference = numeric_difference(left, right)
        structural &= match
        largest = max(largest, difference)
    ok = not differing if tolerance is None else structural and largest <= tolerance
    return {"ok": bool(ok), "differing_files": differing,
            "largest_relative_difference": largest if math.isfinite(largest) else None}


def reference(split_id, tier="quick", partition="validation", device="cpu", storage=None, repo=None, data_root=None):
    require_compute()
    from .runner import REPO, execute, persistence_request
    storage = SharedStorage() if storage is None else storage
    repo = REPO if repo is None else Path(repo)
    storage.require_owner()
    if tier != "quick" or partition != "validation" or device != "cpu":
        raise ValueError("Milestone 2 reference is persistence quick validation on CPU only")
    identifier(split_id)
    split = read_split(repo / "splits" / (split_id + ".json"), data_root)
    if sha256(storage.root / "splits" / (split_id + ".json")) != split.sha256:
        raise ValueError("Shared split identity differs")
    protocol, protocol_sha = read_protocol(repo / "protocols/forecast_v1.json")
    platform = platform_id()
    parent = storage.folder("references", split_id, protocol["protocol_id"], "persistence_quick")
    candidate = storage.exclusive_folder("references", split_id, protocol["protocol_id"], "persistence_quick", ".candidate-" + uuid.uuid4().hex)
    writer = Writer(storage, candidate)
    info = execute(persistence_request(), split, protocol, protocol_sha, partition, writer, torch.device("cpu"), True, repo)
    info.update(split=partition, tier=tier, purpose="reference", status="complete")
    writer.json("run.json", info)
    files = result_files(candidate)
    record = {"result_fingerprint": fingerprint(files), "files": files, "platform": platform,
              "environment_locked": info["environment_locked"], "lock_sha256": info["environment_lock_sha256"],
              "repo_commit": info["repo_commit"], "provisional": not info["environment_locked"]}
    writer.json("golden.json", record)
    target = parent / platform
    if target.exists():
        existing = load_golden(target)
        if existing["result_fingerprint"] != record["result_fingerprint"]:
            raise ValueError("Golden differs; a new protocol or recorded owner decision is required. Candidate retained: " + str(candidate))
        # Identical result bytes may acquire a real lock after a provisional run.
        # Replace metadata atomically, without chmodding any existing entry.
        for filename in ("golden.json", "run.json"):
            temporary = "." + filename + "." + uuid.uuid4().hex
            with storage.new_file(target, temporary) as stream:
                stream.write((candidate / filename).read_bytes())
            os.replace(target / temporary, target / filename)
        _remove_candidate(storage, candidate)
    else:
        no_replace_rename(candidate, target)
    return {"ok": True, "next": "gval conformance --split-id " + split_id,
            "reference_dir": str(target), **record}


def _remove_candidate(storage, candidate):
    # Only paths created in this invocation can be cleaned up.
    for path in sorted((p for p in storage.created if p == candidate or candidate in p.parents), key=lambda p: len(p.parts), reverse=True):
        metadata = path.lstat()
        if storage.created[path] != (metadata.st_dev, metadata.st_ino):
            raise PermissionError("Reference candidate changed before cleanup")
        path.rmdir() if path.is_dir() else path.unlink()


def conformance(split_id, storage=None, repo=None, data_root=None):
    require_compute()
    from .runner import REPO, execute, persistence_request, record_failure
    storage = SharedStorage() if storage is None else storage
    repo = REPO if repo is None else Path(repo)
    identifier(split_id)
    platform = platform_id()
    output = reserve_conformance(storage, platform)
    writer = Writer(storage, output)
    info = {"purpose": "conformance", "platform": platform, "split_id": split_id}
    try:
        protocol, protocol_sha = read_protocol(repo / "protocols/forecast_v1.json")
        golden_dir = reference_path(storage, split_id, protocol, platform)
        cross = False
        if not (golden_dir / "golden.json").is_file():
            alternatives = [name for name in json.loads((repo / "environment/platforms.json").read_text()) if name != platform]
            choices = [reference_path(storage, split_id, protocol, name) for name in alternatives]
            choices = [path for path in choices if (path / "golden.json").is_file()]
            if not choices:
                raise MissingGoldenError("No golden exists; ask the owner to run scripts/reference_delta.sbatch")
            golden_dir, cross = choices[0], True
        golden = load_golden(golden_dir)
        golden_run = json.loads((golden_dir / "run.json").read_text())
        split = read_split(repo / "splits" / (split_id + ".json"), data_root)
        if sha256(storage.root / "splits" / (split_id + ".json")) != split.sha256:
            raise ValueError("Shared split identity differs")
        if golden_run["split_file_sha256"] != split.sha256 or golden_run["protocol_sha256"] != protocol_sha:
            raise ValueError("Golden split/protocol identity differs")
        info = execute(persistence_request(), split, protocol, protocol_sha, "validation", writer, torch.device("cpu"), True, repo)
        info.update(purpose="conformance", split="validation", tier="quick", status="complete")
        writer.json("run.json", info)
        comparison = compare_results(output, golden_dir, tolerance=1e-12 if cross else None)
        verdict = {**comparison, "next": "gval new-adapter <dir>" if comparison["ok"] else "Inspect differing_files and ask the owner to investigate the golden/environment",
                   "status": "warn" if cross and comparison["ok"] else "pass" if comparison["ok"] else "fail",
                   "cross_platform": cross, "platform": platform, "split_id": split_id,
                   "protocol_sha256": protocol_sha, "environment_lock_sha256": info["environment_lock_sha256"],
                   "result_fingerprint": info["result_fingerprint"], "golden_fingerprint": golden["result_fingerprint"],
                   "output_dir": str(output), "end_time": info["end_time"]}
        writer.json("conformance.json", verdict)
        writer.write("COMPLETE" if verdict["ok"] else "FAILED", (info["result_fingerprint"] + "\n").encode())
        return verdict
    except Exception as error:
        record_failure(writer, info, error)
        return {"ok": False, "status": "fail", "next": "ask the owner to run scripts/reference_delta.sbatch" if isinstance(error, MissingGoldenError) else "Inspect the conformance FAILED file",
                "error": str(error), "output_dir": str(output)}


def last_conformance(storage, split_id, protocol_sha, environment):
    root = storage.root / "staging" / getpass.getuser() / "conformance"
    if not root.is_dir():
        raise RuntimeError("Run gval conformance --split-id " + split_id + " before evaluation")
    matches = []
    # Metadata only: never read or discover simulation data here.
    for path in root.glob("*/conformance.json"):
        data = json.loads(path.read_text())
        if data.get("ok") and (path.parent / "COMPLETE").is_file() and data.get("split_id") == split_id and data.get("platform") == environment["platform"] and data.get("environment_lock_sha256") == environment.get("lock_sha256") and data.get("protocol_sha256") == protocol_sha:
            if fingerprint(result_files(path.parent)) == data.get("result_fingerprint"):
                matches.append(data)
    if not matches:
        raise RuntimeError("No passing conformance for this platform/lock/split; run gval conformance")
    last = max(matches, key=lambda d: d["end_time"])
    return {"result_fingerprint": last["result_fingerprint"], "golden_fingerprint": last["golden_fingerprint"],
            "cross_platform": last["cross_platform"], "platform": last["platform"],
            "environment_lock_sha256": last["environment_lock_sha256"], "end_time": last["end_time"]}

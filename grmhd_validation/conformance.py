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
from .protocol import read_protocol,full_tier_ready
from .runstore import no_replace_rename, reserve_conformance
from .slurm import require_compute
from .splits import identifier, read_split


class MissingGoldenError(FileNotFoundError):
    pass


def reference_path(storage, split_id, protocol, platform,tier='quick',protocol_sha=None):
    parent=storage.root/'references'/split_id/protocol['protocol_id']
    if full_tier_ready(protocol):
        if protocol_sha is None:
            import hashlib
            from .artifacts import json_bytes
            protocol_sha=hashlib.sha256(json_bytes(protocol)).hexdigest()
        parent/=protocol_sha
    return parent/('persistence_'+tier)/platform


def load_golden(path):
    if (Path(path)/'.summary-format-v2.json').exists():raise ValueError('Reference summary correction is incomplete; owner must resume --rebuild-summary')
    from .runner import schema_validate
    record = json.loads((path / "golden.json").read_text())
    schema_validate(record, "golden.schema.json")
    files = result_files(path)
    if files != record["files"] or fingerprint(files) != record["result_fingerprint"]:
        raise ValueError("Golden result files do not match golden.json")
    summary=json.loads((Path(path)/'summary.json').read_text())
    if summary.get('schema_version')==2:
        from .summary import validate
        validate(summary)
        if any(files.get('metrics/'+name+'.json')!=digest for name,digest in summary['metric_sha256'].items()):
            raise ValueError('Golden compact-summary metric hashes differ')
    return record


def refresh_reference_figures(storage,reference,figures,expected_fingerprint):
    """Owner-only derived metadata refresh; completed model runs are immutable."""
    storage.require_owner();reference=Path(reference).absolute();figures=Path(figures)
    if not reference.is_relative_to(storage.root/'references') or reference.is_symlink():
        raise ValueError('Figure refresh requires an owner persistence reference')
    golden=load_golden(reference)
    if golden['result_fingerprint']!=expected_fingerprint:raise ValueError('Reference result fingerprint differs')
    info=json.loads((reference/'run.json').read_text())
    if info.get('model')!='persistence' or info.get('purpose')!='reference':raise ValueError('Figure refresh requires a persistence golden')
    from .analyses._port_manifest import FIGURES
    expected={'fig02_rollout_curve.png'}
    if info.get('tier')=='full':expected|={filename for files in FIGURES.values() for filename in files.values()}
    actual={path.name:path for path in figures.glob('*.png')}
    if set(actual)!=expected:raise ValueError('Reference figure refresh is incomplete')
    contents={}
    for name,path in actual.items():
        if path.is_symlink():raise ValueError('Reference presentation cannot contain symlinks')
        content=path.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Reference presentation must contain PNGs')
        contents[name]=content
    destination=storage.folder(*reference.relative_to(storage.root).parts,'figures')
    def replace(name,content):
        temporary='.'+name+'.'+uuid.uuid4().hex
        with storage.new_file(destination,temporary) as stream:stream.write(content)
        os.replace(destination/temporary,destination/name)
    for name,content in contents.items():replace(name,content)
    if load_golden(reference)['result_fingerprint']!=expected_fingerprint:raise RuntimeError('Reference results changed during figure refresh')
    from .artifacts import json_bytes
    record={'result_fingerprint':expected_fingerprint,'figures':{name:sha256(destination/name) for name in sorted(actual)}}
    replace('presentation.json',json_bytes(record))
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
    if not differing:
        return {'ok':True,'differing_files':[],'largest_relative_difference':0.}
    largest, structural = 0., set(ours) == set(theirs)
    for name in sorted(set(ours) & set(theirs)):
        if name.endswith(".json"):
            left, right = (json.loads((root / name).read_text()) for root in (Path(actual), Path(expected)))
            if name=='summary.json':
                # Hash bytes determine differing_files; they are not numerical
                # values for relative-difference diagnostics or tolerance.
                left.pop('metric_sha256',None);right.pop('metric_sha256',None)
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


def reference(split_id, tier="quick", partition="validation", device="cpu", storage=None, repo=None, data_root=None,truth_only=False):
    require_compute()
    from .runner import REPO, execute, persistence_request
    storage = SharedStorage() if storage is None else storage
    repo = REPO if repo is None else Path(repo)
    storage.require_owner()
    if tier not in {'quick','full'} or partition not in {'validation','test'} or device != 'cpu':
        raise ValueError('References require quick/full on CPU and validation/test')
    if not truth_only and partition!='validation':raise ValueError('Persistence goldens use validation only')
    if truth_only:tier='full'
    identifier(split_id)
    split = read_split(repo / "splits" / (split_id + ".json"), data_root)
    if sha256(storage.root / "splits" / (split_id + ".json")) != split.sha256:
        raise ValueError("Shared split identity differs")
    protocol, protocol_sha = read_protocol(repo / "protocols/forecast_v1.json")
    platform = platform_id()
    target=reference_path(storage,split_id,protocol,platform,tier,protocol_sha)
    parent=storage.folder(*target.parent.relative_to(storage.root).parts)
    candidate=storage.exclusive_folder(*parent.relative_to(storage.root).parts,'.candidate-'+uuid.uuid4().hex)
    writer = Writer(storage, candidate)
    from .truth_cache import open_cache,cache_path,TruthCache,identity
    cache=None;cache_candidate=None
    try:
        if tier=='full':
            cache=open_cache(storage,split,partition,protocol,protocol_sha,required=False)
            if cache is None:
                destination=cache_path(storage,split,partition,protocol,protocol_sha)
                cache_candidate=storage.exclusive_folder(*destination.parent.relative_to(storage.root).parts,'.candidate-'+uuid.uuid4().hex)
                with storage.new_file(cache_candidate,'quantities.sqlite3'):pass
                cache=TruthCache(cache_candidate,building=True)
        info = execute(persistence_request(), split, protocol, protocol_sha, partition, writer, torch.device('cpu'), True, repo,tier=tier,truth_cache=cache)
        if cache is not None and cache.building:
            expected=identity(split,partition,protocol,protocol_sha,storage.root/'checksums'/(split_id+'.sha256'))
            cache.finish(storage,expected,info['repo_commit'])
            no_replace_rename(cache_candidate,destination)
            # The published owner cache is no longer a disposable candidate.
            for path in list(storage.created):
                if path==cache_candidate or cache_candidate in path.parents:
                    storage.created.pop(path);storage.original_modes.pop(path,None)
            cache.path=destination
        if cache is not None:
            info['truth_cache_sha256']=cache.manifest['cache_sha256']
            info['truth_cache_manifest_sha256']=sha256(cache.path/'manifest.json')
        if truth_only:
            _remove_candidate(storage,candidate)
            return {'ok':True,'next':'gval reference persistence --split-id '+split_id+' --tier full',
                    'cache_dir':str(cache.path),'cache_sha256':cache.manifest['cache_sha256'],'entries':len(cache.entries)}
    except BaseException:
        # Includes cancellation: remove this invocation's unpublished candidates.
        if cache is not None:cache.close()
        for path in (cache_candidate,candidate):
            if path is not None and path.exists():_remove_candidate(storage,path)
        raise
    finally:
        if cache is not None:cache.close()
    info.update(split=partition, tier=tier, purpose="reference", status="complete")
    from .repo_state import require_clean
    require_clean(info, repo)
    writer.json("run.json", info)
    files = result_files(candidate)
    record = {"result_fingerprint": fingerprint(files), "files": files, "platform": platform,
              "environment_locked": info["environment_locked"], "lock_sha256": info["environment_lock_sha256"],
              "repo_commit": info["repo_commit"], "provisional": not info["environment_locked"], "summary_format_version":2}
    writer.json("golden.json", record)
    if tier=='quick' and full_tier_ready(protocol):
        legacy=storage.root/'references'/split_id/protocol['protocol_id']/'persistence_quick'/platform
        if (legacy/'golden.json').is_file() and {k:v for k,v in load_golden(legacy)['files'].items() if k!='summary.json'}!={k:v for k,v in record['files'].items() if k!='summary.json'}:
            _remove_candidate(storage,candidate)
            raise ValueError('Quick result bytes changed across protocol finalization; refusing a new golden')
    if target.exists():
        existing = load_golden(target)
        if existing["result_fingerprint"] != record["result_fingerprint"]:
            _remove_candidate(storage,candidate)
            raise ValueError('Golden differs; a new protocol or recorded owner decision is required')
        refresh_reference_figures(storage,target,candidate/'figures',record['result_fingerprint'])
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


def conformance(split_id, storage=None, repo=None, data_root=None,tier='quick'):
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
        golden_dir = reference_path(storage, split_id, protocol, platform,tier,protocol_sha)
        cross = False
        if not (golden_dir / "golden.json").is_file():
            alternatives = [name for name in json.loads((repo / "environment/platforms.json").read_text()) if name != platform]
            choices = [reference_path(storage, split_id, protocol, name,tier,protocol_sha) for name in alternatives]
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
        info = execute(persistence_request(), split, protocol, protocol_sha, "validation", writer, torch.device("cpu"), True, repo,tier=tier)
        info.update(purpose="conformance", split="validation", tier=tier, status="complete")
        writer.json("run.json", info)
        comparison = compare_results(output, golden_dir, tolerance=1e-12 if cross else None)
        from .repo_state import require_clean
        require_clean(info, repo)
        verdict = {**comparison, "next": "gval new-adapter <dir>" if comparison["ok"] else "Inspect differing_files and ask the owner to investigate the golden/environment",
                   "status": "warn" if cross and comparison["ok"] else "pass" if comparison["ok"] else "fail",
                   "cross_platform": cross, "platform": platform, "split_id": split_id,
                   "protocol_sha256": protocol_sha, "tier":tier,"environment_lock_sha256": info["environment_lock_sha256"],
                   "result_fingerprint": info["result_fingerprint"], "golden_fingerprint": golden["result_fingerprint"], "summary_format_version":2,
                   "output_dir": str(output), "end_time": info["end_time"], "repo_modified": info['repo_modified']}
        writer.json("conformance.json", verdict)
        writer.write("COMPLETE" if verdict["ok"] else "FAILED", (info["result_fingerprint"] + "\n").encode())
        return verdict
    except Exception as error:
        record_failure(writer, info, error)
        return {"ok": False, "status": "fail", "next": "ask the owner to run scripts/reference_delta.sbatch" if isinstance(error, MissingGoldenError) else "Inspect the conformance FAILED file",
                "error": str(error), "output_dir": str(output)}


def last_conformance(storage, split_id, protocol_sha, environment,tier='quick'):
    root = storage.root / "staging" / getpass.getuser() / "conformance"
    if not root.is_dir():
        raise RuntimeError("Run gval conformance --split-id " + split_id + " before evaluation")
    matches = []
    # Metadata only: never read or discover simulation data here.
    for path in root.glob("*/conformance.json"):
        data = json.loads(path.read_text())
        if data.get('repo_modified') is not True and data.get("summary_format_version")==2 and data.get("ok") and (path.parent / "COMPLETE").is_file() and data.get("split_id") == split_id and data.get('tier','quick')==tier and data.get("platform") == environment["platform"] and data.get("environment_lock_sha256") == environment.get("lock_sha256") and data.get("protocol_sha256") == protocol_sha:
            if fingerprint(result_files(path.parent)) == data.get("result_fingerprint"):
                matches.append(data)
    if not matches:
        raise RuntimeError("No passing conformance for this platform/lock/split; run gval conformance")
    last = max(matches, key=lambda d: d["end_time"])
    return {"result_fingerprint": last["result_fingerprint"], "golden_fingerprint": last["golden_fingerprint"],
            "cross_platform": last["cross_platform"], "platform": last["platform"],
            "environment_lock_sha256": last["environment_lock_sha256"], "end_time": last["end_time"]}

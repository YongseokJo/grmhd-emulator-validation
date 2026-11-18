"""One physical-unit inference path, recorded attempts, and canonical artifacts."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import traceback

import numpy as np
import torch

from .adapter import adapter_source, validate_declaration, validate_history, validate_output
from .artifacts import Writer, fingerprint, result_files
from .data import FrameReader, load_geometry
from .environment import check_environment, platform_id
from .hashing import sha256
from .permissions import SharedStorage
from .protocol import read_protocol,full_tier_ready
from .runstore import UnsealedRunError, publish, reserve
from .slurm import require_compute
from .splits import CHANNELS, identifier, read_split
from .windows import select_starts
from .validation import schema_validate
from .repo_state import inspect as inspect_repository, require_clean

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def git_commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def read_request(path):
    request = json.loads(Path(path).read_text())
    schema_validate(request, "request.schema.json")
    return request


def persistence_request():
    return {"model": "persistence", "version": "v1", "adapter": "persistence", "checkpoint": None,
            "options": {}, "training_record": None, "notes": ""}


def initialize(protocol, device):
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@dataclass
class Context:
    adapter: object
    geometry: object
    reader: FrameReader
    starts: tuple
    protocol: dict
    split: object
    partition: str
    device: object
    writer: Writer
    channels: tuple = CHANNELS
    full: object = None
    channel_scales: object = None
    run_info: object = None
    truth_cache: object = None

    @property
    def constants(self):
        return self.protocol["constants"]

    def step(self, history, prediction_key=None):
        validate_history(history, self.adapter, self.geometry)
        source = getattr(self, 'predictions', None)
        if source is not None:
            if prediction_key is None:
                raise ValueError('Prediction scoring needs an explicit kind/start/step')
            return validate_output(source.prediction(*prediction_key, history), history, self.geometry)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            # An adapter cannot mutate the runner's past states or persistence baseline.
            output = self.adapter.step(history.clone())
            return validate_output(output, history, self.geometry).detach().clone()


def metadata(request, split, protocol, protocol_sha, checkpoint_sha, device, environment, repo):
    return {"request": request, "model": request["model"], "version": request["version"],
            "checkpoint": request["checkpoint"], "checkpoint_sha256": checkpoint_sha,
            "user": getpass.getuser(), "host": socket.gethostname(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "torch": str(torch.__version__), "numpy": np.__version__,
            "python": __import__("platform").python_version(), "platform": platform_id(),
            "environment_lock_sha256": environment.get("lock_sha256"),
            "environment_locked": environment.get("environment_locked", False),
            "repo_commit": git_commit(repo), "adapter_sha256": None, "adapter_repo_commit": None,
            "split_id": split.split_id, "split_file_sha256": split.sha256,
            "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
            "data_verified": False, "start_time": now(), "end_time": None,
            "result_fingerprint": None, "runner_autocast": False, **inspect_repository(repo)}


def execute(request, split, protocol, protocol_sha, partition, writer, device, verify_data, repo=REPO, run_info=None, tier='quick',truth_cache=None,predictions=None):
    """Shared by evaluate, reference and conformance; callers own output placement."""
    environment = check_environment(repo)
    if environment["status"] == "fail":
        raise RuntimeError("Current environment does not match its recorded platform lock")
    checkpoint_sha = sha256(request["checkpoint"]) if request["checkpoint"] is not None else None
    info = metadata(request, split, protocol, protocol_sha, checkpoint_sha, device, environment, repo) if run_info is None else run_info
    writer.json("run.json", info)
    require_clean(info, repo)
    for directory in ("metrics", "tables", "figures", "logs"):
        writer.folder(directory)
    with writer.open("logs/evaluate.log") as raw_log:
        log = io.TextIOWrapper(raw_log, encoding="utf-8", write_through=True)
        with redirect_stdout(log), redirect_stderr(log):
            initialize(protocol, device)
            reader = FrameReader(split, partition, writer.storage.root / "checksums" / (split.split_id + ".sha256"), verify_data)
            geometry = load_geometry(split, device, reader.hashes)
            if predictions is None:
                adapter_class, source = adapter_source(request["adapter"])
                info.update(adapter_sha256=source["source_sha256"], adapter_repo_commit=source["repo_commit"])
                validate_declaration(adapter_class, protocol)
                if adapter_class.name != request["model"] or adapter_class.version != request["version"]:
                    raise ValueError("Run request model/version differs from adapter declaration")
                if adapter_class.requires_checkpoint != (request["checkpoint"] is not None):
                    raise ValueError("Request checkpoint null/path disagrees with adapter.requires_checkpoint")
                adapter = adapter_class(request["checkpoint"], geometry, str(device), request["options"])
                adapter.eval()
                info['model_provenance'] = getattr(adapter, 'provenance', None)
            else:
                from types import SimpleNamespace
                adapter = SimpleNamespace(name=request['model'], version=request['version'],
                                          history_length=predictions.history_length, max_batch=1)
                info['predictions'] = predictions.metadata()
            from .training import training_provenance
            info['training_provenance'] = training_provenance(request, split)
            starts = select_starts(split, partition, protocol)
            info["history_length"] = adapter.history_length
            info["rollout_starts"] = {entry.chunk: list(entry.rollout) for entry in starts}
            ctx = Context(adapter, geometry, reader, starts, protocol, split, partition, device, writer)
            ctx.run_info = info
            ctx.predictions = predictions
            ctx.partial_predictions = predictions is not None and any(not predictions.availability(name)[0] for name in protocol['tiers'][tier])
            from .truth_cache import open_cache
            any_available=predictions is None or any(predictions.availability(name)[0] for name in protocol['tiers'][tier])
            ctx.truth_cache=truth_cache if truth_cache is not None else open_cache(writer.storage,split,partition,protocol,protocol_sha,required=tier=='full' and full_tier_ready(protocol) and any_available) if any_available else None
            if ctx.truth_cache is not None and ctx.truth_cache.manifest is not None:
                info['truth_cache_sha256']=ctx.truth_cache.manifest['cache_sha256']
                info['truth_cache_manifest_sha256']=sha256(ctx.truth_cache.path/'manifest.json')
            from .analyses import single_step, rollout_curve
            analyses = {"single_step": single_step, "rollout_curve": rollout_curve}
            if tier=='full':
                from .analyses._analysis_io import FullStore
                from .channel_scales import read_channel_scales
                import importlib
                ctx.full=FullStore(ctx)
                scale_path=writer.storage.root/'references'/split.split_id/'channel_scales.json'
                if any_available:
                    ctx.channel_scales=read_channel_scales(scale_path,split)
                    info['channel_scales_sha256']=sha256(scale_path)
                for name in protocol['tiers']['full']:
                    if name not in analyses:analyses[name]=importlib.import_module('.analyses.'+name,__package__)
            summary = {}
            unavailable = {}
            memory={}
            for name in protocol["tiers"][tier]:
                print('analysis:',name,flush=True)
                from .memory import measure
                available, reason = (True, None) if predictions is None else predictions.availability(name)
                if not available:
                    summary[name] = {'status':'not evaluated', 'reason':reason}
                    unavailable[name] = reason
                    writer.json('metrics/'+name+'.json', summary[name])
                    continue
                with measure(memory,name):
                    analyses[name].run(ctx)
                writer.json('logs/memory.json',memory)
                import gc
                gc.collect()
            if ctx.full is not None:ctx.full.cleanup()
            if truth_cache is None and ctx.truth_cache is not None:ctx.truth_cache.close()
            from .summary import from_metrics
            summary=from_metrics(writer.root,protocol,tier,split.dt_dump)
            writer.json("summary.json", summary)
            for name in protocol["tiers"][tier]:
                result = json.loads((writer.root / "metrics" / (name + ".json")).read_text())
                schema_validate(result, name + ".schema.json")
            if predictions is not None:
                info['not_evaluated'] = unavailable
                writer.json('coverage.json', {'analyses': {name: {'status':'not evaluated', 'reason':unavailable[name]} if name in unavailable else {'status':'evaluated'} for name in protocol["tiers"][tier]},
                                              'coverage':predictions.coverage})
            info["data_verified"] = verify_data
            info['summary_format_version']=2
            info['result_content_sha256']=result_files(writer.root)
            info["result_fingerprint"] = fingerprint(info['result_content_sha256'])
            info["end_time"] = now()
            require_clean(info, repo)
            writer.json("run.json", info)
            schema_validate(info, "run.schema.json")
            # Human-readable output is rendered from the structured results only.
            writer.write("out_res.txt", (json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())
        log.flush()
    return info


def bundle_checksums(writer, complete=True):
    files = {p.relative_to(writer.root).as_posix(): sha256(p) for p in sorted(writer.root.rglob("*")) if p.is_file() and p.name not in {"checksums.json", "COMPLETE"}}
    writer.json("checksums.json", files)
    if complete:writer.write("COMPLETE", (fingerprint(result_files(writer.root)) + "\n").encode())


def record_failure(writer, info, error):
    if hasattr(error, 'state'):
        info.update(error.state)
    # A rename can fail after freezing; only this invocation's inodes may be restored.
    writer.storage.restore_created_modes(writer.root)
    marker = writer.root / "COMPLETE"
    if marker in writer.storage.created and marker.exists():
        marker.unlink()
    detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    info.update(end_time=now(), status="failed", error=str(error))
    writer.json("run.json", info)
    writer.write("FAILED", detail.encode())
    log_path = writer.root / "logs/evaluate.log"
    previous = log_path.read_bytes() if log_path.exists() else b""
    writer.write("logs/evaluate.log", previous + detail.encode())


def evaluate(request_path, split_id, partition="validation", tier="quick", device=None, data_root=None, no_verify_data=False,
             storage=None, repo=REPO, predictions=None):
    require_compute()
    if tier not in {'quick','full'}:
        raise ValueError('Tier must be quick or full')
    storage = SharedStorage() if storage is None else storage
    request = read_request(request_path)
    identifier(split_id)
    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation requires validation or test")
    checkpoint_sha = sha256(request["checkpoint"]) if request["checkpoint"] is not None else None
    root, run_id, attempt = reserve(storage, request["model"], request["version"], checkpoint_sha, split_id, partition, tier)
    writer = Writer(storage, root)
    info = {"request": request, "run_id": run_id, "attempt": attempt, "split": partition, "tier": tier,
            "model": request['model'], "version": request['version'], "checkpoint_sha256": checkpoint_sha,
            "split_id": split_id, "start_time": now(), "end_time": None, "status": "failed", **inspect_repository(repo)}
    try:
        require_clean(info, repo)
        split = read_split(Path(repo) / "splits" / (split_id + ".json"), data_root)
        if sha256(storage.root / "splits" / (split_id + ".json")) != split.sha256:
            raise ValueError("Shared frozen split differs from repository")
        protocol, protocol_sha = read_protocol(Path(repo) / "protocols/forecast_v1.json")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        environment = check_environment(repo)
        info = {**metadata(request, split, protocol, protocol_sha, checkpoint_sha, device, environment, repo),
                "run_id": run_id, "attempt": attempt, "split": partition, "tier": tier, "status": "running"}
        require_clean(info, repo)
        from .conformance import last_conformance
        info["conformance"] = last_conformance(storage, split_id, protocol_sha, environment,tier='full' if tier=='full' and full_tier_ready(protocol) else 'quick')
        info["no_verify_data"] = no_verify_data
        prediction_source = None
        if predictions is not None:
            if no_verify_data:raise ValueError('Prediction input-chain validation requires verified data')
            from .predictions import Bundle
            prediction_source = Bundle(predictions, request_path, split, protocol, protocol_sha, partition,
                                       storage.root/'checksums'/(split_id+'.sha256'), repo)
        execute(request, split, protocol, protocol_sha, partition, writer, device, not no_verify_data, repo, info,tier,
                predictions=prediction_source)
        if info.get('not_evaluated'):
            info.update(status='not evaluated')
            writer.json('run.json', info)
            schema_validate(info, 'run.schema.json')
            writer.write('NOT_EVALUATED', b'Required analysis inputs are incomplete or unavailable; see coverage.json.\n')
            bundle_checksums(writer, complete=False)
            storage.make_readonly(root)
            return {'ok':False, 'status':'not evaluated', 'next':'Complete the frozen task and record a new evaluation attempt',
                    'run_id':run_id, 'run_dir':str(root), 'analyses':info['not_evaluated']}
        if tier=='full' and not full_tier_ready(protocol):
            reason='Milestone 4 final full-tier semantics are not approved: all-start CIs, transport/coarse-ring handling and truth cache are required'
            info.update(status='preview',preview_reason=reason)
            writer.json('run.json',info)
            schema_validate(info,'run.schema.json')
            writer.write('PREVIEW',(reason+'\n').encode())
            bundle_checksums(writer,complete=False)
            storage.make_readonly(root)
            return {'ok':True,'preview':True,'next':'Review this staging preview; await final full-tier protocol',
                    'run_id':run_id,'run_dir':str(root),'reason':reason,'result_fingerprint':info['result_fingerprint']}
        info["status"] = "complete"
        writer.json("run.json", info)
        bundle_checksums(writer)
        require_clean(info, repo)
        target = publish(storage, root, split_id, request["model"])
    except UnsealedRunError as error:
        return {"ok": False, "next": "gval seal " + str(error.target), "unsealed": True,
                "run_id": run_id, "run_dir": str(error.target), "error": str(error)}
    except Exception as error:
        record_failure(writer, info, error)
        return {"ok": False, "next": "Inspect FAILED in the staging attempt; correct the request or adapter before retrying",
                "run_id": run_id, "run_dir": str(root), "error": str(error)}
    return {"ok": True, "next": "gval verify " + str(target), "run_id": run_id, "run_dir": str(target),
            "result_fingerprint": info["result_fingerprint"], "data_verified": info["data_verified"]}

"""Compute-node checks use only common eligible validation histories."""
import time
import torch
from .adapter import adapter_source,validate_declaration,validate_history,validate_output
from .data import FrameReader,load_geometry
from .permissions import SharedStorage
from .protocol import read_protocol
from .runner import REPO,initialize
from .slurm import require_compute
from .splits import read_split
from .windows import select_starts


def check_adapter(reference,checkpoint,split_id,options=None,device='cpu',data_root=None,storage=None,repo=REPO):
    require_compute();storage=storage or SharedStorage();device=torch.device(device)
    split=read_split(repo/'splits'/(split_id+'.json'),data_root)
    protocol,_=read_protocol(repo/'protocols/forecast_v1.json');initialize(protocol,device)
    cls,source=adapter_source(reference);validate_declaration(cls,protocol)
    if cls.requires_checkpoint!=(checkpoint is not None):raise ValueError('Checkpoint does not match adapter declaration')
    reader=FrameReader(split,'validation',storage.root/'checksums'/(split_id+'.sha256'))
    geometry=load_geometry(split,device,reader.hashes);adapter=cls(checkpoint,geometry,str(device),options or {}).eval()
    starts=select_starts(split,'validation',protocol)
    timings=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for group in starts:
        history=reader.history(group.rollout[0],adapter.history_length,device)
        validate_history(history,adapter,geometry)
        with torch.inference_mode():
            if device.type=='cuda':torch.cuda.synchronize(device)
            begin=time.perf_counter();a=validate_output(adapter.step(history.clone()),history,geometry)
            if device.type=='cuda':torch.cuda.synchronize(device)
            timings.append(time.perf_counter()-begin)
            b=validate_output(adapter.step(history.clone()),history,geometry)
        if not torch.equal(a,b):raise ValueError('Adapter eval calls are not deterministic')
    return {'ok':True,'next':'gval evaluate --tier quick --split validation --request <request> --split-id '+split_id,
            'source':source,'history_length':adapter.history_length,'seconds_per_step':timings,
            'cuda_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}

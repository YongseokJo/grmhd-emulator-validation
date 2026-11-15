"""Owner-computed physical reference scales from every frozen training dump."""
import json
from pathlib import Path
import uuid

import numpy as np

from .data import FrameReader
from .hashing import sha256
from .artifacts import json_bytes
from .permissions import SharedStorage
from .splits import CHANNELS,read_split,identifier
from .slurm import require_compute
from .validation import schema_validate


def read_channel_scales(path, split):
    result=json.loads(Path(path).read_text())
    schema_validate(result,'channel_scales.schema.json')
    if result['split_id']!=split.split_id or result['split_file_sha256']!=split.sha256:
        raise ValueError('Training channel scales do not match the frozen split')
    if result['channels']!=list(CHANNELS):raise ValueError('Training scale channel order differs')
    return result


def generate(split_id, storage=None, repo=None, data_root=None):
    require_compute();identifier(split_id)
    storage=SharedStorage() if storage is None else storage;storage.require_owner()
    repo=Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    split=read_split(repo/'splits'/(split_id+'.json'),data_root)
    if sha256(storage.root/'splits'/(split_id+'.json'))!=split.sha256:raise ValueError('Shared frozen split differs')
    manifest=storage.root/'checksums'/(split_id+'.sha256')
    reader=FrameReader(split,'train',manifest,True,training_truth=True)
    total=np.zeros(8,dtype=np.float64);squares=np.zeros(8,dtype=np.float64);cells=0
    for dump in split.model_dumps('train'):
        frame=np.ascontiguousarray(reader.numpy(dump.index),dtype=np.float64).reshape(8,-1)
        total+=np.sum(np.abs(frame),axis=1,dtype=np.float64)
        squares+=np.sum(frame*frame,axis=1,dtype=np.float64)
        cells+=frame.shape[1]
    if not cells:raise ValueError('No training cells')
    result={'split_id':split_id,'split_file_sha256':split.sha256,'data_manifest_sha256':sha256(manifest),
            'channels':list(CHANNELS),'n_dumps':len(reader.allowed),'n_cells_per_channel':cells,
            'mean_abs':(total/cells).tolist(),'rms':np.sqrt(squares/cells).tolist(),'data_verified':True}
    schema_validate(result,'channel_scales.schema.json')
    directory=storage.folder('references',split_id);path=directory/'channel_scales.json';content=json_bytes(result)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=content:raise FileExistsError('Existing training scales differ; owner must investigate')
    else:
        temporary='.channel-scales-'+uuid.uuid4().hex
        try:
            with storage.new_file(directory,temporary) as stream:stream.write(content)
            import os
            os.link(directory/temporary,path,follow_symlinks=False)
        finally:
            temporary_path=directory/temporary
            if temporary_path in storage.created:temporary_path.unlink()
    return {'ok':True,'next':'gval evaluate --tier full --split validation --request <request> --split-id '+split_id,
            'output':str(path),'sha256':sha256(path),'n_dumps':len(reader.allowed)}

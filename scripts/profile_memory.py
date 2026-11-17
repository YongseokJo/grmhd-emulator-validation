"""Compute-node-only reduced-start memory diagnostic; never publishes results."""
import os
os.umask(0o007)
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import uuid

from grmhd_validation import runner
from grmhd_validation.artifacts import Writer
from grmhd_validation.permissions import SharedStorage
from grmhd_validation.protocol import read_protocol
from grmhd_validation.slurm import require_compute
from grmhd_validation.splits import read_split, Chunk
from grmhd_validation.windows import Starts, select_starts
from grmhd_validation.truth_cache import TruthCache
from grmhd_validation.memory import measure
from grmhd_validation.conformance import _remove_candidate

require_compute()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--starts',type=int,default=3)
parser.add_argument('--output',required=True)
args=parser.parse_args()
storage=SharedStorage()
split=read_split('splits/5000run_v1.json')
protocol,digest=read_protocol('protocols/forecast_v1.json')
selected=[(g.chunk,s) for g in select_starts(split,'validation',protocol) for s in g.rollout][:args.starts]
chunks=[];starts=[]
for index,(name,start) in enumerate(selected):
    dumps=tuple(d for d in split.model_dumps('validation') if start-protocol['H_max']+1<=d.index<=start+protocol['max_steps'])
    label=f'profile_{index}'
    chunks.append(Chunk(label,'validation',dumps[0].index,dumps[-1].index,dumps))
    starts.append(Starts(label,(start,),(start,)))
# Only this diagnostic process has shortened windows; frozen files stay untouched.
split=replace(split,segments=tuple(chunks))
runner.select_starts=lambda *_:tuple(starts)
root=storage.exclusive_folder('staging',__import__('getpass').getuser(),'memory-'+uuid.uuid4().hex)
writer=Writer(storage,root)
cache_root=writer.folder('truth-profile')
with storage.new_file(cache_root,'quantities.sqlite3'):pass
cache=TruthCache(cache_root,building=True)
receipt={'diagnostic_only':True,'ok':False,'starts':args.starts,'root':str(root),'job_id':os.environ.get('SLURM_JOB_ID'),'analyses':{},'cache_build':{}}
def measured(label,fn,group):
    record={}
    try:
        with measure(record,label):return fn()
    finally:
        receipt[group].setdefault(label,[]).append(record[label])
        Path(args.output).write_text(json.dumps(receipt,indent=1,sort_keys=True)+'\n')
original=cache.value
cache.value=lambda kind,key,calculate=None:measured(kind,lambda:original(kind,key,calculate),'cache_build')
# Import analysis entry points before installing the diagnostic wrappers.
for name in protocol['tiers']['full']:
    module=importlib.import_module('grmhd_validation.analyses.'+name)
    operation=module.run
    module.run=lambda ctx,operation=operation,name=name:measured(name,lambda:operation(ctx),'analyses')
try:
    runner.execute(runner.persistence_request(),split,protocol,digest,'validation',writer,__import__('torch').device('cpu'),True,tier='full',truth_cache=cache)
    receipt['ok']=True
finally:
    cache.close()
    _remove_candidate(storage,root)
    receipt['artifacts_removed']=True
    Path(args.output).write_text(json.dumps(receipt,indent=1,sort_keys=True)+'\n')

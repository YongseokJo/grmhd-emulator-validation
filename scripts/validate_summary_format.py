"""Owner format correction and independent conformance; appendable across Slurm jobs."""
import os
os.umask(0o007)
import argparse
import json
from pathlib import Path
import sys
import time

from grmhd_validation.slurm import require_compute
require_compute()
from grmhd_validation.artifacts import result_files
from grmhd_validation.summary_correction import rebuild
from grmhd_validation.conformance import conformance
from grmhd_validation.hashing import sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt',type=Path,default=Path('gval-m6-summary-receipt.log'))
parser.add_argument('--operation',action='append',choices=('correct','conformance'))
parser.add_argument('--split-id',action='append',choices=('rc300_reduced_v1','5000run_v1'))
parser.add_argument('--tier',action='append',choices=('quick','full'))
args=parser.parse_args()
receipt=json.loads(args.receipt.read_text()) if args.receipt.exists() else {'protocol_sha256':sha256('protocols/forecast_v1.json'),'steps':[]}
if receipt['protocol_sha256']!=sha256('protocols/forecast_v1.json'):raise RuntimeError('Protocol changed')
for operation in args.operation or ('correct','conformance'):
 for split_id in args.split_id or ('rc300_reduced_v1','5000run_v1'):
  for tier in args.tier or ('quick','full'):
   print(json.dumps({'starting':operation,'split_id':split_id,'tier':tier}),flush=True)
   begin=time.monotonic()
   verdict=rebuild(split_id,tier) if operation=='correct' else conformance(split_id,tier=tier)
   item={'operation':operation,'split_id':split_id,'tier':tier,'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.monotonic()-begin,'verdict':verdict}
   if operation=='conformance' and verdict.get('ok'):
    # Compare independently recomputed metrics against pre-correction hashes.
    from grmhd_validation.conformance import reference_path
    from grmhd_validation.permissions import SharedStorage
    from grmhd_validation.protocol import read_protocol
    from grmhd_validation.environment import platform_id
    protocol,digest=read_protocol('protocols/forecast_v1.json')
    golden=reference_path(SharedStorage(),split_id,protocol,platform_id(),tier,digest)
    record=json.loads((golden/'golden.json').read_text())['summary_format_correction']
    actual={k:v for k,v in result_files(verdict['output_dir']).items() if k.startswith('metrics/')}
    item['metrics_byte_identical_to_old']=actual==record['metrics_before']
    item['old_fingerprint']=record['old_fingerprint'];item['new_fingerprint']=verdict['result_fingerprint']
    if not item['metrics_byte_identical_to_old']:raise RuntimeError('Independently rerun metric bytes changed')
   receipt['steps'].append(item)
   args.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=1)+'\n')
   print(json.dumps(item,sort_keys=True),flush=True)
   if not verdict.get('ok'):raise SystemExit(1)
receipt['ok']=all(any(s['operation']==operation and s['split_id']==split_id and s['tier']==tier and s['verdict'].get('ok') for s in receipt['steps'])
                  for operation in ('correct','conformance') for split_id in ('rc300_reduced_v1','5000run_v1') for tier in ('quick','full'))
args.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=1)+'\n')

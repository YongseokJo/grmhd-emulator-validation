"""Run the owner milestone-4 receipt inside one CPU/one-thread Slurm allocation."""
import os
os.umask(0o007)
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from grmhd_validation.hashing import sha256
from grmhd_validation.slurm import require_compute

require_compute()
if any(os.environ.get(name)!='1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
    raise RuntimeError('Owner reference requires all three thread controls set to 1')
expected={'rc300_reduced_v1':'6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6',
          '5000run_v1':'cd9598fd3f8380f9ebc862db46eb42d070fcfc7f43e532e7bd0f9fc7c51cf648'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split-id',choices=list(expected),help='Validate one dataset and append to the existing receipt')
parser.add_argument('--operation',action='append',choices=('reference-full','reference-quick','conformance'),help='Run selected existing operations in order; repeat to select several')
args=parser.parse_args()
receipt_path=Path('gval-m4-real-receipt.log');protocol_sha=sha256('protocols/forecast_v1.json')
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'protocol_sha256':protocol_sha,'steps':[]}
if receipt['protocol_sha256']!=protocol_sha:raise RuntimeError('Existing receipt has a different protocol hash')
receipt.pop('ok',None)
for split in ([args.split_id] if args.split_id else expected):
    for operation in (args.operation or ('reference-full','reference-quick','conformance')):
        args=['reference','persistence','--split-id',split,'--tier',operation.split('-')[1]] if operation.startswith('reference') else ['conformance','--split-id',split,'--tier','all']
        print(json.dumps({'starting':operation,'split_id':split}),flush=True)
        start=time.monotonic()
        result=subprocess.run([sys.executable,'-m','grmhd_validation',*args],text=True,capture_output=True)
        try:verdict=json.loads(result.stdout.splitlines()[-1])
        except (ValueError,IndexError):verdict={'ok':False,'stdout':result.stdout,'stderr':result.stderr}
        entry={'operation':operation,'split_id':split,'seconds':time.monotonic()-start,'verdict':verdict}
        receipt['steps'].append(entry)
        Path('gval-m4-real-receipt.log').write_text(json.dumps(receipt,indent=1,sort_keys=True)+'\n')
        print(json.dumps(entry,sort_keys=True),flush=True)
        if result.returncode or not verdict.get('ok'):raise SystemExit(1)
        if operation=='reference-quick' and verdict['result_fingerprint']!=expected[split]:
            raise RuntimeError('Quick result fingerprint differs from the milestone-3 reference')
receipt['ok']=all(any(s['split_id']==split and s['operation']==operation and s['verdict'].get('ok') and
                     (operation!='reference-quick' or s['verdict'].get('result_fingerprint')==expected[split])
                     for s in receipt['steps'])
                  for split in expected for operation in ('reference-full','reference-quick','conformance'))
Path('gval-m4-real-receipt.log').write_text(json.dumps(receipt,indent=1,sort_keys=True)+'\n')

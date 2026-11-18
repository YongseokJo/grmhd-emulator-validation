"""Resumable owner-only six-model rc300 validation; never evaluates test data."""
import os
os.umask(0o007)
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from grmhd_validation.slurm import require_compute
require_compute()
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.hashing import sha256
from grmhd_validation.permissions import SharedStorage
from grmhd_validation.repo_state import require_clean

ROOT=Path(__file__).resolve().parents[1]
MODELS=('persistence','tron','vit','eden-claude','eden-gpt','eden-gemini')
SPLIT='rc300_reduced_v1'
INPUTS=ROOT/'examples/owner_rc300_validation'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare',action='store_true')
parser.add_argument('--preflight',action='store_true')
parser.add_argument('--publish',action='store_true')
parser.add_argument('--model',action='append',choices=MODELS)
parser.add_argument('--tier',action='append',choices=('quick','full'))
parser.add_argument('--receipt',type=Path,default=ROOT/'gval-m7-receipt.log')
args=parser.parse_args()
SharedStorage().require_owner()
require_clean({},ROOT)
receipt=json.loads(args.receipt.read_text()) if args.receipt.exists() else {'split_id':SPLIT,'partition':'validation','steps':[]}
receipt.pop('deferred',None)

def save():
    temporary=args.receipt.with_suffix('.tmp')
    temporary.write_bytes(json_bytes(receipt));os.replace(temporary,args.receipt)


def command(key,argv):
    print(json.dumps({'starting':key,'command':argv}),flush=True)
    begin=time.monotonic()
    log=ROOT/('gval-m7-'+key+'.log')
    with log.open('w') as out:
        result=subprocess.run([sys.executable,'-m','grmhd_validation',*argv],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    try:verdict=json.loads(log.read_text().splitlines()[-1])
    except (ValueError,IndexError):verdict={'ok':False,'error':'No final JSON verdict; inspect '+str(log)}
    item={'operation':key,'command':argv,'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.monotonic()-begin,'returncode':result.returncode,'verdict':verdict}
    receipt['steps'].append(item);save();print(json.dumps(item),flush=True)
    if result.returncode or not verdict.get('ok'):raise RuntimeError('Owner check failed: '+key+'; inspect '+str(log))
    return verdict


def done(key):
    return next((s for s in reversed(receipt['steps']) if s['operation']==key and s['verdict'].get('ok') and s['returncode']==0),None)


def write_immutable(path,value):
    data=json_bytes(value)
    if path.exists() and path.read_bytes()!=data:raise RuntimeError('Conflicting dry-run input: '+str(path))
    if not path.exists():path.write_bytes(data)


if args.prepare:
    INPUTS.mkdir(parents=True,exist_ok=True)
    manifest={}
    source=Path('/work/hdd/bgun/kfortune/fall26_outputs')
    commits={'tron':'2667a5fc405a8467fa77614072786eef63c4eb9a','vit':'ae8d0403c0a4b96068e4a369682d270eb90964a2'}
    for model in MODELS:
        request={'model':model,'version':'v1','adapter':model,'checkpoint':None,'options':{},'training_record':None,'notes':'Owner milestone-7 validation-only dry run; all protocol starts/steps required.'}
        path=INPUTS/(model+'.json')
        if model in commits:
            run=source/(model+'_rc300_v1')/'seed0';checkpoint=run/'best.pth'
            digest=sha256(checkpoint)
            if model=='tron' and digest!='a81d05b44a86752e45c14ad91f6bf827f5213632b36a29d19ce47a16dbc188eb':raise RuntimeError('Pinned TRON checkpoint changed')
            config=json.loads((run/'config.json').read_text())
            with (run/'train.log').open() as log:first=log.readline()
            if 'commit '+commits[model][:7] not in first:raise RuntimeError('Training log commit does not match declaration')
            config['provenance']={'training_code_commit':commits[model],'normalization_split':'fixed_transform_no_fit',
                'source_split':'kevin_fall26_80_10_10','training_dump_ranges':[[3000,10707]],'selection_dump_ranges':[[10708,11671]],
                'known_prior_data_exposure':'Historical checkpoint trained/selected on Kevin 80/10/10 data, overlapping the frozen benchmark. Existing best_blind evaluation artifacts; no blind claim.',
                'prior_data_exposure':['kevin_fall26_80_10_10 training/selection','historical best_blind evaluation artifacts'],
                'original_config_sha256':sha256(run/'config.json'),'original_config_path':str(run/'config.json')}
            declaration=INPUTS/(model+'_training_config.json');write_immutable(declaration,config)
            request['checkpoint']=str(checkpoint)
            if model=='vit':request['options']={'config':{k:v for k,v in config.items() if k!='provenance'}}
            # The request may already link its immutable record on a resumed job.
            if path.exists():request['training_record']=json.loads(path.read_text())['training_record']
            write_immutable(path,request)
            record=command('record-training-'+model,['record-training','--request',str(path),'--split-id',SPLIT,'--config',str(declaration),'--log',str(run/'train.log')])
            request['training_record']=record['training_record'];path.write_bytes(json_bytes(request))
            manifest[model]={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'source_config_sha256':sha256(run/'config.json'),'training_log_sha256':sha256(run/'train.log'),'request_sha256':sha256(path)}
        else:
            if model.startswith('eden-'):request['notes']+=' Protocol A, credited Eden start-11672 method; known pilot exposure, not a blind development claim.'
            write_immutable(path,request);manifest[model]={'checkpoint':None,'checkpoint_sha256':None,'request_sha256':sha256(path)}
    write_immutable(INPUTS/'inputs.json',manifest)
    receipt['inputs']=manifest;save()

if args.preflight:
    command('doctor',['doctor'])
    for tier in ('quick','full'):command('conformance-'+tier,['conformance','--split-id',SPLIT,'--tier',tier])

end=os.environ.get('SLURM_JOB_END_TIME')
if end and end.isdigit():deadline=float(end)
else:
    info=subprocess.run(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],capture_output=True,text=True,check=True).stdout
    match=re.search(r'\bEndTime=(\S+)',info)
    deadline=datetime.fromisoformat(match[1]).timestamp() if match and match[1]!='Unknown' else time.time()+90*60
manifest=json.loads((INPUTS/'inputs.json').read_text()) if (INPUTS/'inputs.json').exists() else {}
stop=False
for model in args.model or ():
    request_path=INPUTS/(model+'.json');request=json.loads(request_path.read_text())
    if sha256(request_path)!=manifest[model]['request_sha256']:raise RuntimeError('Request changed after preparation')
    if request['checkpoint'] and sha256(request['checkpoint'])!=manifest[model]['checkpoint_sha256']:raise RuntimeError('Checkpoint changed after preparation')
    device='cuda' if model in ('tron','vit') else 'cpu'
    if not done('check-adapter-'+model):
        argv=['check-adapter',model,'--split-id',SPLIT,'--device',device]
        if request['checkpoint']:argv+=['--checkpoint',request['checkpoint']]
        if request['options']:argv+=['--options',json.dumps(request['options'])]
        command('check-adapter-'+model,argv)
    for tier in args.tier or ('quick','full'):
        key='evaluate-'+model+'-'+tier
        previous=done(key)
        if previous:
            command('verify-'+model+'-'+tier,['verify',previous['verdict']['run_dir']]);continue
        if deadline-time.time()<(60 if tier=='full' else 15)*60:
            receipt['deferred']={'model':model,'tier':tier,'reason':'Use a fresh interactive allocation; preserve full coverage'};save();stop=True;break
        verdict=command(key,['evaluate','--request',str(request_path),'--split-id',SPLIT,'--split','validation','--tier',tier,'--device',device])
        metadata=json.loads((Path(verdict['run_dir'])/'run.json').read_text())
        if metadata['repo_modified'] or not metadata['environment_locked'] or metadata['checkpoint_sha256']!=manifest[model]['checkpoint_sha256']:
            raise RuntimeError('Published run identity/state differs from approved inputs')
        command('verify-'+model+'-'+tier,['verify',verdict['run_dir']])
    if stop:break
receipt['all_models_complete']=all(done('evaluate-'+model+'-'+tier) and done('verify-'+model+'-'+tier) for model in MODELS for tier in ('quick','full'))
if args.publish:
    if not receipt['all_models_complete']:raise RuntimeError('Complete all six quick/full validation runs before final report')
    for split in (SPLIT,'5000run_v1'):
        command('report-'+split,['report','--split-id',split,'--export-results',str(ROOT)])
        command('replay-'+split,['report','--from-results','--split-id',split,'--check'])
save()
print(json.dumps({'ok':True,'all_models_complete':bool(receipt['all_models_complete']),'deferred':receipt.get('deferred')}),flush=True)

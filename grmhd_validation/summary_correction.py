"""Owner-authorized format-only golden correction; never mutates model runs."""
import json
import os
from pathlib import Path
import shutil
import uuid

from .artifacts import json_bytes,result_files,fingerprint
from .hashing import sha256
from .permissions import SharedStorage
from .protocol import read_protocol
from .slurm import require_compute
from .splits import read_split
from .summary import from_metrics
from .validation import schema_validate


def rebuild(split_id,tier,storage=None,repo=None):
    require_compute()
    from .conformance import reference_path,load_golden
    from .environment import platform_id
    from .runner import REPO,git_commit
    storage=storage or SharedStorage();storage.require_owner();repo=Path(repo or REPO)
    protocol,protocol_sha=read_protocol(repo/'protocols/forecast_v1.json')
    split=read_split(repo/'splits'/(split_id+'.json'))
    target=reference_path(storage,split_id,protocol,platform_id(),tier,protocol_sha)
    marker=target/'.summary-format-v2.json'
    if not marker.exists():
        before=load_golden(target)
        info=json.loads((target/'run.json').read_text())
        if info.get('model')!='persistence' or info.get('purpose')!='reference' or info['protocol_sha256']!=protocol_sha:
            raise ValueError('Summary correction requires the current owner persistence reference')
        if json.loads((target/'summary.json').read_text()).get('schema_version')==2:
            from .summary import validate
            validate(json.loads((target/'summary.json').read_text()))
            return {'ok':True,'reused':True,'reference_dir':str(target),'result_fingerprint':before['result_fingerprint'],
                    'next':'gval conformance --split-id '+split_id+' --tier '+tier}
        backup=storage.folder('reports',split_id,'summary-format-v2',tier,before['result_fingerprint'])
        for name in ('summary.json','run.json','golden.json'):
            destination=backup/name
            if destination.exists():
                if sha256(destination)!=sha256(target/name):raise ValueError('Conflicting summary correction backup')
            else:
                with storage.new_file(backup,name) as out:
                    with (target/name).open('rb') as source:shutil.copyfileobj(source,out)
        with storage.new_file(target,marker.name) as out:
            out.write(json_bytes({'backup':str(backup),'old_fingerprint':before['result_fingerprint']}))
    else:
        record=json.loads(marker.read_text());backup=Path(record['backup'])
        if not backup.is_relative_to(storage.root/'reports'/split_id/'summary-format-v2'):
            raise ValueError('Correction backup is outside its prescribed owner area')
        before=json.loads((backup/'golden.json').read_text());info=json.loads((backup/'run.json').read_text())
    files=result_files(target)
    if {k:v for k,v in files.items() if k!='summary.json'}!={k:v for k,v in before['files'].items() if k!='summary.json'}:
        raise ValueError('Metric/table bytes changed; format correction refuses to proceed')
    if sha256(backup/'summary.json')!=before['files']['summary.json'] or fingerprint(before['files'])!=before['result_fingerprint']:
        raise ValueError('Original summary backup does not match the old golden')
    summary=from_metrics(target,protocol,tier,split.dt_dump)
    content=json_bytes(summary);import hashlib
    corrected={**before['files'],'summary.json':hashlib.sha256(content).hexdigest()}
    new_fp=fingerprint(corrected)
    receipt={'owner_decision':'2026-11-17 result-format correction; protocol unchanged',
             'protocol_sha256':protocol_sha,'old_fingerprint':before['result_fingerprint'],'new_fingerprint':new_fp,
             'old_summary_sha256':before['files']['summary.json'],'new_summary_sha256':corrected['summary.json'],
             'summary_bytes':len(content),'metrics_before':{k:v for k,v in before['files'].items() if k.startswith('metrics/')},
             'metrics_after':{k:v for k,v in corrected.items() if k.startswith('metrics/')},
             'tables_unchanged':True,'backup':str(backup),'writer_repo_commit':git_commit(repo)}
    info.update(result_fingerprint=new_fp,result_content_sha256=corrected,summary_format_version=2,summary_format_correction=receipt)
    golden={**before,'result_fingerprint':new_fp,'files':corrected,'summary_format_version':2,'summary_format_correction':receipt}
    schema_validate(info,'run.schema.json');schema_validate(golden,'golden.schema.json')
    for name,data in [('summary.json',content),('run.json',json_bytes(info)),('golden.json',json_bytes(golden))]:
        temporary='.'+name+'.'+uuid.uuid4().hex
        with storage.new_file(target,temporary) as out:out.write(data)
        os.replace(target/temporary,target/name)
    marker.unlink()
    if load_golden(target)['result_fingerprint']!=new_fp:raise RuntimeError('Corrected golden validation failed')
    with storage.new_file(backup,'correction.json') as out:out.write(json_bytes(receipt))
    return {'ok':True,'reused':False,'reference_dir':str(target),'next':'gval conformance --split-id '+split_id+' --tier '+tier,**receipt}

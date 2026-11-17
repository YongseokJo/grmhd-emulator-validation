"""Checkpoint provenance is an operator declaration, not proof of data isolation."""
import json
from pathlib import Path
from .hashing import sha256
from .validation import schema_validate


def training_provenance(request, split):
    if request['training_record'] is None:
        return {'trained_on_this_split':'unknown','record_sha256':None,'reason':'No training record supplied'}
    path=Path(request['training_record']);record=json.loads(path.read_text())
    schema_validate(record,'training-record.schema.json')
    if request['checkpoint'] is None or record['checkpoint_sha256']!=sha256(request['checkpoint']):
        raise ValueError('Training record checkpoint identity mismatch')
    if record['model']!=request['model'] or record['version']!=request['version']:
        raise ValueError('Training record model/version mismatch')
    if record['split_id']!=split.split_id or record['split_file_sha256']!=split.sha256:
        raise ValueError('Training record split identity mismatch')
    computed=declaration(record['config'],split)
    if computed!=record['computed']:raise ValueError('Training record overlap calculations do not match declaration')
    return {**computed, 'record_sha256':sha256(path),'record':record,'evidence':'operator declaration'}


def merge_ranges(ranges):
    result=[]
    for pair in ranges:
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(v) is not int or v<0 for v in pair) or pair[1]<pair[0]):
            raise ValueError('Dump ranges must be inclusive nonnegative [first,last] pairs')
    for first,last in sorted(ranges):
        if result and first<=result[-1][1]+1:result[-1][1]=max(result[-1][1],last)
        else:result.append([first,last])
    return result


def overlap(ranges,first,last):
    return sum(max(0,min(end,last)-max(begin,first)+1) for begin,end in ranges)


def declaration(config,split):
    value=config.get('provenance')
    if not isinstance(value,dict):raise ValueError('Training config requires a provenance object')
    required={'training_code_commit','normalization_split','source_split','known_prior_data_exposure','prior_data_exposure'}
    if required-set(value):raise ValueError('Missing provenance fields: '+str(sorted(required-set(value))))
    if not all(isinstance(value[key],str) for key in required-{'prior_data_exposure'}):raise ValueError('Provenance labels must be strings')
    import re
    if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',value['training_code_commit']):raise ValueError('Training code commit must be a full Git hash')
    if not isinstance(value['prior_data_exposure'],list) or not all(isinstance(v,str) for v in value['prior_data_exposure']):raise ValueError('prior_data_exposure must be a string list')
    chunks={chunk.name:chunk for chunk in split.segments}
    counts={};ranges={};outside={};totals={}
    for role in ('training','selection'):
        names=value.get(role+'_chunks',[])
        if not isinstance(names,list) or any(name not in chunks for name in names):raise ValueError('Unknown '+role+' chunk name')
        raw=value.get(role+'_dump_ranges',[])
        if not isinstance(raw,list):raise ValueError(role+'_dump_ranges must be a list')
        ranges[role]=merge_ranges(raw+[[chunks[name].first,chunks[name].last] for name in names])
        counts[role]={name:overlap(ranges[role],chunk.first,chunk.last) for name,chunk in chunks.items()}
        totals[role]={partition:sum(counts[role][chunk.name] for chunk in split.chunks(partition)) for partition in ('train','validation','test','excluded')}
        outside[role]=sum(end-begin+1 for begin,end in ranges[role])-sum(counts[role].values())
    reasons=[]
    for role,allowed in [('training','train'),('selection','validation')]:
        if any(count for partition,count in totals[role].items() if partition!=allowed):reasons.append(role+'_outside_'+allowed)
        if outside[role]:reasons.append(role+'_outside_frozen_split')
    if value['normalization_split']!='train':reasons.append('normalization_not_train')
    complete=bool(ranges['training']) and bool(ranges['selection'])
    status='no' if reasons else ('yes' if complete else 'unknown')
    exposed=totals['training']['validation']+totals['training']['test']
    label='no, overlaps validation/test ('+str(exposed)+' dumps)' if exposed else status
    return {'ranges':ranges,'chunk_counts':counts,'partition_counts':totals,'outside_split_counts':outside,
            'trained_on_this_split':status,'label':label,'reasons':reasons,
            'normalization_flagged':value['normalization_split']!='train'}


def record_training(request_path,split_id,config_path,log_path=None,storage=None,repo=None):
    from .permissions import SharedStorage
    from .runner import REPO,read_request
    from .splits import read_split
    from .artifacts import json_bytes
    from .slurm import require_compute
    require_compute();storage=storage or SharedStorage();repo=Path(repo or REPO)
    request=read_request(request_path)
    if request['checkpoint'] is None:raise ValueError('record-training requires a trained checkpoint')
    split=read_split(repo/'splits'/(split_id+'.json'))
    if sha256(storage.root/'splits'/(split_id+'.json'))!=split.sha256:raise ValueError('Shared frozen split mismatch')
    raw_config=Path(config_path).read_bytes()
    config=json.loads(raw_config);computed=declaration(config,split)
    import hashlib
    record={'schema_version':1,'evidence':'operator declaration','verified':False,
            'model':request['model'],'version':request['version'],'split_id':split_id,'split_file_sha256':split.sha256,
            'checkpoint_sha256':sha256(request['checkpoint']),'config_sha256':hashlib.sha256(raw_config).hexdigest(),
            'log_sha256':sha256(log_path) if log_path else None,'config':config,'computed':computed}
    schema_validate(record,'training-record.schema.json')
    parent=storage.folder('training_records',request['model']);name=record['checkpoint_sha256'][:8]+'.json';target=parent/name
    data=json_bytes(record)
    if target.exists():
        if target.is_symlink() or target.read_bytes()!=data:raise FileExistsError('Conflicting checkpoint training record; existing record is immutable')
        reused=True
    else:
        try:
            with storage.new_file(parent,name) as stream:stream.write(data)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes()!=data:raise FileExistsError('Conflicting concurrent training record')
        else:storage.make_readonly(target)
        reused=False
    return {'ok':True,'next':'Set request.training_record to '+str(target)+'; then gval check-adapter',
            'training_record':str(target),'sha256':sha256(target),'reused':reused,'evidence':'operator declaration',**computed}

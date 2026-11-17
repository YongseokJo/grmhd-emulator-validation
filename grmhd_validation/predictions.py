"""Host-owned task, isolated packets, immutable chains and prediction imports.

Packet design adapted from Eden Growney's rc300 primary_ar_onestep contract,
eden/rc300-frontier-15nov@f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a,
eden_rc300_pilot/benchmark_pilot/rc300_packets/ and tools/advance_rc300_ar.py.
Changes: frozen benchmark coverage, physical float32, h-frame histories, strict
hash chains, explicit unknowns, no scoring or future truth in exported packets.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

import numpy as np

from .artifacts import json_bytes
from .hashing import sha256 as _sha256
from .permissions import SHARED_ROOT, SharedStorage
from .protocol import read_protocol
from .splits import CHANNELS, read_split, relative_path
from .windows import select_starts
from .validation import schema_validate

REPO = Path(__file__).resolve().parents[1]
SOURCE = 'f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a'
IDENTITY = ('task_id', 'episode_id', 'prompt_id', 'kind', 'chunk', 'start', 'active_step')
GENERATION_FIELDS = ('provider', 'model', 'model_version', 'client', 'client_version',
                     'session_id', 'request_id', 'response_id', 'parameters', 'seed',
                     'start_time', 'end_time', 'elapsed_seconds', 'allowed_tools',
                     'actual_tools', 'network_access', 'transcript_sha256', 'tool_log_sha256')
RUN_FIELDS = ('source_repository','source_commit','checkpoint_sha256','training_record','known_prior_data_exposure',
              'host_identity','scorer_identity','environment_lock_sha256','implementation_commit')
ATTESTATIONS = ('workspace_isolated', 'history_only', 'no_future_truth', 'no_evaluator',
                'no_other_starts', 'no_scorer_feedback', 'session_policy_followed',
                'frozen_autoregressive_chain', 'source_prompt_frozen', 'complete_without_selection')


def sha256(path):
    # Prediction imports must not charge hundreds of cubes to the Slurm file cache.
    try:return _sha256(path)
    finally:
        from .memory import discard_file_cache
        discard_file_cache(path)


def now():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def write(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    if not frozen:
        with path.open('xb') as stream:stream.write(json_bytes(value))
        return path
    # No accepted JSON file is visible partially written, even after a kill.
    temporary=path.with_name('.pending-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(json_bytes(value));stream.flush();os.fsync(stream.fileno())
        temporary.chmod(0o440)
        os.link(temporary,path)  # exclusive publication, never overwrites
    finally:
        if temporary.exists():temporary.unlink()
    return path


def replace(path, value):
    path = Path(path);temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    write(temporary, value);os.replace(temporary, path)


def workspace(path):
    path = Path(path).expanduser().absolute()
    if path.resolve().is_relative_to(SHARED_ROOT.resolve()):
        raise ValueError('Task and packet workspaces must be outside shared validation storage')
    # Reject symlink ancestors, including an existing output root.
    for parent in (path, *path.parents):
        if parent.is_symlink():raise ValueError('Workspace path contains a symlink')
    return path


def confined(root, name):
    name = relative_path(name);root = Path(root)
    candidate = root/name
    for part in (candidate, *candidate.parents):
        if part == root.parent:break
        if part.is_symlink():raise ValueError('Symlink in prediction workspace: '+name)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Prediction path escapes bundle')
    return candidate


def array(path, shape):
    if Path(path).is_symlink():raise ValueError('Prediction arrays cannot be symlinks')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if value.dtype != np.float32 or tuple(value.shape) != tuple(shape) or not np.isfinite(value).all():
            raise ValueError('Prediction must be finite float32 with exact physical shape')
        return np.array(value, copy=True, order='C')
    finally:
        if isinstance(value,np.memmap):value._mmap.close()
        from .memory import discard_file_cache
        discard_file_cache(path)


def save_array(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False);stream.flush();os.fdatasync(stream.fileno())


def tensor_hash(value):
    value = np.ascontiguousarray(value)
    header = json_bytes({'dtype':str(value.dtype), 'shape':list(value.shape)})
    digest = hashlib.sha256(header);digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


@contextmanager
def locked(task):
    # A file lock serializes create/advance/collect. It never locks shared data.
    with (Path(task)/'.host.lock').open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def prompt(repo, prompt_id, protocol):
    if '/' in prompt_id or prompt_id in {'.','..'}:raise ValueError('Invalid prompt_id')
    directory = Path(repo)/'prompts'/prompt_id
    manifest = read(directory/'manifest.json')
    schema_validate(manifest, 'prompt.schema.json')
    if manifest['prompt_id'] != prompt_id or manifest['protocol'] != protocol:
        raise ValueError('Prompt identity/protocol mismatch')
    if sha256(directory/'template.md') != manifest['template_sha256']:
        raise ValueError('Versioned prompt template changed')
    return manifest, (directory/'template.md').read_text()


def task_create(request_path, split_id, partition, protocol, prompt_id, output,
                data_root=None, storage=None, repo=REPO):
    from .adapter import adapter_source, validate_declaration
    from .runner import read_request
    storage = storage or SharedStorage()
    request = read_request(request_path)
    config, config_sha = read_protocol(Path(repo)/'protocols/forecast_v1.json')
    split = read_split(Path(repo)/'splits'/(split_id+'.json'), data_root)
    if sha256(storage.root/'splits'/(split_id+'.json')) != split.sha256:
        raise ValueError('Shared split differs from repository')
    if partition not in {'validation','test'}:raise ValueError('Tasks use validation or test only')
    if protocol == 'A':
        cls, _ = adapter_source(request['adapter']);validate_declaration(cls, config)
        history = cls.history_length
    elif protocol == 'B':
        history = request['options'].get('history_length')
        if type(history) is not int or not 1 <= history <= config['H_max']:
            raise ValueError('B request options.history_length must lie in 1..H_max')
    else:raise ValueError('Prediction protocol must be A or B')
    manifest, template = prompt(repo, prompt_id, protocol)
    starts = select_starts(split, partition, config)
    root = workspace(output);root.mkdir(parents=True, mode=0o770)
    identity = uuid.uuid4().hex
    task = {'schema_version':1, 'task_id':identity,
            'request_sha256':sha256(request_path), 'split_id':split_id,
            'split_file_sha256':split.sha256, 'protocol_sha256':config_sha,
            'geometry_sha256':split.geometry_sha256, 'partition':partition,
            'protocol':protocol, 'prompt_id':prompt_id,
            'prompt_manifest_sha256':sha256(Path(repo)/'prompts'/prompt_id/'manifest.json'),
            'template_sha256':manifest['template_sha256'], 'session_policy':manifest['session_policy'],
            'history_length':history, 'shape':list(split.shape), 'dtype':'float32',
            'channels':list(CHANNELS), 'seed':config['seed'], 'max_steps':config['max_steps'],
            'starts':[{ 'chunk':group.chunk, 'rollout':list(group.rollout), 'single_step':list(group.every)} for group in starts],
            'required_targets':[{ 'kind':kind, 'chunk':group.chunk, 'start':int(start),
                                  'targets':list(range(start+1,start+count+1))}
                                 for group in starts for kind,ids,count in
                                 [('rollout',group.rollout,config['max_steps']),('single_step',group.every,1)] for start in ids],
            'completeness':{'rollout':'all starts and all steps', 'single_step':'every eligible start'}}
    schema_validate(task, 'task.schema.json')
    write(root/'task.json', task, True)
    write(root/'host.json', {'request_path':str(Path(request_path).resolve()), 'split_file':str(split.path) if hasattr(split,'path') else str(Path(repo)/'splits'/(split_id+'.json')),
                            'data_root':str(split.data_root), 'shared_root':str(storage.root), 'repo':str(Path(repo).resolve()),
                            'task_sha256':sha256(root/'task.json')}, True)
    write(root/'provenance.json', {**{key:None for key in GENERATION_FIELDS}, 'unknown_reasons':{key:'Not supplied by host' for key in (*GENERATION_FIELDS,*RUN_FIELDS)},
                                 'source_repository':None, 'source_commit':None, 'checkpoint_sha256':None,
                                 'training_record':request['training_record'], 'known_prior_data_exposure':None,
                                 'host_identity':None, 'scorer_identity':None, 'host':socket.gethostname(),
                                 'job_id':os.environ.get('SLURM_JOB_ID'), 'environment_lock_sha256':None,
                                 'implementation_commit':None, 'start_time':now(), 'end_time':None})
    write(root/'host_attestation.json', {'operator':None, 'host':socket.gethostname(), 'job_id':os.environ.get('SLURM_JOB_ID'),
                                        'implementation_commit':None, 'attested_at':None, 'isolation_mechanism':None,
                                        'evidence_sha256':[], 'exceptions':[], 'claims':{key:False for key in ATTESTATIONS}})
    return {'ok':True, 'next':'gval packet create --task '+str(root)+' --kind rollout --start <S> --output <packet>',
            'task_dir':str(root), 'task_sha256':sha256(root/'task.json'), 'rollout_entries':sum(len(g.rollout) for g in starts)*config['max_steps']}


def load_task(root):
    root = workspace(root);task = read(root/'task.json');host = read(root/'host.json')
    schema_validate(task, 'task.schema.json')
    if sha256(root/'task.json') != host['task_sha256']:raise ValueError('Immutable task changed')
    if sha256(host['request_path']) != task['request_sha256']:raise ValueError('Task request changed')
    config, digest = read_protocol(Path(host['repo'])/'protocols/forecast_v1.json')
    split = read_split(host['split_file'], host['data_root'])
    if digest != task['protocol_sha256'] or split.sha256 != task['split_file_sha256'] or split.geometry_sha256 != task['geometry_sha256']:
        raise ValueError('Task configuration identity changed')
    prompt(host['repo'], task['prompt_id'], task['protocol'])
    if sha256(Path(host['repo'])/'prompts'/task['prompt_id']/'manifest.json') != task['prompt_manifest_sha256']:
        raise ValueError('Task prompt manifest changed')
    return root, task, host, split


def required(task):
    for group in task['starts']:
        for kind in ('rollout','single_step'):
            for start in group[kind]:
                for step in range(1, (task['max_steps'] if kind=='rollout' else 1)+1):
                    yield kind, group['chunk'], start, step


def relative(kind, chunk, start, step):
    return f'{kind}/{chunk}/start_{start}/prediction_step_{step:04d}.npy'


def episode_root(root, kind, chunk, start):
    return root/'episodes'/kind/chunk/('start_'+str(start))


def initial_history(task, host, split, start):
    from .data import FrameReader
    reader = FrameReader(split, task['partition'], Path(host['shared_root'])/'checksums'/(split.split_id+'.sha256'))
    # Always verify geometry before exporting its bytes too.
    from .checksums import verify_file
    verify_file(split.geometry_path, split.geometry_file, reader.hashes)
    if sha256(split.geometry_path) != task['geometry_sha256']:raise ValueError('Geometry changed')
    return reader.history(start, task['history_length'], 'cpu')[0].numpy()


def frozen_chain(root, task, host, split, kind, chunk, start, until=None):
    history = initial_history(task, host, split, start)
    folder = episode_root(root,kind,chunk,start);previous = None
    limit = task['max_steps'] if kind=='rollout' else 1
    if until is not None:limit = until
    records = []
    for step in range(1,limit+1):
        record_path = folder/'steps'/f'step_{step:04d}.json'
        if not record_path.exists():break
        record = read(record_path);schema_validate(record, 'prediction-step.schema.json')
        expected = {'task_id':task['task_id'], 'task_sha256':sha256(root/'task.json'),
                    'kind':kind, 'chunk':chunk, 'start':start, 'step':step, 'target':start+step,
                    'input_history_sha256':tensor_hash(history), 'previous_output_sha256':previous,
                    'geometry_sha256':task['geometry_sha256']}
        if any(record.get(k) != v for k,v in expected.items()):raise ValueError('Broken frozen input/output chain')
        path = confined(folder, record['prediction_file'])
        if sha256(path) != record['prediction_sha256']:raise ValueError('Frozen output changed')
        code = confined(folder,record['code_directory'])
        actual_code = {p.relative_to(code).as_posix():sha256(p) for p in sorted(code.rglob('*')) if p.is_file()} if code.exists() else {}
        if actual_code != record['code_sha256']:raise ValueError('Frozen method source changed')
        output = array(path,task['shape'])
        history = np.concatenate((history[1:],output[None]),axis=0)
        previous = record['prediction_sha256'];records.append(record)
    return history, records


def render_packet(root, task, host, split, packet, kind, chunk, start):
    history, records = frozen_chain(root,task,host,split,kind,chunk,start)
    step = len(records)+1;count = task['max_steps'] if kind=='rollout' else 1
    if step > count:return
    manifest, template = prompt(host['repo'],task['prompt_id'],task['protocol'])
    rendered = template.format(shape=tuple(task['shape']), dt_dump=split.dt_dump,
                               history_length=task['history_length'], protocol=task['protocol'])
    inputs = packet/'inputs';inputs.mkdir(exist_ok=True)
    for name,value in [('current_state.npy',history[-1]), *([('history.npy',history)] if len(history)>1 else [])]:
        temporary = inputs/('.'+name+'.'+uuid.uuid4().hex);save_array(temporary,value)
        os.replace(temporary,inputs/name)
    if not (inputs/'geometry.npz').exists():
        with (inputs/'geometry.npz').open('xb') as stream:
            with split.geometry_path.open('rb') as source:shutil.copyfileobj(source,stream)
    episode_id = hashlib.sha256(f"{task['task_id']}:{kind}:{chunk}:{start}".encode()).hexdigest()
    episode = {'schema_version':1, 'task_id':task['task_id'], 'episode_id':episode_id,
               'prompt_id':task['prompt_id'], 'kind':kind, 'chunk':chunk, 'start':start,
               'active_step':step, 'required_steps':count, 'state_from_step':step-1,
               'required_prediction_files':[f'prediction_step_{step:04d}.npy'],
               'input_hashes':{p.name:sha256(p) for p in sorted(inputs.iterdir())},
               'input_history_sha256':tensor_hash(history), 'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
    for name,text in [('README.md','Forecasting role: use only this packet and follow prompt.md. Never invoke host or scorer commands.\n'),
                      ('prompt.md',rendered),('format.md',f'Physical float32 {task["shape"]}; channels {list(CHANNELS)}. Cadence {split.dt_dump} R_g/c.\n')]:
        path = packet/name
        if path.exists():
            if path.read_text()!=text:raise ValueError('Packet instruction changed')
        else:path.write_text(text)
    schema = {'schema_version':1, 'task_id':task['task_id'], 'episode_id':episode_id,
              'prompt_id':task['prompt_id'], 'kind':kind, 'chunk':chunk, 'start':start,
              'active_step':step, 'prediction_file':f'prediction_step_{step:04d}.npy'}
    replace(packet/'episode.json',episode)
    if not (packet/'output_schema.json').exists():
        write(packet/'output_schema.json',read(Path(host['repo'])/'schemas/submission.schema.json'))
    submission = packet/'submission';submission.mkdir(exist_ok=True)
    replace(submission/'submission.json',schema)
    generation = {key:None for key in GENERATION_FIELDS}
    generation.update(unknown_reasons={key:'Host must supply actual value or reason' for key in GENERATION_FIELDS})
    # Generation metadata stays private to the host, never in the agent packet.
    folder = episode_root(root,kind,chunk,start)
    generation_path=folder/'generation.json'
    if not generation_path.exists():write(generation_path,generation)
    else:replace(generation_path,generation)


def packet_create(task_dir, kind, start, output):
    from .slurm import require_compute
    require_compute()
    with locked(task_dir):
        root,task,host,split = load_task(task_dir)
        matches = [g for g in task['starts'] if start in g.get(kind,[])]
        if len(matches)!=1:raise ValueError('Start is not in the required task coverage')
        chunk = matches[0]['chunk'];folder = episode_root(root,kind,chunk,start)
        folder.mkdir(parents=True, exist_ok=True)
        packet = workspace(output)
        # A packet is a sibling/external workspace, never inside the private host tree.
        if packet.is_relative_to(root) or root.is_relative_to(packet):raise ValueError('Agent packet must not expose the private task tree')
        registration=folder/'packet.json'
        if packet.exists() and not registration.exists():raise FileExistsError('Packet output already exists and is not registered')
        packet.mkdir(parents=True,mode=0o770,exist_ok=True)
        if registration.exists():
            if read(registration)['path']!=str(packet):raise ValueError('Episode already belongs to another packet workspace')
        else:write(registration,{'path':str(packet)},True)
        render_packet(root,task,host,split,packet,kind,chunk,start)
    return {'ok':True,'next':'Forecasting agent writes the active submission; host fills generation.json, then gval packet advance', 'packet_dir':str(packet)}


def copy_code(source, destination):
    if source.is_symlink():raise ValueError('Source code cannot be a symlink')
    if not source.exists():return {}
    result = {}
    for item in sorted(source.rglob('*')):
        if item.is_symlink():raise ValueError('Source code cannot contain symlinks')
        if item.is_file():
            name = item.relative_to(source).as_posix();target = destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(item.read_bytes())
            target.chmod(0o440);result[name]=sha256(target)
    return result


def validate_unknowns(value):
    for key in GENERATION_FIELDS:
        if key not in value:raise ValueError('Missing generation provenance field: '+key)
        if value[key] is None and not value.get('unknown_reasons',{}).get(key):
            raise ValueError('Unknown generation field needs a reason: '+key)
    if value['elapsed_seconds'] is not None and (type(value['elapsed_seconds']) not in (int,float) or value['elapsed_seconds']<0):
        raise ValueError('Invalid elapsed generation time')


def packet_advance(task_dir, packet_dir):
    from .slurm import require_compute
    require_compute()
    with locked(task_dir):
        root,task,host,split = load_task(task_dir);packet = workspace(packet_dir)
        try:
            episode = read(packet/'episode.json')
            kind,chunk,start = (episode[k] for k in ('kind','chunk','start'))
            if (kind,chunk,start,1) not in set(required(task)):raise ValueError('Unknown episode')
            folder = episode_root(root,kind,chunk,start)
            if read(folder/'packet.json')['path'] != str(packet):raise ValueError('Packet is not registered to this episode')
            history,records = frozen_chain(root,task,host,split,kind,chunk,start)
            step = len(records)+1;limit = task['max_steps'] if kind=='rollout' else 1
            if step>limit:raise ValueError('Episode is already complete')
            expected_episode=hashlib.sha256(f"{task['task_id']}:{kind}:{chunk}:{start}".encode()).hexdigest()
            if episode['episode_id']!=expected_episode or episode['prompt_id']!=task['prompt_id']:
                raise ValueError('Episode identity changed')
            if episode['task_id'] != task['task_id'] or episode['active_step'] != step or episode['input_history_sha256'] != tensor_hash(history):
                raise ValueError('Packet task/active-step/history identity mismatch')
            for name,digest in episode['input_hashes'].items():
                if sha256(confined(packet/'inputs',name)) != digest:raise ValueError('Packet input changed')
            current = array(packet/'inputs/current_state.npy',task['shape'])
            if not np.array_equal(current,history[-1]):raise ValueError('Packet current state changed')
            if len(history)>1 and not np.array_equal(array(packet/'inputs/history.npy',history.shape),history):raise ValueError('Packet history changed')
            if sha256(packet/'inputs/geometry.npz') != task['geometry_sha256']:raise ValueError('Packet geometry changed')
            _,template = prompt(host['repo'],task['prompt_id'],task['protocol'])
            expected = template.format(shape=tuple(task['shape']),dt_dump=split.dt_dump,history_length=task['history_length'],protocol=task['protocol'])
            if (packet/'prompt.md').read_text()!=expected:raise ValueError('Rendered prompt changed')
            submission = read(confined(packet,'submission/submission.json'));schema_validate(submission,'submission.schema.json')
            if any(submission.get(k)!=episode[k] for k in IDENTITY):raise ValueError('Submission identity mismatch')
            name = f'prediction_step_{step:04d}.npy'
            if submission['prediction_file']!=name:raise ValueError('Only the active prediction is accepted')
            path = confined(packet/'submission',name);before = sha256(path)
            output = array(path,task['shape'])
            generation = read(folder/'generation.json');validate_unknowns(generation)
            confined(packet,'submission/code')
            code_relative=f'code/step_{step:04d}'
            code_path=folder/code_relative
            supplied={p.relative_to(packet/'submission/code').as_posix():sha256(p) for p in sorted((packet/'submission/code').rglob('*')) if p.is_file()}
            if any(p.is_symlink() for p in (packet/'submission/code').rglob('*')):
                raise ValueError('Source code cannot contain symlinks')
            if task['protocol']=='A' and (not supplied or (records and supplied!=records[0]['code_sha256'])):
                raise ValueError('Protocol A requires one frozen reproducible method')
            if code_path.exists():
                hashes={p.relative_to(code_path).as_posix():sha256(p) for p in sorted(code_path.rglob('*')) if p.is_file()}
                if hashes!=supplied:raise ValueError('Interrupted step code conflicts with submission')
            else:hashes=copy_code(packet/'submission/code',code_path)
            target = folder/name
            if target.exists():
                if target.is_symlink() or sha256(target)!=before:raise ValueError('Interrupted frozen output conflicts with submission')
            else:
                with target.open('xb') as stream:
                    with path.open('rb') as source:
                        shutil.copyfileobj(source,stream);os.fsync(source.fileno())
                    stream.flush();os.fdatasync(stream.fileno())
                target.chmod(0o440)
            if sha256(target)!=before or sha256(path)!=before:raise ValueError('Submission changed during freezing')
            record = {'schema_version':1, 'task_id':task['task_id'], 'task_sha256':sha256(root/'task.json'),
                      'episode_id':episode['episode_id'], 'prompt_id':task['prompt_id'],
                      'kind':kind, 'chunk':chunk, 'start':start, 'step':step, 'active_step':step,'target':start+step,
                      'input_history_sha256':tensor_hash(history), 'geometry_sha256':task['geometry_sha256'],
                      'previous_output_sha256':records[-1]['prediction_sha256'] if records else None,
                      'rendered_prompt_sha256':hashlib.sha256(expected.encode()).hexdigest(),
                      'prediction_file':name,'prediction_sha256':before,'shape':task['shape'],'dtype':'float32',
                      'code_sha256':hashes,'code_directory':code_relative,'generation':generation,
                      'rejections':[read(p) for p in sorted((root/'rejections').glob('*.json')) if read(p)['packet']==str(packet)],'freeze_time':now(),'validation':'accepted'}
            schema_validate(record,'prediction-step.schema.json')
            write(folder/'steps'/f'step_{step:04d}.json',record,True)
            render_packet(root,task,host,split,packet,kind,chunk,start)
            return {'ok':True, 'next':'gval packet collect' if step==limit else 'Generate the next active step in the declared session policy', 'accepted_step':step,'episode_complete':step==limit}
        except Exception as error:
            write(root/'rejections'/(uuid.uuid4().hex+'.json'),{'time':now(),'packet':str(packet),'error':str(error)},True)
            raise


def packet_collect(task_dir, output):
    from .slurm import require_compute
    require_compute()
    with locked(task_dir):
        root,task,host,split = load_task(task_dir);destination = workspace(output)
        destination.mkdir(parents=True,mode=0o770)
        write(destination/'task.json',task,True)
        entries=[];missing=[];invalid=[]
        for group in task['starts']:
            for kind in ('rollout','single_step'):
                for start in group[kind]:
                    try:
                        _,records = frozen_chain(root,task,host,split,kind,group['chunk'],start)
                    except Exception as error:
                        records=[];invalid.append({'kind':kind,'chunk':group['chunk'],'start':start,'reason':str(error)})
                    folder = episode_root(root,kind,group['chunk'],start)
                    for step in range(1,(task['max_steps'] if kind=='rollout' else 1)+1):
                        identity={'kind':kind,'chunk':group['chunk'],'start':start,'step':step}
                        if step>len(records):missing.append(identity);continue
                        record=records[step-1];name=relative(kind,group['chunk'],start,step)
                        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                        with target.open('xb') as stream:
                            with (folder/record['prediction_file']).open('rb') as source:shutil.copyfileobj(source,stream)
                            stream.flush();os.fdatasync(stream.fileno())
                        target.chmod(0o440)
                        record_name=str(Path(name).parent/'steps'/f'step_{step:04d}.json')
                        write(destination/record_name,record,True)
                        entries.append({**identity,'path':name,'sha256':sha256(target),'record_path':record_name,'record_sha256':sha256(destination/record_name)})
                    copy_code(folder/'code', destination/'code'/kind/group['chunk']/('start_'+str(start)))
        provenance=read(root/'provenance.json');validate_unknowns(provenance)
        for key in RUN_FIELDS:
            if key not in provenance:raise ValueError('Missing run provenance field: '+key)
            if provenance[key] is None and not provenance['unknown_reasons'].get(key):
                raise ValueError('Unknown run provenance needs a reason: '+key)
        provenance.update(task_sha256=sha256(root/'task.json'),request_sha256=task['request_sha256'],
                          split_file_sha256=task['split_file_sha256'],protocol_sha256=task['protocol_sha256'],
                          geometry_sha256=task['geometry_sha256'],protocol=task['protocol'],prompt_id=task['prompt_id'],
                          template_sha256=task['template_sha256'],prompt_manifest_sha256=task['prompt_manifest_sha256'],
                          code_sha256={entry['record_path']:read(destination/entry['record_path'])['code_sha256'] for entry in entries},
                          collection_id=uuid.uuid4().hex,collected_at=now(),
                          rejection_records={p.name:read(p) for p in sorted((root/'rejections').glob('*.json'))})
        write(destination/'provenance.json',provenance,True)
        manifest={'schema_version':1,'task_sha256':sha256(destination/'task.json'),'entries':entries,
                  'coverage':{'required':sum(1 for _ in required(task)),'received':len(entries),'missing':missing,'invalid':invalid},
                  'provenance_sha256':sha256(destination/'provenance.json')}
        schema_validate(manifest,'predictions.schema.json')
        write(destination/'predictions.json',manifest,True)
        attestation=read(root/'host_attestation.json')
        attestation['predictions_sha256']=sha256(destination/'predictions.json')
        schema_validate(attestation,'host-attestation.schema.json')
        write(destination/'host_attestation.json',attestation,True)
    complete=not missing and not invalid
    return {'ok':True,'complete':complete,'next':'gval evaluate --predictions '+str(destination)+' --request <request> --split-id '+task['split_id']+' --split '+task['partition']+' --tier quick', 'prediction_bundle':str(destination),'coverage':manifest['coverage']}


class Bundle:
    """Read-only, one-state-at-a-time prediction source for the common scorer."""
    def __init__(self, directory, request_path, split, protocol, protocol_sha, partition, manifest_path, repo=REPO):
        self.root=workspace(directory)
        for path in self.root.rglob('*'):
            if path.is_symlink():raise ValueError('Prediction bundle contains a symlink')
            if not path.is_file() and not path.is_dir():raise ValueError('Special file in prediction bundle')
        self.task=task=read(self.root/'task.json');schema_validate(task,'task.schema.json')
        self.manifest=manifest=read(self.root/'predictions.json');schema_validate(manifest,'predictions.schema.json')
        expected={'request_sha256':sha256(request_path),'split_file_sha256':split.sha256,
                  'protocol_sha256':protocol_sha,'geometry_sha256':split.geometry_sha256,
                  'partition':partition,'split_id':split.split_id,'shape':list(split.shape),'channels':list(CHANNELS)}
        if any(task.get(k)!=v for k,v in expected.items()):raise ValueError('Prediction task/request/configuration identity mismatch')
        groups=select_starts(split,partition,protocol)
        wanted=[{'chunk':g.chunk,'rollout':list(g.rollout),'single_step':list(g.every)} for g in groups]
        if task['starts']!=wanted or task['max_steps']!=protocol['max_steps'] or not 1<=task['history_length']<=protocol['H_max']:
            raise ValueError('Prediction task shortened or changed frozen coverage')
        expected_targets=[{'kind':kind,'chunk':g.chunk,'start':int(start),'targets':list(range(start+1,start+count+1))}
                          for g in groups for kind,ids,count in [('rollout',g.rollout,protocol['max_steps']),('single_step',g.every,1)] for start in ids]
        if task['required_targets']!=expected_targets:raise ValueError('Prediction task target identities changed')
        pm,_=prompt(repo,task['prompt_id'],task['protocol'])
        if task['template_sha256']!=pm['template_sha256'] or task['prompt_manifest_sha256']!=sha256(Path(repo)/'prompts'/task['prompt_id']/'manifest.json'):
            raise ValueError('Prompt identity changed')
        if manifest['task_sha256']!=sha256(self.root/'task.json'):raise ValueError('Collected task changed')
        if manifest['provenance_sha256']!=sha256(self.root/'provenance.json'):raise ValueError('Collected provenance changed')
        self.provenance=read(self.root/'provenance.json');validate_unknowns(self.provenance)
        for key in ('request_sha256','split_file_sha256','protocol_sha256','geometry_sha256','prompt_id','template_sha256','protocol'):
            if self.provenance[key]!=task[key]:raise ValueError('Generation provenance identity mismatch')
        self.entries={};self.invalid=[];self.missing=[];self.records={}
        allowed=set(required(task))
        for entry in manifest['entries']:
            key=tuple(entry[k] for k in ('kind','chunk','start','step'))
            if key not in allowed or key in self.entries:raise ValueError('Duplicate or unexpected prediction entry')
            if entry['path']!=relative(*key) or entry['record_path']!=str(Path(entry['path']).parent/'steps'/f'step_{key[-1]:04d}.json'):
                raise ValueError('Noncanonical prediction layout')
            self.entries[key]=entry
        from .data import FrameReader
        reader=FrameReader(split,partition,manifest_path)
        if sha256(split.geometry_path)!=task['geometry_sha256']:raise ValueError('Scorer geometry changed')
        session_ids=set();a_code=None
        for group in groups:
            for kind,starts in [('rollout',group.rollout),('single_step',group.every)]:
                for start in starts:
                    history=reader.history(start,task['history_length'],'cpu')[0].numpy();previous=None;chain_ok=True
                    for step in range(1,(task['max_steps'] if kind=='rollout' else 1)+1):
                        key=(kind,group.chunk,start,step);entry=self.entries.get(key)
                        if entry is None:self.missing.append(dict(zip(('kind','chunk','start','step'),key)));chain_ok=False;continue
                        try:
                            if not chain_ok:raise ValueError('Earlier autoregressive entry is missing or invalid')
                            path=confined(self.root,entry['path']);record_path=confined(self.root,entry['record_path'])
                            if sha256(path)!=entry['sha256'] or sha256(record_path)!=entry['record_sha256']:raise ValueError('Prediction/provenance checksum mismatch')
                            record=read(record_path);schema_validate(record,'prediction-step.schema.json');validate_unknowns(record['generation'])
                            identities={'task_id':task['task_id'],'task_sha256':manifest['task_sha256'],
                                        'kind':kind,'chunk':group.chunk,'start':start,'step':step,'active_step':step,'target':start+step,
                                        'input_history_sha256':tensor_hash(history),'previous_output_sha256':previous,
                                        'geometry_sha256':task['geometry_sha256'],'prediction_sha256':entry['sha256'],
                                        'prediction_file':path.name,'shape':task['shape'],'dtype':'float32','prompt_id':task['prompt_id']}
                            if any(record.get(k)!=v for k,v in identities.items()):raise ValueError('Prediction provenance/input chain mismatch')
                            _,template=prompt(repo,task['prompt_id'],task['protocol'])
                            rendered=template.format(shape=tuple(task['shape']),dt_dump=split.dt_dump,history_length=task['history_length'],protocol=task['protocol'])
                            if record['rendered_prompt_sha256']!=hashlib.sha256(rendered.encode()).hexdigest():raise ValueError('Rendered prompt hash mismatch')
                            expected_episode=hashlib.sha256(f"{task['task_id']}:{kind}:{group.chunk}:{start}".encode()).hexdigest()
                            if record['episode_id']!=expected_episode:raise ValueError('Imported episode identity changed')
                            expected_code=f'code/step_{step:04d}'
                            if record['code_directory']!=expected_code:raise ValueError('Noncanonical per-step source layout')
                            code=confined(self.root,f'code/{kind}/{group.chunk}/start_{start}/step_{step:04d}')
                            actual={p.relative_to(code).as_posix():sha256(p) for p in sorted(code.rglob('*')) if p.is_file()}
                            if actual!=record['code_sha256']:raise ValueError('Imported source code changed')
                            if task['protocol']=='A':
                                if not actual:raise ValueError('Protocol A needs reproducible frozen source')
                                if a_code is not None and actual!=a_code:raise ValueError('Protocol A method changed between starts')
                                a_code=actual
                            session=record['generation']['session_id']
                            if task['session_policy']=='fresh_per_step' and session is not None:
                                if session in session_ids:raise ValueError('Session was reused despite fresh-per-step prompt policy')
                                session_ids.add(session)
                            value=array(path,task['shape']);history=np.concatenate((history[1:],value[None]),axis=0)
                            previous=entry['sha256'];self.records[key]=record
                        except Exception as error:
                            self.invalid.append({**dict(zip(('kind','chunk','start','step'),key)),'reason':str(error)});chain_ok=False
        self.coverage={'required':len(allowed),'received':len(self.records),'missing':self.missing,'invalid':self.invalid}
        attestation=read(self.root/'host_attestation.json')
        schema_validate(attestation,'host-attestation.schema.json')
        self.attestation=attestation
        self.attested=(attestation.get('predictions_sha256')==sha256(self.root/'predictions.json')
                       and all(attestation.get('claims',{}).get(k) is True for k in ATTESTATIONS)
                       and all(attestation.get(k) for k in ('operator','host','implementation_commit','attested_at','isolation_mechanism','evidence_sha256'))
                       and not attestation.get('exceptions')
                       and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in attestation.get('evidence_sha256',[]))
                       and all(record['generation'].get('provider') and record['generation'].get('model') and record['generation'].get('session_id') for record in self.records.values()))
        self.hashes={name:sha256(self.root/name) for name in ('task.json','predictions.json','provenance.json','host_attestation.json')}
        self.by_start={start:g.chunk for g in groups for start in g.every}
        self.history_length=task['history_length']

    def availability(self, name):
        if name in {'cost','stability'}:return False,'Prediction files do not provide an executable adapter for this analysis'
        kind='single_step' if name in {'single_step','error_correlation','conditional_error'} else 'rollout'
        expected=[key for key in required(self.task) if key[0]==kind]
        if any(key not in self.records for key in expected):return False,'Required '+kind+' coverage is incomplete or invalid'
        return True,None

    def prediction(self, kind, start, step, history):
        key=(kind,self.by_start[start],int(start),int(step));record=self.records[key];entry=self.entries[key]
        if tensor_hash(history[0].detach().cpu().numpy())!=record['input_history_sha256']:
            raise ValueError('Scorer history differs from the frozen generation history')
        path=confined(self.root,entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('Prediction changed after validation')
        import torch
        return torch.from_numpy(array(path,self.task['shape'])).to(history.device)[None]

    def metadata(self):
        return {'bundle_path':str(self.root),'bundle_sha256':self.hashes['predictions.json'],'files_sha256':self.hashes,
                'protocol':self.task['protocol'],'prompt_id':self.task['prompt_id'],
                'attestation':'host-attested' if self.attested else 'unattested',
                'coverage':self.coverage,'provenance':self.provenance,'host_attestation':self.attestation}

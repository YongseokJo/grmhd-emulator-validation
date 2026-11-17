"""Verified owner snapshots and deterministic, storage-independent report replay."""
from contextlib import contextmanager
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from .artifacts import csv_bytes, json_bytes, fingerprint
from .hashing import sha256
from .permissions import SharedStorage
from .splits import CHANNELS, identifier
from .validation import schema_validate

REPO=Path(__file__).resolve().parents[1]
GROUP_FIELDS=('split_id','protocol_id','protocol_sha256','split','tier')


class ReportSummary(dict):
    """A small projection only constructed after validating the original summary."""


def project(summary):
    return ReportSummary({key:summary[key] for key in ('single_step','rollout_curve','physics_table') if key in summary})


def read(path):return json.loads(Path(path).read_text())


def safe(root,relative):
    from .splits import relative_path
    relative_path(relative);path=Path(root)/relative
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('Symlink in report input/output: '+relative)
    return path


def protocol_blob(repo,protocol_id,digest):
    identifier(protocol_id)
    current=Path(repo)/'protocols'/(protocol_id+'.json')
    if current.is_file() and sha256(current)==digest:return read(current)
    # A SHA256 of file bytes is not a Git object ID. Resolve historical blobs
    # through tracked history without changing worktrees or fetching anything.
    commits=subprocess.run(['git','-C',str(repo),'log','--format=%H','--all','--',str(current.relative_to(repo))],capture_output=True,text=True,check=True)
    import hashlib
    for commit in commits.stdout.splitlines():
        result=subprocess.run(['git','-C',str(repo),'show',commit+':protocols/'+protocol_id+'.json'],capture_output=True)
        if result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==digest:return json.loads(result.stdout)
    raise ValueError('Recorded protocol SHA256 is unavailable in repository history: '+digest)


def valid_record(info,summary,repo):
    schema_validate(info,'run.schema.json')
    if not isinstance(summary,ReportSummary):schema_validate(summary,'summary.schema.json')
    if info.get('status')!='complete' or info.get('not_evaluated'):
        raise ValueError('Only complete evaluated runs are report records')
    if sha256(Path(repo)/'splits'/(info['split_id']+'.json'))!=info['split_file_sha256']:
        raise ValueError('Report split identity differs from frozen repository split')
    protocol=protocol_blob(repo,info['protocol_id'],info['protocol_sha256'])
    for name in ([] if isinstance(summary,ReportSummary) else protocol['tiers'][info['tier']]):
        if name not in summary or summary[name].get('status')=='not evaluated':
            raise ValueError('Completed report record lacks an evaluated analysis: '+name)
    return protocol


def names(path):
    if not path.exists():return []
    if path.is_symlink():raise ValueError('Shared report scan refuses a symlink directory')
    entries=list(path.iterdir())
    if any(p.is_symlink() for p in entries):raise ValueError('Report scan refuses symlink entries')
    return sorted(p for p in entries if p.is_dir())


def attempt_paths(storage,split_id):
    published=[p for model in names(storage.root/'runs'/split_id) for p in names(model)]
    staged=[p for user in names(storage.root/'staging') for p in names(user) if p.name!='conformance']
    return [(p,'runs') for p in published]+[(p,'staging') for p in staged]


def discard_run_cache(path):
    from .memory import discard_file_cache
    for item in path.rglob('*'):
        if item.is_file() and not item.is_symlink():discard_file_cache(item)


def scan(storage,split_id,repo):
    from .verify import verify_run
    from .runstore import reservation_lock
    attempts=[];records=[];seen=set()
    # Capture path and metadata identities while reservation/publication is
    # serialized. Release before hashing/rendering immutable completed bundles.
    snapshots=[]
    with reservation_lock(storage):
        for path,location in attempt_paths(storage,split_id):
            try:info=read(path/'run.json')
            except (OSError,ValueError):info=None
            if not isinstance(info,dict):info=None
            snapshots.append((path,location,info))
    for path,location,info in snapshots:
        if info is not None:
            if info.get('split_id')!=split_id:continue
        elif ('__'+split_id+'__') not in path.name:continue
        if path.name in seen:raise ValueError('Duplicate attempt ID across staging/runs: '+path.name)
        seen.add(path.name)
        reason=None;eligible=False
        if (path/'NOT_EVALUATED').exists():reason='not evaluated'
        elif (path/'FAILED').exists():reason='failed'
        elif (path/'PREVIEW').exists():reason='preview'
        elif info is None:reason='missing or invalid run.json'
        elif info.get('status')!='complete':reason=info.get('status','incomplete')
        elif location!='runs':reason='staging only'
        elif not (path/'COMPLETE').is_file():reason='missing COMPLETE'
        elif path.stat().st_mode&0o222:reason='unsealed'
        else:
            # verify_run binds the current protocol; historical report groups
            # are allowed if that original protocol blob is still in Git.
            try:
                protocol=protocol_blob(repo,info['protocol_id'],info['protocol_sha256'])
                verdict=verify_run(path,repo,allow_historical_protocol=True)
                if not verdict['ok']:raise ValueError('Bundle is not sealed')
                summary=read(path/'summary.json');valid_record(info,summary,repo)
                if info['run_id']!=path.name or info['model']!=path.parent.name:raise ValueError('Run path/model identity mismatch')
                records.append((info,project(summary),path));del summary
                discard_run_cache(path)
                eligible=True
            except Exception as error:
                raise ValueError('Corrupt COMPLETE run excluded; report not replaced: '+str(path)+': '+str(error)) from error
        suffix=re.search(r'__(validation|test)__(quick|full)__a[0-9]+$',path.name)
        attempt={'run_id':path.name,'model':info.get('model') if info else (path.parent.name if location=='runs' else None),
                 'checkpoint_sha256':info.get('checkpoint_sha256') if info else None,
                 'checkpoint_known':info is not None and 'checkpoint_sha256' in info,
                 'split':info.get('split') if info else (suffix[1] if suffix else None),
                 'tier':info.get('tier') if info else (suffix[2] if suffix else None),
                 'status':info.get('status','incomplete') if info else 'unknown','location':location,
                 'eligible':eligible,'reason':reason}
        attempts.append(attempt)
    ledger={'schema_version':1,'split_id':split_id,'attempts':sorted(attempts,key=lambda a:a['run_id'])}
    schema_validate(ledger,'attempts.schema.json')
    return records,ledger


def record_row(info,summary,ledger,protocol):
    curve=summary['rollout_curve']['at_steps'];step=max(int(k) for k in curve);headline=curve[str(step)]
    single=summary['single_step'];training=info.get('training_provenance') or {}
    attestation='not applicable' if info.get('predictions') is None else ('host-attested' if info['predictions'].get('attestation')=='host-attested' else 'unattested')
    flags=[]
    if not info['environment_locked']:flags.append('unlocked')
    if attestation=='unattested':flags.append('unattested')
    if not info['data_verified']:flags.append('data unverified')
    tests=[a for a in ledger['attempts'] if a['split']=='test' and a['model']==info['model'] and a['checkpoint_known'] and a['checkpoint_sha256']==info['checkpoint_sha256']]
    unknown=[a['run_id'] for a in ledger['attempts'] if a['split']=='test' and not a['checkpoint_known']]
    if unknown:flags.append('unattributed test attempts present')
    physics={'mass':None,'energy':None,'angular':None}
    values=summary.get('physics_table',{}).get('data',{}).get('aggregate',{}).get('mean',{})
    samples=values.get('samples',[])
    final=next((s for s in samples if int(s['step'])==step),{})
    physics.update({name:final.get(name+'_error') for name in physics})
    return {**{key:info[key] for key in GROUP_FIELDS},'model':info['model'],'version':info['version'],'run_id':info['run_id'],
            'checkpoint_sha256':info['checkpoint_sha256'],'report_step':step,'skill':headline['skill'],'nmae':headline['mean_nmae'],
            'single_step_skill':single['mean']['skill'],'single_step_nmae':single['mean']['nmae_model'],
            'physical_rel_l1':{name:single['per_channel'][name]['rel_l1'] for name in CHANNELS},'physics_errors':physics,
            'test_attempts':len(tests),'unattributed_test_attempts':unknown,'data_verified':info['data_verified'],
            'training_label':training.get('label',training.get('trained_on_this_split','unknown')),
            'training_evidence':training.get('evidence','unknown'),'environment_locked':info['environment_locked'],
            'prediction_attestation':attestation,'flags':flags,
            'rollout':[{'step':int(k),'time_rg_c':v['time_rg_c'],'skill':v['skill'],'skill_ci':v['skill_ci'],'nmae':v['mean_nmae']} for k,v in sorted(curve.items(),key=lambda p:int(p[0]))]}


def make_report(records,ledger,repo):
    groups={}
    for info,summary,*_ in records:
        protocol=valid_record(info,summary,repo);row=record_row(info,summary,ledger,protocol)
        key=tuple(info[k] for k in GROUP_FIELDS)
        groups.setdefault(key,[]).append(row)
    report={'schema_version':1,'split_id':ledger['split_id'],'comparison_fields':list(GROUP_FIELDS),
            'groups':[{**dict(zip(GROUP_FIELDS,key)),'rows':sorted(rows,key=lambda r:(r['model'],r['run_id']))} for key,rows in sorted(groups.items())],
            'excluded_attempts':[a for a in ledger['attempts'] if not a['eligible']]}
    schema_validate(report,'leaderboard.schema.json')
    return report


def fmt(value):return '—' if value is None else format(value,'.6g')


def cell(value):
    import html
    return html.escape(str(value),quote=False).replace('\\','\\\\').replace('|','\\|').replace('\r',' ').replace('\n',' ')


def markdown(report):
    lines=['# '+report['split_id']+' leaderboard','',
           'Compare only within the same split ID, protocol ID and SHA256, partition, and tier.',
           'Every row is one sealed artifact-verified run. Training evidence is an operator declaration; host attestation does not independently prove blindness.','']
    if not report['groups']:lines+=['No eligible completed runs.','']
    for group in report['groups']:
        lines+=['## '+group['split']+' / '+group['tier']+' / '+group['protocol_id'], '',
                'Protocol SHA256: `'+group['protocol_sha256']+'`. Headline scores use the final report step.','',
                '| Model | Run | Skill | nMAE | Test attempts | Data verified | Training label | Flags |',
                '|---|---|---:|---:|---:|---|---|---|']
        for row in group['rows']:
            values=[row['model'],row['run_id'],fmt(row['skill']),fmt(row['nmae']),row['test_attempts'],
                    'yes' if row['data_verified'] else 'no',row['training_label'],', '.join(row['flags']) or '—']
            lines+=['| '+' | '.join(cell(v) for v in values)+' |']
        lines+=['','Physical relative L1 (per channel):','', '| Run | '+' | '.join(CHANNELS)+' |','|---|'+'---:|'*8]
        for row in group['rows']:lines+=['| '+cell(row['run_id'])+' | '+' | '.join(fmt(row['physical_rel_l1'][c]) for c in CHANNELS)+' |']
        lines+=['','Final-step physics errors (undefined for quick):','', '| Run | Mass | Energy | Angular momentum |','|---|---:|---:|---:|']
        for row in group['rows']:lines+=['| '+cell(row['run_id'])+' | '+' | '.join(fmt(row['physics_errors'][c]) for c in ('mass','energy','angular'))+' |']
        prefix=group['protocol_sha256']+'/'+group['split']+'/'+group['tier']+'/'
        lines+=['','![Rollout comparison]('+prefix+'rollout_comparison.png)',
                '![Skill comparison]('+prefix+'skill_comparison.png)','']
    lines+=['## Excluded attempts','']
    if not report['excluded_attempts']:lines+=['None.','']
    else:
        lines+=['| Attempt | Location | Reason |','|---|---|---|']
        for a in report['excluded_attempts']:lines+=['| '+cell(a['run_id'])+' | '+a['location']+' | '+cell(a['reason'])+' |']
        lines+=['']
    return ('\n'.join(lines)+'\n').encode()


def figures(group,protocol):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import textwrap
    settings=protocol['constants']['figures'];outputs={}
    with matplotlib.rc_context({'font.family':settings['font'],'font.size':settings['font_size']}):
        for filename in ('rollout_comparison.png','skill_comparison.png'):
            fig=Figure(figsize=(14,max(5,1.2+.4*len(group['rows']))),dpi=settings['dpi']);FigureCanvasAgg(fig)
            ax=fig.add_axes((.08,.18,.28,.70) if filename.startswith('rollout') else (.08,.16,.53,.72))
            skill_axis=fig.add_axes((.43,.18,.28,.70)) if filename.startswith('rollout') else None
            palette=matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
            for index,row in enumerate(group['rows']):
                label=textwrap.fill(row['run_id'],55);curve=row['rollout']
                if filename.startswith('rollout'):
                    x=[v['time_rg_c'] for v in curve];y=[v['skill'] for v in curve]
                    line=ax.plot(x,[v['nmae'] for v in curve],marker='o',label=label,color=palette[index%len(palette)])[0]
                    skill_axis.plot(x,y,marker='o',label=label,color=line.get_color())
                    skill_axis.fill_between(x,[v['skill_ci'][0] if v['skill_ci'][0] is not None else float('nan') for v in curve],
                                    [v['skill_ci'][1] if v['skill_ci'][1] is not None else float('nan') for v in curve],alpha=.10,color=line.get_color(),linewidth=0)
                    ax.set(xlabel=r'Time [$R_g/c$]',ylabel='Loss-space nMAE (dimensionless)')
                    skill_axis.set(xlabel=r'Time [$R_g/c$]',ylabel='Skill relative to persistence',title='Skill');skill_axis.grid(alpha=.2)
                else:
                    value=row['skill'];ax.barh(index,float('nan') if value is None else value,label=label,color=palette[index%len(palette)])
                    if value is not None:ax.scatter([value],[index],s=14,color=palette[index%len(palette)],zorder=3)
                    lo,hi=curve[-1]['skill_ci']
                    if value is not None and lo is not None and hi is not None:ax.plot([lo,hi],[index,index],color='.2',linewidth=1.5)
                    ax.set_yticks(range(len(group['rows'])),[r['model']+' / '+r['version'] for r in group['rows']])
                    ax.set(xlabel='Final-step skill relative to persistence',ylabel='Model / version')
            skill_axis.axhline(0,color='.5',linewidth=.6) if skill_axis is not None else ax.axvline(0,color='.5',linewidth=.6)
            ax.set_title(group['split_id']+' / '+group['split']+' / '+group['tier']);ax.grid(alpha=.2)
            (skill_axis if skill_axis is not None else ax).legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=7,frameon=False)
            fig.text(.5,.035,'Protocol '+group['protocol_sha256']+' · Skill bands/bars: 95% CIs; nMAE lines: means',ha='center',fontsize=8)
            stream=io.BytesIO();fig.savefig(stream,format='png',metadata={'Software':None});outputs[filename]=stream.getvalue();fig.clear()
    return outputs


def artifacts(report,repo,render=True):
    result={'leaderboard.json':json_bytes(report),'leaderboard.md':markdown(report)}
    rows=[]
    for group in report['groups']:
        for row in group['rows']:
            flat={key:value for key,value in row.items() if key not in {'physical_rel_l1','physics_errors','rollout','unattributed_test_attempts','flags'}}
            flat.update({'rel_l1_'+c:row['physical_rel_l1'][c] for c in CHANNELS})
            flat.update({'physics_'+c:row['physics_errors'][c] for c in ('mass','energy','angular')})
            flat['flags']='; '.join(row['flags']);rows.append(flat)
        prefix=group['protocol_sha256']+'/'+group['split']+'/'+group['tier']+'/'
        if not render:continue
        result.update({prefix+name:value for name,value in figures(group,protocol_blob(repo,group['protocol_id'],group['protocol_sha256'])).items()})
    fields=list(rows[0]) if rows else ['split_id','protocol_id','protocol_sha256','split','tier','model','run_id','skill','nmae','test_attempts']
    result['leaderboard.csv']=csv_bytes(rows,fields)
    return result


def local_write(root,files,check=False):
    root=Path(root)
    if check:
        differences=[name for name,content in files.items() if not safe(root,name).is_file() or safe(root,name).read_bytes()!=content]
        if differences:raise ValueError('Generated report differs: '+', '.join(differences))
        return
    for name,content in files.items():
        target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('.report-'+uuid.uuid4().hex)
        with temporary.open('xb') as stream:stream.write(content)
        os.replace(temporary,target)


def shared_write(storage,split_id,files):
    parent=storage.folder('reports',split_id)
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        for name,content in files.items():
            target=safe(parent,name)
            relative=target.parent.relative_to(storage.root).parts
            destination=storage.folder(*relative)
            temporary='.report-'+uuid.uuid4().hex
            with storage.new_file(destination,temporary) as stream:stream.write(content)
            os.replace(destination/temporary,target)
    finally:os.close(fd)


def report(split_id,from_results=False,export_results=None,check=False,repo=REPO,storage=None):
    identifier(split_id);repo=Path(repo)
    if from_results:
        records,ledger=read_results(repo,split_id)
    else:
        if check:raise ValueError('--check requires --from-results')
        from .slurm import require_compute
        require_compute();storage=storage or SharedStorage();storage.require_owner()
        records,ledger=scan(storage,split_id,repo)
    payload=make_report(records,ledger,repo);files=artifacts(payload,repo,render=not check)
    if from_results:local_write(repo/'reports'/split_id,files,check)
    else:shared_write(storage,split_id,files)
    if export_results:
        if from_results:raise ValueError('Export requires fully verified shared runs')
        export_snapshot(records,ledger,Path(export_results))
        local_write(Path(export_results)/'reports'/split_id,files)
    return {'ok':True,'next':'Review generated reports and export through an owner PR',
            'split_id':split_id,'groups':len(payload['groups']),'runs':sum(len(g['rows']) for g in payload['groups']),
            'excluded_attempts':len(payload['excluded_attempts']),'checked':check}


def export_snapshot(records,ledger,root):
    """Export exact originals from verified immutable bundles; no ad-hoc copies."""
    split_id=ledger['split_id'];destination=Path(root)/'results'/split_id
    for info,summary,source in records:
        directory=safe(destination,info['model']+'/'+info['run_id']);directory.mkdir(parents=True,exist_ok=True)
        hashes={}
        for name in ('run.json','summary.json'):
            content=(source/name).read_bytes();target=safe(directory,name)
            if target.exists():
                if target.read_bytes()!=content:raise FileExistsError('Conflicting exported result: '+str(target))
            else:
                with target.open('xb') as stream:stream.write(content)
            hashes[name]=sha256(target)
        from .artifacts import result_files
        files=result_files(source)
        manifest={'schema_version':1,'split_id':split_id,'model':info['model'],'run_id':info['run_id'],
                  'run_sha256':hashes['run.json'],'summary_sha256':hashes['summary.json'],
                  'summary_filename':'summary.json','result_files':files,'result_fingerprint':info['result_fingerprint']}
        target=directory/'export.json';content=json_bytes(manifest)
        if target.exists() and target.read_bytes()!=content:raise FileExistsError('Conflicting export manifest')
        if not target.exists():target.write_bytes(content)
        discard_run_cache(source)
    local_write(destination,{'attempts.json':json_bytes(ledger)})


def read_results(repo,split_id):
    directory=safe(repo,'results/'+split_id)
    ledger_path=directory/'attempts.json'
    if not ledger_path.exists():
        if directory.exists() and any(directory.iterdir()):raise ValueError('Result exports require attempts.json')
        return [],{'schema_version':1,'split_id':split_id,'attempts':[]}
    ledger=read(ledger_path);schema_validate(ledger,'attempts.schema.json')
    if ledger['split_id']!=split_id:raise ValueError('Attempt ledger split mismatch')
    ids=[a['run_id'] for a in ledger['attempts']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate attempt ledger identity')
    eligible={a['run_id']:a for a in ledger['attempts'] if a['eligible']}
    records=[];found=set()
    for model in names(directory):
        for path in names(model):
            manifest=read(path/'export.json');schema_validate(manifest,'result-export.schema.json')
            if manifest['summary_filename']!='summary.json':raise ValueError('Unsupported exported summary encoding')
            if sha256(safe(path,'run.json'))!=manifest['run_sha256'] or sha256(safe(path,'summary.json'))!=manifest['summary_sha256']:
                raise ValueError('Exported result file checksum mismatch')
            info=read(path/'run.json');summary=read(path/'summary.json');valid_record(info,summary,repo)
            if (info['split_id'],info['model'],info['run_id'])!=(split_id,model.name,path.name):raise ValueError('Export path identity mismatch')
            if any(manifest[k]!=info[k] for k in ('split_id','model','run_id','result_fingerprint')):raise ValueError('Export identity differs from original run')
            if manifest['result_files'].get('summary.json')!=manifest['summary_sha256'] or fingerprint(manifest['result_files'])!=info['result_fingerprint']:
                raise ValueError('Export result fingerprint is not bound to original summary')
            a=eligible.get(info['run_id'])
            if a is None or (a['model'],a['checkpoint_sha256'],a['split'],a['tier'],a['location'],a['status'])!=(info['model'],info['checkpoint_sha256'],info['split'],info['tier'],'runs','complete'):
                raise ValueError('Result ledger eligibility or identity mismatch')
            found.add(info['run_id']);records.append((info,project(summary),None));del summary
            discard_run_cache(path)
    if found!=set(eligible):raise ValueError('Attempt ledger and exported completed records differ')
    return records,ledger

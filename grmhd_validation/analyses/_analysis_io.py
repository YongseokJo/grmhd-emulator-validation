"""Shared structured I/O for the Kevin/Sachin main@ebcc218 analysis ports.

Inference/window selection remain owned by the benchmark. Legacy scalar/tensor
formulas live in the per-analysis modules and _kevin_math, not an imported script.
"""
from contextlib import contextmanager
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace
import os
import weakref

import numpy as np
import torch

from ..artifacts import csv_bytes
from ..hashing import sha256
from .common import clean
from ._analysis_state import CURRENT, current
from ._geometry_physics import GeometryCriterion, MetricNormalizer


class DiskFrames(MutableMapping):
    """A rollout index, not a container of materialized volume fields."""
    def __init__(self,store,start,count,dtype=None):
        self.store,self.start,self.count,self.dtype=store,int(start),count,dtype
        self.extra={}
    def __iter__(self):return iter((*range(1,self.count+1),*self.extra))
    def __len__(self):return self.count+len(self.extra)
    def __contains__(self,step):return step in range(1,self.count+1) or step in self.extra
    def __getitem__(self,step):
        if step in self.extra:return self.extra[step]
        if step not in range(1,self.count+1):raise KeyError(step)
        value=self.store.load('rollout',self.start,step)
        return value if self.dtype is None else value.astype(self.dtype,copy=False)
    def __setitem__(self,step,value):
        if step in range(1,self.count+1):raise ValueError('Cached rollout frames are immutable')
        self.extra[step]=value
    def __delitem__(self,step):del self.extra[step]


class FullStore:
    """Invocation-owned temporary physical predictions, shared by all analyses."""
    def __init__(self, ctx):
        self.ctx=ctx;self.paths={};self.identities={};self.hashes={};self.clamps={}
        self.root=ctx.writer.root/'.scratch'

    def save(self, kind, start, step, value):
        key=(kind,int(start),int(step))
        if key in self.paths:return
        relative=f'.scratch/{kind}/{start}/{step}.npy'
        with self.ctx.writer.open(relative) as stream:
            np.save(stream,np.asarray(value,dtype=np.float32),allow_pickle=False)
            # Dirty scratch pages cannot be evicted. Make them clean before
            # advising the kernel that this completed frame need not stay in RAM.
            stream.flush();os.fdatasync(stream.fileno())
        path=self.ctx.writer.root/relative
        self.paths[key]=path;self.hashes[key]=sha256(path);self.identities[key]=self.identity(path)
        from ..memory import discard_file_cache
        discard_file_cache(path)
        diagnostic=getattr(self.ctx.adapter,'lorentz_clamp_count',None)
        measured=int(diagnostic(reset=True)) if diagnostic is not None else None
        if kind=='rollout':
            if measured is not None:self.clamps[(int(start),int(step))]=measured
            elif self.ctx.adapter.name=='persistence':self.clamps[(int(start),int(step))]=0

    @staticmethod
    def identity(path):
        stat=path.stat();return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def load(self, kind, start, step):
        key=(kind,int(start),int(step));path=self.paths[key]
        before=self.identity(path)
        if before!=self.identities[key]:
            if sha256(path)!=self.hashes[key]:raise ValueError('Private prediction cache changed')
            self.identities[key]=before
        value=np.load(path,allow_pickle=False)
        if value.dtype!=np.float32 or value.shape!=self.ctx.split.shape or not np.isfinite(value).all():
            raise ValueError('Invalid cached physical prediction')
        if self.identity(path)!=before:raise ValueError('Prediction cache changed while reading')
        from ..memory import discard_file_cache
        discard_file_cache(path)
        return value

    def ensure_rollout(self, start, count):
        if all(('rollout',int(start),step) in self.paths for step in range(1,count+1)):
            # Validate the entire replay without loading its volume fields.
            for step in range(1,count+1):
                key=('rollout',int(start),step);path=self.paths[key];before=self.identity(path)
                if before!=self.identities[key]:
                    if sha256(path)!=self.hashes[key]:raise ValueError('Private prediction cache changed')
                    if self.identity(path)!=before:raise ValueError('Prediction cache changed while reading')
                    self.identities[key]=before
            return DiskFrames(self,start,count)
        history=self.ctx.reader.history(start,self.ctx.adapter.history_length,self.ctx.device)
        for step in range(1,count+1):
            key=('rollout',int(start),step)
            if key in self.paths:
                output=torch.from_numpy(self.load(*key)).to(self.ctx.device)[None]
            else:
                output=self.ctx.step(history)
                self.save('rollout',start,step,output[0].cpu().numpy())
            history=torch.cat((history[:,1:],output[:,None]),dim=1)
        return DiskFrames(self,start,count)

    def cleanup(self):
        storage=self.ctx.writer.storage
        for path in sorted((p for p in storage.created if p==self.root or self.root in p.parents),key=lambda p:len(p.parts),reverse=True):
            info=path.lstat()
            if path.is_symlink() or storage.created[path]!=(info.st_dev,info.st_ino):
                raise PermissionError('Private cache was replaced')
            path.rmdir() if path.is_dir() else path.unlink()
            del storage.created[path];storage.original_modes.pop(path,None)


class ModelView:
    def __init__(self, adapter):self.adapter=adapter
    def parameters(self):
        modules=[value for value in vars(self.adapter).values() if isinstance(value,torch.nn.Module)]
        seen=set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter));yield parameter


@contextmanager
def bind(ctx, name, space, starts):
    if space not in {'loss','physical'}:raise ValueError('Metric space must be loss or physical')
    if getattr(ctx,'full',None) is None:ctx.full=FullStore(ctx)
    if ctx.run_info is None:ctx.run_info={}
    first=ctx.starts[0].rollout[0]
    starts=None if starts is None else tuple(map(int,starts))
    start=first if not starts else starts[0]
    scales=getattr(ctx,'channel_scales',None)
    if space=='physical' and scales is None:
        raise ValueError('Physical full-tier metrics require training-only channel scales')
    state={'ctx':ctx,'analysis':name,'space':space,'starts':starts,'start':start,
           'geometry':ctx.geometry,'scales':scales,'arrays':{}}
    token=CURRENT.set(state)
    try:
        norm=MetricNormalizer(ctx.geometry);crit=GeometryCriterion(ctx.geometry)
        args=SimpleNamespace(start=start,max_steps=ctx.protocol['max_steps'],
                n_starts=len(starts) if starts else 1,seed=ctx.protocol['seed'],save_dir='.')
        state.update(args=args,norm=norm,crit=crit)
        yield args,ModelView(ctx.adapter),norm,norm,crit,ctx.device if name=='cost' else torch.device('cpu'),ctx.reader
    finally:CURRENT.reset(token)


def dump_ids(reader):
    state=current();ctx=state['ctx']
    requested=getattr(ctx,'regression_ids',None)
    return list(requested) if requested is not None else sorted(reader.allowed)


def load_dump(reader, index):
    state=current()
    if index not in reader.allowed or index not in set(dump_ids(reader)):return None
    value=reader.numpy(index)
    if state['space']=='physical':value=value.astype(np.float64)
    state['ctx'].reader.tag(value,index)
    state['arrays'][id(value)]=(weakref.ref(value),index)
    return value


def eligible_single_ids(ctx, first):
    if getattr(ctx,'regression_ids',None) is not None:
        allowed=set(ctx.regression_ids)
        return [index for index in ctx.regression_ids if index>=first and index+1 in allowed]
    return [index for group in ctx.starts for index in group.every if index>=first]


def to_device(raw, device):
    dtype=torch.float64 if current()['space']=='physical' else torch.float32
    if isinstance(raw,torch.Tensor):
        result=raw.detach().to(device=device,dtype=dtype)
        for key in ('_gval_dump_id','_gval_history'):
            if hasattr(raw,key):setattr(result,key,getattr(raw,key))
        return result
    result=torch.as_tensor(np.array(raw,copy=True),device=device,dtype=dtype)
    if result.ndim==4:result=result[None]
    record=current()['arrays'].get(id(raw))
    index=record[1] if record is not None and record[0]() is raw else current()['ctx'].reader.lookup(raw)
    if index is not None:result._gval_dump_id=index
    return result


def to_tensor_cpu(raw):return to_device(raw,torch.device('cpu'))


def forecast_one_step(model, norm, state, device):
    bound=current();ctx=bound['ctx']
    index=getattr(state,'_gval_dump_id',None)
    history=getattr(state,'_gval_history',None)
    if history is None:
        start=index if index is not None else bound['start']
        history=ctx.reader.history(start,ctx.adapter.history_length,ctx.device)
        history[:,-1]=state.to(ctx.device,dtype=torch.float32)
    else:history=history.to(ctx.device)
    key=('single',int(index),1) if index is not None else None
    use_cache=bound['analysis'] not in {'cost','stability'}
    if key is not None and use_cache and key in ctx.full.paths:
        output=torch.from_numpy(ctx.full.load(*key)).to(ctx.device)[None]
    else:
        output=ctx.step(history)
        if key is not None and use_cache:ctx.full.save('single',index,1,output[0].cpu().numpy())
    result=output.to(device)
    result._gval_history=torch.cat((history[:,1:],output[:,None]),dim=1).detach()
    return result


def rollout_predictions(model,norm,init,n_steps,device,dump_id=None):
    state=current();ctx=state['ctx']
    if n_steps>ctx.protocol['max_steps']:raise ValueError('Analysis requested an out-of-protocol horizon')
    values=ctx.full.ensure_rollout(state['start'] if dump_id is None else dump_id,n_steps)
    if isinstance(values,DiskFrames):
        values.dtype=np.float64 if state['space']=='physical' else None
        return values
    return {step:value.astype(np.float64) if state['space']=='physical' else value for step,value in values.items()}


def get_rollout_clamps(dump_id=None):
    state=current();start=state['start'] if dump_id is None else dump_id
    return {step:state['ctx'].full.clamps.get((start,step),np.nan) for step in range(1,state['ctx'].protocol['max_steps']+1)}


def choose_starts(valid,rng,count):
    starts=current()['starts']
    if starts is not None:
        if not set(starts)<=set(valid):raise ValueError('Core start selection is outside eligible data')
        return np.asarray(starts,dtype=int)
    return valid if len(valid)<=count else rng.choice(valid,count,replace=False)


def stability_perturbation(initial,noise):
    state=current();amplitude=state['ctx'].constants['full']['stability_amplitude']
    if state['space']=='loss':return initial*(1+amplitude*noise)
    return initial+amplitude*np.asarray(state['scales']['rms'])[:,None,None,None]*noise


def sample(**values):
    def copy(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
        if isinstance(value,np.ndarray):return value.copy()
        if isinstance(value,dict):return {str(k):copy(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [copy(v) for v in value]
        return value
    return copy(values)


def save_figure(figure, source_path):
    state=current();ctx=state['ctx'];name=state['analysis']
    if getattr(ctx,'suppress_figures',False):return
    from ._port_manifest import FIGURES
    filename=FIGURES[name][Path(source_path).name]
    from matplotlib.text import Text
    for artist in figure.findobj(Text):
        text=artist.get_text().replace('TRON','model')
        if state['space']=='physical':
            text=text.replace('normalized-space','physical-space').replace('norm-space','physical-space').replace('normalized space','physical space').replace('nMAE','MAE (physical)').replace('(dex)','(code units)')
            if r'\Delta' in text:
                text=text.replace(r'\log_{10}', '').replace(r'\mathrm{dex}',r'\mathrm{code\ units}')
        artist.set_text(text)
    with ctx.writer.open('figures/'+filename) as stream:
        figure.savefig(stream,format='png',dpi=ctx.constants['figures']['dpi'],metadata={'Software':None})


def emit(ctx, name, values, space, starts):
    from ..validation import schema_validate
    result=clean({'analysis':name,'space':space,'starts':list(starts),'data':values})
    schema_validate(result,name+'.schema.json')
    ctx.writer.json('metrics/'+name+'.json',result)
    rows=[]
    def flatten(value,path):
        if isinstance(value,dict):
            for key in sorted(value):flatten(value[key],[*path,key])
        elif isinstance(value,list):
            for index,item in enumerate(value):flatten(item,[*path,str(index)])
        else:rows.append({'field':'.'.join(path),'value':value})
    flatten(result['data'],[])
    schema_validate(rows,'full_table.schema.json')
    ctx.writer.write('tables/'+name+'.csv',csv_bytes(rows,['field','value']))
    return result


def geometry_views(geometry):
    raw={name:np.asarray(getattr(geometry,attribute).detach().cpu()) for name,attribute in
         [('r','r'),('h','theta'),('ph','phi'),('gcov','gcov'),('gcon','gcon'),('gdet','gdet'),('dx','dx'),('spin','spin'),('dt_dump','dt_dump'),('resolved','resolved')]}
    if geometry.eos_gamma is not None:raw['eos_gamma']=np.asarray(geometry.eos_gamma.detach().cpu())
    geo={name:getattr(geometry,name).detach().cpu()[None,None] for name in ('gdet','gcov','gcon')}
    geo.update(dx=raw['dx'],dt_dump=float(geometry.dt_dump),eos_gamma=raw.get('eos_gamma'))
    return geo,raw


def execute_core(ctx, name, one, space, starts, native_multi=False):
    selected=None if starts is None else tuple(map(int,starts))
    settings=ctx.constants['figures']
    import matplotlib
    matplotlib.use(settings['backend'])
    with matplotlib.rc_context({'font.family':settings['font'],'font.size':settings['font_size'],
                                'figure.dpi':settings['dpi'],'savefig.dpi':settings['dpi']}):
        if native_multi or selected is None:
            with bind(ctx,name,space,selected) as values:
                data=one(ctx,*values)
            return {'per_start':[{'start':ctx.starts[0].rollout[0] if not selected else selected[0],'values':data}]}
        rows=[]
        for start in selected:
            with bind(ctx,name,space,[start]) as values:
                rows.append({'start':start,'values':one(ctx,*values)})
        return {'per_start':rows}


def production_starts(ctx):
    starts=[start for group in ctx.starts for start in group.rollout]
    return starts[:1] if ctx.constants['full']['milestone3_start_policy']=='single' else starts

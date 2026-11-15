"""Port: Kevin Fortune/Sachin, main@ebcc218, Fall26/eval/evaluate_forecast_ds.py.

channel_decorrelation_times/evaluate_correlation_times: explicit legacy sampling,
metric space and resolved-mask configuration make changed semantics testable.
"""
import numpy as np
import torch

from .common import loss_space,physical,clean,nmae,skill
from ._analysis_io import FullStore,emit,production_starts


def legacy_loss_tensor(raw,constants):
    """Pinned source arithmetic, exclusively for legacy regression."""
    value=torch.from_numpy(np.ascontiguousarray(raw,dtype=np.float32))[None]
    floors=constants['loss_space']
    return torch.cat((torch.log10(torch.clamp(value[:,0:1],min=floors['floor_rho'])),
                      torch.log10(torch.clamp(value[:,1:2],min=floors['floor_ug'])),
                      torch.arcsinh(value[:,2:])),dim=1)


def legacy_nmae(prediction,truth,constants):
    return (legacy_loss_tensor(prediction,constants)-legacy_loss_tensor(truth,constants)).abs().mean(dim=(2,3,4))[0].numpy()


def transform(raw, constants, space, legacy=False):
    if space=='physical':return physical(raw)
    if space!='loss':raise ValueError('Metric space must be loss or physical')
    if legacy:return legacy_loss_tensor(raw,constants)[0].numpy()
    return loss_space(raw,constants)


def disk_mask(geometry, constants, resolved=False):
    r=geometry.r.detach().cpu().numpy();theta=geometry.theta.detach().cpu().numpy()
    mask=(r<=constants['active_disk']['r_max']) & (np.abs(theta-np.pi/2)<constants['active_disk']['theta_halfwidth'])
    if resolved:mask &= geometry.resolved.detach().cpu().numpy().astype(bool)
    if not mask.any():raise ValueError('Correlation-time active disk is empty')
    return mask


def decorrelation(reader, geometry, constants, *, space='loss', ids=None, resolved=False, max_dumps=None, legacy=False):
    """Lag-1 fluctuations with the temporal background removed, as in Kevin."""
    selected=sorted(reader.allowed) if ids is None else list(ids)
    if max_dumps is not None:selected=selected[:max_dumps]
    if len(selected)<2:raise ValueError('Correlation time requires at least two truth dumps')
    mask=disk_mask(geometry,constants,resolved)
    total=np.zeros((8,int(mask.sum())),dtype=np.float64)
    for index in selected:total+=transform(reader.numpy(index),constants,space,legacy)[:,mask]
    background=total/len(selected)
    allowed=set(selected);cov=np.zeros(8);var_a=np.zeros(8);var_b=np.zeros(8);count=0
    for index in selected:
        if index+1 not in allowed or reader.allowed[index].chunk!=reader.allowed[index+1].chunk:continue
        a=transform(reader.numpy(index),constants,space,legacy)[:,mask]-background
        b=transform(reader.numpy(index+1),constants,space,legacy)[:,mask]-background
        for channel in range(8):
            cov[channel]+=np.dot(a[channel],b[channel])
            var_a[channel]+=np.dot(a[channel],a[channel]);var_b[channel]+=np.dot(b[channel],b[channel])
        count+=1
    if not count:raise ValueError('No consecutive truth pairs within an evaluation chunk')
    settings=constants['correlation_time'];epsilon=constants['skill']['epsilon']
    r1=np.clip(cov/(np.sqrt(var_a*var_b)+epsilon),-settings['r1_clip'],settings['r1_clip'])
    saturated=r1>settings['r1_saturated'];tau=np.zeros(8)
    positive=r1>0;tau[positive]=-1/np.log(np.minimum(r1[positive],settings['r1_cap']))
    return {'tau_c':tau,'r1':r1,'saturated':saturated,'n_pairs':count,'n_dumps':len(selected),'n_cells':int(mask.sum())}


def horizon(model, persistence, tau, dt, constants):
    """Original first-negative crossing and recrossing diagnostics, all channels."""
    scores=skill(model.mean(axis=0),persistence.mean(axis=0),constants,threshold=False)
    steps=np.arange(1,scores.shape[0]+1)
    crossing=[];never=[];last=[];after=[];late=[]
    for channel in range(8):
        values=scores[:,channel];negative=np.flatnonzero(values<0);positive=np.flatnonzero(values>0)
        censored=not len(negative);first=len(steps)-1 if censored else int(negative[0])
        crossing.append(int(steps[first]));never.append(censored)
        last.append(int(steps[-1]) if censored else int(steps[positive[-1]]) if len(positive) else 0)
        remainder=values[first+1:]
        after.append(1. if censored else float(np.mean(remainder>0)) if len(remainder) else np.nan)
        late.append(float(np.mean(values[24:min(50,len(values))])) if len(values)>=25 else np.nan)
    crossing=np.asarray(crossing);tau=np.asarray(tau)
    relative=np.full(8,np.nan);np.divide(crossing,tau,out=relative,where=tau>0)
    return {'steps':steps,'skill':scores,'crossing_steps':crossing,'never_crossed':never,
            'last_positive_steps':last,'fraction_positive_after_first':after,'mean_skill_25_50':late,
            'horizon_rg_c':crossing*dt,'horizon_tau_c':relative,'tau_c_rg_c':tau*dt}


def compute(ctx, *, space='loss', starts=None, legacy=False):
    if getattr(ctx,'full',None) is None:ctx.full=FullStore(ctx)
    selected=production_starts(ctx) if starts is None else list(starts)
    ids=getattr(ctx,'regression_ids',None) if legacy else None
    resolved=not legacy and ctx.split.split_id.startswith('rc300')
    truth=decorrelation(ctx.reader,ctx.geometry,ctx.constants,space=space,ids=ids,
                         resolved=resolved,max_dumps=200 if legacy else None,legacy=legacy)
    model=[];persistence=[]
    for start in selected:
        states=ctx.full.ensure_rollout(start,ctx.protocol['max_steps']);initial=ctx.reader.numpy(start)
        errors=[];baseline=[]
        for step in range(1,ctx.protocol['max_steps']+1):
            if legacy and space=='loss':
                true=ctx.reader.numpy(start+step)
                errors.append(legacy_nmae(states[step],true,ctx.constants))
                baseline.append(legacy_nmae(initial,true,ctx.constants))
                continue
            true=transform(ctx.reader.numpy(start+step),ctx.constants,space)
            predicted=transform(states[step],ctx.constants,space)
            held=transform(initial,ctx.constants,space)
            errors.append(np.mean(np.abs(predicted-true).reshape(8,-1),axis=1))
            baseline.append(np.mean(np.abs(held-true).reshape(8,-1),axis=1))
        model.append(errors);persistence.append(baseline)
    result={**truth,**horizon(np.asarray(model,dtype=np.float64),np.asarray(persistence,dtype=np.float64),truth['tau_c'],ctx.split.dt_dump,ctx.constants),
            'resolved_mask':resolved,'starts':selected}
    return {'per_start':[{'start':int(selected[0]),'values':result}]}


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'correlation_times',compute(ctx,space='loss',starts=starts),'loss',starts)

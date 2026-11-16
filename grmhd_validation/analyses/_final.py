"""Final production sampling; legacy cores remain explicit regression APIs.

Pair kernels are adapted from Kevin Fortune/Sachin Venkatesh, main@ebcc218,
Fall26/eval/evaluate_forecast_ds.py (conditional_error/error_correlation).
"""
import numpy as np

from ._aggregate import aggregate,nanmean
from ._analysis_io import bind,to_device,forecast_one_step,emit
from ._kevin_math import rel_l1_per_channel,pairwise_masked_corrcoef
from ._reference_statistics import conditional_edges


def pair_rows(ctx,name,starts):
    from ..truth_cache import truth_value
    rows=[]
    with bind(ctx,name,'physical',starts) as (_,model,norm,_,_,device,_):
        edges=conditional_edges(ctx) if name=='conditional_error' else None
        for start in starts:
            initial=ctx.reader.numpy(start).astype(np.float64);truth=ctx.reader.numpy(start+1).astype(np.float64)
            ctx.reader.tag(initial,start);ctx.reader.tag(truth,start+1)
            predicted=forecast_one_step(model,norm,to_device(initial,device),device).cpu().numpy()[0]
            predicted=predicted.astype(np.float64)
            if name=='error_correlation':
                values={'model_errors':rel_l1_per_channel(predicted,truth),
                        'persistence_errors':truth_value(ctx,'persistence_physical_relative',{'start':int(start)},lambda:rel_l1_per_channel(initial,truth))}
            else:
                count=np.zeros((len(edges[0])-1,8));sums=np.zeros_like(count);held=np.zeros_like(count)
                for channel in range(8):
                    magnitude=truth[channel] if channel<2 else np.abs(truth[channel])
                    for index in range(len(count)):
                        mask=(magnitude>=edges[channel,index]) & ((magnitude<edges[channel,index+1]) if index<len(count)-1 else (magnitude<=edges[channel,index+1]))
                        count[index,channel]=np.count_nonzero(mask)
                        sums[index,channel]=np.sum(np.abs(predicted[channel][mask].astype(np.float64)-truth[channel][mask]),dtype=np.float64)
                        held[index,channel]=np.sum(np.abs(initial[channel][mask].astype(np.float64)-truth[channel][mask]),dtype=np.float64)
                divide=lambda value:np.divide(value,count,out=np.full_like(value,np.nan),where=count>0)
                reference=truth_value(ctx,'conditional_baseline',{'start':int(start)},lambda:{'counts':count,'error_sum':held})
                if not np.array_equal(reference['counts'],count):raise ValueError('Conditional truth-bin counts differ from the verified cache')
                held=reference['error_sum']
                values={'edges':edges,'model':divide(sums),'persistence':divide(held),'counts':count,
                        'model_error_sum':sums,'persistence_error_sum':held}
            rows.append({'start':int(start),'values':values})
    return rows


def run(ctx,name,compute):
    single=name in {'conditional_error','error_correlation'}
    starts=[int(start) for group in ctx.starts for start in (group.every if single else group.rollout)]
    space='loss' if name=='correlation_times' else 'physical'
    previous=getattr(ctx,'suppress_figures',False);ctx.suppress_figures=True
    timings=[]
    try:
        if single:rows=pair_rows(ctx,name,starts)
        else:
            rows=[]
            for start in starts:
                rows.extend(compute(ctx,space=space,starts=[start])['per_start'])
                if name=='cost':timings.append(ctx.run_info['analysis_runtime']['cost']['milliseconds_per_step'])
    finally:ctx.suppress_figures=previous
    reducers={}
    if name=='conditional_error':
        for side in ('model','persistence'):
            def pooled(values,side=side):
                count=np.sum([v['counts'] for v in values],axis=0);total=np.sum([v[side+'_error_sum'] for v in values],axis=0)
                return np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
            reducers[(side,)]=pooled
        for field in ('counts','model_error_sum','persistence_error_sum'):
            reducers[(field,)]=lambda values,field=field:np.sum([v[field] for v in values],axis=0)
    if name=='error_correlation':
        # Per-start rows are the actual vectors; no per-pair correlation exists.
        for row in rows:
            row['values'].update(model=np.full((8,8),np.nan),persistence=np.full((8,8),np.nan))
        for side in ('model','persistence'):
            reducers[(side,)]=lambda values,side=side:pairwise_masked_corrcoef(np.stack([v[side+'_errors'] for v in values]),ctx.constants['full']['correlation_variance_min'])
    if name=='bias_analysis':
        reducers[('rmse',)]=(np.square,np.sqrt)
        reducers[('n_samples',)]=lambda values:sum(v['n_samples'] for v in values)
    if name=='correlation_times':
        from .correlation_times import horizon
        def estimate(values):
            return horizon(np.stack([v['nmae_model'] for v in values]),np.stack([v['nmae_persistence'] for v in values]),
                           values[0]['tau_c'],ctx.split.dt_dump,ctx.constants)
        for field in estimate([row['values'] for row in rows]):
            if field not in {'steps','tau_c_rg_c'}:reducers[(field,)]=lambda values,field=field:estimate(values)[field]
    statistics=aggregate(rows,ctx.split.chunks(ctx.partition),ctx.protocol,reducers)
    if name=='cost':
        timing_rows=[{'start':s,'values':{'milliseconds_per_step':v}} for s,v in zip(starts,timings)]
        from .common import clean
        ctx.run_info['analysis_runtime']['cost']['all_start_statistics']=clean(aggregate(timing_rows,ctx.split.chunks(ctx.partition),ctx.protocol))
    from .correlation_times import decorrelation
    from ..truth_cache import truth_value
    resolved=ctx.split.split_id.startswith('rc300')
    truth=truth_value(ctx,'correlation_time',{'space':'loss','resolved':resolved},
                     lambda:decorrelation(ctx.reader,ctx.geometry,ctx.constants,resolved=resolved))
    dt=float(ctx.split.dt_dump);steps=np.arange(1,ctx.protocol['max_steps']+1);tau=np.asarray(truth['tau_c'])
    ratio=np.divide(steps[:,None],tau[None,:],out=np.full((len(steps),8),np.nan),where=tau[None,:]>0)
    data={'per_start':rows,'aggregate':statistics,'time':{'steps':steps,'rg_c':steps*dt,'tau_c':ratio}}
    from ._figures import render
    render(ctx,name,statistics)
    return emit(ctx,name,data,space,starts)

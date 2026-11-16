"""Shared truth-only bins and spectral support for all model evaluations.

Conditional-bin rule and radial spectra: Kevin Fortune/Sachin Venkatesh,
Fall26/eval/evaluate_forecast_ds.py, main@ebcc218; owner-approved all-start
truth pooling replaces model-dependent edges/support in production.
"""
import numpy as np
import torch

from ..truth_cache import truth_value


def rollout_starts(ctx):
    return [int(start) for group in ctx.starts for start in group.rollout]


def histogram_edges(ctx,channel):
    step=max(1,ctx.protocol['max_steps']//ctx.constants['full']['distribution_step_divisor'])
    def calculate():
        values=np.concatenate([ctx.reader.numpy(start+step)[channel].astype(np.float64).ravel()
                               for start in rollout_starts(ctx)])
        settings=ctx.constants['full'];logarithmic=channel in (0,1)
        if logarithmic:
            values=np.maximum(np.abs(values),settings['epsilon'])
            low,high=np.percentile(values,settings['histogram_log_percentiles'])
        else:low,high=np.percentile(values,settings['histogram_linear_percentiles'])
        if high<=low:
            # An exactly constant truth distribution still has a valid shared bin.
            width=max(abs(low),1.)*np.finfo(np.float64).eps*settings['histogram_bins']
            low=low if logarithmic else low-width;high=high+width
        bins=settings['histogram_bins']
        return np.geomspace(low,high,bins+1) if logarithmic else np.linspace(low,high,bins+1)
    return truth_value(ctx,'histogram_edges',{'channel':int(channel),'step':step},calculate)


def conditional_edges(ctx):
    def calculate():
        settings=ctx.constants['full'];ids=[s for group in ctx.starts for s in group.every]
        count=min(settings['conditional_edge_dumps'],len(ids));spacing=max(1,len(ids)//count)
        pool=[[] for _ in range(8)];rng=np.random.default_rng(ctx.protocol['seed'])
        for start in ids[::spacing][:count]:
            value=ctx.reader.numpy(start).astype(np.float64).reshape(8,-1)
            n=value.shape[1];sample=min(n,max(settings['conditional_min_cells'],int(settings['conditional_cell_fraction']*n)))
            indices=rng.choice(n,size=sample,replace=False)
            for channel in range(8):pool[channel].append(value[channel,indices])
        result=[]
        for channel in range(8):
            values=np.concatenate(pool[channel]);values=values if channel in (0,1) else np.abs(values)
            result.append(np.percentile(values,np.linspace(0,100,settings['conditional_bins']+1)))
        return np.asarray(result,dtype=np.float64)
    return truth_value(ctx,'conditional_edges',{},calculate)


def spectrum_mask(ctx,channel,criterion):
    from ._kevin_math import radial_spectrum
    from ._analysis_io import to_tensor_cpu
    def calculate():
        powers=[radial_spectrum(to_tensor_cpu(ctx.reader.numpy(start+ctx.protocol['max_steps'])),criterion,channel)[1]
                for start in rollout_starts(ctx)]
        mean=np.mean(np.stack(powers),axis=0,dtype=np.float64)[1:]
        return mean>=np.percentile(mean,ctx.constants['full']['spectrum_truth_percentile'])
    return truth_value(ctx,'spectrum_mask',{'channel':int(channel)},calculate)

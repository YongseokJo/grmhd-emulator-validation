"""All-start means and the protocol's within-chunk block bootstrap.

The nonlinear estimators are supplied explicitly by each analysis. No averaging
of correlation matrices, bin means or independently computed horizon crossings
is substituted for recomputing the statistic on sampled starts.
"""
import copy
import warnings

import numpy as np


AXES={'r','theta','X','Y','radius','frequencies','modes','rho_modes','B1_modes',
      'channel','step','steps','n_steps','horizon_index','r10_index','resolved_index',
      'index','radius_rg','coarse_ring','resolved_mask','n_dumps','n_pairs','n_cells',
      'parameters','warmup_steps','timed_steps','edges','starts','tau_c','tau_c_rg_c','r1','saturated',
      'rho_ratio_mask','B1_ratio_mask','report_steps','r_outer','omega_keplerian','centers','widths','lags'}


def blocks_for_starts(starts,groups,length):
    """Blocks contain consecutive selected starts, never crossing a chunk."""
    position={int(start):i for i,start in enumerate(starts)}
    if len(position)!=len(starts):raise ValueError('Duplicate aggregation start')
    blocks=[];covered=[]
    for group in groups:
        indices=[position[start] for start in starts if group.first<=start<=group.last]
        indices.sort(key=lambda index:starts[index])
        covered.extend(indices)
        blocks.extend(np.asarray(indices[i:i+length],dtype=np.int64) for i in range(0,len(indices),length))
    if sorted(covered)!=list(range(len(starts))):raise ValueError('Aggregation start outside evaluated chunks')
    return blocks


def resamples(starts,groups,protocol):
    blocks=blocks_for_starts(starts,groups,protocol['bootstrap_block'])
    if not blocks:raise ValueError('No bootstrap blocks')
    rng=np.random.default_rng(protocol['seed'])
    draws=[np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))])
           for _ in range(protocol['bootstrap_resamples'])]
    return blocks,draws


def leaves(value,path=()):
    if isinstance(value,dict):
        for key in sorted(value):yield from leaves(value[key],(*path,key))
    elif isinstance(value,(tuple,list)) and any(isinstance(item,dict) for item in value):
        for index,item in enumerate(value):yield from leaves(item,(*path,index))
    else:yield path,value


def at(value,path):
    for part in path:value=value[part]
    return value


def put(tree,path,value):
    for part in path[:-1]:tree=tree[part]
    tree[path[-1]]=value


def numbers(value):
    if value is None:return np.asarray(np.nan,dtype=np.float64)
    return np.asarray(value,dtype=np.float64)


def nanmean(values):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(np.ascontiguousarray(values,dtype=np.float64),axis=0)


def mean_bounds(stack,draws,percentiles,transform=None,inverse=None):
    """Bound bootstrap memory independently of the number of spatial cells."""
    shape=stack.shape[1:];flat=stack.reshape(len(stack),-1)
    point=np.empty(flat.shape[1]);lower=np.empty_like(point);upper=np.empty_like(point)
    weights=np.stack([np.bincount(draw,minlength=len(stack)) for draw in draws]).astype(np.float64)
    for offset in range(0,flat.shape[1],2048):
        values=flat[:,offset:offset+2048]
        if transform is not None:values=transform(values)
        valid=np.isfinite(values);numerator=weights@np.where(valid,values,0.);denominator=weights@valid.astype(np.float64)
        distribution=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        estimate=nanmean(values)
        if inverse is not None:estimate=inverse(estimate);distribution=inverse(distribution)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            bounds=np.nanpercentile(distribution,percentiles,axis=0)
        point[offset:offset+2048]=estimate;lower[offset:offset+2048]=bounds[0];upper[offset:offset+2048]=bounds[1]
    return point.reshape(shape),lower.reshape(shape),upper.reshape(shape)


def aggregate(rows,groups,protocol,reducers=None):
    """Pointwise CIs of equal-start means; selected paths may be nonlinear."""
    if not rows:raise ValueError('No starts to aggregate')
    starts=[int(row['start']) for row in rows];values=[row['values'] for row in rows]
    blocks,draws=resamples(starts,groups,protocol)
    reducers={} if reducers is None else reducers
    point=copy.deepcopy(values[0]);lower=copy.deepcopy(point);upper=copy.deepcopy(point);valid_counts={}
    expected_paths={path for path,_ in leaves(point)}
    if any({path for path,_ in leaves(value)}!=expected_paths for value in values):
        raise ValueError('Start outputs have different field layouts')
    for path,first in list(leaves(point)):
        if path[-1]=='starts':
            for tree in (point,lower,upper):put(tree,path,list(starts))
            continue
        if np.asarray(first).dtype.kind in {'U','S'}:
            text=np.stack([np.asarray(at(value,path),dtype=str) for value in values])
            summary=np.where(np.all(text==text[0],axis=0),text[0],'mixed; see per-start reasons')
            for tree in (point,lower,upper):put(tree,path,summary)
            continue
        if path in reducers and not isinstance(reducers[path],tuple):
            reduce=reducers[path]
            raw_estimate=reduce(values);estimate=numbers(raw_estimate)
            distribution=np.stack([numbers(reduce([values[i] for i in draw])) for draw in draws])
        else:
            stack=np.stack([numbers(at(value,path)) for value in values])
            if path[-1] in AXES:
                if not np.all(stack==stack[0]):raise ValueError('Invariant coordinate/metadata differs: '+str(path))
                put(point,path,copy.deepcopy(first));put(lower,path,copy.deepcopy(first));put(upper,path,copy.deepcopy(first))
                continue
            counts=np.isfinite(stack).sum(axis=0)
            if path[-1] in {'eta','eta_jet','phi_normalized'} or np.any(counts<len(rows)):
                valid_counts['.'.join(map(str,path))]=counts
            transforms=reducers.get(path,(None,None))
            estimate,lo,hi=mean_bounds(stack,draws,protocol['constants']['ci']['percentiles'],*transforms)
            put(point,path,estimate);put(lower,path,lo);put(upper,path,hi)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            bounds=np.nanpercentile(distribution,protocol['constants']['ci']['percentiles'],axis=0)
        put(point,path,raw_estimate if path in reducers and np.asarray(raw_estimate).dtype==np.bool_ else estimate)
        put(lower,path,bounds[0]);put(upper,path,bounds[1])
    return {'mean':point,'lower':lower,'upper':upper,'n_starts':len(rows),'valid_counts':valid_counts,
            'bootstrap':{'method':'chunk_block_bootstrap','block_length':protocol['bootstrap_block'],
                         'n_blocks':len(blocks),'resamples':protocol['bootstrap_resamples'],'seed':protocol['seed']}}

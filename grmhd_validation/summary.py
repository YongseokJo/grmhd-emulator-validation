"""Format-2 scalar projections; metric writers and their numerical bytes are untouched."""
import json
from pathlib import Path

from .artifacts import json_bytes
from .hashing import sha256
from .splits import CHANNELS
from .validation import schema_validate

MAX_BYTES=1_000_000
SCALARS={
 'stability':('mean','std','skipped_draws'),
 'cost':('parameters','warmup_steps','timed_steps'),
 'lorentz':('max','max_step','max_p999','clamp_count'),
}
PHYSICS=('mass_error','energy_error','angular_error','divb_pred','divb_truth','normalized_divb_pred','normalized_divb_truth','flux_pred','flux_truth','gamma_max')
TRANSPORT=('mdot','edot','ldot','p_jet','eta','eta_jet','magnetic_flux','phi_normalized')
HORIZON=('crossing_steps','horizon_rg_c','horizon_tau_c','tau_c_rg_c','never_crossed')
BUDGET=('mass','dmass_dt','mdot_inner','mdot_outer','residual')
COVERAGE=('shell_profiles','mean_profiles','azimuthal_spectra','coherence','temporal_spectrum','conditional_error','bias_analysis','spatial_error','meridional_slice','rollout_evolution')


def get(tree,path):
    try:
        for key in path:tree=tree[key]
        if isinstance(tree,(dict,list)):raise ValueError('Summary headline must be a scalar: '+str(path))
        return tree
    except (KeyError,IndexError,TypeError):return None


def project(name,result,protocol,dt_dump=None):
    if result.get('status')=='not evaluated':return result
    if name=='single_step':return result
    if name=='rollout_curve':
        if 'at_steps' in result:return result
        if dt_dump is None:raise ValueError('Compact rollout summary needs the dataset cadence')
        reports={}
        for step in protocol['report_steps']:
            i=step-1
            reports[str(step)]={'time_rg_c':step*dt_dump,'mean_nmae':result['mean_nmae'][i],
                                'std_nmae':result['std_nmae'][i],'mean_persist':result['mean_persist'][i],'skill':result['skill'][i],
                                'skill_ci':result['skill_ci'][i],
                                'per_channel':{c:result['per_channel'][i][j] for j,c in enumerate(CHANNELS)}}
        return {'n_starts':result['n_starts'],'at_steps':reports,'attractor':result['attractor']}
    aggregate=result['data']['aggregate'];mean=aggregate['mean']
    headlines={};intervals={}
    def scalar(key,path):
        headlines[key]=get(mean,path)
        intervals[key]=[get(aggregate[side],path) for side in ('lower','upper')]
    if name in SCALARS:
        for key in SCALARS[name]:scalar(key,(key,))
    elif name=='correlation_times':
        for i,channel in enumerate(CHANNELS):
            for key in HORIZON:scalar(channel+'_'+key,(key,i))
    elif name=='error_correlation':
        for i,channel in enumerate(CHANNELS):
            for role in ('model','persistence'):scalar(channel+'_'+role+'_relative_error',(role+'_errors',i))
    elif name=='physics_table':
        for index,row in enumerate(mean['samples']):
            for key in PHYSICS:scalar('step_'+str(int(row['step']))+'_'+key,('samples',index,key))
    elif name=='accretion_diagnostics':
        for i,step in enumerate(mean['report_steps']):
            for role in ('pred','truth'):
                for key in TRANSPORT:scalar(f'step_{int(step)}_{role}_{key}',('headline_'+role,key,i))
        for key in ('radius_rg','coarse_ring'):headlines['resolved_'+key]=mean['shells']['resolved'][key]
        for role in ('pred','truth'):
            for key in ('eta_reason','eta_jet_reason','phi_normalized_reason'):
                for i,step in enumerate(mean['report_steps']):headlines[f'step_{int(step)}_{role}_{key}']=get(mean,('headline_'+role,key,i))
    elif name=='mdot':
        for i,row in enumerate(mean['samples']):
            for role in ('pred','truth'):scalar(f'step_{int(row["step"])}_{role}_mdot',('headline_'+role,i))
        for key in ('radius_rg','coarse_ring'):headlines['resolved_'+key]=mean['shells']['resolved'][key]
    elif name=='temporal_consistency':
        for quantity in ('mass','energy','angular'):
            for role in ('pred','truth'):
                key=quantity+'_drift_'+role;scalar(key,(key,-1))
    elif name=='conservation_closure':
        for role in ('pred','truth'):
            for key in BUDGET:scalar(role+'_'+key,('headline_'+role,key,-1))
        for key in ('radius_rg','coarse_ring'):headlines['resolved_'+key]=mean['shells']['resolved'][key]
    elif name=='statistical_props':
        for i,channel in enumerate(CHANNELS):
            for key in ('js','pred_mean','true_mean'):scalar(channel+'_'+key,('channels',str(i),key))
    elif name in ('spectrum','turbulence_stats'):
        keys=('slope_pred','slope_truth','fit_r2_pred','fit_r2_truth') if name=='spectrum' else ('s3_pred','s3_truth')
        for index,row in enumerate(mean['samples']):
            for key in keys:scalar(CHANNELS[int(row['channel'])]+'_'+key,('samples',index,key))
    elif name in COVERAGE:
        # Intrinsically spatial diagnostics have no invented global metric.
        headlines={'final_step':1 if name in ('conditional_error','bias_analysis') else mean.get('step',mean.get('n_steps',protocol['max_steps'])),
                   'sample_count':mean.get('n_samples')}
    else:raise ValueError('No compact summary schema for '+name)
    return {'n_starts':aggregate['n_starts'],'space':result['space'],'headlines':headlines,'ci95':intervals}


def from_metrics(root,protocol,tier,dt_dump):
    root=Path(root);summary={'schema_version':2,'metric_sha256':{}}
    for name in protocol['tiers'][tier]:
        path=root/'metrics'/(name+'.json')
        summary[name]=project(name,json.loads(path.read_text()),protocol,dt_dump)
        summary['metric_sha256'][name]=sha256(path)
        from .memory import discard_file_cache
        discard_file_cache(path)
    validate(summary)
    return summary


def validate(value,legacy=False):
    name='summary_legacy.schema.json' if legacy and value.get('schema_version')!=2 else 'summary.schema.json'
    schema_validate(value,name)

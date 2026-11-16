"""Metric-normalized physical transport and owner option-4 shell diagnostics.

Source kernels: Kevin Fortune/Sachin Venkatesh, main@ebcc218,
Fall26/eval/evaluate_forecast_ds.py (bsq_full_components/jet_power_shells)
and common/normalizer.py. Production signs and rest-mass subtraction follow
the owner's milestone-4 convention; original kernels remain in _kevin_math.
"""
import numpy as np

from . import _physical as p
from ..truth_cache import truth_dump_id,truth_value


def tensors(raw,geometry):
    x=p.batched(raw);g=p.array(geometry.gcov)
    velocity=p.ucon(x[:,2:5],g,geometry.gcon)
    covariant=np.einsum('ij...,bj...->bi...',g,velocity)
    B=x[:,5:8];b0=np.sum(B*covariant[:,1:],axis=1,keepdims=True)
    magnetic=np.concatenate((b0,(B+b0*velocity[:,1:])/velocity[:,0:1]),axis=1)
    magnetic_covariant=np.einsum('ij...,bj...->bi...',g,magnetic)
    bsq=np.sum(magnetic*magnetic_covariant,axis=1,keepdims=True)
    enthalpy=x[:,0:1]+float(geometry.eos_gamma)*x[:,1:2]
    trt=(enthalpy+bsq)*velocity[:,1:2]*covariant[:,0:1]-magnetic[:,1:2]*magnetic_covariant[:,0:1]
    trphi=(enthalpy+bsq)*velocity[:,1:2]*covariant[:,3:4]-magnetic[:,1:2]*magnetic_covariant[:,3:4]
    return velocity,covariant,magnetic,magnetic_covariant,bsq,trt,trphi


def shell_fluxes(raw,geometry,constants):
    x=p.batched(raw);u,ucov,b,bcov,bsq,trt,trphi=tensors(x,geometry)
    weight=p.array(geometry.gdet)*float(geometry.dx[1])*float(geometry.dx[2])
    rho_ur=x[:,0:1]*u[:,1:2]
    integral=lambda value:np.sum(value*weight,axis=(-2,-1),dtype=np.float64)[0,0]
    mdot=-integral(rho_ur);edot=integral(trt);ldot=-integral(trphi)
    settings=constants['full'];epsilon=settings['epsilon']
    sigma=bsq/(x[:,0:1]+epsilon)
    polar=np.abs(p.array(geometry.theta)-np.pi/2)>settings['jet_theta_offset']
    funnel=(sigma>settings['jet_sigma_min']) & polar & ((u[:,1:2]>0)|(-trt>0))
    funnel &= x[:,0:1]>settings.get('jet_density_floor_multiple',100)*constants['loss_space']['floor_rho']
    jet=-integral((trt+rho_ur)*funnel)
    phi=integral(np.abs(x[:,5:6]));valid=mdot>0
    divide=lambda numerator:np.divide(numerator,mdot,out=np.full_like(mdot,np.nan),where=valid)
    eta=divide(mdot-edot);eta_jet=divide(jet)
    normalized_flux=np.divide(.5*phi,np.sqrt(np.maximum(mdot,0)),out=np.full_like(mdot,np.nan),where=valid)
    reason=np.where(valid,'','nonpositive_mass_inflow')
    return {'mdot':mdot,'edot':edot,'ldot':ldot,'p_jet':jet,'eta':eta,'eta_jet':eta_jet,
            'magnetic_flux':phi,'phi_normalized':normalized_flux,
            'eta_reason':reason,'eta_jet_reason':reason,'phi_normalized_reason':reason,
            'funnel_fraction':np.mean(funnel[0,0],axis=(-2,-1),dtype=np.float64)}


def shells(ctx):
    r=p.array(ctx.geometry.r)[:,ctx.geometry.r.shape[1]//2,0]
    spin=float(ctx.geometry.spin);horizon=1+np.sqrt(max(0.,1-spin*spin))
    ih=int(np.argmin(np.abs(r-horizon)));rc=ctx.split.split_id.startswith('rc300')
    resolved=p.array(ctx.geometry.resolved).astype(bool).reshape(len(r),-1).all(axis=1)
    if rc:
        minimum=ctx.constants['full'].get('coarse_ring_r_min',6.11)
        choices=np.flatnonzero((r>=minimum)&resolved)
        if not len(choices):raise ValueError('No wholly resolved rc300 shell at r >= 6.11')
        ir=int(choices[0]);coarse=bool(r[ih]<minimum or not resolved[ih])
    else:ir=ih;coarse=False
    return {'horizon':{'index':ih,'radius_rg':float(r[ih]),'coarse_ring':coarse},
            'resolved':{'index':ir,'radius_rg':float(r[ir]),'coarse_ring':False}}


def cached_fluxes(ctx,raw):
    index=truth_dump_id(ctx,raw)
    calculate=lambda:shell_fluxes(raw,ctx.geometry,ctx.constants)
    return calculate() if index is None else truth_value(ctx,'transport',{'dump':int(index)},calculate)


def diagnostics(ctx,start):
    if not hasattr(ctx,'transport_results'):ctx.transport_results={}
    if start in ctx.transport_results:return ctx.transport_results[start]
    from ._kevin_math import mdot_shells
    from ._analysis_state import current
    state=current();criterion=state['crit'];normalizer=state['norm']
    frames=ctx.full.ensure_rollout(start,ctx.protocol['max_steps'])
    selected=list(ctx.protocol['report_steps']);metadata=shells(ctx)
    profiles={'pred':{},'truth':{}};series={'pred':{},'truth':{}};mass_profiles={'pred':[],'truth':[]}
    for step in range(1,ctx.protocol['max_steps']+1):
        true=ctx.reader.numpy(start+step)
        for side,raw in (('pred',frames[step]),('truth',true)):
            mass=mdot_shells(raw,criterion,normalizer,bare=False)[0,0].numpy()
            mass_profiles[side].append(mass)
            for role,info in metadata.items():series[side].setdefault(role,[]).append(float(mass[info['index']]))
            if step in selected:
                flux=cached_fluxes(ctx,raw)
                for key,value in flux.items():profiles[side].setdefault(key,[]).append(value)
    for side in profiles:
        profiles[side]={key:np.stack(values) for key,values in profiles[side].items()}
        series[side]={key:np.asarray(values,dtype=np.float64) for key,values in series[side].items()}
    extract=lambda side,role:{key:value[:,metadata[role]['index']] for key,value in profiles[side].items()}
    r=p.array(ctx.geometry.r)[:,ctx.geometry.r.shape[1]//2,0]
    result={'r':r,'report_steps':selected,'steps':np.arange(1,ctx.protocol['max_steps']+1),
            'shells':metadata,'profiles_pred':profiles['pred'],'profiles_truth':profiles['truth'],
            'horizon_pred':extract('pred','horizon'),'horizon_truth':extract('truth','horizon'),
            'headline_pred':extract('pred','resolved'),'headline_truth':extract('truth','resolved'),
            'mdot_profiles_pred':np.stack(mass_profiles['pred']),'mdot_profiles_truth':np.stack(mass_profiles['truth']),
            'mdot_series_pred':series['pred'],'mdot_series_truth':series['truth']}
    ctx.transport_results[start]=result;return result


def mass_budget(ctx,start):
    from ._kevin_math import mass_shells
    from ._analysis_state import current
    info=diagnostics(ctx,start);metadata=info['shells'];r=info['r']
    outer=int(np.argmin(np.abs(r-ctx.constants['active_disk']['r_max'])))
    volume=float(np.prod(p.array(ctx.geometry.dx)));dt=float(ctx.geometry.dt_dump)
    frames=ctx.full.ensure_rollout(start,ctx.protocol['max_steps'])
    state=current();crit=state['crit'];norm=state['norm']
    initial=ctx.reader.numpy(start)
    initial_shell=mass_shells(initial,crit,norm)[0,0].numpy()
    output={'steps':info['steps'],'shells':metadata,'r_outer':float(r[outer])}
    for side in ('pred','truth'):
        budgets={role:{key:[] for key in ('mass','dmass_dt','mdot_inner','mdot_outer','residual')} for role in metadata}
        previous={role:float(np.sum(initial_shell[entry['index']:outer+1])*volume) for role,entry in metadata.items()}
        if any(entry['index']>outer for entry in metadata.values()):raise ValueError('Resolved inner shell is outside the mass-budget domain')
        for step in range(1,ctx.protocol['max_steps']+1):
            raw=frames[step] if side=='pred' else ctx.reader.numpy(start+step)
            mass=mass_shells(raw,crit,norm)[0,0].numpy()
            mdot=info['mdot_profiles_'+side][step-1]
            for role,entry in metadata.items():
                inner=entry['index'];total=float(np.sum(mass[inner:outer+1])*volume)
                derivative=(total-previous[role])/dt;previous[role]=total
                values=(total,derivative,float(mdot[inner]),float(mdot[outer]),derivative-(mdot[outer]-mdot[inner]))
                for key,value in zip(budgets[role],values):budgets[role][key].append(value)
        for role,label in (('horizon','horizon'),('resolved','headline')):
            output[label+'_'+side]={key:np.asarray(value,dtype=np.float64) for key,value in budgets[role].items()}
    return output

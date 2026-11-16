"""Compact truth-cached physical profiles, adapted from Kevin Fortune/Sachin Venkatesh.

Source: Fall26/eval/evaluate_forecast_ds.py, main@ebcc218,
mean_profiles and temporal_spectrum. No model output enters a truth cache.
"""
import numpy as np

from ..truth_cache import truth_value
from ._transport import tensors


def mean_profiles(ctx,start):
    geo=ctx.geometry;r=geo.r.detach().cpu().numpy()[:,geo.r.shape[1]//2,0];theta=geo.theta.detach().cpu().numpy()[0,:,0]
    settings=ctx.constants['full'];count=max(1,ctx.protocol['max_steps']//settings['mean_step_divisor'])
    band=np.abs(theta-np.pi/2)<settings['mean_theta_halfwidth']
    indices=[int(np.argmin(np.abs(r-radius))) for radius in settings['mean_profile_radii']]
    def profiles(read):
        total=np.zeros(ctx.split.shape,dtype=np.float64)
        for step in range(count+1):total+=read(step)
        state=total/(count+1);u,_,_,_,bsq,_,_=tensors(state,geo)
        b2=bsq[0,0];beta=(float(geo.eos_gamma)-1)*state[1]/(.5*b2+settings['epsilon'])
        sigma=b2/(state[0]+settings['epsilon'])
        return {'omega':np.mean(u[0,3][:,band,:]/u[0,0][:,band,:],axis=(1,2)),
                'rho_theta':np.mean(np.abs(state[0,indices]),axis=-1),
                'beta':np.median(beta[:,band,:],axis=(1,2)),
                'sigma':np.median(sigma[:,band,:],axis=(1,2))}
    truth=truth_value(ctx,'mean_profiles',{'start':int(start)},lambda:profiles(lambda step:ctx.reader.numpy(start+step)))
    states=ctx.full.ensure_rollout(start,count)
    initial=ctx.reader.numpy(start)
    predicted=profiles(lambda step:initial if step==0 else states[step])
    result={'r':r,'theta':theta,'omega_keplerian':1/(r**1.5+float(geo.spin)),'n_steps':count}
    for side,values in (('pred',predicted),('truth',truth)):
        result.update({key+'_'+side:value for key,value in values.items()})
    return result


def temporal_spectrum(ctx,start):
    settings=ctx.constants['full'];count=ctx.protocol['max_steps'];geo=ctx.geometry
    r=geo.r.detach().cpu().numpy();theta=geo.theta.detach().cpu().numpy();epsilon=settings['epsilon']
    disk=(r>=settings['temporal_disk_r_min'])&(r<=ctx.constants['active_disk']['r_max'])&(np.abs(theta-np.pi/2)<settings['temporal_theta_halfwidth'])
    def spectrum(states,masks):
        # Mean of CELL power spectra, not power of a regional mean. Process one
        # channel/region at a time to avoid two float64 full-volume stacks.
        window=np.hanning(count);fs=1/float(geo.dt_dump);result={}
        for role,mask in masks.items():
            output=np.zeros((8,count//2+1),dtype=np.float64)
            if np.any(mask):
                for channel in range(8):
                    cells=np.stack([state[channel,mask] for state in states]).astype(np.float64)
                    cells-=cells.mean(axis=0,keepdims=True)
                    power=np.abs(np.fft.rfft(cells*window[:,None],axis=0))**2/(fs*np.sum(window**2)+epsilon)
                    output[channel]=np.mean(power,axis=1,dtype=np.float64)
            result[role]=output
        return result
    def truth_statistics():
        states=[ctx.reader.numpy(start+step) for step in range(1,count+1)]
        mean=np.zeros(ctx.split.shape,dtype=np.float64)
        for state in states:mean+=state
        mean/=count;bsq=tensors(mean,geo)[4][0,0];sigma=bsq/(mean[0]+epsilon)
        jet=(r>=settings['temporal_jet_r_min'])&(r<=ctx.constants['active_disk']['r_max'])&(np.abs(theta-np.pi/2)>settings['jet_theta_offset'])&(sigma>settings['jet_sigma_min'])
        masks={'disk':disk,'jet':jet}
        return {'masks':masks,'spectra':spectrum(states,masks)}
    truth=truth_value(ctx,'temporal_spectrum',{'start':int(start)},truth_statistics)
    states=ctx.full.ensure_rollout(start,count);predicted=spectrum([states[step] for step in range(1,count+1)],truth['masks'])
    return {'frequencies':np.fft.rfftfreq(count,d=float(geo.dt_dump)),
            'pred_disk':predicted['disk'],'truth_disk':truth['spectra']['disk'],
            'pred_jet':predicted['jet'],'truth_jet':truth['spectra']['jet'],'n_steps':count}

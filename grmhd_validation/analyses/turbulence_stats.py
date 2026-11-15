"""Port: Fall26/eval/evaluate_forecast_ds.py, main@ebcc218.
Authors: Kevin Fortune; Sachin Venkatesh (TRON evaluation originals).
Adapted to common physical adapter inputs, frozen windows and structured outputs.
Metric space and start selection are explicit for faithful legacy regression.
"""
import os,time,warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ._kevin_math import *
from ._analysis_io import (bind,geometry_views,forecast_one_step,rollout_predictions,load_dump,dump_ids,get_rollout_clamps,save_figure,sample,execute_core,emit,production_starts,choose_starts,stability_perturbation)
from ._analysis_state import current,setting
CHANNEL_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#17becf']
CHANNEL_NAMES=['rho','ug','utilde1','utilde2','utilde3','B1','B2','B3']
CHANNEL_NAMES_PLAIN=CHANNEL_NAMES
E_YMAX_CAP=1e5
L_YMAX_CAP=1e6

def _compute(ctx, args, model, norm, norm_cpu, crit, device, data_dir):
    geo, geo_raw = geometry_views(ctx.geometry)
    _records = []
    'Structure-function scaling exponents zeta_2, zeta_3, zeta_4 of the normalized\n    representation (log10 rho, arcsinh velocity/B) along r in the active disk (r in [2, 100] Rg),\n    pred vs truth: whether the model reproduces the turbulence scaling of the solver.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: turbulence_stats')
    n_steps = max(1, args.max_steps // setting('distribution_step_divisor'))
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        raise ValueError('Analysis has insufficient valid data: turbulence_stats')
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]
    pred_n = norm_cpu(to_tensor_cpu(pred))[0].numpy()
    target_n = norm_cpu(to_tensor_cpu(target))[0].numpy()
    r_1d = geo_raw['r'][:, 0, 0]
    h_1d = geo_raw['h'][0, :, 0]
    r_disk = (r_1d >= setting('turbulence_r_min')) & (r_1d <= disk_limit())
    eq_disk = np.abs(h_1d - np.pi / 2.0) < setting('turbulence_theta_halfwidth')
    channels = [0, 4, 5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, c, name in zip(axes, channels, ['$\\rho$', '$u^\\varphi$', '$B^r$']):
        xc_t = target_n[c][r_disk][:, eq_disk]
        xc_p = pred_n[c][r_disk][:, eq_disk]
        for lab, x, col in [('Truth', xc_t, 'b'), ('Forecast', xc_p, 'r')]:
            nr = x.shape[0]
            max_lag = max(setting('structure_min_max_lag'), nr // setting('structure_lag_divisor'))
            lags = np.arange(1, max_lag)
            s2 = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                d = x[lag:] - x[:-lag]
                s2[i] = np.mean(d ** 2)
            ax.loglog(lags, s2 / s2[0], '-', color=col, label=f'{lab} S2')
        ax.set_xlabel('Radial lag $\\Delta\\ln r$ (cells)')
        ax.set_ylabel('$S_2 / S_2(1)$ (normalized space)')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig11_structure_functions.png')
    save_figure(fig, path)
    plt.close(fig)
    for c, name in zip(channels, ['$\\rho$', '$u^\\varphi$', '$B^r$']):
        xc_t = target_n[c][r_disk][:, eq_disk]
        xc_p = pred_n[c][r_disk][:, eq_disk]
        zp, r2p, s3p = structure_function_exponents(xc_p)
        zt, r2t, s3t = structure_function_exponents(xc_t)
        str_z2_p = f'{zp[0]:.3f} ({r2p[0]:.2f})'
        str_z2_t = f'{zt[0]:.3f} ({r2t[0]:.2f})'
        str_s3 = f'{s3p:+.2e}/{s3t:+.2e}'
        _records.append(sample(channel=c, exponents_pred=zp, exponents_truth=zt, fit_r2_pred=r2p, fit_r2_truth=r2t, s3_pred=s3p, s3_truth=s3t))
    return sample(**{'samples': _records, 'step': n_steps})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'turbulence_stats',_compute,space,starts,native_multi=False)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'turbulence_stats',compute(ctx,space="physical",starts=starts),"physical",starts)

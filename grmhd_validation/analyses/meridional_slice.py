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
    'Truth / forecast / error maps of log10 rho and log10 B^2 in the\n    (r sin theta, r cos theta) meridional plane at the mid phi slice —\n    the standard GRMHD-paper flagship figure.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: meridional_slice')
    target_step = args.max_steps
    while target_step > 1 and load_dump(data_dir, args.start + target_step) is None:
        target_step -= 1
    target = load_dump(data_dir, args.start + target_step)
    if target is None:
        raise ValueError('Analysis has insufficient valid data: meridional_slice')
    if target_step < args.max_steps:
        pass
    states = rollout_predictions(model, norm, init, target_step, device, dump_id=args.start)
    pred = states[target_step]
    r = geo_raw['r'][:, :, geo_raw['r'].shape[2] // 2]
    h = geo_raw['h'][:, :, geo_raw['h'].shape[2] // 2]
    X = r * np.sin(h)
    Y = r * np.cos(h)
    pm = pred[:, :, :, pred.shape[3] // 2]
    tm = target[:, :, :, target.shape[3] // 2]
    log_rho_p = np.log10(np.maximum(pm[0], 1e-30))
    log_rho_t = np.log10(np.maximum(tm[0], 1e-30))
    log_rho_e = np.abs(pm[0] - tm[0]) if current()['space'] == 'physical' else np.abs(log_rho_p - log_rho_t)
    b2_p = np.log10(np.maximum(b2_magnitude(pred, geo_raw)[:, :, pred.shape[3] // 2], 1e-30))
    b2_t = np.log10(np.maximum(b2_magnitude(target, geo_raw)[:, :, target.shape[3] // 2], 1e-30))
    b2_e = np.abs(b2_magnitude(pred, geo_raw)[:, :, pred.shape[3] // 2] - b2_magnitude(target, geo_raw)[:, :, target.shape[3] // 2]) if current()['space'] == 'physical' else np.abs(b2_p - b2_t)
    rho_all = np.concatenate([log_rho_t.ravel(), log_rho_p.ravel()])
    rho_lo, rho_hi = (float(np.percentile(rho_all, 1)), float(np.percentile(rho_all, 99.5)))
    b2_all = np.concatenate([b2_t.ravel(), b2_p.ravel()])
    b2_lo, b2_hi = (float(np.percentile(b2_all, 1)), float(np.percentile(b2_all, 99.5)))
    b2_ticks = np.arange(0.0, 3.6, 0.5)
    panels = [('$\\log_{10}\\rho$ Truth', log_rho_t, rho_lo, rho_hi, None), ('$\\log_{10}\\rho$ Forecast', log_rho_p, rho_lo, rho_hi, None), ('$|\\Delta \\log_{10}\\rho|\\ [\\mathrm{dex}]$', log_rho_e, None, None, None), ('$\\log_{10} B^2$ Truth', b2_t, b2_lo, b2_hi, None), ('$\\log_{10} B^2$ Forecast', b2_p, b2_lo, b2_hi, None), ('$|\\Delta \\log_{10} B^2|\\ [\\mathrm{dex}]$', b2_e, 0.0, 3.5, b2_ticks)]
    if current()['space']=='physical':
        title,values,_,_,_=panels[-1]
        panels[-1]=(title,values,0.,None,None)

    def _save_meridional(suffix, xlim=None, ylim=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True, gridspec_kw={'wspace': 0.12, 'hspace': 0.3})
        for ax, (title, Z, vlo, vhi, c_ticks) in zip(axes.ravel(), panels):
            cf = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='auto', vmin=vlo, vmax=vhi)
            cb = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.046)
            if c_ticks is not None:
                cb.set_ticks(c_ticks)
            ax.set_title(title)
            ax.set_xlabel('$r\\sin\\theta\\;[R_g]$')
            ax.set_ylabel('$r\\cos\\theta\\;[R_g]$')
            ax.set_aspect('equal')
        if xlim is not None:
            axes[0, 0].set_xlim(*xlim)
        if ylim is not None:
            axes[0, 0].set_ylim(*ylim)
        zoom_str = f" ({suffix.strip('_')})" if suffix else ''
        fig.suptitle(f'Meridional Slices at t+{target_step} steps{zoom_str} (mid phi, seeded rollout)', fontsize=13)
        plt.tight_layout()
        path = os.path.join(args.save_dir, f'fig8_meridional_slice{suffix}.png')
        save_figure(fig, path)
        plt.close(fig)
    _save_meridional('', xlim=None, ylim=None)
    _save_meridional('_r100', xlim=(0, 100), ylim=(-100, 100))
    _save_meridional('_r250', xlim=(0, 250), ylim=(-250, 250))
    return sample(**{'X': X, 'Y': Y, 'rho_pred': pm[0], 'rho_truth': tm[0], 'rho_error': log_rho_e, 'B2_pred_log10': b2_p, 'B2_truth_log10': b2_t, 'B2_error': b2_e, 'step': target_step})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'meridional_slice',_compute,space,starts,native_multi=False)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'meridional_slice',compute(ctx,space="physical",starts=starts),"physical",starts)

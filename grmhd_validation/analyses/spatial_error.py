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
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: spatial_error')
    horizons = auto_horizons(args.max_steps)
    valid_h = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not valid_h:
        raise ValueError('Analysis has insufficient valid data: spatial_error')
    theta_1d = geo_raw['h'][0, :, 0]
    mask_disk = np.abs(theta_1d - np.pi / 2) < 0.5
    mask_jet = np.abs(theta_1d - np.pi / 2) > 1.0
    preds = rollout_predictions(model, norm, init, max(valid_h), device, dump_id=args.start)
    fig_fl, axes_fl = plt.subplots(len(valid_h), 2, figsize=(14, 4.2 * len(valid_h)), sharex=True)
    if len(valid_h) == 1:
        axes_fl = axes_fl.reshape(1, 2)
    fig_mg, axes_mg = plt.subplots(len(valid_h), 2, figsize=(14, 4.2 * len(valid_h)), sharex=True)
    if len(valid_h) == 1:
        axes_mg = axes_mg.reshape(1, 2)
    for idx, n_steps in enumerate(valid_h):
        target = load_dump(data_dir, args.start + n_steps)
        pred = preds[n_steps]
        centers, prof_disk = compute_radial_error_profile(pred, target, geo_raw, mask_disk)
        _, prof_jet = compute_radial_error_profile(pred, target, geo_raw, mask_jet)
        for c in range(5):
            axes_fl[idx, 0].plot(centers, prof_disk[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
            axes_fl[idx, 1].plot(centers, prof_jet[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
        for col_idx, reg_name in enumerate(['Disk Core', 'Jet Funnel']):
            ax = axes_fl[idx, col_idx]
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Rel. L1 Error')
            ax.set_title(f'Fluid Error — {reg_name} (t+{n_steps})')
            ax.legend(fontsize=8, ncol=2, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
        axes_fl[idx, 0].set_xlabel('$r\\;[R_g]$')
        axes_fl[idx, 1].set_xlabel('$r\\;[R_g]$')
        for c in range(5, 8):
            axes_mg[idx, 0].plot(centers, prof_disk[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
            axes_mg[idx, 1].plot(centers, prof_jet[:, c], linewidth=1.5, color=CHANNEL_COLORS[c], label=CHANNEL_NAMES[c])
        for col_idx, reg_name in enumerate(['Disk Core', 'Jet Funnel']):
            ax = axes_mg[idx, col_idx]
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Rel. L1 Error')
            ax.set_title(f'Magnetic Error — {reg_name} (t+{n_steps})')
            ax.legend(fontsize=8, frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
        axes_mg[idx, 0].set_xlabel('$r\\;[R_g]$')
        axes_mg[idx, 1].set_xlabel('$r\\;[R_g]$')
        _records.append(sample(step=n_steps, radius=centers, disk=prof_disk, jet=prof_jet))
    fig_fl.suptitle('Spatial Error Profiles — Fluid Channels', fontsize=13)
    fig_fl.tight_layout()
    p_fl = os.path.join(args.save_dir, 'spatial_error_fluid.png')
    save_figure(fig_fl, p_fl)
    plt.close(fig_fl)
    fig_mg.suptitle('Spatial Error Profiles — Magnetic Channels', fontsize=13)
    fig_mg.tight_layout()
    p_mg = os.path.join(args.save_dir, 'spatial_error_mag.png')
    save_figure(fig_mg, p_mg)
    plt.close(fig_mg)
    return sample(**{'samples': _records})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'spatial_error',_compute,space,starts,native_multi=False)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'spatial_error',compute(ctx,space="physical",starts=starts),"physical",starts)

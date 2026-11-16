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
    'Truth vs forecast log10 rho in the meridional plane at horizons\n    [1, 10, 25, 50] (clipped to max_steps): the qualitative proof the model\n    does not blow up and tracks the disk structure as it evolves.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: rollout_evolution')
    horizons = auto_horizons(args.max_steps)
    horizons = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not horizons:
        raise ValueError('Analysis has insufficient valid data: rollout_evolution')
    r = geo_raw['r'][:, :, geo_raw['r'].shape[2] // 2]
    h = geo_raw['h'][:, :, geo_raw['h'].shape[2] // 2]
    X = r * np.sin(h)
    Y = r * np.cos(h)
    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    states[0] = init
    fig, axes = plt.subplots(3, len(horizons), figsize=(4 * len(horizons), 11.5))
    if len(horizons) == 1:
        axes = axes.reshape(3, 1)
    mask_inner = active_disk_mask(geo_raw, rmax=disk_limit())[:, :, geo_raw['r'].shape[2] // 2]
    for j, hh in enumerate(horizons):
        t = load_dump(data_dir, args.start + hh)
        f = states[hh]
        phi_idx_t = t.shape[3] // 2
        phi_idx_f = f.shape[3] // 2
        Z_t = np.log10(np.maximum(t[0][:, :, phi_idx_t], 1e-30))
        Z_f = np.log10(np.maximum(f[0][:, :, phi_idx_f], 1e-30))
        Z_err = np.abs(f[0, :, :, phi_idx_f] - t[0, :, :, phi_idx_t]) if current()['space'] == 'physical' else np.abs(Z_f - Z_t)
        vmin_rho = np.percentile(Z_t[mask_inner], 1)
        vmax_rho = np.percentile(Z_t[mask_inner], 99)
        vmax_err = np.percentile(Z_err[mask_inner], 99)
        if vmax_err <= 0:
            vmax_err = 0.5
        im0 = axes[0, j].pcolormesh(X, Y, Z_t, cmap='inferno', vmin=vmin_rho, vmax=vmax_rho, shading='auto')
        fig.colorbar(im0, ax=axes[0, j])
        axes[0, j].set_title(f'Truth t+{hh}')
        axes[0, j].set_aspect('equal')
        axes[0, j].set_xlim(0, 100)
        axes[0, j].set_ylim(-100, 100)
        if j == 0:
            axes[0, j].set_ylabel('$r\\cos\\theta\\;[R_g]$')
        im1 = axes[1, j].pcolormesh(X, Y, Z_f, cmap='inferno', vmin=vmin_rho, vmax=vmax_rho, shading='auto')
        fig.colorbar(im1, ax=axes[1, j])
        axes[1, j].set_title(f'TRON t+{hh}')
        axes[1, j].set_aspect('equal')
        axes[1, j].set_xlim(0, 100)
        axes[1, j].set_ylim(-100, 100)
        if j == 0:
            axes[1, j].set_ylabel('$r\\cos\\theta\\;[R_g]$')
        im2 = axes[2, j].pcolormesh(X, Y, Z_err, cmap='viridis', vmin=0, vmax=vmax_err, shading='auto')
        fig.colorbar(im2, ax=axes[2, j])
        axes[2, j].set_title(f'|Error| t+{hh} (dex)')
        axes[2, j].set_aspect('equal')
        axes[2, j].set_xlim(0, 100)
        axes[2, j].set_ylim(-100, 100)
        axes[2, j].set_xlabel('$r\\sin\\theta\\;[R_g]$')
        if j == 0:
            axes[2, j].set_ylabel('$r\\cos\\theta\\;[R_g]$')
        if current()['space']=='physical':
            _records.append(sample(step=hh,truth=t[0,:,:,phi_idx_t],pred=f[0,:,:,phi_idx_f],error=Z_err))
        else:_records.append(sample(step=hh, truth=Z_t, pred=Z_f, error=Z_err))
    fig.suptitle(f'Meridional log10 rho rollout evolution (start {args.start}, r <= {disk_limit():.0f} Rg)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig16_rollout_evolution.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'X': X, 'Y': Y, 'samples': _records})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'rollout_evolution',_compute,space,starts,native_multi=False)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'rollout_evolution',compute)

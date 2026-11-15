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
from ._analysis_io import (bind,eligible_single_ids,geometry_views,forecast_one_step,rollout_predictions,load_dump,dump_ids,get_rollout_clamps,save_figure,sample,execute_core,emit,production_starts,choose_starts,stability_perturbation)
from ._analysis_state import current,setting
CHANNEL_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#17becf']
CHANNEL_NAMES=['rho','ug','utilde1','utilde2','utilde3','B1','B2','B3']
CHANNEL_NAMES_PLAIN=CHANNEL_NAMES
E_YMAX_CAP=1e5
L_YMAX_CAP=1e6

def _compute(ctx, args, model, norm, norm_cpu, crit, device, data_dir):
    geo, geo_raw = geometry_views(ctx.geometry)
    _records = []
    'Mean signed error (bias) and RMSE maps over multiple starts at h=1,\n    in the meridional plane for 4 key physical channels (rho, u^phi, B^r, B^phi).\n    Splits error into SYSTEMATIC (bias) vs random (scatter) without 16-panel overload.'
    all_ids = dump_ids(data_dir)
    all_ids_set = set(all_ids)
    valid = eligible_single_ids(ctx,args.start)
    if not valid:
        raise ValueError('Analysis has insufficient valid data: bias_analysis')
    rng = np.random.default_rng(args.seed)
    starts = choose_starts(valid, rng, args.n_starts)
    a0, b0 = (load_dump(data_dir, starts[0]), load_dump(data_dir, starts[0] + 1))
    bias = np.zeros((8, a0.shape[1], a0.shape[2]))
    rmse = np.zeros_like(bias)
    n = 0
    for s0 in starts:
        a, b = (load_dump(data_dir, s0), load_dump(data_dir, s0 + 1))
        with torch.no_grad():
            p = forecast_one_step(model, norm, to_device(a, device), device)
        if p is None:
            continue
        p = p.cpu().numpy()[0]
        d = p - b
        d_phi = d.mean(axis=3)
        bias += d_phi
        rmse += d_phi ** 2
        n += 1
    if not n:
        raise ValueError('Analysis has insufficient valid data: bias_analysis')
    bias /= n
    rmse = np.sqrt(rmse / n)
    sel_channels = [0, 4, 5, 7]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    X = geo_raw['r'][:, :, 0] * np.sin(geo_raw['h'][:, :, 0])
    Y = geo_raw['r'][:, :, 0] * np.cos(geo_raw['h'][:, :, 0])
    mask_inner = active_disk_mask(geo_raw, rmax=disk_limit())[:, :, 0]
    for j, c in enumerate(sel_channels):
        vmax = np.percentile(np.abs(bias[c][mask_inner]), 99)
        if vmax <= 0:
            vmax = 1e-06
        im_b = axes[0, j].pcolormesh(X, Y, bias[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        fig.colorbar(im_b, ax=axes[0, j])
        axes[0, j].set_title(f'{CHANNEL_NAMES[c]} Bias (Signed Error)')
        axes[0, j].set_aspect('equal')
        axes[0, j].set_xlim(0, 100)
        axes[0, j].set_ylim(-100, 100)
        axes[0, j].set_ylabel('$r\\cos\\theta\\;[R_g]$')
        vmax_r = np.percentile(rmse[c][mask_inner], 99)
        if vmax_r <= 0:
            vmax_r = 1e-06
        im_r = axes[1, j].pcolormesh(X, Y, rmse[c], cmap='inferno', vmin=0, vmax=vmax_r, shading='auto')
        fig.colorbar(im_r, ax=axes[1, j])
        axes[1, j].set_title(f'{CHANNEL_NAMES[c]} RMSE')
        axes[1, j].set_aspect('equal')
        axes[1, j].set_xlim(0, 100)
        axes[1, j].set_ylim(-100, 100)
        axes[1, j].set_xlabel('$r\\sin\\theta\\;[R_g]$')
        axes[1, j].set_ylabel('$r\\cos\\theta\\;[R_g]$')
    fig.suptitle(f'Spatial Error Decomposition — Bias & RMSE at h=1 ({n} starts, phi-averaged, r <= {disk_limit():.0f} Rg)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig14_error_decomposition.png')
    save_figure(fig, path)
    plt.close(fig)
    for c in range(8):
        pass
    return sample(**{'bias': bias, 'rmse': rmse, 'n_samples': n})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'bias_analysis',_compute,space,starts,native_multi=True)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'bias_analysis',compute(ctx,space="physical",starts=starts),"physical",starts)

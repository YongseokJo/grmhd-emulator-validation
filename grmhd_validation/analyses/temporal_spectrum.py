"""Port: Fall26/eval/evaluate_forecast_ds.py, main@ebcc218.
Authors: Kevin Fortune; Sachin Venkatesh (TRON evaluation originals).
Adapted to common physical adapter inputs, frozen windows and structured outputs.
Metric space and start selection are explicit for faithful legacy regression.
"""
import os, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ._kevin_math import *
from ._analysis_io import bind, geometry_views, forecast_one_step, rollout_predictions, load_dump, dump_ids, get_rollout_clamps, save_figure, sample, execute_core, emit, production_starts, choose_starts, stability_perturbation
from ._analysis_state import current, setting
CHANNEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
CHANNEL_NAMES = ['rho', 'ug', 'utilde1', 'utilde2', 'utilde3', 'B1', 'B2', 'B3']
CHANNEL_NAMES_PLAIN = CHANNEL_NAMES
E_YMAX_CAP = 100000.0
L_YMAX_CAP = 1000000.0

def _compute(ctx, args, model, norm, norm_cpu, crit, device, data_dir):
    if current()['space']=='physical':
        from ._profiles import temporal_spectrum as physical_profiles
        return physical_profiles(ctx,args.start)
    geo, geo_raw = geometry_views(ctx.geometry)
    _records = []
    'Temporal Fourier power spectrum P(f) = |rfft(x*w)|^2 / (fs * sum(w^2)) pred vs truth,\n    evaluated separately in the turbulent disk core and the polar jet funnel.\n    Diagnoses temporal dynamical fidelity: catches artificial laminar freezing\n    (premature high-f decay) and numerical flicker/instability (high-f surge).'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: temporal_spectrum')
    n_steps = max(8, args.max_steps)
    true_states = []
    for step in range(1, n_steps + 1):
        d = load_dump(data_dir, args.start + step)
        if d is None:
            break
        true_states.append(d)
    T_avail = len(true_states)
    if T_avail < 6:
        raise ValueError('Analysis has insufficient valid data: temporal_spectrum')
    states = rollout_predictions(model, norm, init, T_avail, device, dump_id=args.start)
    pred_states = []
    for step in range(1, T_avail + 1):
        if step in states:
            pred_states.append(states[step])
        else:
            break
    T_avail = min(len(true_states), len(pred_states))
    if T_avail < 6:
        raise ValueError('Analysis has insufficient valid data: temporal_spectrum')
    true_states = true_states[:T_avail]
    pred_states = pred_states[:T_avail]
    P_stack = np.stack(pred_states, axis=0)
    T_stack = np.stack(true_states, axis=0)
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(T_avail, d=dt)
    mean_truth = np.mean(T_stack, axis=0)
    bsq_mean = bsq_full(mean_truth, geo_raw, norm_cpu)
    sigma_mean = bsq_mean / (mean_truth[0] + 1e-30)
    theta_3d = geo_raw['h']
    r_3d = geo_raw['r']
    mask_disk = (r_3d >= setting('temporal_disk_r_min')) & (r_3d <= disk_limit()) & (np.abs(theta_3d - np.pi / 2.0) < setting('temporal_theta_halfwidth'))
    mask_jet = (r_3d >= setting('temporal_jet_r_min')) & (r_3d <= disk_limit()) & (np.abs(theta_3d - np.pi / 2.0) > setting('jet_theta_offset')) & (sigma_mean > setting('jet_sigma_min'))
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    w = np.hanning(T_avail)
    w_sum2 = float(np.sum(w ** 2))
    norm_factor = fs * w_sum2

    def compute_psd(data_stack, channel_idx, mask):
        cells = data_stack[:, channel_idx, mask]
        if cells.shape[1] == 0:
            return np.zeros_like(freqs)
        cells_fluct = cells - cells.mean(axis=0, keepdims=True)
        window = w[:, None]
        fft_vals = np.fft.rfft(cells_fluct * window, axis=0)
        psd = np.abs(fft_vals) ** 2 / (norm_factor + 1e-30)
        return np.mean(psd, axis=1)
    psd_pred_disk = np.zeros((8, len(freqs)))
    psd_true_disk = np.zeros((8, len(freqs)))
    psd_pred_jet = np.zeros((8, len(freqs)))
    psd_true_jet = np.zeros((8, len(freqs)))
    for c in range(8):
        row, col = (c // 4, c % 4)
        ax = axes[row, col]
        psd_p_disk = compute_psd(P_stack, c, mask_disk)
        psd_t_disk = compute_psd(T_stack, c, mask_disk)
        psd_p_jet = compute_psd(P_stack, c, mask_jet)
        psd_t_jet = compute_psd(T_stack, c, mask_jet)
        psd_pred_disk[c] = psd_p_disk
        psd_true_disk[c] = psd_t_disk
        psd_pred_jet[c] = psd_p_jet
        psd_true_jet[c] = psd_t_jet
        valid_f = freqs > 0
        f_plot = freqs[valid_f]
        ax.loglog(f_plot, psd_t_disk[valid_f], '-', color='#1f77b4', linewidth=1.6, label='Truth Disk')
        ax.loglog(f_plot, psd_p_disk[valid_f], '--', color='#d62728', linewidth=1.5, label='TRON Disk')
        ax.loglog(f_plot, psd_t_jet[valid_f], ':', color='#17becf', linewidth=1.4, label='Truth Jet')
        ax.loglog(f_plot, psd_p_jet[valid_f], '-.', color='#ff7f0e', linewidth=1.3, label='TRON Jet')
        ax.set_title(f'{CHANNEL_NAMES[c]}')
        ax.set_xlabel('$f\\;[c/R_g]$')
        ax.set_ylabel('PSD $P(f)$')
        ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle(f'Temporal Power Spectra P(f) — Turbulent Dynamics ({T_avail} dumps, steps 1..{T_avail}, start {args.start})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'temporal_power_spectrum.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'frequencies': freqs, 'pred_disk': psd_pred_disk, 'truth_disk': psd_true_disk, 'pred_jet': psd_pred_jet, 'truth_jet': psd_true_jet, 'n_steps': T_avail})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'temporal_spectrum', _compute, space, starts, native_multi=False)

def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'temporal_spectrum',compute)

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
    geo, geo_raw = geometry_views(ctx.geometry)
    _records = []
    "Azimuthal (m) power spectra in the equatorial band at mid radius:\n    Fourier decomposition in phi across multi-horizon forecasts [10, 25, 50, max_steps].\n    The disk's non-axisymmetric structure (spiral density waves, m=1..5 modes) tests\n    whether the model reproduces azimuthal turbulent fluctuations without damping high-m modes."
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: azimuthal_spectra')
    cand_horizons = [10, 25, 50, args.max_steps]
    horizons = sorted(list({h for h in cand_horizons if 1 <= h <= args.max_steps}))
    horizons = [h for h in horizons if load_dump(data_dir, args.start + h) is not None]
    if not horizons:
        raise ValueError('Analysis has insufficient valid data: azimuthal_spectra')
    max_h = max(horizons)
    states = rollout_predictions(model, norm, init, max_h, device, dump_id=args.start)
    horizons = [h for h in horizons if h in states]
    if not horizons:
        raise ValueError('Analysis has insufficient valid data: azimuthal_spectra')
    h3 = geo_raw['h'][:, :, 0]
    band = np.abs(h3[0] - np.pi / 2) < setting('azimuthal_theta_halfwidth')
    nr_mid = slice(init.shape[1] // 4, 3 * init.shape[1] // 4)

    def m_spec(x, c):
        y = x[c][nr_mid][:, band]
        y = y - np.mean(y, axis=-1, keepdims=True)
        return np.abs(np.fft.rfft(y, axis=-1)) ** 2
    channels = [0, 4, 5]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    truth_specs = {}
    pred_specs = {}
    for h in horizons:
        t_dump = load_dump(data_dir, args.start + h)
        p_dump = states[h]
        t_norm = norm_cpu(to_tensor_cpu(t_dump))[0].numpy()
        p_norm = norm_cpu(to_tensor_cpu(p_dump))[0].numpy()
        from ..truth_cache import truth_value
        calculate=lambda: {c:m_spec(t_norm,c).mean(axis=(0,1)) for c in channels}
        truth_specs[h] = truth_value(ctx,'azimuthal_spectra',{'dump':int(args.start+h)},calculate) if current()['space']=='physical' else calculate()
        pred_specs[h] = {c: m_spec(p_norm, c).mean(axis=(0, 1)) for c in channels}
    m_axis = np.arange(len(truth_specs[horizons[0]][channels[0]]))
    m_nyq = len(m_axis) - 1
    h_rep = 50 if 50 in horizons else horizons[-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex='col')
    for col_idx, (c, name) in enumerate(zip(channels, ['$\\rho$', '$u^\\varphi$', '$B^r$'])):
        ax_top = axes[0, col_idx]
        ax_top.loglog(m_axis[1:], truth_specs[h_rep][c][1:], 'b-', linewidth=1.4, alpha=0.9, label=f'Truth (h={h_rep})')
        ax_top.loglog(m_axis[1:], pred_specs[h_rep][c][1:], 'r--', linewidth=1.4, alpha=0.9, label=f'TRON (h={h_rep})')
        ax_top.set_xlim(1, m_nyq)
        ax_top.margins(x=0)
        ax_top.set_ylabel('Power $|F_m|^2$ (equatorial band)')
        ax_top.set_title(f'{name} Spectrum (h={h_rep})')
        ax_top.legend(fontsize=8, loc='lower left')
        ax_top.grid(True, alpha=0.3)
        ax_bot = axes[1, col_idx]
        for idx, h in enumerate(horizons):
            col = colors[idx % len(colors)]
            t_sp = truth_specs[h][c][1:]
            p_sp = pred_specs[h][c][1:]
            p10 = np.percentile(t_sp, setting('spectrum_truth_percentile'))
            m_valid = t_sp >= p10
            ratio = p_sp[m_valid] / (t_sp[m_valid] + 1e-30)
            ax_bot.semilogx(m_axis[1:][m_valid], ratio, '.-', color=col, linewidth=1.4, alpha=0.9, label=f'h={h}')
        ax_bot.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
        ax_bot.set_xlim(1, m_nyq)
        ax_bot.margins(x=0)
        ax_bot.set_xlabel('Azimuthal wavenumber $m$')
        ax_bot.set_ylabel('Power Ratio (Pred / Truth)')
        ax_bot.set_title(f'{name} Power Ratio')
        ax_bot.legend(fontsize=7.5, ncol=2, loc='upper left')
        ax_bot.grid(True, alpha=0.3)
    fig.suptitle(f'Azimuthal Mode Spectra and Multi-Horizon Drift', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig12_azimuthal_spectra.png')
    save_figure(fig, path)
    plt.close(fig)
    modes_to_report = list(range(1, 9))
    high_m_mask = m_axis > m_nyq // 2
    for h in horizons:
        for c, name in zip(channels, ['$\\rho$', '$u^\\varphi$', '$B^r$']):
            t_sp = truth_specs[h][c]
            p_sp = pred_specs[h][c]
            p10 = np.percentile(t_sp[1:], setting('spectrum_truth_percentile'))
            ratio = np.where(t_sp >= p10, p_sp / (t_sp + 1e-30), np.nan)
            parts = [f'm={m} {ratio[m]:.3f}' if not np.isnan(ratio[m]) else f'm={m} n/a' for m in modes_to_report if m < len(ratio)]
            p_high = float(np.sum(pred_specs[h][c][high_m_mask])) / (float(np.sum(pred_specs[h][c])) + 1e-30)
            t_high = float(np.sum(truth_specs[h][c][high_m_mask])) / (float(np.sum(truth_specs[h][c])) + 1e-30)
    return sample(**{'modes': m_axis, 'truth': truth_specs, 'pred': pred_specs, 'steps': horizons})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'azimuthal_spectra', _compute, space, starts, native_multi=False)

def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'azimuthal_spectra',compute)

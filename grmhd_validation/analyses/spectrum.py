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
        raise ValueError('Analysis has insufficient valid data: spectrum')
    n_steps = args.max_steps
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        raise ValueError('Analysis has insufficient valid data: spectrum')
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]
    pred_norm = norm_cpu(to_tensor_cpu(pred))
    target_norm = norm_cpu(to_tensor_cpu(target))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kmax = None
    for ax, ch, name in [(axes[0], 0, '$\\rho$'), (axes[1], 5, '$B^r$')]:
        kp, sp = radial_spectrum(pred_norm, crit, ch)
        kt, st = radial_spectrum(target_norm, crit, ch)
        sl_p, r2_p = spectral_slope(kp[1:], sp[1:])
        sl_t, r2_t = spectral_slope(kt[1:], st[1:])
        ax.loglog(kt[1:], st[1:], 'b-', label=f'Truth (slope {sl_t:.2f}, R²={r2_t:.2f})')
        ax.loglog(kp[1:], sp[1:], 'r--', label=f'TRON (slope {sl_p:.2f}, R²={r2_p:.2f})')
        kmax_ch = int(max(kt.max(), kp.max()))
        if kmax is None:
            kmax = kmax_ch
        ax.set_xlim(1.0, kmax_ch)
        ax.margins(x=0)
        ax.set_xlabel('Radial mode index $|k|$')
        ax.set_ylabel('Power $|\\mathcal{F}[\\widetilde{S}]|^2$ (Hann-windowed, norm-space)')
        ax.set_title(f'{name} radial spectrum (h={n_steps})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if ch == 0:
            k0t, s0t, s0p = (kt, st, sp)
        _records.append(sample(channel=ch, modes=kt, power_pred=sp, power_truth=st, slope_pred=sl_p, slope_truth=sl_t, fit_r2_pred=r2_p, fit_r2_truth=r2_t))
    ax = axes[2]
    p10_rho = np.percentile(s0t[1:], setting('spectrum_truth_percentile'))
    m_rho = s0t[1:] >= p10_rho
    if current()['space']=='physical':
        from ._reference_statistics import spectrum_mask
        m_rho=spectrum_mask(ctx,0,crit)
    ratio_rho = s0p[1:][m_rho] / (s0t[1:][m_rho] + 1e-30)
    ax.plot(k0t[1:][m_rho], ratio_rho, '.-', color=CHANNEL_COLORS[0], label='$\\rho$')
    kp2, sp2 = radial_spectrum(pred_norm, crit, 5)
    kt2, st2 = radial_spectrum(target_norm, crit, 5)
    p10_br = np.percentile(st2[1:], setting('spectrum_truth_percentile'))
    m_br = st2[1:] >= p10_br
    if current()['space']=='physical':m_br=spectrum_mask(ctx,5,crit)
    ratio_br = sp2[1:][m_br] / (st2[1:][m_br] + 1e-30)
    ax.plot(kt2[1:][m_br], ratio_br, '.-', color=CHANNEL_COLORS[5], label='$B^r$')
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0, kmax if kmax is not None else int(kt2.max()))
    ax.margins(x=0)
    ax.set_xlabel('Radial mode index $|k|$')
    ax.set_ylabel('Power ratio $T(|k|) = E_{\\rm pred}/E_{\\rm true}$')
    ax.set_title('Scale-resolved power ratio')
    all_r = np.concatenate([ratio_rho, ratio_br])
    pos_r = all_r[all_r > 0]
    if len(pos_r):
        r_p1 = float(np.percentile(pos_r, 1))
        r_p99 = float(np.percentile(pos_r, 99))
        y_lo = max(0.01, min(0.1, r_p1 * 0.8))
        y_hi = min(100.0, max(10.0, r_p99 * 1.2))
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_ylim(0.1, 10.0)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig6_windowed_spectra.png')
    save_figure(fig, path)
    plt.close(fig)
    if current()['space']=='physical':
        rho_full=np.full(len(m_rho),np.nan);rho_full[m_rho]=ratio_rho;ratio_rho=rho_full
        b_full=np.full(len(m_br),np.nan);b_full[m_br]=ratio_br;ratio_br=b_full
    return sample(**{'samples': _records, 'rho_modes': k0t[1:] if current()['space']=='physical' else k0t, 'rho_ratio': ratio_rho, 'rho_ratio_mask': m_rho, 'B1_modes': kt2[1:] if current()['space']=='physical' else kt2, 'B1_ratio': ratio_br, 'B1_ratio_mask': m_br, 'step': n_steps})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'spectrum',_compute,space,starts,native_multi=False)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'spectrum',compute)

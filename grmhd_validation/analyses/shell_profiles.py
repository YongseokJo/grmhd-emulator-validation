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
    'Conserved shell radial profiles across forecast horizons:\n    M(r) on log scale over full domain [r_H, r_max];\n    binding energy -E(r) on log scale over active disk [r_H, R_DISK_MAX];\n    angular momentum L(r) on log scale positive-only (negative shells counted\n    via n_neg_L in stdout, not drawn) over [r_H, R_DISK_MAX].\n    All rows display dashed ISCO markers at r_ISCO.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: shell_profiles')
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    r_max_M = float(r_mid.max())
    r_max_disk = disk_limit()
    mask_M = (r_mid >= r_H) & (r_mid <= r_max_M)
    mask_EL = (r_mid >= r_H) & (r_mid <= r_max_disk)
    horizons = auto_horizons(args.max_steps)
    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    all_neg_E_t = []
    all_neg_E_p = []
    all_L_t = []
    all_L_p = []
    fc_colors = {1: '#ff7f0e', 10: '#d62728', 25: '#8c564b', 50: '#7f7f7f', 100: '#17becf'}
    err_colors = {1: 'purple', 10: '#ff7f0e', 25: '#2ca02c', 50: '#d62728', 100: '#8c564b'}

    def get_fc_color(h, idx):
        return fc_colors.get(h, plt.cm.tab10((idx + 1) % 10))

    def get_err_color(h, idx):
        if h == 1:
            return 'purple'
        return err_colors.get(h, plt.cm.tab10(idx % 10))
    for idx, n_steps in enumerate(horizons):
        target = load_dump(data_dir, args.start + n_steps)
        if target is None:
            continue
        pred_np = states[n_steps]
        with torch.no_grad():
            M_p = mass_shells(pred_np, crit, norm)[0, 0].cpu().numpy()
            M_t = mass_shells(target, crit, norm)[0, 0].cpu().numpy()
            E_p, L_p = conserved_shells(pred_np, crit, norm)
            E_t, L_t = conserved_shells(target, crit, norm)
            E_p, E_t = (E_p[0, 0].cpu().numpy(), E_t[0, 0].cpu().numpy())
            L_p, L_t = (L_p[0, 0].cpu().numpy(), L_t[0, 0].cpu().numpy())
        n_ang = float(crit.gdet.shape[-2] * crit.gdet.shape[-1])
        M_p = M_p / n_ang
        M_t = M_t / n_ang
        r0 = r_mid[mask_M]
        r1 = r_mid[mask_EL]
        neg_E_t = (-E_t)[mask_EL]
        neg_E_p = (-E_p)[mask_EL]
        L_t_plot = L_t[mask_EL]
        L_p_plot = L_p[mask_EL]
        if n_steps == horizons[0]:
            axes[0, 0].plot(r0, M_t[mask_M], 'b-', linewidth=2.0, alpha=1.0, label='Truth')
            axes[1, 0].plot(r1, neg_E_t, 'b-', linewidth=2.0, alpha=1.0)
            axes[2, 0].plot(r1, L_t_plot, 'b-', linewidth=2.0, alpha=1.0)
        col_fc = get_fc_color(n_steps, idx)
        axes[0, 0].plot(r0, M_p[mask_M], '--', color=col_fc, linewidth=1.5, label=f'TRON ({n_steps} steps)')
        axes[1, 0].plot(r1, neg_E_p, '--', color=col_fc, linewidth=1.5)
        axes[2, 0].plot(r1, L_p_plot, '--', color=col_fc, linewidth=1.5)
        col_err = get_err_color(n_steps, idx)
        sl_err_M = np.abs(np.sign(M_p) * np.log1p(np.abs(M_p)) - np.sign(M_t) * np.log1p(np.abs(M_t)))
        axes[0, 1].plot(r0, sl_err_M[mask_M], color=col_err, linewidth=1, label=f'{n_steps} steps')
        sl_err_E = np.abs(np.sign(-E_p) * np.log1p(np.abs(-E_p)) - np.sign(-E_t) * np.log1p(np.abs(-E_t)))
        axes[1, 1].plot(r1, sl_err_E[mask_EL], color=col_err, linewidth=1, label=f'{n_steps} steps')
        sl_err_L = np.abs(np.sign(L_p) * np.log1p(np.abs(L_p)) - np.sign(L_t) * np.log1p(np.abs(L_t)))
        axes[2, 1].plot(r1, sl_err_L[mask_EL], color=col_err, linewidth=1, label=f'{n_steps} steps')
        all_neg_E_t.extend(neg_E_t)
        all_neg_E_p.extend(neg_E_p)
        all_L_t.extend(L_t_plot)
        all_L_p.extend(L_p_plot)
        _records.append(sample(step=n_steps, mass_pred=M_p, mass_truth=M_t, energy_pred=E_p, energy_truth=E_t, angular_pred=L_p, angular_truth=L_t, mass_error=sl_err_M, energy_error=sl_err_E, angular_error=sl_err_L))
    axes[0, 0].set_ylabel('$M(r)$')
    axes[0, 0].set_yscale('log')
    axes[1, 0].set_ylabel('$\\mathcal{E}(r)$ (rest-mass-subtracted)')
    axes[1, 0].set_yscale('log')
    all_neg_E_arr = np.array(all_neg_E_t + all_neg_E_p) if all_neg_E_t or all_neg_E_p else np.array([1.0])
    n_neg_E = int((all_neg_E_arr < 0).sum())
    pos_E = all_neg_E_arr[all_neg_E_arr > 0]
    if len(pos_E) > 0:
        max_E = float(np.max(pos_E))
        p99_8_E = float(np.percentile(pos_E, 99.8))
        cap_E = E_YMAX_CAP
        y_max_E = min(cap_E, p99_8_E * 1.1)
        y_min_E = max(1e-05, float(np.percentile(pos_E, 0.2)) / 1.1)
        axes[1, 0].set_ylim(y_min_E, y_max_E)
    axes[2, 0].set_ylabel('$L(r)$')
    axes[2, 0].set_yscale('log')
    all_L_arr = np.array(all_L_t + all_L_p) if all_L_t or all_L_p else np.array([1.0])
    n_neg_L = int((all_L_arr < 0).sum())
    pos_L = all_L_arr[all_L_arr > 0]
    if len(pos_L) > 0:
        max_L = float(np.max(pos_L))
        p99_8_L = float(np.percentile(pos_L, 99.8))
        cap_L = L_YMAX_CAP
        y_max_L = min(cap_L, p99_8_L * 1.1)
        y_min_L = max(1e-05, float(np.percentile(pos_L, 0.2)) / 1.1)
        axes[2, 0].set_ylim(y_min_L, y_max_L)
    for row in range(3):
        axes[row, 0].set_xscale('log')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].set_ylabel('Signed-log |err|')
        axes[row, 1].set_xscale('log')
        axes[row, 1].set_yscale('log')
        axes[row, 1].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(r_H, r_max_M)
    axes[0, 0].margins(x=0)
    axes[0, 1].set_xlim(r_H, r_max_M)
    axes[0, 1].margins(x=0)
    for r_idx in [1, 2]:
        axes[r_idx, 0].set_xlim(r_H, r_max_disk)
        axes[r_idx, 0].margins(x=0)
        axes[r_idx, 1].set_xlim(r_H, r_max_disk)
        axes[r_idx, 1].margins(x=0)
    for row in range(3):
        for col in range(2):
            add_isco_line(axes[row, col], r_isco)
    axes[0, 0].legend(fontsize=8, ncol=1)
    axes[0, 1].legend(fontsize=8)
    axes[2, 0].set_xlabel('$r\\;[R_g]$')
    axes[2, 1].set_xlabel('$r\\;[R_g]$')
    fig.suptitle('Conserved Shell Profiles', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig5_conserved_shells.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'samples': _records, 'r': r_mid})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'shell_profiles', _compute, space, starts, native_multi=False)

def run(ctx):
    starts = production_starts(ctx)
    return emit(ctx, 'shell_profiles', compute(ctx, space='physical', starts=starts), 'physical', starts)

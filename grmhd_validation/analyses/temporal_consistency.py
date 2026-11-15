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
    'Total conservation drift over rollout: M, E, L totals pred vs truth.\n    Measures the long-horizon stability of total conserved quantities.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: temporal_consistency')
    target_steps = args.max_steps
    with torch.no_grad():
        M0 = total_sum(mass_shells(init, crit, norm))
        E0, L0 = conserved_shells(init, crit, norm)
        E0 = total_sum(E0)
        L0 = total_sum(L0)
        x0 = to_device(init, crit.gdet.device)
        ug0 = crit._safe_rho_ug(x0[:, 1:2], CONFIG['floor_ug'], CONFIG['ceiling_ug'])
        U0 = total_sum(ug0 * crit.gdet)
        a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
        r_H, r_isco = kerr_horizon_isco(a_spin)
        Lscale = abs(M0) * r_isco
    states = rollout_predictions(model, norm, init, target_steps, device, dump_id=args.start)
    steps, M_p, E_p, L_p, M_t, E_t, L_t = ([], [], [], [], [], [], [])
    for step in range(1, target_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred = states[step]
        with torch.no_grad():
            M_p.append(total_sum(mass_shells(pred, crit, norm)))
            e, l = conserved_shells(pred, crit, norm)
            E_p.append(total_sum(e))
            L_p.append(total_sum(l))
            M_t.append(total_sum(mass_shells(true, crit, norm)))
            e, l = conserved_shells(true, crit, norm)
            E_t.append(total_sum(e))
            L_t.append(total_sum(l))
        steps.append(step)
    if not steps:
        raise ValueError('Analysis has insufficient valid data: temporal_consistency')
    steps = np.array(steps)
    M_p, E_p, L_p = (np.array(M_p), np.array(E_p), np.array(L_p))
    M_t, E_t, L_t = (np.array(M_t), np.array(E_t), np.array(L_t))
    scale_M = abs(M0)
    scale_E = abs(E0)
    if scale_E < 1e-06 * U0:
        scale_E = U0
        lbl_E = 'U_int,0'
    else:
        lbl_E = '|E0|'
    scale_L = abs(L0)
    if scale_L < 1e-06 * Lscale:
        scale_L = Lscale
        lbl_L = 'M0*r_ISCO'
    else:
        lbl_L = '|L0|'
    drift_M = lambda m: np.abs(m - M0) / (scale_M + 1e-30)
    drift_E = lambda e: np.abs(e - E0) / (scale_E + 1e-30)
    drift_L = lambda l: np.abs(l - L0) / (scale_L + 1e-30)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [('Mass M_total (scale |M0|)', M_p, M_t, drift_M), (f'Energy E_total (scale {lbl_E})', E_p, E_t, drift_E), (f'AngMom L_total (scale {lbl_L})', L_p, L_t, drift_L)]
    for ax, (name, p, t, d_fn) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], panels):
        ax.plot(steps, d_fn(p), 'r-', linewidth=1.5, label='TRON drift')
        ax.plot(steps, d_fn(t), 'b--', linewidth=1.5, label='Solver truth drift')
        ax.set_yscale('log')
        ax.set_xlabel('Forecast step')
        ax.set_ylabel('Drift / scale')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[1, 1].axis('off')
    txt = f'Conservation drift after {len(steps)} steps (TRON vs solver):\nScales: M:|M0|={scale_M:.2e}, E:{lbl_E}={scale_E:.2e}, L:{lbl_L}={scale_L:.2e}\n\n  Mass:    {drift_M(M_p)[-1]:.3e}  vs  {drift_M(M_t)[-1]:.3e}\n  Energy:  {drift_E(E_p)[-1]:.3e}  vs  {drift_E(E_t)[-1]:.3e}\n  AngMom:  {drift_L(L_p)[-1]:.3e}  vs  {drift_L(L_t)[-1]:.3e}\n\nmax TRON drift: M {drift_M(M_p).max():.3e} / E {drift_E(E_p).max():.3e} / L {drift_L(L_p).max():.3e}'
    axes[1, 1].text(0.05, 0.5, txt, fontsize=11, family='monospace', verticalalignment='center', transform=axes[1, 1].transAxes)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'temporal_consistency.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'steps': steps, 'mass_pred': M_p, 'mass_truth': M_t, 'energy_pred': E_p, 'energy_truth': E_t, 'angular_pred': L_p, 'angular_truth': L_t, 'mass_scale': scale_M, 'energy_scale': scale_E, 'angular_scale': scale_L, 'mass_drift_pred': drift_M(M_p), 'mass_drift_truth': drift_M(M_t), 'energy_drift_pred': drift_E(E_p), 'energy_drift_truth': drift_E(E_t), 'angular_drift_pred': drift_L(L_p), 'angular_drift_truth': drift_L(L_t)})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'temporal_consistency', _compute, space, starts, native_multi=False)

def run(ctx):
    starts = production_starts(ctx)
    return emit(ctx, 'temporal_consistency', compute(ctx, space='physical', starts=starts), 'physical', starts)

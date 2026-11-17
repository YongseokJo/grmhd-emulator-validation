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
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: mdot')
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    horizons = auto_horizons(args.max_steps)
    states = rollout_predictions(model, norm, init, max(horizons), device, dump_id=args.start)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    r_mask = (r_mid >= r_H) & (r_mid <= disk_limit())
    r_plot = r_mid[r_mask]
    all_M_vals = []
    cmap = plt.cm.tab10
    horizon_colors = {1: '#2ca02c', 10: '#ff7f0e', 25: '#d62728', 50: '#9467bd', 100: '#8c564b'}

    def get_h_color(h, idx):
        return horizon_colors.get(h, cmap(idx % 10))
    truth_lines, truth_labels = ([], [])
    fc_lines, fc_labels = ([], [])
    for idx, h in enumerate(horizons):
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        col = get_h_color(h, idx)
        M_p = mdot_shells(states[h], crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        M_t = mdot_shells(target, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        line_t, = axes[0].plot(r_plot, M_t[r_mask], '-', color=col, linewidth=2.0, alpha=1.0)
        line_p, = axes[0].plot(r_plot, M_p[r_mask], '--', color=col, linewidth=1.5)
        truth_lines.append(line_t)
        truth_labels.append(f'Truth t+{h}')
        fc_lines.append(line_p)
        fc_labels.append(f'Forecast t+{h}')
        err = (np.abs(M_p - M_t) / (np.abs(M_t) + 1e-30))[r_mask]
        axes[1].plot(r_plot, np.maximum(err, 0.001), color=col, linewidth=1, label=f'h={h}')
        all_M_vals.extend(M_t[r_mask])
        all_M_vals.extend(M_p[r_mask])
        _records.append(sample(step=h, pred=M_p, truth=M_t, relative_error=err))
    all_M_arr = np.array(all_M_vals) if all_M_vals else np.array([1.0])
    abs_M = np.abs(all_M_arr)
    pos_M = abs_M[abs_M > 0]
    if len(pos_M) > 0:
        p5 = float(np.percentile(pos_M, 5))
        linthresh_exp = int(np.floor(np.log10(max(0.0001, p5))))
        linthresh = float(10 ** linthresh_exp)
    else:
        linthresh = 0.1
        linthresh_exp = -1
    max_M = float(np.max(abs_M)) if len(abs_M) > 0 else 10.0
    max_decade = int(np.ceil(np.log10(max(max_M, linthresh * 10))))
    min_decade = linthresh_exp
    pos_ticks = [float(10 ** d) for d in range(min_decade, max_decade + 1)]
    min_M = float(np.min(all_M_arr)) if len(all_M_arr) > 0 else 0.0
    neg_ticks = []
    if min_M < -linthresh:
        neg_min_dec = int(np.ceil(np.log10(abs(min_M))))
        neg_ticks = [-float(10 ** d) for d in range(min_decade, neg_min_dec + 1)]
    yticks = sorted(neg_ticks + [0.0] + pos_ticks)
    axes[0].set_ylabel('$\\dot{M}(r) = -\\int \\rho u^r \\sqrt{-g}\\,d\\theta d\\varphi\\;[\\rm code\\;units]$')
    axes[0].set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=linthresh, linscale=1.0)
    axes[0].set_yticks(yticks)
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_xlim(r_H, disk_limit())
    axes[0].margins(x=0)
    add_isco_line(axes[0], r_isco)
    axes[0].set_title('Accretion Rate Profiles')
    axes[0].legend(truth_lines + fc_lines, truth_labels + fc_labels, fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('Rel. error')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(r_H, disk_limit())
    axes[1].margins(x=0)
    axes[1].set_xlabel('$r\\;[R_g]$')
    add_isco_line(axes[1], r_isco)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig7_mdot_profile.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'samples': _records, 'r': r_mid})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    result=execute_core(ctx, 'mdot', _compute, space, starts, native_multi=False)
    if space=='physical':
        from ._transport import shells
        metadata=shells(ctx)
        for row in result['per_start']:
            values=row['values'];values['shells']=metadata
            for role,label in (('horizon','horizon'),('resolved','headline')):
                index=metadata[role]['index']
                values[label+'_pred']=np.array([sample['pred'][index] for sample in values['samples']])
                values[label+'_truth']=np.array([sample['truth'][index] for sample in values['samples']])
    return result

def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'mdot',compute)

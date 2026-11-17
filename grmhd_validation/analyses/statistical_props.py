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
        raise ValueError('Analysis has insufficient valid data: statistical_props')
    n_steps = max(1, args.max_steps // setting('distribution_step_divisor'))
    while n_steps > 1 and load_dump(data_dir, args.start + n_steps) is None:
        n_steps -= 1
    target = load_dump(data_dir, args.start + n_steps)
    if target is None:
        raise ValueError('Analysis has insufficient valid data: statistical_props')
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    pred = states[n_steps]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    metrics = {}
    for c in range(8):
        row, col = (c // 4, c % 4)
        is_log = c in [0, 1]
        centers, pp, pt, widths = pdf_on_shared_bins(pred[c], target[c], log_scale=is_log,channel=c)
        js = js_divergence(pp, pt, widths=widths)
        pred_mean = float(np.sum(centers * pp * widths) / (np.sum(pp * widths) + 1e-30))
        true_mean = float(np.sum(centers * pt * widths) / (np.sum(pt * widths) + 1e-30))
        metrics[c] = {'js': js, 'pred_mean': pred_mean, 'true_mean': true_mean}
        axes[row, col].plot(centers, pt, 'b-', label='Truth', linewidth=1.5)
        axes[row, col].plot(centers, pp, 'r--', label='TRON', linewidth=1.5)
        axes[row, col].set_xlabel(f'${CHANNEL_NAMES_PLAIN[c]}$')
        axes[row, col].set_ylabel('PDF (shared bins)')
        axes[row, col].set_title(f'{CHANNEL_NAMES_PLAIN[c]}: JS={js:.4f}')
        axes[row, col].legend(fontsize=7)
        axes[row, col].grid(True, alpha=0.3)
        _records.append(sample(channel=c, centers=centers, pdf_pred=pp, pdf_truth=pt, widths=widths, js=js))
    plt.tight_layout()
    p1 = os.path.join(args.save_dir, 'statistical_props_pdf.png')
    save_figure(fig, p1)
    plt.close(fig)
    for c in range(8):
        m = metrics[c]
    return sample(**{'channels': metrics, 'samples': _records, 'step': n_steps})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'statistical_props',_compute,space,starts,native_multi=False)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'statistical_props',compute)

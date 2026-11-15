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
    "nMAE binned by TRUTH magnitude quintiles (5 bins), per channel, in normalized\n    space (so bins are comparable across channels). Answers 'where in the\n    dynamic range is the model good?': floor-dominated cells (jets, ISCO\n    vacuum), mid-range disk cells, or the extreme tails. Model vs persistence\n    per bin shows whether the model beats the trivial baseline everywhere."
    ids = dump_ids(data_dir)
    test_ids = eligible_single_ids(ctx,args.start)
    if not test_ids:
        raise ValueError('Analysis has insufficient valid data: conditional_error')
    n_bins = setting('conditional_bins')
    n_sample_dumps = min(setting('conditional_edge_dumps'), len(test_ids))
    step_spacing = max(1, len(test_ids) // n_sample_dumps)
    sampled_ids = test_ids[::step_spacing][:n_sample_dumps]
    rng_pool = np.random.default_rng(args.seed)
    pool = [[] for _ in range(8)]
    for did in sampled_ids:
        dump_data = load_dump(data_dir, did)
        if dump_data is None:
            continue
        nd = norm_cpu(to_tensor_cpu(dump_data))[0].numpy()
        n_cells = nd[0].size
        n_sub = min(n_cells,max(setting('conditional_min_cells'), int(setting('conditional_cell_fraction') * n_cells)))
        sub_idx = rng_pool.choice(n_cells, size=n_sub, replace=False)
        for c in range(8):
            pool[c].append(nd[c].ravel()[sub_idx].astype(np.float64 if current()['space']=='physical' else np.float32))
    edges = np.zeros((8, n_bins + 1))
    for c in range(8):
        data_c = np.concatenate(pool[c]) if pool[c] else np.zeros(10, dtype=np.float32)
        mag = data_c if c in [0, 1] else np.abs(data_c)
        edges[c] = np.percentile(mag, np.linspace(0, 100, n_bins + 1))
    err_m_sum = np.zeros((n_bins, 8))
    err_p_sum = np.zeros((n_bins, 8))
    cnt = np.zeros((n_bins, 8))
    for dump_id in test_ids:
        curr = load_dump(data_dir, dump_id)
        nxt = load_dump(data_dir, dump_id + 1)
        with torch.no_grad():
            pred = forecast_one_step(model, norm, to_device(curr, device), device).cpu().numpy()[0]
        nt = norm_cpu(to_tensor_cpu(nxt))[0].numpy()
        np_ = norm_cpu(to_tensor_cpu(pred))[0].numpy()
        nc = norm_cpu(to_tensor_cpu(curr))[0].numpy()
        for c in range(8):
            mag = nt[c] if c in [0, 1] else np.abs(nt[c])
            for b in range(n_bins):
                if b < n_bins - 1:
                    m = (mag >= edges[c, b]) & (mag < edges[c, b + 1])
                else:
                    m = (mag >= edges[c, b]) & (mag <= edges[c, b + 1])
                n_in = int(m.sum())
                if n_in == 0:
                    continue
                err_m_sum[b, c] += np.abs(np_[c][m] - nt[c][m]).sum()
                err_p_sum[b, c] += np.abs(nc[c][m] - nt[c][m]).sum()
                cnt[b, c] += n_in
    err_m = np.full((n_bins, 8), np.nan)
    err_p = np.full((n_bins, 8), np.nan)
    valid_b = cnt > 0
    err_m[valid_b] = err_m_sum[valid_b] / cnt[valid_b]
    err_p[valid_b] = err_p_sum[valid_b] / cnt[valid_b]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for c in range(8):
        row, col = (c // 4, c % 4)
        x = np.arange(n_bins)
        axes[row, col].plot(x, err_m[:, c], 'r-o', label='TRON')
        axes[row, col].plot(x, err_p[:, c], 'k--', label='Persistence')
        axes[row, col].set_xticks(x)
        axes[row, col].set_xticklabels([f'{b + 1}' for b in range(n_bins)])
        xlabel = 'Truth Quantile (1=Floor -> 5=Dense)' if c in [0, 1] else 'Truth |x| Quantile (1=Weak -> 5=Strong)'
        axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_ylabel('nMAE')
        axes[row, col].set_title(CHANNEL_NAMES[c])
        axes[row, col].legend(fontsize=7)
        axes[row, col].grid(True, alpha=0.3)
        row_s = '  '.join([f'{err_m[b, c]:.4f}'.rjust(13) if not np.isnan(err_m[b, c]) else 'n/a'.rjust(13) for b in range(n_bins)])
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig19_conditional_error.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'edges': edges, 'model': err_m, 'persistence': err_p, 'counts': cnt})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'conditional_error',_compute,space,starts,native_multi=True)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'conditional_error',compute(ctx,space="physical",starts=starts),"physical",starts)

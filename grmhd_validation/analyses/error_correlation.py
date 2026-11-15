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
    ids = dump_ids(data_dir)
    test_ids = eligible_single_ids(ctx,args.start)[:setting('error_correlation_pairs')]
    if len(test_ids) < 2:
        raise ValueError('Analysis has insufficient valid data: error_correlation')
    err_model = np.zeros((len(test_ids), 8))
    err_data = np.zeros((len(test_ids), 8))
    for i, d in enumerate(test_ids):
        curr, nxt = (load_dump(data_dir, d), load_dump(data_dir, d + 1))
        with torch.no_grad():
            pred = forecast_one_step(model, norm, to_device(curr, device), device).cpu().numpy()[0]
        err_model[i] = rel_l1_per_channel(pred, nxt)
        err_data[i] = rel_l1_per_channel(curr, nxt)
    corr_model = pairwise_masked_corrcoef(err_model)
    corr_data = pairwise_masked_corrcoef(err_data)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, corr, title in [(axes[0], corr_model, 'Model Error Correlation'), (axes[1], corr_data, 'Persistence-Error Correlation')]:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(CHANNEL_NAMES_PLAIN, rotation=45, ha='right')
        ax.set_yticklabels(CHANNEL_NAMES_PLAIN)
        for i in range(8):
            for j in range(8):
                val = corr[i, j]
                txt = f'{val:.2f}' if not np.isnan(val) else 'n/a'
                ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)
        ax.set_title(title + f' ({len(test_ids)} test pairs)')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'error_correlation_matrix.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'model': corr_model, 'persistence': corr_data, 'model_errors': err_model, 'persistence_errors': err_data})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'error_correlation',_compute,space,starts,native_multi=True)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'error_correlation',compute(ctx,space="physical",starts=starts),"physical",starts)

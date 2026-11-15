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
    "Scale-resolved coherence C(k,t) over the rollout — the predictability\n    horizon per scale (the k-resolved analog of Table 4).\n    C = |<F_p F_t*>(theta,phi)|^2 / (<|F_p|^2> <|F_t|^2>) per radial mode k\n    and forecast step t. C ~ 1: the model keeps that scale's phase AND\n    amplitude; C -> 0: the scale has decorrelated (unpredictable)."
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: coherence')
    n_steps = max(2, args.max_steps)
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    coh = {0: [], 5: []}
    tt = []
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pn = states[step]
        pn_n = norm_cpu(to_tensor_cpu(pn))[0].numpy()
        true_n = norm_cpu(to_tensor_cpu(true))[0].numpy()
        for c in [0, 5]:
            fp = np.fft.rfftn(pn_n[c])
            ft = np.fft.rfftn(true_n[c])
            num = np.abs(np.mean(fp * np.conj(ft), axis=(1, 2))) ** 2
            den = np.mean(np.abs(fp) ** 2, axis=(1, 2)) * np.mean(np.abs(ft) ** 2, axis=(1, 2))
            coh[c].append(num / (den + 1e-30))
        tt.append(step)
    if len(tt) >= 2:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, c, name in zip(axes2, [0, 5], ['$\\rho$', '$B^r$']):
            C = np.array(coh[c]).T
            im = ax.pcolormesh(np.array(tt), np.arange(C.shape[0]), C, shading='auto', cmap='magma', vmin=0, vmax=1)
            fig2.colorbar(im, ax=ax)
            ax.set_xlabel('Forecast step t')
            ax.set_ylabel('Radial wavenumber k')
            ax.set_title(f'Coherence C(k,t) — {name}')
        plt.tight_layout()
        path2 = os.path.join(args.save_dir, 'fig13_coherence.png')
        save_figure(fig2, path2)
        plt.close(fig2)
    return sample(**{'steps': tt, 'coherence': coh})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'coherence',_compute,space,starts,native_multi=False)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'coherence',compute(ctx,space="physical",starts=starts),"physical",starts)

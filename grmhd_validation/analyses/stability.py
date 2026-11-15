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
    'Stability diagnostic:\n      Single-step empirical Lipschitz estimate: L = ||f(x+eps) - f(x)||_2 / ||eps||_2\n      over random relative perturbations, evaluated explicitly in normalized space.\n      L <= 1.0 indicates discrete non-expansiveness; L in [1.0, 2.0] is typical\n      for chaotic hyperbolic flows with advection CFL ~ 1.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: stability')
    curr = to_device(init, device)
    with torch.no_grad():
        base = forecast_one_step(model, norm, curr, device)
    rng = np.random.default_rng(args.seed)
    ls = []
    skipped_draws = 0
    n_draws = setting('stability_draws')
    norm_curr = norm(curr)[0]
    nb = norm(base)[0]
    for _ in range(n_draws):
        eps_raw = stability_perturbation(init, rng.standard_normal(init.shape))
        with torch.no_grad():
            pert = forecast_one_step(model, norm, to_device(eps_raw, device), device)
        np_ = norm(pert)[0]
        ne = norm(to_device(eps_raw, device))[0]
        denom = np.linalg.norm((ne-norm_curr).numpy().ravel()) if current()['space']=='physical' else (ne - norm_curr).norm().item()
        if denom < 1e-12:
            skipped_draws += 1
            continue
        numerator=np.linalg.norm((np_-nb).numpy().ravel()) if current()['space']=='physical' else (np_ - nb).norm().item()
        l = float(numerator / denom)
        ls.append(l)
    L_mean = float(np.mean(ls)) if ls else float('nan')
    L_std = float(np.std(ls)) if ls else float('nan')
    if skipped_draws > 0:
        pass
    if not np.isnan(L_mean):
        if L_mean <= 1.0:
            pass
        elif L_mean <= 2.0:
            pass
    return sample(**{'draws': ls, 'mean': L_mean, 'std': L_std, 'skipped_draws': skipped_draws})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'stability',_compute,space,starts,native_multi=False)


def run(ctx):
    starts=production_starts(ctx)
    return emit(ctx,'stability',compute(ctx,space="physical",starts=starts),"physical",starts)

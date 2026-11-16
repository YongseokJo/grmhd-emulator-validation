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
        raise ValueError('Analysis has insufficient valid data: lorentz')
    g_t = gamma_stats(init, norm, device)
    n_clamp = 0
    gmax = 0.0
    gmax_step = 0
    p999_max = 0.0
    last_step = 0
    states = rollout_predictions(model, norm, init, args.max_steps, device, dump_id=args.start)
    clamps = get_rollout_clamps(args.start)
    for step in range(1, args.max_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        g = gamma_stats(states[step], norm, device)
        curr_gmax = float(g.max())
        if curr_gmax > gmax:
            gmax = curr_gmax
            gmax_step = step
        p999_max = max(p999_max, float(np.percentile(g, setting('lorentz_percentile'))))
        n_clamp += clamps.get(step, 0)
        last_step = step
        _records.append(sample(step=step, gamma_max=curr_gmax, gamma_p999=float(np.percentile(g, setting('lorentz_percentile')))))
    return sample(**{'samples': _records, 'max': gmax, 'max_step': gmax_step, 'max_p999': p999_max, 'clamp_count': n_clamp})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'lorentz',_compute,space,starts,native_multi=False)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'lorentz',compute)

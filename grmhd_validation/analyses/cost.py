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
    n_params = sum((p.numel() for p in model.parameters()))
    init = load_dump(data_dir, args.start)
    if init is None:
        init = load_dump(data_dir, dump_ids(data_dir)[0])
    if init is not None:
        curr = to_device(init, device)
        with torch.no_grad():
            for _ in range(setting('cost_warmup_steps')):
                curr = forecast_one_step(model, norm, curr, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        n_time = setting('cost_timed_steps')
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_time):
                curr = forecast_one_step(model, norm, curr, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_time * 1000
        dt = float(geo['dt_dump']) if geo['dt_dump'] else 25.0
    ctx.run_info.setdefault('analysis_runtime', {})['cost'] = {'milliseconds_per_step': ms, 'timed_steps': n_time, 'parameters': n_params}
    return sample(**{'parameters': n_params,'warmup_steps':setting('cost_warmup_steps'),'timed_steps':n_time})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'cost',_compute,space,starts,native_multi=True)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'cost',compute)

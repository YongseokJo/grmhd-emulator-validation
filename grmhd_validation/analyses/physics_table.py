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
        raise ValueError('Analysis has insufficient valid data: physics_table')
    d0 = physical_divb(init, crit)
    nd0, nd0_glob = normalized_divb(init, crit, geo_raw=geo_raw, return_global=True)
    horizons = auto_horizons(args.max_steps)
    max_h = max(horizons)
    states_np = rollout_predictions(model, norm, init, max_h, device, dump_id=args.start)
    states_torch = {0: to_device(init, device)}
    for h in horizons:
        if h - 1 not in states_torch and h - 1 in states_np:
            states_torch[h - 1] = to_device(states_np[h - 1], device)
        if h not in states_torch and h in states_np:
            states_torch[h] = to_device(states_np[h], device)
    for h in horizons:
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        pred_np = states_np[h]
        prev_pred = states_torch[h - 1]
        curr_pred = states_torch[h]
        with torch.no_grad():
            n_ang = mass_angular_count(pred_np,crit)
            M_p = mass_shells(pred_np, crit, norm)[0, 0] / n_ang
            M_t = mass_shells(target, crit, norm)[0, 0] / n_ang
            (E_p, L_p), (E_t, L_t) = (conserved_shells(pred_np, crit, norm), conserved_shells(target, crit, norm))
            eM = float(signed_log_err(M_p, M_t).item())
            eE = float(signed_log_err(E_p[0, 0], E_t[0, 0]).item())
            eL = float(signed_log_err(L_p[0, 0], L_t[0, 0]).item())
            dp = physical_divb(pred_np, crit)
            dt = physical_divb(target, crit)
            ndp, ndp_glob = normalized_divb(pred_np, crit, geo_raw=geo_raw, return_global=True)
            ndt, ndt_glob = normalized_divb(target, crit, geo_raw=geo_raw, return_global=True)
            g_pred = gamma_stats(pred_np, norm, device)
            gmax = float(g_pred.max())
            prev_true = load_dump(data_dir, args.start + h - 1)
            if prev_true is not None:
                flux_t = float(flux_balance_residual_local(to_device(prev_true, device), to_device(target, device), crit, norm, geo_raw, relative=True, seed=args.seed))
                flux_p = float(flux_balance_residual_local(prev_pred, curr_pred, crit, norm, geo_raw, relative=True, seed=args.seed))
            else:
                flux_t = flux_p = float('nan')
        _records.append(sample(step=h, mass_error=eM, energy_error=eE, angular_error=eL, divb_pred=dp, divb_truth=dt, normalized_divb_pred=ndp, normalized_divb_truth=ndt, flux_pred=flux_p, flux_truth=flux_t, gamma_max=gmax))
    return sample(**{'samples': _records, 'initial_divb': d0, 'initial_normalized_divb': nd0, 'initial_normalized_divb_global': nd0_glob})

def compute(ctx, *, space="physical", starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx,'physics_table',_compute,space,starts,native_multi=False)


def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'physics_table',compute)

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
    "The decisive physics test: TRON is NOT trained to conserve mass,\n    yet if it has learned the continuity structure of the PDE, its rollouts\n    should satisfy dM_enc/dt = Mdot(100) - Mdot(r_H) to within the solver's own\n    tolerance. Evaluated in the restricted accretion domain r in [r_H, 100] Rg\n    in full physical code units (dV = dx1*dx2*dx3 for mass, dA = dx2*dx3 for flux)."
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: conservation_closure')
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0
    n_steps = args.max_steps
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    idx_rH = int(np.argmin(np.abs(r_mid - r_H)))
    idx_100 = int(np.argmin(np.abs(r_mid - disk_limit())))
    dV = float(crit.dx * crit.dy * crit.dz)
    dA = float(crit.dy * crit.dz)

    def mass_enc_bare(raw):
        m_sh = mass_shells(raw, crit, norm)[0, 0]
        return total_sum(m_sh[idx_rH:idx_100 + 1])

    def mdot_boundaries_bare(raw):
        mdot_sh = mdot_shells(raw, crit, norm=norm, bare=True)[0, 0].cpu().numpy()
        return (float(mdot_sh[idx_rH]), float(mdot_sh[idx_100]))
    M0_b = mass_enc_bare(init)
    rows_p_b, rows_t_b = ([], [])
    rows_p_ph, rows_t_ph = ([], [])
    M_prev_p_b, M_prev_t_b = (M0_b, M0_b)
    states = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred = states[step]
        with torch.no_grad():
            Mp_b, Mt_b = (mass_enc_bare(pred), mass_enc_bare(true))
            mrH_p_b, m100_p_b = mdot_boundaries_bare(pred)
            mrH_t_b, m100_t_b = mdot_boundaries_bare(true)
        dMdt_p_b = (Mp_b - M_prev_p_b) / dt
        dMdt_t_b = (Mt_b - M_prev_t_b) / dt
        rows_p_b.append((dMdt_p_b, mrH_p_b, m100_p_b))
        rows_t_b.append((dMdt_t_b, mrH_t_b, m100_t_b))
        dMdt_p_ph = dMdt_p_b * dV
        dMdt_t_ph = dMdt_t_b * dV
        mrH_p_ph, m100_p_ph = (mrH_p_b * dA, m100_p_b * dA)
        mrH_t_ph, m100_t_ph = (mrH_t_b * dA, m100_t_b * dA)
        rows_p_ph.append((dMdt_p_ph, mrH_p_ph, m100_p_ph))
        rows_t_ph.append((dMdt_t_ph, mrH_t_ph, m100_t_ph))
        M_prev_p_b, M_prev_t_b = (Mp_b, Mt_b)
    if not rows_p_ph:
        raise ValueError('Analysis has insufficient valid data: conservation_closure')
    R_b = np.array(rows_p_b)
    T_b = np.array(rows_t_b)
    R_ph = np.array(rows_p_ph)
    T_ph = np.array(rows_t_ph)
    imb_p_b = np.abs(R_b[:, 0] - (R_b[:, 2] - R_b[:, 1])) / (np.abs(R_b[:, 1]) + 1e-30)
    imb_t_b = np.abs(T_b[:, 0] - (T_b[:, 2] - T_b[:, 1])) / (np.abs(T_b[:, 1]) + 1e-30)
    rms_p = np.sqrt(0.5 * (R_ph[:, 1] ** 2 + R_ph[:, 2] ** 2)) + 1e-30
    rms_t = np.sqrt(0.5 * (T_ph[:, 1] ** 2 + T_ph[:, 2] ** 2)) + 1e-30
    imb_p_ph = np.abs(R_ph[:, 0] - (R_ph[:, 2] - R_ph[:, 1])) / rms_p
    imb_t_ph = np.abs(T_ph[:, 0] - (T_ph[:, 2] - T_ph[:, 1])) / rms_t
    return sample(**{'pred_bare': R_b, 'truth_bare': T_b, 'pred_physical': R_ph, 'truth_physical': T_ph, 'imbalance_pred_bare': imb_p_b, 'imbalance_truth_bare': imb_t_b, 'imbalance_pred_physical': imb_p_ph, 'imbalance_truth_physical': imb_t_ph})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'conservation_closure', _compute, space, starts, native_multi=False)

def run(ctx):
    starts = production_starts(ctx)
    return emit(ctx, 'conservation_closure', compute(ctx, space='physical', starts=starts), 'physical', starts)

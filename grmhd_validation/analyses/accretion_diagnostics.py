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
    'Accretion-physics diagnostics:\n      (a) Magnetic flux threading Phi(r) = full-sphere int |B^r| sqrt(-g) dtheta dphi per shell.\n      (b) Blandford-Znajek jet power P_jet(r) (funnel: sigma > 1, polar cut |theta-pi/2| > 1.0, outward flux).\n      (c) Dimensionless horizon flux phi_BH = (1/2) Phi(r_H) / sqrt(|Mdot(r_H)|) using hemisphere flux (1/2) Phi(r_H).\n      (d) Mdot(t) light curves at r_H and r = 10 R_g over the rollout, and Pearson correlation.\n      Note: Phi is full-sphere integral; phi_BH uses hemisphere flux (1/2) Phi(r_H).'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: accretion_diagnostics')
    r_mid = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    horizons = auto_horizons(args.max_steps)
    dt = float(geo_raw['dt_dump']) if 'dt_dump' in geo_raw else 25.0
    r10 = int(np.argmin(np.abs(r_mid - 10.0)))
    gamma = float(geo_raw['eos_gamma']) if 'eos_gamma' in geo_raw else 5.0 / 3.0
    a_spin = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    r_H, r_isco = kerr_horizon_isco(a_spin)
    r_H_idx = int(np.argmin(np.abs(r_mid - r_H)))
    states = rollout_predictions(model, norm, init, args.max_steps, device, dump_id=args.start)
    table = []
    saved_phi = {}
    saved_pjet = {}
    saved_phi_bh = {}
    for h in horizons:
        target = load_dump(data_dir, args.start + h)
        if target is None:
            continue
        pred_np = states[h]
        with torch.no_grad():
            Phi_p = magnetic_flux_shells(pred_np, crit, bare=False)[0, 0].cpu().numpy()
            Phi_t = magnetic_flux_shells(target, crit, bare=False)[0, 0].cpu().numpy()
        saved_phi[h] = (Phi_p, Phi_t)
        rel = np.abs(Phi_p - Phi_t) / (np.abs(Phi_t) + 1e-30)
        with torch.no_grad():
            Mp = mdot_shells(pred_np, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
            Mt = mdot_shells(target, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        pjet_p, mask_p = jet_power_shells(pred_np, geo_raw, norm_cpu, gamma, bare=False, return_mask=True)
        pjet_t, mask_t = jet_power_shells(target, geo_raw, norm_cpu, gamma, bare=False, return_mask=True)
        saved_pjet[h] = (pjet_p, pjet_t)
        fill_p = float(mask_p[r_H_idx].mean())
        fill_t = float(mask_t[r_H_idx].mean())
        pj_p_val = pjet_p[r_H_idx]
        pj_t_val = pjet_t[r_H_idx]
        Mp_val = Mp[r_H_idx]
        Mt_val = Mt[r_H_idx]
        phi_bh_p = 0.5 * Phi_p[r_H_idx] / np.sqrt(np.abs(Mp_val) + 1e-30)
        phi_bh_t = 0.5 * Phi_t[r_H_idx] / np.sqrt(np.abs(Mt_val) + 1e-30)
        saved_phi_bh[h] = (phi_bh_p, phi_bh_t)
        rel_ph = np.abs(phi_bh_p - phi_bh_t) / (np.abs(phi_bh_t) + 1e-30)
        eta_truth=rel_jet=np.nan
        if np.abs(pj_t_val)>=1e-12 and np.abs(Mt_val)>=1e-12:
            eta_truth=pj_t_val/(np.abs(Mt_val)+1e-30)
            rel_jet=np.abs(pj_p_val-pj_t_val)/(np.abs(pj_t_val)+1e-30)
        eta_pred=np.nan if np.abs(Mt_val)<1e-12 else 0. if np.abs(Mp_val)<1e-12 else pj_p_val/(np.abs(Mp_val)+1e-30)
        table.append(sample(step=h,funnel_fraction_pred=fill_p,funnel_fraction_truth=fill_t,
                            jet_power_pred=pj_p_val,jet_power_truth=pj_t_val,
                            efficiency_pred=eta_pred,efficiency_truth=eta_truth,
                            jet_relative_error=rel_jet,phi_bh_relative_error=rel_ph))
    n_steps = args.max_steps
    mdot_p = {'r_H': [], 'r10': []}
    mdot_t = {'r_H': [], 'r10': []}
    steps = []
    for step in range(1, n_steps + 1):
        true = load_dump(data_dir, args.start + step)
        if true is None or step not in states:
            break
        pred_np = states[step]
        with torch.no_grad():
            mp = mdot_shells(pred_np, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
            mt = mdot_shells(true, crit, norm=norm, bare=False)[0, 0].cpu().numpy()
        mdot_p['r_H'].append(mp[r_H_idx])
        mdot_p['r10'].append(mp[r10])
        mdot_t['r_H'].append(mt[r_H_idx])
        mdot_t['r10'].append(mt[r10])
        steps.append(step)
    correlations={'r_H':np.nan,'r10':np.nan}
    if len(steps) >= 3:
        steps = np.array(steps)
        for lab in ['r_H', 'r10']:
            a, b = (np.array(mdot_p[lab]), np.array(mdot_t[lab]))
            cc = np.corrcoef(a, b)[0, 1] if np.std(a) != 0 and np.std(b) != 0 else float('nan')
            correlations[lab]=cc
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    r_mask = (r_mid >= r_H) & (r_mid <= disk_limit())
    r_plot = r_mid[r_mask]
    cmap = plt.cm.tab10
    horizon_colors = {1: '#2ca02c', 10: '#ff7f0e', 25: '#d62728', 50: '#9467bd', 100: '#8c564b'}

    def get_h_color(h, idx):
        return horizon_colors.get(h, cmap(idx % 10))
    ax_a = axes[0]
    if horizons[0] in saved_phi:
        ax_a.loglog(r_plot, saved_phi[horizons[0]][1][r_mask], 'b-', linewidth=2.0, label='Truth')
    for idx, h in enumerate(horizons):
        if h in saved_phi:
            col = get_h_color(h, idx)
            ax_a.loglog(r_plot, saved_phi[h][0][r_mask], linestyle='--', color=col, linewidth=1.5, label=f'TRON h={h}')
    ax_a.set_xlim(r_H, disk_limit())
    ax_a.margins(x=0)
    add_isco_line(ax_a, r_isco)
    ax_a.set_xlabel('$r\\;[R_g]$')
    ax_a.set_ylabel('$\\Phi(r) = \\int |B^r| \\sqrt{-g}\\,d\\theta d\\varphi\\;[\\rm code\\;units]$')
    ax_a.set_title('(a) Magnetic Flux Threading (cumulative shell flux)')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)
    ax_b = axes[1]
    all_pj_vals = []
    if horizons[0] in saved_pjet:
        pj_t = saved_pjet[horizons[0]][1][r_mask]
        ax_b.plot(r_plot, pj_t, 'b-', linewidth=2.0, label='Truth')
        all_pj_vals.extend(pj_t)
    for idx, h in enumerate(horizons):
        if h in saved_pjet:
            col = get_h_color(h, idx)
            pj_p = saved_pjet[h][0][r_mask]
            ax_b.plot(r_plot, pj_p, linestyle='--', color=col, linewidth=1.5, label=f'TRON h={h}')
            all_pj_vals.extend(pj_p)
    all_pj_arr = np.array(all_pj_vals) if all_pj_vals else np.array([1.0])
    abs_pj = np.abs(all_pj_arr)
    nz_pj = abs_pj[abs_pj != 0]
    if len(nz_pj):
        p5_pj = float(np.percentile(nz_pj, 5))
        linthresh_exp_pj = int(np.floor(np.log10(max(0.0001, p5_pj))))
        linthresh_pj = float(10 ** linthresh_exp_pj)
    else:
        linthresh_pj = 0.1
        linthresh_exp_pj = -1
    p99_8_pj = float(np.percentile(abs_pj, 99.8))
    y_hi_pj = p99_8_pj * 1.1
    max_decade_pj = int(np.ceil(np.log10(max(y_hi_pj, linthresh_pj * 10))))
    min_decade_pj = linthresh_exp_pj
    pos_ticks_pj = [float(10 ** d) for d in range(min_decade_pj, max_decade_pj + 1)]
    neg_pj = all_pj_arr[all_pj_arr < 0]
    n_neg_pj = int(len(neg_pj))
    neg_ticks_pj = []
    if n_neg_pj > 0:
        most_neg_pj = float(np.min(neg_pj))
        y_lo_pj = min(most_neg_pj * 1.1, -linthresh_pj * 1.1)
        neg_ticks_pj = [-float(10 ** d) for d in range(min_decade_pj, max_decade_pj + 1)]
    else:
        most_neg_pj = 0.0
        y_lo_pj = float(10 ** min_decade_pj)
    yticks_pj = [t for t in sorted(neg_ticks_pj + [0.0] + pos_ticks_pj) if y_lo_pj <= t <= y_hi_pj]
    ax_b.set_xscale('log')
    ax_b.set_yscale('symlog', linthresh=linthresh_pj, linscale=1.0)
    ax_b.set_yticks(yticks_pj)
    ax_b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_b.set_ylim(y_lo_pj, y_hi_pj)
    ax_b.set_xlim(r_H, disk_limit())
    ax_b.margins(x=0)
    add_isco_line(ax_b, r_isco)
    ax_b.set_xlabel('$r\\;[R_g]$')
    ax_b.set_ylabel('$P_{\\rm jet}(r) = -\\int_{\\rm funnel} T^r{}_t \\sqrt{-g}\\,d\\theta d\\varphi\\;[\\rm code\\;units]$')
    ax_b.set_title('(b) Blandford-Znajek Jet Power (signed net flux)')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)
    max_pj = float(np.max(abs_pj))
    ax_c = axes[2]
    h_list = [h for h in horizons if h in saved_phi_bh]
    if h_list:
        phi_t_vals = [saved_phi_bh[h][1] for h in h_list]
        phi_p_vals = [saved_phi_bh[h][0] for h in h_list]
        ax_c.plot(h_list, phi_t_vals, 'b-o', linewidth=1.5, markersize=5, label='Truth $\\phi_{\\rm BH}(r_H)$')
        ax_c.plot(h_list, phi_p_vals, 'r--s', linewidth=1.5, markersize=5, label='TRON $\\phi_{\\rm BH}(r_H)$')
        all_phi = phi_t_vals + phi_p_vals + [15.0]
        ax_c.set_ylim(0, max(all_phi) * 1.25)
        max_h = max(h_list) if h_list else args.max_steps
        h_margin = max(1.0, max_h * 0.02)
        ax_c.set_xlim(0, max_h + h_margin)
        ax_c.set_xlabel('Forecast Horizon $h$ [steps]')
        ax_c.set_ylabel('$\\phi_{\\rm BH} = \\frac{1}{2}\\Phi(r_H)/\\sqrt{|\\dot{M}(r_H)|}$')
        ax_c.set_title('(c) Dimensionless Horizon Flux $\\phi_{\\rm BH}$')
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)
    fig.suptitle('Accretion & Jet Diagnostics', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.save_dir, 'fig18_accretion_diagnostics.png')
    save_figure(fig, path)
    plt.close(fig)
    return sample(**{'r':r_mid,'table':table,'mdot_correlation':correlations,'flux_profiles': saved_phi, 'jet_power_profiles': saved_pjet, 'phi_bh': saved_phi_bh, 'mdot_pred': mdot_p, 'mdot_truth': mdot_t, 'steps': steps, 'horizon_index': r_H_idx, 'r10_index': r10})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    if space=='physical':
        from ._transport import diagnostics
        selected=[ctx.starts[0].rollout[0]] if starts is None else starts
        one=lambda ctx,args,*unused:diagnostics(ctx,args.start)
        return execute_core(ctx,'accretion_diagnostics',one,space,selected)
    return execute_core(ctx, 'accretion_diagnostics', _compute, space, starts, native_multi=False)

def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'accretion_diagnostics',compute)

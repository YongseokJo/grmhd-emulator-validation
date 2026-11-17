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
    if current()['space']=='physical':
        from ._profiles import mean_profiles as physical_profiles
        return physical_profiles(ctx,args.start)
    geo, geo_raw = geometry_views(ctx.geometry)
    _records = []
    'Time+phi-averaged radial physics profiles, pred vs truth:\n      (a) rotation curve Omega(r) = u^phi/u^0 in the equatorial band vs the\n          Kerr Keplerian curve 1/(r^{3/2}+a) — does the model learn GR orbital\n          mechanics? (b) vertical density structure rho(theta) at r=10/30/100 —\n          the disk scale height, (c) plasma beta p_gas/p_mag in the equatorial\n          band, (d) magnetization sigma = b^2/rho in the equatorial band.'
    init = load_dump(data_dir, args.start)
    if init is None:
        raise ValueError('Analysis has insufficient valid data: mean_profiles')
    r3 = geo_raw['r'][:, geo_raw['r'].shape[1] // 2, 0]
    h3 = geo_raw['h'][geo_raw['h'].shape[0] // 2, :, 0]
    Nr = r3.shape[0]
    a = float(geo_raw['spin']) if 'spin' in geo_raw else 0.0
    gamma = float(geo_raw['eos_gamma']) if 'eos_gamma' in geo_raw else 5.0 / 3.0
    req_steps = max(1, args.max_steps // setting('mean_step_divisor'))
    h = 0
    while load_dump(data_dir, args.start + h + 1) is not None and h < req_steps:
        h += 1
    n_steps = h
    if n_steps == 0:
        raise ValueError('Analysis has insufficient valid data: mean_profiles')
    r_dict = rollout_predictions(model, norm, init, n_steps, device, dump_id=args.start)
    states = [init] + [r_dict[s] for s in range(1, n_steps + 1)]
    n_t = len(states)
    pred = np.mean(np.stack(states), axis=0)
    tr = np.mean(np.stack([load_dump(data_dir, args.start + i) for i in range(n_t)]), axis=0)
    band_theta = np.abs(h3 - np.pi / 2) < setting('mean_theta_halfwidth')
    with torch.no_grad():
        ucon_p = norm.ucon_of(to_device(pred, device)[:, 2:5])[0].cpu().numpy()
        ucon_t = norm.ucon_of(to_device(tr, device)[:, 2:5])[0].cpu().numpy()
    omega_p = np.mean(ucon_p[3][:, band_theta, :] / ucon_p[0][:, band_theta, :], axis=(1, 2))
    omega_t = np.mean(ucon_t[3][:, band_theta, :] / ucon_t[0][:, band_theta, :], axis=(1, 2))
    omega_k = kerr_keplerian_omega(r3, a)
    r_sel = setting('mean_profile_radii')
    r_idx = [int(np.argmin(np.abs(r3 - rr))) for rr in r_sel]
    rho_t_theta = np.abs(tr[0][r_idx, :, :]).mean(axis=-1)
    rho_p_theta = np.abs(pred[0][r_idx, :, :]).mean(axis=-1)
    rho_t_norm = rho_t_theta / rho_t_theta.max(axis=1, keepdims=True)
    rho_p_norm = rho_p_theta / rho_p_theta.max(axis=1, keepdims=True)

    def beta_sigma(x):
        p_gas = (gamma - 1.0) * x[1]
        b2 = bsq_full(x, geo_raw, norm_cpu)
        beta = p_gas / (0.5 * b2 + 1e-30)
        sigma = b2 / (x[0] + 1e-30)
        return (np.log10(np.maximum(beta[:, band_theta, :], 1e-30)), np.log10(np.maximum(sigma[:, band_theta, :], 1e-30)))
    lb_p, ls_p = beta_sigma(pred)
    lb_t, ls_t = beta_sigma(tr)
    beta_p = np.median(lb_p, axis=(1, 2))
    beta_t = np.median(lb_t, axis=(1, 2))
    sigma_p = np.median(ls_p, axis=(1, 2))
    sigma_t = np.median(ls_t, axis=(1, 2))
    r_H, r_isco = kerr_horizon_isco(a)
    r_mask_100 = (r3 >= r_H) & (r3 <= disk_limit())
    r3_plot = r3[r_mask_100]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.plot(r3_plot, omega_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, omega_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.plot(r3_plot, omega_k[r_mask_100], 'k:', linewidth=1, label=f'Kerr Keplerian (a={a:.3f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r_H, disk_limit())
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel('$r\\;[R_g]$')
    ax.set_ylabel('$\\Omega = u^\\varphi/u^0\\ [c/R_g]\\ (u^0\\equiv u^t\\ \\mathrm{coordinate})$')
    ax.set_title(f'Rotation Curve (equatorial band)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    all_omega = np.concatenate([omega_t[r_mask_100], omega_p[r_mask_100], omega_k[r_mask_100]])
    pos_omega = all_omega[all_omega > 0]
    if len(pos_omega) > 0:
        o_min = float(np.min(pos_omega))
        o_max = float(np.max(pos_omega))
        y_lo_omega = float(10 ** np.floor(np.log10(o_min)))
        y_hi_omega = float(10 ** np.ceil(np.log10(o_max)))
        ax.set_ylim(y_lo_omega, y_hi_omega)
    ax = axes[0, 1]
    theta_deg = np.degrees(h3)
    truth_lines, truth_labels = ([], [])
    fc_lines, fc_labels = ([], [])
    for i, rr in enumerate(r_sel):
        col = f'C{i}'
        lp, = ax.plot(theta_deg, rho_p_norm[i], '-', color=col, linewidth=1.5, alpha=0.65, zorder=2)
        lt, = ax.plot(theta_deg, rho_t_norm[i], ':', color=col, linewidth=2.0, alpha=1.0, zorder=3)
        truth_lines.append(lt)
        truth_labels.append(f'Truth r={rr:.0f}')
        fc_lines.append(lp)
        fc_labels.append(f'TRON r={rr:.0f}')
    ax.set_yscale('log')
    ax.set_xlim(0, 180)
    ax.margins(x=0)
    ax.set_xlabel('$\\theta\\;[\\mathrm{degrees}]$')
    ax.set_ylabel('$\\rho(\\theta)/\\rho_{max}$')
    ax.set_title(f'Vertical Disk Structure (phi-averaged)')
    ax.legend(truth_lines + fc_lines, truth_labels + fc_labels, loc='lower center', fontsize=7.5, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(r3_plot, beta_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, beta_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.set_xscale('log')
    ax.set_xlim(r_H, disk_limit())
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel('$r\\;[R_g]$')
    ax.set_ylabel('$\\log_{10}\\beta = p_{gas}/p_{mag}$')
    ax.set_title(f'Plasma Beta Profile (equatorial band)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    b_lo, b_hi = data_ylim([beta_t[r_mask_100], beta_p[r_mask_100]], p_low=1.0, p_high=99.0, margin=1.15)
    b_lo = min(b_lo, -0.2)
    b_hi = max(b_hi, 0.2)
    ax.set_ylim(b_lo, b_hi)
    ax = axes[1, 1]
    ax.plot(r3_plot, sigma_t[r_mask_100], 'b-', linewidth=1.5, label=f'Truth')
    ax.plot(r3_plot, sigma_p[r_mask_100], 'r--', linewidth=1.5, label=f'TRON')
    ax.set_xscale('log')
    ax.set_xlim(r_H, disk_limit())
    ax.margins(x=0)
    add_isco_line(ax, r_isco)
    ax.set_xlabel('$r\\;[R_g]$')
    ax.set_ylabel('$\\log_{10}\\sigma = b^2/\\rho$')
    ax.set_title(f'Magnetization Profile (equatorial band)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    s_lo, s_hi = data_ylim([sigma_t[r_mask_100], sigma_p[r_mask_100]], p_low=1.0, p_high=99.0, margin=1.15)
    ax.set_ylim(s_lo, s_hi)
    fig.suptitle(f'Mean Radial Physics Profiles', fontsize=13)
    plt.tight_layout()
    path10 = os.path.join(args.save_dir, 'fig10_mean_profiles.png')
    save_figure(fig, path10)
    plt.close(fig)
    r_mask_full = (r3 >= r_H) & (r3 <= 2000.0)
    r3_full = r3[r_mask_full]
    fig_full, axes_full = plt.subplots(1, 3, figsize=(18, 5))
    ax_om = axes_full[0]
    ax_om.plot(r3_full, omega_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_om.plot(r3_full, omega_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_om.plot(r3_full, omega_k[r_mask_full], 'k:', linewidth=1, label=f'Kerr Keplerian (a={a:.3f})')
    ax_om.set_xscale('log')
    ax_om.set_yscale('log')
    ax_om.set_xlim(r_H, 2000.0)
    ax_om.margins(x=0)
    add_isco_line(ax_om, r_isco)
    ax_om.set_xlabel('$r\\;[R_g]$')
    ax_om.set_ylabel('$\\Omega = u^\\varphi/u^0\\ [c/R_g]\\ (u^0\\equiv u^t\\ \\mathrm{coordinate})$')
    ax_om.set_title(f'Rotation Curve (equatorial band, full domain)')
    ax_om.legend(fontsize=8)
    ax_om.grid(True, alpha=0.3)
    pos_om_f = np.concatenate([omega_t[r_mask_full], omega_p[r_mask_full], omega_k[r_mask_full]])
    pos_om_f = pos_om_f[pos_om_f > 0]
    if len(pos_om_f) > 0:
        y_lo_om_f = float(10 ** np.floor(np.log10(np.min(pos_om_f))))
        y_hi_om_f = float(10 ** np.ceil(np.log10(np.max(pos_om_f))))
        ax_om.set_ylim(y_lo_om_f, y_hi_om_f)
    ax_b = axes_full[1]
    ax_b.plot(r3_full, beta_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_b.plot(r3_full, beta_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_b.set_xscale('log')
    ax_b.set_xlim(r_H, 2000.0)
    ax_b.margins(x=0)
    add_isco_line(ax_b, r_isco)
    ax_b.set_xlabel('$r\\;[R_g]$')
    ax_b.set_ylabel('$\\log_{10}\\beta = p_{\\rm gas}/p_{\\rm mag}$')
    ax_b.set_title(f'Plasma Beta Profile (equatorial band, full domain)')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)
    b_lo_f, b_hi_f = data_ylim([beta_t[r_mask_full], beta_p[r_mask_full]], p_low=1.0, p_high=99.0, margin=1.15)
    b_lo_f = min(b_lo_f, -0.2)
    b_hi_f = max(b_hi_f, 0.2)
    ax_b.set_ylim(b_lo_f, b_hi_f)
    ax_s = axes_full[2]
    ax_s.plot(r3_full, sigma_t[r_mask_full], 'b-', linewidth=1.5, label=f'Truth')
    ax_s.plot(r3_full, sigma_p[r_mask_full], 'r--', linewidth=1.5, label=f'TRON')
    ax_s.set_xscale('log')
    ax_s.set_xlim(r_H, 2000.0)
    ax_s.margins(x=0)
    add_isco_line(ax_s, r_isco)
    ax_s.set_xlabel('$r\\;[R_g]$')
    ax_s.set_ylabel('$\\log_{10}\\sigma = b^2/\\rho$')
    ax_s.set_title(f'Magnetization Profile (equatorial band, full domain)')
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.3)
    s_lo_f, s_hi_f = data_ylim([sigma_t[r_mask_full], sigma_p[r_mask_full]], p_low=1.0, p_high=99.0, margin=1.15)
    ax_s.set_ylim(s_lo_f, s_hi_f)
    fig_full.suptitle(f'Mean Radial Physics Profiles — Full Domain (r <= 2000 Rg)', fontsize=13)
    plt.tight_layout()
    path10_full = os.path.join(args.save_dir, 'fig10_mean_profiles_full.png')
    save_figure(fig_full, path10_full)
    plt.close(fig_full)
    return sample(**{'r': r3, 'theta': h3, 'omega_pred': omega_p, 'omega_truth': omega_t, 'omega_keplerian': omega_k, 'rho_theta_pred': rho_p_norm, 'rho_theta_truth': rho_t_norm, 'beta_log10_pred': beta_p, 'beta_log10_truth': beta_t, 'sigma_log10_pred': sigma_p, 'sigma_log10_truth': sigma_t, 'n_steps': n_steps})

def compute(ctx, *, space='physical', starts=None):
    """Numerical core: explicit space and one/all starts for source regression."""
    return execute_core(ctx, 'mean_profiles', _compute, space, starts, native_multi=False)

def run(ctx):
    from ._final import run as final_run
    return final_run(ctx,'mean_profiles',compute)

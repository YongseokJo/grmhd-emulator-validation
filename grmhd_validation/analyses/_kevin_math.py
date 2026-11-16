"""Port: Fall26/eval/evaluate_forecast_ds.py, main@ebcc218.
Authors: Kevin Fortune; Sachin Venkatesh (TRON evaluation originals).
Adapted to common physical adapter inputs, frozen windows and structured outputs.
Metric space and start selection are explicit for faithful legacy regression.
"""
import numpy as np
import torch
from ._analysis_state import CONFIG,current,setting
from ._analysis_io import to_device,to_tensor_cpu,save_figure
from ._geometry_physics import GeometryCriterion as PhysicsLoss3D
from . import _physical as physical

def disk_limit():return current()['ctx'].constants['active_disk']['r_max']
_FLUX_COUNT_PRINTED=False

def auto_horizons(max_steps):
    if current()['space']=='physical':return list(current()['ctx'].protocol['report_steps'])
    if max_steps <= 20:
        horizons = [1, max(1, max_steps // 4), max(1, max_steps // 2), max_steps]
    elif max_steps <= 50:
        horizons = [1, 10, 25, max_steps]
    elif max_steps <= 100:
        horizons = [1, 10, 25, 50, max_steps]
    else:
        horizons = [1, 10, 25, 50, 100, max_steps]
    return sorted(set([h for h in horizons if 1 <= h <= max_steps]))

def dynamic_xticks(max_val):
    if max_val <= 20:
        return list(range(1, max_val + 1))
    elif max_val <= 50:
        return list(range(1, max_val + 1, 5))
    else:
        return list(range(1, max_val + 1, 10))

def active_disk_mask(geo_raw, rmax=None):
    """Active disk boolean mask: (r <= rmax) & (|theta - pi/2| < 0.5)."""
    rmax = disk_limit() if rmax is None else rmax
    r = geo_raw['r']
    h = geo_raw['h']
    return (r <= rmax) & (np.abs(h - np.pi / 2.0) < current()['ctx'].constants['active_disk']['theta_halfwidth'])

def nmae_per_channel(pred_raw, true_raw, norm_cpu):
    """Normalized-space mean absolute error per channel (the training metric)."""
    if current()['space']=='physical':
        return np.mean(np.abs(physical.array(pred_raw)-physical.array(true_raw)).reshape(8,-1),axis=1)
    with torch.no_grad():
        p = norm_cpu(to_tensor_cpu(pred_raw))
        t = norm_cpu(to_tensor_cpu(true_raw))
        return (p - t).abs().mean(dim=(2, 3, 4)).squeeze(0).numpy()

def rel_l1_per_channel(pred_raw, true_raw, reference_scale=None):
    """Physical-space relative L1 per channel."""
    if reference_scale is None and current()['space']=='physical':
        reference_scale=current()['scales']['mean_abs']
    rel = np.zeros(8)
    for c in range(8):
        denom = np.abs(true_raw[c]).mean() if reference_scale is None else reference_scale[c]
        rel[c] = np.nan if denom < current()['ctx'].constants['rel_l1']['denominator_min'] else np.abs(pred_raw[c] - true_raw[c]).mean() / denom
    return rel

def mass_shells(raw_t, crit, norm):
    """M(r) = sum_{theta,phi} rho u^0 sqrt(-g) [bare θφ-sum (no /n_ang, no dx); Fig 5/Table 2 normalize by n_ang at call sites; Table 5 intentionally bare on both sides]."""
    if current()['space']=='physical':
        return torch.from_numpy(physical.mass(raw_t,current()['geometry']))
    x = to_device(raw_t, crit.gdet.device)
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho'])
    u0 = norm.u0_of(x[:, 2:5])
    return torch.sum(rho * u0 * crit.gdet, dim=(3, 4))


def mass_angular_count(raw,crit):
    """Physical cells in production; original geometry-buffer count in legacy."""
    if current()['space']=='loss':return float(crit.gdet.shape[-2]*crit.gdet.shape[-1])
    return float(raw.shape[-2]*raw.shape[-1])

def conserved_shells(raw_t, crit, norm):
    """Per-shell angular-mean conserved densities:
    E(r) = rest-mass-subtracted binding-energy density: (1/N_ang) sum_{theta,phi} sqrt(-g) (T^t_0 + rho u^0)
    L(r) = (1/N_ang) sum_{theta,phi} sqrt(-g) T^t_phi in float64.
    Self-contained out-of-sample physical transport diagnostic (not in TRON training loss)."""
    if current()['space']=='physical':
        return tuple(torch.from_numpy(v) for v in physical.conserved(raw_t,current()['geometry']))
    x = to_device(raw_t, crit.gdet.device).double()
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho'])
    ug = crit._safe_rho_ug(x[:, 1:2], CONFIG['floor_ug'], CONFIG['ceiling_ug'])
    b1, b2, b3 = (x[:, 5:6], x[:, 6:7], x[:, 7:8])
    gdet_d = crit.gdet.double()
    ucon = norm.ucon_of(x[:, 2:5]).double()
    u0 = ucon[:, 0:1]
    alpha = 1.0 / torch.sqrt(-crit.gcon[0, 0, 0, 0].double() + 1e-300)
    gamma = alpha * u0
    eos_gamma = crit.eos_gamma if crit.eos_gamma is not None else float(CONFIG.get('eos_gamma') or 5.0 / 3.0)
    p = (eos_gamma - 1.0) * ug
    w = rho + ug + p
    g = crit.gcov[0, 0].double()
    ucov = torch.einsum('bi...,ij...->bj...', ucon, g)
    Bcon = torch.cat([torch.zeros_like(u0), alpha * b1, alpha * b2, alpha * b3], dim=1)
    u_dot_B = torch.einsum('bi...,bi...->b...', ucov, Bcon).unsqueeze(1)
    bcon = (Bcon + ucon * u_dot_B) / gamma
    bcov = torch.einsum('bi...,ij...->bj...', bcon, g)
    n_dot_b = -alpha * bcon[:, 0:1]
    bsq = torch.einsum('bi...,bi...->b...', bcov, bcon).unsqueeze(1)
    U0 = (gamma * (w + bsq) * ucov[:, 0:1] - (p + bsq / 2.0) * -alpha + n_dot_b * bcov[:, 0:1]) / alpha + rho * u0
    U3 = (gamma * (w + bsq) * ucov[:, 3:4] + n_dot_b * bcov[:, 3:4]) / alpha
    n_ang = float(x.shape[3] * x.shape[4])
    return (torch.sum(U0 * gdet_d, dim=(3, 4)) / n_ang, torch.sum(U3 * gdet_d, dim=(3, 4)) / n_ang)

def signed_log_err(pred_shell, true_shell):
    """mean |sl(pred) - sl(true)| — shell error in signed-log space."""
    if current()['space']=='physical':
        p=physical.array(pred_shell);t=physical.array(true_shell)
        value=np.mean(np.abs(np.sign(p)*np.log1p(np.abs(p))-np.sign(t)*np.log1p(np.abs(t))))
        return torch.tensor(value,dtype=torch.float64)
    return torch.mean(torch.abs(crit_sl(pred_shell) - crit_sl(true_shell)))

def crit_sl(x):
    return PhysicsLoss3D._signed_log(x)

def physical_divb(raw_t, crit):
    """Volume-averaged physical solenoidal residual = mean(|div B|) [code units].
    Exact training loss formulation from uno_simple_ds.py (TRON) and its in-training monitor."""
    x = to_device(raw_t, crit.gdet.device)
    divb = crit.calc_divB(x[:, 5:6], x[:, 6:7], x[:, 7:8])
    if current()['space']=='physical':return float(np.mean(np.abs(physical.array(divb))))
    return float(torch.mean(torch.abs(divb)).item())

def normalized_divb(raw_t, crit, geo_raw=None, return_global=False):
    """Normalized solenoidal residual = volume-mean(|divB| * Delta x_local / |B|_local).
    Delta x_local = (gdet * dx * dy * dz)^(1/3).
    Denominator is max(|B|_local, 1e-3 * median(|B|)) + 1e-12 (|B|_local is coordinate magnitude).
    If geo_raw is provided, evaluates over the active disk pool (r <= R_DISK_MAX, |theta - pi/2| < 0.5,
    rho > 100 * floor_rho) with boundary trimming 1:-1 in r and theta (same triple condition as flux)."""
    if current()['space']=='physical':
        return physical.normalized_divergence(raw_t,current()['geometry'],current()['ctx'].constants,geo_raw,return_global)
    x = to_device(raw_t, crit.gdet.device)
    divb = crit.calc_divB(x[:, 5:6], x[:, 6:7], x[:, 7:8])
    dx_local = torch.pow(crit.gdet * crit.dx * crit.dy * crit.dz, 1.0 / 3.0)
    B_local = torch.sqrt(x[:, 5:6] ** 2 + x[:, 6:7] ** 2 + x[:, 7:8] ** 2)
    b_med = torch.median(B_local)
    denom = torch.maximum(B_local, 0.001 * b_med) + 1e-12
    ndb_field = torch.abs(divb) * dx_local / denom
    ndb_global = float(torch.mean(ndb_field).item())
    if geo_raw is not None:
        ndb_int = ndb_field[0, 0, 1:-1, 1:-1, :].cpu().numpy()
        r_grid = geo_raw['r'][1:-1, 1:-1, :]
        h_grid = geo_raw['h'][1:-1, 1:-1, :]
        rho_safe = crit._safe_rho_ug(x[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho'])
        rho_np = rho_safe[0, 0, 1:-1, 1:-1, :].cpu().numpy()
        floor_rho = float(CONFIG.get('floor_rho', 0.0001))
        geo_trim = {'r': r_grid, 'h': h_grid}
        disk_pool = active_disk_mask(geo_trim, rmax=disk_limit()) & (rho_np > 100.0 * floor_rho)
        if disk_pool.sum() > 0:
            ndb_val = float(np.mean(ndb_int[disk_pool]))
        else:
            ndb_val = ndb_global
    else:
        ndb_val = ndb_global
    if return_global:
        return (ndb_val, ndb_global)
    return ndb_val

def flux_balance_residual_local(x_k, x_k1, crit, norm, geo_raw, relative=True, n_cells=None, seed=0):
    """LOCAL, cellwise truth-free continuity residual between consecutive states:
        M_cell = rho u^0 sqrt(-g)
        F_cell^i = rho u^i sqrt(-g)  (using coordinate velocity u^i)
        R_cell = M_cell(k+1) - M_cell(k) + dt_dump * d_i F_cell^i
    Sampled from active disk pool (r <= 100, |theta - pi/2| < 0.5, rho > 100 * floor_rho)
    with boundary trimming 1:-1 in both r (axis 2) and theta (axis 3).
    Cellwise relative error: mean(|R| / (|M| + eps))."""
    if current()['space']=='physical':
        count=n_cells if n_cells is not None else setting('flux_local_cells')
        return physical.flux_residual(x_k,x_k1,current()['geometry'],current()['ctx'].constants,geo_raw,relative,count,seed)
    dt = crit.dt_dump if crit.dt_dump is not None else float(CONFIG.get('dt_dump', 25.0))
    n_cells = n_cells if n_cells is not None else CONFIG.get('flux_local_cells', 10000)
    rho_k = crit._safe_rho_ug(x_k[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho']).double()
    rho_k1 = crit._safe_rho_ug(x_k1[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho']).double()
    ucon_k = norm.ucon_of(x_k[:, 2:5]).double()
    ucon_k1 = norm.ucon_of(x_k1[:, 2:5]).double()
    gdet_d = crit.gdet.double()
    M_k = rho_k * ucon_k[:, 0:1] * gdet_d
    M_k1 = rho_k1 * ucon_k1[:, 0:1] * gdet_d
    f_k = crit.calc_divB(rho_k * ucon_k[:, 1:2], rho_k * ucon_k[:, 2:3], rho_k * ucon_k[:, 3:4]) * gdet_d
    f_k1 = crit.calc_divB(rho_k1 * ucon_k1[:, 1:2], rho_k1 * ucon_k1[:, 2:3], rho_k1 * ucon_k1[:, 3:4]) * gdet_d
    R_cell = M_k1 - M_k + dt * (f_k + f_k1) / 2.0
    R_int = R_cell[0, 0, 1:-1, 1:-1, :].cpu().numpy()
    M_int = M_k1[0, 0, 1:-1, 1:-1, :].cpu().numpy()
    r_grid = geo_raw['r'][1:-1, 1:-1, :]
    h_grid = geo_raw['h'][1:-1, 1:-1, :]
    rho_np = rho_k1[0, 0, 1:-1, 1:-1, :].cpu().numpy()
    floor_rho = float(CONFIG.get('floor_rho', 0.0001))
    geo_trim = {'r': r_grid, 'h': h_grid}
    disk_pool = active_disk_mask(geo_trim, rmax=disk_limit()) & (rho_np > 100.0 * floor_rho)
    n_qual = int(disk_pool.sum())
    global _FLUX_COUNT_PRINTED
    if not _FLUX_COUNT_PRINTED:
        _FLUX_COUNT_PRINTED = True
    if n_qual > 0:
        R_s = R_int[disk_pool]
        M_s = M_int[disk_pool]
    else:
        R_s = R_int.ravel()
        M_s = M_int.ravel()
    if n_cells is not None and len(R_s) > n_cells:
        rng_f = np.random.default_rng(seed)
        idx = rng_f.choice(len(R_s), size=n_cells, replace=False)
        R_s = R_s[idx]
        M_s = M_s[idx]
    if relative:
        return float(np.mean(np.abs(R_s) / (np.abs(M_s) + 1e-10)))
    return float(np.mean(np.abs(R_s)))

def gamma_stats(raw_t, norm, device):
    """Pre-clamp Lorentz factor stats: gamma_of_raw(utilde^i)."""
    x = to_device(raw_t, device)
    g = norm.gamma_of_raw(x[:, 2:5]).double().cpu().numpy().ravel()
    return g

def radial_spectrum(raw_t, crit, channel):
    """Windowed (Hann r x theta, like the spectral loss) radial power spectrum:
    Mean-subtracted spatial field, rfftn over all three axes, average over (theta, phi) ->
    spectrum folded onto positive radial mode index |k| in [0, Nr//2]. Returns (k, power)."""
    if current()['space']=='physical':return physical.radial_spectrum(raw_t,channel)
    x = to_device(raw_t, crit.gdet.device)
    field = x[:, channel:channel + 1]
    field = field - torch.mean(field, dim=(-3, -2, -1), keepdim=True)
    ch = field * crit.spec_win
    fft = torch.fft.rfftn(ch.contiguous().to(torch.float32), dim=[-3, -2, -1])
    spec_full = (torch.abs(fft) ** 2).mean(dim=(-2, -1))[0, 0].cpu().numpy()
    Nr = len(spec_full)
    N_half = Nr // 2
    spec_folded = np.zeros(N_half + 1, dtype=spec_full.dtype)
    spec_folded[0] = spec_full[0]
    spec_folded[1:N_half] = spec_full[1:N_half] + spec_full[Nr - 1:N_half:-1]
    spec_folded[N_half] = spec_full[N_half]
    k = np.arange(N_half + 1)
    return (k, spec_folded)

def spectral_slope(k, E):
    if len(k) < 10 or len(E) < 10:
        return (0.0, 0.0)
    n = len(k)
    lo, hi = (int(n*part) for part in setting('spectrum_fit_fraction'))
    if lo >= hi:
        return (0.0, 0.0)
    lk = np.log10(k[lo:hi])
    lE = np.log10(np.maximum(E[lo:hi], 1e-30))
    poly = np.polyfit(lk, lE, 1)
    slope = float(poly[0])
    y_fit = poly[0] * lk + poly[1]
    ss_tot = np.sum((lE - np.mean(lE)) ** 2)
    ss_res = np.sum((lE - y_fit) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30)) if ss_tot > 1e-30 else 0.0
    return (slope, r2)

def mdot_shells(raw_t, crit, norm, bare=True):
    """Mdot(r) = -sum_{theta,phi} rho u^r sqrt(-g) (accretion rate per shell;
    the F(r) flux of the continuity machinery; inflow positive). Reconstructs
    coordinate u^r from projected primitive utilde^i via norm.ucon_of.
    When bare=False, multiplies by dA = crit.dy * crit.dz for physical code units."""
    if current()['space']=='physical':
        geometry=current()['geometry'];x=physical.batched(raw_t)
        velocity=physical.ucon(x[:,2:5],geometry.gcov,geometry.gcon)
        res=-np.sum(x[:,0:1]*velocity[:,1:2]*physical.array(geometry.gdet),axis=(3,4))
        if not bare:res=res*(crit.dy*crit.dz)
        return torch.from_numpy(res)
    x = to_device(raw_t, crit.gdet.device)
    rho = crit._safe_rho_ug(x[:, 0:1], CONFIG['floor_rho'], CONFIG['ceiling_rho'])
    ucon = norm.ucon_of(x[:, 2:5])
    ur = ucon[:, 1:2]
    res = -torch.sum(rho * ur * crit.gdet, dim=(3, 4))
    if not bare:
        res = res * (crit.dy * crit.dz)
    return res

def magnetic_flux_shells(raw_t, crit, bare=True):
    """Phi(r) = sum_{theta,phi} |B^r| sqrt(-g) — the magnetic flux threading
    each spherical shell. The near-horizon value Phi(r_H) is the single most
    important BH-observable number (MAD/SANE ordering; jet power ~ Phi^2).
    When bare=False, multiplies by dA = crit.dy * crit.dz for physical code units."""
    if current()['space']=='physical':
        x=physical.batched(raw_t)
        res=np.sum(np.abs(x[:,5:6])*physical.array(current()['geometry'].gdet),axis=(3,4))
        if not bare:res=res*(crit.dy*crit.dz)
        return torch.from_numpy(res)
    x = to_device(raw_t, crit.gdet.device)
    res = torch.sum(torch.abs(x[:, 5:6]) * crit.gdet, dim=(3, 4))
    if not bare:
        res = res * (crit.dy * crit.dz)
    return res

def pdf_on_shared_bins(pred_c, true_c, n_bins=None, log_scale=False, channel=None):
    """PDFs of pred/true on the SAME bin edges (percentiles from the combined
    sample) — the old version used per-distribution edges, making the JS
    divergence meaningless. Returns (bin_centers, pdf_pred, pdf_true, bin_widths)."""
    n_bins=setting('histogram_bins') if n_bins is None else n_bins
    if current()['space']=='physical':pred_c,true_c=physical.array(pred_c),physical.array(true_c)
    if current()['space']=='physical' and channel is not None:
        from ._reference_statistics import histogram_edges
        edges=histogram_edges(current()['ctx'],channel)
        p=pred_c;t=true_c
        if log_scale:p=np.maximum(np.abs(p),setting('epsilon'));t=np.maximum(np.abs(t),setting('epsilon'))
        hp,_=np.histogram(np.clip(p,edges[0],edges[-1]),bins=edges,density=True)
        ht,_=np.histogram(np.clip(t,edges[0],edges[-1]),bins=edges,density=True)
        centers=np.sqrt(edges[:-1]*edges[1:]) if log_scale else (edges[:-1]+edges[1:])/2
        return centers,hp,ht,np.diff(edges)
    flat = np.concatenate([pred_c.ravel(), true_c.ravel()])
    if log_scale:
        abs_f = np.maximum(np.abs(flat), 1e-30)
        p1, p99 = np.percentile(abs_f, setting('histogram_log_percentiles'))
        edges = np.logspace(np.log10(p1), np.log10(p99), n_bins + 1)
        hp, _ = np.histogram(np.clip(np.maximum(np.abs(pred_c), 1e-30), p1, p99), bins=edges, density=True)
        ht, _ = np.histogram(np.clip(np.maximum(np.abs(true_c), 1e-30), p1, p99), bins=edges, density=True)
    else:
        p1, p99 = np.percentile(flat, setting('histogram_linear_percentiles'))
        edges = np.linspace(p1, p99, n_bins + 1)
        hp, _ = np.histogram(np.clip(pred_c, p1, p99), bins=edges, density=True)
        ht, _ = np.histogram(np.clip(true_c, p1, p99), bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:]) if log_scale else (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    return (centers, hp, ht, widths)

def js_divergence(p, q, widths=None):
    if widths is not None:
        p = p * widths
        q = q * widths
    eps = setting('js_epsilon')
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_p = np.where(p > 0, p * np.log2(p / m), 0.0)
        term_q = np.where(q > 0, q * np.log2(q / m), 0.0)
    return 0.5 * (term_p.sum() + term_q.sum())

def compute_radial_error_profile(pred, true, geo_raw, theta_mask=None, reference_scale=None):
    r = geo_raw['r']
    Nr, Nt, Np = r.shape
    r_mid = r[:, Nt // 2, Np // 2]
    n_bins = max(setting('spatial_bins_min'), min(setting('spatial_bins_max'), Nr // setting('spatial_bin_divisor')))
    if reference_scale is None and current()['space']=='physical':
        reference_scale=current()['scales']['mean_abs']
    edges = np.geomspace(r_mid.min(), r_mid.max(), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    prof = np.full((n_bins, 8), np.nan)
    for i in range(n_bins):
        r_mask = (r_mid >= edges[i]) & (r_mid < edges[i + 1])
        if r_mask.sum() == 0:
            continue
        for c in range(8):
            if theta_mask is not None:
                pc = pred[c][r_mask][:, theta_mask, :]
                tc = true[c][r_mask][:, theta_mask, :]
            else:
                pc = pred[c][r_mask]
                tc = true[c][r_mask]
            if tc.size == 0:
                continue
            denom = float(np.abs(tc).mean()) if reference_scale is None else reference_scale[c]
            prof[i, c] = float(np.abs(pc - tc).mean() / denom) if denom > 1e-30 else np.nan
    return (centers, prof)

def b2_magnitude(raw_t, geo_raw):
    """B^2 = gamma_ij B^i B^j using the spatial metric from the geometry.
    NOTE: raw gcov is (4,4,Nr,Nt,1) — the spatial block is the [1:,1:]
    slice of the FULL tensor. Do NOT index [0,0,...] on the raw array:
    that consumes both 4-indices and silently slices Nr and Nt instead
    (the wrapped (1,1,4,4,...) buffer used by the loss needs [0,0,1:,1:];
    the raw array does not)."""
    gam = geo_raw['gcov'][1:, 1:]
    B = raw_t[5:8]
    return np.einsum('iabc,ijabc,jabc->abc', B, gam, B)

def bsq_full_components(raw_t, geo_raw, norm_cpu):
    """Full covariant magnetic pressure b^2 = b^mu b_mu AND the fluid-frame
    field vectors (bcon = b^mu, bcov = b_mu, ucov = u_mu). IDENTICAL to
    analyse_forecast_ds.compute_bsq, so beta/sigma numbers agree between
    the plotting and evaluation scripts. b^mu is the fluid-frame field
    derived from the lab-frame B^i and u^mu (norm_cpu supplies the
    training-identical ucon_of on CPU buffers).
    The fluid-frame vectors are required for the Blandford-Znajek stress
    T^r_t = (w + b^2) u^r u_t - b^r b_t: substituting the LAB-frame B^r for
    b^r there is wrong (b^r = (B^r + b^0 u^r)/u^0), which is the classic
    mistake when evaluating the jet power from primitive output."""
    B1, B2, B3 = (raw_t[5], raw_t[6], raw_t[7])
    g = geo_raw['gcov']
    x = to_device(raw_t, torch.device('cpu'))
    ucon_t = norm_cpu.ucon_of(x[:, 2:5]).numpy()[0]
    u0, u1, u2, u3 = (ucon_t[0], ucon_t[1], ucon_t[2], ucon_t[3])
    ucov = np.zeros((4, *u0.shape))
    ucon = np.array([u0, u1, u2, u3])
    for mu in range(4):
        for nu in range(4):
            ucov[mu] += g[mu, nu] * ucon[nu]
    b0 = B1 * ucov[1] + B2 * ucov[2] + B3 * ucov[3]
    u0_safe = np.where(np.abs(u0) > 1e-30, u0, 1e-30)
    bcon = np.array([b0, (B1 + b0 * u1) / u0_safe, (B2 + b0 * u2) / u0_safe, (B3 + b0 * u3) / u0_safe])
    bcov = np.zeros_like(bcon)
    for mu in range(4):
        for nu in range(4):
            bcov[mu] += g[mu, nu] * bcon[nu]
    bsq = np.maximum(sum((bcon[mu] * bcov[mu] for mu in range(4))), 0.0)
    return (bsq, bcon, bcov, ucov, ucon)

def bsq_full(raw_t, geo_raw, norm_cpu):
    """Full covariant magnetic pressure b^2 = b^mu b_mu (2 p_mag)."""
    return bsq_full_components(raw_t, geo_raw, norm_cpu)[0]

def jet_power_shells(raw_t, geo_raw, norm_cpu, gamma, bare=True, return_mask=False):
    """Blandford-Znajek jet power per shell:
    P_jet(r) = sum_{theta,phi, sigma>1, funnel} (-T^r_t) sqrt(-g)   [code units]
    with -T^r_t = -(rho + gamma u_g + b^2) u^r u_t + b^r b_t using the
    FLUID-frame b^mu (see bsq_full_components). The funnel mask combines
    sigma = b^2/rho > 1 with the polar geometric funnel |theta - pi/2| > 1.0
    (consistent with plot_temporal_power_spectrum), isolating the outflowing jet
    from plunging coronal flow at the horizon. Reports signed net outward energy flux.
    When bare=False, multiplies by dA = dx_2 * dx_3 for physical code units."""
    bsq, bcon, bcov, ucov, ucon = bsq_full_components(raw_t, geo_raw, norm_cpu)
    w = raw_t[0] + gamma * raw_t[1]
    tr_t = (w + bsq) * ucon[1] * ucov[0] - bcon[1] * bcov[0]
    sigma = bsq / (raw_t[0] + 1e-30)
    h_angle = geo_raw['h']
    floor_rho = float(CONFIG.get('floor_rho', 0.0001))
    funnel = (sigma > 1.0) & (np.abs(h_angle - np.pi / 2.0) > 1.0) & ((ucon[1] > 0) | (-tr_t > 0)) & (raw_t[0] > 100 * floor_rho)
    mask = funnel.astype(np.float32)
    gdet = geo_raw['gdet'][..., 0]
    res = -np.sum(tr_t * gdet[:, :, None] * mask, axis=(1, 2))
    if not bare:
        dA = float(geo_raw['dx'][1] * geo_raw['dx'][2])
        res = res * dA
    if return_mask:
        return (res, mask)
    return res

def pairwise_masked_corrcoef(data,variance_min=None):
    """Pairwise-NaN-masked Pearson correlation matrix for data (N_samples, N_channels).
    Masks out samples where either channel is NaN. If variance < 1e-12 or valid samples < 3,
    returns NaN (nan_to_num with zero is strictly forbidden)."""
    if variance_min is None:variance_min=setting('correlation_variance_min')
    n_ch = data.shape[1]
    corr = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            xi = data[:, i]
            xj = data[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 3:
                continue
            xi_m = xi[mask]
            xj_m = xj[mask]
            v_i = float(np.var(xi_m))
            v_j = float(np.var(xj_m))
            if v_i < variance_min or v_j < variance_min:
                continue
            c = np.corrcoef(xi_m, xj_m)[0, 1]
            corr[i, j] = c
    return corr

def kerr_horizon_isco(a_spin):
    """Exact event horizon r_H and prograde ISCO radius r_ISCO (Bardeen 1972)
    for dimensionless Kerr spin parameter a.
    r_H = 1 + sqrt(max(0, 1 - a^2))
    Z_1 = 1 + (1 - a^2)^{1/3} [ (1+a)^{1/3} + (1-a)^{1/3} ]
    Z_2 = sqrt(3 a^2 + Z_1^2)
    r_ISCO = 3 + Z_2 - sqrt((3 - Z_1)(3 + Z_1 + 2 Z_2))
    """
    a = float(a_spin)
    abs_a = min(abs(a), 1.0)
    r_H = 1.0 + np.sqrt(max(0.0, 1.0 - a ** 2))
    z1 = 1.0 + (1.0 - abs_a ** 2) ** (1.0 / 3.0) * ((1.0 + abs_a) ** (1.0 / 3.0) + (1.0 - abs_a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * abs_a ** 2 + z1 ** 2)
    r_isco = 3.0 + z2 - np.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
    return (float(r_H), float(r_isco))

def add_isco_line(ax, r_isco, label='ISCO', color='gray', linestyle='--', alpha=0.7, yloc=0.92):
    """Draw a vertical dashed marker at the ISCO radius with a text label."""
    ax.axvline(r_isco, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.2)
    ax.text(r_isco * 1.04, yloc, label, transform=ax.get_xaxis_transform(), color=color, fontsize=8, alpha=alpha, va='top', ha='left')

def data_ylim(data_list, p_low=1.0, p_high=99.0, margin=1.1, symmetric=False, cap=None):
    """Compute robust data-driven y-limits from percentiles across multiple arrays."""
    clean = []
    for d in data_list:
        if d is not None:
            arr = np.asarray(d).ravel()
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                clean.append(finite)
    if not clean:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    merged = np.concatenate(clean)
    if symmetric:
        vmax = float(np.percentile(np.abs(merged), p_high)) * margin
        if cap is not None:
            vmax = min(float(cap), vmax)
        return (-vmax, vmax)
    else:
        vlo = float(np.percentile(merged, p_low))
        vhi = float(np.percentile(merged, p_high))
        span = vhi - vlo
        if span <= 0:
            span = max(abs(vhi), 1.0)
        low = vlo - (margin - 1.0) * span
        high = vhi + (margin - 1.0) * span
        if cap is not None:
            high = min(float(cap), high)
        return (low, high)

def kerr_keplerian_omega(r, a):
    """GR Keplerian orbital frequency in the equatorial plane of a Kerr BH:
    Omega_K = 1/(r^{3/2} + a) [c/Rg]. This is the analytic curve the disk's
    rotation should follow (frame dragging adds the +a term)."""
    return 1.0 / (r ** 1.5 + a)

def structure_function_exponents(channel, max_order=None):
    """S_p(lag) = <|x(r+lag) - x(r)|^p>_r along r, averaged over (theta, phi);
    scaling exponent zeta_p from a log-log fit over intermediate lags.
    On a log-spaced grid, separation corresponds to lag = Delta ln r.
    Returns (zeta_p, r2_values, signed S3)."""
    max_order=setting('structure_max_order') if max_order is None else max_order
    nr = channel.shape[0]
    max_lag = max(setting('structure_min_max_lag'), nr // setting('structure_lag_divisor'))
    lags = np.arange(1, max_lag)
    zetas = []
    r2s = []
    s3_sign = 0.0
    for p in range(2, max_order + 1):
        sp = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            d = channel[lag:] - channel[:-lag]
            sp[i] = np.mean(np.abs(d) ** p)
        lo, hi = (len(lags) // 4, 3 * len(lags) // 4)
        if hi <= lo:
            zetas.append(0.0)
            r2s.append(0.0)
            continue
        lk = np.log10(lags[lo:hi])
        lE = np.log10(sp[lo:hi] + 1e-30)
        poly = np.polyfit(lk, lE, 1)
        slope = float(poly[0])
        y_fit = poly[0] * lk + poly[1]
        ss_tot = np.sum((lE - np.mean(lE)) ** 2)
        ss_res = np.sum((lE - y_fit) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-30)) if ss_tot > 1e-30 else 0.0
        zetas.append(slope)
        r2s.append(r2)
        if p == 3:
            lag=max_lag//2
            d = channel[lag:] - channel[:-lag] if current()['space']=='physical' else channel[max_lag // 2:] - channel[:-max_lag // 2]
            s3_sign = float(np.mean(d ** 3))
    return (np.array(zetas), np.array(r2s), s3_sign)


def total_sum(value):
    """Float64 NumPy reduction for physical metrics; legacy reduction retained."""
    if current()["space"]=="physical":return float(np.sum(physical.array(value)))
    return float(torch.sum(value).item())


def _truth_cached(function):
    """Cache only registered ground-truth inputs, never adapter predictions."""
    import functools,inspect
    signature=inspect.signature(function)
    @functools.wraps(function)
    def cached(raw,*args,**kwargs):
        state=current();ctx=state['ctx']
        from ..truth_cache import truth_dump_id,truth_value
        index=truth_dump_id(ctx,raw)
        cache=getattr(ctx,'truth_cache',None)
        if state['space']!='physical' or index is None or cache is None:return function(raw,*args,**kwargs)
        arguments=signature.bind(raw,*args,**kwargs);arguments.apply_defaults()
        variant={key:(value.item() if isinstance(value,np.generic) else value)
                 for key,value in arguments.arguments.items()
                 if key in {'channel','bare','return_global','return_mask','gamma'}}
        return truth_value(ctx,'kernel',{'name':function.__name__,'dump':int(index),'variant':variant},
                           lambda:function(raw,*args,**kwargs))
    return cached


for _name in ('mass_shells','conserved_shells','physical_divb','normalized_divb',
              'radial_spectrum','mdot_shells','magnetic_flux_shells','jet_power_shells'):
    globals()[_name]=_truth_cached(globals()[_name])

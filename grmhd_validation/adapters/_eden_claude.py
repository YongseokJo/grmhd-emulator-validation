"""Numerical port: Eden Growney, originating coding agent claude.
Source: eden_rc300_pilot/benchmark_pilot/rc300_packets/start_11672_claude/submission/code/forecast_one_step.py
branch eden/rc300-frontier-15nov @ f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a.
Changes: remove packet/filesystem I/O; shape comes from the supplied state;
use canonical geometry dt_dump instead of the packet-only dt_dump_exact alias.
All numerical transport, smoothing, damping and model-owned floors are retained.
"""
from __future__ import annotations

import numpy as np

from scipy.ndimage import gaussian_filter, map_coordinates

COHERENCE_ORBITS = 3.0

R_EDDY_MIN = 8.0

OMEGA_SMOOTH = (1.5, 1.5)

VX1_SMOOTH = (2.0, 1.5)

DX1_SHIFT_CAP = 0.35

DPHI_CAP = 2.5

def utilde_to_u4(utilde: np.ndarray, gcov: np.ndarray, gcon: np.ndarray) -> np.ndarray:
    """utilde[i=0..2] + metric -> u^mu with g_{mu nu} u^mu u^nu = -1."""
    # utilde: (3, Nr, Nth, Nph); gcov/gcon: (4,4,Nr,Nth)
    util = np.moveaxis(utilde, 0, -1)  # Nr,Nth,Nph,3
    gs = np.transpose(gcov[1:, 1:], (2, 3, 0, 1))  # Nr,Nth,3,3
    gamma2 = 1.0 + np.einsum("...i,...ij,...j->...", util, gs[:, :, None, :, :], util)
    gamma = np.sqrt(np.maximum(gamma2, 1.0))
    alpha = 1.0 / np.sqrt(np.maximum(-gcon[0, 0], 1e-30))
    beta_over_alpha = (alpha * gcon[0, 1:]).transpose(1, 2, 0)  # Nr,Nth,3
    u_t = gamma / alpha[:, :, None]
    u_i = util - gamma[..., None] * beta_over_alpha[:, :, None, :]
    return np.concatenate([u_t[..., None], u_i], axis=-1)

def ring_profiles(state: np.ndarray, u4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density-weighted Omega(r,theta), v^{x1}(r,theta), smoothed."""
    rho = np.clip(state[0], 0.0, None)
    ut = np.clip(u4[..., 0], 1e-12, None)
    omega = u4[..., 3] / ut
    vx1 = u4[..., 1] / ut
    w = rho + 1e-30
    om = np.sum(w * omega, axis=2) / np.sum(w, axis=2)
    vx = np.sum(w * vx1, axis=2) / np.sum(w, axis=2)
    # equatorial reflection symmetrize (even)
    om = 0.5 * (om + om[:, ::-1])
    vx = 0.5 * (vx + vx[:, ::-1])
    om = gaussian_filter(om, sigma=OMEGA_SMOOTH, mode="nearest")
    vx = gaussian_filter(vx, sigma=VX1_SMOOTH, mode="nearest")
    return om, vx

def forecast(state0: np.ndarray, geo: np.lib.npyio.NpzFile, dt: float) -> np.ndarray:
    """Return next dump as float32 (8, Nr, Nth, Nph)."""
    s = state0.astype(np.float64)
    gcov = geo["gcov"][..., 0]
    gcon = geo["gcon"][..., 0]
    r = geo["r"][:, 0, 0].astype(np.float64)
    dx1, dx2, dx3 = [float(x) for x in geo["dx"]]
    spin = float(geo["spin"])

    u4 = utilde_to_u4(s[2:5], gcov, gcon)
    omega, vx1 = ring_profiles(s, u4)

    # Cap |Omega| near horizon spin frequency scale
    omega_h = spin / (1.0 + np.sqrt(max(1.0 - spin * spin, 0.0)))
    omega = np.clip(omega, -omega_h, omega_h)

    n_c, nr, nth, nph = s.shape
    assert n_c == 8

    # Working array: (Nr,Nth,Nph,8) channel-last for FFT convenience
    w0 = np.moveaxis(s, 0, -1)
    m0 = w0.mean(axis=2)
    fluct = w0 - m0[:, :, None, :]

    # Mild radial advection of fluctuations in x1 = ln r
    ir, ith = np.meshgrid(np.arange(nr), np.arange(nth), indexing="ij")
    shift_x1 = (DX1_SHIFT_CAP * np.tanh((vx1 * dt) / DX1_SHIFT_CAP)) / dx1
    # coherence: less radial transport where many orbits elapse
    r_eddy = np.maximum(r, R_EDDY_MIN)
    omega_k = 1.0 / (r_eddy**1.5 + spin)
    n_orbit = (np.abs(omega) * dt) / (2.0 * np.pi)
    n_orbit = np.maximum(n_orbit, (omega_k[:, None] * dt) / (2.0 * np.pi))
    coh_r = 1.0 / (1.0 + (n_orbit / 1.2) ** 2)
    coords = np.array(
        [np.clip(ir - shift_x1 * coh_r, 0.0, nr - 1.0), ith.astype(np.float64)]
    )

    # Axisymmetric background: slight radial advection + soft radial smooth blend
    m0_pred = np.empty_like(m0)
    for iv in range(n_c):
        field = m0[..., iv]
        adv = map_coordinates(field, coords, order=1, mode="nearest")
        # relax tiny snapshot noise toward phi-mean already (m0); keep mostly advected mean
        m0_pred[..., iv] = 0.85 * adv + 0.15 * field

    # Bounded differential rotation phase
    dphi = DPHI_CAP * np.tanh((omega * dt) / DPHI_CAP)
    dphi = gaussian_filter(dphi, sigma=(2.0, 1.5), mode="nearest")

    m_arr = np.arange(nph // 2 + 1, dtype=np.float64)
    # Eddy / coherence damping: high-m dies after a few local orbits
    t_eddy = (2.0 * np.pi) / omega_k  # (Nr,)
    tau = (COHERENCE_ORBITS * t_eddy)[:, None, None] / np.maximum(m_arr[None, None, :], 1.0)
    # Extra shear phase-error damping from Omega uncertainty ~5%
    sig_om = 0.05 * np.abs(omega)
    dph_err = np.abs(sig_om * dt)
    shear = np.exp(-0.5 * (m_arr[None, None, :] * dph_err[:, :, None]) ** 2)
    decay = np.exp(-dt / tau) * shear
    decay[..., 0] = 1.0
    phase = np.exp(-1j * m_arr[None, None, :] * dphi[:, :, None])
    transfer = phase * decay

    pred_cl = np.empty_like(w0)
    for iv in range(n_c):
        f = fluct[..., iv]
        # advect each Fourier coefficient lightly in radius (low m only)
        spec = np.fft.rfft(f, axis=2)
        real_a = np.empty_like(spec.real)
        imag_a = np.empty_like(spec.imag)
        for mi in range(spec.shape[2]):
            if mi <= 12:
                real_a[..., mi] = map_coordinates(
                    spec.real[..., mi], coords, order=1, mode="nearest"
                )
                imag_a[..., mi] = map_coordinates(
                    spec.imag[..., mi], coords, order=1, mode="nearest"
                )
            else:
                real_a[..., mi] = spec.real[..., mi]
                imag_a[..., mi] = spec.imag[..., mi]
        spec_adv = (real_a + 1j * imag_a) * transfer
        fluct_p = np.fft.irfft(spec_adv, n=nph, axis=2)
        pred_cl[..., iv] = m0_pred[:, :, None, iv] + fluct_p

    pred = np.moveaxis(pred_cl, -1, 0)

    # Floors: positivity + soft magnetization using spatial B^2 / gamma^2 proxy
    util = pred[2:5]
    gs = np.transpose(gcov[1:, 1:], (2, 3, 0, 1))
    util_t = np.moveaxis(util, 0, -1)
    gamma2 = 1.0 + np.einsum(
        "...i,...ij,...j->...", util_t, gs[:, :, None, :, :], util_t
    )
    gamma = np.sqrt(np.maximum(gamma2, 1.0))
    B2 = pred[5] ** 2 + pred[6] ** 2 + pred[7] ** 2
    # rough comoving b^2 ~ B^2 / gamma^2 in orthonormal sense (order-of-magnitude floor)
    b2_proxy = B2 / np.maximum(gamma**2, 1.0)

    rho_floor = float(np.min(state0[0])) * 0.98
    ug_floor = float(np.min(state0[1])) * 0.98
    pred[0] = np.maximum(pred[0], np.maximum(rho_floor, b2_proxy / 25.0))
    pred[1] = np.maximum(pred[1], np.maximum(ug_floor, b2_proxy / 750.0))

    out = pred.astype(np.float32)
    if out.shape != state0.shape:
        raise RuntimeError(f"bad shape {out.shape}")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite prediction")
    return out

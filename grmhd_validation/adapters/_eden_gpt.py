"""Numerical port: Eden Growney, originating coding agent gpt.
Source: eden_rc300_pilot/benchmark_pilot/rc300_packets/start_11672_gpt/submission/code/predict_one_step.py
branch eden/rc300-frontier-15nov @ f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a.
Changes: remove packet/filesystem I/O; shape comes from the supplied state;
use canonical geometry dt_dump instead of the packet-only dt_dump_exact alias.
All numerical transport, smoothing, damping and model-owned floors are retained.
"""
from __future__ import annotations

import numpy as np

def recover_four_velocity(state: np.ndarray, geo: np.lib.npyio.NpzFile):
    """Recover u^t, u^i from HARM-style utilde primitives and metric."""
    gcov = geo["gcov"].astype(np.float64)  # (4,4,Nr,Nth,1)
    gcon = geo["gcon"].astype(np.float64)

    u1 = state[2].astype(np.float64)
    u2 = state[3].astype(np.float64)
    u3 = state[4].astype(np.float64)

    g11 = gcov[1, 1, :, :, 0][:, :, None]
    g22 = gcov[2, 2, :, :, 0][:, :, None]
    g33 = gcov[3, 3, :, :, 0][:, :, None]
    g12 = gcov[1, 2, :, :, 0][:, :, None]
    g13 = gcov[1, 3, :, :, 0][:, :, None]
    g23 = gcov[2, 3, :, :, 0][:, :, None]

    q = (
        g11 * u1 * u1
        + g22 * u2 * u2
        + g33 * u3 * u3
        + 2.0 * (g12 * u1 * u2 + g13 * u1 * u3 + g23 * u2 * u3)
    )
    W = np.sqrt(np.maximum(1.0 + q, 1e-16))

    alpha = 1.0 / np.sqrt(np.maximum(-gcon[0, 0, :, :, 0], 1e-16))
    beta1 = -gcon[0, 1, :, :, 0] / gcon[0, 0, :, :, 0]
    beta2 = -gcon[0, 2, :, :, 0] / gcon[0, 0, :, :, 0]
    beta3 = -gcon[0, 3, :, :, 0] / gcon[0, 0, :, :, 0]

    ut = W / alpha[:, :, None]
    ui1 = u1 - W * beta1[:, :, None] / alpha[:, :, None]
    ui2 = u2 - W * beta2[:, :, None] / alpha[:, :, None]
    ui3 = u3 - W * beta3[:, :, None] / alpha[:, :, None]
    return ut, ui1, ui2, ui3

def fourier_shift_phi(field: np.ndarray, shift_cells: np.ndarray) -> np.ndarray:
    """Shift an (Nr,Nth,Nphi) field by a (Nr,Nth) cell offset along phi."""
    nphi = field.shape[-1]
    freqs = np.fft.fftfreq(nphi) * nphi  # integer mode numbers
    spectrum = np.fft.fft(field, axis=-1)
    phase = np.exp(-2j * np.pi * freqs[None, None, :] * shift_cells[:, :, None] / nphi)
    shifted = np.fft.ifft(spectrum * phase, axis=-1)
    return np.real(shifted)

def damp_nonaxisymmetric(field: np.ndarray, wraps: np.ndarray, damp_scale: float = 0.75) -> np.ndarray:
    """
    Phase-mixing: damp m!=0 Fourier modes where |Δφ|/(2π) is large.
    wraps: (Nr,Nth) number of azimuthal wraps over dt.
    """
    spectrum = np.fft.fft(field, axis=-1)
    m = np.abs(np.fft.fftfreq(field.shape[-1]) * field.shape[-1])
    # Retention of m>0 modes decays with wraps; m=0 untouched.
    retain = np.exp(-np.maximum(wraps, 0.0)[:, :, None] / damp_scale)
    retain = np.where(m[None, None, :] < 0.5, 1.0, retain)
    damped = np.fft.ifft(spectrum * retain, axis=-1)
    return np.real(damped)

def upwind_x1(field: np.ndarray, nu_cells: np.ndarray) -> np.ndarray:
    """
    Single first-order upwind step in code-x1 for Courant-limited |nu|<~1.
    nu_cells: (Nr,Nth) signed shift in radial cells (clipped).
    """
    nu = np.clip(nu_cells, -0.95, 0.95)
    # pad radially with edge values
    f = field
    fp = np.concatenate([f[:1], f[:-1]], axis=0)  # i-1
    fn = np.concatenate([f[1:], f[-1:]], axis=0)  # i+1
    nu3 = nu[:, :, None]
    flux_from_left = nu3 * (f - fp)  # for nu>0
    flux_from_right = nu3 * (fn - f)  # for nu<0
    out = f - np.where(nu3 >= 0.0, flux_from_left, flux_from_right)
    return out

def predict_next(state: np.ndarray, geo: np.lib.npyio.NpzFile) -> np.ndarray:
    dt = float(geo["dt_dump"])
    dx = geo["dx"].astype(np.float64)
    ut, ui1, ui2, ui3 = recover_four_velocity(state, geo)

    # Stable axisymmetric transport speeds (phi-median).
    omega = np.median(ui3 / ut, axis=-1)  # dφ/dt
    v1 = np.median(ui1 / ut, axis=-1)  # dx1/dt in code coords

    shift_phi_cells = omega * dt / dx[2]
    wraps = np.abs(omega * dt) / (2.0 * np.pi)
    nu_r = v1 * dt / dx[0]

    pred = np.empty_like(state, dtype=np.float64)
    src = state.astype(np.float64)

    for c in range(state.shape[0]):
        adv = fourier_shift_phi(src[c], shift_phi_cells)
        adv = damp_nonaxisymmetric(adv, wraps, damp_scale=0.85)
        adv = upwind_x1(adv, nu_r)
        pred[c] = adv

    # Physical floors / relationships for primitives.
    rho_floor = max(float(np.min(src[0])) * 0.5, 1e-10)
    ug_floor = max(float(np.min(src[1])) * 0.5, 1e-12)
    pred[0] = np.maximum(pred[0], rho_floor)
    pred[1] = np.maximum(pred[1], ug_floor)

    # Mild blend with source near outer radii where signal is slow and
    # numerical diffusion from damping would otherwise erase structure.
    r = geo["r"][:, 0, 0]
    # Keep more of advected field inside; slight persistence outside.
    blend_persist = np.clip((np.log10(np.maximum(r, 1.0)) - 2.0) / 2.0, 0.0, 0.25)
    w = blend_persist[:, None, None]
    for c in range(state.shape[0]):
        pred[c] = (1.0 - w) * pred[c] + w * src[c]

    # Re-apply floors after blend.
    pred[0] = np.maximum(pred[0], rho_floor)
    pred[1] = np.maximum(pred[1], ug_floor)

    out = pred.astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite values in prediction")
    if out.shape != state.shape:
        raise RuntimeError(f"bad shape {out.shape}")
    return out

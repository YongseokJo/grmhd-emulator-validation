"""Numerical port: Eden Growney, originating coding agent gemini.
Source: eden_rc300_pilot/benchmark_pilot/rc300_packets/start_11672_gemini/submission/code/predict_step.py
branch eden/rc300-frontier-15nov @ f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a.
Changes: remove packet/filesystem I/O; shape comes from the supplied state;
use canonical geometry dt_dump instead of the packet-only dt_dump_exact alias.
All numerical transport, smoothing, damping and model-owned floors are retained.
"""
from __future__ import annotations

import numpy as np

RHO_FLOOR = 1e-6

UG_FLOOR = 1e-8

def four_velocity_and_omega(state: np.ndarray, geom: np.lib.npyio.NpzFile):
    """Return density-weighted Omega(r,theta) from utilde primitives."""
    gcov = geom["gcov"][..., 0].astype(np.float64)
    gcon = geom["gcon"][..., 0].astype(np.float64)

    alpha = 1.0 / np.sqrt(np.maximum(-gcon[0, 0], 1e-30))
    beta = np.stack([(alpha**2) * gcon[0, i + 1] for i in range(3)], axis=0)

    utilde = state[2:5].astype(np.float64)
    gamma2 = np.ones(utilde.shape[1:], dtype=np.float64)
    for i in range(3):
        for j in range(3):
            gamma2 += gcov[i + 1, j + 1][:, :, None] * utilde[i] * utilde[j]
    gamma = np.sqrt(np.maximum(gamma2, 1.0))

    ut = gamma / alpha[:, :, None]
    uphi = utilde[2] - gamma * beta[2][:, :, None] / alpha[:, :, None]
    omega = uphi / np.maximum(ut, 1e-30)

    rho = np.maximum(state[0].astype(np.float64), 0.0)
    weight = rho + 1e-30
    omega_rt = (omega * weight).sum(axis=2) / weight.sum(axis=2)
    return omega_rt

def phi_advect(cube: np.ndarray, shift_cells: np.ndarray) -> np.ndarray:
    """Periodic Fourier shift of shape (C,Nr,Nth,Nph) by shift_cells (Nr,Nth)."""
    nphi = cube.shape[-1]
    k = np.fft.fftfreq(nphi) * nphi
    phase = np.exp(-2j * np.pi * k[None, None, :] * shift_cells[:, :, None] / nphi)
    out = np.empty_like(cube)
    for c in range(cube.shape[0]):
        f = np.fft.fft(cube[c].astype(np.float64), axis=-1)
        shifted = np.fft.ifft(f * phase, axis=-1).real
        out[c] = shifted.astype(cube.dtype, copy=False)
    return out

def predict_next(state: np.ndarray, geom: np.lib.npyio.NpzFile) -> np.ndarray:
    dt = float(geom["dt_dump"])
    dx3 = float(geom["dx"][2])
    omega_rt = four_velocity_and_omega(state, geom)
    # Material with +Omega moves toward +phi; sample origin is -Omega*dt.
    shift_cells = omega_rt * dt / dx3
    pred = phi_advect(state, shift_cells)
    pred[0] = np.maximum(pred[0], RHO_FLOOR)
    pred[1] = np.maximum(pred[1], UG_FLOOR)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite values in prediction")
    return pred.astype(np.float32, copy=False)

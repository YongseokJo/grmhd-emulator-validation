# Ported evaluation code

The owner approved public validation ports on 2026-11-14 (spec §5, e700301).
The private checkout is read-only. No original script is vendored.

| Module | Original source, branch and commit | Authors | Changes |
|---|---|---|---|
| `grmhd_validation/analyses/common.py` | `Fall26/eval/evaluate_forecast_ds.py`, `Fall26/common/normalizer.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON normalization originals) | Float64 NumPy CPU loss transform and reductions; protocol floors/thresholds; JSON null for undefined numbers; no decode ceilings or Lorentz clamp. |
| `grmhd_validation/analyses/single_step.py` | `Fall26/eval/evaluate_forecast_ds.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON originals) | All common eligible starts within each chunk; chunk-block bootstrap; structured JSON/CSV. |
| `grmhd_validation/analyses/rollout_curve.py` | `Fall26/eval/evaluate_forecast_ds.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON originals) | Frozen rollout starts and runner history; start bootstrap; split-wide strided attractor sigma; fixed figure styling and artifact names. Kevin's palette/tick rule retained. |
| `tests/test_quick_metrics.py` | Same evaluator and normalizer; `main@ebcc218` | Kevin Fortune; Sachin (TRON normalization originals) | Loads selected original definitions from the pinned git object at test runtime. Compares quick metrics at the approved 1e-5 relative tolerance and checks palette/ticks exactly. |

The common normalizer's own attribution traces normalization to Sachin's
`fall26/uno_simple_ds_v3.py@ca0288555c99`. The pinned common module is the
reference for these ports. The TRON adapter port is milestone 5; the fixed
checkpoint/full-suite regression belongs to milestone 3.

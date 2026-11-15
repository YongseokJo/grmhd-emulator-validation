# Ported evaluation code

The owner approved public validation ports on 2026-11-14 (spec §5, e700301).
The private checkout is read-only. No original script is vendored.

| Module | Original source, branch and commit | Authors | Changes |
|---|---|---|---|
| `grmhd_validation/analyses/common.py` | `Fall26/eval/evaluate_forecast_ds.py`, `Fall26/common/normalizer.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON normalization originals) | Float64 NumPy CPU loss transform and reductions; protocol floors/thresholds; JSON null for undefined numbers; no decode ceilings or Lorentz clamp. |
| `grmhd_validation/analyses/single_step.py` | `Fall26/eval/evaluate_forecast_ds.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON originals) | All common eligible starts within each chunk; chunk-block bootstrap; structured JSON/CSV. |
| `grmhd_validation/analyses/rollout_curve.py` | `Fall26/eval/evaluate_forecast_ds.py`; `main@ebcc218` | Kevin Fortune; Sachin (TRON originals) | Frozen rollout starts and runner history; start bootstrap; split-wide strided attractor sigma; fixed figure styling and artifact names. Kevin's palette/tick rule retained. |
| `tests/test_quick_metrics.py` | Same evaluator and normalizer; `main@ebcc218` | Kevin Fortune; Sachin (TRON normalization originals) | Loads selected original definitions from the pinned git object at test runtime. Compares quick metrics at the approved 1e-5 relative tolerance and checks palette/ticks exactly. |
| `analyses/error_correlation.py` | `Fall26/eval/evaluate_forecast_ds.py:evaluate_error_correlation`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Pairwise NaN-masked Pearson coefficients retained; physical errors use training mean-abs scales; frozen eligible pairs; JSON/CSV. |
| `analyses/correlation_times.py` | Same evaluator, `channel_decorrelation_times` / `evaluate_correlation_times`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Float64 NumPy, chunk-local truth pairs, resolved rc300 disk, protocol constants, explicit legacy sampling and horizon/censoring diagnostics. |
| `analyses/stability.py` | Same evaluator, `evaluate_stability`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Legacy perturbations retained for regression; physical perturbations and response divided by training RMS; explicit starts. |
| `analyses/cost.py` | Same evaluator, `evaluate_cost`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Adapter inference; 3 warmup/20 timed calls; parameter counts in result, measured latency in run metadata only. |
| `analyses/physics_table.py` | Same evaluator, `evaluate_physics_table`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Fixed physical adapter trajectory; structured per-horizon shell/divergence/continuity/Lorentz diagnostics. |
| `analyses/shell_profiles.py` | Same evaluator, `evaluate_shell_profiles`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical conserved profiles, fixed figure mapping, structured arrays and signed-log shell errors. |
| `analyses/temporal_consistency.py` | Same evaluator, `evaluate_temporal_consistency`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical M/E/L totals and original drift scales; every step emitted. |
| `analyses/conservation_closure.py` | Same evaluator, `evaluate_conservation_closure`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original bare/physical enclosed-mass and boundary-flux arrays; frozen window and JSON/CSV. |
| `analyses/accretion_diagnostics.py` | Same evaluator, `evaluate_accretion_diagnostics`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical magnetic/jet profiles and accretion series; original masks and shell choices retained at this milestone. |
| `analyses/mdot.py` | Same evaluator, `plot_mdot`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original shell reconstruction and errors; physical unit and fixed artifact serialization. |
| `analyses/lorentz.py` | Same evaluator, `evaluate_lorentz`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical pre-clamp gamma; optional adapter clamp counter, unknown is null rather than zero. |
| `analyses/mean_profiles.py` | Same evaluator, `plot_mean_profiles`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original time/phi averages, orbital/density/beta/sigma profiles; physical arrays and two fixed figures. |
| `analyses/statistical_props.py` | Same evaluator, `evaluate_statistical_props`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Shared-bin physical PDFs/JS retained; bin arrays emitted and schema checked. |
| `analyses/spectrum.py` | Same evaluator, `plot_spectrum`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical-field spectrum with float64 NumPy FFT; legacy loss/float32 FFT for regression. |
| `analyses/azimuthal_spectra.py` | Same evaluator, `plot_azimuthal_spectra`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical equatorial m spectra; explicit legacy transform/start option. |
| `analyses/coherence.py` | Same evaluator, `plot_coherence`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical Fourier coherence, every step; explicit legacy transform/start option. |
| `analyses/temporal_spectrum.py` | Same evaluator, `plot_temporal_power_spectrum`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original physical disk/jet Hann periodograms, frequencies in inverse Rg/c, structured arrays. |
| `analyses/turbulence_stats.py` | Same evaluator, `plot_turbulence_stats` / `structure_function_exponents`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical structure functions and fits; original loss-space option. Physical signed-third-order lag fixes odd-lag slice precedence. |
| `analyses/conditional_error.py` | Same evaluator, `evaluate_conditional_error`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical magnitude bins/errors; frozen eligible pairs; sampling capped to available cells on tiny synthetic grids. |
| `analyses/bias_analysis.py` | Same evaluator, `evaluate_bias_analysis`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical phi-averaged bias/RMS; explicit selected starts, structured maps. |
| `analyses/spatial_error.py` | Same evaluator, `evaluate_spatial_error`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Physical radial errors scaled by training mean-abs; legacy target-relative configuration retained. |
| `analyses/meridional_slice.py` | Same evaluator, `plot_meridional_slice`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original coordinate grids/log displays; physical error maps replace log errors; three fixed zoom filenames. |
| `analyses/rollout_evolution.py` | Same evaluator, `plot_rollout_evolution`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Original coordinate/time panels with physical density errors and generic model labels. |
| `analyses/_kevin_math.py`, `_geometry_physics.py`, `_physical.py` | Same evaluator plus `Fall26/common/normalizer.py` / `physics_loss.py`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Selected kernels factored out, original legacy arithmetic retained; independent float64 NumPy physical kernels; common immutable geometry inputs. |
| `analyses/_analysis_io.py`, `_analysis_state.py`, `_port_manifest.py` | Integration around the preceding ports; constants/figure names trace to `main@ebcc218` | Kevin Fortune; Sachin Venkatesh (source formulas); validation implementer (integration) | Runner-owned inference/history, invocation-private prediction scratch, explicit configuration, schema-valid numeric capture and fixed figure mapping. No production import from the private checkout. |
| `tests/reference_support.py`, `tests/test_full_source_regression.py`, `tests/regression_tron.py` | Original evaluator/common at `main@ebcc218`; `Fall26/tron/model.py` at `fall26-tron-rc300@2667a5f` | Kevin Fortune; Sachin Venkatesh | Read pinned definitions at runtime; observe original numeric locals directly; synthetic kernels and regression-only pinned persistence/TRON window. No original script/checkpoint/data cube is committed. |

The common normalizer's own attribution traces normalization to Sachin's
`fall26/uno_simple_ds_v3.py@ca0288555c99`. The pinned common module is the
reference for these ports. The TRON adapter port is milestone 5; the fixed
checkpoint/full-suite regression belongs to milestone 3. Paths abbreviated as
`analyses/` above are under `grmhd_validation/`. Milestone-3 sampling versus the
final milestone-4 aggregate is explicit in `docs/metrics.md` and spec §3.4.

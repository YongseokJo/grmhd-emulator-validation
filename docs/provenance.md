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

| `analyses/_transport.py` | Same evaluator, `bsq_full_components`, `mdot_shells`, `jet_power_shells`; `main@ebcc218`, plus `Fall26/common/normalizer.py` | Kevin Fortune; Sachin Venkatesh | Metric-normalized float64 four-velocity; owner-approved inward signs, rest-mass-subtracted jet power, total/jet efficiencies, resolved-shell headline and endpoint mass budget. Legacy source kernels retained separately. |
| `analyses/_profiles.py` | Same evaluator, `plot_mean_profiles`, `plot_temporal_power_spectrum`; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Compact cached truth profiles/masks/periodograms; physical start averages before display logs; cellwise temporal powers retained with lower peak memory. |
| `analyses/_final.py`, `_reference_statistics.py`, `_aggregate.py` | Same evaluator's conditional bins/error correlation plus approved spec §5.3–9 | Kevin Fortune; Sachin Venkatesh (source estimators); validation implementer (bootstrap/integration) | All common starts, nonlinear pooled estimators, within-chunk block CIs and shared truth-only bins/support. Original single-start APIs remain callable for regression. |
| `analyses/_figures.py` | Same evaluator's panel designs, labels, quantities and sizes; `main@ebcc218` | Kevin Fortune; Sachin Venkatesh | Explicit physics panels and units, shared role/channel/step styles, mean curves at every report step with CI shading only at the final overlay step, physical-before-log maps and split/start captions. Legacy plotting remains in regression cores. |
| `truth_cache.py` | New benchmark integration; spec §5.9–10 | Validation implementer | Owner-only hash-bound pickle-free truth cache, read-only readers, no model predictions. |

The common normalizer's own attribution traces normalization to Sachin's
`fall26/uno_simple_ds_v3.py@ca0288555c99`. The pinned common module is the
reference for these ports. The TRON adapter port is milestone 5; the fixed
checkpoint/full-suite regression belongs to milestone 3. Paths abbreviated as
`analyses/` above are under `grmhd_validation/`. Milestone-3 sampling versus the
final milestone-4 aggregate is explicit in `docs/metrics.md` and spec §3.4.

## Milestone 5 adapters and packet host

| Module | Original source | Changes |
|---|---|---|
| `adapters/learned.py` | Kevin Fortune, `Fall26/eval/evaluate_tron.py`, `fall26-tron-rc300@2667a5f`; `Fall26/eval/evaluate_vit.py`, `Fall26/common/normalizer.py`, `main@ebcc218` (TRON originals: Sachin) | Argument-only geometry and physical history, strict state-dict checkpoint loading, residual decode and exact Lorentz clamp; no data paths, scoring or experiment discovery. |
| `adapters/_tron_model.py` | Kevin Fortune / Sachin, `Fall26/tron/model.py`, `fall26-tron-rc300@2667a5f` | Architecture definitions selected for the standalone adapter; evaluator/training context removed. |
| `adapters/_vit_model.py` | Kevin Fortune, `Fall26/vit/model.py`, `main@ebcc218` | Architecture definitions selected for the standalone adapter; restored checkpoint configuration and argument coordinates. |
| `adapters/eden.py`, `_eden_claude.py` | Eden Growney, originating agent Claude, `eden_rc300_pilot/benchmark_pilot/rc300_packets/start_11672_claude/submission/code/forecast_one_step.py`, `eden/rc300-frontier-15nov@f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a` | Remove packet I/O and fixed-shape guards; validate log-radius and periodic phi. Preserve source index reflection, smoothing, phase/shear damping, radial interpolation and magnetization floors. Milestone 7 removes an extra physical-theta symmetry guard: the original kernel never reads theta and already defines its index reflection on the real rc300 grid. No numerical kernel change or input resampling. |
| `adapters/eden.py`, `_eden_gpt.py` | Eden Growney, originating agent GPT, same branch/commit, `start_11672_gpt/submission/code/predict_one_step.py` | Argument-only input; dynamic shape; unchanged phi shift, phase mixing, Courant-limited radial step, radial persistence blend and floors. |
| `adapters/eden.py`, `_eden_gemini.py` | Eden Growney, originating agent Gemini, same branch/commit, `start_11672_gemini/submission/code/predict_step.py` | Argument-only input; canonical dt_dump replaces packet-only dt_dump_exact; unchanged density-weighted rotation, FFT shift and floors. |
| `predictions.py`, `prompts/eden-{code,step}-v1/` | Eden Growney, same branch/commit, rc300 `primary_ar_onestep` packet contract and `tools/advance_rc300_ar.py` | New implementation of frozen benchmark coverage, h-frame rolling history, strict float32, immutable acceptance, separated host/scorer roles, hashed provenance and explicit host attestation. Dynamic dimensions/cadence. |

The Eden examples are protocol A methods with known pilot exposure. Their historic
start-11672 cubes/scores are not imported. Wrapping them does not establish blind
development on the frozen split. Original scripts are read only in regression
checks, never committed unchanged. No new dependency is introduced.

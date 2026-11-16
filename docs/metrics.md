# Quick-tier metric definitions

These are the final v1 definitions from spec §3.3. All scoring converts raw
physical arrays to C-contiguous NumPy float64 on CPU. Channels are ordered
`rho, ug, utilde1, utilde2, utilde3, B1, B2, B3`. Spatial reductions use all cells,
including the copied coarse ring: quick metrics do not apply the disk mask.

Let `L(x)` take log10 of floored density/internal energy (1e-20/1e-22), and
arcsinh of the other channels. This loss space is dimensionless in the simulation
code-unit convention. It has no decode ceilings or Lorentz clamps. For each
sample/channel, nMAE is `mean_cells(abs(L(pred)-L(truth)))`; the persistence
forecast is the physical frame at the start, held fixed for every rollout step.

Single-step evaluates every start from `first+H_max-1` to `last-max_steps`,
inclusive, separately inside each evaluated chunk. Means are equal-weight sample
means. Skill is `1-M/(P+epsilon)`, undefined where the per-channel persistence
nMAE is below 1e-6. Physical relative L1 is `mean(abs(pred-truth))/mean(abs(truth))`
for each sample/channel, then a nanmean across samples; it is undefined below a
1e-30 denominator. This quick-specific definition uses the target's scale, as
§3.3 prescribes. It does not need the later owner-generated training scales.
The mean row is a nanmean over channels; fluid and magnetic nMAE average channels
0–4 and 5–7 respectively. Undefined JSON numbers are null and CSV cells are empty.

The single-step 95% CI recomputes channel skill after resampling consecutive
blocks of ten starts. Blocks never cross chunks; a last shorter block is retained.
All blocks are pooled, and each draw samples the original number of blocks with
replacement. A fresh `default_rng(42)` makes 1,000 draws, with 2.5/97.5
nanpercentiles. Counts and `chunk_block_bootstrap` are recorded in the metric.

Rollout uses the protocol's rounded, deduplicated linspace starts, without random
selection or shortening. The runner shifts its history and appends its own
prediction. Per-step channel nMAE is averaged across starts. Headline mean and
population standard deviation use each start's channel mean. Rollout skill uses
the ratio of these means with epsilon, without the single-step 1e-6 threshold.
Its CI resamples individual starts using a fresh generator with the same seed,
resample count and percentiles. JSON contains every step; CSV and summary contain
only the protocol's report steps. CSV time is steps times the split's dt_dump,
in R_g/c.

Attractor sigma is the mean of each sampled truth dump's loss-space population
standard deviation over cells. Sampling starts at the first dump of each
evaluated chunk and takes every tenth dump. A final-step channel error strictly
below sigma is `ok`; otherwise it `exceeds`. It measures spatial truth variability,
not temporal variance.

Kevin's equations for nMAE/relative L1, tick rule, channel palette and two-panel
figure are retained. Intended changes are CPU float64 (Kevin normalizes in torch
float32), common starts/history handling, the two explicit bootstrap rules,
strided split-wide sigma, generic labels, fixed output names/style, and structured
serialization. Tests cover these changes; source regression reads the pinned
original at runtime with the approved 1e-5 relative tolerance. Predictability
horizon and correlation times are described below.

## Full-tier ports (milestone 3)

All 23 remaining analyses are separate modules. Each writes a closed-schema
`metrics/<analysis>.json` and `tables/<analysis>.csv`; the CSV flattens numeric
array entries into stable `field,value` rows. `summary.json` contains the same
structured results. Undefined numbers become null/empty CSV cells. Figures are
excluded from result fingerprints. No result-content file contains timing,
paths, host/user names, versions or attempt identifiers.

This milestone establishes the port and its numerical regression. Its production
policy is the **first common protocol rollout start**, as recorded in spec §3.4.
The core accepts `space="loss"|"physical"` and explicit `starts=[...]`; multiple
starts are retained separately. Bias already combines its explicitly selected
starts. Error-correlation/conditional-error retain the original within-analysis
eligible-pair sampling after the selected first start. All windows use the frozen
partition, and no excluded dump reaches the adapter. The final all-start
aggregates, remaining confidence intervals/block bootstraps, corrected/audited
accretion transport, coarse-ring treatment and truth cache are milestone 4.
These intermediate full results are for milestone review, before the leaderboard.

Physical arrays and all physical metric reductions/FFTs use NumPy float64 on CPU.
Torch tensors are used for the adapter interface and as containers at the legacy
kernel boundaries. Source regression explicitly selects the original loss-space
normalizer and original mixed float32/float64 arithmetic. The two configurations
are tested separately: changing metric space is an intended numerical change,
not evidence of a failed port. Model decode floors/ceilings and Lorentz clamps
belong to the adapter; the physical scorer does not apply them again.

Let `x,p,t` denote physical input, prediction and truth; `g=sqrt(-det(g_mu_nu))`.
Let `A_c` and `R_c` be training mean-abs and RMS scales from the owner-generated
`references/<split_id>/channel_scales.json`. They include every cell of every
training dump, with float64 sums in dump-ID order. Their split/manifest hashes,
counts and channel order are recorded; a run records the artifact SHA256.
Zero reference scales make a relative error undefined; stability excludes a
zero-RMS channel from the weighted norm. No evaluation data fit these scales.

| Analysis | Numerical definition, units and within-start averaging | Change from Kevin / legacy configuration |
|---|---|---|
| `error_correlation` | Pearson coefficient of per-channel `mean(abs(p-t))/A_c`, across the first 20 eligible consecutive pairs. Pairwise finite samples; fewer than 3 or variance below 1e-12 is undefined. Dimensionless 8×8 matrices. | Original target-relative denominator is retained in legacy mode. No undefined coefficient is replaced with zero. |
| `correlation_times` | Lag-1 truth correlation after removing the per-cell temporal background. Pool consecutive pairs within each evaluated chunk and cells in `r<=100, abs(theta-pi/2)<0.5`; rc300 also requires resolved cells. Clip r1 to ±0.999; tau=0 for r1<=0, otherwise `-1/log(min(r1,0.99))`; saturation flag is r1>0.99. Horizon is the first negative loss-space skill step, or a censored final step. Record recrossings, last positive step, mean skill at steps 25–50, horizon/tau and times multiplied by dt_dump. | Production uses the whole evaluated truth split and resolved rc300 mask. Legacy selects at most the first 200 supplied truth dumps, without resolved masking, to reproduce the original. Horizon remains loss space. |
| `stability` | Ten Gaussian perturbations of amplitude `0.001*R_c`; ratio `||[f(x+eps)-f(x)]/R||_2 / ||eps/R||_2`. Mean/population std and all draws; dimensionless. | Legacy multiplies the raw state by `(1+0.001*z)` and measures response in the original normalized representation. Physical perturbations and channel weighting change the meaning intentionally. |
| `cost` | Parameter count and measurement configuration; 3 warmup and 20 timed autoregressive steps, with CUDA synchronization around measurement. Latency in ms/step lives only in run.json. | Includes runner history/adapter validation overhead. Timing is not a golden result. Controlled-clock test verifies measurement arithmetic. |
| `physics_table` | Per report horizon: errors in mass/energy/angular shells, physical mean abs(divB), local-length normalized divB, sampled continuity residuals, pre-clamp max gamma. Scalar errors average shells; continuity samples up to 10,000 active-disk interior cells with fixed seed. | Original physical formulas and masks; raw predictions scored without decode clamping. Signed-log shell comparison is retained as a diagnostic transform, not the TRON channel normalization. |
| `shell_profiles` | Angular-mean profiles of mass, binding energy and angular momentum, plus mean absolute signed-log profile errors, at original horizons. Shell densities in simulation code units; transformed errors dimensionless. | Original conserved expressions/figure panels retained. NumPy physical arithmetic; same explicit legacy clamp/float32 path for regression. |
| `temporal_consistency` | Total M/E/L at each step, and absolute change from the initial value divided by original initial magnitude. Near-zero E falls back to initial internal energy, near-zero L to `abs(M0)*r_ISCO`. Dimensionless drift; totals in code units. | Original definitions retained, physical float64 arithmetic. Averaging over starts/CIs remains milestone 4. |
| `conservation_closure` | Enclosed mass between horizon shell and shell nearest r=100; consecutive finite-difference dM/dt versus boundary accretion flux. Emit bare grid sums and physical volume/area factors, and imbalance at every step. | Faithful original snapshot diagnostic. Transport/coarse-ring improvements remain milestone 4. |
| `accretion_diagnostics` | Magnetic flux `sum(abs(B^r)*g)`, fluid-frame stress jet power, normalized horizon magnetic flux, horizon/r10 accretion series and original disk/funnel masks. Physical flux/power code units; normalization and efficiency dimensionless. | Original shell choices/masks retained at this milestone. Final transport/ring changes remain milestone 4. |
| `mdot` | `-sum(rho*u^r*g)` over theta/phi, inflow positive, using original coordinate four-velocity reconstruction; profiles at report horizons with original target-profile relative error. | Original convention retained; the milestone-4 transport audit is still required. |
| `lorentz` | Pre-clamp `sqrt(1+g_ij*utilde^i*utilde^j)` max and 99.9th percentile at each step, worst step, optional model-reported clamp count. Dimensionless. | Unknown clamp counts are null; persistence is known zero. Physical float64 geometry arithmetic. |
| `mean_profiles` | Original time/phi averages over the first max_steps//4 trajectory interval: Omega=u^phi/u^0 versus Kerr Omega_K, vertical density at r=10/30/100, log10 plasma beta and magnetization. Omega in inverse Rg/c, density in code units; beta/sigma dimensionless. | Original within-start averaging retained for regression; arrays/masks and both plot variants exposed. |
| `statistical_props` | Shared-bin physical PDFs, 200 bins, pooled pred/truth percentile edges; log bins for rho/ug, linear for remaining channels. JS divergence uses bin probability mass and epsilon 1e-12; midpoint-weighted means. At max_steps//2. | Original physical distribution definition retained. JS is dimensionless; PDFs/means carry each channel's units. |
| `spectrum` | Mean-subtract physical rho/B1; periodic Hann in r and theta, rfftn, mean over theta/phi Fourier indices, fold ±radial modes. Emit powers, transfer ratios above truth's 10th percentile, log-log slope/R² fit over middle half. | Legacy transforms channels before its float32 FFT. Physical power has squared channel units and uses float64 FFT. |
| `azimuthal_spectra` | Phi-mean-subtracted physical rho/utilde3/B1 in the mid-radius half and equatorial band abs(theta-pi/2)<0.15; phi rFFT power averaged over r/theta at original horizons. | Legacy uses normalized fields. Power is squared physical channel units. |
| `coherence` | At each step/radial mode, `abs(mean(Fp*conj(Ft)))**2/(mean(abs(Fp)**2)*mean(abs(Ft)**2)+epsilon)`, averaging transverse modes. | Physical fields replace normalized fields; dimensionless. Original full radial FFT indexing retained. |
| `temporal_spectrum` | Cellwise temporal mean removal, symmetric Hann time window, `abs(rfft)**2/(fs*sum(window**2))`, averaged over disk/jet cells, all available steps (minimum 6). Original disk r>=5/theta band and jet magnetization masks. | Already physical in original. Frequency inverse Rg/c; power squared channel units times Rg/c. Empty mask retains original zero periodogram. |
| `turbulence_stats` | Radial structure functions `S_p(lag)=mean(abs(x[r+lag]-x[r])**p)` for orders 2–4; intermediate-lag log fits give exponents/R²; signed third-order moment also emitted. Active disk r>=2, theta halfwidth 0.3, at max_steps//2. | Physical fields replace normalized fields. Physical signed-third-order moment uses equal positive/negative slices for odd maximum lag; legacy preserves the original slice expression for regression. |
| `conditional_error` | Physical absolute errors in five truth-magnitude quantile bins per channel; truth rho/ug signed magnitude, other channels abs magnitude. Edge pool uses up to 50 eligible dumps and seeded 1% cell samples (minimum 100, capped to available cells); errors are cell-count weighted. | Original binning on normalized fields becomes physical binning. Each channel's error carries its own units; no cross-channel mean is claimed. |
| `bias_analysis` | Physical signed error averaged in phi then over selected starts; RMS of those phi-mean errors across selected starts, for all channels. | Already physical. RMS is the original phi-mean diagnostic, not cellwise RMSE. |
| `spatial_error` | Radial log bins, disk theta halfwidth 0.5 and polar funnel offset >1; channel mean abs error divided by A_c within each bin/horizon. | Legacy divides by each target region's mean abs value. Physical relative errors use common training scales. |
| `meridional_slice` | Mid-phi physical coordinate `(r*sin(theta),r*cos(theta))`; log10 rho/B² display, physical abs rho/B² residual maps, full/r100/r250 views. | Display logs do not define error space. Legacy log-error panels retained for regression. |
| `rollout_evolution` | Original horizon panels on physical meridional coordinates; log density truth/prediction displays with physical abs density error. | Legacy error is a difference of log displays. Physical error units are density code units. |

The physical float64 implementation preserves the equations rather than the
original rounding. In particular NumPy's median convention is used for the
normalized-divergence magnetic denominator. Spectra/structure/conditional errors
in physical space can rank models differently from normalized-space diagnostics;
this is the intended §4 change and should be reviewed as such.

The milestone-3 mass shell plots/table retained the original normalization by
`gdet.shape[-2]*gdet.shape[-1]`. When geometry stores a singleton phi axis, this
is not the number of physical angular cells. Milestone 4 corrects production
fig 5 and Table 2 to divide by the physical Ntheta*Nphi, as energy/angular
profiles already do. The original expression remains only for `space="loss"`
legacy regression. For rho=1/2 at rest, production mass profiles are 1/2 and
the signed-log error is log(3)-log(2), independent of geometry's phi storage;
legacy profiles are multiplied by Nphi. A synthetic singleton-phi test checks
both profiles and the changed Table-2 error. See the [owner's note for Kevin](notes/mass_shell_normalization_for_kevin.md).
Full all-start/ring/transport
improvements remain deferred as above. Temporal-spectrum frequency labels use
`c/R_g` instead of the original unsupported mathtext `text` command, so the
declared Matplotlib 3.7 minimum can render every full-tier figure.

## Regression-only checkpoint

`tests/fixtures/tron_legacy_regression.json` pins the owner-approved checkpoint
by SHA256. Missing or changed files are hard failures in
`python tests/regression_tron.py`; this explicit compute-node command never skips.
It reads original definitions from the pinned private checkout at runtime.
The original evaluator receives exactly 51 symlinked dumps (4216–4266) plus
geometry in a temporary directory; runner history is 4207–4216. Both persistence
and TRON run the same legacy configuration, comparing original numerical locals
at 1e-5 relative tolerance. All generated artifacts are temporary, outside the
shared runs/references/leaderboard. The older checkpoint trained on Kevin's
80/10/10 split, including this validation chunk; **these are port-regression
checks, never benchmark model scores**. Public CI runs synthetic tests; it cannot
claim the required private-data regression passed.

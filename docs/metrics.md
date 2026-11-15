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
horizon, correlation times and every other analysis are deferred to milestone 3.

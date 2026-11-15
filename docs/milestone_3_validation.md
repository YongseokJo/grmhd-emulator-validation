# Milestone 3 validation receipt

Validated on DeltaAI `gh021` in interactive Slurm allocation `3609288`,
`ghx4-interactive`, account `bgun-dtai-gh`, 2026-11-15. Heavy/data checks ran
inside the allocation. Native base: `python/anaconda3/2.15.0` with the existing
system-site-packages venv. The recorded platform lock was not modified.

The separate prediction-mode/spec commits precede the analysis implementation.
The analysis implementation and subsequent fixes are committed through `3dee560`.

| Check | Result | Coverage / elapsed time |
|---|---|---|
| persistence legacy source regression | pass | 226,552 values; 387.398 s |
| tron legacy source regression | pass | 226,552 values; 290.803 s |
| Native NumPy 2.5.4 / torch 2.15.0 | pass | 67-test full suite, 192.202 s; 2 source-focused tests, 7.989 s (includes the added legacy-arithmetic case) |
| Minimum NumPy 1.24.4 / torch 2.3.1 / matplotlib 3.7.5 | pass | 68-test full suite, 229.853 s |
| Read-only doctor | pass | Locked environment, no mismatches; 114 shared entries scanned in 0.173 s, 7 ACL subprocesses, no unfinished reference candidates |

The source comparisons cover all 23 non-quick analyses, every reported horizon,
all per-step correlation/predictability-horizon fields and direct numerical
intermediates observed in the original source. Both cases pass the approved
**1e-5 relative tolerance with zero absolute tolerance**, including undefined-value
matching. No tolerance was relaxed. The explicit legacy correlation mode retains
Kevin's float32 normalization/reduction before float64 aggregate arithmetic;
production metrics retain float64 NumPy CPU arithmetic.

Checkpoint SHA256:
`a81d05b44a86752e45c14ad91f6bf827f5213632b36a29d19ce47a16dbc188eb`.
It is pinned in `tests/fixtures/tron_legacy_regression.json`; missing or changed
files fail. Hashing/loading checks the same opened checkpoint and detects
concurrent changes. The older checkpoint trained on a split overlapping this
validation chunk. These are **port-regression checks, not benchmark submissions**.
Only temporary artifacts were written for the regressions, using history
4207–4216 and targets 4217–4266. The original evaluator received exactly 51 dump
symlinks plus geometry. No TRON output was published to runs/references/leaderboards.

## Quick reference preservation and conformance

Normal `gval reference` regeneration refused any numerical change; both existing
milestone-2 result fingerprints were reproduced. Only reference metadata was
refreshed through the normal owner command. `gval conformance` then passed on the
same platform and lock with no differing files and largest relative difference 0.

| Split | Reference time | Conformance time | Result fingerprint |
|---|---|---|---|
| rc300_reduced_v1 | 1095.43 s | 655.56 s | `6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6` |
| 5000run_v1 | 1000.73 s | 994.38 s | `cd9598fd3f8380f9ebc862db46eb42d070fcfc7f43e532e7bd0f9fc7c51cf648` |

The conformance artifacts are
`staging/gkerex/conformance/deltaai-aarch64__3` (rc300) and
`staging/gkerex/conformance/deltaai-aarch64__4` (5000run), under the shared root.
Timings include real data I/O and overlapped allocation work; they are validation
wall times, not inference-speed benchmark scores.

Platform lock SHA256:
`35994d88a30a821cac354665ffee35adfc0a5090fe9052d27816d6c241bad525`.
Protocol SHA256:
`22696b4a616fcdea8ca5e67d4d14fcb8e8c20ceaa70d591e9f071ab02f1c0fb3`.

## Review boundary and documented deviations

- Prediction mode is specified in §6.1 and remains milestone 5; no agent-prediction
  implementation or Eden adapter is included in this milestone.
- Full analyses use the intermediate first-common-start policy; final all-start
  aggregation/CIs, transport/coarse-ring changes and truth caching remain milestone 4.
- Physical metric-space changes, training scale denominators, odd-lag third-moment
  correction, original mass-shell normalization and portable figure labels are
  explicit in `docs/metrics.md`; legacy computations remain independently testable.
- Owner-generated real training scale artifacts are still needed before real
  production full-tier evaluations. Their command/format and train-only arithmetic
  are tested on synthetic data. No real production full-tier run was published.
- No frozen split, source dataset, reference checkout, sibling emulator, shared
  layout/permission contract or recorded environment lock was changed.

The owner reviews and merges PR #3. Milestone 4 has not started.

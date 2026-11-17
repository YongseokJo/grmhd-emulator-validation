# Milestone 4: final production semantics

The owner-approved changes and formulas are in [the spec](specs/pipeline.md#510-final-all-start-statistics-and-truth-cache-contract)
and [metrics](metrics.md#milestone-4-production-semantics). The mass-shell issue
has a separate [note for Kevin](notes/mass_shell_normalization_for_kevin.md).
Verified training moments and artifact hashes for both datasets are in
[the channel-scales receipt](milestone_4_channel_scales.md).

Full analyses preserve all per-start values and report means and pointwise 95%
chunk-block intervals. Conditional/error correlation use every common single-step
start; other full analyses use every common rollout start. Quick result bytes
keep their established semantics. Transport uses the owner's inward-positive
fluxes, rest-mass-subtracted jet power, positive-inflow efficiency guard and
resolved rc300 headline.

Before finalization full evaluations remain immutable PREVIEW bundles in staging.
The final protocol hash forms a new comparison/reference cohort; old previews
are never promoted. The owner creates shared truth and persistence artifacts
on a CPU/one-thread Slurm allocation:

```bash
export GVAL_VENV=/u/gkerex/project/gval-envs/deltaai-aarch64-pt2.15
source scripts/load_environment.sh deltaai-aarch64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
gval reference persistence --split-id rc300_reduced_v1 --tier full
gval reference persistence --split-id rc300_reduced_v1 --tier quick
gval conformance --split-id rc300_reduced_v1 --tier all
gval reference persistence --split-id 5000run_v1 --tier full
gval reference persistence --split-id 5000run_v1 --tier quick
gval conformance --split-id 5000run_v1 --tier all
```

The full reference builds a missing verified truth cache in the same evaluation.
`scripts/validate_milestone4.py` records the command verdicts, fingerprints and
wall times in an ignored local receipt. `--split-id <id>` validates one dataset
and appends to that receipt when datasets need separate Slurm allocations. The
overall receipt is successful only after both datasets' three operations pass.
`--operation reference-full`, `--operation reference-quick` and `--operation
conformance` can split that chain at completed-operation boundaries when an
interactive partition's wall limit is shorter than the whole validation chain.
Each completed analysis records process RSS and Slurm cgroup memory in
`logs/memory.json`. A diagnostic-only short memory profile can be run in the
same compute environment with
`python scripts/profile_memory.py --starts 3 --output gval-memory-profile.log`.
Its shortened windows are never benchmark results or shared references.
To refresh presentation from a completed full-tier bundle, run on a compute
node:

```bash
python scripts/render_aggregate_figures.py --source <full-reference-or-run-directory> \
  --receipt-dir docs/artifacts/milestone4
```

It writes a versioned owner report bundle with all figures, the contact sheet,
label/CI audit and source hashes. The source result fingerprint is checked
before and after rendering; no inference is run. Multi-step overlays retain
every mean curve and show bands only at the final report step. The
dimensionless-flux time series retains both role bands across all steps.
For an owner persistence golden, `--refresh-reference` atomically refreshes
its derived PNGs after checking the unchanged numerical fingerprint. Normal
reference regeneration also refreshes presentations when result bytes match.
For error-map colour review, add `--error-map-crops`. This checks visible-cell
colour occupancy and exports the fig08 error columns, fig14 bias/RMS rows and
fig16 error row. With `--receipt-dir`, those crops are copied into repository
documentation instead of the standard contact-sheet/fig18 pair.
For a separate cache (including test truth), the owner runs
`gval reference truth --split-id <id> --split validation|test`. Cache entries are
read-only to readers and pinned by hashes, including training scales. Full
evaluation fails on missing/mismatched/corrupt caches and requires passing full
conformance for the current platform, lock, split and protocol hash. No learned
model output enters a shared truth cache. Goldens cannot be overwritten with
unequal content. Interrupted owner candidates are removed when possible;
doctor reports any candidates left by a process killed before cleanup.

Verification runs on compute nodes:

```bash
PYTHONPATH=tests python -m unittest discover -s tests -v
PYTHONPATH=tests .venv-numpy124/bin/python -m unittest discover -s tests -v
PYTHONPATH=tests python tests/regression_tron.py --kind both
```

The learned regression is only a numerical reproduction check using the pinned
checkpoint/window; its training overlaps validation. It writes to temporary
storage and is never a benchmark submission. Runtime/timing intervals live in
run.json, outside deterministic result content.

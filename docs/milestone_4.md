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

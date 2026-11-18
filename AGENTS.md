# Shared GRMHD validation benchmark

This repository owns the frozen validation contract and common scorer. Models
receive physical arrays through adapters. Every model uses the same split,
starts, history buffer, metrics and artifact writers. Full bundles live in
`/work/nvme/bgun/share/validation`. Read README.md for status and
`docs/specs/pipeline.md` for the complete contract.

## What a participant writes

Write only your model adapter and `run_request.json` in your own workspace.
Keep model code in your model repository. Use an installed adapter when one is
already available. Persistence is installed as `persistence`; its ready request
is `adapters/persistence/run_request.json` (checkpoint null). Do not replace it
with a newly written implementation. `new-adapter` is for a new model only.

## Commands in order

1. `gval doctor` must succeed in the approved environment.
2. Run `gval conformance --split-id <id> --tier quick` before quick evaluation.
   Run full conformance too before a requested full evaluation. Never change a
   golden or turn off checksums to get a pass.
3. For a new model, `gval new-adapter <your_dir>`, implement the adapter and edit
   its request. Skip this for installed persistence/TRON/ViT/Eden adapters.
4. `gval check-adapter <adapter> --split-id <id> [--checkpoint <weights>]`.
5. `gval evaluate --request <request> --split-id <id> --split validation --tier quick`.
6. After quick review, run full validation if requested, then full test only
   after the user requests final testing. Never use test scores to select a
   method, checkpoint or retry.
7. `gval verify <published_run_dir>`. Report the final JSON verdict and stop at
   the user's requested scope. With no tier specified, complete quick validation
   and stop; do not automatically launch full or test.

"rc300" means frozen `rc300_reduced_v1`; the other dataset is `5000run_v1`.
For "validate the persistence adapter on rc300 validation", execute this exact
workflow on CPU, with the existing persistence request, through published quick
validation and verification. No training record or checkpoint is needed.

Evaluation checks Git status for `splits/`, `protocols/`, `grmhd_validation/`,
`schemas/` and `prompts/`, including staged and untracked files. Modified
protected paths produce a failed staging attempt with `repo_modified: true`;
they cannot publish. Shell and Codex edits are covered by this check too.
Changes to your request/adapter workspace outside those paths are allowed.
Do not commit changes to protected benchmark files to bypass participant rules;
pipeline maintenance belongs to the owner-reviewed branch/PR workflow.

### DeltaAI compute setup

Never load torch, render, hash large data, run tests, check-adapter, conformance
or evaluate on a login node. Use `ghx4-interactive`, account `bgun-dtai-gh`, for
short work. The owner shares the one-interactive-job-per-user limit. If occupied,
inspect `squeue -u "$USER"`, wait and resubmit; never cancel the owner's job or
fall back to `ghx4`. Ask only after about two hours waiting. No watcher loops.

From the repository root, obtain one allocation and chain all quick work in it:

```bash
salloc --account=bgun-dtai-gh --partition=ghx4-interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-node=1 \
  --mem=32G --time=02:00:00 \
  srun --cpu-bind=none bash -lc '
    set -euo pipefail
    umask 007
    cd /absolute/path/to/this/repo
    source scripts/load_environment.sh deltaai-aarch64
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export CUDA_VISIBLE_DEVICES=""
    gval doctor
    gval conformance --split-id rc300_reduced_v1 --tier quick
    gval check-adapter persistence --split-id rc300_reduced_v1 --device cpu
    gval evaluate --request adapters/persistence/run_request.json \
      --split-id rc300_reduced_v1 --split validation --tier quick --device cpu
  '
```

Replace only the repository path, adapter/request and dataset as required by
the actual task. The helper loads `python/anaconda3/2.15.0`, activates conda base,
and creates a personal system-site-packages venv with only this package and
missing approved dependencies. `GVAL_VENV` may select an existing compatible
personal venv. Never module purge or use pytorch-conda. Never let pip replace the
module's torch/numpy/scipy/matplotlib. Delta x86_64 has no approved base yet.
After evaluate, use its returned run_dir for `gval verify` in the same approved
compute environment (or add it to your chained job). Each command's last line
is JSON with `ok` and `next`; follow it within the requested scope. A conformance
failure is a failure, not permission to edit the pipeline. Inspect completed
job output at meaningful boundaries; do not poll every second or keep watchers.

## What to report

Report split/partition/tier, adapter and checkpoint identity, Slurm job ID,
doctor/conformance/check-adapter/verify results, published run path and run_id,
result fingerprint and data-verified status. Preserve unlocked, unattested and
training exposure labels. Report failures/NOT_EVALUATED/preview honestly. Do
not claim blind prediction generation merely from a host or agent statement.
The owner publishes reports/results through `gval report`; participants do not
copy bundles to shared storage or edit leaderboard numbers.

Prediction mode has separate roles and instructions:
`docs/prediction_roles/host.md`, `forecasting.md`, and `scorer.md`. A forecasting
agent receives only its isolated exported packet and versioned prompt, never
this benchmark workspace or evaluator. The host freezes all required steps;
the scorer requires all 12 rollout starts × 50 steps and every single-step start
before the corresponding analyses count as evaluated.

## Forbidden participant actions

Do not edit `splits/`, `protocols/`, `grmhd_validation/` or `references/`.
Do not change raw data, shared-folder layout/permissions, completed runs, goldens,
or the read-only source checkout. Do not modify sibling ErgodicEmulator.
Never read dumps outside the history passed by the runner from model code.
Never copy files into the shared folder by hand, fake a missing prediction,
shorten protocol coverage, repeat test runs to improve a number, or cancel
another/owner job. Only gval creates shared artifacts.

These are participant instructions. Explicit owner-authorized pipeline
maintenance takes precedence; it uses reviewed branches/PRs and milestone stops.

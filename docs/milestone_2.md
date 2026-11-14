# Milestone 2: adapter, quick evaluation and conformance

Draft: publication currently fails on Linux under the spec's freeze-before-rename
ordering. The planner must resolve this before milestone 2 can be approved.
See the [validation receipt and pending questions](milestone_2_validation.md).

Milestone 1 is merged. This milestone adds the model-facing API, persistence,
one common inference/metric path, recorded staging attempts, canonical result
artifacts, owner references, conformance, bundle verification and Slurm wrappers.
Full-tier analyses, learned-model examples, training records and reporting remain
in their later milestones.

Install on the chosen base with `python -m pip install --no-deps -e .`; never let
pip replace the module's torch/numpy/scipy/matplotlib. Delta uses
`module reset; module load pytorch-conda/2.8`. DeltaAI's base remains unnamed.
Without a platform lock, doctor warns and goldens stay provisional.

## Owner: create a golden during review

Submit from this repository root on a CPU partition with your CPU account:

```bash
export GVAL_PYTHON=/absolute/path/to/prepared/python
sbatch --account=YOUR_CPU_ACCOUNT --partition=cpu scripts/reference_delta.sbatch rc300_reduced_v1
sbatch --account=YOUR_CPU_ACCOUNT --partition=cpu scripts/reference_delta.sbatch 5000run_v1
```

On DeltaAI, override account/partition and reserve the GPU Slurm requires; the
script hides CUDA and still uses CPU with one thread. It does not invent a
DeltaAI base environment. The wrapper runs:

```bash
gval reference persistence --split-id rc300_reduced_v1 --tier quick --split validation --device cpu
```

Reference output is `references/<split_id>/forecast_v1/persistence_quick/<platform>/`.
`golden.json` hashes only summary, metric JSON and table CSV. Existing different
content is refused and the candidate retained. Identical content can acquire
updated metadata when the owner records a real lock. Figures and runtime
metadata do not enter the fingerprint. Failed or unfinished candidates are not
goldens.

## Everyone: conformance and quick evaluation

Run `gval doctor` first. Enter a Slurm compute node, including for CPU operations;
a job ID in a login shell is insufficient. For example, in an interactive job:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
gval conformance --split-id rc300_reduced_v1
gval evaluate --request adapters/persistence/run_request.json \
  --split-id rc300_reduced_v1 --split validation --tier quick --device cpu
```

Conformance always uses CPU/one thread. It writes exclusively under
`staging/<user>/conformance/<platform>__<n>/`. It fails with the owner-reference
instruction when no golden exists. Same-platform result bytes must match; an
other-platform golden uses 1e-12 relative tolerance and records a warning.
Evaluation requires a passing conformance for that user, split, protocol and
platform/lock, and records its fingerprint. Running conformance does not create
leaderboard attempts.

GPU model requests can use the corresponding wrapper:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_A100_PARTITION scripts/evaluate_delta.sbatch \
  --request /absolute/path/run_request.json --split-id rc300_reduced_v1 \
  --split validation --tier quick --device cuda
# On GH200 use scripts/evaluate_deltaai.sbatch with your prepared GVAL_PYTHON.
```

No protocol settings are CLI flags. `--data-root` relocates physical reads while
preserving split identity. `--no-verify-data` explicitly marks evaluation
unverified; geometry must still match the frozen geometry SHA. The runner never
lists simulation directories and never gives excluded/train/other-partition
dumps to a model.

Each evaluation reserves a numbered staging attempt, including failures. Success
has exact artifact names, checksums and a COMPLETE marker, then is published by
an atomic no-replace rename to `runs/<split_id>/<model>/<run_id>/`. Completed
bundles are read-only. Failures retain FAILED, run metadata and a traceback in
staging. Destination namespaces are created only during final publication.
Existing shared permissions are never repaired, and only invocation-created
inodes can be chmodded.

```bash
gval verify /work/nvme/bgun/share/validation/runs/<split_id>/<model>/<run_id>
gval verify --compare /path/run_a /path/run_b
```

Verification checks coverage and file hashes, schemas, frozen identities,
completion and result fingerprint. Comparison requires the same request and
uses the protocol's 1e-4 relative tolerance. Metadata varies between attempts;
result-content bytes are the CPU determinism contract.

## Adapter contract

```python
from grmhd_validation import ModelAdapter, Geometry
```

`Geometry` holds tensors only, with r/theta/phi, covariant and contravariant
metrics, determinant, dx, spin, dt_dump and resolved mask. `step(history)` takes
physical finite float32 `[B,h,8,Nr,Ntheta,Nphi]` and returns physical float32
`[B,8,Nr,Ntheta,Nphi]` on the same device. h must be in 1..H_max. The runner
maintains history, selects starts, disables gradients, requests eval mode and
checks every result without clamping it. It currently evaluates one start at a
time (within every adapter's declared max_batch).

Use `/absolute/file.py:Adapter` or an installed entry point in
`grmhd_validation.adapters`. A request contains exactly model, version, adapter,
checkpoint, options, training_record and notes. Persistence requires a null
checkpoint and records the `nockpt` run token. Checkpoint-required adapters reject
null. Source hashes and source repository commit are recorded.

`gval new-adapter /your/model-adapter` creates a template and request without
overwriting existing files. The learned-model examples and full walkthrough are
milestone 5. See [metric definitions](metrics.md) and [source credits](provenance.md).

## Tests and doctor scaling

```bash
python -m unittest discover -s tests -v
```

Run the scientific suite inside Slurm on the cluster. It uses synthetic dumps,
geometry and shared-layout temp directories, never the real simulation data.
Use separate environments for NumPy 1.24 and 2.x. Source regressions read selected
definitions at `main@ebcc218` from the read-only private checkout; elsewhere set
`GVAL_REFERENCE_CHECKOUT`, or those two source tests explicitly skip.

Doctor scans mode bits with os.stat and consults getfacl only when POSIX extended
ACL xattrs exist. Identical ACLs share cached parsing. The output includes elapsed
scan seconds, entry count and subprocess count. The 1,000-run synthetic tree test
also checks an unreadable owning-group ACL hidden behind a readable mask and
asserts a scan below 30 seconds.

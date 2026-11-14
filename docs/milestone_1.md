# Milestone 1: foundation commands

This milestone implements only package/CLI setup, frozen schema-v2 split reading,
the approved forecast protocol, doctor, checksums, shared permission helpers, and
the environment recording workflow. Adapter APIs, evaluation, conformance,
metrics, and reporting are later milestones and are not implemented here.

## Install without replacing module packages

Use Python >=3.10 in the approved base environment:

```bash
python -m pip install --no-deps -e .
gval doctor
```

The approved lower bounds are in `environment/requirements.in` and `pyproject.toml`.
The Delta base is `module reset; module load pytorch-conda/2.8`. DeltaAI's base is
deliberately null. No environment lock is invented: doctor returns an unlocked
environment warning until the owner records that machine's lock.

Every CLI invocation sets umask 007 before command imports. Every command's last
line is a JSON verdict containing `ok` and `next`, including usage errors and help.
Doctor is read-only: it checks membership, modes, effective/default ACLs, group
readability, `os.access` on staging, geometry, split identity and checksum-list
coverage. It does not hash every dump. It never creates or repairs shared paths.
The reported `next: gval conformance` becomes available in milestone 2.

## Owner checksum jobs

Submit from the repository root with the account/partition for your machine:

```bash
export GVAL_PYTHON=/absolute/path/to/approved/python
sbatch --account=YOUR_ACCOUNT --partition=cpu scripts/checksums.sbatch splits/rc300_reduced_v1.json
sbatch --account=YOUR_ACCOUNT --partition=cpu scripts/checksums.sbatch splits/5000run_v1.json
```

DeltaAI requires a GPU reservation even though this job uses only CPUs; select
its partition and GPU allocation explicitly when running there. The same wrapper
can run inside an interactive allocation after entering its compute node with
`srun`. A Slurm job ID on a login shell alone does not permit hashing.

`gval checksums --split-file … [--data-root …]` hashes **all** listed dumps,
including excluded dumps, in numeric dump order. Geometry is appended last and
must match the split's `geometry_sha256`. The result is published atomically as
`/work/nvme/bgun/share/validation/checksums/<split_id>.sha256` in sha256sum format.
The root override changes where bytes are read, not the split SHA256. An existing
identical checksum file is reused; a differing file is not overwritten.

Only gval writes shared outputs. Existing shared layout, modes and ACLs are never
modified. New group-area child directories are 3770; new owner-area child
directories are 2750. Completed-tree permission helpers operate only on entries
created and inode-tracked in the current invocation; unknown/replaced/symlink
entries fail rather than being chmodded. The actual staging-to-runs runner is
milestone 2.

`SplitFile.model_dumps` refuses the excluded split, while the checksum path can
read it. The runner/adapter test proving excluded data never reaches a model is
added with the actual runner in milestone 2.

## Record a platform lock

```bash
sbatch --account=YOUR_ACCOUNT --partition=cpu scripts/record_lock.sbatch delta-x86_64
```

The owner runs this on the target machine. The script loads the platform base,
checks all approved lower bounds, refuses missing core packages or any pip plan
that changes them, installs this package with `--no-deps -e .`, and adds only
missing non-core packages. Its `.lock` file is JSON with Python/torch/CUDA
versions, core package versions and sorted `pip freeze --all`. No lock is
committed until it has actually been recorded on that platform. DeltaAI recording
fails while its base is null. Doctor treats core-version differences as failures
and other freeze differences as warnings.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic NumPy files and temporary storage only. Run the suite in
isolated environments with NumPy 1.24 and NumPy 2.x; never replace a module's NumPy
to perform the matrix. ACL inspection tests require `getfacl` (and skip where it
is unavailable). They cover split coverage and path safety, excluded model
access, deterministic checksum order, changed/missing hashes, group readability,
lazy creation, readonly tracking, doctor behavior, schemas, environment mismatch
severity and refusal of unsafe package-install plans.

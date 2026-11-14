# Milestone 1 validation receipt

Scope: milestone 1 only, using the planner's approved clarifications in
`eb438c9`, `d170d1e`, `9f88d5f`, and `5ace1b2`. No milestone 2 implementation has
started. No protocol deviations were introduced.

## Synthetic and contract tests

Command: `python -m unittest discover -s tests -v`.

| Isolated test environment | Result |
|---|---|
| Python 3.11.13, NumPy 1.24.4, jsonschema 4.18.0 | 22 tests passed |
| Python 3.12.11, NumPy 2.5.4, jsonschema 4.18.0 | 22 tests passed |

These are lightweight foundation test environments, not recorded/approved
full-model environments. No module package was replaced, and no platform lock
was fabricated. Shell syntax checks and `git diff --check` also passed.

Coverage includes schema-v2 split identity and chunks, no data directory scans,
root override, excluded model-data rejection, checksum coverage/order/tampering,
geometry identity, atomic publication and cleanup, group/mode inheritance,
read-only inode tracking, no chmod of existing/foreign files, read-only doctor,
unlocked/locked environment severity, approved bounds and refusal of unsafe pip
plans. The actual runner-to-adapter excluded-data test belongs to milestone 2,
when those components exist.

## Real-data Slurm check

Interactive allocation **3604972**, DeltaAI node **gh010**, 32 CPUs, 16 GiB,
one mandatory GPU reservation. CUDA was disabled and hashing used only CPUs.
The allocation was released after the jobs.

The owner ran the supplied `scripts/checksums.sbatch` wrapper for each frozen
split file. Both commands returned `ok: true`. Hashing covered every dump,
including excluded dumps, plus geometry checked against `geometry_sha256`.

| Shared output under `checksums/` | Entries | File SHA256 |
|---|---:|---|
| `rc300_reduced_v1.sha256` | 9637 | `5d9da7ba1c85f45d21911db528d5b8ff09886bae828d2fd235f44f193fd284e8` |
| `5000run_v1.sha256` | 5702 | `a2f651a52f438854a43d625bc4e1062a7fad14f38623a925dcb3f6de28b80ff2` |

The two newly created files are mode 0640 and group `delta_bgun`. Existing shared
directories' owner/group/modes and ACLs were recorded before and after and are
identical. Frozen split files and source datasets were not modified. No shared
directories were created or repaired; no files were copied into shared storage
by hand. Kevin's reference repository was not modified.

## Doctor and environment recording

The real `gval doctor` check after hashing reports:

- Both shared split identities, geometry hashes and checksum-list coverage: pass.
- No group-unreadable shared files found.
- Missing DeltaAI platform lock: warn, `environment_locked: false`, as approved.
- Existing root default ACL: fail. The shared root has default group and default
  mask `rwx`; section 8 requires `r-x`. These existing permissions were left
  unchanged, as instructed. The owner must resolve this storage discrepancy.

`scripts/record_lock.sbatch deltaai-aarch64` was also exercised inside the compute
allocation. It returned the expected failure and JSON instruction to ask the
owner for a base environment, because the approved platform value is still null.
No lock was written. A positive recording test on Delta's
`pytorch-conda/2.8` base remains owner-run on that machine.

Milestone 1 is ready for review. Milestone 2 requires separate approval.

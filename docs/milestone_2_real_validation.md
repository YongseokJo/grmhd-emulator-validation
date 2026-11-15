# Milestone 2: locked DeltaAI real-data validation

All four commands succeeded in interactive Slurm job **3608853**, node **gh152**,
account bgun-dtai-gh, partition ghx4-interactive. Commands ran sequentially in one
allocation: record the lock, commit it, reference rc300, reference 5000run,
conformance rc300, conformance 5000run. Both references and conformance used CPU
with one thread. Data verification was enabled.

The existing personal venv at
`/u/gkerex/project/gval-envs/deltaai-aarch64-pt2.15` was validated and reused.
It has system-site-packages enabled and only pip and the editable validation
package installed locally; jsonschema and scientific packages come from the
approved `python/anaconda3/2.15.0` conda base. No module purge/reset or pytorch-conda
was used. Delta x86_64 is null/unsupported. All job wrappers use the common loader.

## Recorded environment

Lock: `environment/deltaai-aarch64.lock`, SHA256
`35994d88a30a821cac354665ffee35adfc0a5090fe9052d27816d6c241bad525`. Recording took 28 seconds on the compute node. The actual snapshot
was committed before generating the goldens; it was not edited by hand.

| Component | Version |
|---|---|
| Python | 3.12.9 |
| torch | 2.15.0+cu132 |
| CUDA | 13.2 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| matplotlib | 3.11.2 |
| jsonschema | 4.26.0 |

## Results

Both conformance runs pass bit-for-bit on the same platform and lock, with no
differing result files and largest numeric relative difference **0.0**. Both
goldens are **non-provisional**. Fingerprints were also independently recomputed
from the five result-content files in each golden and conformance directory.
Timings below are monotonic wall time per successful command, including reference
wrapper environment setup.

| Split ID | Reference and conformance fingerprint | Reference time | Conformance time |
|---|---|---:|---:|
| rc300_reduced_v1 | `6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6` | 424.752 s | 393.993 s |
| 5000run_v1 | `cd9598fd3f8380f9ebc862db46eb42d070fcfc7f43e532e7bd0f9fc7c51cf648` | 973.010 s | 971.610 s |

Goldens live under
`/work/nvme/bgun/share/validation/references/<split_id>/forecast_v1/persistence_quick/deltaai-aarch64/`.
Conformance paths and command-level timing records are in
`gval-m2-review-results.log` in this checkout; per-command logs are
`gval-m2-real-reference-<split_id>.log` and
`gval-m2-real-conformance-<split_id>.log`. Structured results remain on group storage.

## Mismatches, filesystem fix and cleanup

There are no core-version or numerical result mismatches. Doctor reports only
an expected non-core warning: pip freeze records the editable repository's Git
revision, which changes after committing the recorded lock/code. That warning is
kept visible under the spec's non-core mismatch rule. Run metadata separately
records the actual repository commit.

The first rc300 evaluation completed but could not publish because the shared
filesystem returned EINVAL for renameat2(RENAME_NOREPLACE). That attempt took
398.954 seconds and produced fingerprint
`6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6`.
Publication now prefers the kernel flag and falls back to an atomic POSIX rename
under a destination-directory lock and existence check. Every permitted gval
writer uses this lock. Concurrency tests check that an existing empty target
cannot be replaced, and genuine access errors never trigger fallback. The rc300
retry produced exactly the same fingerprint. The duplicate failed candidate was
removed only after its contents matched the published golden exactly.

The owner's explicitly named cancelled `.candidate-7d9428a055d74986915f5d94e494ea6f`
was deleted. Doctor now reports unfinished reference candidates with status warn
and their paths without deleting or changing them; interruption remnants are
visible. The final doctor storage scan passes: 82 entries in
0.521 seconds, 7 getfacl calls, no unfinished candidates and no
group-unreadable files. No source data, frozen split, read-only reference checkout
or existing shared permissions/layout were changed.

## Tests and scope

All **54 tests pass** in the approved NumPy 2.x base and the isolated NumPy 1.24
minimum-version environment, on this same allocation. The new coverage includes
split-specific run IDs, one-level attempt counting, unfinished-candidate warnings,
and the shared-filesystem publication fallback. Existing quick-metric regression,
bootstrap, deterministic output, checksum, conformance and sealing tests pass.
The 1,000-run doctor test remains below one second with seven getfacl subprocesses.
Compilation, shell syntax and whitespace checks pass.

The implementation follows planner commits 583fdc9/8912407 and does not edit the
planner specs. Milestone 3 has not started; PR #2 remains for the owner to review
and merge.

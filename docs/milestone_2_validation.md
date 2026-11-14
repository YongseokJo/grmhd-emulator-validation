# Milestone 2 validation receipt (draft)

Milestone 2 is **not complete**. The implementation follows the current
freeze-before-rename requirement, and the integration tests expose an OS-level
conflict. The planner's decision is required before publication can pass.

## Compute environment

Scientific tests ran in interactive Slurm allocation 3605039 on DeltaAI node
`gh017`. No real evaluation or golden was generated; reference jobs on the two
real datasets remain the owner's milestone-2 review action.

Two isolated test interpreters were used without replacing module packages:

- Python 3.11.13, NumPy 1.24.4, torch 2.3.1, scipy 1.10.1, matplotlib 3.7.5.
- Python 3.12.11, NumPy 2.5.4, torch 2.15.0, matplotlib 3.11.2.

These are unlocked test environments, not approved platform bases or locks.

## Results

`python -m unittest discover -s tests -v`: 47 tests; 41 pass, six fail at final
publication. The failed tests exercise successful runs after normal, failed,
corrupted-input and explicit unverified attempts. All six currently hit Linux
EACCES from moving the already read-only run directory between parents.

The 41 independent checks pass in both NumPy versions. They cover the foundation
contract, schema validation, port attribution, templates, canonical serialization,
result-only fingerprints, no-replace rename, common starts, chunk boundaries,
excluded-frame refusal, geometry identity, bootstrap oracles, strided sigma,
reference-repeat identity/refusal to replace changed content, same-platform exact
conformance, cross-platform tolerance/warning, and Kevin's pinned source math,
palette and ticks. Runtime source regression uses main@ebcc218 and the approved
1e-5 relative quick-metric tolerance; the full checkpoint regression is milestone 3.

Doctor's 1,000-run synthetic tree contains 9,010 entries, including eight files
per run and an unreadable owning-group ACL with a readable mask. The minimum-env
scan took 0.106 seconds with seven getfacl subprocesses, below the 30-second test
bound. Plain mode ACLs require zero getfacl subprocesses.

Python compilation, shell syntax checks and git diff whitespace checks pass.
Frozen splits and planner specs have no implementation changes. Real geometry
NPZ keys/shapes were inspected read-only on the compute node; no simulation dumps
were altered. No new files or permission changes were made in shared storage.

## Planner decisions pending

1. Spec §7/§8 removes all run-directory write bits before rename. Linux requires
   write permission on a directory moved between parents to update its `..`
   entry. Proposed resolution: freeze contents, retain owner-write on the run
   root for the atomic rename, then immediately freeze the root via an already
   open FD. This proposal is **not implemented** pending a spec update.
2. §6 check-adapter needs a dataset selection and a definition of “a few” dumps.
   Proposed: required --split-id, frozen protocol rollout starts, two calls per
   start, CPU wall time / CUDA peak allocated memory. Alternatively defer this
   helper to milestone 5. No sampling rule has been invented.
3. §11 requires recording adapter-enabled autocast but §6 has no declaration.
   Proposed boolean `uses_autocast = False`, recorded as adapter_autocast, with
   adapters overriding it when enabling autocast. The runner already disables
   autocast around inference. This declaration awaits planner approval.

No milestone-3 work has started. The PR stays draft until publication and the
remaining contract questions are resolved, then the full matrix must pass.

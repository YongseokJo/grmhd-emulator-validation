# Milestone 2 validation receipt

Milestone 2 follows spec §7/§8 at planner commit 91d04ac: write files, checksums
and COMPLETE; freeze the contents bottom-up; atomically publish without replacing
a destination; seal the run root. The implementation uses an already-open root
FD to seal the same invocation-created inode after rename.

## Compute environment

The final scientific suite ran in interactive Slurm allocation 3605215 on
DeltaAI node `gh151`. No real evaluation or golden was generated; reference jobs
on the two real datasets remain the owner's milestone-2 review action.

Two isolated test interpreters were used without replacing module packages:

- Python 3.11.13, NumPy 1.24.4, torch 2.3.1, scipy 1.10.1, matplotlib 3.7.5.
- Python 3.12.11, NumPy 2.5.4, torch 2.15.0, matplotlib 3.11.2.

These are unlocked test environments, not approved platform bases or locks.

## Results

`python -m unittest discover -s tests -v`: **50 tests pass in each environment**,
including all six previously failing publication tests. No tests are skipped on
this machine. Scientific work runs on the allocated compute node, not a login
node.

The tests cover the foundation contract, schema validation, port attribution,
templates, canonical serialization, result-only fingerprints, no-replace rename,
common starts, chunk boundaries, excluded-frame refusal, geometry identity,
bootstrap oracles, strided sigma, repeated reference identity/refusal to replace
changed content, exact same-platform conformance, cross-platform tolerance and
warning, runner autoregression, malformed predictions, failed-attempt retention,
checksum failures and explicit unverified runs, deterministic results and figures,
and verified repeat comparison. Runtime regression reads Kevin's pinned source
at main@ebcc218 using the approved 1e-5 relative quick-metric tolerance; palette
and ticks match exactly. The full fixed-checkpoint regression remains milestone 3.

Publication tests assert that COMPLETE and frozen contents exist while the root
is still owner-writable at rename, and that the root is read-only afterward.
A simulated seal interruption leaves the same COMPLETE run under runs/, without
rewriting metadata/checksums or creating FAILED. Verify returns unsealed and
comparison refuses it. The CLI owner recovery test seals that existing inode,
preserves every child inode/mode/mtime/hash, then verifies successfully.
Recovery also refuses foreign ownership, writable files/directories, changed
hashes, missing COMPLETE, symlink roots and special files. Sealing an already
sealed valid run makes no chmod call.

Doctor's 1,000-run synthetic tree contains 9,010 entries, including eight files
per run and an unreadable owning-group ACL hidden behind a readable mask. Its
scan stays below one second in both test environments using seven getfacl
subprocesses, well below the 30-second test bound. Plain mode ACLs require zero
getfacl subprocesses. Real doctor also succeeds with the expected unlocked
warning and makes no shared writes.

Python compilation, shell syntax checks and git diff whitespace checks pass.
There are no implementation edits to frozen splits or planner specs. No source
data, reference checkout, existing shared permissions or layout were changed.

## Scope and owner review

No milestone-3 work has started. Full analyses, learned-model adapters, training
records, reporting and agent instructions remain their later milestones. The
check-adapter sampling rule and an adapter autocast declaration have not been
invented; those earlier planner questions remain for the later model-integration
work. The milestone-2 API/persistence/quick runner and golden workflow are ready
for review. The runner disables autocast around inference and records that fact.

The owner runs scripts/reference_delta.sbatch for rc300_reduced_v1 and
5000run_v1, then reviews real-data conformance before merging PR #2. Without a
platform lock those references remain provisional, as the spec requires.

# Milestone 3

This milestone adds the 23 remaining full-tier analyses, with closed JSON/CSV
schemas, fixed figures, per-analysis source attribution and numerical regression
against the original evaluator. It also specifies milestone-5 prediction mode
(in separate spec commits) and excludes this package's own editable revision from
environment comparisons without changing recorded locks.

See [the specification](specs/pipeline.md), [metric formulas and intended
deviations](metrics.md) and [source provenance](provenance.md). Full-tier production
currently selects the first common protocol rollout start. Final all-start
aggregation and the remaining §5 improvements are milestone 4; these intermediate
full results are for review before leaderboard use.

The owner first generates training-only physical reference scales, inside a Slurm
allocation with the approved environment:

```bash
gval reference channel-scales --split-id rc300_reduced_v1
gval reference channel-scales --split-id 5000run_v1
```

This hashes every training input against the common checksum manifest and writes
only through `gval` into the owner-only references area. It refuses unequal
existing scale artifacts. Full evaluations require a matching scale artifact;
quick evaluations do not need it and retain their milestone-2 result bytes.

```bash
gval doctor
gval conformance --split-id rc300_reduced_v1
gval evaluate --request adapters/persistence/run_request.json --split-id rc300_reduced_v1 \
  --split validation --tier full --device cpu
```

Every full evaluation uses the same
runner-owned history and physical predictions. It writes invocation-private
prediction scratch, removes it before checksumming, then freezes/publishes/seals
the bundle with the existing permissions contract. Truth caching is deferred.

The pre-merge compute checks run in one `ghx4-interactive` job:

```bash
sbatch scripts/ci_delta.sbatch
```

This loads `python/anaconda3/2.15.0` and the user's system-site-packages venv via
the existing environment helper, runs the suite, runs both required real
legacy regressions, then conformance on both splits and doctor. Set
`GVAL_MINIMUM_PYTHON` to an existing isolated minimum-version interpreter to test
NumPy 1.24 too; no module package is replaced. The native lock tests NumPy 2.x.

The real regression can be rerun inside that allocation with
`python tests/regression_tron.py --kind both`. Its checkpoint must match
`tests/fixtures/tron_legacy_regression.json`; missing/hash-changed files fail.
Original functions are loaded from pinned read-only git objects. The original
evaluator sees a temporary symlink directory of 4216–4266 plus geometry; the
adapter history is 4207–4216. Both persistence and TRON legacy numbers must agree
within 1e-5 relative. All artifacts are temporary, and these checks never write
benchmark runs, references or leaderboard scores. The checkpoint's old training
split overlaps benchmark validation, which makes it unsuitable as a fair
benchmark submission.

Validation results and timings are recorded in the milestone receipt when the
checks finish. The owner reviews and merges the PR; milestone 4 does not start
until approval.

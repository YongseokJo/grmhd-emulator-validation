# Reports and participant workflow

The owner builds reports from sealed, checksum-verified published runs. Groups
separate dataset, protocol ID/hash, partition and tier. Every run gets its own
row; no best-attempt selection or undocumented model averaging is applied.
Test-attempt totals include incomplete and failed attempts for the same model
and full checkpoint hash across tiers. Unattributed metadata remains visible.
Unsealed, NOT_EVALUATED, failed, preview and staging-only attempts stay off rows.
Corrupt COMPLETE bundles abort replacement of the report so the owner can inspect
the actual failure.

Headline values are final-step skill and loss-space nMAE. The report also keeps
single-step scores, physical relative L1 per channel, final-step physics errors,
data verification, computed training labels, operator-declaration evidence,
environment locks and prediction attestation. `no, overlaps validation/test (n
dumps)` is preserved verbatim. Unknown/unlocked/unattested does not become yes.
Undefined quantities remain null/em dashes, including physics for quick runs.
Comparison figures show nMAE and skill against physical time plus skill bars/CIs;
curves and legend labels distinguish every run.

Run short report work on ghx4-interactive, bgun-dtai-gh. Wait for the owner's
other interactive job; never cancel it or fall back to ghx4. From the repository:

```bash
sbatch scripts/report_delta.sbatch --split-id rc300_reduced_v1 --export-results "$PWD"
```

This verifies shared bundles, refreshes only derived reports in the owner-only
reports area, and exports original run.json/summary.json plus source hashes under
results/. An attempt ledger snapshots failures and exclusions; it is necessary
to reproduce test counts without Delta. A conflicting completed result export
is never overwritten. Group directory modes/layout remain unchanged; atomic
report file replacement uses new group-readable inodes.

Review and commit the generated results, attempt ledger, reports and comparison
PNGs through the milestone PR (the owner merges). Never hand-edit numbers or
copy a run bundle into shared storage. Full bundles remain on Delta.

```bash
gval report --from-results --split-id rc300_reduced_v1
gval report --from-results --split-id rc300_reduced_v1 --check
python scripts/sync_agent_instructions.py --check
```

Replay needs only committed result records, split definitions and protocol Git
history, not data paths, Delta storage or environment locks. It validates schemas,
original hashes, identities, the result fingerprint's file-hash map, and ledger
consistency. Exact JSON/CSV/Markdown bytes are checked. Comparison PNGs are bound
to report-group data and hashes in figures.json, checked without rerendering so
FreeType/matplotlib raster differences do not make CI spuriously fail. Empty
datasets have explicit empty reports; owner references are not fabricated runs.

GitHub CI runs the public tests on NumPy 1.24 and 2.x, validates both dataset
reports and checks that docs/agents/generic.md is byte-identical to AGENTS.md.
The checkout fetches all history to resolve older protocol hashes. Current
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) actions use v7.
Pinned private-source comparisons are skipped on public CI; required real-data
regression remains the owner's Slurm check. Published scores are never computed
by GitHub CI from hidden data.

AGENTS.md is the participant source, CLAUDE.md imports it, and the Claude
validate-model skill follows the same sequence. Edit/Write denies protect
splits/, protocols/, grmhd_validation/ and references/; these are not claimed to
sandbox arbitrary shell commands. Explicit owner-authorized maintenance has
precedence. Installed persistence needs no rewritten adapter. Default scope is
quick validation; full/test require the user's requested scope. Prediction mode
retains [separate roles](prediction_roles/host.md); forecasting agents see only
their packet, never the repository-wide participant/host instructions.

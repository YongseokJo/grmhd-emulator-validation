# Milestone 6 validation receipt

Branch: `milestone-6-reporting-instructions`, from merged PR #6 (`b7ffa68`).
PR: https://github.com/YongseokJo/grmhd-emulator-validation/pull/7

Delivered: owner reports, deterministic committed-result replay, schema/hash
checks and public CI, canonical participant instructions, importing Claude
instructions/skill, and separate prediction roles. Spec additions and the
owner-approved summary correction were committed separately before their code.
No frozen splits, protocol settings, source datasets or sibling repositories
were changed. Full bundles and comparison figures remain on shared storage;
Git contains two JSON files per exported run plus text reports/attempt ledgers.

## Owner-approved summary correction

The owner rejected compression and approved compact headline-only summaries,
with a hard 1,000,000-byte bound, fixed scalar schemas and metric SHA256 links.
Maps, profiles and per-start arrays remain in metrics/. Existing completed model
runs are immutable. Their original schema remains readable. Format-2 conformance
certificates are required for new evaluations; old certificates cannot silently
authorize the corrected format.

This is a result-format correction under unchanged protocol SHA256
`74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2`.
Only summary result-content bytes change. Reference run/golden metadata records
the new hashes; metrics, CSVs and figures are preserved. Original metadata and
large summaries are backed up under shared reports/<split>/summary-format-v2/,
not committed. Migration is owner-only, guarded and resumable.

| Split / tier | Old fingerprint | New fingerprint | Summary bytes |
|---|---|---|---:|
| rc300_reduced_v1 / quick | `6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6` | `2e288dd0a15e4480ce8727b76bb07f6c033bcb5c7e14bc48f30120808fe05ca6` | 5,328 |
| rc300_reduced_v1 / full | `aaf9b231403803169969957437cbae66c4f3e22f2fc54df4e986a750ccbf212b` | `d8112a7c97d350144a07cb41ad76fa8898e5caf4a4e6c8ae42410cfde4309d09` | 38,906 |
| 5000run_v1 / quick | `cd9598fd3f8380f9ebc862db46eb42d070fcfc7f43e532e7bd0f9fc7c51cf648` | `e4168be2541f058d88ded28f72bab8d89228ca56c87a7252efc2365b5563c1e4` | 5,349 |
| 5000run_v1 / full | `0b14cbc81327d0ae065b61a68817368833ab440b096a00301e4f6a3fe8baea36` | `9ad8abed8d9d7da593654c73d173a4f873490aa258224f1bce0df0a24e3b8929` | 39,099 |

[Machine-readable proof](artifacts/milestone6/summary_correction.json) records
every metric SHA256 before and after correction, plus independent conformance
comparisons against those original hashes. Equal hashes prove unchanged bytes;
no rounded-number comparison is substituted. The driver is
`scripts/validate_summary_format.py`; it can resume selected operations/tiers
across interactive allocations.

## Actual fresh-client instruction tests

The initial clients each received only a clean repository snapshot and the
prompt `validate the persistence adapter on rc300 validation`. Neither received
this conversation, custom model instructions or coaching. The instruction
snapshot was `690e010`; the canonical instructions have not changed since.
These tests completed before the owner approved format 2, so their published
runs legitimately retain the original quick fingerprint. They are preserved.

| Client | Session | Slurm job | Result |
|---|---|---|---|
| Claude Code 2.1.326, claude-sonnet-5-5 | 6431ed3e-d6c2-4568-ae4c-d952112d3afb | 3619839 | quick conformance passed, a1 published, verify passed |
| Codex CLI 0.169.0, model not exposed by ephemeral trace | 01a1eb0a-2653-7672-8082-1fb2a0bc938e | 3619886 | quick conformance passed, a2 published, verify passed |

Both runs use persistence v1, null checkpoint, frozen rc300 validation, the
locked native environment and data verification. Their original fingerprint is
`6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6`.
Their run IDs are
`persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1` and `...__a2`.
Claude's allocation completed in 11m09s (step 11m02s), MaxRSS 2,305,600 K;
Codex's in 10m55s (step 10m51s), MaxRSS 1,943,744 K, from sacct.

Claude automatically used the skill, inspected the ready request/scheduler,
and chained doctor, quick conformance, check-adapter, evaluation and verification.
It returned a waiting message during its background command, then continued
without human help; one delayed scheduler check was used rather than a poll
loop. Codex read the full spec and CLI helpers before running the recipe,
added run-directory extraction/verification, and gave one progress update after
conformance. It requested no guidance. These hesitations and selected actual
commands are recorded in [Claude's trace](artifacts/milestone6/claude_usability.json)
and [Codex's trace](artifacts/milestone6/codex_usability.json). Raw private traces
are SHA256-pinned there; credentials and full session dumps are not committed.

Harness deviations: an earlier Claude launch used the maintenance cwd instead
of the clean snapshot. Only that harness process was stopped, before any run
was published; no Slurm allocation was cancelled. It is not counted as a pass.
An extra format-2 Claude attempt encountered the occupied interactive slot and
scheduled a delayed retry. The duplicate harness and its sleep were stopped
before retrying; no allocation was cancelled. It is not counted as a completed
usability pass. The two original unassisted passing sessions remain the required
instruction tests; neither instructions nor their generated copy changed.
No owner job was cancelled, and no ghx4 fallback was used.

During migration, the closed summary schema initially required confidence
interval fields for coverage-only diagnostics. Those diagnostics have no scalar
score or CI, so their schema now requires an empty CI object. The backed-up
migration resumed successfully; a dedicated regression covers that case. No
metric data was changed to satisfy the schema.

## Verification

[Native checks](artifacts/milestone6/native_validation.json): 44 tests passed
in 550.610 s on the approved DeltaAI environment, in job 3620119 step 1.
The step completed in 9m56s with sacct MaxRSS 735,360 K. Doctor passed with the
native lock, no mismatches and no unfinished candidates. Its shared-tree scan
covered 6,565 entries in 6.943 s, with nine ACL subprocesses. Both live reports
refreshed successfully and both committed text replays checked exactly.

[Public CI run 44174005913](https://github.com/YongseokJo/grmhd-emulator-validation/actions/runs/44174005913)
passed on implementation/export commit 6986c33: 150 tests, eight expected
private-source/data skips on each matrix leg; NumPy 1.24.4 took 834.437 s and
NumPy 2.x took 707.929 s. Both exact committed leaderboard checks and generated
instruction checks passed without Delta data/storage/environment access.
The native targeted checks exercise the corrected full summary schema and
full publication/repeat-content path, not only quick-tier fixtures.

All four independently recomputed conformance fingerprints match their new
goldens exactly: no differing files, maximum relative difference zero. Each
recomputed metrics/*.json hash also matches its corresponding pre-correction
hash. All four migrations additionally preserve every table file.

| Split | Tier | Conformance seconds | Exact / original metric bytes |
|---|---|---:|---|
| rc300_reduced_v1 | quick | 294.632 | yes / identical |
| rc300_reduced_v1 | full | 1494.041 | yes / identical |
| 5000run_v1 | quick | 657.327 | yes / identical |
| 5000run_v1 | full | 3441.592 | yes / identical |

Job 3620119 completed successfully in 1h39m13s, within its two-hour interactive
allocation. Step 0 completed in 1h39m08s, sacct MaxRSS **6,302,144 K (6.010 GiB)**.
That is a conservative common upper bound over all four sequential checks,
not an isolated per-dataset measurement. The targeted tests/report refresh ran
as step 1 in the same allocation. No owner job was cancelled.

## Real format-2 export

Owner validation job **3620596** published and verified a third rc300 quick run:
`persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3`, fingerprint
`2e288dd0a15e4480ce8727b76bb07f6c033bcb5c7e14bc48f30120808fe05ca6`.
Data is verified, the run is sealed, and its 5,328-byte summary binds every
metric hash. The original run.json/summary.json pair is exported under results/;
no metric or figure files are committed. Report regeneration includes all three
runs without selecting a best attempt. Live report and exact replay both pass.
This is an owner export check, not a third fresh-client usability pass.

The allocation completed in 5m51s; step 0 took 5m46s with sacct MaxRSS 1,087 M.
The [native receipt](artifacts/milestone6/native_validation.json) includes all
adapter/evaluation/verification/report verdicts. The final PR CI checks these
committed format-2 hash bindings as well as the preserved legacy records.

## Review limits

Training provenance and host attestation remain operator declarations, not
proof of isolation. Undefined scores remain null. Completed legacy runs retain
original metadata and result bytes; they are not silently rewritten to format 2.
Full metrics/figures are available in the shared bundle, not Git. Unsealed,
failed, preview and NOT_EVALUATED attempts never enter leaderboard rows, while
the attempt ledger preserves test-attempt counts. Milestone 7 has not started.

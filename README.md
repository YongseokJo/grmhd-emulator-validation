# grmhd-emulator-validation

A shared, agent-friendly pipeline for validating GRMHD forecasting emulators (TRON, ViT, UNO variants, …) end
to end. Every model is scored on the same frozen data splits, with the same inference protocol, the same
metric and figure code, and the same output layout. Results go to one group folder, where anyone can
compare them.

**Status: milestones 1–4 merged; error-map presentation follow-up in progress.** The adapter API,
persistence, recorded runner, owner goldens, conformance and verification are
implemented. Milestone 4 adds final all-start aggregates/CIs, audited transport,
resolved-shell headlines and shared truth caches. The finalized protocol enables
full publication with a verified truth cache and passing full conformance;
historical previews stay in staging. Both datasets pass final quick/full
conformance; the [milestone 4 receipt and figure contact sheet](docs/milestone_4_validation.md)
record fingerprints, memory measurements and presentation checks.
Learned-model examples and prediction mode are milestone 5.
See [milestone 4 usage](docs/milestone_4.md).
See [milestone 3 usage and validation](docs/milestone_3.md), [milestone 2 usage](docs/milestone_2.md),
[metric definitions](docs/metrics.md) and [port provenance](docs/provenance.md). Locked DeltaAI references and real-data
conformance for both datasets pass; see the [validation receipt](docs/milestone_2_real_validation.md).

Short compute work uses `ghx4-interactive`, account `bgun-dtai-gh`. The owner's
jobs share the one-interactive-job-per-user limit. If occupied, inspect
`squeue -u "$USER"`, wait for the owner's job to finish and resubmit; never
cancel it or fall back to `ghx4` for short work. Ask the owner only if the wait
exceeds about two hours.

| Path | Contents |
|---|---|
| `splits/rc300_reduced_v1.json`, `splits/5000run_v1.json` | Frozen train / validation / test splits. Do not edit; changes need a new version. |
| `splits/make_splits.py` | Generator for the split files (interleaved chunks: 3 validation + 3 test, 10-dump gaps). |
| `docs/specs/pipeline.md` | Spec for the validation pipeline (protocol, metrics, adapter API, outputs, shared storage, agent instructions). |
| `docs/specs/movies.md` | Spec for the data movies and timeline. |

Shared results folder (Delta and DeltaAI, group `delta_bgun`): `/work/nvme/bgun/share/validation`.

## Planned prediction mode (milestone 5)

The full [prediction-mode specification](docs/specs/pipeline.md#61-agent-driven-prediction-mode-milestone-5)
defines the forecasting agent, host and scorer roles; task/packet commands;
versioned prompts; fixed prediction layout; complete 12-start × 50-step coverage;
provenance/attestation; and Eden's three credited adapters. Implementation is
milestone 5. Missing required predictions are reported as “not evaluated”.

## Split rules (every model, every agent)

- Use only the dumps listed in a split file. Every input and target frame of a sample must lie in one chunk of one split.
- Fit normalization on train only. Select checkpoints on validation. Use test for final evaluation; every test evaluation is recorded.
- Data, model inputs and model outputs are physical (unnormalized) values.

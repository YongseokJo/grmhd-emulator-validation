# grmhd-emulator-validation

A shared, agent-friendly pipeline for validating GRMHD forecasting emulators (TRON, ViT, UNO variants, …) end
to end. Every model is scored on the same frozen data splits, with the same inference protocol, the same
metric and figure code, and the same output layout. Results go to one group folder, where anyone can
compare them.

**Status: milestone 2 ready for owner review.** Milestone 1 is
merged. The adapter API, persistence, quick analyses, recorded runner, owner
goldens, conformance and verification are implemented. Full-tier analyses and
learned-model examples remain in later milestones. See [milestone 2 usage](docs/milestone_2.md),
[metric definitions](docs/metrics.md) and [port provenance](docs/provenance.md). Locked DeltaAI references and real-data
conformance for both datasets pass; see the [validation receipt](docs/milestone_2_real_validation.md).

| Path | Contents |
|---|---|
| `splits/rc300_reduced_v1.json`, `splits/5000run_v1.json` | Frozen train / validation / test splits. Do not edit; changes need a new version. |
| `splits/make_splits.py` | Generator for the split files (interleaved chunks: 3 validation + 3 test, 10-dump gaps). |
| `docs/specs/pipeline.md` | Spec for the validation pipeline (protocol, metrics, adapter API, outputs, shared storage, agent instructions). |
| `docs/specs/movies.md` | Spec for the data movies and timeline. |

Shared results folder (Delta and DeltaAI, group `delta_bgun`): `/work/nvme/bgun/share/validation`.

## Split rules (every model, every agent)

- Use only the dumps listed in a split file. Every input and target frame of a sample must lie in one chunk of one split.
- Fit normalization on train only. Select checkpoints on validation. Use test for final evaluation; every test evaluation is recorded.
- Data, model inputs and model outputs are physical (unnormalized) values.

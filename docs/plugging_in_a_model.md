# Plugging in a forecasting model

Models receive physical float32 `[B,h,8,Nr,Ntheta,Nphi]` histories and return
physical float32 `[B,8,Nr,Ntheta,Nphi]`. The benchmark owns starts, rolling
buffers, truth access, metrics, figures and publication. Geometry has no data
paths. See [the full contract](specs/pipeline.md#6-plugging-in-a-model-adapter-contract).

TRON and ViT are installed adapter entry points `tron` and `vit`; their residual
forecast and model-owned log10/arcsinh/Lorentz decode are ported from Kevin's
pinned evaluators. State dictionaries load strictly with `weights_only=True`.
TRON uses eight channels. ViT restores its architecture from checkpoint `config`;
if an old checkpoint lacks it, provide the exact training configuration as the
JSON object `options.config`. No experiment is guessed and no data folder is
searched. ViT's own `use_amp` is preserved and recorded in model provenance.

Copy `adapters/tron_example/run_request.json` to your own workspace, set the
checkpoint path and use the immutable split. Historical Kevin checkpoints used
an older split which overlaps current validation/test. Record that declaration
rather than claiming this benchmark's training split:

```json
{
 "provenance": {
  "training_chunks": [], "selection_chunks": [],
  "training_dump_ranges": [[3000,10707]],
  "selection_dump_ranges": [[10708,11671]],
  "source_split": "kevin_fall26_80_10_10",
  "normalization_split": "fixed model constants",
  "training_code_commit": "<full actual Git commit>",
  "known_prior_data_exposure": "Historical split overlaps the current benchmark",
  "prior_data_exposure": ["kevin_fall26_80_10_10 blind test"]
 }
}
```

Keep the full training config around that block. `record-training` computes
counts itself, binds the split/checkpoint/config/log hashes, and labels the
record an **operator declaration**. Overlap is recorded, not rejected. Put the
returned file path into `request.training_record`.

Run compute commands in a Slurm allocation. For short work use
`ghx4-interactive`, account `bgun-dtai-gh`; the owner's jobs share the one-job
limit. Wait for their job rather than cancelling it or switching partitions.

```bash
gval doctor
gval conformance --split-id rc300_reduced_v1
gval record-training --request run_request.json --split-id rc300_reduced_v1 --config training.json --log train.log
gval check-adapter tron --checkpoint /absolute/best.pth --split-id rc300_reduced_v1 --device cuda
gval evaluate --request run_request.json --split-id rc300_reduced_v1 --split validation --tier quick
gval evaluate --request run_request.json --split-id rc300_reduced_v1 --split validation --tier full
# Test follows final validation/model selection, not an optimization loop.
```

`eden-claude`, `eden-gpt`, `eden-gemini` are checkpoint-free protocol-A examples.
Use checkpoint null and options `{}`. They preserve the start-11672 algorithms,
with known pilot exposure recorded. They require an axisymmetric metric and a
uniform periodic phi grid; Claude additionally requires uniform log-radius and
equator symmetry. Unsupported geometry fails. Historical pilot scores are not
benchmark results: these adapters must be rerun on every common start.

For another model, use `gval new-adapter <your directory>`, implement `step`, then
follow the same check/evaluate path. Changing model-owned preprocessing requires
a new model version. No split, protocol or metric changes belong in an adapter.

For agent-generated forecasts, see [host](prediction_roles/host.md),
[forecasting agent](prediction_roles/forecasting.md) and
[scorer](prediction_roles/scorer.md) instructions.

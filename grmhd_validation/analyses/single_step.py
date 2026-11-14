"""Port: Kevin Fortune/Sachin, main@ebcc218, Fall26/eval/evaluate_forecast_ds.py.

Final v1 changes (spec §3.3): frozen starts, float64 CPU metrics, chunk-block CIs,
structured artifacts; see docs/metrics.md and docs/provenance.md.
"""
import numpy as np

from .common import ci, clean, nanmean, nmae, relative_l1, skill


def block_indices(chunk_sizes, block):
    blocks, offset = [], 0
    for size in chunk_sizes:
        for start in range(0, size, block):
            blocks.append(np.arange(offset + start, offset + min(start + block, size)))
        offset += size
    return blocks


def block_skill_ci(model, persistence, chunk_sizes, protocol):
    blocks = block_indices(chunk_sizes, protocol["bootstrap_block"])
    random = np.random.default_rng(protocol["seed"])
    bootstraps = []
    for _ in range(protocol["bootstrap_resamples"]):
        draw = random.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[i] for i in draw])
        bootstraps.append(skill(model[indices].mean(axis=0), persistence[indices].mean(axis=0), protocol["constants"]))
    return ci(bootstraps, protocol["constants"]), len(blocks)


def run(ctx):
    models, persistence, relatives, sizes = [], [], [], []
    for selected in ctx.starts:
        sizes.append(len(selected.every))
        for start in selected.every:
            history = ctx.reader.history(start, ctx.adapter.history_length, ctx.device)
            initial = history[0, -1].cpu().numpy().copy()
            prediction = ctx.step(history)[0].cpu().numpy()
            truth = ctx.reader.numpy(start + 1)
            models.append(nmae(prediction, truth, ctx.constants))
            persistence.append(nmae(initial, truth, ctx.constants))
            relatives.append(relative_l1(prediction, truth, ctx.constants))
    models, persistence = np.ascontiguousarray(models), np.ascontiguousarray(persistence)
    model, baseline = models.mean(axis=0), persistence.mean(axis=0)
    relative = nanmean(relatives)
    scores = skill(model, baseline, ctx.constants)
    intervals, blocks = block_skill_ci(models, persistence, sizes, ctx.protocol)
    per_channel = {name: {"nmae_model": model[i], "nmae_persist": baseline[i], "skill": scores[i],
                          "skill_ci": intervals[:, i], "rel_l1": relative[i]} for i, name in enumerate(ctx.channels)}
    mean = {"nmae_model": nanmean(model), "nmae_persist": nanmean(baseline), "skill": nanmean(scores), "rel_l1": nanmean(relative)}
    groups = {"fluid": {"nmae_model": nanmean(model[:5]), "nmae_persist": nanmean(baseline[:5])},
              "magnetic": {"nmae_model": nanmean(model[5:]), "nmae_persist": nanmean(baseline[5:])}}
    result = clean({"n_samples": len(models), "n_blocks": blocks, "ci_method": "chunk_block_bootstrap",
                    "per_channel": per_channel, "mean": mean, **groups})
    ctx.writer.json("metrics/single_step.json", result)
    rows = []
    for name, data in [*result["per_channel"].items(), ("mean", result["mean"]), ("fluid", result["fluid"]), ("magnetic", result["magnetic"])]:
        low, high = data.get("skill_ci", [None, None])
        rows.append({"channel": name, "nmae_model": data.get("nmae_model"), "nmae_persist": data.get("nmae_persist"),
                     "skill": data.get("skill"), "skill_ci_low": low, "skill_ci_high": high, "rel_l1": data.get("rel_l1")})
    from ..artifacts import csv_bytes
    ctx.writer.write("tables/single_step.csv", csv_bytes(rows, list(rows[0])))
    return result

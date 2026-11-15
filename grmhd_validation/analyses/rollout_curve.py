"""Port: Kevin Fortune/Sachin, main@ebcc218, Fall26/eval/evaluate_forecast_ds.py.

Final spec §3.3 semantics: deterministic chunk starts, CPU float64, start-bootstrap
CIs, split-wide strided truth sigma, fixed artifact names. No original script copy.
"""
import numpy as np

from .common import ci, clean, loss_space, nmae, skill

# The fixed channel palette specified by Kevin's CHANNEL_COLORS in the source.
CHANNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def dynamic_xticks(maximum):
    if maximum <= 20:
        return list(range(1, maximum + 1))
    return list(range(1, maximum + 1, 5 if maximum <= 50 else 10))


def start_skill_ci(model, persistence, protocol):
    random = np.random.default_rng(protocol["seed"])
    count = len(model)
    bootstraps = []
    for _ in range(protocol["bootstrap_resamples"]):
        draw = random.integers(0, count, size=count)
        bootstraps.append(skill(model[draw].mean(axis=0), persistence[draw].mean(axis=0), protocol["constants"], threshold=False))
    return ci(bootstraps, protocol["constants"])


def truth_sigma(ctx):
    values = []
    stride = ctx.constants["attractor"]["sigma_dump_stride"]
    for chunk in ctx.split.chunks(ctx.partition):
        for index in range(chunk.first, chunk.last + 1, stride):
            transformed = loss_space(ctx.reader.numpy(index), ctx.constants)
            values.append(transformed.reshape(8, -1).std(axis=1, ddof=0))
    return np.ascontiguousarray(values).mean(axis=0)


def plot(ctx, mean, std, baseline, per_channel):
    settings = ctx.constants["figures"]
    import matplotlib
    matplotlib.use(settings["backend"])
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({"font.family": settings["font"], "font.size": settings["font_size"]}):
        figure = Figure(figsize=settings["rollout_curve_figsize"], dpi=settings["dpi"])
        FigureCanvasAgg(figure)
        left, right = figure.subplots(1, 2)
        steps = np.arange(1, ctx.protocol["max_steps"] + 1)
        left.plot(steps, mean, "b-o", markersize=3, linewidth=1.5, label="model")
        left.fill_between(steps, mean - std, mean + std, alpha=.2, color="blue")
        left.plot(steps, baseline, "k--", linewidth=1.2, label="Persistence")
        left.set(xlabel="Forecast steps", ylabel="Channel-averaged nMAE", title="(a) Forecast Error")
        left.legend()
        fallback = False
        for channel, name in enumerate(ctx.channels):
            absolute = per_channel[0, channel] < ctx.constants["growth_curve"]["step1_min"]
            fallback |= absolute
            curve = per_channel[:, channel] if absolute else per_channel[:, channel] / per_channel[0, channel]
            right.plot(steps, curve, linewidth=1.4, color=CHANNEL_COLORS[channel], label=name + (" (abs)" if absolute else ""))
        right.set(xlabel="Forecast steps", ylabel="Error growth (rel or abs)" if fallback else "Error growth (relative to step 1)",
                  title="(b) Per-Channel Error Growth")
        right.legend(fontsize=8, ncol=2)
        for axis in (left, right):
            axis.set_xlim(1, max(2, ctx.protocol["max_steps"]))
            axis.set_xticks(dynamic_xticks(ctx.protocol["max_steps"]))
            axis.grid(True, alpha=.3)
        figure.suptitle("Autoregressive Forecast Accuracy")
        figure.tight_layout()
        with ctx.writer.open("figures/fig02_rollout_curve.png") as stream:
            figure.savefig(stream, format="png", dpi=settings["dpi"], metadata={"Software": None})
        figure.clear()


def run(ctx):
    models, persistence = [], []
    for selected in ctx.starts:
        for start in selected.rollout:
            print(f"rollout_curve {selected.chunk}: start {start}", flush=True)
            history = ctx.reader.history(start, ctx.adapter.history_length, ctx.device)
            initial = history[0, -1].cpu().numpy().copy()
            model_curve, baseline_curve = [], []
            for step in range(1, ctx.protocol["max_steps"] + 1):
                output = ctx.step(history)
                truth = ctx.reader.numpy(start + step)
                model_curve.append(nmae(output[0].cpu().numpy(), truth, ctx.constants))
                baseline_curve.append(nmae(initial, truth, ctx.constants))
                import torch
                history = torch.cat((history[:, 1:], output.unsqueeze(1)), dim=1)
            models.append(model_curve); persistence.append(baseline_curve)
    models, persistence = np.ascontiguousarray(models), np.ascontiguousarray(persistence)
    averaged = np.ascontiguousarray(models.mean(axis=-1))
    baseline_starts = np.ascontiguousarray(persistence.mean(axis=-1))
    mean, std = averaged.mean(axis=0), averaged.std(axis=0, ddof=0)
    baseline = baseline_starts.mean(axis=0)
    scores = skill(mean, baseline, ctx.constants, threshold=False)
    per_channel = models.mean(axis=0)
    intervals = start_skill_ci(averaged, baseline_starts, ctx.protocol).T
    sigma = truth_sigma(ctx)
    saturation = {name: {"sigma": sigma[c], "nmae_at_max_steps": per_channel[-1, c],
                         "flag": "ok" if per_channel[-1, c] < sigma[c] else "exceeds"} for c, name in enumerate(ctx.channels)}
    result = clean({"n_starts": len(models), "steps": list(range(1, ctx.protocol["max_steps"] + 1)),
                    "mean_nmae": mean, "std_nmae": std, "mean_persist": baseline,
                    "skill": scores, "skill_ci": intervals, "ci_method": "start_bootstrap",
                    "per_channel": per_channel, "attractor": saturation})
    ctx.writer.json("metrics/rollout_curve.json", result)
    rows, reports = [], {}
    for step in ctx.protocol["report_steps"]:
        index = step - 1
        entry = {"step": step, "time_rg_c": step * ctx.split.dt_dump, "mean_nmae": mean[index],
                 "std_nmae": std[index], "mean_persist": baseline[index], "skill": scores[index],
                 "skill_ci_low": intervals[index, 0], "skill_ci_high": intervals[index, 1]}
        entry.update({"nmae_" + name: per_channel[index, c] for c, name in enumerate(ctx.channels)})
        # Convert NumPy scalar types before repr-based CSV serialization.
        rows.append(clean(entry))
        reports[str(step)] = clean({"time_rg_c": entry["time_rg_c"], "mean_nmae": mean[index], "std_nmae": std[index],
                  "mean_persist": baseline[index], "skill": scores[index], "skill_ci": intervals[index],
                  "per_channel": {name: per_channel[index, c] for c, name in enumerate(ctx.channels)}})
    from ..artifacts import csv_bytes
    ctx.writer.write("tables/rollout_curve.csv", csv_bytes(rows, list(rows[0])))
    plot(ctx, mean, std, baseline, per_channel)
    return clean({"n_starts": len(models), "at_steps": reports, "attractor": saturation})

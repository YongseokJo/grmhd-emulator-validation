import json
from pathlib import Path

from .hashing import sha256


def read_protocol(path):
    path = Path(path)
    data = json.loads(path.read_text())
    import jsonschema
    schema_path = Path(__file__).resolve().parents[1] / "schemas/protocol.schema.json"
    jsonschema.validate(data, json.loads(schema_path.read_text()))
    for name in ("H_max", "max_steps", "n_starts_per_chunk", "bootstrap_block", "bootstrap_resamples"):
        if type(data[name]) is not int or data[name] <= 0:
            raise ValueError("Protocol requires a positive integer: " + name)
    if type(data["seed"]) is not int:
        raise ValueError("Protocol seed must be an integer")
    reports = data["report_steps"]
    if not reports or reports != sorted(set(reports)) or any(type(s) is not int or not 1 <= s <= data["max_steps"] for s in reports):
        raise ValueError("Invalid report_steps")
    if not 0 < data["gpu_tolerance"] < 1:
        raise ValueError("Invalid GPU tolerance")
    if set(data["tiers"]) != {"quick", "full"} or not set(data["tiers"]["quick"]) <= set(data["tiers"]["full"]):
        raise ValueError("Invalid tier analyses")
    for analyses in data["tiers"].values():
        if len(analyses) != len(set(analyses)) or "error_spectra" in analyses:
            raise ValueError("Duplicate or deprecated analysis")
    percentiles = data["constants"]["ci"]["percentiles"]
    if percentiles != sorted(set(percentiles)):
        raise ValueError("CI percentiles must increase")
    correlation = data["constants"]["correlation_time"]
    if correlation["r1_cap"] > correlation["r1_clip"] or correlation["r1_saturated"] > correlation["r1_clip"]:
        raise ValueError("Correlation cap/saturation cannot exceed clipping bound")
    full=data['constants']['full']
    if full['spatial_bins_min']>full['spatial_bins_max']:
        raise ValueError('Spatial bin minimum exceeds maximum')
    for name in ('histogram_linear_percentiles','histogram_log_percentiles','spectrum_fit_fraction'):
        if not full[name][0]<full[name][1]:raise ValueError(name+' must increase')
    return data, sha256(path)

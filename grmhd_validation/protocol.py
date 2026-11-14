import json
from pathlib import Path

from .hashing import sha256


def read_protocol(path):
    path = Path(path)
    data = json.loads(path.read_text())
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
    return data, sha256(path)

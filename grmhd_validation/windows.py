from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Starts:
    chunk: str
    every: tuple
    rollout: tuple


def select_starts(split, partition, protocol):
    result = []
    for chunk in split.chunks(partition):
        lo = chunk.first + protocol["H_max"] - 1
        hi = chunk.last - protocol["max_steps"]
        if hi < lo:
            raise ValueError("Chunk is shorter than max_steps + H_max: " + chunk.name)
        rollout = tuple(sorted(set(int(round(value)) for value in np.linspace(lo, hi, protocol["n_starts_per_chunk"]))))
        result.append(Starts(chunk.name, tuple(range(lo, hi + 1)), rollout))
    if not result:
        raise ValueError("No evaluation chunks")
    return tuple(result)

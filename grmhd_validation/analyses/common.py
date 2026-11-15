"""Loss-space metrics ported from Kevin Fortune and Sachin's evaluation suite.

Source: Fall26/eval/evaluate_forecast_ds.py and Fall26/common/normalizer.py,
main @ ebcc218 (TRON originals: Sachin). Final float64 semantics: spec §3.3.
"""
import warnings

import numpy as np


def physical(value):
    return np.ascontiguousarray(value, dtype=np.float64)


def loss_space(raw, constants):
    result = physical(raw).copy()
    loss = constants["loss_space"]
    result[0] = np.log10(np.maximum(result[0], loss["floor_rho"]))
    result[1] = np.log10(np.maximum(result[1], loss["floor_ug"]))
    result[2:] = np.arcsinh(result[2:])
    return result


def channel_mean(value):
    return np.mean(np.ascontiguousarray(value).reshape(8, -1), axis=1, dtype=np.float64)


def nmae(prediction, truth, constants):
    return channel_mean(np.abs(loss_space(prediction, constants) - loss_space(truth, constants)))


def relative_l1(prediction, truth, constants):
    denominator = channel_mean(np.abs(physical(truth)))
    result = np.full(8, np.nan, dtype=np.float64)
    np.divide(channel_mean(np.abs(physical(prediction) - physical(truth))), denominator,
              out=result, where=denominator >= constants["rel_l1"]["denominator_min"])
    return result


def skill(model, persistence, constants, threshold=True):
    result = 1.0 - model / (persistence + constants["skill"]["epsilon"])
    if threshold:
        result = np.where(persistence < constants["skill"]["persistence_min"], np.nan, result)
    return result


def nanmean(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.ascontiguousarray(values), axis=axis, dtype=np.float64)


def ci(values, constants):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(np.ascontiguousarray(values), constants["ci"]["percentiles"], axis=0)


def clean(value):
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value

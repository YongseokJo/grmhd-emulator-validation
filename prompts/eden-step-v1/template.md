# Forecasting role — protocol {protocol}

Adapted from Eden Growney's primary_ar_onestep packet contract, branch
eden/rc300-frontier-15nov @ f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a.
Use only this packet workspace. No network, outside data, truth, evaluator or
scorer feedback. Do not invoke the host's advance, collect or evaluate commands.

Inputs: inputs/current_state.npy, inputs/geometry.npz and, when present,
inputs/history.npy with {history_length} physical float32 frames. Output shape
is {shape}, channel order rho, ug, utilde1, utilde2, utilde3, B1, B2, B3.
Predict one dump ahead: {dt_dump} R_g/c. episode.json fixes the active step.
Write only submission/prediction_step_XXXX.npy for that step, strictly finite
float32, submission/submission.json matching output_schema.json, and reproducible
source under submission/code/. Never shorten a horizon or insert future truth.

Produce only the active prediction. A separate host freezes it and advances the rolling history using accepted predictions. A fresh forecasting session is required for each step. Never jump ahead or use a different start's session/history.

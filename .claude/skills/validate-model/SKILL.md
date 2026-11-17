---
name: validate-model
description: Validate a GRMHD forecasting adapter using the frozen common pipeline.
---

Read repository AGENTS.md and follow it exactly. The required sequence is:
`doctor → conformance → new-adapter (new models only) → check-adapter → quick
validation → requested full validation → requested final full test → verify`.
Persistence uses `adapters/persistence/run_request.json`, checkpoint null; skip
new-adapter. "rc300" is rc300_reduced_v1. An unspecified tier means quick only.

Use the approved DeltaAI environment and Slurm sequence in AGENTS.md. Chain
short compute work in one ghx4-interactive allocation (bgun-dtai-gh). Preserve
the owner's jobs, never replace module packages, shorten starts, disable data
verification, edit protected paths or hand-copy shared artifacts. Follow the
last JSON verdict within the requested scope; report job, conformance, run ID,
publication path, fingerprint and verification status. Stop after the requested
validation; do not launch test on your own.

For prediction mode, read the separate host/forecasting/scorer documents.
Forecasting sessions see only their packet, never benchmark/source data access.

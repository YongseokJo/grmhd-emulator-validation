# Scorer instructions

Validate a collected bundle using the original frozen request, split and
protocol. Do not execute submitted code or import the request adapter when
scoring prediction files. The scorer reads future truth only after generation
and checks hashes, physical shape/dtype/finiteness and the full autoregressive
chain, including the original checksum-verified history.

```bash
gval evaluate --request run_request.json --predictions /your/prediction-bundle --split-id rc300_reduced_v1 --split validation --tier quick
```

Every one of the 12 rollout starts requires all 50 steps. Single-step coverage
requires every eligible start, independently of rollout step 1. Never score an
available subset, fill missing files or select a preferred retry. An incomplete
request remains sealed in staging with NOT_EVALUATED and coverage.json; available
analyses are labelled partial inspection. It never gets COMPLETE or publication
in runs/. A later completed collection requires another recorded attempt.

File-only cost/stability are not evaluated: an executable adapter is required.
Thus a file-only full-tier request is an inspection attempt, not a completed
full-tier leaderboard submission. A complete quick bundle can be published.
Use protocol A adapter evaluation when executable analyses are needed.

Report exact missing/invalid identities, evaluation attempt, bundle/provenance
hashes and host-attested versus unattested status. Host attestation is an
operator's claim bound to evidence; the pipeline does not independently prove
isolation or infer model versions/seeds. Preserve known prior exposure flags.

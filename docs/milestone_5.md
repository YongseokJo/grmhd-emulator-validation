# Milestone 5 review

This milestone adds TRON/ViT and Eden's three physical-unit adapters,
`record-training`, and the §6.1 task/packet/prediction-file workflow. The common
splits, starts, inference buffer, metric modules, protocol SHA and result
serialization are unchanged. Fig 14's signed colourbar has one major tick per
two decades and zero.

Use [model setup](plugging_in_a_model.md) for normal adapter evaluation. The
[host](prediction_roles/host.md), [forecasting agent](prediction_roles/forecasting.md)
and [scorer](prediction_roles/scorer.md) instructions describe separate roles.
No provider API integration or agent spawning is implicit: the operator launches
the isolated forecasting session and supplies its actual provenance. The host
materializes inputs, freezes accepted outputs and owns advancement. The scorer
never executes submitted source when importing prediction files.

The initial immutable prompts are `eden-code-v1` (A) and `eden-step-v1` (B).
Protocol A tasks read history length from the host's adapter declaration; an
adapter skeleton may declare that contract before generated code is installed.
After generation, the host freezes the method and executes it for all starts.
A new method requires a new version. B requests declare options.history_length;
a fresh forecasting session is required per step by the initial B prompt.

Coverage comes from the frozen split/protocol, never a command-line subset.
Every v1 split has 12 rollout starts × 50 steps. Every single-step start needs
its own prediction; rollout step 1 does not replace that set. A missing/invalid
entry makes its whole analysis group unavailable. Complete groups may produce
labelled partial inspection figures, but any unavailable requested analysis
keeps the attempt sealed in staging as NOT_EVALUATED. There is no COMPLETE or
runs/ publication. File-only cost and stability are unavailable because no
executable adapter was measured; full file-only requests are inspection runs.
Complete quick prediction imports use the normal publish/seal path.

Provenance and host attestation are run metadata, outside result fingerprints.
Unknown generation details stay null with reasons. Missing attestation evidence
is labelled unattested; the pipeline does not independently prove isolation.
Training records are operator declarations. Their split/chunk/range overlap is
computed by the command, and historical exposure never prevents evaluation.
A record for another split or checkpoint is rejected. The owner-approved §9
format and the packet formats are separate spec commits before implementation.

Every port names its source/branch/commit/author in a header and
[provenance table](provenance.md). No original evaluator is copied unchanged,
no historical prediction cubes or pilot scores are imported, and no dependency
was added. Private reference sources are loaded read-only for source comparison;
those comparisons are skipped only in public CI lacking the private checkout.
The explicit real TRON regression requires the pinned checkpoint and fails if
it is absent or its SHA differs. Its output is regression evidence, never a
benchmark submission: that checkpoint overlaps current validation.

Numerical and operational evidence is in the [validation receipt](milestone_5_validation.md).
Milestone 6 (report/leaderboard/CI/participant agent instructions) remains for
owner approval after this review. The milestone-7 learned-model benchmark dry
run is not claimed by adapter source/kernel regression.

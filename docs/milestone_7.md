# Owner rc300 validation dry run

This milestone validates the participant path with persistence, TRON, ViT,
eden-claude, eden-gpt and eden-gemini. Each runs quick and full on frozen
rc300_reduced_v1 validation, with all common starts and all 50 rollout steps.
No test-split evaluation is authorized. Existing persistence goldens and truth
caches are reused; no reference values or completed model bundles are edited.

The preceding protected-path check records real Claude Edit denials under
splits/ and grmhd_validation/. The runner independently inspects Git status on
splits/, protocols/, grmhd_validation/, schemas/ and prompts/, including staged
and untracked files. Dirty attempts fail in staging before inference; changes
during evaluation or before publication also prevent publication. run.json
retains repo_modified/status entries. Reports expose rejected dirty attempts
and keep historical unknown state separate from a recorded clean state.

The owner runs scripts/validate_owner_dry_run.py inside ghx4-interactive,
bgun-dtai-gh, with the approved module/venv and one numerical thread. Use one
GPU for TRON/ViT; persistence and Eden execute on CPU. The driver reserves no
additional allocation. It appends a receipt and defers the next operation when
the remaining allocation is too short, without reducing benchmark coverage.
It does not retry failed methods automatically or select a best score.

```bash
# Inside the approved compute allocation, after native tests pass:
python scripts/validate_owner_dry_run.py --prepare --preflight --model persistence
# In subsequent allocations; verified completed operations are reused:
python scripts/validate_owner_dry_run.py --model tron --model vit
python scripts/validate_owner_dry_run.py --model eden-claude --model eden-gpt --model eden-gemini
# Only after all 12 evaluations and verifications pass:
python scripts/validate_owner_dry_run.py --publish
```

Preparation keeps Kevin's complete original configs and hashes their source
bytes. Training log headers identify TRON commit 2667a5f and ViT commit ae8d040;
full hashes are stored in their declarations. Both declare source_split
kevin_fall26_80_10_10, training 3000–10707 and selection 10708–11671. The fixed
log10/arcsinh transform has no fitted normalization split: the declaration says
fixed_transform_no_fit, rather than inventing a train-only fitted normalizer.
The existing provenance logic retains that flag and computes exact overlap with
our validation/test/excluded chunks. Checkpoints, declaration configs and logs
are hashed; matching records are reused and conflicting records are rejected.

Eden's credited historical source remains protocol A and carries its known
start-11672 pilot exposure; these reruns do not establish blind development.
All models use the common inference, metrics, summaries and artifact writers.
Full bundles/figures stay shared; only original run.json/summary.json pairs,
attempt ledgers and generated text reports enter Git. Final job IDs, memory,
results and limitations belong in milestone_7_validation.md. The owner reviews
and merges; no test evaluation follows this dry run.

Preflight found an extra Eden-Claude wrapper restriction: exact physical theta
reflection symmetry. The original kernel never reads theta; it reverses theta
indices even on the real rc300 grid. The wrapper now preserves that operation
without a new physical-angle condition. Its numerical kernel, model parameters,
input fields and all metrics are unchanged. An asymmetric-theta regression and
the original-source regression verify this source-preserving interface fix.

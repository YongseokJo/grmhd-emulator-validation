# rc300_reduced_v1 leaderboard

Compare only within the same split ID, protocol ID and SHA256, partition, and tier.
Every row is one sealed artifact-verified run. Training evidence is an operator declaration; host attestation does not independently prove blindness.

## validation / quick / forecast_v1

Protocol SHA256: `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2`. Headline scores use the final report step.

| Model | Run | Skill | nMAE | Test attempts | Data verified | Training label | Flags |
|---|---|---:|---:|---:|---|---|---|
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0 | 0.11021 | 0 | yes | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | 0 | 0.11021 | 0 | yes | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | 0 | 0.11021 | 0 | yes | unknown | — |

Physical relative L1 (per channel):

| Run | rho | ug | utilde1 | utilde2 | utilde3 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |

Final-step physics errors (undefined for quick):

| Run | Mass | Energy | Angular momentum |
|---|---:|---:|---:|
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | — | — | — |

Comparison figures (shared reports or local replay; not stored in Git):
`74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/quick/rollout_comparison.png` and `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/quick/skill_comparison.png`.

## Excluded attempts

None.


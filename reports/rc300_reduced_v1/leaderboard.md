# rc300_reduced_v1 leaderboard

Compare only within the same split ID, protocol ID and SHA256, partition, and tier.
Every row is one sealed artifact-verified run. Training evidence is an operator declaration; host attestation does not independently prove blindness.

## validation / full / forecast_v1

Protocol SHA256: `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2`. Headline scores use the final report step.

| Model | Run | Skill | nMAE | Test attempts | Data verified | Repo modified | Training label | Flags |
|---|---|---:|---:|---:|---|---|---|---|
| eden-claude | eden-claude-v1__nockpt__rc300_reduced_v1__validation__full__a1 | -4.63526 | 0.621061 | 0 | yes | no | unknown | — |
| eden-gemini | eden-gemini-v1__nockpt__rc300_reduced_v1__validation__full__a1 | -0.0455062 | 0.115225 | 0 | yes | no | unknown | — |
| eden-gpt | eden-gpt-v1__nockpt__rc300_reduced_v1__validation__full__a1 | -2.90447 | 0.430311 | 0 | yes | no | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0 | 0.11021 | 0 | yes | no | unknown | — |
| tron | tron-v1__a81d05b4__rc300_reduced_v1__validation__full__a1 | 0.000990099 | 0.110101 | 0 | yes | no | no, overlaps validation/test (2379 dumps) | — |
| vit | vit-v1__6ebddd83__rc300_reduced_v1__validation__full__a1 | -0.539178 | 0.169633 | 0 | yes | no | no, overlaps validation/test (2379 dumps) | — |

Physical relative L1 (per channel):

| Run | rho | ug | utilde1 | utilde2 | utilde3 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eden-claude-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.329933 | 0.738206 | 0.117856 | 0.640568 | 0.134706 | 0.0952638 | 0.332558 | 0.134699 |
| eden-gemini-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.347536 | 0.895887 | 0.108438 | 0.742155 | 0.125896 | 0.0877752 | 0.432243 | 0.138526 |
| eden-gpt-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.358862 | 0.729087 | 0.151503 | 0.597859 | 0.147462 | 0.183375 | 0.446635 | 0.235819 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| tron-v1__a81d05b4__rc300_reduced_v1__validation__full__a1 | 0.0708552 | 0.112206 | 0.0398268 | 0.230094 | 0.035108 | 0.00913788 | 0.0650838 | 0.0104675 |
| vit-v1__6ebddd83__rc300_reduced_v1__validation__full__a1 | 0.129978 | 0.324107 | 0.0486763 | 0.282161 | 0.0468664 | 0.0248198 | 0.148581 | 0.0365981 |

Final-step physics errors (undefined for quick):

| Run | Mass | Energy | Angular momentum |
|---|---:|---:|---:|
| eden-claude-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 2.94521 | 11.2187 | 9.11325 |
| eden-gemini-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.13181 | 0.275691 | 0.220226 |
| eden-gpt-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.968043 | 8.42886 | 5.80042 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__full__a1 | 0.131788 | 0.275596 | 0.220164 |
| tron-v1__a81d05b4__rc300_reduced_v1__validation__full__a1 | 0.155835 | 1.81781 | 5.47458 |
| vit-v1__6ebddd83__rc300_reduced_v1__validation__full__a1 | 0.6725 | 4.15768 | 12.3409 |

Comparison figures (shared reports or local replay; not stored in Git):
`74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/full/rollout_comparison.png` and `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/full/skill_comparison.png`.

## validation / quick / forecast_v1

Protocol SHA256: `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2`. Headline scores use the final report step.

| Model | Run | Skill | nMAE | Test attempts | Data verified | Repo modified | Training label | Flags |
|---|---|---:|---:|---:|---|---|---|---|
| eden-claude | eden-claude-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | -4.63526 | 0.621061 | 0 | yes | no | unknown | — |
| eden-gemini | eden-gemini-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | -0.0455062 | 0.115225 | 0 | yes | no | unknown | — |
| eden-gpt | eden-gpt-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | -2.90447 | 0.430311 | 0 | yes | no | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0 | 0.11021 | 0 | yes | unknown | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | 0 | 0.11021 | 0 | yes | unknown | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | 0 | 0.11021 | 0 | yes | unknown | unknown | — |
| persistence | persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a4 | 0 | 0.11021 | 0 | yes | no | unknown | — |
| tron | tron-v1__a81d05b4__rc300_reduced_v1__validation__quick__a1 | 0.000990099 | 0.110101 | 0 | yes | no | no, overlaps validation/test (2379 dumps) | — |
| vit | vit-v1__6ebddd83__rc300_reduced_v1__validation__quick__a1 | -0.539178 | 0.169633 | 0 | yes | no | no, overlaps validation/test (2379 dumps) | — |

Physical relative L1 (per channel):

| Run | rho | ug | utilde1 | utilde2 | utilde3 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eden-claude-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0.329933 | 0.738206 | 0.117856 | 0.640568 | 0.134706 | 0.0952638 | 0.332558 | 0.134699 |
| eden-gemini-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0.347536 | 0.895887 | 0.108438 | 0.742155 | 0.125896 | 0.0877752 | 0.432243 | 0.138526 |
| eden-gpt-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0.358862 | 0.729087 | 0.151503 | 0.597859 | 0.147462 | 0.183375 | 0.446635 | 0.235819 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a4 | 0.331025 | 0.877242 | 0.093556 | 0.624707 | 0.100444 | 0.0843815 | 0.370788 | 0.121143 |
| tron-v1__a81d05b4__rc300_reduced_v1__validation__quick__a1 | 0.0708552 | 0.112206 | 0.0398268 | 0.230094 | 0.035108 | 0.00913788 | 0.0650838 | 0.0104675 |
| vit-v1__6ebddd83__rc300_reduced_v1__validation__quick__a1 | 0.129978 | 0.324107 | 0.0486763 | 0.282161 | 0.0468664 | 0.0248198 | 0.148581 | 0.0365981 |

Final-step physics errors (undefined for quick):

| Run | Mass | Energy | Angular momentum |
|---|---:|---:|---:|
| eden-claude-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| eden-gemini-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| eden-gpt-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a2 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a3 | — | — | — |
| persistence-v1__nockpt__rc300_reduced_v1__validation__quick__a4 | — | — | — |
| tron-v1__a81d05b4__rc300_reduced_v1__validation__quick__a1 | — | — | — |
| vit-v1__6ebddd83__rc300_reduced_v1__validation__quick__a1 | — | — | — |

Comparison figures (shared reports or local replay; not stored in Git):
`74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/quick/rollout_comparison.png` and `74d6f91071413ccc7154770312ea40b2d2c007225d094b0a2bbfba383e3a93e2/validation/quick/skill_comparison.png`.

## Excluded attempts

None.


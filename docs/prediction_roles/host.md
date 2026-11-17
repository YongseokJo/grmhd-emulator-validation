# Host instructions

The host owns source data and audit state. Keep `<task_dir>` private and expose
only `<packet_dir>` in a **separate forecasting-agent session**. A directory
name alone does not enforce isolation. Record actual container/mount/tool/network
controls and exceptions. Never send credentials, future truth, evaluator code,
other starts' context or scorer feedback into generation.

```bash
gval task --request run_request.json --split-id rc300_reduced_v1 --split validation --protocol B --prompt-id eden-step-v1 --output /your/private/task
gval packet create --task /your/private/task --kind rollout --start S --output /isolated/packet
# Start the separate forecasting session in /isolated/packet.
# Fill episodes/rollout/CHUNK/start_S/generation.json in the private host task.
gval packet advance --task /your/private/task --packet /isolated/packet
# Repeat with a fresh session for each active step under eden-step-v1.
gval packet collect --task /your/private/task --output /your/prediction-bundle
```

All packet I/O runs on Slurm compute nodes. B requests declare
`options.history_length` in 1..10 and normally use checkpoint null; the adapter
reference is not imported by the scorer. A tasks use the executable adapter's
history length and `eden-code-v1`: freeze generated reusable code as an adapter,
then execute it host-side for all starts without consulting the agent again.
Normal adapter evaluation is the primary A path; the host may also submit its
outputs through the packet chain with the original generation provenance.
Never describe a code-only host rollout as protocol B.

`task.json` fixes all starts and targets. Do every rollout start (12 on v1), all
50 steps, and separately every single-step start. Only active-step submissions
are accepted. The host never repairs dtype, values, shapes or missing steps.
Each accepted prediction, metadata and per-step code is frozen. A's method must
stay fixed across starts; B's tool code may vary and its hashes are recorded.

`generation.json` records actual provider/model/version/client/session/request/
response IDs, parameters/seed, times and elapsed time, tools/network and log
hashes. Null means unknown and needs a reason. Do not invent values. New B starts
must never reuse another start's context. The initial B template requires a
fresh session per step. Generation wall time is provenance, not inference cost.

To resume after interruption, rerun `packet create` with the **same** task,
kind/start and registered packet path; it verifies the frozen chain and restores
the next rolling input. It cannot replace an accepted output. An interrupted
unrecorded file is reused only with an identical hash. Rejections remain in the
private audit log and collected provenance.

Before collection, fill private `provenance.json` and `host_attestation.json`.
Attest against actual workspace controls, complete collection, source/prompt
freezing, allowed history, no feedback and session rules; include operator,
host/job, commit, evidence-log hashes, time and exceptions. Collect binds the
attestation to the final manifest hash. Missing evidence stays **unattested**.
The forecasting agent cannot attest the host's controls on the host's behalf.
The scorer revalidates the initial history against checksum-verified truth.

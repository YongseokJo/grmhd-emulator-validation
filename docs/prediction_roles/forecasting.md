# Forecasting-agent instructions

Read only your packet's README.md, prompt.md, format.md, output_schema.json,
episode.json and inputs. The versioned prompt controls the session policy and
tool/network permissions. You have no host or scorer role.

Write the active physical prediction as strict finite float32 with the exact
shape/channel order, submission/submission.json matching the supplied schema,
and reproducible source under submission/code/. Predict exactly the active
step. Do not call advance, collect or evaluate. Never seek future truth, other
starts' histories or scoring feedback. The host maintains the rolling buffer.

Protocol A asks for reusable one-step code that the host freezes and executes
without further agent consultation. Protocol B asks for each active prediction;
its host executes no hidden replacement method. Your prompt identifies which
protocol applies. Return control to the host when the active submission is ready.

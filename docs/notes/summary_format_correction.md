# Owner decision: compact summary format 2

On 2026-11-17 the owner approved correcting summary.json under the existing
protocol hash. The former full summary copied maps, profiles and per-start
arrays from metrics/ (129 MB on rc300). New summaries contain bounded headlines
and metric SHA256 values, with a 1,000,000-byte guard. This is a serialization
correction; no metric/table computation changes.

The owner reference migration validates the original golden, backs up original
summary/run/golden metadata under the owner reports area, and replaces only the
summary plus fingerprint metadata. It never modifies completed model runs.
Metric/table files remain unchanged. Independent quick/full conformance reruns
normal inference for both datasets and compares new result fingerprints and
original metric hashes. The validation receipt records old/new fingerprints.

Legacy completed runs keep their original bytes and remain verifiable with the
legacy summary schema. New conformance certificates carry format version 2;
older certificates cannot authorize new-format evaluations. GPU/cross-platform
numeric comparisons validate hash bindings independently, then compare numbers
with the approved tolerance rather than treating differing metric hashes as
numeric errors.

Exports commit original summary.json and run.json, plus the attempt ledger.
There is no gzip or metric/PNG export to Git. Comparison figures stay in shared
reports or ignored local replay outputs. CI checks exact derived text and the
new run/summary hash bindings. Intrinsically spatial diagnostics report coverage
metadata instead of inventing a new global score from profiles or maps.

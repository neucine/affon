# Training observation capture

This benchmark captures one warmed, fixed-shape training step and joins its
forward/backward and optimizer work across Affon telemetry, Compute runtime
evidence, and, on CUDA, an external Nsight Systems timeline.

The worker uses a one-layer decoder with a tiny fixed workload. It is an
observability probe, not a throughput benchmark. The gradient executable owns
the primal forward and backward call, while the optimizer component is the
actual native update executable used by the step.

Build Affon first, then run from the repository root:

```sh
python3 test/benchmark/training/profile-training.py \
  --affon /path/to/affon \
  --device cuda \
  --output /tmp/affon-training-profile
```

Use `--device cpu` or `--device metal` for those backends. When Nsight Systems
is not on `PATH`, pass `--nsys /path/to/nsys`; use `--no-nsys` to collect only
Affon telemetry and runtime evidence. `--capture-records` bounds Compute's
per-Session evidence buffer, and `--telemetry-records` bounds the process-wide
telemetry console ring. Defaults are sized for the model-style CUDA capture,
but both limits should be checked against `missed` and
`dropped_event_count` in the output.

The output directory contains:

- `training-profile.json`: the selected result, compiled explanation,
  composite gradient/optimizer evidence, and application metrics.
- `telemetry-traces.json`: selected-run records and the collector's `missed`
  flag.
- `capture.json`: source joins, invocation and step identities, component
  kernel accounting, phase coverage, and completeness checks.
- With Nsight: the `.nsys-rep`, `profile.sqlite`, and
  `compute-timeline.json` artifacts.

The runner fails on dropped or missed telemetry, a missing selected span, a
missing execution or invocation, mismatched run/executable/step identities, a
non-schema-3 aggregate, invalid kernel-count semantics, exact-count mismatches,
or unmatched external activities.

Compute runtime aggregate schema version 3 distinguishes `exact` custom kernel
counts from `custom_dispatches_exact_library_fanout_unknown`. The latter means
direct launches are exact and library API calls are exact, but a library call
can fan out to more than one device kernel. Do not compare the aggregate count
to an external profiler as though it were an exact device-kernel total when
this value is reported.

Profiler ranges, bridge timing, and per-invocation evidence add overhead. Use
the capture to diagnose attribution and synchronization, then measure training
throughput separately with observation disabled under the same workload,
compiler options, dtype, optimizer, and target.

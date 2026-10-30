# Parity terms and case dimensions

The public workflow is **cases → run → results → compare**. Coverage reporting reads the saved results separately. Existing JSON field names remain compatible with retained runs.

## A case

A case is one concrete thing to check. Its ID is stable, and its dimensions are recorded in each new run's `manifest.json` under `axes`.

| Dimension | Definition | Part of identity? |
| --- | --- | --- |
| `device` | CPU, Metal, or CUDA backend. The exact machine belongs to the run. | Yes |
| `dtype` | Primary value type. Index and output types can be recorded separately in the case contract. | Yes |
| `operation_family` | Browsing group for an operation, such as reduction or indexing. Derived from the operation vocabulary. | No; changing a group must not duplicate the case |
| `operation` | Affon operation under check. | Yes |
| `shape` | Concrete input dimensions. Additional input and output shapes belong in the case contract. | Yes, through the named case definition |
| `layout` | Declared dense, transposed, strided, or offset input view. Legacy catalogs infer this from the case ID unless they specify it. | Yes, through the named case definition |
| `regime` | A named size or behavior class, such as `short_axis` or `ragged`. | Yes |
| `variant` | A further named choice, such as an axis, top-k setting, or compile option. | Yes |
| `scope` | Operation, Program, or model piece. Current operation cases run both an operator check and a composed Program check. Dedicated Program and model piece cases need their own runners; this runner accepts operation cases only. | Yes for a dedicated Program or model case |
| `piece` | Graph or model identity for a dedicated Program or model case. | Yes when scope is Program or model piece |

The case ID uses `device/dtype/operation/regime[/variant]`. The `operation_family` is derived from the operation; it is not a separate combination axis. The JSON case definition supplies concrete shapes, index types, axes, and other operation arguments. A catalog is simply a named list of cases. It is not a pass list. Only declared combinations are run; no implicit Cartesian product is claimed.

## A run and its results

A run records its target machine, software and binary identity, selected cases, and measurement settings. Each case gets three checks:

1. **Numerical:** Affon output against an independent formula, including the implementation Affon selected.
2. **Operator:** matched Affon and PyTorch correctness plus repeated timing.
3. **Program:** a specified composition, with its own correctness and repeated timing.

`manifest.json` summarizes every case and check. New runs also store a hash of each case definition so comparisons can detect changed inputs or options even when a case ID stays the same. `capability.json` is the historical filename for raw numerical results; `operator_peer/` and `program_peer/` contain raw timing results. Failure directories retain failed checks. `report.md` is the readable view. These filenames remain stable because saved links and hashes refer to them.

A check status is an **outcome**, not a case dimension. Run outcomes include passed, incorrect, and error. Coverage reporting separately classifies a planned cell as supported, unsupported, not applicable, or unknown. A passing timing check records samples; it does not accept a performance baseline.

## Internal coverage inputs

`benchmarks/coverage-policy.json` expands the operation vocabulary into planned cells. It defines what to cover, not what passed. `benchmarks/capability-probes.json` is the historical master case list. `coverage-outcomes-current-*.json` links reviewed raw results into the coverage report with file hashes. These inputs are needed for aggregate reporting; a named case list is sufficient to run parity checks.

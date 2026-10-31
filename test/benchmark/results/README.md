# Performance and parity results

Start with the [parity results index](PARITY-INDEX.md) to find named catalogs, retained batch reports, and the current aggregates. Open the [HTML report](performance-report.html) to filter operator and Program timings by device, operation, dtype, and status.

Each `parity-*` directory is one immutable run. It contains the catalog snapshot, `capability.json`, raw `operator_peer` and `program_peer` timings, separate failure directories, `manifest.json`, `outcomes.json`, and `report.md`. Failed numerical probes and timings remain in their original batch directory.

`coverage-outcomes-current-mac-2026-10-29.json` is the reviewed aggregate coverage record. The `cell-peer-current-mac-2026-10-29` and `cell-program-peer-mac-2026-10-29` directories hold reviewed raw results; the adjacent summary JSON files derive from them. `performance-report.html` derives from those summaries and the coverage matrix. Existing file paths are stable because the aggregate record stores paths and SHA-256 hashes.

Run `python3 test/parity/index.py` after adding a retained batch. See [the parity facility guide](../../parity/README.md) for reruns and promotion.

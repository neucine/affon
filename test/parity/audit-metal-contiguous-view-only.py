#!/usr/bin/env python3
"""Review Metal contiguous view-only claims against selected paths and source."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / 'test/benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json'
PROBE = ROOT / 'test/benchmarks/results/parity-metal-remaining-runnable-audit-25-2026-10-31/capability.json'
REPORT = ROOT / 'test/benchmarks/results/parity-metal-contiguous-view-only-audit-2026-10-31.json'
DEVICE = ROOT.parent / 'compute/src/backend/metal/device.zig'
SHADER = ROOT.parent / 'compute/src/backend/metal/ffi.m'


def main():
    device = DEVICE.read_text()
    shader = SHADER.read_text()
    checks = {
        str(DEVICE): 'affon_compute_metal_contiguous(state(context).native_device, input.buffer.handle, output.buffer.handle',
        str(SHADER): '[encoder setBuffer:input.buffer offset:0 atIndex:0]; [encoder setBuffer:output.buffer offset:0 atIndex:1];',
    }
    for path, statement in checks.items():
        if statement not in (device if path == str(DEVICE) else shader):
            raise ValueError(f'{path}: contiguous implementation changed; review decision')
    cases = {case['id']: case for case in json.loads(PROBE.read_text())['cases']}
    ledger = json.loads(LEDGER.read_text())
    decisions = {}
    for dtype in ('f32', 'bf16'):
        cell = f'metal/{dtype}/contiguous/view_only'
        probe = cases[cell]
        if probe['status'] != 'correct' or [node['implementation_id'] for node in probe['selected_path']] != ['metal_contiguous']:
            raise ValueError(f'{cell}: selected path changed; review decision')
        outcome = {
            'support': 'unsupported',
            'reason': 'Metal contiguous selects metal_contiguous and writes a separate output buffer; this view-only path is unavailable. The materializing path has separate parity evidence.',
        }
        if cell in ledger and ledger[cell] != outcome:
            raise ValueError(f'{cell}: existing outcome conflicts with audit')
        decisions[cell] = outcome
        ledger[cell] = outcome
    report = {
        'schema_version': 1,
        'suite': 'affon_parity_contiguous_view_only_audit',
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (DEVICE, SHADER)},
        'probe_artifact': str(PROBE),
        'probe_sha256': hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        'checks': checks,
        'decisions': decisions,
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + '\n')
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(decisions)} reviewed unsupported paths in {REPORT}')


if __name__ == '__main__':
    main()

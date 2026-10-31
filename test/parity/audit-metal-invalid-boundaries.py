#!/usr/bin/env python3
"""Record invalid scatter/topk behavior against PyTorch MPS and source contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / 'test/benchmark/results/coverage-outcomes-current-mac-2026-10-29.json'
REPORT = ROOT / 'test/benchmark/results/parity-metal-invalid-boundaries-2026-10-31.json'
WORKER = ROOT / 'test/parity/invalid-index-boundary.ts'
AFFON = ROOT / 'zig-out/bin/affon'
PROGRAM = ROOT / 'src/js/compute/program.ts'
METAL = ROOT.parent / 'compute/src/backend/metal/ffi.m'


def main():
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('PyTorch MPS is required')
    source = PROGRAM.read_text()
    metal = METAL.read_text()
    if '!Number.isSafeInteger(k) || k < 1 || k > this.spec.shape[normalized]' not in source:
        raise ValueError('topk invalid-k contract changed')
    if 'int check=affon_compute_metal_preflight_scatter_index' not in metal:
        raise ValueError('BF16 scatter preflight contract changed')
    result = subprocess.run([str(AFFON), str(WORKER)], cwd=ROOT, text=True,
                            capture_output=True, check=True)
    affon = {row['id']: row for line in result.stdout.splitlines() if line.startswith('{')
             for row in [json.loads(line)]}
    if len(affon) != 4:
        raise ValueError('expected four Affon boundary results')
    peer = {}
    for dtype in ('f32', 'bf16'):
        kind = torch.float32 if dtype == 'f32' else torch.bfloat16
        try:
            base = torch.tensor([[1, 2], [3, 4]], dtype=kind, device='mps')
            indices = torch.tensor([[3, 0]], dtype=torch.long, device='mps')
            updates = torch.tensor([[1, 1]], dtype=kind, device='mps')
            value = base.scatter_add(0, indices, updates).tolist()
            peer[f'metal/{dtype}/scatter_add/invalid_index'] = {'status': 'returned', 'output': value}
        except Exception as exc:
            peer[f'metal/{dtype}/scatter_add/invalid_index'] = {'status': 'error', 'error': str(exc)}
        try:
            torch.topk(base, 3, dim=1)
            peer[f'metal/{dtype}/topk/invalid_index'] = {'status': 'accepted'}
        except Exception as exc:
            peer[f'metal/{dtype}/topk/invalid_index'] = {'status': 'error', 'error': str(exc)}
    f32 = 'metal/f32/scatter_add/invalid_index'
    bf16 = 'metal/bf16/scatter_add/invalid_index'
    topk = [f'metal/{dtype}/topk/invalid_index' for dtype in ('f32', 'bf16')]
    if not (affon[f32]['status'] == peer[f32]['status'] == 'returned' and
            affon[f32]['output'] == peer[f32]['output'] and
            affon[bf16]['status'] == 'error' and peer[bf16]['status'] == 'returned' and
            all(affon[cell]['status'] == peer[cell]['status'] == 'error' for cell in topk)):
        raise ValueError('invalid-index behavior changed; review outcomes')
    decisions = {bf16: {'support': 'unsupported',
                        'reason': 'Metal BF16 scatter_add rejects an out-of-range index with BackendFailure while PyTorch MPS returns a value; numerical and timing parity are unavailable.'}}
    for cell in topk:
        decisions[cell] = {'support': 'unsupported',
                           'reason': 'An out-of-range topk k is rejected before execution by Affon and PyTorch; no numerical output or timing sample exists.'}
    ledger = json.loads(LEDGER.read_text())
    if ledger.get(f32, {}).get('support') != 'supported':
        raise ValueError('passing f32 scatter parity evidence must be promoted first')
    for cell, decision in decisions.items():
        if cell in ledger and ledger[cell] != decision:
            raise ValueError(f'{cell}: existing outcome conflicts with boundary audit')
        ledger[cell] = decision
    report = {'schema_version': 1, 'suite': 'affon_parity_invalid_boundaries',
              'binary_sha256': hashlib.sha256(AFFON.read_bytes()).hexdigest(),
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (WORKER, PROGRAM, METAL)},
              'peer_engine': 'pytorch', 'peer_version': torch.__version__,
              'affon': affon, 'peer': peer, 'decisions': decisions,
              'f32_scatter_parity_run': str((ROOT / 'test/benchmark/results/parity-metal-invalid-scatter-2-2026-10-31').resolve())}
    LEDGER.write_text(json.dumps(ledger, indent=2) + '\n')
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(decisions)} unsupported boundaries in {REPORT}')


if __name__ == '__main__':
    main()

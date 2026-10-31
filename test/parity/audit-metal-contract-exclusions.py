#!/usr/bin/env python3
"""Record reviewed Metal f32/BF16 exclusions proved by Compute contracts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / 'test/benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json'
REPORT = ROOT / 'test/benchmarks/results/parity-metal-contract-exclusions-38-2026-10-31.json'
ABSTRACT = ROOT.parent / 'compute/src/program/abstract_eval.zig'
SCHEMA = ROOT.parent / 'compute/src/program/schema.zig'


def main():
    abstract = ABSTRACT.read_text()
    schema = SCHEMA.read_text()
    checks = {
        'dot': ('if (lhs.dimensions().len != 1 or rhs.dimensions().len != 1) return error.UnsupportedRank;', abstract),
        'reduce_to_shape': ('if (input.dtype() != .f32 and input.dtype() != .i64) return error.UnsupportedDType;', abstract),
        'reserved': ('.gelu_grad, .cross_entropy_indexed_backward => error.UnsupportedOperation,', schema),
    }
    for name, (statement, source) in checks.items():
        if statement not in source:
            raise ValueError(f'{name} contract changed; review exclusions')
    decisions = {}
    for dtype in ('f32', 'bf16'):
        for regime in ('decode_m1', 'small_batch', 'prefill', 'large_matrix', 'batched', 'transposed'):
            decisions[f'metal/{dtype}/dot/{regime}'] = {
                'support': 'not_applicable',
                'reason': 'Compute dot accepts two rank-one vectors only; this matrix workload regime is not a dot operation.',
                'contract': 'dot',
            }
        for regime in ('scalar', 'small_dense', 'large_dense', 'strided', 'offset', 'ragged'):
            decisions[f'metal/{dtype}/gelu_grad/{regime}'] = {
                'support': 'unsupported',
                'reason': 'Compute schemaFor returns UnsupportedOperation for reserved gelu_grad.',
                'contract': 'reserved',
            }
        for regime in ('small', 'large', 'batched', 'strided'):
            decisions[f'metal/{dtype}/cross_entropy_indexed_backward/{regime}'] = {
                'support': 'unsupported',
                'reason': 'Compute schemaFor returns UnsupportedOperation for reserved cross_entropy_indexed_backward.',
                'contract': 'reserved',
            }
    for regime in ('scalar', 'short_axis', 'long_axis', 'outer_axis', 'strided', 'ragged'):
        decisions[f'metal/bf16/reduce_to_shape/{regime}'] = {
            'support': 'unsupported',
            'reason': 'Compute reduceToShape accepts f32 or i64 input; bf16 returns UnsupportedDType.',
            'contract': 'reduce_to_shape',
        }
    if len(decisions) != 38:
        raise ValueError('unexpected exclusion count')
    ledger = json.loads(LEDGER.read_text())
    for cell, decision in decisions.items():
        outcome = {key: decision[key] for key in ('support', 'reason')}
        if cell in ledger and ledger[cell] != outcome:
            raise ValueError(f'{cell}: existing outcome conflicts with contract audit')
        ledger[cell] = outcome
    report = {
        'schema_version': 1, 'suite': 'affon_parity_contract_exclusions',
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ABSTRACT, SCHEMA)},
        'contracts': {name: {'source': str(ABSTRACT if name != 'reserved' else SCHEMA),
                             'statement': statement}
                      for name, (statement, _) in checks.items()},
        'cells': [{'id': cell, **decision} for cell, decision in sorted(decisions.items())],
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + '\n')
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(decisions)} reviewed exclusions in {REPORT}')


if __name__ == '__main__':
    main()

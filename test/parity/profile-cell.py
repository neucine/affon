#!/usr/bin/env python3
"""Capture one exact parity cell with Affon runtime evidence and PyTorch timing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / 'reference/benchmark-cells-pytorch.py'


def load_reference():
    spec = importlib.util.spec_from_file_location('parity_benchmark_cells', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--kind', choices=('operator_peer', 'program_peer'), default='operator_peer')
    parser.add_argument('--binary', type=Path, default=ROOT / 'zig-out/bin/affon')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--capture-records', type=int, default=65536)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if not args.binary.is_file():
        parser.error('Affon binary missing')
    if args.warmup < 0 or not 1 <= args.capture_records <= 4194304:
        parser.error('invalid warmup or capture record limit')
    catalog = json.loads(args.catalog.read_text())
    matches = [case for case in catalog['cases'] if case['id'] == args.cell]
    if len(matches) != 1:
        parser.error('cell must occur exactly once in catalog')
    case = matches[0]
    if case['op'] in {'topk', 'quantized_matmul', 'scatter_add', 'cross_entropy',
                      'log_softmax_nll', 'reduce_to_shape'} or (case['op'] == 'index_select' and len(case['shape']) == 3):
        parser.error('this cell uses a specialized worker that does not yet support selected-run capture')

    reference = load_reference()
    affon = reference.affon(args.binary, case, 1, 3, args.kind, args.warmup,
                            {'AFFON_CELL_PROFILE': '1', 'AFFON_CELL_CAPTURE_RECORDS': str(args.capture_records)})
    invocation_count = sum(entry.get('invocation_count', 0) for entry in affon['runtime_evidence']['entries'])
    peer_output, peer_timing = reference.pytorch(case, 1, 3, args.kind)
    if (affon['output_count'] != peer_output['output_count'] or
            affon['output_shape'] != peer_output['output_shape'] or
            affon['output_dtype'] != peer_output['output_dtype'] or
            [item['index'] for item in affon['output_samples']] !=
            [item['index'] for item in peer_output['output_samples']]):
        raise ValueError('selected run differs from PyTorch output contract')
    max_sample_error = max(abs(a['value'] - b['value']) for a, b in
                           zip(affon['output_samples'], peer_output['output_samples']))
    relative_tolerance = (0.02 if case['dtype'] == 'bf16' else
                          1e-5 if case['op'] in {'variance_axis', 'std_axis'} else
                          2e-6 if case['op'] in {'dot', 'layer_norm', 'exp'} else 1e-6)
    tolerance = relative_tolerance * max(1, *(abs(item['value']) for item in peer_output['output_samples']))
    checksum_error = abs(affon['output_checksum'] - peer_output['output_checksum'])
    if max_sample_error > tolerance or checksum_error > tolerance * peer_output['output_count']:
        raise ValueError('selected run failed sampled PyTorch parity')
    report = {
        'schema_version': 1, 'suite': 'parity_cell_profile',
        'cell_id': args.cell, 'kind': args.kind, 'case': case,
        'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        'affon_binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        'affon': affon,
        'pytorch': {'output': peer_output, 'timing': peer_timing},
        'comparison': {'max_sample_error': max_sample_error, 'tolerance': tolerance,
                       'checksum_error': checksum_error, 'passed': True},
        'limitations': [
            'Affon selected-run timing includes profiling overhead and must not be compared to PyTorch timing.',
            'PyTorch samples use the parity timing boundary; no PyTorch device activity trace is captured.',
            'Use the uninstrumented parity runner for correctness acceptance and performance ratios.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'cell_id': args.cell, 'kind': args.kind,
                      'invocations': invocation_count,
                      'max_sample_error': max_sample_error}))


if __name__ == '__main__':
    main()

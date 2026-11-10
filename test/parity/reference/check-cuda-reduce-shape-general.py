#!/usr/bin/env python3
"""Check every output of seeded CUDA reduce-to-shape cases outside the catalog.

Integer reductions must match exactly. The scalar and column f32 paths reassociate
within one output, so f32 values are checked against the same tolerance as the
capability probes instead of bit-for-bit sequential order.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LAYOUTS = ('dense', 'offset', 'strided', 'transposed')
SHAPES = ((1, 1), (1, 37), (7, 1), (5, 11), (13, 29), (31, 17),
          (37, 41), (23, 7))


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def physical(logical, layout, noise):
    rows, cols = len(logical), len(logical[0])
    if layout == 'dense':
        return logical, None, None
    if layout == 'transposed':
        return [[logical[row][col] for row in range(rows)] for col in range(cols)], None, [1, 0]
    if layout == 'offset':
        value = [[noise() for _ in range(cols + 3)] for _ in range(rows + 2)]
        for row in range(rows):
            for col in range(cols):
                value[row + 1][col + 2] = logical[row][col]
        return value, [{'start': 1, 'stop': rows + 1}, {'start': 2, 'stop': cols + 2}], None
    value = [[noise() for _ in range(cols * 3 + 2)] for _ in range(rows * 2 + 1)]
    for row in range(rows):
        for col in range(cols):
            value[1 + row * 2][1 + col * 3] = logical[row][col]
    return value, [{'start': 1, 'stop': rows * 2 + 1, 'step': 2},
                   {'start': 1, 'stop': cols * 3 + 1, 'step': 3}], None


def cases(seed):
    rng = random.Random(seed)
    shapes = list(SHAPES) + [(rng.randrange(2, 35), rng.randrange(2, 47)) for _ in range(8)]
    result = []
    for dtype in ('f32', 'i64'):
        for index, (rows, cols) in enumerate(shapes):
            for layout_index, layout in enumerate(LAYOUTS):
                def value():
                    return rng.randrange(-100000, 100001) if dtype == 'i64' else f32(rng.uniform(-1000, 1000))
                logical = [[value() for _ in range(cols)] for _ in range(rows)]
                data, slices, permutation = physical(logical, layout, value)
                targets = ([], [1], [cols], [1, 1], [rows, 1], [1, cols], [rows, cols])
                target = targets[(index * 3 + layout_index + (dtype == 'i64')) % len(targets)]
                case = {'id': f'{dtype}/{seed}/{index}/{layout}', 'dtype': dtype,
                        'shape': [rows, cols], 'target_shape': target,
                        'physical_shape': [len(data), len(data[0])],
                        'physical': data, 'logical': logical, 'layout': layout}
                if slices: case['slices'] = slices
                if permutation: case['permutation'] = permutation
                result.append(case)
    return result


def reference(case, composed=False):
    rows, cols = case['shape']
    target = case['target_shape']
    count = 1
    for dimension in target: count *= dimension
    buckets = [[] for _ in range(count)]
    for row in range(rows):
        for col in range(cols):
            if len(target) == 0: index = 0
            elif len(target) == 1: index = 0 if target[0] == 1 else col
            else: index = (0 if target[0] == 1 else row) * target[1] + (0 if target[1] == 1 else col)
            value = case['logical'][row][col]
            buckets[index].append(f32(value + value) if composed else value)
    if case['dtype'] == 'i64':
        return [sum(bucket) for bucket in buckets]
    return [math.fsum(bucket) for bucket in buckets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = cases(args.seed)
    payload = output / 'inputs.json'
    payload.write_text(json.dumps(inputs) + '\n')
    env = {**os.environ, 'AFFON_REDUCE_CASES': str(payload),
           'AFFON_REDUCE_OUTPUT': str(output / 'outputs.json')}
    process = subprocess.run([str(args.binary.resolve()),
                              str(ROOT / 'test/parity/workers/cuda-reduce-shape-general.ts')],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    (output / 'stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f'Affon exited {process.returncode}; see {output / "stderr.txt"}')
    raw = json.loads((output / 'outputs.json').read_text())
    lookup = {(row['id'], row['composed']): row for row in raw}
    if len(lookup) != len(inputs) * 2 or len(raw) != len(lookup):
        raise RuntimeError('missing or duplicate outputs')
    checks = []
    for case in inputs:
        for composed in (False, True):
            row = lookup[(case['id'], composed)]
            wanted = reference(case, composed)
            tolerance = 0.0 if case['dtype'] == 'i64' else 2e-6 * max(1.0, *(abs(value) for value in wanted))
            check = {'id': case['id'], 'composed': composed, 'dtype': case['dtype'],
                     'layout': case['layout'], 'shape': case['shape'], 'target_shape': case['target_shape']}
            try:
                if 'error' in row: raise AssertionError(row['error'])
                if row['shape'] != case['target_shape'] or row['dtype'] != case['dtype']:
                    raise AssertionError('output shape or dtype differs')
                if len(row['values']) != len(wanted):
                    raise AssertionError('output length differs')
                if case['dtype'] == 'i64':
                    if row['values'] != wanted:
                        raise AssertionError(f'full output differs: expected {wanted[:8]}, actual {row["values"][:8]}')
                else:
                    for actual, wanted_value in zip(row['values'], wanted):
                        if abs(actual - wanted_value) > tolerance:
                            raise AssertionError(f'full output differs: expected {wanted[:8]}, actual {row["values"][:8]}')
                if not any(node == 'cuda_reduce_to_shape' for node in row['path']):
                    raise AssertionError('native reduction path absent')
                if any(node.startswith('host_') for node in row['path']):
                    raise AssertionError('host fallback selected')
                check['status'] = 'passed'
            except AssertionError as error:
                check.update(status='failed', error=str(error))
            checks.append(check)
    report = {'suite': 'cuda_reduce_to_shape_general_checks', 'seed': args.seed,
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'inputs_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
              'reference': 'Exact fsum reference; i64 must match exactly and f32 must stay within the 2e-6 capability-probe tolerance',
              'input_cases': len(inputs), 'total': len(checks),
              'passed': sum(row['status'] == 'passed' for row in checks), 'checks': checks}
    for name in ('inputs.json', 'outputs.json'):
        artifact = output / name
        with gzip.open(str(artifact) + '.gz', 'wb', compresslevel=6) as compressed:
            compressed.write(artifact.read_bytes())
        artifact.unlink()
    report.update(inputs_artifact='inputs.json.gz', outputs_artifact='outputs.json.gz')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'seed': args.seed, 'input_cases': len(inputs),
                      'passed': report['passed'], 'total': len(checks)}))
    if report['passed'] != len(checks): sys.exit(1)


if __name__ == '__main__': main()

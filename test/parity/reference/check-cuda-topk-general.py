#!/usr/bin/env python3
"""Check every CUDA top-k output for seeded shapes, values, ties, and views."""
import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SHAPES = ((5, 47), (37, 9), (3, 7, 41),
          (2, 43, 5), (4, 11, 37), (3, 19), (7, 5))


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def filled(shape, value):
    return value() if not shape else [filled(shape[1:], value) for _ in range(shape[0])]


def get(data, coordinate):
    for index in coordinate: data = data[index]
    return data


def put(data, coordinate, value):
    for index in coordinate[:-1]: data = data[index]
    data[coordinate[-1]] = value


def physical(logical, shape, layout, noise):
    if layout == 'dense': return logical, None, None
    if layout == 'transposed':
        permutation = [1, 0] if len(shape) == 2 else [0, 2, 1]
        physical_shape = [shape[permutation.index(axis)] for axis in range(len(shape))]
        data = filled(physical_shape, noise)
        for coordinate in itertools.product(*(range(n) for n in shape)):
            put(data, tuple(coordinate[permutation.index(axis)] for axis in range(len(shape))), get(logical, coordinate))
        return data, None, permutation
    if layout == 'offset':
        data = filled([n + 2 for n in shape], noise)
        for coordinate in itertools.product(*(range(n) for n in shape)):
            put(data, tuple(index + 1 for index in coordinate), get(logical, coordinate))
        return data, [{'start': 1, 'stop': n + 1} for n in shape], None
    data = filled([2 * n + 1 for n in shape], noise)
    for coordinate in itertools.product(*(range(n) for n in shape)):
        put(data, tuple(1 + 2 * index for index in coordinate), get(logical, coordinate))
    return data, [{'start': 1, 'stop': 2 * n + 1, 'step': 2} for n in shape], None


def cases(seed, include_nan=False):
    rng = random.Random(seed)
    results = []
    for dtype in ('f32', 'bf16'):
        for shape_index, shape in enumerate(SHAPES):
            for axis in range(len(shape)):
                layouts = ('dense', 'offset', 'strided') if len(shape) == 1 else ('dense', 'offset', 'strided', 'transposed')
                for layout_index, layout in enumerate(layouts):
                    ordinal = shape_index * 11 + axis * 5 + layout_index + (dtype == 'bf16')
                    k = (1, max(1, shape[axis] // 2), shape[axis])[ordinal % 3]
                    largest = bool(ordinal % 2)
                    all_equal = ordinal % 7 == 0
                    def value():
                        if all_equal: return 3
                        if dtype == 'bf16': return rng.randrange(-32, 33) / 4
                        return f32(rng.uniform(-10, 10)) if ordinal % 5 == 1 else rng.randrange(-32, 33) / 4
                    logical = filled(shape, value)
                    data, slices, permutation = physical(logical, shape, layout, value)
                    case = {'id': f'{dtype}/{seed}/{shape_index}/{axis}/{layout}', 'dtype': dtype,
                            'shape': list(shape), 'axis': axis, 'k': k, 'largest': largest,
                            'layout': layout, 'logical': logical, 'physical': data,
                            'physical_shape': [len(data)] + ([len(data[0])] if len(shape) >= 2 else []) +
                            ([len(data[0][0])] if len(shape) == 3 else [])}
                    if slices: case['slices'] = slices
                    if permutation: case['permutation'] = permutation
                    results.append(case)
    if include_nan:
        for dtype in ('f32', 'bf16'):
            for shape in ((2, 5), (3, 33)):
                for largest in (False, True):
                    for k in (1, shape[1] // 2, shape[1]):
                        for layout in ('dense', 'offset', 'strided', 'transposed'):
                            logical = [([2, '__NaN__', 1, '__NaN__', 2] * ((shape[1] + 4) // 5))[:shape[1]]
                                       for _ in range(shape[0])]
                            data, slices, permutation = physical(logical, shape, layout, lambda: 0)
                            case = {'id': f'{dtype}/{seed}/nan/{shape[1]}/{int(largest)}/{k}/{layout}',
                                    'dtype': dtype, 'shape': list(shape), 'axis': 1, 'k': k,
                                    'largest': largest, 'layout': layout, 'logical': logical,
                                    'physical': data, 'physical_shape': [len(data), len(data[0])],
                                    'nan_values': True}
                            if slices: case['slices'] = slices
                            if permutation: case['permutation'] = permutation
                            results.append(case)
    return results


def expected(case):
    shape = case['shape']; axis = case['axis']; output_shape = shape.copy();output_shape[axis] = case['k']
    values = [];indices = []
    for coordinate in itertools.product(*(range(n) for n in output_shape)):
        candidates = []
        for index in range(shape[axis]):
            position = list(coordinate);position[axis] = index
            candidates.append((get(case['logical'], position), index))
        def ordering(item):
            value, index = item
            if value == '__NaN__': return (0 if case['largest'] else 1, 0, index)
            return (1 if case['largest'] else 0, -value if case['largest'] else value, index)
        candidates.sort(key=ordering)
        value, index = candidates[coordinate[axis]]
        values.append(value);indices.append(index)
    return output_shape, values, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--include-nan', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve();output.mkdir(parents=True, exist_ok=False)
    inputs = cases(args.seed, args.include_nan);payload = output / 'inputs.json';payload.write_text(json.dumps(inputs) + '\n')
    env = {**os.environ, 'AFFON_TOPK_GENERAL_CASES': str(payload),
           'AFFON_TOPK_GENERAL_OUTPUT': str(output / 'outputs.json')}
    process = subprocess.run([str(args.binary.resolve()), str(ROOT / 'test/parity/workers/cuda-topk-general.ts')],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    (output / 'stderr.txt').write_text(process.stderr)
    if process.returncode: raise RuntimeError(f'Affon exited {process.returncode}; see {output / "stderr.txt"}')
    raw = json.loads((output / 'outputs.json').read_text());lookup = {(x['id'], x['composed']): x for x in raw}
    if len(lookup) != len(raw) or len(raw) != len(inputs) * 2: raise RuntimeError('missing or duplicate outputs')
    checks = []
    for case in inputs:
        shape, values, indices = expected(case)
        for composed in (False, True):
            row = lookup[(case['id'], composed)]
            wanted = (['__NaN__' if value == '__NaN__' else
                       f32(value + value) if case['dtype'] == 'f32' else value + value
                       for value in values]) if composed else values
            check = {'id': case['id'], 'composed': composed, 'dtype': case['dtype'],
                     'shape': case['shape'], 'layout': case['layout'], 'axis': case['axis'], 'k': case['k']}
            try:
                if 'error' in row: raise AssertionError(row['error'])
                if row['value_shape'] != shape or row['value_dtype'] != case['dtype'] or row['index_dtype'] != 'i64':
                    raise AssertionError('output shape or dtype mismatch')
                if row['values'] != wanted or row['indices'] != indices:
                    raise AssertionError(f'full output mismatch: {row["values"][:8]}, {row["indices"][:8]}')
                if 'cuda_topk' not in row['path'] or any(node.startswith('host_') for node in row['path']):
                    raise AssertionError('native top-k path absent')
                check['status'] = 'passed'
            except AssertionError as error: check.update(status='failed', error=str(error))
            checks.append(check)
    report = {'suite': 'cuda_topk_general_full_output_checks', 'seed': args.seed,
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'inputs_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
              'input_cases': len(inputs), 'total': len(checks), 'passed': sum(x['status'] == 'passed' for x in checks),
              'reference': 'Independent Python stable value ordering with lower-index ties; every value and index checked',
              'checks': checks}
    for name in ('inputs.json', 'outputs.json'):
        artifact = output / name
        with gzip.open(str(artifact) + '.gz', 'wb', compresslevel=6) as compressed: compressed.write(artifact.read_bytes())
        artifact.unlink()
    report.update(inputs_artifact='inputs.json.gz', outputs_artifact='outputs.json.gz')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'seed': args.seed, 'passed': report['passed'], 'total': report['total']}))
    if report['passed'] != report['total']: sys.exit(1)


if __name__ == '__main__': main()

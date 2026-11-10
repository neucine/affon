#!/usr/bin/env python3
"""Check every output of seeded, non-catalog CUDA normalization/loss cases.

Rank-four losses are retained as explicit public-API rejection tests; they do
not count as numerical passes. Reference arithmetic uses CPU float64 after
rounding physical inputs to f32/BF16. No tolerance depends on case identity.
"""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]


def cases(seed):
    rng = torch.Generator().manual_seed(seed)
    norms = [([1, 1], 1), ([5, 9], 1), ([9, 31], 1), ([5, 32], 1),
             ([17, 33], 1), ([3, 61], 1), ([9, 127], 1), ([5, 129], 1),
             ([2, 509], 1), ([3, 1025], 1), ([1, 2049], 1), ([5, 17], 0),
             ([2, 5, 19], 1), ([3, 7, 11], 2), ([2, 3, 5, 11], 2),
             ([1, 3, 1, 7], 0)]
    losses = [[1, 1], [1, 2], [5, 9], [9, 31], [5, 32], [17, 33],
              [3, 61], [9, 127], [5, 129], [2, 509], [3, 1025],
              [1, 2049], [2, 5, 19], [3, 7, 11], [2, 3, 5, 11], [1, 17, 1, 7]]
    for _ in range(8):
        norms.append(([int(torch.randint(1, 24, (), generator=rng)), int(torch.randint(1, 2100, (), generator=rng))], 1))
        losses.append([int(torch.randint(1, 24, (), generator=rng)), int(torch.randint(1, 2100, (), generator=rng))])
    result = []
    for dtype in ('f32', 'bf16'):
        for op in ('layer_norm', 'rms_norm', 'cross_entropy', 'log_softmax_nll'):
            for index, item in enumerate(norms if op.endswith('norm') else losses):
                shape, axis = item if op.endswith('norm') else (item, 1)
                layout = ('dense', 'offset', 'strided', 'transposed')[index % 4]
                physical_shape = list(reversed(shape)) if layout == 'transposed' else [n * 2 + 2 for n in shape] if layout == 'strided' else [n + 2 for n in shape] if layout == 'offset' else shape
                x = torch.randn(physical_shape, generator=rng)
                # Large offsets, near-constant rows, zeros, and signed logits
                # exercise numerical stability independently of catalog fixtures.
                distribution = index % 5
                if distribution == 0: x = x * 0.5 + 10000
                elif distribution == 1: x = x * 0.00001 + 0.125
                elif distribution == 2: x = x * 40
                elif distribution == 3: x = torch.zeros_like(x)
                else: x = x * 3
                case = {'id': f'{dtype}/{op}/{index}/{layout}', 'dtype': dtype, 'op': op,
                        'shape': shape, 'physical_shape': physical_shape, 'axis': axis,
                        'epsilon': (1e-8, 1e-5, 0.1)[index % 3], 'x': x.tolist()}
                if layout == 'transposed': case['permutation'] = list(reversed(range(len(shape))))
                elif layout != 'dense': case['slices'] = [{'start': 1, 'stop': 1 + n * (2 if layout == 'strided' else 1), 'step': 2 if layout == 'strided' else 1} for n in shape]
                if not op.endswith('norm'):
                    case['y'] = torch.rand(physical_shape, generator=rng).tolist()
                result.append(case)
    return result


def expected(case):
    dtype = torch.bfloat16 if case['dtype'] == 'bf16' else torch.float32
    def view(values):
        value = torch.tensor(values, dtype=dtype).double()
        if 'permutation' in case: return value.permute(case['permutation'])
        if 'slices' in case: return value[tuple(slice(s['start'], s['stop'], s['step']) for s in case['slices'])]
        return value
    x = view(case['x'])
    if case['op'] == 'layer_norm':
        value = torch.nn.functional.layer_norm(x, tuple(case['shape'][case['axis']:]), eps=case['epsilon'])
    elif case['op'] == 'rms_norm':
        axes = tuple(range(case['axis'], len(case['shape'])))
        value = x / torch.sqrt((x * x).mean(dim=axes, keepdim=True) + case['epsilon'])
    else:
        value = (-(view(case['y']) * torch.log_softmax(x, dim=1)).sum(dim=1).mean()).reshape(1)
    return value.to(dtype).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = cases(args.seed)
    payload = output / 'inputs.json'
    payload.write_text(json.dumps(inputs) + '\n')
    env = {**os.environ, 'AFFON_GENERAL_CASES': str(payload), 'AFFON_GENERAL_OUTPUT': str(output / 'outputs.json')}
    process = subprocess.run([str(args.binary.resolve()), str(ROOT / 'test/parity/workers/cuda-general.ts')],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    (output / 'stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f'Affon exited {process.returncode}; see {output / "stderr.txt"}')
    raw = json.loads((output / 'outputs.json').read_text())
    lookup = {(row['id'], row['composed']): row for row in raw}
    if len(lookup) != len(inputs) * 2: raise RuntimeError('missing or duplicate result')
    checks = []
    for case in inputs:
        reference = expected(case)
        for composed in (False, True):
            row = lookup[(case['id'], composed)]
            wanted = reference * (2 if composed else 1)
            check = {'id': case['id'], 'composed': composed, 'shape': case['shape'], 'layout': case['id'].rsplit('/', 1)[1]}
            try:
                if case['op'] in {'cross_entropy', 'log_softmax_nll'} and len(case['shape']) > 3:
                    if 'two- or three-dimensional logits' not in row.get('error', ''): raise AssertionError('expected public loss rank validation')
                    check.update(status='rejected_as_expected', reason='public dense loss API accepts rank two or three')
                    checks.append(check)
                    continue
                if 'error' in row: raise AssertionError(row['error'])
                if row['dtype'] != case['dtype'] or row['shape'] != list(wanted.shape): raise AssertionError('output dtype/shape mismatch')
                if any('host_' in node for node in row['path']): raise AssertionError('host fallback selected')
                actual = torch.tensor(row['values'], dtype=torch.float32).reshape(wanted.shape)
                tolerance = {'rtol': 0.008, 'atol': 0.008} if case['dtype'] == 'bf16' else {'rtol': 2e-6, 'atol': 2e-6}
                torch.testing.assert_close(actual, wanted, **tolerance, equal_nan=False)
                check.update(status='passed', maximum_error=(actual-wanted).abs().max().item(), **tolerance)
            except AssertionError as error:
                check.update(status='failed', error=str(error))
            checks.append(check)
    report = {'suite': 'cuda_general_case_checks', 'seed': args.seed, 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'inputs_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(), 'pytorch_version': torch.__version__,
              'reference': 'PyTorch CPU float64 after quantizing source values to the declared input dtype; every output checked',
              'checks': checks, 'passed': sum(c['status'] == 'passed' for c in checks),
              'expected_rejections': sum(c['status'] == 'rejected_as_expected' for c in checks), 'total': len(checks)}
    for name in ('inputs.json', 'outputs.json'):
        artifact = output / name
        with gzip.open(str(artifact) + '.gz', 'wb', compresslevel=6) as compressed:
            compressed.write(artifact.read_bytes())
        artifact.unlink()
    report.update(inputs_artifact='inputs.json.gz', outputs_artifact='outputs.json.gz',
                  artifact_encoding='gzip; inputs_sha256 hashes uncompressed JSON bytes')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('seed','passed','expected_rejections','total')}))
    if report['passed'] + report['expected_rejections'] != report['total']: sys.exit(1)


if __name__ == '__main__':
    main()

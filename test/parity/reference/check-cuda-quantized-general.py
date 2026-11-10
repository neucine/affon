#!/usr/bin/env python3
"""Check all outputs of independent mixed-dtype, strided packed-weight products."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
DTYPES = {'f32': torch.float32, 'bf16': torch.bfloat16, 'u32': torch.int64}


def physical(value, dtype, layout):
    rows, columns = value.shape
    result = {'dtype': dtype}
    if layout == 'transposed':
        stored = value.T.contiguous()
        result['permutation'] = [1, 0]
    elif layout in ('offset', 'strided'):
        step = 2 if layout == 'strided' else 1
        stored = torch.zeros((rows * step + 2, columns * step + 2), dtype=value.dtype)
        stored[1:1+rows*step:step, 1:1+columns*step:step] = value
        result['slices'] = [{'start': 1, 'stop': 1 + n * step, 'step': step} for n in (rows, columns)]
    else:
        stored = value
    result.update(shape=list(stored.shape), values=stored.tolist())
    return result


def cases(seed):
    rng = torch.Generator().manual_seed(seed)
    result = []
    for bits in (2, 4, 8):
        per_word = 32 // bits
        factor = int(torch.randint(1, 34, (), generator=rng))
        dimensions = [(1, per_word * 3, 5, 3), (3, per_word * 17, 19, per_word),
                      (int(torch.randint(1, 13, (), generator=rng)), per_word * factor,
                       int(torch.randint(1, 24, (), generator=rng)), factor),
                      (7, per_word * 31, 3, 31)]
        for x_dtype in ('f32', 'bf16'):
            for parameter_dtype in ('f32', 'bf16'):
                for index, (rows, inner, columns, group) in enumerate(dimensions):
                    for layout in ('dense', 'offset', 'strided', 'transposed'):
                        x = torch.rand((rows, inner), generator=rng) - 0.5
                        q = torch.randint(0, 1 << bits, (columns, inner), generator=rng)
                        shifts = torch.arange(per_word) * bits
                        words = (q.reshape(columns, inner // per_word, per_word) << shifts).sum(dim=-1)
                        scales = torch.rand((columns, inner // group), generator=rng) / (1 << bits)
                        biases = (torch.rand((columns, inner // group), generator=rng) - 0.5) / 4
                        result.append({'id': f'q{bits}/{x_dtype}/{parameter_dtype}/{index}/{layout}',
                                       'shape': [rows, inner, columns], 'bits': bits, 'group_size': group,
                                       'inputs': {name: physical(value, dtype, layout) for name, value, dtype in (
                                           ('x', x, x_dtype), ('w', words, 'u32'),
                                           ('s', scales, parameter_dtype), ('b', biases, parameter_dtype))}})
    return result


def view(input):
    value = torch.tensor(input['values'], dtype=DTYPES[input['dtype']])
    if 'permutation' in input:
        return value.permute(input['permutation'])
    if 'slices' in input:
        return value[tuple(slice(s['start'], s['stop'], s['step']) for s in input['slices'])]
    return value


def expected(case):
    inputs = {name: view(value) for name, value in case['inputs'].items()}
    bits, group = case['bits'], case['group_size']
    per_word = 32 // bits
    shifts = torch.arange(per_word) * bits
    digits = ((inputs['w'].unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten(1).double()
    decoded = digits * inputs['s'].double().repeat_interleave(group, dim=1) + inputs['b'].double().repeat_interleave(group, dim=1)
    return (inputs['x'].double() @ decoded.T).float()


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
    process = subprocess.run([str(args.binary.resolve()), str(ROOT / 'test/parity/workers/cuda-quantized-general.ts')],
                             cwd=ROOT, env={**os.environ, 'AFFON_GENERAL_CASES': str(payload),
                             'AFFON_GENERAL_OUTPUT': str(output / 'outputs.json')}, capture_output=True, text=True, timeout=600)
    (output / 'stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f'Affon exited {process.returncode}; see {output / "stderr.txt"}')
    raw = json.loads((output / 'outputs.json').read_text())
    lookup = {(row['id'], row['composed']): row for row in raw}
    if len(lookup) != len(raw) or len(lookup) != len(inputs) * 2:
        raise RuntimeError('missing or duplicate result')
    checks = []
    for case in inputs:
        reference = expected(case)
        for composed in (False, True):
            row = lookup[(case['id'], composed)]
            wanted = reference * (2 if composed else 1)
            # Same f32 peak-scaled absolute tolerance as the catalog oracle.
            # Every output is checked, including near-zero values.
            tolerance = 2e-6 * max(1.0, wanted.abs().max().item())
            check = {'id': case['id'], 'composed': composed, 'shape': case['shape'],
                     'bits': case['bits'], 'group_size': case['group_size'], 'tolerance': tolerance}
            try:
                if 'error' in row: raise AssertionError(row['error'])
                if row['dtype'] != 'f32' or row['shape'] != list(wanted.shape): raise AssertionError('output dtype/shape mismatch')
                if 'cuda_quantized_matmul' not in row['path'] or any('host_' in node for node in row['path']):
                    raise AssertionError('native quantized path not selected')
                actual = torch.tensor(row['values'], dtype=torch.float32).reshape(wanted.shape)
                error = (actual-wanted).abs().max().item()
                if not torch.isfinite(actual).all() or error > tolerance:
                    raise AssertionError(f'maximum error {error} exceeds {tolerance}')
                check.update(status='passed', maximum_error=error)
            except AssertionError as error:
                check.update(status='failed', error=str(error))
            checks.append(check)
    report = {'suite': 'cuda_quantized_general_checks', 'seed': args.seed,
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'inputs_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(), 'pytorch_version': torch.__version__,
              'reference': 'PyTorch CPU float64 explicit bit unpack, affine decode and matmul after source dtype quantization; f32 outputs',
              'tolerance': '2e-6 * max(1, peak absolute reference output), applied to every element',
              'checks': checks, 'passed': sum(c['status'] == 'passed' for c in checks), 'total': len(checks)}
    for name in ('inputs.json', 'outputs.json'):
        artifact = output / name
        with gzip.open(str(artifact) + '.gz', 'wb', compresslevel=6) as compressed:
            compressed.write(artifact.read_bytes())
        artifact.unlink()
    report.update(inputs_artifact='inputs.json.gz', outputs_artifact='outputs.json.gz', artifact_encoding='gzip; inputs_sha256 hashes uncompressed JSON bytes')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('seed', 'passed', 'total')}))
    if report['passed'] != report['total']: sys.exit(1)


if __name__ == '__main__':
    main()

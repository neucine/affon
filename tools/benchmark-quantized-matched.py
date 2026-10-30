#!/usr/bin/env python3
"""Alternating matched M128 GPU comparisons; resident full QMM on both engines."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, default=root/'zig-out/bin/affon')
    p.add_argument('--mlx-python', type=Path, default=Path('/tmp/affon-mlx-bench/bin/python'))
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--warmup-ms', type=int, default=2000)
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.rounds < 3 or args.samples < 100 or args.warmup_ms < 300:
        p.error('Use at least 3 rounds, 100 samples and 300 ms warmup')
    environment = json.loads(subprocess.check_output([str(args.mlx_python.absolute()), '-c',
        'import json,mlx.core as mx,numpy as np; print(json.dumps(dict(mlx_version=mx.__version__,numpy_version=np.__version__,device=mx.device_info())))'], text=True))
    native = root.parent/'compute/src/backend/metal'
    environment['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [native/'quantized_source.h', native/'device.zig', native/'ffi.m']}
    runs = []
    engines = dict(before=('affon', args.before), after=('affon', args.after), mlx=('mlx', args.mlx_python))
    with tempfile.TemporaryDirectory(prefix='qmm-matched-') as folder:
        for round_index in range(args.rounds):
            order = ['before', 'after', 'mlx'] if round_index % 2 == 0 else ['mlx', 'after', 'before']
            # Rotate the leading engine too, avoiding a consistent position bias.
            shift = (round_index // 2) % 3
            order = order[shift:] + order[:shift]
            for label in order:
                engine, executable = engines[label]
                output = Path(folder)/f'{round_index}-{label}.json'
                subprocess.run(['python3', str(root/'tools/profile-quantized-timestamps.py'),
                    '--engine', engine, '--executable', str(executable.absolute()), '--output', str(output),
                    '--m128-only', '--warmup-ms', str(args.warmup_ms), '--samples', str(args.samples)], check=True)
                runs.append(dict(round=round_index, label=label, profile=json.loads(output.read_text())))
    summary = {}
    for label in engines:
        medians = [r['profile']['results'][0]['median_gpu_ms'] for r in runs if r['label'] == label]
        median = statistics.median(medians)
        summary[label] = dict(round_medians_ms=medians, median_gpu_ms=median,
            round_median_range_ms=[min(medians), max(medians)],
            relative_mad=statistics.median(abs(t-median) for t in medians)/median,
            binary_sha256=hashlib.sha256(engines[label][1].read_bytes()).hexdigest())
    report = dict(shape=[128,2048,1024], bits=8, group_size=64, input_dtype='f32', parameter_dtype='bf16',
        platform=platform.platform(), environment=environment, rounds=args.rounds, warmup_ms=args.warmup_ms, samples_per_round=args.samples,
        method='Sequential alternating/rotating processes; same Metal command-buffer timestamp interposer; full QMM; no host readback included; per-round medians avoid pooling away drift.',
        summary=summary, runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

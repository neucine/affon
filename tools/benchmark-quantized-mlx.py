#!/usr/bin/env python3
"""Compare resident Affon/MLX affine qmm; synchronized nested-list readback included."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--affon', type=Path, default=root / 'zig-out/bin/affon')
    parser.add_argument('--affon-telemetry',choices=['on','off'],default='off')
    parser.add_argument('--output', type=Path, default=root / 'test/benchmarks/results/quantized-metal-profiled-vs-mlx-m4.json')
    args = parser.parse_args()
    proc = subprocess.run([str(args.affon.resolve()), str(root / 'test/benchmarks/quantized/quantized-metal.ts')],
                          text=True, capture_output=True, check=True,env=dict(os.environ,AFFON_QMM_TELEMETRY=args.affon_telemetry))
    affon = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
    assert len(affon) == 8, proc.stdout + proc.stderr
    # Import/allocate MLX only after the Affon process exits; do not run both engines concurrently.
    import mlx.core as mx
    import numpy as np
    results = []
    for case in affon:
        m, n, k, bits, gs = (case[key] for key in ['m', 'n', 'k', 'bits', 'group_size'])
        per_word, mask = 32 // bits, (1 << bits) - 1
        r = np.arange(n, dtype=np.uint32)[:, None]
        col = np.arange(k // per_word, dtype=np.uint32)[None, :] * per_word
        packed = np.zeros((n, k // per_word), np.uint32)
        for lane in range(per_word):
            packed |= ((r * 3 + (col + lane) * 7) & mask) << (lane * bits)
        values = ((np.arange(m)[:, None] + np.arange(k)[None, :]) % 17 - 8).astype(np.float32) / 16
        x, w = mx.array(values), mx.array(packed)
        s = mx.full((n, k // gs), 1 / 128, dtype=mx.bfloat16)
        b = mx.full((n, k // gs), -1, dtype=mx.bfloat16)
        mx.eval(x, w, s, b)
        def eager(a):
            return mx.quantized_matmul(a, w, s, b, transpose=True, bits=bits, group_size=gs)
        compiled = mx.compile(eager)
        mlx = {}
        for name, fn in [('eager', eager), ('compiled', compiled), ('eager_completion', eager), ('compiled_completion', compiled)]:
            scalar = name.endswith('_completion')
            check_output = fn(x)
            mx.eval(check_output)
            checked = check_output[:1, :1].tolist() if scalar else check_output.tolist()
            error = 0.0
            for row in ([0] if scalar else [0, m - 1]):
                for output in ([0] if scalar else [0, 17, n - 1]):
                    expected = sum(float(values[row, i]) * (((output * 3 + i * 7) & mask) / 128 - 1) for i in range(k))
                    error = max(error, abs(checked[row][output] - expected))
            assert error <= 1e-4, (name, case, error)
            def consume():
                y = fn(x)
                if scalar:
                    # Materialize the complete QMM before slicing. Putting the
                    # slice inside mx.compile permits work/shape reduction.
                    mx.eval(y)
                    return y[:1, :1].item()
                return y.tolist()
            iterations = case['completion_iterations'] if scalar else case['iterations']
            warm_end=time.perf_counter()+.3
            while time.perf_counter()<warm_end:
                consume()
            samples = []
            for _ in range(5):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    consume()
                samples.append((time.perf_counter_ns() - start) / iterations / 1e6)
            mlx[name] = dict(samples_ms=samples, median_ms=statistics.median(samples), max_error=error)
        case['median_ms'] = statistics.median(case['samples_ms'])
        case['gpu_median_ms'] = statistics.median(case['gpu_samples_ms']) if case['gpu_samples_ms'] else None
        case['completion_median_ms'] = statistics.median(case['completion_samples_ms'])
        results.append(dict(affon=case, mlx=mlx))
        print(f"bits={bits} M={m:3} Affon={case['median_ms']:.3f}ms "
              f"MLX compiled={mlx['compiled']['median_ms']:.3f}ms | completion Affon={case['completion_median_ms']:.3f}ms MLX={mlx['compiled_completion']['median_ms']:.3f}ms", flush=True)
        del compiled, x, w, s, b
        mx.clear_cache()
    report = dict(date=datetime.now(ZoneInfo('Pacific/Auckland')).isoformat(), hardware='Apple M4',
                  affon_base_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                  compute_base_commit=subprocess.check_output(['git','-C',str(root.parent/'compute'),'rev-parse','HEAD'],text=True).strip(),
                  source_state='uncommitted profiled native quantized_matmul and readback changes accompanying this report', build='Zig 0.16.0 ReleaseFast', os=platform.platform(), python=platform.python_version(), mlx=mx.__version__,
                  method=dict(warmup_ms=300, samples=5, iterations=30, resident_weights=True,
                              input_dtype='f32', parameter_dtype='bf16', output_dtype='f32',
                              timed_scope='call + output allocation + synchronization + full nested-list readback',
                              affon_timer='Date.now; 0.0333ms resolution per reported iteration',
                              mlx_timer='perf_counter_ns', affon_gpu_timer='Metal command buffer timestamps; diagnostic only',
                              mlx_variants=['eager quantized_matmul', 'mx.compile quantized_matmul'], affon_backend_timing_enabled=args.affon_telemetry=='on',
                              completion_scope='full QMM evaluated before scalar view/read; MLX slice is outside mx.compile to prevent reduced computation', completion_iterations=100, construction_and_compilation_excluded=True), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export target-only GPU compute-encoder durations from Instruments Metal System Trace.
Raw traces (which contain system metadata) stay in a temporary directory; save only
aggregate QMM durations. Run engines serially, separately from wall-time benchmarks.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['affon', 'mlx'], required=True)
    p.add_argument('--executable', type=Path, required=True, help='Affon executable or Python with MLX installed')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mlx-slice-inside', action='store_true', help='Diagnostic: reproduce the old compiled scalar-slice benchmark')
    args = p.parse_args()
    script = root / ('test/benchmarks/trace-quantized-metal.ts' if args.engine == 'affon' else 'tools/trace-mlx-quantized.py')
    with tempfile.TemporaryDirectory(prefix='affon-gpu-profile-') as folder:
        directory = Path(folder)
        trace, toc, data = (directory / name for name in ['capture.trace', 'toc.xml', 'gpu.xml'])
        subprocess.run(['xcrun', 'xctrace', 'record', '--template', 'Metal System Trace', '--output', str(trace),
                        '--launch', '--', str(args.executable.absolute()), str(script), *(['--slice-inside'] if args.mlx_slice_inside else [])], check=True)
        subprocess.run(['xcrun', 'xctrace', 'export', '--input', str(trace), '--toc', '--output', str(toc)], check=True)
        process = ET.parse(toc).find('.//target/process')
        if process is None or process.get('return-exit-status') != '0':
            raise RuntimeError('profile target failed: ' + ET.tostring(process, encoding='unicode'))
        pid = process.get('pid')
        subprocess.run(['xcrun', 'xctrace', 'export', '--input', str(trace), '--xpath',
                        '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]', '--output', str(data)], check=True)
        document = ET.parse(data).getroot()
        ids = {node.get('id'): node for node in document.iter() if node.get('id')}
        def resolve(node):
            return ids[node.get('ref')] if node.get('ref') else node
        schema = document.find('.//schema')
        columns = {col.findtext('mnemonic'): i for i, col in enumerate(schema.findall('col'))}
        encoders = {}
        for row in document.findall('.//row'):
            cells = list(row)
            def cell(name):
                return resolve(cells[columns[name]])
            if f'({pid})' not in cell('process').get('fmt', '') or cell('channel-name').text != 'Compute':
                continue
            if cell('event-depth').text != '0':
                continue
            key = (cell('cmdbuffer-id').text, cell('encoder-id').text)
            start, duration = int(cell('start').text), int(cell('duration').text)
            latency = int(cell('start-latency').text or '0')
            entry = encoders.setdefault(key, dict(start_ns=start, duration_ns=0, submission_to_gpu_ns=latency))
            entry['start_ns'] = min(start, entry['start_ns'])
            entry['duration_ns'] += duration  # One encoder may be split by GPU preemption.
        groups = []
        for event in sorted(encoders.values(), key=lambda e: e['start_ns']):
            if not groups or event['start_ns'] - groups[-1][-1]['start_ns'] > 80_000_000:
                groups.append([])
            groups[-1].append(event)
        groups = [group for group in groups if len(group) >= 100]
        if len(groups) != 4:
            raise RuntimeError(f'Expected four workloads; observed encoder group sizes {[len(g) for g in groups]}')
        results = []
        for m, group in zip([1, 8, 32, 128], groups):
            samples = group[-100:]  # Drop ten warmups and any setup encoders.
            times = [e['duration_ns'] / 1e6 for e in samples]
            results.append(dict(m=m, n=2048, k=1024, bits=8, group_size=64, samples_ms=times,
                                median_gpu_ms=statistics.median(times),
                                median_submission_to_gpu_ms=statistics.median(e['submission_to_gpu_ns'] / 1e6 for e in samples),
                                observed_encoder_count=len(group)))
    report = dict(engine=args.engine, slice_inside_compile=args.mlx_slice_inside, scope='GPU compute-encoder active intervals from Metal System Trace; excludes CPU encoding, queue wait, synchronization and readback',
                  method='4 workloads, 300ms warmup + 100 measured synchronized QMM calls; target PID filtered; split intervals summed by encoder ID', results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for result in results:
        print(f"{args.engine} M={result['m']} GPU compute encoder {result['median_gpu_ms']:.6f} ms")


if __name__ == '__main__':
    main()

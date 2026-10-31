"""Run fresh native Affon benchmark processes sequentially; Python is orchestration only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--affon', required=True)
parser.add_argument('--cache', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--build-profile', default='unspecified')
parser.add_argument('--families', nargs='+', choices=['gpt2', 'vit'], default=['gpt2', 'vit'])
parser.add_argument('--devices', nargs='+', choices=['cpu', 'metal', 'cuda'], default=['cpu', 'metal'])
parser.add_argument('--target-fingerprint', help='Exact machine identity for normalized performance evidence')
parser.add_argument('--pytorch-python', help='Python with PyTorch and Transformers; enables the matched peer gate')
parser.add_argument('--maximum-peer-ratio', type=float, default=1.25)
parser.add_argument('--baseline-normalized-affon', help='Accepted same-target normalized Affon report')
parser.add_argument('--maximum-self-ratio', type=float, default=1.10)
parser.add_argument('--absolute-noise-ms', type=float, default=0.0)
args = parser.parse_args()
if not 3 <= args.iterations <= 100:
    parser.error('Use 3–100 measured iterations')
if (args.pytorch_python or args.baseline_normalized_affon) and not args.target_fingerprint:
    parser.error('comparison gates require --target-fingerprint')
if args.maximum_peer_ratio <= 0 or args.maximum_self_ratio <= 0 or args.absolute_noise_ms < 0:
    parser.error('comparison limits must be nonnegative; ratios must be positive')
repo = Path(__file__).resolve().parents[3]
output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
executable = Path(args.affon).resolve()
metadata = {'platform': platform.platform(), 'machine': platform.machine(), 'build_profile': args.build_profile,
            'affon_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(), 'iterations': args.iterations,
            'execution': 'Fresh processes, sequential; one warmup then measured runs; existing playground left running; other system activity uncontrolled.',
            'families': args.families, 'devices': args.devices,
            'gpt2_source_sha256': hashlib.sha256((repo / 'packages/@affon/huggingface/src/adapters/gpt2.ts').read_bytes()).hexdigest(), 'runs': []}
if platform.system() == 'Darwin':
    metadata['cpu'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    metadata['system_memory_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
for family in args.families:
    for device in args.devices:
        name = f'{family}-{device}'
        report = output / f'{name}.json'
        env = {**os.environ, 'AFFON_DEVICE': device, 'AFFON_BENCH_FAMILY': family, 'AFFON_BENCH_REPORT': str(report),
               'AFFON_HF_CACHE': str(Path(args.cache).resolve()), 'AFFON_BENCH_ITERATIONS': str(args.iterations), 'AFFON_AUDIT_BUILD': args.build_profile}
        command = [str(executable), 'apps/hf-inference/src/benchmark/inference.ts']
        if platform.system() == 'Darwin': command = ['/usr/bin/time', '-l', *command]
        print(f'Running {name}', flush=True)
        start = time.monotonic()
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
        (output / f'{name}.log').write_text(result.stdout + result.stderr)
        if result.returncode: raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
        peak = re.search(r'(\d+)\s+maximum resident set size', result.stderr)
        metadata['runs'].append({'name': name, 'report': report.name, 'wall_seconds': time.monotonic() - start,
                                 'os_peak_rss_bytes': int(peak.group(1)) if peak else None})
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Finished {name}', flush=True)

subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(output)], check=True)
if args.target_fingerprint:
    normalized = output / 'normalized'
    subprocess.run([sys.executable, str(repo / 'tools/normalize-model-performance.py'),
                    '--source', 'hf', '--input', str(output), '--target', args.target_fingerprint,
                    '--output-dir', str(normalized)], check=True)
    if args.baseline_normalized_affon:
        subprocess.run([sys.executable, str(repo / 'tools/compare-model-performance.py'),
                        '--kind', 'self', '--current', str(normalized / 'affon.json'),
                        '--reference', str(Path(args.baseline_normalized_affon).resolve()),
                        '--maximum-ratio', str(args.maximum_self_ratio),
                        '--absolute-noise-ms', str(args.absolute_noise_ms),
                        '--output', str(normalized / 'self-gate.json')], check=True)
    if args.pytorch_python:
        subprocess.run([args.pytorch_python, str(Path(__file__).with_name('compare-pytorch-models.py')),
                        '--affon-run', str(output), '--normalized-affon', str(normalized / 'affon.json'),
                        '--cache', str(Path(args.cache).resolve()), '--output', str(normalized / 'pytorch.json')], check=True)
        subprocess.run([sys.executable, str(repo / 'tools/compare-model-performance.py'),
                        '--kind', 'peer', '--current', str(normalized / 'affon.json'),
                        '--reference', str(normalized / 'pytorch.json'),
                        '--maximum-ratio', str(args.maximum_peer_ratio),
                        '--output', str(normalized / 'peer-gate.json')], check=True)

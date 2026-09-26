"""Run or summarize the fixed-length native GPT-2 cache scaling benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--affon')
parser.add_argument('--cache')
parser.add_argument('--output', required=True)
parser.add_argument('--summarize-only', action='store_true')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
if not args.summarize_only:
    if not args.affon or not args.cache:
        parser.error('--affon and --cache are required to run')
    binary = Path(args.affon).resolve()
    meta = {'platform': platform.platform(), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'gpt2_source_sha256': hashlib.sha256((repo / 'packages/@affon/huggingface/src/gpt2.ts').read_bytes()).hexdigest(), 'runs': []}
    for device in ['cpu', 'metal']:
        print('Starting', device, flush=True)
        start = time.monotonic()
        command = [str(binary), 'apps/hf-inference/src/benchmark/cache-scaling.ts']
        if platform.system() == 'Darwin': command = ['/usr/bin/time', '-l', *command]
        with (out / f'{device}.log').open('w') as log:
            result = subprocess.run(command, cwd=repo, env={**os.environ, 'AFFON_DEVICE': device,
                'AFFON_HF_CACHE': str(Path(args.cache).resolve()), 'AFFON_BENCH_REPORT': str(out / f'{device}.json')}, stdout=log, stderr=log)
        if result.returncode: raise RuntimeError(f'Benchmark failed: {out / (device + ".log")}')
        peak = re.search(r'(\d+)\s+maximum resident set size', (out / f'{device}.log').read_text())
        meta['runs'].append({'device': device, 'seconds': time.monotonic() - start,
                            'os_peak_rss_bytes': int(peak[1]) if peak else None})
        (out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n')

reports = {device: json.loads((out / f'{device}.json').read_text()) for device in ['cpu', 'metal']}
lines = ['# GPT-2 KV-cache scaling', '', 'Median of three measured runs after one warmup per scenario. Batch one, f32.', '',
         '| Prompt / new tokens | Backend | Prefill + first token | Remaining decode | Total | Cache at final step |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
for cpu, metal in zip(reports['cpu']['results'], reports['metal']['results'], strict=True):
    assert (cpu['prompt_tokens'], cpu['new_tokens']) == (metal['prompt_tokens'], metal['new_tokens'])
    assert cpu['samples'][0]['generated'] == metal['samples'][0]['generated'], 'CPU/Metal greedy tokens differ'
    for device, result in [('CPU', cpu), ('Metal', metal)]:
        samples = result['samples']
        med = lambda key: statistics.median(s[key] for s in samples)
        lines.append(f"| {result['prompt_tokens']} / {result['new_tokens']} | {device} | {med('prefill_ms') / 1000:.3f} s | {med('decode_ms') / 1000:.3f} s | {med('total_ms') / 1000:.3f} s | {med('retained_cache_bytes') / 2**20:.2f} MiB |")
lines += ['', 'All generated token IDs agree across CPU and Metal for these scenarios. Every run checks exact retained cache size and release on reset.', '',
          'Cache size is 36 KiB per processed token. The final predicted token has not been fed back, so the retained length is prompt + new − 1.', '',
          'EOS is deliberately ignored to measure fixed lengths. Inputs repeat a tokenized prose seed; these are systems measurements, not quality tests. Readback and greedy token selection are included. Memory sampling between phases is excluded. Scenarios share one model per backend; process peak RSS includes loading and all scenarios. Background activity is uncontrolled; three samples do not establish tail latency or leak freedom.', '',
          'Raw data: [CPU](cpu.json), [Metal](metal.json), [host/run metadata](run.json).']
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

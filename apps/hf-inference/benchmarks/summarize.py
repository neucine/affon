"""Summarize completed benchmark reports without discarding raw samples."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory')
args = parser.parse_args()
root = Path(args.directory)
run = json.loads((root / 'run.json').read_text())
if len(run['runs']) != len(run.get('families', ['gpt2', 'vit'])) * 2:
    raise ValueError('Expected all requested benchmark processes to complete')
rows = []
for entry in run['runs']:
    report = json.loads((root / entry['report']).read_text())
    metric = 'generate_ms' if report['family'] == 'gpt2' else 'forward_with_readback_ms'
    if len(report['metrics'][metric]['samples']) != report['measured_runs']:
        raise ValueError('Incomplete measurement samples')
    steady = [sample['values'] for sample in report['memory_samples'] if sample['phase'] == 'after_warmup' or sample['phase'].startswith('after_iteration_')]
    live = [sample['compute.storage.live_bytes'] for sample in steady]
    rows.append({'name': entry['name'], 'report': entry['report'], 'median_ms': report['metrics'][metric]['median'],
                 'model_load_ms': report['model_load_ms'], 'cache_validation_ms': report['cache_validation_ms'],
                 'os_peak_rss_bytes': entry['os_peak_rss_bytes'], 'live_tensor_bytes_min': min(live), 'live_tensor_bytes_max': max(live)})
lines = [f"# Local {run.get('cpu', run['machine'])} inference baseline", '',
         f"Build: {run['build_profile']}. One warmup, {run['iterations']} measured runs per process. Batch one; f32.", '',
         '| Workload / backend | Median inference | Model construction | OS peak RSS | Live tensor range after warmup |',
         '| --- | ---: | ---: | ---: | ---: |']
for row in rows:
    rss = f"{row['os_peak_rss_bytes'] / 2**20:.1f} MiB" if row['os_peak_rss_bytes'] is not None else 'unavailable'
    lines.append(f"| [{row['name']}]({row['report']}) | {row['median_ms'] / 1000:.3f} s | {row['model_load_ms']} ms | {rss} | {row['live_tensor_bytes_min'] / 2**20:.1f}–{row['live_tensor_bytes_max'] / 2**20:.1f} MiB |")
lines.extend(['', "GPT-2: five prompt tokens, 16 new greedy tokens, explicit full-prefix Programs with one-token output windows. ViT (when measured): 320×256 synthetic RGB input; forward includes logit readback and excludes preprocessing.", '',
              'OS peak RSS covers the entire process, including startup/loading. Construction excludes snapshot validation; neither is guaranteed cold-disk timing. Memory ranges are six samples for the default five-run benchmark, not a leak-freedom claim.', '',
              'See [methodology](../../README.md) and [host/run metadata](run.json). Raw reports include preprocessing, warmup, all samples, and memory observations.', ''])
(root / 'summary.md').write_text('\n'.join(lines))
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('\n'.join(lines))

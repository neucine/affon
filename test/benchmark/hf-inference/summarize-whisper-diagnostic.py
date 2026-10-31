"""Validate phase telemetry and summarize saved Whisper diagnostic/control runs."""
import argparse
import json
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
root = args.directory
summary = {}
for clip in ('short', 'long'):
    diagnostic = json.loads((root / f'{clip}-diagnostic.json').read_text())
    controls = {mode: json.loads((root / f'{clip}-control-{mode}.json').read_text())
                for mode in ('off', 'on')}
    reference = controls['off']['runs'][0]
    results = [run['result'] for run in diagnostic['runs']]
    results += [run for control in controls.values() for run in control['runs']]
    for result in results:
        assert result['tokens'] == reference['tokens'], (clip, 'tokens')
        assert result['text'] == reference['text'], (clip, 'text')
        assert not result['truncated'], (clip, 'truncated')
    phases = {}
    for name in diagnostic['runs'][0]['phases']:
        samples = []
        for run in diagnostic['runs']:
            phase = run['phases'][name]
            counter = phase['counters']
            def metric(key):
                return counter.get('compute.execution/metal_command_' + key, 0)
            assert metric('count') == metric('gpu_valid_count'), (clip, name, 'timestamps')
            assert metric('submit_ns') + metric('wait_ns') == metric('wall_ns'), (clip, name, 'accounting')
            sample = dict(phase_ms=phase['elapsed_ms'], calls=phase['calls'], commands=metric('count'))
            sample.update({key + '_ms': metric(key + '_ns') / 1e6
                           for key in ('prepare', 'submit', 'wait', 'wall', 'gpu')})
            # Residual is calculated per run, before taking medians. Not a causal attribution.
            sample['outside_command_ms'] = sample['phase_ms'] - sample['prepare_ms'] - sample['wall_ms']
            sample['commands_per_call'] = sample['commands'] / sample['calls']
            sample['allocations'] = counter.get('compute.storage/allocation_count', 0)
            samples.append(sample)
        phases[name] = {key: median(sample[key] for sample in samples) for key in samples[0]}
    summary[clip] = dict(phases=phases, controls={
        mode: {key: median(run[key] for run in control['runs'])
               for key in ('preprocessing_ms', 'encoder_ms', 'decoder_ms', 'inference_ms', 'elapsed_ms')}
        for mode, control in controls.items()
    }, diagnostic_inference_ms=median(run['result']['inference_ms'] for run in diagnostic['runs']))
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

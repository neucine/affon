"""Compare native CPU/Metal real-image results with multiple PyTorch paths."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file
from vit_metrics import compare, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = json.loads((root / 'reference.json').read_text())
    for name, expected in manifest['sha256'].items():
        with (root / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Artifact changed: {name}')
    references = {mode: load_file(str(root / ('reference.safetensors' if mode == 'cpu-eager' else f'reference-{mode}.safetensors')))
                  for mode in manifest['reference_modes']}
    native = {device: load_file(str(root / f'native-hidden-{device}.safetensors')) for device in ['cpu', 'metal']}
    layers = json.loads((root / 'config.json').read_text())['num_hidden_layers'] + 1
    cases = []
    for index, sample in enumerate(manifest['cases']):
        prefix = f'case_{index}'
        comparisons = []
        for device, tensors in native.items():
            preprocessing = compare(tensors[f'{prefix}.pixels'], references['cpu-eager'][f'{prefix}.pixels'])
            preprocessing['exact'] = tensors[f'{prefix}.pixels'].equal(references['cpu-eager'][f'{prefix}.pixels'])
            for mode, expected in references.items():
                metrics = summarize([tensors[f'{prefix}.hidden_{i}'] for i in range(layers)], tensors[f'{prefix}.output'], expected, prefix)
                comparisons.append({'actual': f'affon-{device}', 'reference': f'torch-{mode}', 'preprocessing': preprocessing, **metrics})
        for mode, tensors in references.items():
            if mode == 'cpu-eager':
                continue
            comparisons.append({'actual': f'torch-{mode}', 'reference': 'torch-cpu-eager',
                                **summarize([tensors[f'{prefix}.hidden_{i}'] for i in range(layers)], tensors[f'{prefix}.output'], references['cpu-eager'], prefix)})
        cases.append({**sample, 'comparisons': comparisons})
    report = {'format': 'affon-vit-real-image-calibration/v1', 'model_id': manifest['model_id'],
              'revision': manifest['revision'], 'versions': manifest['versions'],
              'atol': 1e-4, 'rtol': 1e-4, 'scope': manifest['scope'], 'cases': cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for case in cases:
        for result in case['comparisons']:
            print(case['name'], result['actual'], 'vs', result['reference'],
                  'first_bad_hidden=', result['first_failing_hidden_state'],
                  'logits_pass=', result['logits']['passed'], 'top1=', result['top1_matches'],
                  'pixels_exact=', result.get('preprocessing', {}).get('exact', 'n/a'))


if __name__ == '__main__':
    main()

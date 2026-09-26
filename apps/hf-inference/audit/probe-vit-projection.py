"""Calibrate ViT hidden-state sensitivity to equivalent patch projections.

No model implementation or audit tolerance is changed. Every intervention
replaces only the initial embedding; the remaining forward pass is PyTorch.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import ViTForImageClassification


from vit_metrics import compare, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'metal'], default=['cpu', 'metal'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = json.loads((root / 'reference.json').read_text())
    for name in ['model.safetensors', 'config.json']:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != manifest['sha256'][name]:
            raise ValueError(f'Reference artifact hash mismatch: {name}')
    torch.set_num_threads(4)
    reference = load_file(str(root / 'reference.safetensors'))
    native = {device: load_file(str(root / f'native-hidden-{device}.safetensors')) for device in args.devices}
    model = ViTForImageClassification.from_pretrained(root, attn_implementation='eager').cpu().eval()
    rows = []
    with torch.inference_mode():
        for prefix in sorted(key[:-7] for key in reference if key.endswith('.pixels')):
            pixels = reference[f'{prefix}.pixels']
            baseline = model(pixels, output_hidden_states=True)
            baseline_metrics = summarize(baseline.hidden_states, baseline.logits, reference, prefix)
            if any(not row['passed'] for row in baseline_metrics['hidden_states']) or not baseline_metrics['logits']['passed']:
                raise ValueError('Current PyTorch baseline does not reproduce the saved reference')
            projection = model.vit.embeddings.patch_embeddings.projection
            patches = torch.nn.functional.unfold(pixels, projection.kernel_size, stride=projection.stride).transpose(1, 2)
            variants = {
                'torch_linear_f32': torch.nn.functional.linear(patches, projection.weight.flatten(1), projection.bias),
                'torch_projection_f64_then_f32': (patches.double() @ projection.weight.flatten(1).double().T + projection.bias.double()).float(),
                'torch_conv_channels_last_f32': torch.nn.functional.conv2d(
                    pixels.to(memory_format=torch.channels_last), projection.weight.to(memory_format=torch.channels_last),
                    projection.bias, stride=projection.stride).flatten(2).transpose(1, 2),
            }
            embeddings = {name: torch.cat([model.vit.embeddings.cls_token, value], dim=1) + model.vit.embeddings.position_embeddings
                          for name, value in variants.items()}
            embeddings.update({f'native_{device}_embedding': tensors[f'{prefix}.hidden_0'] for device, tensors in native.items()})
            interventions = {}
            for name, embedding in embeddings.items():
                handle = model.vit.embeddings.register_forward_hook(lambda module, inputs, output, embedding=embedding: embedding)
                try:
                    result = model(pixels, output_hidden_states=True)
                finally:
                    handle.remove()
                interventions[name] = summarize(result.hidden_states, result.logits, reference, prefix)
                if name.startswith('native_'):
                    device = name[len('native_'):-len('_embedding')]
                    interventions[name]['native_vs_matched_embedding_torch'] = [
                        compare(native[device][f'{prefix}.hidden_{i}'], value)
                        for i, value in enumerate(result.hidden_states)]
            block_checks = {}
            for device, tensors in native.items():
                checks = []
                for index, block in enumerate(model.vit.encoder.layer):
                    output = block(tensors[f'{prefix}.hidden_{index}'])
                    if isinstance(output, tuple):
                        output = output[0]
                    if index == len(model.vit.encoder.layer) - 1:
                        output = model.vit.layernorm(output)
                    checks.append(compare(tensors[f'{prefix}.hidden_{index + 1}'], output))
                block_checks[device] = checks
            rows.append({
                'native_input_block_checks': block_checks,
                'case': prefix,
                'baseline': baseline_metrics,
                'native': {device: summarize([tensors[f'{prefix}.hidden_{i}'] for i in range(len(baseline.hidden_states))],
                                             tensors[f'{prefix}.output'], reference, prefix)
                           for device, tensors in native.items()},
                'interventions': interventions,
            })
    report = {
        'format': 'affon-vit-projection-calibration/v1',
        'model_id': manifest['model_id'], 'revision': manifest['revision'],
        'artifact_sha256': manifest['sha256'],
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'safetensors']},
        'torch_threads': 4, 'torch_device': 'cpu', 'attention': 'eager',
        'atol': 1e-4, 'rtol': 1e-4,
        'scope': 'Two existing synthetic RGB cases; equivalent patch-projection interventions, not a production compatibility pass.',
        'results': rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in rows:
        for name, metrics in {**row['native'], **row['interventions']}.items():
            print(row['case'], name, 'first_failed_hidden=', metrics['first_failing_hidden_state'],
                  'final_cls_mismatches=', metrics['final_cls_token']['mismatches'],
                  'final_patch_mismatches=', metrics['final_patch_tokens']['mismatches'],
                  'logits_pass=', metrics['logits']['passed'])


if __name__ == '__main__':
    main()

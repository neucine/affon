"""Prepare real-image ViT calibration using local weights and public sample images.

Downloads are reference inputs only. Native inference does not invoke Python.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file
from transformers import ViTForImageClassification, ViTImageProcessor

SOURCES = [
    ('parrots', 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png'),
    ('cats', 'https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039769.jpg'),
    ('dog', 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-manifest', help='Verify image hashes against a previous manifest')
    args = parser.parse_args()
    source, output = Path(args.model_directory), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / 'reference.json').read_text())
    for name in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
        if digest(source / name) != original['sha256'][name]:
            raise ValueError(f'Unexpected model artifact: {name}')
        target = output / name
        if not target.exists():
            os.link(source / name, target)
        if digest(target) != original['sha256'][name]:
            raise ValueError(f'Output contains a different model artifact: {name}')
    expected = json.loads(Path(args.expected_manifest).read_text()) if args.expected_manifest else None
    processor = ViTImageProcessor.from_pretrained(source, local_files_only=True)
    inputs, cases = {}, []
    for index, (name, url) in enumerate(SOURCES):
        path = output / f'{name}.image'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError('Sample image exceeds download limit')
            path.write_bytes(data)
        sha = digest(path)
        if expected and sha != expected['cases'][index]['image_sha256']:
            raise ValueError(f'Sample image changed: {name}')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            pixels = processor(images=rgb, return_tensors='pt')['pixel_values']
            array = np.array(rgb, dtype=np.uint8)
        inputs[f'case_{index}.pixels'] = pixels.contiguous()
        # f32 integers preserve RGB8 exactly and can be loaded by Affon's checkpoint API.
        inputs[f'case_{index}.rgb'] = torch.from_numpy(array.copy()).float()
        cases.append({'name': name, 'source_url': url, 'image_sha256': sha,
                      'height': array.shape[0], 'width': array.shape[1], 'input_kind': 'reference_rgb'})
    torch.set_num_threads(4)
    modes = [('cpu-eager', 'cpu', 'eager'), ('cpu-sdpa', 'cpu', 'sdpa')]
    if torch.backends.mps.is_available():
        modes.append(('mps-eager', 'mps', 'eager'))
    for mode, device, attention in modes:
        model = ViTForImageClassification.from_pretrained(source, local_files_only=True, attn_implementation=attention).to(device).eval()
        tensors = dict(inputs) if mode == 'cpu-eager' else {}
        with torch.inference_mode():
            for index, case in enumerate(cases):
                result = model(inputs[f'case_{index}.pixels'].to(device), output_hidden_states=True)
                tensors[f'case_{index}.output'] = result.logits.cpu().contiguous()
                for layer, hidden in enumerate(result.hidden_states):
                    tensors[f'case_{index}.hidden_{layer}'] = hidden.cpu().contiguous()
                if mode == 'cpu-eager':
                    case['top1'] = result.logits.argmax(-1).tolist()
                    case['top1_label'] = model.config.id2label[case['top1'][0]]
        filename = 'reference.safetensors' if mode == 'cpu-eager' else f'reference-{mode}.safetensors'
        save_file({key: value.clone() for key, value in tensors.items()}, str(output / filename))
        del model, tensors, result
        if device == 'mps':
            torch.mps.empty_cache()
        print('Prepared', mode, flush=True)
    artifact_names = ['model.safetensors', 'config.json', 'preprocessor_config.json', 'reference.safetensors'] + [f'reference-{mode}.safetensors' for mode, _, _ in modes if mode != 'cpu-eager']
    manifest = {
        'format': 'affon-hf-domain-reference/v1', 'family': 'vit',
        'model_id': original['model_id'], 'revision': original['revision'], 'dtype': 'float32',
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers', 'safetensors', 'pillow']},
        'reference_modes': [mode for mode, _, _ in modes], 'torch_threads': 4,
        'scope': 'Three public sample images; decoded RGB, native preprocessing, classification and hidden-state parity. Not a labeled accuracy benchmark.',
        'sha256': {name: digest(output / name) for name in artifact_names},
        'cases': cases,
    }
    (output / 'reference.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

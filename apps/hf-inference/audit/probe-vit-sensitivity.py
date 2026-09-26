"""Measure how native patch-embedding rounding propagates through PyTorch ViT."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import ViTForImageClassification

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', required=True)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
root = Path(args.directory)
torch.set_num_threads(4)
reference = load_file(str(root / 'reference.safetensors'))
native = load_file(str(root / f'native-hidden-{args.device}.safetensors'))
model = ViTForImageClassification.from_pretrained(root, attn_implementation='eager').eval()
handle = model.vit.embeddings.register_forward_hook(lambda module, inputs, output: native['hidden_0'])
with torch.inference_mode():
    result = model(reference['case_0.pixels'], output_hidden_states=True)

def compare(a, b):
    error = (a - b).abs()
    return {'max_absolute_error': error.max().item(), 'rms_error': error.square().mean().sqrt().item(),
            'mismatches': (error > 1e-4 + 1e-4 * b.abs()).sum().item()}

rows = []
for index, hidden in enumerate(result.hidden_states):
    ref = reference[f'case_0.hidden_{index}']
    affon = native[f'hidden_{index}']
    rows.append({'layer': index, 'native_vs_reference': compare(affon, ref),
                 'torch_with_native_embedding_vs_reference': compare(hidden, ref),
                 'native_vs_torch_with_native_embedding': compare(affon, hidden)})
report = {'device': args.device, 'intervention': 'Replace only the PyTorch embedding output with native f32 embedding; all subsequent computation remains PyTorch', 'results': rows}
(root / f'sensitivity-{args.device}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(rows[-1], indent=2))

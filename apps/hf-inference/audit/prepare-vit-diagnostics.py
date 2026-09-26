"""Capture isolated ViT operation inputs/outputs from the prepared local model."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import ViTForImageClassification

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', required=True)
args = parser.parse_args()
root = Path(args.directory)
torch.set_num_threads(4)
model = ViTForImageClassification.from_pretrained(root, attn_implementation='eager').eval()
reference = load_file(str(root / 'reference.safetensors'))
tensors, cases = {}, []
handles = []
for layer in [0, 4, 5, 6, 10]:
    prefix = f'vit.encoder.layer.{layer}'
    for suffix in ['layernorm_before', 'attention.attention', 'attention.output.dense', 'layernorm_after', 'intermediate.dense', 'intermediate.intermediate_act_fn', 'output.dense']:
        name = f'{prefix}.{suffix}'
        module = model.get_submodule(name)
        def capture(module, inputs, output, name=name, suffix=suffix):
            tensors[name + '.input'] = inputs[0].detach().contiguous().clone()
            value = output[0] if isinstance(output, tuple) else output
            tensors[name + '.output'] = value.detach().contiguous().clone()
            cases.append({'name': name, 'kind': 'norm' if 'layernorm' in suffix else 'gelu' if 'act_fn' in suffix else 'attention' if suffix == 'attention.attention' else 'dense', 'eps': getattr(module, 'eps', None), 'heads': model.config.num_attention_heads})
        handles.append(module.register_forward_hook(capture))
with torch.inference_mode():
    model(reference['case_0.pixels'])
save_file(tensors, str(root / 'diagnostics.safetensors'))
(root / 'diagnostics.json').write_text(json.dumps({'cases': cases, 'torch': torch.__version__}, indent=2) + '\n')
print(f'Captured {len(cases)} operations')

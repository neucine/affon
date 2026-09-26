"""Export the already-audited local ViT checkpoint; Python is preparation/oracle only."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file, save_file
from transformers import ViTForImageClassification

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
source, output = Path(a.model_dir), Path(a.output)
output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
model = ViTForImageClassification.from_pretrained(source, local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
# Register the model so the exporter treats weights as parameters.
class ExportModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = model
    def forward(self, pixels):
        return self.model(pixel_values=pixels).logits
reference = load_file(str(source / 'reference.safetensors'))
path = output / 'model.onnx'
with torch.inference_mode():
    torch.onnx.export(ExportModel(), (reference['case_0.pixels'],), str(path),
        input_names=['pixels'], output_names=['logits'], opset_version=17,
        dynamo=False, do_constant_folding=True)
graph = onnx.load(path)
onnx.checker.check_model(graph)
options = ort.SessionOptions(); options.intra_op_num_threads = 4
session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
values, cases = {}, []
for i in range(2):
    pixels = reference[f'case_{i}.pixels']
    actual = session.run(None, {'pixels': pixels.numpy()})[0]
    expected = reference[f'case_{i}.output'].numpy()
    values[f'case_{i}.pixels'] = pixels
    values[f'case_{i}.onnx'] = torch.from_numpy(actual)
    values[f'case_{i}.pytorch'] = torch.from_numpy(expected)
    cases.append({'index': i, 'top1': int(actual.argmax()),
        'pytorch_top1': int(expected.argmax()), 'max_absolute_error': float(np.abs(actual-expected).max()),
        'strict_passed': bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4))})
save_file(values, str(output / 'reference.safetensors'))
source_manifest = json.loads((source/'reference.json').read_text())
report = {'model': source_manifest['model_id'], 'revision': source_manifest['revision'],
    'versions': {name:importlib.metadata.version(name) for name in ['torch','transformers','onnx','onnxruntime']},
    'opset':17, 'input_shape':[1,3,224,224], 'onnx_bytes':path.stat().st_size,
    'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'operators':dict(Counter(node.op_type for node in graph.graph.node)), 'cases':cases}
(output/'export.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

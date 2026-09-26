"""Prepare an independent image-model probe; deployment still uses only Affon."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModelForImageClassification

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
revision=a.revision or HfApi().model_info(a.model).sha
if len(revision)!=40:raise ValueError('Full immutable revision required')
torch.set_num_threads(4)
model=AutoModelForImageClassification.from_pretrained(a.model,revision=revision,trust_remote_code=False,dtype=torch.float32).cpu().eval()
processor=AutoImageProcessor.from_pretrained(a.model,revision=revision,trust_remote_code=False,use_fast=False)
model.save_pretrained(out/'source');processor.save_pretrained(out/'source')
class ExportModel(torch.nn.Module):
    def __init__(self):super().__init__();self.model=model
    def forward(self,pixels):return self.model(pixel_values=pixels).logits
inputs=[]
for height,width in [(224,224),(260,320)]:
    y,x,c=np.indices((height,width,3));rgb=((x*3+y*5+c*47)%256).astype(np.uint8)
    inputs.append(processor(images=Image.fromarray(rgb),return_tensors='pt')['pixel_values'])
with torch.inference_mode():
    torch.onnx.export(ExportModel().eval(),(inputs[0],),str(out/'model.onnx'),opset_version=17,dynamo=False,
        do_constant_folding=True,input_names=['pixels'],output_names=['logits'])
if any(module.training for module in model.modules()):raise ValueError('Export changed model evaluation mode')
graph=onnx.load(out/'model.onnx');onnx.checker.check_model(graph)
options=ort.SessionOptions();options.intra_op_num_threads=4
session=ort.InferenceSession(str(out/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
values={};cases=[]
for i,pixels in enumerate(inputs):
    with torch.inference_mode():expected=model(pixel_values=pixels).logits.contiguous()
    actual=session.run(None,{'pixels':pixels.numpy()})[0]
    values.update({f'case_{i}.pixels':pixels.contiguous(),f'case_{i}.onnx':torch.from_numpy(actual),f'case_{i}.pytorch':expected})
    cases.append({'index':i,'top1':int(actual.argmax()),'pytorch_top1':int(expected.argmax()),
        'max_absolute_error':float(np.abs(actual-expected.numpy()).max()),
        'strict_passed':bool(np.allclose(actual,expected.numpy(),atol=1e-4,rtol=1e-4))})
save_file(values,str(out/'reference.safetensors'))
report={'model':a.model,'revision':revision,'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','onnx','onnxruntime']},
    'opset':17,'input_shape':list(inputs[0].shape),'onnx_bytes':(out/'model.onnx').stat().st_size,
    'onnx_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),
    'operators':dict(Counter(n.op_type for n in graph.graph.node)),'cases':cases}
(out/'export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not all(c['strict_passed'] for c in cases):raise ValueError('Export reference mismatch')

"""Export pinned AST speech-command model and independent waveform/features/logit references."""
import argparse
from collections import Counter
import json
import hashlib
import importlib.metadata
import wave
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from huggingface_hub import HfApi
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', required=True)
p.add_argument('--model', default='MIT/ast-finetuned-speech-commands-v2')
p.add_argument('--revision')
p.add_argument('--wav', help='Optional mono PCM16 16-kHz WAV reference clip')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
revision = a.revision or HfApi().model_info(a.model).sha
torch.set_num_threads(4)
model = AutoModelForAudioClassification.from_pretrained(a.model, revision=revision, trust_remote_code=False, attn_implementation='eager', dtype=torch.float32).cpu().eval()
processor = AutoFeatureExtractor.from_pretrained(a.model, revision=revision, trust_remote_code=False)
model.save_pretrained(out/'source'); processor.save_pretrained(out/'source')
t = np.arange(16000, dtype=np.float64) / 16000
waveforms = [np.zeros(16000, np.float32), (0.3*np.sin(2*np.pi*(220*t+400*t*t)) + 0.05*np.sin(2*np.pi*1700*t)).astype(np.float32)]
if a.wav:
    with wave.open(a.wav) as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=16000: raise ValueError('Reference WAV must be mono PCM16 at 16 kHz')
        waveforms.append(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768)
features = [processor(w, sampling_rate=16000, return_tensors='pt').input_values for w in waveforms]
class Export(torch.nn.Module):
    def __init__(self): super().__init__(); self.model=model
    def forward(self, features): return self.model(input_values=features).logits
with torch.inference_mode():
    torch.onnx.export(Export().eval(), (features[0],), str(out/'model.onnx'), opset_version=17, dynamo=False, do_constant_folding=True, input_names=['features'], output_names=['logits'])
graph = onnx.load(out/'model.onnx'); onnx.checker.check_model(graph)
options=ort.SessionOptions(); options.intra_op_num_threads=4
session=ort.InferenceSession(str(out/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
values={}; cases=[]
for i,(wave,feature) in enumerate(zip(waveforms,features)):
    with torch.inference_mode(): expected=model(input_values=feature).logits.contiguous()
    actual=session.run(None,{'features':feature.numpy()})[0]
    values.update({f'case_{i}.waveform':torch.from_numpy(wave),f'case_{i}.features':feature.contiguous(),f'case_{i}.pytorch':expected,f'case_{i}.onnx':torch.from_numpy(actual)})
    cases.append({'index':i,'top1':int(actual.argmax()),'max_absolute_error':float(np.abs(actual-expected.numpy()).max()),'passed':bool(np.allclose(actual,expected.numpy(),atol=1e-4,rtol=1e-4))})
save_file(values,str(out/'reference.safetensors'))
report={'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','onnx','onnxruntime']},'wav_sha256':hashlib.sha256(Path(a.wav).read_bytes()).hexdigest() if a.wav else None,'model':a.model,'revision':revision,'input_shape':list(features[0].shape),'operators':dict(Counter(n.op_type for n in graph.graph.node)),'cases':cases,'processor_reference':'Transformers ASTFeatureExtractor (NumPy fallback when torchaudio absent)'}
(out/'export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not all(c['passed'] for c in cases):raise ValueError('Export mismatch')

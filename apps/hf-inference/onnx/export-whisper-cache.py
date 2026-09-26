"""Static cache I/O wrappers around HF's Whisper decoder; no replacement attention implementation."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from safetensors.torch import save_file,load_file
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--width',type=int,default=256);a=p.parse_args();out=Path(a.directory)
torch.set_num_threads(4)
m=WhisperForConditionalGeneration.from_pretrained(out/'source',attn_implementation='eager',dtype=torch.float32).eval()
width=a.width
if not 3<=width<=m.config.max_target_positions:raise ValueError('Invalid decoder capacity')
layers=len(m.model.decoder.layers);heads=m.config.decoder_attention_heads;dim=m.config.d_model//heads
positions=m.model.decoder.embed_positions.weight.detach()
class Fixed(DynamicCache):
 def __init__(self,values):super().__init__();self.values=values;self.present=[]
 def update(self,k,v,index,cache_kwargs=None):
  self.present.extend([k,v]);return torch.cat([self.values[index*2],k],2),torch.cat([self.values[index*2+1],v],2)
class CrossCache(DynamicCache):
 def __init__(self,values):
  super().__init__();self.layers=[SimpleNamespace(keys=values[i*2],values=values[i*2+1]) for i in range(layers)]
 def get_seq_length(self,layer_idx=0):return 1500
class Positions(torch.nn.Module):
 def forward(self,*args,**kwargs):return self.value
class Cross(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=m.model.decoder.layers
 def forward(self,encoded):
  outputs=[]
  for layer in self.layers:
   for projection in [layer.encoder_attn.k_proj,layer.encoder_attn.v_proj]:
    outputs.append(projection(encoded).view(1,1500,heads,dim).transpose(1,2).contiguous())
  return tuple(outputs)
class Step(torch.nn.Module):
 def __init__(self):
  super().__init__();self.decoder=m.model.decoder;self.head=m.proj_out;self.decoder.embed_positions=Positions()
 def forward(self,embeddings,position,mask,*values):
  self.decoder.embed_positions.value=position
  self_cache=Fixed(values[:layers*2]);cache=EncoderDecoderCache(self_cache,CrossCache(values[layers*2:]))
  result=self.decoder(inputs_embeds=embeddings,encoder_hidden_states=torch.zeros((1,1,m.config.d_model)),attention_mask=mask,past_key_values=cache,use_cache=True,cache_position=torch.tensor([0])).last_hidden_state
  return (self.head(result),*self_cache.present)
cross=Cross().eval();step=Step().eval()
reference=load_file(out/'reference.safetensors');encoded=reference['encoded']
with torch.inference_mode():
 cross_values=cross(encoded)
 old=[torch.zeros((1,heads,width,dim)) for _ in range(layers*2)]
 embedding=m.model.decoder.embed_tokens(torch.tensor([[50257]]));position=positions[0:1].unsqueeze(0)
 mask=torch.full((1,1,1,width+1),-10000.);mask[:,:,:,-1]=0
 for name,wrapper,args,inputs,outputs in [
  ('cross',cross,(encoded,),['encoded'],[f'cross_{i}' for i in range(layers*2)]),
  ('step',step,(embedding,position,mask,*old,*cross_values),['embeddings','position','mask',*[f'past_{i}' for i in range(layers*2)],*[f'cross_{i}' for i in range(layers*2)]],['logits',*[f'present_{i}' for i in range(layers*2)]])]:
  d=out/name;d.mkdir(exist_ok=True)
  torch.onnx.export(wrapper,args,str(d/'model.onnx'),opset_version=17,dynamo=False,input_names=inputs,output_names=outputs,do_constant_folding=True)
 save_file({'positions':positions.contiguous()},str(out/'positions.safetensors'))
 # Compare every incremental step to the independent full-prefix decoder reference.
 report=json.loads((out/'whisper.json').read_text());cases=[]
 for index,token in enumerate(report['ids'][:-1]):
  mask[:]= -10000.;mask[:,:,:,:index]=0;mask[:,:,:,-1]=0
  result=step(m.model.decoder.embed_tokens(torch.tensor([[token]])),positions[index:index+1].unsqueeze(0),mask,*old,*cross_values)
  for i,value in enumerate(result[1:]):old[i][:,:,index:index+1,:]=value
  if index>=1:
   expected=reference[f'logits_{index-1}'];actual=result[0].reshape(-1)
   error=float((actual-expected).abs().max());passed=bool(torch.allclose(actual,expected,atol=1e-4,rtol=1e-4))
   cases.append({'step':index-1,'max_error':error,'passed':passed})
 (out/'cache-reference.json').write_text(json.dumps(cases,indent=2));print(cases)
 if not all(c['passed'] for c in cases):raise ValueError('Cached/full-prefix parity failed')

 report["width"]=width
 (out/"whisper.json").write_text(json.dumps(report,indent=2))

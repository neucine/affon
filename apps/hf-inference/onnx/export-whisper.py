"""Export bounded Whisper encoder/decoder graphs with an independent greedy reference."""
import argparse,json,wave
from pathlib import Path
from collections import Counter
import numpy as np
import torch,onnx
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingface_hub import HfApi
from safetensors.torch import save_file
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True);p.add_argument('--revision');p.add_argument('--wav',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
revision=a.revision or HfApi().model_info('openai/whisper-tiny.en').sha
torch.set_num_threads(4)
m=WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en',revision=revision,attn_implementation='eager',dtype=torch.float32).eval()
p=WhisperProcessor.from_pretrained('openai/whisper-tiny.en',revision=revision)
m.save_pretrained(out/'source');p.save_pretrained(out/'source')
with wave.open(a.wav) as w:
 assert w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==16000
 audio=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float32)/32768
features=p.feature_extractor(audio,sampling_rate=16000,return_tensors='pt').input_features
width=32
class Encoder(torch.nn.Module):
 def __init__(self):super().__init__();self.encoder=m.model.encoder
 def forward(self,features):return self.encoder(features).last_hidden_state
class Decoder(torch.nn.Module):
 def __init__(self):super().__init__();self.decoder=m.model.decoder;self.head=m.proj_out
 def forward(self,embeddings,encoded):
  # Fixed-length causal decoding; padding tokens cannot affect earlier positions.
  mask=torch.triu(torch.full((1,1,width,width),torch.finfo(torch.float32).min),diagonal=1)
  hidden=self.decoder(inputs_embeds=embeddings,encoder_hidden_states=encoded,attention_mask=mask,use_cache=False).last_hidden_state
  return self.head(hidden)
enc=Encoder().eval();dec=Decoder().eval()
with torch.inference_mode():
 encoded=enc(features)
 embeddings=m.model.decoder.embed_tokens(torch.zeros((1,width),dtype=torch.long))
 for name,wrapper,args,names in [('encoder',enc,(features,),['features']),('decoder',dec,(embeddings,encoded),['embeddings','encoded'])]:
  d=out/name;d.mkdir(exist_ok=True)
  torch.onnx.export(wrapper,args,str(d/'model.onnx'),opset_version=17,dynamo=False,input_names=names,output_names=['output'],do_constant_folding=True)
  graph=onnx.load(d/'model.onnx');print(name,dict(Counter(n.op_type for n in graph.graph.node)),flush=True)
 save_file({'embeddings':m.model.decoder.embed_tokens.weight.contiguous()},str(out/'embeddings.safetensors'))
 ids=[m.config.decoder_start_token_id,m.generation_config.no_timestamps_token_id]
 refs={'features':features.contiguous(),'encoded':encoded.contiguous(),'waveform':torch.from_numpy(audio)}
 for step in range(width-len(ids)):
  batch=ids+[m.config.pad_token_id]*(width-len(ids))
  logits=dec(m.model.decoder.embed_tokens(torch.tensor([batch])),encoded)[0,len(ids)-1]
  refs[f'logits_{step}']=logits.contiguous()
  logits=logits.clone()
  suppress=list(m.generation_config.suppress_tokens or [])
  if step==0:suppress+=list(m.generation_config.begin_suppress_tokens or [])
  logits[suppress]=-float('inf')
  token=int(logits.argmax());ids.append(token)
  if token==m.config.eos_token_id:break
 standard=m.generate(features,max_new_tokens=width-2,do_sample=False,return_timestamps=False)
 save_file(refs,str(out/'reference.safetensors'))
 report={'model':'openai/whisper-tiny.en','revision':revision,'width':width,'prefix':[m.config.decoder_start_token_id,m.generation_config.no_timestamps_token_id],'eos':m.config.eos_token_id,'pad':m.config.pad_token_id,'suppress_tokens':m.generation_config.suppress_tokens,'begin_suppress_tokens':m.generation_config.begin_suppress_tokens,'ids':ids,'text':p.tokenizer.decode(ids,skip_special_tokens=True),'hf_ids':standard[0].tolist(),'hf_text':p.tokenizer.decode(standard[0],skip_special_tokens=True)}
 (out/'whisper.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

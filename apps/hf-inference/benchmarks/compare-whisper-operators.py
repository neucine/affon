"""Generate shared f32 fixtures and benchmark Whisper-shaped MPS operations."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--small',action='store_true')
p.add_argument('--decoder-values',action='store_true',help='Probe short decoder value reductions and dispatch boundaries')
a=p.parse_args();a.fixtures.mkdir(parents=True,exist_ok=True)
assert torch.backends.mps.is_available()
torch.set_num_threads(4)
cases=[]
for name,m,k,n in [('encoder_projection',1500,384,384),('encoder_ffn_up',1500,384,1536),('encoder_ffn_down',1500,1536,384),('decoder_projection',1,384,384),('decoder_ffn_down',1,1536,384)]:
 cases.append(dict(name=name,kind='matmul',a=[1,m,k],b=[k,n],transpose_b=False))
for name,m,k,n,tr in [('encoder_scores',1500,64,1500,True),('encoder_values',1500,1500,64,False),('decoder_cross_scores',1,64,1500,True),('decoder_cross_values',1,1500,64,False),('decoder_self_values',1,257,64,False)]:
 cases.append(dict(name=name,kind='matmul',a=[1,6,m,k],b=[1,6,n,k] if tr else [1,6,k,n],transpose_b=tr))
for name,m,n in [('encoder_attention',1500,1500),('decoder_cross_attention',1,1500)]:
 cases.append(dict(name=name,kind='attention',a=[1,6,m,64],b=[1,6,n,64],v=[1,6,n,64],transpose_b=True))

# Equivalent 2D view isolates rank-based dispatch without changing arithmetic.
cases += [{**c, 'name':c['name']+'_rank2', 'flatten_a':True} for c in cases if len(c['a'])==3]

if a.small:
 cases=[dict(name=f'small_{m}_{k}_{n}',kind='matmul',a=[1,m,k],b=[k,n],transpose_b=False) for m,k,n in [(2,7,3),(4,16,16),(16,32,32),(32,64,64),(64,128,128),(2,384,384),(8,384,1536),(32,384,384)]]

if a.decoder_values:
 cases=[dict(name=f'decoder_values_{k}_{n}',kind='matmul',a=[1,6,1,k],b=[1,6,k,n],transpose_b=False) for k,n in [(16,64),(32,64),(63,64),(64,64),(65,64),(128,64),(257,64),(513,64),(1023,64),(257,127),(257,128),(257,129)]]

def data(shape,offset):
 count=1
 for d in shape:count*=d
 return (((torch.arange(count,dtype=torch.int64)*7+offset)%31-15).float()/128).reshape(shape)

results=[]
with torch.inference_mode():
 for case in cases:
  cpu={'a':data(case['a'],1),'b':data(case['b'],9),'scale':torch.tensor(0.125)}
  if case['kind']=='attention':cpu['v']=data(case['v'],17)
  save_file(cpu,str(a.fixtures/(case['name']+'.safetensors')))
  def calculate(t,sdpa=False):
   if sdpa:return F.scaled_dot_product_attention(t['a'],t['b'],t['v'],dropout_p=0.0,is_causal=False)
   b=t['b'].transpose(-2,-1) if case['transpose_b'] else t['b']
   lhs=t['a'].reshape(t['a'].shape[-2:]) if case.get('flatten_a') else t['a']
   x=torch.matmul(lhs,b)
   return torch.matmul(torch.softmax(x*t['scale'],dim=-1),t['v']) if case['kind']=='attention' else x
  expected=calculate(cpu).reshape(-1)
  indices=sorted(set([0,expected.numel()-1,*[i*(expected.numel()-1)//16 for i in range(17)]]))
  case['samples']=[{'index':i,'expected':float(expected[i])} for i in indices]
  case['numel']=expected.numel();case['runs']=5;case['warmup']=20;case['repeats']=10
  gpu={k:v.to('mps') for k,v in cpu.items()};torch.mps.synchronize()
  for variant in ['eager','sdpa'] if case['kind']=='attention' else ['matmul']:
   for _ in range(20):out=calculate(gpu,variant=='sdpa');torch.mps.synchronize()
   runs=[]
   for _ in range(5):
    torch.mps.synchronize();start=time.perf_counter()
    for repeat in range(10):
     out=calculate(gpu,variant=='sdpa');torch.mps.synchronize()
    runs.append((time.perf_counter()-start)*1000/10)
   actual=out.cpu().reshape(-1)[indices];reference=expected[indices]
   assert torch.allclose(actual,reference,atol=1e-4,rtol=1e-4),(case['name'],variant)
   results.append(dict(name=case['name'],variant=variant,wall_ms=runs,max_sample_error=float((actual-reference).abs().max())))
   print(case['name'],variant,runs,flush=True)
(a.fixtures/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
a.output.write_text(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,device='mps',dtype='float32',cpu_threads=4,warmup=20,runs=5,cases=cases,results=results),indent=2)+'\n')

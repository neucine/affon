// Used by tools/benchmark-quantized-mlx.py. Identical resident data and readback contract.
import { Session, Tensor, program } from 'affon:compute'
import { quantized_matmul, slice } from 'affon:ops'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
const timing=getEnv('AFFON_QMM_TELEMETRY')==='on'
for (const bits of [4, 8] as const) for (const m of [1, 8, 32, 128]) {
  const n=2048, k=1024, group_size=64, perWord=32/bits, mask=(1<<bits)-1
  const session=new Session({device:'metal',telemetry:timing?{backendTiming:true}:false})
  const makeSource=(scalar:boolean)=>program('quantized_metal_benchmark',p=>{const y=quantized_matmul(
    p.argument('x',Tensor.f32([m,k])),p.argument('w',Tensor.u32([n,k/perWord])),
    p.argument('s',Tensor.bf16([n,k/group_size])),p.argument('b',Tensor.bf16([n,k/group_size])),{bits,group_size});return scalar?slice(y,[{start:0,stop:1},{start:0,stop:1}]):y})
  const source=makeSource(false)
  const completionExecutable=session.compile(makeSource(true))
  const executable=session.compile(source,{explanationLevel:'detailed'})
  const x=session.tensor(Array.from({length:m},(_,r)=>Array.from({length:k},(_,c)=>((r+c)%17-8)/16)))
  const w=session.tensor(Array.from({length:n},(_,r)=>Array.from({length:k/perWord},(_,word)=>{
    let v=0;for(let lane=0;lane<perWord;lane++)v|=((r*3+(word*perWord+lane)*7)&mask)<<(lane*bits)
    return v>>>0
  })),{dtype:'u32'})
  const s=session.tensor(Array.from({length:n},()=>Array(k/group_size).fill(1/128)),{dtype:'bf16'})
  const b=session.tensor(Array.from({length:n},()=>Array(k/group_size).fill(-1)),{dtype:'bf16'})
  const run=()=>{const y=executable.run({x,w,s,b});const values=y.to_array() as number[][];y.dispose();return values}
  let checked=run(),max_error=0
  for(const r of [0,m-1])for(const c of [0,17,n-1]){
    let expected=0;for(let i=0;i<k;i++)expected+=((r+i)%17-8)/16*((((c*3+i*7)&mask)/128)-1)
    max_error=Math.max(max_error,Math.abs(checked[r][c]-expected))
  }
  if(max_error>1e-4)throw new Error(`incorrect output ${max_error}`)
  let warmEnd=Date.now()+300;do {run()}while(Date.now()<warmEnd)
  const gpu=()=>timing?telemetry.metrics().find(metric=>metric.scope==='compute.backend'&&metric.name==='elapsed_nanoseconds_total')?.value??0:0
  const iterations=30,samples_ms=[],gpu_samples_ms=[]
  for(let sample=0;sample<5;sample++){
    const before=gpu(),start=Date.now();for(let i=0;i<iterations;i++)run()
    samples_ms.push((Date.now()-start)/iterations);if(timing)gpu_samples_ms.push((gpu()-before)/iterations/1e6)
  }
  const runCompletion=()=>{const y=completionExecutable.run({x,w,s,b});const v=y.item();y.dispose();return v}
  if(Math.abs(runCompletion()-checked[0][0])>1e-4)throw new Error('scalar completion mismatch')
  warmEnd=Date.now()+300;do {runCompletion()}while(Date.now()<warmEnd)
  const completion_samples_ms=[],completion_gpu_samples_ms=[],completion_iterations=100
  for(let sample=0;sample<5;sample++){
    const before=gpu(),start=Date.now();for(let i=0;i<completion_iterations;i++)runCompletion()
    completion_samples_ms.push((Date.now()-start)/completion_iterations)
    if(timing)completion_gpu_samples_ms.push((gpu()-before)/completion_iterations/1e6)
  }
  console.log(JSON.stringify({completion_samples_ms,completion_gpu_samples_ms,completion_iterations,telemetry_enabled:timing,engine:'affon',m,n,k,bits,group_size,iterations,samples_ms,gpu_samples_ms,max_error,explanation:executable.explain()}))
  b.dispose();s.dispose();w.dispose();x.dispose();session.dispose()
}

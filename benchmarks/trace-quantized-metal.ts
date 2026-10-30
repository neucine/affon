// Used by tools/benchmark-quantized-mlx.py. Identical resident data and readback contract.
import { Session, Tensor, program } from 'affon:compute'
import { quantized_matmul, slice } from 'affon:ops'
import { getEnv } from 'std:process'
const warmup=Number(getEnv('AFFON_QMM_WARMUP_MS')??300)
const samples=Number(getEnv('AFFON_QMM_SAMPLES')??100)
const cases=getEnv('AFFON_QMM_M128_ONLY')==='1'?[128]:[1,8,32,128]
for (const bits of [8] as const) for (const m of cases) {
  const n=2048, k=1024, group_size=64, perWord=32/bits, mask=(1<<bits)-1
  const session=new Session({device:'metal',telemetry:{backendTiming:true}})
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
  const check=executable.run({x,w,s,b})
  const actual=check.to_array() as number[][]
  for(const [row,col] of [[0,0],[m-1,n-1]]){
    let expected=0
    for(let inner=0;inner<k;inner++)expected+=((row+inner)%17-8)/16*(((col*3+inner*7)&mask)/128-1)
    if(Math.abs(actual[row][col]-expected)>1e-4)throw new Error('QMM benchmark correctness mismatch')
  }
  check.dispose()
  const warmEnd=Date.now()+warmup
  do {const y=completionExecutable.run({x,w,s,b});y.item();y.dispose()}while(Date.now()<warmEnd)
  for(let i=0;i<samples;i++){const y=completionExecutable.run({x,w,s,b});y.item();y.dispose()}
  b.dispose();s.dispose();w.dispose();x.dispose();session.dispose()
  // Separate workload sizes in the trace; this delay is outside measurements.
  await new Promise(resolve=>setTimeout(resolve,150))
}

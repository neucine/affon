// Used by tools/benchmark-quantized-mlx.py. Identical resident data and readback contract.
import { Session, Tensor, program } from 'affon:compute'
import { quantized_matmul, slice } from 'affon:ops'
import telemetry from 'std:telemetry'
for (const bits of [8] as const) for (const m of [1, 8, 32, 128]) {
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
  const phases=[]
  const gpu=()=>telemetry.metrics().find(t=>t.scope==='compute.backend'&&t.name==='elapsed_nanoseconds_total')?.value??0
  for(const full of [false,true]) {
    const exe=full?executable:completionExecutable
    const warmEnd=Date.now()+300;do {const y=exe.run({x,w,s,b});if(full)y.to_array();else y.item();y.dispose()}while(Date.now()<warmEnd)
    const samples=[]
    for(let sample=0;sample<5;sample++){
      let run_ms=0,read_ms=0,dispose_ms=0;const before=gpu()
      for(let i=0;i<100;i++){
        let t=Date.now();const y=exe.run({x,w,s,b});run_ms+=Date.now()-t
        t=Date.now();if(full)y.to_array();else y.item();read_ms+=Date.now()-t
        t=Date.now();y.dispose();dispose_ms+=Date.now()-t
      }
      samples.push({run_ms:run_ms/100,read_ms:read_ms/100,dispose_ms:dispose_ms/100,gpu_ms:(gpu()-before)/100/1e6})
    }
    phases.push({full_readback:full,samples})
  }
  console.log(JSON.stringify({m,n,k,bits,group_size,phases}))
  b.dispose();s.dispose();w.dispose();x.dispose();session.dispose()
}

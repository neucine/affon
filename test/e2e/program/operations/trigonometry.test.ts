import { test, expect } from 'std:test'
import { Session, Tensor, program, gradient } from 'affon:compute'
import { sin, cos, add, mul, sum, reshape, transpose, slice } from 'affon:ops'
import { conv1d } from 'affon:nn'

for (const device of ['cpu','metal'] as const) {
  for (const dtype of ['f32','bf16'] as const) test(`trigonometry ${device} ${dtype} values and strided views`,()=>{
    const s=new Session({device})
    const data=[[-256,-Math.PI,-.5,0],[.5,Math.PI/2,3,256]]
    const x=s.tensor(data,{dtype})
    const selected=slice(transpose(x,[1,0]),[{start:0,stop:4,step:2},{start:0,stop:2}])
    const actualInput=selected.to_array() as number[][]
    for(const [fn,reference] of [[sin,Math.sin],[cos,Math.cos]] as const){
      const y=fn(selected);const a=y.to_array() as number[][]
      for(let i=0;i<2;i++)for(let j=0;j<2;j++)expect(Math.abs(a[i][j]-reference(actualInput[i][j]))<(dtype==='bf16'?.005:2e-6)).toBe(true)
      y.dispose()
    }
    selected.dispose();x.dispose();s.dispose()
  })
  test(`trigonometry ${device} native fused execution and gradient`,()=>{
    const s=new Session({device})
    const model=program('trig_gradient',p=>sum(add(sin(p.argument('x',Tensor.f32([5]))),cos(p.argument('z',Tensor.f32([5]))))))
    const derivative=gradient(model,'x')
    const x=s.tensor([-3,-1,0,1,3]);const z=s.tensor([0,0,0,0,0])
    const y=s.compile(derivative).run({x,z}) as any
    const a=y.to_array() as number[]
    for(let i=0;i<5;i++)expect(Math.abs(a[i]-Math.cos([-3,-1,0,1,3][i]))<2e-6).toBe(true)
    const cosInput=s.tensor([-.75,-.25,0,.25,.75])
    const cosGradient=s.compile(gradient(model,'z')).run({x,z:cosInput}) as any
    const ca=cosGradient.to_array() as number[]
    for(let i=0;i<5;i++)expect(Math.abs(ca[i]+Math.sin([-.75,-.25,0,.25,.75][i]))<2e-6).toBe(true)
    cosGradient.dispose();cosInput.dispose()
    const chain=program('trig_chain',p=>cos(sin(mul(p.argument('x',Tensor.f32([5])),p.argument('z',Tensor.f32([5]))))))
    for(const optimizationLevel of ['none','safe'] as const){
      const executable=s.compile(chain,{optimizationLevel})
      if(device==='metal')expect(JSON.stringify(executable.explain()).includes('metal_host')).toBe(false)
      const out=executable.run({x,z}) as any
      expect(out.to_array()).toEqual([1,1,1,1,1]);out.dispose();executable.dispose()
    }
    y.dispose();x.dispose();z.dispose();s.dispose()
  })
  test(`layout ${device} inserted copies preserve pending producers`,()=>{
    const s=new Session({device})
    const model=program('pending_layout',p=>{
      const x=p.argument('x',Tensor.f32([2,4]));const a=mul(x,x)
      return reshape(transpose(a,[1,0]),[8])
    })
    const x=s.tensor([[1,2,3,4],[5,6,7,8]])
    for(const optimizationLevel of ['none','safe'] as const){
      const executable=s.compile(model,{optimizationLevel})
      const y=executable.run({x}) as any
      expect(y.to_array()).toEqual([1,25,4,36,9,49,16,64]);y.dispose();executable.dispose()
    }
    x.dispose();s.dispose()
  })
  test(`depthwise ${device} preserves batch channels dilation and gradients`,()=>{
    const layer=conv1d({out_channels:2,kernel_size:2,dilation:2,groups:2,bias:false})
    const model=program('depthwise_batch',p=>layer({x:p.argument('x',Tensor.f32([2,2,4]))},'conv'))
    const s=new Session({device})
    const parameters={'conv.weight':[[[2,3]],[[5,7]]]}
    const state=s.initialize(model,{parameters})
    const x=s.tensor([[[1,2,3,4],[10,20,30,40]],[[2,3,4,5],[3,4,5,6]]])
    const y=s.compile(model).run({x},state) as any
    expect(y.to_array()).toEqual([[[11,16],[260,380]],[[16,21],[50,62]]])
    const objective=program('depthwise_objective',p=>sum(model({x:p.argument('x',Tensor.f32([2,2,4]))})))
    const derivative=gradient(objective,'depthwise_batch.conv.weight')
    const ds=s.initialize(derivative,{parameters:{'depthwise_batch.conv.weight':parameters['conv.weight']}})
    for (const optimizationLevel of ['none','safe'] as const) {
      const dy=s.compile(derivative,{optimizationLevel}).run({x},ds) as any
      expect(dy.to_array()).toEqual([[[8,16]],[[37,81]]]);dy.dispose()
    }
    ds.dispose();y.dispose();x.dispose();state.dispose();s.dispose()
  })
}
test('trigonometry rejects integer semantics',()=>{
  expect(()=>program('invalid_trig',p=>sin(p.argument('x',Tensor.u32([1]))))).toThrow('floating-point')
})
test('CPU f64 trigonometry retains double precision',()=>{
  const s=new Session({device:'cpu'})
  const x=s.tensor([-.75,0,.75],{dtype:'f64'});const a=sin(x);const b=cos(x)
  for(const [out,reference] of [[a,Math.sin],[b,Math.cos]] as const){
    const actual=out.to_array() as number[]
    for(let i=0;i<3;i++)expect(Math.abs(actual[i]-reference([-.75,0,.75][i]))<1e-14).toBe(true)
  }
  a.dispose();b.dispose();x.dispose();s.dispose()
})
test('dense input specialization guards views and keeps a separate executable',()=>{
  const s=new Session({device:'metal'})
  const model=program('guarded_dense',p=>sin(p.argument('x',Tensor.f32([2,2]))))
  const generic=s.compile(model)
  const dense=s.compile(model,{denseInputs:true})
  expect(generic===dense).toBe(false)
  const x=s.tensor([[0,1],[2,3]]);const view=transpose(x,[1,0])
  const y=dense.run({x}) as any;y.dispose()
  expect(()=>dense.run({x:view})).toThrow('GuardMismatch')
  expect(()=>s.compile(model,{denseInputs:'yes' as any})).toThrow('boolean')
  const z=generic.run({x:view}) as any;z.dispose();view.dispose();x.dispose();s.dispose()
})

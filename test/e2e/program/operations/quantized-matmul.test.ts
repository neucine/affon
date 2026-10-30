import { describe, expect, test } from 'std:test'
import { Session, Tensor, program } from 'affon:compute'
import { quantized_matmul, slice, transpose } from 'affon:ops'

for (const device of ['cpu', 'metal'] as const) describe(`quantized_matmul ${device}`, () => {
  test('decodes low-bit-first U32 words and applies per-group affine parameters', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 1, 1]])
    const packed = session.tensor([[0x04030201, 0x08070605], [0x00000000, 0xffffffff]], { dtype: 'u32' })
    const scales = session.tensor([[1, 2], [1, 0.5]], { dtype: 'bf16' })
    const biases = session.tensor([[0, -1], [2, 0]], { dtype: 'bf16' })
    const output = quantized_matmul(input, packed, scales, biases, { bits: 8, group_size: 4 })
    expect(output.shape).toEqual([2, 2])
    expect(output.dtype).toBe('f32')
    expect(output.to_array()).toEqual([[352, 3335], [58, 518]])
    output.dispose(); biases.dispose(); scales.dispose(); packed.dispose(); input.dispose(); session.dispose()
  })

  test('rejects mismatched packing and group metadata', () => {
    const session = new Session({ device })
    const input = session.tensor([[1, 2, 3, 4]])
    const packed = session.tensor([[0x04030201]], { dtype: 'u32' })
    const scales = session.tensor([[1]])
    const biases = session.tensor([[0]])
    expect(() => quantized_matmul(input, packed, scales, biases, { bits: 8, group_size: 3 })).toThrow()
    expect(() => quantized_matmul(input, packed, scales, biases, { bits: 4, group_size: 4 })).toThrow()
    biases.dispose(); scales.dispose(); packed.dispose(); input.dispose(); session.dispose()
  })

  test('supports two and four bits, including high bits of unsigned words', () => {
    const session = new Session({ device })
    for (const [bits, word, width, expected] of [[2, 0xe4e4e4e4, 16, -24], [4, 0x76543210, 8, 48]] as const) {
      const input = session.tensor([Array.from({ length: width }, (_, k) => k + 1)], { dtype: 'bf16' })
      const packed = session.tensor([[word]], { dtype: 'u32' })
      const scales = session.tensor([[0.5]])
      const biases = session.tensor([[-1]])
      const output = quantized_matmul(input, packed, scales, biases, { bits, group_size: width })
      expect(output.to_array()).toEqual([[expected]])
      output.dispose(); biases.dispose(); scales.dispose(); packed.dispose(); input.dispose()
    }
    session.dispose()
  })

  test('rejects cross-Session tensors and invalid bit widths', () => {
    const session = new Session({ device })
    const other = new Session({ device })
    const input = session.tensor([[1, 2, 3, 4]])
    const packed = session.tensor([[0xffffffff]], { dtype: 'u32' })
    const scales = session.tensor([[1]])
    const biases = other.tensor([[0]])
    expect(() => quantized_matmul(input, packed, scales, biases, { bits: 8, group_size: 4 })).toThrow('same Session')
    const localBiases = session.tensor([[0]])
    expect(() => quantized_matmul(input, packed, scales, localBiases, { bits: 3 as any, group_size: 4 })).toThrow('bits')
    localBiases.dispose(); biases.dispose(); scales.dispose(); packed.dispose(); input.dispose(); other.dispose(); session.dispose()
  })

  test('authors a single native operation and runs without a floating weight intermediate', () => {
    const source = program('quantized_linear', p => quantized_matmul(
      p.argument('x', Tensor.f32([1, 4])), p.argument('w', Tensor.u32([1, 1])),
      p.argument('s', Tensor.bf16([1, 1])), p.argument('b', Tensor.bf16([1, 1])),
      { bits: 8, group_size: 4 },
    ))
    expect(source.inspect().nodes.filter(node => node.kind === 'operation').map(node => node.op)).toEqual(['quantized_matmul'])
    const session = new Session({ device })
    const executable = session.compile(source, { explanationLevel: 'detailed' })
    const explanation = JSON.stringify(executable.explain())
    expect(explanation.includes(`${device}_quantized_matmul`)).toBe(true)
    expect(explanation.includes(`${device}_host_`)).toBe(false)
    if (device === 'metal') expect(() => session.compile(source, { numericalPolicy: 'exact_only' })).toThrow()
    const x = session.tensor([[1, 2, 3, 4]])
    const w = session.tensor([[0x04030201]], { dtype: 'u32' })
    const s = session.tensor([[1]], { dtype: 'bf16' })
    const b = session.tensor([[0]], { dtype: 'bf16' })
    const y = executable.run({ x, w, s, b })
    expect(y.to_array()).toEqual([[30]])
    y.dispose(); b.dispose(); s.dispose(); w.dispose(); x.dispose(); session.dispose()
  })

  test('reads transposed input and sliced weights and parameters with byte offsets', () => {
    const session = new Session({ device })
    const baseX = session.tensor([[1, 1], [2, 1], [3, 1], [4, 1]])
    const x = transpose(baseX, [1, 0])
    const baseW = session.tensor([[0, 0], [0, 0x04030201]], { dtype: 'u32' })
    const w = slice(baseW, [{ start: 1, stop: 2 }, { start: 1, stop: 2 }])
    const baseS = session.tensor([[0, 1]], { dtype: 'bf16' })
    const baseB = session.tensor([[0, -1]], { dtype: 'bf16' })
    const s = slice(baseS, [{ start: 0, stop: 1 }, { start: 1, stop: 2 }])
    const b = slice(baseB, [{ start: 0, stop: 1 }, { start: 1, stop: 2 }])
    const y = quantized_matmul(x, w, s, b, { bits: 8, group_size: 4 })
    expect(y.to_array()).toEqual([[20], [6]])
    y.dispose(); b.dispose(); s.dispose(); baseB.dispose(); baseS.dispose(); w.dispose(); baseW.dispose(); x.dispose(); baseX.dispose(); session.dispose()
  })

  test('handles groups crossing word boundaries and zero and negative scales', () => {
    const session = new Session({ device })
    const k = 24, n = 3, m = 2, group_size = 3
    const q = (row: number, col: number) => (row * 5 + col) % 16
    const inputs = Array.from({ length: m }, (_, row) => Array.from({ length: k }, (_, col) => ((row + col) % 7 - 3) / 8))
    const scales = Array.from({ length: n }, () => Array.from({ length: k / group_size }, (_, group) => (group % 3 - 1) / 4))
    const biases = Array.from({ length: n }, () => Array.from({ length: k / group_size }, (_, group) => group % 2 - 0.5))
    const weights = Array.from({ length: n }, (_, row) => Array.from({ length: k / 8 }, (_, word) => {
      let value = 0
      for (let lane = 0; lane < 8; lane++) value |= q(row, word * 8 + lane) << (lane * 4)
      return value >>> 0
    }))
    const expected = inputs.map(x => Array.from({ length: n }, (_, row) => x.reduce((sum, a, col) => {
      const group = Math.floor(col / group_size)
      return sum + a * (scales[row][group] * q(row, col) + biases[row][group])
    }, 0)))
    for (const dtype of ['f32', 'bf16'] as const) {
      const x = session.tensor(inputs, { dtype })
      const w = session.tensor(weights, { dtype: 'u32' })
      const s = session.tensor(scales, { dtype })
      const b = session.tensor(biases, { dtype })
      const y = quantized_matmul(x, w, s, b, { bits: 4, group_size })
      expect(y.to_array()).toEqual(expected)
      expect(w.to_array()).toEqual(weights)
      y.dispose(); b.dispose(); s.dispose(); w.dispose(); x.dispose()
    }
    session.dispose()
  })
  test('SIMD and matrix tiles handle partial M/N/K tiles across all bit widths', () => {
    const session = new Session({ device })
    for (const bits of [2, 4, 8] as const) for (const m of [1, 3, 4, 7, 16, 17, 33, 64, 128]) {
      const n=m>=64?64:19, k=m>=64?96:80, group_size=16, perWord=32/bits, mask=(1<<bits)-1
      const inputs=Array.from({length:m},(_,r)=>Array.from({length:k},(_,c)=>((r+c)%7-3)/8))
      const weights=Array.from({length:n},(_,r)=>Array.from({length:k/perWord},(_,word)=>{
        let v=0;for(let lane=0;lane<perWord;lane++)v|=((r*5+word*perWord+lane)&mask)<<(lane*bits)
        return v>>>0
      }))
      const scales=Array.from({length:n},()=>Array.from({length:k/group_size},(_,g)=>(g%3-1)/4))
      const biases=Array.from({length:n},()=>Array.from({length:k/group_size},(_,g)=>g%2-0.5))
      const expected=inputs.map(row=>Array.from({length:n},(_,out)=>row.reduce((sum,a,c)=>{
        const g=Math.floor(c/group_size)
        return sum+a*(scales[out][g]*((out*5+c)&mask)+biases[out][g])
      },0)))
      const baseX=session.tensor((m===17||m===64)?Array.from({length:k},(_,c)=>inputs.map(row=>row[c])):inputs,{dtype:'bf16'})
      const x=(m===17||m===64)?transpose(baseX,[1,0]):baseX
      const baseW=session.tensor((m===17||m===64)?weights.map(row=>[0,...row]):weights,{dtype:'u32'})
      const w=(m===17||m===64)?slice(baseW,[{start:0,stop:n},{start:1,stop:1+k/perWord}]):baseW
      const s=session.tensor(scales,{dtype:'bf16'}),b=session.tensor(biases,{dtype:'bf16'})
      const y=quantized_matmul(x,w,s,b,{bits,group_size})
      expect(y.to_array()).toEqual(expected)
      y.dispose();b.dispose();s.dispose();if(w!==baseW)w.dispose();baseW.dispose();if(x!==baseX)x.dispose();baseX.dispose()
    }
    session.dispose()
  })

})

for (const device of ['cpu','metal'] as const) test(`native readback preserves ${device} views and dtypes`, () => {
  const session=new Session({device})
  for(const dtype of ['bf16','f32','f64','u32','i64'] as const){
    const high=dtype==='u32'?0xffffffff:6
    const base=session.tensor([[1,2,3],[4,5,high]],{dtype})
    const transposed=transpose(base,[1,0])
    const view=slice(transposed,[{start:1,stop:3},{start:0,stop:2}])
    expect(view.to_array()).toEqual([[2,5],[3,high]])
    const scalar=slice(view,[{start:1,stop:2},{start:1,stop:2}])
    expect(scalar.item()).toBe(high)
    scalar.dispose();view.dispose();transposed.dispose();base.dispose()
    const rankZero=session.tensor(7,{dtype})
    expect(rankZero.to_array()).toBe(7)
    expect(rankZero.item()).toBe(7)
    rankZero.dispose()
  }
  session.dispose()
})

for(const device of ['cpu','metal'] as const) test(`prepared ${device} group and dtype specializations stay distinct`,()=>{
  const session=new Session({device}),m=8,n=3,k=128
  const x=session.tensor(Array.from({length:m},()=>Array(k).fill(1)),{dtype:'bf16'})
  const w=session.tensor(Array.from({length:n},()=>Array(k/4).fill(0x80808080)),{dtype:'u32'})
  for(const group_size of [32,64,128])for(const dtype of ['f32','bf16'] as const){
    const groups=k/group_size
    const scales=session.tensor(Array.from({length:n},()=>Array.from({length:groups},(_,g)=>(g+1)/128)),{dtype})
    const biases=session.tensor(Array.from({length:n},()=>Array(groups).fill(-1)),{dtype})
    const y=quantized_matmul(x,w,scales,biases,{bits:8,group_size})
    expect(y.to_array()).toEqual(Array.from({length:m},()=>Array(n).fill(group_size*groups*(groups-1)/2)))
    y.dispose();biases.dispose();scales.dispose()
  }
  w.dispose();x.dispose();session.dispose()
})

// Packed staging must preserve groups both larger and smaller than its K tile.
for (const group_size of [48, 64]) test(`Metal packed tiles preserve group ${group_size} across K tiles`, () => {
  const session=new Session({device:'metal'}),m=64,n=64,k=group_size===48?96:256
  for(const dtype of ['f32','bf16'] as const){
    const values=Array.from({length:m},(_,r)=>Array.from({length:k},(_,c)=>((r+c)%11-5)/8))
    const weights=Array.from({length:n},(_,r)=>Array.from({length:k/4},(_,c)=>{
      let word=0;for(let lane=0;lane<4;lane++)word|=((r+c*4+lane)%256)<<(lane*8);return word>>>0
    }))
    const scales=Array.from({length:n},(_,r)=>Array.from({length:k/group_size},(_,g)=>(r%3-1)*(g+1)/128))
    const biases=Array.from({length:n},()=>Array.from({length:k/group_size},(_,g)=>(g%3-1)/4))
    const expected=values.map(row=>Array.from({length:n},(_,r)=>row.reduce((sum,a,c)=>sum+a*(scales[r][Math.floor(c/group_size)]*((r+c)%256)+biases[r][Math.floor(c/group_size)]),0)))
    const x=session.tensor(values,{dtype}),w=session.tensor(weights,{dtype:'u32'})
    const s=session.tensor(scales,{dtype}),b=session.tensor(biases,{dtype})
    const y=quantized_matmul(x,w,s,b,{bits:8,group_size})
    expect(y.to_array()).toEqual(expected)
    y.dispose();b.dispose();s.dispose();w.dispose();x.dispose()
  }
  session.dispose()
})

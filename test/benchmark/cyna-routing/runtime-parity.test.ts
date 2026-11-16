import { describe, expect, test } from 'std:test'
import fs from 'std:fs'
import { Session, Tensor, program } from 'affon:core'
import { add, exp, layer_norm, matmul, rms_norm, softmax, sum, transpose } from 'affon:ops'
import { load_graph } from '../../../packages/@affon/onnx/src/runtime.ts'
import { create_bert } from '../../../packages/@affon/models/src/bert/model.ts'
import { create_gpt2 } from '../../../packages/@affon/models/src/gpt2/model.ts'
import { create_llama } from '../../../packages/@affon/models/src/llama/model.ts'
import { create_vit } from '../../../packages/@affon/models/src/vit/model.ts'

const backendSymbol = Symbol.for('affon.compute.executable_backend')
const routingSymbol = Symbol.for('affon.compute.executable_routing')
type NumericTolerance = Readonly<{ absolute: number; relative: number }>
const f32Tolerance: NumericTolerance = { absolute: 3e-4, relative: 3e-4 }
const f64Tolerance: NumericTolerance = { absolute: 1e-12, relative: 1e-12 }
const bf16Tolerance: NumericTolerance = { absolute: 1 / 128, relative: 1 / 128 }

type InputFactory = (session: Session) => Record<string, Tensor>

function nested(shape: readonly number[], seed: number, scale = 0.1): any {
  let state = seed >>> 0
  const next = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return ((state / 0x1_0000_0000) * 2 - 1) * scale
  }
  return shape.length
    ? Array.from({ length: shape[0] }, () => nested(shape.slice(1), state))
    : next()
}

function values(tensor: Tensor): number[] {
  return (tensor.to_array() as any[]).flat(Infinity) as number[]
}

function outputs(result: Tensor | Tensor[]): Tensor[] {
  return Array.isArray(result) ? result : [result]
}

function runPaired(
  name: string,
  source: any,
  parameters: Record<string, Tensor> | undefined,
  inputs: InputFactory,
  comparisonTolerance: NumericTolerance = f32Tolerance,
): void {
  const cyna = new Session({ device: 'cpu' })
  const zig = new Session({ device: 'cpu', telemetry: { captureRecords: 32 } })
  const cynaExecutable = cyna.compile(source)
  const zigExecutable = zig.compile(source)
  const cynaState = cyna.initialize(source, { parameters })
  const zigState = zig.initialize(source, { parameters })
  const cynaInputs = inputs(cyna)
  const zigInputs = inputs(zig)
  let cynaOutputs: Tensor[] = []
  let zigOutputs: Tensor[] = []
  try {
    expect((cynaExecutable as any)[backendSymbol]()).toBe('cyna')
    expect((cynaExecutable as any)[routingSymbol]().reason).toBe('lowered')
    expect((zigExecutable as any)[backendSymbol]()).toBe('zig')
    const zigRouting = (zigExecutable as any)[routingSymbol]()
    expect(zigRouting.reason).toBe('evidence_capture')

    cynaOutputs = outputs(cynaExecutable.run(cynaInputs, cynaState))
    zigOutputs = outputs(zigExecutable.run(zigInputs, zigState))
    expect(cynaOutputs.length).toBe(zigOutputs.length)
    if (cynaOutputs.length === 0) throw new Error(`${name}: runtime returned no outputs`)

    for (let index = 0; index < cynaOutputs.length; index++) {
      const actual = cynaOutputs[index]
      const expected = zigOutputs[index]
      expect(actual.shape).toEqual(expected.shape)
      expect(actual.dtype).toBe(expected.dtype)
      const actualValues = values(actual)
      const expectedValues = values(expected)
      expect(actualValues.length).toBe(expectedValues.length)
      for (let element = 0; element < actualValues.length; element++) {
        const a = actualValues[element]
        const e = expectedValues[element]
        const difference = Math.abs(a - e)
        const limit = comparisonTolerance.absolute + comparisonTolerance.relative * Math.abs(e)
        if (!Number.isFinite(a) || !Number.isFinite(e) || difference > limit) {
          throw new Error(`${name}: output ${index} element ${element}: cyna=${a}, zig=${e}, difference=${difference}, tolerance=${limit}`)
        }
      }
    }
  } finally {
    for (const tensor of cynaOutputs) tensor.dispose()
    for (const tensor of zigOutputs) tensor.dispose()
    for (const tensor of Object.values(cynaInputs)) tensor.dispose()
    for (const tensor of Object.values(zigInputs)) tensor.dispose()
    cynaState.dispose()
    zigState.dispose()
    cynaExecutable.dispose()
    zigExecutable.dispose()
    cyna.dispose()
    zig.dispose()
  }
}

describe('Cyna runtime numerical parity', () => {
  test('transposed f32 matrix multiplication matches Zig Session', () => {
    const source = program('cyna_parity_transposed_matmul', p => matmul(
      p.argument('x', Tensor.f32([2, 3])),
      transpose(p.argument('weight', Tensor.f32([4, 3])), [1, 0]),
    ))
    runPaired('transposed f32 matrix multiplication', source, undefined, session => ({
      x: session.tensor([[1, -2, 3], [4, 5, -6]], { dtype: 'f32' }),
      weight: session.tensor([
        [1, 2, 3], [-4, 5, 6], [7, -8, 9], [10, 11, -12],
      ], { dtype: 'f32' }),
    }))
  })

  test('authored transformer block matches Zig Session', () => {
    const source = program('cyna_parity_transformer_block', p => {
      const input = p.argument('x', Tensor.f32([2, 4, 8]))
      const normalized = layer_norm(input, -1)
      const query = matmul(normalized, p.argument('query_weight', Tensor.f32([8, 8])))
      const key = matmul(normalized, p.argument('key_weight', Tensor.f32([8, 8])))
      const value = matmul(normalized, p.argument('value_weight', Tensor.f32([8, 8])))
      const scores = matmul(query, transpose(key, [0, 2, 1]))
      const attended = matmul(softmax(scores, -1), value)
      const projected = matmul(attended, p.argument('output_weight', Tensor.f32([8, 8])))
      return rms_norm(add(input, projected), -1)
    })
    runPaired('authored transformer block', source, undefined, session => ({
      x: session.tensor(nested([2, 4, 8], 1, 0.2)),
      query_weight: session.tensor(nested([8, 8], 2)),
      key_weight: session.tensor(nested([8, 8], 3)),
      value_weight: session.tensor(nested([8, 8], 4)),
      output_weight: session.tensor(nested([8, 8], 5)),
    }))
  })

  test('f64 unary, reduction, and matmul match Zig Session', () => {
    const unary = program('cyna_parity_f64_unary', p => exp(p.argument('x', Tensor.f64([2, 3]))))
    runPaired('f64 unary', unary, undefined, session => ({
      x: session.tensor([[-1.25, -0.5, -0.125], [0.125, 0.5, 1.25]], { dtype: 'f64' }),
    }), f64Tolerance)

    const reduction = program('cyna_parity_f64_reduction', p => sum(p.argument('x', Tensor.f64([2, 3])), 1))
    runPaired('f64 reduction', reduction, undefined, session => ({
      x: session.tensor([[0.125, -0.5, 1.25], [-1.25, 0.5, -0.125]], { dtype: 'f64' }),
    }), f64Tolerance)

    const matrix = program('cyna_parity_f64_matmul', p => matmul(
      p.argument('left', Tensor.f64([2, 3])),
      p.argument('right', Tensor.f64([3, 4])),
    ))
    runPaired('f64 matmul', matrix, undefined, session => ({
      left: session.tensor([[1, -2, 0.5], [3, 0.25, -1]], { dtype: 'f64' }),
      right: session.tensor([[0.5, 1, -1, 2], [2, -0.5, 0.25, 3], [-1, 4, 1.5, 0]], { dtype: 'f64' }),
    }), f64Tolerance)
  })

  test('bf16 unary, reduction, and matmul match Zig Session', () => {
    const unary = program('cyna_parity_bf16_unary', p => exp(p.argument('x', Tensor.bf16([2, 3]))))
    runPaired('bf16 unary', unary, undefined, session => ({
      x: session.tensor([[-1, -0.5, -0.125], [0.125, 0.5, 1]], { dtype: 'bf16' }),
    }), bf16Tolerance)

    const reduction = program('cyna_parity_bf16_reduction', p => sum(p.argument('x', Tensor.bf16([2, 3])), 1))
    runPaired('bf16 reduction', reduction, undefined, session => ({
      x: session.tensor([[1, -2, 0.5], [-1, 2, -0.5]], { dtype: 'bf16' }),
    }), bf16Tolerance)

    const matrix = program('cyna_parity_bf16_matmul', p => matmul(
      p.argument('left', Tensor.bf16([2, 3])),
      p.argument('right', Tensor.bf16([3, 4])),
    ))
    runPaired('bf16 matmul', matrix, undefined, session => ({
      left: session.tensor([[1, -2, 0.5], [3, 0.25, -1]], { dtype: 'bf16' }),
      right: session.tensor([[0.5, 1, -1, 2], [2, -0.5, 0.25, 3], [-1, 4, 1.5, 0]], { dtype: 'bf16' }),
    }), bf16Tolerance)
  })

  for (const fixture of [
    ['generic', 'packages/@affon/onnx/test/fixtures'],
    ['spatial', 'packages/@affon/onnx/test/fixtures/spatial'],
    ['Conv1d', 'packages/@affon/onnx/test/fixtures/conv1d'],
  ] as const) {
    test(`imported ONNX ${fixture[0]} program with parameters matches Zig Session`, () => {
      const [name, directory] = fixture
      const reference = JSON.parse(fs.readFileSync(`${directory}/reference.json`))
      const model = load_graph(directory)
      try {
        runPaired(`imported ONNX ${name}`, model.forward, model.parameters, session => ({
          x: session.tensor(reference.input),
        }))
      } finally {
        for (const parameter of Object.values(model.parameters)) parameter.dispose()
      }
    })
  }

  test('small GPT-2 prefill matches Zig Session', () => {
    const sourceSession = new Session({ device: 'cpu' })
    try {
      const value = (shape: readonly number[], seed: number, scale = 0.1): Tensor => sourceSession.tensor(nested(shape, seed, scale), { dtype: 'f32' })
      const affine = (input: number, output: number, seed: number) => ({ weight: value([input, output], seed), bias: value([output], seed + 1, 0) })
      const norm = (seed: number) => ({ weight: value([4], seed, 1), bias: value([4], seed + 1, 0) })
      const model = create_gpt2(
        { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 },
        {
          tokenEmbedding: value([7, 4], 20),
          positionEmbedding: value([4, 4], 21),
          finalNorm: norm(22),
          blocks: [{ attentionNorm: norm(23), feedForwardNorm: norm(24), qkv: affine(4, 12, 25), attentionOutput: affine(4, 4, 27), expand: affine(4, 8, 29), contract: affine(8, 4, 31) }],
        },
      )
      const forward = model.forward(3)
      runPaired('GPT-2 prefill', forward, model.parameters, session => ({
        ids: session.tensor([1, 2, 3], { dtype: 'i64' }),
        positions: session.tensor([0, 1, 2], { dtype: 'i64' }),
        mask: session.tensor([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]], { dtype: 'i64' }),
      }))
    } finally {
      sourceSession.dispose()
    }
  })

  test('small BF16-stored GPT-2 prefill matches Zig Session', () => {
    const sourceSession = new Session({ device: 'cpu' })
    try {
      const value = (shape: readonly number[], seed: number, scale = 0.1): Tensor => sourceSession.tensor(nested(shape, seed, scale), { dtype: 'bf16' })
      const affine = (input: number, output: number, seed: number) => ({ weight: value([input, output], seed), bias: value([output], seed + 1, 0) })
      const norm = (seed: number) => ({ weight: value([4], seed, 1), bias: value([4], seed + 1, 0) })
      const model = create_gpt2(
        { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 },
        {
          tokenEmbedding: value([7, 4], 30),
          positionEmbedding: value([4, 4], 31),
          finalNorm: norm(32),
          blocks: [{ attentionNorm: norm(33), feedForwardNorm: norm(34), qkv: affine(4, 12, 35), attentionOutput: affine(4, 4, 37), expand: affine(4, 8, 39), contract: affine(8, 4, 41) }],
        },
      )
      runPaired('BF16-stored GPT-2 prefill', model.forward(3), model.parameters, session => ({
        ids: session.tensor([1, 2, 3], { dtype: 'i64' }),
        positions: session.tensor([0, 1, 2], { dtype: 'i64' }),
        mask: session.tensor([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]], { dtype: 'i64' }),
      }), f32Tolerance)
    } finally {
      sourceSession.dispose()
    }
  })

  test('small BERT encoder matches Zig Session', () => {
    const sourceSession = new Session({ device: 'cpu' })
    try {
      const value = (shape: readonly number[], seed: number, scale = 0.1): Tensor => sourceSession.tensor(nested(shape, seed, scale), { dtype: 'f32' })
      const affine = (input: number, output: number, seed: number) => ({ weight: value([input, output], seed), bias: value([output], seed + 1, 0) })
      const norm = (seed: number) => ({ weight: value([4], seed, 1), bias: value([4], seed + 1, 0) })
      const model = create_bert(
        { width: 4, innerWidth: 8, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, typeVocabSize: 2, epsilon: 1e-5 },
        {
          tokenEmbedding: value([7, 4], 40),
          positionEmbedding: value([4, 4], 41),
          typeEmbedding: value([2, 4], 42),
          embeddingNorm: norm(43),
          pooler: affine(4, 4, 44),
          blocks: [{ query: affine(4, 4, 46), key: affine(4, 4, 48), value: affine(4, 4, 50), attentionOutput: affine(4, 4, 52), attentionNorm: norm(54), feedForwardNorm: norm(56), expand: affine(4, 8, 58), contract: affine(8, 4, 60) }],
        },
      )
      const forward = model.forward(1, 2)
      runPaired('BERT encoder', forward, model.parameters, session => ({
        ids: session.tensor([[1, 2]], { dtype: 'i64' }),
        types: session.tensor([[0, 0]], { dtype: 'i64' }),
        positions: session.tensor([[0, 1]], { dtype: 'i64' }),
        valid: session.tensor([[[1], [1]]]),
        attention_mask: session.tensor([[[[0, 0]]]], { dtype: 'i64' }),
      }))
    } finally {
      sourceSession.dispose()
    }
  })

  test('small ViT image encoder matches Zig Session', () => {
    const sourceSession = new Session({ device: 'cpu' })
    try {
      const value = (shape: readonly number[], seed: number, scale = 0.1): Tensor => sourceSession.tensor(nested(shape, seed, scale), { dtype: 'f32' })
      const affine = (input: number, output: number, seed: number) => ({ weight: value([input, output], seed), bias: value([output], seed + 1, 0) })
      const norm = (seed: number) => ({ weight: value([4], seed, 1), bias: value([4], seed + 1, 0) })
      const model = create_vit(
        { width: 4, innerWidth: 8, heads: 2, layers: 1, imageSize: 2, patchSize: 1, epsilon: 1e-5 },
        {
          classToken: value([1, 1, 4], 70),
          positionEmbedding: value([1, 5, 4], 71),
          patchProjection: { weight: value([4, 3, 1, 1], 72), bias: value([4], 73, 0) },
          finalNorm: norm(74),
          classifier: affine(4, 3, 75),
          blocks: [{ query: affine(4, 4, 77), key: affine(4, 4, 79), value: affine(4, 4, 81), attentionOutput: affine(4, 4, 83), attentionNorm: norm(85), feedForwardNorm: norm(87), expand: affine(4, 8, 89), contract: affine(8, 4, 91) }],
        },
      )
      runPaired('ViT image encoder', model.forward, model.parameters, session => ({
        pixels: session.tensor(nested([1, 3, 2, 2], 100, 0.5)),
      }))
    } finally {
      sourceSession.dispose()
    }
  })

  test('small Llama prefill and cached decode match Zig Session', () => {
    const sourceSession = new Session({ device: 'cpu' })
    try {
      const value = (shape: readonly number[], seed: number, scale = 0.1): Tensor => sourceSession.tensor(nested(shape, seed, scale), { dtype: 'f32' })
      const model = create_llama(
        { width: 4, innerWidth: 8, heads: 2, kvHeads: 1, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5, ropeTheta: 10_000 },
        {
          tokenEmbedding: value([7, 4], 110),
          finalNorm: value([4], 111, 1),
          blocks: [{
            attentionNorm: value([4], 112, 1),
            feedForwardNorm: value([4], 113, 1),
            query: value([4, 4], 114), key: value([4, 2], 115), value: value([4, 2], 116), attentionOutput: value([4, 4], 117),
            gate: value([4, 8], 118), up: value([4, 8], 119), down: value([8, 4], 120),
          }],
        },
      )
      const forward = model.forward(3)
      runPaired('Llama prefill', forward, model.parameters, session => ({
        ids: session.tensor([1, 2, 3], { dtype: 'i64' }),
        mask: session.tensor([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]], { dtype: 'i64' }),
      }))
      const decode = model.decode(1)
      runPaired('Llama cached decode', decode, model.parameters, session => ({
        ids: session.tensor([4], { dtype: 'i64' }),
        past_0_k: session.tensor(nested([1, 1, 1, 2], 130)),
        past_0_v: session.tensor(nested([1, 1, 1, 2], 131)),
      }))
    } finally {
      sourceSession.dispose()
    }
  })
})

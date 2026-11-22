import { describe, expect, test } from 'std:test'
import fs from 'std:fs'
import { Session, Tensor, program } from 'affon:core'
import { abs, add, argmax, cast, cat, clamp, contiguous, cross_entropy, dense_cross_entropy, embedding, eq, exp, gather, gelu, gt, index_select, layer_norm, log_softmax, lt, masked_fill, matmul, max, mean, min, mul, one_hot, permute, reshape, rms_norm, scatter_add, slice, softmax, sqrt, stack, std, sum, topk, transpose, variance, where } from 'affon:ops'
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
  compileOptions: any = {},
): void {
  const cyna = new Session({ device: 'cpu' })
  const zig = new Session({ device: 'cpu', cpuExecution: 'zig', telemetry: { captureRecords: 32 } })
  const cynaExecutable = cyna.compile(source, compileOptions)
  const zigExecutable = zig.compile(source, compileOptions)
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
    expect(zigRouting.reason).toBe('zig_requested')

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
  test('all public search limits match Zig', () => {
    const source = program('cyna_parity_search_limits', p => add(
      add(p.argument('a', Tensor.f32([2])), p.argument('b', Tensor.f32([2]))),
      p.argument('c', Tensor.f32([2])),
    ))
    for (const [name, value] of [
      ['structuralStates', 1],
      ['structuralExpansions', 1],
      ['completeEvaluations', 1],
      ['physicalWork', 1],
      ['readyWork', 1],
      ['maxPeakActiveBytes', 1024],
      ['maxInvocationScratchBytes', 0],
    ] as const) {
      runPaired(`search limit ${name}`, source, undefined, session => ({
        a: session.tensor([1, 2]),
        b: session.tensor([3, 4]),
        c: session.tensor([5, 6]),
      }), f32Tolerance, { searchLimits: { [name]: value } })
    }
  })

  test('reduction axes, extrema, and variance match Zig', () => {
    const operations = [
      ['sum', (x: any) => sum(x, -1, true)],
      ['mean', (x: any) => mean(x, 0)],
      ['min', (x: any) => min(x, -1)],
      ['max', (x: any) => max(x, 0, true)],
      ['argmax', (x: any) => argmax(x, -1)],
      ['variance', (x: any) => variance(x, -1)],
      ['std', (x: any) => std(x, 0)],
    ] as const
    for (const [name, operation] of operations) {
      const source = program(`cyna_parity_reduction_${name}`, p => operation(p.argument('x', Tensor.f32([2, 3]))))
      runPaired(`reduction ${name}`, source, undefined, session => ({
        x: session.tensor([[1, -2, 3], [4, 0, -5]], { dtype: 'f32' }),
      }))
    }
  })

  test('unary, cast, and normalization values match Zig', () => {
    const operations = [
      ['abs', (x: any) => abs(x)],
      ['clamp', (x: any) => clamp(x, -1, 2)],
      ['sqrt', (x: any) => sqrt(abs(x))],
      ['softmax', (x: any) => softmax(x, -1)],
      ['log_softmax', (x: any) => log_softmax(x, 0)],
      ['layer_norm', (x: any) => layer_norm(x, -1)],
      ['rms_norm', (x: any) => rms_norm(x, -1)],
      ['cast_f64', (x: any) => cast(x, 'f64')],
      ['cast_bf16', (x: any) => cast(x, 'bf16')],
    ] as const
    for (const [name, operation] of operations) {
      const source = program(`cyna_parity_unary_${name}`, p => operation(p.argument('x', Tensor.f32([2, 3]))))
      runPaired(`unary ${name}`, source, undefined, session => ({
        x: session.tensor([[1, -2, 3], [4, 0, -5]], { dtype: 'f32' }),
      }), name === 'cast_bf16' ? bf16Tolerance : f32Tolerance)
    }
  })

  test('arithmetic chains match Zig across storage dtypes', () => {
    for (const dtype of ['f64', 'bf16', 'i64', 'u32'] as const) {
      const spec = (Tensor as any)[dtype]([2, 3])
      const source = program(`cyna_parity_arithmetic_${dtype}`, p => add(
        p.argument('a', spec),
        mul(p.argument('b', spec), p.argument('c', spec)),
      ))
      const tolerance = dtype === 'f64' ? f64Tolerance : dtype === 'bf16' ? bf16Tolerance : f32Tolerance
      runPaired(`arithmetic ${dtype}`, source, undefined, session => ({
        a: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype }),
        b: session.tensor([[2, 3, 4], [1, 2, 3]], { dtype }),
        c: session.tensor([[5, 6, 7], [8, 9, 10]], { dtype }),
      }), tolerance)
    }
  })

  test('structural copies and views match Zig', () => {
    const source = program('cyna_parity_structural', p => {
      const x = p.argument('x', Tensor.f32([2, 3]))
      const view = transpose(x, [1, 0])
      const section = slice(view, [{ start: 1, stop: 3 }, { start: 0, stop: 2 }])
      return [view, contiguous(view), reshape(contiguous(view), [2, 3]), section,
        cat([section, section], 1), stack([section, section], 0), permute(x, [1, 0])]
    })
    runPaired('structural copies and views', source, undefined, session => ({
      x: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
    }))
  })

  test('nonzero-offset slices match Zig when consumed by CPU operations', () => {
    const source = program('cyna_parity_offset_view_consumers', p => {
      const left = p.argument('left', Tensor.f32([3, 5]))
      const matrix = p.argument('matrix', Tensor.f32([3, 4]))
      const stridedMatrix = p.argument('stridedMatrix', Tensor.f32([3, 5]))
      return [
        add(
          matmul(
            contiguous(slice(left, [{ start: 1, stop: 3 }, { start: 1, stop: 5 }])),
            p.argument('right', Tensor.f32([4, 2])),
          ),
          p.argument('bias', Tensor.f32([2])),
        ),
        add(
          slice(transpose(matrix, [1, 0]), [
            { start: 0, stop: 3 },
            { start: 1, stop: 3 },
          ]),
          p.argument('transposeBias', Tensor.f32([2])),
        ),
        add(
          gelu(slice(stridedMatrix, [
            { start: 0, stop: 3 },
            { start: 1, stop: 5, step: 2 },
          ])),
          p.argument('stridedBias', Tensor.f32([2])),
        ),
        add(
          p.argument('singletonLeft', Tensor.f32([2, 1])),
          p.argument('singletonRight', Tensor.f32([1, 4])),
        ),
      ]
    })
    runPaired('nonzero-offset view consumers', source, undefined, session => ({
      left: session.tensor([
        [99, 99, 99, 99, 99],
        [99, 1, 2, 3, 4],
        [99, 5, 6, 7, 8],
      ], { dtype: 'f32' }),
      matrix: session.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ], { dtype: 'f32' }),
      stridedMatrix: session.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
      ], { dtype: 'f32' }),
      right: session.tensor([
        [1, 0],
        [0, 1],
        [1, 1],
        [2, -1],
      ], { dtype: 'f32' }),
      bias: session.tensor([0.5, -0.5], { dtype: 'f32' }),
      transposeBias: session.tensor([0, 1], { dtype: 'f32' }),
      stridedBias: session.tensor([1, -1], { dtype: 'f32' }),
      singletonLeft: session.tensor([[1], [2]], { dtype: 'f32' }),
      singletonRight: session.tensor([[3, 4, 5, 6]], { dtype: 'f32' }),
    }))
  })

  test('indexed and dense cross entropy match Zig', () => {
    const indexed = program('cyna_parity_indexed_loss', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])), p.argument('labels', Tensor.i64([2])),
    ))
    runPaired('indexed cross entropy', indexed, undefined, session => ({
      logits: session.tensor([[2, 0, -1], [-2, 1, 3]], { dtype: 'f32' }),
      labels: session.tensor([0, 2], { dtype: 'i64' }),
    }))
    const dense = program('cyna_parity_dense_loss', p => dense_cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])), p.argument('targets', Tensor.f32([2, 3])),
    ))
    runPaired('dense cross entropy', dense, undefined, session => ({
      logits: session.tensor([[2, 0, -1], [-2, 1, 3]], { dtype: 'f32' }),
      targets: session.tensor([[1, 0, 0], [0, 0.25, 0.75]], { dtype: 'f32' }),
    }))
  })

  test('extreme softmax values and view normalization match Zig', () => {
    for (const [name, operation] of [
      ['softmax', (x: any) => softmax(x, -1)],
      ['log_softmax', (x: any) => log_softmax(x, -1)],
      ['layer_norm', (x: any) => layer_norm(x, -1)],
    ] as const) {
      const source = program(`cyna_parity_extreme_${name}`, p => operation(
        transpose(p.argument('x', Tensor.f32([3, 2])), [1, 0]),
      ))
      runPaired(`extreme ${name}`, source, undefined, session => ({
        x: session.tensor([[1000, -1000], [999, -999], [998, -998]], { dtype: 'f32' }),
      }))
    }
  })

  for (const dtype of ['f32', 'f64', 'bf16', 'i64', 'u32'] as const) {
    const spec = (shape: number[]) => (Tensor as any)[dtype](shape)
    const tolerance = dtype === 'f64' ? f64Tolerance : dtype === 'bf16' ? bf16Tolerance : f32Tolerance
    test(`${dtype} selection and indexing match Zig`, () => {
      const select = program(`cyna_parity_${dtype}_where`, p => where(
        p.argument('mask', Tensor.i64([2, 1])),
        p.argument('a', spec([2, 3])), p.argument('b', spec([1, 3])),
      ))
      runPaired(`${dtype} where`, select, undefined, session => ({
        mask: session.tensor([[1], [0]], { dtype: 'i64' }),
        a: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype }),
        b: session.tensor([[7, 8, 9]], { dtype }),
      }), tolerance)

      const selectIndex = program(`cyna_parity_${dtype}_index_select`, p => index_select(
        p.argument('x', spec([3, 2])), 0, p.argument('indices', Tensor.i64([3])),
      ))
      runPaired(`${dtype} index select`, selectIndex, undefined, session => ({
        x: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype }),
        indices: session.tensor([2, 0, 2], { dtype: 'i64' }),
      }), tolerance)

      const scatter = program(`cyna_parity_${dtype}_scatter`, p => scatter_add(
        p.argument('base', spec([3, 2])),
        p.argument('indices', Tensor.i64([2, 2])),
        p.argument('updates', spec([2, 2])), 0,
      ))
      runPaired(`${dtype} scatter-add`, scatter, undefined, session => ({
        base: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype }),
        indices: session.tensor([[1, 1], [1, 2]], { dtype: 'i64' }),
        updates: session.tensor([[2, 3], [4, 5]], { dtype }),
      }), tolerance)

      if (dtype === 'f32' || dtype === 'bf16') {
        const rank = program(`cyna_parity_${dtype}_topk`, p => topk(
          p.argument('x', spec([2, 4])), 2, 1,
        ))
        runPaired(`${dtype} top-k`, rank, undefined, session => ({
          x: session.tensor([[3, 5, 5, 1], [2, 2, 4, 3]], { dtype }),
        }), tolerance)
      }
    })
  }

  test('selection and indexing preserve transposed input values', () => {
    const select = program('cyna_parity_view_where', p => where(
      p.argument('mask', Tensor.i64([2, 1])),
      transpose(p.argument('a', Tensor.f32([3, 2])), [1, 0]),
      p.argument('b', Tensor.f32([1, 3])),
    ))
    runPaired('view where', select, undefined, session => ({
      mask: session.tensor([[1], [0]], { dtype: 'i64' }),
      a: session.tensor([[1, 4], [2, 5], [3, 6]], { dtype: 'f32' }),
      b: session.tensor([[7, 8, 9]], { dtype: 'f32' }),
    }))

    const rank = program('cyna_parity_view_topk', p => topk(
      transpose(p.argument('x', Tensor.f32([4, 2])), [1, 0]), 2, 1,
    ))
    runPaired('view top-k', rank, undefined, session => ({
      x: session.tensor([[3, 2], [5, 2], [5, 4], [1, 3]], { dtype: 'f32' }),
    }))
  })

  test('out-of-range indices fail on both routes and leave executables reusable', () => {
    const source = program('cyna_parity_invalid_index', p => index_select(
      p.argument('x', Tensor.f32([3, 2])), 0,
      p.argument('indices', Tensor.i64([2])),
    ))
    for (const cpuExecution of ['auto', 'zig'] as const) {
      const session = new Session({ device: 'cpu', cpuExecution })
      const executable = session.compile(source)
      const x = session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' })
      const bad = session.tensor([0, 3], { dtype: 'i64' })
      const negative = session.tensor([-1, 0], { dtype: 'i64' })
      const good = session.tensor([2, 0], { dtype: 'i64' })
      try {
        expect(() => executable.run({ x, indices: bad })).toThrow()
        expect(() => executable.run({ x, indices: negative })).toThrow()
        const result = executable.run({ x, indices: good }) as Tensor
        expect(result.to_array()).toEqual([[5, 6], [1, 2]])
        result.dispose()
      } finally {
        good.dispose(); negative.dispose(); bad.dispose(); x.dispose(); executable.dispose(); session.dispose()
      }
    }
  })

  test('indexing failures agree across gather, embedding, scatter, and one-hot', () => {
    const cases = [
      {
        name: 'gather',
        source: program('cyna_invalid_gather', p => gather(p.argument('x', Tensor.f32([3, 2])), 0, p.argument('indices', Tensor.i64([2, 2])))),
        inputs: (session: Session, invalid: boolean) => ({
          x: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
          indices: session.tensor(invalid ? [[0, 3], [1, 2]] : [[0, 2], [1, 2]], { dtype: 'i64' }),
        }),
      },
      {
        name: 'embedding',
        source: program('cyna_invalid_embedding', p => embedding(p.argument('x', Tensor.f32([3, 2])), p.argument('indices', Tensor.i64([2])))),
        inputs: (session: Session, invalid: boolean) => ({
          x: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
          indices: session.tensor(invalid ? [-1, 2] : [0, 2], { dtype: 'i64' }),
        }),
      },
      {
        name: 'one-hot',
        source: program('cyna_invalid_one_hot', p => one_hot(p.argument('indices', Tensor.i64([2])), 3)),
        inputs: (session: Session, invalid: boolean) => ({
          indices: session.tensor(invalid ? [0, 3] : [0, 2], { dtype: 'i64' }),
        }),
      },
      {
        name: 'scatter-add',
        source: program('cyna_invalid_scatter', p => scatter_add(
          p.argument('x', Tensor.f32([3, 2])), p.argument('indices', Tensor.i64([2, 2])),
          p.argument('updates', Tensor.f32([2, 2])), 0,
        )),
        inputs: (session: Session, invalid: boolean) => ({
          x: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
          indices: session.tensor(invalid ? [[0, 3], [1, 2]] : [[0, 2], [1, 2]], { dtype: 'i64' }),
          updates: session.tensor([[1, 1], [1, 1]], { dtype: 'f32' }),
        }),
      },
      {
        name: 'scatter-add-columns',
        source: program('cyna_invalid_scatter_columns', p => scatter_add(
          p.argument('x', Tensor.f32([2, 3])), p.argument('indices', Tensor.i64([2, 2])),
          p.argument('updates', Tensor.f32([2, 2])), 1,
        )),
        inputs: (session: Session, invalid: boolean) => ({
          x: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
          indices: session.tensor(invalid ? [[0, 3], [1, 2]] : [[0, 2], [1, 2]], { dtype: 'i64' }),
          updates: session.tensor([[1, 1], [1, 1]], { dtype: 'f32' }),
        }),
      },
    ]
    for (const item of cases) for (const cpuExecution of ['auto', 'zig'] as const) {
      const session = new Session({ device: 'cpu', cpuExecution })
      const executable = session.compile(item.source)
      const bad = item.inputs(session, true)
      const good = item.inputs(session, false)
      try {
        expect(() => executable.run(bad)).toThrow()
        for (const result of outputs(executable.run(good) as Tensor | Tensor[])) result.dispose()
      } finally {
        for (const tensor of Object.values(bad)) tensor.dispose()
        for (const tensor of Object.values(good)) tensor.dispose()
        executable.dispose(); session.dispose()
      }
    }
  })

  test('broadcast comparison matches Zig Session', () => {
    const source = program('cyna_parity_broadcast_comparison', p => eq(
      p.argument('left', Tensor.f32([2, 3])),
      p.argument('right', Tensor.f32([1, 3])),
    ))
    runPaired('broadcast comparison', source, undefined, session => ({
      left: session.tensor([[1, 2, 3], [3, 2, 1]], { dtype: 'f32' }),
      right: session.tensor([[1, 4, 3]], { dtype: 'f32' }),
    }))
    for (const [name, operation] of [['eq', eq], ['lt', lt], ['gt', gt]] as const) {
      const columns = program(`cyna_parity_broadcast_comparison_columns_${name}`, p => operation(
        p.argument('left', Tensor.f32([1, 3])),
        p.argument('right', Tensor.f32([2, 1])),
      ))
      runPaired(`broadcast ${name} columns`, columns, undefined, session => ({
        left: session.tensor([[1, 2, 3]], { dtype: 'f32' }),
        right: session.tensor([[1], [2]], { dtype: 'f32' }),
      }))
    }
  })

  test('f32 comparison matches Zig for NaN and signed zero', () => {
    const leftBytes = new Uint32Array([
      0x7fc00000, 0x80000000, 0, 0x3f800000,
    ]).buffer
    const rightBytes = new Uint32Array([
      0x3f800000, 0, 0x80000000, 0x7fc00000,
    ]).buffer
    for (const [name, operation] of [['eq', eq], ['lt', lt], ['gt', gt]] as const) {
      const source = program(`cyna_parity_f32_${name}_special_values`, p => operation(
        p.argument('left', Tensor.f32([2, 2])),
        p.argument('right', Tensor.f32([2, 2])),
      ))
      runPaired(`f32 ${name} special values`, source, undefined, session => ({
        left: session.fromBytes(leftBytes, Tensor.f32([2, 2])),
        right: session.fromBytes(rightBytes, Tensor.f32([2, 2])),
      }))
    }
  })

  test('broadcast where matches Zig Session', () => {
    const source = program('cyna_parity_broadcast_where', p => where(
      p.argument('mask', Tensor.i64([2, 1])),
      p.argument('when_true', Tensor.f32([2, 3])),
      p.argument('when_false', Tensor.f32([1, 3])),
    ))
    runPaired('broadcast where', source, undefined, session => ({
      mask: session.tensor([[1], [0]], { dtype: 'i64' }),
      when_true: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
      when_false: session.tensor([[7, 8, 9]], { dtype: 'f32' }),
    }))
    const columns = program('cyna_parity_broadcast_where_columns', p => where(
      p.argument('mask', Tensor.i64([1, 3])),
      p.argument('when_true', Tensor.f32([1, 3])),
      p.argument('when_false', Tensor.f32([2, 1])),
    ))
    runPaired('broadcast where columns', columns, undefined, session => ({
      mask: session.tensor([[1, 0, 1]], { dtype: 'i64' }),
      when_true: session.tensor([[10, 20, 30]], { dtype: 'f32' }),
      when_false: session.tensor([[1], [2]], { dtype: 'f32' }),
    }))
  })

  test('broadcast masked fill matches Zig Session', () => {
    const source = program('cyna_parity_broadcast_masked_fill', p => masked_fill(
      p.argument('value', Tensor.f32([2, 3])),
      p.argument('mask', Tensor.i64([2, 1])),
      -7,
    ))
    runPaired('broadcast masked fill', source, undefined, session => ({
      value: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
      mask: session.tensor([[0], [2]], { dtype: 'i64' }),
    }))
  })

  test('embedding with repeated indices matches Zig Session', () => {
    const source = program('cyna_parity_embedding', p => embedding(
      p.argument('table', Tensor.f32([4, 3])),
      p.argument('indices', Tensor.i64([2, 2])),
    ))
    runPaired('embedding', source, undefined, session => ({
      table: session.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], { dtype: 'f32' }),
      indices: session.tensor([[2, 0], [2, 3]], { dtype: 'i64' }),
    }))
  })

  test('index select with repeated indices matches Zig Session', () => {
    const source = program('cyna_parity_index_select', p => index_select(
      p.argument('value', Tensor.f32([3, 2])), 0,
      p.argument('indices', Tensor.i64([3])),
    ))
    runPaired('index select', source, undefined, session => ({
      value: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
      indices: session.tensor([2, 0, 2], { dtype: 'i64' }),
    }))
  })

  test('gather along a nonfinal axis matches Zig Session', () => {
    const source = program('cyna_parity_gather', p => gather(
      p.argument('value', Tensor.f32([3, 2])), 0,
      p.argument('indices', Tensor.i64([2, 2])),
    ))
    runPaired('gather', source, undefined, session => ({
      value: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
      indices: session.tensor([[2, 0], [1, 2]], { dtype: 'i64' }),
    }))
  })

  test('scatter add with duplicate indices matches Zig Session', () => {
    const source = program('cyna_parity_scatter_add', p => scatter_add(
      p.argument('base', Tensor.f32([3, 2])),
      p.argument('indices', Tensor.i64([2, 2])),
      p.argument('updates', Tensor.f32([2, 2])), 0,
    ))
    runPaired('scatter add', source, undefined, session => ({
      base: session.tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'f32' }),
      indices: session.tensor([[1, 1], [1, 2]], { dtype: 'i64' }),
      updates: session.tensor([[2, 3], [4, 5]], { dtype: 'f32' }),
    }))
    const columns = program('cyna_parity_scatter_add_columns', p => scatter_add(
      p.argument('base', Tensor.f32([2, 3])),
      p.argument('indices', Tensor.i64([2, 2])),
      p.argument('updates', Tensor.f32([2, 2])), 1,
    ))
    runPaired('scatter add columns', columns, undefined, session => ({
      base: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
      indices: session.tensor([[2, 2], [0, 1]], { dtype: 'i64' }),
      updates: session.tensor([[7, 8], [9, 10]], { dtype: 'f32' }),
    }))
  })

  test('top-k ties and both outputs match Zig Session', () => {
    const source = program('cyna_parity_topk', p => topk(
      p.argument('value', Tensor.f32([2, 4])), 2, 1,
    ))
    runPaired('top-k', source, undefined, session => ({
      value: session.tensor([[3, 5, 5, 1], [2, 2, 4, 3]], { dtype: 'f32' }),
    }))
  })

  test('one-hot matches Zig Session', () => {
    const source = program('cyna_parity_one_hot', p => one_hot(
      p.argument('indices', Tensor.i64([2, 2])), 4,
    ))
    runPaired('one-hot', source, undefined, session => ({
      indices: session.tensor([[1, 3], [0, 1]], { dtype: 'i64' }),
    }))
  })

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

  test('transposed f64 matrix multiplication matches Zig Session', () => {
    const source = program('cyna_parity_transposed_f64_matmul', p => matmul(
      p.argument('x', Tensor.f64([2, 3])),
      transpose(p.argument('weight', Tensor.f64([4, 3])), [1, 0]),
    ))
    runPaired('transposed f64 matrix multiplication', source, undefined, session => ({
      x: session.tensor([[1, -2, 3], [4, 5, -6]], { dtype: 'f64' }),
      weight: session.tensor([[1, 2, 3], [-4, 5, 6], [7, -8, 9], [10, 11, -12]], { dtype: 'f64' }),
    }), f64Tolerance)
  })

  test('shared transposed right operand survives two matmul consumers', () => {
    const source = program('cyna_parity_shared_transposed_matmul', p => {
      const right = transpose(p.argument('weight', Tensor.f32([4, 3])), [1, 0])
      return [
        matmul(p.argument('first', Tensor.f32([2, 3])), right),
        matmul(p.argument('second', Tensor.f32([2, 3])), right),
      ]
    })
    runPaired('shared transposed right operand', source, undefined, session => ({
      first: session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }),
      second: session.tensor([[6, 5, 4], [3, 2, 1]], { dtype: 'f32' }),
      weight: session.tensor([[1, 2, 3], [-4, 5, 6], [7, -8, 9], [10, 11, -12]], { dtype: 'f32' }),
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

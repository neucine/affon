import {
  $formalCat,
  $formalOperation,
  $sessionOf,
  FormalTensor,
  Tensor,
  program,
  type ProgramDType,
  type SliceRange,
  type Tensor as EvaluatedTensor,
  type TensorSpec,
} from 'affon:_internal/compute/program'

type AnyTensor = EvaluatedTensor | FormalTensor
type Same<T extends AnyTensor> = T extends FormalTensor ? FormalTensor : EvaluatedTensor

function spec(value: EvaluatedTensor): TensorSpec {
  return Tensor.spec(value.dtype, value.shape, { axes: value.axes })
}
function evaluated(values: readonly EvaluatedTensor[]): { session: NonNullable<ReturnType<typeof $sessionOf>>; specs: TensorSpec[] } {
  if (!values.length) throw new TypeError('operation requires at least one tensor')
  const session = $sessionOf(values[0])
  if (!session) throw new TypeError('tensor is not owned by a Session')
  if (session.disposed) throw new Error('Session has been disposed')
  if (values.some(value => $sessionOf(value) !== session)) throw new TypeError('operation tensors must belong to the same Session')
  return { session, specs: values.map(spec) }
}
function immediate(name: string, values: readonly EvaluatedTensor[], author: (args: FormalTensor[]) => FormalTensor): EvaluatedTensor {
  const { session, specs } = evaluated(values)
  const identifier = `immediate_${name}`.replace(/[^A-Za-z0-9_]/g, '_')
  const source = program(identifier, p => author(specs.map((value, index) => p.argument(`input_${index}`, value))))
  return session.compile(source).run(Object.fromEntries(values.map((value, index) => [`input_${index}`, value]))) as EvaluatedTensor
}
function unary<T extends AnyTensor>(name: string, value: T, formal: (value: FormalTensor) => FormalTensor): Same<T> {
  return (value instanceof FormalTensor ? formal(value) : immediate(name, [value], args => formal(args[0]))) as Same<T>
}
function binary<T extends AnyTensor>(name: string, left: T, right: T, formal: (left: FormalTensor, right: FormalTensor) => FormalTensor): Same<T> {
  if ((left instanceof FormalTensor) !== (right instanceof FormalTensor)) throw new TypeError(`${name} cannot mix formal and evaluated tensors`)
  return (left instanceof FormalTensor ? formal(left, right as FormalTensor) : immediate(name, [left, right as EvaluatedTensor], args => formal(args[0], args[1]))) as Same<T>
}

export const add = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('add', a, b, (x, y) => $formalOperation('add', [x, y]))
export const sub = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('sub', a, b, (x, y) => $formalOperation('sub', [x, y]))
export const mul = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('mul', a, b, (x, y) => $formalOperation('mul', [x, y]))
export const div = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('div', a, b, (x, y) => $formalOperation('div', [x, y]))
export const matmul = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('matmul', a, b, (x, y) => $formalOperation('matmul', [x, y]))
export const dot = <T extends AnyTensor>(a: T, b: T): Same<T> => binary('dot', a, b, (x, y) => $formalOperation('dot', [x, y]))
export const neg = <T extends AnyTensor>(x: T): Same<T> => unary('neg', x, value => $formalOperation('neg', [value]))
export const abs = <T extends AnyTensor>(x: T): Same<T> => unary('abs', x, value => $formalOperation('abs', [value]))
export const exp = <T extends AnyTensor>(x: T): Same<T> => unary('exp', x, value => $formalOperation('exp', [value]))
export const log = <T extends AnyTensor>(x: T): Same<T> => unary('log', x, value => $formalOperation('log', [value]))
export const sqrt = <T extends AnyTensor>(x: T): Same<T> => unary('sqrt', x, value => $formalOperation('sqrt', [value]))
export const relu = <T extends AnyTensor>(x: T): Same<T> => unary('relu', x, value => $formalOperation('relu', [value]))
export const sigmoid = <T extends AnyTensor>(x: T): Same<T> => unary('sigmoid', x, value => $formalOperation('sigmoid', [value]))
export const silu = <T extends AnyTensor>(x: T): Same<T> => unary('silu', x, value => $formalOperation('silu', [value]))
export const tanh = <T extends AnyTensor>(x: T): Same<T> => unary('tanh', x, value => $formalOperation('tanh', [value]))
export const erf = <T extends AnyTensor>(x: T): Same<T> => unary('erf', x, value => $formalOperation('erf', [value]))
export const gelu = <T extends AnyTensor>(x: T): Same<T> => unary('gelu', x, value => $formalOperation('gelu', [value]))
export const softmax = <T extends AnyTensor>(x: T, axis: number): Same<T> => unary(`softmax_${axis}`, x, value => $formalOperation('softmax', [value], [axis]))
export const sum = <T extends AnyTensor>(x: T, axis?: number, keep_dims = false): Same<T> => unary(`sum_${axis}_${keep_dims}`, x, value => $formalOperation('sum', [value], [axis, keep_dims]))
export const mean = <T extends AnyTensor>(x: T, axis?: number, keep_dims = false): Same<T> => unary(`mean_${axis}_${keep_dims}`, x, value => $formalOperation('mean', [value], [axis, keep_dims]))
export const reshape = <T extends AnyTensor>(x: T, shape: readonly number[], axes?: readonly string[]): Same<T> => unary(`reshape_${shape}`, x, value => $formalOperation('reshape', [value], [shape, axes]))
export const contiguous = <T extends AnyTensor>(x: T): Same<T> => unary('contiguous', x, value => $formalOperation('contiguous', [value]))
export const transpose = <T extends AnyTensor>(x: T, permutation?: readonly number[]): Same<T> => unary(`transpose_${permutation}`, x, value => $formalOperation('transpose', [value], [permutation]))
export const cast = <T extends AnyTensor>(x: T, dtype: ProgramDType): Same<T> => unary(`cast_${dtype}`, x, value => $formalOperation('cast', [value], [dtype]))
export const slice = <T extends AnyTensor>(x: T, ranges: readonly SliceRange[]): Same<T> => unary(`slice_${JSON.stringify(ranges)}`, x, value => $formalOperation('slice', [value], [ranges]))
export const squeeze = <T extends AnyTensor>(x: T, axis?: number): Same<T> => unary(`squeeze_${axis}`, x, value => $formalOperation('squeeze', [value], [axis]))
export const unsqueeze = <T extends AnyTensor>(x: T, axis: number, name?: string): Same<T> => unary(`unsqueeze_${axis}_${name}`, x, value => $formalOperation('unsqueeze', [value], [axis, name]))
export const layer_norm = <T extends AnyTensor>(x: T, axis: number, epsilon = 1e-5): Same<T> => unary(`layer_norm_${axis}_${epsilon}`, x, value => $formalOperation('layer_norm', [value], [axis, epsilon]))
export const masked_fill = <T extends AnyTensor>(x: T, mask: T, value: number): Same<T> => binary(`masked_fill_${value}`, x, mask, (left, right) => $formalOperation('masked_fill', [left, right], [value]))
export const embedding = <T extends AnyTensor>(table: T, indices: T): Same<T> => binary('embedding', table, indices, (left, right) => $formalOperation('embedding', [left, right]))
export const index_select = <T extends AnyTensor>(x: T, axis: number, indices: T): Same<T> => binary(`index_select_${axis}`, x, indices, (left, right) => $formalOperation('index_select', [left, right], [axis]))

export function cat<T extends AnyTensor>(values: readonly T[], axis = 0): Same<T> {
  if (!values.length) throw new TypeError('cat requires at least one tensor')
  const formal = values[0] instanceof FormalTensor
  if (values.some(value => (value instanceof FormalTensor) !== formal)) throw new TypeError('cat cannot mix formal and evaluated tensors')
  return (formal
    ? $formalCat(values as readonly FormalTensor[], axis)
    : immediate(`cat_${axis}_${values.length}`, values as readonly EvaluatedTensor[], args => $formalCat(args, axis))) as Same<T>
}

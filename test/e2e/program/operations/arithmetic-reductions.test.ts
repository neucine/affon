// Reference cases migrated from affon-legacy/test/parity/compute. The expected
// values are independent arithmetic fixtures rather than runtime-generated values.
import { abs, add, argmax, argmin, div, exp, log, matmul, max, mean, min, mul, neg, relu, reshape, sigmoid, softmax, sqrt, std, sub, sum, tanh, transpose, variance } from 'affon:ops'
import { Session } from 'affon:compute'
import { describe, expect, test } from 'std:test'

function close(actual: unknown, expected: unknown, tolerance = 1e-5): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true)
    expect((actual as unknown[]).length).toBe(expected.length)
    expected.forEach((entry, index) => close((actual as unknown[])[index], entry, tolerance))
  } else {
    expect(Math.abs((actual as number) - (expected as number)) <= tolerance).toBe(true)
  }
}

describe('Compute reference values on evaluated tensors', () => {
  test('broadcast arithmetic and unary operations', () => {
    const session = new Session({ device: 'cpu' })
    const x = session.tensor([[2, 4, 8], [16, 32, 64]])
    const y = session.tensor([[1, 2, 4]])
    const zero = session.tensor([[0]])
    const cases: [string, () => any, unknown][] = [
      ['add', () => add(x, y), [[3, 6, 12], [17, 34, 68]]],
      ['sub', () => sub(x, y), [[1, 2, 4], [15, 30, 60]]],
      ['mul', () => mul(x, y), [[2, 8, 32], [16, 64, 256]]],
      ['div', () => div(x, y), [[2, 2, 2], [16, 16, 16]]],
      ['neg', () => neg(y), [[-1, -2, -4]]],
      ['abs', () => abs(neg(y)), [[1, 2, 4]]],
      ['sqrt', () => sqrt(y), [[1, Math.SQRT2, 2]]],
      ['log-exp', () => log(exp(y)), [[1, 2, 4]]],
      ['relu', () => relu(neg(y)), [[0, 0, 0]]],
      ['sigmoid', () => sigmoid(zero), [[0.5]]],
      ['tanh', () => tanh(zero), [[0]]],
    ]
    for (const [name, run, expected] of cases) {
      const value = run()
      try { close(value.to_array(), expected) } catch (error) { throw new Error(`${name}: ${error}`) }
      value.dispose()
    }
    x.dispose(); y.dispose(); zero.dispose(); session.dispose()
  })

  test('axis reductions and shape routing', () => {
    const session = new Session({ device: 'cpu' })
    const x = session.tensor([[1, 2, 3], [4, 5, 6]])
    const zeros = session.tensor([[0, 0]])
    const cases: [string, () => any, unknown][] = [
      ['sum', () => sum(x, 1), [6, 15]],
      ['mean', () => mean(x, 0), [2.5, 3.5, 4.5]],
      ['min', () => min(x, 1), [1, 4]],
      ['max', () => max(x, 1), [3, 6]],
      ['argmin', () => argmin(x, 1), [0, 0]],
      ['argmax', () => argmax(x, 1), [2, 2]],
      ['variance', () => variance(x, 1), [2 / 3, 2 / 3]],
      ['std', () => std(x, 1), [Math.sqrt(2 / 3), Math.sqrt(2 / 3)]],
      ['reshape', () => reshape(x, [3, 2]), [[1, 2], [3, 4], [5, 6]]],
      ['transpose', () => transpose(x, [1, 0]), [[1, 4], [2, 5], [3, 6]]],
      ['softmax', () => softmax(zeros, 1), [[0.5, 0.5]]],
    ]
    for (const [name, run, expected] of cases) {
      const value = run()
      try { close(value.to_array(), expected) } catch (error) { throw new Error(`${name}: ${error}`) }
      value.dispose()
    }
    x.dispose(); zeros.dispose(); session.dispose()
  })

  test('matrix multiplication reference', () => {
    const session = new Session({ device: 'cpu' })
    const left = session.tensor([[1, 2, 3], [4, 5, 6]])
    const right = session.tensor([[7, 8], [9, 10], [11, 12]])
    const result = matmul(left, right)
    close(result.to_array(), [[58, 64], [139, 154]])
    result.dispose(); left.dispose(); right.dispose(); session.dispose()
  })
})

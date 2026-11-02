import { expect, test } from 'std:test'
import { getEnv } from 'std:process'
import { Session, Tensor, program } from 'affon:compute'
import { add, matmul, slice, transpose } from 'affon:ops'

test('BF16 matmul handles contiguous, transposed, broadcast, and sliced layouts', () => {
  const session = new Session({device: getEnv('AFFON_TEST_CUDA') === '1' ? 'cuda' : 'cpu'})
  try {
    const left = session.tensor([[1, 2, 3], [4, 5, 6]], {dtype: 'bf16'})
    const right = session.tensor([[7, 8], [9, 10], [11, 12]], {dtype: 'bf16'})
    const rightTransposed = session.tensor([[7, 9, 11], [8, 10, 12]], {dtype: 'bf16'})
    const rightPadded = session.tensor([[7, 0, 8, 0], [9, 0, 10, 0], [11, 0, 12, 0]], {dtype: 'bf16'})
    const batchedLeft = session.tensor([[[1, 2, 3]], [[4, 5, 6]]], {dtype: 'bf16'})
    const batchedRight = session.tensor([[[7, 8], [9, 10], [11, 12]]], {dtype: 'bf16'})
    const view = transpose(rightTransposed, [1, 0])
    const sliced = slice(rightPadded, [{start: 0, stop: 3}, {start: 0, stop: 4, step: 2}])
    const results = [matmul(left, right), matmul(left, view), matmul(left, sliced), matmul(batchedLeft, batchedRight)]
    try {
      for (const result of results.slice(0, 3)) expect(result.to_array()).toEqual([[58, 64], [139, 154]])
      expect(results[3].to_array()).toEqual([[[58, 64]], [[139, 154]]])
    } finally {
      results.forEach(value => value.dispose())
      sliced.dispose(); view.dispose(); batchedRight.dispose(); batchedLeft.dispose()
      rightPadded.dispose(); rightTransposed.dispose(); right.dispose(); left.dispose()
    }
  } finally { session.dispose() }
})

test('Metal tiled BF16 matmul consumes a transposed view inside a Program', () => {
  const source = program('bf16_transposed_tiled_resident', p => {
    const left = p.argument('left', Tensor.bf16([32, 64]))
    const right = transpose(p.argument('right', Tensor.bf16([32, 64])), [1, 0])
    const product = matmul(left, right)
    return add(product, product)
  })
  const session = new Session({ device: 'metal' })
  const executable = session.compile(source)
  const values = Array.from({ length: 32 }, () => Array(64).fill(1))
  const left = session.tensor(values, { dtype: 'bf16' })
  const right = session.tensor(values, { dtype: 'bf16' })
  const output = executable.run({ left, right })
  expect(output.to_array()).toEqual(Array.from({ length: 32 }, () => Array(32).fill(128)))
  output.dispose(); right.dispose(); left.dispose(); executable.dispose(); session.dispose()
})

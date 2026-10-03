import { test, expect, afterAll } from 'std:test'
import { Session } from 'affon:compute'
import { prepare_conv, prepare_pad } from '../src/spatial.ts'

const session = new Session({ device: 'cpu' })
const tensor = (values: any) => session.tensor(values) as any
afterAll(() => session.dispose())

test('canonical spatial padding preserves non-finite interior values and zero borders', () => {
  const run = prepare_pad([1, 1, 1, 2], [1, 1, 1, 1], 'cpu')
  const output = run(tensor([[[[Infinity, NaN]]]]))
  const values = (output.to_array() as number[]).flat(Infinity) as number[]
  expect(values.length).toBe(12)
  expect(values[5]).toBe(Infinity)
  expect(Number.isNaN(values[6])).toBe(true)
  expect(values.every((value, index) => index === 5 || index === 6 || value === 0)).toBe(true)
  output.dispose()
})

test('canonical grouped convolution supports asymmetric padding, stride, dilation, and bias', () => {
  const input = tensor([[[[1, 2, 3], [4, 5, 6]]], [[[2, 1, 0], [1, 2, 3]]]])
  const weight = tensor([[[[1, -1], [2, 0]]]])
  const bias = tensor([0.5])
  const run = prepare_conv([2, 1, 2, 3], weight, bias, {
    kernel: [2, 2], strides: [1, 1], dilations: [1, 1], pads: [1, 0, 0, 1], group: 1,
  }, 'cpu')
  const output = run(input)
  expect(output.shape).toEqual([2, 1, 2, 3])
  const actual = (output.to_array() as number[]).flat(Infinity) as number[]
  const expected = [2.5, 4.5, 6.5, 7.5, 9.5, 15.5, 4.5, 2.5, 0.5, 3.5, 5.5, 6.5]
  expect(actual.every((value, index) => Math.abs(value - expected[index]) < 1e-6)).toBe(true)
  output.dispose(); input.dispose(); weight.dispose(); bias.dispose()
})

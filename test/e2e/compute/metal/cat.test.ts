import { test, expect, values } from 'std:test'
import { tensor, reshape, cat, transpose } from 'affon:compute'

test('Metal batched concatenation preserves unequal chunks across outer rows and dtypes', () => {
  for (const dtype of ['f32', 'i64'] as const) {
    for (const axis of [0, 1, 2, 3]) {
      const shapes = [
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
      ]
      shapes[1][axis] = 1
      shapes[2][axis] = 2
      const inputs = shapes.map((shape, n) =>
        reshape(
          tensor(
            Array.from(
              { length: shape.reduce((a, b) => a * b, 1) },
              (_, i) => n * 1000 + i,
            ),
            { dtype },
          ),
          shape,
        ),
      )
      const expected = values(cat(inputs, axis))
      const actual = cat(
        inputs.map((x) => x.to('metal')),
        axis,
      )
      expect(actual.device).toBe('metal')
      expect(values(actual.to('cpu'))).toEqual(expected)
    }
  }
})

test('Metal concatenation materializes strided inputs correctly', () => {
  const x = tensor(
    [
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [
        [7, 8, 9],
        [10, 11, 12],
      ],
    ],
    { dtype: 'f32' },
  )
  const cpu = transpose(x, 0, 2),
    metal = transpose(x.to('metal'), 0, 2)
  for (const axis of [0, 1, 2])
    expect(values(cat([metal, metal], axis).to('cpu'))).toEqual(
      values(cat([cpu, cpu], axis)),
    )
})

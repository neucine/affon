// Bilinear RGB8 resampling with Pillow's sampling/rounding conventions.
// Reference: Pillow 12.0.0 src/libImaging/Resample.c. This is an experimental
// implementation; other filters, color modes and crop boxes are not supported.
export function resize_rgb(rgb: number[][][], height: number, width: number): number[][][] {
  const sourceHeight = rgb.length, sourceWidth = rgb[0]?.length ?? 0
  if (![height, width, sourceHeight, sourceWidth].every(n => Number.isInteger(n) && n > 0)) throw new Error('Resize dimensions must be positive integers')
  const precision = 2 ** 22
  function weights(source: number, destination: number) {
    const scale = source / destination, support = Math.max(1, scale)
    return Array.from({ length: destination }, (_, index) => {
      const center = (index + 0.5) * scale
      const start = Math.max(0, Math.trunc(center - support + 0.5))
      const end = Math.min(source, Math.trunc(center + support + 0.5))
      const values = Array.from({ length: end - start }, (_, i) => Math.max(0, 1 - Math.abs((start + i - center + 0.5) / support)))
      const total = values.reduce((a, b) => a + b, 0)
      return { start, values: values.map(value => Math.floor(value / total * precision + 0.5)) }
    })
  }
  const horizontal = weights(sourceWidth, width), vertical = weights(sourceHeight, height)
  const rounded = (value: number) => Math.max(0, Math.min(255, Math.floor(value / precision + 0.5)))
  const rows = width === sourceWidth ? rgb : rgb.map(row => horizontal.map(({ start, values }) =>
    [0, 1, 2].map(c => rounded(values.reduce((sum, value, i) => sum + value * row[start + i][c], 0)))))
  if (height === sourceHeight) return rows
  return vertical.map(({ start, values }) => Array.from({ length: width }, (_, x) =>
    [0, 1, 2].map(c => rounded(values.reduce((sum, value, i) => sum + value * rows[start + i][x][c], 0)))))
}

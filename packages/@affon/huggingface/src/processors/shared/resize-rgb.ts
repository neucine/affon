// Bilinear RGB8 resampling with Pillow's sampling/rounding conventions.
// Reference: Pillow 12.0.0 src/libImaging/Resample.c. This is an experimental
// implementation; other filters, color modes and crop boxes are not supported.
export function resize_rgb(
  rgb: number[][][],
  height: number,
  width: number,
): number[][][] {
  const sourceHeight = rgb.length,
    sourceWidth = rgb[0]?.length ?? 0
  if (
    ![height, width, sourceHeight, sourceWidth].every(
      (n) => Number.isInteger(n) && n > 0,
    )
  )
    throw new Error('Resize dimensions must be positive integers')
  const precision = 2 ** 22
  function weights(source: number, destination: number) {
    const scale = source / destination,
      support = Math.max(1, scale)
    return Array.from({ length: destination }, (_, index) => {
      const center = (index + 0.5) * scale
      const start = Math.max(0, Math.trunc(center - support + 0.5))
      const end = Math.min(source, Math.trunc(center + support + 0.5))
      const values = Array.from({ length: end - start }, (_, i) =>
        Math.max(0, 1 - Math.abs((start + i - center + 0.5) / support)),
      )
      const total = values.reduce((a, b) => a + b, 0)
      return {
        start,
        values: values.map((value) =>
          Math.floor((value / total) * precision + 0.5),
        ),
      }
    })
  }
  const horizontal = weights(sourceWidth, width),
    vertical = weights(sourceHeight, height)
  const rounded = (value: number) =>
    Math.max(0, Math.min(255, Math.floor(value / precision + 0.5)))
  // Tight loops avoid allocating a callback/reduction context per pixel/channel.
  let rows = rgb
  if (width !== sourceWidth) {
    rows = new Array(sourceHeight)
    for (let y = 0; y < sourceHeight; y++) {
      const row = (rows[y] = new Array(width))
      for (let x = 0; x < width; x++) {
        const { start, values } = horizontal[x]
        let r = 0,
          g = 0,
          b = 0
        for (let i = 0; i < values.length; i++) {
          const pixel = rgb[y][start + i],
            weight = values[i]
          r += weight * pixel[0]
          g += weight * pixel[1]
          b += weight * pixel[2]
        }
        row[x] = [rounded(r), rounded(g), rounded(b)]
      }
    }
  }
  if (height === sourceHeight) return rows
  const result: number[][][] = new Array(height)
  for (let y = 0; y < height; y++) {
    const { start, values } = vertical[y]
    const row = (result[y] = new Array(width))
    for (let x = 0; x < width; x++) {
      let r = 0,
        g = 0,
        b = 0
      for (let i = 0; i < values.length; i++) {
        const pixel = rows[start + i][x],
          weight = values[i]
        r += weight * pixel[0]
        g += weight * pixel[1]
        b += weight * pixel[2]
      }
      row[x] = [rounded(r), rounded(g), rounded(b)]
    }
  }
  return result
}

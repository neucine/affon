function shuffleItems<T>(items: T[]): T[] {
  const out = items.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    const tmp = out[i]
    out[i] = out[j]
    out[j] = tmp
  }
  return out
}

function requirePositiveInt(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) {
    throw new TypeError(`${name} must be a positive integer`)
  }
}

function requireNonNegativeInt(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    throw new TypeError(`${name} must be a non-negative integer`)
  }
}

function makeClonedBatchLoader<T>(
  items: T[],
  batchSize: number,
  cloneItem: (item: T) => T,
): { length: number; [Symbol.iterator](): Iterator<T[]> } {
  return {
    length: Math.ceil(items.length / batchSize),
    [Symbol.iterator](): Iterator<T[]> {
      let offset = 0
      return {
        next(): IteratorResult<T[]> {
          if (offset >= items.length) return { done: true, value: undefined as any }
          const value = items.slice(offset, offset + batchSize).map((item) => cloneItem(item))
          offset += batchSize
          return { done: false, value }
        }
      }
    }
  }
}

function normalizePaddedBatchOpts(
  opts: { batchSize?: number; padId: number; maxLength?: number },
  batchSizeName: string,
  padIdName: string,
  maxLengthName: string,
): { batchSize: number; padId: number; maxLength: number | undefined } {
  const batchSize = opts.batchSize ?? 32
  requirePositiveInt(batchSize, batchSizeName)
  requireNonNegativeInt(opts.padId, padIdName)
  if (opts.maxLength !== undefined) requirePositiveInt(opts.maxLength, maxLengthName)
  return { batchSize, padId: opts.padId, maxLength: opts.maxLength }
}

function padInputIds(
  rows: readonly number[][],
  padId: number,
  maxLength?: number,
): { inputIds: number[][]; attentionMask: number[][] } {
  let width = 0
  for (let i = 0; i < rows.length; i++) width = Math.max(width, rows[i].length)
  if (maxLength !== undefined) width = Math.min(width, maxLength)

  const inputIds = rows.map((row) => {
    const trimmed = maxLength !== undefined ? row.slice(0, maxLength) : row.slice()
    while (trimmed.length < width) trimmed.push(padId)
    return trimmed
  })
  const attentionMask = rows.map((row) => {
    const used = maxLength !== undefined ? Math.min(row.length, maxLength) : row.length
    const mask = Array.from({ length: used }, () => 1)
    while (mask.length < width) mask.push(0)
    return mask
  })

  return { inputIds, attentionMask }
}

function splitItems<T>(items: T[], ratios: number[]): T[][] {
  if (ratios.length < 1) throw new TypeError('split() requires at least one ratio')
  if (ratios.length > 8) throw new TypeError('split() supports at most 8 ratios')

  let ratioSum = 0
  for (let i = 0; i < ratios.length; i++) {
    const ratio = ratios[i]
    if (!Number.isFinite(ratio) || ratio <= 0 || ratio >= 1) {
      throw new TypeError('split() each ratio must be between 0 and 1')
    }
    ratioSum += ratio
  }
  if (ratioSum > 1) throw new TypeError('split() ratios must sum to at most 1')

  const parts: T[][] = []
  let start = 0
  let cumulative = 0
  for (let i = 0; i < ratios.length; i++) {
    cumulative += ratios[i]
    const end = Math.round(items.length * cumulative)
    parts.push(items.slice(start, end))
    start = end
  }
  parts.push(items.slice(start))
  return parts
}

export {
  makeClonedBatchLoader,
  normalizePaddedBatchOpts,
  padInputIds,
  requireNonNegativeInt,
  requirePositiveInt,
  shuffleItems,
  splitItems,
}

import { readNative } from 'affon:dataset/native'

interface Tensor {
  readonly shape: number[]
  slice(ranges: string[]): Tensor
}

interface Dataset {
  features(...columns: string[]): Dataset
  input(...columns: string[]): Dataset
  toTensor(opts?: { dtype?: 'f32' | 'f64' }): { data: Tensor }
  toTensors(opts?: { dtype?: 'f32' | 'f64' }): { X: Tensor; y: Tensor }
}

class DataLoader {
  private X: Tensor
  private y: Tensor | null
  private batchSize: number

  constructor(dataset: Dataset, opts?: { batchSize?: number }) {
    try {
      const { X, y } = dataset.toTensors()
      this.X = X
      this.y = y
    } catch {
      const { data } = dataset.toTensor()
      this.X = data
      this.y = null
    }
    const batchSize = opts?.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('DataLoader.batchSize must be a positive integer')
    }
    this.batchSize = batchSize
  }

  get length(): number {
    return Math.ceil(this.X.shape[0] / this.batchSize)
  }

  [Symbol.iterator](): Iterator<Tensor[]> {
    const n = this.X.shape[0]
    const bs = this.batchSize
    const X = this.X
    const y = this.y
    const ndim = X.shape.length

    let offset = 0
    return {
      next(): IteratorResult<Tensor[]> {
        if (offset >= n) return { done: true, value: undefined as any }
        const start = offset
        const end = Math.min(offset + bs, n)
        offset = end

        const xRanges: string[] = [`${start}:${end}`]
        for (let d = 1; d < ndim; d++) xRanges.push(':')

        if (y) {
          const yRanges: string[] = [`${start}:${end}`]
          for (let d = 1; d < y.shape.length; d++) yRanges.push(':')
          return { done: false, value: [X.slice(xRanges), y.slice(yRanges)] }
        }
        return { done: false, value: [X.slice(xRanges)] }
      }
    }
  }
}

function addLoader(ds: any): any {
  ds.loader = function(opts?: { batchSize?: number }): DataLoader {
    return new DataLoader(ds, opts)
  }
  ds.tensorLoader = function(opts?: { batchSize?: number }): DataLoader {
    return new DataLoader(ds, opts)
  }

  if (typeof ds.features === 'function') {
    ds.input = function(...args: any[]) {
      return addLoader(ds.features(...args))
    }
  }

  const chainable = [
    'select', 'drop', 'encode', 'normalize', 'standardize', 'log',
    'clip', 'shuffle', 'sample', 'fillna', 'dropna', 'rename',
    'features', 'target', 'concat',
  ]

  for (const method of chainable) {
    if (typeof ds[method] === 'function') {
      const orig = ds[method].bind(ds)
      ds[method] = function(...args: any[]) {
        return addLoader(orig(...args))
      }
    }
  }

  if (typeof ds.split === 'function') {
    const origSplit = ds.split.bind(ds)
    ds.split = function(...args: any[]) {
      return origSplit(...args).map((part: any) => addLoader(part))
    }
  }

  return ds
}

function read(source: string) {
  return addLoader(readNative(source))
}

const tabular = {
  read,
  DataLoader,
}

export { read, DataLoader, tabular }

import 'affon:errors'
import { readNative } from 'affon:dataset/native'
import type { Session, Tensor } from 'affon:compute'
import { slice } from 'affon:ops'

interface TensorExportOpts {
  session: Session
  dtype?: 'f32' | 'f64'
}

interface Dataset {
  features(...columns: string[]): Dataset
  input(...columns: string[]): Dataset
  toTensor(opts: TensorExportOpts): { data: Tensor; schema: Record<string, number[]> }
  toTensors(opts: TensorExportOpts): { X: Tensor; y: Tensor; schema: Record<string, number[]> }
}

interface DataLoaderOpts {
  session: Session
  batchSize?: number
}

class DataLoader {
  private X: Tensor
  private y: Tensor | null
  private batchSize: number

  constructor(dataset: Dataset, opts: DataLoaderOpts) {
    try {
      const { X, y } = dataset.toTensors({ session: opts.session })
      this.X = X
      this.y = y
    } catch {
      const { data } = dataset.toTensor({ session: opts.session })
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

        const xRanges = [{ start, stop: end }]
        for (let d = 1; d < ndim; d++) xRanges.push({ start: 0, stop: X.shape[d] })

        if (y) {
          const yRanges = [{ start, stop: end }]
          for (let d = 1; d < y.shape.length; d++) yRanges.push({ start: 0, stop: y.shape[d] })
          return { done: false, value: [slice(X, xRanges), slice(y, yRanges)] }
        }
        return { done: false, value: [slice(X, xRanges)] }
      }
    }
  }
}

function addLoader(ds: any): any {
  const nativeToTensor = ds.toTensor.bind(ds)
  const nativeToTensors = ds.toTensors.bind(ds)
  ds.toTensor = function(opts: TensorExportOpts) {
    if (!opts?.session) throw new TypeError('toTensor() requires a Session')
    const result = nativeToTensor({ dtype: opts.dtype })
    return { ...result, data: opts.session.tensor(result.data, { dtype: opts.dtype ?? 'f32' }) }
  }
  ds.toTensors = function(opts: TensorExportOpts) {
    if (!opts?.session) throw new TypeError('toTensors() requires a Session')
    const result = nativeToTensors({ dtype: opts.dtype })
    return {
      ...result,
      X: opts.session.tensor(result.X, { dtype: opts.dtype ?? 'f32' }),
      y: opts.session.tensor(result.y, { dtype: opts.dtype ?? 'f32' }),
    }
  }
  ds.loader = function(opts: DataLoaderOpts): DataLoader {
    return new DataLoader(ds, opts)
  }
  ds.tensorLoader = function(opts: DataLoaderOpts): DataLoader {
    return new DataLoader(ds, opts)
  }

  if (typeof ds.features === 'function') {
    ds.input = function(...args: any[]) {
      return ds.features(...args)
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

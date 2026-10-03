import type { Session, Tensor } from 'affon:compute'

interface PaddedTextDataLoaderOpts {
  batchSize?: number
  padId: number
  maxLength?: number
}

interface TensorizedTextDataLoaderOpts extends PaddedTextDataLoaderOpts {
  session: Session
}

interface TensorizedSupervisedTextBatch {
  inputIds: Tensor
  attentionMask: Tensor
  labels: Tensor
}

interface PairPaddedBatch {
  inputIds: number[][]
  attentionMask: number[][]
  tokenTypeIds: number[][]
}

interface TextDatasetLike {
  toArray(): string[]
}

interface EncodedTextDatasetLike {
  toArray(): number[][]
}

interface EncodedLabeledTextDatasetLike {
  toArray(): Array<{ inputIds: number[]; label: string }>
}

class TextDataLoader {
  private items: string[]
  private batchSize: number

  constructor(dataset: TextDatasetLike, opts?: { batchSize?: number }) {
    const batchSize = opts?.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('TextDataLoader.batchSize must be a positive integer')
    }
    this.items = dataset.toArray()
    this.batchSize = batchSize
  }

  get length(): number {
    return Math.ceil(this.items.length / this.batchSize)
  }

  [Symbol.iterator](): Iterator<string[]> {
    const items = this.items
    const bs = this.batchSize
    let offset = 0
    return {
      next(): IteratorResult<string[]> {
        if (offset >= items.length) return { done: true, value: undefined as any }
        const value = items.slice(offset, offset + bs)
        offset += bs
        return { done: false, value }
      }
    }
  }
}

class EncodedTextDataLoader {
  private items: number[][]
  private batchSize: number

  constructor(dataset: EncodedTextDatasetLike, opts?: { batchSize?: number }) {
    const batchSize = opts?.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('EncodedTextDataLoader.batchSize must be a positive integer')
    }
    this.items = dataset.toArray()
    this.batchSize = batchSize
  }

  get length(): number {
    return Math.ceil(this.items.length / this.batchSize)
  }

  [Symbol.iterator](): Iterator<number[][]> {
    const items = this.items
    const bs = this.batchSize
    let offset = 0
    return {
      next(): IteratorResult<number[][]> {
        if (offset >= items.length) return { done: true, value: undefined as any }
        const value = items.slice(offset, offset + bs)
        offset += bs
        return { done: false, value }
      }
    }
  }
}

class PaddedTextDataLoader {
  private items: number[][]
  private batchSize: number
  private padId: number
  private maxLength: number | null

  constructor(dataset: EncodedTextDatasetLike, opts: PaddedTextDataLoaderOpts) {
    const batchSize = opts.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('PaddedTextDataLoader.batchSize must be a positive integer')
    }
    if (!Number.isFinite(opts.padId) || opts.padId < 0 || !Number.isInteger(opts.padId)) {
      throw new TypeError('PaddedTextDataLoader.padId must be a non-negative integer')
    }
    if (opts.maxLength !== undefined && (!Number.isFinite(opts.maxLength) || opts.maxLength <= 0 || !Number.isInteger(opts.maxLength))) {
      throw new TypeError('PaddedTextDataLoader.maxLength must be a positive integer')
    }

    this.items = dataset.toArray()
    this.batchSize = batchSize
    this.padId = opts.padId
    this.maxLength = opts.maxLength ?? null
  }

  get length(): number {
    return Math.ceil(this.items.length / this.batchSize)
  }

  [Symbol.iterator](): Iterator<{ inputIds: number[][]; attentionMask: number[][] }> {
    const items = this.items
    const bs = this.batchSize
    const padId = this.padId
    const maxLength = this.maxLength
    let offset = 0
    return {
      next(): IteratorResult<{ inputIds: number[][]; attentionMask: number[][] }> {
        if (offset >= items.length) return { done: true, value: undefined as any }
        const batch = items.slice(offset, offset + bs)
        offset += bs

        let width = 0
        for (let i = 0; i < batch.length; i++) width = Math.max(width, batch[i].length)
        if (maxLength !== null) width = Math.min(width, maxLength)

        const inputIds = batch.map((row) => {
          const trimmed = maxLength !== null ? row.slice(0, maxLength) : row.slice()
          while (trimmed.length < width) trimmed.push(padId)
          return trimmed
        })
        const attentionMask = batch.map((row) => {
          const used = maxLength !== null ? Math.min(row.length, maxLength) : row.length
          const mask = Array.from({ length: used }, () => 1)
          while (mask.length < width) mask.push(0)
          return mask
        })

        return { done: false, value: { inputIds, attentionMask } }
      }
    }
  }
}

class TensorizedTextDataLoader {
  private loader: PaddedTextDataLoader
  private session: Session

  constructor(dataset: EncodedTextDatasetLike, opts: TensorizedTextDataLoaderOpts) {
    this.loader = new PaddedTextDataLoader(dataset, opts)
    this.session = opts.session
  }

  get length(): number {
    return this.loader.length
  }

  [Symbol.iterator](): Iterator<{ inputIds: Tensor; attentionMask: Tensor }> {
    const inner = this.loader[Symbol.iterator]()
    const session = this.session
    return {
      next(): IteratorResult<{ inputIds: Tensor; attentionMask: Tensor }> {
        const step = inner.next()
        if (step.done) return { done: true, value: undefined as any }
        return {
          done: false,
          value: {
            inputIds: session.tensor(step.value.inputIds, { dtype: 'i64' }),
            attentionMask: session.tensor(step.value.attentionMask, { dtype: 'i64' }),
          },
        }
      }
    }
  }
}

class EncodedLabeledTextDataLoader {
  private items: Array<{ inputIds: number[]; label: string }>
  private batchSize: number

  constructor(dataset: EncodedLabeledTextDatasetLike, opts?: { batchSize?: number }) {
    const batchSize = opts?.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('EncodedLabeledTextDataLoader.batchSize must be a positive integer')
    }
    this.items = dataset.toArray()
    this.batchSize = batchSize
  }

  get length(): number {
    return Math.ceil(this.items.length / this.batchSize)
  }

  [Symbol.iterator](): Iterator<Array<{ inputIds: number[]; label: string }>> {
    const items = this.items
    const bs = this.batchSize
    let offset = 0
    return {
      next(): IteratorResult<Array<{ inputIds: number[]; label: string }>> {
        if (offset >= items.length) return { done: true, value: undefined as any }
        const value = items.slice(offset, offset + bs).map((item) => ({ inputIds: item.inputIds.slice(), label: item.label }))
        offset += bs
        return { done: false, value }
      }
    }
  }
}

class PaddedLabeledTextDataLoader {
  private items: Array<{ inputIds: number[]; label: string }>
  private batchSize: number
  private padId: number
  private maxLength: number | null

  constructor(dataset: EncodedLabeledTextDatasetLike, opts: PaddedTextDataLoaderOpts) {
    const batchSize = opts.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('PaddedLabeledTextDataLoader.batchSize must be a positive integer')
    }
    if (!Number.isFinite(opts.padId) || opts.padId < 0 || !Number.isInteger(opts.padId)) {
      throw new TypeError('PaddedLabeledTextDataLoader.padId must be a non-negative integer')
    }
    if (opts.maxLength !== undefined && (!Number.isFinite(opts.maxLength) || opts.maxLength <= 0 || !Number.isInteger(opts.maxLength))) {
      throw new TypeError('PaddedLabeledTextDataLoader.maxLength must be a positive integer')
    }
    this.items = dataset.toArray()
    this.batchSize = batchSize
    this.padId = opts.padId
    this.maxLength = opts.maxLength ?? null
  }

  get length(): number {
    return Math.ceil(this.items.length / this.batchSize)
  }

  [Symbol.iterator](): Iterator<{ inputIds: number[][]; attentionMask: number[][]; labels: string[] }> {
    const items = this.items
    const bs = this.batchSize
    const padId = this.padId
    const maxLength = this.maxLength
    let offset = 0
    return {
      next(): IteratorResult<{ inputIds: number[][]; attentionMask: number[][]; labels: string[] }> {
        if (offset >= items.length) return { done: true, value: undefined as any }
        const batch = items.slice(offset, offset + bs)
        offset += bs

        let width = 0
        for (let i = 0; i < batch.length; i++) width = Math.max(width, batch[i].inputIds.length)
        if (maxLength !== null) width = Math.min(width, maxLength)

        const inputIds = batch.map((row) => {
          const trimmed = maxLength !== null ? row.inputIds.slice(0, maxLength) : row.inputIds.slice()
          while (trimmed.length < width) trimmed.push(padId)
          return trimmed
        })
        const attentionMask = batch.map((row) => {
          const used = maxLength !== null ? Math.min(row.inputIds.length, maxLength) : row.inputIds.length
          const mask = Array.from({ length: used }, () => 1)
          while (mask.length < width) mask.push(0)
          return mask
        })
        return {
          done: false,
          value: {
            inputIds,
            attentionMask,
            labels: batch.map((row) => row.label),
          },
        }
      }
    }
  }
}

class TensorizedLabeledTextDataLoader {
  private loader: PaddedLabeledTextDataLoader
  private session: Session

  constructor(dataset: EncodedLabeledTextDatasetLike, opts: TensorizedTextDataLoaderOpts) {
    this.loader = new PaddedLabeledTextDataLoader(dataset, opts)
    this.session = opts.session
  }

  get length(): number {
    return this.loader.length
  }

  [Symbol.iterator](): Iterator<{ inputIds: Tensor; attentionMask: Tensor; labels: string[] }> {
    const inner = this.loader[Symbol.iterator]()
    const session = this.session
    return {
      next(): IteratorResult<{ inputIds: Tensor; attentionMask: Tensor; labels: string[] }> {
        const step = inner.next()
        if (step.done) return { done: true, value: undefined as any }
        return {
          done: false,
          value: {
            inputIds: session.tensor(step.value.inputIds, { dtype: 'i64' }),
            attentionMask: session.tensor(step.value.attentionMask, { dtype: 'i64' }),
            labels: step.value.labels.slice(),
          },
        }
      }
    }
  }
}

export {
  EncodedLabeledTextDataLoader,
  EncodedTextDataLoader,
  PaddedLabeledTextDataLoader,
  PaddedTextDataLoader,
  TensorizedLabeledTextDataLoader,
  TensorizedTextDataLoader,
  TextDataLoader,
}

export type {
  PairPaddedBatch,
  PaddedTextDataLoaderOpts,
  TensorizedTextDataLoaderOpts,
  TensorizedSupervisedTextBatch,
}

import { text } from 'affon:dataset/text.ts'
import { type Session, type Tensor } from 'affon:compute'

export type TokenWindow = number[]

export interface PackedCorpusOptions {
  seqLen: number
  stride?: number
  joinWithTokenId?: number
}

export interface TokenBatchOptions {
  batchSize: number
  shuffle?: boolean
  shuffleSeed?: number
  session?: Session
}

interface TokenWindowPrng {
  state: number
}

function createTokenWindowPrng(seed?: number | null): TokenWindowPrng {
  const fallback = Math.floor(Math.random() * 0x100000000)
  const initial = seed === undefined || seed === null ? fallback : seed
  return { state: (initial >>> 0) || 1 }
}

function nextTokenWindowPrng(prng: TokenWindowPrng): number {
  prng.state = (Math.imul(prng.state, 1664525) + 1013904223) >>> 0
  return prng.state
}

function shuffleInPlace<T>(values: T[], prng: TokenWindowPrng): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = nextTokenWindowPrng(prng) % (i + 1)
    const tmp = values[i]
    values[i] = values[j]
    values[j] = tmp
  }
}

function batchToTensor(batch: readonly TokenWindow[], session: Session): Tensor {
  return session.tensor(batch.map((row) => row.slice()), { dtype: 'i64', axes: ['batch', 'token'] })
}

function batchOrder(count: number, shuffle: boolean, prng: TokenWindowPrng): number[] {
  const order = Array.from({ length: count }, (_, index) => index)
  if (shuffle) shuffleInPlace(order, prng)
  return order
}

function batchFromOrder(
  windows: readonly TokenWindow[],
  order: readonly number[],
  start: number,
  batchSize: number,
  session: Session,
): Tensor {
  const rows: TokenWindow[] = []
  const end = Math.min(start + batchSize, order.length)
  for (let i = start; i < end; i++) {
    rows.push(windows[order[i]].slice())
  }
  return batchToTensor(rows, session)
}

export function pack_token_windows(
  rows: readonly (readonly number[])[],
  opts: PackedCorpusOptions,
): TokenWindow[] {
  return text.encoded(rows).window(opts).toArray()
}

export function create_token_batches(
  windows: readonly TokenWindow[],
  opts: TokenBatchOptions,
): Tensor[] {
  if (!Number.isInteger(opts.batchSize) || opts.batchSize <= 0) {
    throw new AffonError('invalid_arg', 'create_token_batches batchSize must be a positive integer')
  }
  const order = batchOrder(windows.length, opts.shuffle ?? false, createTokenWindowPrng(opts.shuffleSeed ?? 0))
  if (!opts.session) throw new AffonError('invalid_arg', 'create_token_batches requires a Session')
  const session = opts.session
  const batches: Tensor[] = []
  for (let i = 0; i < order.length; i += opts.batchSize) {
    batches.push(batchFromOrder(windows, order, i, opts.batchSize, session))
  }
  return batches
}

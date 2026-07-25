import type { SpecialTokens, TextEncodeOpts } from './types.ts'

export function specialTokenSet(specialTokens: Readonly<SpecialTokens>): Set<string> {
  return new Set(
    Object.values(specialTokens).filter((value): value is string => typeof value === 'string')
  )
}

export function specialTokenIds(
  tokenToId: Readonly<Record<string, number>>,
  specialTokens: Readonly<SpecialTokens>,
): Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }> {
  return Object.freeze({
    bos: specialTokens.bos ? tokenToId[specialTokens.bos] : undefined,
    eos: specialTokens.eos ? tokenToId[specialTokens.eos] : undefined,
    pad: specialTokens.pad ? tokenToId[specialTokens.pad] : undefined,
    unk: specialTokens.unk ? tokenToId[specialTokens.unk] : undefined,
    sep: specialTokens.sep ? tokenToId[specialTokens.sep] : undefined,
  })
}

export function tokenIdOrThrow(
  tokenToId: Readonly<Record<string, number>>,
  token: string | undefined,
  kind: 'bos' | 'eos' | 'pad' | 'unk',
): number {
  if (!token) throw new TypeError(`Tokenizer is missing special token: ${kind}`)
  const id = tokenToId[token]
  if (id === undefined) throw new TypeError(`Special token ${kind} is not present in vocabulary: ${token}`)
  return id
}

export function truncateSingleIds(ids: number[], opts: TextEncodeOpts | undefined): number[] {
  const maxLength = opts?.maxLength
  if (maxLength === undefined) return ids
  if (!Number.isFinite(maxLength) || maxLength <= 0 || !Number.isInteger(maxLength)) {
    throw new TypeError('TextEncodeOpts.maxLength must be a positive integer')
  }
  if (ids.length <= maxLength) return ids
  return ids.slice(0, maxLength)
}

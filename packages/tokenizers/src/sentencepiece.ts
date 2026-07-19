import { readFileSync } from 'std:fs'
import { specialTokenIds, specialTokenSet, tokenIdOrThrow, truncateSingleIds } from './common.ts'
import type { SentencePieceModel, SentencePieceTokenizerOpts, SpecialTokens, TextDecodeOpts, TextEncodeOpts, TextTokenizer } from './types.ts'

function makeSentencePieceVocabulary(model: SentencePieceModel): { tokenToId: Record<string, number>; idToToken: Map<number, string> } {
  const tokenToId: Record<string, number> = {}
  const idToToken = new Map<number, string>()
  for (let i = 0; i < model.pieces.length; i++) {
    const token = model.pieces[i].piece
    tokenToId[token] = i
    idToToken.set(i, token)
  }
  return { tokenToId, idToToken }
}

function inferSentencePieceSpecialTokens(
  model: SentencePieceModel,
  idToToken: ReadonlyMap<number, string>,
  opts?: SentencePieceTokenizerOpts,
): Readonly<SpecialTokens> {
  if (opts?.specialTokens) return Object.freeze({ ...opts.specialTokens })

  const tokenAt = (id: number | undefined): string | undefined => {
    if (id === undefined) return undefined
    return idToToken.get(id)
  }

  return Object.freeze({
    bos: tokenAt(model.bosId),
    eos: tokenAt(model.eosId),
    pad: tokenAt(model.padId),
    unk: tokenAt(model.unkId),
  })
}

function normalizeSentencePieceInput(text: string): string {
  if (text.length === 0) return ''
  const words = text.trim().split(/\s+/).filter((part) => part.length > 0)
  if (words.length === 0) return ''
  return `▁${words.join('▁')}`
}

function encodeSentencePieceText(
  text: string,
  tokenToId: Readonly<Record<string, number>>,
  specialTokens: Readonly<SpecialTokens>,
): number[] {
  const normalized = normalizeSentencePieceInput(text)
  if (normalized.length === 0) return []

  const chars = Array.from(normalized)
  const ids: number[] = []
  let start = 0
  while (start < chars.length) {
    let end = chars.length
    let matchedId: number | undefined
    while (end > start) {
      const candidate = chars.slice(start, end).join('')
      const id = tokenToId[candidate]
      if (id !== undefined) {
        matchedId = id
        break
      }
      end -= 1
    }
    if (matchedId === undefined) {
      if (specialTokens.unk !== undefined) return [tokenToId[specialTokens.unk]!]
      throw new TypeError(`Unknown token: ${text}`)
    }
    ids.push(matchedId)
    start = end
  }
  return ids
}

function decodeSentencePieceIds(
  ids: readonly number[],
  idToToken: ReadonlyMap<number, string>,
  specialTokensSet: ReadonlySet<string>,
  opts?: TextDecodeOpts,
): string {
  let text = ''
  for (let i = 0; i < ids.length; i++) {
    const token = idToToken.get(ids[i])
    if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
    if (opts?.skipSpecialTokens && specialTokensSet.has(token)) continue
    text += token
  }
  return text.replaceAll('▁', ' ').trim()
}

class SentencePieceTokenizer implements TextTokenizer {
  private tokenToId: Record<string, number>
  private idToToken: Map<number, string>
  readonly specialTokens: Readonly<SpecialTokens>
  readonly specialTokenIds: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>
  private specialTokensSet: Set<string>

  constructor(private model: SentencePieceModel, opts?: SentencePieceTokenizerOpts) {
    const { tokenToId, idToToken } = makeSentencePieceVocabulary(model)
    this.tokenToId = tokenToId
    this.idToToken = idToToken
    this.specialTokens = inferSentencePieceSpecialTokens(model, idToToken, opts)
    this.specialTokensSet = specialTokenSet(this.specialTokens)
    this.specialTokenIds = specialTokenIds(this.tokenToId, this.specialTokens)
  }

  get vocabSize(): number {
    return this.idToToken.size
  }

  tokenId(token: string): number | undefined {
    return this.tokenToId[token]
  }

  token(id: number): string | undefined {
    return this.idToToken.get(id)
  }

  private tokenIdOrThrow(token: string | undefined, kind: 'bos' | 'eos' | 'pad' | 'unk'): number {
    return tokenIdOrThrow(this.tokenToId, token, kind)
  }

  encode(text: string, opts?: TextEncodeOpts): number[] {
    const ids: number[] = []
    if (opts?.addBos) ids.push(this.tokenIdOrThrow(this.specialTokens.bos, 'bos'))
    ids.push(...encodeSentencePieceText(text, this.tokenToId, this.specialTokens))
    if (opts?.addEos) ids.push(this.tokenIdOrThrow(this.specialTokens.eos, 'eos'))
    return truncateSingleIds(ids, opts)
  }

  decode(ids: readonly number[], opts?: TextDecodeOpts): string {
    return decodeSentencePieceIds(ids, this.idToToken, this.specialTokensSet, opts)
  }
}

export function sentencePieceTokenizer(model: SentencePieceModel, opts?: SentencePieceTokenizerOpts): TextTokenizer {
  return new SentencePieceTokenizer(model, opts)
}

export function sentencePieceTokenizerFromFile(path: string, opts?: SentencePieceTokenizerOpts): TextTokenizer {
  const raw = readFileSync(path)
  return sentencePieceTokenizer(JSON.parse(raw) as SentencePieceModel, opts)
}

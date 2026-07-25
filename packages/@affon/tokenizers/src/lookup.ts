import { specialTokenIds, specialTokenSet, tokenIdOrThrow, truncateSingleIds } from './common.ts'
import type { LookupTokenizerOpts, SpecialTokens, TextDecodeOpts, TextEncodeOpts, TextTokenizer } from './types.ts'

class LookupTokenizer implements TextTokenizer {
  private tokenToId: Record<string, number>
  private idToToken: Map<number, string>
  private splitMode: 'whitespace' | 'char'
  readonly specialTokens: Readonly<SpecialTokens>
  private specialTokensSet: Set<string>
  readonly specialTokenIds: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>

  constructor(tokenToId: Record<string, number>, opts?: LookupTokenizerOpts) {
    this.tokenToId = { ...tokenToId }
    this.idToToken = new Map<number, string>()
    for (const [token, id] of Object.entries(tokenToId)) {
      if (!Number.isInteger(id) || id < 0) {
        throw new TypeError(`Invalid vocabulary id for token ${token}`)
      }
      if (this.idToToken.has(id)) {
        throw new TypeError(`Duplicate vocabulary id: ${id}`)
      }
      this.idToToken.set(id, token)
    }
    this.splitMode = opts?.split ?? 'whitespace'
    this.specialTokens = Object.freeze({ ...(opts?.specialTokens ?? {}) })
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
    const pieces = this.splitMode === 'char'
      ? Array.from(text)
      : text.length === 0
        ? []
        : text.trim().split(/\s+/).filter((part) => part.length > 0)

    const ids: number[] = []
    if (opts?.addBos) ids.push(this.tokenIdOrThrow(this.specialTokens.bos, 'bos'))
    for (let i = 0; i < pieces.length; i++) {
      const piece = pieces[i]
      const id = this.tokenToId[piece]
      if (id !== undefined) {
        ids.push(id)
        continue
      }
      if (this.specialTokens.unk !== undefined) {
        ids.push(this.tokenIdOrThrow(this.specialTokens.unk, 'unk'))
        continue
      }
      throw new TypeError(`Unknown token: ${piece}`)
    }
    if (opts?.addEos) ids.push(this.tokenIdOrThrow(this.specialTokens.eos, 'eos'))
    return truncateSingleIds(ids, opts)
  }

  decode(ids: readonly number[], opts?: TextDecodeOpts): string {
    const tokens: string[] = []
    for (let i = 0; i < ids.length; i++) {
      const token = this.idToToken.get(ids[i])
      if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
      if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
      tokens.push(token)
    }
    return this.splitMode === 'char' ? tokens.join('') : tokens.join(' ')
  }
}

export function lookupTokenizer(tokenToId: Record<string, number>, opts?: LookupTokenizerOpts): TextTokenizer {
  return new LookupTokenizer(tokenToId, opts)
}

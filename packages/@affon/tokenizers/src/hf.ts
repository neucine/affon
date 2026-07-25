import { readFileSync } from 'std:fs'
import { specialTokenIds, specialTokenSet, tokenIdOrThrow, truncateSingleIds } from './common.ts'
import type { HFPreTokenizerSpec, HFTokenizerJSON, LookupTokenizerOpts, SpecialTokens, TextDecodeOpts, TextEncodeOpts, TextTokenizer } from './types.ts'

function inferHFModelType(spec: HFTokenizerJSON): string {
  const explicit = spec.model.type
  if (typeof explicit === 'string' && explicit.length > 0) return explicit
  if (Array.isArray(spec.model.merges)) return 'BPE'
  if (
    spec.model.continuing_subword_prefix !== undefined
    || spec.model.max_input_chars_per_word !== undefined
  ) {
    return 'WordPiece'
  }
  return 'WordLevel'
}

function inferSpecialTokens(
  vocab: Record<string, number>,
  opts: LookupTokenizerOpts | undefined,
  spec: HFTokenizerJSON,
): Readonly<SpecialTokens> {
  if (opts?.specialTokens) {
    return Object.freeze({
      ...opts.specialTokens,
      ...(spec.model.unk_token && opts.specialTokens.unk === undefined ? { unk: spec.model.unk_token } : {}),
    })
  }

  const specials = new Set(
    (spec.added_tokens ?? [])
      .filter((token) => token.special)
      .map((token) => token.content)
  )

  const infer = (candidates: string[]): string | undefined => {
    for (let i = 0; i < candidates.length; i++) {
      const token = candidates[i]
      if (specials.has(token) || vocab[token] !== undefined) return token
    }
    return undefined
  }

  return Object.freeze({
    bos: infer(['<bos>', '<s>', '[BOS]']),
    eos: infer(['<eos>', '</s>', '[EOS]']),
    pad: infer(['<pad>', '[PAD]']),
    unk: spec.model.unk_token ?? infer(['<unk>', '[UNK]']),
    sep: infer(['<sep>', '[SEP]', '</s>']),
  })
}

function splitHFWordLevel(text: string, preTokenizerType: string | undefined): string[] {
  const kind = preTokenizerType ?? 'Whitespace'
  if (kind !== 'Whitespace' && kind !== 'WhitespaceSplit') {
    throw new TypeError(`Unsupported HF pre_tokenizer type: ${kind}`)
  }
  if (text.length === 0) return []
  return text.trim().split(/\s+/).filter((part) => part.length > 0)
}

function utf8Encode(text: string): number[] {
  const bytes: number[] = []
  for (const char of text) {
    const codePoint = char.codePointAt(0) ?? 0
    if (codePoint <= 0x7f) {
      bytes.push(codePoint)
      continue
    }
    if (codePoint <= 0x7ff) {
      bytes.push(0xc0 | (codePoint >> 6))
      bytes.push(0x80 | (codePoint & 0x3f))
      continue
    }
    if (codePoint <= 0xffff) {
      bytes.push(0xe0 | (codePoint >> 12))
      bytes.push(0x80 | ((codePoint >> 6) & 0x3f))
      bytes.push(0x80 | (codePoint & 0x3f))
      continue
    }
    bytes.push(0xf0 | (codePoint >> 18))
    bytes.push(0x80 | ((codePoint >> 12) & 0x3f))
    bytes.push(0x80 | ((codePoint >> 6) & 0x3f))
    bytes.push(0x80 | (codePoint & 0x3f))
  }
  return bytes
}

const hfByteEncoder = (() => {
  const byteToChar = new Map<number, string>()
  const charToByte = new Map<string, number>()
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)

  const cs = bs.slice()
  let extra = 0
  for (let b = 0; b < 256; b++) {
    if (bs.includes(b)) continue
    bs.push(b)
    cs.push(256 + extra)
    extra += 1
  }

  for (let i = 0; i < bs.length; i++) {
    const char = String.fromCodePoint(cs[i])
    byteToChar.set(bs[i], char)
    charToByte.set(char, bs[i])
  }

  return { byteToChar, charToByte }
})()

function encodeHFByteLevelText(text: string): string {
  const bytes = utf8Encode(text)
  let out = ''
  for (let i = 0; i < bytes.length; i++) {
    out += hfByteEncoder.byteToChar.get(bytes[i]) ?? String.fromCodePoint(bytes[i])
  }
  return out
}

function splitHFByteLevel(text: string, addPrefixSpace: boolean | undefined): string[] {
  if (text.length === 0) return []
  const source = addPrefixSpace && text[0] !== ' ' ? ` ${text}` : text
  const pieces: string[] = []
  const matches = source.matchAll(/\S+/g)
  for (const match of matches) {
    const value = match[0]
    const index = match.index ?? 0
    const hasLeadingSpace = index > 0 && /\s/.test(source[index - 1])
    pieces.push(hasLeadingSpace ? ` ${value}` : value)
  }
  return pieces
}

function splitHFText(text: string, preTokenizer: HFPreTokenizerSpec | null | undefined): string[] {
  const kind = preTokenizer?.type ?? 'Whitespace'
  if (kind === 'Whitespace' || kind === 'WhitespaceSplit') {
    return splitHFWordLevel(text, kind)
  }
  if (kind === 'ByteLevel') {
    return splitHFByteLevel(text, preTokenizer?.add_prefix_space)
  }
  if (kind === 'Sequence') {
    const pretokenizers = preTokenizer?.pretokenizers ?? []
    let sawByteLevel = false
    let byteLevelAddPrefixSpace: boolean | undefined
    let sawWhitespace = false
    for (let i = 0; i < pretokenizers.length; i++) {
      const child = pretokenizers[i]
      if (child.type === 'ByteLevel') {
        sawByteLevel = true
        byteLevelAddPrefixSpace = child.add_prefix_space
      } else if (child.type === 'Whitespace' || child.type === 'WhitespaceSplit') {
        sawWhitespace = true
      } else {
        throw new TypeError(`Unsupported HF sequence pre_tokenizer type: ${child.type}`)
      }
    }
    if (sawByteLevel) return splitHFByteLevel(text, byteLevelAddPrefixSpace)
    if (sawWhitespace) return splitHFWordLevel(text, 'Whitespace')
    throw new TypeError('Unsupported HF sequence pre_tokenizer configuration')
  }
  throw new TypeError(`Unsupported HF pre_tokenizer type: ${kind}`)
}

function decodeHFByteLevel(text: string): string {
  let encoded = ''
  for (const char of text) {
    const byte = hfByteEncoder.charToByte.get(char)
    if (byte === undefined) {
      encoded += char
      continue
    }
    encoded += `%${byte.toString(16).padStart(2, '0')}`
  }
  try {
    return decodeURIComponent(encoded).trim()
  } catch {
    return text.replaceAll('Ġ', ' ').trim()
  }
}

function normalizeHFMerge(entry: string | [string, string]): [string, string] {
  if (Array.isArray(entry)) {
    if (entry.length !== 2) {
      throw new TypeError('HF BPE merge entries must contain exactly two tokens')
    }
    return [entry[0], entry[1]]
  }
  const parts = entry.trim().split(/\s+/)
  if (parts.length !== 2) {
    throw new TypeError(`Invalid HF BPE merge entry: ${entry}`)
  }
  return [parts[0], parts[1]]
}

function encodeHFBPEWord(
  word: string,
  vocab: Record<string, number>,
  merges: ReadonlyMap<string, number>,
  specialTokens: { bos?: string; eos?: string; pad?: string; unk?: string },
  prefix: string | undefined,
  suffix: string | undefined,
  byteLevel?: boolean,
): number[] {
  if (word.length === 0) return []
  const source = byteLevel ? encodeHFByteLevelText(word) : word
  const wholeWordId = vocab[source]
  if (wholeWordId !== undefined) return [wholeWordId]

  let pieces = Array.from(source)
  if (prefix && pieces.length > 1) {
    for (let i = 1; i < pieces.length; i++) {
      pieces[i] = `${prefix}${pieces[i]}`
    }
  }
  if (suffix && pieces.length > 0) {
    pieces[pieces.length - 1] = `${pieces[pieces.length - 1]}${suffix}`
  }

  while (pieces.length > 1) {
    let bestIndex = -1
    let bestRank = Number.POSITIVE_INFINITY
    for (let i = 0; i < pieces.length - 1; i++) {
      const rank = merges.get(`${pieces[i]} ${pieces[i + 1]}`)
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank
        bestIndex = i
      }
    }
    if (bestIndex < 0) break
    const merged = `${pieces[bestIndex]}${pieces[bestIndex + 1]}`
    pieces.splice(bestIndex, 2, merged)
  }

  const ids: number[] = []
  for (let i = 0; i < pieces.length; i++) {
    const id = vocab[pieces[i]]
    if (id !== undefined) {
      ids.push(id)
      continue
    }
    if (specialTokens.unk !== undefined) {
      ids.push(vocab[specialTokens.unk]!)
      continue
    }
    throw new TypeError(`Unknown token: ${pieces[i]}`)
  }
  return ids
}

function encodeHFWordPieceWord(
  word: string,
  vocab: Record<string, number>,
  specialTokens: { bos?: string; eos?: string; pad?: string; unk?: string },
  prefix: string,
  maxInputCharsPerWord: number | undefined,
): number[] {
  const chars = Array.from(word)
  if (chars.length === 0) return []
  if (maxInputCharsPerWord !== undefined && chars.length > maxInputCharsPerWord) {
    if (specialTokens.unk !== undefined) return [vocab[specialTokens.unk]!]
    throw new TypeError(`Input token exceeds max_input_chars_per_word: ${word}`)
  }

  const pieces: number[] = []
  let start = 0
  while (start < chars.length) {
    let end = chars.length
    let matchedId: number | undefined
    let matchedEnd = -1
    while (end > start) {
      const chunk = chars.slice(start, end).join('')
      const candidate = start === 0 ? chunk : `${prefix}${chunk}`
      const id = vocab[candidate]
      if (id !== undefined) {
        matchedId = id
        matchedEnd = end
        break
      }
      end -= 1
    }
    if (matchedId === undefined) {
      if (specialTokens.unk !== undefined) return [vocab[specialTokens.unk]!]
      throw new TypeError(`Unknown token: ${word}`)
    }
    pieces.push(matchedId)
    start = matchedEnd
  }
  return pieces
}

class HFTokenizer implements TextTokenizer {
  private vocab: Record<string, number>
  private reverse: Map<number, string>
  private modelType: string
  private preTokenizer: HFPreTokenizerSpec | null | undefined
  private merges: Map<string, number>
  private continuingSubwordPrefix: string | undefined
  private endOfWordSuffix: string | undefined
  private maxInputCharsPerWord: number | undefined
  private isByteLevel: boolean
  readonly specialTokens: Readonly<SpecialTokens>
  private specialTokensSet: Set<string>
  readonly specialTokenIds: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>

  constructor(spec: HFTokenizerJSON, opts?: LookupTokenizerOpts) {
    if (!spec || typeof spec !== 'object' || !spec.model || typeof spec.model !== 'object') {
      throw new TypeError('HF tokenizer spec must include a model object')
    }

    this.modelType = inferHFModelType(spec)
    this.vocab = { ...spec.model.vocab }
    this.reverse = new Map<number, string>()
    for (const [token, id] of Object.entries(this.vocab)) {
      if (!Number.isInteger(id) || id < 0) {
        throw new TypeError(`Invalid vocabulary id for token ${token}`)
      }
      if (this.reverse.has(id)) {
        throw new TypeError(`Duplicate vocabulary id: ${id}`)
      }
      this.reverse.set(id, token)
    }

    this.preTokenizer = spec.pre_tokenizer
    this.merges = new Map<string, number>()
    const mergeEntries = spec.model.merges ?? []
    for (let i = 0; i < mergeEntries.length; i++) {
      const [left, right] = normalizeHFMerge(mergeEntries[i])
      this.merges.set(`${left} ${right}`, i)
    }
    this.continuingSubwordPrefix = spec.model.continuing_subword_prefix
    this.endOfWordSuffix = spec.model.end_of_word_suffix
    this.maxInputCharsPerWord = spec.model.max_input_chars_per_word
    this.isByteLevel =
      this.preTokenizer?.type === 'ByteLevel'
      || (this.preTokenizer?.type === 'Sequence'
        && (this.preTokenizer.pretokenizers ?? []).some((tokenizer) => tokenizer.type === 'ByteLevel'))
    this.specialTokens = inferSpecialTokens(this.vocab, opts, spec)
    this.specialTokensSet = new Set([
      ...specialTokenSet(this.specialTokens),
      ...(spec.added_tokens ?? []).filter((token) => token.special).map((token) => token.content),
    ])
    this.specialTokenIds = specialTokenIds(this.vocab, this.specialTokens)
  }

  get vocabSize(): number {
    return Object.keys(this.vocab).length
  }

  tokenId(token: string): number | undefined {
    return this.vocab[token]
  }

  token(id: number): string | undefined {
    return this.reverse.get(id)
  }

  private tokenIdOrThrow(token: string | undefined, kind: 'bos' | 'eos' | 'pad' | 'unk'): number {
    return tokenIdOrThrow(this.vocab, token, kind)
  }

  encode(text: string, opts?: TextEncodeOpts): number[] {
    const pieces = splitHFText(text, this.preTokenizer)
    const ids: number[] = []
    if (opts?.addBos) ids.push(this.tokenIdOrThrow(this.specialTokens.bos, 'bos'))

    if (this.modelType === 'WordLevel') {
      for (let i = 0; i < pieces.length; i++) {
        const piece = pieces[i]
        const id = this.vocab[piece]
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
    } else if (this.modelType === 'BPE') {
      for (let i = 0; i < pieces.length; i++) {
        ids.push(...encodeHFBPEWord(
          pieces[i],
          this.vocab,
          this.merges,
          this.specialTokens,
          this.continuingSubwordPrefix,
          this.endOfWordSuffix,
          this.isByteLevel,
        ))
      }
    } else if (this.modelType === 'WordPiece') {
      const prefix = this.continuingSubwordPrefix ?? '##'
      for (let i = 0; i < pieces.length; i++) {
        ids.push(...encodeHFWordPieceWord(
          pieces[i],
          this.vocab,
          this.specialTokens,
          prefix,
          this.maxInputCharsPerWord,
        ))
      }
    } else {
      throw new TypeError(`Unsupported HF tokenizer model type: ${this.modelType}`)
    }

    if (opts?.addEos) ids.push(this.tokenIdOrThrow(this.specialTokens.eos, 'eos'))
    return truncateSingleIds(ids, opts)
  }

  decode(ids: readonly number[], opts?: TextDecodeOpts): string {
    if (this.modelType === 'WordPiece') {
      const words: string[] = []
      const prefix = this.continuingSubwordPrefix ?? '##'
      let current = ''
      for (let i = 0; i < ids.length; i++) {
        const token = this.reverse.get(ids[i])
        if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
        if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
        if (token.startsWith(prefix)) {
          current += token.slice(prefix.length)
          continue
        }
        if (current.length > 0) words.push(current)
        current = token
      }
      if (current.length > 0) words.push(current)
      return words.join(' ')
    }

    if (this.modelType === 'BPE') {
      if (this.isByteLevel) {
        let encoded = ''
        for (let i = 0; i < ids.length; i++) {
          const token = this.reverse.get(ids[i])
          if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
          if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
          encoded += token
        }
        return decodeHFByteLevel(encoded)
      }

      if (this.endOfWordSuffix) {
        const words: string[] = []
        let current = ''
        for (let i = 0; i < ids.length; i++) {
          const token = this.reverse.get(ids[i])
          if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
          if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
          const hasSuffix = token.endsWith(this.endOfWordSuffix)
          current += hasSuffix ? token.slice(0, token.length - this.endOfWordSuffix.length) : token
          if (hasSuffix) {
            words.push(current)
            current = ''
          }
        }
        if (current.length > 0) words.push(current)
        return words.join(' ')
      }

      const words: string[] = []
      let current = ''
      for (let i = 0; i < ids.length; i++) {
        const token = this.reverse.get(ids[i])
        if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
        if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
        if (this.continuingSubwordPrefix && token.startsWith(this.continuingSubwordPrefix)) {
          current += token.slice(this.continuingSubwordPrefix.length)
          continue
        }
        if (current.length > 0) words.push(current)
        current = token
      }
      if (current.length > 0) words.push(current)
      return words.join(' ')
    }

    const tokens: string[] = []
    for (let i = 0; i < ids.length; i++) {
      const token = this.reverse.get(ids[i])
      if (token === undefined) throw new TypeError(`Unknown token id: ${ids[i]}`)
      if (opts?.skipSpecialTokens && this.specialTokensSet.has(token)) continue
      tokens.push(token)
    }
    return tokens.join(' ')
  }
}

export function hfTokenizerFromJSON(spec: HFTokenizerJSON, opts?: LookupTokenizerOpts): TextTokenizer {
  return new HFTokenizer(spec, opts)
}

export function hfTokenizerFromFile(path: string, opts?: LookupTokenizerOpts): TextTokenizer {
  const raw = readFileSync(path)
  return hfTokenizerFromJSON(JSON.parse(raw) as HFTokenizerJSON, opts)
}

import { readFileSync } from 'std:fs'
import type {
  TextTokenizer as DatasetTextTokenizer,
} from 'affon:dataset'
import { hfTokenizerFromFile as createInternalHFTokenizerFromFile, hfTokenizerFromJSON as createInternalHFTokenizerFromJSON } from './hf.ts'
import { lookupTokenizer as createInternalLookupTokenizer } from './lookup.ts'
import { sentencePieceTokenizer as createInternalSentencePieceTokenizer, sentencePieceTokenizerFromFile as createInternalSentencePieceTokenizerFromFile } from './sentencepiece.ts'
import type {
  HFPreTokenizerSpec,
  HFTokenizerJSON,
  LookupTokenizerOpts,
  SentencePieceModel,
  SentencePieceTokenizerOpts,
} from './types.ts'

export type TokenizerFamily = 'lookup' | 'hf-tokenizer-json' | 'sentencepiece' | 'tiktoken'

export interface SpecialTokens {
  bos?: string
  eos?: string
  pad?: string
  unk?: string
  sep?: string
}

export interface TokenizerEncodeOptions {
  addBos?: boolean
  addEos?: boolean
}

export interface TokenizerDecodeOptions {
  skipSpecialTokens?: boolean
}

export interface Tokenizer extends DatasetTextTokenizer {
  readonly family: TokenizerFamily
  readonly specialTokens: Readonly<SpecialTokens>
  readonly specialTokenIds: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>
  readonly allSpecialTokenIds: ReadonlyArray<number>
  encode(text: string, opts?: TokenizerEncodeOptions): number[]
  decode(ids: readonly number[], opts?: TokenizerDecodeOptions): string
}

export interface TokenizerArtifact {
  readonly family: TokenizerFamily
  readonly files: ReadonlyArray<string>
}

export type {
  HFPreTokenizerSpec,
  HFTokenizerJSON,
  SentencePieceModel,
}
export type LookupTokenizerOptions = LookupTokenizerOpts
export type HFTokenizerOptions = Pick<LookupTokenizerOpts, 'specialTokens'>
export type SentencePieceTokenizerOptions = SentencePieceTokenizerOpts

function uniqueSortedTokenIds(ids: readonly number[]): ReadonlyArray<number> {
  return Object.freeze(Array.from(new Set(ids)).sort((a, b) => a - b))
}

function wrapDatasetTokenizer(
  tokenizer: DatasetTextTokenizer,
  family: TokenizerFamily,
  extraSpecialTokenIds: readonly number[] = [],
  explicitVocabSize?: number,
): Tokenizer {
  const specialTokens = Object.freeze({ ...(tokenizer.specialTokens ?? {}) })
  const specialTokenIds = Object.freeze({ ...(tokenizer.specialTokenIds ?? {}) })
  const allSpecialTokenIds = uniqueSortedTokenIds([
    ...Object.values(specialTokenIds).filter((value): value is number => typeof value === 'number'),
    ...extraSpecialTokenIds,
  ])
  const vocabSize = explicitVocabSize ?? tokenizer.vocabSize
  return {
    family,
    specialTokens,
    specialTokenIds,
    allSpecialTokenIds,
    vocabSize,
    tokenId(token: string): number | undefined {
      return tokenizer.tokenId(token)
    },
    token(id: number): string | undefined {
      return tokenizer.token(id)
    },
    encode(text: string, opts?: TokenizerEncodeOptions): number[] {
      return tokenizer.encode(text, opts)
    },
    decode(ids: readonly number[], opts?: TokenizerDecodeOptions): string {
      return tokenizer.decode(ids, opts)
    },
  }
}

export function createLookupTokenizer(
  tokenToId: Record<string, number>,
  opts?: LookupTokenizerOptions,
): Tokenizer {
  const explicitVocabSize = Math.max(...Object.values(tokenToId)) + 1
  return wrapDatasetTokenizer(
    createInternalLookupTokenizer(tokenToId, opts),
    'lookup',
    [],
    explicitVocabSize,
  )
}

export function createSentencePieceTokenizer(
  model: SentencePieceModel,
  opts?: SentencePieceTokenizerOptions,
): Tokenizer {
  const explicitVocabSize = model.pieces.length
  return wrapDatasetTokenizer(
    createInternalSentencePieceTokenizer(model, opts),
    'sentencepiece',
    [],
    explicitVocabSize,
  )
}

export function createSentencePieceTokenizerFromFile(
  path: string,
  opts?: SentencePieceTokenizerOptions,
): Tokenizer {
  const raw = readFileSync(path)
  const parsed = JSON.parse(raw) as SentencePieceModel
  return wrapDatasetTokenizer(
    createInternalSentencePieceTokenizerFromFile(path, opts),
    'sentencepiece',
    [],
    parsed.pieces.length,
  )
}

export function createHFTokenizerFromJSON(
  spec: HFTokenizerJSON,
  opts?: HFTokenizerOptions,
): Tokenizer {
  const extraSpecialTokenIds = (spec.added_tokens ?? [])
    .filter((token) => token.special)
    .map((token) => token.id)
  const explicitVocabSize = Math.max(
    ...Object.values(spec.model.vocab),
    ...extraSpecialTokenIds,
  ) + 1
  return wrapDatasetTokenizer(
    createInternalHFTokenizerFromJSON(spec, opts),
    'hf-tokenizer-json',
    extraSpecialTokenIds,
    explicitVocabSize,
  )
}

export function createHFTokenizerFromFile(
  path: string,
  opts?: HFTokenizerOptions,
): Tokenizer {
  const raw = readFileSync(path)
  const parsed = JSON.parse(raw) as HFTokenizerJSON
  const extraSpecialTokenIds = (parsed.added_tokens ?? [])
    .filter((token) => token.special)
    .map((token) => token.id)
  const explicitVocabSize = Math.max(
    ...Object.values(parsed.model.vocab),
    ...extraSpecialTokenIds,
  ) + 1
  return wrapDatasetTokenizer(
    createInternalHFTokenizerFromFile(path, opts),
    'hf-tokenizer-json',
    extraSpecialTokenIds,
    explicitVocabSize,
  )
}

export function tokenizerArtifact(
  family: TokenizerFamily,
  files: string[],
): TokenizerArtifact {
  return {
    family,
    files: Object.freeze(files.slice()),
  }
}

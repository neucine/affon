export interface SpecialTokens {
  bos?: string
  eos?: string
  pad?: string
  unk?: string
  sep?: string
}

export interface LookupTokenizerOpts {
  specialTokens?: SpecialTokens
  split?: 'whitespace' | 'char'
}

export interface TextEncodeOpts {
  addBos?: boolean
  addEos?: boolean
  maxLength?: number
  truncation?: 'longest_first' | 'only_first' | 'only_second'
}

export interface TextDecodeOpts {
  skipSpecialTokens?: boolean
}

export interface HFPreTokenizerSpec {
  type?: string
  add_prefix_space?: boolean
  pretokenizers?: HFPreTokenizerSpec[]
}

export interface HFTokenizerJSON {
  model: {
    type?: string
    vocab: Record<string, number>
    unk_token?: string
    merges?: Array<string | [string, string]>
    continuing_subword_prefix?: string
    end_of_word_suffix?: string
    max_input_chars_per_word?: number
  }
  pre_tokenizer?: HFPreTokenizerSpec | null
  added_tokens?: Array<{
    id: number
    content: string
    special?: boolean
  }>
}

export interface SentencePieceModel {
  pieces: ReadonlyArray<{
    piece: string
    score?: number
    type?: 'normal' | 'unknown' | 'control' | 'user_defined' | 'byte'
  }>
  unkId?: number
  bosId?: number
  eosId?: number
  padId?: number
}

export interface SentencePieceTokenizerOpts {
  specialTokens?: SpecialTokens
}

export interface TextTokenizer {
  readonly specialTokens?: Readonly<SpecialTokens>
  readonly specialTokenIds?: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>
  readonly vocabSize: number
  tokenId(token: string): number | undefined
  token(id: number): string | undefined
  encode(text: string, opts?: TextEncodeOpts): number[]
  decode(ids: readonly number[], opts?: TextDecodeOpts): string
}

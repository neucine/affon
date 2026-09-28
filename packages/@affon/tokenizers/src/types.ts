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
  individual_digits?: boolean
  add_prefix_space?: boolean
  /** Whether ByteLevel applies GPT-2 token boundaries before byte encoding. Defaults to true. */
  use_regex?: boolean
  pretokenizers?: HFPreTokenizerSpec[]
}

export interface HFTokenizerJSON {
  /** Supported BERT normalization; null means no normalization. */
  normalizer?: {
    type: 'BertNormalizer'
    clean_text?: boolean
    handle_chinese_chars?: boolean
    strip_accents?: boolean | null
    lowercase?: boolean
  } | null
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
    /** Match only outside adjacent word characters. */
    single_word?: boolean
    /** Consume whitespace immediately before/after the added token. */
    lstrip?: boolean
    rstrip?: boolean
    /** Serialized HF flag; transformed-text matching requires a normalizer (not implemented). */
    normalized?: boolean
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

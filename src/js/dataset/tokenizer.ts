interface TextEncodeOpts {
  addBos?: boolean
  addEos?: boolean
  maxLength?: number
  truncation?: 'longest_first' | 'only_first' | 'only_second'
}

interface TextDecodeOpts {
  skipSpecialTokens?: boolean
}

interface TextTokenizer {
  readonly specialTokens?: Readonly<{ bos?: string; eos?: string; pad?: string; unk?: string; sep?: string }>
  readonly specialTokenIds?: Readonly<{ bos?: number; eos?: number; pad?: number; unk?: number; sep?: number }>
  readonly vocabSize: number
  tokenId(token: string): number | undefined
  token(id: number): string | undefined
  encode(text: string, opts?: TextEncodeOpts): number[]
  decode(ids: readonly number[], opts?: TextDecodeOpts): string
}

export type {
  TextDecodeOpts,
  TextEncodeOpts,
  TextTokenizer,
}

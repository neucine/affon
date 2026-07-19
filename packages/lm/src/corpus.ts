import fs from 'affon:fs'
import dataset from 'affon:dataset'

import type { Tokenizer, TokenizerEncodeOptions } from '../../tokenizers/src/index.ts'
import type { PackedCorpusOptions, TokenWindow } from './token-windows.ts'

export type { PackedCorpusOptions, TokenWindow } from './token-windows.ts'

export type CorpusSplitMode = 'line' | 'paragraph'

export interface TextCorpusSplitOptions {
  mode?: CorpusSplitMode
  trim?: boolean
  skipEmpty?: boolean
}

export interface TextCorpusReplacement {
  from: string
  to: string
}

export interface TextCorpusPreprocessOptions {
  replacements?: TextCorpusReplacement[]
  normalizeWhitespace?: boolean
}

export interface TokenizeCorpusOptions extends TokenizerEncodeOptions {
  split?: TextCorpusSplitOptions
  preprocess?: TextCorpusPreprocessOptions
}

export interface PackedTextCorpusOptions extends PackedCorpusOptions, TokenizeCorpusOptions {
  validationSplit?: number
  shuffle?: boolean
}

export interface PackedFileCorpusConfig extends PackedTextCorpusOptions {
  path?: string
  paths?: string[]
  tokenCachePath?: string
  tokenCacheKey?: string
  validationPath?: string
  validationPaths?: string[]
  validationTokenCachePath?: string
  validationTokenCacheKey?: string
}

export interface PackedTextCorpus {
  trainRows: number[][]
  validationRows: number[][]
  trainWindows: TokenWindow[]
  validationWindows: TokenWindow[]
}

export function splitTextCorpus(
  text: string,
  opts: TextCorpusSplitOptions = {},
): string[] {
  return dataset.text.fromString(text, {
    mode: opts.mode ?? 'line',
    trim: opts.trim ?? true,
    skipEmpty: opts.skipEmpty ?? true,
  }).toArray()
}

function loadRowsFromPaths(
  paths: readonly string[],
  split: TextCorpusSplitOptions | undefined,
): string[] {
  const rows: string[] = []
  for (let i = 0; i < paths.length; i++) {
    const parts = loadRowsFromPath(paths[i], split)
    for (let j = 0; j < parts.length; j++) rows.push(parts[j])
  }
  return rows
}

function loadRowsFromPath(
  path: string,
  split: TextCorpusSplitOptions | undefined,
): string[] {
  if ((split?.mode ?? 'line') === 'line') {
    return dataset.text.read(path, {
      trim: split?.trim ?? true,
      skipEmpty: split?.skipEmpty ?? true,
    }).toArray()
  }
  return dataset.text.readParagraphs(path, {
    trim: split?.trim ?? true,
    skipEmpty: split?.skipEmpty ?? true,
  }).toArray()
}

function applyTextPreprocess(text: string, opts: TextCorpusPreprocessOptions | undefined): string {
  if (!opts) return text
  let out = text
  const replacements = opts.replacements ?? []
  for (let i = 0; i < replacements.length; i++) {
    const replacement = replacements[i]
    if (replacement.from.length === 0) {
      throw new AffonError('invalid_arg', 'TextCorpusPreprocessOptions replacements.from must be non-empty')
    }
    out = out.split(replacement.from).join(replacement.to)
  }
  if (opts.normalizeWhitespace) {
    out = out.replace(/\s+/g, ' ').trim()
  }
  return out
}

function preprocessRows(
  rows: readonly string[],
  opts: TextCorpusPreprocessOptions | undefined,
): string[] {
  if (!opts) return rows.slice()
  const out: string[] = []
  for (let i = 0; i < rows.length; i++) {
    const row = applyTextPreprocess(rows[i], opts)
    if (row.length > 0) out.push(row)
  }
  return out
}

function tokenizeCorpusPaths(
  tokenizer: Tokenizer,
  paths: readonly string[],
  opts: TokenizeCorpusOptions = {},
): number[][] {
  if ((opts.split?.mode ?? 'line') !== 'line') {
    return tokenizeCorpusRows(tokenizer, loadRowsFromPaths(paths, opts.split), opts)
  }

  const rows: number[][] = []
  for (let i = 0; i < paths.length; i++) {
    let textRows = dataset.text.read(paths[i], {
      trim: opts.split?.trim ?? true,
      skipEmpty: opts.split?.skipEmpty ?? true,
    })
    if (opts.preprocess) {
      textRows = textRows
        .map((text) => applyTextPreprocess(text, opts.preprocess))
        .filter((text) => text.length > 0)
    }
    const encoded = textRows.encode(tokenizer, {
      addBos: opts.addBos,
      addEos: opts.addEos,
    }).toArray()
    for (let j = 0; j < encoded.length; j++) rows.push(encoded[j])
  }
  return rows
}

function normalizePaths(path: string | undefined, paths: readonly string[] | undefined, name: string): string[] {
  const values = paths ? paths.slice() : path ? [path] : []
  if (values.length === 0) {
    throw new AffonError('invalid_arg', `${name} requires at least one path`)
  }
  return values
}

export function tokenizeCorpusRows(
  tokenizer: Tokenizer,
  rows: readonly string[],
  opts: TokenizeCorpusOptions = {},
): number[][] {
  return dataset.text.rows(preprocessRows(rows, opts.preprocess))
    .filter((text) => text.length > 0)
    .encode(tokenizer, {
      addBos: opts.addBos,
      addEos: opts.addEos,
    })
    .toArray()
}

export interface TokenRowsCacheMetadata {
  key?: string
}

export interface TokenRowsCacheFile {
  format: 'affon-lm-token-rows-cache/v1'
  metadata?: TokenRowsCacheMetadata
  rows: number[][]
}

function isTokenRows(value: unknown): value is number[][] {
  return Array.isArray(value)
    && value.every((row) => Array.isArray(row) && row.every((id) => Number.isInteger(id)))
}

export function saveTokenRows(path: string, rows: readonly number[][], metadata?: TokenRowsCacheMetadata): void {
  if (metadata?.key) {
    fs.writeFileSync(path, JSON.stringify({
      format: 'affon-lm-token-rows-cache/v1',
      metadata,
      rows,
    }))
    return
  }
  fs.writeFileSync(path, JSON.stringify(rows))
}

export function loadTokenRows(path: string): number[][] {
  const parsed = JSON.parse(fs.readFileSync(path)) as unknown
  if (isTokenRows(parsed)) return parsed
  if (
    parsed
    && typeof parsed === 'object'
    && (parsed as TokenRowsCacheFile).format === 'affon-lm-token-rows-cache/v1'
    && isTokenRows((parsed as TokenRowsCacheFile).rows)
  ) {
    return (parsed as TokenRowsCacheFile).rows
  }
  throw new AffonError('invalid_arg', `Invalid token rows cache: ${path}`)
}

function tryLoadTokenRows(path: string | undefined, expectedKey?: string): number[][] | null {
  if (!path) return null
  try {
    const parsed = JSON.parse(fs.readFileSync(path)) as unknown
    if (isTokenRows(parsed)) return expectedKey ? null : parsed
    if (
      parsed
      && typeof parsed === 'object'
      && (parsed as TokenRowsCacheFile).format === 'affon-lm-token-rows-cache/v1'
      && isTokenRows((parsed as TokenRowsCacheFile).rows)
    ) {
      const actualKey = (parsed as TokenRowsCacheFile).metadata?.key
      if (expectedKey !== undefined && actualKey !== expectedKey) return null
      return (parsed as TokenRowsCacheFile).rows
    }
    return null
  } catch {
    return null
  }
}

export function createPackedTextCorpus(
  tokenizer: Tokenizer,
  rows: readonly string[],
  opts: PackedTextCorpusOptions,
): PackedTextCorpus {
  if (opts.validationSplit !== undefined && (opts.validationSplit < 0 || opts.validationSplit >= 1)) {
    throw new AffonError('invalid_arg', 'createPackedTextCorpus validationSplit must be in the range [0, 1)')
  }

  const encoded = tokenizeCorpusRows(tokenizer, rows, opts)
  return createPackedTextCorpusFromEncodedRows(tokenizer, encoded, opts)
}

function createPackedTextCorpusFromEncodedRows(
  tokenizer: Tokenizer,
  encodedRows: readonly number[][],
  opts: PackedTextCorpusOptions,
): PackedTextCorpus {
  if (opts.validationSplit !== undefined && (opts.validationSplit < 0 || opts.validationSplit >= 1)) {
    throw new AffonError('invalid_arg', 'createPackedTextCorpus validationSplit must be in the range [0, 1)')
  }

  const prepared = opts.shuffle
    ? dataset.text.encoded(encodedRows).shuffle()
    : dataset.text.encoded(encodedRows)
  const [validationDataset, trainDataset] = opts.validationSplit
    ? prepared.split(opts.validationSplit)
    : [dataset.text.encoded([]), prepared]
  const validationRows = validationDataset.toArray()
  const trainRows = trainDataset.toArray()

  const joinWithTokenId = opts.joinWithTokenId
    ?? (opts.addEos
      ? tokenizer.specialTokenIds.eos
      : undefined)

  return {
    trainRows,
    validationRows,
    trainWindows: dataset.text.encoded(trainRows).window({
      seqLen: opts.seqLen,
      stride: opts.stride,
      joinWithTokenId,
    }).toArray(),
    validationWindows: validationRows.length > 0
      ? dataset.text.encoded(validationRows).window({
        seqLen: opts.seqLen,
        stride: opts.stride,
        joinWithTokenId,
      }).toArray()
      : [],
  }
}

export function createPackedTextCorpusFromFile(
  path: string,
  tokenizer: Tokenizer,
  opts: PackedTextCorpusOptions,
): PackedTextCorpus {
  const encoded = tokenizeCorpusPaths(tokenizer, [path], opts)
  return createPackedTextCorpusFromEncodedRows(tokenizer, encoded, opts)
}

export function createPackedTextCorpusFromFiles(
  paths: readonly string[],
  tokenizer: Tokenizer,
  opts: PackedTextCorpusOptions,
): PackedTextCorpus {
  const encoded = tokenizeCorpusPaths(tokenizer, paths, opts)
  return createPackedTextCorpusFromEncodedRows(tokenizer, encoded, opts)
}

export function createPackedTokenCorpus(
  tokenizer: Tokenizer,
  trainRows: readonly number[][],
  validationRows: readonly number[][],
  opts: PackedTextCorpusOptions,
): PackedTextCorpus {
  const joinWithTokenId = opts.joinWithTokenId
    ?? (opts.addEos
      ? tokenizer.specialTokenIds.eos
      : undefined)
  return {
    trainRows: trainRows.map((row) => row.slice()),
    validationRows: validationRows.map((row) => row.slice()),
    trainWindows: dataset.text.encoded(trainRows).window({
      seqLen: opts.seqLen,
      stride: opts.stride,
      joinWithTokenId,
    }).toArray(),
    validationWindows: validationRows.length > 0
      ? dataset.text.encoded(validationRows).window({
        seqLen: opts.seqLen,
        stride: opts.stride,
        joinWithTokenId,
      }).toArray()
      : [],
  }
}

export function createPackedTextCorpusFromConfig(
  tokenizer: Tokenizer,
  config: PackedFileCorpusConfig,
): PackedTextCorpus {
  const cachedTrainRows = tryLoadTokenRows(config.tokenCachePath, config.tokenCacheKey)
  const trainRows = cachedTrainRows
    ?? tokenizeCorpusPaths(
      tokenizer,
      normalizePaths(config.path, config.paths, 'createPackedTextCorpusFromConfig'),
      config,
    )
  if (config.tokenCachePath && !(config.validationPath || config.validationPaths || config.validationTokenCachePath) && config.validationSplit === undefined) {
    return createPackedTokenCorpus(tokenizer, trainRows, [], config)
  }

  const cachedValidationRows = tryLoadTokenRows(config.validationTokenCachePath, config.validationTokenCacheKey)
  const validationRows = cachedValidationRows
    ?? ((config.validationPath || config.validationPaths)
      ? tokenizeCorpusPaths(
        tokenizer,
        normalizePaths(config.validationPath, config.validationPaths, 'createPackedTextCorpusFromConfig validation'),
        config,
      )
      : [])

  if (cachedTrainRows === null && cachedValidationRows === null && validationRows.length === 0) {
    return createPackedTextCorpus(
      tokenizer,
      loadRowsFromPaths(normalizePaths(config.path, config.paths, 'createPackedTextCorpusFromConfig'), config.split),
      config,
    )
  }

  return createPackedTokenCorpus(tokenizer, trainRows, validationRows, config)
}

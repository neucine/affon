import type { Session, Tensor } from 'affon:compute'
import type { TextEncodeOpts, TextTokenizer } from 'affon:dataset/tokenizer.ts'
import { readDelimitedRows, readLines, readParagraphs, textFromString } from 'affon:dataset/text_io.ts'
import type { DelimitedTextReadOpts, TextReadOpts, TextSourceOpts } from 'affon:dataset/text_io.ts'
import { packTokenWindows } from 'affon:dataset/token_windows.ts'
import type { PackTokenWindowsOpts } from 'affon:dataset/token_windows.ts'
import { makeClonedBatchLoader, normalizePaddedBatchOpts, padInputIds, requirePositiveInt, shuffleItems, splitItems } from 'affon:dataset/text_utils.ts'
import {
  EncodedTextDataLoader,
  PaddedTextDataLoader,
  TensorizedTextDataLoader,
  TextDataLoader,
} from 'affon:dataset/text_loader.ts'
import type { PaddedTextDataLoaderOpts, TensorizedTextDataLoaderOpts } from 'affon:dataset/text_loader.ts'

type PairTemplatePart = 'bos' | 'left' | 'separator' | 'right' | 'eos'
type PairTemplatePreset = 'joined' | 'bert'

interface PairEncodeOpts extends TextEncodeOpts {
  separatorText?: string
  pairTemplatePreset?: PairTemplatePreset
  pairTemplate?: PairTemplatePart[]
  into?: string
  tokenTypesInto?: string
  withTokenTypes?: boolean
}

interface TextDelimitedReadOpts extends DelimitedTextReadOpts {
  columns: string[]
}

interface CloneableRecord {
  [key: string]: any
}

interface PairEncodedExample {
  inputIds: number[]
  tokenTypeIds: number[]
}

function cloneValue<T>(value: T): T {
  if (Array.isArray(value)) return value.map((item) => cloneValue(item)) as T
  if (value !== null && typeof value === 'object') {
    const out: CloneableRecord = {}
    for (const [key, inner] of Object.entries(value as CloneableRecord)) {
      out[key] = cloneValue(inner)
    }
    return out as T
  }
  return value
}

function cloneRecord<T extends CloneableRecord>(item: T): T {
  return cloneValue(item)
}

function specialTokenIds(tokenizer: TextTokenizer): { bos?: number; eos?: number; pad?: number; unk?: number; sep?: number } {
  return tokenizer.specialTokenIds ?? {}
}

function specialTokenArray(
  tokenizer: TextTokenizer,
  kind: 'bos' | 'eos' | 'sep',
  fallbackText?: string,
): number[] {
  const ids = specialTokenIds(tokenizer)
  const id = ids[kind]
  if (typeof id === 'number') return [id]
  if (fallbackText !== undefined) return tokenizer.encode(fallbackText)
  return []
}

function normalizePairEncodeOpts(
  opts?: TextEncodeOpts,
): { maxLength?: number; truncation: 'longest_first' | 'only_first' | 'only_second' } {
  const maxLength = opts?.maxLength
  if (maxLength !== undefined && (!Number.isFinite(maxLength) || maxLength <= 0 || !Number.isInteger(maxLength))) {
    throw new TypeError('TextEncodeOpts.maxLength must be a positive integer')
  }
  return {
    maxLength,
    truncation: opts?.truncation ?? 'longest_first',
  }
}

function truncatePairIds(
  left: number[],
  right: number[],
  available: number,
  strategy: 'longest_first' | 'only_first' | 'only_second',
): { left: number[]; right: number[] } {
  const outLeft = left.slice()
  const outRight = right.slice()
  if (available < 0) throw new TypeError('TextEncodeOpts.maxLength is smaller than required special tokens')

  while (outLeft.length + outRight.length > available) {
    if (strategy === 'only_first') {
      if (outLeft.length === 0) throw new TypeError('TextEncodeOpts.maxLength cannot satisfy pair truncation with only_first')
      outLeft.pop()
      continue
    }
    if (strategy === 'only_second') {
      if (outRight.length === 0) throw new TypeError('TextEncodeOpts.maxLength cannot satisfy pair truncation with only_second')
      outRight.pop()
      continue
    }
    if (outLeft.length >= outRight.length && outLeft.length > 0) {
      outLeft.pop()
      continue
    }
    if (outRight.length > 0) {
      outRight.pop()
      continue
    }
    break
  }

  return { left: outLeft, right: outRight }
}

function normalizePairTemplate(
  parts: PairTemplatePart[] | undefined,
  preset: PairTemplatePreset | undefined,
  defaults: { addBos?: boolean; addEos?: boolean; includeSeparator: boolean },
): PairTemplatePart[] {
  if (parts !== undefined && preset !== undefined) {
    throw new TypeError('pairTemplate and pairTemplatePreset cannot be used together')
  }

  const presetTemplate = preset === undefined
    ? undefined
    : preset === 'joined'
      ? ['left' as const, ...(defaults.includeSeparator ? ['separator' as const] : []), 'right' as const]
      : ['bos' as const, 'left' as const, ...(defaults.includeSeparator ? ['separator' as const] : []), 'right' as const, ...(defaults.includeSeparator ? ['separator' as const] : []), 'eos' as const]

  const template = parts ?? presetTemplate ?? [
    ...(defaults.addBos ? ['bos' as const] : []),
    'left' as const,
    ...(defaults.includeSeparator ? ['separator' as const] : []),
    'right' as const,
    ...(defaults.addEos ? ['eos' as const] : []),
  ]

  let leftCount = 0
  let rightCount = 0
  let bosCount = 0
  let eosCount = 0
  for (let i = 0; i < template.length; i++) {
    const part = template[i]
    if (part === 'left') leftCount += 1
    else if (part === 'right') rightCount += 1
    else if (part === 'bos') bosCount += 1
    else if (part === 'eos') eosCount += 1
    else if (part !== 'separator') throw new TypeError(`Unsupported pair template part: ${part}`)
  }
  if (leftCount !== 1 || rightCount !== 1) {
    throw new TypeError('pairTemplate must contain exactly one left and one right part')
  }
  if (bosCount > 1 || eosCount > 1) {
    throw new TypeError('pairTemplate supports at most one bos and one eos part')
  }
  return template.slice()
}

function buildPairSequence(
  left: number[],
  right: number[],
  template: PairTemplatePart[],
  tokens: { bos: number[]; eos: number[]; separator: number[] },
): PairEncodedExample {
  const inputIds: number[] = []
  const tokenTypeIds: number[] = []
  let currentSegment = 0
  for (let i = 0; i < template.length; i++) {
    const part = template[i]
    let chunk: number[]
    if (part === 'bos') {
      chunk = tokens.bos
    } else if (part === 'left') {
      chunk = left
      currentSegment = 0
    } else if (part === 'separator') {
      chunk = tokens.separator
    } else if (part === 'right') {
      currentSegment = 1
      chunk = right
    } else {
      chunk = tokens.eos
    }
    inputIds.push(...chunk)
    tokenTypeIds.push(...chunk.map(() => currentSegment))
  }
  return { inputIds, tokenTypeIds }
}

function staticTemplateTokenLength(
  template: PairTemplatePart[],
  tokens: { bos: number[]; eos: number[]; separator: number[] },
): number {
  let total = 0
  for (let i = 0; i < template.length; i++) {
    const part = template[i]
    if (part === 'bos') total += tokens.bos.length
    else if (part === 'eos') total += tokens.eos.length
    else if (part === 'separator') total += tokens.separator.length
  }
  return total
}

function remapSelections(fields: string[], previous: string, next: string): string[] {
  return fields.map((field) => field === previous ? next : field)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNumericVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => isFiniteNumber(item))
}

function isNumericMatrix(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every((item) => isNumericVector(item))
}

function collectNumbers(value: unknown, out: number[]): boolean {
  if (isFiniteNumber(value)) {
    out.push(value)
    return true
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      if (!collectNumbers(value[i], out)) return false
    }
    return true
  }
  return false
}

function inferTensorDType(value: unknown): 'i64' | 'f32' {
  const numbers: number[] = []
  if (!collectNumbers(value, numbers)) {
    throw new TypeError('tensorLoader() can only tensorize numeric selected fields')
  }
  return numbers.every((item) => Number.isInteger(item)) ? 'i64' : 'f32'
}

function tensorizeSelectedValue(value: unknown, session: Session): Tensor {
  return session.tensor(cloneValue(value), { dtype: inferTensorDType(value) })
}

function requireString(value: unknown, methodName: string, field: string): string {
  if (typeof value !== 'string') {
    throw new TypeError(`${methodName}() requires field "${field}" to be a string`)
  }
  return value
}

class LabelEncoder {
  private labelToId: Record<string, number>
  private idToLabel: Map<number, string>

  constructor(labelToId: Record<string, number>) {
    this.labelToId = { ...labelToId }
    this.idToLabel = new Map<number, string>()
    for (const [label, id] of Object.entries(labelToId)) {
      if (!Number.isInteger(id) || id < 0) throw new TypeError(`Invalid label id for label ${label}`)
      if (this.idToLabel.has(id)) throw new TypeError(`Duplicate label id: ${id}`)
      this.idToLabel.set(id, label)
    }
  }

  encode(label: string): number {
    const id = this.labelToId[label]
    if (id === undefined) throw new TypeError(`Unknown label: ${label}`)
    return id
  }

  decode(id: number): string {
    const label = this.idToLabel.get(id)
    if (label === undefined) throw new TypeError(`Unknown label id: ${id}`)
    return label
  }

  get size(): number {
    return this.idToLabel.size
  }

  encodeMany(labels: readonly string[]): number[] {
    return Array.from(new Set(labels.map((label) => this.encode(label))))
  }

  decodeMany(ids: readonly number[]): string[] {
    return ids.map((id) => this.decode(id))
  }
}

class EncodedTextDataset {
  private items: number[][]

  constructor(items: number[][]) {
    this.items = items.map((item) => item.slice())
  }

  get length(): number {
    return this.items.length
  }

  toArray(): number[][] {
    return this.items.map((item) => item.slice())
  }

  map(mapper: (ids: number[], index: number) => number[]): EncodedTextDataset {
    return new EncodedTextDataset(this.items.map((item, index) => mapper(item.slice(), index)))
  }

  filter(predicate: (ids: number[], index: number) => boolean): EncodedTextDataset {
    return new EncodedTextDataset(this.items.filter((item, index) => predicate(item.slice(), index)))
  }

  shuffle(): EncodedTextDataset {
    return new EncodedTextDataset(shuffleItems(this.items))
  }

  sample(n: number): EncodedTextDataset {
    if (!Number.isFinite(n) || n < 0 || !Number.isInteger(n)) {
      throw new TypeError('sample() requires a non-negative integer')
    }
    return new EncodedTextDataset(this.items.slice(0, n))
  }

  split(...ratios: number[]): EncodedTextDataset[] {
    return splitItems(this.items, ratios).map((part) => new EncodedTextDataset(part))
  }

  window(opts: PackTokenWindowsOpts): EncodedTextDataset {
    return new EncodedTextDataset(packTokenWindows(this.items, opts))
  }

  loader(opts?: { batchSize?: number }): EncodedTextDataLoader {
    return new EncodedTextDataLoader(this, opts)
  }

  paddedLoader(opts: PaddedTextDataLoaderOpts): PaddedTextDataLoader {
    return new PaddedTextDataLoader(this, opts)
  }

  tensorLoader(opts: TensorizedTextDataLoaderOpts): TensorizedTextDataLoader {
    return new TensorizedTextDataLoader(this, opts)
  }
}

class TextDataset {
  private items: string[]

  constructor(items: string[]) {
    this.items = items.slice()
  }

  get length(): number {
    return this.items.length
  }

  toArray(): string[] {
    return this.items.slice()
  }

  map(mapper: (text: string, index: number) => string): TextDataset {
    return new TextDataset(this.items.map((item, index) => mapper(item, index)))
  }

  filter(predicate: (text: string, index: number) => boolean): TextDataset {
    return new TextDataset(this.items.filter((item, index) => predicate(item, index)))
  }

  shuffle(): TextDataset {
    return new TextDataset(shuffleItems(this.items))
  }

  sample(n: number): TextDataset {
    if (!Number.isFinite(n) || n < 0 || !Number.isInteger(n)) {
      throw new TypeError('sample() requires a non-negative integer')
    }
    return new TextDataset(this.items.slice(0, n))
  }

  split(...ratios: number[]): TextDataset[] {
    return splitItems(this.items, ratios).map((part) => new TextDataset(part))
  }

  loader(opts?: { batchSize?: number }): TextDataLoader {
    return new TextDataLoader(this, opts)
  }

  encode(tokenizer: TextTokenizer, opts?: TextEncodeOpts): EncodedTextDataset {
    return new EncodedTextDataset(this.items.map((item) => tokenizer.encode(item, opts)))
  }

  records(field = 'text'): TextRecordDataset {
    return new TextRecordDataset(this.items.map((text) => ({ [field]: text })))
  }
}

class TextRecordDataset {
  private items: CloneableRecord[]
  private selectedInputs: string[]
  private selectedTargets: string[]

  constructor(items: CloneableRecord[], selectedInputs: string[] = [], selectedTargets: string[] = []) {
    this.items = items.map((item) => cloneRecord(item))
    this.selectedInputs = selectedInputs.slice()
    this.selectedTargets = selectedTargets.slice()
  }

  private with(items: CloneableRecord[], selectedInputs = this.selectedInputs, selectedTargets = this.selectedTargets): TextRecordDataset {
    return new TextRecordDataset(items, selectedInputs, selectedTargets)
  }

  get length(): number {
    return this.items.length
  }

  toArray(): CloneableRecord[] {
    return this.items.map((item) => cloneRecord(item))
  }

  map(mapper: (item: CloneableRecord, index: number) => CloneableRecord): TextRecordDataset {
    return this.with(this.items.map((item, index) => mapper(cloneRecord(item), index)))
  }

  filter(predicate: (item: CloneableRecord, index: number) => boolean): TextRecordDataset {
    return this.with(this.items.filter((item, index) => predicate(cloneRecord(item), index)))
  }

  shuffle(): TextRecordDataset {
    return this.with(shuffleItems(this.items))
  }

  sample(n: number): TextRecordDataset {
    if (!Number.isFinite(n) || n < 0 || !Number.isInteger(n)) {
      throw new TypeError('sample() requires a non-negative integer')
    }
    return this.with(this.items.slice(0, n))
  }

  split(...ratios: number[]): TextRecordDataset[] {
    return splitItems(this.items, ratios).map((part) => this.with(part))
  }

  loader(opts?: { batchSize?: number }): { length: number; [Symbol.iterator](): Iterator<CloneableRecord[]> } {
    const items = this.toArray()
    const batchSize = opts?.batchSize ?? 32
    requirePositiveInt(batchSize, 'TextRecordDataLoader.batchSize')
    return makeClonedBatchLoader(items, batchSize, (item) => cloneRecord(item))
  }

  encode(field: string, tokenizer: TextTokenizer, opts?: TextEncodeOpts & { into?: string }): TextRecordDataset {
    const into = opts?.into ?? field
    return this.with(
      this.items.map((item) => {
        const out = cloneRecord(item)
        out[into] = tokenizer.encode(requireString(item[field], 'encode', field), opts)
        return out
      }),
      remapSelections(this.selectedInputs, field, into),
      remapSelections(this.selectedTargets, field, into),
    )
  }

  splitLabels(field: string, opts?: { delimiter?: string; trim?: boolean; into?: string }): TextRecordDataset {
    const delimiter = opts?.delimiter ?? ','
    const into = opts?.into ?? field
    return this.with(
      this.items.map((item) => {
        const out = cloneRecord(item)
        out[into] = requireString(item[field], 'splitLabels', field)
          .split(delimiter)
          .map((part) => opts?.trim ? part.trim() : part)
          .filter((part) => part.length > 0)
        return out
      }),
      remapSelections(this.selectedInputs, field, into),
      remapSelections(this.selectedTargets, field, into),
    )
  }

  encodeLabels(field: string, labelEncoder: LabelEncoder, opts?: { into?: string }): TextRecordDataset {
    const into = opts?.into ?? field
    return this.with(
      this.items.map((item) => {
        const out = cloneRecord(item)
        out[into] = labelEncoder.encode(requireString(item[field], 'encodeLabels', field))
        return out
      }),
      remapSelections(this.selectedInputs, field, into),
      remapSelections(this.selectedTargets, field, into),
    )
  }

  toMultiHot(field: string, labelEncoder: LabelEncoder, opts?: { into?: string }): TextRecordDataset {
    const into = opts?.into ?? field
    return this.with(
      this.items.map((item) => {
        const labels = item[field]
        if (!Array.isArray(labels) || !labels.every((part) => typeof part === 'string')) {
          throw new TypeError(`toMultiHot() requires field "${field}" to be a string[]`)
        }
        const row = Array.from({ length: labelEncoder.size }, () => 0)
        for (const id of labelEncoder.encodeMany(labels)) row[id] = 1
        const out = cloneRecord(item)
        out[into] = row
        return out
      }),
      remapSelections(this.selectedInputs, field, into),
      remapSelections(this.selectedTargets, field, into),
    )
  }

  cast(field: string, dtype: 'i64' | 'f32', opts?: { into?: string }): TextRecordDataset {
    const into = opts?.into ?? field
    return this.with(
      this.items.map((item) => {
        const numeric = typeof item[field] === 'number' ? item[field] : Number(item[field])
        if (!Number.isFinite(numeric)) throw new TypeError(`cast() requires field "${field}" to be numeric`)
        if (dtype === 'i64' && !Number.isInteger(numeric)) {
          throw new TypeError(`cast() requires integer-compatible values for "${field}"`)
        }
        const out = cloneRecord(item)
        out[into] = numeric
        return out
      }),
      remapSelections(this.selectedInputs, field, into),
      remapSelections(this.selectedTargets, field, into),
    )
  }

  encodePair(fields: { left: string; right: string }, tokenizer: TextTokenizer, opts?: PairEncodeOpts): TextRecordDataset {
    const into = opts?.into ?? 'inputIds'
    const tokenTypesInto = opts?.tokenTypesInto ?? 'tokenTypeIds'
    const includeTokenTypes = opts?.withTokenTypes ?? false
    const separatorText = opts?.separatorText
    const { maxLength, truncation } = normalizePairEncodeOpts(opts)
    return this.with(
      this.items.map((item) => {
        const left = tokenizer.encode(requireString(item[fields.left], 'encodePair', fields.left))
        const right = tokenizer.encode(requireString(item[fields.right], 'encodePair', fields.right))
        const template = normalizePairTemplate(
          opts?.pairTemplate,
          opts?.pairTemplatePreset,
          {
            addBos: opts?.addBos,
            addEos: opts?.addEos,
            includeSeparator: opts?.pairTemplatePreset !== undefined || separatorText !== undefined || specialTokenIds(tokenizer).sep !== undefined,
          },
        )
        const tokens = {
          bos: opts?.addBos || template.includes('bos') ? specialTokenArray(tokenizer, 'bos') : [],
          eos: opts?.addEos || template.includes('eos') ? specialTokenArray(tokenizer, 'eos') : [],
          separator: specialTokenArray(tokenizer, 'sep', opts?.pairTemplatePreset === 'bert' ? separatorText : (separatorText ?? ' ')),
        }
        if (opts?.pairTemplatePreset === 'bert') {
          if (tokens.bos.length < 1) throw new TypeError('bert pairTemplatePreset requires tokenizer BOS support')
          if (tokens.eos.length < 1) throw new TypeError('bert pairTemplatePreset requires tokenizer EOS support')
          if (tokens.separator.length < 1) throw new TypeError('bert pairTemplatePreset requires tokenizer separator metadata or separatorText')
        }
        const available = maxLength === undefined ? undefined : maxLength - staticTemplateTokenLength(template, tokens)
        const truncated = available === undefined ? { left, right } : truncatePairIds(left, right, available, truncation)
        const pair = buildPairSequence(truncated.left, truncated.right, template, tokens)
        const out = cloneRecord(item)
        out[into] = pair.inputIds
        if (includeTokenTypes) out[tokenTypesInto] = pair.tokenTypeIds
        return out
      }),
      remapSelections(this.selectedInputs, fields.left, into),
      this.selectedTargets.slice(),
    )
  }

  input(...fields: string[]): TextRecordDataset {
    return this.with(this.items, fields, this.selectedTargets)
  }

  target(...fields: string[]): TextRecordDataset {
    return this.with(this.items, this.selectedInputs, fields)
  }

  paddedLoader(opts: PaddedTextDataLoaderOpts): { length: number; [Symbol.iterator](): Iterator<CloneableRecord> } {
    if (this.selectedInputs.length < 1 && this.selectedTargets.length < 1) {
      throw new TypeError('paddedLoader() requires selected input or target fields')
    }
    const items = this.toArray()
    const { batchSize, padId, maxLength } = normalizePaddedBatchOpts(
      opts,
      'TextPreparedDataLoader.batchSize',
      'TextPreparedDataLoader.padId',
      'TextPreparedDataLoader.maxLength',
    )
    const inputFields = this.selectedInputs.slice()
    const targetFields = this.selectedTargets.slice()
    return {
      length: Math.ceil(items.length / batchSize),
      [Symbol.iterator](): Iterator<CloneableRecord> {
        let offset = 0
        return {
          next(): IteratorResult<CloneableRecord> {
            if (offset >= items.length) return { done: true, value: undefined as any }
            const batch = items.slice(offset, offset + batchSize)
            offset += batchSize

            const out: CloneableRecord = {}

            for (const field of inputFields) {
              const values = batch.map((item) => cloneValue(item[field]))
              if (values.every((value) => isNumericVector(value))) {
                const padded = padInputIds(values, padId, maxLength)
                out[field] = padded.inputIds
                out[field === 'inputIds' ? 'attentionMask' : `${field}AttentionMask`] = padded.attentionMask
              } else {
                out[field] = values
              }
            }

            for (const field of targetFields) {
              out[field] = batch.map((item) => cloneValue(item[field]))
            }

            return { done: false, value: out }
          }
        }
      }
    }
  }

  tensorLoader(opts: TensorizedTextDataLoaderOpts): { length: number; [Symbol.iterator](): Iterator<CloneableRecord> } {
    const padded = this.paddedLoader(opts)
    const session = opts.session
    return {
      length: padded.length,
      [Symbol.iterator](): Iterator<CloneableRecord> {
        const inner = padded[Symbol.iterator]()
        return {
          next(): IteratorResult<CloneableRecord> {
            const step = inner.next()
            if (step.done) return { done: true, value: undefined as any }
            const out: CloneableRecord = {}
            for (const [key, value] of Object.entries(step.value)) {
              if (
                isNumericVector(value)
                || isNumericMatrix(value)
                || isFiniteNumber(value)
              ) {
                out[key] = tensorizeSelectedValue(value, session)
                continue
              }
              throw new TypeError(`tensorLoader() can only tensorize numeric selected fields; "${key}" is not numeric`)
            }
            return { done: false, value: out }
          }
        }
      }
    }
  }
}

function readText(source: string, opts?: TextReadOpts): TextDataset {
  return new TextDataset(readLines(source, opts))
}

function readParagraphText(source: string, opts?: TextReadOpts): TextDataset {
  return new TextDataset(readParagraphs(source, opts))
}

function textRows(items: readonly string[]): TextDataset {
  return new TextDataset(items.slice())
}

function textFromRawString(raw: string, opts?: TextSourceOpts): TextDataset {
  return new TextDataset(textFromString(raw, opts))
}

function readDelimitedText(source: string, opts: TextDelimitedReadOpts): TextRecordDataset {
  if (!Array.isArray(opts?.columns) || opts.columns.length < 1) {
    throw new TypeError('readDelimited() requires a non-empty columns array')
  }
  const rows = readDelimitedRows(source, opts)
  return new TextRecordDataset(rows.map((row) => {
    if (row.length < opts.columns.length) {
      throw new TypeError(`readDelimited() requires at least ${opts.columns.length} delimited fields per row`)
    }
    const out: CloneableRecord = {}
    for (let i = 0; i < opts.columns.length; i++) out[opts.columns[i]] = row[i]
    return out
  }))
}

function encodedText(items: readonly (readonly number[])[]): EncodedTextDataset {
  return new EncodedTextDataset(items.map((item) => item.slice()))
}

const text = {
  read: readText,
  readParagraphs: readParagraphText,
  fromString: textFromRawString,
  rows: textRows,
  encoded: encodedText,
  readDelimited: readDelimitedText,
  DataLoader: TextDataLoader,
  EncodedTextDataLoader,
  PaddedTextDataLoader,
  TensorizedTextDataLoader,
  labelEncoder(labelToId: Record<string, number>) {
    return new LabelEncoder(labelToId)
  },
}

export { EncodedTextDataset, LabelEncoder, encodedText, readText, readParagraphText, readDelimitedText, textRows, text, TextDataLoader, TextDataset, TextRecordDataset, EncodedTextDataLoader, PaddedTextDataLoader, TensorizedTextDataLoader }

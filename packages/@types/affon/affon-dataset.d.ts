declare module "affon:dataset" {
  import type { Session, Tensor } from "affon:compute"

  interface Schema {
    [column: string]: number[]
  }

  export interface TensorResult {
    data: Tensor
    schema: Schema
  }

  export interface TensorsResult {
    X: Tensor
    y: Tensor
    schema: Schema
  }

  export interface ToTensorOpts {
    /** Session that owns the exported tensor. */
    session: Session
    /** Numeric dtype for exported tensors. Default: `"f32"`. */
    dtype?: 'f32' | 'f64'
  }

  export interface Dataset {
    /** Current column names */
    readonly columns: string[]

    /**
     * Select columns by name.
     * @example ds.select('sepal_length', 'species')
     */
    select(...columns: string[]): Dataset

    /**
     * Drop columns by name.
     * @example ds.drop('id')
     */
    drop(...columns: string[]): Dataset

    /**
     * Encode a string column to numeric. Default strategy: `'onehot'`.
     * @example ds.encode('species', 'label')
     */
    encode(column: string, strategy?: 'onehot' | 'label'): Dataset

    /**
     * Normalize numeric columns to `[0, 1]` range.
     * @example ds.normalize('sepal_length', 'sepal_width')
     */
    normalize(...columns: string[]): Dataset

    /**
     * Standardize numeric columns with z-score scaling.
     * @example ds.standardize('sepal_length')
     */
    standardize(...columns: string[]): Dataset

    /** Log-transform numeric columns. */
    log(...columns: string[]): Dataset

    /**
     * Clip numeric columns to `[min, max]`.
     * @example ds.clip('sepal_length', 'sepal_width', 0, 10)
     */
    clip(...args: [...string[], number, number]): Dataset

    /** Replace `NaN` values in a numeric column with a fill value. */
    fillna(column: string, value: number): Dataset

    /** Drop rows where any specified column has `NaN`. No args checks all columns. */
    dropna(...columns: string[]): Dataset

    /** Rename a column. */
    rename(oldName: string, newName: string): Dataset

    /** Mark feature columns for `toTensors()`. */
    features(...columns: string[]): Dataset

    /** Mark model input columns for `toTensors()`. Alias of `.features(...)`. */
    input(...columns: string[]): Dataset

    /** Mark target column for `toTensors()`. */
    target(column: string): Dataset

    /**
     * Shuffle rows.
     * @example ds.shuffle()
     */
    shuffle(): Dataset

    /** Take the first `n` rows. `n` must be a non-negative integer. */
    sample(n: number): Dataset

    /** Vertically concatenate another dataset. Columns must match. */
    concat(other: Dataset): Dataset

    /**
     * Split into parts by ratios. The last part is the remainder.
     * Supports at most 8 explicit ratio arguments.
     * @example const [train, val, test] = ds.split(0.7, 0.15)
     */
    split<T extends number[]>(...ratios: [...T]): [...{ [K in keyof T]: Dataset }, Dataset]

    /**
     * Collect all selected rows and columns into a single tensor.
     * Exports `f32` by default for canonical `Session` execution and `affon:nn` layers.
     * @example const { data, schema } = ds.select('x1', 'x2').toTensor()
     */
    toTensor(opts: ToTensorOpts): TensorResult

    /**
     * Split into feature and target tensors.
     * Requires `.features()` and `.target()` to be set first.
     * Exports `f32` by default for canonical `Session` execution and `affon:nn` layers.
     */
    toTensors(opts: ToTensorOpts): TensorsResult

    /**
     * Create a `DataLoader` from this dataset.
     * @example const loader = ds.loader({ batchSize: 32 })
     */
    loader(opts: DataLoaderOpts): DataLoader

    /** Alias of `.loader(...)` for cross-domain pipeline consistency. */
    tensorLoader(opts: DataLoaderOpts): DataLoader
  }

  export interface DataLoaderOpts {
    /** Session that owns source and batch tensors. */
    session: Session
    /** Positive integer batch size. Default: `32`. */
    batchSize?: number
  }

  export interface TextReadOpts {
    /** Trim each loaded line before building the dataset. Default: `false`. */
    trim?: boolean
    /** Drop empty lines after optional trimming. Default: `true`. */
    skipEmpty?: boolean
  }

  export interface DelimitedTextReadOpts extends TextReadOpts {
    delimiter?: string
  }

  export interface TextSourceOpts extends TextReadOpts {
    mode?: 'line' | 'paragraph'
  }

  export interface TextDelimitedReadOpts extends DelimitedTextReadOpts {
    columns: string[]
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

  export interface PaddedTextDataLoaderOpts {
    batchSize?: number
    padId: number
    maxLength?: number
  }

  export interface TensorizedTextDataLoaderOpts extends PaddedTextDataLoaderOpts {
    /** Session that owns every tensor yielded by this loader. */
    session: Session
  }

  export interface TextWindowOpts {
    seqLen: number
    stride?: number
    joinWithTokenId?: number
  }

  /** Iterates over tensors in batches for training loops. */
  export class DataLoader {
    constructor(dataset: Dataset, opts?: DataLoaderOpts)
    /** Number of batches */
    readonly length: number;
    [Symbol.iterator](): Iterator<Tensor[]>
  }

  export interface TextDataset {
    readonly length: number
    toArray(): string[]
    map(mapper: (text: string, index: number) => string): TextDataset
    filter(predicate: (text: string, index: number) => boolean): TextDataset
    shuffle(): TextDataset
    sample(n: number): TextDataset
    split<T extends number[]>(...ratios: [...T]): [...{ [K in keyof T]: TextDataset }, TextDataset]
    loader(opts?: DataLoaderOpts): TextDataLoader
    encode(tokenizer: TextTokenizer, opts?: TextEncodeOpts): EncodedTextDataset
    records(field?: string): TextRecordDataset
  }

  export class TextDataLoader {
    constructor(dataset: TextDataset, opts?: DataLoaderOpts)
    readonly length: number
    [Symbol.iterator](): Iterator<string[]>
  }

  export interface EncodedTextDataset {
    readonly length: number
    toArray(): number[][]
    map(mapper: (ids: number[], index: number) => number[]): EncodedTextDataset
    filter(predicate: (ids: number[], index: number) => boolean): EncodedTextDataset
    shuffle(): EncodedTextDataset
    sample(n: number): EncodedTextDataset
    split<T extends number[]>(...ratios: [...T]): [...{ [K in keyof T]: EncodedTextDataset }, EncodedTextDataset]
    window(opts: TextWindowOpts): EncodedTextDataset
    loader(opts?: DataLoaderOpts): EncodedTextDataLoader
    paddedLoader(opts: PaddedTextDataLoaderOpts): PaddedTextDataLoader
    tensorLoader(opts: TensorizedTextDataLoaderOpts): TensorizedTextDataLoader
  }

  export class EncodedTextDataLoader {
    constructor(dataset: EncodedTextDataset, opts?: DataLoaderOpts)
    readonly length: number
    [Symbol.iterator](): Iterator<number[][]>
  }

  export class PaddedTextDataLoader {
    constructor(dataset: EncodedTextDataset, opts: PaddedTextDataLoaderOpts)
    readonly length: number
    [Symbol.iterator](): Iterator<{ inputIds: number[][]; attentionMask: number[][] }>
  }

  export class TensorizedTextDataLoader {
    constructor(dataset: EncodedTextDataset, opts: TensorizedTextDataLoaderOpts)
    readonly length: number
    [Symbol.iterator](): Iterator<{
      inputIds: Tensor
      attentionMask: Tensor
    }>
  }

  export interface TextTokenizer {
    readonly specialTokens?: Readonly<{
      bos?: string
      eos?: string
      pad?: string
      unk?: string
      sep?: string
    }>
    readonly specialTokenIds?: Readonly<{
      bos?: number
      eos?: number
      pad?: number
      unk?: number
      sep?: number
    }>
    readonly vocabSize: number
    tokenId(token: string): number | undefined
    token(id: number): string | undefined
    encode(text: string, opts?: TextEncodeOpts): number[]
    decode(ids: readonly number[], opts?: TextDecodeOpts): string
  }

  export interface LabelEncoder {
    readonly size: number
    encode(label: string): number
    decode(id: number): string
    encodeMany(labels: readonly string[]): number[]
    decodeMany(ids: readonly number[]): string[]
  }

  export interface TextRecord {
    [field: string]: unknown
  }

  export interface TextSplitLabelsOpts {
    delimiter?: string
    trim?: boolean
    into?: string
  }

  export interface TextEncodeFieldOpts extends TextEncodeOpts {
    into?: string
  }

  export type PairTemplatePart = 'bos' | 'left' | 'separator' | 'right' | 'eos'
  export type PairTemplatePreset = 'joined' | 'bert'

  export interface TextEncodePairOpts extends TextEncodeOpts {
    separatorText?: string
    pairTemplatePreset?: PairTemplatePreset
    pairTemplate?: PairTemplatePart[]
    into?: string
    tokenTypesInto?: string
    withTokenTypes?: boolean
  }

  export interface TextPreparedBatch {
    [field: string]: unknown
  }

  export interface TextTensorBatch {
    [field: string]: Tensor | unknown
  }

  export interface TextRecordDataset {
    readonly length: number
    toArray(): TextRecord[]
    map(mapper: (item: TextRecord, index: number) => TextRecord): TextRecordDataset
    filter(predicate: (item: TextRecord, index: number) => boolean): TextRecordDataset
    shuffle(): TextRecordDataset
    sample(n: number): TextRecordDataset
    split<T extends number[]>(...ratios: [...T]): [...{ [K in keyof T]: TextRecordDataset }, TextRecordDataset]
    loader(opts?: DataLoaderOpts): { length: number; [Symbol.iterator](): Iterator<TextRecord[]> }
    encode(field: string, tokenizer: TextTokenizer, opts?: TextEncodeFieldOpts): TextRecordDataset
    splitLabels(field: string, opts?: TextSplitLabelsOpts): TextRecordDataset
    encodeLabels(field: string, labelEncoder: LabelEncoder, opts?: { into?: string }): TextRecordDataset
    toMultiHot(field: string, labelEncoder: LabelEncoder, opts?: { into?: string }): TextRecordDataset
    cast(field: string, dtype: 'i64' | 'f32', opts?: { into?: string }): TextRecordDataset
    encodePair(fields: { left: string; right: string }, tokenizer: TextTokenizer, opts?: TextEncodePairOpts): TextRecordDataset
    input(...fields: string[]): TextRecordDataset
    target(...fields: string[]): TextRecordDataset
    paddedLoader(opts: PaddedTextDataLoaderOpts): {
      length: number
      [Symbol.iterator](): Iterator<TextPreparedBatch>
    }
    tensorLoader(opts: TensorizedTextDataLoaderOpts): {
      length: number
      [Symbol.iterator](): Iterator<TextTensorBatch>
    }
  }

  /**
   * Read data from a CSV file path.
   * @example const ds = read('iris.csv').select('sepal_length', 'species')
   */
  export function read(source: string): Dataset

  export interface TabularModule {
    read(source: string): Dataset
    DataLoader: typeof DataLoader
  }

  export interface TextModule {
    read(source: string, opts?: TextReadOpts): TextDataset
    readParagraphs(source: string, opts?: TextReadOpts): TextDataset
    fromString(text: string, opts?: TextSourceOpts): TextDataset
    rows(items: readonly string[]): TextDataset
    encoded(rows: readonly (readonly number[])[]): EncodedTextDataset
    readDelimited(source: string, opts: TextDelimitedReadOpts): TextRecordDataset
    DataLoader: typeof TextDataLoader
    EncodedTextDataLoader: typeof EncodedTextDataLoader
    PaddedTextDataLoader: typeof PaddedTextDataLoader
    TensorizedTextDataLoader: typeof TensorizedTextDataLoader
    labelEncoder(labelToId: Record<string, number>): LabelEncoder
  }

  export const tabular: TabularModule
  export const text: TextModule

  const datasetModule: {
    read: typeof read
    DataLoader: typeof DataLoader
    tabular: TabularModule
    text: TextModule
  }

  export default datasetModule
}

declare module "affon:dataset/tokenizer.ts" {
  export type {
    TextDecodeOpts,
    TextEncodeOpts,
    TextTokenizer,
  } from "affon:dataset"
}

declare module "affon:dataset/text.ts" {
  import type {
    EncodedTextDataLoader,
    PaddedTextDataLoader,
    TensorizedTextDataLoader,
    TextDataLoader,
    TextModule,
  } from "affon:dataset"

  export const text: TextModule
  export function readText(...args: Parameters<TextModule['read']>): ReturnType<TextModule['read']>
  export { TextDataLoader, EncodedTextDataLoader, PaddedTextDataLoader, TensorizedTextDataLoader }
}

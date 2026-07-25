// Runtime API exposed by the public dataset module (affon:dataset).
// Runtime declaration file for the public dataset module.
// User-facing types are in packages/@types/affon/affon-dataset.d.ts (a curated subset).
// Note: DataLoader is added by the TypeScript layer (index.ts), not native.

interface Dataset {
  readonly columns: string[]

  // Pipeline methods (lazy, return new Dataset)
  select(...cols: string[]): Dataset
  drop(...cols: string[]): Dataset
  encode(col: string, encoding: 'label' | 'onehot'): Dataset
  normalize(...cols: string[]): Dataset
  standardize(...cols: string[]): Dataset
  log(...cols: string[]): Dataset
  clip(col: string, min: number, max: number): Dataset
  shuffle(): Dataset
  sample(n: number): Dataset
  fillna(col: string, value: number): Dataset
  dropna(): Dataset
  rename(from: string, to: string): Dataset
  features(...cols: string[]): Dataset
  input(...cols: string[]): Dataset
  target(col: string): Dataset

  // Terminal methods
  concat(other: Dataset): Dataset
  split(...ratios: number[]): Dataset[]
  toTensor(opts?: { dtype?: "f32" | "f64" }): { data: any; schema: any }
  toTensors(opts?: { dtype?: "f32" | "f64" }): { X: any; y: any; schema: any }
  tensorLoader(opts?: { batchSize?: number }): any
}

interface DatasetModule {
  read(path: string): Dataset
}

import 'affon:errors'
import { readFileSync } from 'std:fs'
import type { Session, Tensor } from 'affon:core'
import { slice } from 'affon:ops'

interface TensorExportOpts {
  session: Session
  dtype?: 'f32' | 'f64'
}

interface DataLoaderOpts {
  session: Session
  batchSize?: number
}

interface Column {
  name: string
  kind: 'numeric' | 'categorical'
  numeric: number[]
  strings: string[]
  categories: string[]
}

interface Selection {
  features: string[] | null
  target: string | null
}

function invalid(message: string): never {
  throw new AffonError('invalid_arg', message)
}

function parseCsv(source: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let index = 0; index < source.length; index++) {
    const character = source[index]
    if (quoted) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          cell += '"'
          index++
        } else {
          quoted = false
        }
      } else {
        cell += character
      }
      continue
    }
    if (character === '"') {
      quoted = true
    } else if (character === ',') {
      row.push(cell)
      cell = ''
    } else if (character === '\n') {
      row.push(cell)
      rows.push(row)
      row = []
      cell = ''
    } else if (character !== '\r') {
      cell += character
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell)
    rows.push(row)
  }
  while (rows.length > 0 && rows[rows.length - 1].every(value => value.length === 0)) rows.pop()
  return rows
}

function numericCell(value: string): number | null {
  const trimmed = value.trim()
  if (trimmed.length === 0) return NaN
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) || trimmed === 'NaN' || trimmed === 'Infinity' || trimmed === '-Infinity'
    ? parsed
    : null
}

function parseColumns(source: string): Column[] {
  const rows = parseCsv(source)
  if (rows.length === 0 || rows[0].length === 0) invalid('read(): EmptyCSV')
  const names = rows[0]
  const dataRows = rows.slice(1)
  const firstRowNumeric = dataRows.length > 0 &&
    dataRows[0].every(value => numericCell(value) !== null)
  return names.map((name, column) => {
    const raw = dataRows.map(row => row[column] ?? '')
    const numeric = raw.map(value => numericCell(value))
    if (firstRowNumeric || numeric.every(value => value !== null)) {
      return {
        name,
        kind: 'numeric',
        numeric: numeric.map(value => {
          if (value === null) invalid('read(): InvalidNumericValue')
          return value
        }),
        strings: [],
        categories: [],
      }
    }
    const categories: string[] = []
    for (const value of raw) if (!categories.includes(value)) categories.push(value)
    return { name, kind: 'categorical', numeric: [], strings: raw, categories }
  })
}

class Dataset {
  #columns: Column[]
  #selection: Selection

  constructor(columns: Column[], selection: Selection = { features: null, target: null }) {
    this.#columns = columns
    this.#selection = {
      features: selection.features ? [...selection.features] : null,
      target: selection.target,
    }
  }

  get columns(): string[] {
    return this.#columns.map(column => column.name)
  }

  #find(name: string): number {
    const index = this.#columns.findIndex(column => column.name === name)
    if (index < 0) invalid(`ColumnNotFound: ${name}`)
    return index
  }

  #withColumns(columns: Column[], selection = this.#selection): Dataset {
    return new Dataset(columns, selection)
  }

  select(...names: string[]): Dataset {
    if (names.length === 0) invalid('select() requires at least one column')
    for (const name of names) this.#find(name)
    return this.#withColumns(names.map(name => this.#columns[this.#find(name)]))
  }

  drop(...names: string[]): Dataset {
    for (const name of names) this.#find(name)
    return this.#withColumns(this.#columns.filter(column => !names.includes(column.name)))
  }

  encode(name: string, strategy: 'onehot' | 'label' = 'onehot'): Dataset {
    const index = this.#find(name)
    const source = this.#columns[index]
    if (strategy !== 'onehot' && strategy !== 'label') invalid(`unsupported encode strategy: ${strategy}`)
    if (source.kind === 'numeric') return this
    if (strategy === 'label') {
      const numeric = source.strings.map(value => source.categories.indexOf(value))
      const columns = [...this.#columns]
      columns[index] = { name: source.name, kind: 'numeric', numeric, strings: [], categories: [] }
      return this.#withColumns(columns)
    }
    const encoded: Column[] = source.categories.map(category => ({
      name: `${source.name}_${category}`,
      kind: 'numeric',
      numeric: source.strings.map(value => value === category ? 1 : 0),
      strings: [],
      categories: [],
    }))
    return this.#withColumns([
      ...this.#columns.slice(0, index),
      ...encoded,
      ...this.#columns.slice(index + 1),
    ])
  }

  #numericColumns(names: string[]): Column[] {
    if (names.length === 0) invalid('numeric transform requires at least one column')
    return names.map(name => {
      const column = this.#columns[this.#find(name)]
      if (column.kind !== 'numeric') invalid(`NotNumericColumn: ${name}`)
      return column
    })
  }

  #replaceColumns(replacements: Column[]): Dataset {
    const byName = new Map(replacements.map(column => [column.name, column]))
    return this.#withColumns(this.#columns.map(column => byName.get(column.name) ?? column))
  }

  normalize(...names: string[]): Dataset {
    return this.#replaceColumns(this.#numericColumns(names).map(source => {
      let minimum = Infinity
      let maximum = -Infinity
      for (const value of source.numeric) {
        if (value < minimum) minimum = value
        if (value > maximum) maximum = value
      }
      const range = maximum - minimum
      return {
        ...source,
        numeric: range === 0
          ? [...source.numeric]
          : source.numeric.map(value => (value - minimum) / range),
      }
    }))
  }

  standardize(...names: string[]): Dataset {
    return this.#replaceColumns(this.#numericColumns(names).map(source => {
      const mean = source.numeric.reduce((sum, value) => sum + value, 0) / source.numeric.length
      const variance = source.numeric.reduce((sum, value) => {
        const difference = value - mean
        return sum + difference * difference
      }, 0) / source.numeric.length
      const spread = Math.sqrt(variance)
      return {
        ...source,
        numeric: spread === 0
          ? [...source.numeric]
          : source.numeric.map(value => (value - mean) / spread),
      }
    }))
  }

  log(...names: string[]): Dataset {
    return this.#replaceColumns(this.#numericColumns(names).map(source => ({
      ...source,
      numeric: source.numeric.map(value => Math.log(value)),
    })))
  }

  clip(...args: any[]): Dataset {
    if (args.length < 3) invalid('clip() requires at least (column, min, max)')
    const maximum = args.pop()
    const minimum = args.pop()
    if (typeof minimum !== 'number' || typeof maximum !== 'number') {
      invalid('clip() requires numeric min/max')
    }
    const names = args as string[]
    if (!names.every(name => typeof name === 'string')) {
      invalid('clip() column arguments must be strings')
    }
    return this.#replaceColumns(this.#numericColumns(names).map(source => ({
      ...source,
      numeric: source.numeric.map(value => Math.max(minimum, Math.min(maximum, value))),
    })))
  }

  fillna(name: string, value: number): Dataset {
    if (typeof value !== 'number') invalid('fillna() requires a numeric value')
    const source = this.#numericColumns([name])[0]
    return this.#replaceColumns([{
      ...source,
      numeric: source.numeric.map(item => Number.isNaN(item) ? value : item),
    }])
  }

  dropna(...names: string[]): Dataset {
    const selected = names.length > 0
      ? this.#numericColumns(names)
      : this.#columns.filter(column => column.kind === 'numeric')
    const rowCount = this.#rowCount()
    const keep: number[] = []
    for (let row = 0; row < rowCount; row++) {
      if (!selected.some(column => Number.isNaN(column.numeric[row]))) keep.push(row)
    }
    return this.#withColumns(this.#columns.map(column => column.kind === 'numeric'
      ? { ...column, numeric: keep.map(row => column.numeric[row]) }
      : { ...column, strings: keep.map(row => column.strings[row]) }))
  }

  rename(oldName: string, newName: string): Dataset {
    const index = this.#find(oldName)
    if (this.#columns.some(column => column.name === newName)) invalid(`duplicate column: ${newName}`)
    const columns = [...this.#columns]
    columns[index] = { ...columns[index], name: newName }
    return this.#withColumns(columns, {
      features: this.#selection.features?.map(name => name === oldName ? newName : name) ?? null,
      target: this.#selection.target === oldName ? newName : this.#selection.target,
    })
  }

  features(...names: string[]): Dataset {
    for (const name of names) this.#find(name)
    return this.#withColumns([...this.#columns], { ...this.#selection, features: [...names] })
  }

  input(...names: string[]): Dataset {
    return this.features(...names)
  }

  target(name: string): Dataset {
    this.#find(name)
    return this.#withColumns([...this.#columns], { ...this.#selection, target: name })
  }

  shuffle(): Dataset {
    const rowCount = this.#rowCount()
    const order = Array.from({ length: rowCount }, (_, index) => index)
    for (let index = rowCount - 1; index > 0; index--) {
      const other = Math.floor(Math.random() * (index + 1))
      const value = order[index]
      order[index] = order[other]
      order[other] = value
    }
    return this.#take(order)
  }

  sample(count: number): Dataset {
    if (!Number.isFinite(count) || count < 0 || !Number.isInteger(count)) {
      invalid('sample() requires a non-negative integer')
    }
    return this.#take(Array.from({ length: Math.min(count, this.#rowCount()) }, (_, index) => index))
  }

  concat(other: Dataset): Dataset {
    if (!(other instanceof Dataset)) invalid('concat() requires a Dataset argument')
    if (this.columns.length !== other.columns.length ||
        this.columns.some((name, index) => name !== other.columns[index])) {
      invalid('concat(): column mismatch')
    }
    const columns = this.#columns.map((left, index) => {
      const right = other.#columns[index]
      if (left.kind !== right.kind) invalid('concat(): column type mismatch')
      if (left.kind === 'numeric') {
        return { ...left, numeric: [...left.numeric, ...right.numeric] }
      }
      const categories = [...left.categories]
      for (const category of right.categories) if (!categories.includes(category)) categories.push(category)
      return { ...left, categories, strings: [...left.strings, ...right.strings] }
    })
    return this.#withColumns(columns)
  }

  split(...ratios: number[]): Dataset[] {
    if (ratios.length === 0) invalid('split() requires at least one ratio')
    if (ratios.length > 8) invalid('split() supports at most 8 ratios')
    let total = 0
    for (const ratio of ratios) {
      if (!Number.isFinite(ratio) || ratio <= 0 || ratio >= 1) {
        invalid('split() each ratio must be between 0 and 1')
      }
      total += ratio
    }
    if (total > 1) invalid('split() ratios must sum to at most 1')
    const rowCount = this.#rowCount()
    const parts: Dataset[] = []
    let start = 0
    let cumulative = 0
    for (const ratio of ratios) {
      cumulative += ratio
      const end = Math.round(rowCount * cumulative)
      parts.push(this.#take(Array.from({ length: end - start }, (_, index) => start + index)))
      start = end
    }
    parts.push(this.#take(Array.from({ length: rowCount - start }, (_, index) => start + index)))
    return parts
  }

  #rowCount(): number {
    return this.#columns.length === 0 ? 0
      : this.#columns[0].kind === 'numeric'
        ? this.#columns[0].numeric.length
        : this.#columns[0].strings.length
  }

  #take(rows: number[]): Dataset {
    return this.#withColumns(this.#columns.map(column => column.kind === 'numeric'
      ? { ...column, numeric: rows.map(row => column.numeric[row]) }
      : { ...column, strings: rows.map(row => column.strings[row]) }))
  }

  #schema(columns: Column[]): Record<string, number[]> {
    const schema: Record<string, number[]> = {}
    let offset = 0
    for (const column of columns) {
      const width = column.kind === 'numeric' ? 1 : column.categories.length
      schema[column.name] = Array.from({ length: width }, (_, index) => offset + index)
      offset += width
    }
    return schema
  }

  #flat(columns: Column[]): { values: number[], width: number } {
    let width = 0
    for (const column of columns) width += column.kind === 'numeric' ? 1 : column.categories.length
    const values: number[] = []
    for (let row = 0; row < this.#rowCount(); row++) {
      for (const column of columns) {
        if (column.kind === 'numeric') {
          values.push(column.numeric[row])
        } else {
          const category = column.categories.indexOf(column.strings[row])
          for (let index = 0; index < column.categories.length; index++) {
            values.push(index === category ? 1 : 0)
          }
        }
      }
    }
    return { values, width }
  }

  toTensor(opts: TensorExportOpts): { data: Tensor; schema: Record<string, number[]> } {
    if (!opts?.session) invalid('toTensor() requires a Session')
    const dtype = opts.dtype ?? 'f32'
    if (dtype !== 'f32' && dtype !== 'f64') invalid("toTensor(): dtype must be 'f32' or 'f64'")
    if (this.#rowCount() === 0) invalid('toTensor(): no data')
    const flat = this.#flat(this.#columns)
    const rows: number[][] = []
    for (let start = 0; start < flat.values.length; start += flat.width) {
      rows.push(flat.values.slice(start, start + flat.width))
    }
    return {
      data: opts.session.tensor(rows, { dtype }),
      schema: this.#schema(this.#columns),
    }
  }

  toTensors(opts: TensorExportOpts): {
    X: Tensor
    y: Tensor
    schema: Record<string, number[]>
  } {
    if (!opts?.session) invalid('toTensors() requires a Session')
    const dtype = opts.dtype ?? 'f32'
    if (dtype !== 'f32' && dtype !== 'f64') invalid("toTensors(): dtype must be 'f32' or 'f64'")
    if (!this.#selection.features || !this.#selection.target) {
      invalid('toTensors() requires .features() and .target() to be set')
    }
    if (this.#rowCount() === 0) invalid('toTensors(): no data')
    const features = this.#selection.features.map(name => this.#columns[this.#find(name)])
    const target = this.#columns[this.#find(this.#selection.target)]
    const flatX = this.#flat(features)
    const flatY = this.#flat([target])
    const rowsX: number[][] = []
    for (let start = 0; start < flatX.values.length; start += flatX.width) {
      rowsX.push(flatX.values.slice(start, start + flatX.width))
    }
    const rowsY: any = flatY.width === 1 ? flatY.values : Array.from(
      { length: this.#rowCount() },
      (_, row) => flatY.values.slice(row * flatY.width, (row + 1) * flatY.width),
    )
    return {
      X: opts.session.tensor(rowsX, { dtype }),
      y: opts.session.tensor(rowsY, { dtype }),
      schema: this.#schema(features),
    }
  }

  loader(opts: DataLoaderOpts): DataLoader {
    return new DataLoader(this, opts)
  }

  tensorLoader(opts: DataLoaderOpts): DataLoader {
    return new DataLoader(this, opts)
  }
}

class DataLoader {
  private X: Tensor
  private y: Tensor | null
  private batchSize: number

  constructor(dataset: Dataset, opts: DataLoaderOpts) {
    try {
      const { X, y } = dataset.toTensors({ session: opts.session })
      this.X = X
      this.y = y
    } catch {
      const { data } = dataset.toTensor({ session: opts.session })
      this.X = data
      this.y = null
    }
    const batchSize = opts?.batchSize ?? 32
    if (!Number.isFinite(batchSize) || batchSize <= 0 || !Number.isInteger(batchSize)) {
      throw new TypeError('DataLoader.batchSize must be a positive integer')
    }
    this.batchSize = batchSize
  }

  get length(): number {
    return Math.ceil(this.X.shape[0] / this.batchSize)
  }

  [Symbol.iterator](): Iterator<Tensor[]> {
    const count = this.X.shape[0]
    const X = this.X
    const y = this.y
    const batchSize = this.batchSize
    let offset = 0
    return {
      next(): IteratorResult<Tensor[]> {
        if (offset >= count) return { done: true, value: undefined as any }
        const start = offset
        const end = Math.min(offset + batchSize, count)
        offset = end
        const xRanges = [{ start, stop: end }]
        for (let axis = 1; axis < X.shape.length; axis++) xRanges.push({ start: 0, stop: X.shape[axis] })
        if (y) {
          const yRanges = [{ start, stop: end }]
          for (let axis = 1; axis < y.shape.length; axis++) yRanges.push({ start: 0, stop: y.shape[axis] })
          return { done: false, value: [slice(X, xRanges), slice(y, yRanges)] }
        }
        return { done: false, value: [slice(X, xRanges)] }
      },
    }
  }
}

function read(source: string): Dataset {
  return new Dataset(parseColumns(readFileSync(source)))
}

const tabular = {
  read,
  DataLoader,
}

export { read, DataLoader, tabular }

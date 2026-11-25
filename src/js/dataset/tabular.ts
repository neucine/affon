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
  numeric: Float64Array
  strings: string[]
  categories: string[]
  codes: Uint32Array
}

interface Selection {
  features: string[] | null
  target: string | null
}

function invalid(message: string): never {
  throw new AffonError('invalid_arg', message)
}

function numericCell(value: string): number | null {
  if (value.length === 0) return NaN
  const parsed = Number(value)
  if (Number.isFinite(parsed)) return parsed === 0 && value.trim().length === 0 ? NaN : parsed
  const trimmed = value.trim()
  if (trimmed.length === 0) return NaN
  return trimmed === 'NaN' || trimmed === 'Infinity' || trimmed === '-Infinity' ? parsed : null
}

function scanCsv(source: string, visit: (row: string[]) => void): void {
  const pendingBlankRows: string[][] = []
  const publish = (row: string[]) => {
    if (row.every(value => value.length === 0)) {
      pendingBlankRows.push(row)
    } else {
      for (const pending of pendingBlankRows) visit(pending)
      pendingBlankRows.length = 0
      visit(row)
    }
  }

  if (!source.includes('"') && !source.includes('\r')) {
    let row: string[] = []
    let start = 0
    for (let index = 0; index < source.length; index++) {
      const character = source[index]
      if (character === ',') {
        row.push(source.slice(start, index))
        start = index + 1
      } else if (character === '\n') {
        row.push(source.slice(start, index))
        publish(row)
        row = []
        start = index + 1
      }
    }
    if (start < source.length || row.length > 0) {
      row.push(source.slice(start))
      publish(row)
    }
    return
  }

  let row: string[] = []
  let cell = ''
  let quoted = false
  const flush = () => {
    row.push(cell)
    publish(row)
    row = []
    cell = ''
  }
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
      flush()
    } else if (character !== '\r') {
      cell += character
    }
  }
  if (cell.length > 0 || row.length > 0) flush()
}

class NumericBuilder {
  #chunks: Float64Array[] = []
  #current = new Float64Array(16_384)
  #offset = 0
  length = 0

  push(value: number): void {
    if (this.#offset === this.#current.length) {
      this.#chunks.push(this.#current)
      this.#current = new Float64Array(this.#current.length)
      this.#offset = 0
    }
    this.#current[this.#offset++] = value
    this.length++
  }

  finish(): Float64Array {
    const values = new Float64Array(this.length)
    let offset = 0
    for (const chunk of this.#chunks) {
      values.set(chunk, offset)
      offset += chunk.length
    }
    values.set(this.#current.subarray(0, this.#offset), offset)
    return values
  }
}

function numericColumns(names: string[], builders: NumericBuilder[]): Column[] {
  return names.map((name, index) => ({
    name,
    kind: 'numeric',
    numeric: builders[index].finish(),
    strings: [],
    categories: [],
    codes: new Uint32Array(0),
  }))
}

function parseColumns(source: string): Column[] {
  let names: string[] | null = null
  let rowCount = 0
  let firstRowNumeric = false
  let numericColumnFlags: boolean[] = []
  let builders: NumericBuilder[] | null = null

  scanCsv(source, row => {
    if (names === null) {
      if (row.length === 0) return
      names = row
      numericColumnFlags = row.map(() => true)
      return
    }
    if (rowCount === 0) {
      const cells = names.map((_, column) => row[column] ?? '')
      const parsedCells = cells.map(numericCell)
      firstRowNumeric = parsedCells.every(value => value !== null)
      if (firstRowNumeric) {
        builders = names.map(() => new NumericBuilder())
        for (let column = 0; column < parsedCells.length; column++) {
          builders[column].push(parsedCells[column]!)
        }
      } else {
        for (let column = 0; column < parsedCells.length; column++) {
          if (parsedCells[column] === null) numericColumnFlags[column] = false
        }
      }
      rowCount++
      return
    }
    if (builders) {
      for (let column = 0; column < names.length; column++) {
        const parsed = numericCell(row[column] ?? '')
        if (parsed === null) invalid('read(): InvalidNumericValue')
        builders[column].push(parsed)
      }
      rowCount++
      return
    }
    for (let column = 0; column < names.length; column++) {
      if (numericColumnFlags[column] && numericCell(row[column] ?? '') === null) {
        numericColumnFlags[column] = false
      }
    }
    rowCount++
  })

  if (names === null || names.length === 0) invalid('read(): EmptyCSV')
  const headers: string[] = names
  if (builders) return numericColumns(headers, builders)

  const kinds = headers.map((_, column) => firstRowNumeric || numericColumnFlags[column] ? 'numeric' : 'categorical')
  const columns: Column[] = headers.map((name, column) => kinds[column] === 'numeric'
    ? {
        name,
        kind: 'numeric',
        numeric: new Float64Array(rowCount),
        strings: [],
        categories: [],
        codes: new Uint32Array(0),
      }
    : {
        name,
        kind: 'categorical',
        numeric: new Float64Array(0),
        strings: new Array(rowCount),
        categories: [],
        codes: new Uint32Array(rowCount),
      })
  const categoryMaps = headers.map(() => new Map<string, number>())
  let dataRow = 0
  let sawHeader = false
  scanCsv(source, row => {
    if (!sawHeader) {
      sawHeader = true
      return
    }
    for (let index = 0; index < headers.length; index++) {
      const value = row[index] ?? ''
      const column = columns[index]
      if (column.kind === 'numeric') {
        const parsed = numericCell(value)
        if (parsed === null) invalid('read(): InvalidNumericValue')
        column.numeric[dataRow] = parsed
      } else {
        let code = categoryMaps[index].get(value)
        if (code === undefined) {
          code = column.categories.length
          categoryMaps[index].set(value, code)
          column.categories.push(value)
        }
        column.strings[dataRow] = value
        column.codes[dataRow] = code
      }
    }
    dataRow++
  })
  return columns
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
      const columns = [...this.#columns]
      columns[index] = {
        name: source.name,
        kind: 'numeric',
        numeric: Float64Array.from(source.codes),
        strings: [],
        categories: [],
        codes: new Uint32Array(0),
      }
      return this.#withColumns(columns)
    }
    const encoded: Column[] = source.categories.map((category, categoryIndex) => {
      const numeric = new Float64Array(source.codes.length)
      for (let row = 0; row < source.codes.length; row++) {
        if (source.codes[row] === categoryIndex) numeric[row] = 1
      }
      return {
        name: `${source.name}_${category}`,
        kind: 'numeric',
        numeric,
        strings: [],
        categories: [],
        codes: new Uint32Array(0),
      }
    })
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
          ? source.numeric.slice()
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
          ? source.numeric.slice()
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
      ? {
          ...column,
          numeric: Float64Array.from(keep, row => column.numeric[row]),
        }
      : {
          ...column,
          strings: keep.map(row => column.strings[row]),
          codes: Uint32Array.from(keep, row => column.codes[row]),
        }))
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
        const numeric = new Float64Array(left.numeric.length + right.numeric.length)
        numeric.set(left.numeric)
        numeric.set(right.numeric, left.numeric.length)
        return { ...left, numeric }
      }
      const categories = [...left.categories]
      const categoryIndexes = new Map(left.categories.map((category, categoryIndex) => [category, categoryIndex]))
      const rightIndexes = new Map<number, number>()
      for (let categoryIndex = 0; categoryIndex < right.categories.length; categoryIndex++) {
        const category = right.categories[categoryIndex]
        let target = categoryIndexes.get(category)
        if (target === undefined) {
          target = categories.length
          categoryIndexes.set(category, target)
          categories.push(category)
        }
        rightIndexes.set(categoryIndex, target)
      }
      const codes = new Uint32Array(left.codes.length + right.codes.length)
      codes.set(left.codes)
      for (let row = 0; row < right.codes.length; row++) {
        codes[left.codes.length + row] = rightIndexes.get(right.codes[row])!
      }
      return {
        ...left,
        categories,
        strings: [...left.strings, ...right.strings],
        codes,
      }
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
      ? { ...column, numeric: Float64Array.from(rows, row => column.numeric[row]) }
      : {
          ...column,
          strings: rows.map(row => column.strings[row]),
          codes: Uint32Array.from(rows, row => column.codes[row]),
        }))
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

  #flat(columns: Column[], dtype: 'f32' | 'f64'): {
    values: Float32Array | Float64Array
    width: number
  } {
    let width = 0
    for (const column of columns) width += column.kind === 'numeric' ? 1 : column.categories.length
    const values = dtype === 'f32'
      ? new Float32Array(this.#rowCount() * width)
      : new Float64Array(this.#rowCount() * width)
    let offset = 0
    for (let row = 0; row < this.#rowCount(); row++) {
      for (const column of columns) {
        if (column.kind === 'numeric') {
          values[offset++] = column.numeric[row]
        } else {
          const category = column.codes[row]
          for (let index = 0; index < column.categories.length; index++) {
            values[offset++] = index === category ? 1 : 0
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
    const flat = this.#flat(this.#columns, dtype)
    return {
      data: opts.session.fromBytes(flat.values.buffer as ArrayBuffer, {
        dtype,
        shape: [this.#rowCount(), flat.width],
      }),
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
    const flatX = this.#flat(features, dtype)
    const flatY = this.#flat([target], dtype)
    return {
      X: opts.session.fromBytes(flatX.values.buffer as ArrayBuffer, {
        dtype,
        shape: [this.#rowCount(), flatX.width],
      }),
      y: opts.session.fromBytes(flatY.values.buffer as ArrayBuffer, {
        dtype,
        shape: flatY.width === 1 ? [this.#rowCount()] : [this.#rowCount(), flatY.width],
      }),
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

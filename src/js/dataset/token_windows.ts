export type TokenWindow = number[]

export interface PackTokenWindowsOpts {
  seqLen: number
  stride?: number
  joinWithTokenId?: number
}

export function packTokenWindows(
  rows: readonly (readonly number[])[],
  opts: PackTokenWindowsOpts,
): TokenWindow[] {
  if (!Number.isInteger(opts.seqLen) || opts.seqLen <= 0) {
    throw new TypeError('packTokenWindows seqLen must be a positive integer')
  }
  const stride = opts.stride ?? opts.seqLen
  if (!Number.isInteger(stride) || stride <= 0) {
    throw new TypeError('packTokenWindows stride must be a positive integer')
  }

  const flat: number[] = []
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]
    for (let j = 0; j < row.length; j++) flat.push(row[j])
    if (opts.joinWithTokenId !== undefined && i < rows.length - 1) {
      flat.push(opts.joinWithTokenId)
    }
  }

  const windowSize = opts.seqLen + 1
  const windows: TokenWindow[] = []
  for (let start = 0; start + windowSize <= flat.length; start += stride) {
    windows.push(flat.slice(start, start + windowSize))
  }
  return windows
}

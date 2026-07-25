import { readFileSync } from 'std:fs'

interface TextReadOpts {
  trim?: boolean
  skipEmpty?: boolean
}

interface TextSourceOpts extends TextReadOpts {
  mode?: 'line' | 'paragraph'
}

interface DelimitedTextReadOpts extends TextReadOpts {
  delimiter?: string
}

function readLines(source: string, opts?: TextReadOpts): string[] {
  const trim = opts?.trim ?? false
  const skipEmpty = opts?.skipEmpty ?? true
  const raw = readFileSync(source)
  return textFromString(raw, { mode: 'line', trim, skipEmpty })
}

function readParagraphs(source: string, opts?: TextReadOpts): string[] {
  const trim = opts?.trim ?? false
  const skipEmpty = opts?.skipEmpty ?? true
  const raw = readFileSync(source)
  return textFromString(raw, { mode: 'paragraph', trim, skipEmpty })
}

function readDelimitedRows(source: string, opts?: DelimitedTextReadOpts): string[][] {
  const delimiter = opts?.delimiter ?? '\t'
  const raw = readFileSync(source)
  return splitLines(raw)
    .map((line) => normalizeLine(line, opts))
    .filter((line) => shouldKeepLine(line, opts))
    .map((line) => line.split(delimiter).map((part) => opts?.trim ? part.trim() : part))
}

function splitLines(raw: string): string[] {
  return raw.split(/\r?\n/)
}

function normalizeLine(line: string, opts?: TextReadOpts): string {
  return opts?.trim ? line.trim() : line
}

function shouldKeepLine(line: string, opts?: TextReadOpts): boolean {
  return !(opts?.skipEmpty ?? true) || line.length > 0
}

function normalizeTextChunks(chunks: string[], trim: boolean, skipEmpty: boolean): string[] {
  return chunks
    .map((line) => trim ? line.trim() : line)
    .filter((line) => !skipEmpty || line.length > 0)
}

function textFromString(raw: string, opts?: TextSourceOpts): string[] {
  const mode = opts?.mode ?? 'line'
  const trim = opts?.trim ?? false
  const skipEmpty = opts?.skipEmpty ?? true
  const chunks = mode === 'paragraph'
    ? raw.split(/\n\s*\n+/)
    : raw.split(/\r?\n/)
  return normalizeTextChunks(chunks, trim, skipEmpty)
}

export { readDelimitedRows, readLines, readParagraphs, textFromString }
export type { DelimitedTextReadOpts, TextReadOpts, TextSourceOpts }

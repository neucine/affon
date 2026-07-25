import { run } from 'std:process'

export interface PythonEvalOptions {
  cwd?: string
  check?: boolean
  python?: string
  maxOutputBytes?: number
}

export interface PythonNumpyResult<T = unknown> {
  value: T
  shape: number[]
  dtype: string
}

export interface PythonTorchResult<T = unknown> {
  value: T
  shape: number[]
  dtype: string
}

function dedent(code: string): string {
  const normalized = code.replace(/\r\n/g, '\n')
  const lines = normalized.split('\n')

  while (lines.length > 0 && lines[0].trim() === '') lines.shift()
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop()

  let indent: number | null = null
  for (const line of lines) {
    if (line.trim() === '') continue
    const width = line.match(/^ */)?.[0].length ?? 0
    indent = indent == null ? width : Math.min(indent, width)
  }

  if (indent == null || indent === 0) {
    return lines.join('\n')
  }

  return lines.map((line) => (line.trim() === '' ? '' : line.slice(indent!))).join('\n')
}

function pythonRun(code: string, options: PythonEvalOptions = {}) {
  const source = dedent(code)
  if (options.python) {
    return run({
      cmd: options.python,
      args: ['-c', source],
      cwd: options.cwd,
      check: options.check,
      maxOutputBytes: options.maxOutputBytes,
    })
  }

  return run({
    cmd: 'sh',
    args: [
      '-lc',
      'if [ -n "$AFFON_TEST_PYTHON" ]; then exec "$AFFON_TEST_PYTHON" -c "$1"; elif [ -x "$HOME/.venvs/affon/bin/python" ]; then exec "$HOME/.venvs/affon/bin/python" -c "$1"; else exec python3 -c "$1"; fi',
      'sh',
      source,
    ],
    cwd: options.cwd,
    check: options.check,
    maxOutputBytes: options.maxOutputBytes,
  })
}

interface PythonHelper {
  (code: string, options?: PythonEvalOptions): ReturnType<typeof run>
  json<T = unknown>(code: string, options?: PythonEvalOptions): Promise<T>
  numpy<T = unknown>(code: string, options?: PythonEvalOptions): Promise<PythonNumpyResult<T>>
  torch: {
    available(options?: PythonEvalOptions): Promise<boolean>
    tensor<T = unknown>(code: string, options?: PythonEvalOptions): Promise<PythonTorchResult<T>>
    json<T = unknown>(code: string, options?: PythonEvalOptions): Promise<T>
  }
}

let torchAvailableCache: boolean | null = null

function buildTorchSource(code: string): string {
  return [
    'import json',
    'import torch',
    '',
    'def _serialize(value):',
    '    if isinstance(value, torch.Tensor):',
    '        data = value.detach().cpu()',
    '        return {',
    '            "value": data.tolist(),',
    '            "shape": list(data.shape),',
    '            "dtype": str(data.dtype),',
    '        }',
    '    raise TypeError(f"emit() expected torch.Tensor, got {type(value).__name__}")',
    '',
    'def emit(value):',
    '    print(json.dumps(_serialize(value)))',
    '',
    'def emit_many(**values):',
    '    print(json.dumps({key: _serialize(value) for key, value in values.items()}))',
    '',
    dedent(code),
  ].join('\n')
}

export const python: PythonHelper = Object.assign(pythonRun, {
  json<T = unknown>(code: string, options: PythonEvalOptions = {}) {
    return pythonRun(code, options).json().then((value) => value as T)
  },

  numpy<T = unknown>(code: string, options: PythonEvalOptions = {}) {
    const source = [
      'import json',
      'import numpy as np',
      '',
      'def emit(value):',
      '    if isinstance(value, np.ndarray):',
      '        payload = {',
      '            "value": value.tolist(),',
      '            "shape": list(value.shape),',
      '            "dtype": str(value.dtype),',
      '        }',
      '    elif isinstance(value, np.generic):',
      '        payload = {',
      '            "value": value.item(),',
      '            "shape": [],',
      '            "dtype": str(value.dtype),',
      '        }',
      '    else:',
      '        raise TypeError(f"emit() expected numpy.ndarray or numpy scalar, got {type(value).__name__}")',
      '    print(json.dumps(payload))',
      '',
      dedent(code),
    ].join('\n')

    return pythonRun(source, options).json().then((value) => value as PythonNumpyResult<T>)
  },

  torch: {
    async available(options: PythonEvalOptions = {}) {
      if (torchAvailableCache != null) return torchAvailableCache
      const result = await pythonRun(
        `
import importlib.util
print("1" if importlib.util.find_spec("torch") is not None else "0")
`,
        { ...options, check: false },
      ).text()
      torchAvailableCache = result.trim() === '1'
      return torchAvailableCache
    },

    tensor<T = unknown>(code: string, options: PythonEvalOptions = {}) {
      return pythonRun(buildTorchSource(code), options).json().then((value) => value as PythonTorchResult<T>)
    },

    json<T = unknown>(code: string, options: PythonEvalOptions = {}) {
      return pythonRun(buildTorchSource(code), options).json().then((value) => value as T)
    },
  },
})

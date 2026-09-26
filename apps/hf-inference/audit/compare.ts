export function compare_values(actual: number[], expected: number[], atol = 1e-4, rtol = 1e-4) {
  if (actual.length !== expected.length || actual.length === 0) {
    throw new Error(`Reference length mismatch: ${actual.length} vs ${expected.length}`)
  }
  let max_absolute_error = 0
  let mismatches = 0
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i])
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) {
      mismatches++
      max_absolute_error = Infinity
    } else {
      max_absolute_error = Math.max(max_absolute_error, error)
      if (error > atol + rtol * Math.abs(expected[i])) mismatches++
    }
  }
  return { passed: mismatches === 0, elements: actual.length, mismatches,
    max_absolute_error: Number.isFinite(max_absolute_error) ? max_absolute_error : null, atol, rtol }
}

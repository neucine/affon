import type { ModelTensor } from '../../../models/src/shared/parameters.ts'

// Some HF GPT-2 checkpoints persist the lower-triangular attention allow-mask.
// Validate all values before replacing it with the runtime-generated mask.
export function validate_causal_buffer(value: ModelTensor, positions: number): void {
  if (value.dtype !== 'f32' || JSON.stringify(value.shape) !== JSON.stringify([1, 1, positions, positions])) {
    throw new Error('Invalid GPT-2 causal buffer shape/dtype')
  }
  const rows = (value.to_array() as number[][][][])[0][0]
  for (let y = 0; y < positions; y++) for (let x = 0; x < positions; x++) {
    if (rows[y][x] !== (x <= y ? 1 : 0)) throw new Error('Invalid GPT-2 causal buffer values')
  }
}

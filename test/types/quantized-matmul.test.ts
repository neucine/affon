import { quantized_matmul } from 'affon:ops'
import type { Tensor, FormalTensor } from 'affon:compute'

declare const evaluated: Tensor
declare const formal: FormalTensor
const output: Tensor = quantized_matmul(evaluated, evaluated, evaluated, evaluated, { bits: 8, group_size: 64 })
const authored: FormalTensor = quantized_matmul(formal, formal, formal, formal, { bits: 4, group_size: 64 })
// @ts-expect-error Explicit packing metadata is required.
quantized_matmul(evaluated, evaluated, evaluated, evaluated)
// @ts-expect-error Three-bit packing is not part of the supported scheme.
quantized_matmul(evaluated, evaluated, evaluated, evaluated, { bits: 3, group_size: 64 })
void output; void authored

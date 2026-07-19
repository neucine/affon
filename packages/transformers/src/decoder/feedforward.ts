import nn from 'affon:nn'
import { add, axes, gelu, matmul, module as computeModule, reshape } from 'affon:compute'
import type { Tensor } from 'affon:compute'


export interface FeedForwardOptions {
  hiddenDim?: number
}

export type FeedForwardModule = nn.Module<[Tensor<number[], 'f32'>], Tensor<number[], 'f32'>> & {
  up: nn.LinearLayer<number, number, 'f32'>
  down: nn.LinearLayer<number, number, 'f32'>
}

function applyLinearStack(
  up: nn.LinearLayer<number, number, 'f32'>,
  down: nn.LinearLayer<number, number, 'f32'>,
  x: Tensor<number[], 'f32'>,
): Tensor<number[], 'f32'> {
  let upOut: Tensor<number[], 'f32'> | null = add(
    matmul(x, up.weight as Tensor<[number, number], 'f32'>, { hint: 'projection', source: 'higher_level_module' }),
    up.bias as Tensor<[1, number], 'f32'>,
  ) as Tensor<number[], 'f32'>
  nn.diagnostics.assert('finite', { path: 'FeedForward.up', value: upOut })
  let activated: Tensor<number[], 'f32'> | null = gelu(upOut)
  upOut = null
  nn.diagnostics.assert('finite', { path: 'FeedForward.gelu', value: activated })
  const out = add(
    matmul(activated, down.weight as Tensor<[number, number], 'f32'>, { hint: 'projection', source: 'higher_level_module' }),
    down.bias as Tensor<[1, number], 'f32'>,
  ) as Tensor<number[], 'f32'>
  activated = null
  nn.diagnostics.assert('finite', { path: 'FeedForward.down', value: out })
  return out
}

export function FeedForward(
  dModel: number,
  opts?: FeedForwardOptions,
): FeedForwardModule {
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new AffonError('invalid_arg', 'FeedForward dModel must be a positive integer')
  }
  const hiddenDim = opts?.hiddenDim ?? dModel * 4
  if (!Number.isInteger(hiddenDim) || hiddenDim <= 0) {
    throw new AffonError('invalid_arg', 'FeedForward hiddenDim must be a positive integer')
  }

  const up = nn.Linear<number, number, 'f32'>(dModel, hiddenDim, { dtype: 'f32' })
  const down = nn.Linear<number, number, 'f32'>(hiddenDim, dModel, { dtype: 'f32' })

	  return computeModule({
	    up,
	    down,
	  } as any, function (_state, x: Tensor<number[], 'f32'>): Tensor<number[], 'f32'> {
      if (x.ndim === 2) {
        if (x.shape[1] !== dModel) {
          throw new AffonError('shape_mismatch', 'FeedForward expects input shaped [tokens, d_model]')
        }
	        return applyLinearStack(up, down, x as Tensor<number[], 'f32'>)
      }
      if (x.ndim === 3) {
        if (x.shape[2] !== dModel) {
          throw new AffonError('shape_mismatch', 'FeedForward expects input shaped [batch, tokens, d_model]')
        }
        const rawOut = applyLinearStack(up, down, x as Tensor<number[], 'f32'>)
        const out = reshape(rawOut, rawOut.shape as number[], { axes: [axes.batch, axes.token, axes.feature] }) as Tensor<number[], 'f32'>
        nn.diagnostics.assert('finite', { path: 'FeedForward.output', value: out })
        return out
      }
      throw new AffonError('invalid_shape', 'FeedForward expects input shaped [tokens, d_model] or [batch, tokens, d_model]')
  }) as FeedForwardModule
}

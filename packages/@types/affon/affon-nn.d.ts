/** Parameterized neural-network factories expanded while authoring Programs. */
declare module "affon:nn" {
  import type { Callable, FormalTensor, ProgramDType } from "affon:compute";

  /** A scalar objective with standardized input and target bindings. */
  export interface LossCallable<Input extends FormalTensor = FormalTensor, Target extends FormalTensor = FormalTensor> extends Callable<{ input: Input; target: Target }, FormalTensor> {}

  export type LinearOptions = Readonly<{
    out_features: number;
    bias?: boolean;
  }>;

  export type Conv1DOptions = Readonly<{
    out_channels: number;
    kernel_size: number;
    stride?: number;
    padding?: number;
    dilation?: number;
    groups?: number;
    bias?: boolean;
  }>;

  export type ConvTranspose1DOptions = Conv1DOptions & Readonly<{ output_padding?: number }>;

  export type EmbeddingOptions = Readonly<{
    num_embeddings: number;
    embedding_dim: number;
    dtype?: ProgramDType;
  }>;

  export type LayerNormOptions = Readonly<{
    normalized_shape?: number;
    epsilon?: number;
    affine?: boolean;
  }>;

  export type LossOptions = Readonly<{
    /** Name exposed for the target argument when optimize() materializes this loss. */
    target?: string;
    /** Loss reduction. Mean is currently the only supported reduction. */
    reduction?: "mean";
  }>;

  /** Configure a reusable affine projection factory. */
  export function linear(options: LinearOptions): Callable<{ x: FormalTensor }, FormalTensor>;
  /** Convolve NCL input with [out_channels, in_channels/groups, kernel_size] weights. */
  export function conv1d(options: Conv1DOptions): Callable<{ x: FormalTensor }, FormalTensor>;
  /** Transposed convolution with [in_channels, out_channels/groups, kernel_size] weights. */
  export function conv_transpose1d(options: ConvTranspose1DOptions): Callable<{ x: FormalTensor }, FormalTensor>;
  /** Configure a reusable learned embedding-table factory. */
  export function embedding(options: EmbeddingOptions): Callable<{ indices: FormalTensor }, FormalTensor>;
  /** Configure reusable normalization over the final tensor dimension. */
  export function layer_norm(options?: LayerNormOptions): Callable<{ x: FormalTensor }, FormalTensor>;
  /** Configure indexed multiclass cross entropy over the final input dimension. */
  export function cross_entropy(options?: LossOptions): LossCallable;
  /** Configure mean squared error for same-shaped input and target tensors. */
  export function mean_squared_error(options?: LossOptions): LossCallable;
  /** Configure binary cross entropy for probabilities and same-shaped targets. */
  export function binary_cross_entropy(options?: LossOptions): LossCallable;
  /** Configure numerically stable binary cross entropy for unnormalized logits. */
  export function binary_cross_entropy_with_logits(options?: LossOptions): LossCallable;
}

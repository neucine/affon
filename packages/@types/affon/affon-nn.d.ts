/** Parameterized neural-network factories expanded while authoring Programs. */
declare module "affon:nn" {
  import type { FormalTensor, ProgramDType } from "affon:compute";

  /** An ordinary callable architecture fragment with named tensor bindings. */
  export interface Layer<Bindings extends Record<string, FormalTensor>, Out extends FormalTensor = FormalTensor> {
    (bindings: Bindings, name?: string): Out;
  }

  export type LinearOptions = Readonly<{
    out_features: number;
    bias?: boolean;
  }>;

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

  /** Configure a reusable affine projection factory. */
  export function linear(options: LinearOptions): Layer<{ x: FormalTensor }>;
  /** Configure a reusable learned embedding-table factory. */
  export function embedding(options: EmbeddingOptions): Layer<{ indices: FormalTensor }>;
  /** Configure reusable normalization over the final tensor dimension. */
  export function layer_norm(options?: LayerNormOptions): Layer<{ x: FormalTensor }>;
}

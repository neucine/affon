declare module "affon:nn/legacy" {
  import type {
    Device,
    DType,
    AxisName,
    Module as ComputeModule,
    Parameter,
    Shape,
    Tensor,
  } from "affon:compute/legacy"

  type NnTensor<S extends Shape = Shape, D extends DType = DType> = Tensor<S, D>
  type NnParameter<S extends Shape = Shape, D extends DType = DType> = Parameter & NnTensor<S, D>
  type ComputeState = any
  type ModuleMode = "train" | "eval"

  namespace nn {
    /**
     * Neural-network oriented model and layer authoring.
     *
     * `nn` is the model-facing surface:
     * - use built-in layers like `Linear`, `Embedding`, `LayerNorm`, and recurrent modules
     * - use `mode("train" | "eval")` on modeful modules when semantics differ by mode
     *
     * Custom model blocks should be authored with `compute.module(...)`.
     * `nn` supplies the layer set, losses, serialization helpers, and modeful modules.
     */
    type TypedSequentialModule<
      InputShape extends Shape,
      OutputShape extends Shape,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > =
      SequentialModule<InputShape, OutputShape, D>
      & SequentialModule<number[], number[], D>
    type ParamEntry<D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">> =
      readonly [string, NnParameter<number[], D>]
    type State = ComputeState
    type ReplaceLastDim<
      S extends readonly number[],
      InFeatures extends number,
      OutFeatures extends number,
    > = S extends readonly [...infer Prefix extends number[], InFeatures]
      ? [...Prefix, OutFeatures]
      : number[]

    interface ParameterCollection<
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends ReadonlyArray<NnParameter<number[], D>> {
      named(): ReadonlyArray<ParamEntry<D>>
    }

    interface SinusoidalEncodingOptions {
      dtype?: Extract<DType, "f32" | "f64">
      device?: Device
    }

    interface EmbeddingOptions<D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">> {
      dtype?: D
      axes?: readonly AxisName[]
    }

    type DiagnosticMode = "off" | "error" | "warn"
    type DiagnosticKind = "finite"

    interface DiagnosticConfig {
      mode?: DiagnosticMode
      include?: string[]
      exclude?: string[]
    }

    interface DiagnosticResolvedConfig {
      mode: DiagnosticMode
      include: string[]
      exclude: string[]
    }

    interface DiagnosticAssertOptions<S extends Shape = Shape, D extends DType = DType> {
      path: string
      value?: NnTensor<S, D>
    }

    namespace diagnostics {
      /**
       * @summary Configure module diagnostics.
       * @category diagnostics
       * @semantics Controls whether matching diagnostic assertion sites are
       * disabled, reported as warnings, or raised as errors.
       */
      function configure(config: DiagnosticConfig): void
      /**
       * @summary Return the current module diagnostics configuration.
       * @category diagnostics
       */
      function get_config(): DiagnosticResolvedConfig
      /**
       * @summary Run a module diagnostic assertion.
       * @category diagnostics
       * @semantics The filter key is `<kind>:<path>`, for example
       * `finite:decoder.blocks.*`.
       */
      function assert<S extends Shape = Shape, D extends DType = DType>(kind: DiagnosticKind, opts: DiagnosticAssertOptions<S, D>): void
    }

    namespace init {
      function xavier_uniform<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
      function xavier_normal<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
      function kaiming_uniform<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
      function kaiming_normal<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
      function zeros<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
      function ones<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(tensor: NnTensor<S, D>): NnTensor<S, D>
    }

    interface Module<
      Args extends readonly NnTensor<any, any>[] = readonly NnTensor<any, any>[],
      Out extends NnTensor<any, any> = Tensor,
    > extends ComputeModule<Args, Out> {
      readonly parameters: ParameterCollection
      mode(): ModuleMode
      mode(mode: ModuleMode): this
    }

    type ModuleListEntry = Module<any, any>

    interface LinearLayer<
      InFeatures extends number = number,
      OutFeatures extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, InFeatures], D>], NnTensor<[number, OutFeatures], D>> {
      readonly parameters: ParameterCollection<D>
      weight: NnTensor<[InFeatures, OutFeatures], D>
      bias: NnTensor<[1, OutFeatures], D>
      <S extends readonly [...number[], InFeatures]>(x: NnTensor<S, D>): NnTensor<ReplaceLastDim<S, InFeatures, OutFeatures>, D>
    }

    interface EmbeddingLayer<
      NumEmbeddings extends number = number,
      EmbeddingDim extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<readonly [NnTensor<[number], D>] | readonly [NnTensor<[number, number], D>], NnTensor<number[], D>> {
      readonly parameters: ParameterCollection<D>
      weight: NnTensor<[NumEmbeddings, EmbeddingDim], D>
      (token_ids: NnTensor<[number], D>): NnTensor<[number, EmbeddingDim], D>
      (token_ids: NnTensor<[number, number], D>): NnTensor<[number, number, EmbeddingDim], D>
    }

    interface SequentialModule<
      InputShape extends Shape = number[],
      OutputShape extends Shape = number[],
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<InputShape, D>], NnTensor<OutputShape, D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<InputShape, D>, ...args: unknown[]): NnTensor<OutputShape, D>
    }

    interface SimpleRNNLayer<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, InputSize], D>], NnTensor<[number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, InputSize], D>, ...args: unknown[]): NnTensor<[number, HiddenSize], D>
      weight_ih: NnTensor<[InputSize, HiddenSize], D>
      weight_hh: NnTensor<[HiddenSize, HiddenSize], D>
      bias: NnTensor<[1, HiddenSize], D> | null
      hidden_state(): NnTensor<[1, HiddenSize], D> | null
    }

    interface SimpleRNNModule<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, InputSize], D>], NnTensor<[number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, InputSize], D>, ...args: unknown[]): NnTensor<[number, HiddenSize], D>
      layers: ModuleList
      hidden_state(): NnTensor<[number, HiddenSize], D> | null
    }

    interface RNNModule<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, number, InputSize], D>], NnTensor<[number, number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, number, InputSize], D>, ...args: unknown[]): NnTensor<[number, number, HiddenSize], D>
      layers: ModuleList
      hidden_state(): NnTensor<[number, number, HiddenSize], D> | null
    }

    interface RNNBatchFirstModule<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, number, InputSize], D>], NnTensor<[number, number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, number, InputSize], D>, ...args: unknown[]): NnTensor<[number, number, HiddenSize], D>
      layers: ModuleList
      hidden_state(): NnTensor<[number, number, HiddenSize], D> | null
    }

    interface LSTMModule<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, number, InputSize], D>], NnTensor<[number, number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, number, InputSize], D>, ...args: unknown[]): NnTensor<[number, number, HiddenSize], D>
      layers: ModuleList
      hidden_state(): NnTensor<[number, number, HiddenSize], D> | null
      cell_state(): NnTensor<[number, number, HiddenSize], D> | null
    }

    interface LSTMBatchFirstModule<
      InputSize extends number = number,
      HiddenSize extends number = number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    > extends Module<[NnTensor<[number, number, InputSize], D>], NnTensor<[number, number, HiddenSize], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, number, InputSize], D>, ...args: unknown[]): NnTensor<[number, number, HiddenSize], D>
      layers: ModuleList
      hidden_state(): NnTensor<[number, number, HiddenSize], D> | null
      cell_state(): NnTensor<[number, number, HiddenSize], D> | null
    }

    interface ModuleList {
      readonly length: number
      readonly [index: number]: Module<any, any>
      readonly parameters: ParameterCollection<Extract<DType, "f32" | "f64">>
      mode(): ModuleMode
      mode(mode: ModuleMode): this
      train(): void
      eval(): void
    }

    interface DropoutLayer<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">> extends Module<[NnTensor<S, D>], NnTensor<S, D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<S, D>, ...args: unknown[]): NnTensor<S, D>
    }

    interface BatchNormLayer<D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">> extends Module<[NnTensor<[number, number], D>], NnTensor<[number, number], D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<[number, number], D>, ...args: unknown[]): NnTensor<[number, number], D>
      gamma: NnTensor<[1, number], D>
      beta: NnTensor<[1, number], D>
    }

    interface LayerNormLayer<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">> extends Module<[NnTensor<S, D>], NnTensor<S, D>> {
      readonly parameters: ParameterCollection<D>
      (x: NnTensor<S, D>, ...args: unknown[]): NnTensor<S, D>
      gamma: NnTensor<[number], D>
      beta: NnTensor<[number], D>
    }

    /**
     * @summary Own a list of submodules.
     * @category module container
     * @semantics Creates a module-owned container whose parameter and state traversal recurse through contained modules.
     * @input Child modules.
     * @output A module-owned list with recursive parameter/state behavior.
     * @usecase Store array-like groups of child modules.
     * @param {Module[]} modules - Child modules.
     * @returns {ModuleList} A module-owned list of child modules.
     * @example
     * const layers = nn.module_list([nn.Linear(2, 4), nn.Linear(4, 1)])
     */
    function module_list(modules: ModuleListEntry[]): ModuleList
    /**
     * @summary Create a list of submodules from a length and factory callback.
     * @category module container
     * @semantics Calls `make(index)` for each position and stores the returned modules in a module-owned list.
     * @input A length and a module factory callback.
     * @output A module-owned list with recursive parameter/state behavior.
     * @usecase Programmatically build repeated stacks of modules.
     * @param {number} length - Number of modules to create.
     * @param {(index: number) => Module} make - Factory callback for each position.
     * @returns {ModuleList} A module-owned list of child modules.
     * @example
     * const layers = nn.module_list(2, () => nn.Linear(4, 4))
     */
    function module_list(length: number, make: (index: number) => ModuleListEntry): ModuleList
    /**
     * @summary Fully connected layer over the last dimension.
     * @category learnable layer
     * @semantics Applies the same affine projection to every leading position using trainable `weight` and `bias`.
     * @input Tensors whose last axis has size `in_features`.
     * @output Tensors whose last axis has size `out_features`.
     * @shape `[..., in_features] -> [..., out_features]`
     * @axis operates on the last dimension of the input
     * @dtype layer parameters and outputs use the configured dtype.
     * @formula y = x @ weight + bias
     * @math y = xW + b
     * @usecase MLPs, classifier heads, and projection layers.
     * @param {number} in_features - Size of the input feature dimension.
     * @param {number} out_features - Size of the output feature dimension.
     * @returns {LinearLayer<InFeatures, OutFeatures, D>} A callable linear layer.
     * @see [basic.md](../docs/ml/nn/basic.md)
     * @example
     * const layer = nn.Linear(2, 4)
     */
    function Linear<
      InFeatures extends number,
      OutFeatures extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(in_features: InFeatures, out_features: OutFeatures, opts?: { dtype?: D }): LinearLayer<InFeatures, OutFeatures, D>
    /**
     * @summary Token embedding lookup table.
     * @category learnable lookup layer
     * @semantics Looks up a trainable dense vector for each token id.
     * @input Token-id tensors shaped `[tokens]` or `[batch, tokens]`.
     * @output Tensors with one trailing embedding axis.
     * @shape `[tokens] -> [tokens, embedding_dim]`, `[batch, tokens] -> [batch, tokens, embedding_dim]`
     * @axis appends a new trailing embedding axis
     * @usecase Trainable token representations for language and sequence models.
     * @param {number} num_embeddings - Vocabulary size.
     * @param {number} embedding_dim - Size of each embedding vector.
     * @returns {EmbeddingLayer<NumEmbeddings, EmbeddingDim, D>} A callable embedding layer.
     * @example
     * const emb = nn.Embedding(32000, 256)
     */
    function Embedding<
      NumEmbeddings extends number,
      EmbeddingDim extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(num_embeddings: NumEmbeddings, embedding_dim: EmbeddingDim, opts?: EmbeddingOptions<D>): EmbeddingLayer<NumEmbeddings, EmbeddingDim, D>
    /**
     * @summary Simple Elman RNN over a single sequence-first input.
     * @category recurrent layer
     * @semantics Iterates over sequence positions and reuses the same recurrent weights at each step.
     * @input Sequence-first tensors shaped `[seq_len, input_size]`.
     * @output Sequence-first tensors shaped `[seq_len, hidden_size]`.
     * @shape `[seq_len, input_size] -> [seq_len, hidden_size]`
     * @axis iterates over the sequence axis
     * @usecase Single-sequence recurrent modeling without an explicit batch dimension.
     * @param {number} input_size - Input feature size at each timestep.
     * @param {number} hidden_size - Hidden state size.
     * @param {{ num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu' }} [opts] - Recurrent layer options.
     * @returns {SimpleRNNModule<InputSize, HiddenSize, D>} A callable simple RNN module.
     * @see [recurrent.md](../docs/ml/nn/recurrent.md)
     * @example
     * const rnn = nn.SimpleRNN(8, 16, { num_layers: 2 })
     */
    function SimpleRNN<
      InputSize extends number,
      HiddenSize extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(
      input_size: InputSize,
      hidden_size: HiddenSize,
      opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu' }
    ): SimpleRNNModule<InputSize, HiddenSize, D>
    /**
     * @summary Classic Elman RNN over batched inputs.
     * @category recurrent layer
     * @semantics Iterates over sequence positions while carrying recurrent hidden state across timesteps.
     * @input Sequence-first or batch-first tensors, depending on `opts.batch_first`.
     * @output Tensors with the same batch/sequence layout and hidden-size trailing axis.
     * @shape sequence-first `[seq_len, batch, input_size] -> [seq_len, batch, hidden_size]`, or batch-first `[batch, seq_len, input_size] -> [batch, seq_len, hidden_size]`
     * @axis loops over sequence positions while processing each batch slice in parallel
     * @usecase Batched recurrent modeling with optional `batch_first` layout.
     * @param {number} input_size - Input feature size at each timestep.
     * @param {number} hidden_size - Hidden state size.
     * @param {{ num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu'; batch_first?: boolean }} [opts] - Recurrent layer options.
     * @returns {RNNModule<InputSize, HiddenSize, D> | RNNBatchFirstModule<InputSize, HiddenSize, D>} A callable RNN module.
     * @see [recurrent.md](../docs/ml/nn/recurrent.md)
     * @example
     * const rnn = nn.RNN(8, 16, { num_layers: 2 })
     */
    function RNN<
      InputSize extends number,
      HiddenSize extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(
      input_size: InputSize,
      hidden_size: HiddenSize,
      opts: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu'; batch_first: true }
    ): RNNBatchFirstModule<InputSize, HiddenSize, D>
    function RNN<
      InputSize extends number,
      HiddenSize extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(
      input_size: InputSize,
      hidden_size: HiddenSize,
      opts?: { num_layers?: number; bias?: boolean; nonlinearity?: 'tanh' | 'relu'; batch_first?: boolean }
    ): RNNModule<InputSize, HiddenSize, D>
    /**
     * @summary Classic LSTM over batched inputs.
     * @category recurrent layer
     * @semantics Iterates over sequence positions while maintaining both hidden and cell state.
     * @input Sequence-first or batch-first tensors, depending on `opts.batch_first`.
     * @output Tensors with the same batch/sequence layout and hidden-size trailing axis.
     * @shape sequence-first `[seq_len, batch, input_size] -> [seq_len, batch, hidden_size]`, or batch-first `[batch, seq_len, input_size] -> [batch, seq_len, hidden_size]`
     * @axis loops over sequence positions while maintaining hidden and cell state
     * @usecase Batched recurrent modeling with gated memory.
     * @param {number} input_size - Input feature size at each timestep.
     * @param {number} hidden_size - Hidden state size.
     * @param {{ num_layers?: number; bias?: boolean; batch_first?: boolean }} [opts] - LSTM options.
     * @returns {LSTMModule<InputSize, HiddenSize, D> | LSTMBatchFirstModule<InputSize, HiddenSize, D>} A callable LSTM module.
     * @see [recurrent.md](../docs/ml/nn/recurrent.md)
     * @example
     * const lstm = nn.LSTM(8, 16, { num_layers: 2 })
     */
    function LSTM<
      InputSize extends number,
      HiddenSize extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(
      input_size: InputSize,
      hidden_size: HiddenSize,
      opts: { num_layers?: number; bias?: boolean; batch_first: true }
    ): LSTMBatchFirstModule<InputSize, HiddenSize, D>
    function LSTM<
      InputSize extends number,
      HiddenSize extends number,
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(
      input_size: InputSize,
      hidden_size: HiddenSize,
      opts?: { num_layers?: number; bias?: boolean; batch_first?: boolean }
    ): LSTMModule<InputSize, HiddenSize, D>
    /**
     * @summary Chain layers or stateless activations in order.
     * @category module composition
     * @semantics Runs each layer in sequence, feeding the output of one stage into the next.
     * @input A variadic ordered list of modules or stateless tensor transforms.
     * @output A callable sequential module.
     * @shape output shape of one layer becomes the input shape of the next
     * @usecase Simple feed-forward model assembly without writing a custom module class.
     * @param {...(Module | ((x: NnTensor<any, D>) => NnTensor<any, D>))} layers - Layers or stateless activations to run in order.
     * @returns {SequentialModule<InputShape, OutputShape, D>} A callable sequential module.
     * @example
     * import { relu } from 'affon:compute/legacy'
     *
     * const model = nn.Sequential(nn.Linear(2, 4), relu, nn.Linear(4, 1))
     */
    function Sequential<S1 extends Shape, S2 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>)
    ): TypedSequentialModule<S1, S2, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>)
    ): TypedSequentialModule<S1, S3, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>)
    ): TypedSequentialModule<S1, S4, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>)
    ): TypedSequentialModule<S1, S5, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>)
    ): TypedSequentialModule<S1, S6, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, S7 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>),
      l6: Module | ((x: NnTensor<S6, D>) => NnTensor<S7, D>)
    ): TypedSequentialModule<S1, S7, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, S7 extends Shape, S8 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>),
      l6: Module | ((x: NnTensor<S6, D>) => NnTensor<S7, D>),
      l7: Module | ((x: NnTensor<S7, D>) => NnTensor<S8, D>)
    ): TypedSequentialModule<S1, S8, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, S7 extends Shape, S8 extends Shape, S9 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>),
      l6: Module | ((x: NnTensor<S6, D>) => NnTensor<S7, D>),
      l7: Module | ((x: NnTensor<S7, D>) => NnTensor<S8, D>),
      l8: Module | ((x: NnTensor<S8, D>) => NnTensor<S9, D>)
    ): TypedSequentialModule<S1, S9, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, S7 extends Shape, S8 extends Shape, S9 extends Shape, S10 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>),
      l6: Module | ((x: NnTensor<S6, D>) => NnTensor<S7, D>),
      l7: Module | ((x: NnTensor<S7, D>) => NnTensor<S8, D>),
      l8: Module | ((x: NnTensor<S8, D>) => NnTensor<S9, D>),
      l9: Module | ((x: NnTensor<S9, D>) => NnTensor<S10, D>)
    ): TypedSequentialModule<S1, S10, D>
    function Sequential<S1 extends Shape, S2 extends Shape, S3 extends Shape, S4 extends Shape, S5 extends Shape, S6 extends Shape, S7 extends Shape, S8 extends Shape, S9 extends Shape, S10 extends Shape, S11 extends Shape, D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(
      l1: Module | ((x: NnTensor<S1, D>) => NnTensor<S2, D>),
      l2: Module | ((x: NnTensor<S2, D>) => NnTensor<S3, D>),
      l3: Module | ((x: NnTensor<S3, D>) => NnTensor<S4, D>),
      l4: Module | ((x: NnTensor<S4, D>) => NnTensor<S5, D>),
      l5: Module | ((x: NnTensor<S5, D>) => NnTensor<S6, D>),
      l6: Module | ((x: NnTensor<S6, D>) => NnTensor<S7, D>),
      l7: Module | ((x: NnTensor<S7, D>) => NnTensor<S8, D>),
      l8: Module | ((x: NnTensor<S8, D>) => NnTensor<S9, D>),
      l9: Module | ((x: NnTensor<S9, D>) => NnTensor<S10, D>),
      l10: Module | ((x: NnTensor<S10, D>) => NnTensor<S11, D>)
    ): TypedSequentialModule<S1, S11, D>
    function Sequential<
      InputShape extends Shape = number[],
      OutputShape extends Shape = number[],
      D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">
    >(...layers: (Module | ((x: NnTensor<any, D>) => NnTensor<any, D>))[]): SequentialModule<InputShape, OutputShape, D>
    /**
     * @summary Mean squared error loss factory.
     * @category loss
     * @semantics Builds a callable loss that compares prediction and target elementwise, then reduces to `[1]`.
     * @input Optional reduction configuration.
     * @output A callable loss function.
     * @errors Prediction and target must have the same shape at call time.
     * @shape `prediction.shape = target.shape`, output `[1]`
     * @formula mean((prediction - target)^2)
     * @math \frac{1}{N}\sum_i(\hat{y}_i-y_i)^2
     * @usecase Regression.
     * @param {{ reduction?: string }} [opts] - Reduction configuration.
     * @returns {(prediction: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>} A callable MSE loss.
     * @see [loss.md](../docs/ml/nn/loss.md)
     * @example
     * const criterion = nn.MSELoss()
     */
    function MSELoss(opts?: { reduction?: string }): <S extends Shape, D extends Extract<DType, "f32" | "f64">>(prediction: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>
    /**
     * @summary Binary cross-entropy loss factory.
     * @category loss
     * @semantics Builds a callable BCE loss that expects probability inputs and binary targets, then reduces to `[1]`.
     * @input Optional reduction configuration.
     * @output A callable loss function.
     * @errors Predictions must already be probabilities in `[0, 1]`; targets must be binary at call time.
     * @shape probabilities and targets shaped `[N]` or `[N, 1]`; output `[1]`
     * @formula -mean(target * log(pred) + (1 - target) * log(1 - pred))
     * @math -\frac{1}{N}\sum_i \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
     * @usecase Binary classification when probabilities are already computed.
     * @param {{ reduction?: string }} [opts] - Reduction configuration.
     * @returns {(prediction: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>} A callable BCE loss.
     * @see [loss.md](../docs/ml/nn/loss.md)
     * @example
     * const criterion = nn.BCELoss()
     */
    function BCELoss(opts?: { reduction?: string }): <S extends Shape, D extends Extract<DType, "f32" | "f64">>(prediction: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>
    /**
     * @summary Stable binary cross-entropy from raw logits.
     * @category loss
     * @semantics Builds a callable BCE loss that consumes raw logits and performs the stable combined formulation internally.
     * @input Optional reduction configuration.
     * @output A callable loss function.
     * @shape logits and targets shaped `[N]` or `[N, 1]`; output `[1]`
     * @formula mean(max(x, 0) - x * target + log(1 + exp(-abs(x))))
     * @math \frac{1}{N}\sum_i \left(\max(x_i,0) - x_i y_i + \log(1 + e^{-|x_i|})\right)
     * @usecase Binary classification with raw logits.
     * @param {{ reduction?: string }} [opts] - Reduction configuration.
     * @returns {(logits: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>} A callable BCE-with-logits loss.
     * @see [loss.md](../docs/ml/nn/loss.md)
     * @example
     * const criterion = nn.BCEWithLogitsLoss()
     */
    function BCEWithLogitsLoss(opts?: { reduction?: string }): <S extends Shape, D extends Extract<DType, "f32" | "f64">>(logits: NnTensor<S, D>, target: NnTensor<S, D>) => NnTensor<[1], D>
    /**
     * @summary Cross-entropy loss factory for one-hot targets.
     * @category loss
     * @semantics Builds a callable cross-entropy loss for 2-D logits and one-hot targets, then reduces to `[1]`.
     * @input Optional reduction configuration.
     * @output A callable loss function.
     * @errors Expects 2-D logits and one-hot/probability targets shaped `[batch, classes]`, or indexed targets shaped `[batch]`.
     * @shape logits shaped `[batch, classes]`; targets shaped `[batch, classes]` or `[batch]`; output `[1]`
     * @axis class dimension is the last axis of the 2D input
     * @formula -mean(target * log_softmax(logits))
     * @math -\frac{1}{N}\sum_i \sum_c y_{ic}\log \mathrm{softmax}(x_{ic})
     * @usecase Multiclass classification with one-hot/probability targets or indexed class ids.
     * @param {{ reduction?: 'mean'; target?: 'auto' | 'index' | 'probability' | 'one_hot'; axis?: number }} [opts] - Loss configuration.
     * @returns {(logits: Tensor, targets: Tensor) => NnTensor<[1], DType>} A callable cross-entropy loss.
     * @see [loss.md](../docs/ml/nn/loss.md)
     * @example
     * const criterion = nn.CrossEntropyLoss()
     */
    function CrossEntropyLoss(opts?: { reduction?: 'mean'; target?: 'auto' | 'index' | 'probability' | 'one_hot'; axis?: number }): (logits: Tensor, targets: Tensor) => NnTensor<[1], Extract<DType, "f32" | "f64">>
    /**
     * @summary Create a causal attention mask.
     * @category sequence helper
     * @semantics Builds a square mask tensor whose strictly upper-triangular
     * entries are `1` and whose visible positions are `0`.
     * @input Positive sequence length plus optional dtype/device placement.
     * @output A square `[length, length]` mask tensor.
     * @usecase Autoregressive attention and sequence models.
     */
    function causal_mask(length: number, opts?: { dtype?: Extract<DType, "f32" | "f64">; device?: Device }): NnTensor<[number, number], Extract<DType, "f32" | "f64">>
    /**
     * @summary Apply a causal attention mask to score tensors.
     * @category sequence helper
     * @semantics Applies a causal mask over the last two dimensions of a score
     * tensor using `masked_fill(...)`.
     * @input A tensor with at least two dimensions.
     * @output A tensor with the same shape and dtype.
     */
    function apply_causal_mask<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(scores: NnTensor<S, D>, value?: number): NnTensor<number[], D>
    /**
     * @summary Create sinusoidal positional encodings.
     * @category sequence helper
     * @semantics Builds the classic sinusoidal encoding table shaped
     * `[length, dim]`.
     * @input Positive sequence length and feature dimension.
     * @output A dense positional encoding tensor.
     */
    function sinusoidal_encoding(length: number, dim: number, opts?: SinusoidalEncodingOptions): NnTensor<[number, number], Extract<DType, "f32" | "f64">>
    /**
     * @summary Create 0-based position ids.
     * @category sequence helper
     * @semantics Builds a one-dimensional tensor `[0, 1, ..., length - 1]`.
     * @input Positive sequence length.
     * @output A one-dimensional `f32` tensor of position ids.
     */
    function position_ids(length: number): NnTensor<[number], "f32">
    /**
     * @summary Dropout layer with train/eval behavior.
     * @category regularization layer
     * @semantics In training mode, randomly zeroes elements and rescales survivors; in eval mode, behaves as identity.
     * @input Tensors of any shape.
     * @output Tensors with the same shape and dtype.
     * @shape preserves input shape
     * @axis all axes are processed elementwise
     * @formula during training, randomly zero elements and rescale survivors by `1 / (1 - p)`
     * @usecase Reduce overfitting during training.
     * @param {number} [p] - Drop probability in `[0, 1)`.
     * @returns {DropoutLayer<S, D>} A callable dropout layer.
     * @example
     * const drop = nn.Dropout(0.5)
     */
    function Dropout<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(p?: number): DropoutLayer<S, D>
    /**
     * @summary Batch normalization layer with train/eval behavior.
     * @category normalization layer
     * @semantics Normalizes 2-D feature inputs using batch statistics in training and running statistics in eval.
     * @input 2-D tensors shaped `[batch, features]`.
     * @output Tensors with the same shape and dtype.
     * @errors Current built-in behavior supports 2-D feature normalization only.
     * @shape `[batch, features] -> [batch, features]`
     * @axis current built-in behavior normalizes over axis `0` only
     * @formula y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
     * @math y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\gamma + \beta
     * @usecase Stabilize 2D MLP-style training activations.
     * @param {number} num_features - Feature dimension size.
     * @param {{ momentum?: number; eps?: number }} [opts] - Batch norm options.
     * @returns {BatchNormLayer<D>} A callable batch norm layer.
     * @example
     * const bn = nn.BatchNorm(16)
     */
    function BatchNorm<D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(num_features: number, opts?: { momentum?: number; eps?: number }): BatchNormLayer<D>
    /**
     * @summary Layer normalization over the last dimension.
     * @category normalization layer
     * @semantics Normalizes each slice independently over the last dimension using its own mean and variance.
     * @input Tensors whose last dimension matches `normalized_shape`.
     * @output Tensors with the same shape and dtype.
     * @shape `[..., normalized_shape] -> [..., normalized_shape]`
     * @axis normalizes only the last dimension
     * @formula y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
     * @math y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\gamma + \beta
     * @usecase Text and sequence models where inputs often look like `[batch, features]` or `[seq, batch, features]`.
     * @param {number} normalized_shape - Size of the normalized trailing dimension.
     * @param {{ eps?: number }} [opts] - Layer norm options.
     * @returns {LayerNormLayer<S, D>} A callable layer norm layer.
     * @example
     * const ln = nn.LayerNorm(8)
     */
    function LayerNorm<S extends Shape = number[], D extends Extract<DType, "f32" | "f64"> = Extract<DType, "f32" | "f64">>(normalized_shape: number, opts?: { eps?: number; dtype?: D }): LayerNormLayer<S, D>
  }

  export default nn
}

import nn from 'affon:nn'
import { compile, copy, tensor, empty, parameter, gelu, relu, softmax, module as computeModule } from 'affon:compute'
import type { ComputeState, DType as ComputeDType, Shape as ComputeShape, Tensor as ComputeTensor } from 'affon:compute'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
type IsAssignable<A, B> = A extends B ? true : false
function assertType<T extends true>() {}
type NNDType = Extract<ComputeDType, 'f32' | 'f64'>
type NNTensor<S extends ComputeShape = number[], D extends NNDType = NNDType> = ComputeTensor<S, D>
type CallableModule<I extends ComputeShape, O extends ComputeShape, D extends NNDType = NNDType> = nn.Module<[ComputeTensor<I, D>], ComputeTensor<O, D>> & {
  (x: ComputeTensor<I, D>, ...args: unknown[]): ComputeTensor<O, D>
}

const ln = nn.LayerNorm(8)
const x = tensor<[1, 8], 'f32'>([[1, 2, 3, 4, 5, 6, 7, 8]], { dtype: 'f32' })
const y = ln(x)
const mask = nn.causal_mask(4, { dtype: 'f32' })
const maskedScores = nn.apply_causal_mask(tensor<[2, 2], 'f32'>([[1, 2], [3, 4]], { dtype: 'f32' }))
const pos = nn.position_ids(8)
const sinusoidal = nn.sinusoidal_encoding(8, 16, { dtype: 'f32' })
const weight = copy(parameter<[2, 1], 'f32'>([2, 1], { dtype: 'f32' }), tensor<[2, 1], 'f32'>([[1], [2]], { dtype: 'f32' }))
const scratch = empty([2, 2] as const, { dtype: 'f32' })
const initWeight = nn.init.xavier_uniform(weight)
const affine = computeModule({
  weight,
  bias: copy(parameter<[1, 1], 'f32'>([1, 1], { dtype: 'f32' }), tensor<[1, 1], 'f32'>([[0]], { dtype: 'f32' })),
} as any, (_state, x) => x) as unknown as CallableModule<[number, 2], [number, 2], 'f32'> & { weight: typeof weight; bias: NNTensor<[1, 1], 'f32'> }
const logistic = computeModule({
  W: weight,
  b: copy(parameter<[1, 1], 'f32'>([1, 1], { dtype: 'f32' }), tensor<[1, 1], 'f32'>([[0]], { dtype: 'f32' })),
} as any, (_state, x) => x) as unknown as CallableModule<[number, 1], [number, 1], 'f32'> & { W: typeof weight; b: NNTensor<[1, 1], 'f32'> }
const params = affine.parameters
const named = affine.parameters.named()
const affinePath = affine.metadata('affine').module_path
const layers = nn.module_list([nn.Linear(2, 3), nn.Linear(3, 1)])
const linear = nn.Linear(2, 3)
const embedding = nn.Embedding(16, 4)
const linearInput = tensor<[1, 2], 'f32'>([[1, 2]], { dtype: 'f32' })
const linearOutput = linear(linearInput)
const linearBatchInput = tensor([[[1, 2], [3, 4]]]) as ComputeTensor<[number, number, 2], 'f32'>
const linearBatchOutput = linear(linearBatchInput)
const embeddingTokens = tensor<[number], 'f32'>([0, 2, 1], { dtype: 'f32' })
const embeddingBatchTokens = tensor<[number, number], 'f32'>([[0, 1], [2, 3]], { dtype: 'f32' })
const embeddingOutput = embedding(embeddingTokens)
const embeddingBatchOutput = embedding(embeddingBatchTokens)
const geluOutput = gelu(tensor<[1, 3], 'f32'>([[-1, 0, 1]], { dtype: 'f32' }))
const softmaxOutput = softmax(tensor<[1, 3], 'f32'>([[1, 2, 3]], { dtype: 'f32' }), 1)
const bceWithLogitsLoss = nn.BCEWithLogitsLoss()
const bceWithLogitsOutput = bceWithLogitsLoss(
  tensor<[2], 'f32'>([1, -1], { dtype: 'f32' }),
  tensor<[2], 'f32'>([1, 0], { dtype: 'f32' }),
)
const seq = nn.Sequential(
  nn.Linear(2, 3),
  relu,
  nn.Linear(3, 1)
)
const seqOutput = seq(linearInput)
const seqModule: nn.SequentialModule<any, any, any> = seq
const affineState = affine.state()
const explicitModule: nn.Module<[ComputeTensor<[number, 2], NNDType>], ComputeTensor<[number, 3], NNDType>> = linear
const explicitLayer: nn.Module<[ComputeTensor<[number, 2], NNDType>], ComputeTensor<[number, 3], NNDType>> | ((x: ComputeTensor<[number, 2], NNDType>) => ComputeTensor<[number, 3], NNDType>) = linear
const linearCompiled = compile(linear as unknown as (...args: ComputeTensor[]) => ComputeTensor)
const tinyL = nn.Linear(2, 1)
const tiny = computeModule({ l: tinyL } as any, (state, x) => state.l(x)) as unknown as CallableModule<[number, 2], [number, 1], NNDType> & { l: typeof tinyL }
const tinyOutput = tiny(linearInput)
const dropout = nn.Dropout(0.5)
const dropoutOutput = dropout(linearInput)
const dropoutLayer: nn.DropoutLayer<any, any> = dropout
const bn = nn.BatchNorm(3)
const bnOutput = bn(tensor<[1, 3], 'f32'>([[1, 2, 3]], { dtype: 'f32' }))
const bnLayer: nn.BatchNormLayer<any> = bn
const rnn = nn.SimpleRNN(2, 4, { num_layers: 2, nonlinearity: 'relu' })
const rnnInput = tensor<[2, 2], 'f32'>([[1, 2], [3, 4]], { dtype: 'f32' })
const rnnOutput = rnn(rnnInput)
const rnnHidden = rnn.hidden_state()
const rnnLayer: nn.SimpleRNNModule<2, 4, NNDType> = rnn
const classicRnn = nn.RNN(2, 4, { num_layers: 2 })
const classicRnnInput = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) as ComputeTensor<[number, number, 2], 'f32'>
const classicRnnOutput = classicRnn(classicRnnInput)
const classicRnnHidden = classicRnn.hidden_state()
const classicRnnLayer: nn.RNNModule<2, 4, NNDType> = classicRnn
const batchFirstRnn = nn.RNN(2, 4, { num_layers: 2, batch_first: true })
const batchFirstRnnInput = tensor([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]) as ComputeTensor<[number, number, 2], 'f32'>
const batchFirstRnnOutput = batchFirstRnn(batchFirstRnnInput)
const batchFirstRnnHidden = batchFirstRnn.hidden_state()
const lstm = nn.LSTM(2, 4, { num_layers: 2 })
const lstmInput = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) as ComputeTensor<[number, number, 2], 'f32'>
const lstmOutput = lstm(lstmInput)
const lstmHidden = lstm.hidden_state()
const lstmCell = lstm.cell_state()
const lstmLayer: nn.LSTMModule<2, 4, NNDType> = lstm
const batchFirstLstm = nn.LSTM(2, 4, { num_layers: 2, batch_first: true })
const batchFirstLstmInput = tensor([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]) as ComputeTensor<[number, number, 2], 'f32'>
const batchFirstLstmOutput = batchFirstLstm(batchFirstLstmInput)
const batchFirstLstmHidden = batchFirstLstm.hidden_state()
const batchFirstLstmCell = batchFirstLstm.cell_state()

assertType<IsAssignable<typeof y, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof mask, ComputeTensor<[number, number], NNDType>>>()
assertType<IsAssignable<typeof maskedScores, ComputeTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof pos, ComputeTensor<[number], 'f32'>>>()
assertType<IsAssignable<typeof sinusoidal, ComputeTensor<[number, number], NNDType>>>()
assertType<IsAssignable<typeof initWeight, NNTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof ln.gamma, NNTensor<[number], NNDType>>>()
assertType<IsAssignable<typeof ln.beta, NNTensor<[number], NNDType>>>()
assertType<IsAssignable<typeof params[number], ComputeTensor<number[], any>>>()
assertType<IsExact<typeof named[number][0], string>>()
assertType<IsAssignable<typeof named[number][1], ComputeTensor<number[], any>>>()
assertType<IsExact<typeof layers.length, number>>()
assertType<IsAssignable<typeof layers[0], nn.Module<any, any>>>()
assertType<IsAssignable<typeof logistic.W, NNTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof logistic.b, NNTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof linearOutput, NNTensor<[1, 3], NNDType>>>()
assertType<IsAssignable<typeof linearBatchOutput, NNTensor<[number, number, 3], NNDType>>>()
assertType<IsAssignable<typeof embedding.weight, NNTensor<[16, 4], NNDType>>>()
assertType<IsAssignable<typeof embeddingOutput, NNTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof embeddingBatchOutput, NNTensor<[number, number, 4], NNDType>>>()
assertType<IsAssignable<typeof geluOutput, NNTensor<[1, 3], NNDType>>>()
assertType<IsAssignable<typeof softmaxOutput, NNTensor<number[], NNDType>>>()
assertType<IsAssignable<typeof bceWithLogitsOutput, NNTensor<[1], NNDType>>>()
assertType<IsAssignable<typeof seqModule, nn.SequentialModule<any, any, any>>>()
assertType<IsExact<typeof affineState, ComputeState>>()
assertType<IsAssignable<typeof explicitModule, nn.Module<[ComputeTensor<[number, 2], NNDType>], ComputeTensor<[number, 3], NNDType>>>>()
assertType<IsAssignable<typeof explicitLayer, nn.Module<[ComputeTensor<[number, 2], NNDType>], ComputeTensor<[number, 3], NNDType>> | ((x: ComputeTensor<[number, 2], NNDType>) => ComputeTensor<[number, 3], NNDType>)>>()
assertType<IsAssignable<typeof linearCompiled, (...args: ComputeTensor[]) => ComputeTensor>>()
assertType<IsAssignable<typeof tinyOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof dropoutLayer, nn.DropoutLayer<any, any>>>()
assertType<IsAssignable<typeof bnLayer, nn.BatchNormLayer<any>>>()
assertType<IsAssignable<typeof rnnOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof rnnHidden, NNTensor<[number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof rnnLayer, nn.SimpleRNNModule<2, 4, NNDType>>>()
assertType<IsAssignable<typeof classicRnnOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof classicRnnHidden, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof classicRnnLayer, nn.RNNModule<2, 4, NNDType>>>()
assertType<IsAssignable<typeof batchFirstRnnOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof batchFirstRnnHidden, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof batchFirstRnn, nn.RNNBatchFirstModule<2, 4, NNDType>>>()
assertType<IsAssignable<typeof lstmOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof lstmHidden, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof lstmCell, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof lstmLayer, nn.LSTMModule<2, 4, NNDType>>>()
assertType<IsAssignable<typeof batchFirstLstmOutput, ComputeTensor<number[], any>>>()
assertType<IsAssignable<typeof batchFirstLstmHidden, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof batchFirstLstmCell, NNTensor<[number, number, 4], NNDType> | null>>()
assertType<IsAssignable<typeof batchFirstLstm, nn.LSTMBatchFirstModule<2, 4, NNDType>>>()

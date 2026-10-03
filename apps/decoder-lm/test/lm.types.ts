import { Session, Tensor, program, type Tensor as EvaluatedTensor } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { DecoderModel } from '../src/model.ts'
import { generate } from '../src/causal-lm.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const model = DecoderModel(32000, 16, { numLayers: 2, numHeads: 4, hiddenDim: 64, causal: true, positional: 'learned', maxSeqLen: 64 })
const session = new Session({ device: 'cpu' })
const modelProgram = program('decoder_lm', p => model({
  token_ids: p.argument('token_ids', Tensor.i64([2, 3], { axes: ['batch', 'token'] })),
}, 'decoder'))
const state = session.initialize(modelProgram)
const tokenIds = session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
const logits = session.compile(modelProgram).run({ token_ids: tokenIds }, state)
assertType<IsExact<typeof logits, EvaluatedTensor>>()
const labels = session.tensor([[2, 3, 4], [5, 6, 7]], { dtype: 'i64' })
const objective = cross_entropy()
const lossProgram = program('decoder_loss', p => objective({
  input: p.argument('logits', Tensor.f32([2, 3, 32000])),
  target: p.argument('labels', Tensor.i64([2, 3])),
}))
const loss = session.compile(lossProgram).run({ logits, labels })
assertType<IsExact<typeof loss, EvaluatedTensor>>()
const generated = generate(model, session, state, tokenIds, { max_new_tokens: 2 })
assertType<IsExact<typeof generated, EvaluatedTensor>>()

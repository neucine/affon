import { Session, type Tensor } from 'affon:compute'
import { DecoderModel } from '../src/model.ts'
import { generate } from '../src/causal-lm.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const model = DecoderModel(32000, 16, { numLayers: 2, numHeads: 4, hiddenDim: 64, causal: true, positional: 'learned', maxSeqLen: 64 })
const session = new Session({ device: 'cpu' })
const forward = model.forward(2, 3)
const state = session.initialize(forward)
const tokenIds = session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
const logits = session.compile(forward).run({ token_ids: tokenIds }, state)
assertType<IsExact<typeof logits, Tensor>>()
const labels = session.tensor([[2, 3, 4], [5, 6, 7]], { dtype: 'i64' })
const loss = session.compile(model.loss(2, 3)).run({ logits, labels })
assertType<IsExact<typeof loss, Tensor>>()
const generated = generate(model, session, state, tokenIds, { max_new_tokens: 2 })
assertType<IsExact<typeof generated, Tensor>>()

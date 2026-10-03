import { Session, Tensor } from 'affon:compute'
import { DecoderModel } from '../src/model.ts'
import { CausalLMLoss, generate } from '../src/causal-lm.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const model = DecoderModel(32000, 16, { numLayers: 2, numHeads: 4, hiddenDim: 64, causal: true, positional: 'learned', maxSeqLen: 64 })
const tokenIds = model.session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
const logits = model(tokenIds)
assertType<IsExact<typeof logits, Tensor>>()
const criterion = CausalLMLoss()
const lossTokens = model.session.tensor([[1, 2, 3, 4], [4, 5, 6, 7]], { dtype: 'i64' })
const lossInputs = model.session.tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'i64' })
const loss = criterion(model(lossInputs), lossTokens)
assertType<IsExact<typeof loss, Tensor>>()
const generated = generate(model, tokenIds, { max_new_tokens: 2 })
assertType<IsExact<typeof generated, Tensor>>()
const generatedWithForbidden = generate(model, tokenIds, { max_new_tokens: 2, forbidden_token_ids: [0, 1] })
assertType<IsExact<typeof generatedWithForbidden, Tensor>>()
const generatedWithSampling = generate(model, tokenIds, { max_new_tokens: 2, temperature: 0.8, top_k: 5 })
assertType<IsExact<typeof generatedWithSampling, Tensor>>()
const otherSession = new Session({ device: 'cpu' })
const otherTokens = otherSession.tensor([[1, 2, 3]], { dtype: 'i64' })
assertType<IsExact<typeof otherTokens, Tensor>>()

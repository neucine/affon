import { expect, test } from 'std:test'
import { tensor, seed, setDevice, grad, clear_grad, adam, clip_grad_norm } from 'affon:compute'
import { DecoderModel } from '../../packages/@affon/lm/src/model.ts'
import { CausalLMLoss, generate } from '../../packages/@affon/lm/src/causal-lm.ts'

test('CUDA decoder training matches CPU with tied embeddings and causal attention', () => {
  function train(device: 'cpu' | 'cuda') {
    setDevice(device)
    seed(42)
    const model = DecoderModel(8, 4, { numLayers: 1, numHeads: 2, hiddenDim: 8, positional: 'learned', maxSeqLen: 8, dropout: 0, causal: true, tieEmbeddings: true })
    const params = model.parameters
    const step = adam({ lr: 0.005 })
    const criterion = CausalLMLoss()
    const tokens = tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    const inputs = tokens.slice([':', '0:3'])
    const losses: number[] = []
    for (let i = 0; i < 3; i++) {
      clear_grad(params)
      const loss = criterion(model(inputs), tokens)
      losses.push(loss.item())
      grad(loss, params)
      clip_grad_norm(params, 1)
      step(params)
    }
    return { losses, weights: params.map(p => p.to_array().flat(Infinity)) }
  }
  try {
    const cpu = train('cpu'), cuda = train('cuda')
    for (let i = 0; i < cpu.losses.length; i++) expect(Math.abs(cuda.losses[i] - cpu.losses[i]) < 1e-4).toBe(true)
    expect(cuda.losses[2] < cuda.losses[0]).toBe(true)
    for (let p = 0; p < cpu.weights.length; p++) {
      for (let i = 0; i < cpu.weights[p].length; i++) expect(Math.abs(cuda.weights[p][i] - cpu.weights[p][i]) < 1e-3).toBe(true)
    }
  } finally { setDevice('cpu') }
})


test('CUDA decoder generates greedy and top-k sampled tokens with forbidden ids', () => {
  setDevice('cuda')
  try {
    seed(42)
    const model = DecoderModel(8, 4, { numLayers: 1, numHeads: 2, hiddenDim: 8, maxSeqLen: 8, dropout: 0, causal: true })
    for (const temperature of [0, 0.7]) {
      const result = generate(model, tensor([[1, 2]]), { max_new_tokens: 2, temperature, top_k: 3, forbidden_token_ids: [0, 1] }).to_array()
      expect(result[0].length).toBe(4)
      for (const token of result[0].slice(2)) {
        expect(Number.isInteger(token) && token >= 2 && token < 8).toBe(true)
      }
    }
  } finally { setDevice('cpu') }
})

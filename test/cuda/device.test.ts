import { describe, expect, test } from 'std:test'
import { contiguous, matmul, move, relu, setDevice, softmax, tensor, transpose } from 'affon:compute/legacy'

// This suite requires CUDA. Fail explicitly if unavailable so GPU CI cannot
// silently report a successful run without exercising the backend.
describe('CUDA device', () => {
  test('packs views for activations and round-trips the result', () => {
    setDevice('cuda')
    try {
      const input = tensor([[1, -2, 3], [4, 5, -6]])
      const view = transpose(input, 0, 1)
      expect(move(contiguous(view), 'cpu').to_array()).toEqual([[1, 4], [-2, 5], [3, -6]])
      expect(move(relu(view), 'cpu').to_array()).toEqual([[1, 4], [0, 5], [3, 0]])
    } finally {
      setDevice('cpu')
    }
  })

  test('uses transposed matrices and normalizes a noncontiguous axis', () => {
    setDevice('cuda')
    try {
      const input = tensor([[1, 2, 3], [4, 5, 6]])
      const transposed = transpose(input, 0, 1)
      expect(move(matmul(input, transposed), 'cpu').to_array()).toEqual([[14, 32], [32, 77]])
      const probabilities = move(softmax(transposed, 0), 'cpu').to_array()
      for (let col = 0; col < 2; col++) {
        const total = probabilities[0][col] + probabilities[1][col] + probabilities[2][col]
        expect(Math.abs(total - 1) < 1e-6).toBe(true)
      }
    } finally {
      setDevice('cpu')
    }
  })
})

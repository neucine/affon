import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn'
import { tensor } from 'affon:compute'
import { python } from '../../support/python.ts'

describe('nn parity embedding', () => {
  test('matches torch embedding lookup and gradient accumulation', async () => {
    if (!(await python.torch.available())) return

    const embedding = nn.Embedding(5, 3)
    const tokenIds = tensor([[0, 2], [2, 4]], { dtype: 'i64' })

    const y = embedding(tokenIds)
    y.sum().backward()
    const weight = values(embedding.weight) as number[][]

    const peer = await python.torch.json<{
      value: number[][][]
      gradWeight: number[][]
    }>(`
      weight = torch.tensor(${JSON.stringify(weight)}, dtype=torch.float32, requires_grad=True)
      token_ids = torch.tensor([[0, 2], [2, 4]], dtype=torch.int64)
      y = torch.nn.functional.embedding(token_ids, weight)
      y.sum().backward()
      print(json.dumps({
        "value": y.detach().cpu().tolist(),
        "gradWeight": weight.grad.detach().cpu().tolist(),
      }))
    `)

    expect(values(y)).toBeAllClose(peer.value)
    expect(values(embedding.weight.grad)).toBeAllClose(peer.gradWeight)
  })
})

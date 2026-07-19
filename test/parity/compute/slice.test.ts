import { describe, test, expect, values } from 'std:test'
import { tensor } from 'affon:compute'
import { python } from '../../support/python.ts'
import { describeParityDevices, materializeOnDevice } from '../../support/parity.ts'

describe('compute parity slice', () => {
  describeParityDevices((device) => {
    test('matches torch stepped and reversed slice semantics', async () => {
      if (!(await python.torch.available())) return

      const x = tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ], { dtype: 'f32', device: device.name })
      const actual = x.slice(['::2', '::-1'])
      const peer = await python.torch.tensor<number[][]>(`
        x = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]], dtype=torch.float32)
        emit(torch.flip(x[::2, :], dims=[1]))
      `)

      if (device.name === 'metal') expect(actual.device).toBe('metal')
      expect(values(materializeOnDevice(actual))).toBeAllClose(peer.value, { rtol: device.rtol, atol: device.atol })
    })
  })
})

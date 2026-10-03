import { describe, test } from 'std:test'
import { add, div, mul, permute, sub } from 'affon:compute/legacy'
import { python } from '../../support/python.ts'
import { describeParityDevices } from '../../support/parity.ts'
import { expectBackwardParity } from '../../support/parity-autograd.ts'
import { internal_tensor } from '../../support/compute.ts'

describe('compute parity broadcast backward', () => {
  describeParityDevices((device) => {
    test('leading-axis and singleton-axis broadcast gradients match torch', async () => {
      if (!(await python.torch.available())) return

      const upstream = internal_tensor([
        [[1, -2, 0.5, 3], [-1, 4, 2, -0.5], [2.5, -3, 1, 0.25]],
        [[-2, 1.5, 3, -4], [0.5, -1, 2.25, 5], [4, 0.75, -2, 1]],
      ], { dtype: 'f32', device: device.name })

      const a = internal_tensor([
        [[1, -2, 3, 4]],
        [[5, 6, -7, 8]],
      ], { dtype: 'f32', device: device.name })
      const b = internal_tensor([
        [2, -1, 0.5, 4],
        [-3, 5, 1.5, -2],
        [6, -4, 2, 3],
      ], { dtype: 'f32', device: device.name })
      const addValue = add(a, b)
      mul(addValue, upstream).sum().backward()

      const subA = internal_tensor([
        [[1, -2, 3, 4]],
        [[5, 6, -7, 8]],
      ], { dtype: 'f32', device: device.name })
      const subB = internal_tensor([
        [2, -1, 0.5, 4],
        [-3, 5, 1.5, -2],
        [6, -4, 2, 3],
      ], { dtype: 'f32', device: device.name })
      const subValue = sub(subA, subB)
      mul(subValue, upstream).sum().backward()

      const mulA = internal_tensor([
        [[1, -2, 3, 4]],
        [[5, 6, -7, 8]],
      ], { dtype: 'f32', device: device.name })
      const mulB = internal_tensor([
        [2, -1, 0.5, 4],
        [-3, 5, 1.5, -2],
        [6, -4, 2, 3],
      ], { dtype: 'f32', device: device.name })
      const mulValue = mul(mulA, mulB)
      mul(mulValue, upstream).sum().backward()

      const divA = internal_tensor([
        [[1, -2, 3, 4]],
        [[5, 6, -7, 8]],
      ], { dtype: 'f32', device: device.name })
      const divB = internal_tensor([
        [2, -1, 0.5, 4],
        [-3, 5, 1.5, -2],
        [6, -4, 2, 3],
      ], { dtype: 'f32', device: device.name })
      const divValue = div(divA, divB)
      mul(divValue, upstream).sum().backward()

      const peer = await python.torch.json<{
        addValue: { value: number[][][]; shape: number[]; dtype: string }
        addGradA: { value: number[][][]; shape: number[]; dtype: string }
        addGradB: { value: number[][]; shape: number[]; dtype: string }
        subValue: { value: number[][][]; shape: number[]; dtype: string }
        subGradA: { value: number[][][]; shape: number[]; dtype: string }
        subGradB: { value: number[][]; shape: number[]; dtype: string }
        mulValue: { value: number[][][]; shape: number[]; dtype: string }
        mulGradA: { value: number[][][]; shape: number[]; dtype: string }
        mulGradB: { value: number[][]; shape: number[]; dtype: string }
        divValue: { value: number[][][]; shape: number[]; dtype: string }
        divGradA: { value: number[][][]; shape: number[]; dtype: string }
        divGradB: { value: number[][]; shape: number[]; dtype: string }
      }>(`
        upstream = torch.tensor([
          [[1., -2., 0.5, 3.], [-1., 4., 2., -0.5], [2.5, -3., 1., 0.25]],
          [[-2., 1.5, 3., -4.], [0.5, -1., 2.25, 5.], [4., 0.75, -2., 1.]],
        ], dtype=torch.float32)

        add_a = torch.tensor([
          [[1., -2., 3., 4.]],
          [[5., 6., -7., 8.]],
        ], dtype=torch.float32, requires_grad=True)
        add_b = torch.tensor([
          [2., -1., 0.5, 4.],
          [-3., 5., 1.5, -2.],
          [6., -4., 2., 3.],
        ], dtype=torch.float32, requires_grad=True)
        add_value = add_a + add_b
        (add_value * upstream).sum().backward()

        sub_a = torch.tensor([
          [[1., -2., 3., 4.]],
          [[5., 6., -7., 8.]],
        ], dtype=torch.float32, requires_grad=True)
        sub_b = torch.tensor([
          [2., -1., 0.5, 4.],
          [-3., 5., 1.5, -2.],
          [6., -4., 2., 3.],
        ], dtype=torch.float32, requires_grad=True)
        sub_value = sub_a - sub_b
        (sub_value * upstream).sum().backward()

        mul_a = torch.tensor([
          [[1., -2., 3., 4.]],
          [[5., 6., -7., 8.]],
        ], dtype=torch.float32, requires_grad=True)
        mul_b = torch.tensor([
          [2., -1., 0.5, 4.],
          [-3., 5., 1.5, -2.],
          [6., -4., 2., 3.],
        ], dtype=torch.float32, requires_grad=True)
        mul_value = mul_a * mul_b
        (mul_value * upstream).sum().backward()

        div_a = torch.tensor([
          [[1., -2., 3., 4.]],
          [[5., 6., -7., 8.]],
        ], dtype=torch.float32, requires_grad=True)
        div_b = torch.tensor([
          [2., -1., 0.5, 4.],
          [-3., 5., 1.5, -2.],
          [6., -4., 2., 3.],
        ], dtype=torch.float32, requires_grad=True)
        div_value = div_a / div_b
        (div_value * upstream).sum().backward()

        emit_many(
          addValue=add_value, addGradA=add_a.grad, addGradB=add_b.grad,
          subValue=sub_value, subGradA=sub_a.grad, subGradB=sub_b.grad,
          mulValue=mul_value, mulGradA=mul_a.grad, mulGradB=mul_b.grad,
          divValue=div_value, divGradA=div_a.grad, divGradB=div_b.grad,
        )
      `)

      expectBackwardParity({
        device,
        actual: addValue,
        gradA: a.grad!,
        gradB: b.grad!,
        peer: { value: peer.addValue, gradA: peer.addGradA, gradB: peer.addGradB },
      })
      expectBackwardParity({
        device,
        actual: subValue,
        gradA: subA.grad!,
        gradB: subB.grad!,
        peer: { value: peer.subValue, gradA: peer.subGradA, gradB: peer.subGradB },
      })
      expectBackwardParity({
        device,
        actual: mulValue,
        gradA: mulA.grad!,
        gradB: mulB.grad!,
        peer: { value: peer.mulValue, gradA: peer.mulGradA, gradB: peer.mulGradB },
      })
      expectBackwardParity({
        device,
        actual: divValue,
        gradA: divA.grad!,
        gradB: divB.grad!,
        peer: { value: peer.divValue, gradA: peer.divGradA, gradB: peer.divGradB },
      })
    })

    test('broadcast parameter gradient matches explicit expanded parameter under non-contiguous upstream', async () => {
      if (!(await python.torch.available())) return

      const aData = [
        [[1, -2, 3, -4], [5, -6, 7, -8], [9, -10, 11, -12]],
        [[-1, 2, -3, 4], [-5, 6, -7, 8], [-9, 10, -11, 12]],
      ]
      const bData = [[[0.25], [-0.5], [0.75]]]
      const upstreamBaseData = [
        [[1, -2, 0.5, 3], [-1, 4, 2, -0.5]],
        [[2.5, -3, 1, 0.25], [-2, 1.5, 3, -4]],
        [[0.5, -1, 2.25, 5], [4, 0.75, -2, 1]],
      ]

      const a = internal_tensor(aData, { dtype: 'f32', device: device.name })
      const b = internal_tensor(bData, { dtype: 'f32', device: device.name })
      const upstream = permute(internal_tensor(upstreamBaseData, { dtype: 'f32', device: device.name }), [1, 0, 2])

      const y = add(a, b)
      mul(y, upstream).sum().backward()

      const peer = await python.torch.json<{
        value: { value: number[][][]; shape: number[]; dtype: string }
        gradA: { value: number[][][]; shape: number[]; dtype: string }
        gradB: { value: number[][][]; shape: number[]; dtype: string }
        explicitReducedGradB: { value: number[][][]; shape: number[]; dtype: string }
      }>(`
        a_data = ${JSON.stringify(aData)}
        b_data = ${JSON.stringify(bData)}
        upstream_base_data = ${JSON.stringify(upstreamBaseData)}

        a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor(upstream_base_data, dtype=torch.float32).permute(1, 0, 2)
        y = a + b
        (y * upstream).sum().backward()

        explicit_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        explicit_b = torch.tensor(b_data, dtype=torch.float32).expand_as(explicit_a).clone().detach().requires_grad_(True)
        explicit_y = explicit_a + explicit_b
        (explicit_y * upstream).sum().backward()
        explicit_reduced_grad_b = explicit_b.grad.sum(dim=(0, 2), keepdim=True)

        emit_many(value=y, gradA=a.grad, gradB=b.grad, explicitReducedGradB=explicit_reduced_grad_b)
      `)

      expectBackwardParity({
        device,
        actual: y,
        gradA: a.grad!,
        gradB: b.grad!,
        peer: { value: peer.value, gradA: peer.gradA, gradB: peer.gradB },
      })
      expectBackwardParity({
        device,
        grad: b.grad!,
        peer: { grad: peer.explicitReducedGradB },
      })
    })
  })
})

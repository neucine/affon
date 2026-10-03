import { clip_grad_norm, grad, mul, parameter, sum } from 'affon:compute/legacy'

const params = Array.from({ length: 96 }, () => parameter([256, 256], { dtype: 'f32' }).randn().to('metal'))

for (let i = 1; i <= 80; i++) {
  let loss = sum(mul(params[0], params[1]))
  for (let p = 2; p < params.length; p += 2) {
    loss = sum(mul(loss, sum(mul(params[p], params[p + 1]))))
  }
  grad(loss, params)
  clip_grad_norm(params, 1.0)
  if (i % 10 === 0) console.log(`metal pressure iteration ${i}/80`)
}

console.log('metal pressure done')

import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const session = new Session({ device: 'cpu' })
const state = { weight: session.tensor([[1]]) as Tensor }
checkpoint.save(state, '/tmp/checkpoint-types.safetensors')

const loaded = checkpoint.load('/tmp/checkpoint-types.safetensors')
checkpoint.saveBundle('/tmp/checkpoint-bundle-types', {
  state: loaded,
  tensorGroups: { optimizer: loaded },
  manifest: { format: 'test' },
})
const loadedBundle = checkpoint.loadBundle('/tmp/checkpoint-bundle-types')

assertType<IsExact<typeof loaded, Record<string, Tensor>>>()
assertType<IsExact<typeof loadedBundle.state, Record<string, Tensor>>>()
assertType<IsExact<typeof loadedBundle.tensorGroups, Record<string, Record<string, Tensor>>>>()

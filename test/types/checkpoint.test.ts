import checkpoint from 'affon:checkpoint'
import { module, parameter, tensor } from 'affon:compute'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const mod = module({
  weight: parameter<[1, 1], 'f32'>([1, 1], { dtype: 'f32' }),
}, (state, x) => tensor([[1]], { dtype: 'f32' }))

checkpoint.save(mod, '/tmp/checkpoint-types.safetensors')
checkpoint.save(mod.state(), '/tmp/checkpoint-state-types.safetensors')

const loaded = checkpoint.load('/tmp/checkpoint-types.safetensors')
const restoredModule = checkpoint.restore(mod, '/tmp/checkpoint-types.safetensors')
const restoredState = checkpoint.restore(mod.state(), loaded)
checkpoint.saveBundle('/tmp/checkpoint-bundle-types', {
  state: loaded,
  tensorGroups: {
    optimizer: loaded,
  },
  manifest: {
    format: 'test',
  },
})
const loadedBundle = checkpoint.loadBundle('/tmp/checkpoint-bundle-types')

assertType<IsExact<typeof loaded, import('affon:compute').ComputeState>>()
assertType<IsExact<typeof restoredModule, typeof mod>>()
assertType<IsExact<typeof restoredState, ReturnType<typeof mod.state>>>()
assertType<IsExact<typeof loadedBundle.state, import('affon:compute').ComputeState>>()
assertType<IsExact<typeof loadedBundle.tensorGroups, Record<string, import('affon:compute').ComputeState>>>()

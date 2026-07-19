import compute, {
  Duration,
  adam,
  compile,
  module,
  parameter,
  scheduled,
  schedules,
  sgd,
  tensor,
  topk,
} from "affon:compute"

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

const input = tensor([[1, 2], [3, 4]])
const values = tensor([1, 2, 3], { dtype: "f32", device: "cpu" })
const result = compute.add(input, input)
const selected = topk(input, 1, 1)
const parameterValue = parameter([1])
const optimizer = sgd()
const adamOptimizer = adam()
const duration = Duration.steps(10)
const schedule = schedules.linear({ start: 0, end: 1, duration })
const scheduledOptimizer = scheduled(optimizer, schedule)
const compiled = compile((value: typeof input) => compute.relu(value))
const layer = module(
  { weight: parameterValue },
  (state: { weight: typeof parameterValue }, value: typeof input) => compute.mul(value, state.weight),
)

assertType<IsExact<typeof input.dtype, "f32" | "f64" | "i64">>()
assertType<IsExact<typeof values.device, "cpu">>()
assertType<IsExact<typeof result, typeof input>>()
assertType<IsExact<typeof selected.values, typeof input>>()
assertType<IsExact<typeof selected.indices, import("affon:compute").Tensor<typeof input.shape, "i64">>>()
assertType<IsExact<typeof parameterValue.grad, import("affon:compute").Tensor | null>>()
assertType<IsExact<typeof optimizer.lr, number>>()
assertType<IsExact<typeof adamOptimizer.lr, number>>()
assertType<IsExact<typeof duration.unit, "step">>()
assertType<IsExact<typeof scheduledOptimizer.context.step, number>>()
assertType<IsExact<ReturnType<typeof compiled.run>, typeof input>>()
assertType<IsExact<typeof layer.training, boolean>>()
assertType<IsExact<typeof layer.parameters, readonly import("affon:compute").Parameter[]>>()

void compiled
void layer
